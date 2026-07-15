from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from steering.domain.models import (
    ArtifactRecord,
    Chunk,
    ExperimentOutcome,
    IssueStatus,
    Project,
    ScoreBreakdown,
    SearchHit,
    SearchQuery,
)
from steering.domain.protocols import ArtifactRepository, EmbeddingProvider
from steering.retrieval.vocabulary import CONCEPT_STRATEGY_FAMILIES, expand_query

TOKEN = re.compile(r"[a-z0-9][a-z0-9_+.-]*")
MAX_RELATION_NEIGHBORS = 8
RELATION_SEED_LIMIT = 12
RETRIEVAL_STOPWORDS = frozenset(
    {
        "a",
        "also",
        "an",
        "and",
        "approach",
        "approaches",
        "architecture",
        "are",
        "as",
        "at",
        "be",
        "been",
        "before",
        "being",
        "but",
        "by",
        "can",
        "could",
        "design",
        "did",
        "different",
        "do",
        "does",
        "engineering",
        "for",
        "from",
        "give",
        "had",
        "has",
        "have",
        "how",
        "i",
        "idea",
        "ideas",
        "if",
        "in",
        "is",
        "it",
        "its",
        "make",
        "may",
        "might",
        "model",
        "models",
        "must",
        "my",
        "need",
        "no",
        "not",
        "of",
        "on",
        "only",
        "option",
        "options",
        "or",
        "our",
        "project",
        "review",
        "run",
        "saved",
        "search",
        "should",
        "so",
        "surface",
        "system",
        "tell",
        "than",
        "that",
        "the",
        "then",
        "these",
        "this",
        "those",
        "to",
        "use",
        "using",
        "want",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "without",
        "work",
        "would",
        "you",
        "your",
    }
)


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def retrieval_tokens(text: str) -> list[str]:
    return [token for token in tokenize(text) if token not in RETRIEVAL_STOPWORDS]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def _normalize_url(url: str | None) -> str:
    if not url:
        return ""
    cleaned = url.strip().rstrip(".,);]")
    try:
        parts = urlsplit(cleaned)
    except ValueError:
        return cleaned.lower().rstrip("/")
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return cleaned.lower().rstrip("/")
    path = parts.path.rstrip("/") or ""
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))


def record_text(record: ArtifactRecord) -> str:
    artifact = record.artifact
    parts = [
        artifact.title,
        artifact.short_name or "",
        artifact.canonical_url or "",
        artifact.summary,
        artifact.strategy_family.replace("_", " "),
        " ".join(artifact.aliases),
        " ".join(artifact.capabilities),
        " ".join(artifact.limitations),
        " ".join(artifact.requirements),
        " ".join(artifact.use_cases),
        " ".join(concept.name for concept in record.concepts),
        " ".join(entity.name for entity in record.entities),
        " ".join(claim.text for claim in record.claims),
        " ".join(chunk.text for chunk in record.chunks),
    ]
    return "\n".join(part for part in parts if part)


@dataclass(slots=True)
class _IndexRow:
    record: ArtifactRecord
    text: str
    term_counts: Counter[str]
    length: int
    embedding: list[float]


class HybridRetriever:
    def __init__(
        self,
        *,
        repository: ArtifactRepository,
        embedding_provider: EmbeddingProvider,
        rrf_k: int = 60,
    ) -> None:
        self.repository = repository
        self.embedding_provider = embedding_provider
        self.rrf_k = rrf_k
        self._lexical_embedding_fallback = embedding_provider.model_id in {
            "steering/hash-embedding-v1",
            "evaluation/blake2-feature-hash-v1",
        }
        self._rows: dict[str, _IndexRow] = {}
        self._document_frequency: Counter[str] = Counter()
        self._average_length = 1.0
        self._relation_neighbors: dict[str, tuple[str, ...]] = {}
        self._dirty = True

    def mark_dirty(self) -> None:
        self._dirty = True

    async def refresh(self) -> None:
        records = self.repository.list_records()
        texts = [record_text(record) for record in records]
        missing_embeddings = [
            text
            for record, text in zip(records, texts, strict=True)
            if not any(chunk.embedding for chunk in record.chunks)
        ]
        generated = await self.embedding_provider.embed(missing_embeddings) if missing_embeddings else []
        generated_index = 0
        rows: dict[str, _IndexRow] = {}
        document_frequency: Counter[str] = Counter()
        for record, text in zip(records, texts, strict=True):
            tokens = retrieval_tokens(text)
            counts = Counter(tokens)
            # BM25 document frequency counts documents containing a term, not total occurrences.
            document_frequency.update(counts.keys())
            chunk_vectors = [chunk.embedding for chunk in record.chunks if chunk.embedding]
            if chunk_vectors:
                dimension = len(chunk_vectors[0])
                embedding = [
                    sum(vector[index] for vector in chunk_vectors) / len(chunk_vectors)
                    for index in range(dimension)
                ]
            else:
                embedding = generated[generated_index]
                generated_index += 1
            rows[record.artifact.id] = _IndexRow(record, text, counts, max(1, len(tokens)), embedding)
        self._rows = rows
        self._document_frequency = document_frequency
        self._average_length = sum(row.length for row in rows.values()) / len(rows) if rows else 1.0
        self._relation_neighbors = self._build_relation_neighbors(records)
        self._dirty = False

    async def search(self, query: SearchQuery) -> list[SearchHit]:
        if self._dirty:
            await self.refresh()
        if not self._rows:
            return []
        expanded, query_concepts = expand_query(query.query)
        raw_query_tokens = retrieval_tokens(query.query)
        query_tokens = retrieval_tokens(expanded)
        query_vector = (await self.embedding_provider.embed([expanded]))[0]
        filtered = {
            artifact_id: row
            for artifact_id, row in self._rows.items()
            if self._matches_filters(row.record, query)
        }
        raw_bm25 = {artifact_id: self._bm25(row, raw_query_tokens) for artifact_id, row in filtered.items()}
        expanded_bm25 = {artifact_id: self._bm25(row, query_tokens) for artifact_id, row in filtered.items()}
        vector = {
            artifact_id: max(0.0, cosine(row.embedding, query_vector))
            for artifact_id, row in filtered.items()
        }
        exact = {
            artifact_id: self._exact_score(row, query.query, query_tokens)
            for artifact_id, row in filtered.items()
        }
        graph = {
            artifact_id: self._graph_score(row.record, query_concepts, query_tokens)
            for artifact_id, row in filtered.items()
        }
        strategy = {
            artifact_id: self._strategy_score(row.record, query_concepts)
            for artifact_id, row in filtered.items()
        }
        self._expand_relation_neighbors(graph, raw_bm25, exact, filtered)
        history = self._history_scores(query.project_id, query.query)
        constraint = {
            artifact_id: self._constraint_fit(row, query.query) for artifact_id, row in filtered.items()
        }
        channels = {
            "bm25": self._rank(raw_bm25, 50),
            "expanded_bm25": self._rank(expanded_bm25, 50),
            "vector": self._rank(vector, 50),
            "exact": self._rank(exact, 50),
            "graph": self._rank(graph, 50),
            "strategy": self._rank(strategy, 50),
            "constraint": self._rank(constraint, 50),
        }
        channel_weights = (
            {
                "bm25": 1.0,
                "expanded_bm25": 0.45,
                "vector": 0.15,
                "exact": 0.35,
                "graph": 0.45,
                "strategy": 0.65,
                "constraint": 0.25,
            }
            if self._lexical_embedding_fallback
            else {
                "bm25": 0.55,
                "expanded_bm25": 0.25,
                "vector": 1.0,
                "exact": 0.35,
                "graph": 0.45,
                "strategy": 0.65,
                "constraint": 0.25,
            }
        )
        scores: dict[str, ScoreBreakdown] = {}
        for artifact_id, row in filtered.items():
            reciprocal_rank = sum(
                channel_weights[name] / (self.rrf_k + ranks[artifact_id])
                for name, ranks in channels.items()
                if artifact_id in ranks
            )
            scores[artifact_id] = ScoreBreakdown(
                exact=exact[artifact_id],
                bm25=raw_bm25[artifact_id],
                vector=vector[artifact_id],
                graph=graph[artifact_id] + strategy[artifact_id],
                project_history=history.get(artifact_id, 0.0),
                rrf=reciprocal_rank,
                constraint_fit=constraint[artifact_id],
                evidence=row.record.artifact.evidence_quality,
            )
        raw = sorted(
            filtered,
            key=lambda artifact_id: (
                scores[artifact_id].rrf + 0.005 * scores[artifact_id].project_history,
                artifact_id,
            ),
            reverse=True,
        )
        deduplicated = self._deduplicate(raw)
        diversified = self._mmr(deduplicated, scores, query.limit, query.breadth)
        hits: list[SearchHit] = []
        for artifact_id in diversified:
            row = filtered[artifact_id]
            breakdown = scores[artifact_id]
            score = breakdown.rrf + 0.005 * breakdown.project_history
            supporting_sources = row.record.artifact.metadata.get("supporting_sources", [])
            preferred_sources = (
                [str(url) for url in supporting_sources] if isinstance(supporting_sources, list) else []
            )
            citations = list(
                dict.fromkeys(
                    [
                        *preferred_sources,
                        *(snapshot.source_url for snapshot in row.record.snapshots if snapshot.source_url),
                    ]
                )
            )
            if not citations and row.record.artifact.canonical_url:
                citations.append(row.record.artifact.canonical_url)
            hits.append(
                SearchHit(
                    artifact=row.record.artifact,
                    score=score,
                    scores=breakdown,
                    matched_chunks=self._matched_chunks(row.record, query_tokens),
                    citation_urls=citations,
                    uncertainty=self._uncertainty(row.record),
                )
            )
        return hits

    def _bm25(self, row: _IndexRow, query_tokens: list[str]) -> float:
        score = 0.0
        total_documents = max(1, len(self._rows))
        k1 = 1.5
        b = 0.75
        for token in set(query_tokens):
            frequency = row.term_counts[token]
            if not frequency:
                continue
            df = self._document_frequency[token]
            idf = math.log(1 + (total_documents - df + 0.5) / (df + 0.5))
            numerator = frequency * (k1 + 1)
            denominator = frequency + k1 * (1 - b + b * row.length / self._average_length)
            score += idf * numerator / denominator
        return score

    @staticmethod
    def _rank(scores: Mapping[str, float], limit: int) -> dict[str, int]:
        ordered = sorted(scores, key=lambda key: (scores[key], key), reverse=True)[:limit]
        return {
            artifact_id: index + 1 for index, artifact_id in enumerate(ordered) if scores[artifact_id] > 0
        }

    @staticmethod
    def _normalize(scores: Mapping[str, float]) -> dict[str, float]:
        maximum = max(scores.values(), default=0.0)
        if maximum <= 0.0:
            return {key: 0.0 for key in scores}
        return {key: max(0.0, value) / maximum for key, value in scores.items()}

    @staticmethod
    def _exact_score(row: _IndexRow, raw_query: str, query_tokens: list[str]) -> float:
        artifact = row.record.artifact
        title = artifact.title.lower()
        aliases = {alias.lower() for alias in artifact.aliases}
        lowered = raw_query.lower()
        score = 1.0 if title in lowered or any(alias in lowered for alias in aliases) else 0.0
        canonical_url = _normalize_url(artifact.canonical_url)
        query_urls = {_normalize_url(match) for match in re.findall(r"https?://[^\s]+", raw_query)}
        if canonical_url and canonical_url in query_urls:
            score += 3.0
        title_tokens = set(tokenize(title))
        if title_tokens:
            score += len(title_tokens & set(query_tokens)) / len(title_tokens)
        return score

    @staticmethod
    def _graph_score(record: ArtifactRecord, concepts: set[str], query_tokens: list[str]) -> float:
        query_set = set(query_tokens)
        names = {concept.name.lower().replace(" ", "_") for concept in record.concepts}
        names.update(entity.name.lower().replace(" ", "_") for entity in record.entities)
        family = record.artifact.strategy_family.lower().replace(" ", "_")
        score = float(len(names & concepts))
        if family in concepts:
            score += 2.0
        family_tokens = set(tokenize(record.artifact.strategy_family))
        score += len(family_tokens & query_set) / max(1, len(family_tokens))
        return score

    @staticmethod
    def _strategy_score(record: ArtifactRecord, concepts: set[str]) -> float:
        family = record.artifact.strategy_family.lower().replace(" ", "_")
        return float(sum(family in CONCEPT_STRATEGY_FAMILIES.get(concept, set()) for concept in concepts))

    @staticmethod
    def _build_relation_neighbors(records: Sequence[ArtifactRecord]) -> dict[str, tuple[str, ...]]:
        node_owners: dict[str, str] = {}
        for record in records:
            artifact_id = record.artifact.id
            node_owners[artifact_id] = artifact_id
            node_owners.update({entity.id: artifact_id for entity in record.entities})
            node_owners.update({concept.id: artifact_id for concept in record.concepts})
        neighbors: defaultdict[str, set[str]] = defaultdict(set)
        for record in records:
            for relation in record.relations:
                if not relation.approved:
                    continue
                subject_owner = node_owners.get(relation.subject_id)
                object_owner = node_owners.get(relation.object_id)
                if not subject_owner or not object_owner or subject_owner == object_owner:
                    continue
                if len(neighbors[subject_owner]) < MAX_RELATION_NEIGHBORS:
                    neighbors[subject_owner].add(object_owner)
                if len(neighbors[object_owner]) < MAX_RELATION_NEIGHBORS:
                    neighbors[object_owner].add(subject_owner)
        return {artifact_id: tuple(sorted(linked)) for artifact_id, linked in neighbors.items()}

    def _expand_relation_neighbors(
        self,
        graph_scores: dict[str, float],
        lexical_scores: Mapping[str, float],
        exact_scores: Mapping[str, float],
        filtered: Mapping[str, _IndexRow],
    ) -> None:
        seed_scores = {
            artifact_id: (
                lexical_scores.get(artifact_id, 0.0)
                + exact_scores.get(artifact_id, 0.0)
                + graph_scores.get(artifact_id, 0.0)
            )
            for artifact_id in filtered
        }
        ranked_seeds = sorted(
            filtered,
            key=lambda artifact_id: (seed_scores[artifact_id], artifact_id),
            reverse=True,
        )[:RELATION_SEED_LIMIT]
        for rank, seed_id in enumerate(ranked_seeds, start=1):
            if seed_scores[seed_id] <= 0.0:
                continue
            for neighbor_id in self._relation_neighbors.get(seed_id, ()):
                if neighbor_id in filtered:
                    graph_scores[neighbor_id] += 1.0 / rank

    @staticmethod
    def _constraint_fit(row: _IndexRow, query: str) -> float:
        query_tokens = set(tokenize(query))
        requirement_tokens = set(tokenize(" ".join(row.record.artifact.requirements)))
        capability_tokens = set(tokenize(" ".join(row.record.artifact.capabilities)))
        local_bonus = 1.0 if "local" in query_tokens and "local" in tokenize(row.text) else 0.0
        return local_bonus + len(query_tokens & (requirement_tokens | capability_tokens)) / max(
            1,
            len(query_tokens),
        )

    def _history_scores(self, project_id: str | None, query: str) -> dict[str, float]:
        if not project_id:
            return {}
        history = self.repository.project_history(project_id)
        current_constraint_tokens = set(retrieval_tokens(query))
        for project in history.get("projects", []):
            if isinstance(project, Project):
                current_constraint_tokens.update(retrieval_tokens(" ".join(project.constraints)))
        scores: defaultdict[str, float] = defaultdict(float)
        for outcome in history.get("outcomes", []):
            if not isinstance(outcome, ExperimentOutcome) or not outcome.artifact_id:
                continue
            outcome_constraint_tokens = set(retrieval_tokens(" ".join(outcome.constraints)))
            constraints_overlap = bool(outcome_constraint_tokens & current_constraint_tokens)
            if outcome_constraint_tokens and not constraints_overlap:
                continue
            if outcome.succeeded is True:
                scores[outcome.artifact_id] += 1.0
            elif outcome.succeeded is False and constraints_overlap:
                scores[outcome.artifact_id] -= 0.5
        return dict(scores)

    def _mmr(
        self,
        ordered: list[str],
        scores: Mapping[str, ScoreBreakdown],
        limit: int,
        breadth: bool,
    ) -> list[str]:
        if not breadth:
            return ordered[:limit]
        candidate_pool = ordered[: min(len(ordered), max(20, limit * 3))]
        available_families = {
            self._rows[artifact_id].record.artifact.strategy_family for artifact_id in candidate_pool
        }
        diversity_target = min(4, limit, len(available_families))
        selected: list[str] = []
        selected_families: set[str] = set()
        relevance = self._normalize({item: scores[item].rrf for item in candidate_pool})
        while candidate_pool and len(selected) < limit:
            enforce_new_family = len(selected_families) < diversity_target
            eligible = [
                artifact_id
                for artifact_id in candidate_pool
                if not enforce_new_family
                or self._rows[artifact_id].record.artifact.strategy_family not in selected_families
            ]
            if not eligible:
                eligible = candidate_pool

            def mmr_score(artifact_id: str) -> tuple[float, float, str]:
                redundancy = max(
                    (
                        cosine(
                            self._rows[artifact_id].embedding,
                            self._rows[selected_id].embedding,
                        )
                        for selected_id in selected
                    ),
                    default=0.0,
                )
                value = 0.72 * relevance[artifact_id] - 0.28 * max(0.0, redundancy)
                return value, relevance[artifact_id], artifact_id

            chosen = max(eligible, key=mmr_score)
            selected.append(chosen)
            selected_families.add(self._rows[chosen].record.artifact.strategy_family)
            candidate_pool.remove(chosen)
        for artifact_id in ordered:
            if len(selected) >= limit:
                break
            if artifact_id not in selected:
                selected.append(artifact_id)
        return selected

    def _deduplicate(self, ordered: Sequence[str]) -> list[str]:
        seen_urls: set[str] = set()
        seen_hashes: set[str] = set()
        deduplicated: list[str] = []
        for artifact_id in ordered:
            artifact = self._rows[artifact_id].record.artifact
            canonical_url = _normalize_url(artifact.canonical_url)
            content_hash = artifact.content_hash.strip().lower()
            if canonical_url and canonical_url in seen_urls:
                continue
            if content_hash and content_hash in seen_hashes:
                continue
            deduplicated.append(artifact_id)
            if canonical_url:
                seen_urls.add(canonical_url)
            if content_hash:
                seen_hashes.add(content_hash)
        return deduplicated

    @staticmethod
    def _matches_filters(record: ArtifactRecord, query: SearchQuery) -> bool:
        artifact = record.artifact
        if query.artifact_types and artifact.artifact_type not in query.artifact_types:
            return False
        if query.minimum_evidence is not None and artifact.evidence_quality < query.minimum_evidence:
            return False
        if query.published_after is not None and (
            artifact.published_at is None or artifact.published_at < query.published_after
        ):
            return False
        if query.concepts:
            names = {concept.name.lower() for concept in record.concepts}
            if not names & {value.lower() for value in query.concepts}:
                return False
        return artifact.review_status.value != "rejected"

    @staticmethod
    def _matched_chunks(record: ArtifactRecord, query_tokens: list[str]) -> list[Chunk]:
        query = set(query_tokens)
        ranked = sorted(
            record.chunks,
            key=lambda chunk: len(query & set(tokenize(chunk.text))),
            reverse=True,
        )
        return ranked[:3]

    @staticmethod
    def _uncertainty(record: ArtifactRecord) -> str | None:
        if any(issue.status == IssueStatus.UNRESOLVED for issue in record.issues):
            return "This artifact has an unresolved source-reconciliation issue."
        if record.artifact.trust_lane.value == "experimental":
            return "Experimental or unreviewed source evidence; validate before adoption."
        if not record.claims:
            return "No structured evidence claims are stored for this artifact."
        return None
