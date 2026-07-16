from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast
from urllib.parse import urlsplit, urlunsplit

from steering.database.native import NativeCandidate, normalize_search_term
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
from steering.domain.protocols import EmbeddingProvider
from steering.retrieval.vocabulary import CONCEPT_STRATEGY_FAMILIES, expand_query

TOKEN = re.compile(r"[a-z0-9][a-z0-9_+.-]*")
URL = re.compile(r"https?://[^\s]+")
RELATION_SEED_LIMIT = 12
CANDIDATE_LIMIT = 50
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


class NativeRetrievalRepository(Protocol):
    def list_records(self) -> list[ArtifactRecord]: ...

    def replace_chunk_embeddings(self, chunks: Sequence[Chunk]) -> int: ...

    def backup_before_reembedding(self) -> str: ...

    def rebuild_search_indexes(self) -> None: ...

    def assert_embedding_compatible(
        self,
        *,
        provider: str,
        model: str,
        revision: str | None,
        dimension: int,
        task_mode: str,
        normalized: bool,
    ) -> None: ...

    def exact_candidates(
        self, terms: Sequence[str], query: SearchQuery, *, limit: int = 50
    ) -> list[NativeCandidate]: ...

    def bm25_candidates(self, text: str, query: SearchQuery, *, limit: int = 50) -> list[NativeCandidate]: ...

    def vector_candidates(
        self, vector: Sequence[float], query: SearchQuery, *, limit: int = 50
    ) -> list[NativeCandidate]: ...

    def graph_candidates(
        self, seed_ids: Sequence[str], query: SearchQuery, *, limit: int = 50
    ) -> list[NativeCandidate]: ...

    def load_records(self, artifact_ids: Sequence[str]) -> list[ArtifactRecord]: ...

    def project_history(self, project_id: str) -> Mapping[str, Sequence[object]]: ...


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


def exact_query_terms(text: str) -> list[str]:
    tokens = tokenize(text)
    terms = {normalize_search_term(text)}
    terms.update(_normalize_url(match) for match in URL.findall(text))
    terms.update(tokens)
    for width in range(2, min(6, len(tokens)) + 1):
        terms.update(" ".join(tokens[start : start + width]) for start in range(len(tokens) - width + 1))
    terms.discard("")
    return sorted(terms)


@dataclass(slots=True)
class _CandidateRow:
    record: ArtifactRecord
    embedding: list[float]


class HybridRetriever:
    def __init__(
        self,
        *,
        repository: NativeRetrievalRepository,
        embedding_provider: EmbeddingProvider,
        rrf_k: int = 60,
    ) -> None:
        self.repository = repository
        self.embedding_provider = embedding_provider
        self.rrf_k = rrf_k
        self._rows: dict[str, _CandidateRow] = {}
        self._dirty = True

    def mark_dirty(self) -> None:
        self._dirty = True

    async def refresh(self) -> None:
        self.repository.rebuild_search_indexes()
        self._rows = {}
        self._dirty = False

    async def reembed_all(self) -> tuple[int, str | None]:
        """Replace every stored vector as one verified maintenance operation."""

        chunks = [chunk for record in self.repository.list_records() for chunk in record.chunks]
        if not chunks:
            await self.refresh()
            return 0, None
        vectors = await self.embedding_provider.embed_documents([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise ValueError("embedding provider returned an incomplete re-embedding batch")
        updated = [
            chunk.model_copy(
                update={
                    "embedding": vector,
                    "embedding_provider": self.embedding_provider.provider_id,
                    "embedding_model": self.embedding_provider.model_id,
                    "embedding_revision": self.embedding_provider.model_revision,
                    "embedding_dimension": self.embedding_provider.dimension,
                    "embedding_task_mode": self.embedding_provider.document_task_mode,
                    "embedding_normalized": self.embedding_provider.normalized,
                    "source_content_hash": hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
                }
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        backup = self.repository.backup_before_reembedding()
        replaced = self.repository.replace_chunk_embeddings(updated)
        if replaced != len(updated):
            raise RuntimeError("complete re-embedding did not replace every chunk")
        await self.refresh()
        return replaced, backup

    async def _embed_query(self, text: str) -> list[float]:
        embed_query = getattr(self.embedding_provider, "embed_query", None)
        if callable(embed_query):
            return cast(list[float], await embed_query(text))
        return (await self.embedding_provider.embed([text]))[0]

    async def search(self, query: SearchQuery) -> list[SearchHit]:
        if self._dirty:
            await self.refresh()
        compatibility_check = getattr(self.repository, "assert_embedding_compatible", None)
        if callable(compatibility_check):
            compatibility_check(
                provider=self.embedding_provider.provider_id,
                model=self.embedding_provider.model_id,
                revision=self.embedding_provider.model_revision,
                dimension=self.embedding_provider.dimension,
                task_mode=self.embedding_provider.document_task_mode,
                normalized=self.embedding_provider.normalized,
            )
        expanded, query_concepts = expand_query(query.query)
        raw_tokens = retrieval_tokens(query.query)
        expanded_tokens = retrieval_tokens(expanded)
        query_vector = await self._embed_query(expanded)
        strategy_terms = {
            family for concept in query_concepts for family in CONCEPT_STRATEGY_FAMILIES.get(concept, set())
        }
        exact_candidates = self.repository.exact_candidates(
            sorted({*exact_query_terms(query.query), *strategy_terms}),
            query,
            limit=CANDIDATE_LIMIT,
        )
        raw_bm25_candidates = self.repository.bm25_candidates(
            " ".join(raw_tokens) or query.query, query, limit=CANDIDATE_LIMIT
        )
        expanded_bm25_candidates = self.repository.bm25_candidates(
            " ".join(expanded_tokens) or expanded, query, limit=CANDIDATE_LIMIT
        )
        vector_candidates = self.repository.vector_candidates(query_vector, query, limit=CANDIDATE_LIMIT)
        exact = self._candidate_scores(exact_candidates)
        raw_bm25 = self._candidate_scores(raw_bm25_candidates)
        expanded_bm25 = self._candidate_scores(expanded_bm25_candidates)
        vector = self._candidate_scores(vector_candidates)
        seed_scores: dict[str, float] = defaultdict(float)
        for channel in (exact, raw_bm25, vector):
            for artifact_id, score in channel.items():
                seed_scores[artifact_id] += score
        seed_ids = sorted(
            seed_scores,
            key=lambda artifact_id: (seed_scores[artifact_id], artifact_id),
            reverse=True,
        )[:RELATION_SEED_LIMIT]
        graph_candidates = self.repository.graph_candidates(seed_ids, query, limit=CANDIDATE_LIMIT)
        graph = self._candidate_scores(graph_candidates)
        artifact_ids = list(
            dict.fromkeys(
                candidate.artifact_id
                for candidates in (
                    exact_candidates,
                    raw_bm25_candidates,
                    expanded_bm25_candidates,
                    vector_candidates,
                    graph_candidates,
                )
                for candidate in candidates
            )
        )
        records = self.repository.load_records(artifact_ids)
        self._rows = {
            record.artifact.id: _CandidateRow(record, self._record_embedding(record)) for record in records
        }
        if not self._rows:
            return []
        strategy = {
            artifact_id: self._strategy_score(row.record, query_concepts)
            for artifact_id, row in self._rows.items()
        }
        local_graph = {
            artifact_id: self._graph_score(row.record, query_concepts, expanded_tokens)
            for artifact_id, row in self._rows.items()
        }
        constraint = {
            artifact_id: self._constraint_fit(row, query.query) for artifact_id, row in self._rows.items()
        }
        history = self._history_scores(query.project_id, query.query)
        channels = {
            "bm25": self._rank(raw_bm25, CANDIDATE_LIMIT),
            "expanded_bm25": self._rank(expanded_bm25, CANDIDATE_LIMIT),
            "vector": self._rank(vector, CANDIDATE_LIMIT),
            "exact": self._rank(exact, CANDIDATE_LIMIT),
            "graph": self._rank(
                {
                    artifact_id: graph.get(artifact_id, 0.0) + local_graph[artifact_id]
                    for artifact_id in self._rows
                },
                CANDIDATE_LIMIT,
            ),
            "strategy": self._rank(strategy, CANDIDATE_LIMIT),
            "constraint": self._rank(constraint, CANDIDATE_LIMIT),
        }
        channel_weights = {
            "bm25": 0.55,
            "expanded_bm25": 0.25,
            "vector": 1.0,
            "exact": 0.35,
            "graph": 0.45,
            "strategy": 1.0,
            "constraint": 0.25,
        }
        scores: dict[str, ScoreBreakdown] = {}
        for artifact_id, row in self._rows.items():
            reciprocal_rank = sum(
                channel_weights[name] / (self.rrf_k + ranks[artifact_id])
                for name, ranks in channels.items()
                if artifact_id in ranks
            )
            scores[artifact_id] = ScoreBreakdown(
                exact=exact.get(artifact_id, 0.0),
                bm25=raw_bm25.get(artifact_id, 0.0),
                vector=vector.get(artifact_id, 0.0),
                graph=graph.get(artifact_id, 0.0) + local_graph[artifact_id] + strategy[artifact_id],
                project_history=history.get(artifact_id, 0.0),
                rrf=reciprocal_rank,
                constraint_fit=constraint[artifact_id],
                evidence=row.record.artifact.evidence_quality,
            )
        ordered = sorted(
            self._rows,
            key=lambda artifact_id: (
                scores[artifact_id].rrf + 0.005 * scores[artifact_id].project_history,
                artifact_id,
            ),
            reverse=True,
        )
        diversified = self._mmr(
            self._deduplicate(ordered),
            scores,
            query.limit,
            query.breadth,
            query_concepts,
        )
        preferred_chunks = self._preferred_chunks(
            raw_bm25_candidates, expanded_bm25_candidates, vector_candidates
        )
        hits: list[SearchHit] = []
        for artifact_id in diversified:
            row = self._rows[artifact_id]
            breakdown = scores[artifact_id]
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
                    score=breakdown.rrf + 0.005 * breakdown.project_history,
                    scores=breakdown,
                    matched_chunks=self._matched_chunks(
                        row.record, expanded_tokens, preferred_chunks.get(artifact_id, [])
                    ),
                    citation_urls=citations,
                    uncertainty=self._uncertainty(row.record),
                )
            )
        return hits

    @staticmethod
    def _candidate_scores(candidates: Sequence[NativeCandidate]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for candidate in candidates:
            scores[candidate.artifact_id] = max(candidate.score, scores.get(candidate.artifact_id, 0.0))
        return scores

    @staticmethod
    def _preferred_chunks(*channels: Sequence[NativeCandidate]) -> dict[str, list[str]]:
        preferred: defaultdict[str, list[str]] = defaultdict(list)
        for candidates in channels:
            for candidate in candidates:
                if candidate.chunk_id and candidate.chunk_id not in preferred[candidate.artifact_id]:
                    preferred[candidate.artifact_id].append(candidate.chunk_id)
        return dict(preferred)

    @staticmethod
    def _record_embedding(record: ArtifactRecord) -> list[float]:
        vectors = [chunk.embedding for chunk in record.chunks if chunk.embedding]
        if not vectors:
            return []
        dimension = len(vectors[0])
        compatible = [vector for vector in vectors if len(vector) == dimension]
        return [sum(vector[index] for vector in compatible) / len(compatible) for index in range(dimension)]

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
    def _constraint_fit(row: _CandidateRow, query: str) -> float:
        query_tokens = set(tokenize(query))
        requirement_tokens = set(tokenize(" ".join(row.record.artifact.requirements)))
        capability_tokens = set(tokenize(" ".join(row.record.artifact.capabilities)))
        chunk_tokens = set(tokenize(" ".join(chunk.text for chunk in row.record.chunks)))
        local_bonus = 1.0 if "local" in query_tokens and "local" in chunk_tokens else 0.0
        return local_bonus + len(query_tokens & (requirement_tokens | capability_tokens)) / max(
            1, len(query_tokens)
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
        concepts: set[str] | None = None,
    ) -> list[str]:
        if not breadth:
            return ordered[:limit]
        candidate_pool = ordered[: min(len(ordered), max(20, limit * 3))]
        available_families = {
            self._rows[artifact_id].record.artifact.strategy_family for artifact_id in candidate_pool
        }
        diversity_target = min(limit, len(available_families))
        selected: list[str] = []
        selected_families: set[str] = set()
        candidate_concepts = {
            artifact_id: {
                concept
                for concept in concepts or set()
                if self._rows[artifact_id].record.artifact.strategy_family
                in CONCEPT_STRATEGY_FAMILIES.get(concept, set())
            }
            for artifact_id in candidate_pool
        }
        available_concept_families = {
            concept: {
                self._rows[artifact_id].record.artifact.strategy_family
                for artifact_id, matched in candidate_concepts.items()
                if concept in matched
            }
            for concept in concepts or set()
        }
        concept_targets = {
            concept: min(2, len(families), limit)
            for concept, families in available_concept_families.items()
            if families
        }
        covered_concept_families: defaultdict[str, set[str]] = defaultdict(set)
        relevance = self._normalize({item: scores[item].rrf for item in candidate_pool})
        while candidate_pool and len(selected) < limit:
            undercovered_concepts = {
                concept
                for concept, target in concept_targets.items()
                if len(covered_concept_families[concept]) < target
            }
            eligible = [
                artifact_id
                for artifact_id in candidate_pool
                if any(
                    concept in candidate_concepts[artifact_id]
                    and self._rows[artifact_id].record.artifact.strategy_family
                    not in covered_concept_families[concept]
                    for concept in undercovered_concepts
                )
            ]
            if not eligible:
                enforce_new_family = len(selected_families) < diversity_target
                eligible = [
                    artifact_id
                    for artifact_id in candidate_pool
                    if not enforce_new_family
                    or self._rows[artifact_id].record.artifact.strategy_family not in selected_families
                ]
            if not eligible:
                eligible = candidate_pool

            def mmr_score(
                artifact_id: str,
                active_concepts: set[str] = undercovered_concepts,
            ) -> tuple[float, float, float, str]:
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
                concept_priority = sum(
                    (concept_targets[concept] - len(covered_concept_families[concept]))
                    / concept_targets[concept]
                    for concept in candidate_concepts[artifact_id] & active_concepts
                )
                return concept_priority, value, relevance[artifact_id], artifact_id

            chosen = max(eligible, key=mmr_score)
            selected.append(chosen)
            chosen_family = self._rows[chosen].record.artifact.strategy_family
            selected_families.add(chosen_family)
            for concept in candidate_concepts[chosen]:
                covered_concept_families[concept].add(chosen_family)
            candidate_pool.remove(chosen)
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
    def _matched_chunks(
        record: ArtifactRecord, query_tokens: list[str], preferred_ids: Sequence[str]
    ) -> list[Chunk]:
        chunks = {chunk.id: chunk for chunk in record.chunks}
        matched = [chunks[identifier] for identifier in preferred_ids if identifier in chunks]
        query = set(query_tokens)
        remaining = sorted(
            (chunk for chunk in record.chunks if chunk.id not in preferred_ids),
            key=lambda chunk: len(query & set(tokenize(chunk.text))),
            reverse=True,
        )
        return [*matched, *remaining][:3]

    @staticmethod
    def _uncertainty(record: ArtifactRecord) -> str | None:
        if any(issue.status == IssueStatus.UNRESOLVED for issue in record.issues):
            return "This artifact has an unresolved source-reconciliation issue."
        if record.artifact.trust_lane.value == "experimental":
            return "Experimental or unreviewed source evidence; validate before adoption."
        if not record.claims:
            return "No structured evidence claims are stored for this artifact."
        return None
