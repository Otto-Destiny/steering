from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

from steering import SCHEMA_VERSION
from steering.domain.credentials import reject_high_confidence_credentials, sanitized_persistence_source
from steering.domain.models import (
    Artifact,
    ArtifactRecord,
    ArtifactType,
    Chunk,
    Claim,
    Concept,
    Entity,
    EvidenceCategory,
    EvidenceSpan,
    Relation,
    ResolvedSource,
    ReviewIssue,
    ReviewStatus,
    Snapshot,
    SourceKind,
    TrustLane,
)
from steering.domain.protocols import EmbeddingProvider, GenerationProvider
from steering.extraction.cache import ExtractionCache
from steering.extraction.prompts import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    extraction_prompt,
    reconciliation_prompt,
)
from steering.extraction.schemas import KnowledgeExtraction

HEADING = re.compile(r"(?m)^(?:#{1,6}\s+.+|[A-Z][A-Z0-9 ]{4,})$")


class EvidenceValidationError(ValueError):
    pass


class ContextBudgetError(ValueError):
    pass


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_id(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.blake2b(value.encode('utf-8'), digest_size=12).hexdigest()}"


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def _chunk_text(text: str, max_chars: int = 6000) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    boundaries = [match.start() for match in HEADING.finditer(text)]
    boundaries = sorted({0, *boundaries, len(text)})
    sections = [text[left:right].strip() for left, right in pairwise(boundaries)]
    sections = [section for section in sections if section]
    chunks: list[str] = []
    current = ""
    for section in sections or [text]:
        paragraphs = section.split("\n\n") if len(section) > max_chars else [section]
        for paragraph in paragraphs:
            if current and len(current) + len(paragraph) + 2 > max_chars:
                chunks.append(current)
                current = ""
            if len(paragraph) > max_chars:
                for start in range(0, len(paragraph), max_chars):
                    if current:
                        chunks.append(current)
                        current = ""
                    chunks.append(paragraph[start : start + max_chars])
            else:
                current = f"{current}\n\n{paragraph}".strip()
    if current:
        chunks.append(current)
    return chunks


class ExtractionService:
    def __init__(
        self,
        *,
        generation: GenerationProvider,
        embedding: EmbeddingProvider,
        cache: ExtractionCache | None = None,
        context_window_tokens: int = 32_000,
        reserved_output_tokens: int = 4_000,
    ) -> None:
        self.generation = generation
        self.embedding = embedding
        self.cache = cache
        self.context_window_tokens = context_window_tokens
        self.reserved_output_tokens = reserved_output_tokens
        if reserved_output_tokens >= context_window_tokens:
            raise ValueError("reserved output tokens must be smaller than the context window")

    async def extract(
        self,
        primary: ResolvedSource,
        supporting_sources: Sequence[ResolvedSource] = (),
    ) -> ArtifactRecord:
        sources = [primary, *supporting_sources]
        cache_key = self._cache_key(sources)
        payload = self.cache.get(cache_key) if self.cache else None
        if payload is None:
            payload = await self._generate(sources)
            if self.cache:
                self.cache.put(cache_key, payload)
        reject_high_confidence_credentials(json.dumps(payload.model_dump(mode="json"), ensure_ascii=False))
        return await self._build_record(sources, payload)

    def _cache_key(self, sources: Sequence[ResolvedSource]) -> str:
        digest = hashlib.sha256()
        for source in sources:
            digest.update(source.canonical_url.encode("utf-8"))
            digest.update(content_hash(source.text).encode("ascii"))
        digest.update(self.generation.model_id.encode("utf-8"))
        digest.update(str(SCHEMA_VERSION).encode("ascii"))
        digest.update(PROMPT_VERSION.encode("utf-8"))
        return digest.hexdigest()

    async def _generate(self, sources: list[ResolvedSource]) -> KnowledgeExtraction:
        prompt = extraction_prompt(sources)
        if self._fits_context(prompt):
            return await self.generation.generate_structured(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
                response_model=KnowledgeExtraction,
            )

        partials: list[KnowledgeExtraction] = []
        for source_index, source in enumerate(sources):
            for fragment in self._source_fragments(source, source_index):
                partial = await self.generation.generate_structured(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=extraction_prompt([fragment]),
                    response_model=KnowledgeExtraction,
                )
                partials.append(self._remap_source_indices(partial, source_index))
        return await self._reconcile(sources, partials)

    @property
    def _safe_input_tokens(self) -> int:
        return self.context_window_tokens - self.reserved_output_tokens

    def _fits_context(self, user_prompt: str) -> bool:
        # UTF-8 bytes / 3 deliberately overestimates ordinary English while
        # remaining conservative for CJK text and emoji without a model-specific tokenizer.
        estimated_tokens = (len((SYSTEM_PROMPT + user_prompt).encode("utf-8")) + 2) // 3
        return estimated_tokens <= self._safe_input_tokens

    def _source_fragments(
        self,
        source: ResolvedSource,
        source_index: int,
    ) -> list[ResolvedSource]:
        max_chars = min(6000, max(128, self._safe_input_tokens * 3))
        pending = _chunk_text(source.text, max_chars=max_chars)
        fitted: list[str] = []
        while pending:
            text = pending.pop(0)
            candidate = source.model_copy(
                update={
                    "text": text,
                    "title": f"{source.title} [part]",
                    "metadata": {**source.metadata, "original_source_index": source_index},
                }
            )
            if self._fits_context(extraction_prompt([candidate])):
                fitted.append(text)
                continue
            if len(text) <= 1:
                raise ContextBudgetError("the configured context window is too small for extraction")
            midpoint = len(text) // 2
            pending[0:0] = [text[:midpoint], text[midpoint:]]
        return [
            source.model_copy(
                update={
                    "text": text,
                    "title": f"{source.title} [part {ordinal + 1}]",
                    "metadata": {**source.metadata, "original_source_index": source_index},
                }
            )
            for ordinal, text in enumerate(fitted)
        ]

    async def _reconcile(
        self,
        sources: list[ResolvedSource],
        partials: list[KnowledgeExtraction],
    ) -> KnowledgeExtraction:
        current = partials
        while len(current) > 1:
            batches: list[list[KnowledgeExtraction]] = []
            batch: list[KnowledgeExtraction] = []
            for partial in current:
                candidate = [*batch, partial]
                prompt = reconciliation_prompt(
                    sources,
                    [item.model_dump(mode="json") for item in candidate],
                )
                if self._fits_context(prompt):
                    batch = candidate
                    continue
                if not batch:
                    raise ContextBudgetError("one extraction fragment is too large for safe reconciliation")
                batches.append(batch)
                batch = [partial]
            if batch:
                batches.append(batch)
            if all(len(item) == 1 for item in batches):
                raise ContextBudgetError(
                    "the configured context window cannot reconcile two extraction fragments"
                )

            consolidated: list[KnowledgeExtraction] = []
            for item in batches:
                if len(item) == 1:
                    consolidated.append(item[0])
                    continue
                prompt = reconciliation_prompt(
                    sources,
                    [partial.model_dump(mode="json") for partial in item],
                )
                consolidated.append(
                    await self.generation.generate_structured(
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt=prompt,
                        response_model=KnowledgeExtraction,
                    )
                )
            current = consolidated
        if not current:
            raise ContextBudgetError("extraction did not produce any fragments")
        return current[0]

    @staticmethod
    def _remap_source_indices(
        payload: KnowledgeExtraction,
        source_index: int,
    ) -> KnowledgeExtraction:
        """Restore original source indices after extracting an isolated chunk."""

        return payload.model_copy(
            update={
                "claims": [
                    claim.model_copy(update={"source_index": source_index}) for claim in payload.claims
                ],
                "relations": [
                    relation.model_copy(update={"source_index": source_index})
                    for relation in payload.relations
                ],
                "reported_results": [
                    result.model_copy(update={"source_index": source_index})
                    for result in payload.reported_results
                ],
            }
        )

    async def _build_record(
        self,
        sources: list[ResolvedSource],
        payload: KnowledgeExtraction,
    ) -> ArtifactRecord:
        artifact_id = stable_id("art", sources[0].canonical_url)
        snapshots = [
            Snapshot(
                id=stable_id("snap", f"{source.canonical_url}:{content_hash(source.text)}"),
                artifact_id=artifact_id,
                source_url=source.canonical_url,
                content_hash=content_hash(source.text),
                mime_type=source.mime_type,
                text=source.text,
                extraction_method=source.extraction_method,
                partial=source.partial,
            )
            for source in sources
        ]
        combined_hash = content_hash("\n".join(snapshot.content_hash for snapshot in snapshots))
        evidence_quality = self._evidence_quality(payload, sources[0].source_kind)
        artifact = Artifact(
            id=artifact_id,
            canonical_url=sources[0].canonical_url,
            source_kind=sources[0].source_kind,
            artifact_type=payload.artifact_type,
            title=payload.title,
            short_name=payload.short_name,
            summary=payload.summary,
            strategy_family=payload.strategy_family,
            review_status=ReviewStatus.CAPTURED,
            trust_lane=self._trust_lane(payload.artifact_type, sources[0].source_kind),
            evidence_quality=evidence_quality,
            maturity=payload.maturity,
            license=payload.license,
            aliases=_unique(payload.aliases),
            capabilities=_unique(payload.capabilities),
            limitations=_unique(payload.limitations),
            requirements=_unique(payload.requirements),
            use_cases=_unique(payload.use_cases),
            published_at=sources[0].published_at,
            content_hash=combined_hash,
            metadata={
                "resolver": sources[0].extraction_method,
                "supporting_sources": [source.canonical_url for source in sources[1:]],
                "media_sources": [sanitized_persistence_source(url) for url in sources[0].media_urls],
                "media_decision": sources[0].metadata.get("media_decision"),
                "reported_results": [row.model_dump(mode="json") for row in payload.reported_results],
            },
        )
        claims: list[Claim] = []
        spans: list[EvidenceSpan] = []
        for claim_data in payload.claims:
            snapshot, start, end = self._locate_quote(
                snapshots,
                claim_data.source_index,
                claim_data.exact_quote,
            )
            claim_id = stable_id("claim", f"{artifact_id}:{claim_data.text}:{claim_data.exact_quote}")
            span_id = stable_id("span", f"{snapshot.id}:{start}:{end}")
            spans.append(
                EvidenceSpan(
                    id=span_id,
                    snapshot_id=snapshot.id,
                    claim_id=claim_id,
                    quote=claim_data.exact_quote,
                    start=start,
                    end=end,
                    locator=claim_data.locator,
                )
            )
            claims.append(
                Claim(
                    id=claim_id,
                    artifact_id=artifact_id,
                    text=claim_data.text,
                    category=claim_data.category,
                    confidence=claim_data.confidence,
                    evidence_span_ids=[span_id],
                )
            )

        result_claims, result_spans = self._reported_result_claims(artifact_id, snapshots, payload)
        claims.extend(result_claims)
        spans.extend(result_spans)
        entities: list[Entity] = []
        concepts: list[Concept] = []
        relations: list[Relation] = []
        for relation_data in payload.relations:
            target_key = relation_data.target_name.strip().lower()
            if relation_data.target_type == "concept":
                target_id = stable_id("concept", target_key)
                concepts.append(Concept(id=target_id, name=relation_data.target_name))
            else:
                target_id = stable_id("entity", f"{relation_data.target_type}:{target_key}")
                entities.append(
                    Entity(
                        id=target_id,
                        name=relation_data.target_name,
                        entity_type=relation_data.target_type,
                    )
                )
            span_ids: list[str] = []
            if relation_data.exact_quote:
                snapshot, start, end = self._locate_quote(
                    snapshots,
                    relation_data.source_index,
                    relation_data.exact_quote,
                )
                relation_claim_id = stable_id("claim", f"relation:{artifact_id}:{target_id}")
                span_id = stable_id("span", f"{snapshot.id}:{start}:{end}:relation")
                spans.append(
                    EvidenceSpan(
                        id=span_id,
                        snapshot_id=snapshot.id,
                        claim_id=relation_claim_id,
                        quote=relation_data.exact_quote,
                        start=start,
                        end=end,
                        locator="relationship evidence",
                    )
                )
                span_ids.append(span_id)
            relations.append(
                Relation(
                    id=stable_id("rel", f"{artifact_id}:{relation_data.predicate}:{target_id}"),
                    subject_id=artifact_id,
                    predicate=relation_data.predicate,
                    object_id=target_id,
                    approved=False,
                    evidence_span_ids=span_ids,
                    rationale=relation_data.rationale,
                )
            )
        chunks = await self._chunks(artifact_id, snapshots)
        issues: list[ReviewIssue] = []
        for issue in payload.issues:
            issue_id = stable_id(
                "issue",
                f"{artifact_id}:{issue.social_statement}:{issue.source_statement}",
            )
            issue_span_ids: list[str] = []
            evidence = (
                (
                    "social",
                    issue.social_statement,
                    issue.social_source_url,
                    issue.social_exact_quote,
                ),
                (
                    "primary",
                    issue.source_statement,
                    issue.primary_source_url,
                    issue.primary_exact_quote,
                ),
            )
            for role, statement, source_url, quote in evidence:
                snapshot, start, end = self._locate_issue_quote(snapshots, source_url, quote)
                claim_id = stable_id("claim", f"issue:{issue_id}:{role}:{statement}")
                span_id = stable_id("span", f"{snapshot.id}:{start}:{end}:issue:{role}")
                category = (
                    EvidenceCategory.SOCIAL_CLAIM
                    if role == "social"
                    else self._source_evidence_category(sources, snapshot.source_url)
                )
                claims.append(
                    Claim(
                        id=claim_id,
                        artifact_id=artifact_id,
                        text=statement,
                        category=category,
                        confidence=0.5 if role == "social" else 0.75,
                        evidence_span_ids=[span_id],
                    )
                )
                spans.append(
                    EvidenceSpan(
                        id=span_id,
                        snapshot_id=snapshot.id,
                        claim_id=claim_id,
                        quote=quote,
                        start=start,
                        end=end,
                        locator=f"conflict {role} evidence",
                    )
                )
                issue_span_ids.append(span_id)
            issues.append(
                ReviewIssue(
                    id=issue_id,
                    artifact_id=artifact_id,
                    social_statement=issue.social_statement,
                    source_statement=issue.source_statement,
                    explanation=issue.explanation,
                    social_source_url=issue.social_source_url,
                    primary_source_url=issue.primary_source_url,
                    evidence_span_ids=issue_span_ids,
                )
            )
        return ArtifactRecord(
            artifact=artifact,
            snapshots=snapshots,
            chunks=chunks,
            claims=claims,
            evidence_spans=spans,
            relations=relations,
            issues=issues,
            entities=list({entity.id: entity for entity in entities}.values()),
            concepts=list({concept.id: concept for concept in concepts}.values()),
        )

    def _locate_quote(
        self,
        snapshots: list[Snapshot],
        source_index: int,
        quote: str,
    ) -> tuple[Snapshot, int, int]:
        if source_index >= len(snapshots):
            raise EvidenceValidationError(f"invalid source index {source_index}")
        snapshot = snapshots[source_index]
        start = snapshot.text.find(quote)
        if start < 0:
            raise EvidenceValidationError("generated evidence quote does not occur exactly in its source")
        return snapshot, start, start + len(quote)

    def _locate_issue_quote(
        self,
        snapshots: list[Snapshot],
        source_url: str,
        quote: str,
    ) -> tuple[Snapshot, int, int]:
        for snapshot in (item for item in snapshots if item.source_url == source_url):
            start = snapshot.text.find(quote)
            if start >= 0:
                return snapshot, start, start + len(quote)
        raise EvidenceValidationError("generated issue evidence quote does not occur in its source")

    @staticmethod
    def _source_evidence_category(
        sources: list[ResolvedSource],
        source_url: str,
    ) -> EvidenceCategory:
        source = next((item for item in sources if item.canonical_url == source_url), None)
        if source is not None and source.source_kind in {SourceKind.PAPER, SourceKind.PDF}:
            return EvidenceCategory.RESEARCH_PAPER
        return EvidenceCategory.MAINTAINER_DOCUMENTATION

    def _reported_result_claims(
        self,
        artifact_id: str,
        snapshots: list[Snapshot],
        payload: KnowledgeExtraction,
    ) -> tuple[list[Claim], list[EvidenceSpan]]:
        claims: list[Claim] = []
        spans: list[EvidenceSpan] = []
        for row in payload.reported_results:
            parts = [row.method_or_model_version, row.dataset_or_benchmark, row.metric, row.result]
            text = " | ".join(part for part in parts if part)
            if not text:
                continue
            snapshot, start, end = self._locate_quote(snapshots, row.source_index, row.exact_quote)
            claim_id = stable_id("claim", f"result:{artifact_id}:{text}")
            span_id = stable_id("span", f"{snapshot.id}:{start}:{end}:result")
            spans.append(
                EvidenceSpan(
                    id=span_id,
                    snapshot_id=snapshot.id,
                    claim_id=claim_id,
                    quote=row.exact_quote,
                    start=start,
                    end=end,
                    locator="reported result",
                )
            )
            claims.append(
                Claim(
                    id=claim_id,
                    artifact_id=artifact_id,
                    text=text,
                    category=EvidenceCategory.BENCHMARK,
                    confidence=0.75,
                    evidence_span_ids=[span_id],
                )
            )
        return claims, spans

    async def _chunks(self, artifact_id: str, snapshots: list[Snapshot]) -> list[Chunk]:
        chunks: list[Chunk] = []
        raw: list[str] = []
        metadata: list[tuple[Snapshot, int]] = []
        for snapshot in snapshots:
            for ordinal, text in enumerate(_chunk_text(snapshot.text, max_chars=2000)):
                raw.append(text)
                metadata.append((snapshot, ordinal))
        embeddings = await self.embedding.embed_documents(raw) if raw else []
        for text, embedding, (snapshot, ordinal) in zip(raw, embeddings, metadata, strict=True):
            source_hash = content_hash(text)
            chunks.append(
                Chunk(
                    id=stable_id("chunk", f"{snapshot.id}:{ordinal}:{source_hash}"),
                    artifact_id=artifact_id,
                    snapshot_id=snapshot.id,
                    ordinal=ordinal,
                    text=text,
                    locator=f"source:{snapshot.source_url}#part-{ordinal + 1}",
                    embedding=embedding,
                    embedding_provider=self.embedding.provider_id,
                    embedding_model=self.embedding.model_id,
                    embedding_revision=self.embedding.model_revision,
                    embedding_dimension=self.embedding.dimension,
                    embedding_task_mode=self.embedding.document_task_mode,
                    embedding_normalized=self.embedding.normalized,
                    source_content_hash=source_hash,
                )
            )
        return chunks

    @staticmethod
    def _trust_lane(artifact_type: ArtifactType, source_kind: SourceKind) -> TrustLane:
        if source_kind in {SourceKind.X, SourceKind.LINKEDIN}:
            return TrustLane.EXPERIMENTAL
        if artifact_type in {ArtifactType.PAPER, ArtifactType.OPEN_SOURCE_TOOL, ArtifactType.REPOSITORY}:
            return TrustLane.PROMISING
        return TrustLane.EXPERIMENTAL

    @staticmethod
    def _evidence_quality(payload: KnowledgeExtraction, source_kind: SourceKind) -> float:
        if not payload.claims:
            return 0.2
        average = sum(claim.confidence for claim in payload.claims) / len(payload.claims)
        if source_kind in {SourceKind.X, SourceKind.LINKEDIN}:
            return min(average, 0.4)
        return min(average, 0.9)


def default_cache(data_directory: Path) -> ExtractionCache:
    return ExtractionCache(data_directory / "extraction-cache")


def extraction_fingerprint(record: ArtifactRecord) -> str:
    payload = record.model_dump(mode="json", exclude={"artifact": {"captured_at", "discovered_at"}})
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
