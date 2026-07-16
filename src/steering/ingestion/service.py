from __future__ import annotations

from collections.abc import Callable, Sequence
from urllib.parse import urlsplit

from steering.domain.credentials import sanitized_persistence_source
from steering.domain.models import (
    Artifact,
    ArtifactRecord,
    ArtifactType,
    IngestionJob,
    JobStatus,
    Relation,
    RelationType,
    ResolvedSource,
    SourceKind,
    TrustLane,
)
from steering.domain.protocols import ArtifactRepository, ImageUnderstandingProvider
from steering.extraction.service import ExtractionService, stable_id
from steering.ingestion.media_fallback import append_social_image_fallback
from steering.ingestion.resolvers import ResolverRegistry
from steering.ingestion.security import SafeFetcher


def _source_priority(url: str) -> int:
    lowered = url.lower()
    if any(token in lowered for token in ("arxiv.org", "openreview.net", "aclanthology.org", "doi.org")):
        return 100
    if "github.com" in lowered:
        return 95
    if "huggingface.co" in lowered or "modelscope" in lowered:
        return 90
    if urlsplit(lowered).path.endswith(".pdf"):
        return 98
    if "docs." in lowered or "readthedocs" in lowered:
        return 80
    return 20


class IngestionService:
    def __init__(
        self,
        *,
        registry: ResolverRegistry,
        extraction: ExtractionService,
        repository: ArtifactRepository,
        media_fetcher: SafeFetcher | None = None,
        image_provider: ImageUnderstandingProvider | None = None,
        on_record_changed: Callable[[], None] | None = None,
    ) -> None:
        self.registry = registry
        self.extraction = extraction
        self.repository = repository
        self.media_fetcher = media_fetcher
        self.image_provider = image_provider
        self.on_record_changed = on_record_changed

    def _store_record(self, record: ArtifactRecord) -> ArtifactRecord:
        stored = self.repository.upsert_record(record)
        if self.on_record_changed is not None:
            self.on_record_changed()
        return stored

    async def add(self, source: str) -> ArtifactRecord:
        job = IngestionJob(source=sanitized_persistence_source(source), status=JobStatus.RUNNING)
        self.repository.save_job(job)
        try:
            resolved = await self.registry.resolve(source)
            return await self._ingest_resolved(resolved, job=job, follow_primary=True)
        except Exception as exc:
            self._fail_job(job, exc)
            raise

    async def add_resolved(self, resolved: ResolvedSource) -> ArtifactRecord:
        """Ingest an already authorized capture without performing any network resolution."""

        job = IngestionJob(
            source=sanitized_persistence_source(resolved.canonical_url),
            status=JobStatus.RUNNING,
        )
        self.repository.save_job(job)
        try:
            follow_primary = resolved.source_kind in {SourceKind.X, SourceKind.LINKEDIN}
            return await self._ingest_resolved(resolved, job=job, follow_primary=follow_primary)
        except Exception as exc:
            self._fail_job(job, exc)
            raise

    async def add_batch(self, sources: Sequence[str]) -> list[ArtifactRecord]:
        results: list[ArtifactRecord] = []
        for source in dict.fromkeys(item.strip() for item in sources if item.strip()):
            results.append(await self.add(source))
        return results

    async def _ingest_resolved(
        self,
        resolved: ResolvedSource,
        *,
        job: IngestionJob,
        follow_primary: bool,
    ) -> ArtifactRecord:
        existing = self.repository.get_by_url(resolved.canonical_url)
        if existing is not None:
            job.status = JobStatus.COMPLETED
            job.artifact_id = existing.artifact.id
            self.repository.save_job(job)
            return existing
        supporting = await self._follow_primary_source(resolved) if follow_primary else None
        if self.media_fetcher is not None:
            resolved = await append_social_image_fallback(
                resolved,
                fetcher=self.media_fetcher,
                image_provider=self.image_provider,
                stronger_source_available=supporting is not None,
            )
        record = await self.extraction.extract(resolved, [supporting] if supporting else [])
        if supporting is not None:
            primary = self._primary_source_record(record, supporting)
            existing_primary = self.repository.get_by_url(supporting.canonical_url)
            primary_artifact_id = (
                existing_primary.artifact.id
                if existing_primary is not None
                else self._store_record(primary).artifact.id
            )
            record.relations.append(
                Relation(
                    id=stable_id(
                        "rel",
                        f"{record.artifact.id}:{RelationType.DERIVED_FROM}:{primary_artifact_id}",
                    ),
                    subject_id=record.artifact.id,
                    predicate=RelationType.DERIVED_FROM,
                    object_id=primary_artifact_id,
                    approved=False,
                    rationale=(
                        "Direct primary source linked from the captured social artifact and retained "
                        "as a separate provenance record without another generation call."
                    ),
                )
            )
        stored = self._store_record(record)
        job.status = JobStatus.NEEDS_REVIEW if stored.issues else JobStatus.COMPLETED
        job.artifact_id = stored.artifact.id
        self.repository.save_job(job)
        return stored

    @staticmethod
    def _primary_source_record(
        combined_record: ArtifactRecord,
        source: ResolvedSource,
    ) -> ArtifactRecord:
        artifact_id = stable_id("art", source.canonical_url)
        source_snapshot = next(
            snapshot for snapshot in combined_record.snapshots if snapshot.source_url == source.canonical_url
        )
        snapshot_id = stable_id("snap", f"{artifact_id}:{source_snapshot.content_hash}")
        snapshot = source_snapshot.model_copy(update={"id": snapshot_id, "artifact_id": artifact_id})
        chunks = [
            chunk.model_copy(
                update={
                    "id": stable_id("chunk", f"{snapshot_id}:{chunk.ordinal}:{chunk.text}"),
                    "artifact_id": artifact_id,
                    "snapshot_id": snapshot_id,
                }
            )
            for chunk in combined_record.chunks
            if chunk.snapshot_id == source_snapshot.id
        ]
        source_types = {
            SourceKind.GITHUB: ArtifactType.REPOSITORY,
            SourceKind.PAPER: ArtifactType.PAPER,
            SourceKind.PDF: ArtifactType.PAPER,
            SourceKind.DOCUMENTATION: ArtifactType.DOCUMENTATION,
            SourceKind.WEBPAGE: ArtifactType.WEBPAGE,
        }
        summary = " ".join(source.text.split())[:500]
        artifact = Artifact(
            id=artifact_id,
            canonical_url=source.canonical_url,
            source_kind=source.source_kind,
            artifact_type=source_types.get(source.source_kind, ArtifactType.UNKNOWN),
            title=source.title,
            summary=summary or "Linked primary source retained for provenance.",
            trust_lane=TrustLane.PROMISING,
            evidence_quality=0.6,
            published_at=source.published_at,
            content_hash=source_snapshot.content_hash,
            metadata={
                "provenance_only": True,
                "linked_social_artifact_id": combined_record.artifact.id,
                "partial": source.partial,
            },
        )
        return ArtifactRecord(artifact=artifact, snapshots=[snapshot], chunks=chunks)

    def _fail_job(self, job: IngestionJob, exc: Exception) -> None:
        job.status = JobStatus.FAILED
        job.error_code = type(exc).__name__
        job.safe_error = "Ingestion failed; inspect the source, provider, and network settings."
        self.repository.save_job(job)

    async def _follow_primary_source(self, source: ResolvedSource) -> ResolvedSource | None:
        if source.source_kind not in {SourceKind.X, SourceKind.LINKEDIN}:
            return None
        candidates = sorted(source.outbound_urls, key=_source_priority, reverse=True)
        for url in candidates:
            if _source_priority(url) < 80:
                continue
            try:
                return await self.registry.resolve(url)
            except (RuntimeError, ValueError):
                continue
        return None
