from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from pydantic import BaseModel

from steering.domain.models import (
    Artifact,
    ArtifactRecord,
    ArtifactType,
    EvidenceCategory,
    IngestionJob,
    JobStatus,
    ResolvedSource,
    ReviewIssue,
    SourceKind,
)
from steering.extraction.schemas import ExtractedClaim, KnowledgeExtraction
from steering.extraction.service import ExtractionService, stable_id
from steering.ingestion.resolvers import ResolverRegistry
from steering.ingestion.service import IngestionService, _source_priority


class MappingResolver:
    name = "fixture"

    def __init__(self, sources: Mapping[str, ResolvedSource]) -> None:
        self.sources = sources
        self.resolved_urls: list[str] = []

    def can_resolve(self, source: str) -> bool:
        return source in self.sources

    async def resolve(self, source: str) -> ResolvedSource:
        self.resolved_urls.append(source)
        return self.sources[source]


class FailingResolver:
    name = "failing-fixture"

    def __init__(self, accepted: Sequence[str]) -> None:
        self.accepted = set(accepted)
        self.resolved_urls: list[str] = []

    def can_resolve(self, source: str) -> bool:
        return source in self.accepted

    async def resolve(self, source: str) -> ResolvedSource:
        self.resolved_urls.append(source)
        raise ValueError("fixture resolution failure")


class MemoryRepository:
    def __init__(self) -> None:
        self.records: dict[str, ArtifactRecord] = {}
        self.jobs: list[IngestionJob] = []

    def upsert_record(self, record: ArtifactRecord) -> ArtifactRecord:
        self.records[record.artifact.id] = record
        return record

    def get_record(self, artifact_id: str) -> ArtifactRecord | None:
        return self.records.get(artifact_id)

    def get_by_url(self, canonical_url: str) -> ArtifactRecord | None:
        return next(
            (record for record in self.records.values() if record.artifact.canonical_url == canonical_url),
            None,
        )

    def list_records(self) -> list[ArtifactRecord]:
        return list(self.records.values())

    def save_job(self, job: IngestionJob) -> None:
        self.jobs.append(job.model_copy(deep=True))

    def list_jobs(self, limit: int = 100) -> list[IngestionJob]:
        return self.jobs[-limit:]

    def list_issues(self, unresolved_only: bool = False) -> list[ReviewIssue]:
        return []

    def resolve_issue(self, issue_id: str, action: str) -> ReviewIssue:
        raise NotImplementedError

    def save_project(self, project: Any) -> Any:
        raise NotImplementedError

    def save_decision(self, decision: Any) -> Any:
        raise NotImplementedError

    def save_outcome(self, outcome: Any) -> Any:
        raise NotImplementedError

    def project_history(self, project_id: str) -> Mapping[str, Sequence[Any]]:
        return {}

    def backup(self, destination: str) -> str:
        raise NotImplementedError


class CountingGeneration:
    model_id = "fixture-generation"

    def __init__(self, payload: KnowledgeExtraction, expected_urls: Sequence[str]) -> None:
        self.payload = payload
        self.expected_urls = expected_urls
        self.calls = 0

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
    ) -> Any:
        self.calls += 1
        assert all(url in user_prompt for url in self.expected_urls)
        return self.payload

    async def test_connection(self) -> None:
        return None


class FixtureEmbedding:
    model_id = "fixture-embedding"
    dimension = 1

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]

    async def test_connection(self) -> None:
        return None


class FailingGeneration(CountingGeneration):
    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
    ) -> Any:
        self.calls += 1
        raise RuntimeError("fixture provider failure")


class RecordingExtraction:
    def __init__(self) -> None:
        self.primary: ResolvedSource | None = None

    async def extract(
        self,
        primary: ResolvedSource,
        supporting: Sequence[ResolvedSource],
    ) -> ArtifactRecord:
        assert supporting == []
        self.primary = primary
        return ArtifactRecord(
            artifact=Artifact(
                canonical_url=primary.canonical_url,
                source_kind=primary.source_kind,
                title=primary.title,
                summary=primary.text,
                content_hash="0" * 64,
            )
        )


@pytest.mark.asyncio
async def test_bounded_media_fallback_runs_before_the_single_extraction_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = ResolvedSource(
        canonical_url="https://x.com/researcher/status/88",
        source_kind=SourceKind.X,
        title="Bundled thread",
        text="Post text",
        extraction_method="authorized_visible_browser",
    )
    extraction = RecordingExtraction()
    marker_fetcher = object()
    marker_provider = object()

    async def fake_fallback(
        resolved: ResolvedSource,
        *,
        fetcher: object,
        image_provider: object,
        stronger_source_available: bool,
    ) -> ResolvedSource:
        assert fetcher is marker_fetcher
        assert image_provider is marker_provider
        assert stronger_source_available is False
        return resolved.model_copy(update={"text": resolved.text + "\nImage evidence"})

    monkeypatch.setattr("steering.ingestion.service.append_social_image_fallback", fake_fallback)
    service = IngestionService(
        registry=ResolverRegistry([]),
        extraction=extraction,  # type: ignore[arg-type]
        repository=MemoryRepository(),  # type: ignore[arg-type]
        media_fetcher=marker_fetcher,  # type: ignore[arg-type]
        image_provider=marker_provider,  # type: ignore[arg-type]
    )

    await service.add_resolved(source)

    assert extraction.primary is not None
    assert extraction.primary.text.endswith("Image evidence")


@pytest.mark.asyncio
async def test_ingestion_follows_one_highest_priority_primary_source_with_one_llm_call() -> None:
    social_url = "https://x.com/researcher/status/7"
    paper_url = "https://papers.example.org/paper.pdf"
    github_url = "https://github.com/example/repository"
    quote = "The paper reports a bounded evaluation result."
    social = ResolvedSource(
        canonical_url=social_url,
        source_kind=SourceKind.X,
        title="Research thread",
        text="A social summary with two primary-source candidates.",
        extraction_method="fixture",
        outbound_urls=[github_url, paper_url],
    )
    paper = ResolvedSource(
        canonical_url=paper_url,
        source_kind=SourceKind.PDF,
        title="Primary paper",
        text=quote,
        extraction_method="fixture",
        mime_type="application/pdf",
    )
    github = ResolvedSource(
        canonical_url=github_url,
        source_kind=SourceKind.GITHUB,
        title="Repository",
        text="Repository README",
        extraction_method="fixture",
    )
    resolver = MappingResolver({social_url: social, paper_url: paper, github_url: github})
    generation = CountingGeneration(
        KnowledgeExtraction(
            artifact_type=ArtifactType.TECHNIQUE,
            title="Bounded technique",
            summary="A paper-backed technique discovered from a social thread.",
            claims=[
                ExtractedClaim(
                    text="A bounded evaluation result was reported.",
                    category=EvidenceCategory.RESEARCH_PAPER,
                    confidence=0.8,
                    exact_quote=quote,
                    source_index=1,
                )
            ],
        ),
        expected_urls=[social_url, paper_url],
    )
    repository = MemoryRepository()
    service = IngestionService(
        registry=ResolverRegistry([resolver]),
        extraction=ExtractionService(generation=generation, embedding=FixtureEmbedding()),
        repository=repository,
    )

    record = await service.add(social_url)

    assert generation.calls == 1
    assert resolver.resolved_urls == [social_url, paper_url]
    assert len(repository.records) == 2
    assert len(record.snapshots) == 2
    assert record.relations[-1].object_id == stable_id("art", paper_url)
    primary = repository.get_by_url(paper_url)
    assert primary is not None
    assert primary.artifact.artifact_type is ArtifactType.PAPER
    assert primary.artifact.metadata["provenance_only"] is True
    assert primary.snapshots[0].text == quote
    assert repository.jobs[-1].status is JobStatus.COMPLETED
    assert [job.status for job in repository.jobs] == [JobStatus.RUNNING, JobStatus.COMPLETED]


@pytest.mark.asyncio
async def test_add_resolved_social_capture_follows_one_primary_without_duplicate_jobs() -> None:
    captured_url = "https://x.com/researcher/status/8"
    linked_paper = "https://papers.example.org/should-not-fetch.pdf"
    quote = "The authorized browser captured this exact technical claim."
    captured = ResolvedSource(
        canonical_url=captured_url,
        source_kind=SourceKind.X,
        title="Authorized browser capture",
        text=quote,
        extraction_method="authorized_visible_browser",
        outbound_urls=[linked_paper],
    )
    resolver = MappingResolver(
        {
            linked_paper: ResolvedSource(
                canonical_url=linked_paper,
                source_kind=SourceKind.PDF,
                title="Linked paper",
                text="The linked paper provides primary evidence.",
                extraction_method="fixture",
            )
        }
    )
    generation = CountingGeneration(
        KnowledgeExtraction(
            artifact_type=ArtifactType.SOCIAL_POST,
            title="Captured post",
            summary="An authorized capture routed through evidence-bound extraction.",
            claims=[
                ExtractedClaim(
                    text="The capture contains a technical claim.",
                    category=EvidenceCategory.SOCIAL_CLAIM,
                    confidence=0.6,
                    exact_quote=quote,
                    source_index=0,
                )
            ],
        ),
        expected_urls=[captured_url, linked_paper],
    )
    repository = MemoryRepository()
    service = IngestionService(
        registry=ResolverRegistry([resolver]),
        extraction=ExtractionService(generation=generation, embedding=FixtureEmbedding()),
        repository=repository,
    )

    first = await service.add_resolved(captured)
    second = await service.add_resolved(captured)

    assert first == second
    assert generation.calls == 1
    assert resolver.resolved_urls == [linked_paper]
    assert len(first.snapshots) == 2
    assert repository.get_by_url(linked_paper) is not None
    assert [job.status for job in repository.jobs] == [
        JobStatus.RUNNING,
        JobStatus.COMPLETED,
        JobStatus.RUNNING,
        JobStatus.COMPLETED,
    ]


@pytest.mark.parametrize(
    ("url", "priority"),
    [
        ("https://arxiv.org/abs/1", 100),
        ("https://papers.example.org/file.PDF?download=1", 98),
        ("https://github.com/example/repo", 95),
        ("https://huggingface.co/example/model", 90),
        ("https://modelscope.cn/models/example/model", 90),
        ("https://docs.example.org/tool", 80),
        ("https://example.org/blog", 20),
    ],
)
def test_primary_source_priority_is_explicit_and_token_bounded(url: str, priority: int) -> None:
    assert _source_priority(url) == priority


@pytest.mark.asyncio
async def test_job_failures_are_safely_recorded_for_resolution_and_extraction() -> None:
    source_url = "https://example.org/failure"
    repository = MemoryRepository()
    failing_resolver = FailingResolver([source_url])
    unused_generation = CountingGeneration(
        KnowledgeExtraction(
            artifact_type=ArtifactType.WEBPAGE,
            title="Unused",
            summary="Unused extraction.",
        ),
        expected_urls=[],
    )
    service = IngestionService(
        registry=ResolverRegistry([failing_resolver]),
        extraction=ExtractionService(generation=unused_generation, embedding=FixtureEmbedding()),
        repository=repository,
    )

    with pytest.raises(ValueError, match="fixture resolution failure"):
        await service.add(source_url)
    assert repository.jobs[-1].status is JobStatus.FAILED
    assert repository.jobs[-1].error_code == "ValueError"
    assert source_url not in (repository.jobs[-1].safe_error or "")

    captured = ResolvedSource(
        canonical_url="text://captured-failure",
        source_kind=SourceKind.TEXT,
        title="Captured input",
        text="Captured text",
        extraction_method="fixture",
    )
    failing_generation = FailingGeneration(
        KnowledgeExtraction(
            artifact_type=ArtifactType.NOTE,
            title="Unused",
            summary="Unused extraction.",
        ),
        expected_urls=[],
    )
    service = IngestionService(
        registry=ResolverRegistry([]),
        extraction=ExtractionService(generation=failing_generation, embedding=FixtureEmbedding()),
        repository=repository,
    )
    with pytest.raises(RuntimeError, match="provider failure"):
        await service.add_resolved(captured)
    assert repository.jobs[-1].status is JobStatus.FAILED
    assert repository.jobs[-1].error_code == "RuntimeError"


@pytest.mark.asyncio
async def test_primary_fallback_skips_failed_and_low_value_links() -> None:
    failed_paper = "https://arxiv.org/abs/fails"
    low_value = "https://example.org/general-post"
    social = ResolvedSource(
        canonical_url="https://x.com/researcher/status/10",
        source_kind=SourceKind.X,
        title="Fallback thread",
        text="A thread whose primary source is unavailable.",
        extraction_method="fixture",
        outbound_urls=[low_value, failed_paper],
    )
    mapping = MappingResolver({social.canonical_url: social})
    failing = FailingResolver([failed_paper])
    generation = CountingGeneration(
        KnowledgeExtraction(
            artifact_type=ArtifactType.SOCIAL_POST,
            title="Fallback thread",
            summary="A social claim retained without an available primary source.",
        ),
        expected_urls=[social.canonical_url],
    )
    repository = MemoryRepository()
    service = IngestionService(
        registry=ResolverRegistry([mapping, failing]),
        extraction=ExtractionService(generation=generation, embedding=FixtureEmbedding()),
        repository=repository,
    )

    record = await service.add(social.canonical_url)
    assert failing.resolved_urls == [failed_paper]
    assert mapping.resolved_urls == [social.canonical_url]
    assert record.relations == []


@pytest.mark.asyncio
async def test_batch_deduplicates_inputs_and_completes_each_job() -> None:
    first_url = "text://batch-one"
    second_url = "text://batch-two"
    first = ResolvedSource(
        canonical_url=first_url,
        source_kind=SourceKind.TEXT,
        title="First",
        text="First batch note.",
        extraction_method="fixture",
    )
    second = ResolvedSource(
        canonical_url=second_url,
        source_kind=SourceKind.TEXT,
        title="Second",
        text="Second batch note.",
        extraction_method="fixture",
    )
    resolver = MappingResolver({first_url: first, second_url: second})
    generation = CountingGeneration(
        KnowledgeExtraction(
            artifact_type=ArtifactType.NOTE,
            title="Batch note",
            summary="A batch-ingested note.",
        ),
        expected_urls=[],
    )
    repository = MemoryRepository()
    service = IngestionService(
        registry=ResolverRegistry([resolver]),
        extraction=ExtractionService(generation=generation, embedding=FixtureEmbedding()),
        repository=repository,
    )

    records = await service.add_batch([f" {first_url} ", first_url, "", second_url])
    assert len(records) == 2
    assert resolver.resolved_urls == [first_url, second_url]
    assert repository.jobs[-1].status is JobStatus.COMPLETED
