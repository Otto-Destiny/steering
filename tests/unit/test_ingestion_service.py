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
from steering.ingestion.browser import BrowserAuthenticationRequired, BrowserCaptureUnavailable
from steering.ingestion.resolvers import ResolverRegistry
from steering.ingestion.service import IngestionService, ThreadPolicy, _source_priority
from steering.ingestion.x import (
    AUTHOR_THREAD,
    BROWSER_METHOD,
    OEMBED_METHOD,
    ROOT_POST_ONLY,
    SYNDICATION_METHOD,
)


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
    provider_id = "test"
    model_id = "fixture-embedding"
    model_revision = "fixture-v1"
    dimension = 1
    document_task_mode = "test-document"
    query_task_mode = "test-query"
    normalized = True

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [1.0]

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return await self.embed_documents(texts)

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
async def test_ingestion_follows_every_primary_source_with_one_llm_call() -> None:
    """A thread that cites a paper *and* its repository should keep both."""

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
    index_changes: list[None] = []
    service = IngestionService(
        registry=ResolverRegistry([resolver]),
        extraction=ExtractionService(generation=generation, embedding=FixtureEmbedding()),
        repository=repository,
        on_record_changed=lambda: index_changes.append(None),
    )

    record = await service.add(social_url)

    # Both linked sources are read, strongest first, on a single generation call.
    assert generation.calls == 1
    assert resolver.resolved_urls == [social_url, paper_url, github_url]
    # One ingestion produces exactly one artifact; a followed source is evidence
    # inside it, not a second record duplicating the same text.
    assert len(repository.records) == 1
    assert len(record.snapshots) == 3
    assert {snapshot.source_url for snapshot in record.snapshots} == {
        social_url,
        paper_url,
        github_url,
    }
    assert record.artifact.metadata["supporting_sources"] == [paper_url, github_url]
    assert any(snapshot.text == quote for snapshot in record.snapshots)
    assert [job.status for job in repository.jobs] == [JobStatus.RUNNING, JobStatus.COMPLETED]
    assert len(index_changes) == 1


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
    # The followed paper is evidence inside this record, not a second artifact.
    assert linked_paper in {snapshot.source_url for snapshot in first.snapshots}
    assert len(repository.records) == 1
    assert [job.status for job in repository.jobs] == [
        JobStatus.RUNNING,
        JobStatus.COMPLETED,
        JobStatus.RUNNING,
        JobStatus.COMPLETED,
    ]


@pytest.mark.asyncio
async def test_signed_in_x_capture_upgrades_an_existing_public_root_capture_once() -> None:
    captured_url = "https://x.com/researcher/status/9"
    quote = "The self-reply links the complete implementation and evaluation."
    repository = MemoryRepository()
    artifact_id = stable_id("art", captured_url)
    repository.records[artifact_id] = ArtifactRecord(
        artifact=Artifact(
            id=artifact_id,
            canonical_url=captured_url,
            source_kind=SourceKind.X,
            artifact_type=ArtifactType.SOCIAL_POST,
            title="Public root post",
            summary="Only the root post was captured.",
            content_hash="0" * 64,
            metadata={"resolver": "x_public_syndication"},
        )
    )
    captured = ResolvedSource(
        canonical_url=captured_url,
        source_kind=SourceKind.X,
        title="Signed-in thread",
        text=f"Root post.\n\n---\n\n{quote}",
        extraction_method="authorized_visible_browser",
        # Real thread captures record their scope; the upgrade is keyed on what
        # the capture reached, not on which reader produced it.
        metadata={"capture_scope": "author_thread", "bundled_self_replies": 1},
    )
    generation = CountingGeneration(
        KnowledgeExtraction(
            artifact_type=ArtifactType.SOCIAL_POST,
            title="Complete signed-in thread",
            summary="The root post and its self-reply were captured together.",
            claims=[
                ExtractedClaim(
                    text="The thread links the implementation and evaluation.",
                    category=EvidenceCategory.SOCIAL_CLAIM,
                    confidence=0.7,
                    exact_quote=quote,
                    source_index=0,
                )
            ],
        ),
        expected_urls=[captured_url],
    )
    service = IngestionService(
        registry=ResolverRegistry([]),
        extraction=ExtractionService(generation=generation, embedding=FixtureEmbedding()),
        repository=repository,
    )

    upgraded = await service.add_resolved(captured)
    duplicate = await service.add_resolved(captured)

    assert generation.calls == 1
    assert upgraded == duplicate
    assert upgraded.artifact.metadata["resolver"] == "authorized_visible_browser"
    assert upgraded.snapshots[0].text == captured.text


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


class RecordingThreadCapture:
    """Stands in for the signed-in browser, which is the only reply-aware path.

    Results are keyed by URL so a run can mix captures that succeed with ones
    that do not, the way an overnight batch over real links actually behaves.
    """

    def __init__(
        self,
        results: Mapping[str, ResolvedSource | Exception] | ResolvedSource | Exception,
        *,
        default: Exception | None = None,
    ) -> None:
        self.results: Mapping[str, ResolvedSource | Exception] = (
            results if isinstance(results, Mapping) else {}
        )
        self.fallback: ResolvedSource | Exception | None = None if isinstance(results, Mapping) else results
        self.default = default or RuntimeError("no capture configured for this URL")
        self.calls: list[str] = []

    async def capture(self, url: str, *, authorized: bool = False) -> ResolvedSource:
        assert authorized is True
        self.calls.append(url)
        result = self.results.get(url, self.fallback if self.fallback is not None else self.default)
        if isinstance(result, Exception):
            raise result
        return result


def _thread_capture_fixture(post_url: str, paper_url: str) -> ResolvedSource:
    """A thread whose paper link lives only in a self-reply, not the root post."""

    return ResolvedSource(
        canonical_url=post_url,
        source_kind=SourceKind.X,
        title="Researcher — X thread",
        text="Root claim with no link.\n\n---\n\nPaper is below.",
        extraction_method="authorized_visible_browser",
        outbound_urls=[paper_url],
        metadata={"capture_scope": "author_thread", "bundled_self_replies": 1},
    )


def _paper_fixture(paper_url: str, quote: str) -> ResolvedSource:
    return ResolvedSource(
        canonical_url=paper_url,
        source_kind=SourceKind.PDF,
        title="Primary paper",
        text=quote,
        extraction_method="fixture",
        mime_type="application/pdf",
    )


def _extraction_for(quote: str, urls: list[str]) -> ExtractionService:
    generation = CountingGeneration(
        KnowledgeExtraction(
            artifact_type=ArtifactType.TECHNIQUE,
            title="Technique from a thread",
            summary="A paper reached through a link the author kept out of the root post.",
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
        expected_urls=urls,
    )
    return ExtractionService(generation=generation, embedding=FixtureEmbedding())


@pytest.mark.asyncio
async def test_bundling_threads_reaches_a_link_the_author_kept_out_of_the_root_post() -> None:
    post_url = "https://x.com/researcher/status/500"
    paper_url = "https://arxiv.org/abs/2501.99999"
    quote = "The paper reports a bounded evaluation result."
    capture = RecordingThreadCapture(_thread_capture_fixture(post_url, paper_url))
    resolver = MappingResolver({paper_url: _paper_fixture(paper_url, quote)})
    service = IngestionService(
        registry=ResolverRegistry([resolver]),
        extraction=_extraction_for(quote, [post_url, paper_url]),
        repository=MemoryRepository(),
        thread_capture=capture,
    )

    record = await service.add(post_url, threads=ThreadPolicy.ALWAYS)

    assert capture.calls == [post_url]
    # The public path never sees this link; only the thread capture surfaces it.
    assert resolver.resolved_urls == [paper_url]
    assert paper_url in {snapshot.source_url for snapshot in record.snapshots}


@pytest.mark.asyncio
async def test_public_capture_is_used_when_thread_bundling_is_not_requested() -> None:
    post_url = "https://x.com/researcher/status/501"
    public = ResolvedSource(
        canonical_url=post_url,
        source_kind=SourceKind.X,
        title="Root only",
        text="Root claim with no link.",
        extraction_method="x_public_syndication",
    )
    capture = RecordingThreadCapture(public)
    resolver = MappingResolver({post_url: public})
    service = IngestionService(
        registry=ResolverRegistry([resolver]),
        extraction=ExtractionService(
            generation=CountingGeneration(
                KnowledgeExtraction(
                    artifact_type=ArtifactType.SOCIAL_POST,
                    title="Root only",
                    summary="Only the root post was captured.",
                ),
                expected_urls=[post_url],
            ),
            embedding=FixtureEmbedding(),
        ),
        repository=MemoryRepository(),  # type: ignore[arg-type]
        thread_capture=capture,
    )

    await service.add(post_url)

    assert capture.calls == []
    assert resolver.resolved_urls == [post_url]


@pytest.mark.asyncio
async def test_a_missing_sign_in_stops_the_batch_instead_of_degrading_every_item() -> None:
    """Silently storing a run's worth of root-post-only captures would hide the problem."""

    posts = [f"https://x.com/researcher/status/{index}" for index in (600, 601, 602)]
    capture = RecordingThreadCapture(BrowserAuthenticationRequired("not signed in to X"))
    service = IngestionService(
        registry=ResolverRegistry([MappingResolver({})]),
        extraction=ExtractionService(
            generation=CountingGeneration(
                KnowledgeExtraction(artifact_type=ArtifactType.NOTE, title="unused", summary="unused"),
                expected_urls=[],
            ),
            embedding=FixtureEmbedding(),
        ),
        repository=MemoryRepository(),
        thread_capture=capture,
    )

    report = await service.add_batch_report(posts, threads=ThreadPolicy.ALWAYS)

    assert report.captured == 0
    assert report.aborted is not None
    assert "not signed in" in report.aborted
    # It stopped at the first item rather than working through the whole list.
    assert capture.calls == [posts[0]]


@pytest.mark.asyncio
async def test_one_bad_source_does_not_discard_the_rest_of_an_unattended_run() -> None:
    post_url = "https://x.com/researcher/status/700"
    dead_url = "https://x.com/researcher/status/999"
    paper_url = "https://arxiv.org/abs/2501.88888"
    quote = "The paper reports a bounded evaluation result."
    capture = RecordingThreadCapture(
        {
            post_url: _thread_capture_fixture(post_url, paper_url),
            dead_url: BrowserCaptureUnavailable("the root X post was not captured"),
        }
    )
    service = IngestionService(
        registry=ResolverRegistry([MappingResolver({paper_url: _paper_fixture(paper_url, quote)})]),
        extraction=_extraction_for(quote, [post_url, paper_url]),
        repository=MemoryRepository(),
        thread_capture=capture,
    )

    report = await service.add_batch_report([dead_url, post_url], threads=ThreadPolicy.ALWAYS)

    # The dead link is recorded and the run carries on to the good one.
    assert report.captured == 1
    assert report.aborted is None
    assert [source for source, _code in report.failures] == [dead_url]
    assert paper_url in {s.source_url for s in report.records[0].snapshots}


def _public_root_capture(post_url: str, *, reply_count: int) -> ResolvedSource:
    """What public capture returns: the root post, plus how many replies exist."""

    return ResolvedSource(
        canonical_url=post_url,
        source_kind=SourceKind.X,
        title="Root post",
        text="Root claim. Paper link is in the replies.",
        extraction_method="x_public_syndication",
        metadata={"capture_scope": "root_post_only", "reply_count": reply_count},
    )


def _signed_in_session(available: bool) -> Any:
    def has_stored_session() -> bool:
        return available

    return has_stored_session


@pytest.mark.asyncio
async def test_a_post_with_replies_escalates_to_thread_capture_automatically() -> None:
    """Pasting the root link is enough to reach a paper buried in a self-reply."""

    post_url = "https://x.com/researcher/status/800"
    paper_url = "https://arxiv.org/abs/2501.77777"
    quote = "The paper reports a bounded evaluation result."
    capture = RecordingThreadCapture({post_url: _thread_capture_fixture(post_url, paper_url)})
    capture.has_stored_session = _signed_in_session(True)  # type: ignore[attr-defined]
    resolver = MappingResolver(
        {
            post_url: _public_root_capture(post_url, reply_count=6),
            paper_url: _paper_fixture(paper_url, quote),
        }
    )
    service = IngestionService(
        registry=ResolverRegistry([resolver]),
        extraction=_extraction_for(quote, [post_url, paper_url]),
        repository=MemoryRepository(),
        thread_capture=capture,
    )

    record = await service.add(post_url)

    assert capture.calls == [post_url]
    assert paper_url in {snapshot.source_url for snapshot in record.snapshots}


@pytest.mark.asyncio
async def test_a_post_with_no_replies_never_opens_the_browser() -> None:
    post_url = "https://x.com/researcher/status/801"
    public = _public_root_capture(post_url, reply_count=0)
    capture = RecordingThreadCapture({})
    capture.has_stored_session = _signed_in_session(True)  # type: ignore[attr-defined]
    service = IngestionService(
        registry=ResolverRegistry([MappingResolver({post_url: public})]),
        extraction=RecordingExtraction(),  # type: ignore[arg-type]
        repository=MemoryRepository(),  # type: ignore[arg-type]
        thread_capture=capture,
    )

    await service.add(post_url)

    assert capture.calls == []


@pytest.mark.asyncio
async def test_escalation_is_skipped_when_no_sign_in_session_exists() -> None:
    """A user who never signed in must not pay a browser launch per post."""

    post_url = "https://x.com/researcher/status/802"
    capture = RecordingThreadCapture({})
    capture.has_stored_session = _signed_in_session(False)  # type: ignore[attr-defined]
    service = IngestionService(
        registry=ResolverRegistry(
            [MappingResolver({post_url: _public_root_capture(post_url, reply_count=9)})]
        ),
        extraction=RecordingExtraction(),  # type: ignore[arg-type]
        repository=MemoryRepository(),  # type: ignore[arg-type]
        thread_capture=capture,
    )

    await service.add(post_url)

    assert capture.calls == []


@pytest.mark.asyncio
async def test_a_failed_escalation_keeps_the_public_capture_and_records_why() -> None:
    """Degrading is acceptable; degrading invisibly is not."""

    post_url = "https://x.com/researcher/status/803"
    capture = RecordingThreadCapture({post_url: BrowserCaptureUnavailable("window closed")})
    capture.has_stored_session = _signed_in_session(True)  # type: ignore[attr-defined]
    extraction = RecordingExtraction()
    service = IngestionService(
        registry=ResolverRegistry(
            [MappingResolver({post_url: _public_root_capture(post_url, reply_count=3)})]
        ),
        extraction=extraction,  # type: ignore[arg-type]
        repository=MemoryRepository(),  # type: ignore[arg-type]
        thread_capture=capture,
    )

    await service.add(post_url)

    assert capture.calls == [post_url]
    assert extraction.primary is not None
    assert extraction.primary.extraction_method == "x_public_syndication"
    assert extraction.primary.metadata["thread_escalation"] == "failed_BrowserCaptureUnavailable"


@pytest.mark.asyncio
async def test_a_missing_session_under_auto_does_not_abort_the_run() -> None:
    """AUTO degrades; only an explicit thread policy is allowed to stop a batch."""

    post_url = "https://x.com/researcher/status/804"
    capture = RecordingThreadCapture({post_url: BrowserAuthenticationRequired("not signed in")})
    capture.has_stored_session = _signed_in_session(True)  # type: ignore[attr-defined]
    service = IngestionService(
        registry=ResolverRegistry(
            [MappingResolver({post_url: _public_root_capture(post_url, reply_count=2)})]
        ),
        extraction=RecordingExtraction(),  # type: ignore[arg-type]
        repository=MemoryRepository(),  # type: ignore[arg-type]
        thread_capture=capture,
    )

    report = await service.add_batch_report([post_url])

    assert report.aborted is None
    assert report.captured == 1


@pytest.mark.asyncio
async def test_the_never_policy_keeps_every_capture_public() -> None:
    post_url = "https://x.com/researcher/status/805"
    capture = RecordingThreadCapture({})
    capture.has_stored_session = _signed_in_session(True)  # type: ignore[attr-defined]
    service = IngestionService(
        registry=ResolverRegistry(
            [MappingResolver({post_url: _public_root_capture(post_url, reply_count=9)})]
        ),
        extraction=RecordingExtraction(),  # type: ignore[arg-type]
        repository=MemoryRepository(),  # type: ignore[arg-type]
        thread_capture=capture,
    )

    await service.add(post_url, threads=ThreadPolicy.NEVER)

    assert capture.calls == []


@pytest.mark.asyncio
async def test_non_x_sources_are_never_escalated() -> None:
    url = "https://example.org/article"
    article = ResolvedSource(
        canonical_url=url,
        source_kind=SourceKind.WEBPAGE,
        title="Article",
        text="An ordinary web article.",
        extraction_method="public_webpage",
    )
    capture = RecordingThreadCapture({})
    capture.has_stored_session = _signed_in_session(True)  # type: ignore[attr-defined]
    service = IngestionService(
        registry=ResolverRegistry([MappingResolver({url: article})]),
        extraction=RecordingExtraction(),  # type: ignore[arg-type]
        repository=MemoryRepository(),  # type: ignore[arg-type]
        thread_capture=capture,
    )

    await service.add(url)

    assert capture.calls == []


class StubReader:
    """A thread reader whose availability and outcome are both controllable."""

    def __init__(self, name: str, *, available: bool, result: Any) -> None:
        self.name = name
        self._available = available
        self._result = result
        self.calls: list[str] = []

    def available(self) -> bool:
        return self._available

    async def read_thread(self, url: str) -> ResolvedSource:
        self.calls.append(url)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def _thread_result(post_url: str, paper_url: str, method: str) -> ResolvedSource:
    return ResolvedSource(
        canonical_url=post_url,
        source_kind=SourceKind.X,
        title="Thread",
        text="Root claim.\n\n---\n\nPaper is below.",
        extraction_method=method,
        outbound_urls=[paper_url],
    )


def _reader_service(post_url: str, paper_url: str, quote: str, readers: list[Any]) -> IngestionService:
    return IngestionService(
        registry=ResolverRegistry(
            [
                MappingResolver(
                    {
                        post_url: _public_root_capture(post_url, reply_count=3),
                        paper_url: _paper_fixture(paper_url, quote),
                    }
                )
            ]
        ),
        extraction=_extraction_for(quote, [post_url, paper_url]),
        repository=MemoryRepository(),
        thread_readers=readers,
    )


@pytest.mark.asyncio
async def test_the_free_browser_is_tried_before_the_billed_api() -> None:
    """The API costs money per read, so it must never pre-empt a working browser."""

    post_url = "https://x.com/researcher/status/900"
    paper_url = "https://arxiv.org/abs/2501.11111"
    quote = "The paper reports a bounded evaluation result."
    browser = StubReader(
        "signed_in_browser",
        available=True,
        result=_thread_result(post_url, paper_url, "authorized_visible_browser"),
    )
    api = StubReader("x_api", available=True, result=_thread_result(post_url, paper_url, "x_api_thread"))
    service = _reader_service(post_url, paper_url, quote, [browser, api])

    record = await service.add(post_url)

    assert browser.calls == [post_url]
    assert api.calls == []
    assert record.artifact.metadata["resolver"] == "authorized_visible_browser"


@pytest.mark.asyncio
async def test_the_api_takes_over_when_the_browser_has_no_session() -> None:
    post_url = "https://x.com/researcher/status/901"
    paper_url = "https://arxiv.org/abs/2501.22222"
    quote = "The paper reports a bounded evaluation result."
    browser = StubReader("signed_in_browser", available=False, result=None)
    api = StubReader("x_api", available=True, result=_thread_result(post_url, paper_url, "x_api_thread"))
    service = _reader_service(post_url, paper_url, quote, [browser, api])

    record = await service.add(post_url)

    assert browser.calls == []
    assert api.calls == [post_url]
    assert record.artifact.metadata["resolver"] == "x_api_thread"


@pytest.mark.asyncio
async def test_a_failing_reader_falls_through_to_the_next() -> None:
    post_url = "https://x.com/researcher/status/902"
    paper_url = "https://arxiv.org/abs/2501.33333"
    quote = "The paper reports a bounded evaluation result."
    browser = StubReader(
        "signed_in_browser", available=True, result=BrowserCaptureUnavailable("window closed")
    )
    api = StubReader("x_api", available=True, result=_thread_result(post_url, paper_url, "x_api_thread"))
    service = _reader_service(post_url, paper_url, quote, [browser, api])

    record = await service.add(post_url)

    assert browser.calls == [post_url]
    assert api.calls == [post_url]
    assert record.artifact.metadata["resolver"] == "x_api_thread"


@pytest.mark.asyncio
async def test_no_available_reader_means_no_escalation_at_all() -> None:
    post_url = "https://x.com/researcher/status/903"
    browser = StubReader("signed_in_browser", available=False, result=None)
    api = StubReader("x_api", available=False, result=None)
    service = IngestionService(
        registry=ResolverRegistry(
            [MappingResolver({post_url: _public_root_capture(post_url, reply_count=5)})]
        ),
        extraction=RecordingExtraction(),  # type: ignore[arg-type]
        repository=MemoryRepository(),  # type: ignore[arg-type]
        thread_readers=[browser, api],
    )

    await service.add(post_url)

    assert browser.calls == []
    assert api.calls == []


# --------------------------------------------------------------------------- #
# Re-capture upgrades
# --------------------------------------------------------------------------- #

POST_URL = "https://x.com/researcher/status/1885026028428681698"
REPO = "https://github.com/example/project"


def stored_capture(**metadata: Any) -> ArtifactRecord:
    return ArtifactRecord(
        artifact=Artifact(
            id="art_stored",
            canonical_url=POST_URL,
            source_kind=SourceKind.X,
            artifact_type=ArtifactType.NOTE,
            title="Stored",
            summary="An earlier capture of the same post.",
            content_hash="stored-hash",
            metadata=metadata,
        )
    )


def recapture(*urls: str, scope: str = ROOT_POST_ONLY) -> ResolvedSource:
    return ResolvedSource(
        canonical_url=POST_URL,
        source_kind=SourceKind.X,
        title="Recapture",
        text="The same post, captured again.",
        extraction_method=SYNDICATION_METHOD,
        outbound_urls=list(urls),
        metadata={"capture_scope": scope},
    )


def test_a_recapture_that_finds_a_new_primary_source_replaces_a_linkless_record() -> None:
    """Records stored before long-form link recovery can never gain a link alone."""

    existing = stored_capture(resolver=OEMBED_METHOD, supporting_sources=[])

    assert IngestionService._upgrades_public_social_capture(existing, recapture(REPO)) is True


def test_a_recapture_that_adds_nothing_leaves_the_stored_record_alone() -> None:
    """Re-submitting a link the record already followed must not rewrite it."""

    existing = stored_capture(resolver=SYNDICATION_METHOD, supporting_sources=[REPO])

    assert IngestionService._upgrades_public_social_capture(existing, recapture(REPO)) is False


def test_a_linkless_recapture_does_not_churn_a_stored_record() -> None:
    existing = stored_capture(resolver=OEMBED_METHOD, supporting_sources=[])

    assert IngestionService._upgrades_public_social_capture(existing, recapture()) is False


def test_a_public_recapture_never_overwrites_a_signed_in_thread_capture() -> None:
    """The signed-in browser reads replies the public payload cannot reach."""

    existing = stored_capture(resolver=BROWSER_METHOD, capture_scope=AUTHOR_THREAD)

    assert IngestionService._upgrades_public_social_capture(existing, recapture(REPO)) is False


def test_a_thread_capture_still_upgrades_a_root_post_record() -> None:
    existing = stored_capture(resolver=SYNDICATION_METHOD, capture_scope=ROOT_POST_ONLY)
    incoming = recapture(scope=AUTHOR_THREAD)

    assert IngestionService._upgrades_public_social_capture(existing, incoming) is True
