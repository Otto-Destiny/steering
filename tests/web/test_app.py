from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from starlette.testclient import TestClient

from steering.database.repository import CorruptRecordError
from steering.domain.models import (
    ArchitectureReview,
    Artifact,
    ArtifactRecord,
    ArtifactType,
    Claim,
    Decision,
    EvidenceCategory,
    EvidenceSpan,
    ExperimentOutcome,
    IdeaCard,
    IngestionJob,
    IssueStatus,
    JobStatus,
    Project,
    ResolvedSource,
    ReviewIssue,
    ScoreBreakdown,
    SearchHit,
    Snapshot,
    SourceKind,
)
from steering.extraction.service import EvidenceValidationError
from steering.ingestion.browser import BrowserDependencyUnavailable, LoginOutcome
from steering.ingestion.login_session import BrowserLoginSession
from steering.ingestion.security import SourceUnavailableError
from steering.ingestion.service import BatchReport, ThreadPolicy
from steering.providers.openai_compatible import ProviderTimeoutError
from steering.web import ProviderView, create_web_app


def make_record(artifact_id: str = "art_memory") -> ArtifactRecord:
    social_quote = "The social post says the method is perfect."
    primary_quote = "The paper reports measured limitations."
    return ArtifactRecord(
        artifact=Artifact(
            id=artifact_id,
            canonical_url="https://example.com/memory",
            source_kind=SourceKind.PAPER,
            artifact_type=ArtifactType.PAPER,
            title="Compact Agent Memory",
            summary="A source-backed memory compression approach.",
            strategy_family="memory_compression",
            content_hash="record-hash",
            capabilities=["Reduces repeated context"],
            limitations=["Requires workload validation"],
            requirements=["A representative recall set"],
        ),
        snapshots=[
            Snapshot(
                id="snap_social",
                artifact_id=artifact_id,
                source_url="https://example.com/post",
                content_hash="social-hash",
                mime_type="text/html",
                text=social_quote,
                extraction_method="fixture",
            ),
            Snapshot(
                id="snap_primary",
                artifact_id=artifact_id,
                source_url="https://example.com/memory",
                content_hash="primary-hash",
                mime_type="text/html",
                text=primary_quote,
                extraction_method="fixture",
            ),
        ],
        claims=[
            Claim(
                id="claim_social",
                artifact_id=artifact_id,
                text="The method is perfect.",
                category=EvidenceCategory.SOCIAL_CLAIM,
                confidence=0.5,
                evidence_span_ids=["span_social"],
            ),
            Claim(
                id="claim_primary",
                artifact_id=artifact_id,
                text="The method has measured limitations.",
                category=EvidenceCategory.RESEARCH_PAPER,
                confidence=0.8,
                evidence_span_ids=["span_primary"],
            ),
        ],
        evidence_spans=[
            EvidenceSpan(
                id="span_social",
                snapshot_id="snap_social",
                claim_id="claim_social",
                quote=social_quote,
                start=0,
                end=len(social_quote),
                locator="social evidence",
            ),
            EvidenceSpan(
                id="span_primary",
                snapshot_id="snap_primary",
                claim_id="claim_primary",
                quote=primary_quote,
                start=0,
                end=len(primary_quote),
                locator="primary evidence",
            ),
        ],
    )


def make_issue() -> ReviewIssue:
    return ReviewIssue(
        id="issue_one",
        artifact_id="art_memory",
        social_statement="The method is perfect.",
        source_statement="The method has measured limitations.",
        explanation="The social post overstates the paper.",
        social_source_url="https://example.com/post",
        primary_source_url="https://example.com/memory",
        evidence_span_ids=["span_social", "span_primary"],
    )


class FakeRepository:
    def __init__(self) -> None:
        self.records = [make_record()]
        self.issues = [make_issue()]
        self.unreadable: list[dict[str, str]] = []
        self.jobs = [
            IngestionJob(
                id="job_one",
                source="https://example.com/memory",
                status=JobStatus.COMPLETED,
                artifact_id="art_memory",
            )
        ]
        self.projects: dict[str, Project] = {}
        self.decisions: list[Decision] = []
        self.outcomes: list[ExperimentOutcome] = []

    def list_records(self) -> list[ArtifactRecord]:
        return self.records

    def list_artifacts(self, *, limit: int | None = None) -> list[Artifact]:
        artifacts = sorted(
            (record.artifact for record in self.records),
            key=lambda artifact: artifact.captured_at,
            reverse=True,
        )
        return artifacts if limit is None else artifacts[:limit]

    def count_artifacts(self) -> int:
        return len(self.records)

    def find_artifact_by_url(self, canonical_url: str) -> Artifact | None:
        return next(
            (r.artifact for r in self.records if r.artifact.canonical_url == canonical_url),
            None,
        )

    def delete_record(self, artifact_id: str) -> bool:
        before = len(self.records)
        self.records = [r for r in self.records if r.artifact.id != artifact_id]
        return len(self.records) != before

    def get_record(self, artifact_id: str) -> ArtifactRecord | None:
        return next((record for record in self.records if record.artifact.id == artifact_id), None)

    def unreadable_records(self) -> list[dict[str, str]]:
        return list(self.unreadable)

    def list_jobs(self, limit: int = 100) -> list[IngestionJob]:
        return self.jobs[:limit]

    def list_issues(self, unresolved_only: bool = False) -> list[ReviewIssue]:
        if unresolved_only:
            return [issue for issue in self.issues if issue.status == IssueStatus.UNRESOLVED]
        return self.issues

    def resolve_issue(self, issue_id: str, action: str) -> ReviewIssue:
        issue = next((item for item in self.issues if item.id == issue_id), None)
        if issue is None:
            raise KeyError(issue_id)
        statuses = {
            "accept_correction": IssueStatus.ACCEPTED_CORRECTION,
            "keep_both": IssueStatus.KEPT_BOTH,
            "dismiss": IssueStatus.DISMISSED,
            "reject": IssueStatus.REJECTED,
        }
        status = statuses.get(action)
        if status is None:
            raise ValueError(action)
        resolved = issue.model_copy(update={"status": status})
        self.issues[self.issues.index(issue)] = resolved
        return resolved

    def save_project(self, project: Project) -> Project:
        self.projects[project.id] = project
        return project

    def list_projects(self) -> list[Project]:
        return list(self.projects.values())

    def save_decision(self, decision: Decision) -> Decision:
        self.decisions.append(decision)
        return decision

    def save_outcome(self, outcome: ExperimentOutcome) -> ExperimentOutcome:
        self.outcomes.append(outcome)
        return outcome

    def project_history(self, project_id: str) -> Mapping[str, Sequence[Any]]:
        return {
            "projects": [self.projects[project_id]] if project_id in self.projects else [],
            "decisions": [item for item in self.decisions if item.project_id == project_id],
            "outcomes": [item for item in self.outcomes if item.project_id == project_id],
            "review_runs": [],
        }

    def backup(self, destination: str) -> str:
        return f"{destination}/steering-backup"


class FakeRetriever:
    def __init__(self) -> None:
        self.dirty = False
        self.refreshes = 0

    def mark_dirty(self) -> None:
        self.dirty = True

    async def refresh(self) -> None:
        self.refreshes += 1
        self.dirty = False

    async def reembed_all(self) -> tuple[int, str]:
        self.refreshes += 1
        self.dirty = False
        return 3, "C:/backups/steering-pre-reembed"


class FakeEngine:
    def __init__(self, repository: FakeRepository) -> None:
        self.repository = repository
        self.retriever = FakeRetriever()
        self.queries: list[Any] = []

    async def search(self, query: Any) -> list[SearchHit]:
        self.queries.append(query)
        return [
            SearchHit(
                artifact=self.repository.records[0].artifact,
                score=0.91,
                scores=ScoreBreakdown(rrf=0.91),
                citation_urls=["https://example.com/memory"],
            )
        ]

    def get_knowledge_record(self, artifact_id: str) -> ArtifactRecord | None:
        return self.repository.get_record(artifact_id)

    async def review_architecture(self, architecture: str, *_: Any, **__: Any) -> ArchitectureReview:
        return ArchitectureReview(
            query=architecture,
            decomposed_areas=["memory"],
            observations=["One source-backed option was found."],
            idea_cards=[
                IdeaCard(
                    artifact_id="art_memory",
                    title="Compact Agent Memory",
                    what_it_is="A compact memory strategy.",
                    why_it_fits="It targets repeated context.",
                    trust_lane=self.repository.records[0].artifact.trust_lane,
                    source_url="https://example.com/memory",
                    published_at=None,
                    advantages=["Lower context use"],
                    limitations=["Needs validation"],
                    compatibility_requirements=["Recall benchmark"],
                    minimal_experiment="Compare against raw history.",
                    success_criteria="Recall improves at lower cost.",
                    failure_criteria="Recall regresses.",
                    related_alternatives=[],
                    uncertainty="Author-reported evidence.",
                    strategy_family="memory_compression",
                )
            ],
            citations=["https://example.com/memory"],
        )

    def record_project_decision(
        self,
        *,
        project: str,
        artifact_id: str | None,
        decision: str,
        rationale: str,
    ) -> Decision:
        history = self.repository.project_history(project)
        saved_project = (
            history["projects"][0]
            if history["projects"]
            else self.repository.save_project(Project(name=project))
        )
        return self.repository.save_decision(
            Decision(
                project_id=saved_project.id,
                artifact_id=artifact_id,
                decision=decision,
                rationale=rationale,
            )
        )

    def record_experiment_outcome(self, **values: Any) -> ExperimentOutcome:
        return self.repository.save_outcome(ExperimentOutcome(**values))


class FakeIngestion:
    def __init__(
        self,
        repository: FakeRepository,
        *,
        fail: bool = False,
        error: Exception | None = None,
    ) -> None:
        self.repository = repository
        self.fail = fail
        self.error = error
        self.thread_policies: list[ThreadPolicy] = []
        self.two_pass_requested: list[bool] = []

    async def add(self, source: str, *, threads: ThreadPolicy = ThreadPolicy.AUTO) -> ArtifactRecord:
        self.thread_policies.append(threads)
        if self.error is not None:
            raise self.error
        if self.fail:
            raise RuntimeError("PRIVATE-PROVIDER-KEY")
        record = make_record("art_added")
        record.artifact.summary = f"Captured from {source[:20]}"
        self.repository.records.append(record)
        return record

    async def add_batch(
        self,
        sources: Sequence[str],
        *,
        threads: ThreadPolicy = ThreadPolicy.AUTO,
    ) -> list[ArtifactRecord]:
        return [await self.add(source, threads=threads) for source in sources]

    async def add_batch_report(
        self,
        sources: Sequence[str],
        *,
        threads: ThreadPolicy = ThreadPolicy.AUTO,
        two_pass: bool = False,
    ) -> BatchReport:
        self.two_pass_requested.append(two_pass)
        report = BatchReport()
        for source in sources:
            try:
                report.records.append(await self.add(source, threads=threads))
            except Exception as exc:
                report.failures.append((source, type(exc).__name__))
        return report


class FakeProviders:
    def __init__(self) -> None:
        self.secrets: dict[str, str | None] = {"generation": None, "embedding": None}
        self.providers: dict[str, ProviderView] = {}
        self.tested: list[str] = []

    def list_providers(self) -> list[ProviderView]:
        return list(self.providers.values())

    async def save_provider(self, **values: Any) -> ProviderView:
        self.secrets["generation"] = values.pop("generation_api_key")
        self.secrets["embedding"] = values.pop("embedding_api_key")
        role = values.pop("role")
        view = ProviderView(
            **values,
            roles=["generation", "embedding"] if role == "both" else [role],
            generation_has_api_key=self.secrets["generation"] is not None,
            generation_key_fingerprint=("****4e738ca5" if self.secrets["generation"] else None),
            embedding_has_api_key=self.secrets["embedding"] is not None,
            embedding_key_fingerprint=("****fa22c991" if self.secrets["embedding"] else None),
        )
        self.providers[view.provider_id] = view
        return view

    async def test_provider(self, provider_id: str, role: str | None = None) -> None:
        if provider_id not in self.providers:
            raise KeyError(provider_id)
        self.tested.append(f"{provider_id}:{role or 'all'}")

    def delete_key(self, provider_id: str, role: str) -> bool:
        view = self.providers.get(provider_id)
        if view is None:
            raise KeyError(provider_id)
        if role not in {"generation", "embedding"}:
            raise ValueError(role)
        existed = self.secrets.get(role) is not None
        self.secrets[role] = None
        prefix = role
        self.providers[provider_id] = view.model_copy(
            update={
                f"{prefix}_has_api_key": False,
                f"{prefix}_key_fingerprint": None,
            }
        )
        return existed


class FakeResolvedIngestion:
    def __init__(self, repository: FakeRepository) -> None:
        self.repository = repository
        self.sources: list[ResolvedSource] = []

    async def add_resolved(self, source: ResolvedSource) -> ArtifactRecord:
        self.sources.append(source)
        record = make_record(f"art_resolved_{len(self.sources)}")
        self.repository.records.append(record)
        return record


class FakeBrowserCapture:
    def __init__(self, outcome: LoginOutcome = LoginOutcome.SIGNED_IN) -> None:
        self.login_urls: list[str] = []
        self.forced: list[bool] = []
        self.outcome = outcome

    async def capture(self, url: str, *, authorized: bool = False) -> ResolvedSource:
        assert authorized is True
        return ResolvedSource(
            canonical_url=url,
            source_kind=SourceKind.X,
            title="Captured thread",
            text="Bundled author thread",
            extraction_method="authorized_visible_browser",
        )

    async def open_login(self, url: str, *, authorized: bool = False, force: bool = False) -> LoginOutcome:
        assert authorized is True
        self.login_urls.append(url)
        self.forced.append(force)
        return self.outcome


def settle_login(client: TestClient, path: str = "/api/browser/login/status") -> dict[str, str]:
    """Poll the status endpoint the way the interface does, until it stops running."""

    for _ in range(50):
        payload = client.get(path).json()
        if payload["state"] != "running":
            return dict(payload)
        time.sleep(0.02)
    raise AssertionError("managed browser sign-in never left the running state")


@pytest.fixture
def web_stack() -> tuple[TestClient, FakeRepository, FakeIngestion, FakeProviders]:
    repository = FakeRepository()
    ingestion = FakeIngestion(repository)
    providers = FakeProviders()
    app = create_web_app(
        engine=FakeEngine(repository),  # type: ignore[arg-type]
        ingestion=ingestion,  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
        provider_settings=providers,
    )
    return TestClient(app, base_url="http://localhost"), repository, ingestion, providers


def test_ui_pages_and_progressive_search_design_and_ingestion(
    web_stack: tuple[TestClient, FakeRepository, FakeIngestion, FakeProviders],
) -> None:
    client, _, _, _ = web_stack
    for path, expected in (
        ("/", "Personal engineering frontier"),
        ("/add", "Add knowledge"),
        ("/search", "Search knowledge"),
        ("/artifacts/art_memory", "Compact Agent Memory"),
        ("/issues", "Needs review"),
        ("/design", "Design Lab"),
        ("/projects", "Projects"),
        ("/settings/providers", "AI providers"),
    ):
        response = client.get(path)
        assert response.status_code == 200
        assert expected in response.text
        assert "/static/htmx.min.js" in response.text

    issues = client.get("/issues")
    assert "The social post says the method is perfect." in issues.text
    assert "The paper reports measured limitations." in issues.text

    search = client.post(
        "/search",
        data={"query": "agent memory", "limit": "10"},
        headers={"HX-Request": "true"},
    )
    assert search.status_code == 200
    assert "Compact Agent Memory" in search.text

    design = client.post(
        "/design",
        data={"architecture": "A memory-heavy agent", "requirements": "Low token cost"},
        headers={"HX-Request": "true"},
    )
    assert design.status_code == 200
    assert "Compare against raw history" in design.text

    added = client.post(
        "/add",
        data={"mode": "url", "source": "https://example.com/new"},
        headers={"HX-Request": "true"},
    )
    assert added.status_code == 200
    assert "Capture succeeded" in added.text


def test_json_api_covers_knowledge_design_review_and_project_history(
    web_stack: tuple[TestClient, FakeRepository, FakeIngestion, FakeProviders],
) -> None:
    client, _, _, _ = web_stack
    assert client.get("/api/health").json()["schema_version"] == 1
    assert client.post("/api/search", json={"query": "memory"}).json()["results"][0]["score"] == 0.91
    assert client.get("/api/records/art_memory").json()["artifact"]["id"] == "art_memory"
    assert client.post("/api/design", json={"architecture": "agent memory"}).json()["idea_cards"]
    assert client.get("/api/issues").json()["unresolved_count"] == 1

    resolved = client.post(
        "/api/issues/issue_one/resolve",
        json={"action": "keep_both"},
    )
    assert resolved.status_code == 200
    assert resolved.json()["status"] == "kept_both"

    captured = client.post("/api/ingestion", json={"sources": ["https://one", "https://two"]})
    assert captured.status_code == 202
    assert len(captured.json()["records"]) == 2
    assert client.get("/api/jobs").json()["jobs"][0]["id"] == "job_one"

    project = client.post(
        "/api/projects",
        json={"name": "Memory agent", "constraints": ["local-first"]},
    ).json()
    history = client.get(f"/api/projects/{project['id']}").json()
    assert history["projects"][0]["name"] == "Memory agent"
    assert client.get("/api/projects").json()["projects"][0]["id"] == project["id"]

    decision = client.post(
        "/api/decisions",
        json={
            "project": "Memory agent",
            "artifact_id": "art_memory",
            "decision": "Run a trial",
            "rationale": "Fits token constraints",
        },
    ).json()
    outcome = client.post(
        "/api/outcomes",
        json={
            "project_id": decision["project_id"],
            "decision_id": decision["id"],
            "artifact_id": "art_memory",
            "outcome": "Recall improved",
            "succeeded": True,
        },
    )
    assert outcome.status_code == 201
    assert outcome.json()["succeeded"] is True

    backup = client.post(
        "/api/maintenance/backup",
        json={"destination": "C:/backups"},
    )
    assert backup.json() == {"backup": "C:/backups/steering-backup"}
    reindex = client.post("/api/maintenance/reindex", json={})
    assert reindex.json() == {
        "reindexed": True,
        "reembedded_chunks": 3,
        "verified_backup": "C:/backups/steering-pre-reembed",
    }


def test_the_artifact_page_states_the_license_or_says_it_is_unstated(
    web_stack: tuple[TestClient, FakeRepository, FakeIngestion, FakeProviders],
) -> None:
    """An unstated license is not permissive, so silence would mislead a reader."""

    client, repository, _, _ = web_stack

    unstated = client.get("/artifacts/art_memory").text
    assert "License" in unstated
    assert "Not stated" in unstated
    assert "treat as all rights reserved" in unstated

    record = repository.records[0]
    record.artifact.license = "Apache-2.0"
    record.artifact.metadata["license_source"] = "https://github.com/example/project"

    stated = client.get("/artifacts/art_memory").text
    assert "Apache-2.0" in stated
    assert "https://github.com/example/project" in stated
    assert "Not stated" not in stated


def test_audit_reports_records_that_cannot_be_opened(
    web_stack: tuple[TestClient, FakeRepository, FakeIngestion, FakeProviders],
) -> None:
    client, repository, _, _ = web_stack

    clean = client.get("/api/maintenance/audit")
    assert clean.status_code == 200
    assert clean.json()["unreadable_count"] == 0

    repository.unreadable = [{"artifact_id": "art_broken", "title": "Broken", "detail": "Invalid JSON: EOF"}]
    damaged = client.get("/api/maintenance/audit").json()
    assert damaged["unreadable_count"] == 1
    assert damaged["unreadable"][0]["artifact_id"] == "art_broken"
    assert damaged["unreadable"][0]["detail"] == "Invalid JSON: EOF"


def test_web_project_workflow_creates_records_history_and_design_selection(
    web_stack: tuple[TestClient, FakeRepository, FakeIngestion, FakeProviders],
) -> None:
    client, repository, _, _ = web_stack
    created = client.post(
        "/projects",
        data={
            "name": "Memory redesign",
            "description": "Bounded project history",
            "constraints": "local-first\nlow latency",
        },
        follow_redirects=False,
    )
    assert created.status_code == 303
    project_id = created.headers["location"].rsplit("/", 1)[-1]

    design = client.get("/design")
    assert f'value="{project_id}"' in design.text
    assert "Memory redesign" in design.text

    decision = client.post(
        f"/projects/{project_id}/decisions",
        data={
            "decision": "Run a bounded prototype",
            "rationale": "It matches the local constraint",
            "artifact_id": "art_memory",
        },
        follow_redirects=False,
    )
    assert decision.status_code == 303
    assert repository.decisions[0].project_id == project_id

    outcome = client.post(
        f"/projects/{project_id}/outcomes",
        data={
            "outcome": "Recall improved",
            "succeeded": "true",
            "constraints": "local-first",
            "decision_id": repository.decisions[0].id,
            "artifact_id": "art_memory",
        },
        follow_redirects=False,
    )
    assert outcome.status_code == 303
    assert repository.outcomes[0].project_id == project_id

    detail = client.get(f"/projects/{project_id}")
    assert "Run a bounded prototype" in detail.text
    assert "Recall improved" in detail.text
    assert client.get("/projects/missing").status_code == 404


def test_explicit_upload_and_authorized_browser_capture() -> None:
    repository = FakeRepository()
    resolved = FakeResolvedIngestion(repository)
    browser = FakeBrowserCapture()
    app = create_web_app(
        engine=FakeEngine(repository),  # type: ignore[arg-type]
        ingestion=FakeIngestion(repository),  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
        provider_settings=FakeProviders(),
        resolved_ingestion=resolved,
        browser_capture=browser,
        login_session=BrowserLoginSession(browser),
    )
    with TestClient(app, base_url="http://localhost") as client:
        denied_upload = client.post(
            "/api/ingestion/upload",
            data={"authorized": "false"},
            files={"file": ("notes.txt", b"useful notes", "text/plain")},
        )
        assert denied_upload.status_code == 403
        uploaded = client.post(
            "/api/ingestion/upload",
            data={"authorized": "true"},
            files={"file": ("notes.txt", b"useful notes", "text/plain")},
        )
        assert uploaded.status_code == 202
        assert resolved.sources[-1].text == "useful notes"

        denied_browser = client.post(
            "/api/ingestion/browser",
            json={"url": "https://x.com/example/status/1", "authorized": False},
        )
        assert denied_browser.status_code == 403
        captured = client.post(
            "/api/ingestion/browser",
            json={"url": "https://x.com/example/status/1", "authorized": True},
        )
        assert captured.status_code == 202
        assert resolved.sources[-1].extraction_method == "authorized_visible_browser"

        denied_login = client.post(
            "/api/browser/login",
            json={"url": "https://www.linkedin.com/login", "authorized": False},
        )
        assert denied_login.status_code == 403

        # Sign-in is human-paced, so the request starts it and returns at once
        # rather than holding the connection open for the whole attempt.
        login = client.post(
            "/api/browser/login",
            json={"url": "https://www.linkedin.com/login", "authorized": True},
        )
        assert login.status_code == 202
        assert login.json()["state"] == "running"
        assert settle_login(client)["state"] == "signed_in"
        assert browser.login_urls == ["https://www.linkedin.com/login"]

        login_ui = client.post(
            "/browser/login",
            data={"login_url": "https://x.com/i/flow/login", "authorize_login": "on"},
            headers={"HX-Request": "true"},
        )
        assert login_ui.status_code == 200
        # The interface is told to poll rather than block on the response.
        assert 'hx-get="/browser/login/status"' in login_ui.text
        assert settle_login(client)["state"] == "signed_in"
        assert browser.login_urls[-1] == "https://x.com/i/flow/login"
        assert "session is saved" in client.get("/browser/login/status").text


def test_provider_api_masks_secret_and_supports_test_replace_and_delete(
    web_stack: tuple[TestClient, FakeRepository, FakeIngestion, FakeProviders],
) -> None:
    client, _, _, providers = web_stack
    secret = "sk-never-return-this-value"
    saved = client.post(
        "/api/providers",
        json={
            "provider_id": "local-openai",
            "role": "both",
            "base_url": "http://127.0.0.1:8000/v1",
            "generation_model": "generation-model",
            "embedding_model": "embedding-model",
            "embedding_dimension": 768,
            "generation_api_key": secret,
            "embedding_api_key": "embed-secret-never-return",
        },
    )
    assert saved.status_code == 201
    assert secret not in saved.text
    assert saved.json()["generation_key_fingerprint"].startswith("****")

    listing = client.get("/api/providers")
    settings = client.get("/settings/providers")
    assert secret not in listing.text
    assert secret not in settings.text
    assert "password" in settings.text

    assert client.post("/api/providers/local-openai/test").json() == {"status": "ok"}
    assert providers.tested == ["local-openai:all"]
    deleted = client.delete("/api/providers/local-openai/key?role=generation")
    assert deleted.json() == {"deleted": True}
    assert providers.secrets["generation"] is None


def test_mutations_require_local_host_same_origin_and_small_body(
    web_stack: tuple[TestClient, FakeRepository, FakeIngestion, FakeProviders],
) -> None:
    client, repository, ingestion, providers = web_stack
    app = create_web_app(
        engine=FakeEngine(repository),  # type: ignore[arg-type]
        ingestion=ingestion,  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
        provider_settings=providers,
    )
    remote = TestClient(app, base_url="http://remote.example")
    assert remote.post("/api/search", json={"query": "memory"}).status_code == 403
    assert (
        client.post(
            "/api/search",
            json={"query": "memory"},
            headers={"Origin": "https://attacker.example"},
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/search",
            json={"query": "memory"},
            headers={"Sec-Fetch-Site": "cross-site"},
        ).status_code
        == 403
    )
    oversized = client.post("/api/ingestion", content=b"x" * (21 * 1_048_576 + 1))
    assert oversized.status_code == 413


def test_unexpected_errors_are_safe_and_validation_never_echoes_secret() -> None:
    repository = FakeRepository()
    app = create_web_app(
        engine=FakeEngine(repository),  # type: ignore[arg-type]
        ingestion=FakeIngestion(repository, fail=True),  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
        provider_settings=FakeProviders(),
    )
    with TestClient(app, base_url="http://localhost", raise_server_exceptions=False) as client:
        failed = client.post("/api/ingestion", json={"source": "https://example.com"})
        assert failed.status_code == 500
        assert failed.json() == {"error": "Request could not be completed."}
        assert "PRIVATE-PROVIDER-KEY" not in failed.text

        failed_html = client.post(
            "/add",
            data={"mode": "url", "source": "https://example.com"},
        )
        assert failed_html.status_code == 500
        assert "Request could not be completed" in failed_html.text
        assert "PRIVATE-PROVIDER-KEY" not in failed_html.text

        invalid_secret = "sk-invalid-never-echo"
        invalid = client.post(
            "/api/providers",
            json={
                "provider_id": "bad id",
                "base_url": "https://example.com",
                "generation_api_key": invalid_secret,
            },
        )
        assert invalid.status_code == 422
        assert invalid_secret not in invalid.text


def test_provider_timeout_returns_clean_retryable_ingestion_failure() -> None:
    repository = FakeRepository()
    app = create_web_app(
        engine=FakeEngine(repository),  # type: ignore[arg-type]
        ingestion=FakeIngestion(
            repository,
            error=ProviderTimeoutError("timeout included PRIVATE-PROVIDER-KEY"),
        ),  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
        provider_settings=FakeProviders(),
    )
    expected = "Ingestion failed because the generation provider timed out. Retry."
    with TestClient(app, base_url="http://localhost", raise_server_exceptions=False) as client:
        fragment = client.post(
            "/add",
            data={"mode": "url", "source": "https://example.com"},
            headers={"HX-Request": "true"},
        )
        assert fragment.status_code == 200
        assert "Capture failed" in fragment.text
        assert expected in fragment.text
        assert "Retry ingestion" in fragment.text

        page = client.post(
            "/add",
            data={"mode": "url", "source": "https://example.com"},
        )
        assert page.status_code == 503
        assert expected in page.text
        assert 'href="/add"' in page.text

        api = client.post("/api/ingestion", json={"source": "https://example.com"})
        assert api.status_code == 503
        assert api.json() == {"error": expected}

        assert "PRIVATE-PROVIDER-KEY" not in fragment.text + page.text + api.text


def test_source_unavailable_returns_clean_retryable_ingestion_failure() -> None:
    repository = FakeRepository()
    app = create_web_app(
        engine=FakeEngine(repository),  # type: ignore[arg-type]
        ingestion=FakeIngestion(
            repository,
            error=SourceUnavailableError("source included PRIVATE-SOURCE-DETAIL"),
        ),  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
        provider_settings=FakeProviders(),
    )
    expected = "Ingestion failed because the source could not be reached. Retry."
    with TestClient(app, base_url="http://localhost", raise_server_exceptions=False) as client:
        fragment = client.post(
            "/add",
            data={"mode": "url", "source": "https://example.com"},
            headers={"HX-Request": "true"},
        )
        assert fragment.status_code == 200
        assert expected in fragment.text
        assert "Retry ingestion" in fragment.text

        api = client.post("/api/ingestion", json={"source": "https://example.com"})
        assert api.status_code == 502
        assert api.json() == {"error": expected}
        assert "PRIVATE-SOURCE-DETAIL" not in fragment.text + api.text


def test_non_htmx_forms_missing_records_and_issue_resolution(
    web_stack: tuple[TestClient, FakeRepository, FakeIngestion, FakeProviders],
) -> None:
    client, repository, _, _ = web_stack

    assert "Compact Agent Memory" in client.get("/search?q=memory").text
    search = client.post("/search", data={"query": "memory", "limit": "5"})
    assert search.status_code == 200
    assert "Results for" in search.text
    design = client.post(
        "/design",
        data={"architecture": "agent", "requirements": "cheap", "concerns": "recall\ncost"},
    )
    assert design.status_code == 200
    assert "Compare against raw history" in design.text

    issue_fragment = client.post(
        "/issues/issue_one/resolve",
        data={"action": "dismiss"},
        headers={"HX-Request": "true"},
    )
    assert issue_fragment.status_code == 200
    assert "dismissed" in issue_fragment.text.lower()
    repository.issues = [make_issue()]
    redirected = client.post(
        "/issues/issue_one/resolve",
        data={"action": "reject"},
        follow_redirects=False,
    )
    assert redirected.status_code == 303
    assert redirected.headers["location"] == "/issues"

    assert client.get("/artifacts/missing").status_code == 404
    assert "Knowledge record not found" in client.get("/artifacts/missing").text
    assert client.get("/api/records/missing").status_code == 404
    assert client.get("/api/records?limit=invalid").status_code == 422
    assert len(client.get("/api/records?limit=0").json()["records"]) == 1
    all_issues = client.get("/api/issues?unresolved=false").json()
    assert len(all_issues["issues"]) == 1
    assert all_issues["unresolved_count"] == 0
    assert client.post("/api/issues/missing/resolve", json={"action": "dismiss"}).status_code == 404


def test_add_form_batch_upload_browser_and_validation_paths() -> None:
    repository = FakeRepository()
    resolved = FakeResolvedIngestion(repository)
    app = create_web_app(
        engine=FakeEngine(repository),  # type: ignore[arg-type]
        ingestion=FakeIngestion(repository),  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
        provider_settings=FakeProviders(),
        resolved_ingestion=resolved,
        browser_capture=FakeBrowserCapture(),
        login_session=BrowserLoginSession(FakeBrowserCapture()),
    )
    with TestClient(app, base_url="http://localhost") as client:
        redirected = client.post(
            "/add",
            data={"mode": "url", "source": "https://example.com/direct"},
            follow_redirects=False,
        )
        assert redirected.status_code == 303
        assert redirected.headers["location"].startswith("/artifacts/")

        batch = client.post(
            "/add",
            data={"mode": "batch", "batch": "https://one\n"},
            files={"batch_file": ("more.txt", b"https://two\nhttps://three", "text/plain")},
            headers={"HX-Request": "true"},
        )
        assert batch.status_code == 200
        assert "3 sources were added" in batch.text
        invalid_batch = client.post(
            "/add",
            data={"mode": "batch"},
            files={"batch_file": ("bad.txt", b"\xff\xfe", "text/plain")},
        )
        assert invalid_batch.status_code == 422
        oversized_batch = client.post(
            "/add",
            data={"mode": "batch"},
            files={"batch_file": ("large.txt", b"x" * 512_001, "text/plain")},
        )
        assert oversized_batch.status_code == 413

        denied_upload = client.post(
            "/add",
            files={"knowledge_file": ("notes.txt", b"content", "text/plain")},
        )
        assert denied_upload.status_code == 403
        uploaded = client.post(
            "/add",
            data={"authorize_upload": "on"},
            files={"knowledge_file": ("notes.txt", b"content", "text/plain")},
            headers={"HX-Request": "true"},
        )
        assert uploaded.status_code == 200
        assert resolved.sources[-1].metadata["binary_retained"] is False

        denied_browser = client.post(
            "/add",
            data={"mode": "browser", "source": "https://x.com/a/status/1"},
        )
        assert denied_browser.status_code == 403
        browser = client.post(
            "/add",
            data={
                "mode": "browser",
                "source": "https://x.com/a/status/1",
                "authorize_browser": "on",
            },
            headers={"HX-Request": "true"},
        )
        assert browser.status_code == 200
        assert resolved.sources[-1].source_kind == SourceKind.X

        assert client.post("/add", data={"mode": "url"}).status_code == 422
        assert client.post("/api/ingestion/upload", data={"authorized": "true"}).status_code == 422
        unsupported = client.post(
            "/api/ingestion/upload",
            data={"authorized": "true"},
            files={"file": ("payload.bin", b"opaque", "application/octet-stream")},
        )
        assert unsupported.status_code == 422


def test_disabled_and_failed_capture_services_return_safe_errors() -> None:
    repository = FakeRepository()
    base = create_web_app(
        engine=FakeEngine(repository),  # type: ignore[arg-type]
        ingestion=FakeIngestion(repository),  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
        provider_settings=FakeProviders(),
    )
    with TestClient(base, base_url="http://localhost") as client:
        disabled_upload = client.post(
            "/api/ingestion/upload",
            data={"authorized": "true"},
            files={"file": ("notes.txt", b"content", "text/plain")},
        )
        assert disabled_upload.status_code == 501
        disabled_browser = client.post(
            "/api/ingestion/browser",
            json={"url": "https://x.com/a/status/1", "authorized": True},
        )
        assert disabled_browser.status_code == 501
        disabled_login = client.post(
            "/api/browser/login",
            json={"url": "https://www.linkedin.com/login", "authorized": True},
        )
        assert disabled_login.status_code == 501

    class RefusingBrowser:
        async def capture(self, url: str, *, authorized: bool = False) -> ResolvedSource:
            raise PermissionError(url)

    class BrokenBrowser:
        async def capture(self, url: str, *, authorized: bool = False) -> ResolvedSource:
            raise RuntimeError("private browser detail")

    for browser, expected in ((RefusingBrowser(), 403), (BrokenBrowser(), 502)):
        app = create_web_app(
            engine=FakeEngine(repository),  # type: ignore[arg-type]
            ingestion=FakeIngestion(repository),  # type: ignore[arg-type]
            repository=repository,  # type: ignore[arg-type]
            provider_settings=FakeProviders(),
            resolved_ingestion=FakeResolvedIngestion(repository),
            browser_capture=browser,
        )
        with TestClient(app, base_url="http://localhost") as client:
            response = client.post(
                "/api/ingestion/browser",
                json={"url": "https://x.com/a/status/1", "authorized": True},
            )
            assert response.status_code == expected
            assert "private browser detail" not in response.text

    class MissingBrowser:
        async def capture(self, url: str, *, authorized: bool = False) -> ResolvedSource:
            raise BrowserDependencyUnavailable(url)

        async def open_login(
            self, url: str, *, authorized: bool = False, force: bool = False
        ) -> LoginOutcome:
            raise BrowserDependencyUnavailable(url)

    missing_browser = MissingBrowser()
    missing_app = create_web_app(
        engine=FakeEngine(repository),  # type: ignore[arg-type]
        ingestion=FakeIngestion(repository),  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
        provider_settings=FakeProviders(),
        resolved_ingestion=FakeResolvedIngestion(repository),
        browser_capture=missing_browser,
        login_session=BrowserLoginSession(missing_browser),
    )
    with TestClient(missing_app, base_url="http://localhost") as client:
        response = client.post(
            "/browser/login",
            data={"login_url": "https://x.com/i/flow/login", "authorize_login": "on"},
            headers={"HX-Request": "true"},
        )
        assert response.status_code == 200
        # A missing dependency is reported through the same status the interface
        # already polls, rather than as a failed form submission.
        assert settle_login(client)["state"] == "unavailable"
        assert "Install the browser extra and Chromium" in client.get("/browser/login/status").text


def test_a_second_sign_in_attempt_is_refused_while_one_is_running() -> None:
    repository = FakeRepository()

    class SlowBrowser(FakeBrowserCapture):
        async def open_login(
            self, url: str, *, authorized: bool = False, force: bool = False
        ) -> LoginOutcome:
            await asyncio.sleep(0.5)
            return await super().open_login(url, authorized=authorized, force=force)

    browser = SlowBrowser()
    app = create_web_app(
        engine=FakeEngine(repository),  # type: ignore[arg-type]
        ingestion=FakeIngestion(repository),  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
        provider_settings=FakeProviders(),
        resolved_ingestion=FakeResolvedIngestion(repository),
        browser_capture=browser,
        login_session=BrowserLoginSession(browser),
    )
    with TestClient(app, base_url="http://localhost") as client:
        first = client.post(
            "/api/browser/login",
            json={"url": "https://x.com/i/flow/login", "authorized": True},
        )
        assert first.status_code == 202
        # Two managed browsers would contend for the same locked profile directory.
        second = client.post(
            "/api/browser/login",
            json={"url": "https://x.com/i/flow/login", "authorized": True},
        )
        assert second.status_code == 409
        assert settle_login(client)["state"] == "signed_in"


def test_evidence_mismatch_returns_visible_ingestion_failure_instead_of_500() -> None:
    repository = FakeRepository()

    class EvidenceFailingIngestion(FakeIngestion):
        async def add(self, source: str, *, threads: ThreadPolicy = ThreadPolicy.AUTO) -> ArtifactRecord:
            raise EvidenceValidationError(source)

    app = create_web_app(
        engine=FakeEngine(repository),  # type: ignore[arg-type]
        ingestion=EvidenceFailingIngestion(repository),  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
        provider_settings=FakeProviders(),
    )
    with TestClient(app, base_url="http://localhost") as client:
        response = client.post(
            "/add",
            data={"mode": "url", "source": "https://example.com/source"},
            headers={"HX-Request": "true"},
        )
        assert response.status_code == 200
        assert "Capture failed" in response.text
        assert "could not be verified" in response.text


def test_provider_html_and_api_failure_paths(
    web_stack: tuple[TestClient, FakeRepository, FakeIngestion, FakeProviders],
) -> None:
    client, _, _, providers = web_stack
    form = {
        "provider_id": "example",
        "role": "both",
        "base_url": "https://api.example.com/v1",
        "generation_model": "gen",
        "embedding_model": "embed",
        "embedding_dimension": "512",
        "generation_api_key": "generation-secret",
        "embedding_api_key": "embedding-secret",
    }
    saved = client.post(
        "/settings/providers",
        data=form,
        headers={"HX-Request": "true"},
    )
    assert saved.status_code == 200
    assert "Provider settings saved" in saved.text
    assert "generation-secret" not in saved.text
    redirected_save = client.post(
        "/settings/providers",
        data=form,
        follow_redirects=False,
    )
    assert redirected_save.status_code == 303
    assert redirected_save.headers["location"] == "/settings/providers"
    tested = client.post(
        "/settings/providers/example/test",
        data={"role": "generation"},
    )
    assert tested.status_code == 200
    deleted = client.post(
        "/settings/providers/example/delete-key",
        data={"role": "embedding"},
    )
    assert deleted.status_code == 200
    assert providers.secrets["embedding"] is None

    assert client.post("/settings/providers/missing/test", data={"role": "generation"}).status_code == 404
    assert (
        client.post(
            "/settings/providers/missing/delete-key",
            data={"role": "generation"},
        ).status_code
        == 404
    )
    assert client.post("/settings/providers/example/delete-key", data={"role": "invalid"}).status_code == 422
    assert client.delete("/api/providers/missing/key?role=generation").status_code == 404
    assert client.delete("/api/providers/example/key").status_code == 422
    assert client.post("/api/providers/missing/test").status_code == 404

    invalid_form = client.post(
        "/settings/providers",
        data={
            "provider_id": "incomplete",
            "role": "both",
            "base_url": "https://api.example.com/v1",
        },
    )
    assert invalid_form.status_code == 422
    assert "correct the submitted fields" in invalid_form.text.lower()

    class FailingProviders(FakeProviders):
        async def save_provider(self, **values: Any) -> ProviderView:
            raise RuntimeError("secret upstream detail")

        async def test_provider(self, provider_id: str, role: str | None = None) -> None:
            raise RuntimeError("secret upstream detail")

    repository = FakeRepository()
    failing = create_web_app(
        engine=FakeEngine(repository),  # type: ignore[arg-type]
        ingestion=FakeIngestion(repository),  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
        provider_settings=FailingProviders(),
    )
    with TestClient(failing, base_url="http://localhost") as failing_client:
        api_save = failing_client.post("/api/providers", json={**form, "embedding_dimension": 512})
        assert api_save.status_code == 502
        assert "secret upstream detail" not in api_save.text
        ui_save = failing_client.post("/settings/providers", data=form)
        assert ui_save.status_code == 502
        assert failing_client.post("/api/providers/example/test").status_code == 502
        assert (
            failing_client.post(
                "/settings/providers/example/test",
                data={"role": "generation"},
            ).status_code
            == 502
        )


def test_batch_ingestion_reports_failures_and_forwards_the_thread_flag() -> None:
    """An unattended run must say what it captured and what it could not."""

    repository = FakeRepository()

    class PartlyFailingIngestion(FakeIngestion):
        async def add(self, source: str, *, threads: ThreadPolicy = ThreadPolicy.AUTO) -> ArtifactRecord:
            self.thread_policies.append(threads)
            if source.endswith("/dead"):
                raise SourceUnavailableError("gone")
            return make_record("art_added")

    ingestion = PartlyFailingIngestion(repository)
    app = create_web_app(
        engine=FakeEngine(repository),  # type: ignore[arg-type]
        ingestion=ingestion,  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
        provider_settings=FakeProviders(),
    )
    with TestClient(app, base_url="http://localhost") as client:
        response = client.post(
            "/api/ingestion",
            json={
                "sources": [
                    "https://x.com/a/status/1",
                    "https://example.com/dead",
                    "https://x.com/a/status/2",
                ],
                "threads": "always",
            },
        )

    assert response.status_code == 202
    body = response.json()
    # One dead link does not discard the rest of the run.
    assert len(body["records"]) == 2
    assert body["failures"] == [
        {"source": "https://example.com/dead", "error_code": "SourceUnavailableError"}
    ]
    assert body["aborted"] is None
    assert ingestion.thread_policies == [ThreadPolicy.ALWAYS] * 3
    # A batch defaults to the free public pass first.
    assert ingestion.two_pass_requested == [False]


def test_single_ingestion_defaults_to_automatic_thread_escalation() -> None:
    repository = FakeRepository()
    ingestion = FakeIngestion(repository)
    app = create_web_app(
        engine=FakeEngine(repository),  # type: ignore[arg-type]
        ingestion=ingestion,  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
        provider_settings=FakeProviders(),
    )
    with TestClient(app, base_url="http://localhost") as client:
        response = client.post("/api/ingestion", json={"source": "https://x.com/a/status/1"})

    assert response.status_code == 202
    assert ingestion.thread_policies == [ThreadPolicy.AUTO]


def test_the_artifact_page_shows_where_the_capture_led() -> None:
    """A post's own URL says where knowledge came from; the paper or repository
    it pointed at is the thing worth opening."""

    repository = FakeRepository()
    record = repository.records[0]
    record.artifact.metadata.update(
        {
            "supporting_sources": ["https://github.com/example/hyperresearch"],
            "outbound_urls": [
                "https://github.com/example/hyperresearch",
                "https://substack.example/only-mentioned",
            ],
        }
    )
    record.snapshots.append(
        Snapshot(
            id="snap_followed",
            artifact_id=record.artifact.id,
            source_url="https://github.com/example/hyperresearch",
            content_hash="followed-hash",
            mime_type="text/markdown",
            text="A" * 4321,
            extraction_method="github_readme_api",
        )
    )

    app = create_web_app(
        engine=FakeEngine(repository),  # type: ignore[arg-type]
        ingestion=FakeIngestion(repository),  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
        provider_settings=FakeProviders(),
    )
    with TestClient(app, base_url="http://localhost") as client:
        page = client.get(f"/artifacts/{record.artifact.id}").text

    assert "Linked sources" in page
    # One line per destination: the link itself, nothing else.
    assert 'href="https://github.com/example/hyperresearch"' in page
    assert "https://substack.example/only-mentioned" in page
    # Both states are labelled: an unlabelled link would be ambiguous.
    assert re.search(r'hyperresearch</a>\s*(?:<!--.*?-->\s*)?<span class="hint">read</span>', page, re.S)
    assert re.search(r'only-mentioned</a>\s*(?:<!--.*?-->\s*)?<span class="hint">not read</span>', page, re.S)


def test_an_artifact_without_linked_sources_omits_the_section() -> None:
    repository = FakeRepository()
    app = create_web_app(
        engine=FakeEngine(repository),  # type: ignore[arg-type]
        ingestion=FakeIngestion(repository),  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
        provider_settings=FakeProviders(),
    )
    with TestClient(app, base_url="http://localhost") as client:
        page = client.get(f"/artifacts/{repository.records[0].artifact.id}").text

    assert "Linked sources" not in page


def _directory_app(repository: FakeRepository, changed: list[int] | None = None) -> Any:
    return create_web_app(
        engine=FakeEngine(repository),  # type: ignore[arg-type]
        ingestion=FakeIngestion(repository),  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
        provider_settings=FakeProviders(),
        on_record_changed=(lambda: changed.append(1)) if changed is not None else None,
    )


def test_the_search_page_lists_all_knowledge_before_any_search() -> None:
    """An empty page hides the knowledge that is already there."""

    repository = FakeRepository()
    with TestClient(_directory_app(repository), base_url="http://localhost") as client:
        page = client.get("/search").text

    assert "All knowledge" in page
    assert repository.records[0].artifact.title in page
    assert 'name="artifact_ids"' in page
    assert 'id="directory-select-all"' in page


def test_the_directory_sorts_the_whole_corpus_not_just_the_first_page() -> None:
    repository = FakeRepository()
    for index, title in enumerate(["Zebra technique", "Alpha method"]):
        record = make_record(f"art_sorted_{index}")
        record.artifact.title = title
        record.artifact.canonical_url = f"https://example.test/{index}"
        repository.records.append(record)

    with TestClient(_directory_app(repository), base_url="http://localhost") as client:
        alphabetical = client.get("/search?sort=title").text
        unknown_sort = client.get("/search?sort=nonsense").text

    titles = re.findall(r'/artifacts/art_\w+">([^<]+)</a>', alphabetical)
    assert titles == sorted(titles, key=str.casefold)
    # An unrecognised sort falls back rather than erroring.
    assert re.search(r'sort-option is-active"\s+href="/search\?sort=newest"', unknown_sort)


def test_bulk_retire_removes_every_selected_record_and_invalidates_retrieval() -> None:
    repository = FakeRepository()
    for index in range(2):
        record = make_record(f"art_retire_{index}")
        record.artifact.canonical_url = f"https://example.test/retire-{index}"
        repository.records.append(record)
    changed: list[int] = []
    before = repository.count_artifacts()

    with TestClient(_directory_app(repository, changed), base_url="http://localhost") as client:
        response = client.post(
            "/artifacts/retire",
            data={"artifact_ids": ["art_retire_0", "art_retire_1"], "sort": "newest"},
            headers={"HX-Request": "true"},
        )

    assert response.status_code == 200
    assert repository.count_artifacts() == before - 2
    assert "Retired 2 records" in response.text
    # Retrieval indexes would still reference the removed artifacts otherwise.
    assert changed == [1]


def test_retiring_nothing_is_rejected_rather_than_silently_succeeding() -> None:
    repository = FakeRepository()
    with TestClient(_directory_app(repository), base_url="http://localhost") as client:
        assert client.post("/artifacts/retire", data={}).status_code == 422


def test_retiring_an_unknown_record_reports_not_found() -> None:
    repository = FakeRepository()
    with TestClient(_directory_app(repository), base_url="http://localhost") as client:
        response = client.post("/artifacts/retire", data={"artifact_ids": ["art_missing"]})
        assert response.status_code == 404
        assert client.delete("/api/records/art_missing").status_code == 404


def test_a_single_record_can_be_retired_from_its_own_page_and_over_the_api() -> None:
    repository = FakeRepository()
    extra = make_record("art_api_retire")
    extra.artifact.canonical_url = "https://example.test/api-retire"
    repository.records.append(extra)

    with TestClient(_directory_app(repository), base_url="http://localhost") as client:
        page = client.get("/artifacts/art_api_retire").text
        assert 'action="/artifacts/art_api_retire/retire"' in page

        response = client.delete("/api/records/art_api_retire")
        assert response.status_code == 200
        assert response.json() == {"retired": ["art_api_retire"]}
    assert repository.get_record("art_api_retire") is None


def test_the_search_bar_can_restrict_results_by_publication_date() -> None:
    repository = FakeRepository()
    engine = FakeEngine(repository)
    app = create_web_app(
        engine=engine,  # type: ignore[arg-type]
        ingestion=FakeIngestion(repository),  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
        provider_settings=FakeProviders(),
    )
    with TestClient(app, base_url="http://localhost") as client:
        client.post(
            "/search",
            data={"query": "agent memory", "limit": "10", "published_after": "2026-01-15"},
            headers={"HX-Request": "true"},
        )

    assert engine.queries[-1].published_after is not None
    assert engine.queries[-1].published_after.date().isoformat() == "2026-01-15"


def test_an_omitted_date_leaves_undated_sources_in_the_results() -> None:
    """Setting the filter excludes sources with no known date, so it stays unset."""

    repository = FakeRepository()
    engine = FakeEngine(repository)
    app = create_web_app(
        engine=engine,  # type: ignore[arg-type]
        ingestion=FakeIngestion(repository),  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
        provider_settings=FakeProviders(),
    )
    with TestClient(app, base_url="http://localhost") as client:
        client.post(
            "/search",
            data={"query": "agent memory", "limit": "10", "published_after": ""},
            headers={"HX-Request": "true"},
        )

    assert engine.queries[-1].published_after is None


def test_an_unreadable_record_explains_itself_and_offers_retirement() -> None:
    """A plain GET reporting 'correct the submitted fields' told the user nothing."""

    repository = FakeRepository()

    class CorruptEngine(FakeEngine):
        def get_knowledge_record(self, artifact_id: str) -> Any:
            raise CorruptRecordError(artifact_id, "source_url: Field required")

    app = create_web_app(
        engine=CorruptEngine(repository),  # type: ignore[arg-type]
        ingestion=FakeIngestion(repository),  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
        provider_settings=FakeProviders(),
    )
    with TestClient(app, base_url="http://localhost") as client:
        response = client.get("/artifacts/art_broken")

    assert response.status_code == 500
    assert "could not be read" in response.text
    assert "source_url: Field required" in response.text
    # The only action that helps is offered directly.
    assert 'action="/artifacts/art_broken/retire"' in response.text
    assert "correct the submitted fields" not in response.text


def test_the_x_api_panel_is_hidden_until_a_client_id_is_configured() -> None:
    """The billed surface must not advertise itself to users who never opted in."""

    repository = FakeRepository()
    with TestClient(_directory_app(repository), base_url="http://localhost") as client:
        page = client.get("/add").text

    assert "Authorize the X API" not in page
    # The free surfaces are always present.
    assert "Save X bookmarks" in page


def test_authorizing_without_a_client_id_is_refused() -> None:
    repository = FakeRepository()
    with TestClient(_directory_app(repository), base_url="http://localhost") as client:
        assert client.post("/oauth/x/authorize", follow_redirects=False).status_code == 501
        assert client.post("/api/ingestion/x-bookmarks").status_code == 409


def test_authorizing_sends_the_user_to_x_with_pkce() -> None:
    repository = FakeRepository()

    class StubApi:
        authorized = False

    app = create_web_app(
        engine=FakeEngine(repository),  # type: ignore[arg-type]
        ingestion=FakeIngestion(repository),  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
        provider_settings=FakeProviders(),
        x_api=StubApi(),  # type: ignore[arg-type]
        x_api_client_id="client-id",
    )
    with TestClient(app, base_url="http://localhost") as client:
        page = client.get("/add").text
        response = client.post("/oauth/x/authorize", follow_redirects=False)

    # Asserted on the route the page offers rather than its wording, so copy can
    # change without pretending the capability did.
    assert 'action="/oauth/x/authorize"' in page
    assert response.status_code == 303
    location = response.headers["location"]
    assert location.startswith("https://x.com/i/oauth2/authorize?")
    assert "code_challenge_method=S256" in location
    assert "client_secret" not in location


def test_a_callback_with_a_mismatched_state_is_rejected() -> None:
    """A callback that did not come from our request must not be exchanged."""

    repository = FakeRepository()

    class StubApi:
        authorized = False

        async def complete_authorization(self, code: str, pending: Any) -> Any:
            raise AssertionError("the code must not be exchanged")

    app = create_web_app(
        engine=FakeEngine(repository),  # type: ignore[arg-type]
        ingestion=FakeIngestion(repository),  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
        provider_settings=FakeProviders(),
        x_api=StubApi(),  # type: ignore[arg-type]
        x_api_client_id="client-id",
    )
    with TestClient(app, base_url="http://localhost") as client:
        client.post("/oauth/x/authorize", follow_redirects=False)
        response = client.get("/oauth/x/callback?code=abc&state=forged", follow_redirects=False)

    assert response.status_code == 400
    assert "did not match" in response.text


def test_a_callback_without_a_started_authorization_is_rejected() -> None:
    repository = FakeRepository()

    class StubApi:
        authorized = False

    app = create_web_app(
        engine=FakeEngine(repository),  # type: ignore[arg-type]
        ingestion=FakeIngestion(repository),  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
        provider_settings=FakeProviders(),
        x_api=StubApi(),  # type: ignore[arg-type]
        x_api_client_id="client-id",
    )
    with TestClient(app, base_url="http://localhost") as client:
        response = client.get("/oauth/x/callback?code=abc&state=s", follow_redirects=False)

    assert response.status_code == 400
    assert "No X authorization was in progress" in response.text


def test_a_bookmark_import_takes_the_free_public_pass_first() -> None:
    """Browser time should be spent only on posts that came back needing it."""

    import json as _json

    repository = FakeRepository()
    ingestion = FakeIngestion(repository)
    app = create_web_app(
        engine=FakeEngine(repository),  # type: ignore[arg-type]
        ingestion=ingestion,  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
        provider_settings=FakeProviders(),
    )
    body = _json.dumps(["https://x.com/a/status/1", "https://x.com/b/status/2"]).encode()
    with TestClient(app, base_url="http://localhost") as client:
        assert client.post("/api/ingestion/bookmarks", content=body).status_code == 202

    assert ingestion.two_pass_requested == [True]


def test_a_non_standard_license_is_distinguished_from_having_none(
    web_stack: tuple[TestClient, FakeRepository, FakeIngestion, FakeProviders],
) -> None:
    """A license file that must be read is a different decision from all-rights-reserved."""

    client, repository, _, _ = web_stack
    repository.records[0].artifact.metadata["license_unidentified"] = True

    page = client.get("/artifacts/art_memory").text

    assert "Non-standard" in page
    assert "read it before adopting" in page
    assert "Not stated" not in page
