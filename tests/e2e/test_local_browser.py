from __future__ import annotations

import os
import socket
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from typing import Any

import pytest
import uvicorn

from steering.domain.models import (
    ArchitectureReview,
    Artifact,
    ArtifactRecord,
    ArtifactType,
    IdeaCard,
    ScoreBreakdown,
    SearchHit,
    SearchQuery,
    SourceKind,
)
from steering.web import ProviderView, create_web_app

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("STEERING_BROWSER_E2E") != "1",
        reason="set STEERING_BROWSER_E2E=1 to run the local Chromium fixture",
    ),
]


def _record() -> ArtifactRecord:
    return ArtifactRecord(
        artifact=Artifact(
            id="art_local_fixture",
            canonical_url="https://example.invalid/local-fixture",
            source_kind=SourceKind.TEXT,
            artifact_type=ArtifactType.TECHNIQUE,
            title="Local Memory Fixture",
            summary="A deterministic local source for browser testing.",
            strategy_family="memory",
            content_hash="local-fixture-hash",
            capabilities=["Keeps the E2E test offline"],
            limitations=["Not real-world evidence"],
        )
    )


class FixtureRepository:
    def __init__(self) -> None:
        self.records: list[ArtifactRecord] = []

    def list_records(self) -> list[ArtifactRecord]:
        return self.records

    def list_artifacts(self, *, limit: int | None = None) -> list[Artifact]:
        artifacts = [record.artifact for record in self.records]
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

    def list_jobs(self, limit: int = 100) -> list[Any]:
        return []

    def list_issues(self, unresolved_only: bool = False) -> list[Any]:
        return []

    def get_record(self, artifact_id: str) -> ArtifactRecord | None:
        return next((item for item in self.records if item.artifact.id == artifact_id), None)

    def project_history(self, project_id: str) -> Mapping[str, Sequence[Any]]:
        return {"projects": [], "decisions": [], "outcomes": [], "review_runs": []}

    def list_projects(self) -> list[Any]:
        return []


class FixtureIngestion:
    def __init__(self, repository: FixtureRepository) -> None:
        self.repository = repository

    async def add(self, source: str) -> ArtifactRecord:
        record = _record()
        if not self.repository.records:
            self.repository.records.append(record)
        return record

    async def add_batch(self, sources: Sequence[str]) -> list[ArtifactRecord]:
        return [await self.add(source) for source in sources]


class FixtureEngine:
    def __init__(self, repository: FixtureRepository) -> None:
        self.repository = repository

    async def search(self, query: SearchQuery) -> list[SearchHit]:
        if not self.repository.records:
            return []
        return [
            SearchHit(
                artifact=self.repository.records[0].artifact,
                score=0.95,
                scores=ScoreBreakdown(rrf=0.95),
                citation_urls=["https://example.invalid/local-fixture"],
            )
        ]

    def get_knowledge_record(self, artifact_id: str) -> ArtifactRecord | None:
        return self.repository.get_record(artifact_id)

    async def review_architecture(self, architecture: str, *_: Any, **__: Any) -> ArchitectureReview:
        artifact = self.repository.records[0].artifact
        return ArchitectureReview(
            query=architecture,
            decomposed_areas=["memory"],
            observations=["A local evidence fixture matched the architecture."],
            idea_cards=[
                IdeaCard(
                    artifact_id=artifact.id,
                    title=artifact.title,
                    what_it_is="A deterministic local memory approach.",
                    why_it_fits="It exercises architecture-active recall without a provider.",
                    trust_lane=artifact.trust_lane,
                    source_url=artifact.canonical_url or "https://example.invalid/local-fixture",
                    published_at=None,
                    advantages=["Offline and reproducible"],
                    limitations=["Fixture evidence only"],
                    compatibility_requirements=["A browser"],
                    minimal_experiment="Compare fixture recall with an empty graph.",
                    success_criteria="The option appears in Design Lab.",
                    failure_criteria="No option is rendered.",
                    related_alternatives=[],
                    uncertainty="Synthetic local fixture.",
                    strategy_family="memory",
                )
            ],
            citations=["https://example.invalid/local-fixture"],
        )


class FixtureProviders:
    def list_providers(self) -> list[ProviderView]:
        return []


@contextmanager
def _loopback_web_server() -> Iterator[str]:
    repository = FixtureRepository()
    app = create_web_app(
        engine=FixtureEngine(repository),  # type: ignore[arg-type]
        ingestion=FixtureIngestion(repository),  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
        provider_settings=FixtureProviders(),  # type: ignore[arg-type]
    )
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off", server_header=False))
        thread = threading.Thread(
            target=server.run,
            kwargs={"sockets": [listener]},
            daemon=True,
        )
        thread.start()
        deadline = time.monotonic() + 10
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not server.started:
            server.should_exit = True
            thread.join(timeout=2)
            raise RuntimeError("local browser fixture server did not start")
        try:
            yield f"http://127.0.0.1:{port}"
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            if thread.is_alive():
                raise RuntimeError("local browser fixture server did not stop")


def test_dashboard_add_search_and_design_in_headless_chromium() -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    with _loopback_web_server() as base_url, playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto(base_url, wait_until="networkidle")
            page.get_by_role("heading", name="Bring saved ideas back").wait_for()

            page.get_by_role("link", name="Add", exact=True).click()
            page.get_by_role("heading", name="Add knowledge").wait_for()
            page.locator("#source").fill("A local note about agent memory")
            page.get_by_role("button", name="Capture source").click()
            page.get_by_text("Capture complete.").wait_for()
            page.get_by_role("heading", name="Local Memory Fixture").wait_for()

            page.get_by_role("link", name="Search", exact=True).click()
            page.locator("#query").fill("memory")
            page.get_by_role("button", name="Search").click()
            page.get_by_role("heading", name="Local Memory Fixture").wait_for()

            page.get_by_role("link", name="Design Lab", exact=True).click()
            page.locator("#architecture").fill("An agent that needs compact memory")
            page.locator("#requirements").fill("Offline and deterministic")
            page.get_by_role("button", name="Review architecture").click()
            page.get_by_role("heading", name="Architecture review").wait_for()
            page.get_by_text("A local evidence fixture matched the architecture.").wait_for()
        finally:
            browser.close()
