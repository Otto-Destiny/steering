"""Drive the capture surface in real Chromium.

The page no longer asks which kind of input you have; a script reads the one
visible field and fills the fields the ingestion route has always read. That
mapping only exists in the browser, so a server-side assertion cannot prove it.
What is checked here is the request that actually leaves the page.
"""

from __future__ import annotations

import contextlib
import socket
import threading
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from tests.web.test_app import (
    FakeEngine,
    FakeIngestion,
    FakeProviders,
    FakeRepository,
    FakeResolvedIngestion,
)

from steering.domain.models import ArtifactRecord
from steering.ingestion.login_session import BrowserLoginSession
from steering.ingestion.service import ThreadPolicy
from steering.web import create_web_app

pytestmark = pytest.mark.e2e

#: A browser normalises textarea line breaks to CRLF when it posts them.
POSTED_BREAK = "\r\n"


class RecordingIngestion(FakeIngestion):
    def __init__(self, repository: Any) -> None:
        super().__init__(repository)
        self.captured: list[str] = []
        self.batches: list[list[str]] = []

    async def add(self, source: str, *, threads: ThreadPolicy = ThreadPolicy.AUTO) -> ArtifactRecord:
        self.captured.append(source)
        return await super().add(source, threads=threads)

    async def add_batch(
        self,
        sources: Sequence[str],
        *,
        threads: ThreadPolicy = ThreadPolicy.AUTO,
    ) -> list[ArtifactRecord]:
        self.batches.append(list(sources))
        return await super().add_batch(sources, threads=threads)


class StubBrowser:
    """Enough of the managed browser for the page to offer deep capture."""

    def has_stored_session(self) -> bool:
        return True


@contextlib.contextmanager
def _serving(*, browser: bool = False) -> Iterator[tuple[str, RecordingIngestion]]:
    repository = FakeRepository()
    ingestion = RecordingIngestion(repository)
    capture = StubBrowser() if browser else None
    app = create_web_app(
        engine=FakeEngine(repository),  # type: ignore[arg-type]
        ingestion=ingestion,  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
        provider_settings=FakeProviders(),
        resolved_ingestion=FakeResolvedIngestion(repository),
        browser_capture=capture,  # type: ignore[arg-type]
        login_session=BrowserLoginSession(capture) if capture else None,  # type: ignore[arg-type]
    )
    with contextlib.closing(socket.socket()) as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(60):
            try:
                with contextlib.closing(socket.create_connection(("127.0.0.1", port), 0.2)):
                    break
            except OSError:
                time.sleep(0.25)
        else:
            raise RuntimeError("the capture fixture server did not start")
        yield f"http://127.0.0.1:{port}", ingestion
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@contextlib.contextmanager
def _page(base_url: str) -> Iterator[tuple[Any, list[str], list[str]]]:
    playwright_api = pytest.importorskip("playwright.sync_api")
    with playwright_api.sync_playwright() as runtime:
        if not Path(runtime.chromium.executable_path).is_file():
            pytest.skip("the managed Chromium runtime is not installed")
        browser = runtime.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            posts: list[str] = []
            errors: list[str] = []
            page.on(
                "request",
                lambda r: posts.append(r.post_data or "") if r.method == "POST" else None,
            )
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.goto(f"{base_url}/add", wait_until="load")
            yield page, posts, errors
        finally:
            browser.close()


def _field(body: str, name: str) -> str | None:
    """Read one field out of a multipart body, which is what the form posts."""

    start = body.find(f'name="{name}"')
    if start == -1:
        return None
    opens = body.find(POSTED_BREAK * 2, start)
    closes = body.find(POSTED_BREAK + "--", opens)
    if opens == -1 or closes == -1:
        return None
    return body[opens + 4 : closes]


def test_one_link_is_read_as_a_single_source() -> None:
    with _serving() as (base_url, ingestion), _page(base_url) as (page, posts, errors):
        page.fill("#capture-input", "https://x.com/researcher/status/1885026028428681698")

        detected = page.locator("#capture-detect").inner_text()
        assert "1 link" in detected
        assert "x.com" in detected
        assert page.locator("#capture-submit").inner_text() == "Capture 1 source"

        page.click("#capture-submit")
        page.wait_for_selector("#add-result >> text=/1 source was added/", timeout=10_000)

        assert ingestion.captured == ["https://x.com/researcher/status/1885026028428681698"]
        assert ingestion.batches == []
        assert _field(posts[-1], "mode") == "url"
        assert errors == []


def test_several_lines_become_a_batch_without_anyone_saying_so() -> None:
    """The mode radio used to ask; the newline already carried the answer."""

    links = [
        "https://github.com/viperrcrypto/Siftly",
        "https://arxiv.org/abs/2501.12948",
        "https://huggingface.co/Qwen/Qwen3-8B",
    ]
    with _serving() as (base_url, ingestion), _page(base_url) as (page, posts, errors):
        page.fill("#capture-input", "\n".join(links))

        assert "3 links" in page.locator("#capture-detect").inner_text()
        assert page.locator("#capture-submit").inner_text() == "Capture 3 sources"

        page.click("#capture-submit")
        page.wait_for_selector("#add-result >> text=/3 sources were added/", timeout=10_000)

        assert ingestion.batches == [links]
        assert _field(posts[-1], "mode") == "batch"
        assert _field(posts[-1], "batch") == POSTED_BREAK.join(links)
        assert errors == []


def test_prose_is_captured_as_text_rather_than_split_into_sources() -> None:
    with _serving() as (base_url, ingestion), _page(base_url) as (page, posts, errors):
        page.fill("#capture-input", "A note about agent memory.\nIt runs to two lines.")

        assert "Pasted text" in page.locator("#capture-detect").inner_text()

        page.click("#capture-submit")
        page.wait_for_selector("#add-result >> text=/1 source was added/", timeout=10_000)

        assert ingestion.batches == []
        assert len(ingestion.captured) == 1
        assert _field(posts[-1], "mode") == "url"
        assert errors == []


def test_deep_capture_asks_for_consent_and_only_offers_itself_for_one_link() -> None:
    """Deep capture reads one post's thread, so a batch cannot use it."""

    with _serving(browser=True) as (base_url, _ingestion), _page(base_url) as (page, _posts, errors):
        page.fill("#capture-input", "https://x.com/researcher/status/1885026028428681698")
        page.check("#depth-deep")

        assert page.locator("#consent-browser").is_visible()
        assert page.locator("#capture-submit").inner_text() == "Capture with the browser"

        page.fill("#capture-input", "https://example.com/one\nhttps://example.com/two")

        assert page.locator("#depth-deep").is_disabled()
        assert not page.locator("#consent-browser").is_visible()
        assert "One link at a time" in page.locator("#depth-deep-note").inner_text()
        assert errors == []


@pytest.mark.parametrize(
    ("filename", "field"),
    [
        ("paper.pdf", "capture-knowledge"),
        ("links.txt", "capture-list"),
        ("x-bookmarks.json", "capture-bookmarks"),
    ],
)
def test_a_dropped_file_lands_in_the_input_its_route_reads(tmp_path: Path, filename: str, field: str) -> None:
    """One drop target, three destinations the route already understood."""

    source = tmp_path / filename
    source.write_bytes(b"%PDF-1.4 " if filename.endswith(".pdf") else b"https://example.com/one")
    with _serving() as (base_url, _ingestion), _page(base_url) as (page, _posts, errors):
        page.set_input_files("#capture-picker", str(source))

        assert filename in page.locator("#capture-detect").inner_text()
        for candidate in ("capture-knowledge", "capture-list", "capture-bookmarks"):
            staged = page.evaluate(f"document.getElementById('{candidate}').files.length")
            assert staged == (1 if candidate == field else 0), candidate
        assert errors == []


def test_consent_for_reading_a_file_is_asked_for_by_name(tmp_path: Path) -> None:
    """The upload gate stays a real tick, worded for the file actually attached."""

    source = tmp_path / "agent-memory.pdf"
    source.write_bytes(b"%PDF-1.4 minimal")
    with _serving() as (base_url, _ingestion), _page(base_url) as (page, _posts, errors):
        assert not page.locator("#consent-upload").is_visible()

        page.set_input_files("#capture-picker", str(source))

        assert page.locator("#consent-upload").is_visible()
        assert "agent-memory.pdf" in page.locator("#consent-upload").inner_text()
        assert page.locator("#authorize-upload").is_checked() is False
        assert errors == []
