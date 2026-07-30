"""Managed browser capture tests.

The doubles here deliberately mirror the real driver's contracts rather than a
convenient subset. An earlier `context.request` double accepted `max_redirects=0`
and returned a `location` header, while real Playwright *rejects* on redirect
with that setting, so a shortlink resolver that could never work in production
passed its tests. Shortlink unwrapping now goes through the guarded fetcher,
which is exercised here against a transport that redirects exactly as t.co does.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import httpx
import pytest

from steering.domain.models import SourceKind
from steering.ingestion.browser import (
    BrowserAuthenticationRequired,
    BrowserCaptureUnavailable,
    LoginOutcome,
    ManagedBrowserCapture,
)
from steering.ingestion.security import SafeFetcher, UnsafeSourceError

PAPER_URL = "https://arxiv.org/abs/2501.12948"


class RecordingGuard:
    def __init__(self) -> None:
        self.urls: list[str] = []

    async def validate_url(self, url: str) -> str:
        self.urls.append(url)
        return url

    def validate_connected_address(self, address: str) -> None:
        assert address == "93.184.216.34"


class BlockingGuard(RecordingGuard):
    async def validate_url(self, url: str) -> str:
        await super().validate_url(url)
        if "127.0.0.1" in url:
            raise UnsafeSourceError("blocked private fixture")
        return url


def shortlink_fetcher(destinations: dict[str, str]) -> SafeFetcher:
    """A guarded fetcher over a transport that redirects the way t.co does."""

    def handler(request: httpx.Request) -> httpx.Response:
        target = destinations.get(str(request.url))
        if target is not None:
            return httpx.Response(301, headers={"location": target})
        return httpx.Response(200, text="destination page")

    return SafeFetcher(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        guard=RecordingGuard(),  # type: ignore[arg-type]
    )


class FakeLocator:
    def __init__(self, count: int) -> None:
        self._count = count
        self.first = self

    async def count(self) -> int:
        return self._count

    async def click(self, *, timeout: int) -> None:
        del timeout


class FakePage:
    def __init__(
        self,
        payload: dict[str, Any],
        *,
        final_url: str | None = None,
        article_batches: list[list[dict[str, Any]]] | None = None,
        logged_in: bool = True,
        visible_post_ids: tuple[str, ...] = ("42",),
    ) -> None:
        self.payload = payload
        self.article_batches = article_batches
        self.article_batch_index = 0
        self.logged_in = logged_in
        self.visible_post_ids = visible_post_ids
        self.url = final_url or "about:blank"
        self.goto_calls: list[tuple[str, str, int]] = []
        self.evaluate_scripts: list[str] = []
        self.handlers: dict[str, Any] = {}
        self.closed = False
        self.default_timeout: int | None = None
        self.waited_ms = 0

    async def goto(self, url: str, *, wait_until: str, timeout: int) -> FakeResponse:
        self.goto_calls.append((url, wait_until, timeout))
        if self.url == "about:blank":
            self.url = url
        return FakeResponse(self.url)

    def set_default_timeout(self, timeout: int) -> None:
        self.default_timeout = timeout

    def locator(self, selector: str) -> FakeLocator:
        if selector in {
            '[data-testid="AppTabBar_Home_Link"]',
            '[data-testid="SideNav_NewTweet_Button"]',
            'a[href="/home"]',
        }:
            return FakeLocator(int(self.logged_in))
        visible = "time" in selector and any(
            f"/status/{post_id}" in selector for post_id in self.visible_post_ids
        )
        return FakeLocator(int(visible))

    def get_by_role(self, _role: str, *, name: str) -> FakeLocator:
        del name
        return FakeLocator(0)

    def on(self, event: str, handler: Any) -> None:
        self.handlers[event] = handler

    def is_closed(self) -> bool:
        return self.closed

    async def close(self) -> None:
        self.closed = True

    async def wait_for_timeout(self, milliseconds: int) -> None:
        # Real `wait_for_timeout` actually sleeps, and the polling loops are paced
        # by it. A double that returned immediately would let those loops spin.
        self.waited_ms += milliseconds
        await asyncio.sleep(milliseconds / 1000)

    async def evaluate(self, script: str, _arguments: dict[str, str] | None = None) -> Any:
        self.evaluate_scripts.append(script)
        if "window.scrollBy" in script:
            return None
        if self.article_batches is not None and "querySelectorAll('article')" in script:
            index = min(self.article_batch_index, len(self.article_batches) - 1)
            self.article_batch_index += 1
            return self.article_batches[index]
        return self.payload


class ClosingPage(FakePage):
    """A page the user closes part-way through polling, as a real user would."""

    def __init__(self, *args: Any, close_after: int = 1, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.close_after = close_after
        self.polls = 0

    def locator(self, selector: str) -> FakeLocator:
        self.polls += 1
        if self.polls > self.close_after:
            self.closed = True
        return super().locator(selector)


class FakeResponse:
    def __init__(self, url: str) -> None:
        self.url = url

    async def server_addr(self) -> dict[str, str]:
        return {"ipAddress": "93.184.216.34", "port": "443"}


class MissingPeerResponse(FakeResponse):
    async def server_addr(self) -> dict[str, str]:
        return {}


class FakeContext:
    def __init__(self, page: FakePage) -> None:
        self.pages = [page]
        self.closed = False
        self.routes: list[tuple[str, Any]] = []

    async def new_page(self) -> FakePage:
        return self.pages[0]

    async def close(self) -> None:
        self.closed = True

    async def route(self, pattern: str, handler: Any) -> None:
        self.routes.append((pattern, handler))


class FakeRoute:
    def __init__(self, url: str) -> None:
        self.request = SimpleNamespace(url=url)
        self.aborted: str | None = None
        self.continued = False

    async def abort(self, *, error_code: str) -> None:
        self.aborted = error_code

    async def continue_(self) -> None:
        self.continued = True


class FakeChromium:
    def __init__(self, context: FakeContext) -> None:
        self.context = context
        self.executable_path = __file__
        self.launches: list[tuple[str, dict[str, Any]]] = []

    async def launch_persistent_context(self, path: str, **options: Any) -> FakeContext:
        self.launches.append((path, options))
        return self.context


class FakePlaywrightManager:
    def __init__(self, chromium: FakeChromium) -> None:
        self.chromium = chromium

    async def __aenter__(self) -> SimpleNamespace:
        return SimpleNamespace(chromium=self.chromium)

    async def __aexit__(self, *_args: object) -> None:
        return None


def install_fake_playwright(monkeypatch: pytest.MonkeyPatch, chromium: FakeChromium) -> None:
    package = ModuleType("playwright")
    api = ModuleType("playwright.async_api")
    api.async_playwright = lambda: FakePlaywrightManager(chromium)  # type: ignore[attr-defined]
    package.async_api = api  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "playwright", package)
    monkeypatch.setitem(sys.modules, "playwright.async_api", api)


def x_thread_fixture() -> list[list[dict[str, Any]]]:
    """Three scroll batches, as X's virtualized timeline really delivers them."""

    root = {
        "post_id": "42",
        "handle": "researcher",
        "status_url": "https://x.com/researcher/status/42",
        "posted_at": "2026-07-16T12:00:00Z",
        "author_label": "Researcher",
        "text": "Root claim",
        "quoted_text": "",
        "links": [{"url": "https://x.com/researcher/status/42", "text": "", "title": None}],
        "media": [{"url": "https://pbs.twimg.com/paper.png", "alt": "First page of a technical paper"}],
    }
    self_reply = {
        "post_id": "43",
        "handle": "researcher",
        "status_url": "https://x.com/researcher/status/43",
        "posted_at": "2026-07-16T13:00:00Z",
        "author_label": "Researcher",
        "text": "Self reply with paper link",
        "quoted_text": "",
        # X shortens every outbound link and renders the destination without a
        # scheme, truncated with an ellipsis when it is long.
        "links": [{"url": "https://t.co/paper", "text": "arxiv.org/abs/2501.1294…", "title": None}],
        "media": [],
    }
    other_author = {
        "post_id": "99",
        "handle": "someone_else",
        "status_url": "https://x.com/someone_else/status/99",
        "posted_at": "2026-07-16T12:30:00Z",
        "author_label": "Someone Else",
        "text": "Other author's comment",
        "quoted_text": "",
        "links": [{"url": "https://github.com/unrelated/project", "text": "Unrelated"}],
        "media": [],
    }
    stale_post = {
        "post_id": "10",
        "handle": "researcher",
        "status_url": "https://x.com/researcher/status/10",
        "posted_at": "2026-07-13T10:00:00Z",
        "author_label": "Researcher",
        "text": "Old unrelated post",
        "quoted_text": "",
        "links": [{"url": "https://huggingface.co/unrelated/model", "text": "Old model"}],
        "media": [],
    }
    return [[root, other_author], [root, self_reply, other_author], [self_reply, stale_post]]


@pytest.mark.asyncio
async def test_browser_capture_requires_authorization_and_uses_isolated_child_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_root = tmp_path / "existing-browser-profile"
    guard = RecordingGuard()
    capture = ManagedBrowserCapture(
        profile_directory=requested_root,
        guard=guard,  # type: ignore[arg-type]
    )

    with pytest.raises(PermissionError, match="explicit user authorization"):
        await capture.capture("https://example.org/article")
    assert not requested_root.exists()
    assert guard.urls == []

    page = FakePage(
        {
            "title": "Public article",
            "text": "A visible article about evaluation.",
            "links": [{"url": "https://example.org/paper.pdf", "text": "paper"}],
            "media": [],
            "author": "Engineer",
        }
    )
    context = FakeContext(page)
    chromium = FakeChromium(context)
    install_fake_playwright(monkeypatch, chromium)

    resolved = await capture.capture("https://example.org/article", authorized=True)

    launched_path, options = chromium.launches[0]
    assert Path(launched_path) == requested_root / "steering-managed-profile"
    assert Path(launched_path) != requested_root
    assert options["headless"] is False
    assert context.closed is True
    assert resolved.source_kind is SourceKind.WEBPAGE
    assert resolved.extraction_method == "authorized_visible_browser"
    assert resolved.outbound_urls == ["https://example.org/paper.pdf"]
    assert context.routes[0][0] == "**/*"


@pytest.mark.asyncio
async def test_browser_capture_revalidates_final_navigation_and_blocks_unsafe_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = BlockingGuard()
    page = FakePage(
        {"title": "Redirected", "text": "should not be captured"},
        final_url="http://127.0.0.1/private",
    )
    context = FakeContext(page)
    install_fake_playwright(monkeypatch, FakeChromium(context))
    capture = ManagedBrowserCapture(
        profile_directory=tmp_path,
        guard=guard,  # type: ignore[arg-type]
    )

    with pytest.raises(BrowserCaptureUnavailable, match="UnsafeSourceError"):
        await capture.capture("https://example.org/redirect", authorized=True)
    assert guard.urls[-1] == "http://127.0.0.1/private"

    handler = context.routes[0][1]
    unsafe_route = FakeRoute("http://127.0.0.1/metadata")
    await handler(unsafe_route)
    assert unsafe_route.aborted == "blockedbyclient"
    assert unsafe_route.continued is False

    safe_route = FakeRoute("https://cdn.example.org/style.css")
    await handler(safe_route)
    assert safe_route.aborted is None
    assert safe_route.continued is True


@pytest.mark.asyncio
async def test_response_guard_does_not_close_login_for_subresource_without_peer_metadata(
    tmp_path: Path,
) -> None:
    page = FakePage({})
    capture = ManagedBrowserCapture(
        profile_directory=tmp_path,
        guard=RecordingGuard(),  # type: ignore[arg-type]
    )
    violations = capture._install_response_guard(page)

    await page.handlers["response"](MissingPeerResponse("https://abs.twimg.com/client.js"))

    assert violations == []
    assert page.closed is False


@pytest.mark.asyncio
async def test_authorized_x_capture_bundles_self_replies_and_unwraps_shortlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = FakePage({}, article_batches=x_thread_fixture())
    context = FakeContext(page)
    chromium = FakeChromium(context)
    install_fake_playwright(monkeypatch, chromium)
    capture = ManagedBrowserCapture(
        profile_directory=tmp_path,
        guard=RecordingGuard(),  # type: ignore[arg-type]
        link_resolver=shortlink_fetcher({"https://t.co/paper": PAPER_URL}),
    )

    resolved = await capture.capture("https://x.com/i/status/42", authorized=True)

    launched_path, options = chromium.launches[0]
    assert Path(launched_path) == tmp_path / "steering-managed-profile"
    assert options["channel"] == "chrome"
    assert options["no_viewport"] is True
    # The X path must carry the same network guarantees as every other capture.
    assert context.routes[0][0] == "**/*"

    assert resolved.source_kind is SourceKind.X
    assert resolved.canonical_url == "https://x.com/researcher/status/42"
    assert resolved.metadata["capture_scope"] == "author_thread"
    assert resolved.metadata["bundled_self_replies"] == 1
    assert resolved.text == "Root claim\n\n---\n\nSelf reply with paper link"
    assert "Other author's comment" not in resolved.text
    assert "Old unrelated post" not in resolved.text
    assert resolved.published_at is not None
    # The destination, not the shortlink: the whole point of the capture.
    assert resolved.outbound_urls == [PAPER_URL]
    assert resolved.media_urls == ["https://pbs.twimg.com/paper.png"]
    assert resolved.metadata["media_inclusion_candidates"] == [
        {"url": "https://pbs.twimg.com/paper.png", "alt": "First page of a technical paper"}
    ]


@pytest.mark.asyncio
async def test_x_capture_without_a_signed_in_session_fails_fast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capture must never hold a request open waiting for a human to sign in."""

    page = FakePage({}, article_batches=[[]], logged_in=False, visible_post_ids=())
    context = FakeContext(page)
    install_fake_playwright(monkeypatch, FakeChromium(context))
    capture = ManagedBrowserCapture(
        profile_directory=tmp_path,
        guard=RecordingGuard(),  # type: ignore[arg-type]
        capture_timeout_seconds=2,
        # A capture must never inherit the human-scale login budget.
        login_timeout_seconds=900,
    )

    started = time.monotonic()
    with pytest.raises(BrowserAuthenticationRequired, match="not signed in to X"):
        await capture.capture("https://x.com/researcher/status/42", authorized=True)

    assert time.monotonic() - started < 10
    assert context.closed is True


@pytest.mark.asyncio
async def test_x_capture_rejects_non_post_urls_before_launching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = FakeContext(FakePage({}))
    chromium = FakeChromium(context)
    install_fake_playwright(monkeypatch, chromium)
    capture = ManagedBrowserCapture(
        profile_directory=tmp_path,
        guard=RecordingGuard(),  # type: ignore[arg-type]
    )

    with pytest.raises(BrowserCaptureUnavailable, match="needs a post URL"):
        await capture.capture("https://x.com/researcher", authorized=True)
    assert chromium.launches == []


@pytest.mark.asyncio
async def test_open_login_requires_authorization_and_reports_a_detected_sign_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = FakePage({}, logged_in=True)
    context = FakeContext(page)
    chromium = FakeChromium(context)
    install_fake_playwright(monkeypatch, chromium)
    capture = ManagedBrowserCapture(
        profile_directory=tmp_path,
        guard=RecordingGuard(),  # type: ignore[arg-type]
    )

    with pytest.raises(PermissionError, match="explicit authorization"):
        await capture.open_login("https://x.com/i/flow/login")

    outcome = await capture.open_login("https://x.com/i/flow/login", authorized=True)

    assert outcome is LoginOutcome.SIGNED_IN
    launched_path, options = chromium.launches[0]
    assert Path(launched_path) == tmp_path / "steering-managed-profile"
    assert options["channel"] == "chrome"
    assert page.goto_calls[0][0] == "https://x.com/i/flow/login"
    assert context.closed is True


@pytest.mark.asyncio
async def test_open_login_treats_a_closed_window_as_a_finished_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closing the window is how users finish; it must not be reported as an error."""

    page = ClosingPage({}, logged_in=False, close_after=1)
    context = FakeContext(page)
    install_fake_playwright(monkeypatch, FakeChromium(context))
    capture = ManagedBrowserCapture(
        profile_directory=tmp_path,
        guard=RecordingGuard(),  # type: ignore[arg-type]
    )

    outcome = await capture.open_login("https://x.com/i/flow/login", authorized=True)

    assert outcome is LoginOutcome.CLOSED_BEFORE_SIGN_IN
    assert context.closed is True


@pytest.mark.parametrize(
    ("displayed", "expected"),
    [
        # X renders destinations without a scheme; the old check required one and
        # therefore never matched real markup.
        ("arxiv.org/abs/2501.12948", "https://arxiv.org/abs/2501.12948"),
        ("https://github.com/example/repo", "https://github.com/example/repo"),
        ("arxiv.org/abs/2501.1294…", None),
        ("Paper", None),
        ("x.com/researcher/status/42", None),
        ("pic.twitter.com/abc", None),
        ("", None),
        (None, None),
    ],
)
def test_displayed_anchor_text_is_read_the_way_x_renders_it(
    displayed: str | None,
    expected: str | None,
) -> None:
    assert ManagedBrowserCapture._displayed_external_url(displayed) == expected


class FailingPage(FakePage):
    async def goto(self, url: str, *, wait_until: str, timeout: int) -> FakeResponse:
        raise TimeoutError("fixture timeout")


@pytest.mark.asyncio
async def test_capture_wraps_browser_failures_without_leaking_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = FailingPage({})
    context = FakeContext(page)
    chromium = FakeChromium(context)
    install_fake_playwright(monkeypatch, chromium)
    capture = ManagedBrowserCapture(
        profile_directory=tmp_path,
        guard=RecordingGuard(),  # type: ignore[arg-type]
    )

    with pytest.raises(BrowserCaptureUnavailable, match="TimeoutError") as error:
        await capture.capture("https://example.org", authorized=True)
    assert "fixture timeout" not in str(error.value)
    assert context.closed is True


@pytest.mark.asyncio
async def test_capture_can_run_headless_so_unattended_batches_need_no_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A capture only replays the session sign-in already stored, so it needs no window."""

    page = FakePage({}, article_batches=x_thread_fixture())
    context = FakeContext(page)
    chromium = FakeChromium(context)
    install_fake_playwright(monkeypatch, chromium)
    capture = ManagedBrowserCapture(
        profile_directory=tmp_path,
        guard=RecordingGuard(),  # type: ignore[arg-type]
        link_resolver=shortlink_fetcher({"https://t.co/paper": PAPER_URL}),
        headless=True,
    )

    resolved = await capture.capture("https://x.com/researcher/status/42", authorized=True)

    _path, options = chromium.launches[0]
    assert options["headless"] is True
    # Headless has no window to maximize, and the thread only renders and scrolls
    # inside a viewport with real height.
    assert options["viewport"] == {"width": 1440, "height": 1000}
    assert "no_viewport" not in options
    assert "--start-maximized" not in options["args"]
    assert resolved.outbound_urls == [PAPER_URL]


@pytest.mark.asyncio
async def test_sign_in_stays_visible_even_when_captures_are_headless(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A human types the credentials, so this step can never be headless."""

    page = FakePage({}, logged_in=True)
    context = FakeContext(page)
    chromium = FakeChromium(context)
    install_fake_playwright(monkeypatch, chromium)
    capture = ManagedBrowserCapture(
        profile_directory=tmp_path,
        guard=RecordingGuard(),  # type: ignore[arg-type]
        headless=True,
    )

    outcome = await capture.open_login("https://x.com/i/flow/login", authorized=True)

    _path, options = chromium.launches[0]
    assert outcome is LoginOutcome.SIGNED_IN
    assert options["headless"] is False
    assert options["no_viewport"] is True


@pytest.mark.asyncio
async def test_visible_capture_remains_the_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = FakePage({}, article_batches=x_thread_fixture())
    chromium = FakeChromium(FakeContext(page))
    install_fake_playwright(monkeypatch, chromium)
    capture = ManagedBrowserCapture(
        profile_directory=tmp_path,
        guard=RecordingGuard(),  # type: ignore[arg-type]
    )

    await capture.capture("https://x.com/researcher/status/42", authorized=True)

    _path, options = chromium.launches[0]
    assert options["headless"] is False


@pytest.mark.asyncio
async def test_open_login_skips_the_window_when_a_session_already_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chromium = FakeChromium(FakeContext(FakePage({})))
    install_fake_playwright(monkeypatch, chromium)
    capture = ManagedBrowserCapture(
        profile_directory=tmp_path,
        guard=RecordingGuard(),  # type: ignore[arg-type]
    )
    session_file = capture.profile_directory / "Default" / "Network"
    session_file.mkdir(parents=True, exist_ok=True)
    (session_file / "Cookies").write_bytes(b"stub cookie store")

    outcome = await capture.open_login("https://x.com/i/flow/login", authorized=True)

    assert outcome is LoginOutcome.ALREADY_SIGNED_IN
    # Nothing was launched, so there is no window to flash and vanish.
    assert chromium.launches == []


@pytest.mark.asyncio
async def test_forcing_sign_in_opens_the_window_despite_a_stored_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chromium = FakeChromium(FakeContext(FakePage({}, logged_in=True)))
    install_fake_playwright(monkeypatch, chromium)
    capture = ManagedBrowserCapture(
        profile_directory=tmp_path,
        guard=RecordingGuard(),  # type: ignore[arg-type]
    )
    cookies = capture.profile_directory / "Default" / "Network"
    cookies.mkdir(parents=True, exist_ok=True)
    (cookies / "Cookies").write_bytes(b"stub cookie store")

    outcome = await capture.open_login("https://x.com/i/flow/login", authorized=True, force=True)

    assert outcome is LoginOutcome.SIGNED_IN
    assert len(chromium.launches) == 1


def test_a_fresh_profile_reports_no_stored_session(tmp_path: Path) -> None:
    capture = ManagedBrowserCapture(profile_directory=tmp_path)

    assert capture.has_stored_session() is False
