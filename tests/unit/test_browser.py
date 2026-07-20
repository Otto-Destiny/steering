from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from steering.domain.models import SourceKind
from steering.ingestion.browser import BrowserCaptureUnavailable, ManagedBrowserCapture
from steering.ingestion.security import UnsafeSourceError


async def _async_value(value: bool) -> bool:
    return value


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


class FakeMouse:
    def __init__(self) -> None:
        self.scrolls = 0

    async def wheel(self, _x: int, _y: int) -> None:
        self.scrolls += 1


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
    ) -> None:
        self.payload = payload
        self.article_batches = article_batches
        self.article_batch_index = 0
        self.url = final_url or "about:blank"
        self.goto_calls: list[tuple[str, str, int]] = []
        self.evaluate_scripts: list[str] = []
        self.mouse = FakeMouse()
        self.handlers: dict[str, Any] = {}
        self.closed = False
        self.default_timeout: int | None = None

    async def goto(self, url: str, *, wait_until: str, timeout: int) -> FakeResponse:
        self.goto_calls.append((url, wait_until, timeout))
        if self.url == "about:blank":
            self.url = url
        return FakeResponse(self.url)

    def set_default_timeout(self, timeout: int) -> None:
        self.default_timeout = timeout

    def locator(self, selector: str) -> FakeLocator:
        visible = selector in {
            '[data-testid="AppTabBar_Home_Link"]',
            '[data-testid="SideNav_NewTweet_Button"]',
            'a[href="/home"]',
        } or ('a[href*="/status/42"]' in selector and "time" in selector)
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

    async def wait_for_timeout(self, _milliseconds: int) -> None:
        return None

    async def evaluate(self, script: str, _arguments: dict[str, str] | None = None) -> Any:
        self.evaluate_scripts.append(script)
        if "window.scrollBy" in script:
            return None
        if self.article_batches is not None and "querySelectorAll('article')" in script:
            index = min(self.article_batch_index, len(self.article_batches) - 1)
            self.article_batch_index += 1
            return self.article_batches[index]
        return self.payload


class FakeResponse:
    def __init__(self, url: str) -> None:
        self.url = url

    async def server_addr(self) -> dict[str, str]:
        return {"ipAddress": "93.184.216.34", "port": "443"}


class MissingPeerResponse(FakeResponse):
    async def server_addr(self) -> dict[str, str]:
        return {}


class FakeRequestContext:
    def __init__(self, redirects: dict[str, str] | None = None) -> None:
        self.redirects = redirects or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def get(self, url: str, **options: Any) -> SimpleNamespace:
        self.calls.append((url, options))
        redirected = self.redirects.get(url)
        return SimpleNamespace(
            url=url,
            headers={"location": redirected} if redirected else {},
        )


class FakeContext:
    def __init__(
        self,
        page: FakePage,
        *,
        redirects: dict[str, str] | None = None,
    ) -> None:
        self.pages = [page]
        self.closed = False
        self.events: list[str] = []
        self.routes: list[tuple[str, Any]] = []
        self.request = FakeRequestContext(redirects)

    async def new_page(self) -> FakePage:
        return self.pages[0]

    async def close(self) -> None:
        self.closed = True

    async def wait_for_event(self, event: str) -> None:
        self.events.append(event)

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
            "links": ["https://example.org/paper.pdf"],
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
    assert guard.urls == ["https://example.org/article", "https://example.org/article"]
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
async def test_authorized_x_capture_bundles_mocked_self_replies_without_live_browser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = {
        "post_id": "42",
        "handle": "researcher",
        "status_url": "https://x.com/researcher/status/42",
        "posted_at": "2026-07-16T12:00:00Z",
        "author_label": "Researcher",
        "text": "Root claim",
        "links": [
            {
                "url": "https://x.com/researcher/status/42",
                "text": "",
                "title": None,
                "aria_label": None,
            }
        ],
        "media": [
            {
                "url": "https://pbs.twimg.com/paper.png",
                "alt": "First page of a technical paper",
            }
        ],
    }
    self_reply = {
        "post_id": "43",
        "handle": "researcher",
        "status_url": "https://x.com/researcher/status/43",
        "posted_at": "2026-07-16T13:00:00Z",
        "author_label": "Researcher",
        "text": "Self reply with paper link",
        "links": [
            {
                "url": "https://t.co/paper",
                "text": "Paper",
                "title": None,
                "aria_label": None,
            }
        ],
        "media": [],
    }
    other_author = {
        "post_id": "99",
        "handle": "someone_else",
        "status_url": "https://x.com/someone_else/status/99",
        "posted_at": "2026-07-16T12:30:00Z",
        "author_label": "Someone Else",
        "text": "Other author's comment",
        "links": [
            {
                "url": "https://github.com/unrelated/project",
                "text": "Unrelated",
                "title": None,
                "aria_label": None,
            }
        ],
        "media": [],
    }
    old_same_author_post = {
        "post_id": "10",
        "handle": "researcher",
        "status_url": "https://x.com/researcher/status/10",
        "posted_at": "2026-07-13T10:00:00Z",
        "author_label": "Researcher",
        "text": "Old unrelated post",
        "links": [
            {
                "url": "https://huggingface.co/unrelated/model",
                "text": "Old model",
                "title": None,
                "aria_label": None,
            }
        ],
        "media": [],
    }
    page = FakePage(
        {},
        article_batches=[
            [root, other_author],
            [root, self_reply, other_author],
            [self_reply, old_same_author_post],
        ],
    )
    context = FakeContext(
        page,
        redirects={"https://t.co/paper": "https://arxiv.org/abs/42"},
    )
    chromium = FakeChromium(context)
    install_fake_playwright(monkeypatch, chromium)
    guard = RecordingGuard()
    capture = ManagedBrowserCapture(
        profile_directory=tmp_path,
        guard=guard,  # type: ignore[arg-type]
    )

    resolved = await capture.capture("https://x.com/i/status/42", authorized=True)

    launched_path, options = chromium.launches[0]
    assert Path(launched_path) == tmp_path / "steering-managed-profile"
    assert options["channel"] == "chrome"
    assert options["no_viewport"] is True
    assert options["locale"] == "en-US"
    assert options["args"] == [
        "--disable-blink-features=AutomationControlled",
        "--start-maximized",
    ]
    assert context.routes == []

    assert resolved.source_kind is SourceKind.X
    assert resolved.metadata["bundled_self_replies"] == 1
    assert resolved.text == "Root claim\n\n---\n\nSelf reply with paper link"
    assert "Other author's comment" not in resolved.text
    assert "Old unrelated post" not in resolved.text
    assert resolved.outbound_urls == ["https://arxiv.org/abs/42"]
    assert resolved.media_urls == ["https://pbs.twimg.com/paper.png"]
    assert resolved.metadata["media_inclusion_candidates"] == [
        {"url": "https://pbs.twimg.com/paper.png", "alt": "First page of a technical paper"}
    ]
    assert context.request.calls == [
        (
            "https://t.co/paper",
            {
                "fail_on_status_code": False,
                "max_redirects": 0,
                "timeout": 20_000,
            },
        )
    ]
    assert "https://arxiv.org/abs/42" in guard.urls


@pytest.mark.asyncio
async def test_open_login_requires_authorization_and_waits_in_isolated_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = FakePage({})
    context = FakeContext(page)
    chromium = FakeChromium(context)
    install_fake_playwright(monkeypatch, chromium)
    capture = ManagedBrowserCapture(
        profile_directory=tmp_path,
        guard=RecordingGuard(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr(capture, "_x_logged_in", lambda _page: _async_value(True))

    with pytest.raises(PermissionError, match="explicit authorization"):
        await capture.open_login("https://x.com/login")
    await capture.open_login("https://x.com/login", authorized=True)

    launched_path, options = chromium.launches[0]
    assert Path(launched_path) == tmp_path / "steering-managed-profile"
    assert options["channel"] == "chrome"
    assert options["args"] == [
        "--disable-blink-features=AutomationControlled",
        "--start-maximized",
    ]
    assert options["no_viewport"] is True
    assert "viewport" not in options
    assert options["locale"] == "en-US"
    assert page.goto_calls[0][0] == "https://x.com/login"
    assert context.closed is True


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
