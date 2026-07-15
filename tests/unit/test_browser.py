from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from steering.domain.models import SourceKind
from steering.ingestion.browser import BrowserCaptureUnavailable, ManagedBrowserCapture
from steering.ingestion.security import UnsafeSourceError


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


class FakePage:
    def __init__(self, payload: dict[str, Any], *, final_url: str | None = None) -> None:
        self.payload = payload
        self.url = final_url or "about:blank"
        self.goto_calls: list[tuple[str, str, int]] = []
        self.evaluate_scripts: list[str] = []
        self.mouse = FakeMouse()
        self.handlers: dict[str, Any] = {}
        self.closed = False

    async def goto(self, url: str, *, wait_until: str, timeout: int) -> FakeResponse:
        self.goto_calls.append((url, wait_until, timeout))
        if self.url == "about:blank":
            self.url = url
        return FakeResponse(self.url)

    def on(self, event: str, handler: Any) -> None:
        self.handlers[event] = handler

    def is_closed(self) -> bool:
        return self.closed

    async def close(self) -> None:
        self.closed = True

    async def wait_for_timeout(self, _milliseconds: int) -> None:
        return None

    async def evaluate(self, _script: str, _arguments: dict[str, str] | None = None) -> dict[str, Any]:
        self.evaluate_scripts.append(_script)
        return self.payload


class FakeResponse:
    def __init__(self, url: str) -> None:
        self.url = url

    async def server_addr(self) -> dict[str, str]:
        return {"ipAddress": "93.184.216.34", "port": "443"}


class FakeContext:
    def __init__(self, page: FakePage) -> None:
        self.pages = [page]
        self.closed = False
        self.events: list[str] = []
        self.routes: list[tuple[str, Any]] = []

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
async def test_authorized_x_capture_bundles_mocked_self_replies_without_live_browser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = FakePage(
        {
            "title": "Researcher - thread",
            "text": "Root claim\n\n---\n\nSelf reply with paper link",
            "links": ["https://arxiv.org/abs/42", "https://arxiv.org/abs/42"],
            "media": [
                {"url": "https://pbs.twimg.com/paper.png", "alt": "First page of a technical paper"},
                {"url": "https://pbs.twimg.com/avatar.png", "alt": "avatar"},
            ],
            "author": "researcher",
            "bundled_self_replies": 1,
        }
    )
    context = FakeContext(page)
    chromium = FakeChromium(context)
    install_fake_playwright(monkeypatch, chromium)
    capture = ManagedBrowserCapture(
        profile_directory=tmp_path,
        guard=RecordingGuard(),  # type: ignore[arg-type]
    )

    resolved = await capture.capture("https://x.com/researcher/status/42", authorized=True)
    assert page.mouse.scrolls == 4
    assert "rootPathParts" in page.evaluate_scripts[0]
    assert "startsWith(`${authorPath}/status/`)" in page.evaluate_scripts[0]
    assert resolved.source_kind is SourceKind.X
    assert resolved.metadata["bundled_self_replies"] == 1
    assert resolved.outbound_urls == ["https://arxiv.org/abs/42"]
    assert resolved.media_urls == [
        "https://pbs.twimg.com/paper.png",
        "https://pbs.twimg.com/avatar.png",
    ]
    assert resolved.metadata["media_inclusion_candidates"] == [
        {"url": "https://pbs.twimg.com/paper.png", "alt": "First page of a technical paper"}
    ]


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

    with pytest.raises(PermissionError, match="explicit authorization"):
        await capture.open_login("https://x.com/login")
    await capture.open_login("https://x.com/login", authorized=True)

    assert Path(chromium.launches[0][0]) == tmp_path / "steering-managed-profile"
    assert page.goto_calls[0][0] == "https://x.com/login"
    assert context.events == ["close"]


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
