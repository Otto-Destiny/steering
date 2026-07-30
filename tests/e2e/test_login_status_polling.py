"""Drive the managed-login panel in real Chromium.

Sign-in status is reported by htmx polling, which is browser behaviour a server
response cannot prove. This caught a defect no server-side assertion could: the
polling element inherits `hx-target` from the enclosing form, so `outerHTML`
replaced the *result container* rather than the element. The container vanished,
the next poll raised `htmx:targetError`, and the panel froze on "Waiting for
sign-in..." while the sign-in had actually finished.
"""

from __future__ import annotations

import contextlib
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
import uvicorn
from tests.web.test_app import (
    FakeEngine,
    FakeIngestion,
    FakeProviders,
    FakeRepository,
    FakeResolvedIngestion,
)

from steering.ingestion.browser import LoginOutcome
from steering.ingestion.login_session import BrowserLoginSession
from steering.web import create_web_app

pytestmark = pytest.mark.e2e

SIGN_IN_SECONDS = 3.0


class StubLogin:
    """A human-paced sign-in, so the panel must poll rather than answer at once."""

    def __init__(self, *, stored_session: bool) -> None:
        self.stored_session = stored_session
        self.opened: list[str] = []
        self.signed_in_hosts = {"x.com"} if stored_session else set()

    def has_stored_session(self) -> bool:
        return self.stored_session

    def is_signed_in_to(self, host: str) -> bool:
        return any(host == saved or host.endswith("." + saved) for saved in self.signed_in_hosts)

    async def capture(self, url: str, *, authorized: bool = False) -> Any:
        raise NotImplementedError

    async def open_login(
        self,
        url: str,
        *,
        authorized: bool = False,
        force: bool = False,
    ) -> LoginOutcome:
        import asyncio

        self.opened.append(url)
        await asyncio.sleep(SIGN_IN_SECONDS)
        self.stored_session = True
        self.signed_in_hosts.add(urlsplit(url).hostname or "")
        return LoginOutcome.SIGNED_IN


@contextlib.contextmanager
def _serving(login: StubLogin) -> Iterator[str]:
    repository = FakeRepository()
    app = create_web_app(
        engine=FakeEngine(repository),  # type: ignore[arg-type]
        ingestion=FakeIngestion(repository),  # type: ignore[arg-type]
        repository=repository,  # type: ignore[arg-type]
        provider_settings=FakeProviders(),
        resolved_ingestion=FakeResolvedIngestion(repository),
        browser_capture=login,
        login_session=BrowserLoginSession(login),
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
            raise RuntimeError("the login fixture server did not start")
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@contextlib.contextmanager
def _panel(base_url: str) -> Iterator[tuple[Any, list[str], list[str]]]:
    playwright_api = pytest.importorskip("playwright.sync_api")
    with playwright_api.sync_playwright() as runtime:
        if not Path(runtime.chromium.executable_path).is_file():
            pytest.skip("the managed Chromium runtime is not installed")
        browser = runtime.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            polls: list[str] = []
            errors: list[str] = []
            page.on(
                "request",
                lambda r: polls.append(r.url) if "/browser/login/status" in r.url else None,
            )
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.goto(f"{base_url}/add", wait_until="load")
            yield page, polls, errors
        finally:
            browser.close()


def test_the_panel_keeps_polling_until_sign_in_completes() -> None:
    login = StubLogin(stored_session=False)
    with _serving(login) as base_url, _panel(base_url) as (page, polls, errors):
        ids = page.locator("[id]").evaluate_all("nodes => nodes.map(node => node.id)")
        assert len(ids) == len(set(ids))

        page.click('[data-source-login="x.com"] button:has-text("Sign in")')

        page.wait_for_selector("#browser-login-result >> text=/Waiting for sign-in/", timeout=5_000)
        page.wait_for_selector("#browser-login-result >> text=/Signed in/", timeout=20_000)
        page.wait_for_selector('[data-source-host="x.com"][data-source-signed-in="true"]')

        # Polling that dies after one response is the defect this guards.
        assert len(polls) >= 2
        assert errors == []
        # The container the panel swaps into must survive every poll.
        assert page.locator("#browser-login-result").count() == 1
        assert page.locator('[data-source-host="x.com"] button').inner_text() == "Sign out"
        assert page.locator('[data-source-host="linkedin.com"] button').inner_text() == "Sign in"


def test_an_existing_session_is_shown_rather_than_discovered_by_clicking() -> None:
    """A signed-in platform states itself, so nothing has to be tried to find out.

    The panel used to offer sign-in regardless and answer "already signed in"
    afterwards. Reporting the session up front is why no browser is opened and
    nothing polls here.
    """

    login = StubLogin(stored_session=True)
    with _serving(login) as base_url, _panel(base_url) as (page, polls, errors):
        signed_in = page.locator('[data-source-host="x.com"]')
        assert signed_in.get_attribute("data-source-signed-in") == "true"
        assert "signed in" in signed_in.inner_text()
        assert signed_in.locator("button").inner_text() == "Sign out"
        # The only way back is deliberate, so no sign-in action is offered here.
        assert page.locator('[data-source-login="x.com"]').count() == 0

        assert page.locator('[data-source-host="linkedin.com"] button').inner_text() == "Sign in"
        assert polls == []
        assert errors == []
        assert login.opened == []
