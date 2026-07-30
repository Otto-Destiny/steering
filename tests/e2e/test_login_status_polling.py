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

    def has_stored_session(self) -> bool:
        return self.stored_session

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
            page.check('input[name="authorize_login"]')
            yield page, polls, errors
        finally:
            browser.close()


def test_the_panel_keeps_polling_until_sign_in_completes() -> None:
    login = StubLogin(stored_session=False)
    with _serving(login) as base_url, _panel(base_url) as (page, polls, errors):
        page.click('button:has-text("Open managed login browser")')

        page.wait_for_selector("#browser-login-result >> text=/Waiting for sign-in/", timeout=5_000)
        page.wait_for_selector("#browser-login-result >> text=/Signed in/", timeout=20_000)

        # Polling that dies after one response is the defect this guards.
        assert len(polls) >= 2
        assert errors == []
        # The container the panel swaps into must survive every poll.
        assert page.locator("#browser-login-result").count() == 1


def test_an_existing_session_answers_without_polling_or_a_waiting_message() -> None:
    login = StubLogin(stored_session=True)
    with _serving(login) as base_url, _panel(base_url) as (page, polls, errors):
        page.click('button:has-text("Open managed login browser")')
        page.wait_for_selector("#browser-login-result >> text=/already exists/", timeout=10_000)

        message = page.locator("#browser-login-result").inner_text()
        assert "Waiting for sign-in" not in message
        assert polls == []
        assert errors == []
        # No browser is opened for a session that is already there.
        assert login.opened == []
