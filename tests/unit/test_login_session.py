from __future__ import annotations

import asyncio

import pytest

from steering.ingestion.browser import (
    BrowserCaptureUnavailable,
    BrowserDependencyUnavailable,
    LoginOutcome,
)
from steering.ingestion.login_session import BrowserLoginSession, LoginState


class StubLogin:
    def __init__(self, outcome: LoginOutcome | Exception, *, delay: float = 0.0) -> None:
        self.outcome = outcome
        self.delay = delay
        self.urls: list[str] = []
        self.forced: list[bool] = []

    async def open_login(self, url: str, *, authorized: bool = False, force: bool = False) -> LoginOutcome:
        assert authorized is True
        self.urls.append(url)
        self.forced.append(force)
        if self.delay:
            await asyncio.sleep(self.delay)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


async def settle(session: BrowserLoginSession) -> LoginState:
    for _ in range(200):
        status = session.status()
        if status.state is not LoginState.RUNNING:
            return status.state
        await asyncio.sleep(0.01)
    raise AssertionError("sign-in never settled")


@pytest.mark.asyncio
async def test_a_new_session_reports_idle_before_anything_starts() -> None:
    session = BrowserLoginSession(StubLogin(LoginOutcome.SIGNED_IN))

    assert session.status().state is LoginState.IDLE


@pytest.mark.asyncio
async def test_start_returns_immediately_and_settles_to_signed_in() -> None:
    login = StubLogin(LoginOutcome.SIGNED_IN, delay=0.05)
    session = BrowserLoginSession(login)

    status = session.start("https://x.com/i/flow/login")

    # The caller is never made to wait for a human-paced sign-in.
    assert status.state is LoginState.RUNNING
    assert await settle(session) is LoginState.SIGNED_IN
    assert session.status().succeeded is True
    assert login.urls == ["https://x.com/i/flow/login"]


@pytest.mark.asyncio
async def test_a_closed_window_settles_without_being_treated_as_a_failure() -> None:
    session = BrowserLoginSession(StubLogin(LoginOutcome.CLOSED_BEFORE_SIGN_IN))
    session.start("https://x.com/i/flow/login")

    state = await settle(session)

    assert state is LoginState.CLOSED_BEFORE_SIGN_IN
    assert "capture will still work" in session.status().message


@pytest.mark.asyncio
async def test_a_missing_browser_dependency_is_reported_as_unavailable() -> None:
    session = BrowserLoginSession(StubLogin(BrowserDependencyUnavailable("no chromium")))
    session.start("https://x.com/i/flow/login")

    assert await settle(session) is LoginState.UNAVAILABLE
    assert "Install the browser extra" in session.status().message


@pytest.mark.asyncio
async def test_a_capture_failure_keeps_its_reason_visible() -> None:
    session = BrowserLoginSession(StubLogin(BrowserCaptureUnavailable("profile is locked")))
    session.start("https://x.com/i/flow/login")

    assert await settle(session) is LoginState.FAILED
    assert "profile is locked" in session.status().message


@pytest.mark.asyncio
async def test_a_concurrent_sign_in_is_refused_while_one_is_running() -> None:
    session = BrowserLoginSession(StubLogin(LoginOutcome.SIGNED_IN, delay=0.2))
    session.start("https://x.com/i/flow/login")

    # Two managed browsers would contend for the same locked profile directory.
    with pytest.raises(RuntimeError, match="already in progress"):
        session.start("https://www.linkedin.com/login")

    assert await settle(session) is LoginState.SIGNED_IN


@pytest.mark.asyncio
async def test_a_settled_session_can_start_again() -> None:
    login = StubLogin(LoginOutcome.SIGNED_IN)
    session = BrowserLoginSession(login)
    session.start("https://x.com/i/flow/login")
    await settle(session)

    session.start("https://www.linkedin.com/login")

    assert await settle(session) is LoginState.SIGNED_IN
    assert login.urls == ["https://x.com/i/flow/login", "https://www.linkedin.com/login"]


@pytest.mark.asyncio
async def test_shutdown_cancels_an_in_flight_sign_in() -> None:
    session = BrowserLoginSession(StubLogin(LoginOutcome.SIGNED_IN, delay=30))
    session.start("https://x.com/i/flow/login")

    await asyncio.wait_for(session.aclose(), timeout=5)

    assert session.status().state is not LoginState.RUNNING


@pytest.mark.asyncio
async def test_an_existing_session_settles_without_opening_a_window() -> None:
    """A window that opens and shuts a second later reads as a crash, not a success."""

    login = StubLogin(LoginOutcome.ALREADY_SIGNED_IN)
    session = BrowserLoginSession(login)
    session.start("https://x.com/i/flow/login")

    assert await settle(session) is LoginState.ALREADY_SIGNED_IN
    status = session.status()
    assert status.succeeded is True
    assert "no browser was opened" in status.message
    assert login.forced == [False]


@pytest.mark.asyncio
async def test_signing_in_again_is_forced_through_to_the_browser() -> None:
    login = StubLogin(LoginOutcome.SIGNED_IN)
    session = BrowserLoginSession(login)

    session.start("https://x.com/i/flow/login", force=True)

    assert await settle(session) is LoginState.SIGNED_IN
    assert login.forced == [True]


class SessionAwareLogin(StubLogin):
    def __init__(self, outcome: LoginOutcome, *, stored: bool) -> None:
        super().__init__(outcome)
        self.stored = stored

    def has_stored_session(self) -> bool:
        return self.stored


@pytest.mark.asyncio
async def test_an_existing_session_answers_in_the_response_without_polling() -> None:
    """Reporting 'waiting for sign-in' then correcting it shows a state never true."""

    login = SessionAwareLogin(LoginOutcome.SIGNED_IN, stored=True)
    session = BrowserLoginSession(login)

    status = session.start("https://x.com/i/flow/login")

    assert status.state is LoginState.ALREADY_SIGNED_IN
    assert status.running is False
    assert status.succeeded is True
    # The browser was never asked to do anything.
    assert login.urls == []


@pytest.mark.asyncio
async def test_no_stored_session_still_runs_the_sign_in_in_the_background() -> None:
    login = SessionAwareLogin(LoginOutcome.SIGNED_IN, stored=False)
    session = BrowserLoginSession(login)

    status = session.start("https://x.com/i/flow/login")

    assert status.state is LoginState.RUNNING
    assert await settle(session) is LoginState.SIGNED_IN
    assert login.urls == ["https://x.com/i/flow/login"]


@pytest.mark.asyncio
async def test_forcing_bypasses_the_stored_session_fast_path() -> None:
    login = SessionAwareLogin(LoginOutcome.SIGNED_IN, stored=True)
    session = BrowserLoginSession(login)

    status = session.start("https://x.com/i/flow/login", force=True)

    assert status.state is LoginState.RUNNING
    assert await settle(session) is LoginState.SIGNED_IN
    assert login.forced == [True]
