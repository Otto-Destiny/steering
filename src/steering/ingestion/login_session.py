"""Track the one human-driven managed-browser sign-in without blocking a request.

Signing in is a human action that can take minutes. Awaiting it inside an HTTP
handler holds the connection open for the whole attempt and gives the user no
feedback until it ends, so the sign-in runs as a background task here and the
interface polls a cheap status endpoint instead.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol
from urllib.parse import urlsplit

from steering.ingestion.browser import (
    BrowserCaptureUnavailable,
    BrowserDependencyUnavailable,
    LoginOutcome,
)

LOGGER = logging.getLogger(__name__)


class ManagedLogin(Protocol):
    async def open_login(
        self,
        url: str,
        *,
        authorized: bool = False,
        force: bool = False,
    ) -> LoginOutcome: ...


class LoginState(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    SIGNED_IN = "signed_in"
    ALREADY_SIGNED_IN = "already_signed_in"
    CLOSED_BEFORE_SIGN_IN = "closed_before_sign_in"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


_MESSAGES: dict[LoginState, str] = {
    LoginState.IDLE: "No managed browser sign-in has been started.",
    LoginState.RUNNING: (
        "The managed browser is open. Sign in there; STEERING finishes as soon as sign-in "
        "is detected, or when you close the window."
    ),
    LoginState.SIGNED_IN: (
        "Signed in. The session is saved to STEERING's isolated profile, so you can now "
        "capture an authorized URL."
    ),
    LoginState.ALREADY_SIGNED_IN: (
        "A signed-in session already exists, so no browser was opened. Capture is ready to "
        "use. If captures start failing with a sign-in error, choose 'Sign in again'."
    ),
    LoginState.CLOSED_BEFORE_SIGN_IN: (
        "The managed browser closed before a completed sign-in was detected. If you did "
        "sign in, capture will still work; otherwise start sign-in again."
    ),
    LoginState.UNAVAILABLE: (
        "Managed browser capture is unavailable. Install the browser extra and Chromium, "
        "then restart STEERING."
    ),
    LoginState.FAILED: "The managed browser sign-in did not complete.",
}

_TERMINAL_STATES = frozenset(
    {
        LoginState.SIGNED_IN,
        LoginState.ALREADY_SIGNED_IN,
        LoginState.CLOSED_BEFORE_SIGN_IN,
        LoginState.UNAVAILABLE,
        LoginState.FAILED,
    }
)
_OUTCOME_STATES: dict[LoginOutcome, LoginState] = {
    LoginOutcome.SIGNED_IN: LoginState.SIGNED_IN,
    LoginOutcome.ALREADY_SIGNED_IN: LoginState.ALREADY_SIGNED_IN,
    LoginOutcome.CLOSED_BEFORE_SIGN_IN: LoginState.CLOSED_BEFORE_SIGN_IN,
}


@dataclass(frozen=True, slots=True)
class LoginStatus:
    state: LoginState
    message: str

    @property
    def running(self) -> bool:
        return self.state is LoginState.RUNNING

    @property
    def succeeded(self) -> bool:
        return self.state in {LoginState.SIGNED_IN, LoginState.ALREADY_SIGNED_IN}


class BrowserLoginSession:
    """Own at most one in-flight managed-browser sign-in for the daemon lifetime."""

    def __init__(self, capture: ManagedLogin) -> None:
        self._capture = capture
        self._task: asyncio.Task[LoginOutcome] | None = None
        self._status = LoginStatus(LoginState.IDLE, _MESSAGES[LoginState.IDLE])

    def start(self, url: str, *, force: bool = False) -> LoginStatus:
        """Begin a sign-in and return immediately.

        Raises ``RuntimeError`` when a sign-in is already in flight, because a
        second one would contend for the same locked browser profile directory.
        """

        if self._task is not None and not self._task.done():
            raise RuntimeError("a managed browser sign-in is already in progress")
        if not force and self._already_signed_in(url):
            # Answer in the response itself. Reporting "waiting for sign-in" and
            # correcting it a poll later shows the user a state that was never
            # true, which is exactly what looks broken.
            state = LoginState.ALREADY_SIGNED_IN
            self._status = LoginStatus(state, _MESSAGES[state])
            return self._status
        self._status = LoginStatus(LoginState.RUNNING, _MESSAGES[LoginState.RUNNING])
        self._task = asyncio.create_task(self._run(url, force=force))
        return self._status

    def _already_signed_in(self, url: str) -> bool:
        """Whether this platform is signed in, not merely whether any site is.

        One profile is shared, so asking only "is there a session" would answer yes
        for LinkedIn because X had been used, and the sign-in would never open.
        """

        host = urlsplit(url).hostname or ""
        per_platform = getattr(self._capture, "is_signed_in_to", None)
        if host and callable(per_platform):
            return bool(per_platform(host))
        probe = getattr(self._capture, "has_stored_session", None)
        return bool(probe()) if callable(probe) else False

    def is_signed_in_to(self, host: str) -> bool:
        probe = getattr(self._capture, "is_signed_in_to", None)
        return bool(probe(host)) if callable(probe) else self.signed_in

    @property
    def signed_in(self) -> bool:
        """Whether any stored session exists in the managed profile."""

        probe = getattr(self._capture, "has_stored_session", None)
        return bool(probe()) if callable(probe) else False

    def sign_out(self, host: str) -> bool:
        """Forget one platform's session, so signing in again is offered."""

        release = getattr(self._capture, "sign_out", None)
        if not callable(release):
            raise RuntimeError("this runtime cannot sign out of the managed browser")
        released = bool(release(host))
        LOGGER.info("signed out of %s in the managed profile (changed=%s)", host, released)
        return released

    def status(self) -> LoginStatus:
        task = self._task
        if task is not None and task.done():
            self._status = self._settle(task)
            self._task = None
        return self._status

    @staticmethod
    def _settle(task: asyncio.Task[LoginOutcome]) -> LoginStatus:
        if task.cancelled():
            return LoginStatus(LoginState.FAILED, _MESSAGES[LoginState.FAILED])
        error = task.exception()
        if error is None:
            state = _OUTCOME_STATES.get(task.result(), LoginState.CLOSED_BEFORE_SIGN_IN)
            return LoginStatus(state, _MESSAGES[state])
        if isinstance(error, BrowserDependencyUnavailable):
            return LoginStatus(LoginState.UNAVAILABLE, _MESSAGES[LoginState.UNAVAILABLE])
        if isinstance(error, (BrowserCaptureUnavailable, PermissionError)):
            return LoginStatus(LoginState.FAILED, f"{_MESSAGES[LoginState.FAILED]} {error}")
        LOGGER.warning("managed browser sign-in raised an unexpected error", exc_info=error)
        return LoginStatus(LoginState.FAILED, _MESSAGES[LoginState.FAILED])

    async def _run(self, url: str, *, force: bool) -> LoginOutcome:
        LOGGER.info("starting managed browser sign-in for %s (force=%s)", url, force)
        return await self._capture.open_login(url, authorized=True, force=force)

    async def aclose(self) -> None:
        task = self._task
        self._task = None
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            LOGGER.debug("managed browser sign-in cancelled during shutdown")
        except Exception:
            LOGGER.debug("managed browser sign-in errored during shutdown", exc_info=True)
        # Leaving the last reported state as RUNNING would strand any poller on a
        # sign-in that can no longer make progress.
        self._status = LoginStatus(LoginState.FAILED, _MESSAGES[LoginState.FAILED])


def is_terminal(state: LoginState) -> bool:
    return state in _TERMINAL_STATES
