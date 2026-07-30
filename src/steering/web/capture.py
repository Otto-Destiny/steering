from __future__ import annotations

from typing import Protocol

from steering.domain.models import ArtifactRecord, ResolvedSource
from steering.ingestion.browser import LoginOutcome


class ResolvedSourceIngestion(Protocol):
    """Runtime-owned bridge for already-resolved disposable sources."""

    async def add_resolved(self, source: ResolvedSource) -> ArtifactRecord: ...


class AuthorizedBrowserCapture(Protocol):
    async def capture(self, url: str, *, authorized: bool = False) -> ResolvedSource: ...

    async def open_login(self, url: str, *, authorized: bool = False) -> LoginOutcome: ...
