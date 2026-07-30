from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from steering.database import DatabaseRuntime
from steering.domain.models import (
    Artifact,
    ArtifactRecord,
    ArtifactType,
    ReviewStatus,
    SearchQuery,
    SourceKind,
)


def _record(identifier: str) -> ArtifactRecord:
    return ArtifactRecord(
        artifact=Artifact(
            id=identifier,
            canonical_url=f"https://example.com/{identifier}",
            source_kind=SourceKind.PAPER,
            artifact_type=ArtifactType.PAPER,
            title=identifier,
            summary=f"summary for {identifier}",
            strategy_family="memory",
            review_status=ReviewStatus.REVIEWED,
            content_hash=f"hash-{identifier}",
        )
    )


class OverlapDetectingConnection:
    """Delegates to the real connection and reports statements that interleave.

    Widens each statement so an unserialised caller is caught reliably rather than
    only under unlucky timing.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._guard = threading.Lock()
        self._occupant: int | None = None
        self.overlaps: list[tuple[int, int]] = []

    def execute(self, query: str, parameters: Any = None) -> Any:
        me = threading.get_ident()
        with self._guard:
            if self._occupant is not None and self._occupant != me:
                self.overlaps.append((self._occupant, me))
            self._occupant = me
        try:
            time.sleep(0.001)
            return self._inner.execute(query, parameters)
        finally:
            with self._guard:
                self._occupant = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


@pytest.mark.integration
def test_writes_and_reads_never_interleave_on_the_shared_connection(tmp_path: Path) -> None:
    """Ingestion writes on a worker thread while the event loop serves reads.

    Both sides share one Ladybug connection. A read landing inside the writer's open
    transaction can commit a node whose payload was never written, which surfaces
    later as a record that lists but cannot be opened.
    """

    with DatabaseRuntime(tmp_path / "concurrency.lbug") as runtime:
        repository = runtime.repository
        probe = OverlapDetectingConnection(repository._connection)
        repository._connection = probe  # type: ignore[assignment]
        repository.upsert_record(_record("seed"))

        failures: list[BaseException] = []

        def write() -> None:
            try:
                for index in range(12):
                    repository.upsert_record(_record(f"written-{index}"))
            except BaseException as exc:  # reported to the assertions below
                failures.append(exc)

        def read() -> None:
            # The retrieval readers are what a browsing user triggers while ingestion
            # is still writing.
            search = SearchQuery(query="memory", limit=5)
            try:
                for _ in range(12):
                    repository.exact_candidates(["memory", "summary"], search)
                    repository.graph_candidates(["seed"], search)
                    repository.count_artifacts()
            except BaseException as exc:  # reported to the assertions below
                failures.append(exc)

        threads = [threading.Thread(target=write), threading.Thread(target=read)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        assert not failures, f"concurrent access raised: {failures[0]!r}"
        assert not probe.overlaps, (
            f"{len(probe.overlaps)} statement(s) executed on the shared connection "
            "while another thread was mid-statement"
        )
        assert repository.count_artifacts() == 13

        # Every written record must still be readable; an interleaved write leaves a
        # node whose payload never landed.
        for index in range(12):
            assert repository.get_record(f"written-{index}") is not None
