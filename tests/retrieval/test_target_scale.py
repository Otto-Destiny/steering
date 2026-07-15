from __future__ import annotations

import math
import os
import time

import pytest

from steering.domain.models import Artifact, ArtifactRecord, Chunk, SearchQuery, SourceKind
from steering.evaluation.embedding import DeterministicBlake2EmbeddingProvider
from steering.retrieval.hybrid import HybridRetriever

ARTIFACT_COUNT = 10_000
CHUNKS_PER_ARTIFACT = 10


class ScaleRepository:
    def __init__(self, records: list[ArtifactRecord]) -> None:
        self.records = records

    def list_records(self) -> list[ArtifactRecord]:
        return self.records

    def project_history(self, _project_id: str) -> dict[str, list[object]]:
        return {}


def _records() -> list[ArtifactRecord]:
    families = ("memory", "evaluation", "inference", "retrieval", "training")
    records: list[ArtifactRecord] = []
    for index in range(ARTIFACT_COUNT):
        artifact_id = f"scale-artifact-{index}"
        family = families[index % len(families)]
        chunks = [
            Chunk(
                id=f"scale-chunk-{index}-{ordinal}",
                artifact_id=artifact_id,
                snapshot_id=f"scale-snapshot-{index}",
                ordinal=ordinal,
                text=(
                    f"{family} technique {index} section {ordinal} discusses bounded context, "
                    "implementation constraints, evaluation, latency, and failure modes."
                ),
                locator=f"section:{ordinal}",
            )
            for ordinal in range(CHUNKS_PER_ARTIFACT)
        ]
        records.append(
            ArtifactRecord(
                artifact=Artifact(
                    id=artifact_id,
                    canonical_url=f"https://example.org/scale/{index}",
                    source_kind=SourceKind.WEBPAGE,
                    title=f"Synthetic {family} option {index}",
                    summary=f"A {family} engineering option with measurable tradeoffs.",
                    strategy_family=family,
                    content_hash=f"{index:064x}",
                    capabilities=[family, "bounded context"],
                    limitations=["requires validation under project constraints"],
                    use_cases=[f"{family} architecture"],
                ),
                chunks=chunks,
            )
        )
    return records


@pytest.mark.performance
@pytest.mark.skipif(
    os.environ.get("STEERING_SCALE_TEST") != "1",
    reason="set STEERING_SCALE_TEST=1 for the opt-in 10k/100k benchmark",
)
async def test_warm_search_sub_two_seconds_at_target_scale() -> None:
    records = _records()
    assert len(records) == ARTIFACT_COUNT
    assert sum(len(record.chunks) for record in records) == ARTIFACT_COUNT * CHUNKS_PER_ARTIFACT

    retriever = HybridRetriever(
        repository=ScaleRepository(records),  # type: ignore[arg-type]
        embedding_provider=DeterministicBlake2EmbeddingProvider(dimension=32),
    )
    await retriever.refresh()

    queries = [
        f"Compare {family} options for bounded context, latency, evaluation, and failure modes {index}"
        for index, family in enumerate(("memory", "evaluation", "inference", "retrieval", "training") * 4)
    ]
    durations: list[float] = []
    for query in queries:
        started = time.perf_counter()
        results = await retriever.search(SearchQuery(query=query, limit=10, breadth=True))
        durations.append(time.perf_counter() - started)
        assert len(results) == 10

    p95 = sorted(durations)[math.ceil(0.95 * len(durations)) - 1]
    assert p95 < 2.0, f"warm search p95 was {p95:.3f}s at 10k artifacts/100k chunks"
