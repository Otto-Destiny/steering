from __future__ import annotations

import math
import os
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.support.providers import DeterministicBlake2EmbeddingProvider

from steering.database import DatabaseRuntime
from steering.domain.models import Artifact, ArtifactRecord, Chunk, SearchQuery, SourceKind
from steering.retrieval.hybrid import HybridRetriever

ARTIFACT_COUNT = 10_000
CHUNKS_PER_ARTIFACT = 10


def _records(
    provider: DeterministicBlake2EmbeddingProvider,
) -> Iterator[ArtifactRecord]:
    families = ("memory", "evaluation", "inference", "retrieval", "training")
    vector = [1.0, *([0.0] * 767)]
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
                embedding=vector,
                embedding_provider=provider.provider_id,
                embedding_model=provider.model_id,
                embedding_revision=provider.model_revision,
                embedding_dimension=provider.dimension,
                embedding_task_mode=provider.document_task_mode,
                embedding_normalized=provider.normalized,
                source_content_hash=f"scale-{index}-{ordinal}",
            )
            for ordinal in range(CHUNKS_PER_ARTIFACT)
        ]
        yield ArtifactRecord(
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


@pytest.mark.performance
@pytest.mark.skipif(
    os.environ.get("STEERING_SCALE_TEST") != "1",
    reason="set STEERING_SCALE_TEST=1 for the opt-in 10k/100k benchmark",
)
async def test_warm_search_sub_two_seconds_at_target_scale(tmp_path: Path) -> None:
    provider = DeterministicBlake2EmbeddingProvider(dimension=768)
    artifact_count = 0
    chunk_count = 0
    with DatabaseRuntime(tmp_path / "target-scale.lbug") as database:
        for record in _records(provider):
            database.repository.upsert_record(record)
            artifact_count += 1
            chunk_count += len(record.chunks)
        assert artifact_count == ARTIFACT_COUNT
        assert chunk_count == ARTIFACT_COUNT * CHUNKS_PER_ARTIFACT

        retriever = HybridRetriever(
            repository=database.repository,
            embedding_provider=provider,
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
