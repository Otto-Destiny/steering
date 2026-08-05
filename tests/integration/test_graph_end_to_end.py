"""Two ordinary captures, then a search that finds one through the other.

Every other test proves a piece: extraction makes a concept, hand-built records
project into edges, a fake repository serves the toggle. None of them proves the
join, which is the part that actually has to hold -- that two separate captures,
each naming an idea in its own words, land on the same node and become reachable
from one another through the retriever the MCP tools call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from tests.support.providers import DeterministicBlake2EmbeddingProvider

from steering.database import DatabaseRuntime
from steering.domain.models import ArtifactType, RelationType, ResolvedSource, SearchQuery, SourceKind
from steering.extraction.schemas import ExtractedRelation, KnowledgeExtraction
from steering.extraction.service import ExtractionService
from steering.ingestion.resolvers import ResolverRegistry
from steering.ingestion.service import IngestionService
from steering.retrieval.hybrid import HybridRetriever

POCKET_URL = "https://example.com/pocketmemory"
SHELF_URL = "https://example.com/shelfcache"

POCKET_TEXT = (
    "PocketMemory is an agent memory store. It solves paged cache pressure by keeping "
    "only recently touched pages resident, and it integrates with SQLite for durability."
)
SHELF_TEXT = (
    "ShelfCache is a retrieval cache for language model servers. It implements Paged Cache "
    "eviction so long sessions stay affordable, and it also stores state in SQLite."
)


class FixedResolver:
    name = "fixture"

    def __init__(self, pages: dict[str, str]) -> None:
        self.pages = pages

    def can_resolve(self, source: str) -> bool:
        return source in self.pages

    async def resolve(self, source: str) -> ResolvedSource:
        return ResolvedSource(
            canonical_url=source,
            source_kind=SourceKind.DOCUMENTATION,
            title=source.rsplit("/", 1)[-1],
            text=self.pages[source],
            extraction_method="fixture",
        )


class ExtractionByUrl:
    """Answers each capture separately, as two independent extractions would.

    The two disagree on wording and on what they call the target type, which is
    the realistic case: nothing coordinates one model call with the next.
    """

    model_id = "fixture-generation"

    def __init__(self, payloads: dict[str, KnowledgeExtraction]) -> None:
        self.payloads = payloads
        self.calls = 0

    async def generate_structured(
        self, *, system_prompt: str, user_prompt: str, response_model: type[Any]
    ) -> Any:
        self.calls += 1
        for url, payload in self.payloads.items():
            if url in user_prompt:
                return payload
        raise AssertionError("no fixture extraction matched the prompt")

    async def test_connection(self) -> None:
        return None


def extraction_for(title: str, *, target_name: str, target_type: str, quote: str) -> KnowledgeExtraction:
    return KnowledgeExtraction(
        artifact_type=ArtifactType.OPEN_SOURCE_TOOL,
        title=title,
        summary=f"{title} keeps recently used pages resident.",
        strategy_family="memory_compression",
        relations=[
            ExtractedRelation(
                predicate=RelationType.SOLVES,
                target_name=target_name,
                target_type=target_type,
                exact_quote=quote,
                source_index=0,
            )
        ],
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_two_captures_naming_one_idea_differently_still_meet(tmp_path: Path) -> None:
    """Casing and target-type wording must not split one idea into two nodes.

    One capture says "paged cache pressure" is a concept; the other writes
    "Paged Cache eviction" and calls it a technique. Left alone those hash to
    different nodes in different namespaces, and the two captures never meet.
    """

    provider = DeterministicBlake2EmbeddingProvider(dimension=768)
    generation = ExtractionByUrl(
        {
            POCKET_URL: extraction_for(
                "PocketMemory",
                target_name="paged cache",
                target_type="concept",
                quote="It solves paged cache pressure by keeping",
            ),
            SHELF_URL: extraction_for(
                "ShelfCache",
                target_name="Paged Cache",
                target_type="Technique",
                quote="It implements Paged Cache eviction so long sessions stay affordable",
            ),
        }
    )

    with DatabaseRuntime(tmp_path / "end-to-end.lbug") as runtime:
        ingestion = IngestionService(
            registry=ResolverRegistry([FixedResolver({POCKET_URL: POCKET_TEXT, SHELF_URL: SHELF_TEXT})]),
            extraction=ExtractionService(generation=generation, embedding=provider),
            repository=runtime.repository,
        )
        first = await ingestion.add(POCKET_URL)
        second = await ingestion.add(SHELF_URL)

        # Both captures produced a live connection to one shared node.
        assert [relation.approved for relation in first.relations] == [True]
        assert [relation.approved for relation in second.relations] == [True]
        assert first.concepts[0].id == second.concepts[0].id

        query = SearchQuery(query="paged cache", limit=5)
        reachable = runtime.repository.graph_candidates([first.artifact.id], query)
        assert [candidate.artifact_id for candidate in reachable] == [second.artifact.id]

        retriever = HybridRetriever(repository=runtime.repository, embedding_provider=provider)
        # Worded for PocketMemory alone. ShelfCache shares nothing with these words,
        # so if it appears at all it can only have arrived through the connection.
        hits = await retriever.search(SearchQuery(query="agent memory store durability", limit=5))

    ranked = {hit.artifact.id: hit for hit in hits}
    assert first.artifact.id in ranked
    assert second.artifact.id in ranked, "the neighbour never surfaced"
    assert ranked[second.artifact.id].scores.graph > 0, "it surfaced, but not through the graph"
