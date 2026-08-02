from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests.support.providers import DeterministicBlake2EmbeddingProvider

from steering.database import DatabaseRuntime, SearchIndexError, verify_backup
from steering.domain.models import (
    Artifact,
    ArtifactRecord,
    ArtifactType,
    Chunk,
    Concept,
    Relation,
    RelationType,
    ReviewStatus,
    SearchQuery,
    SourceKind,
)
from steering.retrieval.hybrid import HybridRetriever


async def _record(
    provider: DeterministicBlake2EmbeddingProvider,
    identifier: str,
    text: str,
    *,
    family: str,
    relations: list[Relation] | None = None,
) -> ArtifactRecord:
    embedding = await provider.embed_documents([text])
    return ArtifactRecord(
        artifact=Artifact(
            id=identifier,
            canonical_url=f"https://example.com/{identifier}",
            source_kind=SourceKind.PAPER,
            artifact_type=ArtifactType.PAPER,
            title=identifier,
            summary=text,
            strategy_family=family,
            review_status=ReviewStatus.REVIEWED,
            content_hash=f"hash-{identifier}",
        ),
        chunks=[
            Chunk(
                id=f"chunk-{identifier}",
                artifact_id=identifier,
                snapshot_id=f"snapshot-{identifier}",
                ordinal=0,
                text=text,
                locator=f"source:{identifier}",
                embedding=embedding[0],
                embedding_provider=provider.provider_id,
                embedding_model=provider.model_id,
                embedding_revision=provider.model_revision,
                embedding_dimension=provider.dimension,
                embedding_task_mode=provider.document_task_mode,
                embedding_normalized=provider.normalized,
                source_content_hash=f"chunk-hash-{identifier}",
            )
        ],
        relations=relations or [],
    )


@pytest.mark.integration
async def test_real_ladybug_fts_hnsw_and_exact_candidates_without_record_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = DeterministicBlake2EmbeddingProvider(dimension=768)
    with DatabaseRuntime(tmp_path / "native-search.lbug") as runtime:
        runtime.repository.upsert_record(
            await _record(provider, "automem", "agent memory context retrieval", family="memory")
        )
        runtime.repository.upsert_record(
            await _record(provider, "webgpu", "gpu kernel optimization", family="inference")
        )
        monkeypatch.setattr(
            runtime.repository,
            "list_records",
            lambda: (_ for _ in ()).throw(AssertionError("full record scan is forbidden")),
        )
        retriever = HybridRetriever(repository=runtime.repository, embedding_provider=provider)

        hits = await retriever.search(SearchQuery(query="agent context memory", limit=2))
        exact = await retriever.search(SearchQuery(query="Review https://example.com/automem", limit=1))
        indexes = runtime.connection.execute("CALL SHOW_INDEXES() RETURN *").get_all()

        assert hits[0].artifact.id == "automem"
        assert hits[0].scores.bm25 > 0.0
        assert hits[0].scores.vector > 0.0
        assert hits[0].matched_chunks[0].id == "chunk-automem"
        assert exact[0].artifact.id == "automem"
        assert exact[0].scores.exact >= 4.0
        assert {row[2] for row in indexes} >= {"FTS", "HNSW"}


@pytest.mark.integration
async def test_reopened_ladybug_loads_persisted_index_extensions_before_insert(
    tmp_path: Path,
) -> None:
    provider = DeterministicBlake2EmbeddingProvider(dimension=768)
    database_path = tmp_path / "reopened-native-search.lbug"

    with DatabaseRuntime(database_path) as runtime:
        runtime.repository.upsert_record(
            await _record(provider, "first", "agent memory retrieval", family="memory")
        )
        retriever = HybridRetriever(repository=runtime.repository, embedding_provider=provider)
        await retriever.search(SearchQuery(query="agent memory", limit=1))

    with DatabaseRuntime(database_path) as reopened:
        reopened.repository.upsert_record(
            await _record(provider, "second", "context compression", family="memory")
        )
        retriever = HybridRetriever(repository=reopened.repository, embedding_provider=provider)
        hits = await retriever.search(SearchQuery(query="context compression", limit=2))

    assert hits[0].artifact.id == "second"


@pytest.mark.integration
async def test_real_ladybug_graph_expansion_is_approved_and_one_hop(tmp_path: Path) -> None:
    provider = DeterministicBlake2EmbeddingProvider(dimension=768)
    first_to_second = Relation(
        id="first-to-second",
        subject_id="first",
        predicate=RelationType.RELATED_TO,
        object_id="second",
        approved=True,
    )
    second_to_third = Relation(
        id="second-to-third",
        subject_id="second",
        predicate=RelationType.RELATED_TO,
        object_id="third",
        approved=True,
    )
    with DatabaseRuntime(tmp_path / "native-graph.lbug") as runtime:
        for record in (
            await _record(
                provider,
                "first",
                "quasar architecture",
                family="first-family",
                relations=[first_to_second],
            ),
            await _record(
                provider,
                "second",
                "neighbor material",
                family="second-family",
                relations=[second_to_third],
            ),
            await _record(provider, "third", "distant material", family="third-family"),
        ):
            runtime.repository.upsert_record(record)
        retriever = HybridRetriever(repository=runtime.repository, embedding_provider=provider)

        hits = await retriever.search(SearchQuery(query="quasar architecture", limit=3))
        knowledge_edges = runtime.connection.execute(
            "MATCH ()-[r:KnowledgeEdges]->() RETURN count(r)"
        ).get_all()[0][0]
        artifact_links = runtime.connection.execute(
            "MATCH ()-[r:ArtifactKnowledgeLinks]->() RETURN count(r)"
        ).get_all()[0][0]

        graph_scores = {hit.artifact.id: hit.scores.graph for hit in hits}
        assert graph_scores["second"] > 0.0
        assert "third" not in graph_scores
        assert knowledge_edges == 2
        assert artifact_links == 2


@pytest.mark.integration
async def test_shared_concept_links_expand_all_owners_and_survive_counterfactual_reupsert(
    tmp_path: Path,
) -> None:
    provider = DeterministicBlake2EmbeddingProvider(dimension=768)
    source = Concept(id="shared-source", name="Shared source concept")
    target = Concept(id="shared-target", name="Shared target concept")
    relation = Relation(
        id="shared-relation",
        subject_id=source.id,
        predicate=RelationType.RELATED_TO,
        object_id=target.id,
        approved=True,
    )
    records: list[ArtifactRecord] = []
    for identifier, concept in (
        ("source-a", source),
        ("source-b", source),
        ("target-a", target),
        ("target-b", target),
    ):
        record = await _record(
            provider,
            identifier,
            f"{concept.name} architecture",
            family=identifier,
            relations=[relation] if identifier == "source-a" else [],
        )
        records.append(record.model_copy(update={"concepts": [concept]}))

    with DatabaseRuntime(tmp_path / "shared-concepts.lbug") as runtime:
        for record in records:
            runtime.repository.upsert_record(record)

        def links() -> list[list[object]]:
            return runtime.connection.execute(
                """MATCH (s:Artifacts)-[r:ArtifactKnowledgeLinks]->(o:Artifacts)
                WHERE r.relation_id = 'shared-relation'
                RETURN s.id, o.id, r.predicate ORDER BY s.id, o.id"""
            ).get_all()

        expected = [
            [subject, object_, RelationType.RELATED_TO.value]
            for subject in ("source-a", "source-b")
            for object_ in ("target-a", "target-b")
        ]
        assert links() == expected

        counterfactual = records[0].model_copy(
            update={
                "artifact": records[0].artifact.model_copy(
                    update={"published_at": datetime(1999, 1, 1, tzinfo=UTC)}
                )
            },
            deep=True,
        )
        runtime.repository.upsert_record(counterfactual)

        assert links() == expected


@pytest.mark.integration
async def test_model_change_requires_verified_complete_reembedding(tmp_path: Path) -> None:
    original = DeterministicBlake2EmbeddingProvider(dimension=768)

    class ReplacementEmbedding(DeterministicBlake2EmbeddingProvider):
        model_id = "test/replacement-embedding-v1"

    replacement = ReplacementEmbedding(dimension=768)
    with DatabaseRuntime(tmp_path / "model-change.lbug") as runtime:
        runtime.repository.upsert_record(
            await _record(original, "memory", "persistent agent memory", family="memory")
        )
        await HybridRetriever(repository=runtime.repository, embedding_provider=original).refresh()
        retriever = HybridRetriever(repository=runtime.repository, embedding_provider=replacement)

        with pytest.raises(SearchIndexError, match="complete re-embedding"):
            await retriever.search(SearchQuery(query="persistent memory", limit=1))

        replaced, backup = await retriever.reembed_all()

        assert replaced == 1
        assert backup is not None
        verify_backup(backup)
        stored = runtime.repository.get_record("memory")
        assert stored is not None
        assert stored.chunks[0].embedding_model == replacement.model_id
        hits = await retriever.search(SearchQuery(query="persistent memory", limit=1))
        assert hits[0].artifact.id == "memory"


@pytest.mark.integration
async def test_search_refuses_partially_embedded_chunk_set(tmp_path: Path) -> None:
    provider = DeterministicBlake2EmbeddingProvider(dimension=768)
    complete = await _record(provider, "complete", "complete vector", family="retrieval")
    missing = await _record(provider, "missing", "missing vector", family="retrieval")
    missing = missing.model_copy(
        update={
            "chunks": [
                missing.chunks[0].model_copy(
                    update={
                        "embedding": [],
                        "embedding_provider": None,
                        "embedding_model": None,
                        "embedding_revision": None,
                        "embedding_dimension": None,
                        "embedding_task_mode": None,
                        "embedding_normalized": None,
                        "source_content_hash": None,
                    }
                )
            ]
        }
    )
    with DatabaseRuntime(tmp_path / "partial-vectors.lbug") as runtime:
        runtime.repository.upsert_record(complete)
        runtime.repository.upsert_record(missing)
        retriever = HybridRetriever(repository=runtime.repository, embedding_provider=provider)

        with pytest.raises(SearchIndexError, match="need embeddings"):
            await retriever.search(SearchQuery(query="vector", limit=2))


def _relating_record(
    identifier: str,
    *,
    concept_id: str,
    concept_name: str,
    predicate: RelationType,
) -> ArtifactRecord:
    """An artifact that points at a shared concept, as extraction now produces."""

    return ArtifactRecord(
        artifact=Artifact(
            id=identifier,
            canonical_url=f"https://example.com/{identifier}",
            source_kind=SourceKind.GITHUB,
            artifact_type=ArtifactType.OPEN_SOURCE_TOOL,
            title=identifier,
            summary=f"{identifier} summary",
            strategy_family="memory",
            review_status=ReviewStatus.REVIEWED,
            content_hash=f"hash-{identifier}",
        ),
        concepts=[Concept(id=concept_id, name=concept_name)],
        relations=[
            Relation(
                id=f"rel-{identifier}",
                subject_id=identifier,
                predicate=predicate,
                object_id=concept_id,
                approved=True,
            )
        ],
    )


@pytest.mark.integration
def test_two_artifacts_sharing_a_concept_become_reachable_from_each_other(tmp_path: Path) -> None:
    """A concept is only worth anything once a second artifact also owns it.

    One artifact pointing at an idea links to nothing, because a record is never
    linked to itself. The connection appears when something else arrives holding
    the same idea, which is the whole reason concepts are hashed from their name.
    """

    concept_id = "concept_paged_cache"
    with DatabaseRuntime(tmp_path / "graph.lbug") as runtime:
        repository = runtime.repository
        repository.upsert_record(
            _relating_record(
                "pocketmemory",
                concept_id=concept_id,
                concept_name="paged cache",
                predicate=RelationType.SOLVES,
            )
        )
        query = SearchQuery(query="paged cache", limit=5)

        # Alone, it has nothing to reach.
        assert repository.graph_candidates(["pocketmemory"], query) == []

        repository.upsert_record(
            _relating_record(
                "shelfcache",
                concept_id=concept_id,
                concept_name="paged cache",
                predicate=RelationType.IMPLEMENTS,
            )
        )

        from_first = repository.graph_candidates(["pocketmemory"], query)
        from_second = repository.graph_candidates(["shelfcache"], query)

    assert [candidate.artifact_id for candidate in from_first] == ["shelfcache"]
    assert [candidate.artifact_id for candidate in from_second] == ["pocketmemory"]


@pytest.mark.integration
def test_an_inactive_relation_is_not_walkable(tmp_path: Path) -> None:
    """Switching a connection off has to remove the edge, not just a flag."""

    concept_id = "concept_paged_cache"
    with DatabaseRuntime(tmp_path / "graph-off.lbug") as runtime:
        repository = runtime.repository
        for name, predicate in (
            ("pocketmemory", RelationType.SOLVES),
            ("shelfcache", RelationType.IMPLEMENTS),
        ):
            repository.upsert_record(
                _relating_record(name, concept_id=concept_id, concept_name="paged cache", predicate=predicate)
            )
        query = SearchQuery(query="paged cache", limit=5)
        assert repository.graph_candidates(["pocketmemory"], query)

        # Each artifact derives its own link from the concept it owns, and the
        # traversal reads both directions, so both have to go for the pair to part.
        repository.set_relation_active("rel-shelfcache", False)
        one_off = repository.graph_candidates(["pocketmemory"], query)
        repository.set_relation_active("rel-pocketmemory", False)
        both_off = repository.graph_candidates(["pocketmemory"], query)

        repository.set_relation_active("rel-pocketmemory", True)
        back_on = repository.graph_candidates(["pocketmemory"], query)

    assert [candidate.artifact_id for candidate in one_off] == ["shelfcache"]
    assert both_off == []
    assert [candidate.artifact_id for candidate in back_on] == ["shelfcache"]
