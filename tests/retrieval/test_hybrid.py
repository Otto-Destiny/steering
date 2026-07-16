from __future__ import annotations

from tests.support.providers import DeterministicBlake2EmbeddingProvider

from steering.database.native import NativeCandidate, normalize_search_term
from steering.domain.models import (
    Artifact,
    ArtifactRecord,
    ArtifactType,
    ExperimentOutcome,
    Project,
    Relation,
    RelationType,
    ReviewStatus,
    ScoreBreakdown,
    SearchQuery,
    SourceKind,
)
from steering.retrieval.hybrid import HybridRetriever, _CandidateRow, retrieval_tokens


class MemoryRepository:
    def __init__(
        self,
        records: list[ArtifactRecord],
        history: dict[str, list[object]] | None = None,
    ) -> None:
        self.records = records
        self.history = history or {}

    def list_records(self) -> list[ArtifactRecord]:
        return self.records

    def rebuild_search_indexes(self) -> None:
        return None

    def exact_candidates(
        self, terms: list[str], _query: SearchQuery, *, limit: int = 50
    ) -> list[NativeCandidate]:
        term_set = set(terms)
        candidates = []
        for item in self.records:
            values = {
                normalize_search_term(item.artifact.title),
                normalize_search_term(item.artifact.canonical_url or ""),
            }
            if values & term_set:
                candidates.append(NativeCandidate(item.artifact.id, None, 3.0))
        return candidates[:limit]

    def bm25_candidates(self, text: str, _query: SearchQuery, *, limit: int = 50) -> list[NativeCandidate]:
        query_tokens = set(retrieval_tokens(text))
        candidates = [
            NativeCandidate(
                item.artifact.id,
                None,
                float(len(query_tokens & set(retrieval_tokens(item.artifact.summary)))),
            )
            for item in self.records
        ]
        return [item for item in candidates if item.score > 0][:limit]

    def vector_candidates(
        self, _vector: list[float], _query: SearchQuery, *, limit: int = 50
    ) -> list[NativeCandidate]:
        return []

    def graph_candidates(
        self, seed_ids: list[str], _query: SearchQuery, *, limit: int = 50
    ) -> list[NativeCandidate]:
        linked: set[str] = set()
        for item in self.records:
            for relation in item.relations:
                if relation.approved and relation.subject_id in seed_ids:
                    linked.add(relation.object_id)
        return [NativeCandidate(identifier, None, 1.0) for identifier in sorted(linked)[:limit]]

    def load_records(self, artifact_ids: list[str]) -> list[ArtifactRecord]:
        requested = set(artifact_ids)
        return [item for item in self.records if item.artifact.id in requested]

    def project_history(self, _project_id: str) -> dict[str, list[object]]:
        return self.history


def record(identifier: str, text: str, family: str) -> ArtifactRecord:
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
            content_hash=identifier,
        )
    )


def _with_identity(
    value: ArtifactRecord,
    *,
    canonical_url: str | None = None,
    content_hash: str | None = None,
) -> ArtifactRecord:
    return value.model_copy(
        update={
            "artifact": value.artifact.model_copy(
                update={
                    "canonical_url": canonical_url or value.artifact.canonical_url,
                    "content_hash": content_hash or value.artifact.content_hash,
                }
            )
        }
    )


async def test_bm25_document_frequency_counts_documents_not_occurrences() -> None:
    repository = MemoryRepository(
        [
            record("rare", "quasar " * 100, "family-a"),
            record("other", "ordinary retrieval", "family-b"),
        ]
    )
    retriever = HybridRetriever(
        repository=repository,  # type: ignore[arg-type]
        embedding_provider=DeterministicBlake2EmbeddingProvider(),
    )
    await retriever.refresh()
    hits = await retriever.search(SearchQuery(query="quasar", limit=2))
    assert hits[0].artifact.id == "rare"
    assert hits[0].scores.bm25 > 0


async def test_breadth_returns_four_families_and_every_hit_has_a_citation() -> None:
    repository = MemoryRepository(
        [record(f"artifact-{index}", "agent context memory", f"family-{index}") for index in range(6)]
    )
    retriever = HybridRetriever(
        repository=repository,  # type: ignore[arg-type]
        embedding_provider=DeterministicBlake2EmbeddingProvider(),
    )
    hits = await retriever.search(SearchQuery(query="agent context memory", limit=5, breadth=True))
    assert len({hit.artifact.strategy_family for hit in hits}) >= 4
    assert all(hit.citation_urls for hit in hits)


async def test_rrf_uses_rank_contributions_and_indexes_exact_urls() -> None:
    target = _with_identity(
        record("target", "otherwise unrelated material", "family-a"),
        canonical_url="https://papers.example.org/work/42",
    )
    repository = MemoryRepository([target, record("other", "retrieval memory", "family-b")])
    retriever = HybridRetriever(
        repository=repository,  # type: ignore[arg-type]
        embedding_provider=DeterministicBlake2EmbeddingProvider(),
    )

    hits = await retriever.search(SearchQuery(query="Review https://papers.example.org/work/42", limit=2))

    assert hits[0].artifact.id == "target"
    assert hits[0].scores.exact >= 3.0
    assert 0.0 < hits[0].scores.rrf < 0.1


async def test_duplicate_canonical_urls_and_content_hashes_collapse_before_selection() -> None:
    canonical = "https://example.com/shared"
    records = [
        _with_identity(record("first", "agent memory", "family-a"), canonical_url=canonical),
        _with_identity(record("second", "agent memory", "family-b"), canonical_url=f"{canonical}/"),
        _with_identity(record("third", "agent memory", "family-c"), content_hash="same-content"),
        _with_identity(record("fourth", "agent memory", "family-d"), content_hash="same-content"),
    ]
    retriever = HybridRetriever(
        repository=MemoryRepository(records),  # type: ignore[arg-type]
        embedding_provider=DeterministicBlake2EmbeddingProvider(),
    )

    hits = await retriever.search(SearchQuery(query="agent memory", limit=4, breadth=True))

    urls = [hit.artifact.canonical_url.rstrip("/") for hit in hits if hit.artifact.canonical_url]
    hashes = [hit.artifact.content_hash for hit in hits]
    assert len(urls) == len(set(urls))
    assert len(hashes) == len(set(hashes))


async def test_approved_relation_expands_only_one_hop() -> None:
    first = record("first", "quasar architecture", "family-a")
    second = record("second", "neighbor material", "family-b")
    third = record("third", "distant material", "family-c")
    first = first.model_copy(
        update={
            "relations": [
                Relation(
                    subject_id="first",
                    predicate=RelationType.RELATED_TO,
                    object_id="second",
                    approved=True,
                )
            ]
        }
    )
    second = second.model_copy(
        update={
            "relations": [
                Relation(
                    subject_id="second",
                    predicate=RelationType.RELATED_TO,
                    object_id="third",
                    approved=True,
                )
            ]
        }
    )
    retriever = HybridRetriever(
        repository=MemoryRepository([first, second, third]),  # type: ignore[arg-type]
        embedding_provider=DeterministicBlake2EmbeddingProvider(),
    )

    hits = await retriever.search(SearchQuery(query="quasar architecture", limit=3))
    graph_scores = {hit.artifact.id: hit.scores.graph for hit in hits}

    assert graph_scores["second"] > 0.0
    assert "third" not in graph_scores


async def test_mmr_uses_embedding_similarity_after_relevance() -> None:
    records = [record(identifier, "same topic", "one-family") for identifier in ["a", "b", "c"]]
    retriever = HybridRetriever(
        repository=MemoryRepository(records),  # type: ignore[arg-type]
        embedding_provider=DeterministicBlake2EmbeddingProvider(),
    )
    await retriever.refresh()
    retriever._rows = {
        "a": _CandidateRow(records[0], [1.0, 0.0]),
        "b": _CandidateRow(records[1], [1.0, 0.0]),
        "c": _CandidateRow(records[2], [0.0, 1.0]),
    }
    scores = {
        "a": ScoreBreakdown(rrf=1.0),
        "b": ScoreBreakdown(rrf=0.99),
        "c": ScoreBreakdown(rrf=0.8),
    }

    selected = retriever._mmr(["a", "b", "c"], scores, limit=2, breadth=True)

    assert selected == ["a", "c"]


async def test_breadth_covers_multiple_strategies_per_requested_engineering_area() -> None:
    records = [
        record("memory-a", "memory option", "memory_lifecycle"),
        record("memory-b", "memory option", "trainable_memory"),
        record("visual-a", "visual option", "visual_page_retrieval"),
        record("visual-b", "visual option", "visual_sparse_retrieval"),
        record("generic-a", "generic option", "other-a"),
        record("generic-b", "generic option", "other-b"),
    ]
    retriever = HybridRetriever(
        repository=MemoryRepository(records),  # type: ignore[arg-type]
        embedding_provider=DeterministicBlake2EmbeddingProvider(),
    )
    retriever._rows = {
        item.artifact.id: _CandidateRow(item, [float(index + 1), 1.0]) for index, item in enumerate(records)
    }
    ordered = ["generic-a", "generic-b", "memory-a", "visual-a", "memory-b", "visual-b"]
    scores = {identifier: ScoreBreakdown(rrf=1.0 - index * 0.1) for index, identifier in enumerate(ordered)}

    selected = retriever._mmr(
        ordered,
        scores,
        limit=4,
        breadth=True,
        concepts={"agent_memory", "visual_retrieval"},
    )

    assert set(selected) == {"memory-a", "memory-b", "visual-a", "visual-b"}


async def test_failed_project_outcome_only_penalizes_matching_constraints() -> None:
    project_id = "project-one"
    target_id = "previously-failed"
    records = [
        record(target_id, "agent memory option", "family-a"),
        record("alternative", "agent memory option", "family-b"),
    ]
    outcome = ExperimentOutcome(
        project_id=project_id,
        artifact_id=target_id,
        outcome="Exceeded latency budget on the local deployment.",
        constraints=["local deployment"],
        succeeded=False,
    )
    repository = MemoryRepository(records, {"outcomes": [outcome]})
    retriever = HybridRetriever(
        repository=repository,  # type: ignore[arg-type]
        embedding_provider=DeterministicBlake2EmbeddingProvider(),
    )

    matching = await retriever.search(
        SearchQuery(query="agent memory for local deployment", project_id=project_id, limit=2)
    )
    changed = await retriever.search(
        SearchQuery(query="agent memory for managed cloud", project_id=project_id, limit=2)
    )

    matching_scores = {hit.artifact.id: hit.scores.project_history for hit in matching}
    changed_scores = {hit.artifact.id: hit.scores.project_history for hit in changed}
    assert matching_scores[target_id] == -0.5
    assert changed_scores[target_id] == 0.0


async def test_project_constraints_are_part_of_outcome_applicability() -> None:
    project_id = "project-one"
    target_id = "previously-failed"
    records = [
        record(target_id, "agent memory option", "family-a"),
        record("alternative", "agent memory option", "family-b"),
    ]
    history = {
        "projects": [Project(id=project_id, name="Local assistant", constraints=["local deployment"])],
        "outcomes": [
            ExperimentOutcome(
                project_id=project_id,
                artifact_id=target_id,
                outcome="Exceeded latency budget.",
                constraints=["local deployment"],
                succeeded=False,
            )
        ],
    }
    retriever = HybridRetriever(
        repository=MemoryRepository(records, history),  # type: ignore[arg-type]
        embedding_provider=DeterministicBlake2EmbeddingProvider(),
    )

    hits = await retriever.search(
        SearchQuery(query="compare agent memory options", project_id=project_id, limit=2)
    )

    scores = {hit.artifact.id: hit.scores.project_history for hit in hits}
    assert scores[target_id] == -0.5


async def test_unconstrained_failure_does_not_create_a_permanent_penalty() -> None:
    project_id = "project-one"
    target_id = "previously-failed"
    records = [
        record(target_id, "agent memory option", "family-a"),
        record("alternative", "agent memory option", "family-b"),
    ]
    outcome = ExperimentOutcome(
        project_id=project_id,
        artifact_id=target_id,
        outcome="The first experiment failed without recorded constraints.",
        constraints=[],
        succeeded=False,
    )
    retriever = HybridRetriever(
        repository=MemoryRepository(records, {"outcomes": [outcome]}),  # type: ignore[arg-type]
        embedding_provider=DeterministicBlake2EmbeddingProvider(),
    )

    hits = await retriever.search(
        SearchQuery(query="compare agent memory options", project_id=project_id, limit=2)
    )

    assert all(hit.scores.project_history == 0.0 for hit in hits)
