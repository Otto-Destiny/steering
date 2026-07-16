from __future__ import annotations

from pathlib import Path

import pytest
from tests.support.providers import FakeGenerationProvider, HashEmbeddingProvider

from steering.database import DatabaseRuntime
from steering.domain.models import (
    Artifact,
    ArtifactRecord,
    ArtifactType,
    Chunk,
    Claim,
    EvidenceCategory,
    EvidenceSpan,
    IssueStatus,
    ReviewIssue,
    ReviewStatus,
    Snapshot,
    SourceKind,
    TrustLane,
)
from steering.intelligence.service import SteeringEngine, decompose_architecture
from steering.retrieval.hybrid import HybridRetriever


def _record(
    identifier: str,
    *,
    title: str,
    family: str,
    summary: str,
    capabilities: list[str],
    limitations: list[str] | None = None,
    requirements: list[str] | None = None,
) -> ArtifactRecord:
    chunk_text = "\n".join(
        [
            title,
            summary,
            *capabilities,
            *(limitations or []),
            *(requirements or []),
        ]
    )
    return ArtifactRecord(
        artifact=Artifact(
            id=identifier,
            canonical_url=f"https://example.com/{identifier}",
            source_kind=SourceKind.PAPER,
            artifact_type=ArtifactType.PAPER,
            title=title,
            summary=summary,
            strategy_family=family,
            review_status=ReviewStatus.REVIEWED,
            trust_lane=TrustLane.PROMISING,
            evidence_quality=0.8,
            capabilities=capabilities,
            limitations=limitations or [],
            requirements=requirements or [],
            use_cases=["agent architecture"],
            content_hash=identifier,
        ),
        chunks=[
            Chunk(
                id=f"chunk-{identifier}",
                artifact_id=identifier,
                snapshot_id=f"snapshot-{identifier}",
                ordinal=0,
                text=chunk_text,
                locator="fixture summary",
            )
        ],
    )


def _seed(database: DatabaseRuntime) -> None:
    records = [
        _record(
            "memory",
            title="Stateful memory",
            family="state_management",
            summary="Retains agent state with explicit decay and provenance.",
            capabilities=["agent memory", "state retention"],
            limitations=["Needs a retention policy"],
            requirements=["Persistent store"],
        ),
        _record(
            "retrieval",
            title="Sparse retrieval",
            family="sparse_retrieval",
            summary="Uses lexical retrieval for identifiers and exact terminology.",
            capabilities=["retrieval", "exact search"],
        ),
        _record(
            "evaluation",
            title="Agent evaluation harness",
            family="evaluation_harness",
            summary="Measures agent reliability using task-level evaluations.",
            capabilities=["agent evaluation", "benchmarking"],
        ),
        _record(
            "inference",
            title="KV cache serving",
            family="inference_optimization",
            summary="Reduces repeated inference work with a managed KV cache.",
            capabilities=["inference latency", "token cost"],
        ),
    ]
    for record in records:
        database.repository.upsert_record(record)


def _engine(database: DatabaseRuntime, generation: FakeGenerationProvider | None = None) -> SteeringEngine:
    retriever = HybridRetriever(
        repository=database.repository,
        embedding_provider=HashEmbeddingProvider(dimension=768),
    )
    return SteeringEngine(
        repository=database.repository,
        retriever=retriever,
        generation=generation,
    )


@pytest.mark.asyncio
async def test_architecture_review_returns_diverse_evidence_bound_cards(tmp_path: Path) -> None:
    with DatabaseRuntime(tmp_path / "knowledge.lbug") as database:
        _seed(database)
        engine = _engine(database)

        review = await engine.explore_design_options(
            "Design an agent with memory, retrieval, evaluation, and low-cost inference",
            constraints="local persistent storage and measurable latency",
        )

        assert {"memory", "retrieval", "evaluation", "inference"} <= set(review.decomposed_areas)
        assert len({card.strategy_family for card in review.idea_cards}) == 4
        assert all(card.source_url and card.minimal_experiment for card in review.idea_cards)
        assert all(card.success_criteria and card.failure_criteria for card in review.idea_cards)
        assert len(review.citations) == 4
        assert review.insufficient_knowledge == []
        assert engine.get_knowledge_record("memory") is not None


@pytest.mark.asyncio
async def test_generation_can_refine_only_proposals_for_retrieved_artifacts(
    tmp_path: Path,
) -> None:
    with DatabaseRuntime(tmp_path / "knowledge.lbug") as database:
        _seed(database)
        baseline = await _engine(database).review_architecture("agent memory retrieval evaluation")
        generated = baseline.model_copy(deep=True)
        generated.observations = ["Unsupported generated factual synthesis"]
        generated.citations = ["https://fabricated.example/claim"]
        generated.idea_cards[0].what_it_is = "Unsupported generated source claim."
        generated.idea_cards[0].why_it_fits = "Unsupported generated fit claim."
        generated.idea_cards[0].related_alternatives = ["Invented Tool"]
        generated.idea_cards[0].minimal_experiment = "Run the generated bounded experiment."
        generated.idea_cards[0].success_criteria = "The stated project metric improves."
        generated.idea_cards[0].failure_criteria = "The stated project metric regresses."
        provider = FakeGenerationProvider(lambda *_args: generated)

        accepted = await _engine(database, provider).review_architecture("agent memory retrieval evaluation")
        assert accepted.observations == baseline.observations
        assert accepted.citations == baseline.citations
        assert accepted.idea_cards[0].what_it_is == baseline.idea_cards[0].what_it_is
        assert accepted.idea_cards[0].why_it_fits == baseline.idea_cards[0].why_it_fits
        assert accepted.idea_cards[0].related_alternatives == baseline.idea_cards[0].related_alternatives
        assert accepted.idea_cards[0].minimal_experiment == "Run the generated bounded experiment."
        assert provider.calls

        bad_cards = generated.model_copy(deep=True)
        bad_cards.idea_cards[0].artifact_id = "unknown"
        rejected_card = await _engine(
            database, FakeGenerationProvider(lambda *_args: bad_cards)
        ).review_architecture("agent memory retrieval evaluation")
        assert all(card.artifact_id != "unknown" for card in rejected_card.idea_cards)

        wrong_lane = generated.model_copy(deep=True)
        wrong_lane.idea_cards[0].trust_lane = TrustLane.EXPERIMENTAL
        rejected_lane = await _engine(
            database, FakeGenerationProvider(lambda *_args: wrong_lane)
        ).review_architecture("agent memory retrieval evaluation")
        assert rejected_lane.idea_cards[0].trust_lane == TrustLane.PROMISING

        wrong_source = generated.model_copy(deep=True)
        wrong_source.idea_cards[0].source_url = generated.idea_cards[1].source_url
        rejected_source = await _engine(
            database, FakeGenerationProvider(lambda *_args: wrong_source)
        ).review_architecture("agent memory retrieval evaluation")
        assert rejected_source.idea_cards[0].source_url != wrong_source.idea_cards[0].source_url


@pytest.mark.asyncio
async def test_unresolved_experimental_knowledge_keeps_lane_and_warning(tmp_path: Path) -> None:
    with DatabaseRuntime(tmp_path / "knowledge.lbug") as database:
        value = _record(
            "unresolved",
            title="Unresolved memory claim",
            family="state_management",
            summary="A social claim that conflicts with its primary source.",
            capabilities=["agent memory"],
        )
        value = value.model_copy(
            update={
                "artifact": value.artifact.model_copy(update={"trust_lane": TrustLane.EXPERIMENTAL}),
                "issues": [
                    ReviewIssue(
                        artifact_id="unresolved",
                        social_statement="The method is lossless.",
                        source_statement="The method may lose information.",
                        explanation="The retained sources disagree.",
                        social_source_url="https://social.example/post",
                        primary_source_url="https://example.com/unresolved",
                    )
                ],
            }
        )
        database.repository.upsert_record(value)
        baseline = await _engine(database).review_architecture("agent memory")
        generated = baseline.model_copy(update={"observations": ["Relabeled synthesis"]}, deep=True)
        generated.idea_cards[0].trust_lane = TrustLane.PROMISING
        generated.idea_cards[0].uncertainty = None

        result = await _engine(
            database, FakeGenerationProvider(lambda *_args: generated)
        ).review_architecture("agent memory")

        assert result.idea_cards[0].trust_lane == TrustLane.EXPERIMENTAL
        assert "unresolved" in (result.idea_cards[0].uncertainty or "").lower()


@pytest.mark.asyncio
async def test_accepted_correction_prefers_primary_evidence_without_removing_claims(
    tmp_path: Path,
) -> None:
    social_quote = "The post claims the method is perfectly lossless."
    primary_quote = "The paper reports measurable information loss."
    with DatabaseRuntime(tmp_path / "knowledge.lbug") as database:
        value = _record(
            "corrected",
            title="Corrected memory claim",
            family="state_management",
            summary="A social post presents a lossless memory method.",
            capabilities=["agent memory"],
        )
        issue = ReviewIssue(
            id="issue_corrected",
            artifact_id="corrected",
            social_statement="The method is perfectly lossless.",
            source_statement="The method has measurable information loss.",
            explanation="The paper contradicts the social claim.",
            social_source_url="https://social.example/post",
            primary_source_url="https://example.com/corrected",
            evidence_span_ids=["span_social", "span_primary"],
        )
        value = value.model_copy(
            update={
                "artifact": value.artifact.model_copy(update={"trust_lane": TrustLane.EXPERIMENTAL}),
                "snapshots": [
                    Snapshot(
                        id="snapshot_social",
                        artifact_id="corrected",
                        source_url=issue.social_source_url,
                        content_hash="social",
                        mime_type="text/html",
                        text=social_quote,
                        extraction_method="fixture",
                    ),
                    Snapshot(
                        id="snapshot_primary",
                        artifact_id="corrected",
                        source_url=issue.primary_source_url,
                        content_hash="primary",
                        mime_type="application/pdf",
                        text=primary_quote,
                        extraction_method="fixture",
                    ),
                ],
                "claims": [
                    Claim(
                        id="claim_social",
                        artifact_id="corrected",
                        text=issue.social_statement,
                        category=EvidenceCategory.SOCIAL_CLAIM,
                        confidence=0.5,
                        evidence_span_ids=["span_social"],
                    ),
                    Claim(
                        id="claim_primary",
                        artifact_id="corrected",
                        text=issue.source_statement,
                        category=EvidenceCategory.RESEARCH_PAPER,
                        confidence=0.8,
                        evidence_span_ids=["span_primary"],
                    ),
                ],
                "evidence_spans": [
                    EvidenceSpan(
                        id="span_social",
                        snapshot_id="snapshot_social",
                        claim_id="claim_social",
                        quote=social_quote,
                        start=0,
                        end=len(social_quote),
                        locator="social evidence",
                    ),
                    EvidenceSpan(
                        id="span_primary",
                        snapshot_id="snapshot_primary",
                        claim_id="claim_primary",
                        quote=primary_quote,
                        start=0,
                        end=len(primary_quote),
                        locator="primary evidence",
                    ),
                ],
                "issues": [issue],
            }
        )
        database.repository.upsert_record(value)

        resolved = database.repository.resolve_issue(issue.id, "accept_correction")
        assert resolved.status == IssueStatus.ACCEPTED_CORRECTION
        stored = database.repository.get_record("corrected")
        assert stored is not None
        assert len(stored.claims) == 2

        review = await _engine(database).review_architecture("agent memory information loss")
        card = next(item for item in review.idea_cards if item.artifact_id == "corrected")
        assert "Source-backed correction accepted" in card.what_it_is
        assert issue.source_statement in card.what_it_is
        assert card.evidence[0].claim == issue.source_statement
        assert card.evidence[0].exact_quote == primary_quote
        assert all(item.exact_quote != social_quote for item in card.evidence)
        assert card.trust_lane == TrustLane.EXPERIMENTAL


@pytest.mark.asyncio
async def test_compare_decisions_outcomes_and_empty_graph_are_explicit(tmp_path: Path) -> None:
    with DatabaseRuntime(tmp_path / "knowledge.lbug") as database:
        _seed(database)
        engine = _engine(database)

        comparison = await engine.compare_entities(
            ["memory", "missing", "memory", "inference"],
            "memory latency persistent",
        )
        assert [row["artifact_id"] for row in comparison] == ["memory", "inference"]
        assert "memory" in comparison[0]["constraint_matches"]

        decision = engine.record_project_decision(
            project="Memory redesign",
            artifact_id="memory",
            decision="Run a bounded prototype",
            rationale="It matches the persistence constraint.",
        )
        follow_up_decision = engine.record_project_decision(
            project="Memory redesign",
            artifact_id="inference",
            decision="Retain the current cache as the baseline",
            rationale="It gives the prototype a measurable comparison.",
        )
        assert follow_up_decision.project_id == decision.project_id
        outcome = engine.record_experiment_outcome(
            project_id=decision.project_id,
            decision_id=decision.id,
            artifact_id="memory",
            outcome="Latency improved on the representative workload.",
            constraints=["local"],
            succeeded=True,
        )
        history = database.repository.project_history(decision.project_id)
        assert outcome in history["outcomes"]
        assert decision in history["decisions"]
        assert follow_up_decision in history["decisions"]

        review = await engine.review_architecture(
            "Improve memory latency",
            project_id=decision.project_id,
        )
        assert any("Previous project decision" in item for item in review.observations)
        assert any("Previous experiment outcome (succeeded)" in item for item in review.observations)

    with DatabaseRuntime(tmp_path / "empty.lbug") as empty_database:
        review = await _engine(empty_database).review_architecture("unknown architecture")
        assert review.idea_cards == []
        assert "no relevant saved knowledge" in review.insufficient_knowledge[0]


def test_architecture_decomposition_has_specific_and_general_fallbacks() -> None:
    assert decompose_architecture("secure private RAG with trace and token budget") == [
        "retrieval",
        "observability",
        "cost",
        "security",
    ]
    assert decompose_architecture("a novel topology") == ["general_architecture"]
