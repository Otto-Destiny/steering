from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

import pytest

from steering.domain.models import (
    ArchitectureReview,
    Artifact,
    ArtifactRecord,
    ArtifactType,
    Decision,
    ExperimentOutcome,
    IdeaCard,
    IssueStatus,
    ReviewIssue,
    ReviewStatus,
    ScoreBreakdown,
    SearchHit,
    SearchQuery,
    Snapshot,
    SourceKind,
    TrustLane,
)
from steering.intelligence.service import SteeringEngine
from steering.mcp import create_mcp_server

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _record() -> ArtifactRecord:
    artifact = Artifact(
        id="art_1",
        canonical_url="https://example.com/paper",
        source_kind=SourceKind.PAPER,
        artifact_type=ArtifactType.PAPER,
        title="Versioned Agent Memory",
        summary="A versioned memory design.",
        strategy_family="versioned_memory",
        review_status=ReviewStatus.REVIEWED,
        trust_lane=TrustLane.EXPERIMENTAL,
        evidence_quality=0.4,
        license="PolyForm-Noncommercial-1.0.0",
        capabilities=["rollback"],
        limitations=["small evaluation"],
        requirements=["local database"],
        content_hash="abc",
    )
    snapshot = Snapshot(
        id="snap_1",
        artifact_id=artifact.id,
        source_url="https://example.com/paper",
        captured_at=NOW,
        content_hash="abc",
        mime_type="text/plain",
        text="evidence",
        extraction_method="test",
    )
    issue = ReviewIssue(
        id="issue_1",
        artifact_id=artifact.id,
        social_statement="unverified speedup",
        source_statement="not in paper",
        explanation="The retained sources differ.",
        social_source_url="https://x.com/example/status/1",
        primary_source_url="https://example.com/paper",
        status=IssueStatus.UNRESOLVED,
        created_at=NOW,
    )
    return ArtifactRecord(artifact=artifact, snapshots=[snapshot], issues=[issue])


def _review(record: ArtifactRecord) -> ArchitectureReview:
    artifact = record.artifact
    return ArchitectureReview(
        query="agent memory",
        decomposed_areas=["memory"],
        observations=["One retained option."],
        idea_cards=[
            IdeaCard(
                artifact_id=artifact.id,
                title=artifact.title,
                what_it_is=artifact.summary,
                why_it_fits="Supports rollback.",
                trust_lane=artifact.trust_lane,
                source_url=artifact.canonical_url,
                published_at=None,
                advantages=artifact.capabilities,
                limitations=artifact.limitations,
                compatibility_requirements=artifact.requirements,
                minimal_experiment="Test one memory workflow.",
                success_criteria="Rollback restores the prior state.",
                failure_criteria="State cannot be restored.",
                related_alternatives=[],
                uncertainty="Unresolved source issue.",
                strategy_family=artifact.strategy_family,
            )
        ],
        citations=["https://example.com/paper"],
    )


def _structured(result: object) -> dict[str, Any]:
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[1], dict)
    return result[1]


class FakeEngine:
    def __init__(self) -> None:
        self.record = _record()
        self.last_query: SearchQuery | None = None

    async def search(self, query: SearchQuery) -> list[SearchHit]:
        self.last_query = query
        return [
            SearchHit(
                artifact=self.record.artifact,
                score=0.9,
                scores=ScoreBreakdown(bm25=0.9),
                citation_urls=["https://example.com/paper"],
                uncertainty="Experimental evidence.",
            )
        ]

    def get_knowledge_record(self, artifact_id: str) -> ArtifactRecord | None:
        return self.record if artifact_id == self.record.artifact.id else None

    async def explore_design_options(self, *_args: Any, **_kwargs: Any) -> ArchitectureReview:
        return _review(self.record)

    async def compare_entities(
        self, artifact_ids: list[str], project_constraints: str = ""
    ) -> list[dict[str, object]]:
        del project_constraints
        return [
            {
                "artifact_id": self.record.artifact.id,
                "title": self.record.artifact.title,
                "source": self.record.artifact.canonical_url,
            }
            for artifact_id in artifact_ids
            if artifact_id == self.record.artifact.id
        ]

    async def review_architecture(self, *_args: Any, **_kwargs: Any) -> ArchitectureReview:
        return _review(self.record)

    def record_project_decision(self, **kwargs: Any) -> Decision:
        return Decision(
            id="decision_1",
            project_id="project_1",
            artifact_id=kwargs.get("artifact_id"),
            decision=str(kwargs["decision"]),
            rationale=str(kwargs["rationale"]),
            created_at=NOW,
        )

    def record_experiment_outcome(self, **kwargs: Any) -> ExperimentOutcome:
        return ExperimentOutcome(
            id="outcome_1",
            project_id=str(kwargs["project_id"]),
            artifact_id=kwargs.get("artifact_id"),
            outcome=str(kwargs["outcome"]),
            constraints=list(kwargs["constraints"]),
            succeeded=kwargs.get("succeeded"),
            recorded_at=NOW,
        )


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def server(engine: FakeEngine):
    return create_mcp_server(cast(SteeringEngine, engine))


async def test_server_contract_and_mount_path(server) -> None:
    tools = await server.list_tools()
    assert {tool.name for tool in tools} == {
        "search_knowledge",
        "get_knowledge_record",
        "explore_design_options",
        "compare_entities",
        "review_architecture",
        "record_project_decision",
        "record_experiment_outcome",
    }
    assert server.settings.stateless_http is True
    assert server.settings.json_response is True
    assert server.settings.streamable_http_path == "/"
    assert [route.path for route in server.streamable_http_app().routes] == ["/"]
    assert {prompt.name for prompt in await server.list_prompts()} == {"design_with_steering"}
    assert {str(resource.uri) for resource in await server.list_resources()} == {"steering://instructions"}


async def test_search_returns_json_safe_citations_and_warnings(server, engine: FakeEngine) -> None:
    result = _structured(
        await server.call_tool("search_knowledge", {"query": "memory rollback", "limit": 5, "breadth": True})
    )
    assert result["result_count"] == 1
    item = result["results"][0]
    assert item["citations"] == ["https://example.com/paper"]
    assert "Experimental evidence: validate before adoption." in item["trust_warnings"]
    assert any("unresolved source" in warning for warning in item["trust_warnings"])
    assert any("noncommercial" in warning.lower() for warning in item["trust_warnings"])
    assert engine.last_query is not None
    assert engine.last_query.limit == 5


async def test_read_and_design_tools_preserve_structured_trust_data(server) -> None:
    record = _structured(await server.call_tool("get_knowledge_record", {"artifact_id": "art_1"}))
    assert record["found"] is True
    assert record["record"]["citations"] == ["https://example.com/paper"]

    review = _structured(
        await server.call_tool(
            "review_architecture",
            {"architecture": "rolling summaries", "requirements": "rollback"},
        )
    )
    assert review["citations"] == ["https://example.com/paper"]
    assert review["trust_warnings_by_artifact"]["art_1"]

    options = _structured(
        await server.call_tool(
            "explore_design_options",
            {"problem": "agent memory", "constraints": "offline rollback"},
        )
    )
    assert options["idea_cards"][0]["artifact_id"] == "art_1"

    comparison = _structured(
        await server.call_tool(
            "compare_entities",
            {"artifact_ids": ["art_1"], "project_constraints": "offline rollback"},
        )
    )
    assert comparison["comparisons"][0]["artifact_id"] == "art_1"
    assert comparison["comparisons"][0]["trust_warnings"]


async def test_mutating_tools_return_serializable_receipts(server) -> None:
    decision = _structured(
        await server.call_tool(
            "record_project_decision",
            {
                "project": "assistant",
                "artifact_id": "art_1",
                "decision": "prototype",
                "rationale": "rollback support",
            },
        )
    )
    assert decision["recorded"] is True
    assert decision["decision"]["created_at"] == "2026-01-01T00:00:00Z"

    outcome = _structured(
        await server.call_tool(
            "record_experiment_outcome",
            {
                "project_id": "project_1",
                "artifact_id": "art_1",
                "outcome": "rollback passed",
                "constraints": ["offline"],
                "succeeded": True,
            },
        )
    )
    assert outcome["recorded"] is True
    assert outcome["outcome"]["constraints"] == ["offline"]


async def test_prompt_includes_problem_and_evidence_rules(server) -> None:
    prompt = await server.get_prompt(
        "design_with_steering", {"problem": "agent memory", "constraints": "offline"}
    )
    rendered = "\n".join(message.content.text for message in prompt.messages)
    assert "agent memory" in rendered
    assert "offline" in rendered
    assert "cite retained sources" in rendered
    assert "after previous approaches fail" in rendered
    assert "before declaring that reasonable options are exhausted" in rendered
