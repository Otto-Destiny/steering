"""Official MCP transport adapter for an injected :class:`SteeringEngine`."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette

from steering.domain.models import (
    ArchitectureReview,
    ArtifactRecord,
    ArtifactType,
    IssueStatus,
    SearchHit,
    SearchQuery,
    TrustLane,
)
from steering.intelligence.service import SteeringEngine
from steering.mcp.instructions import AGENT_WORKFLOW_INSTRUCTIONS

SERVER_INSTRUCTIONS = (
    "STEERING recalls a user's saved AI-engineering knowledge.\n\n" + AGENT_WORKFLOW_INSTRUCTIONS
)


def _trust_warnings(record: ArtifactRecord | None) -> list[str]:
    if record is None:
        return []
    warnings: list[str] = []
    artifact = record.artifact
    if artifact.trust_lane == TrustLane.EXPERIMENTAL:
        warnings.append("Experimental evidence: validate before adoption.")
    unresolved = [issue for issue in record.issues if issue.status == IssueStatus.UNRESOLVED]
    if unresolved:
        warnings.append(f"{len(unresolved)} unresolved source issue(s) require review before recommendation.")
    license_text = (artifact.license or "").lower()
    if "noncommercial" in license_text or "non-commercial" in license_text:
        warnings.append("License includes a noncommercial restriction; it is not unrestricted open source.")
    return warnings


def _record_payload(record: ArtifactRecord) -> dict[str, Any]:
    payload = record.model_dump(mode="json")
    payload["citations"] = list(
        dict.fromkeys(
            [
                url
                for url in [
                    record.artifact.canonical_url,
                    *(snapshot.source_url for snapshot in record.snapshots),
                ]
                if url
            ]
        )
    )
    payload["trust_warnings"] = _trust_warnings(record)
    return payload


def _hit_payload(hit: SearchHit, engine: SteeringEngine) -> dict[str, Any]:
    record = engine.get_knowledge_record(hit.artifact.id)
    citations = list(dict.fromkeys([url for url in [*hit.citation_urls, hit.artifact.canonical_url] if url]))
    return {
        "artifact": hit.artifact.model_dump(mode="json"),
        "score": hit.score,
        "score_breakdown": hit.scores.model_dump(mode="json"),
        "matched_chunks": [chunk.model_dump(mode="json") for chunk in hit.matched_chunks],
        "citations": citations,
        "uncertainty": hit.uncertainty,
        "trust_warnings": _trust_warnings(record),
    }


def _review_payload(review: ArchitectureReview, engine: SteeringEngine) -> dict[str, Any]:
    payload = review.model_dump(mode="json")
    warnings: dict[str, list[str]] = {}
    for card in review.idea_cards:
        card_warnings = _trust_warnings(engine.get_knowledge_record(card.artifact_id))
        if card_warnings:
            warnings[card.artifact_id] = card_warnings
    payload["trust_warnings_by_artifact"] = warnings
    return payload


def create_mcp_server(engine: SteeringEngine) -> FastMCP:
    """Create a stateless Streamable HTTP MCP server around an injected engine."""
    server = FastMCP(
        name="STEERING",
        instructions=SERVER_INSTRUCTIONS,
        stateless_http=True,
        json_response=True,
        streamable_http_path="/",
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[
                "localhost",
                "127.0.0.1",
                "[::1]",
                "localhost:*",
                "127.0.0.1:*",
                "[::1]:*",
            ],
            allowed_origins=[
                "http://localhost",
                "http://127.0.0.1",
                "http://[::1]",
                "http://localhost:*",
                "http://127.0.0.1:*",
                "http://[::1]:*",
            ],
        ),
    )

    @server.tool(
        name="search_knowledge",
        description=(
            "Search saved AI-engineering knowledge. Returns ranked records with citations, "
            "evidence lanes, limitations, and explicit trust warnings."
        ),
        structured_output=True,
    )
    async def search_knowledge(
        query: str,
        limit: int = 10,
        breadth: bool = False,
        artifact_types: list[str] | None = None,
        minimum_evidence: float | None = None,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        parsed_types = (
            [ArtifactType(value) for value in artifact_types] if artifact_types is not None else None
        )
        hits = await engine.search(
            SearchQuery(
                query=query,
                artifact_types=parsed_types,
                minimum_evidence=minimum_evidence,
                limit=limit,
                breadth=breadth,
                project_id=project_id,
            )
        )
        return {
            "query": query,
            "result_count": len(hits),
            "results": [_hit_payload(hit, engine) for hit in hits],
        }

    @server.tool(
        name="get_knowledge_record",
        description="Get one complete retained knowledge record, its evidence, issues, and citations.",
        structured_output=True,
    )
    async def get_knowledge_record(artifact_id: str) -> dict[str, Any]:
        record = engine.get_knowledge_record(artifact_id)
        if record is None:
            return {"found": False, "artifact_id": artifact_id, "record": None}
        return {"found": True, "artifact_id": artifact_id, "record": _record_payload(record)}

    @server.tool(
        name="explore_design_options",
        description=(
            "Retrieve several strategy-family-diverse options for an engineering problem and "
            "propose small validation experiments."
        ),
        structured_output=True,
    )
    async def explore_design_options(
        problem: str,
        constraints: str = "",
        project_id: str | None = None,
        limit: int = 12,
    ) -> dict[str, Any]:
        review = await engine.explore_design_options(
            problem,
            constraints=constraints,
            project_id=project_id,
            limit=limit,
        )
        return _review_payload(review, engine)

    @server.tool(
        name="compare_entities",
        description=(
            "Compare specific saved artifacts by capabilities, limitations, requirements, "
            "evidence quality, license, issues, and fit to project constraints."
        ),
        structured_output=True,
    )
    async def compare_entities(artifact_ids: list[str], project_constraints: str = "") -> dict[str, Any]:
        rows = await engine.compare_entities(artifact_ids, project_constraints)
        enriched: list[dict[str, Any]] = []
        for row in rows:
            artifact_id = str(row["artifact_id"])
            item = dict(row)
            item["trust_warnings"] = _trust_warnings(engine.get_knowledge_record(artifact_id))
            enriched.append(item)
        return {"requested_artifact_ids": artifact_ids, "comparisons": enriched}

    @server.tool(
        name="review_architecture",
        description=(
            "Challenge an architecture using retained evidence, distinct strategy families, "
            "trade-offs, citations, and reversible experiments."
        ),
        structured_output=True,
    )
    async def review_architecture(
        architecture: str,
        requirements: str = "",
        concerns: list[str] | None = None,
        project_id: str | None = None,
        limit: int = 12,
    ) -> dict[str, Any]:
        review = await engine.review_architecture(
            architecture,
            requirements,
            concerns,
            project_id=project_id,
            limit=limit,
        )
        return _review_payload(review, engine)

    @server.tool(
        name="record_project_decision",
        description="Persist a user-approved project decision and its rationale.",
        structured_output=True,
    )
    async def record_project_decision(
        project: str,
        decision: str,
        rationale: str,
        artifact_id: str | None = None,
    ) -> dict[str, Any]:
        stored = engine.record_project_decision(
            project=project,
            artifact_id=artifact_id,
            decision=decision,
            rationale=rationale,
        )
        return {"recorded": True, "decision": stored.model_dump(mode="json")}

    @server.tool(
        name="record_experiment_outcome",
        description="Persist a user-provided experiment outcome so later retrieval can learn from it.",
        structured_output=True,
    )
    async def record_experiment_outcome(
        project_id: str,
        outcome: str,
        artifact_id: str | None = None,
        decision_id: str | None = None,
        constraints: list[str] | None = None,
        succeeded: bool | None = None,
    ) -> dict[str, Any]:
        stored = engine.record_experiment_outcome(
            project_id=project_id,
            outcome=outcome,
            artifact_id=artifact_id,
            decision_id=decision_id,
            constraints=constraints or (),
            succeeded=succeeded,
        )
        return {"recorded": True, "outcome": stored.model_dump(mode="json")}

    @server.prompt(
        name="design_with_steering",
        description="Guide an architecture design session grounded in saved STEERING evidence.",
    )
    async def design_with_steering(
        problem: str, constraints: str = "", current_architecture: str = ""
    ) -> str:
        return (
            f"{AGENT_WORKFLOW_INSTRUCTIONS}\n\n"
            "For this design session, first search the retained knowledge, then "
            "compare genuinely different strategy families. Keep evidence strength and maturity explicit; "
            "cite retained sources; surface unresolved claims as warnings; and "
            "propose small reversible experiments with success and failure criteria.\n\n"
            f"Problem:\n{problem}\n\nConstraints:\n{constraints or 'Not specified.'}\n\n"
            f"Current architecture:\n{current_architecture or 'Not specified.'}"
        )

    @server.resource(
        "steering://instructions",
        name="STEERING usage instructions",
        description="Stable guidance for evidence-bound use of the STEERING tools.",
        mime_type="text/plain",
    )
    async def instructions() -> str:
        return SERVER_INSTRUCTIONS

    return server


def streamable_http_app(engine: SteeringEngine) -> Starlette:
    """Return the ASGI app suitable for ``Mount('/mcp', app=...)``."""
    return create_mcp_server(engine).streamable_http_app()
