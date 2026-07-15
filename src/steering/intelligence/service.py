from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from steering.domain.models import (
    ArchitectureReview,
    ArtifactRecord,
    Decision,
    ExperimentOutcome,
    IdeaCard,
    IdeaEvidence,
    IssueStatus,
    Project,
    SearchHit,
    SearchQuery,
)
from steering.domain.protocols import ArtifactRepository, GenerationProvider
from steering.retrieval.hybrid import HybridRetriever, tokenize
from steering.retrieval.vocabulary import ENGINEERING_CONCEPTS

ARCHITECTURE_AREAS = {
    "memory": {"memory", "remember", "state", "context"},
    "retrieval": {"retrieval", "rag", "search", "vector", "index"},
    "orchestration": {"agent", "workflow", "tool", "harness", "multi-agent"},
    "training": {"train", "training", "fine-tune", "pretraining", "rl", "grpo"},
    "inference": {"inference", "latency", "decode", "serving", "gpu", "cache"},
    "evaluation": {"evaluation", "benchmark", "metric", "test", "accuracy"},
    "observability": {"observability", "trace", "audit", "provenance", "debug"},
    "cost": {"cost", "token", "budget", "memory limit", "storage"},
    "security": {"security", "private", "permission", "approval", "local"},
}

SYNTHESIS_SYSTEM_PROMPT = """You are an evidence-bound AI architecture reviewer.
Use only the supplied retrieved records and preserve their trust and evidence distinctions.
Never present experimental or unresolved social evidence as established.
The application independently supplies all factual descriptions, observations, citations, trust
labels, and evidence. Your useful contribution is limited to small reversible experiment proposals
and their success and failure criteria. Do not add numerical results or claims about a source to
those proposal fields.
Return exactly the requested JSON schema."""


class SteeringEngine:
    def __init__(
        self,
        *,
        repository: ArtifactRepository,
        retriever: HybridRetriever,
        generation: GenerationProvider | None = None,
    ) -> None:
        self.repository = repository
        self.retriever = retriever
        self.generation = generation

    async def search(self, query: SearchQuery) -> list[SearchHit]:
        return await self.retriever.search(query)

    def get_knowledge_record(self, artifact_id: str) -> ArtifactRecord | None:
        return self.repository.get_record(artifact_id)

    async def explore_design_options(
        self,
        problem: str,
        *,
        constraints: str = "",
        project_id: str | None = None,
        limit: int = 12,
    ) -> ArchitectureReview:
        return await self.review_architecture(problem, constraints, project_id=project_id, limit=limit)

    async def review_architecture(
        self,
        architecture: str,
        requirements: str = "",
        concerns: Sequence[str] | None = None,
        *,
        project_id: str | None = None,
        limit: int = 12,
    ) -> ArchitectureReview:
        query_text = "\n".join(
            part for part in [architecture, requirements, " ".join(concerns or [])] if part.strip()
        )
        areas = decompose_architecture(query_text)
        history = self.repository.project_history(project_id) if project_id else {}
        history_observations = self._history_observations(history)
        hits = await self.retriever.search(
            SearchQuery(
                query=query_text,
                limit=min(limit, 12),
                breadth=True,
                project_id=project_id,
            )
        )
        deterministic = self._deterministic_review(
            query_text,
            areas,
            hits,
            history_observations=history_observations,
        )
        if self.generation is None or not hits:
            return deterministic
        prompt = json.dumps(
            {
                "architecture": architecture,
                "requirements": requirements,
                "concerns": list(concerns or []),
                "project_history": self._history_payload(history),
                "retrieved": [
                    {
                        "artifact_id": hit.artifact.id,
                        "title": hit.artifact.title,
                        "summary": hit.artifact.summary,
                        "trust_lane": hit.artifact.trust_lane,
                        "capabilities": hit.artifact.capabilities,
                        "limitations": hit.artifact.limitations,
                        "requirements": hit.artifact.requirements,
                        "citations": hit.citation_urls,
                        "uncertainty": hit.uncertainty,
                    }
                    for hit in hits
                ],
            },
            ensure_ascii=False,
        )
        generated = await self.generation.generate_structured(
            system_prompt=SYNTHESIS_SYSTEM_PROMPT,
            user_prompt=prompt,
            response_model=ArchitectureReview,
        )
        hits_by_artifact = {hit.artifact.id: hit for hit in hits}
        returned_artifacts = [card.artifact_id for card in generated.idea_cards]
        if len(returned_artifacts) != len(set(returned_artifacts)):
            return deterministic
        if any(artifact_id not in hits_by_artifact for artifact_id in returned_artifacts):
            return deterministic
        generated_cards = {card.artifact_id: card for card in generated.idea_cards}
        evidence_bound_cards = [
            card.model_copy(
                update={
                    "minimal_experiment": generated_cards[card.artifact_id].minimal_experiment,
                    "success_criteria": generated_cards[card.artifact_id].success_criteria,
                    "failure_criteria": generated_cards[card.artifact_id].failure_criteria,
                }
            )
            if card.artifact_id in generated_cards
            else card
            for card in deterministic.idea_cards
        ]
        return deterministic.model_copy(
            update={
                "idea_cards": evidence_bound_cards,
            }
        )

    async def compare_entities(
        self,
        artifact_ids: Sequence[str],
        project_constraints: str = "",
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for artifact_id in dict.fromkeys(artifact_ids):
            record = self.repository.get_record(artifact_id)
            if record is None:
                continue
            artifact = record.artifact
            fit_tokens = set(tokenize(project_constraints)) & set(
                tokenize(" ".join([*artifact.capabilities, *artifact.requirements, *artifact.use_cases]))
            )
            rows.append(
                {
                    "artifact_id": artifact.id,
                    "title": artifact.title,
                    "strategy_family": artifact.strategy_family,
                    "trust_lane": artifact.trust_lane,
                    "evidence_quality": artifact.evidence_quality,
                    "capabilities": artifact.capabilities,
                    "limitations": artifact.limitations,
                    "requirements": artifact.requirements,
                    "constraint_matches": sorted(fit_tokens),
                    "source": artifact.canonical_url,
                    "issues": [issue.model_dump(mode="json") for issue in record.issues],
                }
            )
        return rows

    def record_project_decision(
        self,
        *,
        project: str,
        artifact_id: str | None,
        decision: str,
        rationale: str,
    ) -> Decision:
        existing_history = self.repository.project_history(project)
        existing_projects = existing_history.get("projects", [])
        if existing_projects and isinstance(existing_projects[0], Project):
            project_record = existing_projects[0]
        else:
            normalized_name = " ".join(project.lower().split())
            digest = hashlib.sha256(normalized_name.encode("utf-8")).hexdigest()[:24]
            project_record = Project(id=f"project_{digest}", name=project)
        project_record = self.repository.save_project(project_record)
        return self.repository.save_decision(
            Decision(
                project_id=project_record.id,
                artifact_id=artifact_id,
                decision=decision,
                rationale=rationale,
            )
        )

    def record_experiment_outcome(
        self,
        *,
        project_id: str,
        outcome: str,
        artifact_id: str | None = None,
        decision_id: str | None = None,
        constraints: Sequence[str] = (),
        succeeded: bool | None = None,
    ) -> ExperimentOutcome:
        stored = self.repository.save_outcome(
            ExperimentOutcome(
                project_id=project_id,
                decision_id=decision_id,
                artifact_id=artifact_id,
                outcome=outcome,
                constraints=list(constraints),
                succeeded=succeeded,
            )
        )
        self.retriever.mark_dirty()
        return stored

    def _deterministic_review(
        self,
        query: str,
        areas: list[str],
        hits: list[SearchHit],
        *,
        history_observations: Sequence[str] = (),
    ) -> ArchitectureReview:
        cards: list[IdeaCard] = []
        for hit in hits:
            artifact = hit.artifact
            record = self.repository.get_record(artifact.id)
            accepted_corrections = (
                [issue for issue in record.issues if issue.status == IssueStatus.ACCEPTED_CORRECTION]
                if record is not None
                else []
            )
            query_tokens = set(tokenize(query))
            fit_terms = sorted(
                query_tokens & set(tokenize(" ".join(artifact.capabilities + artifact.use_cases)))
            )
            why = (
                f"Matches the current problem through: {', '.join(fit_terms[:6])}."
                if fit_terms
                else (
                    "Provides a distinct "
                    f"{artifact.strategy_family.replace('_', ' ')} strategy worth testing."
                )
            )
            limitations = artifact.limitations or ["No limitations were captured; verify before adoption."]
            requirements = artifact.requirements or ["Confirm compatibility with the current stack."]
            cards.append(
                IdeaCard(
                    artifact_id=artifact.id,
                    title=artifact.title,
                    what_it_is=(
                        f"{artifact.summary} Source-backed correction accepted: "
                        f"{accepted_corrections[0].source_statement}"
                        if accepted_corrections
                        else artifact.summary
                    ),
                    why_it_fits=why,
                    trust_lane=artifact.trust_lane,
                    source_url=hit.citation_urls[0] if hit.citation_urls else artifact.canonical_url,
                    published_at=artifact.published_at,
                    advantages=artifact.capabilities[:5],
                    limitations=limitations[:5],
                    compatibility_requirements=requirements[:5],
                    minimal_experiment=(
                        f"Prototype {artifact.title} on one representative workload and compare it "
                        "with the current baseline."
                    ),
                    success_criteria=(
                        "Improves the target metric without violating stated cost, latency, privacy, "
                        "or quality constraints."
                    ),
                    failure_criteria=(
                        "No material improvement, unacceptable regression, or integration cost "
                        "exceeds the expected benefit."
                    ),
                    related_alternatives=[
                        alternative.artifact.title
                        for alternative in hits
                        if alternative.artifact.id != artifact.id
                        and alternative.artifact.strategy_family == artifact.strategy_family
                    ][:3],
                    uncertainty=hit.uncertainty,
                    strategy_family=artifact.strategy_family,
                    evidence=self._idea_evidence(record) if record is not None else [],
                )
            )
        families = {card.strategy_family for card in cards}
        insufficient = []
        if not cards:
            insufficient.append("The personal graph contains no relevant saved knowledge for this problem.")
        elif len(families) < min(4, len(cards)):
            insufficient.append("The current graph has limited strategy-family diversity for this problem.")
        return ArchitectureReview(
            query=query,
            decomposed_areas=areas,
            observations=[
                *history_observations,
                f"Retrieved {len(cards)} source-backed options across {len(families)} strategy families.",
            ],
            idea_cards=cards,
            insufficient_knowledge=insufficient,
            citations=list(dict.fromkeys(url for hit in hits for url in hit.citation_urls)),
        )

    @staticmethod
    def _idea_evidence(record: ArtifactRecord, limit: int = 5) -> list[IdeaEvidence]:
        spans = {span.id: span for span in record.evidence_spans}
        snapshots = {snapshot.id: snapshot for snapshot in record.snapshots}
        evidence: list[IdeaEvidence] = []
        preferred_span_ids: set[str] = set()
        excluded_span_ids: set[str] = set()
        for issue in record.issues:
            if issue.status != IssueStatus.ACCEPTED_CORRECTION:
                continue
            for span_id in issue.evidence_span_ids:
                span = spans.get(span_id)
                snapshot = snapshots.get(span.snapshot_id) if span is not None else None
                if span is None or snapshot is None:
                    continue
                if snapshot.source_url == issue.social_source_url:
                    excluded_span_ids.add(span_id)
                if snapshot.source_url != issue.primary_source_url:
                    continue
                preferred_span_ids.add(span_id)
                evidence.append(
                    IdeaEvidence(
                        claim=issue.source_statement,
                        exact_quote=span.quote,
                        source_url=snapshot.source_url,
                        snapshot_id=snapshot.id,
                        evidence_span_id=span.id,
                    )
                )
                break
        for claim in record.claims:
            for span_id in claim.evidence_span_ids:
                if span_id in preferred_span_ids or span_id in excluded_span_ids:
                    continue
                span = spans.get(span_id)
                snapshot = snapshots.get(span.snapshot_id) if span is not None else None
                if span is None or snapshot is None:
                    continue
                evidence.append(
                    IdeaEvidence(
                        claim=claim.text,
                        exact_quote=span.quote,
                        source_url=snapshot.source_url,
                        snapshot_id=snapshot.id,
                        evidence_span_id=span.id,
                    )
                )
                break
            if len(evidence) >= limit:
                break
        return evidence[:limit]

    @staticmethod
    def _history_payload(history: Mapping[str, Sequence[Any]]) -> dict[str, list[object]]:
        result: dict[str, list[object]] = {}
        for key in ("projects", "decisions", "outcomes"):
            values = history.get(key, [])
            result[key] = [
                item.model_dump(mode="json") if hasattr(item, "model_dump") else str(item) for item in values
            ]
        return result

    @staticmethod
    def _history_observations(history: Mapping[str, Sequence[Any]]) -> list[str]:
        observations: list[str] = []
        for item in list(history.get("decisions", []))[-3:]:
            if isinstance(item, Decision):
                observations.append(f"Previous project decision: {item.decision} Rationale: {item.rationale}")
        for item in list(history.get("outcomes", []))[-3:]:
            if isinstance(item, ExperimentOutcome):
                result = (
                    "succeeded"
                    if item.succeeded is True
                    else "failed"
                    if item.succeeded is False
                    else "unrated"
                )
                observations.append(f"Previous experiment outcome ({result}): {item.outcome}")
        return observations


def decompose_architecture(text: str) -> list[str]:
    tokens = set(tokenize(text))
    lowered = text.lower()
    areas = [
        area
        for area, terms in ARCHITECTURE_AREAS.items()
        if terms & tokens or any(term in lowered for term in terms if " " in term or "-" in term)
    ]
    for concept, phrases in ENGINEERING_CONCEPTS.items():
        if any(phrase in lowered for phrase in phrases) and concept not in areas:
            areas.append(concept)
    return areas or ["general_architecture"]
