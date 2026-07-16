"""Deterministic, structured scoring for the offline evaluation corpus."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from steering.domain.models import IssueStatus, TrustLane
from steering.evaluation.corpus import EvaluationCorpus


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    prompt: str
    preferred_operation: str
    relevant_candidate_ids: tuple[str, ...]
    minimum_relevant_hits: int
    minimum_strategy_families: int
    multi_item_expected: bool


@dataclass(frozen=True)
class CaseScore:
    case_id: str
    retrieved_candidate_ids: tuple[str, ...]
    relevant_hits: tuple[str, ...]
    recall_at_k: float
    reciprocal_rank: float
    canonical_recall_at_k: float
    relevant_hit_families: tuple[str, ...]
    minimum_hits_satisfied: bool
    family_diversity_satisfied: bool


@dataclass(frozen=True)
class RetrievalScore:
    cases: tuple[CaseScore, ...]
    macro_recall_at_k: float
    micro_recall_at_k: float
    mean_reciprocal_rank: float
    macro_canonical_recall_at_k: float
    minimum_hits_pass_rate: float
    family_diversity_pass_rate: float


@dataclass(frozen=True)
class FactCitation:
    """One externally checkable response fact and its retained evidence pointer."""

    text: str
    externally_checkable: bool = True
    source_url: str | None = None
    snapshot_id: str | None = None
    evidence_span_id: str | None = None


@dataclass(frozen=True)
class TrustPresentation:
    """Structured UI/API presentation fields used by trust assertions."""

    candidate_id: str
    displayed_lane: TrustLane
    recommended: bool
    warning_displayed: bool
    surfaced_issue_ids: tuple[str, ...] = ()
    labeled_unrestricted_open_source: bool = False


@dataclass(frozen=True)
class TrustAssertionResult:
    candidate_id: str
    passed: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class RecencyNeutralityResult:
    exact_order_match: bool
    top_k_set_match: bool
    baseline: tuple[str, ...]
    counterfactual: tuple[str, ...]


def load_evaluation_cases(path: Path) -> tuple[EvaluationCase, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases: list[EvaluationCase] = []
    for item in payload["prompts"]:
        expected = item["expected_retrieval"]
        cases.append(
            EvaluationCase(
                id=str(item["id"]),
                prompt=str(item["prompt"]),
                preferred_operation=str(item["preferred_mcp_operation"]),
                relevant_candidate_ids=tuple(expected["relevant_candidate_ids"]),
                minimum_relevant_hits=int(expected["minimum_relevant_hits"]),
                minimum_strategy_families=int(expected["minimum_strategy_families"]),
                multi_item_expected=bool(item["multi_item_expected"]),
            )
        )
    return tuple(cases)


def _canonical_key(candidate_id: str, corpus: EvaluationCorpus) -> str:
    record = corpus.by_candidate_id[candidate_id]
    return record.artifact.canonical_url or record.artifact.id


def score_rankings(
    cases: Sequence[EvaluationCase],
    rankings: Mapping[str, Sequence[str]],
    corpus: EvaluationCorpus,
    *,
    k: int = 10,
) -> RetrievalScore:
    """Score only seeded positives; unjudged results are not treated as negatives."""
    scored: list[CaseScore] = []
    total_hits = 0
    total_relevant = 0
    canonical_scores: list[float] = []
    for case in cases:
        retrieved = tuple(dict.fromkeys(rankings.get(case.id, ())))[:k]
        relevant = set(case.relevant_candidate_ids)
        hits = tuple(candidate_id for candidate_id in retrieved if candidate_id in relevant)
        families = tuple(
            sorted({corpus.by_candidate_id[candidate_id].artifact.strategy_family for candidate_id in hits})
        )
        relevant_canonical = {
            _canonical_key(candidate_id, corpus) for candidate_id in case.relevant_candidate_ids
        }
        retrieved_canonical = {
            _canonical_key(candidate_id, corpus)
            for candidate_id in retrieved
            if candidate_id in corpus.by_candidate_id
        }
        canonical_recall = len(relevant_canonical & retrieved_canonical) / len(relevant_canonical)
        recall = len(hits) / len(relevant)
        reciprocal_rank = next(
            (1.0 / rank for rank, candidate_id in enumerate(retrieved, start=1) if candidate_id in relevant),
            0.0,
        )
        total_hits += len(hits)
        total_relevant += len(relevant)
        canonical_scores.append(canonical_recall)
        scored.append(
            CaseScore(
                case_id=case.id,
                retrieved_candidate_ids=retrieved,
                relevant_hits=hits,
                recall_at_k=recall,
                reciprocal_rank=reciprocal_rank,
                canonical_recall_at_k=canonical_recall,
                relevant_hit_families=families,
                minimum_hits_satisfied=len(hits) >= case.minimum_relevant_hits,
                family_diversity_satisfied=len(families) >= case.minimum_strategy_families,
            )
        )
    denominator = len(scored) or 1
    return RetrievalScore(
        cases=tuple(scored),
        macro_recall_at_k=sum(item.recall_at_k for item in scored) / denominator,
        micro_recall_at_k=total_hits / (total_relevant or 1),
        mean_reciprocal_rank=sum(item.reciprocal_rank for item in scored) / denominator,
        macro_canonical_recall_at_k=sum(canonical_scores) / denominator,
        minimum_hits_pass_rate=sum(item.minimum_hits_satisfied for item in scored) / denominator,
        family_diversity_pass_rate=(sum(item.family_diversity_satisfied for item in scored) / denominator),
    )


def citation_coverage(facts: Sequence[FactCitation]) -> float:
    checkable = [fact for fact in facts if fact.externally_checkable]
    if not checkable:
        return 1.0
    cited = sum(bool(fact.source_url and fact.snapshot_id and fact.evidence_span_id) for fact in checkable)
    return cited / len(checkable)


def assert_trust_presentations(
    presentations: Sequence[TrustPresentation], corpus: EvaluationCorpus
) -> tuple[TrustAssertionResult, ...]:
    results: list[TrustAssertionResult] = []
    for presentation in presentations:
        record = corpus.by_candidate_id[presentation.candidate_id]
        artifact = record.artifact
        failures: list[str] = []
        unresolved_ids = {issue.id for issue in record.issues if issue.status == IssueStatus.UNRESOLVED}
        is_post_only = artifact.metadata.get("primary_category") == "post_only"
        if is_post_only and presentation.displayed_lane != TrustLane.EXPERIMENTAL:
            failures.append("post-only evidence must remain Experimental")
        if unresolved_ids and presentation.displayed_lane == TrustLane.ESTABLISHED:
            failures.append("unresolved evidence cannot be displayed as Established")
        if (is_post_only or unresolved_ids) and presentation.recommended:
            failures.append("unverified or unresolved evidence must not be recommended")
        if (is_post_only or unresolved_ids) and not presentation.warning_displayed:
            failures.append("unverified or unresolved evidence needs a warning")
        if unresolved_ids - set(presentation.surfaced_issue_ids):
            failures.append("unresolved issue was not surfaced")
        license_text = (artifact.license or "").lower()
        noncommercial = "noncommercial" in license_text or "non-commercial" in license_text
        if noncommercial and presentation.labeled_unrestricted_open_source:
            failures.append("noncommercial license was labeled unrestricted open source")
        results.append(
            TrustAssertionResult(
                candidate_id=presentation.candidate_id,
                passed=not failures,
                failures=tuple(failures),
            )
        )
    return tuple(results)


def compare_recency_neutral_rankings(
    baseline: Sequence[str], counterfactual: Sequence[str], *, k: int = 10
) -> RecencyNeutralityResult:
    baseline_top = tuple(baseline[:k])
    counterfactual_top = tuple(counterfactual[:k])
    return RecencyNeutralityResult(
        exact_order_match=baseline_top == counterfactual_top,
        top_k_set_match=set(baseline_top) == set(counterfactual_top),
        baseline=baseline_top,
        counterfactual=counterfactual_top,
    )
