from pathlib import Path

import pytest

from steering.domain.models import IssueStatus, TrustLane
from steering.evaluation.corpus import EvaluationCorpusLoader
from steering.evaluation.scoring import (
    EvaluationCase,
    FactCitation,
    TrustPresentation,
    assert_trust_presentations,
    citation_coverage,
    compare_recency_neutral_rankings,
    load_evaluation_cases,
    score_rankings,
)


@pytest.fixture(scope="module")
def corpus():
    return EvaluationCorpusLoader(
        results_directory=Path("evaluate/results"),
        categories_path=Path("evaluate/result_categories.yaml"),
        strategy_families_path=Path("evaluate/strategy_families.yaml"),
    ).load()


def test_prompt_loader_reads_all_cases_without_mutating_corpus(corpus) -> None:
    cases = load_evaluation_cases(Path("evaluate/retrieval_prompts.json"))
    assert len(cases) == 26
    assert len(corpus.records) == 51


def test_recall_diversity_and_canonical_recall(corpus) -> None:
    case = EvaluationCase(
        id="duplicate",
        prompt="memory",
        preferred_operation="search_knowledge",
        relevant_candidate_ids=("behavioral-state-decay", "proactive-memory-agent"),
        minimum_relevant_hits=2,
        minimum_strategy_families=1,
        multi_item_expected=True,
    )
    score = score_rankings([case], {case.id: ["behavioral-state-decay", "lmcache"]}, corpus)
    item = score.cases[0]
    assert item.recall_at_k == 0.5
    assert item.reciprocal_rank == 1.0
    assert score.mean_reciprocal_rank == 1.0
    assert item.canonical_recall_at_k == 1.0
    assert item.relevant_hit_families == ("behavioral_state_memory",)
    assert not item.minimum_hits_satisfied
    assert item.family_diversity_satisfied


def test_citation_coverage_counts_only_external_facts() -> None:
    facts = [
        FactCitation("cited", source_url="https://example.com", snapshot_id="s", evidence_span_id="e"),
        FactCitation("uncited"),
        FactCitation("advice", externally_checkable=False),
    ]
    assert citation_coverage(facts) == 0.5


def test_trust_assertions_require_warning_issue_and_license_care(corpus) -> None:
    candidate_id = next(
        candidate_id
        for candidate_id, record in corpus.by_candidate_id.items()
        if any(issue.status == IssueStatus.UNRESOLVED for issue in record.issues)
    )
    failures = assert_trust_presentations(
        [
            TrustPresentation(
                candidate_id=candidate_id,
                displayed_lane=TrustLane.ESTABLISHED,
                recommended=True,
                warning_displayed=False,
            )
        ],
        corpus,
    )[0]
    assert not failures.passed
    assert len(failures.failures) >= 3

    license_failure = assert_trust_presentations(
        [
            TrustPresentation(
                candidate_id="exxperts-memory-governance",
                displayed_lane=corpus.by_candidate_id["exxperts-memory-governance"].artifact.trust_lane,
                recommended=False,
                warning_displayed=True,
                surfaced_issue_ids=tuple(
                    issue.id
                    for issue in corpus.by_candidate_id["exxperts-memory-governance"].issues
                    if issue.status == IssueStatus.UNRESOLVED
                ),
                labeled_unrestricted_open_source=True,
            )
        ],
        corpus,
    )[0]
    assert "noncommercial license was labeled unrestricted open source" in license_failure.failures


def test_recency_neutrality_is_order_sensitive() -> None:
    exact = compare_recency_neutral_rankings(["a", "b"], ["a", "b"])
    reordered = compare_recency_neutral_rankings(["a", "b"], ["b", "a"])
    assert exact.exact_order_match
    assert reordered.top_k_set_match
    assert not reordered.exact_order_match
