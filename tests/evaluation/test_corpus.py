from pathlib import Path

from steering.domain.models import EvidenceCategory, IssueStatus, TrustLane
from steering.evaluation.corpus import EvaluationCorpusLoader, stable_digest


def _load():
    return EvaluationCorpusLoader(
        results_directory=Path("evaluate/results"),
        categories_path=Path("evaluate/result_categories.yaml"),
        strategy_families_path=Path("evaluate/strategy_families.yaml"),
    ).load()


def test_loader_normalizes_all_reviewed_results_deterministically() -> None:
    first = _load()
    second = _load()
    assert len(first.records) == 51
    assert [record.model_dump(mode="json") for record in first.records] == [
        record.model_dump(mode="json") for record in second.records
    ]
    assert len(first.canonical_groups) == 50
    assert stable_digest("same") == stable_digest("same")


def test_spans_are_exact_and_missing_evidence_is_conservative() -> None:
    corpus = _load()
    for record in corpus.records:
        snapshot = record.snapshots[0]
        for span in record.evidence_spans:
            assert snapshot.text[span.start : span.end] == span.quote

    automem = corpus.by_candidate_id["automem"]
    assert automem.artifact.trust_lane == TrustLane.EXPERIMENTAL
    assert automem.claims[0].category == EvidenceCategory.MODEL_INFERENCE
    assert automem.claims[0].confidence <= 0.35


def test_unresolved_issues_and_duplicate_canonical_source_are_preserved() -> None:
    corpus = _load()
    unresolved = [
        issue
        for record in corpus.records
        for issue in record.issues
        if issue.status == IssueStatus.UNRESOLVED
    ]
    assert unresolved
    duplicate_groups = [ids for ids in corpus.canonical_groups.values() if len(ids) > 1]
    assert duplicate_groups == [("behavioral-state-decay", "proactive-memory-agent")]


def test_gold_prompt_labels_are_not_seeded_into_search_chunks() -> None:
    corpus = _load()
    searchable = "\n".join(chunk.text for record in corpus.records for chunk in record.chunks)
    assert "minimum_relevant_hits" not in searchable
    assert "p01_staff_long_running_agent_memory" not in searchable
