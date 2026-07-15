from __future__ import annotations

from pathlib import Path

from steering.domain.models import ArtifactRecord, SearchQuery
from steering.evaluation.corpus import EvaluationCorpus, EvaluationCorpusLoader
from steering.evaluation.embedding import DeterministicBlake2EmbeddingProvider
from steering.evaluation.runner import _counterfactual_records
from steering.evaluation.scoring import load_evaluation_cases, score_rankings
from steering.retrieval.hybrid import HybridRetriever


class CorpusRepository:
    def __init__(self, records: tuple[ArtifactRecord, ...]) -> None:
        self.records = list(records)

    def list_records(self) -> list[ArtifactRecord]:
        return self.records

    def project_history(self, _project_id: str) -> dict[str, list[object]]:
        return {}


async def _rankings(
    records: tuple[ArtifactRecord, ...],
) -> tuple[dict[str, tuple[str, ...]], dict[str, int], int, int]:
    retriever = HybridRetriever(
        repository=CorpusRepository(records),  # type: ignore[arg-type]
        embedding_provider=DeterministicBlake2EmbeddingProvider(),
    )
    cases = load_evaluation_cases(Path("evaluate/retrieval_prompts.json"))
    rankings: dict[str, tuple[str, ...]] = {}
    family_counts: dict[str, int] = {}
    cited = 0
    total = 0
    for case in cases:
        hits = await retriever.search(
            SearchQuery(query=case.prompt, limit=10, breadth=case.multi_item_expected)
        )
        rankings[case.id] = tuple(str(hit.artifact.metadata["candidate_id"]) for hit in hits)
        family_counts[case.id] = len({hit.artifact.strategy_family for hit in hits})
        cited += sum(bool(hit.citation_urls) for hit in hits)
        total += len(hits)
    return rankings, family_counts, cited, total


async def test_committed_scenarios_meet_retrieval_citation_breadth_and_recency_gates() -> None:
    corpus = EvaluationCorpusLoader(
        results_directory=Path("evaluate/results"),
        categories_path=Path("evaluate/result_categories.yaml"),
        strategy_families_path=Path("evaluate/strategy_families.yaml"),
    ).load()
    cases = load_evaluation_cases(Path("evaluate/retrieval_prompts.json"))
    rankings, family_counts, cited, total = await _rankings(corpus.records)
    score = score_rankings(cases, rankings, corpus)
    assert score.micro_recall_at_k >= 0.80
    assert cited == total
    assert all(family_counts[case.id] >= 4 for case in cases if case.multi_item_expected)

    counterfactual = EvaluationCorpus(records=_counterfactual_records(corpus))
    counterfactual_rankings, _, _, _ = await _rankings(counterfactual.records)
    assert rankings == counterfactual_rankings
