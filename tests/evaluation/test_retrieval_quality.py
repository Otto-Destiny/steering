from __future__ import annotations

from pathlib import Path

from tests.support.providers import DeterministicBlake2EmbeddingProvider

from steering.domain.models import SearchQuery
from steering.domain.protocols import KnowledgeRetriever
from steering.evaluation.corpus import EvaluationCorpus, EvaluationCorpusLoader
from steering.evaluation.runner import _counterfactual_records, _embed_records
from steering.evaluation.scoring import load_evaluation_cases, score_rankings
from steering.runtime import evaluation_factory


async def _rankings(
    retriever: KnowledgeRetriever,
) -> tuple[dict[str, tuple[str, ...]], dict[str, int], int, int]:
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
    provider = DeterministicBlake2EmbeddingProvider(dimension=768)
    repository, retriever = evaluation_factory(embedding_provider=provider)
    embedded_records = await _embed_records(corpus.records, provider)
    for record in embedded_records:
        repository.upsert_record(record)
    rankings, family_counts, cited, total = await _rankings(retriever)
    score = score_rankings(cases, rankings, corpus)
    assert score.micro_recall_at_k >= 0.80
    assert cited == total
    assert all(family_counts[case.id] >= 4 for case in cases if case.multi_item_expected)

    counterfactual = EvaluationCorpus(
        records=_counterfactual_records(EvaluationCorpus(records=embedded_records))
    )
    for record in counterfactual.records:
        repository.upsert_record(record)
    counterfactual_rankings, _, _, _ = await _rankings(retriever)
    assert rankings == counterfactual_rankings
