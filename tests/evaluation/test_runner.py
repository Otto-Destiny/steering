from pathlib import Path

import pytest
from tests.support.providers import DeterministicBlake2EmbeddingProvider

from steering.evaluation.corpus import EvaluationCorpusLoader
from steering.evaluation.runner import _architecture_citations
from steering.evaluation.scoring import citation_coverage, load_evaluation_cases
from steering.runtime import evaluation_factory


@pytest.mark.asyncio
async def test_architecture_scenarios_retain_exact_claim_evidence() -> None:
    corpus = EvaluationCorpusLoader(
        results_directory=Path("evaluate/results"),
        categories_path=Path("evaluate/result_categories.yaml"),
        strategy_families_path=Path("evaluate/strategy_families.yaml"),
    ).load()
    cases = load_evaluation_cases(Path("evaluate/retrieval_prompts.json"))[:2]

    result = await _architecture_citations(
        corpus.records,
        cases,
        evaluation_factory,
        DeterministicBlake2EmbeddingProvider(dimension=768),
        scenario_limit=2,
    )

    assert result.scenario_count == 2
    assert len(result.facts) >= 20
    assert citation_coverage(result.facts) == 1.0
