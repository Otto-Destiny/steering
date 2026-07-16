"""Offline corpus seeding and retrieval evaluation orchestration."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, cast

from steering.domain.models import ArtifactRecord, SearchQuery
from steering.domain.protocols import ArtifactRepository, EmbeddingProvider, KnowledgeRetriever
from steering.evaluation.corpus import EvaluationCorpus, EvaluationCorpusLoader
from steering.evaluation.scoring import (
    EvaluationCase,
    FactCitation,
    citation_coverage,
    compare_recency_neutral_rankings,
    load_evaluation_cases,
    score_rankings,
)
from steering.extraction.service import content_hash
from steering.intelligence.service import SteeringEngine
from steering.retrieval.hybrid import HybridRetriever


class EvaluationFactory(Protocol):
    def __call__(
        self, *, embedding_provider: EmbeddingProvider
    ) -> tuple[ArtifactRepository, KnowledgeRetriever]: ...


@dataclass(frozen=True)
class _RetrievalRun:
    rankings: dict[str, tuple[str, ...]]
    family_counts: dict[str, int]
    cited_results: int
    total_results: int


@dataclass(frozen=True)
class _ArchitectureCitationRun:
    scenario_count: int
    facts: tuple[FactCitation, ...]


async def _embed_records(
    records: Sequence[ArtifactRecord], embedding_provider: EmbeddingProvider
) -> tuple[ArtifactRecord, ...]:
    """Embed the reviewed corpus with the provider being evaluated."""

    chunks = [
        chunk
        for record in records
        for chunk in record.chunks
        if not (
            chunk.embedding
            and len(chunk.embedding) == embedding_provider.dimension
            and chunk.embedding_provider == embedding_provider.provider_id
            and chunk.embedding_model == embedding_provider.model_id
            and chunk.embedding_revision == embedding_provider.model_revision
            and chunk.embedding_dimension == embedding_provider.dimension
            and chunk.embedding_task_mode == embedding_provider.document_task_mode
            and chunk.embedding_normalized == embedding_provider.normalized
            and chunk.source_content_hash == content_hash(chunk.text)
        )
    ]
    if not chunks:
        return tuple(records)
    vectors = await embedding_provider.embed_documents([chunk.text for chunk in chunks])
    if len(vectors) != len(chunks):
        raise ValueError("embedding provider returned an incomplete evaluation batch")
    embedded = {
        chunk.id: chunk.model_copy(
            update={
                "embedding": vector,
                "embedding_provider": embedding_provider.provider_id,
                "embedding_model": embedding_provider.model_id,
                "embedding_revision": embedding_provider.model_revision,
                "embedding_dimension": embedding_provider.dimension,
                "embedding_task_mode": embedding_provider.document_task_mode,
                "embedding_normalized": embedding_provider.normalized,
                "source_content_hash": content_hash(chunk.text),
            }
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    }
    return tuple(
        record.model_copy(update={"chunks": [embedded.get(chunk.id, chunk) for chunk in record.chunks]})
        for record in records
    )


def _load_factory(spec: str) -> EvaluationFactory:
    module_name, separator, attribute = spec.partition(":")
    if not separator:
        raise ValueError("factory must use module:function syntax")
    candidate = getattr(importlib.import_module(module_name), attribute)
    if not callable(candidate):
        raise TypeError("factory target is not callable")
    return cast(EvaluationFactory, candidate)


def _counterfactual_records(corpus: EvaluationCorpus) -> tuple[ArtifactRecord, ...]:
    dates = sorted(
        artifact.artifact.published_at
        for artifact in corpus.records
        if artifact.artifact.published_at is not None
    )
    reversed_dates = iter(reversed(dates))
    records: list[ArtifactRecord] = []
    for record in corpus.records:
        published_at: datetime | None = record.artifact.published_at
        if published_at is not None:
            published_at = next(reversed_dates)
        records.append(
            record.model_copy(
                update={"artifact": record.artifact.model_copy(update={"published_at": published_at})},
                deep=True,
            )
        )
    return tuple(records)


async def _retrieve(
    records: Sequence[ArtifactRecord],
    cases: Sequence[EvaluationCase],
    factory: EvaluationFactory,
    embedding_provider: EmbeddingProvider,
) -> _RetrievalRun:
    repository, retriever = factory(embedding_provider=embedding_provider)
    for record in await _embed_records(records, embedding_provider):
        repository.upsert_record(record)
    rankings: dict[str, tuple[str, ...]] = {}
    family_counts: dict[str, int] = {}
    cited_results = 0
    total_results = 0
    for case in cases:
        hits = await retriever.search(
            SearchQuery(query=case.prompt, limit=10, breadth=case.multi_item_expected)
        )
        rankings[case.id] = tuple(
            str(hit.artifact.metadata["candidate_id"])
            for hit in hits
            if "candidate_id" in hit.artifact.metadata
        )
        family_counts[case.id] = len({hit.artifact.strategy_family for hit in hits})
        cited_results += sum(bool(hit.citation_urls) for hit in hits)
        total_results += len(hits)
    return _RetrievalRun(
        rankings=rankings,
        family_counts=family_counts,
        cited_results=cited_results,
        total_results=total_results,
    )


async def _architecture_citations(
    records: Sequence[ArtifactRecord],
    cases: Sequence[EvaluationCase],
    factory: EvaluationFactory,
    embedding_provider: EmbeddingProvider,
    *,
    scenario_limit: int = 20,
) -> _ArchitectureCitationRun:
    repository, candidate = factory(embedding_provider=embedding_provider)
    for record in await _embed_records(records, embedding_provider):
        repository.upsert_record(record)
    retriever = cast(HybridRetriever, candidate)
    engine = SteeringEngine(repository=repository, retriever=retriever)
    facts: list[FactCitation] = []
    selected = list(cases[:scenario_limit])
    for case in selected:
        review = await engine.review_architecture(case.prompt, limit=10)
        for card in review.idea_cards:
            matched_record = repository.get_record(card.artifact_id)
            if matched_record is None or not card.evidence:
                facts.append(FactCitation(text=card.what_it_is))
                continue
            spans = {span.id: span for span in matched_record.evidence_spans}
            snapshots = {snapshot.id: snapshot for snapshot in matched_record.snapshots}
            for item in card.evidence:
                span = spans.get(item.evidence_span_id)
                snapshot = snapshots.get(item.snapshot_id)
                valid = bool(
                    span
                    and snapshot
                    and span.snapshot_id == item.snapshot_id
                    and span.quote == item.exact_quote
                    and snapshot.source_url == item.source_url
                    and item.source_url in review.citations
                )
                facts.append(
                    FactCitation(
                        text=item.claim,
                        source_url=item.source_url if valid else None,
                        snapshot_id=item.snapshot_id if valid else None,
                        evidence_span_id=item.evidence_span_id if valid else None,
                    )
                )
    return _ArchitectureCitationRun(scenario_count=len(selected), facts=tuple(facts))


async def run_evaluation(
    *,
    corpus: EvaluationCorpus,
    cases: Sequence[EvaluationCase],
    factory: EvaluationFactory,
    embedding_provider: EmbeddingProvider,
) -> dict[str, Any]:
    embedded_records = await _embed_records(corpus.records, embedding_provider)
    baseline = await _retrieve(embedded_records, cases, factory, embedding_provider)
    counterfactual = await _retrieve(
        _counterfactual_records(EvaluationCorpus(records=embedded_records)),
        cases,
        factory,
        embedding_provider,
    )
    architecture = await _architecture_citations(embedded_records, cases, factory, embedding_provider)
    retrieval = score_rankings(cases, baseline.rankings, corpus)
    recency = {
        case.id: asdict(
            compare_recency_neutral_rankings(
                baseline.rankings.get(case.id, ()), counterfactual.rankings.get(case.id, ())
            )
        )
        for case in cases
    }
    breadth_cases = [case for case in cases if case.multi_item_expected]
    breadth_passes = {case.id: baseline.family_counts.get(case.id, 0) >= 4 for case in breadth_cases}
    return {
        "schema_version": 3,
        "corpus_size": len(corpus.records),
        "retrieval": asdict(retrieval),
        "retrieval_result_citation_coverage": (
            baseline.cited_results / baseline.total_results if baseline.total_results else 1.0
        ),
        "retrieval_result_citation_denominator": baseline.total_results,
        "architecture_scenario_count": architecture.scenario_count,
        "architecture_claim_evidence_coverage": citation_coverage(architecture.facts),
        "architecture_claim_evidence_denominator": len(architecture.facts),
        "breadth_family_counts": baseline.family_counts,
        "breadth_at_least_four_pass_rate": (
            sum(breadth_passes.values()) / len(breadth_passes) if breadth_passes else 1.0
        ),
        "recency_neutrality": recency,
        "all_recency_exact": all(item["exact_order_match"] for item in recency.values()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory", required=True, help="module:function offline component factory")
    parser.add_argument("--results", type=Path, default=Path("evaluate/results"))
    parser.add_argument("--categories", type=Path, default=Path("evaluate/result_categories.yaml"))
    parser.add_argument("--families", type=Path, default=Path("evaluate/strategy_families.yaml"))
    parser.add_argument("--prompts", type=Path, default=Path("evaluate/retrieval_prompts.json"))
    parser.add_argument("--output", type=Path, default=Path(".work/offline_evaluation.json"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    corpus = EvaluationCorpusLoader(
        results_directory=args.results,
        categories_path=args.categories,
        strategy_families_path=args.families,
    ).load()

    async def execute() -> dict[str, Any]:
        from steering.runtime import create_runtime

        runtime = create_runtime()
        try:
            return await run_evaluation(
                corpus=corpus,
                cases=load_evaluation_cases(args.prompts),
                factory=_load_factory(args.factory),
                embedding_provider=runtime.embedding,
            )
        finally:
            await runtime.aclose()

    report = asyncio.run(execute())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the packaged command path
    raise SystemExit(main())
