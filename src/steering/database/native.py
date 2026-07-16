from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

EMBEDDING_DIMENSION = 768
FTS_STATE_NAME = "chunks_text_fts"
VECTOR_STATE_NAME = "chunks_embedding_hnsw"
INDEX_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class NativeCandidate:
    artifact_id: str
    chunk_id: str | None
    score: float


def normalize_search_term(value: str) -> str:
    return " ".join(value.strip().lower().split())


def artifact_search_terms(
    artifact: dict[str, Any],
    *,
    entity_names: tuple[str, ...] = (),
    concept_names: tuple[str, ...] = (),
) -> dict[str, float]:
    weighted: list[tuple[str, float]] = [
        (str(artifact.get("canonical_url") or ""), 4.0),
        (str(artifact.get("title") or ""), 3.0),
        (str(artifact.get("short_name") or ""), 3.5),
        (str(artifact.get("strategy_family") or ""), 1.5),
    ]
    weighted.extend((str(value), 3.5) for value in artifact.get("aliases", []))
    weighted.extend((value, 2.0) for value in entity_names)
    weighted.extend((value, 2.0) for value in concept_names)
    terms: dict[str, float] = {}
    for value, weight in weighted:
        normalized = normalize_search_term(value)
        if normalized:
            terms[normalized] = max(weight, terms.get(normalized, 0.0))
    return terms
