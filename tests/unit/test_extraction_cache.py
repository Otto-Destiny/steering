from __future__ import annotations

from pathlib import Path

import pytest

from steering.domain.credentials import CredentialDetectedError
from steering.domain.models import ArtifactType
from steering.extraction.cache import ExtractionCache
from steering.extraction.schemas import KnowledgeExtraction
from steering.extraction.service import default_cache


def payload() -> KnowledgeExtraction:
    return KnowledgeExtraction(
        artifact_type=ArtifactType.TECHNIQUE,
        title="Cached technique",
        summary="A deterministic cached extraction.",
    )


def test_extraction_cache_round_trip_miss_and_corruption(tmp_path: Path) -> None:
    cache = ExtractionCache(tmp_path / "cache")
    assert cache.get("missing") is None

    cache.put("valid", payload())
    assert cache.get("valid") == payload()
    assert not (tmp_path / "cache" / ".valid.tmp").exists()

    (tmp_path / "cache" / "corrupt.json").write_text("{not valid json", encoding="utf-8")
    assert cache.get("corrupt") is None


def test_default_cache_uses_dedicated_subdirectory(tmp_path: Path) -> None:
    cache = default_cache(tmp_path)
    assert cache.directory == tmp_path / "extraction-cache"


def test_cache_refuses_high_confidence_credentials_before_writing(tmp_path: Path) -> None:
    cache = ExtractionCache(tmp_path / "cache")
    secret = "sk-proj-" + "A" * 40
    unsafe = payload().model_copy(update={"summary": f"Leaked credential: {secret}"})

    with pytest.raises(CredentialDetectedError, match="refused before persistence"):
        cache.put("unsafe", unsafe)

    assert not (cache.directory / "unsafe.json").exists()
    assert secret not in str(CredentialDetectedError("safe error"))
