from __future__ import annotations

import hashlib
import importlib.machinery
import sys
import types
from pathlib import Path
from typing import ClassVar

import pytest

import steering.providers.fastembed_local as fastembed_local
from steering.providers.fastembed_local import (
    FastEmbedEmbeddingProvider,
    LocalEmbeddingUnavailable,
    install_local_model,
    local_model_status,
)

TEST_MODEL_BYTES = b"reviewed-onnx-model"


def _model_path(cache: Path) -> Path:
    return (
        cache
        / "models--nomic-ai--nomic-embed-text-v1.5"
        / "snapshots"
        / fastembed_local.LOCAL_MODEL_REVISION
        / fastembed_local.LOCAL_MODEL_FILE
    )


def _write_model(cache: Path) -> None:
    path = _model_path(cache)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(TEST_MODEL_BYTES)


class TextEmbedding:
    calls: ClassVar[list[dict[str, object]]] = []

    def __init__(self, **kwargs: object) -> None:
        self.calls.append(kwargs)
        cache = Path(str(kwargs["cache_dir"]))
        cache.mkdir(parents=True, exist_ok=True)
        _write_model(cache)

    def passage_embed(self, texts: list[str], **_kwargs: object) -> list[list[float]]:
        return [[1.0] + [0.0] * 767 for _ in texts]

    def query_embed(self, texts: list[str], **_kwargs: object) -> list[list[float]]:
        return [[0.0, 1.0] + [0.0] * 766 for _ in texts]


@pytest.fixture
def fastembed_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fastembed_local, "LOCAL_MODEL_FILE_SIZE", len(TEST_MODEL_BYTES))
    monkeypatch.setattr(
        fastembed_local,
        "LOCAL_MODEL_SHA256",
        hashlib.sha256(TEST_MODEL_BYTES).hexdigest(),
    )
    module = types.ModuleType("fastembed")
    module.__spec__ = importlib.machinery.ModuleSpec("fastembed", loader=None)
    module.TextEmbedding = TextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", module)
    TextEmbedding.calls.clear()


def test_install_is_explicit_and_records_content_fingerprint(tmp_path: Path, fastembed_module: None) -> None:
    status = install_local_model(tmp_path)
    assert status["model_installed"] is True
    assert status["model_revision"] == (
        f"hf:{fastembed_local.LOCAL_MODEL_REVISION}:sha256:{hashlib.sha256(TEST_MODEL_BYTES).hexdigest()}"
    )
    assert TextEmbedding.calls[0]["local_files_only"] is False


async def test_runtime_loads_only_cached_model_and_uses_retrieval_paths(
    tmp_path: Path, fastembed_module: None
) -> None:
    _write_model(tmp_path)
    provider = FastEmbedEmbeddingProvider(cache_dir=tmp_path)
    documents = await provider.embed_documents(["document"])
    query = await provider.embed_query("query")
    assert len(documents[0]) == len(query) == 768
    assert documents[0][0] == 1.0
    assert query[1] == 1.0
    assert TextEmbedding.calls[0]["local_files_only"] is True
    assert provider.document_task_mode == "retrieval_document"
    assert provider.query_task_mode == "retrieval_query"


async def test_runtime_never_downloads_an_uninstalled_model(tmp_path: Path) -> None:
    provider = FastEmbedEmbeddingProvider(cache_dir=tmp_path)
    with pytest.raises(LocalEmbeddingUnavailable, match="not installed"):
        await provider.embed_query("query")
    assert local_model_status(tmp_path)["model_installed"] is False
