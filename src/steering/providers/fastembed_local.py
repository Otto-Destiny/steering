from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from platformdirs import user_data_path

from steering.providers.openai_compatible import ProviderConnectionError

LOCAL_PROVIDER_ID = "local-fastembed"
LOCAL_MODEL_ID = "nomic-ai/nomic-embed-text-v1.5-Q"
LOCAL_MODEL_DIMENSION = 768
LOCAL_MODEL_LICENSE = "Apache-2.0"
LOCAL_MODEL_SIZE_MB = 130
LOCAL_MODEL_REVISION = "e9b6763023c676ca8431644204f50c2b100d9aab"
LOCAL_MODEL_FILE = "onnx/model_quantized.onnx"
LOCAL_MODEL_FILE_SIZE = 137_296_292
LOCAL_MODEL_SHA256 = "b4342336debaea79de872370664b0aaeb67dea4605513d00ee236ea871a81f27"


class LocalEmbeddingUnavailable(RuntimeError):
    """Actionable failure for an uninstalled optional local embedding runtime."""


def local_model_cache() -> Path:
    return user_data_path("steering", appauthor=False) / "models" / "fastembed"


def _fastembed_class() -> type[Any]:
    try:
        from fastembed import TextEmbedding
    except ImportError:
        raise LocalEmbeddingUnavailable(
            "local embeddings require the optional dependency; install with "
            "'uv sync --extra local-embeddings'"
        ) from None
    return TextEmbedding


def _cache_fingerprint(cache_dir: Path) -> str | None:
    model_file = (
        cache_dir
        / "models--nomic-ai--nomic-embed-text-v1.5"
        / "snapshots"
        / LOCAL_MODEL_REVISION
        / LOCAL_MODEL_FILE
    )
    if not model_file.is_file() or model_file.stat().st_size != LOCAL_MODEL_FILE_SIZE:
        return None
    digest = hashlib.sha256()
    with model_file.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != LOCAL_MODEL_SHA256:
        return None
    return f"hf:{LOCAL_MODEL_REVISION}:sha256:{LOCAL_MODEL_SHA256}"


def local_model_status(cache_dir: Path | None = None) -> dict[str, object]:
    root = cache_dir or local_model_cache()
    fingerprint = _cache_fingerprint(root) if root.is_dir() else None
    return {
        "provider_id": LOCAL_PROVIDER_ID,
        "model_id": LOCAL_MODEL_ID,
        "dimension": LOCAL_MODEL_DIMENSION,
        "license": LOCAL_MODEL_LICENSE,
        "download_size_mb": LOCAL_MODEL_SIZE_MB,
        "cache_path": str(root),
        "dependency_installed": importlib.util.find_spec("fastembed") is not None,
        "model_installed": fingerprint is not None,
        "model_revision": fingerprint,
    }


def install_local_model(cache_dir: Path | None = None) -> dict[str, object]:
    root = cache_dir or local_model_cache()
    root.mkdir(parents=True, exist_ok=True)
    embedding_class = _fastembed_class()
    model = embedding_class(
        model_name=LOCAL_MODEL_ID,
        cache_dir=str(root),
        local_files_only=False,
    )
    # Force both retrieval paths to load and validate the downloaded ONNX model.
    query = list(model.query_embed(["STEERING model installation check"]))
    documents = list(model.passage_embed(["STEERING model installation check"]))
    if not query or not documents:
        raise LocalEmbeddingUnavailable("the local embedding model produced no validation vector")
    status = local_model_status(root)
    if not status["model_installed"]:
        raise LocalEmbeddingUnavailable("the local embedding model download could not be verified")
    return status


def _normalized_rows(rows: Iterable[Any], dimension: int) -> list[list[float]]:
    normalized: list[list[float]] = []
    for row in rows:
        vector = [float(value) for value in row]
        if len(vector) != dimension or any(not math.isfinite(value) for value in vector):
            raise ProviderConnectionError("local embedding dimension or values were invalid")
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            raise ProviderConnectionError("local embedding model returned a zero vector")
        normalized.append([value / norm for value in vector])
    return normalized


class FastEmbedEmbeddingProvider:
    provider_id = LOCAL_PROVIDER_ID
    model_id = LOCAL_MODEL_ID
    dimension = LOCAL_MODEL_DIMENSION
    document_task_mode = "retrieval_document"
    query_task_mode = "retrieval_query"
    normalized = True

    def __init__(self, *, cache_dir: Path | None = None, batch_size: int = 64) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.cache_dir = cache_dir or local_model_cache()
        self.batch_size = batch_size
        self.model_revision = _cache_fingerprint(self.cache_dir)
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            if self.model_revision is None:
                raise LocalEmbeddingUnavailable(
                    "the local embedding model is not installed; run "
                    "'steering local-embeddings install --accept-download'"
                )
            embedding_class = _fastembed_class()
            self._model = embedding_class(
                model_name=self.model_id,
                cache_dir=str(self.cache_dir),
                local_files_only=True,
            )
        return self._model

    def _documents_sync(self, texts: Sequence[str]) -> list[list[float]]:
        rows = self._load().passage_embed(list(texts), batch_size=self.batch_size)
        return _normalized_rows(rows, self.dimension)

    def _query_sync(self, text: str) -> list[float]:
        rows = self._load().query_embed([text])
        vectors = _normalized_rows(rows, self.dimension)
        if not vectors:
            raise ProviderConnectionError("local embedding model returned no query vector")
        return vectors[0]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._documents_sync, texts)

    async def embed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._query_sync, text)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return await self.embed_documents(texts)

    async def test_connection(self) -> None:
        await self.embed_query("STEERING connection test")
