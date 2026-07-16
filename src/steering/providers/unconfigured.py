from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from pydantic import BaseModel

TModel = TypeVar("TModel", bound=BaseModel)


class GenerationProviderNotConfigured(RuntimeError):
    pass


class EmbeddingProviderNotConfigured(RuntimeError):
    pass


class UnconfiguredGenerationProvider:
    """Clear runtime boundary for installs that have not selected an LLM yet."""

    model_id = "steering/unconfigured-generation"

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[TModel],
    ) -> TModel:
        del system_prompt, user_prompt, response_model
        raise GenerationProviderNotConfigured(
            "knowledge construction needs a generation provider; run 'steering configure-provider'"
        )

    async def test_connection(self) -> None:
        raise GenerationProviderNotConfigured(
            "no generation provider is configured; run 'steering configure-provider'"
        )


class UnconfiguredEmbeddingProvider:
    """Fail clearly instead of manufacturing non-semantic production vectors."""

    provider_id = "unconfigured"
    model_id = "unconfigured"
    model_revision: str | None = None
    dimension = 768
    document_task_mode = "unconfigured"
    query_task_mode = "unconfigured"
    normalized = False

    @staticmethod
    def _error() -> EmbeddingProviderNotConfigured:
        return EmbeddingProviderNotConfigured(
            "semantic retrieval needs an embedding provider; run 'steering configure' "
            "or install local embeddings"
        )

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        del texts
        raise self._error()

    async def embed_query(self, text: str) -> list[float]:
        del text
        raise self._error()

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return await self.embed_documents(texts)

    async def test_connection(self) -> None:
        raise self._error()
