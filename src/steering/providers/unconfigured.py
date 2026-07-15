from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

TModel = TypeVar("TModel", bound=BaseModel)


class GenerationProviderNotConfigured(RuntimeError):
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
