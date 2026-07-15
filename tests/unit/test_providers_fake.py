from __future__ import annotations

import math

import pytest
from pydantic import BaseModel

from steering.providers.fake import FakeGenerationProvider, HashEmbeddingProvider
from steering.providers.unconfigured import (
    GenerationProviderNotConfigured,
    UnconfiguredGenerationProvider,
)


class Answer(BaseModel):
    value: int


class CompatibleAnswer(BaseModel):
    value: int


async def test_hash_embedding_is_stable_normalized_and_handles_empty_text() -> None:
    with pytest.raises(ValueError, match="at least 8"):
        HashEmbeddingProvider(7)
    provider = HashEmbeddingProvider(32)
    first, repeated, empty = await provider.embed(["agent memory", "agent memory", ""])
    assert first == repeated
    assert len(first) == 32
    assert math.sqrt(sum(value * value for value in first)) == pytest.approx(1.0)
    assert empty == [0.0] * 32
    await provider.test_connection()


async def test_fake_generation_requires_factory_and_tracks_calls() -> None:
    provider = FakeGenerationProvider()
    with pytest.raises(RuntimeError, match="explicit response factory"):
        await provider.generate_structured(system_prompt="system", user_prompt="user", response_model=Answer)
    assert provider.calls == [("system", "user", "Answer")]
    await provider.test_connection()


@pytest.mark.parametrize(
    "factory",
    [
        lambda *_args: Answer(value=1),
        lambda *_args: CompatibleAnswer(value=1),
        lambda *_args: {"value": 1},
    ],
)
async def test_fake_generation_normalizes_supported_factory_results(factory) -> None:
    provider = FakeGenerationProvider(factory)
    result = await provider.generate_structured(
        system_prompt="system", user_prompt="user", response_model=Answer
    )
    assert result == Answer(value=1)


async def test_unconfigured_generation_provider_has_actionable_failures() -> None:
    provider = UnconfiguredGenerationProvider()
    with pytest.raises(GenerationProviderNotConfigured, match="configure-provider"):
        await provider.generate_structured(system_prompt="system", user_prompt="user", response_model=Answer)
    with pytest.raises(GenerationProviderNotConfigured, match="configure-provider"):
        await provider.test_connection()
