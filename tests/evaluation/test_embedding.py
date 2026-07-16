import math

import pytest
from tests.support.providers import DeterministicBlake2EmbeddingProvider


@pytest.mark.asyncio
async def test_embedding_is_stable_normalized_and_keyless() -> None:
    provider = DeterministicBlake2EmbeddingProvider(dimension=32)
    first = await provider.embed(["agent memory rollback"])
    second = await provider.embed(["agent memory rollback"])
    different = await provider.embed(["visual document retrieval"])
    assert first == second
    assert first != different
    assert len(first[0]) == 32
    assert math.sqrt(sum(value * value for value in first[0])) == pytest.approx(1.0)
    await provider.test_connection()
