from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Sequence
from itertools import pairwise
from typing import Any, TypeVar

from pydantic import BaseModel

TModel = TypeVar("TModel", bound=BaseModel)
TOKEN = re.compile(r"[a-z0-9][a-z0-9_+.-]*")


class HashEmbeddingProvider:
    """Stable local embeddings for tests and no-key installations.

    This is intentionally lexical, not presented as a semantic model. It lets the
    complete application run deterministically until a real embedding provider is
    configured.
    """

    model_id = "steering/hash-embedding-v1"

    def __init__(self, dimension: int = 256) -> None:
        if dimension < 8:
            raise ValueError("embedding dimension must be at least 8")
        self.dimension = dimension

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    async def test_connection(self) -> None:
        return None

    def _embed_one(self, text: str) -> list[float]:
        tokens = TOKEN.findall(text.lower())
        features = tokens + [f"{left}::{right}" for left, right in pairwise(tokens)]
        vector = [0.0] * self.dimension
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "little") % self.dimension
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


ResponseFactory = Callable[[str, str, type[BaseModel]], BaseModel | dict[str, Any]]


class FakeGenerationProvider:
    """Injectable deterministic provider used by tests and local fixtures."""

    model_id = "steering/fake-generation-v1"

    def __init__(self, response_factory: ResponseFactory | None = None) -> None:
        self._response_factory = response_factory
        self.calls: list[tuple[str, str, str]] = []

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[TModel],
    ) -> TModel:
        self.calls.append((system_prompt, user_prompt, response_model.__name__))
        if self._response_factory is None:
            raise RuntimeError("fake generation provider needs an explicit response factory")
        response = self._response_factory(system_prompt, user_prompt, response_model)
        if isinstance(response, response_model):
            return response
        if isinstance(response, BaseModel):
            return response_model.model_validate(response.model_dump(mode="json"))
        return response_model.model_validate(response)

    async def test_connection(self) -> None:
        return None
