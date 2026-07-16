from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Sequence
from hashlib import blake2b
from itertools import pairwise
from typing import Any, TypeVar

from pydantic import BaseModel

TModel = TypeVar("TModel", bound=BaseModel)
TOKEN = re.compile(r"[a-z0-9][a-z0-9_+.-]*")


class DeterministicBlake2EmbeddingProvider:
    provider_id = "test"
    model_id = "test/blake2-feature-hash-v1"
    model_revision = "v1"
    document_task_mode = "test"
    query_task_mode = "test"
    normalized = True

    def __init__(self, dimension: int = 256) -> None:
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self.dimension = dimension

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return await self.embed_documents(texts)

    async def test_connection(self) -> None:
        return None

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in TOKEN.findall(text.lower()):
            digest = blake2b(token.encode(), digest_size=8, person=b"steer-embed").digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            vector[bucket] += 1.0 if digest[4] & 1 else -1.0
        return _normalize(vector)


class HashEmbeddingProvider(DeterministicBlake2EmbeddingProvider):
    model_id = "test/hash-embedding-v1"

    def _embed_one(self, text: str) -> list[float]:
        tokens = TOKEN.findall(text.lower())
        features = tokens + [f"{left}::{right}" for left, right in pairwise(tokens)]
        vector = [0.0] * self.dimension
        for feature in features:
            digest = hashlib.blake2b(feature.encode(), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "little") % self.dimension
            vector[index] += 1.0 if digest[8] & 1 else -1.0
        return _normalize(vector)


ResponseFactory = Callable[[str, str, type[BaseModel]], BaseModel | dict[str, Any]]


class FakeGenerationProvider:
    model_id = "test/fake-generation-v1"

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


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector
