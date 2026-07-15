"""Stable offline embedding provider used by deterministic evaluation runs."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from hashlib import blake2b


class DeterministicBlake2EmbeddingProvider:
    """A keyless signed feature-hashing embedder with stable BLAKE2 buckets."""

    model_id = "evaluation/blake2-feature-hash-v1"

    def __init__(self, dimension: int = 256) -> None:
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self.dimension = dimension

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    async def test_connection(self) -> None:
        return None

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = re.findall(r"[a-z0-9][a-z0-9_+.-]*", text.lower())
        for token in tokens:
            digest = blake2b(
                token.encode("utf-8"),
                digest_size=8,
                person=b"steer-embed",
            ).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        return vector if norm == 0.0 else [value / norm for value in vector]
