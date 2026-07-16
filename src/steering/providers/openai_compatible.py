from __future__ import annotations

import asyncio
import base64
import json
import math
from collections.abc import Sequence
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, SecretStr

TModel = TypeVar("TModel", bound=BaseModel)
_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


class _ConnectionCheck(BaseModel):
    ok: bool


class ProviderConnectionError(RuntimeError):
    """Safe provider error that never contains credentials or response bodies."""


def _authorization_headers(api_key: SecretStr | None) -> dict[str, str]:
    if api_key is None or not api_key.get_secret_value():
        return {}
    return {"Authorization": f"Bearer {api_key.get_secret_value()}"}


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: SecretStr | None = None,
        timeout_seconds: float = 60.0,
        max_attempts: int = 3,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        normalized = base_url.rstrip("/") + "/"
        self._max_attempts = max_attempts
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=normalized,
            headers=_authorization_headers(api_key),
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=False,
            trust_env=False,
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def get_models(self) -> dict[str, Any]:
        return await self._request("GET", "models")

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        for attempt in range(self._max_attempts):
            try:
                response = await self._client.request(method, path, **kwargs)
                if response.status_code in _RETRYABLE_STATUS and attempt + 1 < self._max_attempts:
                    retry_after = response.headers.get("retry-after")
                    delay = min(float(retry_after), 10.0) if retry_after else 0.25 * (2**attempt)
                    await asyncio.sleep(delay)
                    continue
                response.raise_for_status()
                payload = response.json()
            except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
                raise ProviderConnectionError(
                    f"provider request failed ({type(exc).__name__}); verify endpoint, model, and credential"
                ) from None
            if not isinstance(payload, dict):
                raise ProviderConnectionError("provider returned an unexpected response shape")
            return payload
        raise ProviderConnectionError("provider request exhausted its retry budget")


class OpenAICompatibleGenerationProvider:
    def __init__(
        self,
        *,
        client: OpenAICompatibleClient,
        model_id: str,
        temperature: float = 0.0,
    ) -> None:
        self._client = client
        self.model_id = model_id
        self.temperature = temperature

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[TModel],
    ) -> TModel:
        schema = response_model.model_json_schema()
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        request: dict[str, Any] = {
            "model": self.model_id,
            "temperature": self.temperature,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__.lower(),
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        payload = await self._client._request("POST", "chat/completions", json=request)
        try:
            content = payload["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            return response_model.model_validate_json(str(content))
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderConnectionError(
                f"generation response failed schema validation ({type(exc).__name__})"
            ) from None

    async def test_connection(self) -> None:
        result = await self.generate_structured(
            system_prompt="Return the requested JSON only.",
            user_prompt='Set "ok" to true.',
            response_model=_ConnectionCheck,
        )
        if not result.ok:
            raise ProviderConnectionError("structured-output test returned an invalid result")

    async def understand_image(self, *, content: bytes, mime_type: str) -> str:
        if mime_type not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
            raise ValueError("unsupported image media type")
        if len(content) > 5 * 1024 * 1024:
            raise ValueError("image exceeds the 5 MiB multimodal extraction limit")
        encoded = base64.b64encode(content).decode("ascii")
        request = {
            "model": self.model_id,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Extract only technically useful AI-engineering information visible in "
                        "this image. Transcribe claims, labels, metrics, architecture details, and "
                        "paper identifiers exactly. Say 'NO_TECHNICAL_CONTENT' for decorative media."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Inspect this user-authorized uploaded image."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{encoded}", "detail": "high"},
                        },
                    ],
                },
            ],
        }
        payload = await self._client._request("POST", "chat/completions", json=request)
        try:
            value = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderConnectionError("image response had an invalid shape") from exc
        if isinstance(value, list):
            value = "\n".join(str(part.get("text", "")) for part in value if isinstance(part, dict))
        return str(value).strip()


class OpenAICompatibleEmbeddingProvider:
    provider_id = "openai-compatible"
    document_task_mode = "retrieval_document"
    query_task_mode = "retrieval_query"
    normalized = True

    def __init__(
        self,
        *,
        client: OpenAICompatibleClient,
        model_id: str,
        dimension: int,
        provider_id: str = "openai-compatible",
        batch_size: int = 64,
        max_concurrency: int = 2,
    ) -> None:
        if dimension <= 0 or batch_size <= 0 or max_concurrency <= 0:
            raise ValueError("dimension, batch_size, and max_concurrency must be positive")
        self._client = client
        self.provider_id = provider_id
        self.model_id = model_id
        self.model_revision: str | None = None
        self.dimension = dimension
        self._batch_size = batch_size
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        async with self._semaphore:
            payload = await self._client._request(
                "POST",
                "embeddings",
                json={
                    "model": self.model_id,
                    "input": list(texts),
                    "dimensions": self.dimension,
                    "encoding_format": "float",
                },
            )
        try:
            rows = sorted(payload["data"], key=lambda row: int(row["index"]))
            embeddings = [_normalized_embedding(row["embedding"], self.dimension) for row in rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderConnectionError(
                f"embedding response had an invalid shape ({type(exc).__name__})"
            ) from None
        if len(embeddings) != len(texts):
            raise ProviderConnectionError("embedding response count did not match request")
        return embeddings

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        batches = [
            texts[index : index + self._batch_size] for index in range(0, len(texts), self._batch_size)
        ]
        results = await asyncio.gather(*(self._embed_batch(batch) for batch in batches))
        return [embedding for batch in results for embedding in batch]

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed_batch([text]))[0]

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return await self.embed_documents(texts)

    async def test_connection(self) -> None:
        await self.embed_query("STEERING connection test")


def _normalized_embedding(values: Sequence[float], dimension: int) -> list[float]:
    vector = [float(value) for value in values]
    if len(vector) != dimension or any(not math.isfinite(value) for value in vector):
        raise ProviderConnectionError("embedding response dimension or values were invalid")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        raise ProviderConnectionError("embedding provider returned a zero vector")
    return [value / norm for value in vector]
