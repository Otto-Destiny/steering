from __future__ import annotations

import asyncio
import base64
import json
import math
import re
from collections.abc import Sequence
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, SecretStr

from steering.providers.openai_compatible import ProviderConnectionError, ProviderTimeoutError

TModel = TypeVar("TModel", bound=BaseModel)
_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


class _ConnectionCheck(BaseModel):
    ok: bool


def _model_path(model_id: str, operation: str) -> str:
    if not _MODEL_ID.fullmatch(model_id):
        raise ValueError("Gemini model ID contains unsupported characters")
    return f"models/{model_id}:{operation}"


def _normalized(values: Sequence[float], dimension: int) -> list[float]:
    vector = [float(value) for value in values]
    if len(vector) != dimension or any(not math.isfinite(value) for value in vector):
        raise ProviderConnectionError("embedding response dimension or values were invalid")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        raise ProviderConnectionError("embedding provider returned a zero vector")
    return [value / norm for value in vector]


class GeminiClient:
    """Small official Gemini REST client with bounded retry behaviour."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: SecretStr,
        timeout_seconds: float = 90.0,
        max_attempts: int = 3,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key.get_secret_value():
            raise ValueError("Gemini requires an API key")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self._api_key = api_key
        self._max_attempts = max_attempts
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            timeout=httpx.Timeout(
                timeout_seconds,
                connect=min(timeout_seconds, 10.0),
                write=min(timeout_seconds, 30.0),
                pool=min(timeout_seconds, 10.0),
            ),
            follow_redirects=False,
            trust_env=False,
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def request(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(self._max_attempts):
            try:
                response = await self._client.post(
                    path,
                    json=payload,
                    headers={"x-goog-api-key": self._api_key.get_secret_value()},
                )
                if response.status_code in _RETRYABLE_STATUS and attempt + 1 < self._max_attempts:
                    retry_after = response.headers.get("retry-after")
                    delay = min(float(retry_after), 10.0) if retry_after else 0.25 * (2**attempt)
                    await asyncio.sleep(delay)
                    continue
                response.raise_for_status()
                result = response.json()
            except httpx.TransportError as exc:
                if attempt + 1 < self._max_attempts:
                    await asyncio.sleep(0.25 * (2**attempt))
                    continue
                if isinstance(exc, httpx.TimeoutException):
                    raise ProviderTimeoutError(
                        f"Gemini request timed out after {self._max_attempts} attempts"
                    ) from None
                raise ProviderConnectionError(
                    f"Gemini connection failed after {self._max_attempts} attempts"
                ) from None
            except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
                raise ProviderConnectionError(
                    f"Gemini request failed ({type(exc).__name__}); verify model and credential"
                ) from None
            if not isinstance(result, dict):
                raise ProviderConnectionError("Gemini returned an unexpected response shape")
            return result
        raise ProviderConnectionError("Gemini request exhausted its retry budget")


def _response_text(payload: dict[str, Any], operation: str) -> str:
    try:
        parts = payload["candidates"][0]["content"]["parts"]
        text = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict)).strip()
    except (KeyError, IndexError, TypeError):
        text = ""
    if not text:
        raise ProviderConnectionError(f"Gemini {operation} response contained no text")
    return text


class GeminiGenerationProvider:
    def __init__(self, *, client: GeminiClient, model_id: str, temperature: float = 0.0) -> None:
        _model_path(model_id, "generateContent")
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
        payload = await self._client.request(
            _model_path(self.model_id, "generateContent"),
            {
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": {
                    "temperature": self.temperature,
                    "responseMimeType": "application/json",
                    "responseJsonSchema": response_model.model_json_schema(),
                },
            },
        )
        try:
            return response_model.model_validate_json(_response_text(payload, "generation"))
        except ValueError as exc:
            raise ProviderConnectionError(
                f"Gemini generation failed schema validation ({type(exc).__name__})"
            ) from None

    async def test_connection(self) -> None:
        result = await self.generate_structured(
            system_prompt="Return the requested JSON only.",
            user_prompt='Set "ok" to true.',
            response_model=_ConnectionCheck,
        )
        if not result.ok:
            raise ProviderConnectionError("Gemini structured-output test returned an invalid result")

    async def understand_image(self, *, content: bytes, mime_type: str) -> str:
        if mime_type not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
            raise ValueError("unsupported image media type")
        if len(content) > 5 * 1024 * 1024:
            raise ValueError("image exceeds the 5 MiB multimodal extraction limit")
        payload = await self._client.request(
            _model_path(self.model_id, "generateContent"),
            {
                "systemInstruction": {
                    "parts": [
                        {
                            "text": (
                                "Extract only technically useful AI-engineering information. "
                                "Transcribe claims, labels, metrics, architecture details, and "
                                "paper identifiers exactly. Say NO_TECHNICAL_CONTENT for decorative media."
                            )
                        }
                    ]
                },
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": "Inspect this user-authorized uploaded image."},
                            {
                                "inlineData": {
                                    "mimeType": mime_type,
                                    "data": base64.b64encode(content).decode("ascii"),
                                }
                            },
                        ],
                    }
                ],
            },
        )
        return _response_text(payload, "image")


class GeminiEmbeddingProvider:
    """Gemini Embedding 2 using Google's asymmetric retrieval formatting."""

    provider_id = "gemini"
    model_revision: str | None = None
    document_task_mode = "retrieval_document"
    query_task_mode = "retrieval_query"
    normalized = True

    def __init__(
        self,
        *,
        client: GeminiClient,
        model_id: str,
        dimension: int = 768,
        max_concurrency: int = 2,
    ) -> None:
        _model_path(model_id, "embedContent")
        if dimension <= 0 or max_concurrency <= 0:
            raise ValueError("dimension and max_concurrency must be positive")
        self._client = client
        self.model_id = model_id
        self.dimension = dimension
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def _embed_one(self, text: str) -> list[float]:
        async with self._semaphore:
            payload = await self._client.request(
                _model_path(self.model_id, "embedContent"),
                {
                    "model": f"models/{self.model_id}",
                    "content": {"parts": [{"text": text}]},
                    "outputDimensionality": self.dimension,
                },
            )
        try:
            values = payload["embedding"]["values"]
        except (KeyError, TypeError):
            raise ProviderConnectionError("Gemini embedding response had an invalid shape") from None
        if not isinstance(values, list):
            raise ProviderConnectionError("Gemini embedding response had an invalid shape")
        return _normalized(values, self.dimension)

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        # Gemini Embedding 2 requires prompt formatting instead of the legacy taskType field.
        return await asyncio.gather(*(self._embed_one(f"title: none | text: {text}") for text in texts))

    async def embed_query(self, text: str) -> list[float]:
        return await self._embed_one(f"task: search result | query: {text}")

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return await self.embed_documents(texts)

    async def test_connection(self) -> None:
        await self.embed_query("STEERING connection test")
