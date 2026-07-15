from __future__ import annotations

import base64
import json
from collections.abc import Sequence
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, SecretStr

TModel = TypeVar("TModel", bound=BaseModel)


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
        client: httpx.AsyncClient | None = None,
    ) -> None:
        normalized = base_url.rstrip("/") + "/"
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=normalized,
            headers=_authorization_headers(api_key),
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=False,
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def get_models(self) -> dict[str, Any]:
        return await self._request("GET", "models")

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = await self._client.request(method, path, **kwargs)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise ProviderConnectionError(
                f"provider request failed ({type(exc).__name__}); verify endpoint, model, and credential"
            ) from None
        if not isinstance(payload, dict):
            raise ProviderConnectionError("provider returned an unexpected response shape")
        return payload


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
        try:
            payload = await self._client._request("POST", "chat/completions", json=request)
        except ProviderConnectionError:
            request.pop("response_format")
            messages[0]["content"] += " Return one JSON object matching this schema: " + json.dumps(
                schema,
                separators=(",", ":"),
            )
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
        await self._client.get_models()

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
    def __init__(self, *, client: OpenAICompatibleClient, model_id: str, dimension: int) -> None:
        self._client = client
        self.model_id = model_id
        self.dimension = dimension

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        payload = await self._client._request(
            "POST",
            "embeddings",
            json={"model": self.model_id, "input": list(texts)},
        )
        try:
            rows = sorted(payload["data"], key=lambda row: int(row["index"]))
            embeddings = [[float(value) for value in row["embedding"]] for row in rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderConnectionError(
                f"embedding response had an invalid shape ({type(exc).__name__})"
            ) from None
        if len(embeddings) != len(texts) or any(len(row) != self.dimension for row in embeddings):
            raise ProviderConnectionError("embedding response count or dimension did not match configuration")
        return embeddings

    async def test_connection(self) -> None:
        rows = await self.embed(["STEERING connection test"])
        if not rows:
            raise ProviderConnectionError("embedding provider returned no vectors")
