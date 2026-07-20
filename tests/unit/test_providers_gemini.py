from __future__ import annotations

import json
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import BaseModel, SecretStr

from steering.providers.gemini import (
    GeminiClient,
    GeminiEmbeddingProvider,
    GeminiGenerationProvider,
)
from steering.providers.openai_compatible import ProviderConnectionError, ProviderTimeoutError


class Answer(BaseModel):
    value: int


ResponseItem = httpx.Response | Exception


class QueueTransport(httpx.AsyncBaseTransport):
    def __init__(self, responses: list[ResponseItem]) -> None:
        self.responses = responses
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def response(payload: object, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=payload)


def text_payload(text: str) -> dict[str, object]:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


async def test_gemini_structured_generation_uses_official_contract() -> None:
    transport = QueueTransport([response(text_payload('{"value":7}'))])
    async with httpx.AsyncClient(
        base_url="https://generativelanguage.googleapis.com/v1beta/", transport=transport
    ) as raw:
        provider = GeminiGenerationProvider(
            client=GeminiClient(
                base_url="https://ignored.test",
                api_key=SecretStr("secret-canary"),
                client=raw,
            ),
            model_id="gemini-3.5-flash",
        )
        result = await provider.generate_structured(
            system_prompt="system", user_prompt="user", response_model=Answer
        )
    assert result.value == 7
    request = transport.requests[0]
    assert request.url.path.endswith("/models/gemini-3.5-flash:generateContent")
    assert request.headers["x-goog-api-key"] == "secret-canary"
    body = json.loads(request.content)
    assert body["systemInstruction"]["parts"][0]["text"] == "system"
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert body["generationConfig"]["responseJsonSchema"]["type"] == "object"


async def test_gemini_embedding_formats_document_and_query_and_normalizes() -> None:
    transport = QueueTransport(
        [
            response({"embedding": {"values": [3, 4, 0]}}),
            response({"embedding": {"values": [0, 3, 4]}}),
        ]
    )
    async with httpx.AsyncClient(
        base_url="https://generativelanguage.googleapis.com/v1beta/", transport=transport
    ) as raw:
        provider = GeminiEmbeddingProvider(
            client=GeminiClient(base_url="https://ignored.test", api_key=SecretStr("secret"), client=raw),
            model_id="gemini-embedding-2",
            dimension=3,
            max_concurrency=1,
        )
        documents = await provider.embed_documents(["agent memory"])
        query = await provider.embed_query("memory options")
    assert documents == [pytest.approx([0.6, 0.8, 0.0])]
    assert query == pytest.approx([0.0, 0.6, 0.8])
    document_body = json.loads(transport.requests[0].content)
    query_body = json.loads(transport.requests[1].content)
    assert document_body["content"]["parts"][0]["text"] == "title: none | text: agent memory"
    assert query_body["content"]["parts"][0]["text"] == ("task: search result | query: memory options")
    assert document_body["outputDimensionality"] == 3


async def test_gemini_errors_are_redacted_and_invalid_models_are_rejected() -> None:
    transport = QueueTransport([httpx.Response(401, text="secret-canary")])
    async with httpx.AsyncClient(base_url="https://provider.test/", transport=transport) as raw:
        client = GeminiClient(
            base_url="https://ignored.test",
            api_key=SecretStr("secret-canary"),
            client=raw,
            max_attempts=1,
        )
        with pytest.raises(ProviderConnectionError) as captured:
            await client.request("models/model:embedContent", {})
    assert "secret-canary" not in str(captured.value)
    with pytest.raises(ValueError, match="unsupported characters"):
        GeminiGenerationProvider(client=client, model_id="../unsafe")


async def test_gemini_retries_timeout_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    sleep = AsyncMock()
    monkeypatch.setattr("steering.providers.gemini.asyncio.sleep", sleep)
    transport = QueueTransport(
        [httpx.ReadTimeout("slow response contains secret-canary"), response({"ok": True})]
    )
    async with httpx.AsyncClient(base_url="https://provider.test/", transport=transport) as raw:
        client = GeminiClient(
            base_url="https://ignored.test",
            api_key=SecretStr("secret-canary"),
            client=raw,
            max_attempts=2,
        )
        assert await client.request("models/model:generateContent", {}) == {"ok": True}

    assert len(transport.requests) == 2
    sleep.assert_awaited_once()


async def test_gemini_exhausted_timeouts_are_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("steering.providers.gemini.asyncio.sleep", AsyncMock())
    transport = QueueTransport([httpx.ReadTimeout("secret-canary")] * 3)
    async with httpx.AsyncClient(base_url="https://provider.test/", transport=transport) as raw:
        client = GeminiClient(
            base_url="https://ignored.test",
            api_key=SecretStr("secret-canary"),
            client=raw,
            max_attempts=3,
        )
        with pytest.raises(ProviderTimeoutError) as captured:
            await client.request("models/model:generateContent", {})

    assert len(transport.requests) == 3
    assert "secret-canary" not in str(captured.value)
