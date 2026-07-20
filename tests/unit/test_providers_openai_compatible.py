from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import BaseModel, SecretStr

from steering.providers.openai_compatible import (
    OpenAICompatibleClient,
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleGenerationProvider,
    ProviderConnectionError,
    _authorization_headers,
)


class Answer(BaseModel):
    value: int


ResponseItem = httpx.Response | Exception | Callable[[httpx.Request], httpx.Response]


class QueueTransport(httpx.AsyncBaseTransport):
    def __init__(self, items: list[ResponseItem]) -> None:
        self.items = items
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item(request) if callable(item) else item


def response(payload: object, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=payload)


def generation_payload(content: object) -> dict[str, object]:
    return {"choices": [{"message": {"content": content}}]}


def request_json(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content)


def test_authorization_headers_never_materialize_for_missing_keys() -> None:
    assert _authorization_headers(None) == {}
    assert _authorization_headers(SecretStr("")) == {}
    assert _authorization_headers(SecretStr("canary-secret")) == {"Authorization": "Bearer canary-secret"}


async def test_client_success_unexpected_shape_and_injected_lifecycle() -> None:
    transport = QueueTransport([response({"data": []}), response(["not", "an", "object"])])
    async with httpx.AsyncClient(base_url="https://provider.test/v1/", transport=transport) as raw_client:
        client = OpenAICompatibleClient(base_url="https://ignored.test", client=raw_client)
        assert await client.get_models() == {"data": []}
        with pytest.raises(ProviderConnectionError, match="unexpected response shape"):
            await client.get_models()
        await client.close()
        assert raw_client.is_closed is False
    assert [request.url.path for request in transport.requests] == ["/v1/models", "/v1/models"]


@pytest.mark.parametrize(
    "item",
    [
        httpx.ConnectError("network included canary-secret"),
        httpx.Response(500, text="body included canary-secret"),
        httpx.Response(200, text="not-json canary-secret"),
    ],
)
async def test_client_errors_are_redacted(item: ResponseItem) -> None:
    transport = QueueTransport([item])
    async with httpx.AsyncClient(base_url="https://provider.test/v1/", transport=transport) as raw_client:
        client = OpenAICompatibleClient(base_url="https://ignored.test", client=raw_client, max_attempts=1)
        with pytest.raises(ProviderConnectionError) as captured:
            await client.get_models()
    assert "canary-secret" not in str(captured.value)
    assert "body included" not in str(captured.value)


async def test_client_retries_transient_connection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep = AsyncMock()
    monkeypatch.setattr("steering.providers.openai_compatible.asyncio.sleep", sleep)
    transport = QueueTransport([httpx.ConnectError("temporarily unavailable"), response({"data": []})])
    async with httpx.AsyncClient(base_url="https://provider.test/v1/", transport=transport) as raw:
        client = OpenAICompatibleClient(base_url="https://ignored.test", client=raw, max_attempts=2)
        assert await client.get_models() == {"data": []}

    assert len(transport.requests) == 2
    sleep.assert_awaited_once()


async def test_generation_structured_success_and_content_blocks() -> None:
    transport = QueueTransport(
        [
            response(generation_payload('{"value": 7}')),
            response(generation_payload([{"text": '{"value": 8}'}])),
        ]
    )
    async with httpx.AsyncClient(base_url="https://provider.test/v1/", transport=transport) as raw_client:
        provider = OpenAICompatibleGenerationProvider(
            client=OpenAICompatibleClient(base_url="https://ignored.test", client=raw_client),
            model_id="generation-model",
            temperature=0.25,
        )
        assert (
            await provider.generate_structured(
                system_prompt="system", user_prompt="user", response_model=Answer
            )
        ).value == 7
        assert (
            await provider.generate_structured(
                system_prompt="system", user_prompt="user", response_model=Answer
            )
        ).value == 8
    sent = request_json(transport.requests[0])
    assert sent["model"] == "generation-model"
    assert sent["temperature"] == 0.25
    assert sent["response_format"]["type"] == "json_schema"


async def test_generation_does_not_silently_degrade_when_schema_mode_is_rejected() -> None:
    transport = QueueTransport([response({"error": "unsupported"}, 400)])
    async with httpx.AsyncClient(base_url="https://provider.test/v1/", transport=transport) as raw_client:
        provider = OpenAICompatibleGenerationProvider(
            client=OpenAICompatibleClient(base_url="https://ignored.test", client=raw_client),
            model_id="generation-model",
        )
        with pytest.raises(ProviderConnectionError):
            await provider.generate_structured(
                system_prompt="system", user_prompt="user", response_model=Answer
            )
    assert len(transport.requests) == 1
    assert "response_format" in request_json(transport.requests[0])


@pytest.mark.parametrize(
    "payload",
    [{}, {"choices": []}, generation_payload("not-json"), generation_payload('{"value":"bad"}')],
)
async def test_generation_rejects_invalid_response_shapes(payload: dict[str, object]) -> None:
    transport = QueueTransport([response(payload)])
    async with httpx.AsyncClient(base_url="https://provider.test/v1/", transport=transport) as raw_client:
        provider = OpenAICompatibleGenerationProvider(
            client=OpenAICompatibleClient(base_url="https://ignored.test", client=raw_client),
            model_id="generation-model",
        )
        with pytest.raises(ProviderConnectionError, match="schema validation"):
            await provider.generate_structured(
                system_prompt="system", user_prompt="user", response_model=Answer
            )


async def test_generation_connection_and_image_extraction() -> None:
    transport = QueueTransport(
        [
            response(generation_payload('{"ok":true}')),
            response(generation_payload("technical diagram")),
            response(generation_payload([{"text": "first"}, {"text": "second"}])),
        ]
    )
    async with httpx.AsyncClient(base_url="https://provider.test/v1/", transport=transport) as raw_client:
        provider = OpenAICompatibleGenerationProvider(
            client=OpenAICompatibleClient(base_url="https://ignored.test", client=raw_client),
            model_id="vision-model",
        )
        await provider.test_connection()
        assert await provider.understand_image(content=b"png", mime_type="image/png") == "technical diagram"
        assert await provider.understand_image(content=b"webp", mime_type="image/webp") == "first\nsecond"
    image_request = request_json(transport.requests[1])
    data_url = image_request["messages"][1]["content"][1]["image_url"]["url"]
    assert data_url == "data:image/png;base64,cG5n"


async def test_image_limits_and_invalid_shape_are_safe() -> None:
    transport = QueueTransport([response({"choices": []})])
    async with httpx.AsyncClient(base_url="https://provider.test/v1/", transport=transport) as raw_client:
        provider = OpenAICompatibleGenerationProvider(
            client=OpenAICompatibleClient(base_url="https://ignored.test", client=raw_client),
            model_id="vision-model",
        )
        with pytest.raises(ValueError, match="unsupported image"):
            await provider.understand_image(content=b"x", mime_type="image/svg+xml")
        with pytest.raises(ValueError, match="5 MiB"):
            await provider.understand_image(content=b"x" * (5 * 1024 * 1024 + 1), mime_type="image/png")
        with pytest.raises(ProviderConnectionError, match="invalid shape"):
            await provider.understand_image(content=b"x", mime_type="image/gif")


async def test_embedding_orders_rows_converts_values_and_checks_dimensions() -> None:
    transport = QueueTransport(
        [
            response(
                {
                    "data": [
                        {"index": 1, "embedding": ["3", 4]},
                        {"index": 0, "embedding": [1, 2]},
                    ]
                }
            ),
            response({"data": [{"index": 0, "embedding": [1, 2]}]}),
        ]
    )
    async with httpx.AsyncClient(base_url="https://provider.test/v1/", transport=transport) as raw_client:
        provider = OpenAICompatibleEmbeddingProvider(
            client=OpenAICompatibleClient(base_url="https://ignored.test", client=raw_client),
            model_id="embedding-model",
            dimension=2,
        )
        assert await provider.embed(["a", "b"]) == [
            pytest.approx([0.4472135955, 0.894427191]),
            pytest.approx([0.6, 0.8]),
        ]
        assert request_json(transport.requests[0]) == {
            "model": "embedding-model",
            "input": ["a", "b"],
            "dimensions": 2,
            "encoding_format": "float",
        }
        await provider.test_connection()


@pytest.mark.parametrize(
    "payload, message",
    [
        ({}, "invalid shape"),
        ({"data": "bad"}, "invalid shape"),
        ({"data": [{"index": "bad", "embedding": [1, 2]}]}, "invalid shape"),
        ({"data": [{"index": 0, "embedding": [1]}]}, "dimension"),
        ({"data": []}, "count"),
    ],
)
async def test_embedding_rejects_malformed_count_and_dimensions(
    payload: dict[str, object], message: str
) -> None:
    transport = QueueTransport([response(payload)])
    async with httpx.AsyncClient(base_url="https://provider.test/v1/", transport=transport) as raw_client:
        provider = OpenAICompatibleEmbeddingProvider(
            client=OpenAICompatibleClient(base_url="https://ignored.test", client=raw_client),
            model_id="embedding-model",
            dimension=2,
        )
        with pytest.raises(ProviderConnectionError, match=message):
            await provider.embed(["one"])


async def test_embedding_connection_rejects_empty_override(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = QueueTransport([])
    async with httpx.AsyncClient(base_url="https://provider.test/v1/", transport=transport) as raw_client:
        provider = OpenAICompatibleEmbeddingProvider(
            client=OpenAICompatibleClient(base_url="https://ignored.test", client=raw_client),
            model_id="embedding-model",
            dimension=2,
        )

        async def empty(_text: str) -> list[float]:
            raise ProviderConnectionError("embedding provider returned no vectors")

        monkeypatch.setattr(provider, "embed_query", empty)
        with pytest.raises(ProviderConnectionError, match="no vectors"):
            await provider.test_connection()
