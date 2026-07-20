from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

from steering.ingestion.security import (
    NetworkGuard,
    SafeFetcher,
    SourceUnavailableError,
    UnsafeSourceError,
)


@pytest.mark.asyncio
async def test_network_guard_blocks_private_resolution_and_url_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = NetworkGuard()
    monkeypatch.setattr(guard, "_resolve", lambda _host, _port: {"127.0.0.1"})

    with pytest.raises(UnsafeSourceError, match="private"):
        await guard.validate_url("https://public-name.example/resource")
    with pytest.raises(UnsafeSourceError, match="invalid authority"):
        await guard.validate_url("https://user:secret@example.org/resource")
    with pytest.raises(UnsafeSourceError, match="only http"):
        await guard.validate_url("file:///etc/passwd")
    with pytest.raises(UnsafeSourceError, match="local network"):
        await guard.validate_url("http://localhost/resource")


@pytest.mark.asyncio
async def test_network_guard_reports_dns_failure_and_can_explicitly_allow_private_hosts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = NetworkGuard()

    def fail_resolution(_host: str, _port: int | None) -> set[str]:
        raise OSError("fixture DNS failure")

    monkeypatch.setattr(guard, "_resolve", fail_resolution)
    with pytest.raises(SourceUnavailableError, match="could not be resolved"):
        await guard.validate_url("https://missing.example/resource")

    private_guard = NetworkGuard(allow_private_hosts=True)
    monkeypatch.setattr(private_guard, "_resolve", lambda _host, _port: {"10.0.0.1"})
    assert await private_guard.validate_url("http://internal.example/resource") == (
        "http://internal.example/resource"
    )


def test_network_guard_accepts_bracketed_public_ipv6_from_browser_peer() -> None:
    guard = NetworkGuard()

    guard.validate_connected_address("[2606:4700:4700::1111]")
    with pytest.raises(UnsafeSourceError, match="private"):
        guard.validate_connected_address("[::1]")
    with pytest.raises(UnsafeSourceError, match="invalid IP"):
        guard.validate_connected_address("[not-an-ip]")


class RecordingGuard:
    def __init__(self) -> None:
        self.seen: list[str] = []

    async def validate_url(self, url: str) -> str:
        self.seen.append(url)
        if "127.0.0.1" in url:
            raise UnsafeSourceError("blocked redirected private target")
        return url


@pytest.mark.asyncio
async def test_safe_fetcher_revalidates_redirects_and_bounds_redirect_count() -> None:
    requests: list[str] = []

    def private_redirect(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://127.0.0.1/internal"})

    guard = RecordingGuard()
    client = httpx.AsyncClient(transport=httpx.MockTransport(private_redirect))
    fetcher = SafeFetcher(client=client, guard=guard)  # type: ignore[arg-type]
    with pytest.raises(UnsafeSourceError, match="private target"):
        await fetcher.get("https://example.org/start")
    assert requests == ["https://example.org/start"]
    assert guard.seen == ["https://example.org/start", "http://127.0.0.1/internal"]
    await client.aclose()

    def endless_redirect(request: httpx.Request) -> httpx.Response:
        step = int(request.url.params.get("step", "0"))
        return httpx.Response(302, headers={"location": f"/loop?step={step + 1}"})

    guard = RecordingGuard()
    client = httpx.AsyncClient(transport=httpx.MockTransport(endless_redirect))
    fetcher = SafeFetcher(client=client, guard=guard, max_redirects=2)  # type: ignore[arg-type]
    with pytest.raises(SourceUnavailableError, match="redirect limit"):
        await fetcher.get("https://example.org/loop?step=0")
    assert len(guard.seen) == 3
    await client.aclose()


@pytest.mark.asyncio
async def test_safe_fetcher_enforces_streamed_byte_limit() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=b"123456"))
    )
    fetcher = SafeFetcher(
        client=client,
        guard=RecordingGuard(),  # type: ignore[arg-type]
        max_bytes=5,
    )
    with pytest.raises(SourceUnavailableError, match="download limit"):
        await fetcher.get("https://example.org/large")
    await client.aclose()


@pytest.mark.asyncio
async def test_safe_fetcher_reports_missing_redirect_and_http_errors_without_response_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/missing-location":
            return httpx.Response(302)
        return httpx.Response(503, text="sensitive upstream body")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    fetcher = SafeFetcher(client=client, guard=RecordingGuard())  # type: ignore[arg-type]
    with pytest.raises(SourceUnavailableError, match="did not include a location") as redirect_error:
        await fetcher.get("https://example.org/missing-location")
    assert "sensitive" not in str(redirect_error.value)

    with pytest.raises(SourceUnavailableError, match="HTTPStatusError") as http_error:
        await fetcher.get("https://example.org/unavailable")
    assert "sensitive upstream body" not in str(http_error.value)
    await client.aclose()


@pytest.mark.asyncio
async def test_safe_fetcher_retries_transient_transport_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep = AsyncMock()
    monkeypatch.setattr("steering.ingestion.security.asyncio.sleep", sleep)
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ConnectError("temporary connection failure")
        return httpx.Response(200, content=b"source")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    fetcher = SafeFetcher(client=client, guard=RecordingGuard())  # type: ignore[arg-type]
    data, _, _ = await fetcher.get("https://example.org/source")

    assert data == b"source"
    assert attempts == 3
    assert sleep.await_count == 2
    await client.aclose()


@pytest.mark.asyncio
async def test_safe_fetcher_bounds_exhausted_transport_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("steering.ingestion.security.asyncio.sleep", AsyncMock())
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("private failure detail")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    fetcher = SafeFetcher(
        client=client,
        guard=RecordingGuard(),  # type: ignore[arg-type]
        max_attempts=2,
    )
    with pytest.raises(SourceUnavailableError, match="ConnectError") as captured:
        await fetcher.get("https://example.org/source")

    assert attempts == 2
    assert "private failure detail" not in str(captured.value)
    await client.aclose()


@pytest.mark.asyncio
async def test_safe_fetcher_rejects_a_private_connected_peer_after_public_dns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PrivatePeerStream:
        def get_extra_info(self, name: str) -> tuple[str, int] | None:
            return ("127.0.0.1", 443) if name == "server_addr" else None

    guard = NetworkGuard()
    monkeypatch.setattr(guard, "_resolve", lambda _host, _port: {"93.184.216.34"})
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                content=b"private rebinding target",
                extensions={"network_stream": PrivatePeerStream()},
            )
        )
    )
    fetcher = SafeFetcher(client=client, guard=guard)

    with pytest.raises(UnsafeSourceError, match="private"):
        await fetcher.get("https://public-name.example/resource")

    await client.aclose()
