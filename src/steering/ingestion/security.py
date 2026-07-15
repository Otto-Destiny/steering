from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urljoin, urlsplit

import anyio
import httpx


class UnsafeSourceError(ValueError):
    pass


class SourceUnavailableError(RuntimeError):
    pass


class NetworkGuard:
    def __init__(self, *, allow_private_hosts: bool = False) -> None:
        self.allow_private_hosts = allow_private_hosts

    async def validate_url(self, url: str) -> str:
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"}:
            raise UnsafeSourceError("only http and https sources are allowed")
        if not parts.hostname or parts.username or parts.password:
            raise UnsafeSourceError("source URL has an invalid authority")
        if parts.hostname.lower() in {"localhost", "localhost.localdomain"}:
            raise UnsafeSourceError("local network sources are not allowed")
        try:
            addresses = await anyio.to_thread.run_sync(self._resolve, parts.hostname, parts.port)
        except OSError as exc:
            raise SourceUnavailableError("source hostname could not be resolved") from exc
        if not self.allow_private_hosts:
            for address in addresses:
                self.validate_connected_address(address)
        return url

    def validate_connected_address(self, address: str) -> None:
        """Verify the address actually used by the socket after DNS resolution."""

        if self.allow_private_hosts:
            return
        normalized = address.split("%", 1)[0]
        ip = ipaddress.ip_address(normalized)
        if not ip.is_global:
            raise UnsafeSourceError("private, loopback, link-local, or reserved sources are not allowed")

    @staticmethod
    def _resolve(hostname: str, port: int | None) -> set[str]:
        return {
            str(item[4][0]) for item in socket.getaddrinfo(hostname, port or 443, type=socket.SOCK_STREAM)
        }


class SafeFetcher:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        guard: NetworkGuard | None = None,
        max_bytes: int = 20 * 1024 * 1024,
        max_redirects: int = 4,
        require_connected_peer: bool | None = None,
    ) -> None:
        self.client = client
        self.guard = guard or NetworkGuard()
        self.max_bytes = max_bytes
        self.max_redirects = max_redirects
        self.require_connected_peer = (
            not isinstance(getattr(client, "_transport", None), httpx.MockTransport)
            if require_connected_peer is None
            else require_connected_peer
        )

    async def get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
    ) -> tuple[bytes, httpx.Headers, str]:
        current = url
        for _ in range(self.max_redirects + 1):
            await self.guard.validate_url(current)
            try:
                async with self.client.stream(
                    "GET",
                    current,
                    headers=headers,
                    params=params,
                    follow_redirects=False,
                ) as response:
                    network_stream = response.extensions.get("network_stream")
                    get_extra_info = getattr(network_stream, "get_extra_info", None)
                    peer = get_extra_info("server_addr") if callable(get_extra_info) else None
                    if isinstance(peer, (tuple, list)) and peer:
                        self.guard.validate_connected_address(str(peer[0]))
                    elif isinstance(peer, str):
                        self.guard.validate_connected_address(peer)
                    elif self.require_connected_peer:
                        raise SourceUnavailableError("source connection peer could not be verified")
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            raise SourceUnavailableError("source redirect did not include a location")
                        current = urljoin(current, location)
                        params = None
                        continue
                    response.raise_for_status()
                    data = bytearray()
                    async for part in response.aiter_bytes():
                        data.extend(part)
                        if len(data) > self.max_bytes:
                            raise SourceUnavailableError("source exceeded the configured download limit")
                    return bytes(data), response.headers, str(response.url)
            except httpx.HTTPError as exc:
                raise SourceUnavailableError(f"source request failed ({type(exc).__name__})") from None
        raise SourceUnavailableError("source exceeded the redirect limit")
