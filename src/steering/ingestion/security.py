from __future__ import annotations

import asyncio
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
        normalized = address.strip()
        if normalized.startswith("[") and normalized.endswith("]"):
            normalized = normalized[1:-1]
        normalized = normalized.split("%", 1)[0]
        try:
            ip = ipaddress.ip_address(normalized)
        except ValueError:
            raise UnsafeSourceError("source peer returned an invalid IP address") from None
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
        max_attempts: int = 3,
        require_connected_peer: bool | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.client = client
        self.guard = guard or NetworkGuard()
        self.max_bytes = max_bytes
        self.max_redirects = max_redirects
        self.max_attempts = max_attempts
        self.require_connected_peer = (
            not isinstance(getattr(client, "_transport", None), httpx.MockTransport)
            if require_connected_peer is None
            else require_connected_peer
        )

    def _validate_peer(self, response: httpx.Response) -> None:
        network_stream = response.extensions.get("network_stream")
        get_extra_info = getattr(network_stream, "get_extra_info", None)
        peer = get_extra_info("server_addr") if callable(get_extra_info) else None
        if isinstance(peer, (tuple, list)) and peer:
            self.guard.validate_connected_address(str(peer[0]))
        elif isinstance(peer, str):
            self.guard.validate_connected_address(peer)
        elif self.require_connected_peer:
            raise SourceUnavailableError("source connection peer could not be verified")

    @staticmethod
    def _redirect_target(response: httpx.Response, current: str) -> str:
        location = response.headers.get("location")
        if not location:
            raise SourceUnavailableError("source redirect did not include a location")
        return str(urljoin(current, location))

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
            redirected = False
            for attempt in range(self.max_attempts):
                try:
                    async with self.client.stream(
                        "GET",
                        current,
                        headers=headers,
                        params=params,
                        follow_redirects=False,
                    ) as response:
                        self._validate_peer(response)
                        if response.status_code in {301, 302, 303, 307, 308}:
                            current = self._redirect_target(response, current)
                            params = None
                            redirected = True
                            break
                        response.raise_for_status()
                        data = bytearray()
                        async for part in response.aiter_bytes():
                            data.extend(part)
                            if len(data) > self.max_bytes:
                                raise SourceUnavailableError("source exceeded the configured download limit")
                        return bytes(data), response.headers, str(response.url)
                except httpx.TransportError as exc:
                    if attempt + 1 < self.max_attempts:
                        await asyncio.sleep(0.25 * (2**attempt))
                        continue
                    raise SourceUnavailableError(f"source request failed ({type(exc).__name__})") from None
                except httpx.HTTPError as exc:
                    raise SourceUnavailableError(f"source request failed ({type(exc).__name__})") from None
            if redirected:
                continue
        raise SourceUnavailableError("source exceeded the redirect limit")

    async def resolve_redirects(self, url: str) -> str:
        """Follow a shortlink to its destination without downloading the target.

        Every hop is validated exactly as it is in :meth:`get`, and the response
        body is never read, so unwrapping a link costs one request rather than a
        full page download.
        """

        current = url
        for hop in range(self.max_redirects + 1):
            await self.guard.validate_url(current)
            try:
                async with self.client.stream("GET", current, follow_redirects=False) as response:
                    self._validate_peer(response)
                    if response.status_code not in {301, 302, 303, 307, 308}:
                        if hop == 0:
                            raise SourceUnavailableError("source did not redirect to a destination")
                        return current
                    current = self._redirect_target(response, current)
            except httpx.HTTPError as exc:
                raise SourceUnavailableError(f"source request failed ({type(exc).__name__})") from None
        raise SourceUnavailableError("source exceeded the redirect limit")
