from __future__ import annotations

from collections.abc import Awaitable, Callable
from urllib.parse import urlsplit

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
MUTATION_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
# Allows one explicitly authorized 20 MiB knowledge upload plus multipart framing.
# JSON and text fields retain their tighter schema-level limits.
MAX_REQUEST_BYTES = 21 * 1_048_576


def _hostname(value: str) -> str | None:
    return urlsplit(value if "://" in value else f"//{value}").hostname


class LocalMutationGuardMiddleware(BaseHTTPMiddleware):
    """Reject non-local mutations, cross-origin forms, and oversized request bodies."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if request.method not in MUTATION_METHODS:
            return await call_next(request)

        host = request.url.hostname
        if host not in LOCAL_HOSTS:
            return self._reject(request, 403, "Mutations are allowed only through localhost.")

        origin = request.headers.get("origin")
        if origin and _hostname(origin) != host:
            return self._reject(request, 403, "Cross-origin mutation rejected.")
        if request.headers.get("sec-fetch-site", "").lower() == "cross-site":
            return self._reject(request, 403, "Cross-site mutation rejected.")

        length = request.headers.get("content-length")
        if length:
            try:
                if int(length) > MAX_REQUEST_BYTES:
                    return self._reject(request, 413, "Request body is too large.")
            except ValueError:
                return self._reject(request, 400, "Invalid Content-Length header.")
        return await call_next(request)

    @staticmethod
    def _reject(request: Request, status: int, message: str) -> Response:
        if request.url.path.startswith("/api/"):
            return JSONResponse({"error": message}, status_code=status)
        return PlainTextResponse(message, status_code=status)
