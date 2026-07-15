from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

from steering.domain.models import ResolvedSource, SourceKind
from steering.ingestion.resolvers import canonical_http_url
from steering.ingestion.security import NetworkGuard, SourceUnavailableError, UnsafeSourceError

_STATUS_ID = re.compile(r"/status/(\d+)")


class BrowserCaptureUnavailable(RuntimeError):
    """Raised when the optional, explicitly authorized browser is unavailable."""


class ManagedBrowserCapture:
    """Capture one visible page with STEERING's isolated Playwright profile.

    The caller must explicitly authorize every capture. The managed profile is
    never inferred from, or pointed at, a user's normal browser profile.
    """

    def __init__(
        self,
        *,
        profile_directory: str | Path,
        guard: NetworkGuard | None = None,
        timeout_ms: int = 45_000,
    ) -> None:
        profile_root = Path(profile_directory).expanduser().resolve(strict=False)
        self.profile_directory = profile_root / "steering-managed-profile"
        self.guard = guard or NetworkGuard()
        self.timeout_ms = timeout_ms

    async def capture(self, url: str, *, authorized: bool = False) -> ResolvedSource:
        if not authorized:
            raise PermissionError("browser capture requires explicit user authorization")
        canonical = canonical_http_url(url)
        await self.guard.validate_url(canonical)
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise BrowserCaptureUnavailable(
                "browser capture needs the 'browser' extra and an installed Chromium runtime"
            ) from exc

        self.profile_directory.mkdir(parents=True, exist_ok=True)
        try:
            async with async_playwright() as playwright:
                context = await playwright.chromium.launch_persistent_context(
                    str(self.profile_directory),
                    headless=False,
                    viewport={"width": 1280, "height": 900},
                    service_workers="block",
                )
                try:
                    page = context.pages[0] if context.pages else await context.new_page()
                    await context.route("**/*", self._guard_browser_request)
                    peer_violations = self._install_response_guard(page)
                    final_url = await self._navigate(page, canonical)
                    self._raise_peer_violation(peer_violations)
                    await page.wait_for_timeout(1_000)
                    self._raise_peer_violation(peer_violations)
                    payload = await self._capture_page(page, final_url)
                finally:
                    await context.close()
        except BrowserCaptureUnavailable:
            raise
        except Exception as exc:
            raise BrowserCaptureUnavailable(
                f"authorized browser capture failed ({type(exc).__name__})"
            ) from None
        return self._resolved(final_url, payload)

    async def open_login(self, url: str, *, authorized: bool = False) -> None:
        """Open the isolated visible profile so the user can sign in themselves."""

        if not authorized:
            raise PermissionError("opening the managed browser requires explicit authorization")
        canonical = canonical_http_url(url)
        await self.guard.validate_url(canonical)
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise BrowserCaptureUnavailable("browser capture needs the 'browser' extra") from exc
        self.profile_directory.mkdir(parents=True, exist_ok=True)
        async with async_playwright() as playwright:
            context = await playwright.chromium.launch_persistent_context(
                str(self.profile_directory), headless=False, service_workers="block"
            )
            try:
                page = context.pages[0] if context.pages else await context.new_page()
                await context.route("**/*", self._guard_browser_request)
                peer_violations = self._install_response_guard(page)
                await self._navigate(page, canonical)
                self._raise_peer_violation(peer_violations)
                # The browser remains visible until the user closes it. STEERING never
                # reads or requests the password used during this interactive step.
                await context.wait_for_event("close")
            finally:
                if context.pages:
                    await context.close()

    async def _navigate(self, page: Any, canonical: str) -> str:
        response = await page.goto(canonical, wait_until="domcontentloaded", timeout=self.timeout_ms)
        if response is None:
            raise SourceUnavailableError("browser navigation peer could not be verified")
        await self._validate_browser_response(response, required=True)
        final_url = canonical_http_url(str(page.url))
        await self.guard.validate_url(final_url)
        return final_url

    async def _guard_browser_request(self, route: Any) -> None:
        request_url = str(route.request.url)
        scheme = urlsplit(request_url).scheme.lower()
        if scheme in {"http", "https"}:
            try:
                await self.guard.validate_url(request_url)
            except (UnsafeSourceError, SourceUnavailableError):
                await route.abort(error_code="blockedbyclient")
                return
            await route.continue_()
            return
        if scheme in {"about", "blob", "data"}:
            await route.continue_()
            return
        await route.abort(error_code="blockedbyclient")

    def _install_response_guard(self, page: Any) -> list[Exception]:
        violations: list[Exception] = []

        async def inspect(response: Any) -> None:
            try:
                await self._validate_browser_response(response, required=True)
            except (UnsafeSourceError, SourceUnavailableError) as exc:
                violations.append(exc)
                if not page.is_closed():
                    await page.close()

        page.on("response", inspect)
        return violations

    @staticmethod
    def _raise_peer_violation(violations: list[Exception]) -> None:
        if violations:
            raise violations[0]

    async def _validate_browser_response(self, response: Any, *, required: bool) -> None:
        if urlsplit(str(response.url)).scheme.lower() not in {"http", "https"}:
            return
        try:
            peer = await response.server_addr()
        except Exception as exc:
            if required:
                raise SourceUnavailableError("browser response peer could not be verified") from exc
            return
        address = peer.get("ipAddress") if isinstance(peer, dict) else None
        if not address:
            if required:
                raise SourceUnavailableError("browser response peer could not be verified")
            return
        self.guard.validate_connected_address(str(address))

    @staticmethod
    async def _capture_page(page: Any, canonical: str) -> dict[str, Any]:
        host = (urlsplit(canonical).hostname or "").lower()
        if host in {"x.com", "twitter.com", "www.x.com", "www.twitter.com"}:
            return await ManagedBrowserCapture._capture_x_thread(page, canonical)
        return cast(
            dict[str, Any],
            await page.evaluate(
                """() => ({
                title: document.title || location.hostname,
                text: (document.querySelector('main') || document.body)?.innerText || '',
                links: [...document.querySelectorAll('main a[href], article a[href]')]
                    .map(a => a.href).filter(Boolean),
                media: [...document.querySelectorAll('main img[src], article img[src]')]
                    .map(i => ({url: i.currentSrc || i.src, alt: i.alt || ''})),
                author: document.querySelector('[rel="author"]')?.textContent?.trim() || null
            })"""
            ),
        )

    @staticmethod
    async def _capture_x_thread(page: Any, canonical: str) -> dict[str, Any]:
        status_match = _STATUS_ID.search(canonical)
        status_id = status_match.group(1) if status_match else ""
        # A few bounded scrolls expose self-replies in this single conversation;
        # this intentionally does not automate a feed or timeline.
        for _ in range(4):
            await page.mouse.wheel(0, 900)
            await page.wait_for_timeout(350)
        return cast(
            dict[str, Any],
            await page.evaluate(
                r"""({statusId}) => {
                const articles = [...document.querySelectorAll('article')];
                const root = articles.find(a => [...a.querySelectorAll('a[href*="/status/"]')]
                    .some(link => link.href.includes(`/status/${statusId}`))) || articles[0];
                const rootStatusLink = root && [...root.querySelectorAll('a[href*="/status/"]')]
                    .find(link => link.href.includes(`/status/${statusId}`));
                const rootPathParts = rootStatusLink
                    ? new URL(rootStatusLink.href).pathname.split('/').filter(Boolean)
                    : [];
                const authorPath = rootPathParts.length >= 3 ? `/${rootPathParts[0]}` : null;
                const own = articles.filter(article => {
                    if (!authorPath) return article === root;
                    return [...article.querySelectorAll('a[href*="/status/"]')]
                        .some(a => new URL(a.href).pathname.startsWith(`${authorPath}/status/`));
                });
                const selected = own.length ? own : (root ? [root] : []);
                const text = selected.map(a => a.innerText.trim()).filter(Boolean).join('\n\n---\n\n');
                const links = selected.flatMap(a => [...a.querySelectorAll('a[href]')]
                    .map(link => link.href)).filter(Boolean);
                const media = selected.flatMap(a => [...a.querySelectorAll('img[src]')]
                    .map(i => ({url: i.currentSrc || i.src, alt: i.alt || ''})));
                return {
                    title: root?.innerText?.split('\n').slice(0, 2).join(' — ') || document.title,
                    text,
                    links,
                    media,
                    author: authorPath ? authorPath.slice(1) : null,
                    bundled_self_replies: Math.max(0, selected.length - 1)
                };
            }""",
                {"statusId": status_id},
            ),
        )

    @staticmethod
    def _resolved(canonical: str, payload: dict[str, Any]) -> ResolvedSource:
        host = (urlsplit(canonical).hostname or "").lower()
        source_kind = (
            SourceKind.X
            if host in {"x.com", "twitter.com", "www.x.com", "www.twitter.com"}
            else SourceKind.LINKEDIN
            if host.endswith("linkedin.com")
            else SourceKind.WEBPAGE
        )
        links = list(dict.fromkeys(str(link) for link in payload.get("links", []) if link))
        media = payload.get("media", [])
        media_urls = list(
            dict.fromkeys(
                str(item.get("url")) for item in media if isinstance(item, dict) and item.get("url")
            )
        )
        useful_media = [
            item
            for item in media
            if isinstance(item, dict)
            and len(str(item.get("alt", "")).strip()) >= 24
            and not any(word in str(item.get("alt", "")).lower() for word in ("avatar", "logo", "emoji"))
        ]
        return ResolvedSource(
            canonical_url=canonical,
            source_kind=source_kind,
            title=str(payload.get("title") or canonical),
            text=str(payload.get("text") or "").strip(),
            author=str(payload["author"]) if payload.get("author") else None,
            mime_type="text/html",
            extraction_method="authorized_visible_browser",
            partial=not bool(str(payload.get("text") or "").strip()),
            outbound_urls=links,
            media_urls=media_urls,
            metadata={
                "bundled_self_replies": int(payload.get("bundled_self_replies", 0)),
                "media_inclusion_candidates": useful_media,
                "media_decision": (
                    "consider_alt_text_candidates_only_when_no_stronger_linked_source_exists"
                    if useful_media
                    else "skip_no_inexpensive_technical_signal"
                ),
            },
        )
