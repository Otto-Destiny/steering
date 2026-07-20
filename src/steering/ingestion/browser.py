from __future__ import annotations

import asyncio
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import urljoin, urlsplit

from steering.domain.models import ResolvedSource, SourceKind
from steering.ingestion.resolvers import canonical_http_url
from steering.ingestion.security import NetworkGuard, SourceUnavailableError, UnsafeSourceError

_STATUS_ID = re.compile(r"/status/(\d+)")
_X_HOSTS = {"x.com", "www.x.com", "twitter.com", "www.twitter.com"}
_X_MEDIA_HOSTS = {"pbs.twimg.com", "video.twimg.com"}
_X_THREAD_SPAN_HOURS = 48
_X_THREAD_MAX_SCROLLS = 30
_X_ARTICLES_SCRIPT = r"""() => [...document.querySelectorAll('article')].map(element => {
    const time = element.querySelector('time');
    const statusAnchor = time ? time.closest('a[href*="/status/"]') : null;
    const statusUrl = statusAnchor ? statusAnchor.href : null;
    if (!statusUrl) return null;
    const match = statusUrl.match(/(?:x|twitter)\.com\/([^/]+)\/status\/(\d+)/i);
    if (!match) return null;
    const textNode = element.querySelector('[data-testid="tweetText"]');
    const userNameNode = element.querySelector('[data-testid="User-Name"]');
    const links = [...element.querySelectorAll('a[href]')].map(anchor => ({
        url: anchor.href,
        text: (anchor.innerText || anchor.textContent || '').trim(),
        title: anchor.getAttribute('title'),
        aria_label: anchor.getAttribute('aria-label')
    }));
    const images = [...element.querySelectorAll('img[src*="pbs.twimg.com/media"]')].map(image => ({
        url: image.currentSrc || image.src,
        alt: image.alt || ''
    }));
    const videos = [...element.querySelectorAll('video[poster]')].map(video => ({
        url: video.poster,
        alt: ''
    }));
    return {
        post_id: match[2],
        handle: match[1],
        status_url: statusUrl.split('?')[0],
        posted_at: time ? time.getAttribute('datetime') : null,
        author_label: userNameNode ? userNameNode.innerText.trim() : null,
        text: textNode ? textNode.innerText.trim() : '',
        links,
        media: [...images, ...videos]
    };
}).filter(Boolean)"""


class BrowserCaptureUnavailable(RuntimeError):
    """Raised when the optional, explicitly authorized browser is unavailable."""


class BrowserDependencyUnavailable(BrowserCaptureUnavailable):
    """Raised when Playwright or its managed Chromium runtime is not installed."""


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
        login_timeout_seconds: int = 900,
    ) -> None:
        profile_root = Path(profile_directory).expanduser().resolve(strict=False)
        self.profile_directory = profile_root / "steering-managed-profile"
        self.guard = guard or NetworkGuard()
        self.timeout_ms = timeout_ms
        self.login_timeout_seconds = login_timeout_seconds

    async def capture(self, url: str, *, authorized: bool = False) -> ResolvedSource:
        if not authorized:
            raise PermissionError("browser capture requires explicit user authorization")
        canonical = canonical_http_url(url)
        await self.guard.validate_url(canonical)
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise BrowserDependencyUnavailable(
                "browser capture needs the 'browser' extra and an installed Chromium runtime"
            ) from exc

        self.profile_directory.mkdir(parents=True, exist_ok=True)
        try:
            async with async_playwright() as playwright:
                host = (urlsplit(canonical).hostname or "").lower()
                if host in _X_HOSTS:
                    context = await self._launch_x_context(playwright)
                    try:
                        page = context.pages[0] if context.pages else await context.new_page()
                        set_default_timeout = getattr(page, "set_default_timeout", None)
                        if callable(set_default_timeout):
                            set_default_timeout(10_000)
                        await page.goto(canonical, wait_until="domcontentloaded", timeout=60_000)
                        await self._wait_for_x_thread(page, canonical)
                        final_url = canonical_http_url(str(page.url))
                        await self.guard.validate_url(final_url)
                        payload = await self._capture_x_thread(context, page, canonical)
                    finally:
                        await context.close()
                else:
                    if not Path(playwright.chromium.executable_path).is_file():
                        raise BrowserDependencyUnavailable("the managed Chromium runtime is not installed")
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
                        payload = await self._capture_page(page)
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
            raise BrowserDependencyUnavailable("browser capture needs the 'browser' extra") from exc
        self.profile_directory.mkdir(parents=True, exist_ok=True)
        try:
            async with async_playwright() as playwright:
                if (urlsplit(canonical).hostname or "").lower() in _X_HOSTS:
                    await self._open_x_login(playwright, canonical)
                    return
                executable = Path(playwright.chromium.executable_path)
                if not executable.is_file():
                    raise BrowserDependencyUnavailable("the managed Chromium runtime is not installed")
            # Authentication is intentionally not Playwright-controlled. Some providers
            # reject valid credentials in automation-controlled login flows. Launching
            # Chromium directly keeps this step entirely human-driven while preserving
            # the same isolated profile later used for authorized capture.
            process = await asyncio.create_subprocess_exec(
                str(executable),
                f"--user-data-dir={self.profile_directory}",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-background-mode",
                "--new-window",
                canonical,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return_code = await process.wait()
            if return_code != 0:
                raise BrowserCaptureUnavailable(f"managed browser login exited with status {return_code}")
        except BrowserCaptureUnavailable:
            raise
        except Exception as exc:
            raise BrowserCaptureUnavailable(f"managed browser login failed ({type(exc).__name__})") from None

    async def _open_x_login(self, playwright: Any, canonical: str) -> None:
        """Use the proven evaluation login flow in a persistent real-Chrome profile."""

        context = await self._launch_x_context(playwright)
        try:
            page = context.pages[0] if context.pages else await context.new_page()
            set_default_timeout = getattr(page, "set_default_timeout", None)
            if callable(set_default_timeout):
                set_default_timeout(10_000)
            await page.goto(canonical, wait_until="domcontentloaded", timeout=60_000)
            deadline = asyncio.get_running_loop().time() + self.login_timeout_seconds
            login_navigation_attempted = "/i/flow/login" in str(page.url)
            while asyncio.get_running_loop().time() < deadline:
                if await self._x_logged_in(page):
                    return
                await self._dismiss_x_cookie_prompt(page)
                if page.is_closed():
                    raise BrowserCaptureUnavailable(
                        "the X login window closed before authentication completed"
                    )
                if not self._x_login_visible(page) and not login_navigation_attempted:
                    await page.goto(
                        "https://x.com/i/flow/login",
                        wait_until="domcontentloaded",
                        timeout=60_000,
                    )
                    login_navigation_attempted = True
                await page.wait_for_timeout(1_000)
            raise BrowserCaptureUnavailable(
                f"X login did not complete within {self.login_timeout_seconds} seconds"
            )
        finally:
            await context.close()

    async def _launch_x_context(self, playwright: Any) -> Any:
        """Launch the same persistent real-Chrome profile for X login and capture."""

        options = {
            "headless": False,
            "no_viewport": True,
            "locale": "en-US",
            "args": [
                "--disable-blink-features=AutomationControlled",
                "--start-maximized",
            ],
        }
        try:
            context = await playwright.chromium.launch_persistent_context(
                str(self.profile_directory),
                channel="chrome",
                **options,
            )
        except Exception:
            executable = Path(playwright.chromium.executable_path)
            if not executable.is_file():
                raise BrowserDependencyUnavailable(
                    "Google Chrome and the managed Chromium runtime are unavailable"
                ) from None
            try:
                context = await playwright.chromium.launch_persistent_context(
                    str(self.profile_directory),
                    **options,
                )
            except Exception as exc:
                raise BrowserCaptureUnavailable(
                    f"managed X browser could not launch ({type(exc).__name__})"
                ) from None
        return context

    @staticmethod
    def _x_login_visible(page: Any) -> bool:
        return "/login" in str(page.url) or "/i/flow/login" in str(page.url)

    @staticmethod
    async def _x_logged_in(page: Any) -> bool:
        selectors = (
            '[data-testid="AppTabBar_Home_Link"]',
            '[data-testid="SideNav_NewTweet_Button"]',
            'a[href="/home"]',
        )
        return any([await page.locator(selector).count() for selector in selectors])

    @staticmethod
    async def _dismiss_x_cookie_prompt(page: Any) -> None:
        for label in ("Accept all cookies", "Refuse non-essential cookies"):
            button = page.get_by_role("button", name=label)
            if await button.count():
                try:
                    await button.first.click(timeout=2_000)
                    return
                except Exception:
                    return

    async def _wait_for_x_thread(self, page: Any, canonical: str) -> None:
        status_match = _STATUS_ID.search(canonical)
        if status_match is None:
            raise BrowserCaptureUnavailable("signed-in X capture requires a post URL")
        root_post_id = status_match.group(1)
        deadline = asyncio.get_running_loop().time() + self.login_timeout_seconds
        last_navigation = 0.0
        login_navigation_attempted = False

        while asyncio.get_running_loop().time() < deadline:
            if page.is_closed():
                raise BrowserCaptureUnavailable("the X capture window closed before capture completed")
            await self._dismiss_x_cookie_prompt(page)
            logged_in = await self._x_logged_in(page)
            if logged_in and await self._x_thread_visible(page, root_post_id):
                return

            if logged_in and not self._x_login_visible(page):
                now = asyncio.get_running_loop().time()
                if now - last_navigation >= 10:
                    await page.goto(canonical, wait_until="domcontentloaded", timeout=60_000)
                    last_navigation = now
            elif not self._x_login_visible(page) and not login_navigation_attempted:
                await page.goto(
                    "https://x.com/i/flow/login",
                    wait_until="domcontentloaded",
                    timeout=60_000,
                )
                login_navigation_attempted = True
            await page.wait_for_timeout(1_000)

        raise BrowserCaptureUnavailable(
            f"the X thread did not become available within {self.login_timeout_seconds} seconds"
        )

    @staticmethod
    async def _x_thread_visible(page: Any, root_post_id: str) -> bool:
        return bool(await page.locator(f'a[href*="/status/{root_post_id}"] time').count())

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
                # The main document is verified strictly in _navigate. Browser APIs do
                # not expose a peer address for every cached or synthetic subresource.
                await self._validate_browser_response(response, required=False)
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
    async def _capture_page(page: Any) -> dict[str, Any]:
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

    async def _capture_x_thread(
        self,
        context: Any,
        page: Any,
        canonical: str,
    ) -> dict[str, Any]:
        status_match = _STATUS_ID.search(canonical)
        if status_match is None:
            raise BrowserCaptureUnavailable("signed-in X capture requires a post URL")
        root_post_id = status_match.group(1)

        seen: dict[str, dict[str, Any]] = {}
        encounter_order: list[str] = []
        stagnant_scrolls = 0
        for _ in range(_X_THREAD_MAX_SCROLLS):
            if page.is_closed():
                raise BrowserCaptureUnavailable("the X capture window closed before capture completed")
            batch = await page.evaluate(_X_ARTICLES_SCRIPT)
            before = len(seen)
            if isinstance(batch, list):
                for item in batch:
                    if not isinstance(item, dict) or not item.get("post_id"):
                        continue
                    post_id = str(item["post_id"])
                    if post_id not in seen:
                        seen[post_id] = item
                        encounter_order.append(post_id)

            stagnant_scrolls = stagnant_scrolls + 1 if len(seen) == before else 0
            if stagnant_scrolls >= 4:
                break
            await page.evaluate("window.scrollBy(0, Math.max(window.innerHeight * 0.85, 700))")
            await page.wait_for_timeout(1_250)

        root = seen.get(root_post_id)
        if root is None:
            raise BrowserCaptureUnavailable("the root X post was not captured from the visible thread")
        root_handle = str(root.get("handle") or "").strip()
        if not root_handle:
            raise BrowserCaptureUnavailable("the root X post author could not be identified")

        same_author = [
            seen[post_id]
            for post_id in encounter_order
            if str(seen[post_id].get("handle") or "").lower() == root_handle.lower()
        ]
        same_author = [root, *[post for post in same_author if post is not root]]
        retained = self._filter_x_thread_window(same_author, root)
        links = await self._x_outbound_urls(context, retained)
        media = self._x_media(retained)
        texts = [str(post.get("text") or "").strip() for post in retained]
        text = "\n\n---\n\n".join(item for item in texts if item)
        author_label = str(root.get("author_label") or "").strip()

        return {
            "title": f"{author_label or '@' + root_handle} — X thread",
            "text": text,
            "links": links,
            "media": media,
            "author": root_handle,
            "bundled_self_replies": max(0, len(retained) - 1),
        }

    @staticmethod
    def _parse_x_timestamp(value: Any) -> datetime | None:
        if not isinstance(value, str) or not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    @classmethod
    def _filter_x_thread_window(
        cls,
        posts: list[dict[str, Any]],
        root: dict[str, Any],
    ) -> list[dict[str, Any]]:
        root_time = cls._parse_x_timestamp(root.get("posted_at"))
        if root_time is None:
            return posts
        max_seconds = _X_THREAD_SPAN_HOURS * 60 * 60
        retained: list[dict[str, Any]] = []
        for post in posts:
            posted_at = cls._parse_x_timestamp(post.get("posted_at"))
            if posted_at is not None and abs((posted_at - root_time).total_seconds()) > max_seconds:
                continue
            retained.append(post)
        return retained

    @staticmethod
    def _x_media(posts: list[dict[str, Any]]) -> list[dict[str, str]]:
        media: list[dict[str, str]] = []
        seen_urls: set[str] = set()
        for post in posts:
            raw_media = post.get("media")
            if not isinstance(raw_media, list):
                continue
            for item in raw_media:
                if not isinstance(item, dict) or not item.get("url"):
                    continue
                url = str(item["url"])
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                media.append({"url": url, "alt": str(item.get("alt") or "")})
        return media

    async def _x_outbound_urls(
        self,
        context: Any,
        posts: list[dict[str, Any]],
    ) -> list[str]:
        outbound: list[str] = []
        for post in posts:
            raw_links = post.get("links")
            if not isinstance(raw_links, list):
                continue
            for link in raw_links:
                if not isinstance(link, dict):
                    continue
                resolved = await self._resolve_x_link(context, link)
                if resolved and resolved not in outbound:
                    outbound.append(resolved)
        return outbound

    async def _resolve_x_link(self, context: Any, link: dict[str, Any]) -> str | None:
        raw_url = str(link.get("url") or "").strip()
        parts = urlsplit(raw_url)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            return None
        host = parts.hostname.lower()
        if host in _X_HOSTS or host in _X_MEDIA_HOSTS:
            return None

        resolved = raw_url
        if host == "t.co":
            displayed = next(
                (
                    candidate
                    for value in (link.get("text"), link.get("title"), link.get("aria_label"))
                    if (candidate := self._displayed_external_url(value)) is not None
                ),
                None,
            )
            resolved = displayed or await self._resolve_tco(context, raw_url) or ""
        if not resolved:
            return None

        resolved_parts = urlsplit(resolved)
        resolved_host = (resolved_parts.hostname or "").lower()
        if (
            resolved_parts.scheme not in {"http", "https"}
            or not resolved_host
            or resolved_host in _X_HOSTS
            or resolved_host in _X_MEDIA_HOSTS
            or resolved_host == "t.co"
        ):
            return None
        return canonical_http_url(resolved)

    @staticmethod
    def _displayed_external_url(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        candidate = re.sub(r"\s+", "", value)
        if "…" in candidate or not candidate.startswith(("http://", "https://")):
            return None
        parts = urlsplit(candidate)
        host = (parts.hostname or "").lower()
        if not host or host in _X_HOSTS or host in _X_MEDIA_HOSTS or host == "t.co":
            return None
        return candidate

    async def _resolve_tco(self, context: Any, url: str) -> str | None:
        current = canonical_http_url(url)
        for _ in range(10):
            try:
                await self.guard.validate_url(current)
                response = await context.request.get(
                    current,
                    fail_on_status_code=False,
                    max_redirects=0,
                    timeout=20_000,
                )
            except Exception:
                return None
            location = response.headers.get("location")
            if not location:
                final_url = canonical_http_url(str(response.url))
                if final_url == current:
                    return None
                try:
                    await self.guard.validate_url(final_url)
                except Exception:
                    return None
                return final_url
            candidate = canonical_http_url(urljoin(current, location))
            try:
                await self.guard.validate_url(candidate)
            except Exception:
                return None
            if (urlsplit(candidate).hostname or "").lower() != "t.co":
                return candidate
            current = candidate
        return None

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
