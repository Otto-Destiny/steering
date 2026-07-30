from __future__ import annotations

import asyncio
import logging
import re
import subprocess
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

from steering.domain.models import ResolvedSource, SourceKind
from steering.ingestion.resolvers import canonical_http_url
from steering.ingestion.security import (
    NetworkGuard,
    SafeFetcher,
    SourceUnavailableError,
    UnsafeSourceError,
)
from steering.ingestion.x import (
    AUTHOR_THREAD,
    BROWSER_METHOD,
    ROOT_POST_ONLY,
    X_HOSTS,
    is_x_owned_url,
    parse_x_post_url,
)

LOGGER = logging.getLogger(__name__)

_X_THREAD_SPAN_HOURS = 48
_X_THREAD_MAX_SCROLLS = 30
_X_STAGNANT_SCROLL_LIMIT = 4
# X fills an article in stages: text first, then the link-preview card that holds
# the author's outbound URL, which can take several seconds. Scrolling past an
# article removes it from the virtualized DOM before that card ever attaches, so
# the top of the thread is settled in place first. The root and the author's own
# replies sit there, and they are the only posts a capture keeps.
_X_SETTLE_SECONDS = 10.0
_X_SETTLE_INTERVAL_MS = 1_500
_AUTH_GRACE_SECONDS = 10
_MIN_DESCRIPTIVE_ALT_CHARS = 24
_DECORATIVE_ALT_HINTS = ("avatar", "emoji", "headshot", "logo", "profile photo")

# A quoted post renders inside a nested `div[role="link"]` card that carries its
# own `<time>` and status anchor. Selecting the first `<time>` in the article
# would attribute the post to the quoted author and lose the root post entirely,
# so the article's own timestamp is the first one outside any nested card.
_X_ARTICLES_SCRIPT = r"""() => [...document.querySelectorAll('article')].map(element => {
    const time = [...element.querySelectorAll('time')]
        .find(node => !node.closest('div[role="link"]')) || null;
    const statusAnchor = time ? time.closest('a[href*="/status/"]') : null;
    const statusUrl = statusAnchor ? statusAnchor.href : null;
    if (!statusUrl) return null;
    const match = statusUrl.match(/(?:x|twitter)\.com\/([^/]+)\/status\/(\d+)/i);
    if (!match) return null;
    const quoted = element.querySelector('div[role="link"] [data-testid="tweetText"]');
    const textNodes = [...element.querySelectorAll('[data-testid="tweetText"]')]
        .filter(node => !node.closest('div[role="link"]'));
    const userNameNode = element.querySelector('[data-testid="User-Name"]');
    const links = [...element.querySelectorAll('a[href]')]
        .filter(anchor => !anchor.closest('div[role="link"]'))
        .map(anchor => ({
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
        text: textNodes.map(node => node.innerText.trim()).filter(Boolean).join('\n'),
        quoted_text: quoted ? quoted.innerText.trim() : '',
        links,
        media: [...images, ...videos]
    };
}).filter(Boolean)"""


class BrowserCaptureUnavailable(RuntimeError):
    """Raised when the optional, explicitly authorized browser is unavailable."""


class BrowserDependencyUnavailable(BrowserCaptureUnavailable):
    """Raised when Playwright or its managed Chromium runtime is not installed."""


class BrowserAuthenticationRequired(BrowserCaptureUnavailable):
    """Raised when the managed profile holds no usable session for the platform."""


class LoginOutcome(StrEnum):
    SIGNED_IN = "signed_in"
    CLOSED_BEFORE_SIGN_IN = "closed_before_sign_in"
    ALREADY_SIGNED_IN = "already_signed_in"


class _ValidatedHosts:
    """Memoize host checks so guarding every subresource stays affordable.

    A single X post issues hundreds of subresource requests. Resolving DNS for
    each one would dominate capture time, while resolving once per host keeps the
    same guarantee for the duration of one capture.
    """

    def __init__(self, guard: NetworkGuard) -> None:
        self._guard = guard
        self._decisions: dict[str, bool] = {}

    async def allows(self, url: str) -> bool:
        host = (urlsplit(url).hostname or "").lower()
        if not host:
            return False
        cached = self._decisions.get(host)
        if cached is not None:
            return cached
        try:
            await self._guard.validate_url(url)
        except (UnsafeSourceError, SourceUnavailableError) as exc:
            LOGGER.debug("browser request to %s blocked (%s)", host, exc)
            self._decisions[host] = False
            return False
        self._decisions[host] = True
        return True


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
        link_resolver: SafeFetcher | None = None,
        timeout_ms: int = 45_000,
        capture_timeout_seconds: int = 120,
        login_timeout_seconds: int = 900,
        headless: bool = False,
    ) -> None:
        profile_root = Path(profile_directory).expanduser().resolve(strict=False)
        self.profile_directory = profile_root / "steering-managed-profile"
        self.guard = guard or NetworkGuard()
        self.link_resolver = link_resolver
        self.timeout_ms = timeout_ms
        self.capture_timeout_seconds = capture_timeout_seconds
        self.login_timeout_seconds = login_timeout_seconds
        self.headless = headless

    def has_stored_session(self) -> bool:
        """Report whether a previous sign-in left a session in the managed profile.

        This is a filesystem check on purpose. Automatic escalation asks it before
        every candidate post, so it must not cost a browser launch; a user who has
        never signed in should pay nothing for the feature existing.
        """

        if not self.profile_directory.is_dir():
            return False
        return any(
            (self.profile_directory / relative).exists()
            for relative in (
                Path("Default") / "Network" / "Cookies",
                Path("Default") / "Cookies",
                Path("Network") / "Cookies",
                Path("Cookies"),
            )
        )

    async def capture(self, url: str, *, authorized: bool = False) -> ResolvedSource:
        if not authorized:
            raise PermissionError("browser capture requires explicit user authorization")
        canonical = canonical_http_url(url)
        await self.guard.validate_url(canonical)
        playwright_module = self._playwright_module()

        self.profile_directory.mkdir(parents=True, exist_ok=True)
        try:
            async with playwright_module() as playwright:
                post = parse_x_post_url(canonical)
                is_x = (urlsplit(canonical).hostname or "").lower() in X_HOSTS
                if is_x and post is None:
                    raise BrowserCaptureUnavailable(
                        "signed-in X capture needs a post URL such as https://x.com/<author>/status/<id>"
                    )
                context = await self._launch_context(playwright, prefer_installed_chrome=is_x)
                try:
                    page = context.pages[0] if context.pages else await context.new_page()
                    self._set_default_timeout(page, 10_000)
                    validated = await self._install_guards(context, page)
                    peer_violations = self._install_response_guard(page)
                    if is_x and post is not None:
                        await page.goto(canonical, wait_until="domcontentloaded", timeout=self.timeout_ms)
                        await self._wait_for_x_thread(page, post.post_id)
                        self._raise_peer_violation(peer_violations)
                        payload = await self._capture_x_thread(page, post.post_id)
                        # The handle read from the thread canonicalizes the URL even
                        # when the user pasted an /i/status/ link, so this capture
                        # deduplicates against the public one for the same post.
                        final_url = post.canonical_url(str(payload.get("author") or "") or None)
                    else:
                        final_url = await self._navigate(page, canonical)
                        self._raise_peer_violation(peer_violations)
                        await page.wait_for_timeout(1_000)
                        self._raise_peer_violation(peer_violations)
                        payload = await self._capture_page(page)
                    payload["links"] = await self._resolve_links(payload.get("links", []), validated)
                finally:
                    await context.close()
        except BrowserCaptureUnavailable:
            raise
        except Exception as exc:
            LOGGER.warning("authorized browser capture failed for %s", canonical, exc_info=True)
            raise BrowserCaptureUnavailable(
                f"authorized browser capture failed ({type(exc).__name__})"
            ) from None
        return self._resolved(final_url, payload)

    async def open_login(
        self,
        url: str,
        *,
        authorized: bool = False,
        force: bool = False,
    ) -> LoginOutcome:
        """Open the isolated visible profile so the user can sign in themselves.

        Returns once sign-in is detected or the user closes the window. Closing
        the window is a normal way to finish, not a failure.

        When a session already exists this returns immediately without opening
        anything. Launching a browser that detects the existing session and shuts
        itself a second later looks like a crash, and shows no sign-in form
        because none was needed. Pass ``force`` to replace a stale session.
        """

        if not authorized:
            raise PermissionError("opening the managed browser requires explicit authorization")
        if not force and self.has_stored_session():
            LOGGER.info("managed profile already holds a session; skipping the sign-in window")
            return LoginOutcome.ALREADY_SIGNED_IN
        canonical = canonical_http_url(url)
        await self.guard.validate_url(canonical)
        playwright_module = self._playwright_module()
        self.profile_directory.mkdir(parents=True, exist_ok=True)
        try:
            async with playwright_module() as playwright:
                if (urlsplit(canonical).hostname or "").lower() in X_HOSTS:
                    return await self._open_x_login(playwright, canonical, force=force)
                executable = Path(playwright.chromium.executable_path)
                if not executable.is_file():
                    raise BrowserDependencyUnavailable("the managed Chromium runtime is not installed")
            # Authentication is intentionally not Playwright-controlled here. Some
            # providers reject valid credentials in automation-controlled login
            # flows. Launching Chromium directly keeps this step entirely
            # human-driven while preserving the same isolated profile that
            # authorized capture uses later.
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
            return LoginOutcome.SIGNED_IN
        except BrowserCaptureUnavailable:
            raise
        except Exception as exc:
            LOGGER.warning("managed browser login failed for %s", canonical, exc_info=True)
            raise BrowserCaptureUnavailable(f"managed browser login failed ({type(exc).__name__})") from None

    @staticmethod
    def _playwright_module() -> Any:
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise BrowserDependencyUnavailable(
                "browser capture needs the 'browser' extra and an installed Chromium runtime"
            ) from exc
        return async_playwright

    @staticmethod
    def _set_default_timeout(page: Any, timeout_ms: int) -> None:
        set_default_timeout = getattr(page, "set_default_timeout", None)
        if callable(set_default_timeout):
            set_default_timeout(timeout_ms)

    async def _open_x_login(
        self,
        playwright: Any,
        canonical: str,
        *,
        force: bool = False,
    ) -> LoginOutcome:
        """Drive X sign-in in the persistent real-Chrome profile capture reuses."""

        # A human has to type credentials, so sign-in is never headless even when
        # captures are configured to be.
        context = await self._launch_context(playwright, prefer_installed_chrome=True, headless=False)
        try:
            page = context.pages[0] if context.pages else await context.new_page()
            self._set_default_timeout(page, 10_000)
            if force:
                # Asking to sign in again must actually produce a sign-in form.
                # Keeping the old cookies would make the very next check report
                # "already signed in" and close the window before the user can
                # type, which is indistinguishable from a crash.
                await self._clear_session(context)
            await page.goto(canonical, wait_until="domcontentloaded", timeout=self.timeout_ms)
            deadline = asyncio.get_running_loop().time() + self.login_timeout_seconds
            while asyncio.get_running_loop().time() < deadline:
                if page.is_closed():
                    LOGGER.info("managed X login window closed before sign-in was detected")
                    return LoginOutcome.CLOSED_BEFORE_SIGN_IN
                try:
                    if await self._x_logged_in(page):
                        LOGGER.info("managed X sign-in detected; session saved to the isolated profile")
                        return LoginOutcome.SIGNED_IN
                    await self._dismiss_x_cookie_prompt(page)
                except Exception:
                    # The user closing the window mid-poll surfaces as a driver
                    # error rather than a clean `is_closed`; that is a normal end.
                    LOGGER.info("managed X login window closed during polling")
                    return LoginOutcome.CLOSED_BEFORE_SIGN_IN
                await page.wait_for_timeout(1_000)
            raise BrowserCaptureUnavailable(
                f"X login did not complete within {self.login_timeout_seconds} seconds"
            )
        finally:
            await context.close()

    @staticmethod
    async def _clear_session(context: Any) -> None:
        """Discard the stored session so X presents its sign-in form again."""

        try:
            await context.clear_cookies()
            LOGGER.info("cleared the managed profile session before signing in again")
        except Exception:
            LOGGER.warning("could not clear the managed profile session", exc_info=True)

    async def _launch_context(
        self,
        playwright: Any,
        *,
        prefer_installed_chrome: bool,
        headless: bool | None = None,
    ) -> Any:
        """Launch the one persistent isolated profile used for login and capture.

        Signing in is always visible because a human types the credentials, but a
        capture only replays the session that sign-in already stored. Once that
        session exists, capture can run headless, which is what makes unattended
        overnight batches possible without an API key.
        """

        run_headless = self.headless if headless is None else headless
        options: dict[str, Any] = {
            "headless": run_headless,
            "locale": "en-US",
            "args": ["--disable-blink-features=AutomationControlled"],
        }
        if run_headless:
            # Headless has no window to maximize, and the thread only renders
            # (and only scrolls) inside a viewport with real height.
            options["viewport"] = {"width": 1440, "height": 1000}
        else:
            options["no_viewport"] = True
            options["args"].append("--start-maximized")
        if prefer_installed_chrome:
            try:
                return await playwright.chromium.launch_persistent_context(
                    str(self.profile_directory), channel="chrome", **options
                )
            except Exception:
                LOGGER.info("installed Google Chrome unavailable; falling back to managed Chromium")
        executable = Path(playwright.chromium.executable_path)
        if not executable.is_file():
            raise BrowserDependencyUnavailable("the managed Chromium runtime is not installed")
        try:
            return await playwright.chromium.launch_persistent_context(str(self.profile_directory), **options)
        except Exception as exc:
            raise BrowserCaptureUnavailable(
                f"the managed browser could not launch ({type(exc).__name__}); "
                "close any other STEERING browser window and retry"
            ) from None

    async def _install_guards(self, context: Any, page: Any) -> _ValidatedHosts:
        """Apply the same network guarantees to every capture, X included."""

        del page
        validated = _ValidatedHosts(self.guard)

        async def guard_request(route: Any) -> None:
            request_url = str(route.request.url)
            scheme = urlsplit(request_url).scheme.lower()
            if scheme in {"about", "blob", "data"}:
                await route.continue_()
                return
            if scheme in {"http", "https"} and await validated.allows(request_url):
                await route.continue_()
                return
            await route.abort(error_code="blockedbyclient")

        await context.route("**/*", guard_request)
        return validated

    @staticmethod
    async def _x_logged_in(page: Any) -> bool:
        selectors = (
            '[data-testid="AppTabBar_Home_Link"]',
            '[data-testid="SideNav_NewTweet_Button"]',
            'a[href="/home"]',
        )
        for selector in selectors:
            if await page.locator(selector).count():
                return True
        return False

    @staticmethod
    async def _dismiss_x_cookie_prompt(page: Any) -> None:
        for label in ("Accept all cookies", "Refuse non-essential cookies"):
            button = page.get_by_role("button", name=label)
            if await button.count():
                try:
                    await button.first.click(timeout=2_000)
                except Exception:
                    LOGGER.debug("X cookie prompt could not be dismissed")
                return

    async def _wait_for_x_thread(self, page: Any, root_post_id: str) -> None:
        """Wait only for rendering; never hold a request open for a human to sign in."""

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.capture_timeout_seconds
        auth_deadline = loop.time() + min(_AUTH_GRACE_SECONDS, self.capture_timeout_seconds)
        signed_in_seen = False
        while loop.time() < deadline:
            if page.is_closed():
                raise BrowserCaptureUnavailable("the X capture window closed before capture completed")
            await self._dismiss_x_cookie_prompt(page)
            if await self._x_thread_visible(page, root_post_id):
                return
            signed_in_seen = signed_in_seen or await self._x_logged_in(page)
            if not signed_in_seen and loop.time() >= auth_deadline:
                raise self._authentication_required()
            await page.wait_for_timeout(1_000)
        if not signed_in_seen:
            # Never having seen a session is a more accurate diagnosis than a
            # render timeout, and it points at the action that actually fixes it.
            raise self._authentication_required()
        raise BrowserCaptureUnavailable(
            f"the X post did not render within {self.capture_timeout_seconds} seconds"
        )

    @staticmethod
    def _authentication_required() -> BrowserAuthenticationRequired:
        return BrowserAuthenticationRequired(
            "the managed browser is not signed in to X; use 'Prepare signed-in browser capture' "
            "to sign in once, then retry this capture"
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
        payload = cast(
            dict[str, Any],
            await page.evaluate(
                """() => ({
                title: document.title || location.hostname,
                text: (document.querySelector('main') || document.body)?.innerText || '',
                links: [...document.querySelectorAll('main a[href], article a[href]')]
                    .map(a => ({url: a.href, text: (a.innerText || '').trim()})).filter(l => l.url),
                media: [...document.querySelectorAll('main img[src], article img[src]')]
                    .map(i => ({url: i.currentSrc || i.src, alt: i.alt || ''})),
                author: document.querySelector('[rel="author"]')?.textContent?.trim() || null
            })"""
            ),
        )
        return payload

    async def _capture_x_thread(self, page: Any, root_post_id: str) -> dict[str, Any]:
        seen: dict[str, dict[str, Any]] = {}
        encounter_order: list[str] = []

        await self._settle_thread_head(page, seen, encounter_order)

        stagnant_scrolls = 0
        for _ in range(_X_THREAD_MAX_SCROLLS):
            if page.is_closed():
                raise BrowserCaptureUnavailable("the X capture window closed before capture completed")
            await page.evaluate("window.scrollBy(0, Math.max(window.innerHeight * 0.85, 700))")
            await page.wait_for_timeout(1_250)
            before = self._thread_signature(seen)
            self._absorb(await page.evaluate(_X_ARTICLES_SCRIPT), seen, encounter_order)
            # A later sighting of the same post can carry links the first did not,
            # so growth is measured over links too, not just post count.
            stagnant_scrolls = stagnant_scrolls + 1 if self._thread_signature(seen) == before else 0
            if stagnant_scrolls >= _X_STAGNANT_SCROLL_LIMIT:
                break

        root = seen.get(root_post_id)
        if root is None:
            raise BrowserCaptureUnavailable("the root X post was not captured from the visible thread")
        root_handle = str(root.get("handle") or "").strip()
        if not root_handle:
            raise BrowserCaptureUnavailable("the root X post author could not be identified")

        retained = self._filter_x_thread_window(
            [
                root,
                *[
                    seen[post_id]
                    for post_id in encounter_order
                    if seen[post_id] is not root
                    and str(seen[post_id].get("handle") or "").lower() == root_handle.lower()
                ],
            ],
            root,
        )
        LOGGER.info(
            "captured X thread %s: %d post(s) by @%s",
            root_post_id,
            len(retained),
            root_handle,
        )
        return {
            "title": f"{str(root.get('author_label') or '').strip() or '@' + root_handle} — X thread",
            "text": "\n\n---\n\n".join(item for item in map(self._x_post_text, retained) if item),
            "links": [link for post in retained for link in self._post_links(post)],
            "media": self._x_media(retained),
            "author": root_handle,
            "published_at": root.get("posted_at"),
            "bundled_self_replies": max(0, len(retained) - 1),
        }

    def _absorb(
        self,
        batch: Any,
        seen: dict[str, dict[str, Any]],
        encounter_order: list[str],
    ) -> None:
        for item in batch if isinstance(batch, list) else []:
            if not isinstance(item, dict) or not item.get("post_id"):
                continue
            post_id = str(item["post_id"])
            if post_id not in seen:
                seen[post_id] = item
                encounter_order.append(post_id)
            else:
                seen[post_id] = self._richer_sighting(seen[post_id], item)

    async def _settle_thread_head(
        self,
        page: Any,
        seen: dict[str, dict[str, Any]],
        encounter_order: list[str],
    ) -> None:
        """Let the top of the thread finish rendering before scrolling past it.

        Link-preview cards attach seconds after an article's text, and scrolling
        evicts the article from X's virtualized DOM before that happens. Reading
        the same articles in place until they stop gaining links is what makes a
        link the author put in a reply survive the capture.
        """

        loop = asyncio.get_running_loop()
        deadline = loop.time() + _X_SETTLE_SECONDS
        while True:
            if page.is_closed():
                raise BrowserCaptureUnavailable("the X capture window closed before capture completed")
            self._absorb(await page.evaluate(_X_ARTICLES_SCRIPT), seen, encounter_order)
            if loop.time() >= deadline:
                return
            await page.wait_for_timeout(_X_SETTLE_INTERVAL_MS)

    @staticmethod
    def _thread_signature(seen: dict[str, dict[str, Any]]) -> tuple[int, int]:
        """Summarize progress as (posts, links) so late-arriving links count as growth."""

        links = sum(len(post.get("links") or ()) for post in seen.values())
        return len(seen), links

    @staticmethod
    def _richer_sighting(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
        """Keep the fuller version of a post seen more than once.

        A post first appears with its text and gains its link-preview card a few
        seconds later. Treating the first sighting as final would discard exactly
        the link the author put in the reply.
        """

        merged = dict(existing)
        for field in ("links", "media"):
            if len(incoming.get(field) or ()) > len(existing.get(field) or ()):
                merged[field] = incoming[field]
        for field in ("text", "quoted_text", "author_label", "posted_at"):
            if len(str(incoming.get(field) or "")) > len(str(existing.get(field) or "")):
                merged[field] = incoming[field]
        return merged

    @staticmethod
    def _post_links(post: dict[str, Any]) -> list[dict[str, Any]]:
        raw = post.get("links")
        return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []

    @staticmethod
    def _x_post_text(post: dict[str, Any]) -> str:
        body = str(post.get("text") or "").strip()
        quoted = str(post.get("quoted_text") or "").strip()
        if quoted:
            body = f"{body}\n\n[Quoted post]\n{quoted}".strip()
        return body

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

    async def _resolve_links(
        self,
        links: list[Any],
        validated: _ValidatedHosts,
    ) -> list[str]:
        """Turn captured anchors into external destinations, unwrapping shortlinks."""

        outbound: list[str] = []
        # X renders one destination as several anchors (the card, its caption, the
        # "From host" label). Resolving each would unwrap the same shortlink over
        # and over, costing a network round trip per duplicate.
        resolved_by_raw: dict[str, str | None] = {}
        for link in links:
            item = link if isinstance(link, dict) else {"url": str(link)}
            raw = str(item.get("url") or "")
            if raw in resolved_by_raw:
                resolved = resolved_by_raw[raw]
            else:
                resolved = await self._resolve_link(item)
                resolved_by_raw[raw] = resolved
            if not resolved or resolved in outbound:
                continue
            if not await validated.allows(resolved):
                LOGGER.info("captured link %s did not pass network validation; dropped", resolved)
                continue
            outbound.append(resolved)
        return outbound

    async def _resolve_link(self, link: dict[str, Any]) -> str | None:
        raw_url = str(link.get("url") or "").strip()
        parts = urlsplit(raw_url)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            return None
        if is_x_owned_url(raw_url) and (parts.hostname or "").lower() != "t.co":
            return None

        destination = raw_url
        if (parts.hostname or "").lower() == "t.co":
            displayed = next(
                (
                    candidate
                    for value in (link.get("text"), link.get("title"), link.get("aria_label"))
                    if (candidate := self._displayed_external_url(value)) is not None
                ),
                None,
            )
            destination = displayed or await self._unwrap_shortlink(raw_url) or ""
        if not destination or is_x_owned_url(destination):
            return None
        return canonical_http_url(destination)

    async def _unwrap_shortlink(self, url: str) -> str | None:
        """Unwrap a t.co link through the guarded fetcher, not the browser.

        The guarded fetcher validates every redirect hop and never downloads the
        destination body, so unwrapping stays cheap and keeps the same network
        guarantees the rest of ingestion relies on.
        """

        if self.link_resolver is None:
            return None
        try:
            return await self.link_resolver.resolve_redirects(url)
        except (SourceUnavailableError, UnsafeSourceError) as exc:
            LOGGER.info("X shortlink %s could not be unwrapped (%s)", url, exc)
            return None

    @staticmethod
    def _displayed_external_url(value: Any) -> str | None:
        """Read the destination X renders as anchor text, which omits the scheme.

        Only a value that is a bare URL on its own is accepted. X also renders
        prose around a host, such as "From huggingface.co" or the card's aria
        label "huggingface.co microsoft/Mage-VL - Hugging Face". Collapsing the
        spaces in those manufactures a hostname that does not exist
        ("fromhuggingface.co", "huggingface.comicrosoft"), which then fails DNS
        validation and silently loses the author's link. Interior whitespace is
        therefore a rejection, not something to strip.
        """

        if not isinstance(value, str):
            return None
        candidate = value.strip()
        if not candidate or "…" in candidate or re.search(r"\s", candidate):
            return None
        if not candidate.startswith(("http://", "https://")):
            candidate = f"https://{candidate}"
        parts = urlsplit(candidate)
        host = (parts.hostname or "").lower()
        if not host or "." not in host or is_x_owned_url(candidate):
            return None
        return candidate

    @staticmethod
    def _resolved(canonical: str, payload: dict[str, Any]) -> ResolvedSource:
        host = (urlsplit(canonical).hostname or "").lower()
        source_kind = (
            SourceKind.X
            if host in X_HOSTS
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
            and len(str(item.get("alt", "")).strip()) >= _MIN_DESCRIPTIVE_ALT_CHARS
            and not any(hint in str(item.get("alt", "")).lower() for hint in _DECORATIVE_ALT_HINTS)
        ]
        text = str(payload.get("text") or "").strip()
        return ResolvedSource(
            canonical_url=canonical,
            source_kind=source_kind,
            title=str(payload.get("title") or canonical),
            text=text,
            author=str(payload["author"]) if payload.get("author") else None,
            published_at=ManagedBrowserCapture._parse_x_timestamp(payload.get("published_at")),
            mime_type="text/html",
            extraction_method=BROWSER_METHOD,
            partial=not text,
            outbound_urls=links,
            media_urls=media_urls,
            metadata={
                "capture_scope": AUTHOR_THREAD if source_kind is SourceKind.X else ROOT_POST_ONLY,
                "bundled_self_replies": int(payload.get("bundled_self_replies", 0)),
                "media_inclusion_candidates": useful_media,
                "media_decision": (
                    "consider_alt_text_candidates_only_when_no_stronger_linked_source_exists"
                    if useful_media
                    else "skip_no_inexpensive_technical_signal"
                ),
            },
        )
