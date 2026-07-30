"""Capture an X post and its author's self-reply thread in a visible browser.

This is the corpus-building tool for `evaluate/`, not the product code path.
Production capture lives in `steering.ingestion.browser`, and shortlink unwrapping
lives in `steering.ingestion.x`. Keep behavioural fixes in those modules: the two
implementations have diverged once already, and a redirect-handling change made
here was ported into the product in a form that could never work against the real
Playwright API.

The browser profile remains local and must never be committed. The capture stores
post text, links, and media URLs; it does not download images, videos, or papers.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

try:
    from playwright.sync_api import BrowserContext, Page, Playwright, sync_playwright
except ImportError as exc:  # pragma: no cover - actionable CLI error
    raise SystemExit("Playwright is not installed. Run: uv sync --extra browser") from exc


STATUS_URL = re.compile(r"https?://(?:www\.)?x\.com/([^/?#]+)/status/(\d+)", re.IGNORECASE)
INTERNAL_X_HOSTS = {"x.com", "www.x.com", "twitter.com", "www.twitter.com"}
MEDIA_HOSTS = {"pbs.twimg.com", "video.twimg.com"}
DEFAULT_THREAD_SPAN_HOURS = 48
LOGGER = logging.getLogger(__name__)


def parse_status_url(url: str) -> tuple[str, str]:
    match = STATUS_URL.search(url)
    if not match:
        raise ValueError(f"Expected a canonical X status URL, received: {url}")
    return match.group(1), match.group(2)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def update_status(path: Path, state: str, **details: object) -> None:
    write_json(
        path,
        {
            "state": state,
            "updated_at_unix": int(time.time()),
            **details,
        },
    )


def launch_context(
    playwright: Playwright,
    profile_dir: Path,
    *,
    headless: bool,
) -> BrowserContext:
    profile_dir.mkdir(parents=True, exist_ok=True)
    options = {
        "user_data_dir": str(profile_dir.resolve()),
        "headless": headless,
        "viewport": {"width": 1440, "height": 1000},
        "locale": "en-US",
        "args": ["--disable-blink-features=AutomationControlled"],
    }
    try:
        return playwright.chromium.launch_persistent_context(channel="chrome", **options)
    except Exception:
        try:
            return playwright.chromium.launch_persistent_context(**options)
        except Exception as chromium_error:
            raise RuntimeError(
                "Could not launch Google Chrome or Playwright Chromium. "
                "Install Chrome, or run: python -m playwright install chromium"
            ) from chromium_error


def dismiss_cookie_prompt(page: Page) -> None:
    for label in ("Accept all cookies", "Refuse non-essential cookies"):
        button = page.get_by_role("button", name=label)
        if button.count():
            try:
                button.first.click(timeout=2_000)
                return
            except Exception as exc:
                LOGGER.debug("cookie prompt could not be dismissed: %s", type(exc).__name__)


def clear_browser_cache(context: BrowserContext, page: Page) -> None:
    """Discard fetched page/media bytes while preserving the login cookies."""

    try:
        session = context.new_cdp_session(page)
        session.send("Network.clearBrowserCache")
        session.detach()
    except Exception as exc:
        # Cache cleanup must not destroy an otherwise valid capture.
        LOGGER.debug("browser cache cleanup failed: %s", type(exc).__name__)


def thread_is_visible(page: Page, root_post_id: str) -> bool:
    return page.locator(f'a[href*="/status/{root_post_id}"] time').count() > 0


def login_is_visible(page: Page) -> bool:
    return "/login" in page.url or "/i/flow/login" in page.url


def logged_in(page: Page) -> bool:
    selectors = (
        '[data-testid="AppTabBar_Home_Link"]',
        '[data-testid="SideNav_NewTweet_Button"]',
        'a[href="/home"]',
    )
    return any(page.locator(selector).count() for selector in selectors)


def wait_for_thread(
    page: Page,
    target_url: str,
    root_post_id: str,
    status_path: Path,
    timeout_seconds: int,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_navigation = 0.0
    login_navigation_attempted = False
    while time.monotonic() < deadline:
        dismiss_cookie_prompt(page)
        session_is_logged_in = logged_in(page)
        if session_is_logged_in and thread_is_visible(page, root_post_id):
            update_status(status_path, "capturing", url=target_url)
            return

        if session_is_logged_in and not login_is_visible(page):
            now = time.monotonic()
            if now - last_navigation > 10:
                page.goto(target_url, wait_until="domcontentloaded", timeout=60_000)
                last_navigation = now
        else:
            update_status(
                status_path,
                "awaiting_login",
                message="Sign in to X in the visible Chrome window; capture will continue automatically.",
                url=target_url,
            )
            if not login_is_visible(page) and not login_navigation_attempted:
                page.goto(
                    "https://x.com/i/flow/login",
                    wait_until="domcontentloaded",
                    timeout=60_000,
                )
                login_navigation_attempted = True
        page.wait_for_timeout(1_000)

    raise TimeoutError(f"The X thread did not become available within {timeout_seconds} seconds.")


def extract_article(article: object) -> dict[str, object] | None:
    result = article.evaluate(
        """element => {
            const time = element.querySelector('time');
            const statusAnchor = time ? time.closest('a[href*="/status/"]') : null;
            const statusUrl = statusAnchor ? statusAnchor.href : null;
            if (!statusUrl) return null;

            const match = statusUrl.match(/x\\.com\\/([^/]+)\\/status\\/(\\d+)/i);
            if (!match) return null;

            const textNode = element.querySelector('[data-testid="tweetText"]');
            const userNameNode = element.querySelector('[data-testid="User-Name"]');
            const links = Array.from(element.querySelectorAll('a[href]')).map(anchor => ({
                url: anchor.href,
                text: (anchor.innerText || anchor.textContent || '').trim(),
                title: anchor.getAttribute('title'),
                aria_label: anchor.getAttribute('aria-label'),
            }));
            const images = Array.from(
                element.querySelectorAll('img[src*="pbs.twimg.com/media"]')
            ).map(image => ({
                url: image.src,
                alt: image.alt || null,
            }));
            const videoPosters = Array.from(element.querySelectorAll('video[poster]')).map(video => ({
                url: video.poster,
                alt: null,
            }));

            return {
                post_id: match[2],
                handle: match[1],
                status_url: statusUrl.split('?')[0],
                posted_at: time ? time.getAttribute('datetime') : null,
                author_label: userNameNode ? userNameNode.innerText.trim() : null,
                text: textNode ? textNode.innerText.trim() : '',
                links,
                media: [...images, ...videoPosters],
            };
        }"""
    )
    return result


def capture_visible_articles(page: Page) -> list[dict[str, object]]:
    captured: list[dict[str, object]] = []
    for article in page.locator("article").all():
        try:
            item = extract_article(article)
        except Exception as exc:
            LOGGER.debug("visible article extraction failed: %s", type(exc).__name__)
            continue
        if item:
            captured.append(item)
    return captured


def displayed_external_url(link_text: object) -> str | None:
    if not isinstance(link_text, str):
        return None
    candidate = re.sub(r"\s+", "", link_text).rstrip("…")
    if "…" in link_text or not candidate.startswith(("http://", "https://")):
        return None
    if is_external_material(candidate):
        return candidate
    return None


def resolve_url(context: BrowserContext, url: str, link_text: object = None) -> str:
    if urlsplit(url).scheme not in {"http", "https"}:
        return url
    if urlsplit(url).netloc.lower() != "t.co":
        return url

    displayed_url = displayed_external_url(link_text)
    if displayed_url:
        return displayed_url

    try:
        response = context.request.get(
            url,
            fail_on_status_code=False,
            max_redirects=10,
            timeout=20_000,
        )
        if response.url != url:
            return response.url
    except Exception as exc:
        LOGGER.debug("Playwright redirect resolution failed: %s", type(exc).__name__)

    try:
        request = Request(  # noqa: S310 - scheme is restricted above
            url,
            method="HEAD",
            headers={"User-Agent": "Mozilla/5.0 STEERING-link-resolver/0.1"},
        )
        with urlopen(request, timeout=20) as response:  # noqa: S310 - HTTP(S) only
            resolved_url = response.geturl()
            if resolved_url != url:
                return resolved_url
    except Exception as exc:
        LOGGER.debug("HEAD redirect resolution failed: %s", type(exc).__name__)

    resolver_page = context.new_page()
    try:
        resolver_page.goto(url, wait_until="commit", timeout=20_000)
        resolver_page.wait_for_timeout(500)
        return resolver_page.url
    except Exception:
        return url
    finally:
        resolver_page.close()


def is_external_material(url: str) -> bool:
    parts = urlsplit(url)
    host = parts.netloc.lower()
    if not parts.scheme.startswith("http") or not host:
        return False
    return host not in INTERNAL_X_HOSTS and host not in MEDIA_HOSTS


def is_linked_social_status(url: str) -> bool:
    return STATUS_URL.search(url) is not None


def classify_source(url: str) -> tuple[str, int]:
    parts = urlsplit(url)
    host = parts.netloc.lower()
    if (
        host == "arxiv.org"
        or host.endswith(".arxiv.org")
        or host == "openreview.net"
        or host.endswith(".openreview.net")
        or host == "aclanthology.org"
        or host.endswith(".aclanthology.org")
        or host in {"doi.org", "dx.doi.org", "pubmed.ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov"}
        or parts.path.lower().endswith(".pdf")
    ):
        return "paper", 100
    if host == "github.com" or host.endswith(".github.com"):
        return "repository", 95
    if (
        host == "huggingface.co"
        or host.endswith(".huggingface.co")
        or host == "modelscope.ai"
        or host.endswith(".modelscope.ai")
    ):
        return "model_or_dataset", 90
    if host.endswith("readthedocs.io") or "docs." in host:
        return "documentation", 85
    if host.endswith("substack.com"):
        return "author_explanation", 70
    return "web_source", 60


def parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def filter_thread_window(
    posts: list[dict[str, object]],
    root_post_id: str,
    max_span_hours: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    root = next(post for post in posts if str(post["post_id"]) == root_post_id)
    root_time = parse_timestamp(root.get("posted_at"))
    if root_time is None:
        return posts, []

    kept: list[dict[str, object]] = []
    discarded: list[dict[str, object]] = []
    max_seconds = max_span_hours * 60 * 60
    for post in posts:
        post_time = parse_timestamp(post.get("posted_at"))
        if post_time is None or abs((post_time - root_time).total_seconds()) <= max_seconds:
            kept.append(post)
            continue
        discarded.append(
            {
                "post_id": post["post_id"],
                "status_url": post["status_url"],
                "reason": f"outside_{max_span_hours}h_root_thread_window",
            }
        )
    return kept, discarded


def build_capture(
    context: BrowserContext,
    page: Page,
    *,
    candidate_id: str,
    target_url: str,
    root_handle: str,
    root_post_id: str,
    max_scrolls: int,
    max_thread_span_hours: int,
) -> dict[str, object]:
    seen: dict[str, dict[str, object]] = {}
    encounter_order: list[str] = []
    stagnant_scrolls = 0

    for _ in range(max_scrolls):
        before = len(seen)
        for item in capture_visible_articles(page):
            if str(item["handle"]).lower() != root_handle.lower():
                continue
            post_id = str(item["post_id"])
            if post_id not in seen:
                seen[post_id] = item
                encounter_order.append(post_id)

        stagnant_scrolls = stagnant_scrolls + 1 if len(seen) == before else 0
        if stagnant_scrolls >= 4:
            break
        page.evaluate("window.scrollBy(0, Math.max(window.innerHeight * 0.85, 700))")
        page.wait_for_timeout(1_250)

    posts = [seen[post_id] for post_id in encounter_order]
    if root_post_id not in seen:
        raise RuntimeError("The root post was not captured from the visible thread.")
    raw_same_author_post_count = len(posts)
    posts, discarded_posts = filter_thread_window(
        posts,
        root_post_id,
        max_thread_span_hours,
    )

    outbound_by_url: dict[str, dict[str, object]] = {}
    linked_social_by_url: dict[str, dict[str, object]] = {}
    retained_post_ids = {str(post["post_id"]) for post in posts}
    for post in posts:
        for link in post.pop("links", []):
            raw_url = str(link.get("url") or "")
            resolved_url = resolve_url(context, raw_url, link.get("text"))
            status_match = STATUS_URL.search(resolved_url)
            if status_match:
                linked_post_id = status_match.group(2)
                if linked_post_id not in retained_post_ids and resolved_url not in linked_social_by_url:
                    linked_social_by_url[resolved_url] = {
                        "url": resolved_url,
                        "artifact_type": "linked_social_post",
                        "found_in_post_id": post["post_id"],
                        "link_text": link.get("text") or None,
                    }
                continue
            if not is_external_material(resolved_url):
                continue
            if resolved_url not in outbound_by_url:
                artifact_type, priority = classify_source(resolved_url)
                outbound_by_url[resolved_url] = {
                    "url": resolved_url,
                    "artifact_type": artifact_type,
                    "priority": priority,
                    "found_in_post_id": post["post_id"],
                    "link_text": link.get("text") or None,
                }

    ranked_sources = sorted(
        outbound_by_url.values(),
        key=lambda source: (-int(source["priority"]), str(source["url"])),
    )
    for source in ranked_sources:
        source.pop("priority", None)
    linked_social_sources = list(linked_social_by_url.values())

    return {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "capture_method": "logged_in_visible_playwright",
        "root_url": target_url,
        "root_author_handle": root_handle,
        "root_post_id": root_post_id,
        "capture_policy": {
            "self_replies_bundled": True,
            "other_authors_excluded": True,
            "media_downloaded": False,
            "documents_downloaded": False,
            "max_thread_span_hours": max_thread_span_hours,
        },
        "thread": {
            "post_count": len(posts),
            "raw_same_author_post_count": raw_same_author_post_count,
            "posts": posts,
            "discarded_same_author_posts": discarded_posts,
        },
        "source_selection": {
            "selected_source": ranked_sources[0] if ranked_sources else None,
            "ranked_sources": ranked_sources,
            "linked_social_sources": linked_social_sources,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="Canonical X status URL")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path("evaluate/.browser-profile/x"),
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        default=Path("evaluate/.capture-status.json"),
    )
    parser.add_argument("--login-timeout", type=int, default=900)
    parser.add_argument("--max-scrolls", type=int, default=30)
    parser.add_argument(
        "--max-thread-span-hours",
        type=int,
        default=DEFAULT_THREAD_SPAN_HOURS,
    )
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root_handle, root_post_id = parse_status_url(args.url)
    update_status(args.status_file, "launching", url=args.url)

    try:
        with sync_playwright() as playwright:
            context = launch_context(
                playwright,
                args.profile_dir,
                headless=args.headless,
            )
            try:
                page = context.pages[0] if context.pages else context.new_page()
                page.set_default_timeout(10_000)
                page.goto(args.url, wait_until="domcontentloaded", timeout=60_000)
                wait_for_thread(
                    page,
                    args.url,
                    root_post_id,
                    args.status_file,
                    args.login_timeout,
                )
                capture = build_capture(
                    context,
                    page,
                    candidate_id=args.candidate_id,
                    target_url=args.url,
                    root_handle=root_handle,
                    root_post_id=root_post_id,
                    max_scrolls=args.max_scrolls,
                    max_thread_span_hours=args.max_thread_span_hours,
                )
                write_json(args.output, capture)
                update_status(
                    args.status_file,
                    "complete",
                    output=str(args.output),
                    post_count=capture["thread"]["post_count"],
                )
                clear_browser_cache(context, page)
                page.wait_for_timeout(2_000)
            finally:
                context.close()
    except Exception as exc:
        update_status(args.status_file, "failed", error=str(exc))
        print(f"Capture failed: {exc}", file=sys.stderr)
        return 1

    print(f"Capture written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
