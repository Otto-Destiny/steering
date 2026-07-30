"""Run the X article-extraction script against an X-shaped DOM in real Chromium.

The script is JavaScript that only ever runs inside a browser, so a Python double
cannot check it. This exercises it against markup mirroring X's post-detail page,
including the two structures that previously broke extraction: a quoted post
nested in a `div[role="link"]` card carrying its own `<time>` and status anchor,
and a focal post whose own timestamp appears *after* that card in DOM order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from steering.ingestion.browser import _X_ARTICLES_SCRIPT

pytestmark = pytest.mark.e2e

ROOT_POST_ID = "1885026028428681698"
QUOTED_POST_ID = "1770000000000000001"
REPLY_POST_ID = "1885026028428681699"

# A quote-tweet post detail view: the quoted card renders *before* the focal
# post's own timestamp, which is exactly why selecting the first <time> in the
# article attributed the post to the quoted author.
QUOTE_TWEET_ARTICLE = f"""
<article data-testid="tweet">
  <div data-testid="User-Name"><span>A Researcher</span><span>@researcher</span></div>
  <div data-testid="tweetText">This is the result I have been waiting for.
    <a href="https://t.co/SHORTPAPER">arxiv.org/abs/2501.1294…</a>
  </div>
  <div role="link" tabindex="0">
    <div data-testid="User-Name"><span>Lab Account</span><span>@labaccount</span></div>
    <a href="/labaccount/status/{QUOTED_POST_ID}">
      <time datetime="2024-03-19T09:00:00.000Z">Mar 19, 2024</time>
    </a>
    <div data-testid="tweetText">Releasing our paper today.
      <a href="https://t.co/QUOTEDLINK">example.org/quoted</a>
    </div>
    <img src="https://pbs.twimg.com/media/quoted.jpg" alt="Quoted media">
  </div>
  <a href="/researcher/status/{ROOT_POST_ID}">
    <time datetime="2025-01-30T18:03:21.000Z">6:03 PM &middot; Jan 30, 2025</time>
  </a>
  <img src="https://pbs.twimg.com/media/root.jpg" alt="Benchmark table comparing four systems">
</article>
"""

# A plain self-reply card, where the timestamp sits in the header instead.
SELF_REPLY_ARTICLE = f"""
<article data-testid="tweet">
  <div data-testid="User-Name">
    <span>A Researcher</span><span>@researcher</span>
    <a href="/researcher/status/{REPLY_POST_ID}">
      <time datetime="2025-01-30T18:20:00.000Z">Jan 30, 2025</time>
    </a>
  </div>
  <div data-testid="tweetText">Follow-up detail in the same thread.</div>
</article>
"""


@pytest.fixture(scope="module")
def articles() -> list[dict[str, Any]]:
    playwright_api = pytest.importorskip("playwright.sync_api")
    with playwright_api.sync_playwright() as runtime:
        if not Path(runtime.chromium.executable_path).is_file():
            pytest.skip("the managed Chromium runtime is not installed")
        browser = runtime.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            # A real origin so anchor `href` values resolve to x.com absolute URLs.
            page.route(
                "https://x.com/**",
                lambda route: route.fulfill(
                    # X serves UTF-8; without the charset Chromium falls back to
                    # windows-1252 and the ellipsis in display text becomes mojibake.
                    content_type="text/html; charset=utf-8",
                    body=f"<html><body>{QUOTE_TWEET_ARTICLE}{SELF_REPLY_ARTICLE}</body></html>",
                ),
            )
            page.goto(f"https://x.com/researcher/status/{ROOT_POST_ID}")
            return list(page.evaluate(_X_ARTICLES_SCRIPT))
        finally:
            browser.close()


def test_the_script_finds_both_posts(articles: list[dict[str, Any]]) -> None:
    assert [item["post_id"] for item in articles] == [ROOT_POST_ID, REPLY_POST_ID]


def test_a_quoted_post_does_not_steal_the_articles_identity(articles: list[dict[str, Any]]) -> None:
    """The regression: the quoted card's <time> came first and hijacked attribution."""

    root = articles[0]

    assert root["post_id"] == ROOT_POST_ID
    assert root["handle"] == "researcher"
    assert root["posted_at"] == "2025-01-30T18:03:21.000Z"


def test_quoted_text_is_kept_separate_from_the_authors_own_words(
    articles: list[dict[str, Any]],
) -> None:
    root = articles[0]

    assert "This is the result I have been waiting for." in root["text"]
    assert "Releasing our paper today." not in root["text"]
    assert "Releasing our paper today." in root["quoted_text"]


def test_only_the_authors_own_links_are_collected(articles: list[dict[str, Any]]) -> None:
    root = articles[0]
    urls = [link["url"] for link in root["links"]]

    assert "https://t.co/SHORTPAPER" in urls
    # A link that belongs to the quoted post is not this author's outbound link.
    assert "https://t.co/QUOTEDLINK" not in urls


def test_scheme_less_display_text_is_captured_for_shortlink_unwrapping(
    articles: list[dict[str, Any]],
) -> None:
    root = articles[0]
    shortlink = next(link for link in root["links"] if link["url"] == "https://t.co/SHORTPAPER")

    assert shortlink["text"] == "arxiv.org/abs/2501.1294…"


def test_header_timestamps_still_identify_ordinary_cards(articles: list[dict[str, Any]]) -> None:
    reply = articles[1]

    assert reply["post_id"] == REPLY_POST_ID
    assert reply["handle"] == "researcher"
    assert reply["quoted_text"] == ""
