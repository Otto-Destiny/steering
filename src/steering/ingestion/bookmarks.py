"""Import a list of saved X posts exported by the user's own browser.

X does not publish bookmarks without an authenticated API call, but the user can
read their own bookmarks page themselves. A bookmarklet run by hand collects the
post links there and writes a file, which this module turns into ingestible
sources.

That keeps the whole path under the user's control: their browser, their
session, an action they invoke deliberately. No process drives a signed-in
browser on their behalf, and nothing is sent anywhere to obtain the list.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from steering.ingestion.x import parse_x_post_url

LOGGER = logging.getLogger(__name__)

MAX_EXPORT_BYTES = 8 * 1024 * 1024
#: Keys an export may use for the post link, covering the shapes different
#: bookmarklets emit rather than insisting on one of them.
_URL_KEYS = ("url", "link", "href", "tweet_url", "tweetUrl", "post_url", "postUrl", "permalink")


class UnsupportedBookmarkExportError(ValueError):
    """Raised when an export cannot be read as a list of saved posts."""


def _candidate_urls(value: Any) -> list[str]:
    """Walk an arbitrary export shape and collect anything that looks like a link."""

    if isinstance(value, str):
        return [value] if value.startswith(("http://", "https://")) else []
    if isinstance(value, list):
        return [url for item in value for url in _candidate_urls(item)]
    if isinstance(value, dict):
        found: list[str] = []
        for key in _URL_KEYS:
            entry = value.get(key)
            if isinstance(entry, str) and entry.startswith(("http://", "https://")):
                found.append(entry)
        if found:
            return found
        # No recognised key: the export may nest its records, so keep looking.
        return [url for item in value.values() for url in _candidate_urls(item)]
    return []


def bookmark_sources(payload: str) -> list[str]:
    """Extract canonical X post URLs from a bookmark export.

    Accepts JSON in any shape a bookmarklet is likely to produce, and a plain
    newline-separated list of links. Anything that is not a single X post is
    dropped, so a stray profile or search link cannot become an artifact.
    """

    text = payload.strip()
    if not text:
        raise UnsupportedBookmarkExportError("the export is empty")

    if text.startswith(("{", "[")):
        try:
            candidates = _candidate_urls(json.loads(text))
        except json.JSONDecodeError as exc:
            raise UnsupportedBookmarkExportError(f"the export is not valid JSON ({exc.msg})") from None
    else:
        candidates = [line.strip() for line in text.splitlines() if line.strip()]

    sources: list[str] = []
    for candidate in candidates:
        reference = parse_x_post_url(candidate)
        if reference is None:
            continue
        canonical = reference.canonical_url()
        if canonical not in sources:
            sources.append(canonical)
    if not sources:
        raise UnsupportedBookmarkExportError("the export contained no X post links")
    LOGGER.info("bookmark export yielded %d post(s)", len(sources))
    return sources


def bookmark_sources_from_file(path: Path) -> list[str]:
    if not path.is_file():
        raise UnsupportedBookmarkExportError(f"no such export file: {path}")
    if path.stat().st_size > MAX_EXPORT_BYTES:
        raise UnsupportedBookmarkExportError("the export exceeds the 8 MiB safety limit")
    return bookmark_sources(path.read_text(encoding="utf-8-sig"))
