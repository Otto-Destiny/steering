"""Public X (Twitter) post capture.

Two public representations exist and neither needs an account:

* the syndication payload that X's own embed widget requests, which carries the
  post body, the author's unwrapped destination URLs, media with author-supplied
  alt text, and the publication timestamp;
* the oEmbed blockquote, which truncates the body and leaves every destination
  as a ``t.co`` shortlink.

Syndication is preferred because the shortlinks oEmbed returns carry no signal a
downstream resolver can rank or follow, so an oEmbed-only capture silently loses
the paper or repository the post was written to point at. oEmbed remains the
fallback and unwraps its shortlinks before returning.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit

from bs4 import BeautifulSoup

from steering import __version__
from steering.domain.models import ResolvedSource, SourceKind
from steering.ingestion.security import SafeFetcher, SourceUnavailableError

LOGGER = logging.getLogger(__name__)

X_HOSTS = frozenset(
    {
        "x.com",
        "www.x.com",
        "mobile.x.com",
        "m.x.com",
        "twitter.com",
        "www.twitter.com",
        "mobile.twitter.com",
        "m.twitter.com",
    }
)
X_MEDIA_HOSTS = frozenset(
    {
        "pbs.twimg.com",
        "video.twimg.com",
        "abs.twimg.com",
        # The domains X renders as the visible label for an attached photo or video.
        "pic.twitter.com",
        "pic.x.com",
    }
)
SHORTLINK_HOST = "t.co"

SYNDICATION_ENDPOINT = "https://cdn.syndication.twimg.com/tweet-result"
OEMBED_ENDPOINT = "https://publish.x.com/oembed"
# FixTweet is the open-source embed service behind fxtwitter.com. It is the only
# free, unauthenticated source that returns the *complete* body of a long-form
# post; X's own syndication payload truncates those to their opening and exposes
# the remainder nowhere. It is used only to recover that missing text, never as
# the primary capture, and never for a post X already returned in full.
LONG_FORM_ENDPOINT = "https://api.fxtwitter.com"
#: FixTweet rejects the HTTP client's default agent outright, so a capture that
#: does not name itself gets 403 and silently loses every long-form body. Naming
#: the tool honestly is enough; no browser impersonation is involved.
LONG_FORM_USER_AGENT = f"STEERING/{__version__} (+https://github.com/steering-knowledge/steering)"

SYNDICATION_METHOD = "x_public_syndication"
OEMBED_METHOD = "x_public_oembed"
BROWSER_METHOD = "authorized_visible_browser"
PUBLIC_METHODS = frozenset({SYNDICATION_METHOD, OEMBED_METHOD})

# X publishes one post at a time. Only an authorized signed-in capture can read
# the author's self-reply thread, so every capture records which it was.
ROOT_POST_ONLY = "root_post_only"
AUTHOR_THREAD = "author_thread"

_ANONYMOUS_STATUS = re.compile(r"^/i/(?:web/)?status(?:es)?/(?P<post_id>\d{1,25})(?:/|$)")
_HANDLE_STATUS = re.compile(r"^/(?P<handle>[A-Za-z0-9_]{1,15})/status(?:es)?/(?P<post_id>\d{1,25})(?:/|$)")
_BASE36_DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"
# A float64 fraction carries at most 53 bits, and 36**11 > 2**56, so eleven
# base-36 fraction digits reproduce every digit the browser widget can emit.
_BASE36_FRACTION_DIGITS = 11


def is_x_url(source: str) -> bool:
    return (urlsplit(source).hostname or "").lower() in X_HOSTS


def is_x_owned_url(url: str) -> bool:
    """Report whether a URL points back at X itself rather than at external material."""

    host = (urlsplit(url).hostname or "").lower()
    return host in X_HOSTS or host in X_MEDIA_HOSTS or host == SHORTLINK_HOST


@dataclass(frozen=True, slots=True)
class XPostRef:
    """One X post identified independently of the many URL shapes that address it."""

    post_id: str
    handle: str | None = None

    def canonical_url(self, handle: str | None = None) -> str:
        resolved = (handle or self.handle or "i").lower()
        return f"https://x.com/{resolved}/status/{self.post_id}"


def parse_x_post_url(source: str) -> XPostRef | None:
    """Reduce any X post URL shape to its author handle and post id.

    Returns ``None`` for X URLs that do not address a single post, such as
    profiles, search, or the home timeline.
    """

    parts = urlsplit(source)
    if (parts.hostname or "").lower() not in X_HOSTS:
        return None
    path = parts.path if parts.path.startswith("/") else f"/{parts.path}"
    anonymous = _ANONYMOUS_STATUS.match(path)
    if anonymous is not None:
        return XPostRef(post_id=anonymous.group("post_id"))
    handled = _HANDLE_STATUS.match(path)
    if handled is not None:
        return XPostRef(post_id=handled.group("post_id"), handle=handled.group("handle"))
    return None


def syndication_token(post_id: str) -> str:
    """Reproduce the cache-busting token X's embed widget derives from the post id.

    The endpoint does not reject a missing or altered token, but sending the
    value the official widget sends keeps this client indistinguishable from the
    supported embed path.
    """

    value = (int(post_id) / 1e15) * math.pi
    whole = int(value)
    fraction = value - whole
    digits = ""
    while whole:
        digits = _BASE36_DIGITS[whole % 36] + digits
        whole //= 36
    rendered = digits or "0"
    if fraction:
        encoded = ""
        for _ in range(_BASE36_FRACTION_DIGITS):
            fraction *= 36
            index = int(fraction)
            encoded += _BASE36_DIGITS[index]
            fraction -= index
        rendered = f"{rendered}.{encoded.rstrip('0')}"
    return re.sub(r"[0.]", "", rendered)


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _entities(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    entities = payload.get("entities")
    if not isinstance(entities, dict):
        return []
    items = entities.get(key)
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def _expanded_body(payload: dict[str, Any]) -> str:
    """Rewrite the body so it carries real destinations instead of shortlinks."""

    body = _text(payload.get("text"))
    for item in _entities(payload, "urls"):
        shortlink, expanded = _text(item.get("url")), _text(item.get("expanded_url"))
        if shortlink and expanded:
            body = body.replace(shortlink, expanded)
    for item in _entities(payload, "media"):
        shortlink = _text(item.get("url"))
        if shortlink:
            body = body.replace(shortlink, "")
    return body.strip()


#: A URL written in a post body. Trailing sentence punctuation is stripped
#: separately because it is far more often prose than part of the address.
_URL_IN_TEXT = re.compile(r"https?://[^\s<>\"'\]}]+")


def _links_in_text(text: str) -> list[str]:
    """Read destinations out of the body itself.

    A long-form post arrives with ``entities.urls`` set to null and ``note_tweet``
    reduced to a bare id, so the payload records no destinations at all. Once the
    full body is recovered the prose is the only surviving evidence of where the
    author was sending the reader. Shortlinks are left intact for the caller to
    unwrap, which is also what filters X's own hosts back out.
    """

    found = [match.rstrip(".,;:!?)'\"") for match in _URL_IN_TEXT.findall(text)]
    return list(dict.fromkeys(url for url in found if url))


def _outbound_urls(payload: dict[str, Any]) -> list[str]:
    """Collect the author's external destinations, already unwrapped by X."""

    found = [
        expanded
        for item in _entities(payload, "urls")
        if (expanded := _text(item.get("expanded_url")))
        and expanded.startswith(("http://", "https://"))
        and not is_x_owned_url(expanded)
    ]
    return list(dict.fromkeys(found))


def _media(payload: dict[str, Any]) -> tuple[list[str], list[dict[str, str]]]:
    """Return media URLs plus the subset carrying author-written alt text."""

    urls: list[str] = []
    described: list[dict[str, str]] = []
    details = payload.get("mediaDetails")
    for item in details if isinstance(details, list) else []:
        if not isinstance(item, dict):
            continue
        url = _text(item.get("media_url_https"))
        if not url or url in urls:
            continue
        urls.append(url)
        alt = _text(item.get("ext_alt_text"))
        if alt:
            described.append({"url": url, "alt": alt})
    return urls, described


def _published_at(payload: dict[str, Any]) -> datetime | None:
    raw = _text(payload.get("created_at"))
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        LOGGER.debug("X post carried an unparsable created_at value: %r", raw)
        return None


def _author(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    user = payload.get("user")
    if not isinstance(user, dict):
        return None, None
    return _text(user.get("screen_name")) or None, _text(user.get("name")) or None


def _quoted_block(payload: dict[str, Any]) -> tuple[str, list[str]]:
    """Fold one level of quoted post into the body; quotes usually carry the source."""

    quoted = payload.get("quoted_tweet")
    if not isinstance(quoted, dict):
        return "", []
    body = _expanded_body(quoted)
    if not body:
        return "", []
    handle, _display = _author(quoted)
    attribution = f"@{handle}" if handle else "an unnamed account"
    return f"\n\n[Quoted post by {attribution}]\n{body}", _outbound_urls(quoted)


def _conversation_count(payload: dict[str, Any]) -> int:
    value = payload.get("conversation_count")
    return value if isinstance(value, int) and value > 0 else 0


def _capture_advice(*, long_form_truncated: bool, has_conversation: bool) -> str | None:
    """State plainly what a public capture could not reach.

    X publishes one post at a time. Replies are not in any public payload, so a
    link the author put in a self-reply is invisible here. Saying so is the
    difference between a known limit and silent data loss.
    """

    reasons = []
    if long_form_truncated:
        reasons.append("This is a long-form X post and the public payload carries only its opening")
    if has_conversation:
        reasons.append(
            "This post has replies, and public capture reads the root post only, so links "
            "the author added in a self-reply are not included"
        )
    if not reasons:
        return None
    return f"{'. '.join(reasons)}. Use authorized signed-in capture to bundle the author's full thread."


def _tombstone_reason(payload: dict[str, Any]) -> str | None:
    if _text(payload.get("__typename")) != "TweetTombstone":
        return None
    tombstone = payload.get("tombstone")
    if isinstance(tombstone, dict):
        nested = tombstone.get("text")
        if isinstance(nested, dict) and (reason := _text(nested.get("text"))):
            return reason
    return "X reports this post as unavailable"


class XPostUnavailable(SourceUnavailableError):
    """Raised when X states the post is deleted, protected, or otherwise withdrawn."""


class XResolver:
    """Resolve a public X post without an account, preserving author links."""

    name = "x"

    def __init__(self, fetcher: SafeFetcher, *, recover_long_form: bool = True) -> None:
        self.fetcher = fetcher
        self.recover_long_form = recover_long_form

    def can_resolve(self, source: str) -> bool:
        return is_x_url(source)

    async def resolve(self, source: str) -> ResolvedSource:
        ref = parse_x_post_url(source)
        if ref is None:
            raise SourceUnavailableError(
                "this X URL does not address a single post; paste a post link such as "
                "https://x.com/<author>/status/<id>"
            )
        try:
            return await self._resolve_syndication(ref)
        except XPostUnavailable:
            # X stated the post is gone. A weaker public path cannot contradict
            # that, and retrying would replace a precise reason with a vague one.
            raise
        except SourceUnavailableError as exc:
            LOGGER.info(
                "X syndication capture failed for post %s (%s); falling back to oEmbed",
                ref.post_id,
                exc,
            )
        return await self._resolve_oembed(ref)

    async def _resolve_syndication(self, ref: XPostRef) -> ResolvedSource:
        data, _headers, _final_url = await self.fetcher.get(
            SYNDICATION_ENDPOINT,
            params={"id": ref.post_id, "token": syndication_token(ref.post_id), "lang": "en"},
            headers={"Accept": "application/json"},
        )
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            raise SourceUnavailableError("X syndication returned a non-JSON payload") from exc
        if not isinstance(payload, dict):
            raise SourceUnavailableError("X syndication returned an unexpected payload shape")
        if (reason := _tombstone_reason(payload)) is not None:
            raise XPostUnavailable(reason)

        handle, display_name = _author(payload)
        body = _expanded_body(payload)
        quoted_text, quoted_urls = _quoted_block(payload)
        media_urls, described_media = _media(payload)
        long_form_truncated = isinstance(payload.get("note_tweet"), dict)
        has_conversation = _conversation_count(payload) > 0
        if not body and not quoted_text:
            raise SourceUnavailableError("X syndication returned a post with no readable text")

        recovered_urls: list[str] = []
        if (
            long_form_truncated
            and self.recover_long_form
            and (full := await self._long_form_body(ref, len(body))) is not None
        ):
            body = full
            long_form_truncated = False
            # Recovering the text without its links would leave the destination
            # readable as prose but never actually visited, which is the whole
            # point of capturing the post.
            recovered_urls = await self._unwrap_shortlinks(_links_in_text(full))

        return ResolvedSource(
            canonical_url=ref.canonical_url(handle),
            source_kind=SourceKind.X,
            title=f"X post by {display_name or (f'@{handle}' if handle else 'unknown author')}",
            text=f"{body}{quoted_text}".strip(),
            author=handle,
            published_at=_published_at(payload),
            mime_type="application/json",
            extraction_method=SYNDICATION_METHOD,
            partial=long_form_truncated,
            outbound_urls=list(dict.fromkeys([*_outbound_urls(payload), *quoted_urls, *recovered_urls])),
            media_urls=media_urls,
            metadata={
                "author_display_name": display_name,
                "capture_scope": ROOT_POST_ONLY,
                "long_form_truncated": long_form_truncated,
                "reply_count": _conversation_count(payload),
                "media_inclusion_candidates": described_media,
                "media_decision": (
                    "consider_author_described_media_when_no_stronger_linked_source_exists"
                    if described_media
                    else "skip_no_author_described_media"
                ),
                **(
                    {"capture_advice": advice}
                    if (
                        advice := _capture_advice(
                            long_form_truncated=long_form_truncated,
                            has_conversation=has_conversation,
                        )
                    )
                    else {}
                ),
            },
        )

    async def _resolve_oembed(self, ref: XPostRef) -> ResolvedSource:
        data, _headers, _final_url = await self.fetcher.get(
            OEMBED_ENDPOINT,
            params={"url": ref.canonical_url(), "omit_script": "1", "dnt": "1"},
        )
        try:
            payload = json.loads(data)
            html = str(payload["html"])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise SourceUnavailableError("X oEmbed returned an invalid payload") from exc

        soup = BeautifulSoup(html, "html.parser")
        body = soup.get_text(" ", strip=True)
        handle = _handle_from_author_url(_text(payload.get("author_url"))) or ref.handle
        shortlinks = [
            href
            for anchor in soup.find_all("a", href=True)
            if (href := str(anchor["href"])) and not is_x_url(href)
        ]
        outbound = await self._unwrap_shortlinks(shortlinks)
        return ResolvedSource(
            canonical_url=ref.canonical_url(handle),
            source_kind=SourceKind.X,
            title=f"X post by {_text(payload.get('author_name')) or 'unknown author'}",
            text=body,
            author=handle,
            mime_type="text/html",
            extraction_method=OEMBED_METHOD,
            partial=True,
            outbound_urls=outbound,
            metadata={
                "author_display_name": _text(payload.get("author_name")) or None,
                "capture_scope": ROOT_POST_ONLY,
                "capture_degraded": True,
                "capture_advice": (
                    "X syndication was unavailable, so this capture came from the oEmbed "
                    "blockquote and may be truncated. It covers the root post only; use "
                    "authorized signed-in capture to bundle the author's full thread."
                ),
            },
        )

    async def _long_form_body(self, ref: XPostRef, truncated_length: int) -> str | None:
        """Recover the body of a long-form post that X truncates everywhere.

        Returns ``None`` on any failure or if the result is not actually longer,
        so the capture falls back to the honest truncated body plus its advice
        rather than trading a known limit for a silent one.
        """

        try:
            data, _headers, _final = await self.fetcher.get(
                f"{LONG_FORM_ENDPOINT}/x/status/{ref.post_id}",
                headers={"Accept": "application/json", "User-Agent": LONG_FORM_USER_AGENT},
            )
            payload = json.loads(data)
        except (SourceUnavailableError, json.JSONDecodeError) as exc:
            LOGGER.info("long-form recovery for post %s failed (%s)", ref.post_id, exc)
            return None
        post = payload.get("tweet") if isinstance(payload, dict) else None
        if not isinstance(post, dict):
            return None
        body = _text(post.get("text"))
        if len(body) <= truncated_length:
            return None
        LOGGER.info(
            "recovered full long-form body for post %s (%d chars, was %d)",
            ref.post_id,
            len(body),
            truncated_length,
        )
        return body

    async def _unwrap_shortlinks(self, urls: list[str]) -> list[str]:
        """Turn t.co shortlinks into destinations a downstream resolver can rank."""

        unwrapped: list[str] = []
        for url in dict.fromkeys(urls):
            destination = url
            if (urlsplit(url).hostname or "").lower() == SHORTLINK_HOST:
                try:
                    destination = await self.fetcher.resolve_redirects(url)
                except SourceUnavailableError as exc:
                    LOGGER.info("X shortlink %s could not be unwrapped (%s)", url, exc)
                    continue
            if not is_x_owned_url(destination) and destination not in unwrapped:
                unwrapped.append(destination)
        return unwrapped


def _handle_from_author_url(author_url: str) -> str | None:
    if not author_url:
        return None
    segments = [segment for segment in urlsplit(author_url).path.split("/") if segment]
    return segments[0] if segments else None
