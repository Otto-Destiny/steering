"""Authenticated X API access: threads and bookmarks, without a browser.

This is the one capture surface that costs money. X moved to pay-per-use in
February 2026, so reads are billed per post rather than behind a subscription,
but they are still billed. Everything here is therefore opt-in: with no client
id configured the reader reports itself unavailable and the free paths run
exactly as they did before.

It earns its place by doing what neither free path can: reading a thread
unattended, with no browser, no signed-in session, and no bot-detection risk.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlencode

import httpx

from steering.domain.models import ResolvedSource, SourceKind
from steering.ingestion.security import SourceUnavailableError
from steering.ingestion.x import (
    AUTHOR_THREAD,
    BROWSER_METHOD,
    ROOT_POST_ONLY,
    XPostRef,
    is_x_owned_url,
    parse_x_post_url,
)

LOGGER = logging.getLogger(__name__)

AUTHORIZE_URL = "https://x.com/i/oauth2/authorize"
TOKEN_URL = "https://api.x.com/2/oauth2/token"  # noqa: S105 - endpoint, not a secret
API_ROOT = "https://api.x.com/2"
API_METHOD = "x_api_thread"
API_BOOKMARK_METHOD = "x_api_bookmark"
#: `offline.access` is what makes an unattended run possible: without it the
#: token expires in two hours and every batch would need a human again.
SCOPES = ("tweet.read", "users.read", "bookmark.read", "offline.access")
DEFAULT_REDIRECT_PATH = "/oauth/x/callback"
_THREAD_SPAN_HOURS = 48
_MAX_THREAD_POSTS = 100
_TWEET_FIELDS = "created_at,entities,author_id,conversation_id,note_tweet,attachments"


class XApiError(SourceUnavailableError):
    """Raised when the X API cannot satisfy a request."""


class XApiNotAuthorized(XApiError):
    """Raised when no usable token exists and the user must authorize again."""


@dataclass(slots=True)
class XApiToken:
    access_token: str
    refresh_token: str | None
    expires_at: datetime

    @property
    def expired(self) -> bool:
        # Refresh a minute early so a long request cannot straddle expiry.
        return datetime.now(UTC) >= self.expires_at - timedelta(seconds=60)

    def serialize(self) -> str:
        return json.dumps(
            {
                "access_token": self.access_token,
                "refresh_token": self.refresh_token,
                "expires_at": self.expires_at.isoformat(),
            }
        )

    @classmethod
    def deserialize(cls, raw: str) -> XApiToken | None:
        try:
            payload = json.loads(raw)
            return cls(
                access_token=str(payload["access_token"]),
                refresh_token=payload.get("refresh_token"),
                expires_at=datetime.fromisoformat(str(payload["expires_at"])),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            LOGGER.warning("stored X API token could not be read; reauthorization is required")
            return None


@dataclass(slots=True)
class PendingAuthorization:
    """One in-flight PKCE exchange, held only until the callback returns."""

    state: str
    code_verifier: str
    redirect_uri: str


def start_authorization(client_id: str, redirect_uri: str) -> tuple[str, PendingAuthorization]:
    """Build the consent URL and the verifier its callback must be checked against.

    PKCE is used rather than a client secret because STEERING runs on the user's
    own machine, where a secret could not be kept.
    """

    verifier = secrets.token_urlsafe(64)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
        .decode("ascii")
        .rstrip("=")
    )
    state = secrets.token_urlsafe(24)
    query = urlencode(
        {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": " ".join(SCOPES),
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
    )
    pending = PendingAuthorization(state=state, code_verifier=verifier, redirect_uri=redirect_uri)
    return f"{AUTHORIZE_URL}?{query}", pending


def _token_from_response(payload: dict[str, Any]) -> XApiToken:
    try:
        expires_in = int(payload.get("expires_in", 7200))
        return XApiToken(
            access_token=str(payload["access_token"]),
            refresh_token=payload.get("refresh_token"),
            expires_at=datetime.now(UTC) + timedelta(seconds=expires_in),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise XApiError("the X API returned a token response that could not be read") from exc


class XApiClient:
    """Authenticated X API v2 client that refreshes its own token."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        client_id: str,
        token: XApiToken | None = None,
        on_token_refreshed: Any = None,
    ) -> None:
        self._client = client
        self._client_id = client_id
        self._token = token
        self._on_token_refreshed = on_token_refreshed

    @property
    def authorized(self) -> bool:
        return self._token is not None

    async def complete_authorization(self, code: str, pending: PendingAuthorization) -> XApiToken:
        token = await self._post_token(
            {
                "grant_type": "authorization_code",
                "code": code,
                "client_id": self._client_id,
                "redirect_uri": pending.redirect_uri,
                "code_verifier": pending.code_verifier,
            }
        )
        self._store(token)
        return token

    async def _post_token(self, data: dict[str, str]) -> XApiToken:
        try:
            response = await self._client.post(
                TOKEN_URL,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        except httpx.HTTPError as exc:
            raise XApiError(f"the X API token request failed ({type(exc).__name__})") from None
        if response.status_code >= 400:
            raise XApiError(f"the X API rejected the token request ({response.status_code})")
        return _token_from_response(response.json())

    def _store(self, token: XApiToken) -> None:
        self._token = token
        if callable(self._on_token_refreshed):
            self._on_token_refreshed(token)

    async def _authorization(self) -> str:
        if self._token is None:
            raise XApiNotAuthorized("STEERING is not authorized to use the X API")
        if self._token.expired:
            if not self._token.refresh_token:
                raise XApiNotAuthorized("the X API token expired and cannot be refreshed")
            LOGGER.info("refreshing the expired X API token")
            self._store(
                await self._post_token(
                    {
                        "grant_type": "refresh_token",
                        "refresh_token": self._token.refresh_token,
                        "client_id": self._client_id,
                    }
                )
            )
        assert self._token is not None
        return f"Bearer {self._token.access_token}"

    async def _get(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        header = await self._authorization()
        try:
            response = await self._client.get(
                f"{API_ROOT}{path}",
                params=params,
                headers={"Authorization": header},
            )
        except httpx.HTTPError as exc:
            raise XApiError(f"the X API request failed ({type(exc).__name__})") from None
        if response.status_code == 401:
            raise XApiNotAuthorized("the X API rejected the stored token; authorize again")
        if response.status_code == 429:
            raise XApiError("the X API rate limit was reached; retry later")
        if response.status_code >= 400:
            raise XApiError(f"the X API returned {response.status_code} for {path}")
        payload = response.json()
        if not isinstance(payload, dict):
            raise XApiError("the X API returned an unexpected payload shape")
        return payload

    async def thread(self, reference: XPostRef) -> ResolvedSource:
        """Read a post and the author's replies to it as one source."""

        root_payload = await self._get(
            f"/tweets/{reference.post_id}",
            {"tweet.fields": _TWEET_FIELDS, "expansions": "author_id", "user.fields": "username,name"},
        )
        root = root_payload.get("data")
        if not isinstance(root, dict):
            raise XApiError("the X API returned no post for that id")
        users = {
            str(user["id"]): user
            for user in (root_payload.get("includes", {}) or {}).get("users", [])
            if isinstance(user, dict) and "id" in user
        }
        author_id = str(root.get("author_id") or "")
        author = users.get(author_id, {})
        handle = str(author.get("username") or "") or None

        replies = await self._self_replies(root, author_id)
        posts = [root, *replies]
        body = "\n\n---\n\n".join(text for post in posts if (text := _post_text(post)))
        outbound = _outbound_urls(posts)
        return ResolvedSource(
            canonical_url=reference.canonical_url(handle),
            source_kind=SourceKind.X,
            title=f"{author.get('name') or ('@' + handle if handle else 'X post')} — X thread",
            text=body,
            author=handle,
            published_at=_created_at(root),
            mime_type="application/json",
            extraction_method=API_METHOD,
            partial=not body,
            outbound_urls=outbound,
            metadata={
                "capture_scope": AUTHOR_THREAD,
                "bundled_self_replies": len(replies),
                "author_display_name": author.get("name"),
            },
        )

    async def _self_replies(self, root: dict[str, Any], author_id: str) -> list[dict[str, Any]]:
        """Fetch the author's own replies within the thread window.

        `search/recent` only reaches back seven days, so older threads return the
        root alone rather than failing; that is a smaller loss than no capture.
        """

        conversation_id = str(root.get("conversation_id") or root.get("id") or "")
        if not conversation_id or not author_id:
            return []
        try:
            payload = await self._get(
                "/tweets/search/recent",
                {
                    "query": f"conversation_id:{conversation_id} from:{author_id}",
                    "tweet.fields": _TWEET_FIELDS,
                    "max_results": str(_MAX_THREAD_POSTS),
                },
            )
        except XApiError as exc:
            LOGGER.info("could not read replies for conversation %s (%s)", conversation_id, exc)
            return []
        found = payload.get("data")
        if not isinstance(found, list):
            return []
        root_time = _created_at(root)
        replies = [
            post
            for post in found
            if isinstance(post, dict)
            and str(post.get("id")) != str(root.get("id"))
            and _within_window(root_time, _created_at(post))
        ]
        replies.sort(key=lambda post: str(post.get("id")))
        return replies

    async def bookmarks(self, *, limit: int = 100) -> list[ResolvedSource]:
        """Return the authorized user's bookmarked posts, content included.

        The bookmarks response already carries each post's text, entities, and
        timestamp. Returning only ids would mean fetching all of that a second
        time, and on a billed API that is paying twice for the same read.
        """

        me = await self._get("/users/me", {})
        user = me.get("data")
        if not isinstance(user, dict) or "id" not in user:
            raise XApiError("the X API did not identify the authorized user")
        collected: list[ResolvedSource] = []
        token: str | None = None
        while len(collected) < limit:
            params = {
                "max_results": str(min(100, limit - len(collected))),
                "tweet.fields": _TWEET_FIELDS,
                "expansions": "author_id",
                "user.fields": "username,name",
            }
            if token:
                params["pagination_token"] = token
            payload = await self._get(f"/users/{user['id']}/bookmarks", params)
            users = {
                str(item["id"]): item
                for item in (payload.get("includes", {}) or {}).get("users", [])
                if isinstance(item, dict) and "id" in item
            }
            rows = payload.get("data")
            if not isinstance(rows, list) or not rows:
                break
            for row in rows:
                if not isinstance(row, dict):
                    continue
                source = _bookmark_source(row, users)
                if source is not None and all(
                    source.canonical_url != existing.canonical_url for existing in collected
                ):
                    collected.append(source)
            meta = payload.get("meta")
            token = meta.get("next_token") if isinstance(meta, dict) else None
            if not token:
                break
        LOGGER.info("read %d bookmarked post(s) from the X API", len(collected))
        return collected


def _bookmark_source(post: dict[str, Any], users: dict[str, dict[str, Any]]) -> ResolvedSource | None:
    """Turn one bookmarked post into a capture, using content already returned."""

    author = users.get(str(post.get("author_id") or ""), {})
    handle = str(author.get("username") or "") or None
    reference = XPostRef(post_id=str(post.get("id") or ""), handle=handle)
    if not reference.post_id:
        return None
    body = _post_text(post)
    if not body:
        return None
    return ResolvedSource(
        canonical_url=reference.canonical_url(handle),
        source_kind=SourceKind.X,
        title=f"X post by {author.get('name') or ('@' + handle if handle else 'unknown author')}",
        text=body,
        author=handle,
        published_at=_created_at(post),
        mime_type="application/json",
        extraction_method=API_BOOKMARK_METHOD,
        outbound_urls=_outbound_urls([post]),
        metadata={
            "capture_scope": ROOT_POST_ONLY,
            "author_display_name": author.get("name"),
            # The bookmarks endpoint returns the post, never its replies, so the
            # reply count is unknown here rather than zero.
            "reply_count": 1,
        },
    )


def _post_text(post: dict[str, Any]) -> str:
    """Prefer the long-form body, which the API returns in full."""

    note = post.get("note_tweet")
    if isinstance(note, dict) and isinstance(note.get("text"), str) and note["text"].strip():
        return _expanded(note["text"], note.get("entities"))
    return _expanded(str(post.get("text") or ""), post.get("entities"))


def _expanded(text: str, entities: Any) -> str:
    """Replace shortlinks with the destinations the API already resolved."""

    if not isinstance(entities, dict):
        return text.strip()
    for item in entities.get("urls") or []:
        if not isinstance(item, dict):
            continue
        shortlink, expanded = str(item.get("url") or ""), str(item.get("expanded_url") or "")
        if not shortlink or not expanded:
            continue
        text = text.replace(shortlink, "" if is_x_owned_url(expanded) else expanded)
    return text.strip()


def _outbound_urls(posts: list[dict[str, Any]]) -> list[str]:
    found: list[str] = []
    for post in posts:
        for source in (post.get("entities"), (post.get("note_tweet") or {}).get("entities")):
            if not isinstance(source, dict):
                continue
            for item in source.get("urls") or []:
                if not isinstance(item, dict):
                    continue
                expanded = str(item.get("expanded_url") or "")
                if (
                    expanded.startswith(("http://", "https://"))
                    and not is_x_owned_url(expanded)
                    and expanded not in found
                ):
                    found.append(expanded)
    return found


def _created_at(post: dict[str, Any]) -> datetime | None:
    raw = post.get("created_at")
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _within_window(root_time: datetime | None, post_time: datetime | None) -> bool:
    if root_time is None or post_time is None:
        return True
    return abs((post_time - root_time).total_seconds()) <= _THREAD_SPAN_HOURS * 3600


class ApiThreadReader:
    """Reads a thread through the authenticated API, with no browser at all."""

    name = "x_api"

    def __init__(self, client: XApiClient | None) -> None:
        self._client = client

    def available(self) -> bool:
        return self._client is not None and self._client.authorized

    async def read_thread(self, url: str) -> ResolvedSource:
        reference = parse_x_post_url(url)
        if reference is None or self._client is None:
            raise XApiError("the X API can only read a single post URL")
        captured = await self._client.thread(reference)
        LOGGER.info(
            "X API read %s with %s self-repl(ies)",
            captured.canonical_url,
            captured.metadata.get("bundled_self_replies"),
        )
        return captured


__all__ = [
    "API_BOOKMARK_METHOD",
    "API_METHOD",
    "BROWSER_METHOD",
    "DEFAULT_REDIRECT_PATH",
    "SCOPES",
    "ApiThreadReader",
    "PendingAuthorization",
    "XApiClient",
    "XApiError",
    "XApiNotAuthorized",
    "XApiToken",
    "XPostRef",
    "start_authorization",
]
