"""X API surface tests.

The API is the one billed capture path, so the properties that matter most are
that it stays inert until a user opts in, and that when it does run it produces
the same shape of source the free paths do.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from steering.domain.models import SourceKind
from steering.ingestion.x import parse_x_post_url
from steering.ingestion.x_api import (
    API_METHOD,
    SCOPES,
    ApiThreadReader,
    XApiClient,
    XApiNotAuthorized,
    XApiToken,
    start_authorization,
)

POST_ID = "1885026028428681698"
PAPER_URL = "https://arxiv.org/abs/2501.12948"
REPO_URL = "https://github.com/example/agent-memory"


def live_token() -> XApiToken:
    return XApiToken(
        access_token="access",
        refresh_token="refresh",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )


def api_client(handler: Any, token: XApiToken | None = None, **kwargs: Any) -> XApiClient:
    return XApiClient(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        client_id="client-id",
        token=token,
        **kwargs,
    )


def thread_handler(
    *,
    root: dict[str, Any] | None = None,
    replies: list[dict[str, Any]] | None = None,
) -> Any:
    """Respond the way the API does: the post, then a conversation search."""

    root_post = root or {
        "id": POST_ID,
        "text": "Root claim with the link kept out. https://t.co/SHORT",
        "author_id": "42",
        "conversation_id": POST_ID,
        "created_at": "2026-07-16T12:00:00.000Z",
        "entities": {"urls": [{"url": "https://t.co/SHORT", "expanded_url": PAPER_URL}]},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == f"/2/tweets/{POST_ID}":
            return httpx.Response(
                200,
                json={
                    "data": root_post,
                    "includes": {"users": [{"id": "42", "username": "Researcher", "name": "A Researcher"}]},
                },
            )
        if request.url.path == "/2/tweets/search/recent":
            return httpx.Response(200, json={"data": replies or []})
        raise AssertionError(f"unexpected path {request.url.path}")

    return handler


# --------------------------------------------------------------------------- #
# Opt-in behaviour
# --------------------------------------------------------------------------- #


def test_the_reader_is_unavailable_without_a_client() -> None:
    """With no API configured the billed path must never be attempted."""

    assert ApiThreadReader(None).available() is False


def test_the_reader_is_unavailable_until_authorized() -> None:
    assert ApiThreadReader(api_client(thread_handler())).available() is False


def test_the_reader_becomes_available_once_a_token_exists() -> None:
    assert ApiThreadReader(api_client(thread_handler(), live_token())).available() is True


@pytest.mark.asyncio
async def test_an_unauthorized_request_names_the_fix() -> None:
    with pytest.raises(XApiNotAuthorized, match="not authorized"):
        await api_client(thread_handler()).thread(parse_x_post_url(f"https://x.com/i/status/{POST_ID}"))


# --------------------------------------------------------------------------- #
# Authorization
# --------------------------------------------------------------------------- #


def test_authorization_uses_pkce_and_requests_offline_access() -> None:
    """Without offline access the token expires in hours and batches break."""

    url, pending = start_authorization("client-id", "http://127.0.0.1:8765/oauth/x/callback")
    query = parse_qs(urlsplit(url).query)

    assert query["code_challenge_method"] == ["S256"]
    assert query["code_challenge"][0] and query["code_challenge"][0] != pending.code_verifier
    assert query["state"] == [pending.state]
    assert set(query["scope"][0].split()) == set(SCOPES)
    assert "offline.access" in query["scope"][0]
    # No client secret is sent: STEERING runs where a secret could not be kept.
    assert "client_secret" not in query


def test_each_authorization_is_unique() -> None:
    first, first_pending = start_authorization("client-id", "http://127.0.0.1:8765/cb")
    second, second_pending = start_authorization("client-id", "http://127.0.0.1:8765/cb")

    assert first != second
    assert first_pending.state != second_pending.state
    assert first_pending.code_verifier != second_pending.code_verifier


@pytest.mark.asyncio
async def test_a_completed_authorization_is_handed_back_for_storage() -> None:
    stored: list[XApiToken] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = parse_qs(request.content.decode())
        assert body["grant_type"] == ["authorization_code"]
        assert body["code_verifier"] == ["verifier"]
        return httpx.Response(200, json={"access_token": "new", "refresh_token": "r", "expires_in": 7200})

    from steering.ingestion.x_api import PendingAuthorization

    client = api_client(handler, on_token_refreshed=stored.append)
    token = await client.complete_authorization(
        "code",
        PendingAuthorization(state="s", code_verifier="verifier", redirect_uri="http://127.0.0.1/cb"),
    )

    assert token.access_token == "new"
    assert client.authorized is True
    # The runtime persists it in the keyring, never in the JSON config.
    assert stored == [token]


@pytest.mark.asyncio
async def test_an_expired_token_is_refreshed_before_use() -> None:
    refreshed: list[XApiToken] = []
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/2/oauth2/token":
            return httpx.Response(
                200, json={"access_token": "fresh", "refresh_token": "r2", "expires_in": 7200}
            )
        return thread_handler()(request)

    expired = XApiToken("stale", "r1", datetime.now(UTC) - timedelta(minutes=5))
    client = api_client(handler, expired, on_token_refreshed=refreshed.append)

    await client.thread(parse_x_post_url(f"https://x.com/i/status/{POST_ID}"))

    assert calls[0] == "/2/oauth2/token"
    assert refreshed and refreshed[0].access_token == "fresh"


@pytest.mark.asyncio
async def test_an_expired_token_without_a_refresh_token_asks_for_reauthorization() -> None:
    expired = XApiToken("stale", None, datetime.now(UTC) - timedelta(minutes=5))

    with pytest.raises(XApiNotAuthorized, match="cannot be refreshed"):
        await api_client(thread_handler(), expired).thread(
            parse_x_post_url(f"https://x.com/i/status/{POST_ID}")
        )


# --------------------------------------------------------------------------- #
# Thread reading
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_thread_is_read_with_the_authors_replies_and_real_destinations() -> None:
    replies = [
        {
            "id": "1885026028428681699",
            "text": "Code here https://t.co/REPO",
            "author_id": "42",
            "conversation_id": POST_ID,
            "created_at": "2026-07-16T12:05:00.000Z",
            "entities": {"urls": [{"url": "https://t.co/REPO", "expanded_url": REPO_URL}]},
        }
    ]
    reader = ApiThreadReader(api_client(thread_handler(replies=replies), live_token()))

    resolved = await reader.read_thread(f"https://x.com/i/status/{POST_ID}")

    assert resolved.extraction_method == API_METHOD
    assert resolved.source_kind is SourceKind.X
    assert resolved.canonical_url == f"https://x.com/researcher/status/{POST_ID}"
    assert resolved.author == "Researcher"
    assert resolved.metadata["capture_scope"] == "author_thread"
    assert resolved.metadata["bundled_self_replies"] == 1
    # Shortlinks are replaced by the destinations, as the browser path does.
    assert resolved.outbound_urls == [PAPER_URL, REPO_URL]
    assert "t.co" not in resolved.text
    assert "Code here" in resolved.text


@pytest.mark.asyncio
async def test_a_long_form_body_is_preferred_over_the_truncated_text() -> None:
    root = {
        "id": POST_ID,
        "text": "Opening that the API truncates https://t.co/SHORT",
        "author_id": "42",
        "conversation_id": POST_ID,
        "created_at": "2026-07-16T12:00:00.000Z",
        "note_tweet": {
            "text": "The complete long-form body the API returns in full.",
            "entities": {"urls": [{"url": "https://t.co/SHORT", "expanded_url": PAPER_URL}]},
        },
        "entities": {"urls": [{"url": "https://t.co/SHORT", "expanded_url": PAPER_URL}]},
    }
    reader = ApiThreadReader(api_client(thread_handler(root=root), live_token()))

    resolved = await reader.read_thread(f"https://x.com/i/status/{POST_ID}")

    assert resolved.text == "The complete long-form body the API returns in full."
    assert resolved.partial is False


@pytest.mark.asyncio
async def test_replies_outside_the_thread_window_are_excluded() -> None:
    replies = [
        {
            "id": "999",
            "text": "An unrelated post days later",
            "author_id": "42",
            "conversation_id": POST_ID,
            "created_at": "2026-07-25T12:00:00.000Z",
        }
    ]
    reader = ApiThreadReader(api_client(thread_handler(replies=replies), live_token()))

    resolved = await reader.read_thread(f"https://x.com/i/status/{POST_ID}")

    assert resolved.metadata["bundled_self_replies"] == 0
    assert "unrelated" not in resolved.text


@pytest.mark.asyncio
async def test_a_failed_reply_search_still_returns_the_root_post() -> None:
    """`search/recent` reaches back seven days; an older thread is not a failure."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/2/tweets/search/recent":
            return httpx.Response(400, json={"title": "too old"})
        return thread_handler()(request)

    reader = ApiThreadReader(api_client(handler, live_token()))

    resolved = await reader.read_thread(f"https://x.com/i/status/{POST_ID}")

    assert resolved.metadata["bundled_self_replies"] == 0
    assert resolved.outbound_urls == [PAPER_URL]


# --------------------------------------------------------------------------- #
# Bookmarks
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_bookmarks_arrive_with_their_content_so_nothing_is_read_twice() -> None:
    """The response already carries each post; refetching would bill again."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/2/users/me":
            return httpx.Response(200, json={"data": {"id": "42", "username": "me"}})
        if request.url.path == "/2/users/42/bookmarks":
            assert "entities" in request.url.params["tweet.fields"]
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "111",
                            "author_id": "7",
                            "text": "A saved post https://t.co/SHORT",
                            "created_at": "2026-07-16T12:00:00.000Z",
                            "entities": {"urls": [{"url": "https://t.co/SHORT", "expanded_url": PAPER_URL}]},
                        },
                        {"id": "222", "author_id": "8", "text": "Another saved post"},
                    ],
                    "includes": {
                        "users": [
                            {"id": "7", "username": "Alpha", "name": "Alpha Author"},
                            {"id": "8", "username": "Beta"},
                        ]
                    },
                    "meta": {},
                },
            )
        raise AssertionError(f"unexpected path {request.url.path}")

    sources = await api_client(handler, live_token()).bookmarks()

    assert [source.canonical_url for source in sources] == [
        "https://x.com/alpha/status/111",
        "https://x.com/beta/status/222",
    ]
    # Content, destinations, and dates all arrive with the listing.
    assert sources[0].text == f"A saved post {PAPER_URL}"
    assert sources[0].outbound_urls == [PAPER_URL]
    assert sources[0].published_at is not None
    assert sources[0].author == "Alpha"
    assert sources[0].metadata["capture_scope"] == "root_post_only"
    # Every one is a shape the resolver already understands.
    assert all(parse_x_post_url(source.canonical_url) is not None for source in sources)


@pytest.mark.asyncio
async def test_bookmark_pagination_stops_at_the_requested_limit() -> None:
    pages = {
        None: {"data": [{"id": "1", "author_id": "7"}], "meta": {"next_token": "p2"}},
        "p2": {"data": [{"id": "2", "author_id": "7"}], "meta": {}},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/2/users/me":
            return httpx.Response(200, json={"data": {"id": "42"}})
        token = request.url.params.get("pagination_token")
        payload = dict(pages[token])
        payload["data"] = [{**row, "text": f"post {row['id']}"} for row in payload["data"]]
        payload["includes"] = {"users": [{"id": "7", "username": "Alpha"}]}
        return httpx.Response(200, json=payload)

    assert len(await api_client(handler, live_token()).bookmarks(limit=100)) == 2


def test_a_stored_token_survives_a_round_trip() -> None:
    token = live_token()

    restored = XApiToken.deserialize(token.serialize())

    assert restored is not None
    assert restored.access_token == token.access_token
    assert restored.refresh_token == token.refresh_token


def test_an_unreadable_stored_token_is_discarded_rather_than_crashing() -> None:
    assert XApiToken.deserialize("not json") is None
    assert XApiToken.deserialize(json.dumps({"access_token": "a"})) is None
