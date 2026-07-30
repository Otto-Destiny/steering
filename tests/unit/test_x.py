"""X capture tests.

Every fixture here mirrors the shape X actually returns, which matters more than
usual for this resolver: X wraps *every* outbound link in a ``t.co`` shortlink and
exposes the real destination only through ``entities.urls[].expanded_url``. A
fixture that inlines a direct ``https://arxiv.org/...`` href would let a resolver
that never unwraps shortlinks pass, which is exactly how the primary-source
follow-through regressed before. ``test_fixtures_wrap_every_link_like_x_does``
locks that property in.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from steering.domain.models import SourceKind
from steering.ingestion.resolvers import WebResolver, default_registry
from steering.ingestion.security import SafeFetcher, SourceUnavailableError
from steering.ingestion.service import FOLLOW_PRIORITY_FLOOR, _source_priority
from steering.ingestion.x import (
    OEMBED_METHOD,
    ROOT_POST_ONLY,
    SYNDICATION_METHOD,
    XResolver,
    _links_in_text,
    parse_x_post_url,
    syndication_token,
)

POST_ID = "1885026028428681698"
PAPER_URL = "https://arxiv.org/abs/2501.12948"
REPO_URL = "https://github.com/example/agent-memory"


class AllowPublicFixtureGuard:
    async def validate_url(self, url: str) -> str:
        return url

    def validate_connected_address(self, address: str) -> None:
        del address


def fetcher_for(handler: Any) -> SafeFetcher:
    return SafeFetcher(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        guard=AllowPublicFixtureGuard(),  # type: ignore[arg-type]
    )


def syndication_payload(**overrides: Any) -> dict[str, Any]:
    """A standard post whose links are shortened exactly as X shortens them."""

    payload: dict[str, Any] = {
        "__typename": "Tweet",
        "id_str": POST_ID,
        "lang": "en",
        "created_at": "2025-01-30T18:03:21.000Z",
        "text": (
            "New work on agent memory with explicit decay.\n\n"
            "Paper: https://t.co/PAPERSHORT\nCode: https://t.co/REPOSHORT"
        ),
        "user": {"screen_name": "Researcher", "name": "A Researcher", "id_str": "42"},
        "entities": {
            "urls": [
                {
                    "url": "https://t.co/PAPERSHORT",
                    "expanded_url": PAPER_URL,
                    "display_url": "arxiv.org/abs/2501.12948",
                    "indices": [46, 69],
                },
                {
                    "url": "https://t.co/REPOSHORT",
                    "expanded_url": REPO_URL,
                    "display_url": "github.com/example/agent-…",
                    "indices": [76, 99],
                },
            ]
        },
        "conversation_count": 12,
    }
    payload.update(overrides)
    return payload


def oembed_payload(html: str) -> dict[str, Any]:
    return {
        "url": f"https://x.com/Researcher/status/{POST_ID}",
        "author_name": "A Researcher",
        "author_url": "https://x.com/Researcher",
        "html": html,
        "provider_name": "X",
        "version": "1.0",
    }


def syndication_only(payload: dict[str, Any]) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "cdn.syndication.twimg.com"
        assert request.url.params["id"] == POST_ID
        return httpx.Response(200, json=payload)

    return handler


# --------------------------------------------------------------------------- #
# URL handling (X-4, X-5)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "source",
    [
        f"https://x.com/Researcher/status/{POST_ID}",
        f"https://x.com/researcher/status/{POST_ID}?s=46&t=AbCdEf",
        f"https://x.com/researcher/status/{POST_ID}/photo/1",
        f"https://x.com/researcher/status/{POST_ID}/video/1",
        f"https://twitter.com/researcher/status/{POST_ID}",
        f"https://www.twitter.com/researcher/statuses/{POST_ID}",
        f"https://mobile.twitter.com/RESEARCHER/status/{POST_ID}/",
        f"https://m.x.com/researcher/status/{POST_ID}",
    ],
)
def test_every_post_url_shape_reduces_to_one_canonical_url(source: str) -> None:
    reference = parse_x_post_url(source)

    assert reference is not None
    assert reference.post_id == POST_ID
    assert reference.canonical_url() == f"https://x.com/researcher/status/{POST_ID}"


def test_anonymous_post_urls_parse_without_an_author() -> None:
    for source in (f"https://x.com/i/web/status/{POST_ID}", f"https://x.com/i/status/{POST_ID}"):
        reference = parse_x_post_url(source)
        assert reference is not None
        assert reference.post_id == POST_ID
        assert reference.handle is None


@pytest.mark.parametrize(
    "source",
    [
        "https://x.com/researcher",
        "https://x.com/",
        "https://x.com/search?q=agents",
        "https://x.com/i/lists/12345",
        "https://example.com/researcher/status/123",
    ],
)
def test_non_post_urls_are_not_parsed_as_posts(source: str) -> None:
    assert parse_x_post_url(source) is None


def test_syndication_token_is_deterministic_and_url_safe() -> None:
    token = syndication_token(POST_ID)

    assert token == syndication_token(POST_ID)
    assert token and token.isalnum()
    assert "0" not in token and "." not in token


# --------------------------------------------------------------------------- #
# Syndication capture (X-1, X-3)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_syndication_capture_unwraps_shortlinks_into_followable_sources() -> None:
    fetcher = fetcher_for(syndication_only(syndication_payload()))

    resolved = await XResolver(fetcher).resolve(f"https://x.com/i/web/status/{POST_ID}?s=20")

    assert resolved.extraction_method == SYNDICATION_METHOD
    assert resolved.source_kind is SourceKind.X
    # The author handle is recovered from the payload even though the URL omitted it.
    assert resolved.canonical_url == f"https://x.com/researcher/status/{POST_ID}"
    assert resolved.author == "Researcher"
    assert resolved.published_at is not None
    assert resolved.published_at.year == 2025
    # Destinations, not shortlinks: this is what makes primary-source follow work.
    assert resolved.outbound_urls == [PAPER_URL, REPO_URL]
    assert "t.co" not in resolved.text
    assert PAPER_URL in resolved.text
    assert resolved.partial is False


@pytest.mark.asyncio
async def test_unwrapped_sources_outrank_the_follow_floor() -> None:
    """A capture is only useful downstream if its links clear the follow threshold."""

    fetcher = fetcher_for(syndication_only(syndication_payload()))

    resolved = await XResolver(fetcher).resolve(f"https://x.com/researcher/status/{POST_ID}")

    assert [_source_priority(url) for url in resolved.outbound_urls] == [100, 95]


@pytest.mark.asyncio
async def test_long_form_posts_are_marked_partial_with_actionable_advice() -> None:
    payload = syndication_payload(note_tweet={"id": "Tm90ZVR3ZWV0"})
    fetcher = fetcher_for(syndication_only(payload))

    resolved = await XResolver(fetcher, recover_long_form=False).resolve(
        f"https://x.com/researcher/status/{POST_ID}"
    )

    assert resolved.partial is True
    assert resolved.metadata["long_form_truncated"] is True
    assert "signed-in capture" in resolved.metadata["capture_advice"]


@pytest.mark.asyncio
async def test_public_capture_declares_that_it_read_the_root_post_only() -> None:
    """X publishes one post at a time; a reply-borne link is silently missing otherwise."""

    fetcher = fetcher_for(syndication_only(syndication_payload(conversation_count=9)))

    resolved = await XResolver(fetcher).resolve(f"https://x.com/researcher/status/{POST_ID}")

    assert resolved.metadata["capture_scope"] == ROOT_POST_ONLY
    assert resolved.metadata["reply_count"] == 9
    advice = resolved.metadata["capture_advice"]
    assert "public capture reads the root post only" in advice
    assert "signed-in capture" in advice


@pytest.mark.asyncio
async def test_a_post_with_no_replies_carries_no_thread_advice() -> None:
    fetcher = fetcher_for(syndication_only(syndication_payload(conversation_count=0)))

    resolved = await XResolver(fetcher).resolve(f"https://x.com/researcher/status/{POST_ID}")

    assert resolved.metadata["capture_scope"] == ROOT_POST_ONLY
    assert "capture_advice" not in resolved.metadata


@pytest.mark.asyncio
async def test_long_form_and_thread_limits_are_reported_together() -> None:
    payload = syndication_payload(note_tweet={"id": "Tm90ZVR3ZWV0"}, conversation_count=4)
    fetcher = fetcher_for(syndication_only(payload))

    resolved = await XResolver(fetcher, recover_long_form=False).resolve(
        f"https://x.com/researcher/status/{POST_ID}"
    )

    advice = resolved.metadata["capture_advice"]
    assert "long-form" in advice
    assert "self-reply" in advice


@pytest.mark.asyncio
async def test_media_alt_text_becomes_an_inclusion_candidate() -> None:
    payload = syndication_payload(
        mediaDetails=[
            {
                "media_url_https": "https://pbs.twimg.com/media/described.jpg",
                "type": "photo",
                "ext_alt_text": "Benchmark table comparing retrieval latency across four systems",
            },
            {"media_url_https": "https://pbs.twimg.com/media/plain.jpg", "type": "photo"},
        ]
    )
    fetcher = fetcher_for(syndication_only(payload))

    resolved = await XResolver(fetcher).resolve(f"https://x.com/researcher/status/{POST_ID}")

    assert resolved.media_urls == [
        "https://pbs.twimg.com/media/described.jpg",
        "https://pbs.twimg.com/media/plain.jpg",
    ]
    assert resolved.metadata["media_inclusion_candidates"] == [
        {
            "url": "https://pbs.twimg.com/media/described.jpg",
            "alt": "Benchmark table comparing retrieval latency across four systems",
        }
    ]


@pytest.mark.asyncio
async def test_quoted_post_text_and_links_are_folded_into_the_capture() -> None:
    payload = syndication_payload(
        text="This is the result I have been waiting for.",
        entities={},
        quoted_tweet={
            "__typename": "Tweet",
            "text": "Releasing our paper https://t.co/QUOTESHORT",
            "user": {"screen_name": "labaccount", "name": "Lab"},
            "entities": {"urls": [{"url": "https://t.co/QUOTESHORT", "expanded_url": PAPER_URL}]},
        },
    )
    fetcher = fetcher_for(syndication_only(payload))

    resolved = await XResolver(fetcher).resolve(f"https://x.com/researcher/status/{POST_ID}")

    assert "[Quoted post by @labaccount]" in resolved.text
    assert resolved.outbound_urls == [PAPER_URL]


@pytest.mark.asyncio
async def test_media_shortlinks_are_stripped_rather_than_left_in_the_body() -> None:
    payload = syndication_payload(
        text="Findings below https://t.co/MEDIASHORT",
        entities={
            "media": [
                {
                    "url": "https://t.co/MEDIASHORT",
                    "expanded_url": f"https://x.com/researcher/status/{POST_ID}/photo/1",
                }
            ]
        },
    )
    fetcher = fetcher_for(syndication_only(payload))

    resolved = await XResolver(fetcher).resolve(f"https://x.com/researcher/status/{POST_ID}")

    assert resolved.text == "Findings below"
    assert resolved.outbound_urls == []


@pytest.mark.asyncio
async def test_unavailable_posts_fail_cleanly_instead_of_storing_a_shell() -> None:
    payload = {
        "__typename": "TweetTombstone",
        "tombstone": {"text": {"text": "This Post was deleted by the Post author."}},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "cdn.syndication.twimg.com":
            return httpx.Response(200, json=payload)
        return httpx.Response(404, text="not found")

    with pytest.raises(SourceUnavailableError, match="deleted by the Post author"):
        await XResolver(fetcher_for(handler)).resolve(f"https://x.com/researcher/status/{POST_ID}")


@pytest.mark.asyncio
async def test_profile_urls_report_an_actionable_error_and_never_capture_a_login_wall() -> None:
    """A profile URL used to reach WebResolver and store X's login chrome as knowledge."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"no network call should be made, got {request.url}")

    with pytest.raises(SourceUnavailableError, match="does not address a single post"):
        await XResolver(fetcher_for(handler)).resolve("https://x.com/researcher")


# --------------------------------------------------------------------------- #
# oEmbed fallback (X-1 on the degraded path)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_oembed_fallback_unwraps_shortlinks_through_guarded_redirects() -> None:
    html = (
        '<blockquote class="twitter-tweet"><p lang="en" dir="ltr">New work on agent memory '
        '<a href="https://t.co/PAPERSHORT">arxiv.org/abs/2501.1294…</a></p>'
        f'&mdash; A Researcher (@Researcher) <a href="https://x.com/Researcher/status/{POST_ID}'
        '?ref_src=twsrc%5Etfw">January 30, 2025</a></blockquote>'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "cdn.syndication.twimg.com":
            return httpx.Response(503, text="unavailable")
        if request.url.host == "publish.x.com":
            return httpx.Response(200, json=oembed_payload(html))
        if request.url.host == "t.co":
            return httpx.Response(301, headers={"location": PAPER_URL})
        if request.url.host == "arxiv.org":
            return httpx.Response(200, text="paper landing page")
        raise AssertionError(f"unexpected host {request.url.host}")

    resolved = await XResolver(fetcher_for(handler)).resolve(f"https://x.com/researcher/status/{POST_ID}")

    assert resolved.extraction_method == OEMBED_METHOD
    assert resolved.canonical_url == f"https://x.com/researcher/status/{POST_ID}"
    assert resolved.outbound_urls == [PAPER_URL]
    assert resolved.partial is True
    assert resolved.metadata["capture_degraded"] is True


@pytest.mark.asyncio
async def test_both_public_paths_failing_surfaces_the_oembed_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "cdn.syndication.twimg.com":
            return httpx.Response(404, text="missing")
        return httpx.Response(200, json={"author_name": "no html key"})

    with pytest.raises(SourceUnavailableError, match="invalid payload"):
        await XResolver(fetcher_for(handler)).resolve(f"https://x.com/researcher/status/{POST_ID}")


# --------------------------------------------------------------------------- #
# Registry routing and fixture realism
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "source",
    [
        f"https://x.com/researcher/status/{POST_ID}",
        "https://x.com/researcher",
        "https://mobile.twitter.com/researcher/status/1",
    ],
)
def test_every_x_url_is_claimed_by_the_x_resolver_never_the_web_resolver(source: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    registry = default_registry(fetcher_for(handler))
    resolver = registry.resolver_for(source)

    assert isinstance(resolver, XResolver)
    assert not isinstance(resolver, WebResolver)


def test_fixtures_wrap_every_link_like_x_does() -> None:
    """Guard the fixtures themselves against drifting away from X's real shape."""

    payload = syndication_payload()
    body = json.dumps(payload["text"])

    assert "t.co" in body
    assert PAPER_URL not in body
    assert REPO_URL not in body
    assert [item["expanded_url"] for item in payload["entities"]["urls"]] == [PAPER_URL, REPO_URL]


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://arxiv.org/abs/2501.12948", 100),
        ("https://export.arxiv.org/abs/2501.12948", 100),
        ("https://example.org/paper.pdf", 98),
        ("https://github.com/example/repo", 95),
        ("https://huggingface.co/example/model", 90),
        ("https://docs.example.com/guide", 80),
        # A hostile URL must not borrow a trusted host's rank through substring matching.
        ("https://attacker.example/?ref=arxiv.org", 20),
        ("https://arxiv.org.attacker.example/abs/1", 20),
        ("https://t.co/SHORTLINK", 20),
    ],
)
def test_source_priority_ranks_by_host_not_substring(url: str, expected: int) -> None:
    assert _source_priority(url) == expected


# --------------------------------------------------------------------------- #
# Long-form recovery
# --------------------------------------------------------------------------- #

FULL_LONG_FORM = (
    "Opening that X truncates. " + "Body that only the long-form endpoint returns. " * 20
).strip()


def long_form_handler(*, available: bool = True, body: str = FULL_LONG_FORM) -> Any:
    payload = syndication_payload(
        text="Opening that X truncates.",
        entities={},
        note_tweet={"id": "Tm90ZVR3ZWV0"},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "cdn.syndication.twimg.com":
            return httpx.Response(200, json=payload)
        if request.url.host == "api.fxtwitter.com":
            if not available:
                return httpx.Response(503, text="unavailable")
            # FixTweet refuses a client that does not name itself. Answering 200
            # regardless made this double more capable than the real service and
            # hid a 403 that silently cost every long-form body.
            agent = request.headers.get("user-agent", "")
            if not agent or agent.startswith("python-httpx"):
                return httpx.Response(403, text="forbidden")
            return httpx.Response(200, json={"tweet": {"text": body, "is_note_tweet": True}})
        raise AssertionError(f"unexpected host {request.url.host}")

    return handler


@pytest.mark.asyncio
async def test_long_form_body_is_recovered_and_the_post_stops_being_partial() -> None:
    """X truncates long-form posts everywhere and exposes the remainder nowhere."""

    resolver = XResolver(fetcher_for(long_form_handler()))

    resolved = await resolver.resolve(f"https://x.com/researcher/status/{POST_ID}")

    assert resolved.text == FULL_LONG_FORM
    assert resolved.partial is False
    assert resolved.metadata["long_form_truncated"] is False


@pytest.mark.asyncio
async def test_recovery_can_be_switched_off_and_the_truncation_is_then_declared() -> None:
    resolver = XResolver(fetcher_for(long_form_handler()), recover_long_form=False)

    resolved = await resolver.resolve(f"https://x.com/researcher/status/{POST_ID}")

    assert resolved.text == "Opening that X truncates."
    assert resolved.partial is True
    assert "long-form" in resolved.metadata["capture_advice"]


@pytest.mark.asyncio
async def test_a_failed_recovery_keeps_the_honest_truncated_capture() -> None:
    """Trading a declared limit for a silent one would be the worse outcome."""

    resolver = XResolver(fetcher_for(long_form_handler(available=False)))

    resolved = await resolver.resolve(f"https://x.com/researcher/status/{POST_ID}")

    assert resolved.text == "Opening that X truncates."
    assert resolved.partial is True
    assert resolved.metadata["long_form_truncated"] is True


@pytest.mark.asyncio
async def test_a_shorter_recovery_result_is_rejected() -> None:
    resolver = XResolver(fetcher_for(long_form_handler(body="tiny")))

    resolved = await resolver.resolve(f"https://x.com/researcher/status/{POST_ID}")

    assert resolved.text == "Opening that X truncates."
    assert resolved.partial is True


@pytest.mark.asyncio
async def test_a_standard_post_never_calls_the_recovery_endpoint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "cdn.syndication.twimg.com":
            return httpx.Response(200, json=syndication_payload())
        raise AssertionError("a post X returned in full needs no recovery call")

    resolved = await XResolver(fetcher_for(handler)).resolve(f"https://x.com/researcher/status/{POST_ID}")

    assert resolved.partial is False


LONG_FORM_WITH_LINK = (
    "Opening that X truncates. A tool worth reading about, described at length "
    "so the recovered body is longer than the opening.\n\nProject address::\n"
    "https://github.com/example/project"
)


@pytest.mark.asyncio
async def test_links_written_in_a_recovered_long_form_body_become_followable() -> None:
    """A long-form payload carries no URL entities at all.

    ``entities.urls`` arrives null and ``note_tweet`` is reduced to a bare id, so
    once the body is recovered the prose is the only surviving record of where the
    author pointed. Recovering the text without the links left the destination
    readable but never visited.
    """

    resolver = XResolver(fetcher_for(long_form_handler(body=LONG_FORM_WITH_LINK)))

    resolved = await resolver.resolve(f"https://x.com/researcher/status/{POST_ID}")

    assert resolved.outbound_urls == ["https://github.com/example/project"]
    assert _source_priority(resolved.outbound_urls[0]) >= FOLLOW_PRIORITY_FLOOR


@pytest.mark.asyncio
async def test_long_form_recovery_names_the_tool_rather_than_being_refused() -> None:
    """The default client agent is rejected outright, losing the body in silence."""

    agents: list[str] = []
    inner = long_form_handler(body=LONG_FORM_WITH_LINK)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.fxtwitter.com":
            agents.append(request.headers.get("user-agent", ""))
        return inner(request)

    resolved = await XResolver(fetcher_for(handler)).resolve(f"https://x.com/researcher/status/{POST_ID}")

    assert agents and agents[0].startswith("STEERING/")
    assert resolved.partial is False


@pytest.mark.asyncio
async def test_a_refused_recovery_keeps_the_honest_truncation_instead_of_guessing() -> None:
    resolver = XResolver(fetcher_for(long_form_handler(available=False)))

    resolved = await resolver.resolve(f"https://x.com/researcher/status/{POST_ID}")

    assert resolved.partial is True
    assert resolved.outbound_urls == []
    assert "long-form" in resolved.metadata["capture_advice"]


def test_body_links_drop_trailing_prose_punctuation() -> None:
    text = "See https://github.com/example/project. Also https://arxiv.org/abs/2401.00001, twice."

    assert _links_in_text(text) == [
        "https://github.com/example/project",
        "https://arxiv.org/abs/2401.00001",
    ]
