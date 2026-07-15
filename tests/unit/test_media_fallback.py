from __future__ import annotations

from typing import cast

import httpx
import pytest

from steering.domain.models import ResolvedSource, SourceKind
from steering.ingestion.media_fallback import (
    MAX_IMAGE_BYTES,
    MAX_VISION_TEXT_CHARS,
    append_social_image_fallback,
)
from steering.ingestion.security import SafeFetcher


class RecordingFetcher:
    def __init__(
        self,
        *,
        content: bytes = b"IMAGE_BINARY_CANARY",
        content_type: str = "image/png",
        failure: Exception | None = None,
    ) -> None:
        self.content = content
        self.content_type = content_type
        self.failure = failure
        self.urls: list[str] = []

    async def get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
    ) -> tuple[bytes, httpx.Headers, str]:
        del headers, params
        self.urls.append(url)
        if self.failure is not None:
            raise self.failure
        return self.content, httpx.Headers({"content-type": self.content_type}), url + "?final=1"


class RecordingImageProvider:
    def __init__(self, result: str, *, failure: Exception | None = None) -> None:
        self.result = result
        self.failure = failure
        self.calls: list[tuple[bytes, str]] = []

    async def understand_image(self, *, content: bytes, mime_type: str) -> str:
        self.calls.append((content, mime_type))
        if self.failure is not None:
            raise self.failure
        return self.result


def social_source(
    *,
    kind: SourceKind = SourceKind.X,
    outbound_urls: list[str] | None = None,
    candidates: list[dict[str, str]] | None = None,
) -> ResolvedSource:
    image_url = "https://pbs.twimg.com/media/diagram.png"
    return ResolvedSource(
        canonical_url="https://x.com/researcher/status/1",
        source_kind=kind,
        title="Bundled thread",
        text="The author bundled technical context across the thread.",
        extraction_method="authorized_visible_browser",
        outbound_urls=outbound_urls or [],
        media_urls=[image_url],
        metadata={
            "bundled_self_replies": 3,
            "media_inclusion_candidates": candidates
            if candidates is not None
            else [
                {
                    "url": image_url,
                    "alt": "Architecture diagram comparing two agent memory strategies",
                }
            ],
            "media_decision": "consider",
        },
    )


def as_safe_fetcher(fetcher: RecordingFetcher) -> SafeFetcher:
    return cast(SafeFetcher, fetcher)


@pytest.mark.asyncio
async def test_non_social_source_is_not_touched() -> None:
    source = social_source(kind=SourceKind.WEBPAGE)
    fetcher = RecordingFetcher()
    provider = RecordingImageProvider("Useful diagram text")

    result = await append_social_image_fallback(
        source,
        fetcher=as_safe_fetcher(fetcher),
        image_provider=provider,
    )

    assert result is source
    assert fetcher.urls == []
    assert provider.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "strong_url",
    [
        "https://arxiv.org/abs/2601.00001",
        "https://github.com/example/agent-memory",
        "https://huggingface.co/example/model",
        "https://docs.example.org/guide",
        "https://example.org/paper.pdf",
    ],
)
async def test_strong_outbound_source_skips_image_without_fetching(strong_url: str) -> None:
    source = social_source(outbound_urls=[strong_url])
    fetcher = RecordingFetcher()
    provider = RecordingImageProvider("Useful diagram text")

    result = await append_social_image_fallback(
        source,
        fetcher=as_safe_fetcher(fetcher),
        image_provider=provider,
        stronger_source_available=True,
    )

    assert fetcher.urls == []
    assert provider.calls == []
    assert result.canonical_url == source.canonical_url
    assert result.text == source.text
    assert result.outbound_urls == [strong_url]
    assert result.media_urls == source.media_urls
    assert result.metadata == {"bundled_self_replies": 3, "media_decision": "skip_strong_source"}


@pytest.mark.asyncio
async def test_one_plausible_image_is_disposable_and_appended_once() -> None:
    first_url = "https://pbs.twimg.com/media/first.png"
    second_url = "https://pbs.twimg.com/media/second.png"
    source = social_source(
        candidates=[
            {"url": first_url, "alt": "Technical benchmark chart with accuracy and latency axes"},
            {"url": second_url, "alt": "A second technically plausible architecture diagram"},
        ]
    )
    fetcher = RecordingFetcher()
    provider = RecordingImageProvider("Model A reaches 91% accuracy at 120 ms latency.")

    result = await append_social_image_fallback(
        source,
        fetcher=as_safe_fetcher(fetcher),
        image_provider=provider,
    )

    assert fetcher.urls == [first_url]
    assert provider.calls == [(b"IMAGE_BINARY_CANARY", "image/png")]
    assert result.canonical_url == source.canonical_url
    assert result.text.startswith(source.text)
    assert "Model A reaches 91% accuracy" in result.text
    assert result.media_urls == source.media_urls
    assert result.metadata == {
        "bundled_self_replies": 3,
        "media_decision": "include_technical_image",
    }
    serialized = result.model_dump_json()
    assert "pbs.twimg.com" in serialized
    assert "IMAGE_BINARY_CANARY" not in serialized
    assert "?final=1" not in serialized


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content_type", "vision_result", "expected_decision", "provider_calls"),
    [
        ("text/html", "unused", "skip_unsupported_image", 0),
        ("image/png", "NO_TECHNICAL_CONTENT", "skip_no_technical_content", 1),
        ("image/png", "", "skip_no_technical_content", 1),
    ],
)
async def test_unsupported_or_empty_image_content_returns_sanitized_source(
    content_type: str,
    vision_result: str,
    expected_decision: str,
    provider_calls: int,
) -> None:
    source = social_source()
    fetcher = RecordingFetcher(content_type=content_type)
    provider = RecordingImageProvider(vision_result)

    result = await append_social_image_fallback(
        source,
        fetcher=as_safe_fetcher(fetcher),
        image_provider=provider,
    )

    assert result.text == source.text
    assert len(fetcher.urls) == 1
    assert len(provider.calls) == provider_calls
    assert result.media_urls == source.media_urls
    assert result.metadata["media_decision"] == expected_decision
    assert "media_inclusion_candidates" not in result.metadata


@pytest.mark.asyncio
async def test_failure_and_missing_provider_are_safe_and_do_not_retry() -> None:
    source = social_source()
    failed_fetcher = RecordingFetcher(failure=RuntimeError("upstream detail"))
    provider = RecordingImageProvider("unused")

    failed = await append_social_image_fallback(
        source,
        fetcher=as_safe_fetcher(failed_fetcher),
        image_provider=provider,
    )
    assert len(failed_fetcher.urls) == 1
    assert provider.calls == []
    assert failed.text == source.text
    assert failed.metadata["media_decision"] == "skip_image_failure"

    unused_fetcher = RecordingFetcher()
    no_provider = await append_social_image_fallback(
        source,
        fetcher=as_safe_fetcher(unused_fetcher),
        image_provider=None,
    )
    assert unused_fetcher.urls == []
    assert no_provider.metadata["media_decision"] == "skip_no_provider"

    provider_fetcher = RecordingFetcher()
    failed_provider = RecordingImageProvider("unused", failure=RuntimeError("provider detail"))
    provider_failure = await append_social_image_fallback(
        source,
        fetcher=as_safe_fetcher(provider_fetcher),
        image_provider=failed_provider,
    )
    assert len(provider_fetcher.urls) == 1
    assert len(failed_provider.calls) == 1
    assert provider_failure.text == source.text
    assert provider_failure.metadata["media_decision"] == "skip_image_failure"


@pytest.mark.asyncio
async def test_no_plausible_candidate_and_oversized_image_do_not_call_provider() -> None:
    decorative = social_source(
        candidates=[
            {
                "url": "https://pbs.twimg.com/media/avatar.png",
                "alt": "Profile photo avatar for the post author",
            }
        ]
    )
    unused_fetcher = RecordingFetcher()
    provider = RecordingImageProvider("unused")
    no_candidate = await append_social_image_fallback(
        decorative,
        fetcher=as_safe_fetcher(unused_fetcher),
        image_provider=provider,
    )
    assert unused_fetcher.urls == []
    assert provider.calls == []
    assert no_candidate.metadata["media_decision"] == "skip_no_candidate"

    oversized_fetcher = RecordingFetcher(content=b"x" * (MAX_IMAGE_BYTES + 1))
    oversized = await append_social_image_fallback(
        social_source(),
        fetcher=as_safe_fetcher(oversized_fetcher),
        image_provider=provider,
    )
    assert len(oversized_fetcher.urls) == 1
    assert provider.calls == []
    assert oversized.metadata["media_decision"] == "skip_unsupported_image"


@pytest.mark.asyncio
async def test_vision_text_is_bounded_before_single_extraction_input() -> None:
    source = social_source(kind=SourceKind.LINKEDIN)
    fetcher = RecordingFetcher(content_type="image/webp")
    provider = RecordingImageProvider("technical " * (MAX_VISION_TEXT_CHARS + 100))

    result = await append_social_image_fallback(
        source,
        fetcher=as_safe_fetcher(fetcher),
        image_provider=provider,
    )

    appended = result.text.split("[Technical content extracted from one attached image]\n", 1)[1]
    assert len(appended) <= MAX_VISION_TEXT_CHARS
    assert len(provider.calls) == 1
