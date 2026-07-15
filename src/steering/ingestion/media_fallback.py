from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from steering.domain.models import ResolvedSource, SourceKind
from steering.domain.protocols import ImageUnderstandingProvider
from steering.ingestion.security import SafeFetcher

SUPPORTED_IMAGE_MIME_TYPES = frozenset({"image/jpeg", "image/png", "image/webp", "image/gif"})
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_VISION_TEXT_CHARS = 4_000

_DECORATIVE_HINTS = ("avatar", "emoji", "headshot", "logo", "profile photo")


def _candidate(source: ResolvedSource) -> tuple[str, str] | None:
    candidates = source.metadata.get("media_inclusion_candidates", [])
    if not isinstance(candidates, list):
        return None
    for item in candidates:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url", "")).strip()
        alt = str(item.get("alt", "")).strip()
        parsed = urlsplit(url)
        if (
            parsed.scheme in {"http", "https"}
            and parsed.hostname
            and len(alt) >= 24
            and not any(hint in alt.lower() for hint in _DECORATIVE_HINTS)
        ):
            return url, alt
    return None


def _sanitized(source: ResolvedSource, decision: str) -> ResolvedSource:
    metadata: dict[str, Any] = {
        key: value
        for key, value in source.metadata.items()
        if key not in {"media_inclusion_candidates", "media_decision"}
    }
    metadata["media_decision"] = decision
    return source.model_copy(
        update={
            "metadata": metadata,
        }
    )


def _bounded_vision_text(value: str) -> str:
    compact = value.strip()
    if len(compact) <= MAX_VISION_TEXT_CHARS:
        return compact
    boundary = compact.rfind(" ", 0, MAX_VISION_TEXT_CHARS)
    return compact[: boundary if boundary > 0 else MAX_VISION_TEXT_CHARS].rstrip()


async def append_social_image_fallback(
    source: ResolvedSource,
    *,
    fetcher: SafeFetcher,
    image_provider: ImageUnderstandingProvider | None,
    stronger_source_available: bool = False,
) -> ResolvedSource:
    """Append useful content from at most one disposable social image.

    Call this after self-replies are bundled and before selecting the single
    extraction request. Image bytes are discarded; the source URL and compact
    inclusion decision remain as provenance.
    """

    if source.source_kind not in {SourceKind.X, SourceKind.LINKEDIN}:
        return source
    if stronger_source_available:
        return _sanitized(source, "skip_strong_source")
    candidate = _candidate(source)
    if candidate is None:
        return _sanitized(source, "skip_no_candidate")
    if image_provider is None:
        return _sanitized(source, "skip_no_provider")

    image_url, _alt = candidate
    try:
        content, headers, _final_url = await fetcher.get(image_url)
        mime_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if mime_type not in SUPPORTED_IMAGE_MIME_TYPES or len(content) > MAX_IMAGE_BYTES:
            return _sanitized(source, "skip_unsupported_image")
        vision_text = _bounded_vision_text(
            await image_provider.understand_image(content=content, mime_type=mime_type)
        )
        if not vision_text or vision_text == "NO_TECHNICAL_CONTENT":
            return _sanitized(source, "skip_no_technical_content")
        sanitized = _sanitized(source, "include_technical_image")
        return sanitized.model_copy(
            update={
                "text": (
                    f"{sanitized.text.rstrip()}\n\n"
                    "[Technical content extracted from one attached image]\n"
                    f"{vision_text}"
                ).strip()
            }
        )
    except Exception:
        return _sanitized(source, "skip_image_failure")
