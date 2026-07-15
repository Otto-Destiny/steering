from __future__ import annotations

import hashlib
import re
from pathlib import Path
from urllib.parse import quote

from bs4 import BeautifulSoup

from steering.domain.models import ResolvedSource, SourceKind
from steering.domain.protocols import ImageUnderstandingProvider
from steering.ingestion.pdf import extract_bounded_pdf

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
SUPPORTED_IMAGES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


class UnsupportedUploadError(ValueError):
    pass


async def resolve_upload(
    *,
    filename: str,
    content: bytes,
    mime_type: str,
    image_provider: ImageUnderstandingProvider | None = None,
) -> ResolvedSource:
    """Convert an authorized in-memory upload to disposable source text.

    The binary is never written or placed in the graph. Only its digest-backed
    source identifier and extracted text are returned for normal ingestion.
    """

    if not content:
        raise UnsupportedUploadError("upload is empty")
    if len(content) > MAX_UPLOAD_BYTES:
        raise UnsupportedUploadError("upload exceeds the 20 MiB limit")
    safe_name = Path(filename).name or "upload"
    digest = hashlib.sha256(content).hexdigest()
    canonical = f"text://upload/{digest}/{quote(safe_name)}"
    normalized_mime = mime_type.split(";", 1)[0].strip().lower()

    if normalized_mime == "application/pdf" or safe_name.lower().endswith(".pdf"):
        try:
            reader, extracted_pages = extract_bounded_pdf(content)
            pages = [f"[Page {number}]\n{page}" for number, page in enumerate(extracted_pages, start=1)]
        except Exception as exc:
            raise UnsupportedUploadError(f"PDF extraction failed ({type(exc).__name__})") from None
        text = "\n\n".join(pages).strip()
        return ResolvedSource(
            canonical_url=canonical,
            source_kind=SourceKind.PDF,
            title=safe_name,
            text=text,
            mime_type="application/pdf",
            extraction_method="authorized_pdf_upload_pypdf",
            partial=not bool(text),
            metadata={"content_sha256": digest, "page_count": len(reader.pages), "binary_retained": False},
        )

    if normalized_mime in SUPPORTED_IMAGES:
        if image_provider is None:
            raise UnsupportedUploadError(
                "image knowledge extraction requires a configured multimodal generation provider"
            )
        text = await image_provider.understand_image(content=content, mime_type=normalized_mime)
        useful = bool(text and text != "NO_TECHNICAL_CONTENT")
        return ResolvedSource(
            canonical_url=canonical,
            source_kind=SourceKind.TEXT,
            title=safe_name,
            text=text if useful else "No technically useful content was detected in the image.",
            mime_type=normalized_mime,
            extraction_method="authorized_multimodal_image_upload",
            partial=not useful,
            metadata={
                "content_sha256": digest,
                "binary_retained": False,
                "media_inclusion_decision": "include" if useful else "skip_decorative_or_empty",
            },
        )

    if normalized_mime in {"text/html", "application/xhtml+xml"}:
        decoded = content.decode("utf-8", errors="replace")
        soup = BeautifulSoup(decoded, "html.parser")
        for node in soup(["script", "style", "noscript", "svg"]):
            node.decompose()
        text = soup.get_text("\n", strip=True)
    elif normalized_mime.startswith("text/") or normalized_mime in {
        "application/json",
        "application/xml",
    }:
        text = content.decode("utf-8", errors="replace")
    else:
        raise UnsupportedUploadError(f"unsupported upload media type: {normalized_mime or 'unknown'}")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return ResolvedSource(
        canonical_url=canonical,
        source_kind=SourceKind.TEXT,
        title=safe_name,
        text=text,
        mime_type=normalized_mime or "text/plain",
        extraction_method="authorized_document_upload",
        partial=not bool(text),
        metadata={"content_sha256": digest, "binary_retained": False},
    )
