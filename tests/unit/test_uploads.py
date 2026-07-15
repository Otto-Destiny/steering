from __future__ import annotations

import io

import pytest
from pypdf import PdfWriter

from steering.domain.models import SourceKind
from steering.ingestion import uploads
from steering.ingestion.uploads import UnsupportedUploadError, resolve_upload


def pdf_upload_fixture() -> bytes:
    output = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_metadata({"/Title": "Uploaded Fixture"})
    writer.write(output)
    return output.getvalue()


class RecordingImageProvider:
    def __init__(self, result: str) -> None:
        self.result = result
        self.seen_content: bytes | None = None
        self.seen_mime: str | None = None

    async def understand_image(self, *, content: bytes, mime_type: str) -> str:
        self.seen_content = content
        self.seen_mime = mime_type
        return self.result


@pytest.mark.asyncio
async def test_pdf_upload_extracts_in_memory_without_retaining_binary() -> None:
    canary = b"BINARY_PDF_CANARY_NOT_TO_RETAIN"
    content = pdf_upload_fixture() + canary
    resolved = await resolve_upload(
        filename="../unsafe/fixture.pdf",
        content=content,
        mime_type="application/pdf",
    )

    serialized = resolved.model_dump_json().encode()
    assert resolved.source_kind is SourceKind.PDF
    assert resolved.title == "fixture.pdf"
    assert resolved.metadata["binary_retained"] is False
    assert resolved.metadata["page_count"] == 1
    assert canary not in serialized
    assert resolved.canonical_url.startswith("text://upload/")


@pytest.mark.asyncio
async def test_image_upload_is_disposable_and_requires_explicit_provider() -> None:
    content = b"\x89PNG\r\nIMAGE_BINARY_CANARY"
    with pytest.raises(UnsupportedUploadError, match="multimodal"):
        await resolve_upload(filename="diagram.png", content=content, mime_type="image/png")

    provider = RecordingImageProvider("The diagram compares two bounded context strategies.")
    resolved = await resolve_upload(
        filename="diagram.png",
        content=content,
        mime_type="image/png",
        image_provider=provider,
    )
    assert provider.seen_content == content
    assert provider.seen_mime == "image/png"
    assert resolved.metadata["binary_retained"] is False
    assert resolved.metadata["media_inclusion_decision"] == "include"
    assert content not in resolved.model_dump_json().encode()

    decorative = RecordingImageProvider("NO_TECHNICAL_CONTENT")
    skipped = await resolve_upload(
        filename="logo.webp",
        content=b"RIFF_IMAGE_CANARY",
        mime_type="image/webp",
        image_provider=decorative,
    )
    assert skipped.partial is True
    assert skipped.metadata["media_inclusion_decision"] == "skip_decorative_or_empty"


@pytest.mark.asyncio
async def test_document_upload_types_and_size_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(UnsupportedUploadError, match="empty"):
        await resolve_upload(filename="empty.txt", content=b"", mime_type="text/plain")

    monkeypatch.setattr(uploads, "MAX_UPLOAD_BYTES", 3)
    with pytest.raises(UnsupportedUploadError, match="20 MiB"):
        await resolve_upload(filename="large.txt", content=b"1234", mime_type="text/plain")
    monkeypatch.setattr(uploads, "MAX_UPLOAD_BYTES", 20 * 1024 * 1024)

    html = await resolve_upload(
        filename="page.html",
        content=b"<html><script>noise</script><body><h1>Guide</h1><p>Useful text.</p></body></html>",
        mime_type="text/html; charset=utf-8",
    )
    assert html.text == "Guide\nUseful text."
    assert html.extraction_method == "authorized_document_upload"

    data = await resolve_upload(
        filename="data.json",
        content=b'{"method": "bounded memory"}',
        mime_type="application/json",
    )
    assert data.text == '{"method": "bounded memory"}'

    with pytest.raises(UnsupportedUploadError, match="unsupported upload"):
        await resolve_upload(
            filename="archive.zip",
            content=b"PK fixture",
            mime_type="application/zip",
        )
    with pytest.raises(UnsupportedUploadError, match="PDF extraction failed"):
        await resolve_upload(
            filename="broken.pdf",
            content=b"not a pdf",
            mime_type="application/pdf",
        )
