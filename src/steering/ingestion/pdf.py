from __future__ import annotations

import io

from pypdf import PdfReader

MAX_PDF_PAGES = 500
MAX_PDF_TEXT_CHARS = 2_000_000


class PdfExtractionLimitError(ValueError):
    pass


def extract_bounded_pdf(content: bytes) -> tuple[PdfReader, list[str]]:
    reader = PdfReader(io.BytesIO(content))
    if len(reader.pages) > MAX_PDF_PAGES:
        raise PdfExtractionLimitError("PDF exceeds the page limit")
    pages: list[str] = []
    total_chars = 0
    for page in reader.pages:
        text = page.extract_text() or ""
        total_chars += len(text)
        if total_chars > MAX_PDF_TEXT_CHARS:
            raise PdfExtractionLimitError("PDF exceeds the extracted-text limit")
        pages.append(text)
    return reader, pages
