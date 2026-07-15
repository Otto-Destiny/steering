from __future__ import annotations

import pytest
from pydantic import ValidationError

from steering.domain.models import ResolvedSource, Snapshot, SourceKind


def test_source_and_snapshot_refuse_high_confidence_credentials_without_echoing() -> None:
    secret = "sk-proj-" + "A" * 40
    with pytest.raises(ValidationError) as source_error:
        ResolvedSource(
            canonical_url="https://example.org/source",
            source_kind=SourceKind.WEBPAGE,
            title="Unsafe source",
            text=f"Accidental credential: {secret}",
            extraction_method="fixture",
        )
    assert "refused before persistence" in str(source_error.value)
    assert secret not in str(source_error.value)

    with pytest.raises(ValidationError) as snapshot_error:
        Snapshot(
            artifact_id="art_one",
            source_url="https://example.org/source",
            content_hash="hash",
            mime_type="text/plain",
            text=f"Accidental credential: {secret}",
            extraction_method="fixture",
        )
    assert "refused before persistence" in str(snapshot_error.value)
    assert secret not in str(snapshot_error.value)


def test_credential_guard_allows_documentation_placeholders() -> None:
    source = ResolvedSource(
        canonical_url="https://example.org/docs",
        source_kind=SourceKind.DOCUMENTATION,
        title="Configuration guide",
        text="Set API_KEY=sk-example or use your-provider-token here.",
        extraction_method="fixture",
    )

    assert "sk-example" in source.text
