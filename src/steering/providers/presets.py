from __future__ import annotations

from dataclasses import dataclass

from steering.domain.models import ProviderConfig


@dataclass(frozen=True, slots=True)
class ProviderPreset:
    provider_id: str
    base_url: str
    generation_model: str
    embedding_model: str
    embedding_dimension: int = 768

    def configuration(self, *, generation_model: str | None = None) -> ProviderConfig:
        return ProviderConfig(
            base_url=self.base_url,
            generation_model=generation_model or self.generation_model,
            embedding_model=self.embedding_model,
            embedding_dimension=self.embedding_dimension,
        )


# Reviewed against the providers' stable model catalogs on 2026-07-16. The
# OpenAI snapshot is pinned so a future alias update cannot silently change
# extraction behaviour. Gemini currently publishes its stable model as an alias.
GEMINI_PRESET = ProviderPreset(
    provider_id="gemini",
    base_url="https://generativelanguage.googleapis.com/v1beta",
    generation_model="gemini-3.5-flash",
    embedding_model="gemini-embedding-2",
)
OPENAI_PRESET = ProviderPreset(
    provider_id="openai",
    base_url="https://api.openai.com/v1",
    generation_model="gpt-5.4-mini-2026-03-17",
    embedding_model="text-embedding-3-large",
)

PROVIDER_PRESETS = {
    GEMINI_PRESET.provider_id: GEMINI_PRESET,
    OPENAI_PRESET.provider_id: OPENAI_PRESET,
}


def provider_preset(provider_id: str) -> ProviderPreset:
    normalized = provider_id.strip().lower()
    try:
        return PROVIDER_PRESETS[normalized]
    except KeyError:
        supported = ", ".join(sorted(PROVIDER_PRESETS))
        raise ValueError(f"unknown provider preset '{provider_id}'; choose {supported}") from None
