from __future__ import annotations

from pathlib import Path

from platformdirs import user_data_path
from pydantic import BaseModel, ConfigDict, Field, model_validator

from steering.domain.models import ProviderConfig


def default_database_path() -> str:
    return str(user_data_path("steering", appauthor=False) / "steering.lbug")


class AppConfig(BaseModel):
    """Serializable application settings. Secrets are deliberately absent."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    database_path: str = Field(default_factory=default_database_path)
    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1, le=65535)
    log_level: str = "INFO"
    generation_context_window_tokens: int = Field(default=32_000, ge=2_048, le=10_000_000)
    generation_reserved_output_tokens: int = Field(default=4_000, ge=256, le=1_000_000)
    generation_provider: str | None = None
    embedding_provider: str | None = None
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)

    @property
    def database_file(self) -> Path:
        return Path(self.database_path).expanduser()

    @model_validator(mode="after")
    def output_budget_fits_context(self) -> AppConfig:
        if self.generation_reserved_output_tokens >= self.generation_context_window_tokens:
            raise ValueError("generation output budget must be smaller than the context window")
        return self
