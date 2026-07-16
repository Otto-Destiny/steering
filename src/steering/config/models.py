from __future__ import annotations

from pathlib import Path

from platformdirs import user_data_path
from pydantic import BaseModel, ConfigDict, Field

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
    generation_provider: str | None = None
    embedding_provider: str | None = None
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)

    @property
    def database_file(self) -> Path:
        return Path(self.database_path).expanduser()
