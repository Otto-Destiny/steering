from __future__ import annotations

from pathlib import Path
from typing import Literal

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
    # Signing in is always visible because a human types the credentials. Once the
    # session is stored, captures can replay it without a window, which is what
    # makes unattended batch runs possible.
    browser_headless: bool = False
    #: X API OAuth client id. Absent by default: the API is billed per read, so
    #: the free capture paths stay in charge unless a user opts in.
    x_api_client_id: str | None = None
    #: When to read an X post's replies.
    #: "when-needed" reads them only if the root post has no source worth
    #: following, or says the link is in the replies. "always" reads them
    #: whenever a post has replies. "never" keeps every capture to the root post.
    read_threads: Literal["when-needed", "always", "never"] = "when-needed"

    @property
    def database_file(self) -> Path:
        return Path(self.database_path).expanduser()
