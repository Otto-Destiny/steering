from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from steering.ingestion.service import ThreadPolicy


class WebInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SearchInput(WebInput):
    query: str = Field(min_length=1, max_length=4_000)
    limit: int = Field(default=10, ge=1, le=50)
    breadth: bool = False
    project_id: str | None = Field(default=None, max_length=200)
    #: Sources with no known publication date are excluded once this is set, so
    #: it is left unset unless the user asks for it.
    published_after: datetime | None = None


class DesignInput(WebInput):
    architecture: str = Field(min_length=1, max_length=50_000)
    requirements: str = Field(default="", max_length=20_000)
    concerns: list[str] = Field(default_factory=list, max_length=30)
    project_id: str | None = Field(default=None, max_length=200)
    limit: int = Field(default=12, ge=1, le=12)


class IngestionInput(WebInput):
    source: str | None = Field(default=None, max_length=250_000)
    sources: list[str] = Field(default_factory=list, max_length=1_000)
    threads: ThreadPolicy = ThreadPolicy.AUTO
    #: Capture everything publicly first, then read threads only where one is
    #: still missing. Free pass first, browser time only where it is earned.
    two_pass: bool = False

    @model_validator(mode="after")
    def require_content(self) -> IngestionInput:
        if not self.source and not self.sources:
            raise ValueError("provide source or sources")
        return self


class RetireInput(WebInput):
    artifact_ids: list[str] = Field(min_length=1, max_length=500)


class BrowserCaptureInput(WebInput):
    url: str = Field(min_length=1, max_length=2_000)
    authorized: bool


class IssueResolutionInput(WebInput):
    action: Literal["accept_correction", "keep_both", "dismiss", "reject"]


class ProviderInput(WebInput):
    provider_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    role: Literal["generation", "embedding", "both"] = "both"
    base_url: str = Field(min_length=1, max_length=2_000)
    generation_model: str | None = Field(default=None, max_length=500)
    embedding_model: str | None = Field(default=None, max_length=500)
    embedding_dimension: int = Field(default=768, ge=8, le=65_536)
    generation_api_key: SecretStr | None = None
    embedding_api_key: SecretStr | None = None

    @model_validator(mode="after")
    def require_models_for_selected_roles(self) -> ProviderInput:
        if self.role in {"generation", "both"} and not self.generation_model:
            raise ValueError("generation_model is required for the generation role")
        if self.role in {"embedding", "both"} and not self.embedding_model:
            raise ValueError("embedding_model is required for the embedding role")
        return self


class ProjectInput(WebInput):
    name: str = Field(min_length=1, max_length=300)
    description: str | None = Field(default=None, max_length=10_000)
    constraints: list[str] = Field(default_factory=list, max_length=100)


class DecisionInput(WebInput):
    project: str = Field(min_length=1, max_length=300)
    artifact_id: str | None = Field(default=None, max_length=200)
    decision: str = Field(min_length=1, max_length=20_000)
    rationale: str = Field(min_length=1, max_length=20_000)


class OutcomeInput(WebInput):
    project_id: str = Field(min_length=1, max_length=200)
    outcome: str = Field(min_length=1, max_length=20_000)
    artifact_id: str | None = Field(default=None, max_length=200)
    decision_id: str | None = Field(default=None, max_length=200)
    constraints: list[str] = Field(default_factory=list, max_length=100)
    succeeded: bool | None = None


class BackupInput(WebInput):
    destination: str = Field(min_length=1, max_length=4_000)
