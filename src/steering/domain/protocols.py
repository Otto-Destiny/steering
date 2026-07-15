from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from steering.domain.models import (
    ArchitectureReview,
    ArtifactRecord,
    Decision,
    ExperimentOutcome,
    IngestionJob,
    Project,
    ResolvedSource,
    ReviewIssue,
    SearchHit,
    SearchQuery,
)

TModel = TypeVar("TModel", bound=BaseModel)


class SourceResolver(Protocol):
    name: str

    def can_resolve(self, source: str) -> bool: ...

    async def resolve(self, source: str) -> ResolvedSource: ...


class GenerationProvider(Protocol):
    model_id: str

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[TModel],
    ) -> TModel: ...

    async def test_connection(self) -> None: ...


class EmbeddingProvider(Protocol):
    model_id: str
    dimension: int

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def test_connection(self) -> None: ...


class ImageUnderstandingProvider(Protocol):
    async def understand_image(self, *, content: bytes, mime_type: str) -> str: ...


class ArtifactRepository(Protocol):
    def upsert_record(self, record: ArtifactRecord) -> ArtifactRecord: ...

    def get_record(self, artifact_id: str) -> ArtifactRecord | None: ...

    def get_by_url(self, canonical_url: str) -> ArtifactRecord | None: ...

    def list_records(self) -> list[ArtifactRecord]: ...

    def save_job(self, job: IngestionJob) -> None: ...

    def list_jobs(self, limit: int = 100) -> list[IngestionJob]: ...

    def list_issues(self, unresolved_only: bool = False) -> list[ReviewIssue]: ...

    def resolve_issue(self, issue_id: str, action: str) -> ReviewIssue: ...

    def save_project(self, project: Project) -> Project: ...

    def list_projects(self) -> list[Project]: ...

    def save_decision(self, decision: Decision) -> Decision: ...

    def save_outcome(self, outcome: ExperimentOutcome) -> ExperimentOutcome: ...

    def project_history(self, project_id: str) -> Mapping[str, Sequence[Any]]: ...

    def backup(self, destination: str) -> str: ...


class KnowledgeRetriever(Protocol):
    async def search(self, query: SearchQuery) -> list[SearchHit]: ...

    async def review_architecture(
        self,
        architecture: str,
        requirements: str = "",
        concerns: Sequence[str] | None = None,
    ) -> ArchitectureReview: ...
