from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

from steering.domain.credentials import (
    reject_credential_bearing_source,
    reject_high_confidence_credentials,
)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def utc_now() -> datetime:
    return datetime.now(UTC)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, hide_input_in_errors=True)


class ArtifactType(StrEnum):
    PAPER = "paper"
    OPEN_SOURCE_TOOL = "open_source_tool"
    SOCIAL_POST = "social_post"
    TECHNIQUE = "technique"
    REPOSITORY = "repository"
    MODEL = "model"
    DATASET = "dataset"
    DOCUMENTATION = "documentation"
    WEBPAGE = "webpage"
    NOTE = "note"
    UNKNOWN = "unknown"


class SourceKind(StrEnum):
    X = "x"
    LINKEDIN = "linkedin"
    GITHUB = "github"
    PAPER = "paper"
    PDF = "pdf"
    DOCUMENTATION = "documentation"
    WEBPAGE = "webpage"
    TEXT = "text"
    TELEGRAM = "telegram"


class ReviewStatus(StrEnum):
    CAPTURED = "captured"
    REVIEWED = "reviewed"
    REJECTED = "rejected"
    STALE = "stale"


class TrustLane(StrEnum):
    ESTABLISHED = "established"
    RECENT = "recent"
    PROMISING = "promising"
    EXPERIMENTAL = "experimental"
    DEPRECATED_OR_INCOMPATIBLE = "deprecated_or_incompatible"


class EvidenceCategory(StrEnum):
    MAINTAINER_DOCUMENTATION = "maintainer_documentation"
    RESEARCH_PAPER = "research_paper"
    BENCHMARK = "benchmark"
    SOCIAL_CLAIM = "social_claim"
    USER_OBSERVATION = "user_observation"
    MODEL_INFERENCE = "model_inference"


class RelationType(StrEnum):
    SOLVES = "solves"
    IMPLEMENTS = "implements"
    ALTERNATIVE_TO = "alternative_to"
    REQUIRES = "requires"
    INTEGRATES_WITH = "integrates_with"
    LIMITED_BY = "limited_by"
    INTRODUCES = "introduces"
    EVALUATES = "evaluates"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    IMPROVES = "improves"
    APPLIES_TO = "applies_to"
    EXTENDS = "extends"
    SUPERSEDES = "supersedes"
    DEPRECATED_BY = "deprecated_by"
    RECOMMENDED_OVER = "recommended_over"
    MENTIONS = "mentions"
    DERIVED_FROM = "derived_from"
    RELATED_TO = "related_to"


SENSITIVE_RELATIONS = {
    RelationType.SUPERSEDES,
    RelationType.DEPRECATED_BY,
    RelationType.RECOMMENDED_OVER,
}


class IssueStatus(StrEnum):
    UNRESOLVED = "unresolved"
    ACCEPTED_CORRECTION = "accepted_correction"
    KEPT_BOTH = "kept_both"
    DISMISSED = "dismissed"
    REJECTED = "rejected"


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"


class Artifact(StrictModel):
    id: str = Field(default_factory=lambda: new_id("art"))
    canonical_url: str | None = None
    source_kind: SourceKind
    artifact_type: ArtifactType = ArtifactType.UNKNOWN
    title: str
    short_name: str | None = None
    summary: str
    strategy_family: str = "uncategorized"
    review_status: ReviewStatus = ReviewStatus.CAPTURED
    trust_lane: TrustLane = TrustLane.EXPERIMENTAL
    evidence_quality: float = Field(default=0.3, ge=0.0, le=1.0)
    maturity: str | None = None
    license: str | None = None
    aliases: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    use_cases: list[str] = Field(default_factory=list)
    published_at: datetime | None = None
    source_updated_at: datetime | None = None
    discovered_at: datetime = Field(default_factory=utc_now)
    captured_at: datetime = Field(default_factory=utc_now)
    last_verified_at: datetime | None = None
    deprecated_at: datetime | None = None
    content_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("canonical_url")
    @classmethod
    def canonical_url_is_http(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith(("https://", "http://", "text://")):
            raise ValueError("canonical_url must use http, https, or text scheme")
        return reject_credential_bearing_source(value) if value is not None else None

    @field_validator("trust_lane")
    @classmethod
    def unreviewed_claims_are_experimental(cls, value: TrustLane, info: Any) -> TrustLane:
        status = info.data.get("review_status")
        if status == ReviewStatus.CAPTURED and value == TrustLane.ESTABLISHED:
            raise ValueError("captured artifacts cannot enter the established lane")
        return value


class Snapshot(StrictModel):
    id: str = Field(default_factory=lambda: new_id("snap"))
    artifact_id: str
    source_url: str
    captured_at: datetime = Field(default_factory=utc_now)
    content_hash: str
    mime_type: str
    text: str
    extraction_method: str
    partial: bool = False

    _reject_credentials = field_validator("text")(reject_high_confidence_credentials)
    _reject_source_credentials = field_validator("source_url")(reject_credential_bearing_source)


class Chunk(StrictModel):
    id: str = Field(default_factory=lambda: new_id("chunk"))
    artifact_id: str
    snapshot_id: str
    ordinal: int = Field(ge=0)
    text: str
    locator: str
    embedding: list[float] = Field(default_factory=list)
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_revision: str | None = None
    embedding_dimension: int | None = Field(default=None, ge=1)
    embedding_task_mode: str | None = None
    embedding_normalized: bool | None = None
    source_content_hash: str | None = None

    @model_validator(mode="after")
    def embedding_dimension_matches_vector(self) -> Chunk:
        if (
            self.embedding
            and self.embedding_dimension is not None
            and len(self.embedding) != self.embedding_dimension
        ):
            raise ValueError("embedding dimension does not match the stored vector")
        return self


class Claim(StrictModel):
    id: str = Field(default_factory=lambda: new_id("claim"))
    artifact_id: str
    text: str
    category: EvidenceCategory
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_span_ids: list[str] = Field(default_factory=list)


class EvidenceSpan(StrictModel):
    id: str = Field(default_factory=lambda: new_id("span"))
    snapshot_id: str
    claim_id: str
    quote: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    locator: str

    @field_validator("end")
    @classmethod
    def end_after_start(cls, value: int, info: Any) -> int:
        if value <= int(info.data.get("start", -1)):
            raise ValueError("evidence span end must be after start")
        return value


class Relation(StrictModel):
    id: str = Field(default_factory=lambda: new_id("rel"))
    subject_id: str
    predicate: RelationType
    object_id: str
    approved: bool = False
    evidence_span_ids: list[str] = Field(default_factory=list)
    rationale: str | None = None

    @model_validator(mode="after")
    def sensitive_relations_need_evidence(self) -> Relation:
        if self.approved and self.predicate in SENSITIVE_RELATIONS and not self.evidence_span_ids:
            raise ValueError("sensitive relations require explicit evidence")
        return self


class ReviewIssue(StrictModel):
    id: str = Field(default_factory=lambda: new_id("issue"))
    artifact_id: str
    social_statement: str
    source_statement: str
    explanation: str
    social_source_url: str
    primary_source_url: str
    evidence_span_ids: list[str] = Field(default_factory=list)
    status: IssueStatus = IssueStatus.UNRESOLVED
    created_at: datetime = Field(default_factory=utc_now)
    resolved_at: datetime | None = None

    _reject_source_credentials = field_validator("social_source_url", "primary_source_url")(
        reject_credential_bearing_source
    )


class Entity(StrictModel):
    id: str = Field(default_factory=lambda: new_id("entity"))
    name: str
    entity_type: str
    aliases: list[str] = Field(default_factory=list)
    canonical_entity_id: str | None = None


class Concept(StrictModel):
    id: str = Field(default_factory=lambda: new_id("concept"))
    name: str
    description: str | None = None


class Mention(StrictModel):
    id: str = Field(default_factory=lambda: new_id("mention"))
    artifact_id: str
    entity_id: str
    evidence_span_id: str | None = None


class Project(StrictModel):
    id: str = Field(default_factory=lambda: new_id("project"))
    name: str
    description: str | None = None
    constraints: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class Decision(StrictModel):
    id: str = Field(default_factory=lambda: new_id("decision"))
    project_id: str
    artifact_id: str | None = None
    decision: str
    rationale: str
    created_at: datetime = Field(default_factory=utc_now)


class ExperimentOutcome(StrictModel):
    id: str = Field(default_factory=lambda: new_id("outcome"))
    project_id: str
    decision_id: str | None = None
    artifact_id: str | None = None
    outcome: str
    constraints: list[str] = Field(default_factory=list)
    succeeded: bool | None = None
    recorded_at: datetime = Field(default_factory=utc_now)


class ReviewRun(StrictModel):
    id: str = Field(default_factory=lambda: new_id("review"))
    project_id: str | None = None
    query: str
    result_artifact_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class IngestionJob(StrictModel):
    id: str = Field(default_factory=lambda: new_id("job"))
    source: str
    status: JobStatus = JobStatus.PENDING
    artifact_id: str | None = None
    error_code: str | None = None
    safe_error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ArtifactRecord(StrictModel):
    artifact: Artifact
    snapshots: list[Snapshot] = Field(default_factory=list)
    chunks: list[Chunk] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    issues: list[ReviewIssue] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    concepts: list[Concept] = Field(default_factory=list)

    @model_validator(mode="after")
    def unresolved_issues_are_not_established(self) -> ArtifactRecord:
        if self.artifact.trust_lane == TrustLane.ESTABLISHED and any(
            issue.status == IssueStatus.UNRESOLVED for issue in self.issues
        ):
            raise ValueError("artifacts with unresolved issues cannot enter the established lane")
        return self


class SearchQuery(StrictModel):
    query: str = Field(min_length=1)
    artifact_types: list[ArtifactType] | None = None
    concepts: list[str] | None = None
    minimum_evidence: float | None = Field(default=None, ge=0.0, le=1.0)
    published_after: datetime | None = None
    limit: int = Field(default=10, ge=1, le=50)
    breadth: bool = False
    project_id: str | None = None


class ScoreBreakdown(StrictModel):
    exact: float = 0.0
    bm25: float = 0.0
    vector: float = 0.0
    graph: float = 0.0
    project_history: float = 0.0
    rrf: float = 0.0
    constraint_fit: float = 0.0
    evidence: float = 0.0


class SearchHit(StrictModel):
    artifact: Artifact
    score: float
    scores: ScoreBreakdown
    matched_chunks: list[Chunk] = Field(default_factory=list)
    citation_urls: list[str] = Field(default_factory=list)
    uncertainty: str | None = None


class IdeaEvidence(StrictModel):
    claim: str
    exact_quote: str
    source_url: str
    snapshot_id: str
    evidence_span_id: str


class IdeaCard(StrictModel):
    artifact_id: str
    title: str
    what_it_is: str
    why_it_fits: str
    trust_lane: TrustLane
    source_url: str | None
    published_at: datetime | None
    advantages: list[str]
    limitations: list[str]
    compatibility_requirements: list[str]
    minimal_experiment: str
    success_criteria: str
    failure_criteria: str
    related_alternatives: list[str]
    uncertainty: str | None
    strategy_family: str
    evidence: list[IdeaEvidence] = Field(default_factory=list)


class ArchitectureReview(StrictModel):
    query: str
    decomposed_areas: list[str]
    observations: list[str]
    idea_cards: list[IdeaCard]
    insufficient_knowledge: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)


class ResolvedSource(StrictModel):
    canonical_url: str
    source_kind: SourceKind
    title: str
    text: str
    author: str | None = None
    published_at: datetime | None = None
    mime_type: str = "text/plain"
    extraction_method: str
    partial: bool = False
    outbound_urls: list[str] = Field(default_factory=list)
    media_urls: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    _reject_credentials = field_validator("text")(reject_high_confidence_credentials)
    _reject_source_credentials = field_validator("canonical_url")(reject_credential_bearing_source)


class ProviderConfig(StrictModel):
    base_url: HttpUrl
    generation_model: str | None = None
    embedding_model: str | None = None
    embedding_dimension: int = Field(default=768, ge=8, le=65536)
    api_key_fingerprint: str | None = None

    @field_validator("base_url", mode="before")
    @classmethod
    def base_url_has_no_userinfo(cls, value: Any) -> Any:
        parsed = urlsplit(str(value))
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("provider base URL must not contain user information")
        reject_credential_bearing_source(str(value))
        return value
