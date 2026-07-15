from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from steering.domain.models import ArtifactType, EvidenceCategory, RelationType


class ExtractionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExtractedClaim(ExtractionModel):
    text: str = Field(min_length=1)
    category: EvidenceCategory
    confidence: float = Field(ge=0.0, le=1.0)
    exact_quote: str = Field(min_length=1)
    source_index: int = Field(default=0, ge=0)
    locator: str = "document"


class ExtractedRelation(ExtractionModel):
    predicate: RelationType
    target_name: str = Field(min_length=1)
    target_type: str = "concept"
    rationale: str | None = None
    exact_quote: str | None = None
    source_index: int = Field(default=0, ge=0)


class ReportedResult(ExtractionModel):
    method_or_model_version: str | None = None
    dataset_or_benchmark: str | None = None
    metric: str | None = None
    result: str | None = None
    experimental_conditions: str | None = None
    exact_quote: str = Field(min_length=1)
    source_index: int = Field(default=0, ge=0)


class ExtractedIssue(ExtractionModel):
    social_statement: str
    source_statement: str
    explanation: str
    social_source_url: str
    primary_source_url: str
    social_exact_quote: str = Field(min_length=1)
    primary_exact_quote: str = Field(min_length=1)


class KnowledgeExtraction(ExtractionModel):
    artifact_type: ArtifactType
    title: str = Field(min_length=1)
    short_name: str | None = None
    summary: str = Field(min_length=1)
    strategy_family: str = "uncategorized"
    maturity: str | None = None
    license: str | None = None
    aliases: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    use_cases: list[str] = Field(default_factory=list)
    claims: list[ExtractedClaim] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
    reported_results: list[ReportedResult] = Field(default_factory=list)
    issues: list[ExtractedIssue] = Field(default_factory=list)
