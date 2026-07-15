"""Normalize the reviewed evaluation YAML corpus into domain records."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from hashlib import blake2b
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

from steering.domain.models import (
    Artifact,
    ArtifactRecord,
    ArtifactType,
    Chunk,
    Claim,
    Concept,
    Entity,
    EvidenceCategory,
    EvidenceSpan,
    IssueStatus,
    Relation,
    RelationType,
    ReviewIssue,
    ReviewStatus,
    Snapshot,
    SourceKind,
    TrustLane,
)

EVALUATION_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


def stable_digest(*parts: object, digest_size: int = 20) -> str:
    """Return a stable BLAKE2 digest for fixture identity and content."""
    payload = "\x00".join(str(part) for part in parts).encode("utf-8")
    return blake2b(payload, digest_size=digest_size, person=b"steering-eval").hexdigest()


def stable_id(prefix: str, *parts: object) -> str:
    return f"{prefix}_{stable_digest(*parts, digest_size=12)}"


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: object) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _text(value: object, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def _parse_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    text = str(value).strip()
    if not text:
        return None
    if len(text) == 4 and text.isdigit():
        return datetime(int(text), 1, 1, tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _artifact_type(category: str) -> ArtifactType:
    return {
        "paper": ArtifactType.PAPER,
        "open_source_tool": ArtifactType.OPEN_SOURCE_TOOL,
        "tip_or_trick": ArtifactType.TECHNIQUE,
        "post_only": ArtifactType.SOCIAL_POST,
    }.get(category, ArtifactType.UNKNOWN)


def _source_kind(url: str, selected_type: str) -> SourceKind:
    host = urlsplit(url).netloc.lower()
    if host == "x.com" or host.endswith(".x.com"):
        return SourceKind.X
    if host == "linkedin.com" or host.endswith(".linkedin.com"):
        return SourceKind.LINKEDIN
    if host == "github.com" or host.endswith(".github.com"):
        return SourceKind.GITHUB
    if "arxiv.org" in host or "openreview.net" in host or "paper" in selected_type:
        return SourceKind.PAPER
    if "pdf" in selected_type:
        return SourceKind.PDF
    if "documentation" in selected_type or "docs" in host:
        return SourceKind.DOCUMENTATION
    return SourceKind.WEBPAGE


def _confidence_score(evidence_quality: Mapping[str, Any]) -> float:
    confidence = _text(evidence_quality.get("confidence")).lower()
    if confidence == "low":
        score = 0.2
    elif confidence == "medium":
        score = 0.6
    elif confidence == "medium_high":
        score = 0.7
    elif confidence.startswith("high"):
        score = 0.8
    else:
        score = 0.3
    if evidence_quality.get("independently_validated") is True:
        score = max(score, 0.9)
    return score


def _trust_lane(*, category: str, evidence_quality: Mapping[str, Any], unresolved_issue: bool) -> TrustLane:
    confidence = _text(evidence_quality.get("confidence")).lower()
    if category == "post_only" or confidence == "low":
        return TrustLane.EXPERIMENTAL
    if evidence_quality.get("independently_validated") is True and not unresolved_issue:
        return TrustLane.ESTABLISHED
    if not evidence_quality:
        return TrustLane.EXPERIMENTAL
    return TrustLane.PROMISING


def _evidence_category(category: str, *, inferred: bool) -> EvidenceCategory:
    if inferred:
        return EvidenceCategory.MODEL_INFERENCE
    if category == "paper":
        return EvidenceCategory.RESEARCH_PAPER
    if category == "post_only":
        return EvidenceCategory.SOCIAL_CLAIM
    return EvidenceCategory.MAINTAINER_DOCUMENTATION


def _relation_type(predicate: str) -> RelationType:
    normalized = predicate.strip().lower()
    direct = {
        "supports": RelationType.SUPPORTS,
        "extends": RelationType.EXTENDS,
        "integrates_with": RelationType.INTEGRATES_WITH,
        "requires": RelationType.REQUIRES,
        "introduces": RelationType.INTRODUCES,
        "evaluates": RelationType.EVALUATES,
        "evaluated_on": RelationType.EVALUATES,
        "improves": RelationType.IMPROVES,
        "applies": RelationType.APPLIES_TO,
        "applies_to": RelationType.APPLIES_TO,
        "implements": RelationType.IMPLEMENTS,
    }
    return direct.get(normalized, RelationType.RELATED_TO)


@dataclass(frozen=True)
class _EvidenceInput:
    claim_text: str
    quote: str
    locator: str
    inferred: bool = False


class _SnapshotTextBuilder:
    def __init__(self) -> None:
        self._parts: list[str] = []
        self._length = 0

    def append(self, text: str) -> tuple[int, int]:
        start = self._length
        self._parts.append(text)
        self._length += len(text)
        return start, self._length

    def text(self) -> str:
        return "".join(self._parts)


@dataclass(frozen=True)
class EvaluationCorpus:
    records: tuple[ArtifactRecord, ...]

    @property
    def by_candidate_id(self) -> dict[str, ArtifactRecord]:
        return {str(record.artifact.metadata["candidate_id"]): record for record in self.records}

    @property
    def by_artifact_id(self) -> dict[str, ArtifactRecord]:
        return {record.artifact.id: record for record in self.records}

    @property
    def canonical_groups(self) -> dict[str, tuple[str, ...]]:
        groups: dict[str, list[str]] = {}
        for record in self.records:
            url = record.artifact.canonical_url or record.artifact.id
            groups.setdefault(url, []).append(str(record.artifact.metadata["candidate_id"]))
        return {url: tuple(sorted(ids)) for url, ids in groups.items()}


class EvaluationCorpusLoader:
    """Load evaluation result fixtures without accessing the network."""

    def __init__(
        self,
        *,
        results_directory: Path,
        categories_path: Path,
        strategy_families_path: Path,
    ) -> None:
        self.results_directory = results_directory
        self.categories_path = categories_path
        self.strategy_families_path = strategy_families_path

    def load(self) -> EvaluationCorpus:
        categories = self._load_categories()
        families = self._load_families()
        result_paths = sorted(self.results_directory.glob("*.yaml"))
        candidate_ids = {path.stem for path in result_paths}
        if candidate_ids != set(categories):
            raise ValueError("Result categories must cover exactly the result files")
        if candidate_ids != set(families):
            raise ValueError("Strategy families must cover exactly the result files")
        records = tuple(
            self._record_from_payload(
                candidate_id=path.stem,
                payload=_mapping(yaml.safe_load(path.read_text(encoding="utf-8"))),
                category=categories[path.stem],
                strategy_family=families[path.stem],
            )
            for path in result_paths
        )
        return EvaluationCorpus(records=records)

    def _load_categories(self) -> dict[str, str]:
        payload = _mapping(yaml.safe_load(self.categories_path.read_text(encoding="utf-8")))
        category_groups = _mapping(payload.get("categories"))
        result: dict[str, str] = {}
        for category, candidate_ids in category_groups.items():
            for candidate_id in _strings(candidate_ids):
                if candidate_id in result:
                    raise ValueError(f"Duplicate category assignment: {candidate_id}")
                result[candidate_id] = str(category)
        return result

    def _load_families(self) -> dict[str, str]:
        payload = _mapping(yaml.safe_load(self.strategy_families_path.read_text(encoding="utf-8")))
        families = _mapping(payload.get("families"))
        return {str(candidate_id): _text(family) for candidate_id, family in families.items()}

    def _record_from_payload(
        self,
        *,
        candidate_id: str,
        payload: Mapping[str, Any],
        category: str,
        strategy_family: str,
    ) -> ArtifactRecord:
        if _text(payload.get("candidate_id")) != candidate_id:
            raise ValueError(f"Candidate mismatch in {candidate_id}")

        ingestion = _mapping(payload.get("ingestion"))
        selected = _mapping(ingestion.get("selected_artifact"))
        knowledge = _mapping(payload.get("knowledge"))
        artifact_data = _mapping(knowledge.get("artifact"))
        engineering = _mapping(knowledge.get("engineering_relevance"))
        evidence_quality = _mapping(knowledge.get("evidence_quality"))
        adoption = _mapping(knowledge.get("adoption_readiness"))
        reconciliation = _mapping(payload.get("source_reconciliation"))
        raw_issues = [item for item in _list(reconciliation.get("issues")) if isinstance(item, Mapping)]
        unresolved = any(_text(item.get("status")) == "needs_review" for item in raw_issues)

        canonical_url = _text(selected.get("canonical_url"))
        if not canonical_url.startswith(("https://", "http://", "text://")):
            raise ValueError(f"Missing canonical URL for {candidate_id}")
        selected_type = _text(selected.get("type"))
        title = _text(artifact_data.get("title"), candidate_id)
        summary = _text(knowledge.get("concise_idea"))
        if not summary:
            raise ValueError(f"Missing concise idea for {candidate_id}")
        published_at = _parse_datetime(artifact_data.get("published_at"))
        evidence_score = _confidence_score(evidence_quality)
        lane = _trust_lane(
            category=category,
            evidence_quality=evidence_quality,
            unresolved_issue=unresolved,
        )
        artifact_id = stable_id("art", candidate_id)
        snapshot_id = stable_id("snap", candidate_id, canonical_url)
        evidence_inputs = self._evidence_inputs(knowledge, summary)
        snapshot_text, span_offsets = self._snapshot_text(
            candidate_id=candidate_id,
            title=title,
            canonical_url=canonical_url,
            knowledge=knowledge,
            evidence_inputs=evidence_inputs,
        )
        snapshot_hash = stable_digest(snapshot_text)

        snapshot = Snapshot(
            id=snapshot_id,
            artifact_id=artifact_id,
            source_url=canonical_url,
            captured_at=EVALUATION_EPOCH,
            content_hash=snapshot_hash,
            mime_type="text/yaml; profile=steering-evaluation-fixture",
            text=snapshot_text,
            extraction_method="reviewed_evaluation_fixture_normalization",
            partial="partial_social_capture" in _text(payload.get("evaluation_status")),
        )

        claims: list[Claim] = []
        spans: list[EvidenceSpan] = []
        for index, evidence_input in enumerate(evidence_inputs):
            claim_id = stable_id("claim", candidate_id, index, evidence_input.claim_text)
            span_id = stable_id("span", candidate_id, index, evidence_input.quote)
            start, end = span_offsets[index]
            spans.append(
                EvidenceSpan(
                    id=span_id,
                    snapshot_id=snapshot_id,
                    claim_id=claim_id,
                    quote=evidence_input.quote,
                    start=start,
                    end=end,
                    locator=evidence_input.locator,
                )
            )
            claims.append(
                Claim(
                    id=claim_id,
                    artifact_id=artifact_id,
                    text=evidence_input.claim_text,
                    category=_evidence_category(category, inferred=evidence_input.inferred),
                    confidence=min(evidence_score, 0.35) if evidence_input.inferred else evidence_score,
                    evidence_span_ids=[span_id],
                )
            )

        chunks = self._chunks(
            candidate_id=candidate_id,
            artifact_id=artifact_id,
            snapshot_id=snapshot_id,
            title=title,
            knowledge=knowledge,
            graph_ready=_mapping(payload.get("graph_ready")),
        )
        entities, concepts, relations = self._graph_objects(
            candidate_id=candidate_id,
            artifact_id=artifact_id,
            graph_ready=_mapping(payload.get("graph_ready")),
            evidence_span_ids=[span.id for span in spans[:1]],
        )
        issues = self._issues(
            candidate_id=candidate_id,
            artifact_id=artifact_id,
            ingestion=ingestion,
            reconciliation=reconciliation,
            raw_issues=raw_issues,
            evidence_span_ids=[span.id for span in spans],
        )

        license_value = _text(adoption.get("license")) or _text(artifact_data.get("license")) or None
        aliases = [
            value
            for value in {_text(artifact_data.get("short_name")), candidate_id}
            if value and value != title
        ]
        artifact = Artifact(
            id=artifact_id,
            canonical_url=canonical_url,
            source_kind=_source_kind(canonical_url, selected_type),
            artifact_type=_artifact_type(category),
            title=title,
            short_name=_text(artifact_data.get("short_name")) or None,
            summary=summary,
            strategy_family=strategy_family,
            review_status=ReviewStatus.REVIEWED,
            trust_lane=lane,
            evidence_quality=evidence_score,
            maturity=_text(artifact_data.get("maturity")) or None,
            license=license_value,
            aliases=sorted(aliases),
            capabilities=_strings(knowledge.get("capabilities")),
            limitations=_strings(knowledge.get("limitations")),
            requirements=_strings(knowledge.get("requirements")),
            use_cases=_strings(engineering.get("applicability")),
            published_at=published_at,
            discovered_at=EVALUATION_EPOCH,
            captured_at=EVALUATION_EPOCH,
            last_verified_at=EVALUATION_EPOCH,
            content_hash=snapshot_hash,
            metadata={
                "candidate_id": candidate_id,
                "primary_category": category,
                "evaluation_status": _text(payload.get("evaluation_status")),
                "selected_artifact_type": selected_type,
                "implementation_available": adoption.get("implementation_available"),
                "minimum_validation": _text(adoption.get("minimum_validation")),
                "adoption_posture": _text(engineering.get("adoption_posture")),
                "independently_validated": evidence_quality.get("independently_validated"),
                "fixture_snapshot": True,
            },
        )
        return ArtifactRecord(
            artifact=artifact,
            snapshots=[snapshot],
            chunks=chunks,
            claims=claims,
            evidence_spans=spans,
            relations=relations,
            issues=issues,
            entities=entities,
            concepts=concepts,
        )

    def _evidence_inputs(self, knowledge: Mapping[str, Any], summary: str) -> list[_EvidenceInput]:
        raw_evidence = knowledge.get("evidence")
        result: list[_EvidenceInput] = []
        for item in _list(raw_evidence):
            if isinstance(item, str) and item.strip():
                text = item.strip()
                result.append(_EvidenceInput(claim_text=text, quote=text, locator="fixture evidence"))
                continue
            evidence = _mapping(item)
            if not evidence:
                continue
            claim_text = _text(evidence.get("claim"))
            result_text = _text(evidence.get("result"))
            context = _text(evidence.get("context")) or _text(evidence.get("scope"))
            quote_parts = [part for part in (claim_text, result_text, context) if part]
            if not quote_parts:
                continue
            result.append(
                _EvidenceInput(
                    claim_text=claim_text or result_text,
                    quote=" ".join(quote_parts),
                    locator=_text(evidence.get("source_location"))
                    or _text(evidence.get("source"))
                    or "fixture evidence",
                )
            )
        if not result:
            result.append(
                _EvidenceInput(
                    claim_text=summary,
                    quote=summary,
                    locator="normalized concise idea",
                    inferred=True,
                )
            )
        return result

    def _snapshot_text(
        self,
        *,
        candidate_id: str,
        title: str,
        canonical_url: str,
        knowledge: Mapping[str, Any],
        evidence_inputs: list[_EvidenceInput],
    ) -> tuple[str, list[tuple[int, int]]]:
        builder = _SnapshotTextBuilder()
        builder.append(f"Candidate: {candidate_id}\n")
        builder.append(f"Title: {title}\n")
        builder.append(f"Canonical source: {canonical_url}\n")
        builder.append(f"Summary: {_text(knowledge.get('concise_idea'))}\n")
        builder.append(f"Problem: {_text(knowledge.get('problem_addressed'))}\n")
        for heading, value in (
            ("Method", knowledge.get("method")),
            ("Capabilities", knowledge.get("capabilities")),
            ("Requirements", knowledge.get("requirements")),
            ("Limitations", knowledge.get("limitations")),
        ):
            builder.append(f"{heading}:\n")
            for item in _strings(value):
                builder.append(f"- {item}\n")
        offsets: list[tuple[int, int]] = []
        builder.append("Evidence:\n")
        for index, evidence_input in enumerate(evidence_inputs, start=1):
            builder.append(f"[{index}] ")
            offsets.append(builder.append(evidence_input.quote))
            builder.append(f"\nLocator: {evidence_input.locator}\n")
        return builder.text(), offsets

    def _chunks(
        self,
        *,
        candidate_id: str,
        artifact_id: str,
        snapshot_id: str,
        title: str,
        knowledge: Mapping[str, Any],
        graph_ready: Mapping[str, Any],
    ) -> list[Chunk]:
        engineering = _mapping(knowledge.get("engineering_relevance"))
        sections: list[tuple[str, str]] = [
            ("identity", title),
            ("summary", _text(knowledge.get("concise_idea"))),
            ("problem", _text(knowledge.get("problem_addressed"))),
            ("method", "\n".join(_strings(knowledge.get("method")))),
            ("capabilities", "\n".join(_strings(knowledge.get("capabilities")))),
            ("requirements", "\n".join(_strings(knowledge.get("requirements")))),
            ("limitations", "\n".join(_strings(knowledge.get("limitations")))),
            (
                "engineering_relevance",
                "\n".join(
                    part
                    for part in (
                        _text(engineering.get("applicability")),
                        _text(engineering.get("not_a_fit_for")),
                    )
                    if part
                ),
            ),
            ("topics", " ".join(_strings(graph_ready.get("topics")))),
        ]
        chunks: list[Chunk] = []
        for ordinal, (locator, text) in enumerate(item for item in sections if item[1]):
            chunks.append(
                Chunk(
                    id=stable_id("chunk", candidate_id, ordinal, locator, text),
                    artifact_id=artifact_id,
                    snapshot_id=snapshot_id,
                    ordinal=ordinal,
                    text=text,
                    locator=locator,
                )
            )
        return chunks

    def _graph_objects(
        self,
        *,
        candidate_id: str,
        artifact_id: str,
        graph_ready: Mapping[str, Any],
        evidence_span_ids: list[str],
    ) -> tuple[list[Entity], list[Concept], list[Relation]]:
        primary = _mapping(graph_ready.get("primary_entity"))
        raw_primary_id = _text(primary.get("id")) or f"artifact:{candidate_id}"
        entity_id = stable_id("entity", candidate_id, raw_primary_id)
        entity = Entity(
            id=entity_id,
            name=_text(primary.get("name"), candidate_id),
            entity_type=_text(primary.get("type"), "artifact"),
            aliases=[candidate_id],
        )
        concept_ids: dict[str, str] = {}

        def concept_id(name: str) -> str:
            if name not in concept_ids:
                concept_ids[name] = stable_id("concept", name)
            return concept_ids[name]

        for topic in _strings(graph_ready.get("topics")):
            concept_id(topic)
        relations: list[Relation] = []
        for index, raw_relation in enumerate(_list(graph_ready.get("relations"))):
            relation = _mapping(raw_relation)
            subject = _text(relation.get("subject"))
            object_name = _text(relation.get("object"))
            predicate = _text(relation.get("predicate"))
            if not subject or not object_name or not predicate:
                continue
            subject_id = entity_id if subject == raw_primary_id else concept_id(subject)
            object_id = entity_id if object_name == raw_primary_id else concept_id(object_name)
            relations.append(
                Relation(
                    id=stable_id("rel", candidate_id, index, subject, predicate, object_name),
                    subject_id=subject_id,
                    predicate=_relation_type(predicate),
                    object_id=object_id,
                    approved=True,
                    evidence_span_ids=evidence_span_ids,
                    rationale=f"Evaluation fixture relation: {predicate}",
                )
            )
        concepts = [
            Concept(id=concept_id_value, name=name) for name, concept_id_value in sorted(concept_ids.items())
        ]
        return [entity], concepts, relations

    def _issues(
        self,
        *,
        candidate_id: str,
        artifact_id: str,
        ingestion: Mapping[str, Any],
        reconciliation: Mapping[str, Any],
        raw_issues: Iterable[Mapping[str, Any]],
        evidence_span_ids: list[str],
    ) -> list[ReviewIssue]:
        social_url = _text(ingestion.get("social_source"))
        primary_url = _text(reconciliation.get("authoritative_source"))
        issues: list[ReviewIssue] = []
        for index, issue in enumerate(raw_issues):
            raw_status = _text(issue.get("status"))
            status = {
                "needs_review": IssueStatus.UNRESOLVED,
                "corrected_from_repository": IssueStatus.ACCEPTED_CORRECTION,
                "excluded": IssueStatus.DISMISSED,
            }.get(raw_status, IssueStatus.UNRESOLVED)
            source_statement = (
                _text(issue.get("source_statement"))
                or _text(issue.get("artifact_statement"))
                or _text(issue.get("finding"))
                or "No source statement was retained."
            )
            issues.append(
                ReviewIssue(
                    id=stable_id("issue", candidate_id, index),
                    artifact_id=artifact_id,
                    social_statement=_text(issue.get("social_statement"), "No social statement retained."),
                    source_statement=source_statement,
                    explanation=_text(issue.get("explanation"), source_statement),
                    social_source_url=social_url or primary_url,
                    primary_source_url=primary_url or social_url,
                    evidence_span_ids=evidence_span_ids,
                    status=status,
                    created_at=EVALUATION_EPOCH,
                    resolved_at=None if status == IssueStatus.UNRESOLVED else EVALUATION_EPOCH,
                )
            )
        return issues
