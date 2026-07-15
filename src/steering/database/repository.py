from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any, TypeVar, cast

import ladybug as lb
from pydantic import BaseModel

from steering.database.backup import create_backup
from steering.domain.credentials import reject_high_confidence_credentials
from steering.domain.models import (
    Artifact,
    ArtifactRecord,
    Chunk,
    Claim,
    Concept,
    Decision,
    Entity,
    EvidenceSpan,
    ExperimentOutcome,
    IngestionJob,
    IssueStatus,
    Mention,
    Project,
    Relation,
    ReviewIssue,
    ReviewRun,
    ReviewStatus,
    Snapshot,
    new_id,
    utc_now,
)

TModel = TypeVar("TModel", bound=BaseModel)


class ImmutableRecordError(ValueError):
    pass


class CanonicalMappingError(ValueError):
    pass


MODEL_TABLES: dict[type[BaseModel], str] = {
    Snapshot: "Snapshots",
    Chunk: "Chunks",
    Claim: "Claims",
    EvidenceSpan: "EvidenceSpans",
    Relation: "Relations",
    ReviewIssue: "ReviewIssues",
    Entity: "Entities",
    Concept: "Concepts",
}


ISSUE_ACTIONS = {
    "accept_correction": IssueStatus.ACCEPTED_CORRECTION,
    "accepted_correction": IssueStatus.ACCEPTED_CORRECTION,
    "keep_both": IssueStatus.KEPT_BOTH,
    "kept_both": IssueStatus.KEPT_BOTH,
    "dismiss": IssueStatus.DISMISSED,
    "dismissed": IssueStatus.DISMISSED,
    "reject": IssueStatus.REJECTED,
    "rejected": IssueStatus.REJECTED,
    "reject_artifact": IssueStatus.REJECTED,
}


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _payload(model: BaseModel) -> str:
    payload = model.model_dump_json()
    reject_high_confidence_credentials(payload)
    return payload


def _record_id(model: BaseModel) -> str:
    record_id = model.model_dump().get("id")
    if not isinstance(record_id, str):
        raise ValueError("persisted models must have a string id")
    return record_id


class LadybugArtifactRepository:
    """Synchronous repository sharing the daemon's single Ladybug connection."""

    def __init__(self, connection: lb.Connection, database_path: str | Path) -> None:
        self._connection = connection
        self.database_path = Path(database_path).expanduser().resolve(strict=False)
        self._lock = RLock()
        self._closed = False

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._connection.close()
            self._closed = True

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        with self._lock:
            self._ensure_open()
            self._connection.execute("BEGIN TRANSACTION")
            try:
                yield
            except BaseException:
                self._connection.execute("ROLLBACK")
                raise
            else:
                self._connection.execute("COMMIT")

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("repository is closed")

    def _rows(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self._ensure_open()
        result = self._connection.execute(query, parameters)
        if isinstance(result, list):
            raise RuntimeError("repository query unexpectedly returned multiple result sets")
        return cast(list[dict[str, Any]], result.rows_as_dict().get_all())

    def _load_by_id(self, table: str, model_type: type[TModel], record_id: str) -> TModel | None:
        rows = self._rows(
            f"MATCH (n:{table} {{id: $id}}) RETURN n.payload AS payload",
            {"id": record_id},
        )
        if not rows:
            return None
        return model_type.model_validate_json(rows[0]["payload"])

    def _load_members(
        self,
        artifact_id: str,
        member_type: str,
        table: str,
        model_type: type[TModel],
    ) -> list[TModel]:
        query = f"""
            MATCH (m:ArtifactMembers)
            WHERE m.artifact_id = $artifact_id AND m.member_type = $member_type
            MATCH (n:{table})
            WHERE n.id = m.member_id
            RETURN n.payload AS payload
            ORDER BY n.id
            """
        rows = self._rows(
            query,
            {"artifact_id": artifact_id, "member_type": member_type},
        )
        return [model_type.model_validate_json(row["payload"]) for row in rows]

    def _save_member_link(self, artifact_id: str, member_type: str, member_id: str) -> None:
        link_id = f"{artifact_id}:{member_type}:{member_id}"
        self._connection.execute(
            """
            MERGE (m:ArtifactMembers {id: $id})
            SET m.artifact_id = $artifact_id,
                m.member_type = $member_type,
                m.member_id = $member_id
            """,
            {
                "id": link_id,
                "artifact_id": artifact_id,
                "member_type": member_type,
                "member_id": member_id,
            },
        )

    def _save_model(self, model: BaseModel) -> None:
        payload = _payload(model)
        if isinstance(model, Snapshot):
            self._connection.execute(
                """MERGE (n:Snapshots {id: $id})
                SET n.artifact_id = $artifact_id, n.content_hash = $content_hash, n.payload = $payload""",
                {
                    "id": model.id,
                    "artifact_id": model.artifact_id,
                    "content_hash": model.content_hash,
                    "payload": payload,
                },
            )
        elif isinstance(model, Chunk):
            self._connection.execute(
                """MERGE (n:Chunks {id: $id})
                SET n.artifact_id = $artifact_id, n.snapshot_id = $snapshot_id,
                    n.ordinal = $ordinal, n.text = $text, n.locator = $locator, n.payload = $payload""",
                {
                    "id": model.id,
                    "artifact_id": model.artifact_id,
                    "snapshot_id": model.snapshot_id,
                    "ordinal": model.ordinal,
                    "text": model.text,
                    "locator": model.locator,
                    "payload": payload,
                },
            )
        elif isinstance(model, Claim):
            self._connection.execute(
                """MERGE (n:Claims {id: $id})
                SET n.artifact_id = $artifact_id, n.category = $category, n.payload = $payload""",
                {
                    "id": model.id,
                    "artifact_id": model.artifact_id,
                    "category": model.category.value,
                    "payload": payload,
                },
            )
        elif isinstance(model, EvidenceSpan):
            self._connection.execute(
                """MERGE (n:EvidenceSpans {id: $id})
                SET n.snapshot_id = $snapshot_id, n.claim_id = $claim_id, n.payload = $payload""",
                {
                    "id": model.id,
                    "snapshot_id": model.snapshot_id,
                    "claim_id": model.claim_id,
                    "payload": payload,
                },
            )
        elif isinstance(model, Relation):
            self._connection.execute(
                """MERGE (n:Relations {id: $id})
                SET n.subject_id = $subject_id, n.predicate = $predicate,
                    n.object_id = $object_id, n.approved = $approved, n.payload = $payload""",
                {
                    "id": model.id,
                    "subject_id": model.subject_id,
                    "predicate": model.predicate.value,
                    "object_id": model.object_id,
                    "approved": model.approved,
                    "payload": payload,
                },
            )
        elif isinstance(model, ReviewIssue):
            self._connection.execute(
                """MERGE (n:ReviewIssues {id: $id})
                SET n.artifact_id = $artifact_id, n.status = $status, n.created_at = $created_at,
                    n.resolved_at = $resolved_at, n.payload = $payload""",
                {
                    "id": model.id,
                    "artifact_id": model.artifact_id,
                    "status": model.status.value,
                    "created_at": _iso(model.created_at),
                    "resolved_at": _iso(model.resolved_at),
                    "payload": payload,
                },
            )
        elif isinstance(model, Entity):
            self._connection.execute(
                """MERGE (n:Entities {id: $id})
                SET n.name = $name, n.entity_type = $entity_type,
                    n.canonical_entity_id = $canonical_entity_id, n.payload = $payload""",
                {
                    "id": model.id,
                    "name": model.name,
                    "entity_type": model.entity_type,
                    "canonical_entity_id": model.canonical_entity_id,
                    "payload": payload,
                },
            )
        elif isinstance(model, Concept):
            self._connection.execute(
                "MERGE (n:Concepts {id: $id}) SET n.name = $name, n.payload = $payload",
                {"id": model.id, "name": model.name, "payload": payload},
            )
        else:
            raise TypeError(f"unsupported aggregate member: {type(model).__name__}")

    def _save_artifact(self, artifact: Artifact) -> None:
        self._connection.execute(
            """
            MERGE (a:Artifacts {id: $id})
            SET a.canonical_url = $canonical_url,
                a.content_hash = $content_hash,
                a.payload = $payload
            """,
            {
                "id": artifact.id,
                "canonical_url": artifact.canonical_url,
                "content_hash": artifact.content_hash,
                "payload": _payload(artifact),
            },
        )
        if artifact.canonical_url is not None:
            self._connection.execute(
                """
                MERGE (u:UrlIdentities {canonical_url: $canonical_url})
                SET u.artifact_id = $artifact_id
                """,
                {"canonical_url": artifact.canonical_url, "artifact_id": artifact.id},
            )

    @staticmethod
    def _merge_items(
        existing: Sequence[TModel],
        incoming: Sequence[TModel],
        *,
        immutable: bool = False,
    ) -> list[TModel]:
        by_id = {_record_id(item): item for item in existing}
        for item in incoming:
            item_id = _record_id(item)
            previous = by_id.get(item_id)
            if immutable and previous is not None and _payload(previous) != _payload(item):
                raise ImmutableRecordError(f"immutable record {item_id} cannot be changed")
            by_id[item_id] = item
        return [by_id[key] for key in sorted(by_id)]

    @staticmethod
    def _rebind_record(record: ArtifactRecord, artifact_id: str) -> ArtifactRecord:
        old_id = record.artifact.id
        if old_id == artifact_id:
            return record
        artifact = record.artifact.model_copy(update={"id": artifact_id})
        snapshots = [item.model_copy(update={"artifact_id": artifact_id}) for item in record.snapshots]
        chunks = [item.model_copy(update={"artifact_id": artifact_id}) for item in record.chunks]
        claims = [item.model_copy(update={"artifact_id": artifact_id}) for item in record.claims]
        issues = [item.model_copy(update={"artifact_id": artifact_id}) for item in record.issues]
        relations = [
            item.model_copy(
                update={
                    "subject_id": artifact_id if item.subject_id == old_id else item.subject_id,
                    "object_id": artifact_id if item.object_id == old_id else item.object_id,
                }
            )
            for item in record.relations
        ]
        return record.model_copy(
            update={
                "artifact": artifact,
                "snapshots": snapshots,
                "chunks": chunks,
                "claims": claims,
                "issues": issues,
                "relations": relations,
            }
        )

    def upsert_record(self, record: ArtifactRecord) -> ArtifactRecord:
        with self._lock:
            self._ensure_open()
            canonical = record
            if record.artifact.canonical_url is not None:
                same_url = self.get_by_url(record.artifact.canonical_url)
                if same_url is not None and same_url.artifact.id != record.artifact.id:
                    canonical = self._rebind_record(record, same_url.artifact.id)
            existing = self.get_record(canonical.artifact.id)
            if existing is not None:
                canonical = canonical.model_copy(
                    update={
                        "snapshots": self._merge_items(
                            existing.snapshots, canonical.snapshots, immutable=True
                        ),
                        "chunks": self._merge_items(existing.chunks, canonical.chunks),
                        "claims": self._merge_items(existing.claims, canonical.claims),
                        "evidence_spans": self._merge_items(
                            existing.evidence_spans,
                            canonical.evidence_spans,
                            immutable=True,
                        ),
                        "relations": self._merge_items(existing.relations, canonical.relations),
                        "issues": self._merge_items(existing.issues, canonical.issues),
                        "entities": self._merge_items(existing.entities, canonical.entities),
                        "concepts": self._merge_items(existing.concepts, canonical.concepts),
                    }
                )
            canonical = ArtifactRecord.model_validate(canonical.model_dump(mode="python"))
            with self._transaction():
                self._save_artifact(canonical.artifact)
                member_groups: tuple[tuple[str, Sequence[BaseModel]], ...] = (
                    ("snapshot", canonical.snapshots),
                    ("chunk", canonical.chunks),
                    ("claim", canonical.claims),
                    ("evidence_span", canonical.evidence_spans),
                    ("relation", canonical.relations),
                    ("issue", canonical.issues),
                    ("entity", canonical.entities),
                    ("concept", canonical.concepts),
                )
                for member_type, members in member_groups:
                    for member in members:
                        self._save_model(member)
                        self._save_member_link(
                            canonical.artifact.id,
                            member_type,
                            _record_id(member),
                        )
            return canonical

    def get_record(self, artifact_id: str) -> ArtifactRecord | None:
        with self._lock:
            rows = self._rows(
                "MATCH (a:Artifacts {id: $id}) RETURN a.payload AS payload",
                {"id": artifact_id},
            )
            if not rows:
                return None
            artifact = Artifact.model_validate_json(rows[0]["payload"])
            return ArtifactRecord(
                artifact=artifact,
                snapshots=self._load_members(artifact_id, "snapshot", "Snapshots", Snapshot),
                chunks=self._load_members(artifact_id, "chunk", "Chunks", Chunk),
                claims=self._load_members(artifact_id, "claim", "Claims", Claim),
                evidence_spans=self._load_members(
                    artifact_id, "evidence_span", "EvidenceSpans", EvidenceSpan
                ),
                relations=self._load_members(artifact_id, "relation", "Relations", Relation),
                issues=self._load_members(artifact_id, "issue", "ReviewIssues", ReviewIssue),
                entities=self._load_members(artifact_id, "entity", "Entities", Entity),
                concepts=self._load_members(artifact_id, "concept", "Concepts", Concept),
            )

    def get_by_url(self, canonical_url: str) -> ArtifactRecord | None:
        with self._lock:
            rows = self._rows(
                """MATCH (u:UrlIdentities {canonical_url: $canonical_url})
                RETURN u.artifact_id AS artifact_id""",
                {"canonical_url": canonical_url},
            )
            return None if not rows else self.get_record(str(rows[0]["artifact_id"]))

    def list_records(self) -> list[ArtifactRecord]:
        with self._lock:
            rows = self._rows("MATCH (a:Artifacts) RETURN a.id AS id ORDER BY a.id")
            records = [self.get_record(str(row["id"])) for row in rows]
            return [record for record in records if record is not None]

    def save_job(self, job: IngestionJob) -> None:
        with self._transaction():
            self._connection.execute(
                """MERGE (j:IngestionJobs {id: $id})
                SET j.status = $status, j.created_at = $created_at,
                    j.updated_at = $updated_at, j.payload = $payload""",
                {
                    "id": job.id,
                    "status": job.status.value,
                    "created_at": _iso(job.created_at),
                    "updated_at": _iso(job.updated_at),
                    "payload": _payload(job),
                },
            )

    def list_jobs(self, limit: int = 100) -> list[IngestionJob]:
        if limit < 1:
            raise ValueError("limit must be positive")
        with self._lock:
            rows = self._rows(
                """MATCH (j:IngestionJobs) RETURN j.payload AS payload
                ORDER BY j.created_at DESC LIMIT $limit""",
                {"limit": limit},
            )
            return [IngestionJob.model_validate_json(row["payload"]) for row in rows]

    def list_issues(self, unresolved_only: bool = False) -> list[ReviewIssue]:
        with self._lock:
            query = "MATCH (i:ReviewIssues)"
            parameters: dict[str, Any] | None = None
            if unresolved_only:
                query += " WHERE i.status = $status"
                parameters = {"status": IssueStatus.UNRESOLVED.value}
            query += " RETURN i.payload AS payload ORDER BY i.created_at DESC"
            return [ReviewIssue.model_validate_json(row["payload"]) for row in self._rows(query, parameters)]

    def resolve_issue(self, issue_id: str, action: str) -> ReviewIssue:
        normalized = action.strip().lower().replace("-", "_")
        status = ISSUE_ACTIONS.get(normalized)
        if status is None:
            raise ValueError(f"unsupported issue action: {action}")
        with self._transaction():
            issue = self._load_by_id("ReviewIssues", ReviewIssue, issue_id)
            if issue is None:
                raise KeyError(issue_id)
            if status == IssueStatus.ACCEPTED_CORRECTION:
                record = self.get_record(issue.artifact_id)
                snapshots = (
                    {snapshot.id: snapshot for snapshot in record.snapshots} if record is not None else {}
                )
                spans = {span.id: span for span in record.evidence_spans} if record is not None else {}
                has_primary_evidence = any(
                    span_id in spans
                    and spans[span_id].snapshot_id in snapshots
                    and snapshots[spans[span_id].snapshot_id].source_url == issue.primary_source_url
                    for span_id in issue.evidence_span_ids
                )
                if not has_primary_evidence:
                    raise ValueError("a correction requires retained primary-source evidence")
            resolved = issue.model_copy(update={"status": status, "resolved_at": utc_now()})
            self._save_model(resolved)
            if status == IssueStatus.REJECTED:
                record = self.get_record(issue.artifact_id)
                if record is not None:
                    rejected = record.artifact.model_copy(update={"review_status": ReviewStatus.REJECTED})
                    self._save_artifact(rejected)
            return resolved

    def save_project(self, project: Project) -> Project:
        with self._transaction():
            self._connection.execute(
                """MERGE (p:Projects {id: $id})
                SET p.name = $name, p.created_at = $created_at, p.payload = $payload""",
                {
                    "id": project.id,
                    "name": project.name,
                    "created_at": _iso(project.created_at),
                    "payload": _payload(project),
                },
            )
        return project

    def list_projects(self) -> list[Project]:
        with self._lock:
            return self._load_payloads("Projects", Project, "TRUE", {}, "p")

    def save_decision(self, decision: Decision) -> Decision:
        with self._transaction():
            self._connection.execute(
                """MERGE (d:Decisions {id: $id})
                SET d.project_id = $project_id, d.artifact_id = $artifact_id,
                    d.created_at = $created_at, d.payload = $payload""",
                {
                    "id": decision.id,
                    "project_id": decision.project_id,
                    "artifact_id": decision.artifact_id,
                    "created_at": _iso(decision.created_at),
                    "payload": _payload(decision),
                },
            )
        return decision

    def save_outcome(self, outcome: ExperimentOutcome) -> ExperimentOutcome:
        with self._transaction():
            self._connection.execute(
                """MERGE (o:ExperimentOutcomes {id: $id})
                SET o.project_id = $project_id, o.decision_id = $decision_id,
                    o.artifact_id = $artifact_id, o.recorded_at = $recorded_at,
                    o.payload = $payload""",
                {
                    "id": outcome.id,
                    "project_id": outcome.project_id,
                    "decision_id": outcome.decision_id,
                    "artifact_id": outcome.artifact_id,
                    "recorded_at": _iso(outcome.recorded_at),
                    "payload": _payload(outcome),
                },
            )
        return outcome

    def save_review_run(self, review_run: ReviewRun) -> ReviewRun:
        with self._transaction():
            self._connection.execute(
                """MERGE (r:ReviewRuns {id: $id})
                SET r.project_id = $project_id, r.created_at = $created_at, r.payload = $payload""",
                {
                    "id": review_run.id,
                    "project_id": review_run.project_id,
                    "created_at": _iso(review_run.created_at),
                    "payload": _payload(review_run),
                },
            )
        return review_run

    def save_mention(self, mention: Mention) -> Mention:
        with self._transaction():
            self._connection.execute(
                """MERGE (m:Mentions {id: $id})
                SET m.artifact_id = $artifact_id, m.entity_id = $entity_id, m.payload = $payload""",
                {
                    "id": mention.id,
                    "artifact_id": mention.artifact_id,
                    "entity_id": mention.entity_id,
                    "payload": _payload(mention),
                },
            )
        return mention

    def project_history(self, project_id: str) -> Mapping[str, Sequence[Any]]:
        with self._lock:
            projects = self._load_payloads(
                "Projects", Project, "p.id = $project_id", {"project_id": project_id}, "p"
            )
            decisions = self._load_payloads(
                "Decisions", Decision, "p.project_id = $project_id", {"project_id": project_id}, "p"
            )
            outcomes = self._load_payloads(
                "ExperimentOutcomes",
                ExperimentOutcome,
                "p.project_id = $project_id",
                {"project_id": project_id},
                "p",
            )
            review_runs = self._load_payloads(
                "ReviewRuns", ReviewRun, "p.project_id = $project_id", {"project_id": project_id}, "p"
            )
            return {
                "projects": projects,
                "decisions": decisions,
                "outcomes": outcomes,
                "review_runs": review_runs,
            }

    def _load_payloads(
        self,
        table: str,
        model_type: type[TModel],
        where: str,
        parameters: dict[str, Any],
        alias: str,
    ) -> list[TModel]:
        rows = self._rows(
            f"MATCH ({alias}:{table}) WHERE {where} RETURN {alias}.payload AS payload",
            parameters,
        )
        return [model_type.model_validate_json(row["payload"]) for row in rows]

    def get_entity(self, entity_id: str) -> Entity | None:
        with self._lock:
            return self._load_by_id("Entities", Entity, entity_id)

    def save_entity(self, entity: Entity) -> Entity:
        with self._transaction():
            self._save_model(entity)
        return entity

    def set_canonical_entity(
        self,
        entity_id: str,
        canonical_entity_id: str,
        *,
        reason: str | None = None,
    ) -> Entity:
        if reason is not None:
            reject_high_confidence_credentials(reason)
        if entity_id == canonical_entity_id:
            raise CanonicalMappingError("an entity cannot be canonicalized to itself")
        with self._transaction():
            entity = self._load_by_id("Entities", Entity, entity_id)
            canonical = self._load_by_id("Entities", Entity, canonical_entity_id)
            if entity is None or canonical is None:
                raise KeyError(entity_id if entity is None else canonical_entity_id)
            cursor = canonical
            visited = {entity_id}
            while cursor.canonical_entity_id is not None:
                if cursor.canonical_entity_id in visited:
                    raise CanonicalMappingError("canonical mapping would create a cycle")
                visited.add(cursor.canonical_entity_id)
                next_entity = self._load_by_id("Entities", Entity, cursor.canonical_entity_id)
                if next_entity is None:
                    break
                cursor = next_entity
            changed_at = datetime.now(UTC)
            mapping_id = new_id("canonical")
            self._connection.execute(
                """CREATE (m:CanonicalMappings {
                    id: $id, entity_id: $entity_id, canonical_entity_id: $canonical_entity_id,
                    previous_canonical_entity_id: $previous, changed_at: $changed_at,
                    reverted_at: NULL, reason: $reason, active: true
                })""",
                {
                    "id": mapping_id,
                    "entity_id": entity_id,
                    "canonical_entity_id": canonical_entity_id,
                    "previous": entity.canonical_entity_id,
                    "changed_at": changed_at.isoformat(),
                    "reason": reason,
                },
            )
            updated = entity.model_copy(update={"canonical_entity_id": canonical_entity_id})
            self._save_model(updated)
            return updated

    def undo_canonical_mapping(self, entity_id: str) -> Entity:
        with self._transaction():
            rows = self._rows(
                """MATCH (m:CanonicalMappings)
                WHERE m.entity_id = $entity_id AND m.active = true
                RETURN m.id AS id, m.previous_canonical_entity_id AS previous
                ORDER BY m.changed_at DESC LIMIT 1""",
                {"entity_id": entity_id},
            )
            if not rows:
                raise CanonicalMappingError(f"entity has no active canonical mapping: {entity_id}")
            entity = self._load_by_id("Entities", Entity, entity_id)
            if entity is None:
                raise KeyError(entity_id)
            mapping = rows[0]
            reverted_at = datetime.now(UTC).isoformat()
            self._connection.execute(
                """MATCH (m:CanonicalMappings {id: $id})
                SET m.active = false, m.reverted_at = $reverted_at""",
                {"id": mapping["id"], "reverted_at": reverted_at},
            )
            updated = entity.model_copy(update={"canonical_entity_id": mapping["previous"]})
            self._save_model(updated)
            return updated

    def canonical_mapping_history(self, entity_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return self._rows(
                """MATCH (m:CanonicalMappings)
                WHERE m.entity_id = $entity_id
                RETURN m.id AS id, m.entity_id AS entity_id,
                       m.canonical_entity_id AS canonical_entity_id,
                       m.previous_canonical_entity_id AS previous_canonical_entity_id,
                       m.changed_at AS changed_at, m.reverted_at AS reverted_at,
                       m.reason AS reason, m.active AS active
                ORDER BY m.changed_at""",
                {"entity_id": entity_id},
            )

    def backup(self, destination: str) -> str:
        with self._lock:
            self._ensure_open()
            return str(create_backup(self._connection, destination))
