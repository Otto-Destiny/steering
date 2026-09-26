from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from time import time_ns
from typing import Any, TypeVar, cast

import ladybug as lb
from pydantic import BaseModel, ValidationError

from steering.database.backup import create_backup
from steering.database.native import (
    EMBEDDING_DIMENSION,
    FTS_STATE_NAME,
    INDEX_IDENTIFIER,
    VECTOR_STATE_NAME,
    NativeCandidate,
    artifact_search_terms,
)
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
    SearchQuery,
    Snapshot,
    new_id,
    utc_now,
)

LOGGER = logging.getLogger(__name__)
TModel = TypeVar("TModel", bound=BaseModel)


class ImmutableRecordError(ValueError):
    pass


class CanonicalMappingError(ValueError):
    pass


class SearchIndexError(RuntimeError):
    pass


class CorruptRecordError(RuntimeError):
    """Raised when a stored payload no longer validates against its model."""

    def __init__(self, artifact_id: str, detail: str) -> None:
        super().__init__(f"stored data for artifact {artifact_id} could not be read: {detail}")
        self.artifact_id = artifact_id
        self.detail = detail


#: Members owned outright by one artifact. Entities and concepts are shared with
#: other artifacts, so removal takes their membership rows but never the nodes.
_MEMBER_TABLES: dict[str, str] = {
    "snapshot": "Snapshots",
    "chunk": "Chunks",
    "claim": "Claims",
    "evidence_span": "EvidenceSpans",
    "relation": "Relations",
    "issue": "ReviewIssues",
}


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


def _aggregate_members(
    record: ArtifactRecord,
) -> tuple[tuple[str, Sequence[BaseModel]], ...]:
    return (
        ("snapshot", record.snapshots),
        ("chunk", record.chunks),
        ("claim", record.claims),
        ("evidence_span", record.evidence_spans),
        ("relation", record.relations),
        ("issue", record.issues),
        ("entity", record.entities),
        ("concept", record.concepts),
    )


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
                with suppress(RuntimeError):
                    self._connection.execute("ROLLBACK")
                raise
            else:
                self._connection.execute("COMMIT")

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("repository is closed")

    def _execute(self, query: str, parameters: dict[str, Any] | None = None) -> Any:
        """Run one statement under the shared lock.

        The daemon owns a single Ladybug connection, but ingestion runs on a worker
        thread while web requests run on the event loop. Executing on that one
        connection from two threads interleaves statements -- including statements
        landing inside another thread's open transaction, which can commit a node
        whose payload was never written. Every caller goes through here; the lock is
        re-entrant, so methods already holding it nest freely and a transaction keeps
        it for its whole span.
        """

        with self._lock:
            self._ensure_open()
            return self._connection.execute(query, parameters)

    def _rows(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        result = self._execute(query, parameters)
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
        self._execute(
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
            self._execute(
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
            embedding = model.embedding or None
            if embedding is not None and len(embedding) != EMBEDDING_DIMENSION:
                raise ValueError(
                    f"chunk {model.id} embedding has {len(embedding)} dimensions; "
                    f"expected {EMBEDDING_DIMENSION}"
                )
            self._execute(
                """MERGE (n:Chunks {id: $id})
                SET n.artifact_id = $artifact_id, n.snapshot_id = $snapshot_id,
                    n.ordinal = $ordinal, n.text = $text, n.locator = $locator,
                    n.embedding = $embedding, n.embedding_provider = $embedding_provider,
                    n.embedding_model = $embedding_model, n.embedding_revision = $embedding_revision,
                    n.embedding_dimension = $embedding_dimension,
                    n.embedding_task_mode = $embedding_task_mode,
                    n.embedding_normalized = $embedding_normalized,
                    n.source_content_hash = $source_content_hash, n.payload = $payload""",
                {
                    "id": model.id,
                    "artifact_id": model.artifact_id,
                    "snapshot_id": model.snapshot_id,
                    "ordinal": model.ordinal,
                    "text": model.text,
                    "locator": model.locator,
                    "embedding": embedding,
                    "embedding_provider": getattr(model, "embedding_provider", None),
                    "embedding_model": getattr(model, "embedding_model", None),
                    "embedding_revision": getattr(model, "embedding_revision", None),
                    "embedding_dimension": (EMBEDDING_DIMENSION if embedding is not None else None),
                    "embedding_task_mode": getattr(model, "embedding_task_mode", None),
                    "embedding_normalized": getattr(model, "embedding_normalized", None),
                    "source_content_hash": getattr(model, "source_content_hash", None),
                    "payload": payload,
                },
            )
        elif isinstance(model, Claim):
            self._execute(
                """MERGE (n:Claims {id: $id})
                SET n.artifact_id = $artifact_id, n.category = $category,
                    n.text = $text, n.confidence = $confidence, n.payload = $payload""",
                {
                    "id": model.id,
                    "artifact_id": model.artifact_id,
                    "category": model.category.value,
                    "text": model.text,
                    "confidence": model.confidence,
                    "payload": payload,
                },
            )
        elif isinstance(model, EvidenceSpan):
            self._execute(
                """MERGE (n:EvidenceSpans {id: $id})
                SET n.snapshot_id = $snapshot_id, n.claim_id = $claim_id,
                    n.quote = $quote, n.start_offset = $start_offset,
                    n.end_offset = $end_offset, n.locator = $locator, n.payload = $payload""",
                {
                    "id": model.id,
                    "snapshot_id": model.snapshot_id,
                    "claim_id": model.claim_id,
                    "quote": model.quote,
                    "start_offset": model.start,
                    "end_offset": model.end,
                    "locator": model.locator,
                    "payload": payload,
                },
            )
        elif isinstance(model, Relation):
            self._execute(
                """MERGE (n:Relations {id: $id})
                SET n.subject_id = $subject_id, n.predicate = $predicate,
                    n.object_id = $object_id, n.approved = $approved,
                    n.evidence_span_ids = $evidence_span_ids, n.rationale = $rationale,
                    n.payload = $payload""",
                {
                    "id": model.id,
                    "subject_id": model.subject_id,
                    "predicate": model.predicate.value,
                    "object_id": model.object_id,
                    "approved": model.approved,
                    "evidence_span_ids": model.evidence_span_ids,
                    "rationale": model.rationale,
                    "payload": payload,
                },
            )
        elif isinstance(model, ReviewIssue):
            self._execute(
                """MERGE (n:ReviewIssues {id: $id})
                SET n.artifact_id = $artifact_id, n.status = $status, n.created_at = $created_at,
                    n.resolved_at = $resolved_at, n.social_statement = $social_statement,
                    n.source_statement = $source_statement, n.explanation = $explanation,
                    n.social_source_url = $social_source_url,
                    n.primary_source_url = $primary_source_url,
                    n.evidence_span_ids = $evidence_span_ids, n.payload = $payload""",
                {
                    "id": model.id,
                    "artifact_id": model.artifact_id,
                    "status": model.status.value,
                    "created_at": _iso(model.created_at),
                    "resolved_at": _iso(model.resolved_at),
                    "social_statement": model.social_statement,
                    "source_statement": model.source_statement,
                    "explanation": model.explanation,
                    "social_source_url": model.social_source_url,
                    "primary_source_url": model.primary_source_url,
                    "evidence_span_ids": model.evidence_span_ids,
                    "payload": payload,
                },
            )
        elif isinstance(model, Entity):
            self._execute(
                """MERGE (n:Entities {id: $id})
                SET n.name = $name, n.entity_type = $entity_type,
                    n.canonical_entity_id = $canonical_entity_id,
                    n.aliases = $aliases, n.payload = $payload""",
                {
                    "id": model.id,
                    "name": model.name,
                    "entity_type": model.entity_type,
                    "canonical_entity_id": model.canonical_entity_id,
                    "aliases": model.aliases,
                    "payload": payload,
                },
            )
        elif isinstance(model, Concept):
            self._execute(
                """MERGE (n:Concepts {id: $id})
                SET n.name = $name, n.description = $description, n.payload = $payload""",
                {
                    "id": model.id,
                    "name": model.name,
                    "description": model.description,
                    "payload": payload,
                },
            )
        else:
            raise TypeError(f"unsupported aggregate member: {type(model).__name__}")

    def _save_artifact(self, artifact: Artifact) -> None:
        self._execute(
            """
            MERGE (a:Artifacts {id: $id})
            SET a.canonical_url = $canonical_url,
                a.content_hash = $content_hash,
                a.source_kind = $source_kind,
                a.artifact_type = $artifact_type,
                a.title = $title,
                a.short_name = $short_name,
                a.summary = $summary,
                a.strategy_family = $strategy_family,
                a.review_status = $review_status,
                a.trust_lane = $trust_lane,
                a.evidence_quality = $evidence_quality,
                a.maturity = $maturity,
                a.license = $license,
                a.aliases = $aliases,
                a.capabilities = $capabilities,
                a.limitations = $limitations,
                a.requirements = $requirements,
                a.use_cases = $use_cases,
                a.published_at = $published_at,
                a.source_updated_at = $source_updated_at,
                a.last_verified_at = $last_verified_at,
                a.deprecated_at = $deprecated_at,
                a.payload = $payload
            """,
            {
                "id": artifact.id,
                "canonical_url": artifact.canonical_url,
                "content_hash": artifact.content_hash,
                "source_kind": artifact.source_kind.value,
                "artifact_type": artifact.artifact_type.value,
                "title": artifact.title,
                "short_name": artifact.short_name,
                "summary": artifact.summary,
                "strategy_family": artifact.strategy_family,
                "review_status": artifact.review_status.value,
                "trust_lane": artifact.trust_lane.value,
                "evidence_quality": artifact.evidence_quality,
                "maturity": artifact.maturity,
                "license": artifact.license,
                "aliases": artifact.aliases,
                "capabilities": artifact.capabilities,
                "limitations": artifact.limitations,
                "requirements": artifact.requirements,
                "use_cases": artifact.use_cases,
                "published_at": _iso(artifact.published_at),
                "source_updated_at": _iso(artifact.source_updated_at),
                "last_verified_at": _iso(artifact.last_verified_at),
                "deprecated_at": _iso(artifact.deprecated_at),
                "payload": _payload(artifact),
            },
        )
        if artifact.canonical_url is not None:
            self._execute(
                """
                MERGE (u:UrlIdentities {canonical_url: $canonical_url})
                SET u.artifact_id = $artifact_id
                """,
                {"canonical_url": artifact.canonical_url, "artifact_id": artifact.id},
            )

    def _replace_native_links(self, record: ArtifactRecord) -> None:
        artifact_id = record.artifact.id
        memberships: tuple[tuple[str, str, Sequence[BaseModel]], ...] = (
            ("ArtifactHasSnapshot", "Snapshots", record.snapshots),
            ("ArtifactHasChunk", "Chunks", record.chunks),
            ("ArtifactHasClaim", "Claims", record.claims),
            ("ArtifactHasEntity", "Entities", record.entities),
            ("ArtifactHasConcept", "Concepts", record.concepts),
            ("ArtifactHasIssue", "ReviewIssues", record.issues),
        )
        for relation_table, node_table, members in memberships:
            self._execute(
                f"MATCH (a:Artifacts {{id: $artifact_id}})-[r:{relation_table}]->() DELETE r",
                {"artifact_id": artifact_id},
            )
            for member in members:
                self._execute(
                    f"""MATCH (a:Artifacts {{id: $artifact_id}}),
                    (n:{node_table} {{id: $member_id}})
                    MERGE (a)-[:{relation_table}]->(n)""",
                    {"artifact_id": artifact_id, "member_id": _record_id(member)},
                )

        self._execute(
            """MATCH (t:SearchTerms)-[r:TermReferencesArtifact]->
            (a:Artifacts {id: $artifact_id}) DELETE r""",
            {"artifact_id": artifact_id},
        )
        terms = artifact_search_terms(
            record.artifact.model_dump(mode="json"),
            entity_names=tuple(entity.name for entity in record.entities),
            concept_names=tuple(concept.name for concept in record.concepts),
        )
        for term, weight in terms.items():
            self._execute(
                """MERGE (t:SearchTerms {term: $term}) SET t.kind = 'artifact_term'
                WITH t MATCH (a:Artifacts {id: $artifact_id})
                MERGE (t)-[r:TermReferencesArtifact]->(a)
                SET r.weight = $weight, r.kind = 'artifact_term'""",
                {"term": term, "artifact_id": artifact_id, "weight": weight},
            )

        for claim in record.claims:
            self._execute(
                "MATCH (c:Claims {id: $id})-[r:ClaimHasEvidence]->() DELETE r",
                {"id": claim.id},
            )
            for span_id in claim.evidence_span_ids:
                self._execute(
                    """MATCH (c:Claims {id: $claim_id}), (s:EvidenceSpans {id: $span_id})
                    MERGE (c)-[:ClaimHasEvidence]->(s)""",
                    {"claim_id": claim.id, "span_id": span_id},
                )
        for snapshot in record.snapshots:
            self._execute(
                "MATCH (s:Snapshots {id: $id})-[r:SnapshotHasEvidence]->() DELETE r",
                {"id": snapshot.id},
            )
        for span in record.evidence_spans:
            self._execute(
                """MATCH (s:Snapshots {id: $snapshot_id}), (e:EvidenceSpans {id: $span_id})
                MERGE (s)-[:SnapshotHasEvidence]->(e)""",
                {"snapshot_id": span.snapshot_id, "span_id": span.id},
            )

        new_node_ids = [
            artifact_id,
            *(entity.id for entity in record.entities),
            *(concept.id for concept in record.concepts),
        ]
        relation_payloads = {relation.id: relation for relation in record.relations}
        rows = self._rows(
            """MATCH (r:Relations)
            WHERE r.subject_id IN $node_ids OR r.object_id IN $node_ids
            RETURN r.payload AS payload""",
            {"node_ids": new_node_ids},
        )
        for row in rows:
            relation = Relation.model_validate_json(row["payload"])
            relation_payloads[relation.id] = relation
        for relation in relation_payloads.values():
            self._sync_native_knowledge_edge(relation)

    def _native_node_table(self, node_id: str) -> str | None:
        for table in ("Artifacts", "Entities", "Concepts"):
            if self._rows(f"MATCH (n:{table} {{id: $id}}) RETURN n.id AS id", {"id": node_id}):
                return table
        return None

    def _artifact_owners(self, node_id: str, node_table: str) -> list[str]:
        if node_table == "Artifacts":
            return [node_id]
        relation_table = "ArtifactHasEntity" if node_table == "Entities" else "ArtifactHasConcept"
        rows = self._rows(
            f"""MATCH (a:Artifacts)-[:{relation_table}]->(n:{node_table} {{id: $id}})
            RETURN DISTINCT a.id AS artifact_id ORDER BY artifact_id""",
            {"id": node_id},
        )
        return [str(row["artifact_id"]) for row in rows]

    def _sync_native_knowledge_edge(self, relation: Relation) -> None:
        self._execute(
            "MATCH ()-[r:KnowledgeEdges]->() WHERE r.id = $id DELETE r",
            {"id": relation.id},
        )
        self._execute(
            """MATCH ()-[r:ArtifactKnowledgeLinks]->()
            WHERE r.relation_id = $id DELETE r""",
            {"id": relation.id},
        )
        if not relation.approved:
            return
        subject_table = self._native_node_table(relation.subject_id)
        object_table = self._native_node_table(relation.object_id)
        if subject_table is None or object_table is None:
            return
        self._execute(
            f"""MATCH (s:{subject_table} {{id: $subject_id}}),
            (o:{object_table} {{id: $object_id}})
            MERGE (s)-[r:KnowledgeEdges {{id: $id}}]->(o)
            SET r.predicate = $predicate, r.evidence_span_ids = $evidence_span_ids,
                r.rationale = $rationale, r.payload = $payload""",
            {
                "subject_id": relation.subject_id,
                "object_id": relation.object_id,
                "id": relation.id,
                "predicate": relation.predicate.value,
                "evidence_span_ids": relation.evidence_span_ids,
                "rationale": relation.rationale,
                "payload": _payload(relation),
            },
        )
        subject_owners = self._artifact_owners(relation.subject_id, subject_table)
        object_owners = self._artifact_owners(relation.object_id, object_table)
        for subject_owner in subject_owners:
            for object_owner in object_owners:
                if subject_owner == object_owner:
                    continue
                self._execute(
                    """MATCH (s:Artifacts {id: $subject_owner}),
                    (o:Artifacts {id: $object_owner})
                    MERGE (s)-[r:ArtifactKnowledgeLinks {relation_id: $relation_id}]->(o)
                    SET r.predicate = $predicate, r.subject_node_id = $subject_id,
                        r.object_node_id = $object_id""",
                    {
                        "subject_owner": subject_owner,
                        "object_owner": object_owner,
                        "relation_id": relation.id,
                        "predicate": relation.predicate.value,
                        "subject_id": relation.subject_id,
                        "object_id": relation.object_id,
                    },
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
            existing_member_payloads: dict[tuple[str, str], str] = {}
            if existing is not None:
                existing_member_payloads = {
                    (member_type, _record_id(member)): _payload(member)
                    for member_type, members in _aggregate_members(existing)
                    for member in members
                }
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
                for member_type, members in _aggregate_members(canonical):
                    for member in members:
                        member_key = (member_type, _record_id(member))
                        if existing_member_payloads.get(member_key) != _payload(member):
                            self._save_model(member)
                        self._save_member_link(
                            canonical.artifact.id,
                            member_type,
                            _record_id(member),
                        )
                self._replace_native_links(canonical)
            return canonical

    def get_record(self, artifact_id: str) -> ArtifactRecord | None:
        try:
            return self._get_record(artifact_id)
        except ValidationError as exc:
            # Surfacing this as a validation failure made a plain GET report that
            # the user should "correct the submitted fields". Name the real
            # problem so the interface can offer the only fix: retire the record.
            raise CorruptRecordError(artifact_id, self._first_validation_detail(exc)) from exc

    @staticmethod
    def _first_validation_detail(exc: ValidationError) -> str:
        errors = exc.errors(include_input=False, include_url=False)
        if not errors:
            return "payload did not match its stored model"
        first = errors[0]
        location = ".".join(str(part) for part in first.get("loc", ())) or "payload"
        return f"{location}: {first.get('msg', 'invalid value')}"

    def _get_record(self, artifact_id: str) -> ArtifactRecord | None:
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

    def set_relation_active(self, artifact_id: str, relation_id: str, active: bool) -> Relation:
        """Turn one of an artifact's relations into a traversable edge, or take it out.

        Rebuilds the edge rather than only flipping the flag, because the flag is
        what the projection is derived from: a relation nobody can walk is not a
        different record, it is an absent edge.

        The artifact is part of the lookup, not decoration. Matching on the relation
        alone would let a request name one artifact and change another's connection,
        and then be shown the untouched page of the artifact it named.
        """

        with self._transaction():
            rows = self._rows(
                "MATCH (n:Relations {id: $id}) RETURN n.payload AS payload", {"id": relation_id}
            )
            if not rows:
                raise KeyError(relation_id)
            stored = Relation.model_validate_json(rows[0]["payload"])
            if stored.subject_id != artifact_id:
                raise KeyError(relation_id)
            # Revalidated rather than copied, so a sensitive relation cannot be
            # activated without the quote its own model insists on.
            updated = Relation.model_validate({**stored.model_dump(mode="json"), "approved": active})
            self._save_model(updated)
            self._sync_native_knowledge_edge(updated)
        LOGGER.info("relation %s is now %s", relation_id, "active" if active else "inactive")
        return updated

    def unreadable_records(self) -> list[dict[str, str]]:
        """Find every record whose stored data can no longer be read.

        Damage to a member payload leaves the artifact listable but unopenable,
        so it only reveals itself when someone clicks it. Scanning finds them all
        in one pass instead of one accident at a time.
        """

        with self._lock:
            self._ensure_open()
            rows = self._rows("MATCH (a:Artifacts) RETURN a.id AS id, a.title AS title ORDER BY a.id")
        damaged: list[dict[str, str]] = []
        for row in rows:
            artifact_id = str(row["id"])
            try:
                self.get_record(artifact_id)
            except CorruptRecordError as exc:
                damaged.append(
                    {
                        "artifact_id": artifact_id,
                        "title": str(row.get("title") or ""),
                        "detail": exc.detail,
                    }
                )
        if damaged:
            LOGGER.warning("%d stored record(s) could not be read", len(damaged))
        return damaged

    def delete_record(self, artifact_id: str) -> bool:
        """Remove one artifact and everything stored only for it.

        Evidence is immutable while an artifact exists, so removal is the only
        way to correct a bad capture. Members, membership rows, native edges, the
        URL identity, and search terms all go with it; leaving any behind would
        keep the artifact half-present in retrieval.

        Nothing here parses a stored payload. A record whose payload no longer
        validates is exactly the record a user most needs to remove, and reading
        it first would make the repair depend on the damage.
        """

        with self._lock:
            self._ensure_open()
            rows = self._rows(
                "MATCH (a:Artifacts {id: $id}) RETURN a.canonical_url AS canonical_url",
                {"id": artifact_id},
            )
            if not rows:
                return False
            canonical_url = rows[0]["canonical_url"]
            members = self._rows(
                """MATCH (m:ArtifactMembers)
                WHERE m.artifact_id = $id
                RETURN m.member_type AS member_type, m.member_id AS member_id""",
                {"id": artifact_id},
            )
            owned: dict[str, list[str]] = {}
            for row in members:
                table = _MEMBER_TABLES.get(str(row["member_type"]))
                if table is not None:
                    owned.setdefault(table, []).append(str(row["member_id"]))
            with self._transaction():
                # Membership rows go first, so what remains answers whether any
                # other artifact still owns a member.
                self._execute(
                    "MATCH (m:ArtifactMembers) WHERE m.artifact_id = $id DELETE m",
                    {"id": artifact_id},
                )
                # DETACH DELETE rather than enumerating edge tables: these nodes
                # are being removed outright, and listing every relationship type
                # by hand breaks silently whenever the schema gains another one.
                for table, member_ids in owned.items():
                    for member_id in member_ids:
                        if self._member_is_shared(member_id):
                            # Another artifact still references this node. Its
                            # membership row is gone; the node stays for them.
                            continue
                        self._execute(
                            f"MATCH (n:{table} {{id: $id}}) DETACH DELETE n",
                            {"id": member_id},
                        )
                if canonical_url is not None:
                    self._execute(
                        "MATCH (u:UrlIdentities {canonical_url: $url}) DELETE u",
                        {"url": str(canonical_url)},
                    )
                self._execute(
                    "MATCH (a:Artifacts {id: $id}) DETACH DELETE a",
                    {"id": artifact_id},
                )
            return True

    def _member_is_shared(self, member_id: str) -> bool:
        """Report whether another artifact still owns this member node."""

        return bool(
            self._rows(
                "MATCH (m:ArtifactMembers) WHERE m.member_id = $id RETURN m.id AS id LIMIT 1",
                {"id": member_id},
            )
        )

    def find_artifact_by_url(self, canonical_url: str) -> Artifact | None:
        """Look up an artifact header by URL without hydrating its whole record."""

        with self._lock:
            rows = self._rows(
                """MATCH (u:UrlIdentities {canonical_url: $canonical_url})
                RETURN u.artifact_id AS artifact_id""",
                {"canonical_url": canonical_url},
            )
            if not rows:
                return None
            payloads = self._rows(
                "MATCH (a:Artifacts {id: $id}) RETURN a.payload AS payload",
                {"id": str(rows[0]["artifact_id"])},
            )
            if not payloads:
                return None
            return Artifact.model_validate_json(str(payloads[0]["payload"]))

    def count_artifacts(self) -> int:
        with self._lock:
            return int(self._rows("MATCH (a:Artifacts) RETURN count(a) AS count")[0]["count"])

    def list_artifacts(self, *, limit: int | None = None) -> list[Artifact]:
        """Read artifact headers only, newest first.

        Listing views need titles and lanes, not evidence. Hydrating whole
        records for them costs nine queries per artifact and grows with the
        graph, so overviews stay on this single-query path.
        """

        with self._lock:
            rows = self._rows("MATCH (a:Artifacts) RETURN a.payload AS payload")
            artifacts = [Artifact.model_validate_json(str(row["payload"])) for row in rows]
        artifacts.sort(key=lambda artifact: artifact.captured_at, reverse=True)
        return artifacts if limit is None else artifacts[:limit]

    def load_records(self, artifact_ids: Sequence[str]) -> list[ArtifactRecord]:
        with self._lock:
            records = [self.get_record(artifact_id) for artifact_id in dict.fromkeys(artifact_ids)]
            return [record for record in records if record is not None]

    def _load_search_extension(self, extension: str) -> None:
        try:
            self._execute(f"LOAD {extension}")
            return
        except RuntimeError:
            pass
        try:
            self._execute(f"INSTALL {extension}")
            self._execute(f"LOAD {extension}")
        except RuntimeError as exc:
            raise SearchIndexError(
                f"Ladybug {extension.lower()} extension is unavailable; "
                "restore network access, then run 'steering reindex'"
            ) from exc

    def load_active_search_extensions(self) -> None:
        """Load extensions required by indexes persisted from an earlier process."""

        with self._lock:
            rows = self._rows(
                """MATCH (s:SearchIndexState)
                WHERE s.status = 'ready' AND s.active_index_name IS NOT NULL
                RETURN s.index_type AS index_type"""
            )
            active_types = {str(row["index_type"]).upper() for row in rows}
            if "FTS" in active_types:
                self._load_search_extension("FTS")
            if "HNSW" in active_types:
                self._load_search_extension("VECTOR")

    def _active_index_name(self, state_name: str) -> str | None:
        rows = self._rows(
            """MATCH (s:SearchIndexState {name: $name})
            RETURN s.active_index_name AS active_index_name, s.status AS status""",
            {"name": state_name},
        )
        if not rows or rows[0]["status"] != "ready":
            return None
        name = str(rows[0]["active_index_name"] or "")
        if not INDEX_IDENTIFIER.fullmatch(name):
            raise SearchIndexError("stored Ladybug index identifier is invalid")
        return name

    def _embedding_identity(self) -> tuple[dict[str, Any], int]:
        rows = self._rows(
            """MATCH (c:Chunks)
            WHERE c.embedding IS NOT NULL
            RETURN c.embedding_provider AS provider, c.embedding_model AS model,
                c.embedding_revision AS revision, c.embedding_dimension AS dimension,
                c.embedding_task_mode AS task_mode, c.embedding_normalized AS normalized,
                c.source_content_hash AS source_content_hash"""
        )
        if not rows:
            return {}, 0
        required = ("provider", "model", "dimension", "task_mode", "normalized", "source_content_hash")
        if any(any(row[key] is None or row[key] == "" for key in required) for row in rows):
            raise SearchIndexError("embedded chunks need complete model and source-hash provenance")
        identities = {
            (
                str(row["provider"]),
                str(row["model"]),
                None if row["revision"] is None else str(row["revision"]),
                int(row["dimension"]),
                str(row["task_mode"]),
                bool(row["normalized"]),
            )
            for row in rows
        }
        if len(identities) != 1:
            raise SearchIndexError("chunks contain incompatible embedding model provenance")
        provider, model, revision, dimension, task_mode, normalized = identities.pop()
        if dimension != EMBEDDING_DIMENSION:
            raise SearchIndexError(f"native index requires {EMBEDDING_DIMENSION}-dimensional embeddings")
        return {
            "embedding_provider": provider,
            "embedding_model": model,
            "embedding_revision": revision,
            "embedding_dimension": dimension,
            "embedding_task_mode": task_mode,
            "embedding_normalized": normalized,
        }, len(rows)

    def rebuild_search_indexes(self) -> None:
        """Build replacements first, then atomically switch the active index metadata."""

        with self._lock:
            self._ensure_open()
            self._load_search_extension("FTS")
            self._load_search_extension("VECTOR")
            identity, vector_count = self._embedding_identity()
            chunk_count = int(self._rows("MATCH (c:Chunks) RETURN count(c) AS count")[0]["count"])
            if vector_count != chunk_count:
                raise SearchIndexError(
                    f"{chunk_count - vector_count} stored chunks need embeddings; "
                    "run 'steering reindex' before searching"
                )
            suffix = str(time_ns())
            new_fts = f"chunks_text_fts_{suffix}"
            new_vector = f"chunks_embedding_hnsw_{suffix}"
            old_fts = self._active_index_name(FTS_STATE_NAME)
            old_vector = self._active_index_name(VECTOR_STATE_NAME)
            try:
                self._execute(
                    f"""CALL CREATE_FTS_INDEX('Chunks', '{new_fts}', ['text'],
                    stemmer := 'porter', stopwords := 'SearchStopwords')"""
                )
                self._execute(
                    f"""CALL CREATE_VECTOR_INDEX('Chunks', '{new_vector}', 'embedding',
                    metric := 'cosine')"""
                )
            except RuntimeError as exc:
                for index_type, name in (("fts", new_fts), ("vector", new_vector)):
                    with suppress(RuntimeError):
                        self._drop_search_index(index_type, name)
                raise SearchIndexError("Ladybug failed to build replacement search indexes") from exc
            built_at = datetime.now(UTC).isoformat()
            with self._transaction():
                self._execute(
                    """MERGE (s:SearchIndexState {name: $name})
                    SET s.index_type = 'FTS', s.active_index_name = $index_name,
                        s.status = 'ready', s.built_at = $built_at, s.row_count = $row_count""",
                    {
                        "name": FTS_STATE_NAME,
                        "index_name": new_fts,
                        "built_at": built_at,
                        "row_count": chunk_count,
                    },
                )
                self._execute(
                    """MERGE (s:SearchIndexState {name: $name})
                    SET s.index_type = 'HNSW', s.active_index_name = $index_name,
                        s.status = 'ready', s.built_at = $built_at, s.row_count = $row_count,
                        s.embedding_provider = $embedding_provider,
                        s.embedding_model = $embedding_model,
                        s.embedding_revision = $embedding_revision,
                        s.embedding_dimension = $embedding_dimension,
                        s.embedding_task_mode = $embedding_task_mode,
                        s.embedding_normalized = $embedding_normalized""",
                    {
                        "name": VECTOR_STATE_NAME,
                        "index_name": new_vector,
                        "built_at": built_at,
                        "row_count": vector_count,
                        **{
                            "embedding_provider": identity.get("embedding_provider"),
                            "embedding_model": identity.get("embedding_model"),
                            "embedding_revision": identity.get("embedding_revision"),
                            "embedding_dimension": identity.get("embedding_dimension"),
                            "embedding_task_mode": identity.get("embedding_task_mode"),
                            "embedding_normalized": identity.get("embedding_normalized"),
                        },
                    },
                )
            if old_fts and old_fts != new_fts:
                self._drop_search_index("fts", old_fts)
            if old_vector and old_vector != new_vector:
                self._drop_search_index("vector", old_vector)

    def _drop_search_index(self, index_type: str, name: str) -> None:
        if not INDEX_IDENTIFIER.fullmatch(name):
            raise SearchIndexError("refusing to use an invalid Ladybug index identifier")
        procedure = "DROP_FTS_INDEX" if index_type == "fts" else "DROP_VECTOR_INDEX"
        self._execute(f"CALL {procedure}('Chunks', '{name}')")

    def search_index_status(self) -> list[dict[str, Any]]:
        with self._lock:
            return self._rows(
                """MATCH (s:SearchIndexState)
                RETURN s.name AS name, s.index_type AS index_type,
                    s.active_index_name AS active_index_name, s.status AS status,
                    s.built_at AS built_at, s.row_count AS row_count,
                    s.embedding_provider AS embedding_provider,
                    s.embedding_model AS embedding_model,
                    s.embedding_revision AS embedding_revision,
                    s.embedding_dimension AS embedding_dimension,
                    s.embedding_task_mode AS embedding_task_mode,
                    s.embedding_normalized AS embedding_normalized
                ORDER BY s.name"""
            )

    def backup_before_reembedding(self) -> str:
        destination = self.database_path.with_name(
            f"{self.database_path.name}.pre-reembed-{time_ns()}.backup"
        )
        return self.backup(str(destination))

    def replace_chunk_embeddings(self, chunks: Sequence[Chunk]) -> int:
        """Atomically replace the complete existing chunk-vector set."""

        chunk_ids = [chunk.id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("re-embedding batch contains duplicate chunk IDs")
        with self._transaction():
            rows = self._rows(
                "MATCH (c:Chunks) WHERE c.id IN $ids RETURN c.id AS id",
                {"ids": chunk_ids},
            )
            existing = {str(row["id"]) for row in rows}
            if existing != set(chunk_ids):
                raise ValueError("re-embedding batch does not match stored chunks")
            for chunk in chunks:
                self._save_model(chunk)
        return len(chunks)

    def assert_embedding_compatible(
        self,
        *,
        provider: str,
        model: str,
        revision: str | None,
        dimension: int,
        task_mode: str,
        normalized: bool,
    ) -> None:
        """Refuse semantic search across vectors from a different model contract."""

        with self._lock:
            rows = self._rows(
                """MATCH (s:SearchIndexState {name: $name})
                WHERE s.status = 'ready' AND s.row_count > 0
                RETURN s.embedding_provider AS provider, s.embedding_model AS model,
                    s.embedding_revision AS revision, s.embedding_dimension AS dimension,
                    s.embedding_task_mode AS task_mode,
                    s.embedding_normalized AS normalized""",
                {"name": VECTOR_STATE_NAME},
            )
            if not rows:
                return
            row = rows[0]
            stored = (
                str(row["provider"]),
                str(row["model"]),
                None if row["revision"] is None else str(row["revision"]),
                int(row["dimension"]),
                str(row["task_mode"]),
                bool(row["normalized"]),
            )
            requested = (provider, model, revision, dimension, task_mode, normalized)
            if stored != requested:
                raise SearchIndexError(
                    "the active embedding provider is incompatible with stored vectors; "
                    "run a verified complete re-embedding before searching"
                )

    @staticmethod
    def _search_filter(query: SearchQuery, alias: str = "a") -> tuple[str, str, dict[str, Any]]:
        concept_match = ""
        clauses = [f"{alias}.review_status <> $rejected"]
        parameters: dict[str, Any] = {"rejected": ReviewStatus.REJECTED.value}
        if query.artifact_types:
            clauses.append(f"{alias}.artifact_type IN $artifact_types")
            parameters["artifact_types"] = [item.value for item in query.artifact_types]
        if query.minimum_evidence is not None:
            clauses.append(f"{alias}.evidence_quality >= $minimum_evidence")
            parameters["minimum_evidence"] = query.minimum_evidence
        if query.published_after is not None:
            clauses.append(f"{alias}.published_at IS NOT NULL")
            clauses.append(f"{alias}.published_at >= $published_after")
            parameters["published_after"] = query.published_after.isoformat()
        if query.concepts:
            concept_match = f"MATCH ({alias})-[:ArtifactHasConcept]->(filter_concept:Concepts)"
            clauses.append("lower(filter_concept.name) IN $filter_concepts")
            parameters["filter_concepts"] = [item.lower() for item in query.concepts]
        return concept_match, " AND ".join(clauses), parameters

    def exact_candidates(
        self, terms: Sequence[str], query: SearchQuery, *, limit: int = 50
    ) -> list[NativeCandidate]:
        if not terms:
            return []
        concept_match, filters, parameters = self._search_filter(query)
        parameters.update({"terms": list(dict.fromkeys(terms)), "limit": limit})
        rows = self._rows(
            f"""MATCH (t:SearchTerms)-[r:TermReferencesArtifact]->(a:Artifacts)
            {concept_match}
            WHERE t.term IN $terms AND {filters}
            RETURN a.id AS artifact_id, max(r.weight) AS score
            ORDER BY score DESC, artifact_id LIMIT $limit""",
            parameters,
        )
        return [NativeCandidate(str(row["artifact_id"]), None, float(row["score"])) for row in rows]

    def bm25_candidates(self, text: str, query: SearchQuery, *, limit: int = 50) -> list[NativeCandidate]:
        index_name = self._active_index_name(FTS_STATE_NAME)
        if index_name is None:
            raise SearchIndexError("Ladybug full-text index is not ready")
        concept_match, filters, parameters = self._search_filter(query)
        parameters.update({"text": text, "top": limit * 2, "limit": limit})
        rows = self._rows(
            f"""CALL QUERY_FTS_INDEX('Chunks', '{index_name}', $text, top := $top)
            WITH node, score
            MATCH (a:Artifacts) {concept_match}
            WHERE a.id = node.artifact_id AND {filters}
            RETURN a.id AS artifact_id, node.id AS chunk_id, score
            ORDER BY score DESC, artifact_id LIMIT $limit""",
            parameters,
        )
        return [
            NativeCandidate(str(row["artifact_id"]), str(row["chunk_id"]), float(row["score"]))
            for row in rows
        ]

    def vector_candidates(
        self, vector: Sequence[float], query: SearchQuery, *, limit: int = 50
    ) -> list[NativeCandidate]:
        if len(vector) != EMBEDDING_DIMENSION:
            raise ValueError(f"query embedding must have {EMBEDDING_DIMENSION} dimensions")
        index_name = self._active_index_name(VECTOR_STATE_NAME)
        if index_name is None:
            raise SearchIndexError("Ladybug vector index is not ready")
        concept_match, filters, parameters = self._search_filter(query)
        parameters.update({"vector": list(vector), "top": limit * 2, "limit": limit})
        rows = self._rows(
            f"""CALL QUERY_VECTOR_INDEX(
                'Chunks', '{index_name}', $vector, $top, efs := 200)
            WITH node, distance
            MATCH (a:Artifacts) {concept_match}
            WHERE a.id = node.artifact_id AND {filters}
            RETURN a.id AS artifact_id, node.id AS chunk_id, distance
            ORDER BY distance, artifact_id LIMIT $limit""",
            parameters,
        )
        candidates: list[NativeCandidate] = []
        for row in rows:
            similarity = max(0.0, 1.0 - float(row["distance"]))
            if similarity > 0.0:
                candidates.append(NativeCandidate(str(row["artifact_id"]), str(row["chunk_id"]), similarity))
        return candidates

    def graph_candidates(
        self, seed_ids: Sequence[str], query: SearchQuery, *, limit: int = 50
    ) -> list[NativeCandidate]:
        if not seed_ids:
            return []
        concept_match, filters, parameters = self._search_filter(query)
        parameters.update({"seed_ids": list(dict.fromkeys(seed_ids)), "limit": limit})
        outgoing = self._rows(
            f"""MATCH (seed:Artifacts)-[r:ArtifactKnowledgeLinks]->(a:Artifacts)
            {concept_match}
            WHERE seed.id IN $seed_ids AND NOT (a.id IN $seed_ids) AND {filters}
            RETURN a.id AS artifact_id, count(r) AS paths
            ORDER BY paths DESC, artifact_id LIMIT $limit""",
            parameters,
        )
        incoming = self._rows(
            f"""MATCH (a:Artifacts)-[r:ArtifactKnowledgeLinks]->(seed:Artifacts)
            {concept_match}
            WHERE seed.id IN $seed_ids AND NOT (a.id IN $seed_ids) AND {filters}
            RETURN a.id AS artifact_id, count(r) AS paths
            ORDER BY paths DESC, artifact_id LIMIT $limit""",
            parameters,
        )
        scores: dict[str, float] = {}
        for direction_weight, rows in ((1.0, outgoing), (0.85, incoming)):
            for row in rows:
                artifact_id = str(row["artifact_id"])
                scores[artifact_id] = scores.get(artifact_id, 0.0) + direction_weight * float(row["paths"])
        return [
            NativeCandidate(artifact_id, None, score)
            for artifact_id, score in sorted(
                scores.items(), key=lambda item: (item[1], item[0]), reverse=True
            )[:limit]
        ]

    def save_job(self, job: IngestionJob) -> None:
        with self._transaction():
            self._execute(
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
            self._execute(
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
            self._execute(
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
            self._execute(
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
            self._execute(
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
            self._execute(
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
            self._execute(
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
            self._execute(
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
