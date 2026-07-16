from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, cast

import ladybug as lb

from steering.database.native import EMBEDDING_DIMENSION, artifact_search_terms

SCHEMA_REVISION = 2


REVISION_1_STATEMENTS = (
    ("CREATE NODE TABLE IF NOT EXISTS SchemaMigrations(revision INT64 PRIMARY KEY, applied_at STRING)"),
    (
        "CREATE NODE TABLE IF NOT EXISTS Artifacts("
        "id STRING PRIMARY KEY, canonical_url STRING, content_hash STRING, payload STRING)"
    ),
    "CREATE NODE TABLE IF NOT EXISTS UrlIdentities(canonical_url STRING PRIMARY KEY, artifact_id STRING)",
    (
        "CREATE NODE TABLE IF NOT EXISTS Snapshots("
        "id STRING PRIMARY KEY, artifact_id STRING, content_hash STRING, payload STRING)"
    ),
    (
        "CREATE NODE TABLE IF NOT EXISTS Chunks("
        "id STRING PRIMARY KEY, artifact_id STRING, snapshot_id STRING, ordinal INT64, "
        "text STRING, locator STRING, payload STRING)"
    ),
    (
        "CREATE NODE TABLE IF NOT EXISTS Claims("
        "id STRING PRIMARY KEY, artifact_id STRING, category STRING, payload STRING)"
    ),
    (
        "CREATE NODE TABLE IF NOT EXISTS EvidenceSpans("
        "id STRING PRIMARY KEY, snapshot_id STRING, claim_id STRING, payload STRING)"
    ),
    (
        "CREATE NODE TABLE IF NOT EXISTS Relations("
        "id STRING PRIMARY KEY, subject_id STRING, predicate STRING, object_id STRING, "
        "approved BOOL, payload STRING)"
    ),
    (
        "CREATE NODE TABLE IF NOT EXISTS ReviewIssues("
        "id STRING PRIMARY KEY, artifact_id STRING, status STRING, created_at STRING, "
        "resolved_at STRING, payload STRING)"
    ),
    (
        "CREATE NODE TABLE IF NOT EXISTS Entities("
        "id STRING PRIMARY KEY, name STRING, entity_type STRING, "
        "canonical_entity_id STRING, payload STRING)"
    ),
    "CREATE NODE TABLE IF NOT EXISTS Concepts(id STRING PRIMARY KEY, name STRING, payload STRING)",
    (
        "CREATE NODE TABLE IF NOT EXISTS Mentions("
        "id STRING PRIMARY KEY, artifact_id STRING, entity_id STRING, payload STRING)"
    ),
    (
        "CREATE NODE TABLE IF NOT EXISTS ArtifactMembers("
        "id STRING PRIMARY KEY, artifact_id STRING, member_type STRING, member_id STRING)"
    ),
    (
        "CREATE NODE TABLE IF NOT EXISTS Projects("
        "id STRING PRIMARY KEY, name STRING, created_at STRING, payload STRING)"
    ),
    (
        "CREATE NODE TABLE IF NOT EXISTS Decisions("
        "id STRING PRIMARY KEY, project_id STRING, artifact_id STRING, "
        "created_at STRING, payload STRING)"
    ),
    (
        "CREATE NODE TABLE IF NOT EXISTS ExperimentOutcomes("
        "id STRING PRIMARY KEY, project_id STRING, decision_id STRING, "
        "artifact_id STRING, recorded_at STRING, payload STRING)"
    ),
    (
        "CREATE NODE TABLE IF NOT EXISTS ReviewRuns("
        "id STRING PRIMARY KEY, project_id STRING, created_at STRING, payload STRING)"
    ),
    (
        "CREATE NODE TABLE IF NOT EXISTS IngestionJobs("
        "id STRING PRIMARY KEY, status STRING, created_at STRING, "
        "updated_at STRING, payload STRING)"
    ),
    (
        "CREATE NODE TABLE IF NOT EXISTS CanonicalMappings("
        "id STRING PRIMARY KEY, entity_id STRING, canonical_entity_id STRING, "
        "previous_canonical_entity_id STRING, changed_at STRING, reverted_at STRING, "
        "reason STRING, active BOOL)"
    ),
)


REVISION_2_STATEMENTS = (
    "ALTER TABLE Artifacts ADD IF NOT EXISTS source_kind STRING",
    "ALTER TABLE Artifacts ADD IF NOT EXISTS artifact_type STRING",
    "ALTER TABLE Artifacts ADD IF NOT EXISTS title STRING",
    "ALTER TABLE Artifacts ADD IF NOT EXISTS short_name STRING",
    "ALTER TABLE Artifacts ADD IF NOT EXISTS summary STRING",
    "ALTER TABLE Artifacts ADD IF NOT EXISTS strategy_family STRING",
    "ALTER TABLE Artifacts ADD IF NOT EXISTS review_status STRING",
    "ALTER TABLE Artifacts ADD IF NOT EXISTS trust_lane STRING",
    "ALTER TABLE Artifacts ADD IF NOT EXISTS evidence_quality DOUBLE",
    "ALTER TABLE Artifacts ADD IF NOT EXISTS maturity STRING",
    "ALTER TABLE Artifacts ADD IF NOT EXISTS license STRING",
    "ALTER TABLE Artifacts ADD IF NOT EXISTS aliases STRING[]",
    "ALTER TABLE Artifacts ADD IF NOT EXISTS capabilities STRING[]",
    "ALTER TABLE Artifacts ADD IF NOT EXISTS limitations STRING[]",
    "ALTER TABLE Artifacts ADD IF NOT EXISTS requirements STRING[]",
    "ALTER TABLE Artifacts ADD IF NOT EXISTS use_cases STRING[]",
    "ALTER TABLE Artifacts ADD IF NOT EXISTS published_at STRING",
    "ALTER TABLE Artifacts ADD IF NOT EXISTS source_updated_at STRING",
    "ALTER TABLE Artifacts ADD IF NOT EXISTS last_verified_at STRING",
    "ALTER TABLE Artifacts ADD IF NOT EXISTS deprecated_at STRING",
    f"ALTER TABLE Chunks ADD IF NOT EXISTS embedding FLOAT[{EMBEDDING_DIMENSION}]",
    "ALTER TABLE Chunks ADD IF NOT EXISTS embedding_provider STRING",
    "ALTER TABLE Chunks ADD IF NOT EXISTS embedding_model STRING",
    "ALTER TABLE Chunks ADD IF NOT EXISTS embedding_revision STRING",
    "ALTER TABLE Chunks ADD IF NOT EXISTS embedding_dimension INT64",
    "ALTER TABLE Chunks ADD IF NOT EXISTS embedding_task_mode STRING",
    "ALTER TABLE Chunks ADD IF NOT EXISTS embedding_normalized BOOL",
    "ALTER TABLE Chunks ADD IF NOT EXISTS source_content_hash STRING",
    "ALTER TABLE Claims ADD IF NOT EXISTS text STRING",
    "ALTER TABLE Claims ADD IF NOT EXISTS confidence DOUBLE",
    "ALTER TABLE EvidenceSpans ADD IF NOT EXISTS quote STRING",
    "ALTER TABLE EvidenceSpans ADD IF NOT EXISTS start_offset INT64",
    "ALTER TABLE EvidenceSpans ADD IF NOT EXISTS end_offset INT64",
    "ALTER TABLE EvidenceSpans ADD IF NOT EXISTS locator STRING",
    "ALTER TABLE Relations ADD IF NOT EXISTS evidence_span_ids STRING[]",
    "ALTER TABLE Relations ADD IF NOT EXISTS rationale STRING",
    "ALTER TABLE ReviewIssues ADD IF NOT EXISTS social_statement STRING",
    "ALTER TABLE ReviewIssues ADD IF NOT EXISTS source_statement STRING",
    "ALTER TABLE ReviewIssues ADD IF NOT EXISTS explanation STRING",
    "ALTER TABLE ReviewIssues ADD IF NOT EXISTS social_source_url STRING",
    "ALTER TABLE ReviewIssues ADD IF NOT EXISTS primary_source_url STRING",
    "ALTER TABLE ReviewIssues ADD IF NOT EXISTS evidence_span_ids STRING[]",
    "ALTER TABLE Entities ADD IF NOT EXISTS aliases STRING[]",
    "ALTER TABLE Concepts ADD IF NOT EXISTS description STRING",
    "CREATE NODE TABLE IF NOT EXISTS SearchTerms(term STRING PRIMARY KEY, kind STRING)",
    "CREATE NODE TABLE IF NOT EXISTS SearchStopwords(word STRING PRIMARY KEY)",
    (
        "CREATE NODE TABLE IF NOT EXISTS SearchIndexState("
        "name STRING PRIMARY KEY, index_type STRING, active_index_name STRING, status STRING, "
        "built_at STRING, row_count INT64, embedding_provider STRING, embedding_model STRING, "
        "embedding_revision STRING, embedding_dimension INT64, embedding_task_mode STRING, "
        "embedding_normalized BOOL)"
    ),
    "CREATE REL TABLE IF NOT EXISTS ArtifactHasSnapshot(FROM Artifacts TO Snapshots)",
    "CREATE REL TABLE IF NOT EXISTS ArtifactHasChunk(FROM Artifacts TO Chunks)",
    "CREATE REL TABLE IF NOT EXISTS ArtifactHasClaim(FROM Artifacts TO Claims)",
    "CREATE REL TABLE IF NOT EXISTS ArtifactHasEntity(FROM Artifacts TO Entities)",
    "CREATE REL TABLE IF NOT EXISTS ArtifactHasConcept(FROM Artifacts TO Concepts)",
    "CREATE REL TABLE IF NOT EXISTS ArtifactHasIssue(FROM Artifacts TO ReviewIssues)",
    "CREATE REL TABLE IF NOT EXISTS ClaimHasEvidence(FROM Claims TO EvidenceSpans)",
    "CREATE REL TABLE IF NOT EXISTS SnapshotHasEvidence(FROM Snapshots TO EvidenceSpans)",
    (
        "CREATE REL TABLE IF NOT EXISTS TermReferencesArtifact("
        "FROM SearchTerms TO Artifacts, weight DOUBLE, kind STRING)"
    ),
    (
        "CREATE REL TABLE IF NOT EXISTS KnowledgeEdges("
        "FROM Artifacts TO Artifacts, FROM Artifacts TO Entities, FROM Artifacts TO Concepts, "
        "FROM Entities TO Artifacts, FROM Entities TO Entities, FROM Entities TO Concepts, "
        "FROM Concepts TO Artifacts, FROM Concepts TO Entities, FROM Concepts TO Concepts, "
        "id STRING, predicate STRING, evidence_span_ids STRING[], rationale STRING, payload STRING)"
    ),
    (
        "CREATE REL TABLE IF NOT EXISTS ArtifactKnowledgeLinks("
        "FROM Artifacts TO Artifacts, relation_id STRING, predicate STRING, "
        "subject_node_id STRING, object_node_id STRING)"
    ),
)

TECHNICAL_FTS_STOPWORDS = (
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
)


def current_revision(connection: lb.Connection) -> int:
    result = connection.execute("MATCH (m:SchemaMigrations) RETURN max(m.revision) AS revision")
    if isinstance(result, list):
        raise RuntimeError("migration query unexpectedly returned multiple result sets")
    row = cast(dict[str, Any] | None, result.rows_as_dict().get_next())
    if row is None or row["revision"] is None:
        return 0
    return int(row["revision"])


def apply_migrations(connection: lb.Connection) -> int:
    revision = current_revision(connection) if _migration_table_exists(connection) else 0
    if revision > SCHEMA_REVISION:
        raise RuntimeError(
            f"database schema revision {revision} is newer than supported revision {SCHEMA_REVISION}"
        )
    if revision < 1:
        for statement in REVISION_1_STATEMENTS:
            connection.execute(statement)
        connection.execute(
            "CREATE (m:SchemaMigrations {revision: $revision, applied_at: $applied_at})",
            {
                "revision": 1,
                "applied_at": datetime.now(UTC).isoformat(),
            },
        )
        revision = 1
    if revision < 2:
        _apply_revision_2(connection)
    return current_revision(connection)


def _rows(
    connection: lb.Connection,
    query: str,
    parameters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    result = connection.execute(query, parameters)
    if isinstance(result, list):
        raise RuntimeError("migration query unexpectedly returned multiple result sets")
    return cast(list[dict[str, Any]], result.rows_as_dict().get_all())


def _apply_revision_2(connection: lb.Connection) -> None:
    before = _source_integrity(connection)
    connection.execute("BEGIN TRANSACTION")
    try:
        for statement in REVISION_2_STATEMENTS:
            connection.execute(statement)
        for word in TECHNICAL_FTS_STOPWORDS:
            connection.execute("MERGE (:SearchStopwords {word: $word})", {"word": word})
        rebuild_native_projections(connection)
        after = _source_integrity(connection)
        if after != before:
            raise RuntimeError("revision 2 migration changed source record counts or content hashes")
        connection.execute(
            "CREATE (m:SchemaMigrations {revision: $revision, applied_at: $applied_at})",
            {"revision": 2, "applied_at": datetime.now(UTC).isoformat()},
        )
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise


def _source_integrity(connection: lb.Connection) -> tuple[tuple[str, int], ...]:
    result: list[tuple[str, int]] = []
    for table in (
        "Artifacts",
        "Snapshots",
        "Chunks",
        "Claims",
        "EvidenceSpans",
        "Relations",
        "ReviewIssues",
        "Entities",
        "Concepts",
        "ArtifactMembers",
    ):
        rows = _rows(connection, f"MATCH (n:{table}) RETURN count(n) AS count")
        result.append((table, int(rows[0]["count"])))
    hashes = _rows(
        connection,
        "MATCH (a:Artifacts) RETURN a.id AS id, a.content_hash AS content_hash ORDER BY a.id",
    )
    result.extend((f"artifact:{row['id']}:{row['content_hash']}", 1) for row in hashes)
    return tuple(result)


def rebuild_native_projections(connection: lb.Connection) -> None:
    """Rebuild revision-2 typed properties and derived graph/search structures from payloads."""

    payloads: dict[str, dict[str, dict[str, Any]]] = {}
    for table in (
        "Artifacts",
        "Snapshots",
        "Chunks",
        "Claims",
        "EvidenceSpans",
        "Relations",
        "ReviewIssues",
        "Entities",
        "Concepts",
    ):
        table_payloads: dict[str, dict[str, Any]] = {}
        for row in _rows(connection, f"MATCH (n:{table}) RETURN n.id AS id, n.payload AS payload"):
            table_payloads[str(row["id"])] = json.loads(str(row["payload"]))
        payloads[table] = table_payloads

    for artifact_id, artifact in payloads["Artifacts"].items():
        connection.execute(
            """MATCH (a:Artifacts {id: $id}) SET
            a.source_kind = $source_kind, a.artifact_type = $artifact_type,
            a.title = $title, a.short_name = $short_name, a.summary = $summary,
            a.strategy_family = $strategy_family, a.review_status = $review_status,
            a.trust_lane = $trust_lane, a.evidence_quality = $evidence_quality,
            a.maturity = $maturity, a.license = $license, a.aliases = $aliases,
            a.capabilities = $capabilities, a.limitations = $limitations,
            a.requirements = $requirements, a.use_cases = $use_cases,
            a.published_at = $published_at, a.source_updated_at = $source_updated_at,
            a.last_verified_at = $last_verified_at, a.deprecated_at = $deprecated_at""",
            {
                "id": artifact_id,
                "source_kind": artifact.get("source_kind"),
                "artifact_type": artifact.get("artifact_type"),
                "title": artifact.get("title"),
                "short_name": artifact.get("short_name"),
                "summary": artifact.get("summary"),
                "strategy_family": artifact.get("strategy_family"),
                "review_status": artifact.get("review_status"),
                "trust_lane": artifact.get("trust_lane"),
                "evidence_quality": artifact.get("evidence_quality"),
                "maturity": artifact.get("maturity"),
                "license": artifact.get("license"),
                "aliases": artifact.get("aliases", []),
                "capabilities": artifact.get("capabilities", []),
                "limitations": artifact.get("limitations", []),
                "requirements": artifact.get("requirements", []),
                "use_cases": artifact.get("use_cases", []),
                "published_at": artifact.get("published_at"),
                "source_updated_at": artifact.get("source_updated_at"),
                "last_verified_at": artifact.get("last_verified_at"),
                "deprecated_at": artifact.get("deprecated_at"),
            },
        )

    for chunk_id, chunk in payloads["Chunks"].items():
        embedding = chunk.get("embedding") or None
        if embedding is not None and len(embedding) != EMBEDDING_DIMENSION:
            embedding = None
        connection.execute(
            """MATCH (n:Chunks {id: $id}) SET n.embedding = $embedding,
            n.embedding_provider = $embedding_provider, n.embedding_model = $embedding_model,
            n.embedding_revision = $embedding_revision,
            n.embedding_dimension = $embedding_dimension,
            n.embedding_task_mode = $embedding_task_mode,
            n.embedding_normalized = $embedding_normalized,
            n.source_content_hash = $source_content_hash""",
            {
                "id": chunk_id,
                "embedding": embedding,
                "embedding_provider": chunk.get("embedding_provider"),
                "embedding_model": chunk.get("embedding_model"),
                "embedding_revision": chunk.get("embedding_revision"),
                "embedding_dimension": (
                    EMBEDDING_DIMENSION if embedding is not None else chunk.get("embedding_dimension")
                ),
                "embedding_task_mode": chunk.get("embedding_task_mode"),
                "embedding_normalized": chunk.get("embedding_normalized"),
                "source_content_hash": chunk.get("source_content_hash"),
            },
        )

    _set_payload_properties(connection, "Claims", payloads["Claims"], ("text", "confidence"))
    _set_payload_properties(
        connection,
        "EvidenceSpans",
        payloads["EvidenceSpans"],
        (("quote", "quote"), ("start_offset", "start"), ("end_offset", "end"), ("locator", "locator")),
    )
    _set_payload_properties(
        connection,
        "Relations",
        payloads["Relations"],
        ("evidence_span_ids", "rationale"),
    )
    _set_payload_properties(
        connection,
        "ReviewIssues",
        payloads["ReviewIssues"],
        (
            "social_statement",
            "source_statement",
            "explanation",
            "social_source_url",
            "primary_source_url",
            "evidence_span_ids",
        ),
    )
    _set_payload_properties(connection, "Entities", payloads["Entities"], ("aliases",))
    _set_payload_properties(connection, "Concepts", payloads["Concepts"], ("description",))
    _rebuild_native_relationships(connection, payloads)


def _set_payload_properties(
    connection: lb.Connection,
    table: str,
    payloads: dict[str, dict[str, Any]],
    properties: tuple[str | tuple[str, str], ...],
) -> None:
    assignments: list[str] = []
    mappings: list[tuple[str, str]] = []
    for item in properties:
        native_name, payload_name = (item, item) if isinstance(item, str) else item
        assignments.append(f"n.{native_name} = ${native_name}")
        mappings.append((native_name, payload_name))
    statement = f"MATCH (n:{table} {{id: $id}}) SET " + ", ".join(assignments)
    for record_id, payload in payloads.items():
        parameters = {native: payload.get(source) for native, source in mappings}
        parameters["id"] = record_id
        connection.execute(statement, parameters)


def _rebuild_native_relationships(
    connection: lb.Connection,
    payloads: dict[str, dict[str, dict[str, Any]]],
) -> None:
    for relation_table in (
        "ArtifactHasSnapshot",
        "ArtifactHasChunk",
        "ArtifactHasClaim",
        "ArtifactHasEntity",
        "ArtifactHasConcept",
        "ArtifactHasIssue",
        "ClaimHasEvidence",
        "SnapshotHasEvidence",
        "TermReferencesArtifact",
        "KnowledgeEdges",
        "ArtifactKnowledgeLinks",
    ):
        connection.execute(f"MATCH ()-[r:{relation_table}]->() DELETE r")
    connection.execute("MATCH (t:SearchTerms) DELETE t")

    member_types = {
        "snapshot": ("Snapshots", "ArtifactHasSnapshot"),
        "chunk": ("Chunks", "ArtifactHasChunk"),
        "claim": ("Claims", "ArtifactHasClaim"),
        "entity": ("Entities", "ArtifactHasEntity"),
        "concept": ("Concepts", "ArtifactHasConcept"),
        "issue": ("ReviewIssues", "ArtifactHasIssue"),
    }
    ownership: dict[str, dict[str, list[str]]] = {}
    members = _rows(
        connection,
        "MATCH (m:ArtifactMembers) RETURN m.artifact_id AS artifact_id, "
        "m.member_type AS member_type, m.member_id AS member_id",
    )
    for member in members:
        member_type = str(member["member_type"])
        mapping = member_types.get(member_type)
        if mapping is None:
            continue
        artifact_id = str(member["artifact_id"])
        member_id = str(member["member_id"])
        node_table, relation_table = mapping
        connection.execute(
            f"""MATCH (a:Artifacts {{id: $artifact_id}}), (n:{node_table} {{id: $member_id}})
            MERGE (a)-[:{relation_table}]->(n)""",
            {"artifact_id": artifact_id, "member_id": member_id},
        )
        ownership.setdefault(artifact_id, {}).setdefault(member_type, []).append(member_id)

    for claim_id, claim in payloads["Claims"].items():
        for span_id in claim.get("evidence_span_ids", []):
            if span_id in payloads["EvidenceSpans"]:
                connection.execute(
                    """MATCH (c:Claims {id: $claim_id}), (s:EvidenceSpans {id: $span_id})
                    MERGE (c)-[:ClaimHasEvidence]->(s)""",
                    {"claim_id": claim_id, "span_id": span_id},
                )
    for span_id, span in payloads["EvidenceSpans"].items():
        snapshot_id = str(span.get("snapshot_id") or "")
        if snapshot_id in payloads.get("Snapshots", {}):
            connection.execute(
                """MATCH (s:Snapshots {id: $snapshot_id}), (e:EvidenceSpans {id: $span_id})
                MERGE (s)-[:SnapshotHasEvidence]->(e)""",
                {"snapshot_id": snapshot_id, "span_id": span_id},
            )

    for artifact_id, artifact in payloads["Artifacts"].items():
        owned = ownership.get(artifact_id, {})
        entity_names = tuple(
            str(payloads["Entities"][identifier].get("name") or "")
            for identifier in owned.get("entity", [])
            if identifier in payloads["Entities"]
        )
        concept_names = tuple(
            str(payloads["Concepts"][identifier].get("name") or "")
            for identifier in owned.get("concept", [])
            if identifier in payloads["Concepts"]
        )
        for term, weight in artifact_search_terms(
            artifact,
            entity_names=entity_names,
            concept_names=concept_names,
        ).items():
            connection.execute(
                """MERGE (t:SearchTerms {term: $term}) SET t.kind = 'artifact_term'
                WITH t MATCH (a:Artifacts {id: $artifact_id})
                MERGE (t)-[r:TermReferencesArtifact]->(a)
                SET r.weight = $weight, r.kind = 'artifact_term'""",
                {"term": term, "artifact_id": artifact_id, "weight": weight},
            )

    node_tables = {
        **{identifier: "Artifacts" for identifier in payloads["Artifacts"]},
        **{identifier: "Entities" for identifier in payloads["Entities"]},
        **{identifier: "Concepts" for identifier in payloads["Concepts"]},
    }
    node_owners: dict[str, set[str]] = {identifier: {identifier} for identifier in payloads["Artifacts"]}
    for artifact_id, owned in ownership.items():
        for member_type in ("entity", "concept"):
            for member_id in owned.get(member_type, []):
                node_owners.setdefault(member_id, set()).add(artifact_id)
    for relation_id, relation in payloads["Relations"].items():
        if not relation.get("approved"):
            continue
        subject_id = str(relation.get("subject_id") or "")
        object_id = str(relation.get("object_id") or "")
        subject_table = node_tables.get(subject_id)
        object_table = node_tables.get(object_id)
        if subject_table is None or object_table is None:
            raise RuntimeError(f"approved relation {relation_id} references a missing node")
        connection.execute(
            f"""MATCH (s:{subject_table} {{id: $subject_id}}), (o:{object_table} {{id: $object_id}})
            MERGE (s)-[r:KnowledgeEdges {{id: $id}}]->(o)
            SET r.predicate = $predicate, r.evidence_span_ids = $evidence_span_ids,
                r.rationale = $rationale, r.payload = $payload""",
            {
                "subject_id": subject_id,
                "object_id": object_id,
                "id": relation_id,
                "predicate": relation.get("predicate"),
                "evidence_span_ids": relation.get("evidence_span_ids", []),
                "rationale": relation.get("rationale"),
                "payload": json.dumps(relation, ensure_ascii=False, separators=(",", ":")),
            },
        )
        subject_owners = sorted(node_owners.get(subject_id, set()))
        object_owners = sorted(node_owners.get(object_id, set()))
        for subject_owner in subject_owners:
            for object_owner in object_owners:
                if subject_owner == object_owner:
                    continue
                connection.execute(
                    """MATCH (s:Artifacts {id: $subject_owner}),
                    (o:Artifacts {id: $object_owner})
                    MERGE (s)-[r:ArtifactKnowledgeLinks {relation_id: $relation_id}]->(o)
                    SET r.predicate = $predicate, r.subject_node_id = $subject_id,
                        r.object_node_id = $object_id""",
                    {
                        "subject_owner": subject_owner,
                        "object_owner": object_owner,
                        "relation_id": relation_id,
                        "predicate": relation.get("predicate"),
                        "subject_id": subject_id,
                        "object_id": object_id,
                    },
                )


def _migration_table_exists(connection: lb.Connection) -> bool:
    result = connection.execute("CALL SHOW_TABLES() RETURN name")
    if isinstance(result, list):
        raise RuntimeError("table query unexpectedly returned multiple result sets")
    rows = result.get_all()
    return any(
        str(row.get("name") if isinstance(row, dict) else row[0]) == "SchemaMigrations" for row in rows
    )
