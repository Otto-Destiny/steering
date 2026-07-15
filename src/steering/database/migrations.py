from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

import ladybug as lb

SCHEMA_REVISION = 1


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
                "revision": SCHEMA_REVISION,
                "applied_at": datetime.now(UTC).isoformat(),
            },
        )
    return current_revision(connection)


def _migration_table_exists(connection: lb.Connection) -> bool:
    result = connection.execute("CALL SHOW_TABLES() RETURN name")
    if isinstance(result, list):
        raise RuntimeError("table query unexpectedly returned multiple result sets")
    rows = result.get_all()
    return any(
        str(row.get("name") if isinstance(row, dict) else row[0]) == "SchemaMigrations" for row in rows
    )
