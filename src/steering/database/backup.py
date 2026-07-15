from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import ladybug as lb

from steering.database.migrations import SCHEMA_REVISION, apply_migrations, current_revision
from steering.domain.credentials import reject_high_confidence_credentials

BACKUP_FORMAT = 1
RECORDS_NAME = "records.json"
MANIFEST_NAME = "manifest.json"

# Explicit columns prevent accidental inclusion of future tables or local configuration.
BACKUP_TABLES: dict[str, tuple[str, ...]] = {
    "Artifacts": ("id", "canonical_url", "content_hash", "payload"),
    "UrlIdentities": ("canonical_url", "artifact_id"),
    "Snapshots": ("id", "artifact_id", "content_hash", "payload"),
    "Chunks": ("id", "artifact_id", "snapshot_id", "ordinal", "text", "locator", "payload"),
    "Claims": ("id", "artifact_id", "category", "payload"),
    "EvidenceSpans": ("id", "snapshot_id", "claim_id", "payload"),
    "Relations": ("id", "subject_id", "predicate", "object_id", "approved", "payload"),
    "ReviewIssues": (
        "id",
        "artifact_id",
        "status",
        "created_at",
        "resolved_at",
        "payload",
    ),
    "Entities": ("id", "name", "entity_type", "canonical_entity_id", "payload"),
    "Concepts": ("id", "name", "payload"),
    "Mentions": ("id", "artifact_id", "entity_id", "payload"),
    "ArtifactMembers": ("id", "artifact_id", "member_type", "member_id"),
    "Projects": ("id", "name", "created_at", "payload"),
    "Decisions": ("id", "project_id", "artifact_id", "created_at", "payload"),
    "ExperimentOutcomes": (
        "id",
        "project_id",
        "decision_id",
        "artifact_id",
        "recorded_at",
        "payload",
    ),
    "ReviewRuns": ("id", "project_id", "created_at", "payload"),
    "IngestionJobs": ("id", "status", "created_at", "updated_at", "payload"),
    "CanonicalMappings": (
        "id",
        "entity_id",
        "canonical_entity_id",
        "previous_canonical_entity_id",
        "changed_at",
        "reverted_at",
        "reason",
        "active",
    ),
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def create_backup(connection: lb.Connection, destination: str | Path) -> Path:
    """Create a consistent logical backup containing only revision-1 database records."""

    target = Path(destination).expanduser().resolve(strict=False)
    if target.exists():
        raise FileExistsError(f"backup destination already exists: {target}")
    target.mkdir(parents=True)
    try:
        records: dict[str, list[dict[str, Any]]] = {}
        connection.execute("BEGIN TRANSACTION")
        try:
            for table, columns in BACKUP_TABLES.items():
                projections = ", ".join(f"n.{column} AS {column}" for column in columns)
                query = f"MATCH (n:{table}) RETURN {projections} ORDER BY n.{columns[0]}"
                result = connection.execute(query)
                if isinstance(result, list):
                    raise RuntimeError("backup query returned multiple result sets")
                records[table] = cast(list[dict[str, Any]], result.rows_as_dict().get_all())
            revision = current_revision(connection)
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise

        serialized = (
            json.dumps(records, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
        ).encode("utf-8")
        reject_high_confidence_credentials(serialized.decode("utf-8"))
        (target / RECORDS_NAME).write_bytes(serialized)
        manifest = {
            "backup_format": BACKUP_FORMAT,
            "schema_revision": revision,
            "created_at": datetime.now(UTC).isoformat(),
            "records_sha256": _sha256_bytes(serialized),
        }
        (target / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return target


def restore_backup(backup: str | Path, database_path: str | Path) -> Path:
    """Restore a verified logical backup to a new path; never overwrite a database."""

    source = Path(backup).expanduser().resolve(strict=True)
    target = Path(database_path).expanduser().resolve(strict=False)
    if target.exists():
        raise FileExistsError(f"restore target already exists: {target}")
    manifest = json.loads((source / MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("backup_format") != BACKUP_FORMAT:
        raise ValueError("unsupported backup format")
    if int(manifest.get("schema_revision", -1)) > SCHEMA_REVISION:
        raise ValueError("backup schema is newer than this application supports")
    serialized = (source / RECORDS_NAME).read_bytes()
    if _sha256_bytes(serialized) != manifest.get("records_sha256"):
        raise ValueError("backup records checksum does not match manifest")
    reject_high_confidence_credentials(serialized.decode("utf-8"))
    records = json.loads(serialized)
    target.parent.mkdir(parents=True, exist_ok=True)
    database = lb.Database(str(target))
    connection = lb.Connection(database)
    try:
        apply_migrations(connection)
        connection.execute("BEGIN TRANSACTION")
        try:
            for table, columns in BACKUP_TABLES.items():
                primary_key = columns[0]
                assignments = ", ".join(
                    f"n.{column} = ${column}" for column in columns if column != primary_key
                )
                query = f"MERGE (n:{table} {{{primary_key}: ${primary_key}}})"
                if assignments:
                    query += f" SET {assignments}"
                for row in records.get(table, []):
                    connection.execute(query, row)
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    except BaseException:
        connection.close()
        database.close()
        target.unlink(missing_ok=True)
        raise
    connection.close()
    database.close()
    return target
