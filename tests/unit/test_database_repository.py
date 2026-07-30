from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from steering.database import (
    SCHEMA_REVISION,
    CanonicalMappingError,
    DatabaseRuntime,
    ImmutableRecordError,
    apply_migrations,
    current_revision,
    restore_backup,
)
from steering.database.repository import CorruptRecordError
from steering.database.runtime import DatabaseCorruptedError
from steering.domain.models import (
    Artifact,
    ArtifactRecord,
    ArtifactType,
    Claim,
    Concept,
    Decision,
    Entity,
    EvidenceCategory,
    EvidenceSpan,
    ExperimentOutcome,
    IngestionJob,
    IssueStatus,
    JobStatus,
    Project,
    Relation,
    RelationType,
    ReviewIssue,
    ReviewRun,
    Snapshot,
    SourceKind,
    TrustLane,
)


def make_record(*, artifact_id: str = "art_one", url: str = "https://example.com/paper") -> ArtifactRecord:
    artifact = Artifact(
        id=artifact_id,
        canonical_url=url,
        source_kind=SourceKind.PAPER,
        artifact_type=ArtifactType.PAPER,
        title="A paper",
        summary="A useful method.",
        content_hash="artifact-hash",
    )
    snapshot = Snapshot(
        id="snap_one",
        artifact_id=artifact_id,
        source_url=url,
        content_hash="snapshot-hash",
        mime_type="text/html",
        text="Supported claim.",
        extraction_method="html",
    )
    claim = Claim(
        id="claim_one",
        artifact_id=artifact_id,
        text="Supported claim.",
        category=EvidenceCategory.RESEARCH_PAPER,
        confidence=0.9,
        evidence_span_ids=["span_one"],
    )
    span = EvidenceSpan(
        id="span_one",
        snapshot_id=snapshot.id,
        claim_id=claim.id,
        quote="Supported",
        start=0,
        end=9,
        locator="section:abstract",
    )
    entity = Entity(id="entity_one", name="Method", entity_type="method")
    concept = Concept(id="concept_one", name="Reasoning")
    relation = Relation(
        id="relation_one",
        subject_id=artifact_id,
        predicate=RelationType.INTRODUCES,
        object_id=entity.id,
        approved=True,
        evidence_span_ids=[span.id],
    )
    issue = ReviewIssue(
        id="issue_one",
        artifact_id=artifact_id,
        social_statement="The result is perfect.",
        source_statement="The result has limitations.",
        explanation="The post overstates the paper.",
        social_source_url="https://example.com/post",
        primary_source_url=url,
        evidence_span_ids=[span.id],
    )
    return ArtifactRecord(
        artifact=artifact,
        snapshots=[snapshot],
        claims=[claim],
        evidence_spans=[span],
        relations=[relation],
        issues=[issue],
        entities=[entity],
        concepts=[concept],
    )


@pytest.mark.integration
def test_revision_one_is_idempotent_and_covers_core_records(tmp_path: Path) -> None:
    with DatabaseRuntime(tmp_path / "schema.lbug") as runtime:
        assert current_revision(runtime.connection) == SCHEMA_REVISION
        assert apply_migrations(runtime.connection) == SCHEMA_REVISION
        tables = {row[0] for row in runtime.connection.execute("CALL SHOW_TABLES() RETURN name").get_all()}

    assert {
        "Artifacts",
        "Snapshots",
        "Chunks",
        "Claims",
        "EvidenceSpans",
        "Relations",
        "ReviewIssues",
        "Entities",
        "Concepts",
        "Mentions",
        "Projects",
        "Decisions",
        "ExperimentOutcomes",
        "ReviewRuns",
        "IngestionJobs",
        "CanonicalMappings",
    } <= tables


@pytest.mark.integration
def test_aggregate_round_trip_url_dedup_and_immutable_evidence(tmp_path: Path) -> None:
    with DatabaseRuntime(tmp_path / "records.lbug") as runtime:
        repository = runtime.repository
        original = repository.upsert_record(make_record())
        assert repository.get_record("art_one") == original
        assert repository.get_by_url("https://example.com/paper") == original

        duplicate_artifact = make_record(artifact_id="art_duplicate").artifact
        duplicate_artifact.title = "Updated paper title"
        duplicate = ArtifactRecord(artifact=duplicate_artifact)
        deduplicated = repository.upsert_record(duplicate)
        assert deduplicated.artifact.id == "art_one"
        assert deduplicated.artifact.title == "Updated paper title"
        assert len(repository.list_records()) == 1

        changed_snapshot = deduplicated.model_copy(deep=True)
        changed_snapshot.snapshots[0].text = "Changed source text."
        with pytest.raises(ImmutableRecordError, match="snap_one"):
            repository.upsert_record(changed_snapshot)

        changed_span = deduplicated.model_copy(deep=True)
        changed_span.evidence_spans[0].quote = "Different"
        with pytest.raises(ImmutableRecordError, match="span_one"):
            repository.upsert_record(changed_span)


@pytest.mark.integration
def test_metadata_update_skips_unchanged_member_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with DatabaseRuntime(tmp_path / "unchanged-members.lbug") as runtime:
        repository = runtime.repository
        original = repository.upsert_record(make_record())
        saved_member_ids: list[str] = []
        save_model = repository._save_model

        def record_save(model: BaseModel) -> None:
            saved_member_ids.append(str(model.model_dump()["id"]))
            save_model(model)

        monkeypatch.setattr(repository, "_save_model", record_save)
        metadata_update = original.model_copy(deep=True)
        metadata_update.artifact.summary = "Updated artifact metadata only."
        repository.upsert_record(metadata_update)
        assert saved_member_ids == []

        changed_member = metadata_update.model_copy(deep=True)
        changed_member.claims[0].confidence = 0.8
        repository.upsert_record(changed_member)
        assert saved_member_ids == ["claim_one"]


@pytest.mark.integration
def test_jobs_issues_project_history_and_reversible_canonical_mapping(tmp_path: Path) -> None:
    with DatabaseRuntime(tmp_path / "workflow.lbug") as runtime:
        repository = runtime.repository
        repository.upsert_record(make_record())

        job = IngestionJob(id="job_one", source="https://example.com", status=JobStatus.COMPLETED)
        repository.save_job(job)
        assert repository.list_jobs(limit=1) == [job]

        issue = repository.resolve_issue("issue_one", "keep_both")
        assert issue.status == IssueStatus.KEPT_BOTH
        assert issue.resolved_at is not None
        assert repository.list_issues(unresolved_only=True) == []

        project = repository.save_project(Project(id="project_one", name="Agent architecture"))
        decision = repository.save_decision(
            Decision(
                id="decision_one",
                project_id=project.id,
                artifact_id="art_one",
                decision="Trial the method",
                rationale="Evidence fits the constraints",
            )
        )
        outcome = repository.save_outcome(
            ExperimentOutcome(
                id="outcome_one",
                project_id=project.id,
                decision_id=decision.id,
                artifact_id="art_one",
                outcome="Latency was too high",
                succeeded=False,
            )
        )
        review = repository.save_review_run(
            ReviewRun(id="review_one", project_id=project.id, query="memory architecture")
        )
        history = repository.project_history(project.id)
        assert history["projects"] == [project]
        assert history["decisions"] == [decision]
        assert history["outcomes"] == [outcome]
        assert history["review_runs"] == [review]

        alias = repository.save_entity(Entity(id="entity_alias", name="PTRM", entity_type="method"))
        canonical = repository.save_entity(
            Entity(id="entity_canonical", name="Probabilistic Tiny Recursive Model", entity_type="method")
        )
        mapped = repository.set_canonical_entity(alias.id, canonical.id, reason="expanded alias")
        assert mapped.canonical_entity_id == canonical.id
        with pytest.raises(CanonicalMappingError):
            repository.set_canonical_entity(canonical.id, alias.id)
        restored = repository.undo_canonical_mapping(alias.id)
        assert restored.canonical_entity_id is None
        mapping_history = repository.canonical_mapping_history(alias.id)
        assert len(mapping_history) == 1
        assert mapping_history[0]["active"] is False
        assert mapping_history[0]["reverted_at"] is not None


@pytest.mark.integration
def test_backup_restore_round_trip(tmp_path: Path) -> None:
    database_path = tmp_path / "source.lbug"
    backup_path = tmp_path / "backup"
    with DatabaseRuntime(database_path) as runtime:
        runtime.repository.upsert_record(make_record())
        assert Path(runtime.repository.backup(str(backup_path))) == backup_path

    restored_path = restore_backup(backup_path, tmp_path / "restored.lbug")
    with DatabaseRuntime(restored_path) as restored:
        record = restored.repository.get_by_url("https://example.com/paper")
        assert record is not None
        assert record.artifact.id == "art_one"

    with pytest.raises(FileExistsError):
        restore_backup(backup_path, restored_path)


def test_sensitive_relation_approval_requires_and_accepts_explicit_evidence() -> None:
    with pytest.raises(ValidationError, match="sensitive relations require explicit evidence"):
        Relation(
            subject_id="art_one",
            predicate=RelationType.SUPERSEDES,
            object_id="art_two",
            approved=True,
        )

    relation = Relation(
        subject_id="art_one",
        predicate=RelationType.SUPERSEDES,
        object_id="art_two",
        approved=True,
        evidence_span_ids=["span_one"],
    )
    assert relation.approved is True


@pytest.mark.integration
def test_repository_refuses_established_artifact_with_unresolved_issue(tmp_path: Path) -> None:
    record = make_record()
    record.artifact.review_status = "reviewed"
    record.artifact.trust_lane = TrustLane.ESTABLISHED

    with (
        DatabaseRuntime(tmp_path / "trust-invariant.lbug") as runtime,
        pytest.raises(ValidationError, match="unresolved issues cannot enter"),
    ):
        runtime.repository.upsert_record(record)


@pytest.mark.integration
def test_listing_views_read_headers_without_hydrating_whole_records(tmp_path: Path) -> None:
    """Overview pages must not cost a full graph load that grows with the corpus."""

    with DatabaseRuntime(tmp_path / "listing.lbug") as runtime:
        repository = runtime.repository
        assert repository.count_artifacts() == 0
        assert repository.list_artifacts() == []

        for index in range(5):
            record = make_record(
                artifact_id=f"art_{index}",
                url=f"https://example.com/paper-{index}",
            )
            record.artifact.captured_at = datetime(2026, 1, index + 1, tzinfo=UTC)
            record.artifact.title = f"Paper {index}"
            repository.upsert_record(record)

        assert repository.count_artifacts() == 5

        newest_first = repository.list_artifacts()
        assert [artifact.title for artifact in newest_first] == [
            "Paper 4",
            "Paper 3",
            "Paper 2",
            "Paper 1",
            "Paper 0",
        ]
        assert [artifact.title for artifact in repository.list_artifacts(limit=2)] == [
            "Paper 4",
            "Paper 3",
        ]
        # Headers carry what listing views render, and nothing they do not need.
        assert all(isinstance(artifact, Artifact) for artifact in newest_first)


def test_a_corrupt_write_ahead_log_explains_how_to_recover(tmp_path: Path) -> None:
    """The raw storage error leaves a user one command from a working database."""

    database_path = tmp_path / "damaged.lbug"
    with DatabaseRuntime(database_path) as runtime:
        runtime.repository.upsert_record(make_record())
    wal = Path(f"{database_path}.wal")
    wal.write_bytes(b"not a valid write-ahead log")

    with pytest.raises(DatabaseCorruptedError) as error:
        DatabaseRuntime(database_path)

    message = str(error.value)
    assert str(wal) in message
    assert "do not delete" in message
    assert "last checkpoint" in message


def test_recovery_succeeds_once_the_corrupt_log_is_set_aside(tmp_path: Path) -> None:
    database_path = tmp_path / "recovered.lbug"
    with DatabaseRuntime(database_path) as runtime:
        runtime.repository.upsert_record(make_record())
    wal = Path(f"{database_path}.wal")
    wal.write_bytes(b"not a valid write-ahead log")
    wal.rename(tmp_path / "set-aside.wal")

    with DatabaseRuntime(database_path) as runtime:
        assert runtime.repository.count_artifacts() == 1


def _distinct_record(tag: str) -> ArtifactRecord:
    """A record whose members are its own, so removal can be observed cleanly."""

    record = make_record(artifact_id=f"art_{tag}", url=f"https://example.com/{tag}")
    for snapshot in record.snapshots:
        snapshot.id, snapshot.artifact_id = f"snap_{tag}", f"art_{tag}"
    for claim in record.claims:
        claim.id, claim.artifact_id = f"claim_{tag}", f"art_{tag}"
        claim.evidence_span_ids = [f"span_{tag}"]
    for span in record.evidence_spans:
        span.id, span.snapshot_id, span.claim_id = f"span_{tag}", f"snap_{tag}", f"claim_{tag}"
    for relation in record.relations:
        relation.id, relation.subject_id = f"rel_{tag}", f"art_{tag}"
        relation.evidence_span_ids = [f"span_{tag}"]
    for issue in record.issues:
        issue.id, issue.artifact_id = f"issue_{tag}", f"art_{tag}"
        issue.evidence_span_ids = [f"span_{tag}"]
    return record


@pytest.mark.integration
def test_retiring_a_record_removes_it_and_frees_its_url(tmp_path: Path) -> None:
    with DatabaseRuntime(tmp_path / "retire.lbug") as runtime:
        repository = runtime.repository
        repository.upsert_record(_distinct_record("solo"))

        assert repository.delete_record("art_solo") is True
        assert repository.count_artifacts() == 0
        assert repository.get_record("art_solo") is None
        # The URL is released, so the same source can be captured again.
        assert repository.get_by_url("https://example.com/solo") is None
        # Members owned only by that artifact go with it.
        assert repository._rows("MATCH (n:Snapshots) RETURN count(n) AS c")[0]["c"] == 0
        assert repository.delete_record("art_solo") is False


@pytest.mark.integration
def test_a_record_whose_stored_data_is_unreadable_can_still_be_retired(tmp_path: Path) -> None:
    """The damaged record is exactly the one a user most needs to remove, so
    removal must not depend on being able to read it."""

    with DatabaseRuntime(tmp_path / "corrupt.lbug") as runtime:
        repository = runtime.repository
        repository.upsert_record(_distinct_record("ok"))
        repository.upsert_record(_distinct_record("bad"))
        # An interrupted write leaves a payload that no longer matches its model.
        repository._connection.execute(
            "MATCH (n:Snapshots {id: 'snap_bad'}) SET n.payload = $payload",
            {"payload": '{"id":"snap_bad","artifact_id":"art_bad"}'},
        )

        with pytest.raises(CorruptRecordError) as error:
            repository.get_record("art_bad")
        assert error.value.artifact_id == "art_bad"
        assert "source_url" in error.value.detail

        assert repository.delete_record("art_bad") is True
        assert repository.count_artifacts() == 1
        # The healthy record is untouched.
        healthy = repository.get_record("art_ok")
        assert healthy is not None
        assert len(healthy.snapshots) == 1


@pytest.mark.integration
def test_a_member_shared_with_another_artifact_survives_retirement(tmp_path: Path) -> None:
    with DatabaseRuntime(tmp_path / "shared.lbug") as runtime:
        repository = runtime.repository
        # make_record reuses fixed member ids, so these two share every member.
        repository.upsert_record(make_record(artifact_id="art_a", url="https://example.com/a"))
        repository.upsert_record(make_record(artifact_id="art_b", url="https://example.com/b"))

        assert repository.delete_record("art_a") is True

        survivor = repository.get_record("art_b")
        assert survivor is not None
        assert len(survivor.snapshots) == 1
        assert len(survivor.claims) == 1
