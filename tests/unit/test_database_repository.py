from __future__ import annotations

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
