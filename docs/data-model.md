# Data Model

STEERING stores provenance before interpretation. Source material is immutable; derived knowledge is reviewable
and replaceable.

| Record | Purpose |
| --- | --- |
| `Artifact` | Canonical social post, paper, repository, webpage, file, or text item. |
| `Snapshot` | Immutable captured representation identified by content hash. |
| `Chunk` | Structure-aware retrieval unit linked to a snapshot. |
| `Claim` | Structured statement extracted from an artifact. |
| `EvidenceSpan` | Exact snapshot location supporting a claim. |
| `Entity` | Distinct tool, model, method, organization, benchmark, or other object. |
| `Concept` | Topic or engineering concern used for discovery. |
| `Mention` | Evidence-backed reference from a snapshot to an entity or concept. |
| `Project` | User project and its relevant constraints. |
| `Decision` | Reviewable architecture or training choice. |
| `ExperimentOutcome` | Result and context of testing a decision or idea. |
| `ReviewRun` | Auditable review operation and status changes. |
| `IngestionJob` | Capture/extraction lifecycle, errors, and retry state. |

## Provenance rules

- Every external factual assertion points to an exact evidence span in an immutable snapshot.
- Missing fields remain `null` or empty; extraction never invents values for schema completeness.
- Linked primary sources are separate artifacts, not flattened into the social post.
- Review and evidence status are independent.
- Conflict resolution does not rewrite original claims or snapshots.

## Time and trust

`published_at`, `source_updated_at`, `discovered_at`, `captured_at`, `last_verified_at`, and `deprecated_at` have
different meanings and are stored separately. Dates cannot create quality, supersession, or deprecation edges by
themselves.

The trust lanes are Established, Recent, Promising, Experimental, and Deprecated or incompatible. An unreviewed
social claim can appear only as an explicitly labelled Experimental item.

Schema revision 1 is changed only through explicit migrations and contract tests.
