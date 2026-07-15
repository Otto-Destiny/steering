# ADR 0003: Source-backed social ingestion

- Status: Accepted
- Date: 2026-07-15

## Context

Social posts are useful discovery context but frequently compress, omit, or overstate a linked technical source.

## Decision

Bundle same-author social replies into one artifact. Preserve a linked paper, repository, benchmark, or
documentation page as a separate primary artifact. Reconcile claims with exact evidence while retaining both
immutable snapshots and a simple review issue for material inconsistencies.

## Consequences

Recommendations can distinguish the author's framing from primary evidence. Ingestion is more involved, but
conflicts remain visible and reviewable instead of being silently flattened.
