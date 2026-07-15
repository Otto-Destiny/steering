# ADR 0002: Operating-system secret storage

- Status: Accepted
- Date: 2026-07-15

## Context

Provider credentials must survive restarts without leaking into portable project data, logs, or backups.

## Decision

Use Python `keyring` with the operating-system credential store. Allow environment overrides for CI and headless
use. Never persist secret values in Ladybug, application configuration, snapshots, browser profiles, or backups.

## Consequences

Configuration can store non-secret provider metadata and masked fingerprints. Cross-platform behavior needs tests
on each supported operating system, and headless Linux deployments may need an explicitly configured backend.
