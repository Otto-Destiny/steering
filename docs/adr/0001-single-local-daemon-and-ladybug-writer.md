# ADR 0001: Single local daemon and Ladybug writer

- Status: Accepted
- Date: 2026-07-15

## Context

STEERING needs CLI, web, and MCP access to one embedded knowledge graph without competing writers.

## Decision

One localhost Starlette/Uvicorn daemon serves all interfaces and owns one writable Ladybug `Database` object for
its lifetime. Fetching and model work happens outside database transactions.

## Consequences

All state-changing clients go through the daemon. This simplifies consistency and backup behavior, but the daemon
is a required local process and long external calls must never hold write transactions.
