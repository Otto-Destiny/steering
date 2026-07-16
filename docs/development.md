# Development

## Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- Git

## Environment

```text
uv sync --all-extras --dev
uv run pre-commit install
```

`pyproject.toml` is the only dependency and tool-configuration source. Commit `uv.lock`; do not add a manually
maintained `requirements.txt`. `--all-extras` installs contributor coverage for browser capture and the optional
FastEmbed runtime; the base application does not depend on either extra.

To install and verify the pinned ONNX model, use the explicit local-model workflow:

```text
uv run steering local-embeddings status
uv run steering local-embeddings install --accept-download
```

The model is downloaded only by the explicit install command. Never add model files, caches, API keys, or
`.env.local` to the repository.

## Checks

The canonical local and CI command is:

```text
uv run python scripts/check.py
uv run pip-audit --local --skip-editable --progress-spinner off
```

It checks the lockfile, lint, formatting, strict source typing, tests and coverage, package build, wheel metadata,
and a clean-wheel import in that order. Useful focused commands are:

```text
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest tests/unit
uv run pytest -m integration
uv run pytest -m "e2e and not live"
```

Live tests are opt-in, require explicit credentials, and must never run for forks or upload unsanitized logs.
Native retrieval integration tests must exercise Ladybug FTS/BM25, HNSW, exact-term lookup, and relationship
traversal against a disposable database.

## Test layout

- `tests/unit`: isolated behavior.
- `tests/contract`: schemas and contributor-facing protocols.
- `tests/integration`: disposable Ladybug databases and composed services.
- `tests/e2e`: interface flows using local fixture pages.
- `tests/fixtures`: synthetic, redistributable data with no secrets or private captures.

## Pull requests

Document behavior, user impact, evidence, security/privacy implications, and any schema or MCP change. Add a short
ADR for a significant architectural decision. See [CONTRIBUTING.md](../CONTRIBUTING.md).
