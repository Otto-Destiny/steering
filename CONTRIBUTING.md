# Contributing to STEERING

Thank you for helping build a trustworthy, local-first knowledge system for AI engineers.

## Set up the project

STEERING uses Python 3.11 and [uv](https://docs.astral.sh/uv/).

```text
git clone https://github.com/Otto-Destiny/steering.git
cd steering
uv sync --all-extras --dev
uv run pre-commit install
```

Run the same checks used by CI:

```text
uv run python scripts/check.py
uv run pip-audit --local --skip-editable --progress-spinner off
```

See [docs/development.md](docs/development.md) for focused test commands and project structure.

## Make a change

1. Open or reference an issue for substantial work.
2. Create a focused branch from `main`.
3. Add tests for behavior changes and update relevant documentation.
4. Run the canonical check command.
5. Open a pull request describing user impact, test evidence, privacy/security implications, and contract changes.

Keep pull requests small enough to review. Do not include credentials, captured private content, database files,
or planning files. Significant architectural decisions need a short ADR under `docs/adr/`.

## Resolver and provider contributions

New resolvers require canonical URL tests, successful and unavailable/private fixtures, malformed-input handling,
provenance-locator tests, prompt-injection tests, and documented platform or credential limitations. Providers and
resolvers register through internal protocols; these extension points are experimental in 0.1 and are not a stable
third-party plugin API.

## Testing expectations

- Unit tests cover isolated logic.
- Contract tests protect schemas and contributor interfaces.
- Integration tests use disposable local data stores.
- Browser E2E tests use local fixture pages, never live accounts.
- Live tests are explicit opt-in checks and must sanitize output.

Bug fixes should include a regression test. The repository coverage floor is 85%.

## Conduct, security, and licensing

Follow the [Code of Conduct](CODE_OF_CONDUCT.md). Report vulnerabilities through the private process in
[SECURITY.md](SECURITY.md), not a public issue. Contributions are submitted under the repository's
[Apache-2.0 license](LICENSE); no CLA is required.
