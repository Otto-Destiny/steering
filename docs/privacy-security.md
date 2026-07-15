# Privacy and Security

STEERING is local-first, but it processes untrusted web content and valuable personal research history. Local
execution does not remove the need for strict boundaries.

## Secrets

Provider keys are stored through Python `keyring` in Windows Credential Manager, macOS Keychain, or Linux Secret
Service. Environment variables are supported for CI and headless deployments. Secrets must never enter database
records, snapshots, configuration files, logs, backups, exceptions, fixtures, or browser responses. The UI can
replace or delete a key but can never read it back.

## Browser capture

Authorized capture uses a visible Playwright Chromium profile owned by STEERING. It never reads a normal browser
profile, asks for a password, automates bulk timelines, or bypasses a paywall or access control. Public extraction
is attempted before browser capture. Downloaded media is discarded after extraction; provenance retains the URL
and inclusion decision.

## Untrusted sources

- Validate URL schemes, redirects, resolved hosts, response size, and content type before following links.
- Treat page instructions as data, never as authority to call tools, disclose secrets, or change review state.
- Follow at most the authorized source-selection path; do not crawl broadly.
- Validate exact evidence spans before promoting generated claims.
- Mark inaccessible or incomplete content as partial instead of bypassing controls.

## Local service and backups

The web service binds only to loopback. State-changing browser requests require a localhost Host header,
same-origin validation, cross-site rejection, bounded request bodies, and typed input validation. MCP transport
also enables DNS-rebinding protection. Backups contain knowledge data but not credential-store secrets or browser
profiles. Restore and migration paths preserve snapshot hashes and review history.

Report vulnerabilities through [SECURITY.md](../SECURITY.md).
