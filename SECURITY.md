# Security Policy

## Supported versions

| Version | Supported |
| --- | --- |
| 0.1.x | Yes |
| Older pre-release snapshots | No |

Until 1.0, security fixes may include necessary schema or interface changes. These changes will be documented.

## Report a vulnerability privately

Use GitHub's **Security → Report a vulnerability** flow for this repository. If private vulnerability reporting is
temporarily unavailable, contact the repository owner through their GitHub profile to request a private channel.
Do not include vulnerability details in a public issue, discussion, or pull request.

Include the affected version, reproduction steps, impact, and any suggested mitigation. Remove API keys, personal
content, browser profiles, databases, and other secrets from reports.

We aim to acknowledge reports within three business days and provide an initial assessment within seven business
days. Timelines for a fix depend on severity and complexity. We will coordinate disclosure and credit with the
reporter.

## Security boundaries

STEERING stores provider secrets in the operating-system credential store, isolates authorized browser capture,
and treats fetched content as untrusted. Reports involving secret leakage, SSRF, access-control bypass, prompt
injection across trust boundaries, unsafe archive/file handling, or evidence corruption are especially important.
