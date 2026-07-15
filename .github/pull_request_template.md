## What changed

Describe the focused change.

## Why

Explain the problem or decision this addresses.

## User impact

Describe visible behavior, migration needs, and compatibility implications.

## Test evidence

- [ ] `uv run python scripts/check.py`
- [ ] `uv run pip-audit --local --skip-editable --progress-spinner off`
- [ ] Added or updated relevant tests
- [ ] Included manual evidence where automation is insufficient

## Trust and maintenance

- Privacy/security implications:
- Documentation changes:
- Schema revision or migration change: Yes / No
- MCP interface change: Yes / No
- Breaking change: Yes / No

## Checklist

- [ ] No secrets, private captures, databases, or normal browser profiles are included.
- [ ] External factual assertions retain source provenance.
- [ ] Significant architecture changes include an ADR.
- [ ] The changelog is updated when users need to know about the change.
