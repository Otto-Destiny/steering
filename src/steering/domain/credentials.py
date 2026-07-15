from __future__ import annotations

import re
from hashlib import sha256
from urllib.parse import parse_qsl, urlsplit


class CredentialDetectedError(ValueError):
    """Raised without echoing credential-bearing input."""


_HIGH_CONFIDENCE_CREDENTIALS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\bsk-(?:ant-api\d{2}-|proj-|svcacct-)?[A-Za-z0-9_-]{32,}\b",
        r"\bgithub_pat_[A-Za-z0-9_]{30,}\b",
        r"\bgh[pousr]_[A-Za-z0-9]{36,}\b",
        r"\bhf_[A-Za-z0-9]{30,}\b",
        r"\bAIza[0-9A-Za-z_-]{35}\b",
        r"\bAKIA[0-9A-Z]{16}\b",
        r"\bxox[baprs]-[0-9A-Za-z-]{24,}\b",
        r"\bsk_live_[0-9A-Za-z]{20,}\b",
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    )
)


def contains_high_confidence_credential(value: str) -> bool:
    return any(pattern.search(value) is not None for pattern in _HIGH_CONFIDENCE_CREDENTIALS)


def reject_high_confidence_credentials(value: str) -> str:
    if contains_high_confidence_credential(value):
        raise CredentialDetectedError(
            "content appears to contain a credential and was refused before persistence"
        )
    return value


_SENSITIVE_QUERY_KEYS = frozenset(
    {"access_token", "api_key", "apikey", "authorization", "key", "password", "secret", "token"}
)


def credential_bearing_source(value: str) -> bool:
    if contains_high_confidence_credential(value):
        return True
    parsed = urlsplit(value.strip())
    if parsed.username is not None or parsed.password is not None:
        return True
    return any(
        key.lower() in _SENSITIVE_QUERY_KEYS and bool(item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
    )


def sanitized_persistence_source(value: str) -> str:
    if not credential_bearing_source(value):
        return value
    digest = sha256(value.encode("utf-8")).hexdigest()
    return f"redacted://sha256/{digest}"


def reject_credential_bearing_source(value: str) -> str:
    if credential_bearing_source(value):
        raise CredentialDetectedError(
            "source URL appears to contain a credential and was refused before persistence"
        )
    return value
