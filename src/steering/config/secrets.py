from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Protocol

import keyring

SERVICE_NAME = "steering"
SECRET_ROLES = frozenset({"generation", "embedding"})


class KeyringBackend(Protocol):
    def get_password(self, service: str, username: str) -> str | None: ...

    def set_password(self, service: str, username: str, password: str) -> None: ...

    def delete_password(self, service: str, username: str) -> None: ...


class SecretStore(Protocol):
    def get(self, provider_id: str) -> str | None: ...

    def set(self, provider_id: str, secret: str) -> str: ...

    def delete(self, provider_id: str) -> bool: ...

    def fingerprint(self, provider_id: str) -> str | None: ...

    def get_for_role(self, role: str, provider_id: str) -> str | None: ...

    def set_for_role(self, role: str, provider_id: str, secret: str) -> str: ...

    def delete_for_role(self, role: str, provider_id: str) -> bool: ...

    def fingerprint_for_role(self, role: str, provider_id: str) -> str | None: ...


@dataclass(frozen=True, slots=True)
class SecretDescriptor:
    provider_id: str
    fingerprint: str | None
    source: str

    @property
    def masked_fingerprint(self) -> str | None:
        if self.fingerprint is None:
            return None
        return f"****{self.fingerprint.rsplit(':', 1)[-1][-8:]}"


def secret_fingerprint(secret: str) -> str:
    digest = hashlib.sha256(secret.encode("utf-8")).hexdigest()[:16]
    return f"sha256:{digest}"


def provider_secret_env_name(provider_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", provider_id).strip("_").upper()
    if not normalized:
        raise ValueError("provider_id must contain at least one letter or digit")
    return f"STEERING_{normalized}_API_KEY"


def role_secret_env_name(role: str) -> str:
    normalized = role.lower()
    if normalized not in SECRET_ROLES:
        raise ValueError(f"unknown secret role: {role}")
    return f"STEERING_{normalized.upper()}_API_KEY"


def _role_account(role: str, provider_id: str) -> str:
    role_secret_env_name(role)
    return f"{role.lower()}:{provider_id}"


class KeyringSecretStore:
    """Environment-first secret access backed by the operating-system keyring."""

    def __init__(
        self,
        backend: KeyringBackend | None = None,
        *,
        environ: Mapping[str, str] | None = None,
        service_name: str = SERVICE_NAME,
    ) -> None:
        self._backend = backend or keyring.get_keyring()
        self._environ = os.environ if environ is None else environ
        self._service_name = service_name

    def _environment_secret(self, provider_id: str) -> str | None:
        value = self._environ.get(provider_secret_env_name(provider_id))
        return value or None

    def _role_environment_secret(self, role: str, provider_id: str) -> str | None:
        value = self._environ.get(role_secret_env_name(role))
        return value or self._environment_secret(provider_id)

    def get(self, provider_id: str) -> str | None:
        environment_value = self._environment_secret(provider_id)
        if environment_value is not None:
            return environment_value
        return self._backend.get_password(self._service_name, provider_id)

    def get_for_role(self, role: str, provider_id: str) -> str | None:
        environment_value = self._role_environment_secret(role, provider_id)
        if environment_value is not None:
            return environment_value
        role_value = self._backend.get_password(
            self._service_name,
            _role_account(role, provider_id),
        )
        return role_value or self._backend.get_password(self._service_name, provider_id)

    def set(self, provider_id: str, secret: str) -> str:
        if not secret:
            raise ValueError("secret must not be empty")
        self._backend.set_password(self._service_name, provider_id, secret)
        return secret_fingerprint(secret)

    def set_for_role(self, role: str, provider_id: str, secret: str) -> str:
        if not secret:
            raise ValueError("secret must not be empty")
        self._backend.set_password(
            self._service_name,
            _role_account(role, provider_id),
            secret,
        )
        return secret_fingerprint(secret)

    def delete_for_role(self, role: str, provider_id: str) -> bool:
        account = _role_account(role, provider_id)
        if self._backend.get_password(self._service_name, account) is None:
            return False
        self._backend.delete_password(self._service_name, account)
        return True

    def delete(self, provider_id: str) -> bool:
        if self._backend.get_password(self._service_name, provider_id) is None:
            return False
        self._backend.delete_password(self._service_name, provider_id)
        return True

    def fingerprint(self, provider_id: str) -> str | None:
        secret = self.get(provider_id)
        return None if secret is None else secret_fingerprint(secret)

    def fingerprint_for_role(self, role: str, provider_id: str) -> str | None:
        secret = self.get_for_role(role, provider_id)
        return None if secret is None else secret_fingerprint(secret)

    def describe(self, provider_id: str) -> SecretDescriptor:
        environment_value = self._environment_secret(provider_id)
        source = "environment" if environment_value is not None else "keyring"
        return SecretDescriptor(provider_id, self.fingerprint(provider_id), source)

    def describe_for_role(self, role: str, provider_id: str) -> SecretDescriptor:
        environment_value = self._role_environment_secret(role, provider_id)
        source = "environment" if environment_value is not None else "keyring"
        return SecretDescriptor(
            provider_id,
            self.fingerprint_for_role(role, provider_id),
            source,
        )

    @property
    def backend_name(self) -> str:
        return type(self._backend).__name__


class MemorySecretBackend:
    """Small injectable backend for tests and non-persistent development."""

    def __init__(self) -> None:
        self._values: MutableMapping[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self._values.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self._values[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        del self._values[(service, username)]
