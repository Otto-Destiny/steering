from __future__ import annotations

from pathlib import Path

from steering.config import (
    AppConfig,
    ConfigStore,
    KeyringSecretStore,
    MemorySecretBackend,
    provider_secret_env_name,
    role_secret_env_name,
)
from steering.database import DatabaseRuntime
from steering.domain.models import ProviderConfig


def test_keyring_store_is_injectable_masked_and_environment_first() -> None:
    backend = MemorySecretBackend()
    environment: dict[str, str] = {}
    secrets = KeyringSecretStore(backend, environ=environment)

    fingerprint = secrets.set("openai-compatible", "keyring-canary-secret")
    assert fingerprint.startswith("sha256:")
    assert secrets.get("openai-compatible") == "keyring-canary-secret"
    descriptor = secrets.describe("openai-compatible")
    assert descriptor.masked_fingerprint is not None
    assert "keyring-canary-secret" not in descriptor.masked_fingerprint
    assert descriptor.source == "keyring"

    environment[provider_secret_env_name("openai-compatible")] = "environment-canary-secret"
    assert secrets.get("openai-compatible") == "environment-canary-secret"
    assert secrets.describe("openai-compatible").source == "environment"
    del environment[provider_secret_env_name("openai-compatible")]
    assert secrets.delete("openai-compatible") is True
    assert secrets.delete("openai-compatible") is False


def test_role_specific_secrets_stay_separate_for_one_provider() -> None:
    backend = MemorySecretBackend()
    environment: dict[str, str] = {}
    secrets = KeyringSecretStore(backend, environ=environment)

    secrets.set_for_role("generation", "shared-provider", "generation-keyring-key")
    secrets.set_for_role("embedding", "shared-provider", "embedding-keyring-key")
    assert secrets.get_for_role("generation", "shared-provider") == "generation-keyring-key"
    assert secrets.get_for_role("embedding", "shared-provider") == "embedding-keyring-key"

    environment[role_secret_env_name("generation")] = "generation-environment-key"
    environment[role_secret_env_name("embedding")] = "embedding-environment-key"
    environment[provider_secret_env_name("shared-provider")] = "provider-alias-key"
    assert secrets.get_for_role("generation", "shared-provider") == "generation-environment-key"
    assert secrets.get_for_role("embedding", "shared-provider") == "embedding-environment-key"


def test_simple_provider_key_is_available_to_both_roles_only_for_selected_provider() -> None:
    secrets = KeyringSecretStore(
        MemorySecretBackend(),
        environ={
            "STEERING_PROVIDER": "gemini",
            "STEERING_API_KEY": "simple-key",
        },
    )
    assert secrets.get_for_role("generation", "gemini") == "simple-key"
    assert secrets.get_for_role("embedding", "gemini") == "simple-key"
    assert secrets.get_for_role("generation", "openai") is None


def test_secret_canary_never_reaches_config_database_or_backup(tmp_path: Path) -> None:
    canary = "STEERING_SECRET_CANARY_7e36f089"
    backend = MemorySecretBackend()
    secrets = KeyringSecretStore(backend, environ={})
    fingerprint = secrets.set("provider", canary)

    config_path = tmp_path / "config.json"
    ConfigStore(config_path).save(
        AppConfig(
            database_path=str(tmp_path / "steering.lbug"),
            generation_provider="provider",
            providers={
                "provider": ProviderConfig(
                    base_url="https://provider.example/v1",
                    generation_model="generation-model",
                    api_key_fingerprint=fingerprint,
                )
            },
        )
    )
    with DatabaseRuntime(tmp_path / "steering.lbug") as runtime:
        runtime.repository.backup(str(tmp_path / "backup"))

    assert canary.encode() not in config_path.read_bytes()
    for path in (tmp_path / "backup").iterdir():
        assert canary.encode() not in path.read_bytes()
