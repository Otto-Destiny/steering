from __future__ import annotations

from pathlib import Path

import pytest

from steering.config.models import AppConfig
from steering.config.secrets import KeyringSecretStore, MemorySecretBackend
from steering.config.store import ConfigStore
from steering.domain.models import ProviderConfig
from steering.web.providers import LocalProviderSettings


@pytest.mark.asyncio
async def test_local_provider_settings_store_only_masked_fingerprint(tmp_path: Path) -> None:
    seen: list[tuple[str, str, str | None]] = []

    async def tester(
        provider_id: str,
        role: str,
        config: ProviderConfig,
        api_key: str | None,
    ) -> None:
        assert str(config.base_url) == "https://api.example.com/v1"
        seen.append((provider_id, role, api_key))

    config_path = tmp_path / "config.json"
    config_store = ConfigStore(config_path)
    secret_store = KeyringSecretStore(MemorySecretBackend(), environ={})
    settings = LocalProviderSettings(
        config_store=config_store,
        secret_store=secret_store,
        connection_tester=tester,
    )
    secret = "sk-stored-in-keyring-only"
    view = await settings.save_provider(
        provider_id="example",
        role="both",
        base_url="https://api.example.com/v1",
        generation_model="gen",
        embedding_model="embed",
        embedding_dimension=1024,
        generation_api_key=secret,
        embedding_api_key="embed-secret",
    )

    assert view.generation_has_api_key is True
    assert view.generation_key_fingerprint is not None
    assert secret not in view.model_dump_json()
    assert secret not in config_path.read_text(encoding="utf-8")
    assert secret_store.get_for_role("generation", "example") == secret

    await settings.test_provider("example")
    assert seen == [
        ("example", "generation", secret),
        ("example", "embedding", "embed-secret"),
        ("example", "generation", secret),
        ("example", "embedding", "embed-secret"),
    ]
    assert settings.delete_key("example", "generation") is True
    updated = settings.list_providers()[0]
    assert updated.generation_has_api_key is False
    assert updated.embedding_has_api_key is True


@pytest.mark.asyncio
async def test_provider_connection_failure_writes_nothing(tmp_path: Path) -> None:
    async def failing_tester(
        provider_id: str,
        role: str,
        config: ProviderConfig,
        api_key: str | None,
    ) -> None:
        raise RuntimeError("connection refused")

    config_store = ConfigStore(tmp_path / "config.json")
    secret_store = KeyringSecretStore(MemorySecretBackend(), environ={})
    settings = LocalProviderSettings(
        config_store=config_store,
        secret_store=secret_store,
        connection_tester=failing_tester,
    )

    with pytest.raises(RuntimeError, match="connection refused"):
        await settings.save_provider(
            provider_id="broken",
            role="generation",
            base_url="https://api.example.com/v1",
            generation_model="gen",
            embedding_model=None,
            embedding_dimension=256,
            generation_api_key="must-not-be-saved",
            embedding_api_key=None,
        )

    assert "broken" not in config_store.load(apply_env=False).providers
    assert secret_store.get_for_role("generation", "broken") is None


@pytest.mark.asyncio
async def test_provider_settings_reject_unknown_provider_role_and_inactive_test(tmp_path: Path) -> None:
    async def tester(
        provider_id: str,
        role: str,
        config: ProviderConfig,
        api_key: str | None,
    ) -> None:
        raise AssertionError("inactive providers must not be tested")

    config_store = ConfigStore(tmp_path / "config.json")
    provider = ProviderConfig(base_url="https://api.example.com/v1", generation_model="gen")
    config_store.save(AppConfig(providers={"inactive": provider}))
    settings = LocalProviderSettings(
        config_store=config_store,
        secret_store=KeyringSecretStore(MemorySecretBackend(), environ={}),
        connection_tester=tester,
    )

    with pytest.raises(KeyError):
        await settings.test_provider("missing")
    with pytest.raises(KeyError):
        settings.delete_key("missing", "generation")
    with pytest.raises(ValueError, match="not active"):
        await settings.test_provider("inactive")
    with pytest.raises(ValueError, match="role must be"):
        settings.delete_key("inactive", "invalid")


@pytest.mark.asyncio
async def test_shared_provider_role_updates_preserve_the_other_model(tmp_path: Path) -> None:
    async def tester(
        provider_id: str,
        role: str,
        config: ProviderConfig,
        api_key: str | None,
    ) -> None:
        del provider_id, role, config, api_key

    store = ConfigStore(tmp_path / "config.json")
    settings = LocalProviderSettings(
        config_store=store,
        secret_store=KeyringSecretStore(MemorySecretBackend(), environ={}),
        connection_tester=tester,
    )
    await settings.save_provider(
        provider_id="shared",
        role="generation",
        base_url="https://api.example.com/v1",
        generation_model="gen-v1",
        embedding_model=None,
        embedding_dimension=256,
        generation_api_key="generation-key",
        embedding_api_key=None,
    )
    await settings.save_provider(
        provider_id="shared",
        role="embedding",
        base_url="https://api.example.com/v1",
        generation_model=None,
        embedding_model="embed-v1",
        embedding_dimension=1536,
        generation_api_key=None,
        embedding_api_key="embedding-key",
    )

    provider = store.load(apply_env=False).providers["shared"]
    assert provider.generation_model == "gen-v1"
    assert provider.embedding_model == "embed-v1"
    assert provider.embedding_dimension == 1536
