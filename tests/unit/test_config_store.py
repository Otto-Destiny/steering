from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from steering.config import AppConfig, ConfigStore, apply_environment_overrides
from steering.domain.models import ProviderConfig


def test_config_round_trip_contains_only_non_secret_settings(tmp_path: Path) -> None:
    path = tmp_path / "config" / "config.json"
    config = AppConfig(
        database_path=str(tmp_path / "steering.lbug"),
        generation_provider="openai",
        embedding_provider="local",
        providers={
            "openai": ProviderConfig(
                base_url="https://api.openai.com/v1",
                generation_model="example-generation-model",
                api_key_fingerprint="sha256:1234567890abcdef",
            ),
            "local": ProviderConfig(
                base_url="http://127.0.0.1:11434/v1",
                embedding_model="example-embedding-model",
                embedding_dimension=384,
            ),
        },
    )
    store = ConfigStore(path)
    assert store.save(config) == path
    assert store.load(apply_env=False) == config
    serialized = path.read_text(encoding="utf-8").lower()
    assert 'api_key"' not in serialized
    assert "password" not in serialized
    assert "authorization" not in serialized


def test_environment_overrides_scalar_and_provider_settings(tmp_path: Path) -> None:
    base = AppConfig()
    environment = {
        "STEERING_DATABASE_PATH": str(tmp_path / "env.lbug"),
        "STEERING_PORT": "9123",
        "STEERING_GENERATION_PROVIDER": "compatible",
        "STEERING_GENERATION_BASE_URL": "https://provider.example/v1",
        "STEERING_GENERATION_MODEL": "generation-model",
        "STEERING_EMBEDDING_PROVIDER": "compatible",
        "STEERING_EMBEDDING_MODEL": "embedding-model",
        "STEERING_EMBEDDING_DIMENSION": "768",
    }
    resolved = apply_environment_overrides(base, environment)
    assert resolved.database_path == environment["STEERING_DATABASE_PATH"]
    assert resolved.port == 9123
    provider = resolved.providers["compatible"]
    assert str(provider.base_url) == "https://provider.example/v1"
    assert provider.generation_model == "generation-model"
    assert provider.embedding_model == "embedding-model"
    assert provider.embedding_dimension == 768


def test_provider_base_url_rejects_userinfo_without_echoing_it() -> None:
    secret_url = "https://URL_SECRET_CANARY@example.com/v1"

    with pytest.raises(ValidationError) as raised:
        ProviderConfig(base_url=secret_url)

    rendered = str(raised.value)
    assert "provider base URL must not contain user information" in rendered
    assert "URL_SECRET_CANARY" not in rendered
