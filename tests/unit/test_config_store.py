from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from steering.config import AppConfig, ConfigStore, apply_environment_overrides, load_local_environment
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


def test_simple_gemini_environment_applies_reviewed_preset() -> None:
    resolved = apply_environment_overrides(
        AppConfig(),
        {
            "STEERING_PROVIDER": "gemini",
            "STEERING_API_KEY": "not-serialized",
        },
    )
    assert resolved.generation_provider == resolved.embedding_provider == "gemini"
    provider = resolved.providers["gemini"]
    assert provider.generation_model == "gemini-3.5-flash"
    assert provider.embedding_model == "gemini-embedding-2"
    assert provider.embedding_dimension == 768
    assert "not-serialized" not in resolved.model_dump_json()


def test_local_environment_is_allowlisted_and_process_environment_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = tmp_path / ".env.local"
    local.write_text(
        "# local settings\n"
        "STEERING_PROVIDER=gemini\n"
        "STEERING_API_KEY='local-secret'\n"
        "UNRELATED_SECRET=ignored\n",
        encoding="utf-8",
    )
    assert load_local_environment(local) == {
        "STEERING_PROVIDER": "gemini",
        "STEERING_API_KEY": "local-secret",
    }
    monkeypatch.setattr(
        "steering.config.store.os.environ",
        {"STEERING_PROVIDER": "openai", "STEERING_API_KEY": "process-secret"},
    )
    store = ConfigStore(tmp_path / "config.json", env_file=local)
    environment = store.environment()
    assert environment["STEERING_PROVIDER"] == "openai"
    assert environment["STEERING_API_KEY"] == "process-secret"
    resolved = store.load()
    assert resolved.generation_provider == resolved.embedding_provider == "openai"
    assert resolved.providers["openai"].embedding_dimension == 768
