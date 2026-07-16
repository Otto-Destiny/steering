from __future__ import annotations

from pathlib import Path

import pytest
from tests.support.providers import DeterministicBlake2EmbeddingProvider

from steering.config import AppConfig, ConfigStore, KeyringSecretStore, MemorySecretBackend
from steering.database import DatabaseRuntime
from steering.domain.models import ProviderConfig
from steering.providers import (
    GeminiEmbeddingProvider,
    GeminiGenerationProvider,
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleGenerationProvider,
    UnconfiguredEmbeddingProvider,
    UnconfiguredGenerationProvider,
)
from steering.runtime import (
    _EVALUATION_RUNTIMES,
    _browser_extra_available,
    create_runtime,
    evaluation_factory,
)


@pytest.mark.asyncio
async def test_runtime_owns_one_database_and_starts_without_api_keys(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.json")
    config = AppConfig(database_path=str(tmp_path / "knowledge.lbug"))
    store.save(config)
    secrets = KeyringSecretStore(backend=MemorySecretBackend(), environ={})

    runtime = create_runtime(config_store=store, secret_store=secrets)
    report = runtime.doctor()

    assert report["status"] == "setup_required"
    assert report["generation_key_fingerprint"] is None
    assert report["embedding_provider"] == "unconfigured"
    assert await runtime.reindex() == 0

    await runtime.close()
    assert runtime.database._closed is True


@pytest.mark.asyncio
async def test_runtime_context_manager_closes_resources(tmp_path: Path) -> None:
    runtime = create_runtime(
        config=AppConfig(database_path=str(tmp_path / "knowledge.lbug")),
        config_store=ConfigStore(tmp_path / "config.json"),
        secret_store=KeyringSecretStore(backend=MemorySecretBackend(), environ={}),
    )

    async with runtime as active:
        assert active.repository.list_records() == []

    assert runtime.database._closed is True


@pytest.mark.asyncio
async def test_runtime_configures_distinct_provider_roles_and_doctor_fingerprints(
    tmp_path: Path,
) -> None:
    provider_id = "provider"
    config = AppConfig(
        database_path=str(tmp_path / "configured.lbug"),
        generation_provider=provider_id,
        embedding_provider=provider_id,
        providers={
            provider_id: ProviderConfig(
                base_url="https://provider.test/v1",
                generation_model="generation-model",
                embedding_model="embedding-model",
                embedding_dimension=32,
            )
        },
    )
    secrets = KeyringSecretStore(backend=MemorySecretBackend(), environ={})
    secrets.set_for_role("generation", provider_id, "generation-secret")
    secrets.set_for_role("embedding", provider_id, "embedding-secret")
    runtime = create_runtime(
        config=config,
        config_store=ConfigStore(tmp_path / "config.json"),
        secret_store=secrets,
    )
    assert isinstance(runtime.generation, OpenAICompatibleGenerationProvider)
    assert isinstance(runtime.embedding, OpenAICompatibleEmbeddingProvider)
    assert runtime.embedding.dimension == 32
    assert len(runtime._provider_clients) == 2
    assert runtime._provider_clients[0]._client.headers["authorization"] == ("Bearer generation-secret")
    assert runtime._provider_clients[1]._client.headers["authorization"] == ("Bearer embedding-secret")
    report = runtime.doctor()
    assert report["status"] == "ready"
    assert report["generation_provider"] == provider_id
    assert str(report["generation_key_fingerprint"]).startswith("****")
    assert str(report["embedding_key_fingerprint"]).startswith("****")
    assert report["artifact_count"] == 0
    assert report["unresolved_issue_count"] == 0

    clients = list(runtime._provider_clients)
    await runtime.close()
    await runtime.close()
    assert all(client._client.is_closed for client in clients)


@pytest.mark.parametrize(
    "config, expected",
    [
        (AppConfig(generation_provider="missing"), "needs a generation model"),
        (
            AppConfig(
                generation_provider="provider",
                providers={
                    "provider": ProviderConfig(
                        base_url="https://provider.test/v1",
                        embedding_model="embedding-model",
                    )
                },
            ),
            "needs a generation model",
        ),
        (
            AppConfig(
                embedding_provider="provider",
                providers={
                    "provider": ProviderConfig(
                        base_url="https://provider.test/v1",
                        generation_model="generation-model",
                    )
                },
            ),
            "needs an embedding model",
        ),
    ],
)
def test_runtime_provider_configuration_failures_close_database(
    tmp_path: Path, config: AppConfig, expected: str
) -> None:
    config.database_path = str(tmp_path / f"{len(list(tmp_path.iterdir()))}.lbug")
    with pytest.raises(ValueError, match=expected):
        create_runtime(
            config=config,
            config_store=ConfigStore(tmp_path / "config.json"),
            secret_store=KeyringSecretStore(backend=MemorySecretBackend(), environ={}),
        )
    reopened = DatabaseRuntime(config.database_file)
    reopened.close()


class LegacySecretStore:
    def get(self, provider_id: str) -> str | None:
        return f"secret-for-{provider_id}"

    def set(self, provider_id: str, secret: str) -> str:
        return f"fingerprint-{provider_id}-{secret}"

    def delete(self, provider_id: str) -> bool:
        return bool(provider_id)

    def fingerprint(self, provider_id: str) -> str | None:
        return f"sha256:12345678{provider_id}"


class MinimalSecretStore:
    def get(self, _provider_id: str) -> None:
        return None


@pytest.mark.asyncio
async def test_runtime_legacy_secret_fallbacks_and_missing_provider_client(tmp_path: Path) -> None:
    runtime = create_runtime(
        config=AppConfig(database_path=str(tmp_path / "legacy.lbug")),
        config_store=ConfigStore(tmp_path / "config.json"),
        secret_store=LegacySecretStore(),  # type: ignore[arg-type]
    )
    assert isinstance(runtime.generation, UnconfiguredGenerationProvider)
    assert isinstance(runtime.embedding, UnconfiguredEmbeddingProvider)
    assert runtime._role_secret("generation", "provider") == "secret-for-provider"
    runtime.config.providers["provider"] = ProviderConfig(
        base_url="https://provider.test/v1", generation_model="model"
    )
    client = runtime._provider_client("generation", "provider")
    assert client._client.headers["authorization"] == "Bearer secret-for-provider"
    with pytest.raises(ValueError, match="has no configuration"):
        runtime._provider_client("generation", "missing")
    assert runtime._masked_fingerprint("generation", None) is None
    assert runtime._masked_fingerprint("generation", "provider") == "****provider"
    runtime.secret_store = MinimalSecretStore()  # type: ignore[assignment]
    assert runtime._masked_fingerprint("generation", "provider") is None
    await runtime.close()


def test_runtime_sync_context_manager_and_browser_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = create_runtime(
        config=AppConfig(database_path=str(tmp_path / "sync.lbug")),
        config_store=ConfigStore(tmp_path / "config.json"),
        secret_store=KeyringSecretStore(backend=MemorySecretBackend(), environ={}),
    )
    with runtime as active:
        assert active is runtime
    assert runtime.database._closed is True

    monkeypatch.setitem(__import__("sys").modules, "playwright", object())
    assert _browser_extra_available() is True


def test_evaluation_factory_returns_disposable_repository_and_retriever() -> None:
    before = len(_EVALUATION_RUNTIMES)
    repository, retriever = evaluation_factory(embedding_provider=DeterministicBlake2EmbeddingProvider())
    assert repository.list_records() == []
    assert retriever.repository is repository
    assert len(_EVALUATION_RUNTIMES) == before + 1
    _EVALUATION_RUNTIMES.pop().close()


def test_create_runtime_uses_default_collaborators(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = ConfigStore(tmp_path / "default-config.json")
    store.save(AppConfig(database_path=str(tmp_path / "default.lbug")))
    secrets = KeyringSecretStore(backend=MemorySecretBackend(), environ={})
    monkeypatch.setattr("steering.runtime.ConfigStore", lambda: store)
    monkeypatch.setattr("steering.runtime.KeyringSecretStore", lambda **_kwargs: secrets)
    runtime = create_runtime()
    assert runtime.config_store is store
    assert runtime.secret_store is secrets
    runtime.database.close()


@pytest.mark.asyncio
async def test_runtime_wires_simple_env_file_to_real_gemini_providers(tmp_path: Path) -> None:
    env_file = tmp_path / ".env.local"
    env_file.write_text(
        "STEERING_PROVIDER=gemini\nSTEERING_API_KEY=test-only-key\n",
        encoding="utf-8",
    )
    store = ConfigStore(tmp_path / "config.json", env_file=env_file)
    config = AppConfig(database_path=str(tmp_path / "gemini.lbug"))
    store.save(config)
    runtime = create_runtime(config_store=store)
    assert isinstance(runtime.generation, GeminiGenerationProvider)
    assert isinstance(runtime.embedding, GeminiEmbeddingProvider)
    assert runtime.embedding.dimension == 768
    assert runtime.embedding.document_task_mode == "retrieval_document"
    assert runtime.embedding.query_task_mode == "retrieval_query"
    assert runtime.doctor()["status"] == "ready"
    await runtime.aclose()
