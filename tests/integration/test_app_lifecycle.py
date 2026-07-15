from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import steering.app as app_module
from steering.config import AppConfig, ConfigStore, KeyringSecretStore, MemorySecretBackend
from steering.domain.models import ProviderConfig
from steering.runtime import create_runtime


@pytest.mark.asyncio
async def test_provider_connection_probe_routes_by_role_and_always_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeClient:
        def __init__(self, **values: Any) -> None:
            events.append(f"client:{values['base_url']}:{values['api_key'] is not None}")

        async def close(self) -> None:
            events.append("closed")

    class FakeGeneration:
        def __init__(self, **values: Any) -> None:
            events.append(f"generation:{values['model_id']}")

        async def test_connection(self) -> None:
            events.append("generation-tested")

    class FakeEmbedding:
        def __init__(self, **values: Any) -> None:
            events.append(f"embedding:{values['model_id']}:{values['dimension']}")

        async def test_connection(self) -> None:
            events.append("embedding-tested")

    monkeypatch.setattr(app_module, "OpenAICompatibleClient", FakeClient)
    monkeypatch.setattr(app_module, "OpenAICompatibleGenerationProvider", FakeGeneration)
    monkeypatch.setattr(app_module, "OpenAICompatibleEmbeddingProvider", FakeEmbedding)
    config = ProviderConfig(
        base_url="https://api.example.com/v1",
        generation_model="gen",
        embedding_model="embed",
        embedding_dimension=384,
    )

    await app_module._test_provider("example", "generation", config, "secret")
    await app_module._test_provider("example", "embedding", config, None)
    assert events == [
        "client:https://api.example.com/v1:True",
        "generation:gen",
        "generation-tested",
        "closed",
        "client:https://api.example.com/v1:False",
        "embedding:embed:384",
        "embedding-tested",
        "closed",
    ]

    incomplete = config.model_copy(update={"generation_model": None})
    with pytest.raises(ValueError, match="configured generation model"):
        await app_module._test_provider("example", "generation", incomplete, None)
    assert events[-1] == "closed"


def test_create_app_can_own_factory_runtime_without_network(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ConfigStore(tmp_path / "config.json")
    runtime = create_runtime(
        config=AppConfig(database_path=str(tmp_path / "factory.lbug")),
        config_store=store,
        secret_store=KeyringSecretStore(MemorySecretBackend(), environ={}),
    )
    monkeypatch.setattr(app_module, "create_runtime", lambda **_: runtime)
    app = app_module.create_app(config_store=store, close_runtime=False)
    assert app.state.runtime is runtime
    assert runtime.database._closed is False
    runtime.database.close()


def test_loopback_host_validation() -> None:
    assert app_module._is_loopback_host("localhost") is True
    assert app_module._is_loopback_host("127.0.0.1") is True
    assert app_module._is_loopback_host("::1") is True
    assert app_module._is_loopback_host("0.0.0.0") is False  # noqa: S104 - rejection test
    assert app_module._is_loopback_host("not-an-ip") is False


@pytest.mark.asyncio
async def test_run_server_rejects_non_loopback_and_closes_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRuntime:
        config = SimpleNamespace(
            host="0.0.0.0",  # noqa: S104 - deliberate unsafe-bind rejection test
            port=8765,
            log_level="INFO",
        )
        closed = False

        async def aclose(self) -> None:
            self.closed = True

    runtime = FakeRuntime()
    monkeypatch.setattr(app_module, "create_runtime", lambda **_: runtime)
    with pytest.raises(ValueError, match="loopback"):
        await app_module.run_server(config_store=SimpleNamespace())  # type: ignore[arg-type]
    assert runtime.closed is True


@pytest.mark.asyncio
async def test_run_server_builds_local_uvicorn_without_starting_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[Any] = []

    class FakeRuntime:
        config = SimpleNamespace(host="127.0.0.1", port=9123, log_level="DEBUG")

    class FakeConfig:
        def __init__(self, app: Any, **values: Any) -> None:
            events.append(("config", app, values))

    class FakeServer:
        def __init__(self, config: FakeConfig) -> None:
            events.append(("server", config))

        async def serve(self) -> None:
            events.append("served")

    runtime = FakeRuntime()
    sentinel_app = object()
    monkeypatch.setattr(app_module, "create_runtime", lambda **_: runtime)
    monkeypatch.setattr(app_module, "create_app", lambda **_: sentinel_app)
    monkeypatch.setattr(app_module.uvicorn, "Config", FakeConfig)
    monkeypatch.setattr(app_module.uvicorn, "Server", FakeServer)

    await app_module.run_server(config_store=SimpleNamespace())  # type: ignore[arg-type]
    assert events[0] == (
        "config",
        sentinel_app,
        {
            "host": "127.0.0.1",
            "port": 9123,
            "log_level": "debug",
            "server_header": False,
        },
    )
    assert events[-1] == "served"


def test_uvicorn_application_factory_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    monkeypatch.setattr(app_module, "create_app", lambda: sentinel)
    assert app_module.application() is sentinel
