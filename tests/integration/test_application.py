from __future__ import annotations

from pathlib import Path

from starlette.testclient import TestClient

from steering.app import create_app
from steering.config import AppConfig, ConfigStore, KeyringSecretStore, MemorySecretBackend
from steering.runtime import create_runtime


def test_single_daemon_serves_ui_api_static_and_mcp(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.json")
    config = AppConfig(database_path=str(tmp_path / "knowledge.lbug"))
    store.save(config)
    runtime = create_runtime(
        config_store=store,
        secret_store=KeyringSecretStore(backend=MemorySecretBackend(), environ={}),
    )
    app = create_app(runtime=runtime, close_runtime=True)

    with TestClient(app, base_url="http://localhost") as client:
        assert client.get("/").status_code == 200
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        static = client.get("/static/htmx.min.js")
        assert static.status_code == 200
        assert len(static.content) > 40_000
        assert client.get("/mcp/").status_code in {400, 406}

    assert runtime.database._closed is True


def test_web_mutations_reject_cross_origin_requests(tmp_path: Path) -> None:
    runtime = create_runtime(
        config=AppConfig(database_path=str(tmp_path / "knowledge.lbug")),
        config_store=ConfigStore(tmp_path / "config.json"),
        secret_store=KeyringSecretStore(backend=MemorySecretBackend(), environ={}),
    )
    app = create_app(runtime=runtime, close_runtime=True)

    with TestClient(app, base_url="http://localhost") as client:
        response = client.post(
            "/api/search",
            json={"query": "agent memory"},
            headers={"origin": "https://attacker.example"},
        )

    assert response.status_code == 403
