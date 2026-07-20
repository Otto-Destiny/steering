from __future__ import annotations

import importlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from steering.config.secrets import KeyringSecretStore, MemorySecretBackend
from steering.config.store import ConfigStore
from steering.domain.models import (
    Artifact,
    ArtifactRecord,
    ArtifactType,
    IngestionJob,
    JobStatus,
    ReviewStatus,
    SourceKind,
    TrustLane,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)
cli = importlib.import_module("steering.cli.main")


def test_cli_loads_the_real_server_entrypoint() -> None:
    server_module = cli._runtime_module()

    assert server_module.__name__ == "steering.app"
    assert callable(server_module.run_server)


def _record(source: str = "https://example.com/paper") -> ArtifactRecord:
    return ArtifactRecord(
        artifact=Artifact(
            id="art_1",
            canonical_url=source,
            source_kind=SourceKind.PAPER,
            artifact_type=ArtifactType.PAPER,
            title="Test paper",
            summary="A retained idea.",
            review_status=ReviewStatus.REVIEWED,
            trust_lane=TrustLane.PROMISING,
            content_hash="abc",
        )
    )


class FakeIngestion:
    def __init__(self) -> None:
        self.sources: list[str] = []

    async def add(self, source: str) -> ArtifactRecord:
        self.sources.append(source)
        return _record(source)

    async def add_batch(self, sources: list[str]) -> list[ArtifactRecord]:
        self.sources.extend(sources)
        return [_record(source) for source in sources]


class FakeRepository:
    def __init__(self) -> None:
        self.backups: list[str] = []

    def list_jobs(self, limit: int = 100) -> list[IngestionJob]:
        assert limit > 0
        return [
            IngestionJob(
                id="job_1",
                source="https://example.com/paper",
                status=JobStatus.COMPLETED,
                artifact_id="art_1",
                created_at=NOW,
                updated_at=NOW,
            )
        ]

    def backup(self, destination: str) -> str:
        self.backups.append(destination)
        return destination


class FakeRuntime:
    def __init__(self) -> None:
        self.ingestion = FakeIngestion()
        self.repository = FakeRepository()
        self.closed = 0
        self.reindexed = 0

    def doctor(self) -> dict[str, str]:
        return {"runtime": "ok", "database": "ok"}

    async def reindex(self) -> dict[str, int]:
        self.reindexed += 1
        return {"records": 1}

    def close(self) -> None:
        self.closed += 1


class FakeRuntimeModule:
    def __init__(self, runtime: FakeRuntime) -> None:
        self.runtime = runtime
        self.served = 0
        self.stores: list[ConfigStore] = []

    def create_runtime(self, *, config_store: ConfigStore) -> FakeRuntime:
        self.stores.append(config_store)
        return self.runtime

    def run_server(self, *, config_store: ConfigStore) -> None:
        self.stores.append(config_store)
        self.served += 1


def _invoke(
    argv: list[str],
    *,
    store: ConfigStore,
    secrets: KeyringSecretStore,
    read_secret: Any = lambda _prompt: "unused",
) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = cli.main(
        argv,
        config_store=store,
        secret_store=secrets,
        read_secret=read_secret,
        stdout=stdout,
        stderr=stderr,
    )
    return code, stdout.getvalue(), stderr.getvalue()


@pytest.fixture
def stores(tmp_path: Path) -> tuple[ConfigStore, KeyringSecretStore]:
    return (
        ConfigStore(tmp_path / "config.json"),
        KeyringSecretStore(MemorySecretBackend(), environ={}),
    )


def test_configure_provider_tests_before_saving_and_never_prints_secret(
    monkeypatch: pytest.MonkeyPatch,
    stores: tuple[ConfigStore, KeyringSecretStore],
) -> None:
    store, secrets = stores
    secret = "secret-canary-value"
    tested: list[str] = []

    async def test_connection(provider_id: str, role: str, provider: Any, supplied_secret: str) -> None:
        assert provider_id == "provider-1"
        assert not store.path.exists()
        assert secrets.get_for_role(role, "provider-1") is None
        assert str(provider.base_url) == "https://api.example.com/v1"
        tested.append(supplied_secret)

    monkeypatch.setattr(cli, "_test_provider_connection", test_connection)
    code, output, error = _invoke(
        [
            "configure-provider",
            "--role",
            "generation",
            "--provider-id",
            "provider-1",
            "--base-url",
            "https://api.example.com/v1",
            "--model",
            "model-1",
        ],
        store=store,
        secrets=secrets,
        read_secret=lambda _prompt: secret,
    )
    assert code == 0
    assert error == ""
    assert tested == [secret]
    assert secret not in output
    assert "****" in output
    assert secret not in store.path.read_text(encoding="utf-8")
    assert secrets.get_for_role("generation", "provider-1") == secret
    config = store.load(apply_env=False)
    assert config.generation_provider == "provider-1"
    assert config.providers["provider-1"].generation_model == "model-1"


def test_failed_provider_test_saves_nothing_and_redacts_exception(
    monkeypatch: pytest.MonkeyPatch,
    stores: tuple[ConfigStore, KeyringSecretStore],
) -> None:
    store, secrets = stores
    secret = "secret-canary-value"

    async def fail(*_args: Any) -> None:
        raise RuntimeError(f"upstream echoed {secret}")

    monkeypatch.setattr(cli, "_test_provider_connection", fail)
    code, output, error = _invoke(
        [
            "configure-provider",
            "--role",
            "embedding",
            "--provider-id",
            "provider-1",
            "--base-url",
            "https://api.example.com/v1",
            "--model",
            "embed-1",
            "--dimension",
            "64",
        ],
        store=store,
        secrets=secrets,
        read_secret=lambda _prompt: secret,
    )
    assert code == 2
    assert output == ""
    assert secret not in error
    assert not store.path.exists()
    assert secrets.get_for_role("embedding", "provider-1") is None


def test_parser_has_no_secret_argument() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            [
                "configure-provider",
                "--role",
                "generation",
                "--base-url",
                "https://example.com",
                "--model",
                "model",
                "--api-key",
                "forbidden",
            ]
        )


def test_simple_configure_applies_reviewed_preset_and_tests_both_roles(
    monkeypatch: pytest.MonkeyPatch,
    stores: tuple[ConfigStore, KeyringSecretStore],
) -> None:
    store, secrets = stores
    calls: list[tuple[str, str]] = []

    async def test_connection(provider_id: str, role: str, provider: Any, supplied_secret: str) -> None:
        assert supplied_secret == "simple-secret"
        assert provider.embedding_dimension == 768
        calls.append((provider_id, role))

    monkeypatch.setattr(cli, "_test_provider_connection", test_connection)
    code, output, error = _invoke(
        ["configure", "--provider", "gemini"],
        store=store,
        secrets=secrets,
        read_secret=lambda _prompt: "simple-secret",
    )
    assert code == 0
    assert error == ""
    assert "simple-secret" not in output
    assert calls == [("gemini", "generation"), ("gemini", "embedding")]
    config = store.load(apply_env=False)
    assert config.generation_provider == config.embedding_provider == "gemini"
    assert config.providers["gemini"].generation_model == "gemini-3.5-flash"
    assert config.providers["gemini"].embedding_model == "gemini-embedding-2"
    assert secrets.get_for_role("generation", "gemini") == "simple-secret"
    assert secrets.get_for_role("embedding", "gemini") == "simple-secret"


def test_configure_provider_supports_local_endpoint_without_authentication(
    monkeypatch: pytest.MonkeyPatch,
    stores: tuple[ConfigStore, KeyringSecretStore],
) -> None:
    store, secrets = stores
    tested: list[str] = []

    async def test_connection(provider_id: str, role: str, provider: Any, supplied_secret: str) -> None:
        del provider_id, role, provider
        tested.append(supplied_secret)

    monkeypatch.setattr(cli, "_test_provider_connection", test_connection)
    code, output, error = _invoke(
        [
            "configure-provider",
            "--role",
            "generation",
            "--provider-id",
            "local",
            "--base-url",
            "http://127.0.0.1:11434/v1",
            "--model",
            "local-model",
        ],
        store=store,
        secrets=secrets,
        read_secret=lambda _prompt: "",
    )

    assert code == 0
    assert error == ""
    assert tested == [""]
    assert "no API key" in output
    assert secrets.get_for_role("generation", "local") is None


def test_configure_shared_provider_preserves_both_role_models(
    monkeypatch: pytest.MonkeyPatch,
    stores: tuple[ConfigStore, KeyringSecretStore],
) -> None:
    store, secrets = stores

    async def test_connection(provider_id: str, role: str, provider: Any, supplied_secret: str) -> None:
        del provider_id, role, provider, supplied_secret

    monkeypatch.setattr(cli, "_test_provider_connection", test_connection)
    common = [
        "--provider-id",
        "shared",
        "--base-url",
        "https://api.example.com/v1",
    ]
    first, _, _ = _invoke(
        ["configure-provider", "--role", "generation", *common, "--model", "gen-v1"],
        store=store,
        secrets=secrets,
        read_secret=lambda _prompt: "generation-key",
    )
    second, _, _ = _invoke(
        [
            "configure-provider",
            "--role",
            "embedding",
            *common,
            "--model",
            "embed-v1",
            "--dimension",
            "1536",
        ],
        store=store,
        secrets=secrets,
        read_secret=lambda _prompt: "embedding-key",
    )

    assert (first, second) == (0, 0)
    config = store.load(apply_env=False)
    provider = config.providers["shared"]
    assert provider.generation_model == "gen-v1"
    assert provider.embedding_model == "embed-v1"
    assert provider.embedding_dimension == 1536
    assert config.generation_provider == config.embedding_provider == "shared"
    assert secrets.get_for_role("generation", "shared") == "generation-key"
    assert secrets.get_for_role("embedding", "shared") == "embedding-key"


def test_add_batch_telegram_jobs_doctor_backup_and_reindex_use_daemon(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stores: tuple[ConfigStore, KeyringSecretStore],
) -> None:
    store, secrets = stores
    requests: list[tuple[str, str, dict[str, object] | None]] = []

    async def daemon_request(
        _context: object,
        method: str,
        path: str,
        *,
        payload: dict[str, object] | None = None,
        timeout: float = 120.0,
    ) -> object:
        del timeout
        requests.append((method, path, payload))
        if path == "/api/ingestion":
            sources = payload["sources"] if payload else []
            return {
                "records": [
                    {"artifact": _record(str(source)).artifact.model_dump(mode="json")} for source in sources
                ]
            }
        if path.startswith("/api/jobs"):
            return {"jobs": [{"id": "job_1"}]}
        if path == "/api/health":
            return {"status": "ok"}
        if path == "/api/maintenance/backup":
            return {"backup": str(payload["destination"])}
        if path == "/api/maintenance/reindex":
            return {"reindexed": True, "result": 1}
        raise AssertionError(path)

    monkeypatch.setattr(cli, "_daemon_request", daemon_request)

    code, output, _ = _invoke(["add", "https://example.com/one"], store=store, secrets=secrets)
    assert code == 0
    assert json.loads(output)["added"] == 1

    batch = tmp_path / "sources.txt"
    batch.write_text("# saved\nhttps://example.com/two\nhttps://example.com/two\n", encoding="utf-8")
    code, output, _ = _invoke(["add", "--batch", str(batch)], store=store, secrets=secrets)
    assert code == 0
    assert json.loads(output)["added"] == 1

    export = tmp_path / "result.json"
    export.write_text(json.dumps({"messages": [{"text": "https://example.com/three"}]}), encoding="utf-8")
    code, output, _ = _invoke(["import-telegram", str(export)], store=store, secrets=secrets)
    assert code == 0
    assert json.loads(output)["discovered_sources"] == 1

    code, output, _ = _invoke(["jobs", "--limit", "5"], store=store, secrets=secrets)
    assert code == 0
    assert json.loads(output)[0]["id"] == "job_1"

    code, output, _ = _invoke(["doctor"], store=store, secrets=secrets)
    assert code == 0
    assert json.loads(output)["status"] == "ok"

    destination = tmp_path / "backup"
    code, output, _ = _invoke(["backup", str(destination)], store=store, secrets=secrets)
    assert code == 0
    assert json.loads(output)["backup"] == str(destination)

    code, output, _ = _invoke(["reindex"], store=store, secrets=secrets)
    assert code == 0
    assert json.loads(output)["reindexed"] is True
    assert all(path.startswith("/api/") for _, path, _ in requests)


def test_restore_serve_and_agent_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stores: tuple[ConfigStore, KeyringSecretStore],
) -> None:
    store, secrets = stores
    runtime = FakeRuntime()
    module = FakeRuntimeModule(runtime)
    monkeypatch.setattr(cli, "_runtime_module", lambda: module)

    async def daemon_unavailable(*_args: object, **_kwargs: object) -> object:
        raise cli.DaemonUnavailableError("not running")

    monkeypatch.setattr(cli, "_daemon_request", daemon_unavailable)

    calls: list[tuple[Path, Path]] = []

    def fake_restore(source: Path, target: Path) -> Path:
        calls.append((source, target))
        return target

    monkeypatch.setattr("steering.database.backup.restore_backup", fake_restore)
    source = tmp_path / "backup"
    target = tmp_path / "restored.lbug"
    code, output, _ = _invoke(["restore", str(source), "--target", str(target)], store=store, secrets=secrets)
    assert code == 0
    assert calls == [(source, target)]
    assert json.loads(output)["restored_database"] == str(target)

    code, _, _ = _invoke(["serve"], store=store, secrets=secrets)
    assert code == 0
    assert module.served == 1

    code, codex, _ = _invoke(["agent-config", "codex"], store=store, secrets=secrets)
    assert code == 0
    assert "[mcp_servers.steering]" in codex
    assert "http://127.0.0.1:8765/mcp" in codex
    assert "after previous approaches fail" in codex
    assert "before declaring that reasonable options are exhausted" in codex

    code, claude, _ = _invoke(["agent-config", "claude"], store=store, secrets=secrets)
    assert code == 0
    assert "claude mcp add --transport http steering" in claude
    assert "start of architecture planning" in claude
