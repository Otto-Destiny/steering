from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

import steering.cli.main as cli
from steering.config.secrets import KeyringSecretStore, MemorySecretBackend
from steering.config.store import ConfigStore
from steering.domain.models import ProviderConfig


async def test_cli_provider_connection_uses_role_specific_provider_and_closes_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Client:
        def __init__(self, **_kwargs: object) -> None:
            events.append("client")

        async def close(self) -> None:
            events.append("close")

    class Generation:
        def __init__(self, **_kwargs: object) -> None:
            events.append("generation")

        async def test_connection(self) -> None:
            events.append("generation-test")

    class Embedding:
        def __init__(self, **_kwargs: object) -> None:
            events.append("embedding")

        async def test_connection(self) -> None:
            events.append("embedding-test")

    monkeypatch.setattr(cli, "OpenAICompatibleClient", Client)
    monkeypatch.setattr(cli, "OpenAICompatibleGenerationProvider", Generation)
    monkeypatch.setattr(cli, "OpenAICompatibleEmbeddingProvider", Embedding)
    generation = ProviderConfig(base_url="https://provider.test/v1", generation_model="generation-model")
    embedding = ProviderConfig(
        base_url="https://provider.test/v1",
        embedding_model="embedding-model",
        embedding_dimension=32,
    )
    await cli._test_provider_connection("generation", generation, "secret")
    await cli._test_provider_connection("embedding", embedding, "secret")
    assert events == [
        "client",
        "generation",
        "generation-test",
        "close",
        "client",
        "embedding",
        "embedding-test",
        "close",
    ]


def test_json_helpers_and_missing_fingerprint_are_safe(tmp_path: Path) -> None:
    provider = ProviderConfig(base_url="https://provider.test/v1")
    assert cli._json_safe(provider)["base_url"] == "https://provider.test/v1"
    assert cli._json_safe(tmp_path) == str(tmp_path)
    with pytest.raises(cli.CliError, match="did not return"):
        cli._masked_fingerprint("provider", None)
    assert cli._runtime_module().__name__ == "steering.runtime"


async def test_daemon_request_uses_local_config_and_handles_invalid_responses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    requests: list[tuple[str, str, object]] = []
    responses = [
        cli.httpx.Response(200, json={"status": "ok"}),
        cli.httpx.Response(200, content=b"not-json"),
    ]

    class Client:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["base_url"] == "http://127.0.0.1:8765"
            assert kwargs["trust_env"] is False
            assert kwargs["headers"] == {"Origin": "http://127.0.0.1:8765"}

        async def __aenter__(self) -> Client:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def request(self, method: str, path: str, *, json: object) -> cli.httpx.Response:
            requests.append((method, path, json))
            return responses.pop(0)

    monkeypatch.setattr(cli.httpx, "AsyncClient", Client)
    daemon_context = cli._CliContext(
        config_store=ConfigStore(tmp_path / "config.json"),
        secret_store=KeyringSecretStore(MemorySecretBackend(), environ={}),
        read_secret=lambda _prompt: "",
        stdout=io.StringIO(),
    )

    assert await cli._daemon_request(daemon_context, "GET", "/api/health") == {"status": "ok"}
    with pytest.raises(cli.CliError, match="invalid response"):
        await cli._daemon_request(daemon_context, "GET", "/api/health")
    assert requests == [
        ("GET", "/api/health", None),
        ("GET", "/api/health", None),
    ]


def invoke(
    argv: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path, module: object
) -> tuple[int, str, str]:
    monkeypatch.setattr(cli, "_runtime_module", lambda: module)
    stdout, stderr = io.StringIO(), io.StringIO()
    code = cli.main(
        argv,
        config_store=ConfigStore(tmp_path / "config.json"),
        secret_store=KeyringSecretStore(MemorySecretBackend(), environ={}),
        read_secret=lambda _prompt: "",
        stdout=stdout,
        stderr=stderr,
    )
    return code, stdout.getvalue(), stderr.getvalue()


def test_safe_cli_errors_for_files_empty_import_and_runtime_capabilities(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class BareRuntime:
        repository = object()

    class Module:
        def create_runtime(self, **_kwargs: object) -> BareRuntime:
            return BareRuntime()

    missing = tmp_path / "missing.txt"
    code, _, error = invoke(["add", "--batch", str(missing)], monkeypatch, tmp_path, Module())
    assert code == 2
    assert "does not exist" in error

    empty = tmp_path / "empty.txt"
    empty.write_text("# comments only\n", encoding="utf-8")
    code, _, error = invoke(["add", "--batch", str(empty)], monkeypatch, tmp_path, Module())
    assert code == 2
    assert "contains no sources" in error

    export = tmp_path / "telegram.json"
    export.write_text(json.dumps({"messages": []}), encoding="utf-8")
    code, _, error = invoke(["import-telegram", str(export)], monkeypatch, tmp_path, Module())
    assert code == 2
    assert "contains no supported URLs" in error

    async def daemon_request(_context: object, _method: str, path: str, **_kwargs: object) -> object:
        if path == "/api/health":
            return {"runtime": "ok"}
        if path == "/api/maintenance/reindex":
            return {"reindexed": True}
        raise AssertionError(path)

    monkeypatch.setattr(cli, "_daemon_request", daemon_request)
    code, output, _ = invoke(["doctor"], monkeypatch, tmp_path, Module())
    assert code == 0
    assert json.loads(output)["runtime"] == "ok"

    code, output, error = invoke(["reindex"], monkeypatch, tmp_path, Module())
    assert code == 0
    assert error == ""
    assert json.loads(output)["reindexed"] is True


def test_async_server_and_generic_failures_are_handled_without_leaking(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class AsyncServerModule:
        async def run_server(self, **_kwargs: object) -> None:
            return None

    code, _, error = invoke(["serve"], monkeypatch, tmp_path, AsyncServerModule())
    assert code == 0
    assert error == ""

    async def failing_daemon(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("upstream included secret-canary")

    monkeypatch.setattr(cli, "_daemon_request", failing_daemon)
    code, _, error = invoke(["doctor"], monkeypatch, tmp_path, object())
    assert code == 1
    assert "RuntimeError" in error
    assert "secret-canary" not in error
