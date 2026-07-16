"""Dependency-light command-line interface for STEERING."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import importlib
import inspect
import json
import shutil
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO, cast

import httpx
from pydantic import SecretStr

from steering.config.models import AppConfig
from steering.config.secrets import (
    KeyringSecretStore,
    SecretDescriptor,
    SecretStore,
)
from steering.config.store import ConfigStore
from steering.domain.models import ProviderConfig
from steering.providers.fastembed_local import (
    LOCAL_PROVIDER_ID,
    install_local_model,
    local_model_cache,
    local_model_status,
)
from steering.providers.gemini import (
    GeminiClient,
    GeminiEmbeddingProvider,
    GeminiGenerationProvider,
)
from steering.providers.openai_compatible import (
    OpenAICompatibleClient,
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleGenerationProvider,
)
from steering.providers.presets import PROVIDER_PRESETS, provider_preset


class CliError(RuntimeError):
    """An error whose message is safe to display to the user."""


class DaemonUnavailableError(CliError):
    """Raised when a command requiring the localhost daemon cannot connect."""


@dataclass(slots=True)
class _CliContext:
    config_store: ConfigStore
    secret_store: SecretStore
    read_secret: Callable[[str], str]
    stdout: TextIO


def _json_safe(value: object) -> object:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    return value


def _print_json(value: object, output: TextIO) -> None:
    print(json.dumps(_json_safe(value), indent=2, ensure_ascii=False, default=_json_safe), file=output)


async def _test_provider_connection(
    provider_id: str,
    role: str,
    config: ProviderConfig,
    secret: str,
) -> None:
    if provider_id == "gemini":
        if not secret:
            raise ValueError("Gemini requires an API key")
        gemini_client = GeminiClient(
            base_url=str(config.base_url),
            api_key=SecretStr(secret),
        )
        try:
            if role == "generation":
                await GeminiGenerationProvider(
                    client=gemini_client,
                    model_id=config.generation_model or "",
                ).test_connection()
            else:
                await GeminiEmbeddingProvider(
                    client=gemini_client,
                    model_id=config.embedding_model or "",
                    dimension=config.embedding_dimension,
                ).test_connection()
        finally:
            await gemini_client.close()
        return

    client = OpenAICompatibleClient(base_url=str(config.base_url), api_key=SecretStr(secret))
    try:
        if role == "generation":
            generation_provider = OpenAICompatibleGenerationProvider(
                client=client,
                model_id=config.generation_model or "",
            )
            await generation_provider.test_connection()
        else:
            embedding_provider = OpenAICompatibleEmbeddingProvider(
                client=client,
                model_id=config.embedding_model or "",
                dimension=config.embedding_dimension,
                provider_id=provider_id,
            )
            await embedding_provider.test_connection()
    finally:
        await client.close()


def _masked_fingerprint(provider_id: str, fingerprint: str) -> str:
    masked = SecretDescriptor(provider_id, fingerprint, "keyring").masked_fingerprint
    if masked is None:
        raise CliError("Credential storage did not return a fingerprint.")
    return masked


async def _configure_provider(args: argparse.Namespace, context: _CliContext) -> None:
    secret = context.read_secret("API key (hidden, leave blank for local no-auth endpoints): ")
    config = context.config_store.load(apply_env=False)
    existing = config.providers.get(args.provider_id)
    provider = ProviderConfig(
        base_url=args.base_url,
        generation_model=(
            args.model
            if args.role == "generation"
            else existing.generation_model
            if existing is not None
            else None
        ),
        embedding_model=(
            args.model
            if args.role == "embedding"
            else existing.embedding_model
            if existing is not None
            else None
        ),
        embedding_dimension=(
            args.dimension
            if args.role == "embedding"
            else existing.embedding_dimension
            if existing is not None
            else args.dimension
        ),
        api_key_fingerprint=existing.api_key_fingerprint if existing is not None else None,
    )
    try:
        await _test_provider_connection(args.provider_id, args.role, provider, secret)
    except Exception:
        raise CliError(
            "Provider connection test failed; verify the endpoint, model, and credential."
        ) from None

    fingerprint = context.secret_store.set_for_role(args.role, args.provider_id, secret) if secret else None
    provider.api_key_fingerprint = fingerprint
    config.providers[args.provider_id] = provider
    if args.role == "generation":
        config.generation_provider = args.provider_id
    else:
        config.embedding_provider = args.provider_id
    context.config_store.save(config)
    credential = (
        _masked_fingerprint(args.provider_id, fingerprint) if fingerprint is not None else "no API key"
    )
    print(
        f"Configured {args.role} provider '{args.provider_id}' ({credential}).",
        file=context.stdout,
    )


async def _configure(args: argparse.Namespace, context: _CliContext) -> None:
    preset = provider_preset(args.provider)
    secret = context.read_secret("API key (hidden): ")
    if not secret:
        raise CliError("API key is required.")
    provider = preset.configuration(generation_model=args.model)
    try:
        await _test_provider_connection(preset.provider_id, "generation", provider, secret)
        await _test_provider_connection(preset.provider_id, "embedding", provider, secret)
    except Exception:
        raise CliError(
            "Provider validation failed; verify the API key, model availability, and rate limit."
        ) from None

    config = context.config_store.load(apply_env=False)
    fingerprint = context.secret_store.set_for_role("generation", preset.provider_id, secret)
    context.secret_store.set_for_role("embedding", preset.provider_id, secret)
    config.providers[preset.provider_id] = provider
    config.generation_provider = preset.provider_id
    config.embedding_provider = preset.provider_id
    context.config_store.save(config)
    print(
        f"Configured {preset.provider_id} for generation and embeddings "
        f"({_masked_fingerprint(preset.provider_id, fingerprint)}).",
        file=context.stdout,
    )


async def _local_embeddings(args: argparse.Namespace, context: _CliContext) -> None:
    status = local_model_status()
    if args.action == "status":
        _print_json(status, context.stdout)
        return
    if args.action == "install":
        _print_json(status, context.stdout)
        if not args.accept_download:
            raise CliError(
                "Review the model details above, then rerun with --accept-download to download it."
            )
        installed = await asyncio.to_thread(install_local_model)
        if args.activate:
            config = context.config_store.load(apply_env=False)
            config.embedding_provider = LOCAL_PROVIDER_ID
            context.config_store.save(config)
        _print_json(installed, context.stdout)
        return
    if not args.confirm_remove:
        raise CliError("Rerun with --confirm-remove to remove the local model cache.")
    cache = local_model_cache().resolve(strict=False)
    if cache.exists():
        shutil.rmtree(cache)
    config = context.config_store.load(apply_env=False)
    if config.embedding_provider == LOCAL_PROVIDER_ID:
        config.embedding_provider = None
        context.config_store.save(config)
    _print_json(local_model_status(), context.stdout)


def _runtime_module() -> Any:
    return importlib.import_module("steering.runtime")


def _daemon_base_url(context: _CliContext) -> str:
    config = context.config_store.load()
    host = f"[{config.host}]" if ":" in config.host and not config.host.startswith("[") else config.host
    return f"http://{host}:{config.port}"


async def _daemon_request(
    context: _CliContext,
    method: str,
    path: str,
    *,
    payload: dict[str, object] | None = None,
    timeout: float = 120.0,
) -> Any:
    base_url = _daemon_base_url(context)
    try:
        async with httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout),
            trust_env=False,
            headers={"Origin": base_url},
        ) as client:
            response = await client.request(method, path, json=payload)
    except httpx.HTTPError:
        raise DaemonUnavailableError(
            "STEERING's localhost daemon is not reachable; run 'steering serve' first."
        ) from None
    if response.status_code >= 400:
        try:
            detail = response.json().get("error") or response.json().get("detail")
        except (ValueError, AttributeError):
            detail = None
        message = str(detail)[:300] if detail else f"daemon request failed with HTTP {response.status_code}"
        raise CliError(message)
    try:
        return response.json()
    except ValueError:
        raise CliError("The localhost daemon returned an invalid response.") from None


def _batch_sources(path: Path) -> list[str]:
    if not path.is_file():
        raise CliError(f"Batch file does not exist: {path}")
    return list(
        dict.fromkeys(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    )


async def _add(args: argparse.Namespace, context: _CliContext) -> None:
    if args.batch is not None:
        sources = _batch_sources(args.batch)
        if not sources:
            raise CliError("Batch file contains no sources.")

    else:
        source = cast(str, args.source)
        sources = [source]

    result = await _daemon_request(context, "POST", "/api/ingestion", payload={"sources": sources})
    records = result.get("records", []) if isinstance(result, dict) else []
    _print_json(
        {
            "added": len(records),
            "artifacts": [record["artifact"] for record in records if isinstance(record, dict)],
        },
        context.stdout,
    )


async def _import_telegram(args: argparse.Namespace, context: _CliContext) -> None:
    from steering.ingestion.telegram import telegram_sources

    sources = telegram_sources(args.export)
    if not sources:
        raise CliError("Telegram export contains no supported URLs.")

    result = await _daemon_request(context, "POST", "/api/ingestion", payload={"sources": sources})
    records = result.get("records", []) if isinstance(result, dict) else []
    _print_json(
        {
            "discovered_sources": len(sources),
            "imported_artifacts": len(records),
            "artifact_ids": [
                record["artifact"]["id"]
                for record in records
                if isinstance(record, dict) and isinstance(record.get("artifact"), dict)
            ],
        },
        context.stdout,
    )


async def _jobs(args: argparse.Namespace, context: _CliContext) -> None:
    result = await _daemon_request(context, "GET", f"/api/jobs?limit={args.limit}")
    jobs = result.get("jobs", []) if isinstance(result, dict) else []
    _print_json(jobs, context.stdout)


async def _doctor(_args: argparse.Namespace, context: _CliContext) -> None:
    report = await _daemon_request(context, "GET", "/api/health", timeout=10.0)
    _print_json(report, context.stdout)


async def _backup(args: argparse.Namespace, context: _CliContext) -> None:
    result = await _daemon_request(
        context,
        "POST",
        "/api/maintenance/backup",
        payload={"destination": str(args.destination)},
    )
    _print_json(result, context.stdout)


async def _restore(args: argparse.Namespace, context: _CliContext) -> None:
    from steering.database.backup import restore_backup

    config = context.config_store.load(apply_env=False)
    target = args.target or config.database_file
    if target.expanduser().resolve(strict=False) == config.database_file.resolve(strict=False):
        try:
            await _daemon_request(context, "GET", "/api/health", timeout=2.0)
        except DaemonUnavailableError:
            pass
        else:
            raise CliError("Stop the STEERING daemon before restoring its active database.")
    restored = restore_backup(args.backup, target)
    _print_json({"restored_database": str(restored)}, context.stdout)


async def _reindex(_args: argparse.Namespace, context: _CliContext) -> None:
    result = await _daemon_request(context, "POST", "/api/maintenance/reindex")
    _print_json(result, context.stdout)


def _agent_configuration(agent: str, config: AppConfig) -> str:
    from steering.mcp.instructions import AGENT_WORKFLOW_INSTRUCTIONS

    url = f"http://{config.host}:{config.port}/mcp"
    if agent == "codex":
        setup = f'Add this to ~/.codex/config.toml:\n\n[mcp_servers.steering]\nurl = "{url}"\n'
    else:
        setup = f"Run this command:\n\nclaude mcp add --transport http steering {url}\n"
    return f"{setup}\nReview and add these project instructions:\n\n{AGENT_WORKFLOW_INSTRUCTIONS}\n"


async def _agent_config(args: argparse.Namespace, context: _CliContext) -> None:
    print(
        _agent_configuration(args.agent, context.config_store.load()),
        file=context.stdout,
        end="",
    )


async def _serve(_args: argparse.Namespace, context: _CliContext) -> None:
    result = _runtime_module().run_server(config_store=context.config_store)
    if inspect.isawaitable(result):
        await result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="steering", description="Local AI-engineering recall")
    commands = parser.add_subparsers(dest="command", required=True)

    simple = commands.add_parser("configure", help="configure a tested Gemini or OpenAI preset")
    simple.add_argument("--provider", choices=tuple(sorted(PROVIDER_PRESETS)), required=True)
    simple.add_argument("--model", help="optional generation-model override")

    configure = commands.add_parser("configure-provider", help="configure and test an AI provider")
    configure.add_argument("--role", choices=("generation", "embedding"), required=True)
    configure.add_argument("--provider-id", default="openai-compatible")
    configure.add_argument("--base-url", required=True)
    configure.add_argument("--model", required=True)
    configure.add_argument("--dimension", type=int, default=768)

    local = commands.add_parser("local-embeddings", help="manage the optional local ONNX model")
    local.add_argument("action", choices=("status", "install", "remove"))
    local.add_argument("--accept-download", action="store_true")
    local.add_argument("--activate", action="store_true")
    local.add_argument("--confirm-remove", action="store_true")

    add = commands.add_parser("add", help="ingest one URL or a newline-delimited batch")
    source = add.add_mutually_exclusive_group(required=True)
    source.add_argument("source", nargs="?")
    source.add_argument("--batch", type=Path)

    telegram = commands.add_parser("import-telegram", help="ingest URLs from a Telegram export")
    telegram.add_argument("export", type=Path)

    jobs = commands.add_parser("jobs", help="show recent ingestion jobs")
    jobs.add_argument("--limit", type=int, default=100)
    commands.add_parser("serve", help="run the local web and MCP daemon")
    commands.add_parser("doctor", help="check local configuration and runtime health")

    backup = commands.add_parser("backup", help="create a verified logical backup")
    backup.add_argument("destination", type=Path)
    restore = commands.add_parser("restore", help="restore a backup without overwriting a database")
    restore.add_argument("backup", type=Path)
    restore.add_argument("--target", type=Path)
    commands.add_parser("reindex", help="back up, re-embed, and rebuild retrieval indexes")

    agent = commands.add_parser("agent-config", help="print MCP configuration for a coding agent")
    agent.add_argument("agent", choices=("codex", "claude"))
    return parser


async def _dispatch(args: argparse.Namespace, context: _CliContext) -> None:
    handlers: dict[str, Callable[[argparse.Namespace, _CliContext], Awaitable[None]]] = {
        "configure": _configure,
        "configure-provider": _configure_provider,
        "local-embeddings": _local_embeddings,
        "add": _add,
        "import-telegram": _import_telegram,
        "jobs": _jobs,
        "serve": _serve,
        "doctor": _doctor,
        "backup": _backup,
        "restore": _restore,
        "reindex": _reindex,
        "agent-config": _agent_config,
    }
    await handlers[args.command](args, context)


def main(
    argv: Sequence[str] | None = None,
    *,
    config_store: ConfigStore | None = None,
    secret_store: SecretStore | None = None,
    read_secret: Callable[[str], str] = getpass.getpass,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    context = _CliContext(
        config_store=config_store or ConfigStore(),
        secret_store=secret_store or KeyringSecretStore(),
        read_secret=read_secret,
        stdout=stdout,
    )
    try:
        asyncio.run(_dispatch(build_parser().parse_args(argv), context))
    except CliError as exc:
        print(f"error: {exc}", file=stderr)
        return 2
    except Exception as exc:
        print(f"error: command failed ({type(exc).__name__})", file=stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
