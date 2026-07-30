from __future__ import annotations

import ipaddress
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal, cast

import uvicorn
from pydantic import SecretStr
from starlette.applications import Starlette
from starlette.routing import Mount

from steering.config import ConfigStore, SecretStore
from steering.domain.models import ProviderConfig
from steering.domain.protocols import ImageUnderstandingProvider
from steering.mcp import create_mcp_server
from steering.observability import configure_logging
from steering.providers import (
    GeminiClient,
    GeminiEmbeddingProvider,
    GeminiGenerationProvider,
    OpenAICompatibleClient,
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleGenerationProvider,
)
from steering.runtime import SteeringRuntime, create_runtime
from steering.web import LocalProviderSettings, create_web_app


async def _test_provider(
    provider_id: str,
    role: Literal["generation", "embedding"],
    config: ProviderConfig,
    api_key: str | None,
) -> None:
    if provider_id == "gemini":
        if not api_key:
            raise ValueError("Gemini requires an API key")
        gemini = GeminiClient(
            base_url=str(config.base_url),
            api_key=SecretStr(api_key),
        )
        try:
            if role == "generation" and config.generation_model:
                await GeminiGenerationProvider(
                    client=gemini,
                    model_id=config.generation_model,
                ).test_connection()
            elif role == "embedding" and config.embedding_model:
                await GeminiEmbeddingProvider(
                    client=gemini,
                    model_id=config.embedding_model,
                    dimension=config.embedding_dimension,
                ).test_connection()
            else:
                raise ValueError(f"provider needs a configured {role} model")
        finally:
            await gemini.close()
        return
    client = OpenAICompatibleClient(
        base_url=str(config.base_url),
        api_key=SecretStr(api_key) if api_key else None,
    )
    try:
        if role == "generation" and config.generation_model:
            await OpenAICompatibleGenerationProvider(
                client=client,
                model_id=config.generation_model,
            ).test_connection()
        elif role == "embedding" and config.embedding_model:
            await OpenAICompatibleEmbeddingProvider(
                client=client,
                model_id=config.embedding_model,
                dimension=config.embedding_dimension,
                provider_id=provider_id,
            ).test_connection()
        else:
            raise ValueError(f"provider needs a configured {role} model")
    finally:
        await client.close()


def create_app(
    *,
    runtime: SteeringRuntime | None = None,
    config_store: ConfigStore | None = None,
    secret_store: SecretStore | None = None,
    close_runtime: bool | None = None,
) -> Starlette:
    """Create the one Starlette app serving UI, typed API, and MCP."""

    owns_runtime = runtime is None if close_runtime is None else close_runtime
    active = runtime or create_runtime(config_store=config_store, secret_store=secret_store)
    provider_settings = LocalProviderSettings(
        config_store=active.config_store,
        secret_store=active.secret_store,
        connection_tester=_test_provider,
    )
    image_provider = (
        cast(ImageUnderstandingProvider, active.generation)
        if hasattr(active.generation, "understand_image")
        else None
    )
    web = create_web_app(
        engine=active.engine,
        ingestion=active.ingestion,
        repository=active.repository,
        provider_settings=provider_settings,
        resolved_ingestion=active.ingestion,
        browser_capture=active.browser,
        login_session=active.browser_login,
        image_provider=image_provider,
        on_record_changed=active.retriever.mark_dirty,
        x_api=active.x_api,
        x_api_client_id=active.config.x_api_client_id,
    )
    mcp_server = create_mcp_server(active.engine)
    mcp_app = mcp_server.streamable_http_app()

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        try:
            async with mcp_server.session_manager.run():
                yield
        finally:
            if owns_runtime:
                await active.aclose()

    app = Starlette(
        debug=False,
        routes=[
            Mount("/mcp", app=mcp_app, name="mcp"),
            Mount("/", app=web, name="web"),
        ],
        lifespan=lifespan,
    )
    app.state.runtime = active
    return app


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


async def run_server(*, config_store: ConfigStore | None = None) -> None:
    """Run the localhost daemon without nesting an event loop."""

    store = config_store or ConfigStore()
    runtime = create_runtime(config_store=store)
    configure_logging(runtime.config.log_level)
    if not _is_loopback_host(runtime.config.host):
        await runtime.aclose()
        raise ValueError("STEERING 0.1 may bind only to a loopback host")
    app = create_app(runtime=runtime, close_runtime=True)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=runtime.config.host,
            port=runtime.config.port,
            log_level=runtime.config.log_level.lower(),
            server_header=False,
        )
    )
    await server.serve()


def application() -> Any:
    """Uvicorn factory entry point using the default local configuration."""

    return create_app()
