from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import httpx
from platformdirs import user_cache_path, user_data_path
from pydantic import SecretStr

from steering import __version__
from steering.config import AppConfig, ConfigStore, KeyringSecretStore, SecretStore
from steering.database import SCHEMA_REVISION, DatabaseRuntime
from steering.domain.protocols import (
    ArtifactRepository,
    EmbeddingProvider,
    GenerationProvider,
    ImageUnderstandingProvider,
)
from steering.extraction.service import ExtractionService, default_cache
from steering.ingestion.browser import ManagedBrowserCapture
from steering.ingestion.resolvers import default_registry
from steering.ingestion.security import SafeFetcher
from steering.ingestion.service import IngestionService
from steering.intelligence.service import SteeringEngine
from steering.providers import (
    HashEmbeddingProvider,
    OpenAICompatibleClient,
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleGenerationProvider,
    UnconfiguredGenerationProvider,
)
from steering.retrieval.hybrid import HybridRetriever

_EVALUATION_RUNTIMES: list[DatabaseRuntime] = []


class SteeringRuntime:
    """Own every process-lifetime resource behind the one local daemon."""

    def __init__(
        self,
        *,
        config: AppConfig,
        config_store: ConfigStore,
        secret_store: SecretStore,
    ) -> None:
        self.config = config
        self.config_store = config_store
        self.secret_store = secret_store
        self.database = DatabaseRuntime(config.database_file)
        self.repository = self.database.repository
        self._provider_clients: list[OpenAICompatibleClient] = []
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            headers={"User-Agent": f"STEERING/{__version__}"},
            follow_redirects=False,
            trust_env=False,
        )
        try:
            self.generation = self._generation_provider()
            self.embedding = self._embedding_provider()
            self.fetcher = SafeFetcher(client=self._http_client)
            self.registry = default_registry(self.fetcher)
            cache_root = user_cache_path("steering", appauthor=False)
            self.extraction = ExtractionService(
                generation=self.generation,
                embedding=self.embedding,
                cache=default_cache(cache_root / "extraction"),
                context_window_tokens=config.generation_context_window_tokens,
                reserved_output_tokens=config.generation_reserved_output_tokens,
            )
            image_provider = (
                cast(ImageUnderstandingProvider, self.generation)
                if hasattr(self.generation, "understand_image")
                else None
            )
            self.ingestion = IngestionService(
                registry=self.registry,
                extraction=self.extraction,
                repository=self.repository,
                media_fetcher=self.fetcher,
                image_provider=image_provider,
            )
            self.retriever = HybridRetriever(
                repository=self.repository,
                embedding_provider=self.embedding,
            )
            configured_generation = self.generation if config.generation_provider is not None else None
            self.engine = SteeringEngine(
                repository=self.repository,
                retriever=self.retriever,
                generation=configured_generation,
            )
            self.browser = ManagedBrowserCapture(
                profile_directory=user_data_path("steering", appauthor=False) / "browser-profile"
            )
        except BaseException:
            self.database.close()
            raise
        self._closed = False

    def _role_secret(self, role: str, provider_id: str | None) -> str | None:
        getter = getattr(self.secret_store, "get_for_role", None)
        if callable(getter):
            return cast(str | None, getter(role, provider_id))
        return self.secret_store.get(provider_id or role)

    def _provider_client(self, role: str, provider_id: str) -> OpenAICompatibleClient:
        provider = self.config.providers.get(provider_id)
        if provider is None:
            raise ValueError(f"{role} provider '{provider_id}' has no configuration")
        secret = self._role_secret(role, provider_id)
        client = OpenAICompatibleClient(
            base_url=str(provider.base_url),
            api_key=SecretStr(secret) if secret else None,
        )
        self._provider_clients.append(client)
        return client

    def _generation_provider(self) -> GenerationProvider:
        provider_id = self.config.generation_provider
        if provider_id is None:
            return UnconfiguredGenerationProvider()
        provider = self.config.providers.get(provider_id)
        if provider is None or not provider.generation_model:
            raise ValueError(f"generation provider '{provider_id}' needs a generation model")
        return OpenAICompatibleGenerationProvider(
            client=self._provider_client("generation", provider_id),
            model_id=provider.generation_model,
        )

    def _embedding_provider(self) -> EmbeddingProvider:
        provider_id = self.config.embedding_provider
        if provider_id is None:
            return HashEmbeddingProvider()
        provider = self.config.providers.get(provider_id)
        if provider is None or not provider.embedding_model:
            raise ValueError(f"embedding provider '{provider_id}' needs an embedding model")
        return OpenAICompatibleEmbeddingProvider(
            client=self._provider_client("embedding", provider_id),
            model_id=provider.embedding_model,
            dimension=provider.embedding_dimension,
        )

    async def reindex(self) -> int:
        self.retriever.mark_dirty()
        await self.retriever.refresh()
        return len(self.repository.list_records())

    def doctor(self) -> Mapping[str, Any]:
        generation_id = self.config.generation_provider
        embedding_id = self.config.embedding_provider
        return {
            "status": "ready" if generation_id else "setup_required",
            "version": __version__,
            "schema_revision": SCHEMA_REVISION,
            "database_path": str(self.config.database_file),
            "artifact_count": len(self.repository.list_records()),
            "unresolved_issue_count": len(self.repository.list_issues(unresolved_only=True)),
            "generation_provider": generation_id,
            "embedding_provider": embedding_id or "local_hash_fallback",
            "generation_key_fingerprint": self._masked_fingerprint("generation", generation_id),
            "embedding_key_fingerprint": self._masked_fingerprint("embedding", embedding_id),
            "browser_extra_available": _browser_extra_available(),
            "host": self.config.host,
            "port": self.config.port,
        }

    def _masked_fingerprint(self, role: str, provider_id: str | None) -> str | None:
        if provider_id is None:
            return None
        describer = getattr(self.secret_store, "describe_for_role", None)
        if callable(describer):
            descriptor = describer(role, provider_id)
            return cast(str | None, descriptor.masked_fingerprint)
        fingerprinter = getattr(self.secret_store, "fingerprint", None)
        if not callable(fingerprinter):
            return None
        value = fingerprinter(provider_id or role)
        return None if value is None else f"****{str(value)[-8:]}"

    async def aclose(self) -> None:
        if self._closed:
            return
        for client in self._provider_clients:
            await client.close()
        await self._http_client.aclose()
        self.database.close()
        self._closed = True

    async def close(self) -> None:
        await self.aclose()

    def __enter__(self) -> SteeringRuntime:
        return self

    def __exit__(self, *_: object) -> None:
        asyncio.run(self.aclose())

    async def __aenter__(self) -> SteeringRuntime:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()


def create_runtime(
    *,
    config_store: ConfigStore | None = None,
    secret_store: SecretStore | None = None,
    config: AppConfig | None = None,
) -> SteeringRuntime:
    store = config_store or ConfigStore()
    resolved_config = config or store.load()
    resolved_secrets = secret_store or KeyringSecretStore()
    return SteeringRuntime(
        config=resolved_config,
        config_store=store,
        secret_store=resolved_secrets,
    )


def _browser_extra_available() -> bool:
    try:
        import playwright  # noqa: F401
    except ImportError:
        return False
    return True


def evaluation_factory(
    *, embedding_provider: EmbeddingProvider
) -> tuple[ArtifactRepository, HybridRetriever]:
    """Disposable repository/retriever factory used by the committed eval suite."""

    import tempfile

    root = Path(tempfile.mkdtemp(prefix="steering-eval-"))
    database = DatabaseRuntime(root / "evaluation.lbug")
    repository = database.repository
    # Keep the database owner alive for the repository's evaluation lifetime.
    _EVALUATION_RUNTIMES.append(database)
    return repository, HybridRetriever(
        repository=repository,
        embedding_provider=embedding_provider,
    )
