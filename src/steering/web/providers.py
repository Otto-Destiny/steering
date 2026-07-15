from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict

from steering.config.models import AppConfig
from steering.config.secrets import SecretStore
from steering.config.store import ConfigStore
from steering.domain.models import ProviderConfig


class ProviderView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_id: str
    base_url: str
    generation_model: str | None
    embedding_model: str | None
    embedding_dimension: int
    roles: list[str]
    generation_has_api_key: bool
    generation_key_fingerprint: str | None
    embedding_has_api_key: bool
    embedding_key_fingerprint: str | None


class ProviderSettingsService(Protocol):
    def list_providers(self) -> Sequence[ProviderView]: ...

    async def save_provider(
        self,
        *,
        provider_id: str,
        role: str,
        base_url: str,
        generation_model: str | None,
        embedding_model: str | None,
        embedding_dimension: int,
        generation_api_key: str | None,
        embedding_api_key: str | None,
    ) -> ProviderView: ...

    async def test_provider(self, provider_id: str, role: str | None = None) -> None: ...

    def delete_key(self, provider_id: str, role: str) -> bool: ...


class ProviderConnectionTester(Protocol):
    async def __call__(
        self,
        provider_id: str,
        role: Literal["generation", "embedding"],
        config: ProviderConfig,
        api_key: str | None,
    ) -> None: ...


def _masked(fingerprint: str | None) -> str | None:
    if fingerprint is None:
        return None
    return f"****{fingerprint.rsplit(':', 1)[-1][-8:]}"


class LocalProviderSettings:
    """Secret-safe adapter around the serialized settings and OS credential store."""

    def __init__(
        self,
        *,
        config_store: ConfigStore,
        secret_store: SecretStore,
        connection_tester: ProviderConnectionTester,
    ) -> None:
        self._config_store = config_store
        self._secret_store = secret_store
        self._connection_tester = connection_tester

    def list_providers(self) -> Sequence[ProviderView]:
        config = self._config_store.load()
        return [
            self._view(provider_id, provider, config) for provider_id, provider in config.providers.items()
        ]

    async def save_provider(
        self,
        *,
        provider_id: str,
        role: str,
        base_url: str,
        generation_model: str | None,
        embedding_model: str | None,
        embedding_dimension: int,
        generation_api_key: str | None,
        embedding_api_key: str | None,
    ) -> ProviderView:
        config = self._config_store.load(apply_env=False)
        existing = config.providers.get(provider_id)
        provider = ProviderConfig(
            base_url=base_url,
            generation_model=(
                generation_model
                if role in {"generation", "both"}
                else existing.generation_model
                if existing is not None
                else None
            ),
            embedding_model=(
                embedding_model
                if role in {"embedding", "both"}
                else existing.embedding_model
                if existing is not None
                else None
            ),
            embedding_dimension=(
                embedding_dimension
                if role in {"embedding", "both"}
                else existing.embedding_dimension
                if existing is not None
                else embedding_dimension
            ),
            api_key_fingerprint=None,
        )
        selected_roles = self._selected_roles(role)
        candidate_keys = {
            "generation": generation_api_key or self._secret_store.get_for_role("generation", provider_id),
            "embedding": embedding_api_key or self._secret_store.get_for_role("embedding", provider_id),
        }
        # Connection validation is deliberately completed before config or keyring writes.
        for selected_role in selected_roles:
            await self._connection_tester(
                provider_id,
                selected_role,
                provider,
                candidate_keys[selected_role],
            )

        if generation_api_key and "generation" in selected_roles:
            self._secret_store.set_for_role("generation", provider_id, generation_api_key)
        if embedding_api_key and "embedding" in selected_roles:
            self._secret_store.set_for_role("embedding", provider_id, embedding_api_key)
        providers = dict(config.providers)
        providers[provider_id] = provider
        updates: dict[str, object] = {"providers": providers}
        if role in {"generation", "both"}:
            updates["generation_provider"] = provider_id
        if role in {"embedding", "both"}:
            updates["embedding_provider"] = provider_id
        saved = config.model_copy(update=updates)
        self._config_store.save(saved)
        return self._view(provider_id, provider, saved)

    async def test_provider(self, provider_id: str, role: str | None = None) -> None:
        config = self._config_store.load()
        provider = config.providers.get(provider_id)
        if provider is None:
            raise KeyError(provider_id)
        roles = self._selected_roles(role) if role else self._active_roles(provider_id, config)
        for selected_role in roles:
            await self._connection_tester(
                provider_id,
                selected_role,
                provider,
                self._secret_store.get_for_role(selected_role, provider_id),
            )

    def delete_key(self, provider_id: str, role: str) -> bool:
        config = self._config_store.load(apply_env=False)
        provider = config.providers.get(provider_id)
        if provider is None:
            raise KeyError(provider_id)
        selected_role = self._selected_roles(role)[0]
        return self._secret_store.delete_for_role(selected_role, provider_id)

    def _view(self, provider_id: str, provider: ProviderConfig, config: AppConfig) -> ProviderView:
        roles = []
        if config.generation_provider == provider_id:
            roles.append("generation")
        if config.embedding_provider == provider_id:
            roles.append("embedding")
        generation_fingerprint = self._fingerprint("generation", provider_id)
        embedding_fingerprint = self._fingerprint("embedding", provider_id)
        return ProviderView(
            provider_id=provider_id,
            base_url=str(provider.base_url),
            generation_model=provider.generation_model,
            embedding_model=provider.embedding_model,
            embedding_dimension=provider.embedding_dimension,
            roles=roles,
            generation_has_api_key=generation_fingerprint is not None,
            generation_key_fingerprint=_masked(generation_fingerprint),
            embedding_has_api_key=embedding_fingerprint is not None,
            embedding_key_fingerprint=_masked(embedding_fingerprint),
        )

    @staticmethod
    def _selected_roles(role: str) -> list[Literal["generation", "embedding"]]:
        if role == "both":
            return ["generation", "embedding"]
        if role in {"generation", "embedding"}:
            return [cast(Literal["generation", "embedding"], role)]
        raise ValueError("role must be generation, embedding, or both")

    @staticmethod
    def _active_roles(provider_id: str, config: AppConfig) -> list[Literal["generation", "embedding"]]:
        roles: list[Literal["generation", "embedding"]] = []
        if config.generation_provider == provider_id:
            roles.append("generation")
        if config.embedding_provider == provider_id:
            roles.append("embedding")
        if not roles:
            raise ValueError("provider is not active for generation or embedding")
        return roles

    def _fingerprint(self, role: str, provider_id: str) -> str | None:
        return self._secret_store.fingerprint_for_role(role, provider_id)
