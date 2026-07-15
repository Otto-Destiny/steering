from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from platformdirs import user_config_path

from steering.config.models import AppConfig
from steering.domain.credentials import reject_high_confidence_credentials

FORBIDDEN_SECRET_KEYS = frozenset({"api_key", "apikey", "secret", "password", "token", "authorization"})


def default_config_path() -> Path:
    return user_config_path("steering", appauthor=False) / "config.json"


def _assert_secret_free(value: Any, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in FORBIDDEN_SECRET_KEYS or normalized.endswith("_api_key"):
                raise ValueError(f"secret-bearing field is not serializable: {path}.{key}")
            _assert_secret_free(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_secret_free(child, f"{path}[{index}]")
    elif isinstance(value, str):
        reject_high_confidence_credentials(value)


def _set_provider_override(
    data: dict[str, Any],
    environ: Mapping[str, str],
    role: str,
) -> None:
    role_upper = role.upper()
    provider_key = f"{role}_provider"
    provider_id = environ.get(f"STEERING_{role_upper}_PROVIDER") or data.get(provider_key)
    if provider_id is None:
        return
    data[provider_key] = provider_id
    providers = data.setdefault("providers", {})
    provider = dict(providers.get(provider_id, {}))
    base_url = environ.get(f"STEERING_{role_upper}_BASE_URL")
    if base_url is not None:
        provider["base_url"] = base_url
    model_key = "generation_model" if role == "generation" else "embedding_model"
    model_value = environ.get(f"STEERING_{role_upper}_MODEL")
    if model_value is not None:
        provider[model_key] = model_value
    if role == "embedding" and "STEERING_EMBEDDING_DIMENSION" in environ:
        provider["embedding_dimension"] = int(environ["STEERING_EMBEDDING_DIMENSION"])
    if provider:
        providers[provider_id] = provider


def apply_environment_overrides(
    config: AppConfig,
    environ: Mapping[str, str] | None = None,
) -> AppConfig:
    env = os.environ if environ is None else environ
    data = config.model_dump(mode="json")
    scalar_overrides: tuple[tuple[str, str, Any], ...] = (
        ("STEERING_DATABASE_PATH", "database_path", str),
        ("STEERING_HOST", "host", str),
        ("STEERING_PORT", "port", int),
        ("STEERING_LOG_LEVEL", "log_level", str),
        ("STEERING_GENERATION_CONTEXT_WINDOW_TOKENS", "generation_context_window_tokens", int),
        (
            "STEERING_GENERATION_RESERVED_OUTPUT_TOKENS",
            "generation_reserved_output_tokens",
            int,
        ),
    )
    for env_name, field_name, converter in scalar_overrides:
        if env_name in env:
            data[field_name] = converter(env[env_name])
    _set_provider_override(data, env, "generation")
    _set_provider_override(data, env, "embedding")
    return AppConfig.model_validate(data)


class ConfigStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_config_path()

    def load(
        self,
        *,
        apply_env: bool = True,
        environ: Mapping[str, str] | None = None,
    ) -> AppConfig:
        if self.path.exists():
            config = AppConfig.model_validate_json(self.path.read_text(encoding="utf-8"))
        else:
            config = AppConfig()
        return apply_environment_overrides(config, environ) if apply_env else config

    def save(self, config: AppConfig) -> Path:
        payload = config.model_dump(mode="json")
        _assert_secret_free(payload)
        serialized = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            Path(temporary_name).replace(self.path)
        except BaseException:
            Path(temporary_name).unlink(missing_ok=True)
            raise
        return self.path
