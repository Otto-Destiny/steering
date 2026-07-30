from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from platformdirs import user_config_path

from steering.config.models import AppConfig
from steering.domain.credentials import reject_high_confidence_credentials
from steering.providers.presets import provider_preset

FORBIDDEN_SECRET_KEYS = frozenset({"api_key", "apikey", "secret", "password", "token", "authorization"})
_ENV_FILE_LIMIT = 64 * 1024
_RECOGNIZED_ENV = frozenset(
    {
        "STEERING_PROVIDER",
        "STEERING_API_KEY",
        "STEERING_MODEL",
        "STEERING_DATABASE_PATH",
        "STEERING_HOST",
        "STEERING_PORT",
        "STEERING_LOG_LEVEL",
        "STEERING_BROWSER_HEADLESS",
        "STEERING_X_API_CLIENT_ID",
        "STEERING_READ_THREADS",
        "STEERING_GENERATION_PROVIDER",
        "STEERING_GENERATION_BASE_URL",
        "STEERING_GENERATION_MODEL",
        "STEERING_GENERATION_API_KEY",
        "STEERING_EMBEDDING_PROVIDER",
        "STEERING_EMBEDDING_BASE_URL",
        "STEERING_EMBEDDING_MODEL",
        "STEERING_EMBEDDING_DIMENSION",
        "STEERING_EMBEDDING_API_KEY",
    }
)
_PROVIDER_SECRET_ENV = re.compile(r"^STEERING_[A-Z0-9_]+_API_KEY$")


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


def _boolean(value: str) -> bool:
    """Read a flag the way a person would write one in `.env.local`.

    Silently treating an unrecognised value as false would leave a user who
    wrote `yes` believing a setting is on when it is not.
    """

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"expected a true/false value, received {value!r}")


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


def _apply_simple_provider(data: dict[str, Any], environ: Mapping[str, str]) -> None:
    selected = environ.get("STEERING_PROVIDER")
    if selected is None:
        if "STEERING_MODEL" in environ:
            raise ValueError("STEERING_MODEL requires STEERING_PROVIDER")
        return
    preset = provider_preset(selected)
    provider = preset.configuration(generation_model=environ.get("STEERING_MODEL"))
    providers = data.setdefault("providers", {})
    providers[preset.provider_id] = provider.model_dump(mode="json")
    data["generation_provider"] = preset.provider_id
    data["embedding_provider"] = preset.provider_id


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
        ("STEERING_X_API_CLIENT_ID", "x_api_client_id", str),
        ("STEERING_READ_THREADS", "read_threads", str),
    )
    for env_name, field_name, converter in scalar_overrides:
        if env_name in env:
            data[field_name] = converter(env[env_name])
    if "STEERING_BROWSER_HEADLESS" in env:
        data["browser_headless"] = _boolean(env["STEERING_BROWSER_HEADLESS"])
    _apply_simple_provider(data, env)
    _set_provider_override(data, env, "generation")
    _set_provider_override(data, env, "embedding")
    return AppConfig.model_validate(data)


def _recognized_environment_name(name: str) -> bool:
    return name in _RECOGNIZED_ENV or _PROVIDER_SECRET_ENV.fullmatch(name) is not None


def load_local_environment(path: Path) -> dict[str, str]:
    """Read a small, non-interpolating `.env.local` without exposing values."""

    if not path.is_file():
        return {}
    if path.stat().st_size > _ENV_FILE_LIMIT:
        raise ValueError(".env.local exceeds the 64 KiB safety limit")
    loaded: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"invalid .env.local assignment on line {line_number}")
        name, value = line.split("=", 1)
        name = name.strip()
        if not _recognized_environment_name(name):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        loaded[name] = value
    return loaded


class ConfigStore:
    def __init__(
        self,
        path: str | Path | None = None,
        *,
        env_file: str | Path | None = None,
    ) -> None:
        self.path = Path(path) if path is not None else default_config_path()
        self.env_file = (
            Path(env_file) if env_file is not None else Path.cwd() / ".env.local" if path is None else None
        )

    def environment(self, environ: Mapping[str, str] | None = None) -> dict[str, str]:
        if environ is not None:
            return dict(environ)
        resolved = load_local_environment(self.env_file) if self.env_file is not None else {}
        process_provider = os.environ.get("STEERING_PROVIDER")
        local_provider = resolved.get("STEERING_PROVIDER")
        if process_provider and local_provider and process_provider.lower() != local_provider.lower():
            if "STEERING_API_KEY" not in os.environ:
                resolved.pop("STEERING_API_KEY", None)
            if "STEERING_MODEL" not in os.environ:
                resolved.pop("STEERING_MODEL", None)
        # The real process environment deliberately has the final say.
        resolved.update(os.environ)
        return resolved

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
        return apply_environment_overrides(config, self.environment(environ)) if apply_env else config

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
