from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


PROVIDER_CONFIG_SCHEMA = "arc.llm.providers.v1"
PROVIDER_CONFIG_ENV = "ARC_LLM_PROVIDER_CONFIG"
LOCAL_PROVIDER_CONFIG_PATH = Path("llm-providers.json")
DEFAULT_PROVIDER_CONFIG_PATH = Path("~/.config/arc/llm-providers.json")
OPENAI_COMPATIBLE_TYPE = "openai-compatible"
VALID_JSON_MODES = frozenset({"json_schema", "json_object", "none"})
VALID_MODEL_TIERS = frozenset({"low", "medium", "high"})
LEGACY_SELECTION_FIELDS = frozenset({"default", "auto_provider_priority"})


class ProviderConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ConfiguredProvider:
    id: str
    type: str
    base_url: str
    api_key: str | None = None
    api_key_optional: bool = False
    models: dict[str, str] | None = None
    json_mode: str = "json_schema"

    def is_usable(self, *, env: Mapping[str, str] | None = None) -> bool:
        env = env if env is not None else os.environ
        if not self.base_url:
            return False
        if self.has_api_key(env=env):
            return True
        return self.api_key_optional

    def has_api_key(self, *, env: Mapping[str, str] | None = None) -> bool:
        return bool(_usable_api_key(self.api_key))

    def resolved_api_key(self, *, env: Mapping[str, str] | None = None) -> str | None:
        if api_key := _usable_api_key(self.api_key):
            return api_key
        if self.api_key_optional:
            return "not-needed"
        return None

    def model_for_tier(self, tier: str) -> str | None:
        models = self.models or {}
        return models.get(tier)


@dataclass(frozen=True)
class ProviderConfig:
    path: str
    providers: list[ConfiguredProvider]

    def provider(self, provider_id: str) -> ConfiguredProvider | None:
        for provider in self.providers:
            if provider.id == provider_id:
                return provider
        return None


def provider_config_path(*, env: Mapping[str, str] | None = None) -> Path:
    return provider_config_paths(env=env)[0]


def provider_config_paths(*, env: Mapping[str, str] | None = None) -> list[Path]:
    env = env if env is not None else os.environ
    if path := env.get(PROVIDER_CONFIG_ENV):
        return [Path(path).expanduser()]
    return [Path.cwd() / LOCAL_PROVIDER_CONFIG_PATH, DEFAULT_PROVIDER_CONFIG_PATH.expanduser()]


def load_provider_config(*, env: Mapping[str, str] | None = None) -> ProviderConfig:
    env = env if env is not None else os.environ
    path = None
    for candidate in provider_config_paths(env=env):
        if candidate.exists():
            path = candidate
            break
    if path is None:
        path = provider_config_path(env=env)
    if not path.exists():
        return ProviderConfig(path=str(path), providers=[])
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProviderConfigError(f"Provider config is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProviderConfigError("Provider config must be a JSON object")
    return parse_provider_config(payload, path=str(path))


def parse_provider_config(payload: Mapping[str, Any], *, path: str = "") -> ProviderConfig:
    schema_version = str(payload.get("schema_version") or "").strip()
    if schema_version != PROVIDER_CONFIG_SCHEMA:
        raise ProviderConfigError(f"schema_version must be {PROVIDER_CONFIG_SCHEMA}")
    legacy_fields = sorted(LEGACY_SELECTION_FIELDS.intersection(payload))
    if legacy_fields:
        names = ", ".join(legacy_fields)
        raise ProviderConfigError(
            f"Provider config must not set run selection fields: {names}; "
            "choose provider/model_tier per run instead"
        )
    raw_providers = payload.get("providers")
    if raw_providers is None:
        raw_providers = []
    if not isinstance(raw_providers, list):
        raise ProviderConfigError("providers must be a list")
    providers = [_parse_provider(item, index) for index, item in enumerate(raw_providers)]
    seen: set[str] = set()
    for provider in providers:
        if provider.id in seen:
            raise ProviderConfigError(f"duplicate provider id: {provider.id}")
        seen.add(provider.id)
    return ProviderConfig(
        path=path,
        providers=providers,
    )


def configured_provider(provider_id: str, *, env: Mapping[str, str] | None = None) -> ConfiguredProvider | None:
    return load_provider_config(env=env).provider(provider_id)


def usable_configured_providers(*, env: Mapping[str, str] | None = None) -> list[ConfiguredProvider]:
    config = load_provider_config(env=env)
    return [provider for provider in config.providers if provider.is_usable(env=env)]


def select_configured_provider(*, env: Mapping[str, str] | None = None) -> ConfiguredProvider | None:
    config = load_provider_config(env=env)
    for provider in config.providers:
        if provider.has_api_key(env=env):
            return provider
    for provider in config.providers:
        if provider.api_key_optional and provider.is_usable(env=env):
            return provider
    return None


def _parse_provider(raw: Any, index: int) -> ConfiguredProvider:
    if not isinstance(raw, dict):
        raise ProviderConfigError(f"providers[{index}] must be an object")
    provider_id = _safe_id(_required_text(raw, "id", index), f"providers[{index}].id")
    provider_type = str(raw.get("type") or OPENAI_COMPATIBLE_TYPE).strip()
    if provider_type != OPENAI_COMPATIBLE_TYPE:
        raise ProviderConfigError(f"providers[{index}].type must be {OPENAI_COMPATIBLE_TYPE}")
    base_url = _required_text(raw, "base_url", index)
    if "api_key_env" in raw:
        raise ProviderConfigError(f"providers[{index}].api_key_env is not supported; use api_key")
    api_key = raw.get("api_key")
    if api_key is not None:
        api_key = str(api_key).strip()
        if not api_key:
            api_key = None
    api_key_optional = _bool(raw.get("api_key_optional", False), f"providers[{index}].api_key_optional")
    if not api_key and not api_key_optional:
        raise ProviderConfigError(f"providers[{index}] must set api_key or api_key_optional=true")
    models = _models(raw.get("models"), index)
    json_mode = str(raw.get("json_mode") or "json_schema").strip()
    if json_mode not in VALID_JSON_MODES:
        raise ProviderConfigError(f"providers[{index}].json_mode must be one of: json_schema, json_object, none")
    return ConfiguredProvider(
        id=provider_id,
        type=provider_type,
        base_url=base_url,
        api_key=api_key,
        api_key_optional=api_key_optional,
        models=models,
        json_mode=json_mode,
    )


def _required_text(raw: Mapping[str, Any], key: str, index: int) -> str:
    value = raw.get(key)
    if value is None:
        raise ProviderConfigError(f"providers[{index}].{key} is required")
    text = str(value).strip()
    if not text:
        raise ProviderConfigError(f"providers[{index}].{key} is required")
    return text


def _models(value: Any, index: int) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ProviderConfigError(f"providers[{index}].models must be an object")
    models: dict[str, str] = {}
    for key, model in value.items():
        text_key = str(key).strip()
        if text_key not in VALID_MODEL_TIERS:
            valid = ", ".join(sorted(VALID_MODEL_TIERS))
            raise ProviderConfigError(f"providers[{index}].models keys must be one of: {valid}")
        text_model = str(model).strip()
        if text_model:
            models[text_key] = text_model
    return models or None


def _bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ProviderConfigError(f"{field_name} must be a boolean")


def _safe_id(value: str, field_name: str) -> str:
    if not value:
        raise ProviderConfigError(f"{field_name} is required")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
    if value[0] not in allowed or any(char not in allowed for char in value):
        raise ProviderConfigError(f"{field_name} must contain only letters, numbers, dot, underscore, or dash")
    return value


def _usable_api_key(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if text.lower().startswith("replace-with-"):
        return None
    return text
