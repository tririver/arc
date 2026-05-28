from __future__ import annotations

import json

import pytest

from arc_llm.host import select_llm_provider
from arc_llm.model import resolve_model
from arc_llm.providers.config import ProviderConfigError, load_provider_config, provider_config_path, usable_configured_providers
from arc_llm.providers.select import select_provider


def write_config(tmp_path, providers, *, default=None, auto_provider_priority=None):
    path = tmp_path / "llm-providers.json"
    write_payload(path, providers, default=default, auto_provider_priority=auto_provider_priority)
    return str(path)


def write_payload(path, providers, *, default=None, auto_provider_priority=None):
    payload = {
        "schema_version": "arc.llm.providers.v1",
        "providers": providers,
    }
    if default is not None:
        payload["default"] = default
    if auto_provider_priority is not None:
        payload["auto_provider_priority"] = auto_provider_priority
    path.write_text(json.dumps(payload), encoding="utf-8")


def provider_payload(provider_id):
    return [
        {
            "id": provider_id,
            "type": "openai-compatible",
            "base_url": f"https://{provider_id}.example/v1",
            "api_key": "secret-value",
        }
    ]


def test_provider_config_path_defaults_to_project_local_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert provider_config_path(env={}) == tmp_path / "llm-providers.json"


def test_load_provider_config_uses_project_local_default_before_user_config(tmp_path, monkeypatch):
    project = tmp_path / "project"
    home = tmp_path / "home"
    project.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setenv("HOME", str(home))
    user_config = home / ".config" / "arc" / "llm-providers.json"
    user_config.parent.mkdir(parents=True)
    write_payload(user_config, provider_payload("user-config"))
    project_config = project / "llm-providers.json"
    write_payload(project_config, provider_payload("project-config"))

    config = load_provider_config(env={})

    assert config.path == str(project_config)
    assert [provider.id for provider in config.providers] == ["project-config"]


def test_load_provider_config_falls_back_to_user_default_when_project_file_is_missing(tmp_path, monkeypatch):
    project = tmp_path / "project"
    home = tmp_path / "home"
    project.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setenv("HOME", str(home))
    user_config = home / ".config" / "arc" / "llm-providers.json"
    user_config.parent.mkdir(parents=True)
    write_payload(user_config, provider_payload("user-config"))

    config = load_provider_config(env={})

    assert config.path == str(user_config)
    assert [provider.id for provider in config.providers] == ["user-config"]


def test_load_provider_config_supports_file_api_key(tmp_path):
    path = write_config(
        tmp_path,
        [
            {
                "id": "deepseek",
                "type": "openai-compatible",
                "base_url": "https://api.deepseek.example/v1",
                "api_key": "secret-value",
                "models": {"medium": "deepseek-chat", "high": "deepseek-reasoner"},
                "json_mode": "json_schema",
            }
        ],
    )

    config = load_provider_config(env={"ARC_LLM_PROVIDER_CONFIG": path})

    assert config.path == path
    assert not hasattr(config, "default")
    assert not hasattr(config, "auto_provider_priority")
    assert config.providers[0].id == "deepseek"
    assert config.providers[0].api_key == "secret-value"


@pytest.mark.parametrize("legacy_field", ["default", "auto_provider_priority"])
def test_load_provider_config_rejects_legacy_selection_fields(tmp_path, legacy_field):
    kwargs = {legacy_field: "deepseek" if legacy_field == "default" else "configured-first"}
    path = write_config(
        tmp_path,
        [
            {
                "id": "deepseek",
                "type": "openai-compatible",
                "base_url": "https://api.deepseek.example/v1",
                "api_key": "secret-value",
            }
        ],
        **kwargs,
    )

    with pytest.raises(ProviderConfigError, match="run selection fields"):
        load_provider_config(env={"ARC_LLM_PROVIDER_CONFIG": path})


def test_provider_config_rejects_api_key_env(tmp_path):
    path = write_config(
        tmp_path,
        [
            {
                "id": "deepseek",
                "type": "openai-compatible",
                "base_url": "https://api.deepseek.example/v1",
                "api_key_env": "DEEPSEEK_API_KEY",
            }
        ],
    )

    with pytest.raises(ProviderConfigError, match="api_key"):
        load_provider_config(env={"ARC_LLM_PROVIDER_CONFIG": path})


def test_usable_configured_providers_require_file_api_key_unless_optional(tmp_path):
    path = write_config(
        tmp_path,
        [
            {
                "id": "deepseek",
                "type": "openai-compatible",
                "base_url": "https://api.deepseek.example/v1",
                "api_key": "replace-with-your-deepseek-api-key",
            },
            {
                "id": "ollama",
                "type": "openai-compatible",
                "base_url": "http://127.0.0.1:11434/v1",
                "api_key_optional": True,
                "models": {"medium": "llama3.1"},
                "json_mode": "json_object",
            },
        ],
    )

    assert [item.id for item in usable_configured_providers(env={"ARC_LLM_PROVIDER_CONFIG": path})] == ["ollama"]
    configured_path = write_config(
        tmp_path,
        [
            {
                "id": "deepseek",
                "type": "openai-compatible",
                "base_url": "https://api.deepseek.example/v1",
                "api_key": "secret-value",
            },
            {
                "id": "ollama",
                "type": "openai-compatible",
                "base_url": "http://127.0.0.1:11434/v1",
                "api_key_optional": True,
            },
        ],
    )
    assert [item.id for item in usable_configured_providers(env={"ARC_LLM_PROVIDER_CONFIG": configured_path})] == [
        "deepseek",
        "ollama",
    ]


def test_usable_configured_providers_treat_inline_api_key_as_key_present(tmp_path):
    path = write_config(
        tmp_path,
        [
            {
                "id": "deepseek",
                "type": "openai-compatible",
                "base_url": "https://api.deepseek.example/v1",
                "api_key": "secret-value",
            }
        ],
    )

    assert [item.id for item in usable_configured_providers(env={"ARC_LLM_PROVIDER_CONFIG": path})] == ["deepseek"]


def test_auto_provider_selection_prefers_native_host_before_configured_provider(tmp_path):
    path = write_config(
        tmp_path,
        [
            {
                "id": "ollama",
                "type": "openai-compatible",
                "base_url": "http://127.0.0.1:11434/v1",
                "api_key_optional": True,
            },
            {
                "id": "deepseek",
                "type": "openai-compatible",
                "base_url": "https://api.deepseek.example/v1",
                "api_key": "secret-value",
            },
        ],
    )

    selected = select_llm_provider(
        env={"ARC_LLM_PROVIDER_CONFIG": path},
        process_chain=["codex exec"],
    )

    assert selected.provider == "codex-cli"
    assert selected.host.host == "codex"
    assert selected.signals == ["parent:codex exec"]


def test_auto_provider_selection_rejects_configured_first_priority(tmp_path):
    path = write_config(
        tmp_path,
        [
            {
                "id": "ollama",
                "type": "openai-compatible",
                "base_url": "http://127.0.0.1:11434/v1",
                "api_key_optional": True,
            },
            {
                "id": "deepseek",
                "type": "openai-compatible",
                "base_url": "https://api.deepseek.example/v1",
                "api_key": "secret-value",
            },
        ],
        default="deepseek",
        auto_provider_priority="configured-first",
    )

    with pytest.raises(ProviderConfigError, match="run selection fields"):
        usable_configured_providers(env={"ARC_LLM_PROVIDER_CONFIG": path})


def test_auto_provider_selection_uses_usable_configured_provider_when_no_host(tmp_path):
    path = write_config(
        tmp_path,
        [
            {
                "id": "openrouter",
                "type": "openai-compatible",
                "base_url": "https://openrouter.example/api/v1",
                "api_key": "replace-with-your-openrouter-api-key",
            },
            {
                "id": "ollama",
                "type": "openai-compatible",
                "base_url": "http://127.0.0.1:11434/v1",
                "api_key_optional": True,
            },
        ],
    )

    selected = select_llm_provider(env={"ARC_LLM_PROVIDER_CONFIG": path}, process_chain=[])

    assert selected.provider == "ollama"


def test_auto_provider_selection_prefers_file_api_key_over_earlier_optional_provider(tmp_path):
    path = write_config(
        tmp_path,
        [
            {
                "id": "ollama",
                "type": "openai-compatible",
                "base_url": "http://127.0.0.1:11434/v1",
                "api_key_optional": True,
            },
            {
                "id": "deepseek",
                "type": "openai-compatible",
                "base_url": "https://api.deepseek.example/v1",
                "api_key": "secret-value",
            },
        ],
    )

    selected = select_llm_provider(
        env={"ARC_LLM_PROVIDER_CONFIG": path},
        process_chain=[],
    )

    assert selected.provider == "deepseek"


def test_auto_provider_selection_falls_back_to_host_when_no_configured_provider_is_usable(tmp_path):
    path = write_config(
        tmp_path,
        [
            {
                "id": "deepseek",
                "type": "openai-compatible",
                "base_url": "https://api.deepseek.example/v1",
                "api_key": "replace-with-your-deepseek-api-key",
            }
        ],
    )

    selected = select_llm_provider(env={"ARC_LLM_PROVIDER_CONFIG": path}, process_chain=["claude -p"])

    assert selected.provider == "claude-cli"


def test_explicit_provider_still_wins_over_configured_auto(tmp_path):
    path = write_config(
        tmp_path,
        [
            {
                "id": "deepseek",
                "type": "openai-compatible",
                "base_url": "https://api.deepseek.example/v1",
                "api_key": "secret-value",
            }
        ],
    )

    selected = select_llm_provider(
        explicit_provider="manual",
        env={"ARC_LLM_PROVIDER_CONFIG": path},
        process_chain=["codex exec"],
    )

    assert selected.provider == "manual"
    assert selected.signals == ["explicit"]


def test_configured_provider_model_resolution_uses_tiers_only(tmp_path):
    path = write_config(
        tmp_path,
        [
            {
                "id": "deepseek",
                "type": "openai-compatible",
                "base_url": "https://api.deepseek.example/v1",
                "api_key": "secret-value",
                "models": {
                    "low": "deepseek-chat",
                    "medium": "deepseek-chat",
                    "high": "deepseek-reasoner",
                },
            }
        ],
    )
    env = {"ARC_LLM_PROVIDER_CONFIG": path}

    assert resolve_model("deepseek", model_tier="high", env=env) == "deepseek-reasoner"
    assert resolve_model("deepseek", env=env) == "deepseek-chat"
    assert resolve_model("deepseek", "explicit", env=env) == "explicit"


def test_configured_provider_rejects_legacy_default_model(tmp_path):
    path = write_config(
        tmp_path,
        [
            {
                "id": "openaiish",
                "type": "openai-compatible",
                "base_url": "https://api.openaiish.example/v1",
                "api_key": "secret-value",
                "models": {
                    "default": "gpt-5.4-mini",
                },
            }
        ],
    )

    with pytest.raises(ProviderConfigError, match="models keys"):
        load_provider_config(env={"ARC_LLM_PROVIDER_CONFIG": path})


def test_configured_provider_default_uses_medium_tier_when_available(tmp_path):
    path = write_config(
        tmp_path,
        [
            {
                "id": "openaiish",
                "type": "openai-compatible",
                "base_url": "https://api.openaiish.example/v1",
                "api_key": "secret-value",
                "models": {
                    "low": "gpt-5.3-codex-spark",
                    "medium": "gpt-5.4",
                    "high": "gpt-5.5",
                },
            }
        ],
    )

    assert resolve_model("openaiish", env={"ARC_LLM_PROVIDER_CONFIG": path}) == "gpt-5.4"


def test_select_provider_can_return_configured_provider(tmp_path):
    path = write_config(
        tmp_path,
        [
            {
                "id": "deepseek",
                "type": "openai-compatible",
                "base_url": "https://api.deepseek.example/v1",
                "api_key": "secret-value",
            }
        ],
    )

    provider = select_provider(
        "deepseek",
        env={"ARC_LLM_PROVIDER_CONFIG": path},
        process_chain=[],
    )

    assert provider.name == "deepseek"
