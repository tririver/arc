import json

import pytest

from arc_llm import runner
from arc_llm.runner import resolve_llm_config, run_json, run_text


def no_provider_config(tmp_path):
    return {"ARC_LLM_PROVIDER_CONFIG": str(tmp_path / "missing.json")}


class FakeProvider:
    name = "codex-cli"

    def generate_json(self, prompt, *, schema=None, model=None):
        return {"prompt": prompt, "schema": schema, "model": model}

    def generate_text(self, prompt, *, model=None):
        return f"{model}:{prompt}"


class FlakyJsonProvider:
    def __init__(self, *, name, failures_before_success=None, result=None):
        self.name = name
        self.failures_before_success = failures_before_success
        self.result = result or {"ok": True}
        self.attempts = 0

    def generate_json(self, prompt, *, schema=None, model=None):
        self.attempts += 1
        if self.failures_before_success is None or self.attempts <= self.failures_before_success:
            raise RuntimeError(f"{self.name} failed")
        return {**self.result, "model": model}


class FlakyTextProvider:
    def __init__(self, *, name, failures_before_success=None):
        self.name = name
        self.failures_before_success = failures_before_success
        self.attempts = 0

    def generate_text(self, prompt, *, model=None):
        self.attempts += 1
        if self.failures_before_success is None or self.attempts <= self.failures_before_success:
            raise RuntimeError(f"{self.name} failed")
        return f"{self.name}:{model}:{prompt}"


def test_resolve_llm_config_uses_host_and_default_model(tmp_path):
    config = resolve_llm_config(env={**no_provider_config(tmp_path), "ARC_AGENT_HOST": "codex"}, process_chain=[])
    assert config.provider == "codex-cli"
    assert config.model == "gpt-5.4"
    assert config.host.host == "codex"


def test_run_json_uses_selected_provider_and_model(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "select_provider", lambda provider, **kwargs: FakeProvider())

    result = run_json(
        "prompt",
        schema={"type": "object"},
        env={**no_provider_config(tmp_path), "ARC_AGENT_HOST": "codex", "ARC_CODEX_MODEL": "fast"},
        process_chain=[],
    )

    assert result["prompt"] == "prompt"
    assert result["schema"] == {"type": "object"}
    assert result["model"] == "fast"


def test_run_json_uses_model_tier_when_exact_model_is_not_set(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "select_provider", lambda provider, **kwargs: FakeProvider())

    result = run_json(
        "prompt",
        schema={"type": "object"},
        model_tier="high",
        env={**no_provider_config(tmp_path), "ARC_AGENT_HOST": "codex"},
        process_chain=[],
    )

    assert result["model"] == "gpt-5.5"


def test_run_text_uses_selected_provider_and_model(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "select_provider", lambda provider, **kwargs: FakeProvider())

    result = run_text("prompt", env={**no_provider_config(tmp_path), "ARC_AGENT_HOST": "codex"}, process_chain=[])

    assert result == "gpt-5.4:prompt"


def test_run_text_retries_selected_provider_twice_before_success(tmp_path, monkeypatch):
    flaky = FlakyTextProvider(name="codex-cli", failures_before_success=2)
    monkeypatch.setattr(runner, "select_provider", lambda provider, **kwargs: flaky)

    result = run_text(
        "prompt",
        env={**no_provider_config(tmp_path), "ARC_AGENT_HOST": "codex"},
        process_chain=[],
    )

    assert result == "codex-cli:gpt-5.4:prompt"
    assert flaky.attempts == 3


def test_run_json_retries_selected_provider_twice_before_success(tmp_path, monkeypatch):
    flaky = FlakyJsonProvider(name="codex-cli", failures_before_success=2, result={"provider": "codex-cli"})
    monkeypatch.setattr(runner, "select_provider", lambda provider, **kwargs: flaky)

    result = run_json(
        "prompt",
        env={**no_provider_config(tmp_path), "ARC_AGENT_HOST": "codex"},
        process_chain=[],
    )

    assert result["provider"] == "codex-cli"
    assert flaky.attempts == 3


def test_run_json_auto_falls_back_to_configured_provider_after_retries(tmp_path, monkeypatch):
    provider_config = tmp_path / "llm-providers.json"
    provider_config.write_text(
        json.dumps(
            {
                "schema_version": "arc.llm.providers.v1",
                "providers": [
                    {
                        "id": "deepseek",
                        "type": "openai-compatible",
                        "base_url": "https://deepseek.example/v1",
                        "api_key": "secret-value",
                        "models": {"default": "deepseek-chat"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    codex = FlakyJsonProvider(name="codex-cli")
    deepseek = FlakyJsonProvider(name="deepseek", failures_before_success=0, result={"provider": "deepseek"})
    providers = {"codex-cli": codex, "deepseek": deepseek}
    monkeypatch.setattr(runner, "select_provider", lambda provider, **kwargs: providers[provider])

    result = run_json(
        "prompt",
        env={"ARC_AGENT_HOST": "codex", "ARC_LLM_PROVIDER_CONFIG": str(provider_config)},
        process_chain=[],
    )

    assert result == {"provider": "deepseek", "model": "deepseek-chat"}
    assert codex.attempts == 3
    assert deepseek.attempts == 1


def test_run_json_explicit_provider_retries_without_fallback(tmp_path, monkeypatch):
    provider_config = tmp_path / "llm-providers.json"
    provider_config.write_text(
        json.dumps(
            {
                "schema_version": "arc.llm.providers.v1",
                "providers": [
                    {
                        "id": "deepseek",
                        "type": "openai-compatible",
                        "base_url": "https://deepseek.example/v1",
                        "api_key": "secret-value",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    codex = FlakyJsonProvider(name="codex-cli")
    deepseek = FlakyJsonProvider(name="deepseek", failures_before_success=0, result={"provider": "deepseek"})
    providers = {"codex-cli": codex, "deepseek": deepseek}
    monkeypatch.setattr(runner, "select_provider", lambda provider, **kwargs: providers[provider])

    with pytest.raises(RuntimeError, match="LLM task failed after 3 attempt\\(s\\) across 1 provider\\(s\\)"):
        run_json(
            "prompt",
            provider="codex-cli",
            env={"ARC_AGENT_HOST": "codex", "ARC_LLM_PROVIDER_CONFIG": str(provider_config)},
            process_chain=[],
        )

    assert codex.attempts == 3
    assert deepseek.attempts == 0
