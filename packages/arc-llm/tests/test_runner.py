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


def test_resolve_llm_config_uses_host_and_default_model(tmp_path):
    config = resolve_llm_config(env={**no_provider_config(tmp_path), "ARC_AGENT_HOST": "codex"}, process_chain=[])
    assert config.provider == "codex-cli"
    assert config.model == "gpt-5.4-mini"
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

    assert result == "gpt-5.4-mini:prompt"
