from arc_llm.model import resolve_model


def test_explicit_model_wins():
    assert resolve_model("codex-cli", "custom", env={"ARC_CODEX_MODEL": "ignored"}) == "custom"


def test_provider_specific_env_wins():
    assert resolve_model("codex-cli", env={"ARC_CODEX_MODEL": "codex-env", "ARC_LLM_MODEL": "generic"}) == "codex-env"


def test_generic_env_is_fallback():
    assert resolve_model("claude-cli", env={"ARC_LLM_MODEL": "generic"}) == "generic"


def test_fast_defaults():
    assert resolve_model("codex-cli", env={}) == "gpt-5.4-mini"
    assert resolve_model("claude-cli", env={}) == "haiku"
    assert resolve_model("manual", env={}) is None
