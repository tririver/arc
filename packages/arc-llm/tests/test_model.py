import pytest

from arc_llm.model import ModelTierError, resolve_model


def test_explicit_model_wins():
    assert resolve_model("codex-cli", "custom", env={"ARC_CODEX_MODEL": "ignored"}) == "custom"


def test_model_env_vars_do_not_select_model():
    env = {
        "ARC_CODEX_MODEL": "codex-env",
        "ARC_LLM_MODEL": "generic",
        "ARC_CODEX_MODEL_TIER": "low",
        "ARC_LLM_MODEL_TIER": "high",
    }
    assert resolve_model("codex-cli", env=env) == "gpt-5.4"


def test_defaults_use_medium_tier():
    assert resolve_model("codex-cli", env={}) == "gpt-5.4"
    assert resolve_model("claude-cli", env={}) == "sonnet"
    assert resolve_model("manual", env={}) is None


def test_model_tier_resolves_provider_specific_model_when_no_exact_model_is_set():
    assert resolve_model("codex-cli", model_tier="high", env={}) == "gpt-5.5"
    assert resolve_model("codex-cli", model_tier="medium", env={}) == "gpt-5.4"
    assert resolve_model("codex-cli", model_tier="low", env={}) == "gpt-5.3-codex-spark"
    assert resolve_model("claude-cli", model_tier="high", env={}) == "opus"


def test_model_tier_aliases_can_be_overridden_by_env():
    env = {
        "ARC_LLM_CODEX_HIGH_MODEL": "codex-high-custom",
        "ARC_LLM_CLAUDE_LOW_MODEL": "claude-low-custom",
    }

    assert resolve_model("codex-cli", model_tier="high", env=env) == "codex-high-custom"
    assert resolve_model("claude-cli", model_tier="low", env=env) == "claude-low-custom"


def test_explicit_model_tier_is_only_tier_selector():
    assert resolve_model("codex-cli", model_tier="high", env={"ARC_LLM_MODEL_TIER": "low"}) == "gpt-5.5"


def test_unknown_model_tier_fails():
    with pytest.raises(ModelTierError, match="model_tier must be one of"):
        resolve_model("codex-cli", model_tier="strong", env={})
