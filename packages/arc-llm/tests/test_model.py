import pytest

from arc_llm.model import ModelTierError, resolve_model


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


def test_model_tier_resolves_provider_specific_model_when_no_exact_model_is_set():
    assert resolve_model("codex-cli", model_tier="high", env={}) == "gpt-5.5"
    assert resolve_model("codex-cli", model_tier="medium", env={}) == "gpt-5.4"
    assert resolve_model("codex-cli", model_tier="low", env={}) == "gpt-5.4-mini"
    assert resolve_model("claude-cli", model_tier="high", env={}) == "opus"


def test_model_tier_env_is_fallback_after_exact_model_env():
    assert resolve_model("codex-cli", env={"ARC_LLM_MODEL_TIER": "high"}) == "gpt-5.5"
    assert resolve_model(
        "codex-cli",
        env={"ARC_CODEX_MODEL": "exact", "ARC_LLM_MODEL_TIER": "high"},
    ) == "exact"


def test_provider_specific_model_tier_env_wins_over_generic_tier_env():
    assert resolve_model(
        "codex-cli",
        env={"ARC_CODEX_MODEL_TIER": "medium", "ARC_LLM_MODEL_TIER": "high"},
    ) == "gpt-5.4"


def test_unknown_model_tier_fails():
    with pytest.raises(ModelTierError, match="model_tier must be one of"):
        resolve_model("codex-cli", model_tier="strong", env={})
