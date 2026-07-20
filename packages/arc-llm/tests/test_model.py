import pytest

from arc_llm.model import (
    DEFAULT_MODEL_TIER,
    VALID_MODEL_TIERS,
    ModelTierError,
    KIMI_TIER_UNMAPPED_WARNING,
    reasoning_effort_for_model_tier,
    resolve_model,
    resolve_model_with_warnings,
)


def test_explicit_model_wins():
    assert resolve_model("codex-cli", "custom", env={"ARC_CODEX_MODEL": "ignored"}) == "custom"


def test_model_env_vars_do_not_select_model():
    env = {
        "ARC_CODEX_MODEL": "codex-env",
        "ARC_LLM_MODEL": "generic",
        "ARC_CODEX_MODEL_TIER": "low",
        "ARC_LLM_MODEL_TIER": "high",
    }
    assert resolve_model("codex-cli", env=env) == "gpt-5.6-luna"


def test_defaults_use_medium_tier():
    assert DEFAULT_MODEL_TIER == "medium"
    assert DEFAULT_MODEL_TIER != "max"
    assert resolve_model("codex-cli", env={}) == "gpt-5.6-luna"
    assert resolve_model("claude-cli", env={}) == "sonnet"
    assert resolve_model("manual", env={}) is None


def test_model_tier_resolves_provider_specific_model_when_no_exact_model_is_set():
    assert VALID_MODEL_TIERS == {"low", "medium", "high", "max"}
    assert resolve_model("codex-cli", model_tier="max", env={}) == "gpt-5.6-sol"
    assert resolve_model("codex-cli", model_tier="high", env={}) == "gpt-5.6-sol"
    assert resolve_model("codex-cli", model_tier="medium", env={}) == "gpt-5.6-luna"
    assert resolve_model("codex-cli", model_tier="low", env={}) == "gpt-5.6-luna"
    assert resolve_model("claude-cli", model_tier="high", env={}) == "opus"


def test_codex_model_tiers_resolve_requested_reasoning_effort():
    assert reasoning_effort_for_model_tier("codex-cli", "low") == "medium"
    assert reasoning_effort_for_model_tier("codex-cli", "medium") == "high"
    assert reasoning_effort_for_model_tier("codex-cli", "high") == "high"
    assert reasoning_effort_for_model_tier("codex-cli", "max") == "max"


def test_model_tier_aliases_can_be_overridden_by_env():
    env = {
        "ARC_LLM_CODEX_MAX_MODEL": "codex-max-custom",
        "ARC_LLM_CODEX_HIGH_MODEL": "codex-high-custom",
        "ARC_LLM_CLAUDE_LOW_MODEL": "claude-low-custom",
    }

    assert resolve_model("codex-cli", model_tier="max", env=env) == "codex-max-custom"
    assert resolve_model("codex-cli", model_tier="high", env=env) == "codex-high-custom"
    assert resolve_model("claude-cli", model_tier="low", env=env) == "claude-low-custom"


def test_kimi_model_tier_uses_only_explicit_alias_or_default_model():
    mapped = resolve_model_with_warnings(
        "kimi-code-cli",
        model_tier="high",
        env={"ARC_LLM_KIMI_HIGH_MODEL": "kimi-high-alias"},
    )
    fallback = resolve_model_with_warnings("kimi-code-cli", model_tier="high", env={})

    assert mapped.model == "kimi-high-alias"
    assert mapped.warnings == ()
    assert fallback.model == "default_model"
    assert fallback.warnings == (KIMI_TIER_UNMAPPED_WARNING,)


def test_kimi_exact_model_does_not_report_unmapped_tier():
    resolution = resolve_model_with_warnings("kimi-code-cli", "kimi-explicit", model_tier="max", env={})

    assert resolution.model == "kimi-explicit"
    assert resolution.warnings == ()


def test_model_tier_aliases_use_process_env_when_env_not_passed(monkeypatch):
    monkeypatch.setenv("ARC_LLM_CODEX_HIGH_MODEL", "codex-high-from-process-env")

    assert resolve_model("codex-cli", model_tier="high") == "codex-high-from-process-env"


def test_explicit_model_tier_is_only_tier_selector():
    assert resolve_model("codex-cli", model_tier="high", env={"ARC_LLM_MODEL_TIER": "low"}) == "gpt-5.6-sol"


def test_unknown_model_tier_fails():
    with pytest.raises(ModelTierError, match="model_tier must be one of"):
        resolve_model("codex-cli", model_tier="strong", env={})


def test_old_xhigh_model_tier_is_rejected():
    with pytest.raises(ModelTierError, match="low, medium, high, max"):
        resolve_model("codex-cli", model_tier="xhigh", env={})
