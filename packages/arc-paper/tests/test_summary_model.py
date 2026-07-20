from arc_paper.summary.model import resolve_summary_model


def test_codex_default_summary_model_is_low():
    assert resolve_summary_model("codex-cli", env={}) == "gpt-5.6-luna"


def test_claude_default_summary_model_is_low():
    assert resolve_summary_model("claude-cli", env={}) == "haiku"


def test_kimi_default_summary_model_uses_provider_default_alias():
    assert resolve_summary_model("kimi-code-cli", env={}) == "default_model"


def test_kimi_default_summary_model_honors_low_tier_mapping():
    env = {"ARC_LLM_KIMI_LOW_MODEL": "kimi-low-custom"}

    assert resolve_summary_model("kimi-code-cli", env=env) == "kimi-low-custom"


def test_summary_model_env_vars_do_not_select_model():
    env = {
        "ARC_LLM_MODEL": "shared",
        "ARC_CODEX_MODEL": "codex-specific",
    }

    assert resolve_summary_model("codex-cli", "explicit", env=env) == "explicit"
    assert resolve_summary_model("codex-cli", env=env) == "gpt-5.6-luna"
    assert resolve_summary_model("claude-cli", env=env) == "haiku"
