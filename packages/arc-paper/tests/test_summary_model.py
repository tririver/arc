from arc_paper.summary.model import resolve_summary_model


def test_codex_default_summary_model_is_medium():
    assert resolve_summary_model("codex-cli", env={}) == "gpt-5.4"


def test_claude_default_summary_model_is_medium():
    assert resolve_summary_model("claude-cli", env={}) == "sonnet"


def test_summary_model_env_vars_do_not_select_model():
    env = {
        "ARC_LLM_MODEL": "shared",
        "ARC_CODEX_MODEL": "codex-specific",
    }

    assert resolve_summary_model("codex-cli", "explicit", env=env) == "explicit"
    assert resolve_summary_model("codex-cli", env=env) == "gpt-5.4"
    assert resolve_summary_model("claude-cli", env=env) == "sonnet"
