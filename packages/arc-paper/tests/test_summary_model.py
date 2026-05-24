from arc_paper.summary.model import resolve_summary_model


def test_codex_default_summary_model_is_mini():
    assert resolve_summary_model("codex-cli", env={}) == "gpt-5.4-mini"


def test_claude_default_summary_model_is_haiku():
    assert resolve_summary_model("claude-cli", env={}) == "haiku"


def test_summary_model_precedence():
    env = {
        "ARC_LLM_MODEL": "shared",
        "ARC_CODEX_MODEL": "codex-specific",
    }

    assert resolve_summary_model("codex-cli", "explicit", env=env) == "explicit"
    assert resolve_summary_model("codex-cli", env=env) == "codex-specific"
    assert resolve_summary_model("claude-cli", env=env) == "shared"
