from arc_paper.summary.providers.select import select_summary_provider


def test_select_summary_provider_explicit():
    provider = select_summary_provider("manual", env={"ARC_AGENT_HOST": "codex"}, process_chain=[])
    assert provider.name == "manual"


def test_select_summary_provider_auto_uses_agent_host_env():
    provider = select_summary_provider(
        "auto",
        env={"ARC_AGENT_HOST": "claude-code"},
        process_chain=[],
    )
    assert provider.name == "claude-cli"


def test_select_summary_provider_auto_falls_back_to_manual():
    provider = select_summary_provider("auto", env={}, process_chain=[])
    assert provider.name == "manual"


def test_select_summary_provider_passes_env_to_codex_native_provider():
    env = {"ARC_CODEX_SANDBOX": "workspace-write", "CUSTOM_SETTING": "value"}

    provider = select_summary_provider("codex-cli", env=env, process_chain=[])

    assert provider.prompt_provider.env is env


def test_select_summary_provider_passes_env_to_claude_native_provider():
    env = {"ARC_CLAUDE_EFFORT": "medium", "CUSTOM_SETTING": "value"}

    provider = select_summary_provider("claude-cli", env=env, process_chain=[])

    assert provider.prompt_provider.env is env
