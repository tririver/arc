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
