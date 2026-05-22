from arc_llm_worker.providers.select import select_provider


def test_select_provider_explicit():
    assert select_provider("manual", env={"ARC_AGENT_HOST": "codex"}, process_chain=[]).name == "manual"


def test_select_provider_auto_uses_agent_host_env():
    assert select_provider("auto", env={"ARC_AGENT_HOST": "claude-code"}, process_chain=[]).name == "claude-cli"


def test_select_provider_auto_falls_back_to_manual():
    assert select_provider("auto", env={}, process_chain=[]).name == "manual"
