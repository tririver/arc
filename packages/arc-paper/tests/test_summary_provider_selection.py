from arc_paper.summary.providers import select as select_module
from arc_paper.summary.providers.prompt import PromptProviderSummaryAdapter
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


def test_select_summary_provider_auto_uses_kimi_code_host():
    provider = select_summary_provider(
        "auto",
        env={"ARC_AGENT_HOST": "kimi-code"},
        process_chain=[],
    )

    assert isinstance(provider, PromptProviderSummaryAdapter)
    assert provider.name == "kimi-code-cli"


def test_select_summary_provider_auto_falls_back_to_manual():
    provider = select_summary_provider("auto", env={}, process_chain=[])
    assert provider.name == "manual"


def test_select_summary_provider_passes_env_to_codex_native_provider():
    env = {"ARC_CODEX_SANDBOX": "workspace-write", "CUSTOM_SETTING": "value"}

    provider = select_summary_provider("codex-cli", env=env, process_chain=[])

    assert provider.env is env
    assert provider.prompt_provider is None


def test_select_summary_provider_passes_env_to_claude_native_provider():
    env = {"ARC_CLAUDE_EFFORT": "medium", "CUSTOM_SETTING": "value"}

    provider = select_summary_provider("claude-cli", env=env, process_chain=[])

    assert provider.env is env
    assert provider.prompt_provider is None


def test_select_summary_provider_passes_env_to_kimi_prompt_provider():
    env = {"ARC_KIMI_BIN": "/opt/kimi/bin/kimi", "CUSTOM_SETTING": "value"}

    provider = select_summary_provider("kimi-code-cli", env=env, process_chain=[])

    assert isinstance(provider, PromptProviderSummaryAdapter)
    assert provider.prompt_provider is None
    assert provider.env is env
    assert provider.process_chain == []
