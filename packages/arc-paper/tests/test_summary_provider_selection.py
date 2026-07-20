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


def test_select_summary_provider_auto_uses_kimi_code_host(monkeypatch):
    prompt_provider = type("PromptProvider", (), {"name": "kimi-code-cli"})()
    monkeypatch.setattr(select_module, "select_prompt_provider", lambda *args, **kwargs: prompt_provider)

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


def test_select_summary_provider_passes_env_to_kimi_prompt_provider(monkeypatch):
    env = {"ARC_KIMI_BIN": "/opt/kimi/bin/kimi", "CUSTOM_SETTING": "value"}
    captured = {}
    prompt_provider = type("PromptProvider", (), {"name": "kimi-code-cli"})()

    def select_prompt_provider(name, **kwargs):
        captured["name"] = name
        captured.update(kwargs)
        return prompt_provider

    monkeypatch.setattr(select_module, "select_prompt_provider", select_prompt_provider)

    provider = select_summary_provider("kimi-code-cli", env=env, process_chain=[])

    assert isinstance(provider, PromptProviderSummaryAdapter)
    assert provider.prompt_provider is prompt_provider
    assert provider.env is env
    assert captured == {"name": "kimi-code-cli", "env": env, "process_chain": []}
