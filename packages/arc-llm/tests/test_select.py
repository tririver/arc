from arc_llm.providers.select import select_provider


def no_provider_config(tmp_path):
    return {"ARC_LLM_PROVIDER_CONFIG": str(tmp_path / "missing.json")}


def test_select_provider_explicit():
    assert select_provider("manual", env={"ARC_AGENT_HOST": "codex"}, process_chain=[]).name == "manual"


def test_select_provider_auto_uses_agent_host_env(tmp_path):
    assert (
        select_provider(
            "auto",
            env={**no_provider_config(tmp_path), "ARC_AGENT_HOST": "claude-code"},
            process_chain=[],
        ).name
        == "claude-cli"
    )


def test_select_provider_auto_falls_back_to_manual(tmp_path):
    assert select_provider("auto", env=no_provider_config(tmp_path), process_chain=[]).name == "manual"
