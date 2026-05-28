from arc_paper.summary.providers.select import select_summary_provider


def no_provider_config(tmp_path):
    return {"ARC_LLM_PROVIDER_CONFIG": str(tmp_path / "missing.json")}


def test_select_summary_provider_explicit():
    provider = select_summary_provider("manual", env={"ARC_AGENT_HOST": "codex"}, process_chain=[])
    assert provider.name == "manual"


def test_select_summary_provider_auto_uses_agent_host_env(tmp_path):
    provider = select_summary_provider(
        "auto",
        env={**no_provider_config(tmp_path), "ARC_AGENT_HOST": "claude-code"},
        process_chain=[],
    )
    assert provider.name == "claude-cli"


def test_select_summary_provider_auto_falls_back_to_manual(tmp_path):
    provider = select_summary_provider("auto", env=no_provider_config(tmp_path), process_chain=[])
    assert provider.name == "manual"


def test_select_summary_provider_can_wrap_configured_arc_llm_provider(tmp_path, monkeypatch):
    config_path = tmp_path / "llm-providers.json"
    config_path.write_text(
        """
        {
          "schema_version": "arc.llm.providers.v1",
          "providers": [
            {
              "id": "deepseek",
              "type": "openai-compatible",
              "base_url": "https://api.deepseek.example/v1",
              "api_key": "secret-value",
              "models": {"medium": "deepseek-chat"}
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    provider = select_summary_provider(
        "deepseek",
        env={"ARC_LLM_PROVIDER_CONFIG": str(config_path)},
        process_chain=[],
    )

    assert provider.name == "deepseek"
    assert provider.prompt_provider.name == "deepseek"
