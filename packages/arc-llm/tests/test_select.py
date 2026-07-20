from arc_llm.providers.select import select_provider
from arc_llm.providers.registry import get_provider_spec, provider_diagnostic


def test_select_provider_explicit_builtin():
    assert select_provider("manual", env={"ARC_AGENT_HOST": "codex"}, process_chain=[]).name == "manual"


def test_select_provider_auto_uses_agent_host_env():
    assert (
        select_provider(
            "auto",
            env={"ARC_AGENT_HOST": "claude-code"},
            process_chain=[],
        ).name
        == "claude-cli"
    )


def test_select_provider_auto_falls_back_to_manual():
    assert select_provider("auto", env={}, process_chain=[]).name == "manual"


def test_kimi_provider_spec_records_experimental_capabilities_and_risks():
    spec = get_provider_spec("kimi-code-cli")

    assert spec.experimental is True
    assert spec.supports_sessions is True
    assert spec.supports_usage is False
    assert spec.supports_native_schema is False
    assert spec.provider_side_persistence is True
    assert "kimi_code_cli.experimental" in spec.warning_codes
    assert spec.risk_warning and "may access the network" in spec.risk_warning


def test_kimi_provider_diagnostic_reports_only_existing_risk_paths(tmp_path):
    user_home = tmp_path / "home"
    kimi_home = tmp_path / "kimi-home"
    project = tmp_path / "project"
    (project / ".git").mkdir(parents=True)
    (project / ".kimi-code" / "skills").mkdir(parents=True)
    (project / ".kimi-code" / "mcp.json").write_text("secret-value", encoding="utf-8")
    (kimi_home / "plugins").mkdir(parents=True)
    (kimi_home / "config.toml").write_text("api_key = 'secret'", encoding="utf-8")
    (user_home / ".agents").mkdir(parents=True)
    (user_home / ".agents" / "AGENTS.md").write_text("instructions", encoding="utf-8")

    diagnostic = provider_diagnostic(
        "kimi-code-cli",
        env={"HOME": str(user_home), "KIMI_CODE_HOME": str(kimi_home)},
        cwd=project,
    )

    risks = diagnostic["risks"]
    assert {item["category"] for item in risks} == {
        "configuration",
        "instructions",
        "mcp",
        "plugins",
        "skills",
    }
    assert all(set(item) == {"category", "path"} for item in risks)
    assert "secret" not in repr(diagnostic)


def test_non_kimi_provider_diagnostic_has_no_inherited_risk_scan():
    diagnostic = provider_diagnostic("codex-cli", env={}, cwd="/")

    assert diagnostic["experimental"] is False
    assert diagnostic["risks"] == []
