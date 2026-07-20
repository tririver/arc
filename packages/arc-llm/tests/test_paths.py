from arc_llm.paths import arc_home, llm_cache_root, llm_tmp_root, schema_cache_root
from arc_llm.providers.claude_cli import _arc_only_mcp_config_path
from arc_llm.providers.codex_cli import _default_schema_cache_dir


def test_arc_home_layout_drives_llm_cache_tmp_schema_and_claude_mcp(tmp_path):
    env = {"ARC_HOME": str(tmp_path / "arc")}

    assert arc_home(env) == tmp_path / "arc"
    assert llm_cache_root(env) == tmp_path / "arc/cache/arc-llm"
    assert llm_tmp_root(env) == tmp_path / "arc/tmp/arc-llm"
    assert schema_cache_root(env) == tmp_path / "arc/cache/arc-llm/schemas"
    assert _default_schema_cache_dir(env) == tmp_path / "arc/cache/arc-llm/schemas"
    assert _arc_only_mcp_config_path(env) == tmp_path / "arc/cache/arc-llm/mcp/arc-claude-mcp.json"


def test_arc_llm_specific_paths_override_arc_home(tmp_path):
    env = {
        "ARC_HOME": str(tmp_path / "arc"),
        "ARC_LLM_CACHE": str(tmp_path / "cache"),
        "ARC_LLM_TMP_DIR": str(tmp_path / "tmp"),
        "ARC_LLM_SCHEMA_CACHE_DIR": str(tmp_path / "schemas"),
        "ARC_CLAUDE_ARC_MCP_CONFIG_PATH": str(tmp_path / "claude.json"),
    }

    assert llm_cache_root(env) == tmp_path / "cache"
    assert llm_tmp_root(env) == tmp_path / "tmp"
    assert schema_cache_root(env) == tmp_path / "schemas"
    assert _arc_only_mcp_config_path(env) == tmp_path / "claude.json"


def test_arc_home_falls_back_to_xdg_data_then_system_home(tmp_path):
    assert arc_home({"XDG_DATA_HOME": str(tmp_path / "data")}) == tmp_path / "data/arc"
    assert arc_home({"HOME": str(tmp_path / "home")}) == tmp_path / "home/.local/share/arc"
    assert schema_cache_root({"HOME": str(tmp_path / "home")}) == (
        tmp_path / "home/.local/share/arc/cache/arc-llm/schemas"
    )
