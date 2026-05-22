from arc_llm.host import detect_host, select_llm_provider


def test_env_host_has_priority():
    detected = detect_host(env={"ARC_AGENT_HOST": "codex"}, process_chain=[])
    assert detected.host == "codex"
    assert detected.confidence == 1.0
    assert detected.signals == ["env:ARC_AGENT_HOST=codex"]


def test_parent_process_detects_codex():
    detected = detect_host(
        env={},
        process_chain=[
            "/bin/bash -c arc-paper",
            "/usr/local/lib/node_modules/@openai/codex/bin/codex",
        ],
    )
    assert detected.host == "codex"
    assert detected.confidence >= 0.8


def test_parent_process_detects_claude_code():
    detected = detect_host(
        env={},
        process_chain=[
            "/bin/bash -c source /home/user/.claude/shell-snapshots/snapshot.sh",
            "claude -p --bare",
        ],
    )
    assert detected.host == "claude-code"
    assert detected.confidence >= 0.8


def test_provider_selection_prefers_env():
    provider = select_llm_provider(
        env={"ARC_AGENT_HOST": "codex", "ARC_LLM_PROVIDER": "openai"},
        process_chain=[],
    )
    assert provider.provider == "openai"
    assert provider.host.host == "codex"


def test_provider_selection_uses_detected_host():
    provider = select_llm_provider(env={}, process_chain=["claude -p"])
    assert provider.provider == "claude-cli"
