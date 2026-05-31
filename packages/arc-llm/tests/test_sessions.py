from __future__ import annotations

import json
import threading

import pytest

from arc_llm.sessions import LLMSessionManager, runtime_fingerprint


def test_session_manager_persists_native_id_and_turns(tmp_path):
    manager = LLMSessionManager(tmp_path / "sessions")

    ref = manager.get_or_create(
        key="calculate/run/step/proposer/proposer_001",
        provider="codex-cli",
        model="gpt-5.5",
        runtime_fingerprint="fp-1",
        name="proposer",
        metadata={"step_id": "step"},
    )
    updated = manager.update_native_session_id(ref.key, "native-123")
    manager.record_turn(
        ref.key,
        call_label="round_001/proposer_001",
        prompt_sha256="prompt-sha",
        static_prefix_sha256="static-sha",
        schema_sha256="schema-sha",
        usage={"input_tokens": 10, "cached_input_tokens": 7},
        provider_used="codex-cli",
        model_used="gpt-5.5",
        native_session_id="native-123",
    )

    reloaded = LLMSessionManager(tmp_path / "sessions")
    same = reloaded.get_or_create(
        key=ref.key,
        provider="codex-cli",
        model="gpt-5.5",
        runtime_fingerprint="fp-1",
    )
    calls = [
        json.loads(line)
        for line in (tmp_path / "sessions" / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert updated.native_session_id == "native-123"
    assert same.native_session_id == "native-123"
    assert reloaded.turn_count(ref.key) == 1
    assert calls[0]["session_key"] == ref.key
    assert calls[0]["usage"]["cached_input_tokens"] == 7


def test_session_manager_rejects_runtime_mismatch(tmp_path):
    manager = LLMSessionManager(tmp_path / "sessions")
    manager.get_or_create(key="k", provider="codex-cli", model="m1", runtime_fingerprint="fp-1")

    with pytest.raises(ValueError, match="runtime fingerprint changed"):
        manager.get_or_create(key="k", provider="codex-cli", model="m1", runtime_fingerprint="fp-2")


def test_session_manager_reloads_state_written_by_other_manager(tmp_path):
    root = tmp_path / "sessions"
    first = LLMSessionManager(root)
    second = LLMSessionManager(root)

    ref = first.get_or_create(key="k", provider="codex-cli", model="m1", runtime_fingerprint="fp-1")
    first.update_native_session_id(ref.key, "native-1")
    same = second.get_or_create(key="k", provider="codex-cli", model="m1", runtime_fingerprint="fp-1")

    assert same.native_session_id == "native-1"


def test_session_manager_refuses_native_id_overwrite_from_other_manager(tmp_path):
    root = tmp_path / "sessions"
    first = LLMSessionManager(root)
    second = LLMSessionManager(root)

    ref = first.get_or_create(key="k", provider="codex-cli", model="m1", runtime_fingerprint="fp-1")
    first.update_native_session_id(ref.key, "native-1")

    with pytest.raises(ValueError, match="native_session_id changed"):
        second.update_native_session_id(ref.key, "native-2")


def test_session_lock_serializes_threads(tmp_path):
    manager = LLMSessionManager(tmp_path / "sessions")
    events: list[str] = []

    def worker(name: str) -> None:
        with manager.lock("shared"):
            events.append(f"{name}:enter")
            events.append(f"{name}:exit")

    first = threading.Thread(target=worker, args=("a",))
    second = threading.Thread(target=worker, args=("b",))
    first.start()
    second.start()
    first.join()
    second.join()

    assert events in (["a:enter", "a:exit", "b:enter", "b:exit"], ["b:enter", "b:exit", "a:enter", "a:exit"])


def test_locked_turn_serializes_turn_count_across_manager_instances(tmp_path):
    root = tmp_path / "sessions"
    managers = [LLMSessionManager(root), LLMSessionManager(root)]
    barrier = threading.Barrier(2)
    turn_counts: list[int] = []

    def worker(manager: LLMSessionManager) -> None:
        barrier.wait()
        with manager.locked_turn(
            key="shared",
            provider="codex-cli",
            model="m1",
            runtime_fingerprint="fp-1",
        ) as (_ref, turn_count):
            turn_counts.append(turn_count)
            manager.record_turn(
                "shared",
                call_label=f"turn-{turn_count}",
                prompt_sha256=f"prompt-{turn_count}",
                static_prefix_sha256=None,
                schema_sha256=None,
                usage={},
                provider_used="codex-cli",
                model_used="m1",
                native_session_id=None,
            )

    threads = [threading.Thread(target=worker, args=(manager,)) for manager in managers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(turn_counts) == [0, 1]


def test_runtime_fingerprint_includes_runtime_but_not_prompt_or_run_values():
    first = runtime_fingerprint(
        provider="codex-cli",
        model="gpt-5.5",
        model_tier="high",
        env={
            "ARC_CODEX_SANDBOX": "read-only",
            "ARC_CODEX_WORK_DIR": "/tmp/project",
            "ARC_CODEX_REASONING_EFFORT": "high",
        },
        process_chain=["codex", "bash"],
    )
    second = runtime_fingerprint(
        provider="codex-cli",
        model="gpt-5.5",
        model_tier="high",
        env={
            "ARC_CODEX_SANDBOX": "workspace-write",
            "ARC_CODEX_WORK_DIR": "/tmp/project",
            "ARC_CODEX_REASONING_EFFORT": "high",
        },
        process_chain=["codex", "bash"],
    )

    assert first != second
    assert "prompt" not in first
    assert "attempt" not in first


def test_runtime_fingerprint_includes_provider_config_env():
    base = runtime_fingerprint(provider="codex-cli", model="m", model_tier=None, env={}, process_chain=[])
    codex_config = runtime_fingerprint(
        provider="codex-cli",
        model="m",
        model_tier=None,
        env={"ARC_CODEX_CONFIG": 'model="x"'},
        process_chain=[],
    )
    claude_mcp = runtime_fingerprint(
        provider="claude-cli",
        model="m",
        model_tier=None,
        env={"ARC_CLAUDE_MCP_CONFIG": "/tmp/mcp.json", "ARC_CLAUDE_MCP_MODE": "arc-only"},
        process_chain=[],
    )

    assert codex_config != base
    assert claude_mcp != base


def test_runtime_fingerprint_includes_claude_mcp_config_file_contents(tmp_path):
    config_path = tmp_path / "mcp.json"
    config_path.write_text('{"mcpServers":{"a":{"command":"a"}}}', encoding="utf-8")
    first = runtime_fingerprint(
        provider="claude-cli",
        model="m",
        model_tier=None,
        env={"ARC_CLAUDE_MCP_CONFIG": str(config_path)},
        process_chain=[],
    )

    config_path.write_text('{"mcpServers":{"b":{"command":"b"}}}', encoding="utf-8")
    second = runtime_fingerprint(
        provider="claude-cli",
        model="m",
        model_tier=None,
        env={"ARC_CLAUDE_MCP_CONFIG": str(config_path)},
        process_chain=[],
    )

    assert first != second
