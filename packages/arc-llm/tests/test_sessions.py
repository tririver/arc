from __future__ import annotations

import json
import threading

import pytest

from arc_llm import sessions as sessions_module
from arc_llm.sessions import LLMSessionManager, runtime_fingerprint


def _create_session_for_process(args):
    root_text, key = args
    from pathlib import Path

    from arc_llm.sessions import LLMSessionManager

    manager = LLMSessionManager(Path(root_text))
    manager.get_or_create(
        key=key,
        provider="codex-cli",
        model="test-model",
        runtime_fingerprint="runtime",
    )
    return key


def _record_turn_for_process(args):
    root_text, key = args
    from pathlib import Path

    from arc_llm.sessions import LLMSessionManager

    manager = LLMSessionManager(Path(root_text))
    manager.get_or_create(
        key=key,
        provider="codex-cli",
        model="test-model",
        runtime_fingerprint="runtime",
    )
    manager.record_turn(
        key,
        call_label=f"call/{key}",
        prompt_sha256="prompt",
        static_prefix_sha256=None,
        schema_sha256=None,
        usage={},
        provider_used="codex-cli",
        model_used="test-model",
        native_session_id=None,
    )
    return key


def _attempt_lock_for_process(root_text, key, timeout, queue):
    import os
    from pathlib import Path

    from arc_llm.sessions import LLMSessionManager

    os.environ["ARC_LLM_SESSION_LOCK_TIMEOUT_SECONDS"] = str(timeout)
    manager = LLMSessionManager(Path(root_text))
    try:
        with manager.lock(key):
            queue.put("acquired")
    except TimeoutError as exc:
        queue.put(f"timeout:{exc}")


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


def test_session_lock_times_out_on_unrecoverable_foreign_lock(tmp_path):
    from multiprocessing import get_context

    root = tmp_path / "sessions"
    key = "blocked"
    lock_path = root / "locks" / f"{sessions_module._safe_lock_name(key)}.lock"  # noqa: SLF001
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text(
        json.dumps({"pid": 1, "thread_id": 1, "host": "other-host", "created_at": "2026-01-01T00:00:00+00:00"})
        + "\n",
        encoding="utf-8",
    )
    ctx = get_context()
    queue = ctx.Queue()
    process = ctx.Process(target=_attempt_lock_for_process, args=(str(root), key, 0.05, queue))

    process.start()
    process.join(1)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("lock acquisition hung instead of timing out")

    assert process.exitcode == 0
    assert str(queue.get()).startswith("timeout:")


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


def test_session_store_preserves_concurrent_process_keys(tmp_path):
    from concurrent.futures import ProcessPoolExecutor

    keys = [f"worker_{i:03d}" for i in range(32)]
    with ProcessPoolExecutor(max_workers=8) as pool:
        returned = list(pool.map(_create_session_for_process, [(str(tmp_path), key) for key in keys]))

    payload = json.loads((tmp_path / "sessions.json").read_text(encoding="utf-8"))
    assert sorted(returned) == sorted(keys)
    assert set(payload["sessions"]) == set(keys)


def test_calls_jsonl_preserves_concurrent_process_records(tmp_path):
    from concurrent.futures import ProcessPoolExecutor

    keys = [f"worker_{i:03d}" for i in range(32)]
    with ProcessPoolExecutor(max_workers=8) as pool:
        returned = list(pool.map(_record_turn_for_process, [(str(tmp_path), key) for key in keys]))

    lines = (tmp_path / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    seen = {json.loads(line)["session_key"] for line in lines}
    assert sorted(returned) == sorted(keys)
    assert seen == set(keys)
    assert len(lines) == len(keys)


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


@pytest.mark.parametrize(
    "changed_env",
    [
        {"ARC_KIMI_BIN": "/opt/kimi"},
        {"ARC_KIMI_WORK_DIR": "/tmp/kimi-project"},
        {"KIMI_CODE_HOME": "/tmp/kimi-home"},
        {"ARC_LLM_KIMI_HIGH_MODEL": "kimi-high"},
        {"ARC_KIMI_IDLE_TIMEOUT_SECONDS": "42"},
        {"ARC_LLM_IDLE_TIMEOUT_SECONDS": "43"},
    ],
)
def test_kimi_runtime_fingerprint_includes_runtime_inputs(changed_env):
    base = runtime_fingerprint(provider="kimi-code-cli", model="default_model", model_tier="high", env={})
    changed = runtime_fingerprint(
        provider="kimi-code-cli",
        model="default_model",
        model_tier="high",
        env=changed_env,
    )

    assert changed != base


def test_kimi_runtime_fingerprint_uses_provider_idle_timeout_before_generic_fallback():
    first = runtime_fingerprint(
        provider="kimi-code-cli",
        model="default_model",
        model_tier=None,
        env={
            "ARC_KIMI_IDLE_TIMEOUT_SECONDS": "42",
            "ARC_LLM_IDLE_TIMEOUT_SECONDS": "10",
        },
    )
    second = runtime_fingerprint(
        provider="kimi-code-cli",
        model="default_model",
        model_tier=None,
        env={
            "ARC_KIMI_IDLE_TIMEOUT_SECONDS": "42",
            "ARC_LLM_IDLE_TIMEOUT_SECONDS": "20",
        },
    )

    assert first == second


def test_kimi_runtime_fingerprint_empty_provider_idle_timeout_uses_generic_fallback():
    first = runtime_fingerprint(
        provider="kimi-code-cli",
        model="default_model",
        model_tier=None,
        env={
            "ARC_KIMI_IDLE_TIMEOUT_SECONDS": "",
            "ARC_LLM_IDLE_TIMEOUT_SECONDS": "10",
        },
    )
    second = runtime_fingerprint(
        provider="kimi-code-cli",
        model="default_model",
        model_tier=None,
        env={
            "ARC_KIMI_IDLE_TIMEOUT_SECONDS": "",
            "ARC_LLM_IDLE_TIMEOUT_SECONDS": "20",
        },
    )

    assert first != second


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


def test_generated_claude_arc_mcp_path_does_not_change_fingerprint_when_file_appears(tmp_path):
    config_path = tmp_path / "arc-claude-mcp.json"
    env = {
        "ARC_CLAUDE_MCP_MODE": "arc-only",
        "ARC_CLAUDE_ARC_MCP_CONFIG_PATH": str(config_path),
        "ARC_CLAUDE_ARC_MCP_COMMAND": "arc-mcp",
        "ARC_CLAUDE_ARC_MCP_ENV_JSON": '{"FOO":"bar"}',
    }
    first = runtime_fingerprint(provider="claude-cli", model="m", model_tier=None, env=env)

    config_path.write_text('{"mcpServers":{"arc":{"command":"arc-mcp"}}}', encoding="utf-8")
    second = runtime_fingerprint(provider="claude-cli", model="m", model_tier=None, env=env)

    assert first == second
