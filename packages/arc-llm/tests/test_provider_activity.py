from __future__ import annotations

import json
import os
import sys

import pytest

from arc_llm.providers.activity import ActivityTracker, resolve_idle_timeout_seconds
from arc_llm.providers.base import LLMSubmissionState, LLMWorkerError, LLMWorkerTimeout
from arc_llm.providers.claude_cli import (
    _claude_terminal_json,
    _record_claude_activity,
    _require_stream_json_support,
)
from arc_llm.providers.codex_cli import (
    _record_codex_activity,
    _require_codex_json_stream_support,
)
from arc_llm.providers.lifecycle import run_streaming_process_group
from arc_llm.providers.kimi_code_cli import _AcpProcess


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_activity_uses_idle_not_total_runtime_and_emits_reviews() -> None:
    clock = Clock()
    events: list[dict] = []
    tracker = ActivityTracker(
        provider="test", idle_timeout_seconds=30, review_interval_seconds=30,
        progress_callback=events.append, clock=clock,
    )
    tracker.submitted()
    clock.now = 29
    tracker.record("assistant", text="finished section one")
    clock.now = 31
    tracker.check()
    assert any(event["event"] == "review_due" for event in events)
    clock.now = 58
    tracker.check()
    clock.now = 60
    tracker.record("tool", text="saved checkpoint")
    clock.now = 89
    tracker.check()
    with pytest.raises(LLMWorkerTimeout, match="no meaningful output") as caught:
        clock.now = 90
        tracker.check()
    assert caught.value.submission_state == LLMSubmissionState.SUBMITTED


def test_noise_and_duplicate_progress_do_not_refresh_idle() -> None:
    clock = Clock()
    tracker = ActivityTracker(provider="test", idle_timeout_seconds=10, clock=clock)
    tracker.submitted()
    clock.now = 5
    assert not tracker.record("heartbeat", text="still alive")
    assert not tracker.record("reasoning", text="hidden")
    assert tracker.record("assistant", text="result A")
    clock.now = 9
    assert not tracker.record("assistant", text="result A")
    clock.now = 15
    with pytest.raises(LLMWorkerTimeout):
        tracker.check()


def test_empty_assistant_heartbeat_does_not_refresh_idle() -> None:
    clock = Clock()
    tracker = ActivityTracker(provider="test", idle_timeout_seconds=10, clock=clock)
    tracker.submitted()
    clock.now = 9
    assert not tracker.record("assistant", text="still alive")
    clock.now = 10
    with pytest.raises(LLMWorkerTimeout):
        tracker.check()


def test_repeated_assistant_heartbeat_does_not_refresh_idle() -> None:
    clock = Clock()
    tracker = ActivityTracker(provider="test", idle_timeout_seconds=10, clock=clock)
    tracker.submitted()
    clock.now = 9
    assert not tracker.record("assistant", text="still alive " * 600)
    clock.now = 10
    with pytest.raises(LLMWorkerTimeout):
        tracker.check()


def test_progress_persistence_failure_aborts_submitted_call() -> None:
    def broken(_event: dict) -> None:
        raise OSError("disk full")

    tracker = ActivityTracker(provider="test", progress_callback=broken)
    tracker.submitted()
    with pytest.raises(LLMWorkerError, match="persist provider progress") as caught:
        tracker.check()
    assert caught.value.submission_state == LLMSubmissionState.SUBMITTED


def test_artifact_progress_requires_an_existing_path(tmp_path) -> None:
    events: list[dict] = []
    tracker = ActivityTracker(provider="test", progress_callback=events.append)
    tracker.submitted()
    assert not tracker.record_artifact(tmp_path / "missing.json")
    artifact = tmp_path / "checkpoint.json"
    artifact.write_text("{}", encoding="utf-8")
    assert tracker.record_artifact(artifact)
    assert events[-1]["artifact_paths"] == [str(artifact.resolve())]


def test_idle_timeout_resolution_precedence() -> None:
    assert resolve_idle_timeout_seconds(None, env={}, provider="codex-cli") == 1800
    assert resolve_idle_timeout_seconds(
        None, env={"ARC_LLM_IDLE_TIMEOUT_SECONDS": "20"}, provider="codex-cli"
    ) == 20
    assert resolve_idle_timeout_seconds(
        None,
        env={"ARC_LLM_IDLE_TIMEOUT_SECONDS": "20", "ARC_CODEX_IDLE_TIMEOUT_SECONDS": "10"},
        provider="codex-cli",
    ) == 10
    assert resolve_idle_timeout_seconds(5, env={}, provider="codex-cli") == 5


def test_removed_total_timeout_environment_fails_before_provider_submission() -> None:
    with pytest.raises(ValueError, match="ARC_CODEX_IDLE_TIMEOUT_SECONDS"):
        resolve_idle_timeout_seconds(
            None,
            env={"ARC_CODEX_TIMEOUT_SECONDS": "60"},
            provider="codex-cli",
        )


@pytest.mark.parametrize(
    ("preflight", "env"),
    [
        (_require_codex_json_stream_support, {"ARC_CODEX_JSON_STREAM_SUPPORT": "0"}),
        (_require_stream_json_support, {"ARC_CLAUDE_STREAM_JSON_SUPPORT": "0"}),
    ],
)
def test_unsupported_streaming_fails_before_submission(preflight, env) -> None:
    with pytest.raises(LLMWorkerError, match="upgrade") as caught:
        preflight(env)
    assert caught.value.submission_state == LLMSubmissionState.NOT_SUBMITTED


def test_codex_and_claude_event_classification() -> None:
    codex_events: list[dict] = []
    codex = ActivityTracker(provider="codex", progress_callback=codex_events.append)
    codex.submitted()
    _record_codex_activity(
        json.dumps({"type": "item.completed", "item": {"type": "reasoning", "text": "secret"}}), codex
    )
    _record_codex_activity(
        json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "milestone"}}), codex
    )
    assert [event.get("summary") for event in codex_events if event["event"] == "provider_progress"] == ["milestone"]

    claude_events: list[dict] = []
    claude = ActivityTracker(provider="claude", progress_callback=claude_events.append)
    claude.submitted()
    _record_claude_activity(
        json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "draft ready"}]}}),
        claude,
    )
    assert any(event.get("summary") == "draft ready" for event in claude_events)


def test_codex_tool_progress_never_exposes_arguments_or_queries() -> None:
    events: list[dict] = []
    tracker = ActivityTracker(provider="codex", progress_callback=events.append)
    tracker.submitted()
    _record_codex_activity(
        json.dumps({
            "type": "item.completed",
            "item": {
                "id": "tool-1",
                "type": "command_execution",
                "status": "completed",
                "command": "curl -H Authorization:super-secret https://private.invalid",
                "query": "confidential search terms",
                "name": "sensitive-tool-name",
            },
        }),
        tracker,
    )
    rendered = json.dumps(events)
    assert "super-secret" not in rendered
    assert "confidential" not in rendered
    assert "sensitive-tool-name" not in rendered
    assert events[-1]["summary"] == "command_execution completed"


def test_claude_split_heartbeat_deltas_do_not_refresh_idle() -> None:
    clock = Clock()
    tracker = ActivityTracker(provider="claude", idle_timeout_seconds=10, clock=clock)
    tracker.submitted()
    clock.now = 5
    for fragment in ("still ", "alive"):
        _record_claude_activity(
            json.dumps({
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": fragment},
                },
            }),
            tracker,
        )
    clock.now = 10
    with pytest.raises(LLMWorkerTimeout):
        tracker.check()


def test_session_metadata_is_visible_but_not_substantive() -> None:
    events: list[dict] = []
    tracker = ActivityTracker(provider="codex", progress_callback=events.append)
    tracker.submitted()

    _record_codex_activity(
        json.dumps({"type": "thread.started", "thread_id": "session-1"}), tracker
    )

    session_event = next(event for event in events if event.get("activity_kind") == "session")
    assert session_event["substantive"] is False
    assert session_event["resumable"] is True


def test_claude_terminal_result_is_selected_from_jsonl() -> None:
    stdout = "\n".join([
        json.dumps({"type": "system", "subtype": "init"}),
        json.dumps({"type": "result", "result": "done", "session_id": "s1"}),
    ])
    assert json.loads(_claude_terminal_json(stdout))["result"] == "done"


def test_kimi_acp_forwards_assistant_and_tool_progress() -> None:
    events: list[dict] = []
    activity = ActivityTracker(provider="kimi", progress_callback=events.append)
    client = _AcpProcess({}, artifact_dir=None, cancel_check=None, activity=activity)
    client.current_session_id = "s1"
    activity.submitted()
    client._capture_update({
        "method": "session/update",
        "params": {
            "sessionId": "s1",
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "section complete"},
            },
        },
    })
    client.flush_assistant_progress()
    client._capture_update({
        "method": "session/update",
        "params": {
            "sessionId": "s1",
            "update": {"sessionUpdate": "tool_call_update", "title": "secret args", "status": "completed"},
        },
    })
    assert client.message_chunks == ["section complete"]
    assert [
        event.get("summary") for event in events if event["event"] == "provider_progress"
    ] == ["section complete", "kimi_tool completed"]


def test_kimi_split_heartbeat_and_plan_updates_do_not_refresh_idle() -> None:
    clock = Clock()
    events: list[dict] = []
    activity = ActivityTracker(
        provider="kimi", idle_timeout_seconds=10, progress_callback=events.append, clock=clock
    )
    client = _AcpProcess({}, artifact_dir=None, cancel_check=None, activity=activity)
    client.current_session_id = "s1"
    activity.submitted()
    clock.now = 5
    for chunk in ("still ", "alive\n"):
        client._capture_update({
            "method": "session/update",
            "params": {
                "sessionId": "s1",
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": chunk},
                },
            },
        })
    client._capture_update({
        "method": "session/update",
        "params": {
            "sessionId": "s1",
            "update": {"sessionUpdate": "plan_update", "title": "looks useful but is not"},
        },
    })
    assert all(not event.get("substantive") for event in events)
    assert "looks useful" not in json.dumps(events)
    clock.now = 10
    with pytest.raises(LLMWorkerTimeout):
        activity.check()


def test_streaming_process_idle_timeout_terminates_process_group() -> None:
    tracker = ActivityTracker(provider="fake", idle_timeout_seconds=0.15)
    with pytest.raises(LLMWorkerTimeout):
        run_streaming_process_group(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            input_text="", env=os.environ, activity=tracker,
            stdout_line_callback=lambda _line: None, poll_interval_seconds=0.02,
            terminate_grace_seconds=0.1,
        )
