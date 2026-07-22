from __future__ import annotations

import pytest

from arc_llm.raw_completion import RawCompletionError, validate_raw_completion
from arc_llm import raw_completion as raw_module


def _kimi_prefix() -> list[dict]:
    return [
        {"direction": "request", "id": 1, "method": "initialize"},
        {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": 1}},
        {"direction": "request", "id": 2, "method": "authenticate"},
        {"jsonrpc": "2.0", "id": 2, "result": {}},
        {"direction": "request", "id": 3, "method": "session/new"},
        {"jsonrpc": "2.0", "id": 3, "result": {"sessionId": "session-1"}},
    ]


def test_codex_requires_unique_final_completion() -> None:
    events = [
        {"type": "thread.started", "thread_id": "thread-1", "ordinal": 1},
        {"type": "turn.started", "ordinal": 2},
        {"type": "item.completed", "ordinal": 3, "item": {
            "type": "agent_message", "text": '{"ok":true}',
        }},
        {"type": "turn.completed", "ordinal": 4},
    ]
    completion = validate_raw_completion(
        "codex-cli", events, native_session_id="thread-1",
    )
    assert completion.terminal_ordinal == 4
    assert completion.material[0].source == "codex.completed_message"

    with pytest.raises(RawCompletionError, match="ordinals"):
        validate_raw_completion("codex-cli", [events[0], {**events[-1], "ordinal": 1}])
    with pytest.raises(RawCompletionError, match="turn.completed"):
        validate_raw_completion("codex-cli", events[:-1])
    with pytest.raises(RawCompletionError, match="thread.started"):
        validate_raw_completion(
            "codex-cli",
            [{key: value for key, value in event.items() if key != "ordinal"}
             for event in events[1:]],
            native_session_id="thread-1",
        )


def test_claude_requires_one_success_result_at_end() -> None:
    events = [
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": '{"ok":true}'},
        ]}},
        {"type": "result", "subtype": "success", "is_error": False,
         "session_id": "session-1", "structured_output": {"ok": True}},
    ]
    completion = validate_raw_completion(
        "claude-cli", events, native_session_id="session-1",
    )
    assert completion.terminal_ordinal == 2
    assert completion.material[-1].source == "claude.terminal_structured_output"

    with pytest.raises(RawCompletionError, match="error result"):
        validate_raw_completion("claude-cli", [
            events[0], {**events[1], "is_error": True},
        ])
    with pytest.raises(RawCompletionError, match="assistant|final"):
        validate_raw_completion("claude-cli", [events[1], events[0]])
    missing_session = [events[0], {**events[1], "session_id": None}]
    with pytest.raises(RawCompletionError, match="expected session"):
        validate_raw_completion(
            "claude-cli", missing_session, native_session_id="session-1",
        )


def test_kimi_binds_prompt_id_session_chunks_and_end_turn() -> None:
    events = [
        *_kimi_prefix(),
        {"direction": "request", "id": 4, "method": "session/prompt"},
        {"jsonrpc": "2.0", "method": "session/update", "params": {
            "sessionId": "session-1", "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": '{"ok":true}'},
            },
        }},
        {"jsonrpc": "2.0", "id": 4, "result": {"stopReason": "end_turn"}},
    ]
    completion = validate_raw_completion(
        "kimi-code-cli", events, native_session_id="session-1",
    )
    assert completion.terminal_ordinal == len(events)
    assert completion.material[0].source == "kimi.session_prompt_message"

    with pytest.raises(RawCompletionError, match="lacks one final"):
        validate_raw_completion("kimi-code-cli", events[:-1])
    with pytest.raises(RawCompletionError, match="session changed"):
        validate_raw_completion(
            "kimi-code-cli", events, native_session_id="other-session",
        )
    with pytest.raises(RawCompletionError, match="after completion"):
        validate_raw_completion("kimi-code-cli", [*events, dict(events[-1])])


@pytest.mark.parametrize("invalid_id", [True, False, None, ""])
def test_kimi_rejects_malformed_json_rpc_ids(invalid_id) -> None:
    events = [
        {"direction": "request", "id": invalid_id, "method": "session/prompt"},
        {
            "jsonrpc": "2.0", "id": invalid_id,
            "result": {"stopReason": "end_turn"},
        },
    ]
    with pytest.raises(RawCompletionError, match="unknown or duplicated"):
        validate_raw_completion("kimi-code-cli", events)


def test_kimi_boolean_response_id_cannot_alias_integer_request_id() -> None:
    events = [
        *_kimi_prefix(),
        {"direction": "request", "id": 4, "method": "session/prompt"},
        {
            "jsonrpc": "2.0", "id": True,
            "result": {"stopReason": "end_turn"},
        },
    ]
    with pytest.raises(RawCompletionError, match="unknown or duplicated"):
        validate_raw_completion("kimi-code-cli", events)


@pytest.mark.parametrize("provider,events,message", [
    (
        "codex-cli",
        [
            {"type": "item.completed", "item": {"type": "agent_message", "text": "{}"}},
            {"type": "turn.completed"},
        ],
        "thread.started",
    ),
    (
        "claude-cli",
        [
            {"type": "result", "subtype": "success", "is_error": False},
        ],
        "assistant",
    ),
    (
        "kimi-code-cli",
        [
            {"direction": "request", "id": 1, "method": "session/prompt"},
            {"jsonrpc": "2.0", "id": 1, "result": {"stopReason": "end_turn"}},
        ],
        "lifecycle",
    ),
])
def test_provider_lifecycle_is_closed(provider, events, message) -> None:
    with pytest.raises(RawCompletionError, match=message):
        validate_raw_completion(provider, events)


def test_raw_event_count_line_and_aggregate_caps(monkeypatch) -> None:
    valid = [
        {"type": "thread.started", "thread_id": "thread-1"},
        {"type": "turn.started"},
        {"type": "item.completed", "item": {
            "type": "agent_message", "text": '{"ok":true}',
        }},
        {"type": "turn.completed"},
    ]
    monkeypatch.setattr(raw_module, "MAX_RAW_EVENTS", 2)
    with pytest.raises(RawCompletionError, match="count"):
        validate_raw_completion("codex-cli", valid)
    monkeypatch.setattr(raw_module, "MAX_RAW_EVENTS", 10)
    monkeypatch.setattr(raw_module, "MAX_RAW_EVENT_BYTES", 20)
    with pytest.raises(RawCompletionError, match="line"):
        validate_raw_completion("codex-cli", valid)
    monkeypatch.setattr(raw_module, "MAX_RAW_EVENT_BYTES", 1024)
    monkeypatch.setattr(raw_module, "MAX_RAW_AGGREGATE_BYTES", 40)
    with pytest.raises(RawCompletionError, match="aggregate"):
        validate_raw_completion("codex-cli", valid)


def test_payload_and_request_closure_rejects_unknown_fields_and_open_requests() -> None:
    with pytest.raises(RawCompletionError, match="unknown fields"):
        validate_raw_completion("codex-cli", [
            {"type": "thread.started", "thread_id": "thread-1", "secret": "x"},
            {"type": "turn.completed"},
        ])
    events = [
        *_kimi_prefix(),
        {"direction": "request", "id": 4, "method": "session/set_config_option"},
        {"direction": "request", "id": 5, "method": "session/prompt"},
    ]
    with pytest.raises(RawCompletionError, match="prior response"):
        validate_raw_completion("kimi-code-cli", events)

    reverse = [
        *_kimi_prefix(),
        {"direction": "request", "id": 4, "method": "session/prompt"},
        {
            "jsonrpc": "2.0",
            "id": "reverse-1",
            "method": "session/request_permission",
            "params": {
                "sessionId": "session-1",
                "toolCall": {"toolCallId": "tool-1", "status": "pending"},
                "options": [{
                    "optionId": "allow", "name": "Allow", "kind": "allow_once",
                }],
            },
        },
    ]
    with pytest.raises(RawCompletionError, match="response closure"):
        validate_raw_completion("kimi-code-cli", reverse)


def _kimi_permission_turn() -> list[dict]:
    return [
        *_kimi_prefix(),
        {"direction": "request", "id": 4, "method": "session/prompt"},
        {
            "jsonrpc": "2.0", "id": "reverse-1",
            "method": "session/request_permission",
            "params": {
                "sessionId": "session-1",
                "toolCall": {"toolCallId": "tool-1", "status": "pending"},
                "options": [{
                    "optionId": "allow", "name": "Allow", "kind": "allow_once",
                }],
            },
        },
        {
            "direction": "reverse_response", "id": "reverse-1",
            "method": "session/request_permission",
            "disposition": "permission_cancelled",
        },
        {"jsonrpc": "2.0", "method": "session/update", "params": {
            "sessionId": "session-1", "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": '{"ok":true}'},
            },
        }},
        {"jsonrpc": "2.0", "id": 4, "result": {"stopReason": "end_turn"}},
    ]


def test_kimi_reverse_denial_marker_closes_request_without_closing_prompt() -> None:
    completion = validate_raw_completion(
        "kimi-code-cli", _kimi_permission_turn(), native_session_id="session-1",
    )
    assert completion.terminal_ordinal == len(_kimi_permission_turn())
    assert completion.material[0].source == "kimi.session_prompt_message"


@pytest.mark.parametrize("mutation", ["missing", "wrong_id", "wrong_disposition", "duplicate", "reordered"])
def test_kimi_reverse_denial_marker_is_exactly_once_and_ordered(mutation: str) -> None:
    events = _kimi_permission_turn()
    marker = dict(events[8])
    if mutation == "missing":
        del events[8]
    elif mutation == "wrong_id":
        events[8] = {**marker, "id": "other"}
    elif mutation == "wrong_disposition":
        events[8] = {**marker, "disposition": "filesystem_denied"}
    elif mutation == "duplicate":
        events.insert(9, marker)
    elif mutation == "reordered":
        events[7], events[8] = events[8], events[7]
    with pytest.raises(RawCompletionError, match="reverse"):
        validate_raw_completion("kimi-code-cli", events)


def test_nested_provider_payloads_and_ordinal_coverage_are_closed() -> None:
    with pytest.raises(RawCompletionError, match="item payload"):
        validate_raw_completion("codex-cli", [
            {"type": "thread.started", "thread_id": "thread"},
            {"type": "turn.started"},
            {"type": "item.completed", "item": {
                "type": "agent_message", "text": "{}", "hidden": "x",
            }},
            {"type": "turn.completed"},
        ])
    with pytest.raises(RawCompletionError, match="content block"):
        validate_raw_completion("claude-cli", [
            {"type": "assistant", "message": {"content": [
                {"type": "text", "text": "{}", "hidden": "x"},
            ]}},
            {"type": "result", "subtype": "success", "is_error": False},
        ])
    with pytest.raises(RawCompletionError, match="coverage"):
        validate_raw_completion("codex-cli", [
            {"type": "thread.started", "thread_id": "thread", "ordinal": 1},
            {"type": "turn.started"},
            {"type": "turn.completed", "ordinal": 3},
        ])
