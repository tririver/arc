from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence

from .response_candidates import material_from_claude, material_from_codex, material_from_kimi
from .usage import ResponseCandidateMaterial


class RawCompletionError(ValueError):
    pass


MAX_RAW_EVENTS = 100_000
MAX_RAW_EVENT_BYTES = 1024 * 1024
MAX_RAW_AGGREGATE_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class ValidatedRawCompletion:
    material: tuple[ResponseCandidateMaterial, ...]
    terminal_ordinal: int


def validate_raw_completion(
    provider: str,
    events: Sequence[Mapping[str, Any]],
    *,
    native_session_id: str | None = None,
) -> ValidatedRawCompletion:
    if not events:
        raise RawCompletionError("provider raw event stream is empty")
    _validate_raw_limits(events)
    _validate_ordinals(events)
    if provider == "codex-cli":
        return _codex(events, native_session_id)
    if provider == "claude-cli":
        return _claude(events, native_session_id)
    if provider == "kimi-code-cli":
        return _kimi(events, native_session_id)
    raise RawCompletionError("provider has no recovery raw-event grammar")


def _validate_raw_limits(events: Sequence[Mapping[str, Any]]) -> None:
    if len(events) > MAX_RAW_EVENTS:
        raise RawCompletionError("provider raw event count exceeds its limit")
    aggregate = 0
    for event in events:
        if not isinstance(event, Mapping):
            raise RawCompletionError("provider raw event is not an object")
        try:
            raw = json.dumps(
                dict(event), ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError, RecursionError) as exc:
            raise RawCompletionError("provider raw event is not bounded JSON") from exc
        if len(raw) > MAX_RAW_EVENT_BYTES:
            raise RawCompletionError("provider raw event line exceeds its byte limit")
        aggregate += len(raw) + 1
        if aggregate > MAX_RAW_AGGREGATE_BYTES:
            raise RawCompletionError("provider raw event aggregate exceeds its byte limit")


def _validate_ordinals(events: Sequence[Mapping[str, Any]]) -> None:
    previous = 0
    seen: set[int] = set()
    marker_mode: bool | None = None
    for event in events:
        values = [event[key] for key in ("sequence", "ordinal") if key in event]
        if len(values) == 2 and values[0] != values[1]:
            raise RawCompletionError("provider event sequence and ordinal conflict")
        marked = bool(values)
        if marker_mode is None:
            marker_mode = marked
        elif marked != marker_mode:
            raise RawCompletionError("provider event ordinal coverage is incomplete")
        if not values:
            continue
        value = values[0]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value != previous + 1
            or value in seen
        ):
            raise RawCompletionError(
                "provider event ordinals are not unique, contiguous, and increasing"
            )
        seen.add(value)
        previous = value


def _codex(
    events: Sequence[Mapping[str, Any]], native_session_id: str | None,
) -> ValidatedRawCompletion:
    allowed = {
        "thread.started", "turn.started", "item.started", "item.updated",
        "item.completed", "message.completed", "turn.completed",
        "turn.failed", "error",
    }
    terminals: list[int] = []
    observed_native: str | None = None
    thread_seen = False
    turn_open = False
    started_items: set[str] = set()
    for ordinal, event in enumerate(events, 1):
        if terminals:
            raise RawCompletionError("Codex raw stream contains an event after completion")
        kind = event.get("type")
        if kind not in allowed:
            raise RawCompletionError("Codex raw stream contains an unknown event")
        common = {"type", "sequence", "ordinal"}
        allowed_fields = {
            "thread.started": common | {"thread_id", "session_id"},
            "turn.started": common,
            "item.started": common | {"item"},
            "item.updated": common | {"item"},
            "item.completed": common | {"item"},
            "message.completed": common | {"message", "item"},
            "turn.completed": common | {"usage"},
            "turn.failed": common | {"error"},
            "error": common | {"error", "message"},
        }[str(kind)]
        if set(event) - allowed_fields:
            raise RawCompletionError("Codex raw event payload contains unknown fields")
        if kind == "thread.started":
            if ordinal != 1 or thread_seen:
                raise RawCompletionError("Codex thread.started lifecycle is invalid")
            thread_id = event.get("thread_id")
            session_id = event.get("session_id")
            if (
                thread_id is not None and not isinstance(thread_id, str)
                or session_id is not None and not isinstance(session_id, str)
                or thread_id and session_id and thread_id != session_id
            ):
                raise RawCompletionError("Codex thread.started payload is malformed")
            candidate = str(thread_id or session_id or "") or None
            if candidate is None:
                raise RawCompletionError("Codex thread.started lacks a session")
            if observed_native not in {None, candidate}:
                raise RawCompletionError("Codex raw stream contains conflicting sessions")
            observed_native = candidate
            thread_seen = True
            continue
        if not thread_seen:
            raise RawCompletionError("Codex raw stream does not start with thread.started")
        if kind == "turn.started":
            if turn_open or ordinal == len(events):
                raise RawCompletionError("Codex turn.started lifecycle is invalid")
            turn_open = True
            continue
        if not turn_open:
            raise RawCompletionError("Codex event occurred outside an open turn")
        if kind in {"item.started", "item.updated", "item.completed"}:
            item = event.get("item")
            if not isinstance(item, Mapping) or not isinstance(item.get("type"), str):
                raise RawCompletionError("Codex item event payload is malformed")
            _validate_codex_item(item)
            item_id = str(item.get("id") or item.get("item_id") or "")
            if kind == "item.started":
                if not item_id or item_id in started_items:
                    raise RawCompletionError("Codex item.started is duplicated")
                started_items.add(item_id)
            elif not item_id or item_id not in started_items:
                if kind == "item.updated" or item_id:
                    raise RawCompletionError(f"Codex {kind} has no open item")
            if kind == "item.completed" and item_id in started_items:
                started_items.discard(item_id)
        if kind == "message.completed":
            payload = event.get("message", event.get("item"))
            if not isinstance(payload, Mapping):
                raise RawCompletionError("Codex message.completed payload is malformed")
            _validate_codex_item(payload)
        if kind in {"turn.failed", "error"}:
            error = event.get("error")
            if error is not None and not isinstance(error, (str, Mapping)):
                raise RawCompletionError("Codex failure payload is malformed")
            raise RawCompletionError("Codex raw stream contains a failure terminal")
        if kind == "turn.completed":
            usage = event.get("usage")
            if usage is not None and not _closed_number_mapping(usage):
                raise RawCompletionError("Codex turn usage payload is malformed")
            if started_items:
                raise RawCompletionError("Codex raw stream contains unclosed item lifecycles")
            turn_open = False
            terminals.append(ordinal)
    if started_items:
        raise RawCompletionError("Codex raw stream contains unclosed item lifecycles")
    if terminals != [len(events)]:
        raise RawCompletionError("Codex raw stream lacks one final turn.completed event")
    if native_session_id and observed_native != native_session_id:
        raise RawCompletionError("Codex raw stream lacks the expected session")
    material = tuple(
        item for item in material_from_codex(events, "")
        if item.source != "codex.output_last_message" and (item.value is not None or str(item.text or "").strip())
    )
    return ValidatedRawCompletion(material, terminals[0])


def _validate_codex_item(item: Mapping[str, Any]) -> None:
    item_type = str(item.get("type") or "")
    common = {"id", "item_id", "type", "status"}
    allowed_by_type = {
        "agent_message": common | {"text", "content"},
        "assistant_message": common | {"text", "content"},
        "message": common | {"text", "content"},
        "reasoning": common | {"text", "content", "summary"},
        "analysis": common | {"text", "content", "summary"},
        "command_execution": common | {
            "command", "aggregated_output", "exit_code",
        },
        "file_change": common | {"changes"},
        "tool_call": common | {
            "call_id", "name", "arguments", "result", "error",
        },
        "mcp_tool_call": common | {
            "call_id", "server", "tool", "arguments", "result", "error",
        },
        "web_search": common | {"query", "result", "error"},
    }
    allowed = allowed_by_type.get(item_type)
    if allowed is None or set(item) - allowed:
        raise RawCompletionError("Codex item payload contains unknown fields")
    for key in ("id", "item_id", "status"):
        if key in item and not isinstance(item[key], str):
            raise RawCompletionError("Codex item identity payload is malformed")
    if item_type in {"agent_message", "assistant_message", "message"}:
        text = item.get("text", item.get("content"))
        if not isinstance(text, str):
            raise RawCompletionError("Codex message item lacks string text")
    if not _closed_json_value(dict(item)):
        raise RawCompletionError("Codex item payload is not closed JSON")


def _closed_number_mapping(value: Any) -> bool:
    return bool(
        isinstance(value, Mapping)
        and all(
            isinstance(key, str)
            and not isinstance(item, bool)
            and isinstance(item, (int, float))
            and item >= 0
            for key, item in value.items()
        )
    )


def _closed_json_value(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return value == value and value not in {float("inf"), float("-inf")}
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str) and _closed_json_value(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return all(_closed_json_value(item) for item in value)
    return False


def _claude(
    events: Sequence[Mapping[str, Any]], native_session_id: str | None,
) -> ValidatedRawCompletion:
    allowed = {"system", "assistant", "user", "result", "stream_event", "rate_limit_event"}
    terminals: list[int] = []
    observed_native: str | None = None
    system_seen = False
    assistant_seen = False
    for ordinal, event in enumerate(events, 1):
        if terminals:
            raise RawCompletionError("Claude raw stream contains an event after completion")
        kind = event.get("type")
        if kind not in allowed:
            raise RawCompletionError("Claude raw stream contains an unknown event")
        common = {"type", "session_id", "uuid"}
        allowed_fields = {
            "system": common | {
                "subtype", "apiKeySource", "cwd", "tools", "mcp_servers",
                "model", "permissionMode", "slash_commands", "output_style",
                "skills", "plugins",
            },
            "assistant": common | {"message", "parent_tool_use_id"},
            "user": common | {"message", "parent_tool_use_id", "tool_use_result"},
            "result": common | {
                "subtype", "is_error", "duration_ms", "duration_api_ms",
                "num_turns", "result", "total_cost_usd", "usage", "modelUsage",
                "permission_denials", "structured_output", "errors",
            },
            "stream_event": common | {"event", "parent_tool_use_id"},
            "rate_limit_event": common | {"rate_limit_info"},
        }[str(kind)] | {"sequence", "ordinal"}
        if set(event) - allowed_fields:
            raise RawCompletionError("Claude raw event payload contains unknown fields")
        candidate_session = str(event.get("session_id") or "") or None
        if candidate_session is not None:
            if observed_native not in {None, candidate_session}:
                raise RawCompletionError("Claude raw stream contains conflicting sessions")
            observed_native = candidate_session
        if kind == "system":
            if system_seen or assistant_seen or terminals:
                raise RawCompletionError("Claude system lifecycle is invalid")
            system_seen = True
            continue
        if kind == "assistant":
            message = event.get("message")
            _validate_claude_message(message, expected_role="assistant")
            assistant_seen = True
        elif kind == "user":
            if not assistant_seen:
                raise RawCompletionError("Claude user lifecycle is invalid")
            _validate_claude_message(event.get("message"), expected_role="user")
        elif kind == "stream_event":
            _validate_claude_stream_event(event.get("event"))
        elif kind == "rate_limit_event" and not _closed_json_value(
            event.get("rate_limit_info")
        ):
            raise RawCompletionError("Claude rate-limit payload is malformed")
        if kind == "result":
            terminals.append(ordinal)
            if event.get("is_error") is True or event.get("subtype") not in {None, "success"}:
                raise RawCompletionError("Claude raw stream contains an error result")
            if not assistant_seen:
                raise RawCompletionError("Claude result has no assistant response")
            if event.get("is_error") is not False:
                raise RawCompletionError("Claude successful result lacks a false error marker")
            for key in ("usage", "modelUsage", "structured_output", "errors"):
                if key in event and not _closed_json_value(event[key]):
                    raise RawCompletionError("Claude result nested payload is malformed")
    if terminals != [len(events)]:
        raise RawCompletionError("Claude raw stream lacks one final successful result")
    if native_session_id and observed_native != native_session_id:
        raise RawCompletionError("Claude raw stream lacks the expected session")
    material = tuple(
        item for item in material_from_claude("\n".join(
            json.dumps(dict(event), sort_keys=True) for event in events
        )) if item.value is not None or str(item.text or "").strip()
    )
    return ValidatedRawCompletion(material, terminals[0])


def _validate_claude_message(value: Any, *, expected_role: str) -> None:
    if not isinstance(value, Mapping):
        raise RawCompletionError("Claude message payload is malformed")
    allowed = {
        "id", "type", "role", "model", "content", "stop_reason",
        "stop_sequence", "usage",
    }
    if set(value) - allowed:
        raise RawCompletionError("Claude message payload contains unknown fields")
    role = value.get("role")
    if role is not None and role != expected_role:
        raise RawCompletionError("Claude message role is inconsistent")
    blocks = value.get("content")
    if not isinstance(blocks, list) or not all(isinstance(block, Mapping) for block in blocks):
        raise RawCompletionError("Claude message content is malformed")
    for block in blocks:
        _validate_claude_content_block(block)
    usage = value.get("usage")
    if usage is not None and not _closed_json_value(usage):
        raise RawCompletionError("Claude message usage is malformed")


def _validate_claude_content_block(block: Mapping[str, Any]) -> None:
    block_type = block.get("type")
    allowed = {
        "text": {"type", "text", "citations"},
        "thinking": {"type", "thinking", "signature"},
        "redacted_thinking": {"type", "data"},
        "tool_use": {"type", "id", "name", "input"},
        "tool_result": {"type", "tool_use_id", "content", "is_error"},
    }.get(str(block_type or ""))
    if allowed is None or set(block) - allowed:
        raise RawCompletionError("Claude content block contains unknown fields")
    required_string = {
        "text": "text",
        "thinking": "thinking",
        "redacted_thinking": "data",
        "tool_use": "id",
        "tool_result": "tool_use_id",
    }[str(block_type)]
    if not isinstance(block.get(required_string), str):
        raise RawCompletionError("Claude content block is malformed")
    if block_type == "tool_use" and not isinstance(block.get("name"), str):
        raise RawCompletionError("Claude tool-use block is malformed")
    if not _closed_json_value(dict(block)):
        raise RawCompletionError("Claude content block is not closed JSON")


def _validate_claude_stream_event(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise RawCompletionError("Claude stream_event payload is malformed")
    allowed = {"type", "index", "delta", "content_block", "message", "usage"}
    if set(value) - allowed or not isinstance(value.get("type"), str):
        raise RawCompletionError("Claude stream_event payload contains unknown fields")
    delta = value.get("delta")
    if delta is not None:
        if not isinstance(delta, Mapping):
            raise RawCompletionError("Claude stream delta is malformed")
        delta_type = delta.get("type")
        allowed_delta = {
            "text_delta": {"type", "text"},
            "thinking_delta": {"type", "thinking"},
            "signature_delta": {"type", "signature"},
            "input_json_delta": {"type", "partial_json"},
        }.get(str(delta_type or ""))
        if allowed_delta is None or set(delta) - allowed_delta:
            raise RawCompletionError("Claude stream delta contains unknown fields")
    if not _closed_json_value(dict(value)):
        raise RawCompletionError("Claude stream event is not closed JSON")


def _kimi(
    events: Sequence[Mapping[str, Any]], native_session_id: str | None,
) -> ValidatedRawCompletion:
    request_methods = {
        "initialize", "authenticate", "session/new", "session/resume",
        "session/set_config_option", "session/prompt", "session/cancel",
    }
    prompt_id: int | str | None = None
    prompt_ordinal = 0
    terminal_ordinal = 0
    chunks: list[str] = []
    seen_ids: set[int | str] = set()
    seen_response_ids: set[int | str] = set()
    request_methods_by_id: dict[int | str, str] = {}
    pending_request: int | str | None = None
    pending_reverse: tuple[int | str, str] | None = None
    phase = "initialize"
    observed_native: str | None = None
    for ordinal, event in enumerate(events, 1):
        if terminal_ordinal:
            raise RawCompletionError("Kimi raw stream contains an event after completion")
        if event.get("direction") == "request":
            if set(event) - {"direction", "id", "method", "sequence", "ordinal"}:
                raise RawCompletionError("Kimi request marker contains unknown fields")
            method = event.get("method")
            request_id = event.get("id")
            if (
                method not in request_methods
                or not _valid_kimi_rpc_id(request_id)
                or request_id in seen_ids
            ):
                raise RawCompletionError("Kimi request marker is unknown or duplicated")
            seen_ids.add(request_id)
            if pending_request is not None or pending_reverse is not None:
                raise RawCompletionError("Kimi request began before the prior response")
            expected = {
                "initialize": {"initialize"},
                "authenticate": {"authenticate"},
                "session": {"session/new", "session/resume"},
                "config_or_prompt": {"session/set_config_option", "session/prompt"},
                "prompt": set(),
            }[phase]
            if method not in expected:
                raise RawCompletionError("Kimi request lifecycle is invalid")
            request_methods_by_id[request_id] = str(method)
            pending_request = request_id
            if method == "session/prompt":
                if prompt_id is not None:
                    raise RawCompletionError("Kimi raw stream contains multiple prompt requests")
                prompt_id = request_id
                prompt_ordinal = ordinal
            continue
        if event.get("direction") == "reverse_response":
            if set(event) - {
                "direction", "id", "method", "disposition", "sequence", "ordinal",
            }:
                raise RawCompletionError("Kimi reverse response marker is malformed")
            reverse_id = event.get("id")
            reverse_method = event.get("method")
            if (
                pending_reverse != (reverse_id, reverse_method)
                or event.get("disposition")
                != {
                    "session/request_permission": "permission_cancelled",
                    "fs/read_text_file": "filesystem_denied",
                    "fs/write_text_file": "filesystem_denied",
                }.get(str(reverse_method or ""))
            ):
                raise RawCompletionError("Kimi reverse response does not close its request")
            pending_reverse = None
            continue
        if event.get("jsonrpc") != "2.0":
            raise RawCompletionError("Kimi raw stream contains an unknown event")
        if event.get("method") in {
            "session/request_permission", "fs/read_text_file", "fs/write_text_file",
        } and "id" in event:
            if not _valid_kimi_rpc_id(event.get("id")):
                raise RawCompletionError("Kimi reverse request ID is malformed")
            if set(event) - {"jsonrpc", "id", "method", "params", "sequence", "ordinal"}:
                raise RawCompletionError("Kimi reverse request payload is malformed")
            if pending_request != prompt_id or pending_reverse is not None:
                raise RawCompletionError("Kimi reverse request lifecycle is invalid")
            _validate_kimi_reverse_request(event)
            pending_reverse = (event["id"], str(event["method"]))
            continue
        if event.get("method") == "session/update":
            if set(event) - {"jsonrpc", "method", "params", "sequence", "ordinal"}:
                raise RawCompletionError("Kimi session update payload is malformed")
            if pending_reverse is not None:
                raise RawCompletionError(
                    "Kimi reverse request has no captured response closure"
                )
            if prompt_id is None or pending_request != prompt_id:
                raise RawCompletionError("Kimi session update occurred outside the prompt")
            params = event.get("params")
            update = params.get("update") if isinstance(params, Mapping) else None
            if not isinstance(update, Mapping) or not isinstance(params, Mapping):
                raise RawCompletionError("Kimi session update is malformed")
            if set(params) != {"sessionId", "update"}:
                raise RawCompletionError("Kimi session update params are not closed")
            candidate_native = str(params.get("sessionId") or "") or None
            if candidate_native is None:
                raise RawCompletionError("Kimi session update lacks a session")
            if observed_native not in {None, candidate_native}:
                raise RawCompletionError("Kimi raw stream contains conflicting sessions")
            observed_native = candidate_native
            if native_session_id and candidate_native != native_session_id:
                raise RawCompletionError("Kimi raw stream session changed")
            update_type = update.get("sessionUpdate")
            if update_type not in {
                "available_commands_update", "agent_thought_chunk",
                "agent_message_chunk", "plan", "plan_update",
                "current_mode_update", "tool_call", "tool_call_update",
            }:
                raise RawCompletionError("Kimi session update type is unknown")
            _validate_kimi_update(update)
            if update_type == "agent_message_chunk":
                content = update.get("content")
                if (
                    set(update) != {"sessionUpdate", "content"}
                    or not isinstance(content, Mapping)
                    or set(content) != {"type", "text"}
                    or content.get("type") != "text"
                    or not isinstance(content.get("text"), str)
                ):
                    raise RawCompletionError("Kimi message chunk is malformed")
                chunks.append(content["text"])
            continue
        if "id" in event and ("result" in event or "error" in event):
            if pending_reverse is not None:
                raise RawCompletionError("Kimi reverse request has no captured response closure")
            response_id = event.get("id")
            if (
                not _valid_kimi_rpc_id(response_id)
                or response_id not in seen_ids
                or response_id in seen_response_ids
            ):
                raise RawCompletionError("Kimi response marker is unknown or duplicated")
            if set(event) - {"jsonrpc", "id", "result", "error", "sequence", "ordinal"}:
                raise RawCompletionError("Kimi response payload contains unknown fields")
            if ("result" in event) == ("error" in event):
                raise RawCompletionError("Kimi response must contain exactly one outcome")
            if response_id != pending_request:
                raise RawCompletionError("Kimi response order does not close the pending request")
            seen_response_ids.add(response_id)
            method = request_methods_by_id[response_id]
            if "error" in event:
                raise RawCompletionError("Kimi request completed with an error")
            result = event.get("result")
            _validate_kimi_result(str(method), result)
            if method == "initialize":
                phase = "authenticate"
            elif method == "authenticate":
                phase = "session"
            elif method in {"session/new", "session/resume"}:
                returned_native = str(result.get("sessionId") or "") or None
                if method == "session/new" and returned_native is None:
                    raise RawCompletionError("Kimi session/new lacks a sessionId")
                if returned_native is not None:
                    if observed_native not in {None, returned_native}:
                        raise RawCompletionError("Kimi raw stream contains conflicting sessions")
                    observed_native = returned_native
                phase = "config_or_prompt"
            elif method == "session/set_config_option":
                phase = "config_or_prompt"
            if response_id == prompt_id:
                if not isinstance(result, Mapping) or result.get("stopReason") != "end_turn":
                    raise RawCompletionError("Kimi prompt did not complete an end_turn")
                terminal_ordinal = ordinal
                phase = "prompt"
            pending_request = None
            continue
        raise RawCompletionError("Kimi raw stream contains an unknown event")
    if pending_reverse is not None:
        raise RawCompletionError("Kimi reverse request has no captured response closure")
    if (
        prompt_id is None
        or terminal_ordinal != len(events)
        or terminal_ordinal <= prompt_ordinal
        or pending_request is not None
        or seen_response_ids != seen_ids
    ):
        raise RawCompletionError("Kimi raw stream lacks one final prompt completion")
    if native_session_id and observed_native != native_session_id:
        raise RawCompletionError("Kimi raw stream lacks the expected session")
    material = material_from_kimi("".join(chunks)) if chunks else ()
    return ValidatedRawCompletion(material, terminal_ordinal)


def _validate_kimi_reverse_request(event: Mapping[str, Any]) -> None:
    method = str(event.get("method") or "")
    params = event.get("params")
    if not isinstance(params, Mapping):
        raise RawCompletionError("Kimi reverse request params are malformed")
    expected_fields = {
        "session/request_permission": {"sessionId", "toolCall", "options"},
        "fs/read_text_file": {"sessionId", "path"},
        "fs/write_text_file": {"sessionId", "path", "content"},
    }[method]
    if set(params) != expected_fields or not isinstance(params.get("sessionId"), str):
        raise RawCompletionError("Kimi reverse request params are not closed")
    if method == "session/request_permission":
        tool_call = params.get("toolCall")
        options = params.get("options")
        if (
            not isinstance(tool_call, Mapping)
            or set(tool_call) != {"toolCallId", "status"}
            or not all(isinstance(tool_call[key], str) for key in tool_call)
            or not isinstance(options, list)
        ):
            raise RawCompletionError("Kimi permission request is malformed")
        for option in options:
            if (
                not isinstance(option, Mapping)
                or set(option) != {"optionId", "name", "kind"}
                or not all(isinstance(option[key], str) for key in option)
            ):
                raise RawCompletionError("Kimi permission option is malformed")
    elif not isinstance(params.get("path"), str):
        raise RawCompletionError("Kimi filesystem request path is malformed")
    if method == "fs/write_text_file" and not isinstance(params.get("content"), str):
        raise RawCompletionError("Kimi filesystem write content is malformed")


def _validate_kimi_update(update: Mapping[str, Any]) -> None:
    update_type = str(update.get("sessionUpdate") or "")
    allowed = {
        "available_commands_update": {
            "sessionUpdate", "availableCommands", "commands",
        },
        "agent_thought_chunk": {"sessionUpdate", "content"},
        "agent_message_chunk": {"sessionUpdate", "content"},
        "plan": {"sessionUpdate", "entries", "content"},
        "plan_update": {"sessionUpdate", "entries", "content"},
        "current_mode_update": {"sessionUpdate", "currentMode", "mode"},
        "tool_call": {
            "sessionUpdate", "toolCallId", "title", "kind", "status",
            "content", "locations", "rawInput", "rawOutput",
        },
        "tool_call_update": {
            "sessionUpdate", "toolCallId", "title", "kind", "status",
            "content", "locations", "rawInput", "rawOutput",
        },
    }.get(update_type)
    if allowed is None or set(update) - allowed or not _closed_json_value(dict(update)):
        raise RawCompletionError("Kimi session update payload is not closed")
    if update_type in {"agent_message_chunk", "agent_thought_chunk"}:
        content = update.get("content")
        if (
            not isinstance(content, Mapping)
            or set(content) != {"type", "text"}
            or content.get("type") != "text"
            or not isinstance(content.get("text"), str)
        ):
            raise RawCompletionError("Kimi text chunk is malformed")


def _validate_kimi_result(method: str, value: Any) -> None:
    if not isinstance(value, Mapping):
        raise RawCompletionError(f"Kimi {method} response is malformed")
    allowed = {
        "initialize": {
            "protocolVersion", "agentInfo", "agentCapabilities", "authMethods",
        },
        "authenticate": set(),
        "session/new": {"sessionId", "configOptions", "modes"},
        "session/resume": {"sessionId", "configOptions", "modes"},
        "session/set_config_option": {"configOptions", "modes"},
        "session/prompt": {"stopReason"},
        "session/cancel": set(),
    }.get(method)
    if allowed is None or set(value) - allowed:
        raise RawCompletionError(f"Kimi {method} response contains unknown fields")
    if method == "initialize":
        version = value.get("protocolVersion")
        if isinstance(version, bool) or not isinstance(version, (str, int)):
            raise RawCompletionError("Kimi initialize protocol version is malformed")
        agent_info = value.get("agentInfo")
        if agent_info is not None and (
            not isinstance(agent_info, Mapping)
            or set(agent_info) - {"name", "version", "title"}
            or not all(isinstance(item, str) for item in agent_info.values())
        ):
            raise RawCompletionError("Kimi initialize agent info is malformed")
        capabilities = value.get("agentCapabilities")
        if capabilities is not None:
            if (
                not isinstance(capabilities, Mapping)
                or set(capabilities) != {"sessionCapabilities"}
            ):
                raise RawCompletionError("Kimi initialize capabilities are malformed")
            session_capabilities = capabilities.get("sessionCapabilities")
            if (
                not isinstance(session_capabilities, Mapping)
                or set(session_capabilities) - {"resume"}
                or any(
                    not isinstance(capability, Mapping) or bool(capability)
                    for capability in session_capabilities.values()
                )
            ):
                raise RawCompletionError("Kimi session capabilities are malformed")
        auth_methods = value.get("authMethods")
        if auth_methods is not None:
            if not isinstance(auth_methods, list):
                raise RawCompletionError("Kimi initialize auth methods are malformed")
            for auth_method in auth_methods:
                if (
                    not isinstance(auth_method, Mapping)
                    or set(auth_method) - {"id", "type", "args", "name", "description"}
                    or not _closed_json_value(dict(auth_method))
                    or not isinstance(auth_method.get("id"), str)
                    or not isinstance(auth_method.get("type"), str)
                    or (
                        "args" in auth_method
                        and (
                            not isinstance(auth_method["args"], list)
                            or not all(isinstance(item, str) for item in auth_method["args"])
                        )
                    )
                ):
                    raise RawCompletionError("Kimi initialize auth method is malformed")
    if method in {"session/new", "session/resume"} and "sessionId" in value:
        if not isinstance(value.get("sessionId"), str) or not value.get("sessionId"):
            raise RawCompletionError("Kimi session response has an invalid sessionId")
    config_options = value.get("configOptions")
    if config_options is not None:
        if not isinstance(config_options, list):
            raise RawCompletionError("Kimi config options are malformed")
        for option in config_options:
            if (
                not isinstance(option, Mapping)
                or set(option) - {
                    "id", "name", "description", "type", "currentValue", "options",
                }
                or not _closed_json_value(dict(option))
            ):
                raise RawCompletionError("Kimi config option is malformed")
            choices = option.get("options")
            if choices is not None:
                if not isinstance(choices, list):
                    raise RawCompletionError("Kimi config option choices are malformed")
                for choice in choices:
                    if (
                        not isinstance(choice, Mapping)
                        or set(choice) - {"value", "name", "description"}
                        or not _closed_json_value(dict(choice))
                    ):
                        raise RawCompletionError("Kimi config option choice is malformed")
    if not _closed_json_value(dict(value)):
        raise RawCompletionError(f"Kimi {method} response is not closed JSON")


def _valid_kimi_rpc_id(value: Any) -> bool:
    """ACP request IDs are nonempty JSON-RPC string or integer scalars."""

    return (isinstance(value, str) and bool(value)) or type(value) is int
