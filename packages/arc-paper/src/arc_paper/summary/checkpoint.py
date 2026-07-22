from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from ..cache import CachePaths, content_lock, now_iso, read_json, write_json

CALL_CHECKPOINT_SCHEMA = "arc.llm.call_checkpoint.v1"
UNCERTAIN_RETRY_SECONDS = 3600
MAX_UNCERTAIN_ATTEMPTS = 2
_CALL_CONTEXT: ContextVar[tuple[Path, str] | None] = ContextVar("arc_paper_llm_call_context", default=None)
_SCHEMA_CANARY_ROOT: ContextVar[Path | None] = ContextVar(
    "arc_paper_schema_canary_root", default=None
)


class CallCheckpointUncertain(RuntimeError):
    pass


def run_json_checkpointed(
    *,
    paper_id: str,
    call_kind: str,
    identity: dict[str, Any],
    prompt: str,
    schema: dict[str, Any],
    model: str | None,
    run_json: Callable[[str, dict[str, Any], str | None], dict[str, Any]],
    validate: Callable[[dict[str, Any]], None],
    use_cache: bool = True,
) -> dict[str, Any]:
    """Persist a paid JSON response before local validation and replay it safely."""

    key_payload = {
        "call_kind": call_kind,
        "identity": identity,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "schema": schema,
        "model": model,
    }
    key_json = json.dumps(key_payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    key = hashlib.sha256(key_json.encode("utf-8")).hexdigest()
    prompt_sha256 = key_payload["prompt_sha256"]
    schema_sha256 = hashlib.sha256(
        json.dumps(schema, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    path = _checkpoint_path(paper_id, call_kind, key)
    with content_lock("llm-call-checkpoints", key):
        checkpoint = read_json(path)
        response = _replayable_response(checkpoint) if use_cache else None
        if response is not None:
            validate(response)
            if checkpoint.get("status") != "validated":
                write_json(path, {**checkpoint, "status": "validated", "validated_at": now_iso()})
            return response

        attempts = _next_attempt(checkpoint, path)
        started_at = now_iso()
        write_json(
            path,
            {
                "schema_version": CALL_CHECKPOINT_SCHEMA,
                "status": "started",
                "submission_state": "unknown",
                "key": key,
                "call_kind": call_kind,
                "identity": identity,
                "model": model,
                "prompt_sha256": prompt_sha256,
                "schema_sha256": schema_sha256,
                "attempts": attempts,
                "started_at": started_at,
                "updated_at": started_at,
            },
        )
        token: Token[tuple[Path, str] | None] = _CALL_CONTEXT.set((path.parent / "provider", f"{call_kind}-{key[:16]}"))
        try:
            response = run_json(prompt, schema, model)
        except BaseException as exc:
            # Ordinary returned failures are known outcomes. A hard process death leaves
            # `started`, which is deliberately quarantined before one recovery attempt.
            submission_state = _submission_state(exc)
            write_json(
                path,
                {
                    **(read_json(path) or {}),
                    "status": "failed" if submission_state == "not_submitted" else "uncertain",
                    "submission_state": submission_state,
                    "error": str(exc),
                    "updated_at": now_iso(),
                },
            )
            raise
        finally:
            _CALL_CONTEXT.reset(token)

        # This write is intentionally before schema/business validation.
        write_json(
            path,
            {
                **(read_json(path) or {}),
                "status": "response_received",
                "submission_state": "response_received",
                "response": response,
                "response_received_at": now_iso(),
                "updated_at": now_iso(),
            },
        )
        validate(response)
        write_json(
            path,
            {
                **(read_json(path) or {}),
                "status": "validated",
                "validated_at": now_iso(),
                "updated_at": now_iso(),
            },
        )
        return response


def current_provider_checkpoint() -> tuple[Path | None, str | None]:
    context = _CALL_CONTEXT.get()
    return context if context is not None else (None, None)


def current_schema_canary_root() -> Path | None:
    """Return the Controller-owned root shared by the active summary batch."""

    return _SCHEMA_CANARY_ROOT.get()


@contextmanager
def schema_canary_scope(root: Path):
    """Address every nested summary call to one batch-local schema proof root."""

    token = _SCHEMA_CANARY_ROOT.set(root)
    try:
        yield
    finally:
        _SCHEMA_CANARY_ROOT.reset(token)


def _checkpoint_path(paper_id: str, call_kind: str, key: str) -> Path:
    safe_kind = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in call_kind)
    return CachePaths.for_paper(paper_id).paper_dir / "llm-call-checkpoints" / safe_kind / f"{key}.json"


def _replayable_response(checkpoint: Any) -> dict[str, Any] | None:
    if not isinstance(checkpoint, dict):
        return None
    if checkpoint.get("status") not in {"response_received", "validated"}:
        return None
    response = checkpoint.get("response")
    return response if isinstance(response, dict) else None


def _next_attempt(checkpoint: Any, path: Path) -> int:
    if not isinstance(checkpoint, dict) or checkpoint.get("status") not in {"started", "uncertain"}:
        return 1
    attempts = max(1, int(checkpoint.get("attempts") or 1))
    started = _parse_time(checkpoint.get("started_at"))
    if started is not None and datetime.now(timezone.utc) < started + timedelta(seconds=UNCERTAIN_RETRY_SECONDS):
        raise CallCheckpointUncertain(
            f"LLM call outcome is uncertain; retry is quarantined for {UNCERTAIN_RETRY_SECONDS} seconds: {path}"
        )
    if attempts >= MAX_UNCERTAIN_ATTEMPTS:
        raise CallCheckpointUncertain(f"LLM call remained uncertain after recovery attempt: {path}")
    return attempts + 1


def _parse_time(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _submission_state(exc: BaseException) -> str:
    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        value = getattr(current, "submission_state", None)
        if value:
            return str(getattr(value, "value", value))
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None and current.__context__ is not current.__cause__:
            pending.append(current.__context__)
    return "unknown"
