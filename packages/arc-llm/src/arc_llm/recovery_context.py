from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema_cache import sha256_text
from .sessions import LLMSessionManager


@dataclass(frozen=True)
class LLMRecoveryContext:
    idempotency_key: str
    checkpoint_path: Path | None
    checkpoint_state: str | None
    submission_state: str | None
    native_session_id: str | None
    resumable: bool
    progress_journal: Path | None
    latest_progress: dict[str, Any] | None
    session_key: str | None
    generation: int | None


def read_recovery_context(
    artifact_dir: Path | str,
    *,
    idempotency_key: str,
    session_manager: LLMSessionManager | None = None,
    session_key: str | None = None,
) -> LLMRecoveryContext:
    """Read provider-neutral supervision context for one logical call."""

    root = Path(artifact_dir)
    candidate = root / "call-checkpoints" / f"idempotency-{sha256_text(idempotency_key)}.json"
    checkpoint_path = candidate if candidate.exists() else None
    checkpoint = _read_object(checkpoint_path) if checkpoint_path else None
    progress_path = root / "progress.jsonl"
    if checkpoint and checkpoint.get("progress_journal"):
        progress_path = Path(str(checkpoint["progress_journal"]))
    latest = _last_json_object(progress_path)
    response = checkpoint.get("response") if isinstance(checkpoint, dict) else None
    native_id = response.get("native_session_id") if isinstance(response, dict) else None
    if not native_id and latest:
        native_id = latest.get("native_session_id")
    generation = None
    if session_manager is not None and session_key:
        ref = session_manager.get_existing(session_key)
        if ref is not None:
            generation = ref.generation
            native_id = native_id or ref.native_session_id
    resumable = bool(
        (checkpoint and checkpoint.get("resumable"))
        or (latest and latest.get("resumable"))
        or native_id
    )
    return LLMRecoveryContext(
        idempotency_key=idempotency_key,
        checkpoint_path=checkpoint_path,
        checkpoint_state=str(checkpoint.get("state")) if checkpoint else None,
        submission_state=str(checkpoint.get("submission_state")) if checkpoint else None,
        native_session_id=str(native_id) if native_id else None,
        resumable=resumable,
        progress_journal=progress_path if progress_path.exists() else None,
        latest_progress=latest,
        session_key=session_key,
        generation=generation,
    )


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read recovery checkpoint {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"recovery checkpoint is not an object: {path}")
    return value


def _last_json_object(path: Path) -> dict[str, Any] | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None
