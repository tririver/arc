from __future__ import annotations

import json
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from .schema_cache import sha256_text
from .secure_io import SecureReadError, read_bounded_file, read_bounded_json, safe_relative_path
from .sessions import LLMSessionManager


_MAX_RECOVERY_CHECKPOINT_BYTES = 16 * 1024 * 1024
_MAX_RECOVERY_PROGRESS_BYTES = 16 * 1024 * 1024


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
    provider: str | None
    model: str | None
    runtime_fingerprint: str | None


def read_recovery_context(
    artifact_dir: Path | str,
    *,
    idempotency_key: str,
    session_manager: LLMSessionManager | None = None,
    session_key: str | None = None,
) -> LLMRecoveryContext:
    """Read provider-neutral supervision context for one logical call."""

    root = Path(os.path.abspath(os.fspath(Path(artifact_dir).expanduser())))
    candidate = root / "call-checkpoints" / f"idempotency-{sha256_text(idempotency_key)}.json"
    checkpoint = _read_object(root, candidate)
    checkpoint_path = candidate if checkpoint is not None else None
    progress_path = root / "progress.jsonl"
    if checkpoint and checkpoint.get("progress_journal"):
        progress_path = _contained_recovery_path(
            root, checkpoint["progress_journal"], suffixes=(".jsonl",),
        )
    session_ref = session_manager.get_existing(session_key) if session_manager and session_key else None
    expected = {
        "idempotency_key": idempotency_key,
        "session_key": session_key,
        "generation": session_ref.generation if session_ref is not None else None,
        "provider": session_ref.provider if session_ref is not None else None,
        "model": session_ref.model if session_ref is not None else None,
        "runtime_fingerprint": session_ref.runtime_fingerprint if session_ref is not None else None,
    }
    latest, progress_present = _last_matching_json_object(
        root, progress_path, expected=expected,
    )
    response = checkpoint.get("response") if isinstance(checkpoint, dict) else None
    native_id = response.get("native_session_id") if isinstance(response, dict) else None
    if not native_id and latest:
        native_id = latest.get("native_session_id")
    generation = None
    provider = str(latest.get("provider")) if latest and latest.get("provider") else None
    model = None
    runtime_fp = None
    if session_manager is not None and session_key:
        ref = session_ref
        if ref is not None:
            generation = ref.generation
            native_id = native_id or ref.native_session_id
            provider = provider or ref.provider
            model = ref.model
            runtime_fp = ref.runtime_fingerprint
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
        progress_journal=progress_path if progress_present else None,
        latest_progress=latest,
        session_key=session_key,
        generation=generation,
        provider=provider,
        model=model,
        runtime_fingerprint=runtime_fp,
    )


def _read_object(root: Path, path: Path) -> dict[str, Any] | None:
    try:
        relative = path.relative_to(root)
        value = read_bounded_json(
            root, relative,
            max_bytes=_MAX_RECOVERY_CHECKPOINT_BYTES,
            suffixes=(".json",),
        )
    except SecureReadError as exc:
        # Safe absence is the common first-call case. Anything present but
        # unsafe/corrupt is surfaced instead of being trusted as recovery data.
        try:
            os.lstat(path)
        except FileNotFoundError:
            return None
        raise ValueError(f"could not read recovery checkpoint {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"recovery checkpoint is not an object: {path}")
    return value


def _last_matching_json_object(
    root: Path, path: Path, *, expected: dict[str, Any]
) -> tuple[dict[str, Any] | None, bool]:
    try:
        raw = read_bounded_file(
            root, path.relative_to(root),
            max_bytes=_MAX_RECOVERY_PROGRESS_BYTES,
            suffixes=(".jsonl",),
        )
        lines = raw.decode("utf-8").splitlines()
    except (SecureReadError, UnicodeError):
        try:
            os.lstat(path)
        except FileNotFoundError:
            return None, False
        raise ValueError(f"could not read recovery progress journal {path}")
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and all(
            expected_value is None or value.get(key) == expected_value
            for key, expected_value in expected.items()
        ):
            return value, True
    return None, True


def _contained_recovery_path(
    root: Path, value: Any, *, suffixes: tuple[str, ...],
) -> Path:
    supplied = Path(str(value or ""))
    lexical = (
        supplied if supplied.is_absolute()
        else root / safe_relative_path(str(value or ""), suffixes=suffixes)
    )
    lexical = Path(os.path.abspath(os.fspath(lexical.expanduser())))
    try:
        relative = lexical.relative_to(root)
        safe_relative_path(relative.as_posix(), suffixes=suffixes)
    except (ValueError, SecureReadError) as exc:
        raise ValueError("recovery progress journal escapes its artifact root") from exc
    return root / relative
