from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .io import read_json, write_json


SCHEMA_VERSION = "arc.companion.resume-transaction.v1"


def transaction_path(project_dir: Path) -> Path:
    return project_dir / ".arc-companion" / "resume-transaction.json"


def load_transaction(project_dir: Path) -> dict[str, Any] | None:
    path = transaction_path(project_dir)
    if not path.is_file():
        return None
    value = read_json(path)
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Invalid resume transaction journal: {path}")
    return value


def begin_transaction(
    project_dir: Path,
    *,
    action: str,
    recovery_options: Mapping[str, Any],
    entries: list[Mapping[str, Any]],
    native_resume_contexts: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    existing = load_transaction(project_dir)
    if existing and existing.get("status") != "complete":
        if existing.get("action") != action:
            raise ValueError("An incomplete resume transaction uses a different action")
        return existing
    value = {
        "schema_version": SCHEMA_VERSION,
        "action": action,
        "status": "applying",
        "recovery_options": dict(recovery_options),
        "entries": [{**dict(entry), "status": "pending"} for entry in entries],
        "native_resume_contexts": [
            dict(item) for item in (native_resume_contexts or [])
        ],
        "started_at": _now(),
        "updated_at": _now(),
    }
    write_json(transaction_path(project_dir), value)
    return value


def mark_entry(project_dir: Path, index: int, *, status: str, **receipt: Any) -> dict[str, Any]:
    value = load_transaction(project_dir)
    if value is None:
        raise ValueError("Resume transaction journal is missing")
    entries = list(value.get("entries") or [])
    entries[index] = {**dict(entries[index]), **receipt, "status": status}
    value.update({"entries": entries, "updated_at": _now()})
    write_json(transaction_path(project_dir), value)
    return value


def mark_transaction(project_dir: Path, status: str) -> dict[str, Any]:
    value = load_transaction(project_dir)
    if value is None:
        raise ValueError("Resume transaction journal is missing")
    value.update({"status": status, "updated_at": _now()})
    write_json(transaction_path(project_dir), value)
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
