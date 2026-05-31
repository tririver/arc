from __future__ import annotations

import json
import os
import socket
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, RLock, get_ident
from typing import Any, Iterator, Mapping, Sequence

from .schema_cache import sha256_text


SessionPolicy = str


@dataclass(frozen=True)
class LLMSessionRef:
    key: str
    provider: str
    model: str | None
    runtime_fingerprint: str
    native_session_id: str | None = None
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    generation: int = 1


@dataclass(frozen=True)
class LLMSessionTurn:
    call_label: str
    prompt_sha256: str
    static_prefix_sha256: str | None
    schema_sha256: str | None
    usage: dict[str, Any]
    created_at: str


_PROCESS_LOCKS: dict[Path, RLock] = {}
_PROCESS_LOCKS_GUARD = Lock()
_HELD_FILE_LOCKS: dict[tuple[Path, int, int], int] = {}
_HELD_FILE_LOCKS_GUARD = Lock()


class LLMSessionManager:
    def __init__(self, root: Path | str):
        self.root = Path(root).expanduser()
        self.sessions_path = self.root / "sessions.json"
        self.calls_path = self.root / "calls.jsonl"
        self.root.mkdir(parents=True, exist_ok=True)
        self._state_lock = Lock()
        self._sessions = self._load_sessions()

    def get_or_create(
        self,
        *,
        key: str,
        provider: str,
        model: str | None,
        runtime_fingerprint: str,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> LLMSessionRef:
        with self.lock(key):
            with self._state_lock:
                self._reload_sessions_from_disk_locked()
                return self._get_or_create_locked(
                    key=key,
                    provider=provider,
                    model=model,
                    runtime_fingerprint=runtime_fingerprint,
                    name=name,
                    metadata=metadata,
                )

    def update_native_session_id(
        self,
        key: str,
        native_session_id: str,
        *,
        allow_overwrite: bool = False,
    ) -> LLMSessionRef:
        with self.lock(key):
            with self._state_lock:
                self._reload_sessions_from_disk_locked()
                ref = self._require_ref(key)
                if ref.native_session_id and ref.native_session_id != native_session_id and not allow_overwrite:
                    raise ValueError(f"session native_session_id changed for {key}")
                if ref.native_session_id == native_session_id:
                    return ref
                updated = LLMSessionRef(
                    key=ref.key,
                    provider=ref.provider,
                    model=ref.model,
                    runtime_fingerprint=ref.runtime_fingerprint,
                    native_session_id=native_session_id,
                    name=ref.name,
                    metadata=dict(ref.metadata),
                    generation=ref.generation,
                )
                self._sessions[key] = updated
                self._write_sessions()
                return updated

    def reload(self) -> None:
        with self._state_lock:
            self._reload_sessions_from_disk_locked()

    def record_turn(
        self,
        key: str,
        *,
        call_label: str,
        prompt_sha256: str,
        static_prefix_sha256: str | None,
        schema_sha256: str | None,
        usage: Mapping[str, Any],
        provider_used: str,
        model_used: str | None,
        native_session_id: str | None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        item = {
            "session_key": key,
            "call_label": call_label,
            "prompt_sha256": prompt_sha256,
            "static_prefix_sha256": static_prefix_sha256,
            "schema_sha256": schema_sha256,
            "usage": dict(usage),
            "provider_used": provider_used,
            "model_used": model_used,
            "native_session_id": native_session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            item.update(dict(extra))
        with self.lock(key):
            _append_jsonl(self.calls_path, item)

    def turn_count(self, key: str) -> int:
        try:
            lines = self.calls_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return 0
        count = 0
        for line in lines:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and payload.get("session_key") == key:
                count += 1
        return count

    def has_native_session(self, key: str) -> bool:
        with self._state_lock:
            self._reload_sessions_from_disk_locked()
            ref = self._sessions.get(key)
            return bool(ref and ref.native_session_id)

    @contextmanager
    def lock(self, key: str) -> Iterator[None]:
        safe = _safe_lock_name(key)
        lock_path = self.root / "locks" / f"{safe}.lock"
        process_lock = _process_lock(lock_path)
        with process_lock:
            with _file_lock(lock_path):
                yield

    @contextmanager
    def locked_turn(
        self,
        *,
        key: str,
        provider: str,
        model: str | None,
        runtime_fingerprint: str,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Iterator[tuple[LLMSessionRef, int]]:
        with self.lock(key):
            with self._state_lock:
                self._reload_sessions_from_disk_locked()
                ref = self._get_or_create_locked(
                    key=key,
                    provider=provider,
                    model=model,
                    runtime_fingerprint=runtime_fingerprint,
                    name=name,
                    metadata=metadata,
                )
            yield ref, self.turn_count(key)

    def _get_or_create_locked(
        self,
        *,
        key: str,
        provider: str,
        model: str | None,
        runtime_fingerprint: str,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> LLMSessionRef:
        existing = self._sessions.get(key)
        if existing is not None:
            if existing.provider != provider or existing.model != model:
                raise ValueError(f"session provider/model changed for {key}")
            if existing.runtime_fingerprint != runtime_fingerprint:
                raise ValueError(f"session runtime fingerprint changed for {key}")
            return existing
        ref = LLMSessionRef(
            key=key,
            provider=provider,
            model=model,
            runtime_fingerprint=runtime_fingerprint,
            name=name,
            metadata=dict(metadata or {}),
        )
        self._sessions[key] = ref
        self._write_sessions()
        return ref

    def _require_ref(self, key: str) -> LLMSessionRef:
        ref = self._sessions.get(key)
        if ref is None:
            raise KeyError(f"unknown LLM session key: {key}")
        return ref

    def _reload_sessions_from_disk_locked(self) -> None:
        self._sessions = self._load_sessions()

    def _load_sessions(self) -> dict[str, LLMSessionRef]:
        try:
            payload = json.loads(self.sessions_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        sessions = payload.get("sessions") if isinstance(payload, dict) else None
        if not isinstance(sessions, dict):
            return {}
        result: dict[str, LLMSessionRef] = {}
        for key, item in sessions.items():
            if isinstance(item, dict):
                result[str(key)] = LLMSessionRef(
                    key=str(item.get("key") or key),
                    provider=str(item.get("provider") or ""),
                    model=item.get("model") if item.get("model") is None else str(item.get("model")),
                    runtime_fingerprint=str(item.get("runtime_fingerprint") or ""),
                    native_session_id=item.get("native_session_id")
                    if item.get("native_session_id") is None
                    else str(item.get("native_session_id")),
                    name=item.get("name") if item.get("name") is None else str(item.get("name")),
                    metadata=dict(item.get("metadata") or {}),
                    generation=int(item.get("generation") or 1),
                )
        return result

    def _write_sessions(self) -> None:
        payload = {"schema_version": "arc.llm.sessions.v1", "sessions": {k: asdict(v) for k, v in self._sessions.items()}}
        _atomic_write_json(self.sessions_path, payload)


def runtime_fingerprint(
    *,
    provider: str,
    model: str | None,
    model_tier: str | None,
    env: Mapping[str, str] | None,
    process_chain: Sequence[str] | None = None,
) -> str:
    env = env or {}
    interesting = {
        key: env.get(key)
        for key in sorted(
            {
                "ARC_CODEX_SANDBOX",
                "ARC_CODEX_CONFIG",
                "ARC_CODEX_CONFIG_JSON",
                "ARC_CODEX_HISTORY_PERSISTENCE",
                "ARC_CODEX_EPHEMERAL",
                "ARC_CODEX_WORK_DIR",
                "ARC_CODEX_ADD_DIRS",
                "ARC_CODEX_PROFILE",
                "ARC_CODEX_PROFILE_V2",
                "ARC_CODEX_ENABLE_MCP",
                "ARC_CODEX_MCP_MODE",
                "ARC_CODEX_ARC_MCP_COMMAND",
                "ARC_CODEX_ARC_MCP_ENV_JSON",
                "ARC_CODEX_ALLOW_INTERNET",
                "ARC_CODEX_NETWORK_ACCESS",
                "ARC_CODEX_WEB_SEARCH",
                "ARC_CODEX_REASONING_EFFORT",
                "ARC_CODEX_REASONING_SUMMARY",
                "ARC_CODEX_MODEL_VERBOSITY",
                "ARC_CODEX_IGNORE_USER_CONFIG",
                "ARC_CODEX_IGNORE_RULES",
                "ARC_CLAUDE_TOOLS",
                "ARC_CLAUDE_ALLOW_MCP",
                "ARC_CLAUDE_MCP_MODE",
                "ARC_CLAUDE_MCP_CONFIG",
                "ARC_CLAUDE_MCP_CONFIG_JSON",
                "ARC_CLAUDE_STRICT_MCP_CONFIG",
                "ARC_CLAUDE_ARC_MCP_COMMAND",
                "ARC_CLAUDE_ARC_MCP_ARGS_JSON",
                "ARC_CLAUDE_ARC_MCP_ENV_JSON",
                "ARC_CLAUDE_ARC_MCP_CONFIG_PATH",
                "ARC_CLAUDE_TEXT_OUTPUT_FORMAT_JSON",
                "ARC_CLAUDE_EFFORT",
                "ARC_CLAUDE_BARE",
                "ARC_CLAUDE_EXCLUDE_DYNAMIC_SYSTEM_PROMPT_SECTIONS",
                "ARC_CLAUDE_ALLOW_INTERNET",
            }
        )
        if env.get(key) is not None
    }
    payload = {
        "provider": provider,
        "model": model,
        "model_tier": model_tier,
        "env": interesting,
        "file_hashes": _runtime_file_hashes(env),
        "process_chain": list(process_chain or []),
    }
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _runtime_file_hashes(env: Mapping[str, str]) -> dict[str, list[dict[str, str | None]]]:
    paths_by_key = {
        "ARC_CLAUDE_MCP_CONFIG": _newline_paths(env.get("ARC_CLAUDE_MCP_CONFIG")),
        "ARC_CLAUDE_MCP_CONFIG_JSON": _json_paths(env.get("ARC_CLAUDE_MCP_CONFIG_JSON")),
        "ARC_CLAUDE_ARC_MCP_CONFIG_PATH": _single_path(env.get("ARC_CLAUDE_ARC_MCP_CONFIG_PATH")),
    }
    result: dict[str, list[dict[str, str | None]]] = {}
    for key, paths in paths_by_key.items():
        if not paths:
            continue
        result[key] = [_file_hash_entry(path) for path in paths]
    return result


def _single_path(value: str | None) -> list[str]:
    return [value.strip()] if value and value.strip() else []


def _newline_paths(value: str | None) -> list[str]:
    if value is None:
        return []
    return [line.strip() for line in value.splitlines() if line.strip()]


def _json_paths(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, str) and item.strip()]
    if isinstance(parsed, str) and parsed.strip():
        return [parsed.strip()]
    return []


def _file_hash_entry(path_text: str) -> dict[str, str | None]:
    path = Path(path_text).expanduser()
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        return {"path": str(path), "sha256": None, "error": type(exc).__name__}
    return {"path": str(path), "sha256": sha256_text(content), "error": None}


def _safe_lock_name(key: str) -> str:
    return sha256_text(key)[:32]


def _hostname() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return ""


def _process_lock(lock_path: Path) -> RLock:
    resolved = lock_path.resolve()
    with _PROCESS_LOCKS_GUARD:
        lock = _PROCESS_LOCKS.get(resolved)
        if lock is None:
            lock = RLock()
            _PROCESS_LOCKS[resolved] = lock
        return lock


@contextmanager
def _file_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    resolved = lock_path.resolve()
    held_key = (resolved, os.getpid(), get_ident())
    already_held = False
    with _HELD_FILE_LOCKS_GUARD:
        held_count = _HELD_FILE_LOCKS.get(held_key, 0)
        if held_count:
            _HELD_FILE_LOCKS[held_key] = held_count + 1
            already_held = True
    if already_held:
        try:
            yield
        finally:
            with _HELD_FILE_LOCKS_GUARD:
                remaining = _HELD_FILE_LOCKS[held_key] - 1
                if remaining:
                    _HELD_FILE_LOCKS[held_key] = remaining
                else:
                    del _HELD_FILE_LOCKS[held_key]
        return
    payload = {
        "pid": os.getpid(),
        "thread_id": get_ident(),
        "host": _hostname(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    while True:
        try:
            fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            break
        except FileExistsError:
            if not _recover_dead_process_lock(lock_path):
                time.sleep(0.01)
    try:
        with _HELD_FILE_LOCKS_GUARD:
            _HELD_FILE_LOCKS[held_key] = 1
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        unlink = False
        with _HELD_FILE_LOCKS_GUARD:
            remaining = _HELD_FILE_LOCKS.get(held_key, 1) - 1
            if remaining:
                _HELD_FILE_LOCKS[held_key] = remaining
            else:
                _HELD_FILE_LOCKS.pop(held_key, None)
                unlink = True
        if unlink:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def _recover_dead_process_lock(lock_path: Path) -> bool:
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("host") != _hostname():
        return False
    pid = payload.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return False
    except ProcessLookupError:
        try:
            lock_path.unlink()
            return True
        except OSError:
            return False
    except PermissionError:
        return False


def _append_jsonl(path: Path, item: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write_json(path: Path, data: Any) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{get_ident()}.{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)
