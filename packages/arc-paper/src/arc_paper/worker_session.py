from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import fcntl

from .cache import now_iso
from .ids import normalize_paper_id


SCHEMA_VERSION = "arc.paper.worker-session.v1"
_STATE_DIR = ".arc-paper-worker"
_SECRET_KEYS = ("authorization", "credential", "password", "secret", "token", "api_key", "api-key")
_TERMINAL_FETCH_STATUS = {401, 403, 429}
_SAFE_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_ALLOWED_CACHE_NAMESPACES = {
    "papers", "sources", "rich-sources", "source-annotations", "paper-aliases", "queries"
}


class WorkerSessionError(RuntimeError):
    """Base error for worker-cache session failures."""


class CachedFetchError(WorkerSessionError):
    """A fetch was suppressed because the run already saw a terminal failure."""


@dataclass(frozen=True)
class PromotionResult:
    promoted: tuple[str, ...] = ()
    deduplicated: tuple[str, ...] = ()
    conflicted: tuple[str, ...] = ()
    quarantined: tuple[str, ...] = ()
    deleted: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "promoted": list(self.promoted),
            "deduplicated": list(self.deduplicated),
            "conflicted": list(self.conflicted),
            "quarantined": list(self.quarantined),
            "deleted": list(self.deleted),
        }


class WorkerCacheSession:
    """Run-scoped writable overlay over a read-only ARC paper cache.

    The overlay is shared by all workers in one run.  ``environment`` returns
    the variables needed by an ordinary ``arc-paper`` subprocess.  Callers
    should invoke :meth:`finish_call` even for failed or cancelled model calls;
    it records an audit event and promotes every valid staged artifact.
    """

    def __init__(
        self,
        *,
        base_root: str | Path,
        run_root: str | Path,
        session_id: str,
        max_parallel_fetches: int = 4,
    ) -> None:
        if not _SAFE_SESSION_ID.fullmatch(session_id) or session_id in {".", ".."} or ".." in session_id:
            raise ValueError("session_id must be a safe 1-128 character identifier")
        if max_parallel_fetches < 1:
            raise ValueError("max_parallel_fetches must be positive")
        self.base_root = Path(base_root).expanduser().resolve()
        self.run_root = Path(run_root).expanduser().resolve()
        self.session_id = session_id
        self.max_parallel_fetches = max_parallel_fetches
        self.overlay_root = self.run_root / "paper-cache-overlay"
        self.state_root = self.overlay_root / _STATE_DIR
        self.tombstone_root = self.state_root / "tombstones"
        self.record_root = self.state_root / "records"
        self.quarantine_root = self.state_root / "quarantine"
        self.fetch_root = self.state_root / "fetches"
        self.audit_path = self.state_root / "audit.jsonl"
        for directory in (
            self.overlay_root,
            self.tombstone_root,
            self.record_root,
            self.quarantine_root,
            self.fetch_root,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self._write_session_manifest()

    @classmethod
    def from_environment(cls) -> "WorkerCacheSession | None":
        """Reopen the controller-created session described by worker env vars."""

        overlay_value = os.environ.get("ARC_PAPER_CACHE")
        if not overlay_value:
            return None
        session = cls.open_or_create_from_environment()
        if session is None:
            return None
        overlay = Path(overlay_value).expanduser().resolve()
        if session.overlay_root != overlay:
            raise WorkerSessionError("worker session overlay path mismatch")
        return session

    @classmethod
    def open_or_create_from_environment(cls) -> "WorkerCacheSession | None":
        """Open/create a session from controller paths, before overlay activation.

        ``arc-llm`` cannot import ``arc-paper`` without creating a package cycle,
        so it supplies only the base cache, run/session directory, and session
        ID.  The worker wrapper uses this constructor and then ``activated()``.
        A completely absent configuration means paper sessions are disabled;
        a partial configuration fails closed.
        """

        names = (
            "ARC_PAPER_WORKER_BASE_CACHE",
            "ARC_PAPER_WORKER_SESSION_DIR",
            "ARC_PAPER_WORKER_SESSION_ID",
        )
        values = {name: os.environ.get(name) for name in names}
        if not any(values.values()):
            return None
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise WorkerSessionError(f"incomplete worker session environment: missing {', '.join(missing)}")
        run_root = Path(str(values["ARC_PAPER_WORKER_SESSION_DIR"])).expanduser().resolve()
        expected_overlay = run_root / "paper-cache-overlay"
        overlay_value = os.environ.get("ARC_PAPER_CACHE")
        if overlay_value and Path(overlay_value).expanduser().resolve() != expected_overlay:
            raise WorkerSessionError("ARC_PAPER_CACHE does not match the worker session overlay")
        manifest_path = expected_overlay / _STATE_DIR / "session.json"
        manifest_exists = manifest_path.exists()
        manifest = _read_json(manifest_path)
        max_fetches = int(manifest.get("max_parallel_fetches") or 4) if isinstance(manifest, dict) else 4
        session = cls(
            base_root=str(values["ARC_PAPER_WORKER_BASE_CACHE"]),
            run_root=run_root,
            session_id=str(values["ARC_PAPER_WORKER_SESSION_ID"]),
            max_parallel_fetches=max_fetches,
        )
        if manifest_exists:
            if not isinstance(manifest, dict):
                raise WorkerSessionError("worker session manifest is invalid JSON")
            expected = {
                "schema_version": SCHEMA_VERSION,
                "session_id": session.session_id,
                "base_root": str(session.base_root),
                "overlay_root": str(session.overlay_root),
            }
            if any(manifest.get(key) != value for key, value in expected.items()):
                raise WorkerSessionError("worker session manifest does not match the environment")
        return session

    def environment(self) -> dict[str, str]:
        return {
            "ARC_PAPER_CACHE": str(self.overlay_root),
            "ARC_PAPER_WORKER_BASE_CACHE": str(self.base_root),
            "ARC_PAPER_WORKER_TOMBSTONE_DIR": str(self.tombstone_root),
            "ARC_PAPER_WORKER_SESSION_ID": self.session_id,
            "ARC_PAPER_WORKER_SESSION_DIR": str(self.run_root),
        }

    @contextmanager
    def activated(self) -> Iterator[None]:
        """Temporarily activate this session in the current process.

        This helper is intended for serial controller code and tests.  Worker
        subprocesses should receive :meth:`environment` instead.
        """

        values = self.environment()
        previous = {key: os.environ.get(key) for key in values}
        os.environ.update(values)
        try:
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    @contextmanager
    def call_scope(self, call_id: str) -> Iterator[None]:
        if not _SAFE_SESSION_ID.fullmatch(call_id) or call_id in {".", ".."} or ".." in call_id:
            raise WorkerSessionError("call_id must be a safe 1-128 character identifier")
        key = "ARC_PAPER_WORKER_CALL_ID"
        previous = os.environ.get(key)
        os.environ[key] = call_id
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous

    def overlay_path(self, relative: str | Path) -> Path:
        return self.overlay_root / self._safe_relative(relative)

    def read_path(self, relative: str | Path) -> Path | None:
        relative_path = self._safe_relative(relative)
        overlay = self.overlay_root / relative_path
        if overlay.exists():
            return overlay
        if self._tombstone_path(relative_path).exists():
            return None
        base = self.base_root / relative_path
        return base if base.exists() else None

    def stage_bytes(
        self,
        relative: str | Path,
        content: bytes,
        *,
        source: Mapping[str, Any] | None = None,
        parser_version: str | int | None = None,
    ) -> Path:
        relative_path = self._safe_relative(relative)
        target = self.overlay_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._artifact_lock(relative_path):
            self._atomic_write(target, content)
            self._tombstone_path(relative_path).unlink(missing_ok=True)
            self._record_artifact(
                relative_path,
                source=source or {"operation": "stage_bytes"},
                parser_version=parser_version,
                writer_call_id=os.environ.get("ARC_PAPER_WORKER_CALL_ID", "session-direct"),
                operation="stage_bytes",
            )
        return target

    def tombstone(self, relative: str | Path, *, source: Mapping[str, Any] | None = None) -> None:
        relative_path = self._safe_relative(relative)
        with self._artifact_lock(relative_path):
            overlay = self.overlay_root / relative_path
            if overlay.exists():
                trash = self.state_root / "trash" / relative_path
                trash.parent.mkdir(parents=True, exist_ok=True)
                os.replace(overlay, trash)
            data = {
                "schema_version": SCHEMA_VERSION,
                "session_id": self.session_id,
                "relative_path": relative_path.as_posix(),
                "created_at": now_iso(),
                "source": _redact(dict(source or {})),
                "writer_call_id": os.environ.get("ARC_PAPER_WORKER_CALL_ID", "session-direct"),
                "operation": str((source or {}).get("operation") or "cache remove"),
            }
            self._atomic_json(self._tombstone_path(relative_path), data)

    def finish_call(
        self,
        *,
        worker_id: str,
        call_id: str,
        operation: str,
        status: str,
        paper_ids: list[str] | tuple[str, ...] = (),
        parameters: Mapping[str, Any] | None = None,
        source: Mapping[str, Any] | None = None,
        artifact_hash: str = "",
        result_hash: str = "",
    ) -> PromotionResult:
        """Record, validate, and promote artifacts after any call outcome."""

        self._validate_call_id(call_id)
        source_record = {
            "worker_id": worker_id,
            "call_id": call_id,
            "operation": operation,
            **dict(source or {}),
        }
        self._enrich_call_records(call_id, operation, source_record)
        result = self.promote()
        self.audit(
            worker_id=worker_id, call_id=call_id, operation=operation, status=status,
            paper_ids=paper_ids, parameters=parameters, source=source,
            artifact_hash=artifact_hash, result_hash=result_hash,
            promotion=result, promotion_status="complete",
        )
        return result

    def record_call(
        self,
        *,
        worker_id: str,
        call_id: str,
        operation: str,
        status: str,
        paper_ids: list[str] | tuple[str, ...] = (),
        parameters: Mapping[str, Any] | None = None,
        source: Mapping[str, Any] | None = None,
        artifact_hash: str = "",
        result_hash: str = "",
    ) -> None:
        """Attribute this call's completed writes and audit them as pending promotion."""
        self._validate_call_id(call_id)
        source_record = {
            "worker_id": worker_id,
            "call_id": call_id,
            "operation": operation,
            **dict(source or {}),
        }
        self._enrich_call_records(call_id, operation, source_record)
        self.audit(
            worker_id=worker_id,
            call_id=call_id,
            operation=operation,
            status=status,
            paper_ids=paper_ids,
            parameters=parameters,
            source=source,
            artifact_hash=artifact_hash,
            result_hash=result_hash,
            promotion_status="pending",
        )

    def promote(self) -> PromotionResult:
        promoted: list[str] = []
        deduplicated: list[str] = []
        conflicted: list[str] = []
        quarantined: list[str] = []
        deleted: list[str] = []
        with self._locked(self.base_root / ".arc-paper-worker-locks" / "promotion.lock"):
            for path in sorted(self._artifact_files()):
                relative = path.relative_to(self.overlay_root)
                with self._artifact_lock(relative):
                    error = self._validation_error(relative, path)
                    if error:
                        self._quarantine(relative, path, error)
                        quarantined.append(relative.as_posix())
                        continue
                    payload = self._promotion_bytes(path)
                    digest = hashlib.sha256(payload).hexdigest()
                    target = self.base_root / relative
                    if target.exists():
                        if target.is_file() and _sha256_file(target) == digest:
                            deduplicated.append(relative.as_posix())
                            path.unlink()
                        else:
                            conflict = self.base_root / ".arc-paper-worker-conflicts" / self.session_id / relative
                            conflict = conflict.with_name(f"{conflict.name}.{digest}")
                            conflict.parent.mkdir(parents=True, exist_ok=True)
                            self._write_atomic_bytes(conflict, payload)
                            self._write_conflict_record(relative, digest, target, conflict)
                            conflicted.append(relative.as_posix())
                            path.unlink()
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    self._write_atomic_bytes(target, payload)
                    promoted.append(relative.as_posix())
                    path.unlink()

            for marker in sorted(self.tombstone_root.glob("*.json")):
                data = _read_json(marker)
                if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
                    continue
                try:
                    relative = self._safe_relative(str(data["relative_path"]))
                except (KeyError, TypeError, ValueError):
                    continue
                if (
                    marker != self._tombstone_path(relative)
                    or data.get("session_id") != self.session_id
                    or not data.get("writer_call_id")
                    or not data.get("operation")
                    or not isinstance(data.get("source"), dict)
                    or not data.get("source")
                ):
                    continue
                with self._artifact_lock(relative):
                    target = self.base_root / relative
                    if target.exists():
                        trash = self.base_root / ".arc-paper-worker-trash" / self.session_id / relative
                        trash.parent.mkdir(parents=True, exist_ok=True)
                        if trash.exists():
                            trash = trash.with_name(f"{trash.name}.{time.time_ns()}")
                        os.replace(target, trash)
                    deleted.append(relative.as_posix())
                    marker.unlink()
        return PromotionResult(
            tuple(promoted), tuple(deduplicated), tuple(conflicted), tuple(quarantined), tuple(deleted)
        )

    def audit(
        self,
        *,
        worker_id: str,
        call_id: str,
        operation: str,
        status: str,
        paper_ids: list[str] | tuple[str, ...] = (),
        parameters: Mapping[str, Any] | None = None,
        source: Mapping[str, Any] | None = None,
        artifact_hash: str = "",
        result_hash: str = "",
        promotion: PromotionResult | None = None,
        promotion_status: str = "complete",
    ) -> None:
        event = {
            "schema_version": SCHEMA_VERSION,
            "timestamp": now_iso(),
            "session_id": self.session_id,
            "worker_id": worker_id,
            "call_id": call_id,
            "operation": operation,
            "paper_ids": [normalize_paper_id(item) for item in paper_ids],
            "parameters": _redact(dict(parameters or {})),
            "status": status,
            "source": _redact(dict(source or {})),
            "artifact_hash": artifact_hash,
            "result_hash": result_hash,
            "promotion": (promotion or PromotionResult()).as_dict(),
            "promotion_status": promotion_status,
        }
        payload = json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
        with self._locked(self.state_root / "audit.lock"):
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())

    def fetch_once(
        self,
        paper_id: str,
        fetcher: Callable[[], Any],
        *,
        operation: str = "paper",
        replay_success: bool = True,
    ) -> Any:
        """Run one fetch per canonical paper ID, with four run-wide slots by default.

        Successful JSON-compatible results are replayed to later workers.  HTTP
        401/403/429-like failures are recorded and fail closed for the rest of
        the run, preventing hidden retry amplification.
        """

        canonical_id = normalize_paper_id(paper_id)
        operation_key = str(operation or "paper").strip().lower()
        canonical_key = hashlib.sha256(canonical_id.encode("utf-8")).hexdigest()
        operation_key_hash = hashlib.sha256(
            f"{operation_key}\0{canonical_id}".encode("utf-8")
        ).hexdigest()
        marker = self.fetch_root / "results" / f"{operation_key_hash}.json"
        terminal_marker = self.fetch_root / "terminal" / f"{canonical_key}.json"
        with self._locked(self.state_root / "fetch-locks" / f"{canonical_key}.lock"):
            terminal = _read_json(terminal_marker)
            if isinstance(terminal, dict):
                raise CachedFetchError(str(terminal.get("error") or "terminal fetch failure"))
            prior = _read_json(marker)
            if replay_success and isinstance(prior, dict):
                if prior.get("status") == "ok":
                    return prior.get("result")
            with self._fetch_slot():
                try:
                    result = fetcher()
                except Exception as exc:
                    status_code = getattr(exc, "status_code", None)
                    if status_code is None and getattr(exc, "response", None) is not None:
                        status_code = getattr(exc.response, "status_code", None)
                    terminal = status_code in _TERMINAL_FETCH_STATUS
                    failure = {
                            "schema_version": SCHEMA_VERSION,
                            "paper_id": canonical_id,
                            "operation": operation_key,
                            "status": "error",
                            "terminal": terminal,
                            "status_code": status_code,
                            "error": type(exc).__name__,
                            "updated_at": now_iso(),
                        }
                    self._atomic_json(terminal_marker if terminal else marker, failure)
                    raise
                if replay_success:
                    self._atomic_json(marker, {
                        "schema_version": SCHEMA_VERSION,
                        "paper_id": canonical_id,
                        "operation": operation_key,
                        "status": "ok",
                        "result": result,
                        "updated_at": now_iso(),
                    })
                return result

    def _enrich_call_records(self, call_id: str, operation: str, source: Mapping[str, Any]) -> None:
        for record_path in self.record_root.glob("*.json"):
            record = _read_json(record_path)
            if not isinstance(record, dict) or record.get("writer_call_id") != call_id:
                continue
            try:
                relative = self._safe_relative(str(record["relative_path"]))
            except (KeyError, TypeError, ValueError):
                continue
            path = self.overlay_root / relative
            if not path.is_file():
                continue
            with self._artifact_lock(relative):
                current = _read_json(record_path)
                if (
                    not isinstance(current, dict)
                    or current.get("writer_call_id") != call_id
                    or current.get("session_id") != self.session_id
                    or current.get("relative_path") != relative.as_posix()
                    or current.get("content_hash") != _sha256_file(path)
                    or current.get("writer") != "arc_paper.cache.v1"
                ):
                    continue
                self._record_artifact(
                    relative, source=source, writer_call_id=call_id, operation=operation
                )

    def _record_artifact(
        self,
        relative: Path,
        *,
        source: Mapping[str, Any] | None,
        parser_version: str | int | None = None,
        writer_call_id: str = "",
        operation: str = "",
    ) -> None:
        path = self.overlay_root / relative
        schema_version: Any = None
        embedded_parser_version: Any = None
        if path.suffix == ".json":
            value = _read_json(path)
            if isinstance(value, dict):
                schema_version = value.get("schema_version")
                embedded_parser_version = value.get("parser_version") or value.get("rich_parser_version")
        self._atomic_json(
            self._record_path(relative),
            {
                "schema_version": SCHEMA_VERSION,
                "session_id": self.session_id,
                "relative_path": relative.as_posix(),
                "content_hash": _sha256_file(path),
                "artifact_schema_version": schema_version,
                "parser_version": parser_version if parser_version is not None else embedded_parser_version,
                "source": _redact(dict(source or {})),
                "writer_call_id": writer_call_id,
                "writer": "arc_paper.cache.v1" if writer_call_id != "session-direct" else "arc_paper.session.v1",
                "operation": operation,
                "created_at": now_iso(),
            },
        )

    def _validation_error(self, relative: Path, path: Path) -> str:
        if relative.parts[0] not in _ALLOWED_CACHE_NAMESPACES:
            return "cache_namespace_forbidden"
        if path.is_symlink() or not path.is_file():
            return "artifact_not_regular_file"
        record = _read_json(self._record_path(relative))
        if not isinstance(record, dict):
            return "source_record_missing"
        if record.get("schema_version") != SCHEMA_VERSION:
            return "source_record_schema_invalid"
        if record.get("session_id") != self.session_id or record.get("relative_path") != relative.as_posix():
            return "source_record_ownership_invalid"
        if not record.get("writer_call_id") or not record.get("operation"):
            return "operation_ownership_missing"
        if record.get("writer") not in {"arc_paper.cache.v1", "arc_paper.session.v1"}:
            return "writer_provenance_invalid"
        if record.get("content_hash") != _sha256_file(path):
            return "content_hash_mismatch"
        if not isinstance(record.get("source"), dict) or not record["source"]:
            return "source_provenance_missing"
        if path.suffix == ".json":
            data = _read_json(path)
            if data is None:
                return "invalid_json"
            structural_error = self._json_contract_error(relative, data)
            if structural_error:
                return structural_error
        return ""

    def _json_contract_error(self, relative: Path, data: Any) -> str:
        namespace = relative.parts[0]
        if namespace == "sources":
            if not isinstance(data, dict) or not data.get("paper_id") or not data.get("source_hash"):
                return "parsed_source_contract_invalid"
            if not isinstance(data.get("parser_version"), int) or data["parser_version"] < 1:
                return "parser_version_invalid"
        elif namespace == "rich-sources":
            if not isinstance(data, dict) or not isinstance(data.get("rich_parser_version"), int):
                return "rich_parser_version_invalid"
            document = data.get("document")
            if not isinstance(document, dict) or not document.get("schema_version"):
                return "rich_document_schema_invalid"
        elif namespace == "source-annotations":
            if not isinstance(data, dict) or data.get("schema_version") != "arc.parsed_source.annotations.v1":
                return "annotation_schema_invalid"
        elif namespace == "paper-aliases":
            if not isinstance(data, dict) or data.get("schema_version") != "arc.paper_alias.v1":
                return "paper_alias_schema_invalid"
        elif relative.name == "manifest.json":
            if not isinstance(data, dict) or not str(data.get("schema_version") or "").startswith("arc."):
                return "manifest_schema_invalid"
        return ""

    def _quarantine(self, relative: Path, path: Path, reason: str) -> None:
        digest = _sha256_file(path) if path.is_file() else "unknown"
        target = self.quarantine_root / relative
        target = target.with_name(f"{target.name}.{digest}")
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, target)
        self._atomic_json(
            target.with_name(f"{target.name}.record.json"),
            {
                "schema_version": SCHEMA_VERSION,
                "relative_path": relative.as_posix(),
                "content_hash": digest,
                "reason": reason,
                "quarantined_at": now_iso(),
            },
        )

    def _write_conflict_record(self, relative: Path, digest: str, target: Path, conflict: Path) -> None:
        self._atomic_json(
            conflict.with_name(f"{conflict.name}.record.json"),
            {
                "schema_version": SCHEMA_VERSION,
                "relative_path": relative.as_posix(),
                "incoming_hash": digest,
                "existing_hash": _sha256_file(target) if target.is_file() else "non-file",
                "conflict_path": str(conflict),
                "created_at": now_iso(),
            },
        )

    def _artifact_files(self) -> Iterator[Path]:
        for path in self.overlay_root.rglob("*"):
            relative = path.relative_to(self.overlay_root)
            if (
                path.is_file()
                and _STATE_DIR not in relative.parts
                and "locks" not in relative.parts
                and not path.name.endswith(".tmp")
            ):
                yield path

    @contextmanager
    def _artifact_lock(self, relative: Path) -> Iterator[None]:
        digest = hashlib.sha256(relative.as_posix().encode("utf-8")).hexdigest()
        with self._locked(self.state_root / "write-locks" / f"{digest}.lock"):
            yield

    def _promotion_bytes(self, path: Path) -> bytes:
        payload = path.read_bytes()
        if path.suffix != ".json":
            return payload
        data = _read_json(path)
        rewritten = _rewrite_path_prefix(data, str(self.overlay_root), str(self.base_root))
        if rewritten == data:
            return payload
        return json.dumps(rewritten, indent=2, ensure_ascii=False).encode("utf-8")

    def _write_session_manifest(self) -> None:
        path = self.state_root / "session.json"
        if path.exists():
            return
        self._atomic_json(
            path,
            {
                "schema_version": SCHEMA_VERSION,
                "session_id": self.session_id,
                "base_root": str(self.base_root),
                "overlay_root": str(self.overlay_root),
                "max_parallel_fetches": self.max_parallel_fetches,
                "created_at": now_iso(),
            },
        )

    def _safe_relative(self, relative: str | Path) -> Path:
        value = Path(relative)
        if value.is_absolute() or not value.parts or any(part in {"", ".", ".."} for part in value.parts):
            raise ValueError("cache path must be a non-empty relative path without traversal")
        if value.parts[0] == _STATE_DIR:
            raise ValueError("cache path belongs to worker session state")
        if value.parts[0] not in _ALLOWED_CACHE_NAMESPACES:
            raise ValueError("cache path is outside allowed ARC paper namespaces")
        return value

    @staticmethod
    def _validate_call_id(call_id: str) -> None:
        if not _SAFE_SESSION_ID.fullmatch(call_id) or call_id in {".", ".."} or ".." in call_id:
            raise WorkerSessionError("call_id must be a safe 1-128 character identifier")

    def _record_path(self, relative: Path) -> Path:
        digest = hashlib.sha256(relative.as_posix().encode("utf-8")).hexdigest()
        return self.record_root / f"{digest}.json"

    def _tombstone_path(self, relative: Path) -> Path:
        digest = hashlib.sha256(relative.as_posix().encode("utf-8")).hexdigest()
        return self.tombstone_root / f"{digest}.json"

    @contextmanager
    def _fetch_slot(self) -> Iterator[None]:
        slots = self.state_root / "fetch-slots"
        slots.mkdir(parents=True, exist_ok=True)
        while True:
            for index in range(self.max_parallel_fetches):
                handle = (slots / f"{index}.lock").open("a+b")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    handle.close()
                return
            time.sleep(0.01)

    @contextmanager
    def _locked(self, path: Path) -> Iterator[None]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        temporary.write_bytes(content)
        os.replace(temporary, path)

    @classmethod
    def _atomic_json(cls, path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cls._atomic_write(path, json.dumps(value, indent=2, ensure_ascii=False).encode("utf-8"))

    @classmethod
    def _copy_atomic(cls, source: Path, target: Path) -> None:
        temporary = target.with_name(f"{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)

    @classmethod
    def _write_atomic_bytes(cls, target: Path, payload: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        cls._atomic_write(target, payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]" if any(secret in str(key).lower() for secret in _SECRET_KEYS) else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    if isinstance(value, str) and len(value) > 512:
        return value[:512] + "…"
    return value


def _rewrite_path_prefix(value: Any, source_root: str, target_root: str) -> Any:
    if isinstance(value, dict):
        return {key: _rewrite_path_prefix(item, source_root, target_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_path_prefix(item, source_root, target_root) for item in value]
    if isinstance(value, str) and (value == source_root or value.startswith(source_root + os.sep)):
        return target_root + value[len(source_root):]
    return value


def worker_fetch_once(
    paper_id: str,
    fetcher: Callable[[], Any],
    *,
    operation: str,
    replay_success: bool = True,
) -> Any:
    """Use run-wide fetch deduplication when called inside a worker session."""

    session = WorkerCacheSession.from_environment()
    if session is None:
        return fetcher()
    return session.fetch_once(
        paper_id, fetcher, operation=operation, replay_success=replay_success
    )
