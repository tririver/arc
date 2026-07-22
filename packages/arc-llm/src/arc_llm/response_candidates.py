from __future__ import annotations

import hashlib
import ctypes
import json
import os
import stat
import threading
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .call_record import strip_arc_llm_call_records
from .providers.base import (
    LLMAbortScope,
    LLMFailureCategory,
    LLMSubmissionState,
    LLMWorkerError,
)
from .schema_cache import canonical_json, sha256_text
from .usage import LLMProviderResponse, ResponseCandidateMaterial


SELECTION_SCHEMA_VERSION = "arc.llm.response_candidate_selection.v1"
MAX_RESPONSE_MATERIAL = 256
MAX_COMPLETE_CANDIDATES = 256
MAX_CANDIDATE_ORIGINS = 64
MAX_SUPERSEDED_POSITIONS = 64
MAX_SELECTION_RECEIPT_BYTES = 1024 * 1024
_KNOWN_SOURCES = {
    "codex.completed_message",
    "codex.output_last_message",
    "claude.completed_assistant_text",
    "claude.terminal_result",
    "claude.terminal_structured_output",
    "kimi.session_prompt_message",
    "generic.provider_value",
    "recovery.response_candidate_stream",
}
_SELECTION_THREAD_LOCK = threading.Lock()


def _selection_write_fault(_cutpoint: str) -> None:
    """Test-only crash hook for immutable receipt publication."""


class LLMResponseCandidateConflict(LLMWorkerError):
    """Multiple paid, schema-valid results need human selection."""

    def __init__(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        replayed: bool = False,
    ) -> None:
        self.candidates = tuple(dict(candidate) for candidate in candidates)
        self.replayed = replayed
        rendered = ", ".join(
            f"{item['sha256']}@{item['protocol_position']}"
            + (f"#{item['event_id']}" if item.get("event_id") else "")
            for item in self.candidates
        )
        super().__init__(
            f"Conflicting substantive structured responses need supervision: {rendered}",
            retryable=False,
            category=LLMFailureCategory.OUTPUT_INVALID,
            abort_scope=LLMAbortScope.BATCH,
            submission_state=(
                LLMSubmissionState.NOT_SUBMITTED
                if replayed
                else LLMSubmissionState.SUBMITTED
            ),
        )


class LLMResponseCandidateReceiptError(LLMWorkerError):
    def __init__(self, message: str, *, replayed: bool = False) -> None:
        super().__init__(
            message,
            retryable=False,
            category=LLMFailureCategory.LOCAL_IO,
            abort_scope=LLMAbortScope.BATCH,
            submission_state=(
                LLMSubmissionState.NOT_SUBMITTED
                if replayed
                else LLMSubmissionState.SUBMITTED
            ),
        )


@dataclass(frozen=True)
class CandidateSelection:
    response: LLMProviderResponse[dict[str, Any]]
    receipt: dict[str, Any]
    conflict: LLMResponseCandidateConflict | None = None
    diagnostic_candidates: tuple[tuple[int, str, str, dict[str, Any]], ...] = ()


@dataclass
class _Candidate:
    ordinal: int
    source: str
    protocol_position: int
    text_offset: int
    extraction_ordinal: int
    event_id: str | None
    value: dict[str, Any]
    sha256: str
    schema_valid: bool
    substantive: bool
    semantic_mass: int
    supersedes: tuple[int, ...]
    origins: list[dict[str, Any]]


class _CandidateLimitExceeded(ValueError):
    pass


def select_response_candidate(
    response: LLMProviderResponse[dict[str, Any]],
    *,
    schema: Mapping[str, Any] | None,
    checkpoint_identity: str | None,
    replayed: bool = False,
) -> CandidateSelection:
    """Select a complete schema-valid object without using relaxed fragments."""

    material = list(response.candidate_material)
    if not material:
        material.append(
            ResponseCandidateMaterial(
                source="generic.provider_value",
                protocol_position=0,
                value=response.value if isinstance(response.value, dict) else None,
            )
        )
    if len(material) > MAX_RESPONSE_MATERIAL:
        raise LLMResponseCandidateReceiptError(
            f"Response candidate material exceeds the {MAX_RESPONSE_MATERIAL}-item audit limit",
            replayed=replayed,
        )
    if any(len(item.supersedes) > MAX_SUPERSEDED_POSITIONS for item in material):
        raise LLMResponseCandidateReceiptError(
            f"Response candidate supersession metadata exceeds the {MAX_SUPERSEDED_POSITIONS}-position audit limit",
            replayed=replayed,
        )
    try:
        candidates, extracted = _enumerate_candidates(material, schema)
    except _CandidateLimitExceeded as exc:
        raise LLMResponseCandidateReceiptError(
            f"Complete response candidates exceed the {MAX_COMPLETE_CANDIDATES}-item audit limit",
            replayed=replayed,
        ) from exc
    valid = [candidate for candidate in candidates if candidate.schema_valid]
    substantive = [candidate for candidate in valid if candidate.substantive]
    selected: _Candidate | None = None
    conflict: LLMResponseCandidateConflict | None = None
    if substantive:
        distinct = {candidate.sha256 for candidate in substantive}
        selected = substantive[-1]
        if len(distinct) == 1:
            decision = "last_substantive"
        elif _supersedes_all(selected, substantive[:-1]):
            decision = "protocol_supersession"
        else:
            decision = "ambiguous_substantive_conflict"
            conflict_items = [
                {
                    "sha256": candidate.sha256,
                    "protocol_position": candidate.protocol_position,
                    "event_id": candidate.event_id,
                }
                for candidate in substantive
                if candidate.sha256 in distinct
            ]
            conflict = LLMResponseCandidateConflict(conflict_items, replayed=replayed)
    elif valid:
        selected = valid[-1]
        decision = "last_valid_empty"
    else:
        decision = "no_schema_valid_candidate"

    receipt = _receipt(
        checkpoint_identity=checkpoint_identity,
        schema=schema,
        material=material,
        candidates=candidates,
        decision=decision,
        selected=selected,
        conflict=conflict,
    )
    selected_response = replace(
        response,
        value=(selected.value if selected is not None and conflict is None else response.value),
        candidate_material=tuple(material),
        candidate_selection=receipt,
    )
    return CandidateSelection(
        response=selected_response,
        receipt=receipt,
        conflict=conflict,
        diagnostic_candidates=tuple(
            (item.ordinal, item.sha256, item.source, item.value)
            for item in extracted
        ),
    )


def has_complete_candidate(
    material: Sequence[ResponseCandidateMaterial],
) -> bool:
    for item in material:
        if isinstance(item.value, dict):
            return True
        if item.text is not None:
            try:
                if _complete_json_objects(item.text, stop_after_first=True):
                    return True
            except _CandidateLimitExceeded:
                return False
    return False


def persist_selection_receipt(
    checkpoint_path: Path,
    receipt: Mapping[str, Any],
    replayed: bool = False,
) -> tuple[str, str]:
    """Create or verify the immutable, body-free selection receipt."""

    path = checkpoint_path.with_name(f"{checkpoint_path.stem}.candidate-selection.json")
    encoded = (canonical_json(dict(receipt)) + "\n").encode("utf-8")
    if len(encoded) > MAX_SELECTION_RECEIPT_BYTES:
        raise LLMResponseCandidateReceiptError(
            "Response candidate receipt exceeds its byte limit", replayed=replayed
        )
    try:
        with _secure_selection_parent(path, create=True) as (directory_fd, name, binding):
            published, persisted = _atomic_publish_exclusive_at(
                directory_fd, name, encoded, binding=binding,
            )
    except (OSError, ValueError) as exc:
        raise LLMResponseCandidateReceiptError(
            f"Could not create response candidate receipt {path}: {exc}",
            replayed=replayed,
        ) from exc
    if not published:
        try:
            existing_value = _loads_receipt_strict(persisted)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise LLMResponseCandidateReceiptError(
                f"Could not read response candidate receipt {path}: {exc}",
                replayed=replayed,
            ) from exc
        if not isinstance(existing_value, Mapping) or canonical_json(existing_value) != canonical_json(dict(receipt)):
            raise LLMResponseCandidateReceiptError(
                f"Response candidate receipt changed or is incompatible: {path}",
                replayed=replayed,
            )
    return path.name, hashlib.sha256(persisted).hexdigest()


def _atomic_publish_exclusive(path: Path, payload: bytes) -> bool:
    """Compatibility wrapper around the held-dirfd immutable publisher."""

    with _secure_selection_parent(path, create=True) as (directory_fd, name, binding):
        published, _persisted = _atomic_publish_exclusive_at(
            directory_fd, name, payload, binding=binding,
        )
        return published


@dataclass(frozen=True)
class _ParentBinding:
    absolute: Path
    identity: tuple[int, int]
    lock_fd: int
    lock_identity: tuple[int, int]


def _directory_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


def _open_absolute_directory(path: Path, *, create: bool) -> int:
    absolute = path.absolute()
    if not absolute.is_absolute():
        raise ValueError("selection receipt parent must be absolute")
    descriptor = os.open(absolute.anchor, _directory_flags())
    try:
        for part in absolute.parts[1:]:
            if part in {"", ".", ".."}:
                raise ValueError("selection receipt parent is unsafe")
            try:
                child = os.open(part, _directory_flags(), dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(part, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                os.fsync(descriptor)
                child = os.open(part, _directory_flags(), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _verify_parent_address(binding: _ParentBinding, directory_fd: int) -> None:
    held = os.fstat(directory_fd)
    if (held.st_dev, held.st_ino) != binding.identity:
        raise ValueError("selection receipt parent binding changed")
    current_fd = _open_absolute_directory(binding.absolute, create=False)
    try:
        current = os.fstat(current_fd)
        if (current.st_dev, current.st_ino) != binding.identity:
            raise ValueError("selection receipt parent address changed")
    finally:
        os.close(current_fd)
    named = os.stat(".candidate-selection.lock", dir_fd=directory_fd, follow_symlinks=False)
    lock_stat = os.fstat(binding.lock_fd)
    if (
        not stat.S_ISREG(named.st_mode)
        or named.st_nlink != 1
        or (named.st_dev, named.st_ino) != binding.lock_identity
        or (lock_stat.st_dev, lock_stat.st_ino) != binding.lock_identity
    ):
        raise ValueError("selection receipt lock binding changed")


@contextmanager
def _secure_selection_parent(
    path: Path, *, create: bool,
) -> Any:
    absolute = path.absolute()
    if absolute.name != path.name or not absolute.name.endswith(".candidate-selection.json"):
        raise ValueError("selection receipt address is invalid")
    with _SELECTION_THREAD_LOCK:
        directory_fd = _open_absolute_directory(absolute.parent, create=create)
        lock_flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        lock_fd = os.open(
            ".candidate-selection.lock", lock_flags, 0o600, dir_fd=directory_fd,
        )
        try:
            if os.name == "nt":  # pragma: no cover
                import msvcrt
                msvcrt.locking(lock_fd, msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
            parent_stat = os.fstat(directory_fd)
            lock_stat = os.fstat(lock_fd)
            binding = _ParentBinding(
                absolute=absolute.parent,
                identity=(parent_stat.st_dev, parent_stat.st_ino),
                lock_fd=lock_fd,
                lock_identity=(lock_stat.st_dev, lock_stat.st_ino),
            )
            _verify_parent_address(binding, directory_fd)
            try:
                yield directory_fd, absolute.name, binding
                _verify_parent_address(binding, directory_fd)
            finally:
                try:
                    _verify_parent_address(binding, directory_fd)
                finally:
                    if os.name == "nt":  # pragma: no cover
                        import msvcrt
                        os.lseek(lock_fd, 0, os.SEEK_SET)
                        msvcrt.locking(lock_fd, msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)
            os.close(directory_fd)


def _read_receipt_at(directory_fd: int, name: str, *, allow_two_links: bool) -> bytes | None:
    try:
        leaf = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    allowed = {1, 2} if allow_two_links else {1}
    if not stat.S_ISREG(leaf.st_mode) or leaf.st_nlink not in allowed:
        raise ValueError("selection receipt leaf is unsafe")
    descriptor = os.open(
        name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd,
    )
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        remaining = MAX_SELECTION_RECEIPT_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if len(raw) > MAX_SELECTION_RECEIPT_BYTES:
            raise ValueError("selection receipt exceeds byte limit")
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or after.st_size != len(raw)
        ):
            raise ValueError("selection receipt changed while reading")
        return raw
    finally:
        os.close(descriptor)


def _atomic_publish_exclusive_at(
    directory_fd: int,
    name: str,
    payload: bytes,
    *,
    binding: _ParentBinding,
) -> tuple[bool, bytes]:
    stage = f".{name}.{hashlib.sha256(payload).hexdigest()[:24]}.staged"
    try:
        return _atomic_publish_exclusive_at_inner(
            directory_fd, name, payload, binding=binding,
        )
    except BaseException:
        # Cleanup is limited to this payload's exact deterministic stage. The
        # held directory fd remains safe even if its pathname was renamed.
        try:
            _verify_parent_address(binding, directory_fd)
            stage_stat = os.stat(stage, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISLNK(stage_stat.st_mode):
                os.unlink(stage, dir_fd=directory_fd)
                os.fsync(directory_fd)
            else:
                staged = _read_receipt_at(directory_fd, stage, allow_two_links=True)
                try:
                    final = _read_receipt_at(
                        directory_fd, name, allow_two_links=True,
                    )
                except ValueError:
                    final = None
                    _selection_exchange_at(directory_fd, stage, name)
                    restored = _read_receipt_at(
                        directory_fd, name, allow_two_links=False,
                    )
                    if restored != payload:
                        raise ValueError("selection receipt rollback binding changed")
                    os.unlink(name, dir_fd=directory_fd)
                    _unlink_exact_selection_stage(directory_fd, stage)
                    os.fsync(directory_fd)
                    raise
                stage_value = os.stat(
                    stage, dir_fd=directory_fd, follow_symlinks=False,
                )
                final_value = (
                    os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                    if final is not None else None
                )
                if (
                    final is not None
                    and final_value is not None
                    and (stage_value.st_dev, stage_value.st_ino)
                    != (final_value.st_dev, final_value.st_ino)
                ):
                    _selection_exchange_at(directory_fd, stage, name)
                    restored = _read_receipt_at(
                        directory_fd, name, allow_two_links=False,
                    )
                    if restored != payload:
                        raise ValueError("selection receipt rollback binding changed")
                    os.unlink(name, dir_fd=directory_fd)
                    _unlink_exact_selection_stage(directory_fd, stage)
                    os.fsync(directory_fd)
                    raise ValueError("selection receipt publication was rolled back")
                if staged is not None and (
                    final is None
                    or (
                        final == staged
                        and final_value is not None
                        and (stage_value.st_dev, stage_value.st_ino)
                        == (final_value.st_dev, final_value.st_ino)
                    )
                ):
                    os.unlink(stage, dir_fd=directory_fd)
                    os.fsync(directory_fd)
        except (FileNotFoundError, OSError, ValueError):
            pass
        raise


def _selection_exchange_at(directory_fd: int, left: str, right: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:  # pragma: no cover - non-Linux fail-closed path
        raise ValueError("atomic selection receipt exchange is unavailable")
    renameat2.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(
        directory_fd, os.fsencode(left), directory_fd, os.fsencode(right), 2,
    ) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), f"{left}<->{right}")


def _unlink_exact_selection_stage(directory_fd: int, name: str) -> None:
    value = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if not (stat.S_ISLNK(value.st_mode) or stat.S_ISREG(value.st_mode)):
        raise ValueError("selection receipt rollback stage is unsafe")
    os.unlink(name, dir_fd=directory_fd)


def _atomic_publish_exclusive_at_inner(
    directory_fd: int,
    name: str,
    payload: bytes,
    *,
    binding: _ParentBinding,
) -> tuple[bool, bytes]:
    """Crash-recoverable immutable no-clobber publication under a named lock."""

    if len(payload) > MAX_SELECTION_RECEIPT_BYTES:
        raise ValueError("selection receipt exceeds byte limit")
    digest = hashlib.sha256(payload).hexdigest()
    stage = f".{name}.{digest[:24]}.staged"
    _verify_parent_address(binding, directory_fd)
    _reconcile_selection_stage(directory_fd, name, stage, payload, binding)
    existing = _read_receipt_at(directory_fd, name, allow_two_links=True)
    if existing is not None:
        return False, existing
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(stage, flags, 0o400, dir_fd=directory_fd)
    except FileExistsError:
        _reconcile_selection_stage(directory_fd, name, stage, payload, binding)
        descriptor = -1
    if descriptor >= 0:
        created = os.fstat(descriptor)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                _selection_write_fault("stage:after_write")
                os.fsync(handle.fileno())
                _selection_write_fault("stage:after_fsync")
            os.fsync(directory_fd)
        except BaseException:
            try:
                _verify_parent_address(binding, directory_fd)
                current = os.stat(stage, dir_fd=directory_fd, follow_symlinks=False)
                if (current.st_dev, current.st_ino) == (created.st_dev, created.st_ino):
                    os.unlink(stage, dir_fd=directory_fd)
                    os.fsync(directory_fd)
            except FileNotFoundError:
                pass
            raise
        current_stage = os.stat(stage, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(current_stage.st_mode)
            or current_stage.st_nlink != 1
            or (current_stage.st_dev, current_stage.st_ino)
            != (created.st_dev, created.st_ino)
        ):
            raise ValueError("selection receipt staging leaf binding changed")
    _verify_parent_address(binding, directory_fd)
    existing = _read_receipt_at(directory_fd, name, allow_two_links=True)
    if existing is not None:
        _reconcile_selection_stage(directory_fd, name, stage, payload, binding)
        return False, existing
    try:
        os.link(
            stage,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        published = True
    except FileExistsError:
        published = False
    except BaseException:
        if _read_receipt_at(directory_fd, name, allow_two_links=True) is None:
            try:
                os.unlink(stage, dir_fd=directory_fd)
                os.fsync(directory_fd)
            except FileNotFoundError:
                pass
        raise
    _selection_write_fault("publish:after_link")
    os.fsync(directory_fd)
    _selection_write_fault("publish:after_fsync")
    _reconcile_selection_stage(directory_fd, name, stage, payload, binding)
    _verify_parent_address(binding, directory_fd)
    persisted = _read_receipt_at(directory_fd, name, allow_two_links=False)
    if persisted is None:
        raise ValueError("selection receipt publication disappeared")
    return published, persisted


def _reconcile_selection_stage(
    directory_fd: int,
    name: str,
    stage: str,
    payload: bytes,
    binding: _ParentBinding,
) -> None:
    staged = _read_receipt_at(directory_fd, stage, allow_two_links=True)
    if staged is None:
        return
    stage_stat = os.stat(stage, dir_fd=directory_fd, follow_symlinks=False)
    final = _read_receipt_at(directory_fd, name, allow_two_links=True)
    if staged != payload:
        if final is None and stage_stat.st_nlink == 1:
            _verify_parent_address(binding, directory_fd)
            os.unlink(stage, dir_fd=directory_fd)
            os.fsync(directory_fd)
            return
        raise ValueError("selection receipt staging collision")
    if final is not None:
        final_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if final != payload or (stage_stat.st_dev, stage_stat.st_ino) != (
            final_stat.st_dev, final_stat.st_ino,
        ):
            raise ValueError("selection receipt publication collision")
        _verify_parent_address(binding, directory_fd)
        os.unlink(stage, dir_fd=directory_fd)
        os.fsync(directory_fd)


def material_from_codex(
    raw_events: Sequence[Mapping[str, Any]], output_last_message: str
) -> tuple[ResponseCandidateMaterial, ...]:
    items: list[ResponseCandidateMaterial] = []
    position = 0
    for event in raw_events:
        event_type = str(event.get("type") or "")
        item = event.get("item")
        if event_type not in {"item.completed", "message.completed"} or not isinstance(item, Mapping):
            continue
        if str(item.get("type") or "") not in {"agent_message", "assistant_message", "message"}:
            continue
        text = item.get("text") or item.get("content")
        if isinstance(text, str):
            items.append(
                ResponseCandidateMaterial(
                    source="codex.completed_message",
                    protocol_position=position,
                    text=text,
                    event_id=_event_id(event, item),
                )
            )
            position += 1
    items.append(
        ResponseCandidateMaterial(
            source="codex.output_last_message",
            protocol_position=position,
            text=output_last_message,
        )
    )
    return tuple(items)


def material_from_claude(
    stdout: str,
) -> tuple[ResponseCandidateMaterial, ...]:
    items: list[ResponseCandidateMaterial] = []
    position = 0
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, Mapping):
            continue
        if event.get("type") == "assistant":
            message = event.get("message")
            blocks = message.get("content") if isinstance(message, Mapping) else None
            if isinstance(blocks, list):
                text_blocks = [
                    block["text"]
                    for block in blocks
                    if isinstance(block, Mapping)
                    and block.get("type") == "text"
                    and isinstance(block.get("text"), str)
                ]
                if text_blocks:
                    items.append(
                        ResponseCandidateMaterial(
                            source="claude.completed_assistant_text",
                            protocol_position=position,
                            text="".join(text_blocks),
                            event_id=_event_id(event),
                        )
                    )
                    position += 1
        if event.get("type") != "result":
            continue
        result_position: int | None = None
        result = event.get("result")
        if isinstance(result, Mapping):
            result_position = position
            items.append(
                ResponseCandidateMaterial(
                    source="claude.terminal_result",
                    protocol_position=position,
                    value=dict(result),
                    event_id=_event_id(event),
                )
            )
            position += 1
        elif isinstance(result, str):
            result_position = position
            items.append(
                ResponseCandidateMaterial(
                    source="claude.terminal_result",
                    protocol_position=position,
                    text=result,
                    event_id=_event_id(event),
                )
            )
            position += 1
        structured = event.get("structured_output")
        if isinstance(structured, Mapping):
            items.append(
                ResponseCandidateMaterial(
                    source="claude.terminal_structured_output",
                    protocol_position=position,
                    value=dict(structured),
                    event_id=_event_id(event),
                    supersedes=(result_position,) if result_position is not None else (),
                )
            )
            position += 1
    return tuple(items)


def material_from_kimi(text: str) -> tuple[ResponseCandidateMaterial, ...]:
    return (
        ResponseCandidateMaterial(
            source="kimi.session_prompt_message",
            protocol_position=0,
            text=text,
        ),
    )


def _enumerate_candidates(
    material: Sequence[ResponseCandidateMaterial],
    schema: Mapping[str, Any] | None,
) -> tuple[list[_Candidate], list[_Candidate]]:
    ordered = sorted(enumerate(material), key=lambda pair: (pair[1].protocol_position, pair[0]))
    extracted: list[_Candidate] = []
    extraction_ordinal = 0
    for _, item in ordered:
        values: list[tuple[int, dict[str, Any]]] = []
        if isinstance(item.value, dict):
            values.append((-1, dict(item.value)))
        if item.text is not None:
            values.extend(
                _complete_json_objects(
                    item.text,
                    limit=MAX_COMPLETE_CANDIDATES - extraction_ordinal - len(values),
                )
            )
        for text_offset, value in values:
            clean = strip_arc_llm_call_records(value)
            if _contains_nonfinite(clean):
                continue
            if extraction_ordinal >= MAX_COMPLETE_CANDIDATES:
                raise _CandidateLimitExceeded
            extraction_ordinal += 1
            encoded = canonical_json(clean)
            digest = sha256_text(encoded)
            valid = _validates(clean, schema)
            mass = _semantic_mass(clean, schema, root_schema=schema) if valid else 0
            extracted.append(
                _Candidate(
                    ordinal=extraction_ordinal,
                    source=_safe_source(item.source),
                    protocol_position=item.protocol_position,
                    text_offset=text_offset,
                    extraction_ordinal=extraction_ordinal,
                    event_id=_safe_event_id(item.event_id),
                    value=clean,
                    sha256=digest,
                    schema_valid=valid,
                    substantive=valid and mass > 0,
                    semantic_mass=mass,
                    supersedes=tuple(int(position) for position in item.supersedes),
                    origins=[
                        {
                            "source": _safe_source(item.source),
                            "protocol_position": item.protocol_position,
                            "text_offset": text_offset,
                            "event_id": _safe_event_id(item.event_id),
                        }
                    ],
                )
            )
    merged: dict[str, _Candidate] = {}
    for candidate in extracted:
        previous = merged.get(candidate.sha256)
        if previous is not None:
            candidate.origins = [*previous.origins, *candidate.origins]
            candidate.supersedes = tuple(dict.fromkeys((*previous.supersedes, *candidate.supersedes)))
            del merged[candidate.sha256]
        merged[candidate.sha256] = candidate
    merged_candidates = sorted(
        merged.values(),
        key=lambda candidate: (
            candidate.protocol_position,
            candidate.text_offset,
            candidate.extraction_ordinal,
        ),
    )
    return merged_candidates, extracted


def _complete_json_objects(
    text: str,
    *,
    limit: int | None = None,
    stop_after_first: bool = False,
) -> list[tuple[int, dict[str, Any]]]:
    found: list[tuple[int, dict[str, Any]]] = []
    stripped = text.strip()
    if stripped:
        try:
            direct = _loads_strict(stripped)
        except (json.JSONDecodeError, RecursionError, ValueError):
            pass
        else:
            if isinstance(direct, dict):
                if limit is not None and limit <= 0:
                    raise _CandidateLimitExceeded
                return [(text.find(stripped), direct)]
    covering_end = -1
    parse_attempts = 0
    for start, end in sorted(_balanced_object_ranges(text), key=lambda item: (item[0], -item[1])):
        if end <= covering_end:
            continue
        parse_attempts += 1
        if parse_attempts > MAX_COMPLETE_CANDIDATES + 1:
            raise _CandidateLimitExceeded
        try:
            value = _loads_strict(text[start:end])
        except (json.JSONDecodeError, RecursionError, ValueError):
            continue
        if isinstance(value, dict):
            if limit is not None and len(found) >= limit:
                raise _CandidateLimitExceeded
            found.append((start, value))
            covering_end = max(covering_end, end)
            if stop_after_first:
                return found
    # Whole objects, fenced objects, and balanced prose objects converge here.
    unique: dict[tuple[int, str], dict[str, Any]] = {}
    for offset, value in found:
        unique[(offset, canonical_json(value))] = value
    return [(offset, value) for (offset, _), value in sorted(unique.items(), key=lambda item: item[0][0])]


def _balanced_object_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    starts: list[int] = []
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            starts.append(index)
        elif char == "}" and starts:
            ranges.append((starts.pop(), index + 1))
    return sorted(ranges)


def _validates(value: Mapping[str, Any], schema: Mapping[str, Any] | None) -> bool:
    if schema is None:
        return True
    from jsonschema import ValidationError
    from jsonschema.exceptions import SchemaError
    from jsonschema.validators import validator_for

    try:
        validator = validator_for(schema)
        validator.check_schema(schema)
        validator(schema).validate(value)
    except (ValidationError, SchemaError):
        return False
    return True


def _semantic_mass(
    value: Any,
    schema: Mapping[str, Any] | None,
    *,
    root_schema: Mapping[str, Any] | None = None,
) -> int:
    root = root_schema if root_schema is not None else schema
    active = _resolved_schema(schema, root_schema=root)
    for keyword in ("oneOf", "anyOf"):
        branches = active.get(keyword) if isinstance(active, Mapping) else None
        if isinstance(branches, list):
            masses = [
                _semantic_mass(
                    value,
                    _merge_schema(active, _resolve_ref(branch, root or active), keyword),
                    root_schema=root,
                )
                for branch in branches
                if isinstance(branch, Mapping)
                and _validates_branch(value, branch, root or active)
            ]
            if masses:
                return max(masses)
    if value is None:
        return 0
    if isinstance(value, str):
        return 1 if value.strip() else 0
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return 1
    if isinstance(value, list):
        item_schema = active.get("items") if isinstance(active, Mapping) else None
        return sum(
            _semantic_mass(
                item,
                item_schema if isinstance(item_schema, Mapping) else None,
                root_schema=root,
            )
            for item in value
        )
    if isinstance(value, Mapping):
        if isinstance(active, Mapping):
            properties = active.get("properties")
            required = active.get("required")
            if isinstance(required, list) and required:
                keys = [key for key in required if isinstance(key, str)]
            else:
                keys = list(value)
            return sum(
                _semantic_mass(
                    value.get(key),
                    properties.get(key) if isinstance(properties, Mapping) and isinstance(properties.get(key), Mapping) else None,
                    root_schema=root,
                )
                for key in keys
            )
        return sum(_semantic_mass(item, None, root_schema=root) for item in value.values())
    return 0


def _resolved_schema(
    schema: Mapping[str, Any] | None,
    *,
    root_schema: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if not isinstance(schema, Mapping):
        return {}
    root = root_schema if isinstance(root_schema, Mapping) else schema
    return _resolve_ref(schema, root)


def _validates_branch(value: Any, branch: Mapping[str, Any], root: Mapping[str, Any]) -> bool:
    from jsonschema import ValidationError
    from jsonschema.exceptions import SchemaError
    from jsonschema.validators import validator_for

    try:
        validator = validator_for(root)
        branch_schema = dict(_resolve_ref(branch, root))
        for definitions_key in ("$defs", "definitions"):
            if definitions_key in root and definitions_key not in branch_schema:
                branch_schema[definitions_key] = root[definitions_key]
        validator(branch_schema).validate(value)
    except (ValidationError, SchemaError):
        return False
    return True


def _resolve_ref(schema: Mapping[str, Any], root: Mapping[str, Any]) -> Mapping[str, Any]:
    ref = schema.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return schema
    current: Any = root
    for token in ref[2:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, Mapping) or token not in current:
            return schema
        current = current[token]
    if not isinstance(current, Mapping):
        return schema
    # Modern JSON Schema drafts apply siblings of ``$ref`` as additional
    # constraints. Preserve both required/property sets for semantic scoring
    # instead of silently discarding the sibling half of the schema.
    resolved = dict(current)
    siblings = {key: value for key, value in schema.items() if key != "$ref"}
    if isinstance(resolved.get("required"), list) and isinstance(
        siblings.get("required"), list
    ):
        siblings["required"] = list(
            dict.fromkeys((*resolved["required"], *siblings["required"]))
        )
    if isinstance(resolved.get("properties"), Mapping) and isinstance(
        siblings.get("properties"), Mapping
    ):
        siblings["properties"] = {
            **dict(resolved["properties"]),
            **dict(siblings["properties"]),
        }
    resolved.update(siblings)
    return resolved


def _merge_schema(base: Mapping[str, Any], branch: Mapping[str, Any], keyword: str) -> dict[str, Any]:
    merged = {key: value for key, value in base.items() if key != keyword}
    merged.update(branch)
    return merged


def _supersedes_all(selected: _Candidate, earlier: Sequence[_Candidate]) -> bool:
    conflicts = {
        candidate.protocol_position
        for candidate in earlier
        if candidate.sha256 != selected.sha256
    }
    return bool(conflicts) and conflicts.issubset(set(selected.supersedes))


def _receipt(
    *,
    checkpoint_identity: str | None,
    schema: Mapping[str, Any] | None,
    material: Sequence[ResponseCandidateMaterial],
    candidates: Sequence[_Candidate],
    decision: str,
    selected: _Candidate | None,
    conflict: LLMResponseCandidateConflict | None,
) -> dict[str, Any]:
    material_payload = [item.to_json() for item in material]
    return {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "checkpoint_identity": checkpoint_identity,
        "business_schema_sha256": sha256_text(canonical_json(dict(schema))) if schema is not None else None,
        "material_sha256": sha256_text(canonical_json(material_payload)),
        "candidates": [
            {
                "ordinal": item.ordinal,
                "source": item.source,
                "protocol_position": item.protocol_position,
                "text_offset": item.text_offset,
                "event_id": item.event_id,
                "sha256": item.sha256,
                "schema_valid": item.schema_valid,
                "substantive": item.substantive,
                "semantic_mass": item.semantic_mass,
                "supersedes": list(item.supersedes[:MAX_SUPERSEDED_POSITIONS]),
                "supersedes_count": len(item.supersedes),
                "supersedes_truncated": len(item.supersedes) > MAX_SUPERSEDED_POSITIONS,
                "origins": _bounded_origins(item.origins),
                "origin_count": len(item.origins),
                "origins_truncated": len(item.origins) > MAX_CANDIDATE_ORIGINS,
            }
            for item in candidates
        ],
        "decision": decision,
        "selected_ordinal": selected.ordinal if selected is not None and conflict is None else None,
        "selected_sha256": selected.sha256 if selected is not None and conflict is None else None,
        "conflict_hashes": list(dict.fromkeys(item["sha256"] for item in (conflict.candidates if conflict else ()))),
    }


def _event_id(*values: Mapping[str, Any]) -> str | None:
    for value in values:
        for key in ("id", "item_id", "message_id", "uuid"):
            candidate = value.get(key)
            if isinstance(candidate, (str, int)) and str(candidate):
                return str(candidate)
    return None


def _safe_source(source: object) -> str:
    rendered = str(source)
    if rendered in _KNOWN_SOURCES:
        return rendered
    return "unknown." + hashlib.sha256(rendered.encode("utf-8", errors="replace")).hexdigest()[:16]


def _safe_event_id(event_id: object) -> str | None:
    if event_id is None:
        return None
    digest = hashlib.sha256(str(event_id).encode("utf-8", errors="replace")).hexdigest()
    return f"event.{digest[:16]}"


def _bounded_origins(origins: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(origins) <= MAX_CANDIDATE_ORIGINS:
        return [dict(origin) for origin in origins]
    half = MAX_CANDIDATE_ORIGINS // 2
    return [
        *(dict(origin) for origin in origins[:half]),
        *(dict(origin) for origin in origins[-half:]),
    ]


def _loads_strict(text: str) -> Any:
    return json.loads(
        text,
        parse_constant=lambda constant: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant {constant}")
        ),
    )


def _loads_receipt_strict(payload: bytes) -> Any:
    def object_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate receipt key {key!r}")
            result[key] = value
        return result

    return json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=object_pairs,
        parse_constant=lambda constant: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant {constant}")
        ),
    )


def _contains_nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not (float("-inf") < value < float("inf"))
    if isinstance(value, Mapping):
        return any(_contains_nonfinite(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_nonfinite(item) for item in value)
    return False


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
