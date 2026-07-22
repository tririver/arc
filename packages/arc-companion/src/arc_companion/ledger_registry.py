from __future__ import annotations

from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import threading
from typing import Any, Callable, Iterator, Mapping

from .secure_io import SecureReadError, read_bounded_file


REGISTRY_SCHEMA_VERSION = "arc.companion.lane-ledger-registry.v1"
MUTATION_JOURNAL_SCHEMA_VERSION = "arc.companion.lane-ledger-mutation.v1"
LANE_LEDGER_SCHEMA_VERSION = "arc.companion.chapter-lane-ledger.v2"
REGISTRY_FILE_NAME = "lane-ledger-registry.json"
MUTATION_JOURNAL_FILE_NAME = ".lane-ledger-mutation.json"
MAX_REGISTRY_ENTRIES = 4096
MAX_REGISTRY_BYTES = 4 * 1024 * 1024
MAX_LEDGER_BYTES = 16 * 1024 * 1024

_REGISTRY_THREAD_LOCK = threading.RLock()
_REGISTRY_LOCK_NAME = f".{REGISTRY_FILE_NAME}.lock"
_OWNED_LAYOUTS = {
    "chapters": "arc-companion.chapter-lane",
    "recovery-controls": "arc-companion.pipeline-recovery-control",
}


def _before_registered_ledger_replace(_path: Path) -> None:
    """Fault-injection cutpoint after staging and before address revalidation."""


def _before_registered_ledger_exchange(_path: Path) -> None:
    """Fault-injection cutpoint in the final check-to-publication window."""


def _after_registered_ledger_replace(_path: Path) -> None:
    """Fault-injection cutpoint after ledger durability but before registry update."""


def _after_mutation_journal_publish(_path: Path) -> None:
    """Fault-injection cutpoint after the transaction intent is durable."""


def _after_registered_registry_update(_path: Path) -> None:
    """Fault-injection cutpoint after registry durability but before journal clear."""


def _before_mutation_journal_clear(_path: Path) -> None:
    """Fault-injection cutpoint immediately before durable journal removal."""


def _after_mutation_journal_clear(_path: Path) -> None:
    """Fault-injection cutpoint after the whole transaction is durable."""


def _before_control_leaf_replace(_path: Path) -> None:
    """Fault-injection cutpoint after staging and before registry/journal CAS."""


def _before_control_leaf_exchange(_path: Path) -> None:
    """Fault-injection cutpoint at the final atomic CAS publication."""


def _before_journal_clear_rename(_path: Path) -> None:
    """Fault-injection cutpoint at the final atomic journal clear."""


def _after_control_leaf_publish_before_stage_cleanup(_path: Path) -> None:
    """Fault-injection cutpoint after atomic publish retains recovery metadata."""


def _after_journal_clear_rename(_path: Path) -> None:
    """Fault-injection cutpoint after journal detach and before exact cleanup."""


class LaneLedgerRegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class _LeafSnapshot:
    identity: tuple[int, int, int, int, int, int] | None
    raw: bytes | None


@dataclass
class _RegistryTransaction:
    root: Path
    root_fd: int
    root_identity: tuple[int, int, int]
    lock_fd: int
    lock_identity: tuple[int, int, int, int]


def create_lane_ledger(
    path: Path,
    ledger: Mapping[str, Any],
    *,
    checkpoint_dir: Path | None = None,
) -> bool:
    """Create and register a production ledger only at an absent address."""

    return persist_lane_ledger(
        path, ledger, checkpoint_dir=checkpoint_dir, _operation="create",
    )


def persist_lane_ledger(
    path: Path,
    ledger: Mapping[str, Any],
    *,
    checkpoint_dir: Path | None = None,
    expected_existing_sha256: str | None = None,
    _operation: str = "replace_registered",
) -> bool:
    """Replace one already-registered ledger, or serve create_lane_ledger.

    Adoption is deliberately owned by ``adopt_lane_ledger_exact`` and never
    rewrites ledger bytes. Paths outside ARC-owned layouts return ``False``.
    """

    located = _owned_location(path, checkpoint_dir=checkpoint_dir)
    if located is None:
        return False
    root, relative, owner = located
    if _operation not in {"create", "replace_registered"}:
        raise LaneLedgerRegistryError("lane ledger persistence operation is invalid")
    if _operation == "replace_registered" and expected_existing_sha256 is None:
        raise LaneLedgerRegistryError(
            "registered ledger replacement requires an exact expected digest"
        )
    encoded = _encode_ledger(ledger)
    new_entry = _entry_for(relative, owner, ledger, encoded)
    _ensure_owned_parent(root, relative.parent)
    with _registry_lock(root) as transaction:
        _reconcile_mutation_journal(transaction)
        registry, registry_snapshot = _read_registry(transaction)
        matches = [
            dict(item) for item in registry["entries"]
            if isinstance(item, Mapping) and item.get("path") == relative.as_posix()
        ]
        if len(matches) > 1:
            raise LaneLedgerRegistryError("lane ledger registry entry is ambiguous")
        old_entry = matches[0] if matches else None
        if old_entry is None and len(registry["entries"]) >= MAX_REGISTRY_ENTRIES:
            raise LaneLedgerRegistryError("lane ledger registry entry limit exceeded")
        parent_fd, identities = _open_verified_parent(transaction, relative.parent)
        leaf_fd: int | None = None
        temporary_name = _ledger_stage_name(relative.name, new_entry["ledger_sha256"])
        journal_snapshot: _LeafSnapshot | None = None
        try:
            before: os.stat_result | None = None
            old_raw: bytes | None = None
            leaf_flags = os.O_RDONLY
            if hasattr(os, "O_NONBLOCK"):
                leaf_flags |= os.O_NONBLOCK
            if hasattr(os, "O_NOFOLLOW"):
                leaf_flags |= os.O_NOFOLLOW
            try:
                leaf_fd = os.open(relative.name, leaf_flags, dir_fd=parent_fd)
            except FileNotFoundError:
                leaf_fd = None
            if leaf_fd is not None:
                before = os.fstat(leaf_fd)
                if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                    raise LaneLedgerRegistryError(
                        "lane ledger is not a singly-linked regular file"
                    )
                old_raw = _read_fd_bounded(leaf_fd, MAX_LEDGER_BYTES)
            if _operation == "create" and (old_entry is not None or old_raw is not None):
                raise LaneLedgerRegistryError(
                    "lane ledger create-only address already exists"
                )
            if _operation == "replace_registered" and old_entry is None:
                raise LaneLedgerRegistryError(
                    "lane ledger replacement requires an existing registry entry"
                )
            if (
                expected_existing_sha256 is not None
                and (
                    old_raw is None
                    or hashlib.sha256(old_raw).hexdigest()
                    != expected_existing_sha256
                )
            ):
                raise LaneLedgerRegistryError(
                    "lane ledger changed before registration"
                )
            if old_entry is not None:
                if old_raw is None:
                    raise LaneLedgerRegistryError("registered lane ledger is missing")
                try:
                    old_ledger = json.loads(old_raw)
                except (UnicodeError, json.JSONDecodeError) as exc:
                    raise LaneLedgerRegistryError("lane ledger is invalid JSON") from exc
                current_entry = (
                    _entry_for(relative, owner, old_ledger, old_raw)
                    if isinstance(old_ledger, dict)
                    else None
                )
                if current_entry != old_entry:
                    raise LaneLedgerRegistryError(
                        "lane ledger bytes changed after registration"
                    )
            if old_raw == encoded and old_entry == new_entry:
                return True

            journal = {
                "schema_version": MUTATION_JOURNAL_SCHEMA_VERSION,
                "path": relative.as_posix(),
                "owner": owner,
                "old_entry": old_entry,
                "old_ledger_sha256": (
                    hashlib.sha256(old_raw).hexdigest() if old_raw is not None else None
                ),
                "new_entry": new_entry,
                "ledger_stage_name": temporary_name if old_raw != encoded else None,
            }
            journal_snapshot = _publish_mutation_journal(transaction, journal)
            _after_mutation_journal_publish(root / relative)

            if old_raw != encoded:
                _stage_exact_bytes(parent_fd, temporary_name, encoded)
                _before_registered_ledger_replace(root / relative)
                _revalidate_parent(transaction, relative.parent, identities)
                _verify_exact_stage(parent_fd, temporary_name, encoded)
                if leaf_fd is None:
                    try:
                        os.stat(relative.name, dir_fd=parent_fd, follow_symlinks=False)
                    except FileNotFoundError:
                        pass
                    else:
                        raise LaneLedgerRegistryError(
                            "lane ledger appeared before initial creation"
                        )
                else:
                    assert before is not None
                    current_raw = _read_fd_bounded(leaf_fd, MAX_LEDGER_BYTES)
                    if current_raw != old_raw:
                        raise LaneLedgerRegistryError(
                            "lane ledger bytes changed before replacement"
                        )
                    current = os.stat(
                        relative.name, dir_fd=parent_fd, follow_symlinks=False,
                    )
                    if _stat_identity(current) != _stat_identity(before):
                        raise LaneLedgerRegistryError(
                            "lane ledger leaf changed before replacement"
                        )
                _publish_ledger_stage(
                    parent_fd,
                    temporary_name,
                    relative.name,
                    encoded,
                    expected=(
                        _LeafSnapshot(_stat_identity(before), old_raw)
                        if before is not None and old_raw is not None
                        else _LeafSnapshot(None, None)
                    ),
                    logical_path=root / relative,
                )
                _after_registered_ledger_replace(root / relative)

            registry_snapshot = _write_registry_entry(
                transaction, registry, registry_snapshot, new_entry,
            )
            _after_registered_registry_update(root / relative)
            _before_mutation_journal_clear(root / relative)
            assert journal_snapshot is not None
            _clear_mutation_journal(transaction, journal_snapshot)
            _after_mutation_journal_clear(root / relative)
            return True
        except (OSError, TypeError) as exc:
            raise LaneLedgerRegistryError(
                "owned lane ledger address changed or cannot be updated safely"
            ) from exc
        finally:
            if leaf_fd is not None:
                try:
                    os.close(leaf_fd)
                except OSError:
                    pass
            if temporary_name:
                _remove_exact_stage(parent_fd, temporary_name)
            try:
                os.close(parent_fd)
            except OSError:
                pass


def adopt_lane_ledger_exact(
    path: Path,
    ledger: Mapping[str, Any],
    *,
    checkpoint_dir: Path | None = None,
    expected_existing_sha256: str | None = None,
) -> bool:
    """Adopt exact existing bytes without rewriting the ledger leaf.

    An already registered exact entry is idempotent. Any different registry
    identity or byte snapshot fails closed.
    """

    located = _owned_location(path, checkpoint_dir=checkpoint_dir)
    if located is None:
        return False
    root, relative, owner = located
    with _registry_lock(root) as transaction:
        _reconcile_mutation_journal(transaction)
        registry, registry_snapshot = _read_registry(transaction)
        matches = [
            dict(item) for item in registry["entries"]
            if isinstance(item, Mapping) and item.get("path") == relative.as_posix()
        ]
        if len(matches) > 1:
            raise LaneLedgerRegistryError("lane ledger registry entry is ambiguous")
        snapshot = _read_relative_snapshot(transaction, relative, MAX_LEDGER_BYTES)
        if snapshot.raw is None:
            raise LaneLedgerRegistryError("lane ledger adoption requires existing bytes")
        digest = hashlib.sha256(snapshot.raw).hexdigest()
        if expected_existing_sha256 is not None and digest != expected_existing_sha256:
            raise LaneLedgerRegistryError("lane ledger changed before exact adoption")
        try:
            persisted = json.loads(snapshot.raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise LaneLedgerRegistryError("lane ledger is not valid UTF-8 JSON") from exc
        if not isinstance(persisted, dict) or persisted != dict(ledger):
            raise LaneLedgerRegistryError(
                "persisted lane ledger differs from exact adoption payload"
            )
        new_entry = _entry_for(relative, owner, persisted, snapshot.raw)
        if matches:
            if matches[0] != new_entry:
                raise LaneLedgerRegistryError(
                    "lane ledger address already has a different registry identity"
                )
            return True
        if len(registry["entries"]) >= MAX_REGISTRY_ENTRIES:
            raise LaneLedgerRegistryError("lane ledger registry entry limit exceeded")
        journal_snapshot = _publish_mutation_journal(transaction, {
            "schema_version": MUTATION_JOURNAL_SCHEMA_VERSION,
            "path": relative.as_posix(),
            "owner": owner,
            "old_entry": None,
            "old_ledger_sha256": digest,
            "new_entry": new_entry,
            "ledger_stage_name": None,
        })
        _after_mutation_journal_publish(root / relative)
        _write_registry_entry(
            transaction, registry, registry_snapshot, new_entry,
        )
        _after_registered_registry_update(root / relative)
        _before_mutation_journal_clear(root / relative)
        _clear_mutation_journal(transaction, journal_snapshot)
        _after_mutation_journal_clear(root / relative)
        return True


def register_lane_ledger(
    path: Path,
    ledger: Mapping[str, Any],
    *,
    checkpoint_dir: Path | None = None,
) -> bool:
    """Compatibility alias for exact, non-rewriting adoption."""

    located = _owned_location(path, checkpoint_dir=checkpoint_dir)
    if located is None:
        return False
    root, relative, _owner = located
    raw = _read_regular_bounded(root, relative, MAX_LEDGER_BYTES)
    return adopt_lane_ledger_exact(
        path,
        ledger,
        checkpoint_dir=checkpoint_dir,
        expected_existing_sha256=hashlib.sha256(raw).hexdigest(),
    )


def registered_lane_ledger_paths(checkpoint_dir: Path) -> list[Path]:
    """Return only hash-bound, currently valid ARC-owned ledgers."""

    root = _checkpoint_root(checkpoint_dir)
    try:
        with _registry_lock(root) as transaction:
            _reconcile_mutation_journal(transaction)
            registry, _snapshot = _read_registry(transaction)
            return _registered_paths_from_snapshot(transaction, registry)
    except LaneLedgerRegistryError:
        return []


def _registered_paths_from_snapshot(
    transaction: _RegistryTransaction, registry: Mapping[str, Any],
) -> list[Path]:
    root = transaction.root
    output: list[Path] = []
    for raw_entry in registry["entries"]:
        if not isinstance(raw_entry, Mapping):
            continue
        try:
            relative = _safe_relative_path(raw_entry.get("path"))
            owner = _owner_for_relative(relative)
            if owner is None or raw_entry.get("owner") != owner:
                continue
            raw = _read_relative_snapshot(
                transaction, relative, MAX_LEDGER_BYTES,
            ).raw
            if raw is None:
                continue
            if hashlib.sha256(raw).hexdigest() != raw_entry.get("ledger_sha256"):
                continue
            ledger = json.loads(raw)
            if not isinstance(ledger, dict):
                continue
            if _entry_for(relative, owner, ledger, raw) != dict(raw_entry):
                continue
        except (LaneLedgerRegistryError, UnicodeError, json.JSONDecodeError, OSError, ValueError):
            continue
        output.append(root / relative)
    return output


def read_registered_lane_ledger(
    checkpoint_dir: Path, path: Path,
) -> tuple[dict[str, Any], str]:
    """Return the exact hash-validated bytes that authorize auto mutation."""

    root = _checkpoint_root(checkpoint_dir)
    try:
        relative = _lexical_absolute(path).relative_to(root)
    except ValueError as exc:
        raise LaneLedgerRegistryError("lane ledger escapes its checkpoint") from exc
    with _registry_lock(root) as transaction:
        _reconcile_mutation_journal(transaction)
        registry, _snapshot = _read_registry(transaction)
        matches = [
            dict(item) for item in registry["entries"]
            if isinstance(item, Mapping) and item.get("path") == relative.as_posix()
        ]
        if len(matches) != 1:
            raise LaneLedgerRegistryError("lane ledger is not registered")
        entry = matches[0]
        owner = _owner_for_relative(relative)
        if owner is None or entry.get("owner") != owner:
            raise LaneLedgerRegistryError("lane ledger registry owner changed")
        raw = _read_relative_snapshot(
            transaction, relative, MAX_LEDGER_BYTES,
        ).raw
        if raw is None:
            raise LaneLedgerRegistryError("registered lane ledger is missing")
        digest = hashlib.sha256(raw).hexdigest()
        if digest != entry.get("ledger_sha256"):
            raise LaneLedgerRegistryError("lane ledger bytes changed after registration")
        try:
            ledger = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise LaneLedgerRegistryError("lane ledger is invalid JSON") from exc
        if not isinstance(ledger, dict) or _entry_for(relative, owner, ledger, raw) != entry:
            raise LaneLedgerRegistryError("lane ledger identity changed after registration")
        return ledger, digest


def mutate_lane_ledger(
    path: Path,
    *,
    mutate: Callable[[dict[str, Any]], Mapping[str, Any]],
    checkpoint_dir: Path | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any] | None:
    """Reapply one production ledger mutation to the latest locked snapshot."""

    located = _owned_location(path, checkpoint_dir=checkpoint_dir)
    if located is None:
        return None
    root, relative, owner = located
    return _mutate_owned_lane_ledger(
        root,
        relative,
        owner,
        mutate=mutate,
        expected_sha256=expected_sha256,
    )


def owned_lane_ledger_root(
    path: Path, *, checkpoint_dir: Path | None = None,
) -> Path | None:
    """Return the canonical registry root for a production-owned ledger path."""

    located = _owned_location(path, checkpoint_dir=checkpoint_dir)
    return located[0] if located is not None else None


def mutate_registered_lane_ledger(
    checkpoint_dir: Path,
    path: Path,
    *,
    expected_sha256: str,
    mutate: Callable[[dict[str, Any]], Mapping[str, Any]],
) -> dict[str, Any]:
    """Strict CAS-update retained for automatic-recovery callers."""

    result = mutate_lane_ledger(
        path,
        checkpoint_dir=checkpoint_dir,
        expected_sha256=expected_sha256,
        mutate=mutate,
    )
    if result is None:  # pragma: no cover - explicit checkpoint rejects this
        raise LaneLedgerRegistryError("lane ledger is outside an ARC-owned layout")
    return result


def _mutate_owned_lane_ledger(
    root: Path,
    relative: Path,
    owner: str,
    *,
    expected_sha256: str | None,
    mutate: Callable[[dict[str, Any]], Mapping[str, Any]],
) -> dict[str, Any]:
    with _registry_lock(root) as transaction:
        _reconcile_mutation_journal(transaction)
        registry, registry_snapshot = _read_registry(transaction)
        matches = [
            dict(item) for item in registry["entries"]
            if isinstance(item, Mapping) and item.get("path") == relative.as_posix()
        ]
        if len(matches) != 1:
            raise LaneLedgerRegistryError("lane ledger is not registered")
        entry = matches[0]
        parent_fd, identities = _open_verified_parent(transaction, relative.parent)
        leaf_fd: int | None = None
        temporary_name = ""
        journal_snapshot: _LeafSnapshot | None = None
        try:
            leaf_flags = os.O_RDONLY
            if hasattr(os, "O_NONBLOCK"):
                leaf_flags |= os.O_NONBLOCK
            if hasattr(os, "O_NOFOLLOW"):
                leaf_flags |= os.O_NOFOLLOW
            leaf_fd = os.open(relative.name, leaf_flags, dir_fd=parent_fd)
            before = os.fstat(leaf_fd)
            if not stat.S_ISREG(before.st_mode):
                raise LaneLedgerRegistryError("lane ledger is not a regular file")
            if before.st_nlink != 1:
                raise LaneLedgerRegistryError("lane ledger must have exactly one link")
            raw = _read_fd_bounded(leaf_fd, MAX_LEDGER_BYTES)
            digest = hashlib.sha256(raw).hexdigest()
            if (
                (expected_sha256 is not None and digest != expected_sha256)
                or digest != entry.get("ledger_sha256")
            ):
                raise LaneLedgerRegistryError("lane ledger snapshot changed before mutation")
            try:
                ledger = json.loads(raw)
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise LaneLedgerRegistryError("lane ledger is invalid JSON") from exc
            if not isinstance(ledger, dict) or _entry_for(relative, owner, ledger, raw) != entry:
                raise LaneLedgerRegistryError("lane ledger identity changed before mutation")
            updated = dict(mutate(dict(ledger)))
            encoded = _encode_ledger(updated)
            new_entry = _entry_for(relative, owner, updated, encoded)
            if encoded == raw and new_entry == entry:
                return updated
            temporary_name = _ledger_stage_name(
                relative.name, new_entry["ledger_sha256"],
            )
            journal_snapshot = _publish_mutation_journal(transaction, {
                "schema_version": MUTATION_JOURNAL_SCHEMA_VERSION,
                "path": relative.as_posix(),
                "owner": owner,
                "old_entry": entry,
                "old_ledger_sha256": digest,
                "new_entry": new_entry,
                "ledger_stage_name": temporary_name,
            })
            _after_mutation_journal_publish(root / relative)
            _stage_exact_bytes(parent_fd, temporary_name, encoded)
            _before_registered_ledger_replace(root / relative)
            _revalidate_parent(transaction, relative.parent, identities)
            current_raw = _read_fd_bounded(leaf_fd, MAX_LEDGER_BYTES)
            if hashlib.sha256(current_raw).hexdigest() != digest:
                raise LaneLedgerRegistryError("lane ledger bytes changed before mutation")
            current = os.stat(relative.name, dir_fd=parent_fd, follow_symlinks=False)
            if (
                current.st_dev, current.st_ino, current.st_mode,
                current.st_nlink, current.st_size, current.st_mtime_ns,
            ) != (
                before.st_dev, before.st_ino, before.st_mode,
                before.st_nlink, before.st_size, before.st_mtime_ns,
            ):
                raise LaneLedgerRegistryError("lane ledger leaf changed before mutation")
            _verify_exact_stage(parent_fd, temporary_name, encoded)
            _publish_ledger_stage(
                parent_fd,
                temporary_name,
                relative.name,
                encoded,
                expected=_LeafSnapshot(_stat_identity(before), raw),
                logical_path=root / relative,
            )
            _after_registered_ledger_replace(root / relative)
            registry_snapshot = _write_registry_entry(
                transaction, registry, registry_snapshot, new_entry,
            )
            _after_registered_registry_update(root / relative)
            _before_mutation_journal_clear(root / relative)
            assert journal_snapshot is not None
            _clear_mutation_journal(transaction, journal_snapshot)
            _after_mutation_journal_clear(root / relative)
            return updated
        except (OSError, TypeError) as exc:
            raise LaneLedgerRegistryError(
                "registered lane ledger address changed or cannot be updated safely"
            ) from exc
        finally:
            if leaf_fd is not None:
                try:
                    os.close(leaf_fd)
                except OSError:
                    pass
            if temporary_name:
                _remove_exact_stage(parent_fd, temporary_name)
            try:
                os.close(parent_fd)
            except OSError:
                pass


def lane_ledger_is_registered(
    checkpoint_dir: Path,
    path: Path,
) -> bool:
    root = _checkpoint_root(checkpoint_dir)
    try:
        relative = _lexical_absolute(path).relative_to(root)
    except ValueError:
        return False
    return relative.as_posix() in {
        candidate.relative_to(root).as_posix()
        for candidate in registered_lane_ledger_paths(root)
    }


def legacy_lane_ledger_paths(checkpoint_dir: Path) -> list[Path]:
    """Read-only discovery for explicit ``resume-native`` compatibility only."""

    root = _checkpoint_root(checkpoint_dir)
    output: list[Path] = []
    for candidate in sorted(root.rglob("*-ledger.json")):
        try:
            relative = _lexical_absolute(candidate).relative_to(root)
            raw = _read_regular_bounded(root, relative, MAX_LEDGER_BYTES)
            value = json.loads(raw)
        except (LaneLedgerRegistryError, UnicodeError, json.JSONDecodeError, OSError, ValueError):
            continue
        if (
            isinstance(value, dict)
            and value.get("schema_version") == LANE_LEDGER_SCHEMA_VERSION
            and value.get("chapter_id")
            and value.get("lane")
        ):
            output.append(root / relative)
    return output


def _entry_for(
    relative: Path,
    owner: str,
    ledger: Mapping[str, Any],
    raw: bytes,
) -> dict[str, Any]:
    if ledger.get("schema_version") != LANE_LEDGER_SCHEMA_VERSION:
        raise LaneLedgerRegistryError("lane ledger schema is unsupported")
    chapter_id = str(ledger.get("chapter_id") or "")
    lane = str(ledger.get("lane") or "")
    try:
        generation = int(ledger.get("generation") or 0)
    except (TypeError, ValueError) as exc:
        raise LaneLedgerRegistryError("lane ledger generation is invalid") from exc
    blocks = ledger.get("blocks")
    if not chapter_id or not lane or generation < 1 or not isinstance(blocks, list):
        raise LaneLedgerRegistryError("lane ledger identity is incomplete")
    segment_ids = [
        str(item.get("segment_id") or "") if isinstance(item, Mapping) else ""
        for item in blocks
    ]
    if not segment_ids or not all(segment_ids) or len(segment_ids) != len(set(segment_ids)):
        raise LaneLedgerRegistryError("lane ledger segment identity is invalid")
    identity = {
        "owner": owner,
        "path": relative.as_posix(),
        "ledger_schema_version": LANE_LEDGER_SCHEMA_VERSION,
        "chapter_id": chapter_id,
        "lane": lane,
        "generation": generation,
        "ordered_segment_ids": segment_ids,
    }
    return {
        **identity,
        "ledger_identity_sha256": _sha256_json(identity),
        "ledger_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _encode_ledger(ledger: Mapping[str, Any]) -> bytes:
    try:
        encoded = (
            json.dumps(dict(ledger), ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LaneLedgerRegistryError("lane ledger is not JSON serializable") from exc
    if len(encoded) > MAX_LEDGER_BYTES:
        raise LaneLedgerRegistryError("lane ledger exceeds its byte limit")
    return encoded


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
    )


def _write_registry_entry(
    transaction: _RegistryTransaction,
    registry: Mapping[str, Any],
    expected: _LeafSnapshot,
    entry: Mapping[str, Any],
) -> _LeafSnapshot:
    path = str(entry.get("path") or "")
    entries = [
        dict(item) for item in registry["entries"]
        if isinstance(item, Mapping) and item.get("path") != path
    ]
    if len(entries) >= MAX_REGISTRY_ENTRIES:
        raise LaneLedgerRegistryError("lane ledger registry entry limit exceeded")
    entries.append(dict(entry))
    entries.sort(key=lambda item: str(item.get("path") or ""))
    return _atomic_write_json(transaction, REGISTRY_FILE_NAME, {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "entries": entries,
    }, expected=expected)


def _owned_location(
    path: Path,
    *,
    checkpoint_dir: Path | None,
) -> tuple[Path, Path, str] | None:
    lexical = _lexical_absolute(path)
    if checkpoint_dir is not None:
        root = _checkpoint_root(checkpoint_dir)
        try:
            relative = lexical.relative_to(root)
        except ValueError as exc:
            raise LaneLedgerRegistryError("lane ledger escapes its checkpoint") from exc
        owner = _owner_for_relative(relative)
        if owner is None:
            raise LaneLedgerRegistryError("lane ledger is outside an ARC-owned layout")
        return root, relative, owner
    for parent in lexical.parents:
        try:
            relative = lexical.relative_to(parent)
        except ValueError:
            continue
        owner = _owner_for_relative(relative)
        if owner is not None:
            return _checkpoint_root(parent), relative, owner
    return None


def _owner_for_relative(relative: Path) -> str | None:
    if relative.is_absolute() or not relative.parts:
        return None
    owner = _OWNED_LAYOUTS.get(relative.parts[0])
    if owner is None:
        return None
    if relative.parts[0] == "chapters" and len(relative.parts) != 3:
        return None
    if relative.parts[0] == "recovery-controls" and len(relative.parts) != 3:
        return None
    if not relative.name.endswith("-ledger.json"):
        return None
    return owner


def _checkpoint_root(path: Path) -> Path:
    root = _lexical_absolute(path)
    if root.is_symlink():
        raise LaneLedgerRegistryError("checkpoint root must not be a symlink")
    return root


def _ensure_owned_parent(root: Path, relative_parent: Path) -> None:
    """Create the bounded owned parent chain without following components."""

    root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(root, _directory_flags())
    try:
        _directory_identity(descriptor)
        for part in relative_parent.parts:
            try:
                child = os.open(part, _directory_flags(), dir_fd=descriptor)
            except FileNotFoundError:
                try:
                    os.mkdir(part, 0o700, dir_fd=descriptor)
                    os.fsync(descriptor)
                except FileExistsError:
                    pass
                child = os.open(part, _directory_flags(), dir_fd=descriptor)
            try:
                _directory_identity(child)
            except BaseException:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
    except BaseException:
        os.close(descriptor)
        raise
    os.close(descriptor)


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _safe_relative_path(value: Any) -> Path:
    text = str(value or "")
    if not text or "\\" in text or "\x00" in text:
        raise LaneLedgerRegistryError("registry path is not a safe relative path")
    relative = Path(text)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise LaneLedgerRegistryError("registry path is not a safe relative path")
    return relative


def _read_registry(
    transaction: _RegistryTransaction,
) -> tuple[dict[str, Any], _LeafSnapshot]:
    snapshot = _read_root_leaf_snapshot(
        transaction, REGISTRY_FILE_NAME, MAX_REGISTRY_BYTES,
    )
    if snapshot.raw is None:
        return ({"schema_version": REGISTRY_SCHEMA_VERSION, "entries": []}, snapshot)
    raw = snapshot.raw
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise LaneLedgerRegistryError("lane ledger registry is invalid JSON") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != REGISTRY_SCHEMA_VERSION
        or not isinstance(value.get("entries"), list)
        or len(value["entries"]) > MAX_REGISTRY_ENTRIES
    ):
        raise LaneLedgerRegistryError("lane ledger registry is invalid")
    return value, snapshot


def _read_mutation_journal(
    transaction: _RegistryTransaction,
) -> tuple[dict[str, Any] | None, _LeafSnapshot]:
    snapshot = _read_root_leaf_snapshot(
        transaction, MUTATION_JOURNAL_FILE_NAME, MAX_REGISTRY_BYTES,
    )
    if snapshot.raw is None:
        return None, snapshot
    raw = snapshot.raw
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise LaneLedgerRegistryError("lane ledger mutation journal is invalid JSON") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != MUTATION_JOURNAL_SCHEMA_VERSION
        or (
            value.get("old_entry") is not None
            and not isinstance(value.get("old_entry"), Mapping)
        )
        or not isinstance(value.get("new_entry"), Mapping)
        or (
            value.get("old_ledger_sha256") is not None
            and not isinstance(value.get("old_ledger_sha256"), str)
        )
        or (
            value.get("ledger_stage_name") is not None
            and not isinstance(value.get("ledger_stage_name"), str)
        )
    ):
        raise LaneLedgerRegistryError("lane ledger mutation journal is invalid")
    return value, snapshot


def _reconcile_mutation_journal(transaction: _RegistryTransaction) -> None:
    """Finish or discard one interrupted ledger/registry mutation under lock."""

    root = transaction.root
    journal, journal_snapshot = _read_mutation_journal(transaction)
    if journal is None:
        return
    relative = _safe_relative_path(journal.get("path"))
    owner = _owner_for_relative(relative)
    old_entry = (
        dict(journal["old_entry"])
        if isinstance(journal.get("old_entry"), Mapping)
        else None
    )
    new_entry = dict(journal["new_entry"])
    old_digest = journal.get("old_ledger_sha256")
    if old_entry is not None and old_digest is None:
        # Backward-compatible recovery for a journal written by the earlier
        # registered-only mutation implementation.
        old_digest = old_entry.get("ledger_sha256")
    if (
        owner is None
        or journal.get("owner") != owner
        or new_entry.get("path") != relative.as_posix()
        or new_entry.get("owner") != owner
        or (
            old_entry is not None
            and (
                old_entry.get("path") != relative.as_posix()
                or old_entry.get("owner") != owner
            )
        )
        or (
            old_digest is not None
            and (
                len(old_digest) != 64
                or any(character not in "0123456789abcdef" for character in old_digest)
            )
        )
    ):
        raise LaneLedgerRegistryError("lane ledger mutation journal identity is invalid")
    raw = _read_relative_snapshot(
        transaction, relative, MAX_LEDGER_BYTES,
    ).raw
    current_entry: dict[str, Any] | None = None
    current_digest = hashlib.sha256(raw).hexdigest() if raw is not None else None
    if raw is not None:
        try:
            ledger = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError):
            ledger = None
        if isinstance(ledger, dict):
            try:
                current_entry = _entry_for(relative, owner, ledger, raw)
            except LaneLedgerRegistryError:
                current_entry = None
    registry, registry_snapshot = _read_registry(transaction)
    matching = [
        dict(item) for item in registry["entries"]
        if isinstance(item, Mapping) and item.get("path") == relative.as_posix()
    ]
    if len(matching) > 1:
        raise LaneLedgerRegistryError("journaled lane ledger registry entry is ambiguous")
    registry_entry = matching[0] if matching else None
    registry_is_old = registry_entry == old_entry
    registry_is_new = registry_entry == new_entry
    ledger_is_new = current_entry == new_entry
    ledger_is_old = (
        (old_digest is None and raw is None)
        or (old_digest is not None and current_digest == old_digest)
    )
    # Prefer completing a published adoption when old and new bytes are
    # identical: the durable journal is the commit intent.
    if ledger_is_new and registry_is_old:
        _write_registry_entry(transaction, registry, registry_snapshot, new_entry)
        _clear_ledger_stage(transaction, relative, journal)
        _clear_mutation_journal(transaction, journal_snapshot)
        return
    if ledger_is_new and registry_is_new:
        _clear_ledger_stage(transaction, relative, journal)
        _clear_mutation_journal(transaction, journal_snapshot)
        return
    if ledger_is_old and registry_is_old:
        _clear_ledger_stage(transaction, relative, journal)
        _clear_mutation_journal(transaction, journal_snapshot)
        return
    raise LaneLedgerRegistryError("journaled lane ledger mutation is in conflict")


def _clear_mutation_journal(
    transaction: _RegistryTransaction, expected: _LeafSnapshot,
) -> None:
    _assert_transaction(transaction)
    current = _read_root_leaf_snapshot(
        transaction, MUTATION_JOURNAL_FILE_NAME, MAX_REGISTRY_BYTES,
    )
    if current != expected or current.raw is None:
        raise LaneLedgerRegistryError(
            "lane ledger mutation journal changed before clear"
        )
    quarantine = (
        f".{MUTATION_JOURNAL_FILE_NAME}.arc-clear-"
        f"{hashlib.sha256(expected.raw).hexdigest()}"
    )
    if _read_named_fd_snapshot(
        transaction.root_fd, quarantine, MAX_REGISTRY_BYTES,
    ).raw is not None:
        raise LaneLedgerRegistryError("journal clear quarantine already exists")
    _before_journal_clear_rename(
        transaction.root / MUTATION_JOURNAL_FILE_NAME,
    )
    _assert_transaction(transaction)
    os.rename(
        MUTATION_JOURNAL_FILE_NAME,
        quarantine,
        src_dir_fd=transaction.root_fd,
        dst_dir_fd=transaction.root_fd,
    )
    _after_journal_clear_rename(
        transaction.root / MUTATION_JOURNAL_FILE_NAME,
    )
    moved = _read_named_fd_snapshot(
        transaction.root_fd, quarantine, MAX_REGISTRY_BYTES,
    )
    if moved != expected:
        # Preserve an unexpected leaf. Restore its public name only when doing
        # so cannot overwrite another writer.
        if _read_named_fd_snapshot(
            transaction.root_fd, MUTATION_JOURNAL_FILE_NAME, MAX_REGISTRY_BYTES,
        ).raw is None:
            os.rename(
                quarantine,
                MUTATION_JOURNAL_FILE_NAME,
                src_dir_fd=transaction.root_fd,
                dst_dir_fd=transaction.root_fd,
            )
        raise LaneLedgerRegistryError(
            "lane ledger mutation journal changed during clear"
        )
    os.unlink(quarantine, dir_fd=transaction.root_fd)
    os.fsync(transaction.root_fd)
    _assert_transaction(transaction)


def _publish_mutation_journal(
    transaction: _RegistryTransaction,
    journal: Mapping[str, Any],
) -> _LeafSnapshot:
    expected = _read_root_leaf_snapshot(
        transaction, MUTATION_JOURNAL_FILE_NAME, MAX_REGISTRY_BYTES,
    )
    if expected.raw is not None:
        raise LaneLedgerRegistryError(
            "lane ledger mutation journal unexpectedly already exists"
        )
    return _atomic_write_json(
        transaction,
        MUTATION_JOURNAL_FILE_NAME,
        journal,
        expected=expected,
    )


def _ledger_stage_name(ledger_name: str, digest: str) -> str:
    return f".{ledger_name}.arc-stage-{digest}"


def _control_stage_name(name: str, encoded: bytes) -> str:
    return f".{name}.arc-stage-{hashlib.sha256(encoded).hexdigest()}"


def _stage_exact_bytes(parent_fd: int, name: str, encoded: bytes) -> None:
    """Create or reuse one content-addressed staged leaf."""

    expected_digest = name.rsplit("-", 1)[-1]
    if hashlib.sha256(encoded).hexdigest() != expected_digest:
        raise LaneLedgerRegistryError("staged control name does not bind its bytes")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=parent_fd)
    except FileExistsError:
        existing = _read_named_fd_snapshot(parent_fd, name, MAX_LEDGER_BYTES)
        if existing.raw != encoded:
            raise LaneLedgerRegistryError("staged control leaf was replaced")
        return
    try:
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(parent_fd)


def _verify_exact_stage(parent_fd: int, name: str, encoded: bytes) -> None:
    snapshot = _read_named_fd_snapshot(parent_fd, name, MAX_LEDGER_BYTES)
    if snapshot.raw != encoded:
        raise LaneLedgerRegistryError("staged control leaf changed before publication")


def _publish_ledger_stage(
    parent_fd: int,
    staged_name: str,
    ledger_name: str,
    encoded: bytes,
    *,
    expected: _LeafSnapshot,
    logical_path: Path,
) -> None:
    """Atomically CAS one ledger leaf, rolling back a late replacement."""

    _before_registered_ledger_exchange(logical_path)
    if expected.raw is None:
        try:
            os.link(
                staged_name,
                ledger_name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise LaneLedgerRegistryError(
                "lane ledger appeared in the final publication window"
            ) from exc
        staged = os.stat(staged_name, dir_fd=parent_fd, follow_symlinks=False)
        target = os.stat(ledger_name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            staged.st_dev, staged.st_ino, staged.st_nlink
        ) != (target.st_dev, target.st_ino, 2):
            raise LaneLedgerRegistryError("initial ledger hardlink publication changed")
        os.unlink(staged_name, dir_fd=parent_fd)
    elif _rename_exchange(parent_fd, staged_name, ledger_name):
        displaced = _read_named_fd_snapshot(
            parent_fd, staged_name, MAX_LEDGER_BYTES,
        )
        if displaced != expected:
            if not _rename_exchange(parent_fd, staged_name, ledger_name):
                raise LaneLedgerRegistryError(
                    "late ledger replacement could not be restored after CAS"
                )
            raise LaneLedgerRegistryError(
                "lane ledger changed in the final publication window"
            )
        _remove_exact_leaf(
            parent_fd,
            staged_name,
            expected,
            error="displaced ledger changed before cleanup",
        )
    else:  # pragma: no cover - fail closed on platforms without atomic exchange
        raise LaneLedgerRegistryError(
            "atomic ledger exchange is unsupported on this platform"
        )
    os.fsync(parent_fd)


def _remove_exact_stage(parent_fd: int, name: str) -> None:
    """Clean a content-addressed stage without unlinking a replacement."""

    digest = name.rsplit("-", 1)[-1]
    try:
        snapshot = _read_named_fd_snapshot(parent_fd, name, MAX_LEDGER_BYTES)
    except (OSError, LaneLedgerRegistryError):
        return
    if (
        snapshot.raw is None
        or hashlib.sha256(snapshot.raw).hexdigest() != digest
    ):
        return
    current = _read_named_fd_snapshot(parent_fd, name, MAX_LEDGER_BYTES)
    if current != snapshot:
        return
    try:
        os.unlink(name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except FileNotFoundError:
        pass


def _clear_ledger_stage(
    transaction: _RegistryTransaction,
    relative: Path,
    journal: Mapping[str, Any],
) -> None:
    raw_name = journal.get("ledger_stage_name")
    if raw_name is None:
        return
    name = str(raw_name)
    new_entry = journal.get("new_entry")
    digest = (
        str(new_entry.get("ledger_sha256") or "")
        if isinstance(new_entry, Mapping)
        else ""
    )
    if name != _ledger_stage_name(relative.name, digest):
        raise LaneLedgerRegistryError("journaled ledger stage identity is invalid")
    parent_fd, _identities = _open_verified_parent(transaction, relative.parent)
    try:
        snapshot = _read_named_fd_snapshot(
            parent_fd, name, MAX_LEDGER_BYTES, allowed_links=(1, 2),
        )
        if snapshot.raw is None:
            return
        target = _read_named_fd_snapshot(
            parent_fd, relative.name, MAX_LEDGER_BYTES, allowed_links=(1, 2),
        )
        stage_matches = hashlib.sha256(snapshot.raw).hexdigest() == digest
        target_matches = (
            target.raw is not None
            and hashlib.sha256(target.raw).hexdigest() == digest
        )
        hardlink_publish = (
            stage_matches
            and snapshot.identity is not None
            and snapshot.identity[3] == 2
            and target.identity == snapshot.identity
        )
        exchange_publish = (
            not stage_matches
            and snapshot.identity is not None
            and snapshot.identity[3] == 1
            and target_matches
        )
        unpublished = (
            stage_matches
            and snapshot.identity is not None
            and snapshot.identity[3] == 1
        )
        if not (hardlink_publish or exchange_publish or unpublished):
            raise LaneLedgerRegistryError("journaled ledger stage was replaced")
        current = _read_named_fd_snapshot(
            parent_fd, name, MAX_LEDGER_BYTES, allowed_links=(1, 2),
        )
        if current != snapshot:
            raise LaneLedgerRegistryError("journaled ledger stage changed")
        os.unlink(name, dir_fd=parent_fd)
        os.fsync(parent_fd)
        _assert_transaction(transaction)
    finally:
        os.close(parent_fd)


def _read_named_fd_snapshot(
    parent_fd: int,
    name: str,
    limit: int,
    *,
    allowed_links: tuple[int, ...] = (1,),
) -> _LeafSnapshot:
    flags = os.O_RDONLY
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except FileNotFoundError:
        return _LeafSnapshot(None, None)
    try:
        raw, value = _read_fd_snapshot_bounded(
            descriptor, limit, allowed_links=allowed_links,
        )
        return _LeafSnapshot(_stat_identity(value), raw)
    finally:
        os.close(descriptor)


def _read_root_leaf_snapshot(
    transaction: _RegistryTransaction, name: str, limit: int,
) -> _LeafSnapshot:
    _assert_transaction(transaction)
    return _read_named_fd_snapshot(transaction.root_fd, name, limit)


def _read_relative_snapshot(
    transaction: _RegistryTransaction, relative: Path, limit: int,
) -> _LeafSnapshot:
    parent_fd, _identities = _open_verified_parent(transaction, relative.parent)
    try:
        return _read_named_fd_snapshot(parent_fd, relative.name, limit)
    finally:
        os.close(parent_fd)


def _read_regular_bounded(root: Path, relative: Path, limit: int) -> bytes:
    try:
        return read_bounded_file(root, relative, max_bytes=limit, suffixes=(".json",))
    except SecureReadError as exc:
        raise LaneLedgerRegistryError("registered control path is unsafe or unavailable") from exc


def _directory_flags() -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _directory_identity(descriptor: int) -> tuple[int, int, int]:
    value = os.fstat(descriptor)
    if not stat.S_ISDIR(value.st_mode):
        raise LaneLedgerRegistryError("registered ledger parent is not a directory")
    return value.st_dev, value.st_ino, value.st_mode


def _open_verified_parent(
    transaction: _RegistryTransaction,
    relative_parent: Path,
) -> tuple[int, tuple[tuple[int, int, int], ...]]:
    """Open and retain the addressed parent without following any component."""

    _assert_transaction(transaction)
    descriptor = os.dup(transaction.root_fd)
    identities: list[tuple[int, int, int]] = []
    try:
        identities.append(_directory_identity(descriptor))
        for part in relative_parent.parts:
            child = os.open(part, _directory_flags(), dir_fd=descriptor)
            try:
                identity = _directory_identity(child)
            except BaseException:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
            identities.append(identity)
        return descriptor, tuple(identities)
    except BaseException:
        os.close(descriptor)
        raise


def _revalidate_parent(
    transaction: _RegistryTransaction,
    relative_parent: Path,
    expected: tuple[tuple[int, int, int], ...],
) -> None:
    """Verify that the lexical address still names the retained parent chain."""

    _assert_transaction(transaction)
    descriptor = os.dup(transaction.root_fd)
    try:
        actual = [_directory_identity(descriptor)]
        if actual[0] != expected[0]:
            raise LaneLedgerRegistryError(
                "registered ledger parent address changed before mutation"
            )
        for index, part in enumerate(relative_parent.parts, start=1):
            child = os.open(part, _directory_flags(), dir_fd=descriptor)
            try:
                identity = _directory_identity(child)
            except BaseException:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
            actual.append(identity)
            if index >= len(expected) or identity != expected[index]:
                raise LaneLedgerRegistryError(
                    "registered ledger parent address changed before mutation"
                )
        if len(actual) != len(expected):
            raise LaneLedgerRegistryError(
                "registered ledger parent address changed before mutation"
            )
    finally:
        os.close(descriptor)


def _read_fd_bounded(
    descriptor: int,
    limit: int,
    *,
    allowed_links: tuple[int, ...] = (1,),
) -> bytes:
    raw, _snapshot = _read_fd_snapshot_bounded(
        descriptor, limit, allowed_links=allowed_links,
    )
    return raw


def _read_fd_snapshot_bounded(
    descriptor: int,
    limit: int,
    *,
    allowed_links: tuple[int, ...] = (1,),
) -> tuple[bytes, os.stat_result]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink not in allowed_links:
        raise LaneLedgerRegistryError("lane ledger is not a singly-linked regular file")
    chunks: list[bytes] = []
    remaining = limit + 1
    while remaining:
        chunk = os.read(descriptor, min(1024 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    raw = b"".join(chunks)
    if len(raw) > limit:
        raise LaneLedgerRegistryError("lane ledger exceeds its byte limit")
    after = os.fstat(descriptor)
    before_identity = (
        before.st_dev, before.st_ino, before.st_mode, before.st_nlink,
        before.st_size, before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev, after.st_ino, after.st_mode, after.st_nlink,
        after.st_size, after.st_mtime_ns,
    )
    if before_identity != after_identity or after.st_size != len(raw):
        raise LaneLedgerRegistryError("lane ledger changed while reading")
    return raw, after


def _write_all(descriptor: int, value: bytes) -> None:
    offset = 0
    while offset < len(value):
        written = os.write(descriptor, value[offset:])
        if written <= 0:
            raise OSError("short write while staging registered lane ledger")
        offset += written


def _assert_transaction(transaction: _RegistryTransaction) -> None:
    """Bind every control mutation to one root and one named lock inode."""

    try:
        reopened = os.open(transaction.root, _directory_flags())
    except OSError as exc:
        raise LaneLedgerRegistryError(
            "checkpoint root changed during ledger transaction"
        ) from exc
    try:
        if _directory_identity(reopened) != transaction.root_identity:
            raise LaneLedgerRegistryError(
                "checkpoint root changed during ledger transaction"
            )
    finally:
        os.close(reopened)
    retained = os.fstat(transaction.root_fd)
    if (
        retained.st_dev, retained.st_ino, retained.st_mode
    ) != transaction.root_identity:
        raise LaneLedgerRegistryError(
            "retained checkpoint root changed during ledger transaction"
        )
    held = os.fstat(transaction.lock_fd)
    held_identity = held.st_dev, held.st_ino, held.st_mode, held.st_nlink
    try:
        named = os.stat(
            _REGISTRY_LOCK_NAME,
            dir_fd=transaction.root_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError as exc:
        raise LaneLedgerRegistryError(
            "lane ledger registry lock disappeared during transaction"
        ) from exc
    named_identity = named.st_dev, named.st_ino, named.st_mode, named.st_nlink
    if (
        held_identity != transaction.lock_identity
        or named_identity != transaction.lock_identity
        or not stat.S_ISREG(named.st_mode)
        or named.st_nlink != 1
    ):
        raise LaneLedgerRegistryError(
            "lane ledger registry lock inode changed during transaction"
        )


def _cleanup_control_stages(transaction: _RegistryTransaction) -> None:
    """Remove only self-hash-valid control stages abandoned by a dead owner."""

    prefixes = {
        f".{REGISTRY_FILE_NAME}.arc-stage-": REGISTRY_FILE_NAME,
        f".{MUTATION_JOURNAL_FILE_NAME}.arc-stage-": MUTATION_JOURNAL_FILE_NAME,
        f".{MUTATION_JOURNAL_FILE_NAME}.arc-clear-": None,
    }
    for name in os.listdir(transaction.root_fd):
        prefix = next((item for item in prefixes if name.startswith(item)), None)
        if prefix is None:
            continue
        digest = name[len(prefix):]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            continue
        snapshot = _read_named_fd_snapshot(
            transaction.root_fd,
            name,
            MAX_REGISTRY_BYTES,
            allowed_links=(1, 2),
        )
        if snapshot.raw is None:
            continue
        stage_matches = hashlib.sha256(snapshot.raw).hexdigest() == digest
        target_name = prefixes[prefix]
        target = (
            _read_named_fd_snapshot(
                transaction.root_fd,
                target_name,
                MAX_REGISTRY_BYTES,
                allowed_links=(1, 2),
            )
            if target_name is not None
            else _LeafSnapshot(None, None)
        )
        target_matches = (
            target.raw is not None
            and hashlib.sha256(target.raw).hexdigest() == digest
        )
        hardlink_publish = (
            stage_matches
            and snapshot.identity is not None
            and snapshot.identity[3] == 2
            and target.identity == snapshot.identity
        )
        exchange_publish = (
            not stage_matches
            and snapshot.identity is not None
            and snapshot.identity[3] == 1
            and target_matches
        )
        unpublished = (
            stage_matches
            and snapshot.identity is not None
            and snapshot.identity[3] == 1
        )
        if not (hardlink_publish or exchange_publish or unpublished):
            raise LaneLedgerRegistryError("staged registry control was replaced")
        _assert_transaction(transaction)
        current = _read_named_fd_snapshot(
            transaction.root_fd,
            name,
            MAX_REGISTRY_BYTES,
            allowed_links=(1, 2),
        )
        if current != snapshot:
            raise LaneLedgerRegistryError("staged registry control changed before cleanup")
        os.unlink(name, dir_fd=transaction.root_fd)
        os.fsync(transaction.root_fd)
        _assert_transaction(transaction)


@contextmanager
def _registry_lock(root: Path) -> Iterator[_RegistryTransaction]:
    root.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    with _REGISTRY_THREAD_LOCK:
        root_fd = os.open(root, _directory_flags())
        transaction: _RegistryTransaction | None = None
        validation_error: BaseException | None = None
        try:
            descriptor = os.open(_REGISTRY_LOCK_NAME, flags, 0o600, dir_fd=root_fd)
        except BaseException:
            os.close(root_fd)
            raise
        try:
            if os.name == "nt":  # pragma: no cover - exercised on Windows CI
                import msvcrt

                msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX)
            root_value = os.fstat(root_fd)
            lock_value = os.fstat(descriptor)
            if (
                not stat.S_ISREG(lock_value.st_mode)
                or lock_value.st_nlink != 1
            ):
                raise LaneLedgerRegistryError(
                    "lane ledger registry lock is not a singly-linked regular file"
                )
            transaction = _RegistryTransaction(
                root=root,
                root_fd=root_fd,
                root_identity=(root_value.st_dev, root_value.st_ino, root_value.st_mode),
                lock_fd=descriptor,
                lock_identity=(
                    lock_value.st_dev, lock_value.st_ino,
                    lock_value.st_mode, lock_value.st_nlink,
                ),
            )
            _assert_transaction(transaction)
            _cleanup_control_stages(transaction)
            yield transaction
            _assert_transaction(transaction)
        finally:
            if transaction is not None:
                try:
                    _assert_transaction(transaction)
                except BaseException as exc:
                    validation_error = exc
            try:
                if os.name == "nt":  # pragma: no cover - exercised on Windows CI
                    import msvcrt

                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
                os.close(root_fd)
            if validation_error is not None:
                raise validation_error


def _atomic_write_json(
    transaction: _RegistryTransaction,
    name: str,
    value: Mapping[str, Any],
    *,
    expected: _LeafSnapshot,
) -> _LeafSnapshot:
    encoded = (json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ) + "\n").encode("utf-8")
    if len(encoded) > MAX_REGISTRY_BYTES:
        raise LaneLedgerRegistryError("lane ledger registry exceeds its byte limit")
    _assert_transaction(transaction)
    if _read_root_leaf_snapshot(transaction, name, MAX_REGISTRY_BYTES) != expected:
        raise LaneLedgerRegistryError(f"{name} changed before atomic publication")
    temporary = _control_stage_name(name, encoded)
    _stage_exact_bytes(transaction.root_fd, temporary, encoded)
    _before_control_leaf_replace(transaction.root / name)
    _assert_transaction(transaction)
    if _read_root_leaf_snapshot(transaction, name, MAX_REGISTRY_BYTES) != expected:
        raise LaneLedgerRegistryError(f"{name} changed before atomic publication")
    staged = _read_named_fd_snapshot(
        transaction.root_fd, temporary, MAX_REGISTRY_BYTES,
    )
    if staged.raw != encoded:
        raise LaneLedgerRegistryError(f"{name} staged bytes changed")
    _before_control_leaf_exchange(transaction.root / name)
    _assert_transaction(transaction)
    if expected.raw is None:
        try:
            os.link(
                temporary,
                name,
                src_dir_fd=transaction.root_fd,
                dst_dir_fd=transaction.root_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise LaneLedgerRegistryError(
                f"{name} appeared before atomic publication"
            ) from exc
        _after_control_leaf_publish_before_stage_cleanup(transaction.root / name)
        current_stage = os.stat(
            temporary, dir_fd=transaction.root_fd, follow_symlinks=False,
        )
        current_identity = _stat_identity(current_stage)
        if (
            staged.identity is None
            or current_identity[:3] != staged.identity[:3]
            or current_identity[4:] != staged.identity[4:]
            or current_stage.st_nlink != 2
        ):
            raise LaneLedgerRegistryError(f"{name} staged leaf changed after link")
        os.unlink(temporary, dir_fd=transaction.root_fd)
    elif _rename_exchange(transaction.root_fd, temporary, name):
        _after_control_leaf_publish_before_stage_cleanup(transaction.root / name)
        displaced = _read_named_fd_snapshot(
            transaction.root_fd, temporary, MAX_REGISTRY_BYTES,
        )
        if displaced != expected:
            if not _rename_exchange(transaction.root_fd, temporary, name):
                raise LaneLedgerRegistryError(
                    f"{name} changed and could not be restored after CAS"
                )
            raise LaneLedgerRegistryError(f"{name} changed during atomic publication")
        _remove_exact_leaf(
            transaction.root_fd, temporary, expected,
            error="displaced registry control changed before cleanup",
        )
    else:  # pragma: no cover - non-Linux portability fallback
        if _read_root_leaf_snapshot(transaction, name, MAX_REGISTRY_BYTES) != expected:
            raise LaneLedgerRegistryError(f"{name} changed before atomic publication")
        os.replace(
            temporary, name,
            src_dir_fd=transaction.root_fd, dst_dir_fd=transaction.root_fd,
        )
    os.fsync(transaction.root_fd)
    _assert_transaction(transaction)
    published = _read_root_leaf_snapshot(transaction, name, MAX_REGISTRY_BYTES)
    if published.raw != encoded:
        raise LaneLedgerRegistryError(f"{name} publication could not be verified")
    return published


def _rename_exchange(parent_fd: int, first: str, second: str) -> bool:
    """Atomically exchange two names where Linux renameat2 is available."""

    if os.name != "posix":
        return False
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        return False
    renameat2.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_fd, os.fsencode(first), parent_fd, os.fsencode(second), 2,
    )
    if result == 0:
        return True
    error = ctypes.get_errno()
    if error in {getattr(os, "ENOSYS", 38), 22, 95}:
        return False
    raise OSError(error, os.strerror(error))


def _remove_exact_leaf(
    parent_fd: int,
    name: str,
    expected: _LeafSnapshot,
    *,
    error: str,
) -> None:
    current = _read_named_fd_snapshot(parent_fd, name, MAX_REGISTRY_BYTES)
    if current != expected:
        raise LaneLedgerRegistryError(error)
    os.unlink(name, dir_fd=parent_fd)


def _sha256_json(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
