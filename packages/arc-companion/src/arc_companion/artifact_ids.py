from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Iterator, Literal, Mapping

from .io import sha256_json


ARTIFACT_ID_RECEIPT_VERSION = "arc.companion.directory-identity.v1"
ARTIFACT_ID_RECEIPT_NAME = "directory-identity.json"
ARTIFACT_ID_LOCK_NAME = ".artifact-ids.lock"
ARTIFACT_ID_RECEIPT_MAX_BYTES = 64 * 1024
_IDENTITY = re.compile(r"[0-9a-f]{64}")
_NONCE = re.compile(r"[0-9a-f]{32}")
_PREFIX_LENGTHS = (*range(12, 64, 4), 64)
_ALIAS_KEYS = {
    "schema_version",
    "kind",
    "alias_identity",
    "legacy_fingerprint",
    "content_fingerprint",
    "legacy_checkpoint_dir",
    "legacy_workers_per_lane",
}
_PDF_RENDER_PAYLOAD_KEYS = {
    "content_sha256",
    "render_recipe_sha256",
    "validator_version",
    "stem",
}


class ArtifactIdError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArtifactAllocation:
    path: Path
    identity: str
    display_id: str
    prefix_length: int
    disposition: Literal["created", "adopted", "legacy"]
    receipt_path: Path | None
    receipt_sha256: str | None
    payload: Mapping[str, object] | None = None
    nonce: str | None = None

    @property
    def legacy(self) -> bool:
        return self.disposition == "legacy"


@dataclass(frozen=True)
class ArtifactAliasReceipt:
    path: Path
    value: Mapping[str, object]
    sha256: str
    disposition: Literal["created", "adopted"]


def render_artifact_identity(
    *, kind: str, payload: Mapping[str, object], nonce: str,
) -> str:
    if not isinstance(nonce, str) or not _NONCE.fullmatch(nonce):
        raise ArtifactIdError("render artifact nonce is invalid")
    if kind == "pdf-render":
        _validate_pdf_render_payload(payload)
    return sha256_json({
        "kind": str(kind),
        "payload": dict(payload),
        "nonce": nonce,
    })


def allocate_artifact_dir(
    root: Path,
    identity: str,
    *,
    kind: str,
    stem: str | None = None,
    payload: Mapping[str, object] | None = None,
    nonce: str | None = None,
    allow_legacy: bool = True,
) -> ArtifactAllocation:
    root = _prepare_root(Path(root))
    identity = _full_identity(identity)
    _validate_stem(stem)
    _validate_requested_identity(
        identity, kind=kind, stem=stem, payload=payload, nonce=nonce,
    )
    with _allocation_lock(root):
        full_candidate = root / identity
        try:
            full_mode = full_candidate.lstat().st_mode
        except FileNotFoundError:
            full_mode = None
        if full_mode is not None:
            if stat.S_ISLNK(full_mode) or not stat.S_ISDIR(full_mode):
                raise ArtifactIdError("full identity candidate is invalid")
            full_receipt = full_candidate / ARTIFACT_ID_RECEIPT_NAME
            try:
                full_receipt.lstat()
            except FileNotFoundError:
                if allow_legacy and kind == "checkpoint" and stem is None:
                    return ArtifactAllocation(
                        full_candidate, identity, identity, 64, "legacy",
                        None, None,
                    )
            else:
                allocation, receipt = _inspect_artifact_dir(
                    root,
                    full_candidate,
                    expected_identity=identity,
                    kind=kind,
                    stem=stem,
                )
                _match_requested_render(receipt, payload=payload, nonce=nonce)
                return allocation
        for length in _PREFIX_LENGTHS:
            display_id = identity[:length]
            name = f"{stem}-{display_id}" if stem is not None else display_id
            candidate = root / name
            if candidate.parent != root:
                raise ArtifactIdError("artifact candidate escapes its root")
            occupied = False
            for existing in _prefix_occupants(root, display_id):
                mode = existing.lstat().st_mode
                if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                    raise ArtifactIdError("artifact candidate is invalid")
                receipt_path = existing / ARTIFACT_ID_RECEIPT_NAME
                try:
                    receipt_path.lstat()
                except FileNotFoundError:
                    receipt_present = False
                else:
                    receipt_present = True
                if receipt_present:
                    allocation, receipt = _inspect_artifact_dir(
                        root, existing,
                    )
                    if allocation.display_id != display_id:
                        continue
                    if allocation.identity != identity:
                        occupied = True
                        continue
                    if existing != candidate:
                        raise ArtifactIdError(
                            "artifact identity is bound to a different name"
                        )
                    if allocation.path.name != name:
                        raise ArtifactIdError(
                            "artifact directory and request differ"
                        )
                    if str(receipt.get("kind") or "") != kind:
                        raise ArtifactIdError("artifact kind differs")
                    if str(receipt.get("stem") or "") != (stem or ""):
                        raise ArtifactIdError("artifact stem differs")
                    _match_requested_render(
                        receipt, payload=payload, nonce=nonce,
                    )
                    return allocation
                if (
                    allow_legacy
                    and kind == "checkpoint"
                    and length == 64
                    and stem is None
                    and existing.name == identity
                ):
                    return ArtifactAllocation(
                        existing, identity, identity, 64, "legacy",
                        None, None,
                    )
                occupied = True
            if occupied:
                continue
            candidate.mkdir(mode=0o700)
            os.chmod(candidate, 0o700, follow_symlinks=False)
            receipt_value = _receipt_value(
                identity=identity,
                display_id=display_id,
                prefix_length=length,
                kind=kind,
                stem=stem,
                payload=payload,
                nonce=nonce,
            )
            receipt_path = candidate / ARTIFACT_ID_RECEIPT_NAME
            try:
                _write_receipt(receipt_path, receipt_value)
                _fsync_directory(candidate)
                _fsync_directory(root)
                _value, receipt_sha = _read_receipt(receipt_path)
            except BaseException:
                try:
                    candidate.rmdir()
                except OSError:
                    pass
                raise
            return ArtifactAllocation(
                candidate,
                identity,
                display_id,
                length,
                "created",
                receipt_path,
                receipt_sha,
                dict(payload) if payload is not None else None,
                nonce,
            )
    raise ArtifactIdError("artifact identity allocation exhausted")


def _prefix_occupants(root: Path, display_id: str) -> tuple[Path, ...]:
    """Return direct children occupying one root-global display prefix."""

    suffix = f"-{display_id}"
    try:
        entries = tuple(root.iterdir())
    except OSError as exc:
        raise ArtifactIdError("artifact root is unreadable") from exc
    return tuple(sorted(
        (
            entry for entry in entries
            if entry.name == display_id or entry.name.endswith(suffix)
        ),
        key=lambda entry: entry.name,
    ))


def resolve_artifact_dir(
    root: Path,
    value: str | Path,
    *,
    expected_identity: str | None = None,
    kind: str | None = None,
    stem: str | None = None,
    payload: Mapping[str, object] | None = None,
    nonce: str | None = None,
    allow_legacy: bool = True,
) -> ArtifactAllocation:
    root = _existing_root(Path(root))
    _validate_stem(stem)
    raw = Path(value)
    candidate = raw if raw.is_absolute() else root / raw
    _require_contained_components(root, candidate)
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ArtifactIdError("artifact path escapes its root") from exc
    if candidate.parent != root:
        raise ArtifactIdError("artifact directory must be a direct root child")
    if not stat.S_ISDIR(candidate.lstat().st_mode):
        raise ArtifactIdError("artifact directory is missing or invalid")
    expected = _full_identity(expected_identity) if expected_identity else None
    receipt_path = candidate / ARTIFACT_ID_RECEIPT_NAME
    try:
        receipt_path.lstat()
    except FileNotFoundError:
        if (
            allow_legacy
            and kind == "checkpoint"
            and expected is not None
            and candidate.name == expected
            and _IDENTITY.fullmatch(candidate.name)
            and stem is None
            and payload is None
            and nonce is None
        ):
            return ArtifactAllocation(
                candidate, expected, expected, 64, "legacy", None, None,
            )
        raise ArtifactIdError("artifact identity receipt is missing")
    allocation, receipt = _inspect_artifact_dir(
        root,
        candidate,
        expected_identity=expected,
        kind=kind,
        stem=stem,
    )
    _match_requested_render(receipt, payload=payload, nonce=nonce)
    return allocation


def ensure_artifact_alias_receipt(
    root: Path,
    identity: str,
    value: Mapping[str, object],
) -> ArtifactAliasReceipt:
    """Create or strictly adopt one identity-keyed alias receipt."""
    root = _prepare_root(Path(root))
    identity = _full_identity(identity)
    normalized = dict(value)
    _validate_alias_value(identity, normalized)
    with _allocation_lock(root):
        alias_root = root / "aliases"
        try:
            mode = alias_root.lstat().st_mode
        except FileNotFoundError:
            alias_root.mkdir(mode=0o700)
            os.chmod(alias_root, 0o700, follow_symlinks=False)
            _fsync_directory(root)
        else:
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise ArtifactIdError("artifact alias root is invalid")
        path = alias_root / f"{identity}.json"
        try:
            path.lstat()
        except FileNotFoundError:
            pass
        else:
            existing, receipt_sha = _read_receipt(path)
            _validate_alias_value(identity, existing)
            if dict(existing) != normalized:
                raise ArtifactIdError("artifact alias receipt conflicts")
            return ArtifactAliasReceipt(
                path, existing, receipt_sha, "adopted",
            )
        _write_receipt(path, normalized)
        _fsync_directory(alias_root)
        _fsync_directory(root)
        existing, receipt_sha = _read_receipt(path)
        if dict(existing) != normalized:
            raise ArtifactIdError("artifact alias receipt changed")
        return ArtifactAliasReceipt(
            path, existing, receipt_sha, "created",
        )


def read_artifact_alias_receipt(
    root: Path,
    identity: str,
) -> ArtifactAliasReceipt | None:
    """Read one identity-keyed alias receipt without following links."""
    root = _existing_root(Path(root))
    identity = _full_identity(identity)
    alias_root = root / "aliases"
    try:
        mode = alias_root.lstat().st_mode
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ArtifactIdError("artifact alias root is invalid")
    path = alias_root / f"{identity}.json"
    try:
        path.lstat()
    except FileNotFoundError:
        return None
    value, receipt_sha = _read_receipt(path)
    _validate_alias_value(identity, value)
    return ArtifactAliasReceipt(
        path, value, receipt_sha, "adopted",
    )


def _inspect_artifact_dir(
    root: Path,
    candidate: Path,
    *,
    expected_identity: str | None = None,
    kind: str | None = None,
    stem: str | None = None,
) -> tuple[ArtifactAllocation, Mapping[str, object]]:
    if candidate.parent != root:
        raise ArtifactIdError("artifact directory must be a direct root child")
    receipt_path = candidate / ARTIFACT_ID_RECEIPT_NAME
    receipt, receipt_sha = _read_receipt(receipt_path)
    identity = _full_identity(str(receipt.get("full_identity") or ""))
    prefix_length = receipt.get("prefix_length")
    if prefix_length not in _PREFIX_LENGTHS:
        raise ArtifactIdError("artifact identity prefix length is invalid")
    display_id = identity[:prefix_length]
    receipt_kind = str(receipt.get("kind") or "")
    receipt_stem = str(receipt.get("stem") or "")
    _validate_stem(receipt_stem if receipt_stem else None)
    expected_keys = {
        "schema_version",
        "kind",
        "full_identity",
        "display_id",
        "prefix_length",
        "stem",
    }
    if receipt_kind == "pdf-render":
        expected_keys |= {"payload", "nonce"}
    if set(receipt) != expected_keys:
        raise ArtifactIdError("artifact identity receipt shape is invalid")
    if receipt.get("schema_version") != ARTIFACT_ID_RECEIPT_VERSION:
        raise ArtifactIdError("artifact identity receipt version is invalid")
    expected_name = (
        f"{receipt_stem}-{display_id}" if receipt_stem else display_id
    )
    if receipt.get("display_id") != display_id or candidate.name != expected_name:
        raise ArtifactIdError("artifact directory and receipt differ")
    if receipt_kind == "checkpoint":
        if receipt_stem:
            raise ArtifactIdError("artifact checkpoint stem is invalid")
    elif receipt_kind == "pdf-render":
        recorded_payload = receipt.get("payload")
        recorded_nonce = receipt.get("nonce")
        if not isinstance(recorded_payload, Mapping):
            raise ArtifactIdError("artifact render identity payload is invalid")
        _validate_pdf_render_payload(
            recorded_payload, expected_stem=receipt_stem,
        )
        if (
            not isinstance(recorded_nonce, str)
            or not _NONCE.fullmatch(recorded_nonce)
        ):
            raise ArtifactIdError("render artifact nonce is invalid")
        if render_artifact_identity(
            kind=receipt_kind,
            payload=recorded_payload,
            nonce=recorded_nonce,
        ) != identity:
            raise ArtifactIdError("render artifact full identity differs")
    elif "payload" in receipt or "nonce" in receipt:
        raise ArtifactIdError("non-render artifact has render identity fields")
    if expected_identity is not None and identity != expected_identity:
        raise ArtifactIdError("artifact identity differs")
    if kind is not None and receipt_kind != kind:
        raise ArtifactIdError("artifact kind differs")
    if stem is not None and receipt_stem != stem:
        raise ArtifactIdError("artifact stem differs")
    if stem is None and kind == "checkpoint" and receipt_stem:
        raise ArtifactIdError("artifact stem differs")
    return (
        ArtifactAllocation(
            candidate,
            identity,
            display_id,
            prefix_length,
            "adopted",
            receipt_path,
            receipt_sha,
            (
                dict(receipt["payload"])
                if isinstance(receipt.get("payload"), Mapping)
                else None
            ),
            (
                str(receipt["nonce"])
                if receipt.get("nonce") is not None else None
            ),
        ),
        receipt,
    )


def _receipt_value(
    *,
    identity: str,
    display_id: str,
    prefix_length: int,
    kind: str,
    stem: str | None,
    payload: Mapping[str, object] | None,
    nonce: str | None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": ARTIFACT_ID_RECEIPT_VERSION,
        "kind": str(kind),
        "full_identity": identity,
        "display_id": display_id,
        "prefix_length": prefix_length,
        "stem": stem or "",
    }
    if kind == "pdf-render":
        value.update({"payload": dict(payload or {}), "nonce": nonce})
    return value


def _validate_requested_identity(
    identity: str,
    *,
    kind: str,
    stem: str | None,
    payload: Mapping[str, object] | None,
    nonce: str | None,
) -> None:
    if kind == "pdf-render":
        if payload is None or nonce is None:
            raise ArtifactIdError(
                "render identity payload and nonce are required"
            )
        _validate_pdf_render_payload(payload, expected_stem=stem)
        if render_artifact_identity(
            kind=kind, payload=payload, nonce=nonce,
        ) != identity:
            raise ArtifactIdError("render artifact full identity differs")
    elif payload is not None or nonce is not None:
        raise ArtifactIdError(
            "non-render artifact cannot have render identity fields"
        )


def _match_requested_render(
    receipt: Mapping[str, object],
    *,
    payload: Mapping[str, object] | None,
    nonce: str | None,
) -> None:
    if payload is None and nonce is None:
        return
    if receipt.get("payload") != dict(payload or {}) or receipt.get("nonce") != nonce:
        raise ArtifactIdError("artifact render identity payload differs")


def _validate_pdf_render_payload(
    payload: Mapping[str, object],
    *,
    expected_stem: str | None = None,
) -> None:
    if set(payload) != _PDF_RENDER_PAYLOAD_KEYS:
        raise ArtifactIdError("pdf render identity payload shape is invalid")
    if not _IDENTITY.fullmatch(str(payload.get("content_sha256") or "")):
        raise ArtifactIdError("pdf render content identity is invalid")
    if not _IDENTITY.fullmatch(
        str(payload.get("render_recipe_sha256") or "")
    ):
        raise ArtifactIdError("pdf render recipe identity is invalid")
    validator = payload.get("validator_version")
    if not isinstance(validator, str) or not validator or len(validator) > 256:
        raise ArtifactIdError("pdf render validator identity is invalid")
    render_stem = payload.get("stem")
    if not isinstance(render_stem, str):
        raise ArtifactIdError("pdf render stem identity is invalid")
    _validate_stem(render_stem)
    if expected_stem is not None and render_stem != expected_stem:
        raise ArtifactIdError("pdf render stem identity differs")


def _validate_alias_value(
    identity: str, value: Mapping[str, object],
) -> None:
    if set(value) != _ALIAS_KEYS:
        raise ArtifactIdError("artifact alias receipt shape is invalid")
    if value.get("schema_version") != "arc.companion.checkpoint-alias.v1":
        raise ArtifactIdError("artifact alias receipt version is invalid")
    if value.get("kind") != "workers-to-total-concurrency-budget":
        raise ArtifactIdError("artifact alias receipt kind is invalid")
    if value.get("alias_identity") != identity:
        raise ArtifactIdError("artifact alias receipt identity differs")
    if value.get("content_fingerprint") != identity:
        raise ArtifactIdError("artifact alias content identity differs")
    _full_identity(str(value.get("legacy_fingerprint") or ""))
    legacy_path = value.get("legacy_checkpoint_dir")
    if not isinstance(legacy_path, str) or not legacy_path:
        raise ArtifactIdError("artifact alias checkpoint path is invalid")
    workers = value.get("legacy_workers_per_lane")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ArtifactIdError("artifact alias worker count is invalid")


def _validate_stem(stem: str | None) -> None:
    if stem is None:
        return
    if (
        not isinstance(stem, str)
        or not stem
        or stem in {".", ".."}
        or "/" in stem
        or "\\" in stem
        or Path(stem).name != stem
    ):
        raise ArtifactIdError("artifact stem is invalid")


def _full_identity(value: str | None) -> str:
    normalized = str(value or "").casefold()
    if not _IDENTITY.fullmatch(normalized):
        raise ArtifactIdError("artifact identity must be a full SHA-256")
    return normalized


def _prepare_root(root: Path) -> Path:
    _reject_symlink_components(root)
    try:
        root.mkdir(parents=True, mode=0o700)
    except FileExistsError:
        created = False
    else:
        created = True
    if created:
        os.chmod(root, 0o700, follow_symlinks=False)
    return _existing_root(root)


def _existing_root(root: Path) -> Path:
    _reject_symlink_components(root)
    if root.is_symlink():
        raise ArtifactIdError("artifact root is invalid")
    root = root.resolve()
    if not root.is_dir() or not stat.S_ISDIR(root.lstat().st_mode):
        raise ArtifactIdError("artifact root is invalid")
    return root


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ArtifactIdError("artifact path contains a symbolic link")


def _require_contained_components(root: Path, candidate: Path) -> None:
    absolute = candidate if candidate.is_absolute() else root / candidate
    try:
        relative = absolute.relative_to(root)
    except ValueError as exc:
        raise ArtifactIdError("artifact path escapes its root") from exc
    if ".." in relative.parts:
        raise ArtifactIdError("artifact path escapes its root")
    current = root
    for part in relative.parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as exc:
            raise ArtifactIdError("artifact directory is missing") from exc
        if stat.S_ISLNK(mode):
            raise ArtifactIdError("artifact path contains a symbolic link")


def _read_receipt(path: Path) -> tuple[Mapping[str, object], str]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ArtifactIdError("artifact identity receipt is unreadable") from exc
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode):
            raise ArtifactIdError("artifact identity receipt is not regular")
        if details.st_size > ARTIFACT_ID_RECEIPT_MAX_BYTES:
            raise ArtifactIdError("artifact identity receipt is oversized")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(
                descriptor,
                min(8192, ARTIFACT_ID_RECEIPT_MAX_BYTES + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > ARTIFACT_ID_RECEIPT_MAX_BYTES:
                raise ArtifactIdError(
                    "artifact identity receipt is oversized"
                )
        raw = b"".join(chunks)
    finally:
        os.close(descriptor)
    try:
        value = json.loads(raw)
    except (UnicodeError, ValueError) as exc:
        raise ArtifactIdError("artifact identity receipt is unreadable") from exc
    if not isinstance(value, dict):
        raise ArtifactIdError("artifact identity receipt is invalid")
    return value, hashlib.sha256(raw).hexdigest()


def _write_receipt(path: Path, value: Mapping[str, object]) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent,
    )
    os.fchmod(descriptor, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


@contextmanager
def _allocation_lock(root: Path) -> Iterator[None]:
    path = root / ARTIFACT_ID_LOCK_NAME
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise ArtifactIdError("artifact allocation lock is invalid") from exc
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode):
            raise ArtifactIdError("artifact allocation lock is not regular")
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
