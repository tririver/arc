"""Portable, verified archives for logical ARC paper-cache entries."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable, Mapping, Sequence

from ac_document import FullTextCatalogAdminEntry
from ac_jobs import canonical_json_bytes as _canonical_json_bytes

from ._cache_admin import CACHE_INDEX_SCHEMA, CacheAdministrator, CacheEntry
from ._cache_root import resolve_cache_root
from .html_dependencies import (
    AR5IV_HTML_ACQUISITION_NAMESPACE,
    AR5IV_HTML_DEPENDENCY_NAMESPACE,
    ARXIV_HTML_ACQUISITION_NAMESPACE,
    ARXIV_HTML_DEPENDENCY_NAMESPACE,
    bundle_resource_identities,
)
from .reference_cache import CachedResourceRef, ReferenceMaterialCache
from ac_jobs import file_matches_sha256
from .source_repository import SourceRepository
from .sources import SourceFormat


CACHE_ARCHIVE_SCHEMA = "arc.paper.cache_archive.v2"
_ARCHIVE_ROOT = "arc-paper-cache"
_MANIFEST_NAME = f"{_ARCHIVE_ROOT}/manifest.json"
_CACHE_PREFIX = f"{_ARCHIVE_ROOT}/cache/"
_MANIFEST_FIELDS = {"schema_version", "selection", "files"}
_SELECTION_FIELDS = {"mode", "entry_ids"}
_FILE_FIELDS = {"path", "size", "sha256"}


class CacheArchiveError(RuntimeError):
    """Stable cache archive failure."""

    def __init__(self, code: str, message: str, *, paths: Sequence[str] = ()):
        super().__init__(message)
        self.code = code
        self.message = message
        self.paths = tuple(paths)


@dataclass(frozen=True)
class CacheExportResult:
    archive_path: str
    archive_sha256: str
    selection_mode: str
    entry_ids: tuple[str, ...]
    file_count: int
    total_bytes: int


@dataclass(frozen=True)
class CacheImportResult:
    archive_path: str
    archive_sha256: str
    selection_mode: str
    entry_ids: tuple[str, ...]
    added_count: int
    reused_count: int
    replaced_count: int
    total_bytes: int


@dataclass(frozen=True)
class _ArchiveFile:
    path: str
    size: int
    sha256: str


def export_cache_archive(
    output: str | Path,
    *,
    cache_root: str | Path | None = None,
    entry_ids: Sequence[str] = (),
    all_entries: bool = False,
) -> CacheExportResult:
    root = _absolute_path(resolve_cache_root(cache_root))
    destination = _absolute_path(Path(output).expanduser())
    requested = tuple(dict.fromkeys(str(item) for item in entry_ids if str(item)))
    if all_entries == bool(requested):
        raise CacheArchiveError(
            "cache_archive_selection_invalid",
            "cache export requires either --all or at least one exact entry ID",
        )
    if destination.exists():
        raise CacheArchiveError(
            "cache_archive_exists", f"archive output already exists: {destination}"
        )
    if _is_within(destination, root):
        raise CacheArchiveError(
            "cache_archive_output_inside_cache",
            "archive output must be outside the cache root",
        )

    if all_entries:
        relative_paths = _stable_cache_files(root)
        selected_ids: tuple[str, ...] = ()
        selection_mode = "all"
    else:
        relative_paths, selected_ids = _selected_cache_files(root, requested)
        selection_mode = "entries"

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(destination)
    try:
        last_error: CacheArchiveError | None = None
        for _ in range(2):
            temporary.unlink(missing_ok=True)
            try:
                files = tuple(_describe_file(root, item) for item in relative_paths)
                manifest = _manifest_document(selection_mode, selected_ids, files)
                _write_archive(temporary, root, files, manifest)
                last_error = None
                break
            except CacheArchiveError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise CacheArchiveError(
                "cache_archive_exists",
                f"archive output already exists: {destination}",
            ) from exc
        archive_sha256 = _file_sha256(destination)
        return CacheExportResult(
            str(destination),
            archive_sha256,
            selection_mode,
            selected_ids,
            len(files),
            sum(item.size for item in files),
        )
    finally:
        temporary.unlink(missing_ok=True)


def import_cache_archive(
    archive: str | Path,
    *,
    cache_root: str | Path | None = None,
    replace_conflicts: bool = False,
) -> CacheImportResult:
    source = _absolute_path(Path(archive).expanduser())
    if not source.is_file():
        raise CacheArchiveError(
            "cache_archive_not_found", f"cache archive is not a file: {source}"
        )
    root = _absolute_path(resolve_cache_root(cache_root))
    if root.is_symlink():
        raise CacheArchiveError(
            "cache_archive_structural_conflict", "cache root must not be a symlink"
        )
    root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".arc-paper-import.", dir=root.parent))
    try:
        manifest, files = _stage_archive(source, stage)
        conflicts: list[str] = []
        structural: list[str] = []
        added: list[_ArchiveFile] = []
        reused: list[_ArchiveFile] = []
        replaced: list[_ArchiveFile] = []
        for item in files:
            destination = root / PurePosixPath(item.path)
            issue = _structural_conflict(root, destination)
            if issue is not None:
                structural.append(item.path)
                continue
            if destination.exists():
                if not destination.is_file() or destination.is_symlink():
                    structural.append(item.path)
                elif file_matches_sha256(destination, item.sha256, item.size):
                    reused.append(item)
                elif replace_conflicts:
                    replaced.append(item)
                else:
                    conflicts.append(item.path)
            else:
                added.append(item)
        if structural:
            raise CacheArchiveError(
                "cache_archive_structural_conflict",
                "cache import encountered file, directory, or symlink conflicts",
                paths=sorted(structural),
            )
        if conflicts:
            raise CacheArchiveError(
                "cache_archive_conflict",
                "cache import found differing destination files; use --replace-conflicts",
                paths=sorted(conflicts),
            )

        for item in sorted((*added, *replaced), key=_import_order):
            _atomic_copy(stage / PurePosixPath(item.path), root / PurePosixPath(item.path))

        selection = manifest["selection"]
        return CacheImportResult(
            str(source),
            _file_sha256(source),
            selection["mode"],
            tuple(selection["entry_ids"]),
            len(added),
            len(reused),
            len(replaced),
            sum(item.size for item in files),
        )
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _selected_cache_files(
    root: Path, requested: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    administrator = CacheAdministrator(root)
    available = {item.entry_id: item for item in administrator.list().entries}
    missing = sorted(set(requested) - set(available))
    if missing:
        raise CacheArchiveError(
            "cache_archive_entry_not_found",
            "one or more selected cache entries do not exist",
            paths=missing,
        )
    selected = tuple(available[item] for item in requested)
    paths: set[str] = set()
    catalog = {
        item.entry_id: item for item in administrator.catalog.admin_entries()
    }
    for entry in selected:
        _add_index_entry(root, entry, paths)
        for component in entry.components:
            for storage_id in component.storage_entry_ids:
                if storage_id.startswith("remote:"):
                    _add_remote_entry(root, storage_id, paths)
                elif storage_id in catalog:
                    _add_catalog_entry(root, catalog[storage_id], paths)
                elif storage_id.startswith("term-inventory:"):
                    _add_term_inventory(root, storage_id, entry, paths)
    if not paths:
        raise CacheArchiveError(
            "cache_archive_entry_incomplete",
            "selected cache entries have no exportable files",
        )
    return tuple(sorted(paths)), tuple(item.entry_id for item in selected)


def _add_index_entry(root: Path, entry: CacheEntry, paths: set[str]) -> None:
    digest = hashlib.sha256(entry.entry_id.encode("utf-8")).hexdigest()
    directory = root / "cache-admin" / "v3" / "entries" / digest[:2] / digest
    path = directory / "entry.json"
    if path.is_file():
        value = _read_json_object(path)
        if (
            value.get("schema_version") != CACHE_INDEX_SCHEMA
            or value.get("entry_id") != entry.entry_id
        ):
            raise CacheArchiveError(
                "cache_archive_dependency_corrupt",
                f"cache index dependency is invalid: {entry.entry_id}",
            )
        _add_directory_files(root, directory, paths)


def _add_remote_entry(root: Path, entry_id: str, paths: set[str]) -> None:
    parts = entry_id.split(":", 3)
    if len(parts) != 4 or parts[0] != "remote" or parts[1] not in {"json", "source"}:
        raise CacheArchiveError(
            "cache_archive_dependency_invalid", f"invalid remote cache entry ID: {entry_id}"
        )
    _, kind, namespace, digest = parts
    if not _is_sha256(digest):
        raise CacheArchiveError(
            "cache_archive_dependency_invalid", f"invalid remote cache digest: {entry_id}"
        )
    directory = root / "remote-request-cache" / "v2" / kind / namespace / digest[:2] / digest
    manifest_path = directory / "manifest.json"
    admin_path = directory / "admin.json"
    if not manifest_path.is_file() or not admin_path.is_file():
        raise CacheArchiveError(
            "cache_archive_dependency_missing", f"remote cache dependency is missing: {entry_id}"
        )
    manifest = _read_json_object(manifest_path)
    if manifest.get("request_digest") != digest or manifest.get("namespace") != namespace:
        raise CacheArchiveError(
            "cache_archive_dependency_corrupt", f"remote cache dependency is invalid: {entry_id}"
        )
    if kind == "json":
        payload_name = manifest.get("payload_file")
        payload_digest = manifest.get("artifact_digest")
        size = manifest.get("size")
        if payload_name != f"payloads/{payload_digest}.json":
            raise CacheArchiveError(
                "cache_archive_dependency_corrupt", f"remote JSON dependency is invalid: {entry_id}"
            )
        if (
            not _is_sha256(payload_digest)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not file_matches_sha256(directory / payload_name, payload_digest, size)
        ):
            raise CacheArchiveError(
                "cache_archive_dependency_corrupt", f"remote JSON payload is corrupt: {entry_id}"
            )
        if namespace in {
            ARXIV_HTML_DEPENDENCY_NAMESPACE,
            AR5IV_HTML_DEPENDENCY_NAMESPACE,
        }:
            try:
                payload = json.loads((directory / payload_name).read_text(encoding="utf-8"))
                references = bundle_resource_identities(payload)
            except (
                OSError,
                UnicodeError,
                json.JSONDecodeError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                raise CacheArchiveError(
                    "cache_archive_dependency_corrupt",
                    f"HTML dependency bundle is invalid: {entry_id}",
                ) from exc
            for reference in references:
                _add_reference_resource(root, reference, paths)
        if namespace in {
            ARXIV_HTML_ACQUISITION_NAMESPACE,
            AR5IV_HTML_ACQUISITION_NAMESPACE,
        }:
            try:
                from .html_acquisition import (
                    html_acquisition_sidecar_from_document,
                )

                payload = json.loads(
                    (directory / payload_name).read_text(encoding="utf-8")
                )
                request_key = _read_json_object(admin_path).get("request_key")
                if not isinstance(request_key, str):
                    raise ValueError("HTML acquisition sidecar request key is invalid")
                bundle = html_acquisition_sidecar_from_document(
                    payload,
                    request_key=request_key,
                )
                references = tuple(
                    CachedResourceRef(
                        resource_sha256=item.artifact_digest,
                        resource_size=item.size,
                        media_type=item.media_type,
                    )
                    for item in bundle.dependencies
                    if item.availability == "available"
                )
            except (
                OSError,
                UnicodeError,
                json.JSONDecodeError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                raise CacheArchiveError(
                    "cache_archive_dependency_corrupt",
                    f"HTML acquisition sidecar is invalid: {entry_id}",
                ) from exc
            for reference in references:
                _add_reference_resource(root, reference, paths)
    else:
        _add_source_object(root, manifest, paths)
    _add_directory_files(root, directory, paths)


def _add_catalog_entry(
    root: Path, admin: FullTextCatalogAdminEntry, paths: set[str]
) -> None:
    entry = admin.entry
    identities: Iterable[str | Mapping[str, Any]] = (
        entry.document_ids
        if entry.kind == "identified"
        else (entry.local_source_identity or {},)
    )
    for identity in identities:
        key = hashlib.sha256(
            _canonical_json_bytes(
                {
                    "kind": entry.kind,
                    "identity": dict(identity)
                    if isinstance(identity, Mapping)
                    else identity,
                }
            )
        ).hexdigest()
        directory = (
            root
            / "document-full-text-catalog"
            / "v2"
            / "entries"
            / key[:2]
            / key
        )
        if not (directory / "locator.json").is_file() or not (directory / "admin.json").is_file():
            raise CacheArchiveError(
                "cache_archive_dependency_missing",
                f"full-text catalog dependency is missing: {admin.entry_id}",
            )
        _add_directory_files(root, directory, paths)
    for representation in entry.representations:
        _add_source_object(root, representation.source_identity, paths)
        key = representation.parsed_cache_key
        directory = root / "parsed-document-cache" / "v1" / "sha256" / key[:2] / key
        manifest = directory / "manifest.json"
        document = directory / "document.json"
        value = _read_json_object(manifest)
        payload_digest = value.get("payload_digest")
        payload_size = value.get("payload_size")
        if (
            value.get("source_identity") != dict(representation.source_identity)
            or value.get("parser_contract") != representation.parser_contract
            or value.get("document_digest") != representation.document_digest
            or not _is_sha256(payload_digest)
            or not isinstance(payload_size, int)
            or isinstance(payload_size, bool)
            or payload_size < 0
            or not file_matches_sha256(document, payload_digest, payload_size)
        ):
            raise CacheArchiveError(
                "cache_archive_dependency_corrupt",
                f"parsed cache dependency is corrupt: {admin.entry_id}",
            )
        _add_directory_files(root, directory, paths)


def _add_term_inventory(
    root: Path, storage_entry_id: str, entry: CacheEntry, paths: set[str]
) -> None:
    key = storage_entry_id.removeprefix("term-inventory:")
    if not _is_sha256(key):
        raise CacheArchiveError(
            "cache_archive_dependency_invalid",
            f"invalid term inventory ID: {storage_entry_id}",
        )
    directory = root / "term-inventory" / "v1" / "lineages" / key[:2] / key
    if not (directory / "current.json").is_file():
        raise CacheArchiveError(
            "cache_archive_dependency_missing",
            f"term inventory is missing: {storage_entry_id}",
        )
    _add_directory_files(root, directory, paths)
    if entry.local_source_identity is not None:
        _add_source_object(root, entry.local_source_identity, paths)


def _add_source_object(root: Path, identity: Mapping[str, Any], paths: set[str]) -> None:
    source_format = identity.get("source_format")
    digest = identity.get("artifact_digest")
    size = identity.get("size")
    media_type = identity.get("media_type")
    try:
        resolved_format = SourceFormat(str(source_format))
    except ValueError as exc:
        raise CacheArchiveError(
            "cache_archive_dependency_invalid", "source dependency has an invalid format"
        ) from exc
    if not _is_sha256(digest):
        raise CacheArchiveError(
            "cache_archive_dependency_invalid", "source dependency has an invalid digest"
        )
    repository = SourceRepository(root)
    try:
        source = repository.get(resolved_format, str(digest))
        repository.read_bytes(source)
    except Exception as exc:
        raise CacheArchiveError(
            "cache_archive_dependency_corrupt",
            f"source dependency is missing or corrupt: {resolved_format.value}/{digest}",
        ) from exc
    if source.size != size or source.media_type != media_type:
        raise CacheArchiveError(
            "cache_archive_dependency_corrupt", "source dependency metadata conflicts"
        )
    directory = (
        root / "source-repository" / "v1" / resolved_format.value / "sha256"
        / str(digest)[:2] / str(digest)
    )
    _add_directory_files(root, directory, paths)


def _add_reference_resource(
    root: Path,
    reference: Any,
    paths: set[str],
) -> None:
    cache = ReferenceMaterialCache(root)
    try:
        payload = cache.read_resource(reference)
    except Exception as exc:
        raise CacheArchiveError(
            "cache_archive_dependency_corrupt",
            f"HTML dependency resource is missing or corrupt: {reference.resource_sha256}",
        ) from exc
    if (
        len(payload) != reference.resource_size
        or hashlib.sha256(payload).hexdigest() != reference.resource_sha256
    ):
        raise CacheArchiveError(
            "cache_archive_dependency_corrupt",
            f"HTML dependency resource metadata conflicts: {reference.resource_sha256}",
        )
    directory = (
        root
        / "reference-material-cache"
        / "v1"
        / "resources"
        / "sha256"
        / reference.resource_sha256[:2]
        / reference.resource_sha256
    )
    _add_directory_files(root, directory, paths)


def _stable_cache_files(root: Path) -> tuple[str, ...]:
    if not root.is_dir():
        return ()
    values: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        if _excluded(relative):
            continue
        values.append(relative.as_posix())
    return tuple(sorted(values))


def _add_directory_files(root: Path, directory: Path, paths: set[str]) -> None:
    if not directory.is_dir() or directory.is_symlink():
        raise CacheArchiveError(
            "cache_archive_dependency_missing",
            f"cache dependency directory is missing: {directory.relative_to(root)}",
        )
    found = False
    for path in directory.rglob("*"):
        if path.is_file() and not path.is_symlink():
            relative = path.relative_to(root)
            if not _excluded(relative):
                paths.add(relative.as_posix())
                found = True
    if not found:
        raise CacheArchiveError(
            "cache_archive_dependency_missing",
            f"cache dependency has no stable files: {directory.relative_to(root)}",
        )


def _excluded(relative: Path) -> bool:
    parts = relative.parts
    if "locks" in parts or "host-gates" in parts:
        return True
    return relative.name.endswith(".lock") or any(part.startswith(".") for part in parts)


def _describe_file(root: Path, relative: str) -> _ArchiveFile:
    path = root / PurePosixPath(relative)
    if not path.is_file() or path.is_symlink():
        raise CacheArchiveError(
            "cache_archive_changed", f"cache file changed during export: {relative}"
        )
    size = path.stat().st_size
    return _ArchiveFile(relative, size, _file_sha256(path))


def _write_archive(
    output: Path,
    root: Path,
    files: Sequence[_ArchiveFile],
    manifest: Mapping[str, Any],
) -> None:
    with tarfile.open(output, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        payload = _canonical_json_bytes(manifest)
        info = _tar_info(_MANIFEST_NAME, len(payload))
        archive.addfile(info, _BytesReader(payload))
        for item in files:
            path = root / PurePosixPath(item.path)
            before = path.stat()
            if before.st_size != item.size or not stat.S_ISREG(before.st_mode):
                raise CacheArchiveError(
                    "cache_archive_changed", f"cache file changed during export: {item.path}"
                )
            with path.open("rb") as handle:
                reader = _DigestingReader(handle)
                archive.addfile(_tar_info(_CACHE_PREFIX + item.path, item.size), reader)
            after = path.stat()
            if (
                reader.digest != item.sha256
                or before.st_mtime_ns != after.st_mtime_ns
                or before.st_size != after.st_size
            ):
                raise CacheArchiveError(
                    "cache_archive_changed", f"cache file changed during export: {item.path}"
                )


def _stage_archive(
    source: Path, stage: Path
) -> tuple[dict[str, Any], tuple[_ArchiveFile, ...]]:
    try:
        with tarfile.open(source, "r:gz") as archive:
            return _stage_open_archive(archive, stage)
    except (OSError, tarfile.TarError) as exc:
        raise CacheArchiveError(
            "cache_archive_invalid", "cache archive is not a readable tar.gz"
        ) from exc


def _stage_open_archive(
    archive: tarfile.TarFile, stage: Path
) -> tuple[dict[str, Any], tuple[_ArchiveFile, ...]]:
    members = archive.getmembers()
    names = [item.name for item in members]
    if len(names) != len(set(names)) or any(not item.isfile() for item in members):
        raise CacheArchiveError(
            "cache_archive_invalid", "cache archive must contain unique regular files"
        )
    try:
        manifest_member = archive.getmember(_MANIFEST_NAME)
    except KeyError as exc:
        raise CacheArchiveError(
            "cache_archive_invalid", "cache archive manifest is missing"
        ) from exc
    if manifest_member.size > 16 * 1024 * 1024:
        raise CacheArchiveError("cache_archive_invalid", "cache archive manifest is too large")
    manifest_stream = archive.extractfile(manifest_member)
    if manifest_stream is None:
        raise CacheArchiveError("cache_archive_invalid", "cache archive manifest is unreadable")
    try:
        manifest = json.loads(manifest_stream.read().decode("utf-8"))
        files = _decode_manifest(manifest)
    except (UnicodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        raise CacheArchiveError(
            "cache_archive_invalid", f"cache archive manifest is invalid: {exc}"
        ) from exc
    expected_names = {_MANIFEST_NAME, *(_CACHE_PREFIX + item.path for item in files)}
    if set(names) != expected_names:
        raise CacheArchiveError(
            "cache_archive_invalid", "cache archive members do not match its manifest"
        )
    for item in files:
        member = archive.getmember(_CACHE_PREFIX + item.path)
        if member.size != item.size:
            raise CacheArchiveError(
                "cache_archive_invalid", f"cache archive size mismatch: {item.path}"
            )
        stream = archive.extractfile(member)
        if stream is None:
            raise CacheArchiveError(
                "cache_archive_invalid", f"cache archive member is unreadable: {item.path}"
            )
        destination = stage / PurePosixPath(item.path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        written = 0
        with destination.open("wb") as handle:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                handle.write(chunk)
                written += len(chunk)
        if written != item.size or digest.hexdigest() != item.sha256:
            raise CacheArchiveError(
                "cache_archive_invalid", f"cache archive digest mismatch: {item.path}"
            )
    return manifest, files


def _decode_manifest(value: object) -> tuple[_ArchiveFile, ...]:
    if not isinstance(value, dict) or set(value) != _MANIFEST_FIELDS:
        raise ValueError("manifest fields are invalid")
    if value.get("schema_version") != CACHE_ARCHIVE_SCHEMA:
        raise ValueError("manifest schema is unsupported")
    selection = value.get("selection")
    if not isinstance(selection, dict) or set(selection) != _SELECTION_FIELDS:
        raise ValueError("selection is invalid")
    mode = selection.get("mode")
    entry_ids = selection.get("entry_ids")
    if mode not in {"all", "entries"} or not isinstance(entry_ids, list) or not all(
        isinstance(item, str) and item for item in entry_ids
    ):
        raise ValueError("selection values are invalid")
    if (mode == "all" and entry_ids) or (mode == "entries" and not entry_ids):
        raise ValueError("selection mode conflicts with entry IDs")
    raw_files = value.get("files")
    if not isinstance(raw_files, list):
        raise ValueError("files must be a list")
    files: list[_ArchiveFile] = []
    seen: set[str] = set()
    for raw in raw_files:
        if not isinstance(raw, dict) or set(raw) != _FILE_FIELDS:
            raise ValueError("file record is invalid")
        path = _safe_relative_path(raw.get("path"))
        size = raw.get("size")
        digest = raw.get("sha256")
        if (
            path in seen
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not _is_sha256(digest)
        ):
            raise ValueError("file record values are invalid")
        seen.add(path)
        files.append(_ArchiveFile(path, size, digest))
    if [item.path for item in files] != sorted(item.path for item in files):
        raise ValueError("file records must be sorted")
    return tuple(files)


def _manifest_document(
    mode: str, entry_ids: Sequence[str], files: Sequence[_ArchiveFile]
) -> dict[str, Any]:
    return {
        "schema_version": CACHE_ARCHIVE_SCHEMA,
        "selection": {"mode": mode, "entry_ids": list(entry_ids)},
        "files": [
            {"path": item.path, "size": item.size, "sha256": item.sha256}
            for item in files
        ],
    }


def _safe_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("file path is invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("file path is unsafe")
    return value


def _structural_conflict(root: Path, destination: Path) -> str | None:
    current = root
    if current.is_symlink() or (current.exists() and not current.is_dir()):
        return "root"
    for part in destination.relative_to(root).parts[:-1]:
        current = current / part
        if current.is_symlink() or (current.exists() and not current.is_dir()):
            return str(current)
    if destination.is_symlink():
        return str(destination)
    return None


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as output, source.open("rb") as input_file:
            shutil.copyfileobj(input_file, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _import_order(item: _ArchiveFile) -> tuple[int, str]:
    metadata = item.path.endswith(".json")
    return (1 if metadata else 0, item.path)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CacheArchiveError(
            "cache_archive_dependency_corrupt", f"cache manifest is unreadable: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise CacheArchiveError(
            "cache_archive_dependency_corrupt", f"cache manifest is invalid: {path}"
        )
    return value


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o600
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def _temporary_path(destination: Path) -> Path:
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    path = Path(name)
    path.unlink()
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class _BytesReader:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.offset = 0

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self.payload) - self.offset
        value = self.payload[self.offset : self.offset + size]
        self.offset += len(value)
        return value


class _DigestingReader:
    def __init__(self, handle: BinaryIO):
        self.handle = handle
        self.hasher = hashlib.sha256()

    @property
    def digest(self) -> str:
        return self.hasher.hexdigest()

    def read(self, size: int = -1) -> bytes:
        value = self.handle.read(size)
        self.hasher.update(value)
        return value


__all__ = [
    "CACHE_ARCHIVE_SCHEMA",
    "CacheArchiveError",
    "CacheExportResult",
    "CacheImportResult",
    "export_cache_archive",
    "import_cache_archive",
]
