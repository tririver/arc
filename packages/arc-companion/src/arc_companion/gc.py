from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from .artifact_ids import (
    ARTIFACT_ID_RECEIPT_NAME,
    ArtifactIdError,
    resolve_artifact_dir,
)
from .io import canonical_json
from .pdf import (
    PDF_VALIDATION_RECEIPT_VERSION,
    managed_run_root_pdf_path,
    match_validated_pdf_revision,
    normalize_run_root_pdf_state,
)
from .results import err, ok
from .run_lock import BuildInProgressError, ProjectBuildLock, inspect_lock
from .web import (
    WEB_MANIFEST_VERSION,
    WebReaderError,
    inspect_reader_publish,
    validate_reader_project,
)


GC_REPORT_VERSION = "arc.companion.gc-report.v1"
GC_TRANSACTION_VERSION = "arc.companion.gc-transaction.v1"
GC_RECEIPT_VERSION = "arc.companion.gc-receipt.v1"
GC_RECOGNIZER_VERSION = "arc.companion.gc-recognizer.v1"

MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_CANDIDATE_BYTES = 2 * 1024 * 1024 * 1024
MAX_RECOGNIZED_ENTRIES = 20_000
MAX_MANIFEST_REFERENCES = 8_192
MAX_DIRECTORY_DEPTH = 16
MAX_WARNING_CODES = 128

_SHA256 = re.compile(r"[0-9a-f]{64}")
_MANIFEST_NAME = re.compile(r"manifest-([0-9a-f]{64})\.json")
_SNAPSHOT_NAME = re.compile(r"snapshot-([0-9a-f]{64})\.json")
_DATA_NAME = re.compile(r"snapshot-([0-9a-f]{64})\.js")
_BUILTIN_NAME = re.compile(r"builtin-([0-9a-f]{64})")
_SOURCE_NAME = re.compile(
    r"([0-9a-f]{64})(\.(?:png|jpg|jpeg|gif|webp|svg))",
    re.IGNORECASE,
)
_VALIDATION_PAGE = re.compile(
    r".+\.validation-page-([1-9][0-9]*)\.png"
)
_VALIDATION_TEXT = re.compile(r".+\.validation\.txt")
_STAGING = re.compile(
    r"arc-companion-(?:building|rendering)-[A-Za-z0-9._-]+-"
    r"[0-9a-f]{12}(?:\.tex|\.pdf|-manifest\.json|-validation\.json)"
)
_BUSY_STATES = {
    "loading_source",
    "building_intent_guidance",
    "segmenting",
    "generating",
    "reviewing",
    "typesetting",
    "finalizing",
    "needs_supervision",
    "first_chapter_ready",
}
_NONTERMINAL_TRANSACTION_STATES = {
    "planned",
    "moving",
    "quarantined",
    "deleting",
}
_RETAINED_ROOTS = {
    "checkpoints": ".arc-companion/checkpoints",
    "objects": ".arc-companion/objects",
    "broker": ".arc-companion/paper-broker",
    "jobs": ".arc-companion/jobs",
    "translation_references": ".arc-companion/translation-references",
    "arbitration": ".arc-companion/review-arbitration",
    "review_objects": ".arc-companion/review-segments",
    "intent_guidance": ".arc-companion/intent-guidance",
    "attempts": ".arc-companion/pdf-validation-attempts",
    "resume_history": ".arc-companion/resume-transactions",
    "gc_audit": ".arc-companion/gc",
    "provenance": ".arc-companion/provenance",
}


class CompanionGCError(RuntimeError):
    """A stable, non-retryable refusal at the local GC ownership boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(str(message)[:512])
        self.code = code


@dataclass(frozen=True)
class GCCandidate:
    category: str
    path: str
    kind: str
    bytes: int
    sha256: str
    recognizer_version: str = GC_RECOGNIZER_VERSION

    def __post_init__(self) -> None:
        _safe_relative(self.path)
        if self.kind not in {"file", "directory"}:
            raise ValueError("GC candidate kind is invalid")
        if isinstance(self.bytes, bool) or self.bytes < 0:
            raise ValueError("GC candidate byte count is invalid")
        if not _SHA256.fullmatch(self.sha256):
            raise ValueError("GC candidate digest is invalid")
        if not self.category or not re.fullmatch(
            r"[a-z][a-z0-9_]*", self.category
        ):
            raise ValueError("GC candidate category is invalid")
        if self.recognizer_version != GC_RECOGNIZER_VERSION:
            raise ValueError("GC candidate recognizer version is invalid")

    def as_dict(self) -> dict[str, object]:
        return {
            "category": self.category,
            "path": self.path,
            "kind": self.kind,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "recognizer_version": self.recognizer_version,
        }


@dataclass(frozen=True)
class GCReport:
    project_identity_sha256: str
    root_snapshot_sha256: str
    candidates: tuple[GCCandidate, ...]
    candidate_set_sha256: str
    category_totals: tuple[tuple[str, int, int], ...]
    retained_class_totals: tuple[tuple[str, int, int], ...]
    total_reclaimable_bytes: int
    warnings: tuple[str, ...] = ()
    status: str = "ready"
    schema_version: str = GC_REPORT_VERSION
    recognizer_version: str = GC_RECOGNIZER_VERSION
    mode: str = "dry_run"

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "recognizer_version": self.recognizer_version,
            "project_identity_sha256": self.project_identity_sha256,
            "root_snapshot_sha256": self.root_snapshot_sha256,
            "mode": self.mode,
            "status": self.status,
            "candidates": [
                candidate.as_dict() for candidate in self.candidates
            ],
            "candidate_set_sha256": self.candidate_set_sha256,
            "category_totals": {
                category: {"count": count, "bytes": size}
                for category, count, size in self.category_totals
            },
            "retained_class_totals": {
                category: {"count": count, "bytes": size}
                for category, count, size in self.retained_class_totals
            },
            "total_reclaimable_bytes": self.total_reclaimable_bytes,
            "warnings": list(self.warnings),
            "refusals": [],
        }


@dataclass(frozen=True)
class _Discovery:
    report: GCReport
    root: Path
    state: Mapping[str, Any]
    retained_paths: frozenset[str]
    root_snapshot_payload: Mapping[str, Any]


def discover_gc(
    project_dir: Path,
    *,
    extra_roots: Iterable[str | Path] = (),
) -> GCReport:
    """Plan latest-only cleanup without writing or acquiring a lock."""

    return _discover_gc(
        project_dir,
        extra_roots=extra_roots,
        allow_active_build_lock=False,
        allow_gc_transaction=False,
    ).report


def gc_project(
    project_dir: Path,
    *,
    apply: bool = False,
    candidate_digest: str | None = None,
    extra_roots: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Result-envelope facade used by the CLI."""

    try:
        if apply:
            receipt = apply_gc(
                project_dir,
                candidate_digest=candidate_digest,
                extra_roots=extra_roots,
            )
            return ok(receipt)
        if candidate_digest is not None:
            raise CompanionGCError(
                "gc_candidate_digest_invalid",
                "--candidate-digest is valid only with --apply",
            )
        return ok(discover_gc(project_dir, extra_roots=extra_roots).as_dict())
    except CompanionGCError as exc:
        return err(exc.code, str(exc))


def _discover_gc(
    project_dir: Path,
    *,
    extra_roots: Iterable[str | Path],
    allow_active_build_lock: bool,
    allow_gc_transaction: bool,
    scan_candidates: bool = True,
) -> _Discovery:
    root = _project_root(project_dir)
    root_before = root.stat()
    state_path = root / "state.json"
    state_bytes = _read_regular_bytes(
        root, state_path, max_bytes=MAX_JSON_BYTES,
        code="gc_state_invalid",
    )
    state = _json_object(state_bytes, "gc_state_invalid")
    status = str(state.get("status") or "")
    if status in _BUSY_STATES:
        raise CompanionGCError(
            "gc_state_invalid" if status != "needs_supervision"
            else "gc_build_active",
            "companion state is not at a safe final boundary",
        )
    if status not in {"complete", "failed"}:
        raise CompanionGCError(
            "gc_state_invalid",
            "companion state has no safe latest-publication boundary",
        )

    build_lock = root / ".arc-companion-build.lock"
    if build_lock.exists() or build_lock.is_symlink():
        _require_regular_or_missing(
            root, build_lock, code="gc_project_unsafe",
        )
        lock_state = inspect_lock(build_lock)
        if (
            not allow_active_build_lock
            and isinstance(lock_state, Mapping)
            and lock_state.get("active") is True
        ):
            raise CompanionGCError(
                "gc_build_active",
                "the companion project build lock is active",
            )
    render_lock = root / ".arc-companion" / "render.lock"
    if render_lock.exists() or render_lock.is_symlink():
        _require_regular_or_missing(
            root, render_lock, code="gc_project_unsafe",
        )
        render_owner = inspect_lock(render_lock)
        if (
            not allow_active_build_lock
            and
            isinstance(render_owner, Mapping)
            and render_owner.get("active") is True
        ):
            raise CompanionGCError(
                "gc_transaction_active",
                "an explicit render transaction is active",
            )

    transaction_hashes = _active_transaction_hashes(
        root, allow_gc_transaction=allow_gc_transaction,
    )
    explicit_roots = _extra_root_records(root, extra_roots)
    retained_paths = set(explicit_roots)
    warnings: set[str] = set()

    reader_roots, reader_candidates, reader_snapshot = _reader_discovery(
        root, state, retained_paths, warnings,
        scan_candidates=scan_candidates,
    )
    retained_paths.update(reader_roots)
    render_roots, render_candidates, render_snapshot = _render_discovery(
        root, state, retained_paths, warnings,
        scan_candidates=scan_candidates,
    )
    retained_paths.update(render_roots)

    checkpoint_snapshot = _checkpoint_snapshot(root, state)
    content_snapshot = _content_snapshot(root, state)
    run_pdf_snapshot = _managed_pdf_snapshot(root, state)
    t19_identities = _directory_identity_snapshot(root)
    extra_snapshot = {
        relative: _identity_record(root, root / relative)
        for relative in sorted(explicit_roots)
        if (root / relative).exists()
    }
    root_snapshot_payload = {
        "state_sha256": hashlib.sha256(state_bytes).hexdigest(),
        "state_bytes": len(state_bytes),
        "reader": reader_snapshot,
        "render": render_snapshot,
        "run_root_pdf": run_pdf_snapshot,
        "checkpoint": checkpoint_snapshot,
        "content": content_snapshot,
        "active_transactions": transaction_hashes,
        "directory_identities": t19_identities,
        "extra_roots": extra_snapshot,
    }
    root_snapshot_sha256 = _sha_json(root_snapshot_payload)
    candidates = tuple(sorted(
        (*reader_candidates, *render_candidates),
        key=lambda item: (
            item.category, item.path, item.kind, item.sha256,
        ),
    ))
    _require_antichain(candidates)
    candidate_records = [item.as_dict() for item in candidates]
    candidate_set_sha256 = _sha_json(candidate_records)
    categories = _totals(candidates)
    retained_totals = _retained_totals(root)
    warning_tuple = tuple(sorted(warnings))[:MAX_WARNING_CODES]
    state_after = _read_regular_bytes(
        root, state_path, max_bytes=MAX_JSON_BYTES,
        code="gc_state_invalid",
    )
    root_after = root.stat()
    if (
        state_after != state_bytes
        or (root_before.st_dev, root_before.st_ino)
        != (root_after.st_dev, root_after.st_ino)
    ):
        raise CompanionGCError(
            "gc_candidate_changed",
            "project roots changed during GC discovery",
        )
    report = GCReport(
        project_identity_sha256=hashlib.sha256(
            str(root).encode("utf-8")
        ).hexdigest(),
        root_snapshot_sha256=root_snapshot_sha256,
        candidates=candidates,
        candidate_set_sha256=candidate_set_sha256,
        category_totals=categories,
        retained_class_totals=retained_totals,
        total_reclaimable_bytes=sum(item.bytes for item in candidates),
        warnings=warning_tuple,
        status="ready" if candidates else "no_op",
    )
    return _Discovery(
        report=report,
        root=root,
        state=state,
        retained_paths=frozenset(retained_paths),
        root_snapshot_payload=root_snapshot_payload,
    )


def _reader_discovery(
    root: Path,
    state: Mapping[str, Any],
    extra_roots: set[str],
    warnings: set[str],
    *,
    scan_candidates: bool,
) -> tuple[set[str], tuple[GCCandidate, ...], Mapping[str, Any]]:
    reader = root / "reader"
    index = reader / "index.html"
    published = state.get("published")
    published = published if isinstance(published, Mapping) else {}
    web_state = published.get("web")
    web_state = web_state if isinstance(web_state, Mapping) else {}
    effective_web = {**state, **dict(web_state)}
    retained: set[str] = set()
    current_manifest: Path | None = None
    current_graph: set[str] = set()
    inspection: Mapping[str, Any] | None = None
    if index.exists() or index.is_symlink():
        try:
            inspection = inspect_reader_publish(root)
        except (OSError, RuntimeError, ValueError, WebReaderError) as exc:
            raise CompanionGCError(
                "gc_reader_invalid",
                "the committed Reader graph is invalid",
            ) from exc
        if inspection is None:
            raise CompanionGCError(
                "gc_reader_invalid",
                "the committed Reader index has no validated graph",
            )
        selected_manifest = effective_web.get("web_manifest_path")
        selected_hash = str(
            effective_web.get("web_manifest_sha256") or ""
        )
        if selected_manifest and selected_hash:
            current_manifest = _state_file(
                root, selected_manifest, selected_hash,
                code="gc_reader_invalid",
            )
            try:
                validate_reader_project(root, state=effective_web)
            except (OSError, RuntimeError, ValueError, WebReaderError) as exc:
                raise CompanionGCError(
                    "gc_reader_invalid",
                    "published Reader state and committed index disagree",
                ) from exc
            for key in (
                "output_html_sha256",
                "reader_snapshot_sha256",
                "web_render_version",
                "source_credit_sha256",
                "source_credit_observation_sha256",
            ):
                if (
                    effective_web.get(key) is not None
                    and inspection.get(key) != effective_web.get(key)
                ):
                    raise CompanionGCError(
                        "gc_reader_invalid",
                        "published Reader identity and committed index disagree",
                    )
        else:
            current_manifest = _legacy_current_manifest(
                root, effective_web, inspection,
            )
        current_graph = _manifest_graph(root, current_manifest)
        current_graph.add("reader/index.html")
        current_graph.add(_relative(root, current_manifest))
        retained.update(current_graph)
    elif any(
        effective_web.get(key)
        for key in (
            "output_html",
            "web_manifest_path",
            "reader_snapshot_path",
        )
    ):
        raise CompanionGCError(
            "gc_reader_invalid",
            "published Reader state is missing its committed index",
        )

    if not scan_candidates:
        return (
            retained,
            (),
            _reader_snapshot(root, inspection, current_manifest, current_graph),
        )
    candidates: list[GCCandidate] = []
    data_dir = reader / "data"
    if data_dir.exists() or data_dir.is_symlink():
        _require_directory(root, data_dir, code="gc_reader_invalid")
        for entry in _bounded_children(data_dir):
            relative = _relative(root, entry)
            match = _MANIFEST_NAME.fullmatch(entry.name)
            snapshot_match = _SNAPSHOT_NAME.fullmatch(entry.name)
            data_match = _DATA_NAME.fullmatch(entry.name)
            if match:
                _require_filename_hash(root, entry, match.group(1))
                _manifest_graph(root, entry)
                if relative not in current_graph and not _rooted(
                    relative, extra_roots,
                ):
                    candidates.append(_file_candidate(
                        root, entry, "reader_manifest_history",
                    ))
            elif snapshot_match or data_match:
                digest = (snapshot_match or data_match).group(1)
                _require_filename_hash(root, entry, digest)
                if snapshot_match:
                    snapshot = _json_object(
                        _read_regular_bytes(
                            root, entry, max_bytes=MAX_JSON_BYTES,
                            code="gc_reader_invalid",
                        ),
                        "gc_reader_invalid",
                    )
                    if not isinstance(snapshot.get("schema_version"), str):
                        raise CompanionGCError(
                            "gc_reader_invalid",
                            "Reader snapshot history has an invalid schema",
                        )
                if relative not in current_graph and not _rooted(
                    relative, extra_roots,
                ):
                    candidates.append(_file_candidate(
                        root,
                        entry,
                        (
                            "reader_snapshot_history"
                            if snapshot_match
                            else "reader_data_history"
                        ),
                    ))
            elif entry.name == ".DS_Store":
                warnings.add("gc_unrecognized_owned_path")
            else:
                warnings.add("gc_unrecognized_owned_path")

    assets = reader / "assets"
    if assets.exists() or assets.is_symlink():
        _require_directory(root, assets, code="gc_reader_invalid")
        for entry in _bounded_children(assets):
            relative = _relative(root, entry)
            builtin = _BUILTIN_NAME.fullmatch(entry.name)
            if builtin:
                _require_directory(root, entry, code="gc_reader_invalid")
                if _builtin_identity(entry) != builtin.group(1):
                    raise CompanionGCError(
                        "gc_reader_invalid",
                        "Reader builtin asset directory identity is invalid",
                    )
                if (
                    not any(
                        path == relative or path.startswith(relative + "/")
                        for path in current_graph
                    )
                    and not _rooted(relative, extra_roots)
                ):
                    candidates.append(_directory_candidate(
                        root, entry, "reader_builtin_history",
                    ))
            elif entry.name == "source":
                _require_directory(root, entry, code="gc_reader_invalid")
                for source in _bounded_children(entry):
                    source_relative = _relative(root, source)
                    match = _SOURCE_NAME.fullmatch(source.name)
                    if not match:
                        warnings.add("gc_unrecognized_owned_path")
                        continue
                    _require_filename_hash(root, source, match.group(1))
                    if (
                        source_relative not in current_graph
                        and not _rooted(source_relative, extra_roots)
                    ):
                        candidates.append(_file_candidate(
                            root, source, "reader_source_history",
                        ))
            else:
                warnings.add("gc_unrecognized_owned_path")
    return (
        retained,
        tuple(candidates),
        _reader_snapshot(root, inspection, current_manifest, current_graph),
    )


def _reader_snapshot(
    root: Path,
    inspection: Mapping[str, Any] | None,
    current_manifest: Path | None,
    current_graph: set[str],
) -> Mapping[str, Any]:
    return {
        "committed": {
            key: inspection.get(key)
            for key in (
                "output_html_sha256",
                "reader_snapshot_sha256",
                "web_render_version",
                "reader_semantic_sha256",
                "source_credit_sha256",
                "source_credit_observation_sha256",
            )
        } if inspection is not None else {},
        "manifest_path": (
            _relative(root, current_manifest)
            if current_manifest is not None else None
        ),
        "manifest_sha256": (
            _hash_file(current_manifest)[0]
            if current_manifest is not None else None
        ),
        "graph_sha256": _sha_json(sorted(current_graph)),
    }


def _legacy_current_manifest(
    root: Path,
    state: Mapping[str, Any],
    inspection: Mapping[str, Any],
) -> Path:
    data_dir = root / "reader" / "data"
    _require_directory(root, data_dir, code="gc_reader_invalid")
    matches: list[Path] = []
    for path in _bounded_children(data_dir):
        match = _MANIFEST_NAME.fullmatch(path.name)
        if match is None:
            continue
        _require_filename_hash(root, path, match.group(1))
        candidate_state = {
            **dict(inspection),
            **dict(state),
            "web_manifest_path": str(path),
            "web_manifest_sha256": match.group(1),
        }
        try:
            validate_reader_project(root, state=candidate_state)
        except (OSError, RuntimeError, ValueError, WebReaderError):
            continue
        matches.append(path)
    if len(matches) != 1:
        raise CompanionGCError(
            "gc_reader_invalid",
            "legacy Reader index has no unique authoritative manifest",
        )
    return matches[0]


def _render_discovery(
    root: Path,
    state: Mapping[str, Any],
    extra_roots: set[str],
    warnings: set[str],
    *,
    scan_candidates: bool,
) -> tuple[set[str], tuple[GCCandidate, ...], Mapping[str, Any]]:
    render_root = root / ".arc-companion" / "renders" / "pdf"
    retained: set[str] = set()
    candidates: list[GCCandidate] = []
    published = state.get("published")
    published = published if isinstance(published, Mapping) else {}
    pdf_state = published.get("pdf")
    pdf_state = pdf_state if isinstance(pdf_state, Mapping) else {}
    effective = normalize_run_root_pdf_state({**state, **dict(pdf_state)})
    current_dir: Path | None = None
    current_new = False
    if effective.get("output_pdf"):
        current_pdf = _state_file(
            root,
            effective.get("output_pdf"),
            str(effective.get("output_pdf_sha256") or ""),
            code="gc_render_invalid",
        )
        current_dir = current_pdf.parent
        if current_dir.parent != render_root.resolve(strict=False):
            raise CompanionGCError(
                "gc_render_invalid",
                "current PDF revision is outside the render root",
            )
        for path_key, hash_key in (
            ("output_tex", "output_tex_sha256"),
            ("source_manifest_path", "source_manifest_sha256"),
            ("validation_path", "validation_sha256"),
        ):
            path = _state_file(
                root, effective.get(path_key),
                str(effective.get(hash_key) or ""),
                code="gc_render_invalid",
            )
            if path.parent != current_dir:
                raise CompanionGCError(
                    "gc_render_invalid",
                    "current PDF revision files do not share one directory",
                )
        identity = effective.get("render_identity")
        if identity is not None:
            try:
                allocation = resolve_artifact_dir(
                    render_root,
                    current_dir,
                    expected_identity=str(identity),
                    kind="pdf-render",
                    stem=str(effective.get("render_stem") or ""),
                    allow_legacy=False,
                )
            except (ArtifactIdError, OSError, ValueError) as exc:
                raise CompanionGCError(
                    "gc_render_invalid",
                    "current render directory identity is invalid",
                ) from exc
            receipt_path = Path(str(
                effective.get("render_identity_receipt_path") or ""
            ))
            if (
                allocation.receipt_path is None
                or receipt_path.absolute()
                != allocation.receipt_path.absolute()
                or effective.get("render_identity_receipt_sha256")
                != allocation.receipt_sha256
            ):
                raise CompanionGCError(
                    "gc_render_invalid",
                    "current render receipt binding is invalid",
                )
            current_new = True
            content_sha = str(
                published.get("content_sha256")
                or effective.get("content_sha256")
                or ""
            )
            if _SHA256.fullmatch(content_sha):
                match = match_validated_pdf_revision(
                    root, state, content_sha256=content_sha,
                )
                if not match.reusable:
                    raise CompanionGCError(
                        "gc_render_invalid",
                        "current render revision is not exactly reusable",
                    )
        else:
            _recognize_legacy_render(root, current_dir, state=effective)
        retained.add(_relative(root, current_dir))

    if not scan_candidates:
        if current_dir is not None and current_new and _legacy_sidecars(
            current_dir
        ):
            raise CompanionGCError(
                "gc_render_invalid",
                "current render contains durable validation temporaries",
            )
        return retained, (), _render_snapshot(current_dir, root, effective)

    if render_root.exists() or render_root.is_symlink():
        _require_directory(root, render_root, code="gc_render_invalid")
        for child in _bounded_children(render_root):
            if child.name in {".artifact-ids.lock", "aliases"}:
                continue
            if not child.is_dir() or child.is_symlink():
                warnings.add("gc_unrecognized_owned_path")
                continue
            relative = _relative(root, child)
            if current_dir is not None and child == current_dir:
                sidecars = _legacy_sidecars(child)
                if current_new and sidecars:
                    raise CompanionGCError(
                        "gc_render_invalid",
                        "current render contains durable validation temporaries",
                    )
                if not current_new:
                    for sidecar in sidecars:
                        candidates.append(_file_candidate(
                            root, sidecar, "validation_temporary",
                        ))
                continue
            if _rooted(relative, extra_roots):
                retained.add(relative)
                continue
            receipt = child / ARTIFACT_ID_RECEIPT_NAME
            if receipt.is_file():
                try:
                    resolve_artifact_dir(
                        render_root, child,
                        kind="pdf-render", allow_legacy=False,
                    )
                except (ArtifactIdError, OSError, ValueError) as exc:
                    raise CompanionGCError(
                        "gc_render_invalid",
                        "historical render identity is invalid",
                    ) from exc
            else:
                try:
                    _recognize_legacy_render(root, child, state=None)
                except CompanionGCError:
                    warnings.add("gc_unrecognized_owned_path")
                    continue
            candidates.append(_directory_candidate(
                root, child, "render_history",
            ))
    return (
        retained,
        tuple(candidates),
        _render_snapshot(current_dir, root, effective),
    )


def _render_snapshot(
    current_dir: Path | None,
    root: Path,
    effective: Mapping[str, Any],
) -> Mapping[str, Any]:
    return {
        "current_directory": (
            _relative(root, current_dir)
            if current_dir is not None else None
        ),
        "current_identity": effective.get("render_identity"),
        "current_pdf_sha256": effective.get("output_pdf_sha256"),
    }


def _safe_relative(value: str | Path) -> str:
    raw = str(value)
    if (
        not raw
        or raw in {".", ".."}
        or "\x00" in raw
        or "\\" in raw
    ):
        raise CompanionGCError(
            "gc_project_unsafe", "GC path is not a safe project-relative path",
        )
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise CompanionGCError(
            "gc_project_unsafe", "GC path is not a safe project-relative path",
        )
    return path.as_posix()


def _project_root(project_dir: Path) -> Path:
    raw = Path(project_dir).absolute()
    try:
        mode = raw.lstat().st_mode
    except OSError as exc:
        raise CompanionGCError(
            "gc_project_unsafe", "companion project root is unavailable",
        ) from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise CompanionGCError(
            "gc_project_unsafe", "companion project root is not a directory",
        )
    return raw.resolve()


def _relative(root: Path, path: Path) -> str:
    try:
        relative = path.absolute().relative_to(root)
    except ValueError as exc:
        raise CompanionGCError(
            "gc_project_unsafe", "owned path escapes the companion project",
        ) from exc
    return _safe_relative(relative.as_posix())


def _rooted(path: str, roots: Iterable[str]) -> bool:
    return any(path == root or path.startswith(root + "/") for root in roots)


def _require_regular_or_missing(
    root: Path, path: Path, *, code: str,
) -> None:
    _reject_symlink_components(root, path, code=code)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    except OSError as exc:
        raise CompanionGCError(code, "owned file is unavailable") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise CompanionGCError(code, "owned file is not a regular file")


def _require_directory(root: Path, path: Path, *, code: str) -> None:
    _reject_symlink_components(root, path, code=code)
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise CompanionGCError(code, "owned directory is unavailable") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise CompanionGCError(code, "owned path is not a directory")


def _reject_symlink_components(
    root: Path, path: Path, *, code: str,
) -> None:
    try:
        relative = path.absolute().relative_to(root)
    except ValueError as exc:
        raise CompanionGCError(code, "owned path escapes the project") from exc
    current = root
    for component in relative.parts:
        current = current / component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            return
        except OSError as exc:
            raise CompanionGCError(code, "owned path is unavailable") from exc
        if stat.S_ISLNK(mode):
            raise CompanionGCError(code, "owned path contains a symbolic link")


def _bounded_children(path: Path) -> tuple[Path, ...]:
    try:
        children = tuple(sorted(path.iterdir(), key=lambda item: item.name))
    except OSError as exc:
        raise CompanionGCError(
            "gc_project_unsafe", "owned directory cannot be enumerated",
        ) from exc
    if len(children) > MAX_RECOGNIZED_ENTRIES:
        raise CompanionGCError(
            "gc_project_unsafe", "owned directory exceeds the GC entry bound",
        )
    return children


def _hash_file(path: Path, *, candidate: bool = False) -> tuple[str, int]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CompanionGCError(
            "gc_project_unsafe", "owned file cannot be opened safely",
        ) from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CompanionGCError(
                "gc_project_unsafe", "owned file is not regular",
            )
        if candidate and before.st_nlink != 1:
            raise CompanionGCError(
                "gc_candidate_unsafe",
                "cleanup candidate has multiple hard links",
            )
        if before.st_size > MAX_CANDIDATE_BYTES:
            raise CompanionGCError(
                "gc_project_unsafe", "owned file exceeds the GC size bound",
            )
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mode,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mode,
        ):
            raise CompanionGCError(
                "gc_candidate_changed", "owned file changed while hashing",
            )
        return digest.hexdigest(), size
    finally:
        os.close(descriptor)


def _read_regular_bytes(
    root: Path,
    path: Path,
    *,
    max_bytes: int,
    code: str,
) -> bytes:
    _reject_symlink_components(root, path, code=code)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CompanionGCError(code, "required owned file is unreadable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CompanionGCError(code, "required owned file is not regular")
        if before.st_size > max_bytes:
            raise CompanionGCError(code, "owned JSON exceeds its size bound")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > max_bytes:
                raise CompanionGCError(code, "owned JSON exceeds its size bound")
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mode,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mode,
        ):
            raise CompanionGCError(code, "owned file changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _json_object(value: bytes, code: str) -> Mapping[str, Any]:
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise CompanionGCError(code, "owned JSON is malformed") from exc
    if not isinstance(parsed, dict):
        raise CompanionGCError(code, "owned JSON is not an object")
    return parsed


def _sha_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _state_file(
    root: Path, value: object, expected_sha256: str, *, code: str,
) -> Path:
    if not value or not _SHA256.fullmatch(expected_sha256):
        raise CompanionGCError(code, "published file identity is incomplete")
    raw = Path(str(value))
    path = raw if raw.is_absolute() else root / _safe_relative(raw)
    try:
        relative = path.absolute().relative_to(root)
    except ValueError as exc:
        raise CompanionGCError(code, "published file escapes the project") from exc
    path = root / _safe_relative(relative.as_posix())
    _require_regular_or_missing(root, path, code=code)
    if not path.exists():
        raise CompanionGCError(code, "published file is missing")
    actual, _size = _hash_file(path)
    if actual != expected_sha256:
        raise CompanionGCError(code, "published file digest differs")
    return path


def _require_filename_hash(root: Path, path: Path, expected: str) -> None:
    _require_regular_or_missing(root, path, code="gc_reader_invalid")
    actual, _size = _hash_file(path, candidate=True)
    if actual != expected:
        raise CompanionGCError(
            "gc_reader_invalid", "Reader object filename hash differs",
        )


def _file_candidate(root: Path, path: Path, category: str) -> GCCandidate:
    relative = _relative(root, path)
    _require_regular_or_missing(root, path, code="gc_candidate_unsafe")
    digest, size = _hash_file(path, candidate=True)
    return GCCandidate(category, relative, "file", size, digest)


def _tree_identity(
    root: Path, path: Path, *, candidate: bool,
) -> tuple[str, int]:
    _require_directory(root, path, code="gc_candidate_unsafe")
    records: list[dict[str, object]] = []
    total = 0
    count = 0

    def visit(directory: Path, depth: int) -> None:
        nonlocal total, count
        if depth > MAX_DIRECTORY_DEPTH:
            raise CompanionGCError(
                "gc_candidate_unsafe", "owned tree exceeds the GC depth bound",
            )
        for child in _bounded_children(directory):
            count += 1
            if count > MAX_RECOGNIZED_ENTRIES:
                raise CompanionGCError(
                    "gc_candidate_unsafe", "owned tree exceeds the GC entry bound",
                )
            relative = child.relative_to(path).as_posix()
            mode = child.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise CompanionGCError(
                    "gc_candidate_unsafe", "owned tree contains a symbolic link",
                )
            if stat.S_ISDIR(mode):
                records.append({"path": relative, "kind": "directory"})
                visit(child, depth + 1)
            elif stat.S_ISREG(mode):
                digest, size = _hash_file(child, candidate=candidate)
                total += size
                records.append({
                    "path": relative,
                    "kind": "file",
                    "bytes": size,
                    "sha256": digest,
                })
            else:
                raise CompanionGCError(
                    "gc_candidate_unsafe", "owned tree contains a special file",
                )

    visit(path, 0)
    return _sha_json(records), total


def _directory_candidate(
    root: Path, path: Path, category: str,
) -> GCCandidate:
    digest, size = _tree_identity(root, path, candidate=True)
    return GCCandidate(
        category, _relative(root, path), "directory", size, digest,
    )


def _manifest_graph(root: Path, manifest_path: Path) -> set[str]:
    value = _json_object(
        _read_regular_bytes(
            root, manifest_path, max_bytes=MAX_JSON_BYTES,
            code="gc_reader_invalid",
        ),
        "gc_reader_invalid",
    )
    if value.get("schema_version") != WEB_MANIFEST_VERSION:
        raise CompanionGCError(
            "gc_reader_invalid", "Reader history manifest schema is invalid",
        )
    records: list[tuple[Mapping[str, Any], bool]] = []
    for key in ("snapshot", "data_script"):
        record = value.get(key)
        if not isinstance(record, Mapping):
            raise CompanionGCError(
                "gc_reader_invalid", "Reader manifest record is missing",
            )
        records.append((record, True))
    index = value.get("index")
    if not isinstance(index, Mapping):
        raise CompanionGCError(
            "gc_reader_invalid", "Reader manifest index record is missing",
        )
    records.append((index, False))
    assets = value.get("assets")
    if not isinstance(assets, list) or len(assets) > MAX_MANIFEST_REFERENCES:
        raise CompanionGCError(
            "gc_reader_invalid", "Reader manifest assets are invalid",
        )
    if not all(isinstance(record, Mapping) for record in assets):
        raise CompanionGCError(
            "gc_reader_invalid", "Reader manifest asset record is invalid",
        )
    records.extend((record, True) for record in assets)
    graph: set[str] = set()
    for record, immutable in records:
        keys = {"path", "sha256", "bytes"}
        if not keys.issubset(record):
            raise CompanionGCError(
                "gc_reader_invalid", "Reader manifest file identity is incomplete",
            )
        relative = _safe_relative(str(record["path"]))
        digest = str(record["sha256"])
        size = record["bytes"]
        if (
            not _SHA256.fullmatch(digest)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise CompanionGCError(
                "gc_reader_invalid", "Reader manifest file identity is invalid",
            )
        if not relative.startswith("reader/"):
            raise CompanionGCError(
                "gc_reader_invalid", "Reader manifest reference escapes Reader",
            )
        graph.add(relative)
        if not immutable:
            if relative != "reader/index.html":
                raise CompanionGCError(
                    "gc_reader_invalid", "Reader manifest index path is invalid",
                )
            continue
        path = root / relative
        _require_regular_or_missing(root, path, code="gc_reader_invalid")
        if not path.exists():
            raise CompanionGCError(
                "gc_reader_invalid", "Reader manifest object is missing",
            )
        actual, actual_size = _hash_file(path)
        if actual != digest or actual_size != size:
            raise CompanionGCError(
                "gc_reader_invalid", "Reader manifest object identity differs",
            )
    return graph


def _builtin_identity(path: Path) -> str:
    digest = hashlib.sha256()
    count = 0
    for child in sorted(path.rglob("*")):
        count += 1
        if count > MAX_RECOGNIZED_ENTRIES:
            raise CompanionGCError(
                "gc_reader_invalid", "Reader asset tree exceeds the entry bound",
            )
        mode = child.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise CompanionGCError(
                "gc_reader_invalid", "Reader builtin assets are not regular files",
            )
        relative = child.relative_to(path).as_posix()
        file_digest, _size = _hash_file(child, candidate=True)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_digest))
    return digest.hexdigest()


def _extra_root_records(
    root: Path, extra_roots: Iterable[str | Path],
) -> set[str]:
    records: set[str] = set()
    for value in extra_roots:
        relative = _safe_relative(value)
        if relative in {"state.json", ".arc-companion-build.lock"}:
            raise CompanionGCError(
                "gc_extra_root_invalid", "extra root cannot name project control state",
            )
        path = root / relative
        if not path.exists() and not path.is_symlink():
            raise CompanionGCError(
                "gc_extra_root_invalid", "extra root does not exist",
            )
        try:
            mode = path.lstat().st_mode
        except OSError as exc:
            raise CompanionGCError(
                "gc_extra_root_invalid", "extra root is unavailable",
            ) from exc
        if stat.S_ISLNK(mode):
            raise CompanionGCError(
                "gc_extra_root_invalid", "extra root cannot be a symbolic link",
            )
        if not stat.S_ISREG(mode) and not stat.S_ISDIR(mode):
            raise CompanionGCError(
                "gc_extra_root_invalid", "extra root has an unsupported type",
            )
        records.add(relative)
    return records


def _identity_record(root: Path, path: Path) -> Mapping[str, Any]:
    mode = path.lstat().st_mode
    if stat.S_ISREG(mode):
        digest, size = _hash_file(path)
        return {"kind": "file", "sha256": digest, "bytes": size}
    if stat.S_ISDIR(mode):
        digest, size = _tree_identity(root, path, candidate=False)
        return {"kind": "directory", "sha256": digest, "bytes": size}
    raise CompanionGCError(
        "gc_project_unsafe", "retained root has an unsupported type",
    )


def _checkpoint_snapshot(
    root: Path, state: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    raw = state.get("checkpoint_dir")
    if not raw:
        return None
    try:
        relative = _safe_relative(
            Path(str(raw)).absolute().relative_to(root).as_posix()
            if Path(str(raw)).is_absolute() else str(raw)
        )
    except (CompanionGCError, ValueError) as exc:
        raise CompanionGCError(
            "gc_checkpoint_invalid", "checkpoint path is outside the project",
        ) from exc
    checkpoint = root / relative
    _require_directory(root, checkpoint, code="gc_checkpoint_invalid")
    try:
        from .pipeline import _resolve_checkpoint_state_identity

        allocation = _resolve_checkpoint_state_identity(
            root, state, checkpoint,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise CompanionGCError(
            "gc_checkpoint_invalid", "checkpoint state identity is invalid",
        ) from exc
    identity = str(
        state.get("checkpoint_identity")
        or state.get("fingerprint")
        or (state.get("active_run") or {}).get("fingerprint")
        or ""
    )
    return {
        "path": _relative(root, allocation.path),
        "identity": identity,
        "directory": _identity_record(root, allocation.path),
    }


def _content_snapshot(
    root: Path, state: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    published = state.get("published")
    published = published if isinstance(published, Mapping) else {}
    digest = str(published.get("content_sha256") or "")
    if not digest:
        return None
    if not _SHA256.fullmatch(digest):
        raise CompanionGCError(
            "gc_content_invalid", "published content identity is invalid",
        )
    try:
        from .content import content_object_path, load_reader_content

        path = content_object_path(root, digest)
        load_reader_content(root, digest)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise CompanionGCError(
            "gc_content_invalid", "published content object is invalid",
        ) from exc
    file_digest, size = _hash_file(path)
    return {
        "content_sha256": digest,
        "path": _relative(root, path),
        "file_sha256": file_digest,
        "bytes": size,
    }


def _managed_pdf_snapshot(
    root: Path, state: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    normalized = normalize_run_root_pdf_state(state)
    path = managed_run_root_pdf_path(normalized)
    if path is None:
        return None
    path = path if path.is_absolute() else root / _safe_relative(path)
    expected = str(
        normalized.get("output_run_pdf_sha256")
        or (
            (normalized.get("published") or {}).get("pdf") or {}
        ).get("output_run_pdf_sha256")
        or ""
    )
    if not _SHA256.fullmatch(expected):
        raise CompanionGCError(
            "gc_render_invalid", "managed run-root PDF identity is incomplete",
        )
    selected = _state_file(root, path, expected, code="gc_render_invalid")
    digest, size = _hash_file(selected)
    return {"path": _relative(root, selected), "sha256": digest, "bytes": size}


def _directory_identity_snapshot(root: Path) -> Mapping[str, Any]:
    checkpoint_root = root / ".arc-companion" / "checkpoints"
    if not checkpoint_root.exists():
        return {}
    _require_directory(root, checkpoint_root, code="gc_checkpoint_invalid")
    records: dict[str, Any] = {}
    for child in _bounded_children(checkpoint_root):
        receipt = child / ARTIFACT_ID_RECEIPT_NAME
        if child.is_dir() and not child.is_symlink() and receipt.is_file():
            digest, size = _hash_file(receipt)
            records[_relative(root, receipt)] = {
                "sha256": digest,
                "bytes": size,
            }
    return records


def _recognize_legacy_render(
    root: Path,
    path: Path,
    *,
    state: Mapping[str, Any] | None,
) -> None:
    _require_directory(root, path, code="gc_render_invalid")
    allowed = {
        "source-manifest.json",
        "validation.json",
        ARTIFACT_ID_RECEIPT_NAME,
    }
    tex: list[Path] = []
    pdf: list[Path] = []
    for child in _bounded_children(path):
        if child.is_symlink() or not child.is_file():
            raise CompanionGCError(
                "gc_render_invalid", "legacy render contains an unsupported entry",
            )
        if child.suffix == ".tex" and not _STAGING.fullmatch(child.name):
            tex.append(child)
        elif child.suffix == ".pdf" and not _STAGING.fullmatch(child.name):
            pdf.append(child)
        elif (
            child.name not in allowed
            and not _VALIDATION_PAGE.fullmatch(child.name)
            and not _VALIDATION_TEXT.fullmatch(child.name)
            and not _STAGING.fullmatch(child.name)
        ):
            raise CompanionGCError(
                "gc_render_invalid", "legacy render shape is not recognized",
            )
    if len(tex) != 1 or len(pdf) != 1:
        raise CompanionGCError(
            "gc_render_invalid", "legacy render must have one TeX and one PDF",
        )
    manifest = path / "source-manifest.json"
    validation = path / "validation.json"
    for required in (manifest, validation):
        if not required.is_file():
            raise CompanionGCError(
                "gc_render_invalid", "legacy render receipt is missing",
            )
        _json_object(
            _read_regular_bytes(
                root, required, max_bytes=MAX_JSON_BYTES,
                code="gc_render_invalid",
            ),
            "gc_render_invalid",
        )
    validation_value = _json_object(
        validation.read_bytes(), "gc_render_invalid",
    )
    if validation_value.get("ok") is not True and validation_value.get(
        "result"
    ) != "success":
        raise CompanionGCError(
            "gc_render_invalid", "legacy render validation did not succeed",
        )
    if state is not None:
        expected = {
            Path(str(state.get("output_pdf") or "")).name,
            Path(str(state.get("output_tex") or "")).name,
            Path(str(state.get("source_manifest_path") or "")).name,
            Path(str(state.get("validation_path") or "")).name,
        }
        if not {pdf[0].name, tex[0].name, manifest.name, validation.name}.issubset(
            expected
        ):
            raise CompanionGCError(
                "gc_render_invalid", "legacy render and published state differ",
            )


def _legacy_sidecars(path: Path) -> tuple[Path, ...]:
    return tuple(
        child for child in _bounded_children(path)
        if (
            _VALIDATION_PAGE.fullmatch(child.name)
            or _VALIDATION_TEXT.fullmatch(child.name)
            or _STAGING.fullmatch(child.name)
        )
    )


def _require_antichain(candidates: Sequence[GCCandidate]) -> None:
    paths = sorted(candidate.path for candidate in candidates)
    if len(paths) != len(set(paths)):
        raise CompanionGCError(
            "gc_candidate_unsafe", "cleanup candidates overlap exactly",
        )
    for index, path in enumerate(paths):
        for other in paths[index + 1:]:
            if other.startswith(path + "/"):
                raise CompanionGCError(
                    "gc_candidate_unsafe", "cleanup candidates are not an antichain",
                )
            if not other.startswith(path):
                break


def _totals(
    candidates: Sequence[GCCandidate],
) -> tuple[tuple[str, int, int], ...]:
    totals: dict[str, list[int]] = {}
    for candidate in candidates:
        values = totals.setdefault(candidate.category, [0, 0])
        values[0] += 1
        values[1] += candidate.bytes
    return tuple(
        (category, values[0], values[1])
        for category, values in sorted(totals.items())
    )


def _measure_tree(path: Path) -> tuple[int, int]:
    if not path.exists() or path.is_symlink():
        return (0, 0)
    if path.is_file():
        _digest, size = _hash_file(path)
        return (1, size)
    count = 0
    size = 0

    def visit(directory: Path, depth: int) -> None:
        nonlocal count, size
        if depth > MAX_DIRECTORY_DEPTH:
            raise CompanionGCError(
                "gc_project_unsafe", "retained tree exceeds the depth bound",
            )
        for child in _bounded_children(directory):
            mode = child.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise CompanionGCError(
                    "gc_project_unsafe", "retained tree contains a symbolic link",
                )
            if stat.S_ISDIR(mode):
                visit(child, depth + 1)
            elif stat.S_ISREG(mode):
                count += 1
                if count > MAX_RECOGNIZED_ENTRIES:
                    raise CompanionGCError(
                        "gc_project_unsafe",
                        "retained tree exceeds the entry bound",
                    )
                _digest, file_size = _hash_file(child)
                size += file_size
            else:
                raise CompanionGCError(
                    "gc_project_unsafe", "retained tree contains a special file",
                )

    visit(path, 0)
    return count, size


def _retained_totals(root: Path) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (category, *_measure_tree(root / relative))
        for category, relative in sorted(_RETAINED_ROOTS.items())
    )


def _active_transaction_hashes(
    root: Path, *, allow_gc_transaction: bool,
) -> Mapping[str, Any]:
    records: dict[str, Any] = {}
    resume = root / ".arc-companion" / "resume-transaction.json"
    if resume.exists() or resume.is_symlink():
        value_bytes = _read_regular_bytes(
            root, resume, max_bytes=MAX_JSON_BYTES,
            code="gc_transaction_invalid",
        )
        value = _json_object(value_bytes, "gc_transaction_invalid")
        if value.get("status") != "complete":
            raise CompanionGCError(
                "gc_transaction_active",
                "an incomplete resume transaction is active",
            )
        records[_relative(root, resume)] = hashlib.sha256(value_bytes).hexdigest()
    transactions = root / ".arc-companion" / "gc" / "transactions"
    if transactions.exists() or transactions.is_symlink():
        _require_directory(root, transactions, code="gc_transaction_invalid")
        for path in _bounded_children(transactions):
            if not path.name.endswith(".json"):
                raise CompanionGCError(
                    "gc_transaction_invalid",
                    "GC transaction directory contains an unknown entry",
                )
            value_bytes = _read_regular_bytes(
                root, path, max_bytes=MAX_JSON_BYTES,
                code="gc_transaction_invalid",
            )
            value = _json_object(value_bytes, "gc_transaction_invalid")
            status = str(value.get("status") or "")
            if status in _NONTERMINAL_TRANSACTION_STATES:
                if allow_gc_transaction:
                    continue
                raise CompanionGCError(
                    "gc_transaction_active",
                    "an incomplete GC transaction requires recovery",
                )
            if status != "complete":
                raise CompanionGCError(
                    "gc_transaction_invalid", "GC transaction status is invalid",
                )
    return records


def apply_gc(
    project_dir: Path,
    *,
    candidate_digest: str | None = None,
    extra_roots: Iterable[str | Path] = (),
    lock_already_held: bool = False,
) -> dict[str, Any]:
    """Apply one exact discovery set with journaled forward-only recovery."""

    if candidate_digest is not None and not _SHA256.fullmatch(candidate_digest):
        raise CompanionGCError(
            "gc_candidate_digest_invalid", "candidate digest is not SHA-256",
        )
    root = _project_root(project_dir)
    roots = tuple(extra_roots)
    active = _find_nonterminal_gc_transaction(root)
    pre_lock: _Discovery | None = None
    if active is None:
        pre_lock = _discover_gc(
            root,
            extra_roots=roots,
            allow_active_build_lock=lock_already_held,
            allow_gc_transaction=False,
        )
        if (
            candidate_digest is not None
            and candidate_digest
            != pre_lock.report.candidate_set_sha256
        ):
            raise CompanionGCError(
                "gc_candidate_set_changed",
                "candidate digest differs from the pre-lock discovery set",
            )
    lock: ProjectBuildLock | None = None
    try:
        if not lock_already_held:
            lock = ProjectBuildLock(root / ".arc-companion-build.lock")
            try:
                lock.acquire()
            except BuildInProgressError as exc:
                raise CompanionGCError(
                    "gc_build_active", "the companion project build lock is active",
                ) from exc
        if active is not None:
            return _recover_transaction(
                root,
                active,
                extra_roots=roots,
                candidate_digest=candidate_digest,
            )
        under_lock = _discover_gc(
            root,
            extra_roots=roots,
            allow_active_build_lock=True,
            allow_gc_transaction=False,
        )
        report = under_lock.report
        assert pre_lock is not None
        if (
            pre_lock.report.root_snapshot_sha256
            != report.root_snapshot_sha256
            or pre_lock.report.candidate_set_sha256
            != report.candidate_set_sha256
        ):
            raise CompanionGCError(
                "gc_candidate_set_changed",
                "publication roots or candidates changed while acquiring the lock",
            )
        if (
            candidate_digest is not None
            and candidate_digest != report.candidate_set_sha256
        ):
            raise CompanionGCError(
                "gc_candidate_set_changed",
                "candidate digest differs from the current discovery set",
            )
        for candidate in report.candidates:
            _require_candidate_match(root, candidate)
        transaction_id = _sha_json({
            "schema_version": GC_TRANSACTION_VERSION,
            "recognizer_version": GC_RECOGNIZER_VERSION,
            "project_identity_sha256": report.project_identity_sha256,
            "root_snapshot_sha256": report.root_snapshot_sha256,
            "candidate_set_sha256": report.candidate_set_sha256,
        })
        transaction_path = _transaction_path(root, transaction_id)
        receipt_path = _receipt_path(root, transaction_id)
        existing_receipt = _read_terminal_receipt(
            root, receipt_path, transaction_id=transaction_id,
        )
        if existing_receipt is not None:
            return _receipt_result(root, receipt_path, existing_receipt)
        transaction = _new_transaction(
            root, under_lock, transaction_id=transaction_id,
        )
        _write_json_atomic(root, transaction_path, transaction)
        _gc_fault_point("transaction_planned")
        return _execute_transaction(
            root,
            transaction_path,
            transaction,
            extra_roots=roots,
        )
    finally:
        if lock is not None:
            lock.release()


def run_post_publication_gc(
    project_dir: Path,
    *,
    state_merger: Any,
    extra_roots: Iterable[str | Path] = (),
    lock_already_held: bool = True,
) -> Mapping[str, Any]:
    """Run best-effort GC after a durable publication and persist its outcome."""

    try:
        receipt = apply_gc(
            project_dir,
            extra_roots=extra_roots,
            lock_already_held=lock_already_held,
        )
        update = {
            "artifact_gc": {
                "status": "complete",
                "receipt_path": receipt["receipt_path"],
                "receipt_sha256": receipt["receipt_sha256"],
                "candidate_set_sha256": receipt["candidate_set_sha256"],
                "reclaimed_bytes": receipt["reclaimed_bytes"],
            },
            "artifact_gc_warning": None,
        }
    except CompanionGCError as exc:
        update = {
            "artifact_gc": None,
            "artifact_gc_warning": {
                "code": exc.code,
                "message": str(exc)[:256],
            },
        }
    except Exception as exc:  # publication must survive unexpected GC failure
        update = {
            "artifact_gc": None,
            "artifact_gc_warning": {
                "code": "gc_failed",
                "message": str(exc)[:256] or exc.__class__.__name__,
            },
        }
    except KeyboardInterrupt:
        update = {
            "artifact_gc": None,
            "artifact_gc_warning": {
                "code": "gc_interrupted",
                "message": "automatic artifact cleanup was interrupted",
            },
        }
    try:
        state_merger(update)
    except Exception as exc:
        return {
            **(
                {"artifact_gc": update["artifact_gc"]}
                if "artifact_gc" in update else {}
            ),
            "artifact_gc_warning": {
                "code": "gc_state_update_failed",
                "message": str(exc)[:256] or exc.__class__.__name__,
            },
        }
    return update


def _find_nonterminal_gc_transaction(root: Path) -> Path | None:
    directory = root / ".arc-companion" / "gc" / "transactions"
    if not directory.exists():
        return None
    _require_directory(root, directory, code="gc_transaction_invalid")
    active: list[Path] = []
    for path in _bounded_children(directory):
        if not path.name.endswith(".json"):
            raise CompanionGCError(
                "gc_transaction_invalid", "GC transaction entry is invalid",
            )
        value = _json_object(
            _read_regular_bytes(
                root, path, max_bytes=MAX_JSON_BYTES,
                code="gc_transaction_invalid",
            ),
            "gc_transaction_invalid",
        )
        if value.get("status") in _NONTERMINAL_TRANSACTION_STATES:
            active.append(path)
    if len(active) > 1:
        raise CompanionGCError(
            "gc_transaction_invalid", "multiple incomplete GC transactions exist",
        )
    return active[0] if active else None


def _transaction_path(root: Path, transaction_id: str) -> Path:
    return (
        root / ".arc-companion" / "gc" / "transactions"
        / f"{transaction_id}.json"
    )


def _receipt_path(root: Path, transaction_id: str) -> Path:
    return (
        root / ".arc-companion" / "gc" / "receipts"
        / f"{transaction_id}.json"
    )


def _new_transaction(
    root: Path, discovery: _Discovery, *, transaction_id: str,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for candidate in discovery.report.candidates:
        record = candidate.as_dict()
        record.update({
            "original_path": candidate.path,
            "quarantine_path": (
                f".arc-companion/gc-trash/{transaction_id}/payload/"
                f"{candidate.path}"
            ),
            "state": "planned",
        })
        record.pop("path")
        candidates.append(record)
    return {
        "schema_version": GC_TRANSACTION_VERSION,
        "recognizer_version": GC_RECOGNIZER_VERSION,
        "transaction_id": transaction_id,
        "project_identity_sha256": (
            discovery.report.project_identity_sha256
        ),
        "root_snapshot_sha256": discovery.report.root_snapshot_sha256,
        "root_snapshot_payload": dict(discovery.root_snapshot_payload),
        "candidate_set_sha256": discovery.report.candidate_set_sha256,
        "category_totals": {
            category: {"count": count, "bytes": size}
            for category, count, size in discovery.report.category_totals
        },
        "retained_class_totals": {
            category: {"count": count, "bytes": size}
            for category, count, size in (
                discovery.report.retained_class_totals
            )
        },
        "warnings": list(discovery.report.warnings),
        "status": "planned",
        "candidates": candidates,
    }


def _recover_transaction(
    root: Path,
    transaction_path: Path,
    *,
    extra_roots: Iterable[str | Path],
    candidate_digest: str | None,
) -> dict[str, Any]:
    transaction = _read_transaction(root, transaction_path)
    if (
        candidate_digest is not None
        and transaction.get("candidate_set_sha256") != candidate_digest
    ):
        raise CompanionGCError(
            "gc_candidate_set_changed",
            "candidate digest differs from the recoverable transaction",
        )
    return _execute_transaction(
        root, transaction_path, transaction, extra_roots=extra_roots,
    )


def _read_transaction(root: Path, path: Path) -> dict[str, Any]:
    value = dict(_json_object(
        _read_regular_bytes(
            root, path, max_bytes=MAX_JSON_BYTES,
            code="gc_transaction_invalid",
        ),
        "gc_transaction_invalid",
    ))
    transaction_id = str(value.get("transaction_id") or "")
    base_keys = {
        "schema_version",
        "recognizer_version",
        "transaction_id",
        "project_identity_sha256",
        "root_snapshot_sha256",
        "root_snapshot_payload",
        "candidate_set_sha256",
        "category_totals",
        "retained_class_totals",
        "warnings",
        "status",
        "candidates",
    }
    allowed_keys = (
        base_keys | {"receipt_path", "receipt_sha256"}
        if value.get("status") == "complete" else base_keys
    )
    if (
        set(value) != allowed_keys
        or value.get("schema_version") != GC_TRANSACTION_VERSION
        or value.get("recognizer_version") != GC_RECOGNIZER_VERSION
        or not _SHA256.fullmatch(transaction_id)
        or path != _transaction_path(root, transaction_id)
        or value.get("status") not in (
            *_NONTERMINAL_TRANSACTION_STATES, "complete",
        )
        or not isinstance(value.get("candidates"), list)
        or not isinstance(value.get("root_snapshot_payload"), Mapping)
    ):
        raise CompanionGCError(
            "gc_transaction_invalid", "GC transaction journal is invalid",
        )
    if _sha_json(value["root_snapshot_payload"]) != value.get(
        "root_snapshot_sha256"
    ):
        raise CompanionGCError(
            "gc_transaction_invalid", "GC root snapshot journal digest differs",
        )
    expected_transaction_id = _sha_json({
        "schema_version": GC_TRANSACTION_VERSION,
        "recognizer_version": GC_RECOGNIZER_VERSION,
        "project_identity_sha256": value.get("project_identity_sha256"),
        "root_snapshot_sha256": value.get("root_snapshot_sha256"),
        "candidate_set_sha256": value.get("candidate_set_sha256"),
    })
    if expected_transaction_id != transaction_id:
        raise CompanionGCError(
            "gc_transaction_invalid", "GC transaction identity differs",
        )
    expected_candidates: list[dict[str, object]] = []
    for record in value["candidates"]:
        if (
            not isinstance(record, Mapping)
            or set(record) != {
                "category",
                "kind",
                "bytes",
                "sha256",
                "recognizer_version",
                "original_path",
                "quarantine_path",
                "state",
            }
        ):
            raise CompanionGCError(
                "gc_transaction_invalid", "GC candidate journal is invalid",
            )
        original = _safe_relative(str(record.get("original_path") or ""))
        quarantine = _safe_relative(str(record.get("quarantine_path") or ""))
        if quarantine != (
            f".arc-companion/gc-trash/{transaction_id}/payload/{original}"
        ):
            raise CompanionGCError(
                "gc_transaction_invalid", "GC quarantine binding is invalid",
            )
        candidate = _candidate_from_record(record, path_key="original_path")
        _validate_candidate_binding(candidate)
        expected_candidates.append(candidate.as_dict())
        if record.get("state") not in {"planned", "moved", "deleted"}:
            raise CompanionGCError(
                "gc_transaction_invalid", "GC candidate state is invalid",
            )
    if _sha_json(expected_candidates) != value.get("candidate_set_sha256"):
        raise CompanionGCError(
            "gc_transaction_invalid", "GC candidate journal digest differs",
        )
    expected_totals = {
        category: {"count": count, "bytes": size}
        for category, count, size in _totals([
            _candidate_from_record(record, path_key="original_path")
            for record in value["candidates"]
        ])
    }
    if (
        value.get("category_totals") != expected_totals
        or not _valid_totals(value.get("retained_class_totals"))
        or not _valid_warnings(value.get("warnings"))
        or not _SHA256.fullmatch(
            str(value.get("project_identity_sha256") or "")
        )
        or not _SHA256.fullmatch(
            str(value.get("root_snapshot_sha256") or "")
        )
    ):
        raise CompanionGCError(
            "gc_transaction_invalid", "GC transaction totals are invalid",
        )
    if value.get("status") == "complete":
        expected_receipt = _receipt_path(root, transaction_id)
        if (
            value.get("receipt_path") != _relative(root, expected_receipt)
            or not _SHA256.fullmatch(str(value.get("receipt_sha256") or ""))
            or _hash_file(expected_receipt)[0] != value.get("receipt_sha256")
        ):
            raise CompanionGCError(
                "gc_transaction_invalid",
                "GC transaction terminal receipt binding is invalid",
            )
    return value


def _validate_candidate_binding(candidate: GCCandidate) -> None:
    path = PurePosixPath(candidate.path)
    valid = False
    if candidate.category == "reader_manifest_history":
        valid = (
            candidate.kind == "file"
            and len(path.parts) == 3
            and path.parts[:2] == ("reader", "data")
            and _MANIFEST_NAME.fullmatch(path.name) is not None
        )
    elif candidate.category == "reader_snapshot_history":
        valid = (
            candidate.kind == "file"
            and len(path.parts) == 3
            and path.parts[:2] == ("reader", "data")
            and _SNAPSHOT_NAME.fullmatch(path.name) is not None
        )
    elif candidate.category == "reader_data_history":
        valid = (
            candidate.kind == "file"
            and len(path.parts) == 3
            and path.parts[:2] == ("reader", "data")
            and _DATA_NAME.fullmatch(path.name) is not None
        )
    elif candidate.category == "reader_builtin_history":
        valid = (
            candidate.kind == "directory"
            and len(path.parts) == 3
            and path.parts[:2] == ("reader", "assets")
            and _BUILTIN_NAME.fullmatch(path.name) is not None
        )
    elif candidate.category == "reader_source_history":
        valid = (
            candidate.kind == "file"
            and len(path.parts) == 4
            and path.parts[:3] == ("reader", "assets", "source")
            and _SOURCE_NAME.fullmatch(path.name) is not None
        )
    elif candidate.category == "render_history":
        valid = (
            candidate.kind == "directory"
            and len(path.parts) == 4
            and path.parts[:3] == (
                ".arc-companion", "renders", "pdf",
            )
        )
    elif candidate.category == "validation_temporary":
        valid = (
            candidate.kind == "file"
            and len(path.parts) == 5
            and path.parts[:3] == (
                ".arc-companion", "renders", "pdf",
            )
            and (
                _VALIDATION_PAGE.fullmatch(path.name)
                or _VALIDATION_TEXT.fullmatch(path.name)
                or _STAGING.fullmatch(path.name)
            )
        )
    if not valid:
        raise CompanionGCError(
            "gc_transaction_invalid",
            "GC candidate path does not match its recognizer category",
        )


def _candidate_from_record(
    record: Mapping[str, Any], *, path_key: str,
) -> GCCandidate:
    try:
        return GCCandidate(
            category=str(record["category"]),
            path=str(record[path_key]),
            kind=str(record["kind"]),
            bytes=int(record["bytes"]),
            sha256=str(record["sha256"]),
            recognizer_version=str(record["recognizer_version"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CompanionGCError(
            "gc_transaction_invalid", "GC candidate record is invalid",
        ) from exc


def _execute_transaction(
    root: Path,
    transaction_path: Path,
    transaction: dict[str, Any],
    *,
    extra_roots: Iterable[str | Path],
) -> dict[str, Any]:
    transaction_id = str(transaction["transaction_id"])
    _validate_trash_shape(root, transaction_id, transaction["candidates"])
    receipt_path = _receipt_path(root, transaction_id)
    existing = _read_terminal_receipt(
        root, receipt_path, transaction_id=transaction_id,
    )
    if existing is not None:
        if transaction.get("status") != "complete":
            transaction["status"] = "complete"
            transaction["receipt_path"] = _relative(root, receipt_path)
            transaction["receipt_sha256"] = _hash_file(receipt_path)[0]
            _write_json_atomic(root, transaction_path, transaction)
        return _receipt_result(root, receipt_path, existing)

    recovering_partial_move = (
        transaction.get("status") != "planned"
        or any(
            (root / _safe_relative(record["quarantine_path"])).exists()
            for record in transaction["candidates"]
        )
    )
    discovery = _discover_gc(
        root,
        extra_roots=extra_roots,
        allow_active_build_lock=True,
        allow_gc_transaction=True,
        scan_candidates=not recovering_partial_move,
    )
    if (
        discovery.report.project_identity_sha256
        != transaction.get("project_identity_sha256")
        or discovery.root_snapshot_payload
        != transaction.get("root_snapshot_payload")
    ):
        raise CompanionGCError(
            "gc_publication_changed",
            "published roots changed during the GC transaction",
        )

    records = transaction["candidates"]
    resuming_deletion = transaction.get("status") == "deleting"
    if resuming_deletion and any(
        record["state"] == "planned" for record in records
    ):
        raise CompanionGCError(
            "gc_transaction_invalid",
            "deleting GC transaction contains an unmoved candidate",
        )
    if not resuming_deletion:
        transaction["status"] = "moving"
        _write_json_atomic(root, transaction_path, transaction)
        for record in records:
            if record["state"] != "planned":
                continue
            original = root / _safe_relative(record["original_path"])
            quarantine = root / _safe_relative(record["quarantine_path"])
            original_exists = original.exists() or original.is_symlink()
            quarantine_exists = quarantine.exists() or quarantine.is_symlink()
            if original_exists and quarantine_exists:
                raise CompanionGCError(
                    "gc_transaction_invalid",
                    "both original and quarantined candidate exist",
                )
            candidate = _candidate_from_record(record, path_key="original_path")
            if quarantine_exists:
                quarantine_candidate = GCCandidate(
                    candidate.category,
                    _relative(root, quarantine),
                    candidate.kind,
                    candidate.bytes,
                    candidate.sha256,
                )
                _require_candidate_match(root, quarantine_candidate)
            elif original_exists:
                _require_candidate_match(root, candidate)
                _ensure_directory_chain(root, quarantine.parent)
                _gc_fault_point("before_move")
                os.replace(original, quarantine)
                _fsync_directory(original.parent)
                _fsync_directory(quarantine.parent)
                _gc_fault_point("after_move")
            else:
                raise CompanionGCError(
                    "gc_transaction_invalid",
                    "planned GC candidate is missing from both locations",
                )
            record["state"] = "moved"
            _write_json_atomic(root, transaction_path, transaction)
            _gc_fault_point("candidate_moved")

    discovery = _discover_gc(
        root,
        extra_roots=extra_roots,
        allow_active_build_lock=True,
        allow_gc_transaction=True,
        scan_candidates=False,
    )
    if discovery.root_snapshot_payload != transaction["root_snapshot_payload"]:
        raise CompanionGCError(
            "gc_publication_changed",
            "published roots changed after GC quarantine",
        )
    for record in records:
        if record["state"] == "moved":
            original = root / _safe_relative(record["original_path"])
            quarantine = root / _safe_relative(record["quarantine_path"])
            if original.exists() or original.is_symlink():
                raise CompanionGCError(
                    "gc_transaction_invalid",
                    "quarantined GC source still exists",
                )
            if (
                resuming_deletion
                and not quarantine.exists()
                and not quarantine.is_symlink()
            ):
                record["state"] = "deleted"
                _write_json_atomic(root, transaction_path, transaction)
                continue
            candidate = _candidate_from_record(
                record, path_key="quarantine_path",
            )
            _require_candidate_match(root, candidate)
    if not resuming_deletion:
        transaction["status"] = "quarantined"
        _write_json_atomic(root, transaction_path, transaction)
        _gc_fault_point("transaction_quarantined")

    transaction["status"] = "deleting"
    _write_json_atomic(root, transaction_path, transaction)
    for record in records:
        if record["state"] == "deleted":
            continue
        quarantine = root / _safe_relative(record["quarantine_path"])
        if quarantine.exists() or quarantine.is_symlink():
            candidate = _candidate_from_record(
                record, path_key="quarantine_path",
            )
            _require_candidate_match(root, candidate)
            _gc_fault_point("before_delete")
            if candidate.kind == "file":
                quarantine.unlink()
            else:
                _delete_tree(quarantine)
            _fsync_directory(quarantine.parent)
            _gc_fault_point("after_delete")
        elif record["state"] != "moved":
            raise CompanionGCError(
                "gc_transaction_invalid",
                "unmoved GC candidate disappeared",
            )
        record["state"] = "deleted"
        _write_json_atomic(root, transaction_path, transaction)
        _gc_fault_point("candidate_deleted")

    receipt = {
        "schema_version": GC_RECEIPT_VERSION,
        "recognizer_version": GC_RECOGNIZER_VERSION,
        "transaction_id": transaction_id,
        "project_identity_sha256": transaction["project_identity_sha256"],
        "root_snapshot_sha256": transaction["root_snapshot_sha256"],
        "candidate_set_sha256": transaction["candidate_set_sha256"],
        "status": "complete",
        "category_totals": transaction["category_totals"],
        "retained_class_totals": transaction["retained_class_totals"],
        "warnings": transaction["warnings"],
        "moved_count": len(records),
        "deleted_count": len(records),
        "reclaimed_bytes": sum(int(record["bytes"]) for record in records),
        "candidates": [
            {
                "category": record["category"],
                "path": record["original_path"],
                "kind": record["kind"],
                "bytes": record["bytes"],
                "sha256": record["sha256"],
                "recognizer_version": record["recognizer_version"],
            }
            for record in records
        ],
    }
    _create_or_adopt_json(root, receipt_path, receipt)
    receipt_sha256, _size = _hash_file(receipt_path)
    transaction.update({
        "status": "complete",
        "receipt_path": _relative(root, receipt_path),
        "receipt_sha256": receipt_sha256,
    })
    _write_json_atomic(root, transaction_path, transaction)
    _cleanup_trash(root, transaction_id)
    _gc_fault_point("transaction_complete")
    return _receipt_result(root, receipt_path, receipt)


def _require_candidate_match(root: Path, candidate: GCCandidate) -> None:
    path = root / _safe_relative(candidate.path)
    if candidate.kind == "file":
        actual = _file_candidate(root, path, candidate.category)
    else:
        actual = _directory_candidate(root, path, candidate.category)
    if actual.bytes != candidate.bytes or actual.sha256 != candidate.sha256:
        raise CompanionGCError(
            "gc_candidate_changed", "cleanup candidate identity changed",
        )


def _ensure_directory_chain(root: Path, parent: Path) -> None:
    try:
        relative = parent.relative_to(root)
    except ValueError as exc:
        raise CompanionGCError(
            "gc_project_unsafe", "GC transaction directory escapes the project",
        ) from exc
    current = root
    for component in relative.parts:
        current = current / component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            current.mkdir(mode=0o700)
            mode = current.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise CompanionGCError(
                "gc_project_unsafe", "GC transaction parent is not a directory",
            )


def _write_json_atomic(
    root: Path, path: Path, value: Mapping[str, Any],
) -> None:
    encoded = canonical_json(value).encode("utf-8") + b"\n"
    if len(encoded) > MAX_JSON_BYTES:
        raise CompanionGCError(
            "gc_transaction_invalid", "GC control JSON exceeds its size bound",
        )
    _ensure_directory_chain(root, path.parent)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent,
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _create_or_adopt_json(
    root: Path, path: Path, value: Mapping[str, Any],
) -> None:
    expected = canonical_json(value).encode("utf-8") + b"\n"
    if path.exists() or path.is_symlink():
        actual = _read_regular_bytes(
            root, path, max_bytes=MAX_JSON_BYTES,
            code="gc_transaction_invalid",
        )
        if actual != expected:
            raise CompanionGCError(
                "gc_transaction_invalid", "terminal GC receipt conflicts",
            )
        return
    _write_json_atomic(root, path, value)


def _read_terminal_receipt(
    root: Path, path: Path, *, transaction_id: str,
) -> Mapping[str, Any] | None:
    if not path.exists() and not path.is_symlink():
        return None
    value = _json_object(
        _read_regular_bytes(
            root, path, max_bytes=MAX_JSON_BYTES,
            code="gc_transaction_invalid",
        ),
        "gc_transaction_invalid",
    )
    expected_keys = {
        "schema_version",
        "recognizer_version",
        "transaction_id",
        "project_identity_sha256",
        "root_snapshot_sha256",
        "candidate_set_sha256",
        "status",
        "category_totals",
        "retained_class_totals",
        "warnings",
        "moved_count",
        "deleted_count",
        "reclaimed_bytes",
        "candidates",
    }
    if (
        set(value) != expected_keys
        or value.get("schema_version") != GC_RECEIPT_VERSION
        or value.get("recognizer_version") != GC_RECOGNIZER_VERSION
        or value.get("transaction_id") != transaction_id
        or value.get("status") != "complete"
    ):
        raise CompanionGCError(
            "gc_transaction_invalid", "terminal GC receipt is invalid",
        )
    raw_candidates = value.get("candidates")
    if not isinstance(raw_candidates, list):
        raise CompanionGCError(
            "gc_transaction_invalid", "terminal GC candidates are invalid",
        )
    candidates: list[GCCandidate] = []
    for raw in raw_candidates:
        if (
            not isinstance(raw, Mapping)
            or set(raw) != {
                "category",
                "path",
                "kind",
                "bytes",
                "sha256",
                "recognizer_version",
            }
        ):
            raise CompanionGCError(
                "gc_transaction_invalid", "terminal GC candidate is invalid",
            )
        candidate = _candidate_from_record(raw, path_key="path")
        _validate_candidate_binding(candidate)
        candidates.append(candidate)
    candidate_records = [candidate.as_dict() for candidate in candidates]
    category_totals = {
        category: {"count": count, "bytes": size}
        for category, count, size in _totals(candidates)
    }
    reclaimed = sum(candidate.bytes for candidate in candidates)
    expected_transaction_id = _sha_json({
        "schema_version": GC_TRANSACTION_VERSION,
        "recognizer_version": GC_RECOGNIZER_VERSION,
        "project_identity_sha256": value.get("project_identity_sha256"),
        "root_snapshot_sha256": value.get("root_snapshot_sha256"),
        "candidate_set_sha256": value.get("candidate_set_sha256"),
    })
    if (
        _sha_json(candidate_records) != value.get("candidate_set_sha256")
        or value.get("category_totals") != category_totals
        or value.get("moved_count") != len(candidates)
        or value.get("deleted_count") != len(candidates)
        or value.get("reclaimed_bytes") != reclaimed
        or expected_transaction_id != transaction_id
        or not _valid_totals(value.get("retained_class_totals"))
        or not _valid_warnings(value.get("warnings"))
    ):
        raise CompanionGCError(
            "gc_transaction_invalid", "terminal GC receipt identity differs",
        )
    return value


def _valid_totals(value: object) -> bool:
    return isinstance(value, Mapping) and all(
        isinstance(category, str)
        and re.fullmatch(r"[a-z][a-z0-9_]*", category) is not None
        and isinstance(record, Mapping)
        and set(record) == {"count", "bytes"}
        and isinstance(record["count"], int)
        and not isinstance(record["count"], bool)
        and record["count"] >= 0
        and isinstance(record["bytes"], int)
        and not isinstance(record["bytes"], bool)
        and record["bytes"] >= 0
        for category, record in value.items()
    )


def _valid_warnings(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) <= MAX_WARNING_CODES
        and all(
            isinstance(item, str)
            and re.fullmatch(r"gc_[a-z0-9_]+", item) is not None
            for item in value
        )
    )


def _receipt_result(
    root: Path, path: Path, receipt: Mapping[str, Any],
) -> dict[str, Any]:
    digest, _size = _hash_file(path)
    return {
        **dict(receipt),
        "receipt_path": _relative(root, path),
        "receipt_sha256": digest,
    }


def _delete_tree(path: Path) -> None:
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise CompanionGCError(
            "gc_candidate_changed", "quarantined tree type changed",
        )
    for child in _bounded_children(path):
        mode = child.lstat().st_mode
        if stat.S_ISDIR(mode) and not stat.S_ISLNK(mode):
            _delete_tree(child)
        elif stat.S_ISREG(mode) and not stat.S_ISLNK(mode):
            child.unlink()
        else:
            raise CompanionGCError(
                "gc_candidate_changed", "quarantined tree type changed",
            )
    path.rmdir()


def _validate_trash_shape(
    root: Path,
    transaction_id: str,
    records: Sequence[Mapping[str, Any]],
) -> None:
    transaction_root = root / ".arc-companion" / "gc-trash" / transaction_id
    if not transaction_root.exists() and not transaction_root.is_symlink():
        return
    _require_directory(root, transaction_root, code="gc_transaction_invalid")
    payload = transaction_root / "payload"
    if not payload.exists() and not payload.is_symlink():
        if _bounded_children(transaction_root):
            raise CompanionGCError(
                "gc_transaction_invalid", "GC trash contains an unknown entry",
            )
        return
    _require_directory(root, payload, code="gc_transaction_invalid")
    roots = [
        root / _safe_relative(str(record["quarantine_path"]))
        for record in records
    ]
    count = 0
    for path in payload.rglob("*"):
        count += 1
        if count > MAX_RECOGNIZED_ENTRIES:
            raise CompanionGCError(
                "gc_transaction_invalid", "GC trash exceeds its entry bound",
            )
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or (
            not stat.S_ISREG(mode) and not stat.S_ISDIR(mode)
        ):
            raise CompanionGCError(
                "gc_transaction_invalid", "GC trash contains a special entry",
            )
        if not any(
            path == candidate
            or path in candidate.parents
            or candidate in path.parents
            for candidate in roots
        ):
            raise CompanionGCError(
                "gc_transaction_invalid", "GC trash contains an unjournaled entry",
            )


def _cleanup_trash(root: Path, transaction_id: str) -> None:
    transaction_root = (
        root / ".arc-companion" / "gc-trash" / transaction_id
    )
    current = transaction_root / "payload"
    while current != transaction_root.parent:
        try:
            current.rmdir()
        except (FileNotFoundError, OSError):
            break
        _fsync_directory(current.parent)
        if current == transaction_root:
            break
        current = current.parent


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _gc_fault_point(_label: str) -> None:
    """No-op seam used by tests to inject deterministic transaction crashes."""
