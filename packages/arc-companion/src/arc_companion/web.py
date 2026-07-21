from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib import resources
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any
import unicodedata
from urllib.parse import urlparse

from .io import canonical_json, read_json, sha256_file, sha256_json, write_json, write_text
from .reader_text import clean_reader_annotation, clean_reader_translation
from .source import asset_path, block_id


READER_SNAPSHOT_VERSION = "arc.companion.reader-snapshot.v2"
READER_FINAL_VERSION = "arc.companion.reader-final.v2"
WEB_MANIFEST_VERSION = "arc.companion.web-manifest.v2"
WEB_RENDER_VERSION = "arc.companion.web-render.v2"
WEB_VALIDATION_VERSION = "arc.companion.web-validation.v2"

_STATE_VERSIONS = {
    "arc.companion.state.v1",
    "arc.companion.state.v2",
    "arc.companion.state.v3",
}
_LEDGER_VERSIONS = {
    "arc.companion.chapter-lane-ledger.v1",
    "arc.companion.chapter-lane-ledger.v2",
}
_SEGMENTATION_PREFIX = "arc.companion.segmentation."
_CHAPTERS_PREFIX = "arc.companion.chapters."
_GUIDE_PREFIX = "arc.companion.chapter-guide."
_ANNOTATION_PREFIX = "arc.companion.annotation-checkpoint."
_TRANSLATION_PREFIX = "arc.companion.translation-checkpoint."
_GLOSSARY_PREFIXES = (
    "arc.companion.glossary.",
    "arc.companion.index-glossary.",
)
_OPAQUE_INLINE_PATTERN = re.compile(r"\[\[ARC_INLINE:([^\]\s]+):([0-9a-f]{64})\]\]")
_MATH_TOKEN_PATTERN = re.compile(
    r"(?<!\\)(?P<display_dollar>\$\$(?:\\.|[^$])*?\$\$)"
    r"|(?P<display_bracket>\\\[(?:\\.|[^\\])*?\\\])"
    r"|(?P<inline_paren>\\\((?:\\.|[^\\])*?\\\))"
    r"|(?P<inline_dollar>\$(?:\\.|[^$\n])+?\$)",
    flags=re.DOTALL,
)
_WEB_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
_TERMINAL_READER_STATUSES = {"complete", "first_chapter_ready"}
_SVG_BLOCKED_ELEMENTS = "script|foreignObject|object|embed|iframe|style"


class WebReaderError(RuntimeError):
    """The project cannot be safely represented as a companion reader."""


def build_reader_snapshot(
    project_dir: Path,
    *,
    state: Mapping[str, Any] | None = None,
    final_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Discover the current accepted project prefix and build a browser-safe view.

    ``state.json`` is the only stable root entrypoint.  Segment payload names are
    derived from their IDs and chaptered payloads are exposed only after their
    lane ledger records an accepted value with the same output hash.
    """
    root = Path(project_dir).resolve()
    current_state = _state(root, state)
    checkpoint = _checkpoint(root, current_state)
    saved_overrides = _reader_final_overrides(checkpoint)
    _require_final_reader_payload(
        current_state,
        checkpoint=checkpoint,
        final_overrides=final_overrides,
    )
    overrides = _deep_merge(saved_overrides, final_overrides or {})
    trusted_overrides = deepcopy(saved_overrides)
    if _is_explicit_final_publish(current_state, final_overrides):
        trusted_overrides = deepcopy(overrides)

    document_envelope = _document_envelope(checkpoint, overrides)
    document = _document(document_envelope, overrides)
    metadata = _metadata(document_envelope, document, overrides)
    translation_mode = str(
        overrides.get("translation_mode")
        or current_state.get("translation_mode")
        or "pending"
    )
    # Same-language mode is authoritative.  In particular, never revive a
    # glossary left in an older checkpoint when rendering or resuming it.
    glossary = {} if translation_mode == "skipped" else _glossary(checkpoint, overrides)
    glossary_view = _glossary_view(glossary)
    chapters = _chapters(checkpoint, document, overrides)
    segments_by_chapter = _segments(checkpoint, chapters, overrides)
    guides = _guides(checkpoint, chapters, overrides)
    translations, annotations, lane_states = _lane_values(
        checkpoint,
        chapters=chapters,
        segments_by_chapter=segments_by_chapter,
        overrides=overrides,
        trusted_overrides=trusted_overrides,
    )
    if translation_mode == "skipped":
        translations = {}
        for state_value in lane_states.values():
            state_value.pop("translation", None)
    blocks_by_id = {
        block_id(item): dict(item)
        for item in document.get("blocks") or []
        if isinstance(item, Mapping) and block_id(dict(item))
    }
    entities = _entity_indexes(document)
    language = str(
        overrides.get("language")
        or current_state.get("annotation_language")
        or current_state.get("language")
        or ""
    )

    rendered_chapters: list[dict[str, Any]] = []
    ordered_segment_ids: list[str] = []
    translated_ids: list[str] = []
    annotated_ids: list[str] = []
    for chapter in chapters:
        chapter_id = str(chapter.get("chapter_id") or "")
        rendered_segments: list[dict[str, Any]] = []
        for segment in segments_by_chapter.get(chapter_id, []):
            segment_id = str(segment.get("segment_id") or "")
            if not segment_id:
                continue
            ordered_segment_ids.append(segment_id)
            source_blocks = [
                blocks_by_id[value]
                for value in segment.get("block_ids") or []
                if value in blocks_by_id
            ]
            translation = translations.get(segment_id)
            annotation = annotations.get(segment_id)
            if translation is not None:
                translated_ids.append(segment_id)
            if annotation is not None:
                annotated_ids.append(segment_id)
            rendered_segments.append(
                {
                    "segment_id": segment_id,
                    "title": str(segment.get("title") or ""),
                    "source": [
                        _source_block(item, entities=entities) for item in source_blocks
                    ],
                    "translation": _translation_view(
                        translation, source_blocks=source_blocks, entities=entities
                    ),
                    "companion": _annotation_view(annotation, language=language),
                    "lane_status": lane_states.get(segment_id, {}),
                }
            )
        rendered_chapters.append(
            {
                "chapter_id": chapter_id,
                "title": str(chapter.get("title") or ""),
                "page_start": _optional_int(chapter.get("page_start")),
                "page_end": _optional_int(chapter.get("page_end")),
                "guide": _guide_view(guides.get(chapter_id)),
                "segments": rendered_segments,
            }
        )

    appendices = _source_only_appendices(
        document,
        overrides=overrides,
        entities=entities,
        translation_mode=translation_mode,
    )
    if translation_mode != "skipped" and translation_mode == "pending" and translated_ids:
        translation_mode = "enabled"
    if translation_mode == "enabled" and glossary_view:
        _annotate_term_runs(rendered_chapters, glossary_view)
        _annotate_term_runs(appendices, glossary_view)

    title = str(
        overrides.get("title")
        or metadata.get("title")
        or (document.get("front_matter") or {}).get("title")
        or current_state.get("paper_id")
        or "Companion Reader"
    )
    snapshot: dict[str, Any] = {
        "schema_version": READER_SNAPSHOT_VERSION,
        "web_render_version": WEB_RENDER_VERSION,
        "status": str(overrides.get("status") or current_state.get("status") or "preparing"),
        "updated_at": str(current_state.get("updated_at") or ""),
        "paper_id": str(current_state.get("paper_id") or ""),
        "title": title,
        "authors": _author_names(metadata.get("authors") or []),
        "language": language,
        "translation_mode": translation_mode,
        "glossary": glossary_view,
        "chapters": rendered_chapters,
        "appendices": appendices,
        "coverage": {
            "chapter_ids": [str(item.get("chapter_id") or "") for item in chapters],
            "segment_ids": ordered_segment_ids,
            "translation_segment_ids": translated_ids,
            "annotation_segment_ids": annotated_ids,
        },
    }
    snapshot["revision"] = sha256_json(snapshot)
    return snapshot


def publish_reader(
    project_dir: Path,
    *,
    snapshot: Mapping[str, Any] | None = None,
    state: Mapping[str, Any] | None = None,
    final_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically publish a static ``file://`` reader, replacing index last.

    Every candidate file except ``index.html`` is immutable/content-addressed.
    The complete candidate is validated before the index commit, and a failed
    post-commit validation restores the previous index bytes.
    """
    root = Path(project_dir).resolve()
    current_state = _state(root, state)
    checkpoint = _checkpoint(root, current_state)
    # Keep the publish entrypoint safe even when a caller supplies a snapshot
    # built earlier while the workflow was still in preview state.
    _reader_final_overrides(checkpoint)
    _require_final_reader_payload(
        current_state,
        checkpoint=checkpoint,
        final_overrides=final_overrides,
    )
    value = deepcopy(
        dict(snapshot)
        if snapshot is not None
        else build_reader_snapshot(
            root, state=current_state, final_overrides=final_overrides
        )
    )
    if value.get("schema_version") != READER_SNAPSHOT_VERSION:
        raise WebReaderError("reader snapshot has an unsupported schema")

    reader_dir = root / "reader"
    data_dir = reader_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (reader_dir / "assets").mkdir(parents=True, exist_ok=True)
    index_path = reader_dir / "index.html"
    previous_index = index_path.read_bytes() if index_path.is_file() else None

    builtin_assets, builtin_root = _publish_builtin_assets(reader_dir)
    source_assets = _publish_source_assets(
        root,
        current_state,
        value,
        final_overrides=final_overrides,
    )
    value["revision"] = sha256_json({key: item for key, item in value.items() if key != "revision"})

    snapshot_bytes = _json_file_bytes(value)
    snapshot_hash = hashlib.sha256(snapshot_bytes).hexdigest()
    snapshot_path = data_dir / f"snapshot-{snapshot_hash}.json"
    _publish_fault_point("snapshot")
    write_json(snapshot_path, value)
    _assert_file_identity(snapshot_path, sha256=snapshot_hash, size=len(snapshot_bytes))
    data_text = "window.__ARC_COMPANION_SNAPSHOT__ = " + _safe_script_json(value) + ";\n"
    data_hash = hashlib.sha256(data_text.encode("utf-8")).hexdigest()
    data_name = f"snapshot-{data_hash}.js"
    data_path = data_dir / data_name
    _publish_fault_point("data-script")
    write_text(data_path, data_text)
    _assert_file_identity(
        data_path, sha256=data_hash, size=len(data_text.encode("utf-8"))
    )

    index_text = _index_html(
        data_script=f"data/{data_name}",
        asset_root=builtin_root,
        title=str(value.get("title") or "Companion Reader"),
    )
    index_hash = hashlib.sha256(index_text.encode("utf-8")).hexdigest()
    manifest = {
        "schema_version": WEB_MANIFEST_VERSION,
        "web_render_version": WEB_RENDER_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "snapshot": _file_record(snapshot_path, root=root),
        "data_script": _file_record(data_path, root=root),
        "index": {
            "path": index_path.relative_to(root).as_posix(),
            "sha256": index_hash,
            "bytes": len(index_text.encode("utf-8")),
        },
        "assets": [*builtin_assets, *source_assets],
        "coverage": deepcopy(value.get("coverage") or {}),
    }
    manifest_bytes = _json_file_bytes(manifest)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest_path = data_dir / f"manifest-{manifest_hash}.json"
    _publish_fault_point("manifest")
    write_json(manifest_path, manifest)
    _assert_file_identity(
        manifest_path, sha256=manifest_hash, size=len(manifest_bytes)
    )

    # Validate the exact on-disk candidate plus the in-memory index before the
    # only mutable path is touched.
    disk_manifest = _read_object(manifest_path, label="candidate web manifest")
    _validate_reader_bundle(
        root,
        index_path=index_path,
        snapshot_path=snapshot_path,
        manifest=disk_manifest,
        index_text=index_text,
    )

    # Publishing index last is the commit point: an open reader sees either the
    # previous complete bundle or this complete bundle, never half an update.
    published_state = {
            **current_state,
            "output_html": str(index_path),
            "output_html_sha256": index_hash,
            "reader_snapshot_path": str(snapshot_path),
            "reader_snapshot_sha256": snapshot_hash,
            "web_manifest_path": str(manifest_path),
            "web_manifest_sha256": manifest_hash,
            "web_render_version": WEB_RENDER_VERSION,
    }
    index_commit_attempted = False
    try:
        _publish_fault_point("index")
        index_commit_attempted = True
        write_text(index_path, index_text)
        _assert_file_identity(
            index_path,
            sha256=index_hash,
            size=len(index_text.encode("utf-8")),
        )
        _publish_fault_point("post-index-validation")
        report = validate_reader_project(root, state=published_state)
    except BaseException:
        if index_commit_attempted:
            _restore_index(index_path, previous_index)
        raise
    return {
        "output_html": str(index_path),
        "output_html_sha256": index_hash,
        "reader_snapshot_path": str(snapshot_path),
        "reader_snapshot_sha256": snapshot_hash,
        "web_manifest_path": str(manifest_path),
        "web_manifest_sha256": manifest_hash,
        "web_render_version": WEB_RENDER_VERSION,
        "web": report,
    }


def validate_reader_project(
    project_dir: Path,
    *,
    state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the committed web bundle without executing browser code."""
    root = Path(project_dir).resolve()
    current_state = _state(root, state)
    index_path = _state_output_path(root, current_state, "output_html", root / "reader" / "index.html")
    if current_state.get("web_manifest_path"):
        manifest_path = _state_output_path(
            root, current_state, "web_manifest_path", root / "reader" / "manifest.json"
        )
    else:
        manifest_path = _discover_manifest_for_index(root, index_path)
    manifest = _read_object(manifest_path, label="web manifest")
    if current_state.get("reader_snapshot_path"):
        snapshot_path = _state_output_path(
            root, current_state, "reader_snapshot_path", root / "reader" / "snapshot.json"
        )
    else:
        snapshot_record = manifest.get("snapshot")
        if not isinstance(snapshot_record, Mapping):
            raise WebReaderError("web manifest snapshot record is invalid")
        snapshot_path = _inside(root, Path(str(snapshot_record.get("path") or "")))
    for path in (index_path, snapshot_path, manifest_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise WebReaderError(f"reader artifact is missing or empty: {path}")

    index = index_path.read_text(encoding="utf-8")
    snapshot, segment_ids = _validate_reader_bundle(
        root,
        index_path=index_path,
        snapshot_path=snapshot_path,
        manifest=manifest,
        index_text=index,
    )

    for key, hash_key, path in (
        ("output_html", "output_html_sha256", index_path),
        ("reader_snapshot_path", "reader_snapshot_sha256", snapshot_path),
        ("web_manifest_path", "web_manifest_sha256", manifest_path),
    ):
        expected = str(current_state.get(hash_key) or "")
        if current_state.get(key) and expected and sha256_file(path) != expected:
            raise WebReaderError(f"state hash mismatch for {key}")
    return {
        "ok": True,
        "schema_version": WEB_VALIDATION_VERSION,
        "output_html": str(index_path),
        "snapshot_revision": snapshot["revision"],
        "chapter_count": len(snapshot.get("chapters") or []),
        "segment_count": len(segment_ids),
    }


def _validate_reader_bundle(
    root: Path,
    *,
    index_path: Path,
    snapshot_path: Path,
    manifest: Mapping[str, Any],
    index_text: str,
) -> tuple[dict[str, Any], list[str]]:
    """Validate one candidate without requiring its index to be committed."""
    snapshot = _read_object(snapshot_path, label="reader snapshot")
    if snapshot.get("schema_version") != READER_SNAPSHOT_VERSION:
        raise WebReaderError("reader snapshot schema is invalid")
    if manifest.get("schema_version") != WEB_MANIFEST_VERSION:
        raise WebReaderError("web manifest schema is invalid")
    if manifest.get("web_render_version") != WEB_RENDER_VERSION:
        raise WebReaderError("web manifest render version is stale")
    expected_revision = sha256_json(
        {key: item for key, item in snapshot.items() if key != "revision"}
    )
    if snapshot.get("revision") != expected_revision:
        raise WebReaderError("reader snapshot revision is invalid")

    assets = manifest.get("assets")
    if not isinstance(assets, list):
        raise WebReaderError("web manifest assets must be an array")
    for record in (manifest.get("snapshot"), manifest.get("data_script"), *assets):
        _validate_file_record(root, record)
    _validate_memory_file_record(
        root,
        manifest.get("index"),
        expected_path=index_path,
        content=index_text.encode("utf-8"),
    )
    if str((manifest.get("snapshot") or {}).get("path") or "") != snapshot_path.relative_to(root).as_posix():
        raise WebReaderError("web manifest points to a different snapshot")

    data_relative = str((manifest.get("data_script") or {}).get("path") or "")
    data_from_reader = _index_relative_path(root, index_path, data_relative)
    if data_from_reader not in index_text:
        raise WebReaderError("reader index does not reference the current data script")
    for required_suffix in (
        "/reader.css",
        "/reader.js",
        "/katex/katex.min.css",
        "/katex/katex.min.js",
    ):
        matching = [
            item for item in assets
            if isinstance(item, Mapping)
            and str(item.get("path") or "").endswith(required_suffix)
        ]
        if len(matching) != 1:
            raise WebReaderError(
                f"web manifest must contain exactly one {required_suffix.lstrip('/')}"
            )
        relative = _index_relative_path(
            root, index_path, str(matching[0].get("path") or "")
        )
        if relative not in index_text:
            raise WebReaderError(f"reader index does not reference {required_suffix}")

    coverage = snapshot.get("coverage") or {}
    if manifest.get("coverage") != coverage:
        raise WebReaderError("web manifest coverage differs from the reader snapshot")
    segment_ids = [
        str(segment.get("segment_id") or "")
        for chapter in snapshot.get("chapters") or []
        if isinstance(chapter, Mapping)
        for segment in chapter.get("segments") or []
        if isinstance(segment, Mapping)
    ]
    if segment_ids != list(coverage.get("segment_ids") or []):
        raise WebReaderError("reader chapter content differs from declared segment coverage")
    _validate_snapshot_terms(snapshot)
    return snapshot, segment_ids


def _validate_snapshot_terms(snapshot: Mapping[str, Any]) -> None:
    glossary = snapshot.get("glossary")
    if not isinstance(glossary, list):
        raise WebReaderError("reader glossary must be an array")
    mode = str(snapshot.get("translation_mode") or "")
    entries: dict[str, Mapping[str, Any]] = {}
    for item in glossary:
        if not isinstance(item, Mapping):
            raise WebReaderError("reader glossary contains an invalid entry")
        entry_id = str(item.get("entry_id") or "")
        if not entry_id or entry_id in entries:
            raise WebReaderError("reader glossary entry identities are invalid")
        entries[entry_id] = item
    term_runs = list(_walk_term_runs(snapshot.get("chapters") or []))
    term_runs.extend(_walk_term_runs(snapshot.get("appendices") or []))
    if mode == "skipped":
        if glossary:
            raise WebReaderError("skipped reader must not expose a glossary")
        if term_runs:
            raise WebReaderError("skipped reader must not expose term runs")
        return
    for run in term_runs:
        entry = entries.get(str(run.get("entry_id") or ""))
        if entry is None:
            raise WebReaderError("reader term run refers to an unknown glossary entry")
        source = str(entry.get("source") or "")
        target = str(entry.get("target") or "")
        if (
            not source or not target or _fold_term(source) == _fold_term(target)
            or run.get("source") != source or run.get("target") != target
        ):
            raise WebReaderError("reader term run refers to a non-bilingual glossary entry")


def _walk_term_runs(value: Any):
    if isinstance(value, list):
        for item in value:
            yield from _walk_term_runs(item)
    elif isinstance(value, Mapping):
        if value.get("type") == "term":
            yield value
        for item in value.values():
            yield from _walk_term_runs(item)


def _discover_manifest_for_index(root: Path, index_path: Path) -> Path:
    """Find the immutable manifest whose index record matches the commit."""
    if not index_path.is_file():
        raise WebReaderError(f"reader artifact is missing or empty: {index_path}")
    index_hash = sha256_file(index_path)
    index_size = index_path.stat().st_size
    legacy = root / "reader" / "manifest.json"
    candidates = ([legacy] if legacy.is_file() else []) + sorted(
        (root / "reader" / "data").glob("manifest-*.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    expected_path = index_path.relative_to(root).as_posix()
    for candidate in candidates:
        try:
            value = _read_object(candidate, label="web manifest candidate")
        except WebReaderError:
            continue
        record = value.get("index")
        if not isinstance(record, Mapping):
            continue
        if (
            str(record.get("path") or "") == expected_path
            and str(record.get("sha256") or "") == index_hash
            and record.get("bytes") == index_size
        ):
            return candidate.resolve()
    raise WebReaderError("no web manifest matches the committed reader index")


def _index_relative_path(root: Path, index_path: Path, value: str) -> str:
    path = _inside(root, Path(value))
    try:
        return path.relative_to(index_path.parent).as_posix()
    except ValueError as exc:
        raise WebReaderError("reader dependency is outside the reader directory") from exc


def _validate_memory_file_record(
    root: Path,
    value: Any,
    *,
    expected_path: Path,
    content: bytes,
) -> None:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256", "bytes"}:
        raise WebReaderError("web manifest contains an invalid file record")
    path = _inside(root, Path(str(value.get("path") or "")))
    if path != expected_path.resolve():
        raise WebReaderError("web manifest points to a different index")
    if str(value.get("sha256") or "") != hashlib.sha256(content).hexdigest():
        raise WebReaderError("web manifest index hash is invalid")
    if value.get("bytes") != len(content):
        raise WebReaderError("web manifest index byte size is invalid")


def _json_file_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n"
    ).encode("utf-8")


def _assert_file_identity(path: Path, *, sha256: str, size: int) -> None:
    if not path.is_file() or path.stat().st_size != size or sha256_file(path) != sha256:
        raise WebReaderError(f"candidate reader write changed unexpectedly: {path}")


def _publish_fault_point(_label: str) -> None:
    """No-op seam used by tests to inject a failure at every publish write."""


def _restore_index(index_path: Path, previous: bytes | None) -> None:
    if previous is None:
        index_path.unlink(missing_ok=True)
        return
    _atomic_write_bytes(index_path, previous)


def _state(root: Path, supplied: Mapping[str, Any] | None) -> dict[str, Any]:
    if supplied is not None:
        value = dict(supplied)
    else:
        path = root / "state.json"
        if not path.is_file():
            raise WebReaderError(f"No companion state found in {root}")
        value = _read_object(path, label="companion state")
    schema = value.get("schema_version")
    if schema is not None and schema not in _STATE_VERSIONS:
        raise WebReaderError("companion state schema is invalid")
    return value


def _checkpoint(root: Path, state: Mapping[str, Any]) -> Path | None:
    raw = state.get("checkpoint_dir")
    if not raw:
        return None
    path = _inside(root, Path(str(raw)))
    return path if path.is_dir() else None


def _reader_final_overrides(checkpoint: Path | None) -> dict[str, Any]:
    if checkpoint is None:
        return {}
    path = checkpoint / "reader-final.json"
    if not path.is_file():
        return {}
    value = _read_object(path, label="reader final checkpoint")
    if value.get("schema_version") != READER_FINAL_VERSION:
        raise WebReaderError("reader final checkpoint schema is invalid")
    overrides = value.get("final_overrides")
    if not isinstance(overrides, Mapping):
        raise WebReaderError("reader final checkpoint has no final_overrides object")
    return deepcopy(dict(overrides))


def _require_final_reader_payload(
    state: Mapping[str, Any],
    *,
    checkpoint: Path | None,
    final_overrides: Mapping[str, Any] | None,
) -> None:
    """Prevent accepted pre-review lanes from masquerading as final output."""
    status = str(state.get("status") or "").casefold()
    if status not in _TERMINAL_READER_STATUSES or final_overrides is not None:
        return
    if checkpoint is not None and (checkpoint / "reader-final.json").is_file():
        return
    raise WebReaderError(
        "terminal companion state requires final_overrides or a valid reader-final.json"
    )


def _is_explicit_final_publish(
    state: Mapping[str, Any], final_overrides: Mapping[str, Any] | None
) -> bool:
    """Recognize explicit final publication without trusting active previews."""
    if final_overrides is None:
        return False
    state_status = str(state.get("status") or "").casefold()
    override_status = str(final_overrides.get("status") or "").casefold()
    return (
        state_status in _TERMINAL_READER_STATUSES
        or (
            state_status == "typesetting"
            and override_status in _TERMINAL_READER_STATUSES
        )
    )


def _document_envelope(
    checkpoint: Path | None, overrides: Mapping[str, Any]
) -> dict[str, Any]:
    supplied = overrides.get("document_envelope")
    if isinstance(supplied, Mapping):
        return deepcopy(dict(supplied))
    if checkpoint is None or not (checkpoint / "document.json").is_file():
        return {}
    return _read_object(checkpoint / "document.json", label="source document checkpoint")


def _document(
    envelope: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    supplied = overrides.get("document")
    if isinstance(supplied, Mapping):
        return deepcopy(dict(supplied))
    nested = envelope.get("document")
    if isinstance(nested, Mapping):
        return deepcopy(dict(nested))
    return deepcopy(dict(envelope)) if isinstance(envelope.get("blocks"), list) else {}


def _metadata(
    envelope: Mapping[str, Any],
    document: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    supplied = overrides.get("metadata")
    if isinstance(supplied, Mapping):
        return deepcopy(dict(supplied))
    if isinstance(envelope.get("metadata"), Mapping):
        return deepcopy(dict(envelope["metadata"]))
    if isinstance(document.get("metadata"), Mapping):
        return deepcopy(dict(document["metadata"]))
    return {}


def _glossary(checkpoint: Path | None, overrides: Mapping[str, Any]) -> dict[str, Any]:
    supplied = overrides.get("glossary")
    if isinstance(supplied, Mapping):
        return deepcopy(dict(supplied))
    if checkpoint is None:
        return {}
    candidates: list[dict[str, Any]] = []
    for path in sorted(checkpoint.glob("*glossary.json")):
        try:
            value = _read_object(path, label="glossary checkpoint")
        except WebReaderError:
            continue
        if any(str(value.get("schema_version") or "").startswith(prefix) for prefix in _GLOSSARY_PREFIXES):
            candidates.append(value)
    return candidates[0] if candidates else {}


def _chapters(
    checkpoint: Path | None,
    document: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> list[dict[str, Any]]:
    supplied = overrides.get("chapters")
    if isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes)):
        normalized = [
            deepcopy(dict(item)) for item in supplied if isinstance(item, Mapping)
        ]
        if normalized or not overrides.get("segments"):
            return normalized
    if checkpoint is not None and (checkpoint / "chapters.json").is_file():
        value = _read_object(checkpoint / "chapters.json", label="chapter checkpoint")
        if not str(value.get("schema_version") or "").startswith(_CHAPTERS_PREFIX):
            raise WebReaderError("chapter checkpoint schema is invalid")
        return [dict(item) for item in value.get("chapters") or [] if isinstance(item, Mapping)]
    block_ids = [block_id(dict(item)) for item in document.get("blocks") or [] if isinstance(item, Mapping)]
    block_ids = [value for value in block_ids if value]
    return ([{
        "chapter_id": "ch-0001",
        "title": "",
        "block_ids": block_ids,
        "start_block_id": block_ids[0] if block_ids else "",
        "end_block_id": block_ids[-1] if block_ids else "",
    }] if block_ids else [])


def _segments(
    checkpoint: Path | None,
    chapters: Sequence[Mapping[str, Any]],
    overrides: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    supplied = overrides.get("segments")
    if isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes)):
        output: dict[str, list[dict[str, Any]]] = {str(item.get("chapter_id") or ""): [] for item in chapters}
        fallback = str(chapters[0].get("chapter_id") or "ch-0001") if chapters else "ch-0001"
        for raw in supplied:
            if not isinstance(raw, Mapping):
                continue
            item = dict(raw)
            chapter_id = str(item.get("chapter_id") or fallback)
            output.setdefault(chapter_id, []).append(item)
        return output

    output = {str(item.get("chapter_id") or ""): [] for item in chapters}
    if checkpoint is None:
        return output
    chaptered = (checkpoint / "chapters.json").is_file()
    if chaptered:
        for chapter in chapters:
            chapter_id = str(chapter.get("chapter_id") or "")
            path = checkpoint / "chapters" / chapter_id / "segmentation.json"
            if not path.is_file():
                continue
            value = _read_object(path, label=f"segmentation for {chapter_id}")
            if not str(value.get("schema_version") or "").startswith(_SEGMENTATION_PREFIX):
                raise WebReaderError(f"segmentation schema is invalid for {chapter_id}")
            normalized: list[dict[str, Any]] = []
            for index, raw in enumerate(value.get("segments") or [], 1):
                if not isinstance(raw, Mapping):
                    continue
                item = dict(raw)
                item["chapter_id"] = chapter_id
                item["segment_id"] = f"{chapter_id}.seg-{index:04d}"
                normalized.append(item)
            output[chapter_id] = normalized
        return output
    path = checkpoint / "segmentation.json"
    if not path.is_file():
        return output
    value = _read_object(path, label="segmentation checkpoint")
    if not str(value.get("schema_version") or "").startswith(_SEGMENTATION_PREFIX):
        raise WebReaderError("segmentation checkpoint schema is invalid")
    fallback = str(chapters[0].get("chapter_id") or "ch-0001") if chapters else "ch-0001"
    output.setdefault(fallback, []).extend(
        dict(item) for item in value.get("segments") or [] if isinstance(item, Mapping)
    )
    return output


def _guides(
    checkpoint: Path | None,
    chapters: Sequence[Mapping[str, Any]],
    overrides: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    supplied = overrides.get("chapter_guides") or overrides.get("guides")
    if isinstance(supplied, Mapping):
        return {
            str(key): deepcopy(dict(item))
            for key, item in supplied.items()
            if isinstance(item, Mapping)
        }
    output: dict[str, dict[str, Any]] = {}
    if checkpoint is None:
        return output
    for chapter in chapters:
        chapter_id = str(chapter.get("chapter_id") or "")
        path = checkpoint / "chapters" / chapter_id / "chapter-guide.json"
        if not path.is_file():
            continue
        value = _read_object(path, label=f"guide for {chapter_id}")
        if not str(value.get("schema_version") or "").startswith(_GUIDE_PREFIX):
            raise WebReaderError(f"guide schema is invalid for {chapter_id}")
        if str(value.get("chapter_id") or "") != chapter_id:
            raise WebReaderError(f"guide identity changed for {chapter_id}")
        output[chapter_id] = value
    return output


def _lane_values(
    checkpoint: Path | None,
    *,
    chapters: Sequence[Mapping[str, Any]],
    segments_by_chapter: Mapping[str, Sequence[Mapping[str, Any]]],
    overrides: Mapping[str, Any],
    trusted_overrides: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    final_translations = _mapping_of_objects(overrides.get("translations"))
    final_annotations = _mapping_of_objects(overrides.get("annotations"))
    trusted_translations = _mapping_of_objects(trusted_overrides.get("translations"))
    trusted_annotations = _mapping_of_objects(trusted_overrides.get("annotations"))
    states: dict[str, dict[str, str]] = {}
    all_ids = [
        str(item.get("segment_id") or "")
        for chapter in chapters
        for item in segments_by_chapter.get(str(chapter.get("chapter_id") or ""), [])
    ]
    if checkpoint is None:
        for segment_id in all_ids:
            states[segment_id] = {
                lane: (
                    "accepted"
                    if segment_id in target
                    and segment_id in trusted
                    and sha256_json(target[segment_id]) == sha256_json(trusted[segment_id])
                    else "preview" if segment_id in target else "unknown"
                )
                for lane, target, trusted in (
                    ("translation", final_translations, trusted_translations),
                    ("companion", final_annotations, trusted_annotations),
                )
            }
        return final_translations, final_annotations, states

    ledgers: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted((checkpoint / "chapters").glob("*/*-ledger.json")):
        value = _read_object(path, label="chapter lane ledger")
        if value.get("schema_version") not in _LEDGER_VERSIONS:
            continue
        chapter_id = str(value.get("chapter_id") or "")
        lane = str(value.get("lane") or "")
        if chapter_id and lane in {"translation", "companion"}:
            ledgers[(chapter_id, lane)] = value
    chaptered = bool(ledgers) or (checkpoint / "chapters.json").is_file()

    translations = dict(final_translations)
    annotations = dict(final_annotations)
    for chapter in chapters:
        chapter_id = str(chapter.get("chapter_id") or "")
        for lane, target, trusted, directory, prefix, field in (
            ("translation", translations, trusted_translations, "translations", _TRANSLATION_PREFIX, "translation"),
            ("companion", annotations, trusted_annotations, "annotations", _ANNOTATION_PREFIX, "annotation"),
        ):
            ledger = ledgers.get((chapter_id, lane))
            block_records = {
                str(item.get("segment_id") or ""): dict(item)
                for item in (ledger or {}).get("blocks") or []
                if isinstance(item, Mapping)
            }
            for segment in segments_by_chapter.get(chapter_id, []):
                segment_id = str(segment.get("segment_id") or "")
                state = str(block_records.get(segment_id, {}).get("state") or ("pending" if chaptered else "unknown"))
                states.setdefault(segment_id, {})[lane] = state
                supplied = target.get(segment_id)
                supplied_is_trusted = (
                    supplied is not None
                    and segment_id in trusted
                    and sha256_json(supplied) == sha256_json(trusted[segment_id])
                )
                if supplied_is_trusted:
                    states[segment_id][lane] = "accepted"
                    continue
                record = block_records.get(segment_id)
                if chaptered and (record is None or state != "accepted"):
                    if supplied is not None:
                        states[segment_id][lane] = "preview"
                    continue
                embedded = record.get(field) if isinstance(record, Mapping) else None
                if isinstance(embedded, Mapping):
                    candidate = dict(embedded)
                else:
                    path = checkpoint / directory / f"{_segment_checkpoint_name(segment_id)}.json"
                    if not path.is_file():
                        continue
                    envelope = _read_object(path, label=f"{lane} checkpoint for {segment_id}")
                    if not str(envelope.get("schema_version") or "").startswith(prefix):
                        continue
                    if str(envelope.get("segment_id") or "") != segment_id:
                        raise WebReaderError(f"{lane} checkpoint identity changed for {segment_id}")
                    candidate = envelope.get(field)
                    if not isinstance(candidate, Mapping):
                        continue
                    candidate = dict(candidate)
                expected = str((record or {}).get("output_sha256") or "")
                if expected and sha256_json(candidate) != expected:
                    raise WebReaderError(f"accepted {lane} hash mismatch for {segment_id}")
                # An active override is merely a preview until an accepted
                # checkpoint proves it.  If it differs, prefer the proven
                # checkpoint rather than labeling unreviewed data accepted.
                target[segment_id] = candidate
                states[segment_id][lane] = "accepted"
    for segment_id in all_ids:
        states.setdefault(segment_id, {})
    return translations, annotations, states


def _source_block(block: Mapping[str, Any], *, entities: Mapping[str, Mapping[str, dict[str, Any]]]) -> dict[str, Any]:
    value = dict(block)
    kind = str(value.get("type") or value.get("kind") or "text").casefold()
    output: dict[str, Any] = {
        "kind": kind,
        "title": str(value.get("title") or ""),
        "runs": _inline_runs(value),
    }
    if kind in {"equation", "math", "display_math"}:
        entity = _entity_for(value, entities.get("equations") or {})
        tex = (entity or value).get("tex")
        output["math"] = [
            {"type": "math", "tex": item, "display": True}
            for item in _tex_values(tex)
            if item
        ]
        output["number"] = str(
            (entity or value).get("number")
            or (entity or value).get("display_number")
            or (entity or value).get("tag")
            or ""
        )
    elif kind in {"figure", "image"}:
        entity = _entity_for(value, entities.get("figures") or {}) or value
        output["caption"] = str(entity.get("caption") or value.get("caption") or "")
        output["asset_ids"] = [
            str(item) for item in (entity.get("asset_ids") or ([entity.get("asset_id")] if entity.get("asset_id") else [])) if item
        ]
    elif kind == "table":
        entity = _entity_for(value, entities.get("tables") or {}) or value
        output["caption"] = str(entity.get("caption") or value.get("caption") or "")
        output["rows"] = _table_rows(entity)
    elif kind in {"list", "itemize", "enumerate", "ordered_list", "unordered_list"}:
        output["items"] = _list_items(value.get("items") or value.get("list_items") or [])
        output["ordered"] = kind in {"enumerate", "ordered_list"} or bool(value.get("ordered"))
    return output


def _translation_view(
    value: Mapping[str, Any] | None,
    *,
    source_blocks: Sequence[Mapping[str, Any]],
    entities: Mapping[str, Mapping[str, dict[str, Any]]],
) -> dict[str, Any] | None:
    if value is None:
        return None
    cleaned = clean_reader_translation(dict(value))
    translated = {
        str(item.get("block_id") or ""): item
        for item in cleaned.get("blocks") or []
        if isinstance(item, Mapping) and item.get("block_id")
    }
    blocks: list[dict[str, Any]] = []
    for source in source_blocks:
        source_id = block_id(dict(source))
        kind = str(source.get("type") or source.get("kind") or "text").casefold()
        if kind in {"equation", "math", "display_math"}:
            entity = _entity_for(source, entities.get("equations") or {}) or source
            blocks.append({"kind": "equation", "runs": [
                {"type": "math", "tex": _strip_equation_identity(item), "display": True}
                for item in _tex_values(entity.get("tex")) if item
            ]})
            continue
        item = translated.get(source_id)
        if not item or item.get("translate") is False:
            continue
        text = item.get("text")
        if text in {None, ""}:
            text = item.get("translated_text") or item.get("translation") or ""
        if text:
            blocks.append({"kind": kind, "runs": _translation_runs(str(text), source)})
    return {"blocks": blocks}


def _annotation_view(value: Mapping[str, Any] | None, *, language: str) -> dict[str, Any] | None:
    if value is None:
        return None
    annotation = clean_reader_annotation(dict(value), language=language)
    explanation = str(annotation.get("explanation") or "").strip()
    commentary = str(annotation.get("commentary") or "").strip()
    if explanation and commentary and explanation != commentary:
        prose = commentary if explanation in commentary else (explanation if commentary in explanation else f"{explanation}\n\n{commentary}")
    else:
        prose = explanation or commentary
    sections: list[dict[str, Any]] = []
    if prose:
        sections.append({
            "kind": "explanation",
            "runs": _text_math_runs(prose),
            "sources": _safe_sources(annotation.get("commentary_sources")),
        })
    for field in ("prior_work", "later_work"):
        claims = annotation.get(field)
        if not isinstance(claims, list) or not claims:
            continue
        rendered = []
        for claim in claims:
            if isinstance(claim, Mapping):
                rendered.append({
                    "runs": _text_math_runs(str(claim.get("text") or "")),
                    "sources": _safe_sources(claim.get("sources")),
                })
            else:
                rendered.append({"runs": _text_math_runs(str(claim)), "sources": []})
        sections.append({"kind": field, "claims": rendered})
    return {"sections": sections}


def _guide_view(value: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not value:
        return []
    fields = ("motivation", "main_content", "section_logic", "book_position", "prerequisites")
    return [
        {"kind": key, "runs": _text_math_runs(str(value.get(key) or ""))}
        for key in fields
        if str(value.get(key) or "").strip()
    ]


def _glossary_view(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for ordinal, item in enumerate(value.get("entries") or [], 1):
        if not isinstance(item, Mapping):
            continue
        source = str(item.get("source_term") or item.get("source") or item.get("term") or "").strip()
        target = str(item.get("target_term") or item.get("target") or item.get("translation") or "").strip()
        if not source:
            continue
        entry_id = str(item.get("entry_id") or f"term-{ordinal:04d}")
        if entry_id in used_ids:
            entry_id = f"term-{ordinal:04d}"
        used_ids.add(entry_id)
        lineage = item.get("parent_path") or item.get("lineage") or []
        aliases = item.get("source_aliases") or item.get("aliases") or []
        output.append({
            "entry_id": entry_id,
            "source": source,
            "target": target,
            "source_aliases": _unique_folded_strings(aliases, excluding=source),
            "explanation": str(item.get("explanation") or ""),
            "lineage": deepcopy(lineage) if isinstance(lineage, list) else [],
        })
    return output


def _source_only_appendices(
    document: Mapping[str, Any],
    *,
    overrides: Mapping[str, Any],
    entities: Mapping[str, Mapping[str, dict[str, Any]]],
    translation_mode: str,
) -> list[dict[str, Any]]:
    """Expose the source Index in same-language mode without lane placeholders."""
    if translation_mode != "skipped":
        return []
    supplied = overrides.get("source_only_appendices")
    if isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes)):
        output: list[dict[str, Any]] = []
        for ordinal, raw in enumerate(supplied, 1):
            if not isinstance(raw, Mapping):
                continue
            blocks = raw.get("blocks") or raw.get("source") or []
            output.append({
                "appendix_id": str(raw.get("appendix_id") or f"source-appendix-{ordinal:04d}"),
                "kind": str(raw.get("kind") or "source_only"),
                "title": str(raw.get("title") or "Index"),
                "source": [
                    _source_block(item, entities=entities)
                    for item in blocks if isinstance(item, Mapping)
                ],
            })
        return output
    blocks = [
        item for item in document.get("blocks") or []
        if isinstance(item, Mapping)
        and str(item.get("source_role") or item.get("role") or "").casefold() == "index"
    ]
    if not blocks:
        return []
    title = next((
        str(item.get("title") or item.get("text") or "").strip()
        for item in blocks
        if str(item.get("type") or item.get("kind") or "").casefold()
        in {"heading", "section", "chapter"}
    ), "") or "Index"
    return [{
        "appendix_id": "source-index",
        "kind": "source_only_index",
        "title": title,
        "source": [_source_block(item, entities=entities) for item in blocks],
    }]


def _unique_folded_strings(values: Any, *, excluding: str = "") -> list[str]:
    if not isinstance(values, list):
        return []
    excluded = _fold_term(excluding)
    seen = {excluded} if excluded else set()
    output: list[str] = []
    for value in values:
        text = str(value).strip()
        folded = _fold_term(text)
        if text and folded and folded not in seen:
            seen.add(folded)
            output.append(text)
    return output


def _annotate_term_runs(value: Any, glossary: Sequence[Mapping[str, Any]]) -> None:
    """Split ordinary text runs into deterministic, browser-safe term runs."""
    if isinstance(value, list):
        for item in value:
            _annotate_term_runs(item, glossary)
        return
    if not isinstance(value, dict):
        return
    runs = value.get("runs")
    if isinstance(runs, list):
        annotated: list[dict[str, Any]] = []
        for run in runs:
            if isinstance(run, Mapping) and run.get("type") == "text":
                annotated.extend(_term_runs(str(run.get("text") or ""), glossary))
            else:
                annotated.append(dict(run) if isinstance(run, Mapping) else run)
        value["runs"] = annotated
    for key, item in list(value.items()):
        if key != "runs":
            _annotate_term_runs(item, glossary)


def _term_runs(text: str, glossary: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    folded, spans = _fold_with_spans(text)
    if not folded:
        return ([{"type": "text", "text": text}] if text else [])
    candidates: list[tuple[int, int, int, int, Mapping[str, Any]]] = []
    for ordinal, entry in enumerate(glossary):
        source = str(entry.get("source") or "")
        target = str(entry.get("target") or "")
        if not source or not target or _fold_term(source) == _fold_term(target):
            continue
        terms = [(source, 0), *(
            (str(alias), 1) for alias in entry.get("source_aliases") or []
        ), (target, 0)]
        for term, alias_rank in terms:
            needle = _fold_term(term)
            if not needle:
                continue
            start = 0
            while (offset := folded.find(needle, start)) >= 0:
                end = offset + len(needle)
                before = folded[offset - 1] if offset else ""
                after = folded[end] if end < len(folded) else ""
                if (
                    (not _is_latin_or_decimal(needle[0]) or not _is_latin_or_decimal(before))
                    and (not _is_latin_or_decimal(needle[-1]) or not _is_latin_or_decimal(after))
                ):
                    original_start = spans[offset][0]
                    original_end = spans[end - 1][1]
                    candidates.append((
                        original_start, original_end, ordinal, alias_rank, entry
                    ))
                start = offset + 1
    # Resolve every overlap by longest match, glossary order, then canonical
    # over alias.  Re-sort the non-overlapping winners for source-order output.
    candidates.sort(key=lambda item: (-(item[1] - item[0]), item[2], item[3], item[0]))
    selected: list[tuple[int, int, Mapping[str, Any]]] = []
    for start, end, _ordinal, _alias_rank, entry in candidates:
        if any(start < chosen_end and end > chosen_start for chosen_start, chosen_end, _ in selected):
            continue
        selected.append((start, end, entry))
    selected.sort(key=lambda item: item[0])
    if not selected:
        return [{"type": "text", "text": text}] if text else []
    output: list[dict[str, Any]] = []
    cursor = 0
    for start, end, entry in selected:
        if start > cursor:
            output.append({"type": "text", "text": text[cursor:start]})
        output.append({
            "type": "term",
            "text": text[start:end],
            "entry_id": str(entry.get("entry_id") or ""),
            "source": str(entry.get("source") or ""),
            "target": str(entry.get("target") or ""),
        })
        cursor = end
    if cursor < len(text):
        output.append({"type": "text", "text": text[cursor:]})
    return output


def _fold_with_spans(text: str) -> tuple[str, list[tuple[int, int]]]:
    # Per-codepoint spans retain exact source slices for full-width/casefold
    # expansions (for example, ligatures).  NFKC is deterministic and mirrors
    # segment-glossary matching for ordinary source text.
    folded: list[str] = []
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        end = index + 1
        while end < len(text) and unicodedata.combining(text[end]):
            end += 1
        normalized = unicodedata.normalize("NFKC", text[index:end]).casefold()
        folded.extend(normalized)
        spans.extend([(index, end)] * len(normalized))
        index = end
    return "".join(folded), spans


def _fold_term(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _is_latin_or_decimal(value: str) -> bool:
    if not value:
        return False
    return unicodedata.category(value) == "Nd" or (
        unicodedata.category(value).startswith("L")
        and "LATIN" in unicodedata.name(value, "")
    )


def _inline_runs(block: Mapping[str, Any]) -> list[dict[str, Any]]:
    runs = block.get("inline_runs")
    if isinstance(runs, list) and runs:
        output: list[dict[str, Any]] = []
        for raw in runs:
            if not isinstance(raw, Mapping):
                continue
            kind = str(raw.get("kind") or "text").casefold()
            if kind == "math":
                output.append({"type": "math", "tex": str(raw.get("tex") or raw.get("content") or ""), "display": False})
            elif kind == "link":
                href = _safe_href(raw.get("href"))
                output.append({"type": "link" if href else "text", "text": str(raw.get("content") or href or ""), **({"href": href} if href else {})})
            else:
                output.extend(_text_math_runs(str(raw.get("content") or "")))
        return output
    return _text_math_runs(str(block.get("text") or block.get("title") or ""))


def _translation_runs(text: str, source: Mapping[str, Any]) -> list[dict[str, Any]]:
    owned = {
        (str(item.get("token_id") or ""), str(item.get("content_hash") or "")): item
        for item in source.get("inline_runs") or []
        if isinstance(item, Mapping) and str(item.get("kind") or "") != "text"
    }
    output: list[dict[str, Any]] = []
    cursor = 0
    for match in _OPAQUE_INLINE_PATTERN.finditer(text):
        output.extend(_text_math_runs(text[cursor:match.start()]))
        run = owned.get((match.group(1), match.group(2)))
        if run is None:
            output.append({"type": "text", "text": match.group(0)})
        elif str(run.get("kind") or "").casefold() == "math":
            output.append({"type": "math", "tex": _strip_equation_identity(str(run.get("tex") or run.get("content") or "")), "display": False})
        elif str(run.get("kind") or "").casefold() == "link":
            href = _safe_href(run.get("href"))
            output.append({"type": "link" if href else "text", "text": str(run.get("content") or href or ""), **({"href": href} if href else {})})
        else:
            output.append({"type": "text", "text": str(run.get("content") or "")})
        cursor = match.end()
    output.extend(_text_math_runs(text[cursor:]))
    return output


def _text_math_runs(text: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    cursor = 0
    for match in _MATH_TOKEN_PATTERN.finditer(str(text)):
        if match.start() > cursor:
            output.append({"type": "text", "text": text[cursor:match.start()]})
        token = match.group(0)
        display = token.startswith(("$$", r"\["))
        trim = 1 if token.startswith("$") and not token.startswith("$$") else 2
        output.append({"type": "math", "tex": token[trim:-trim], "display": display})
        cursor = match.end()
    if cursor < len(text):
        output.append({"type": "text", "text": text[cursor:]})
    return [item for item in output if item.get("type") != "text" or item.get("text")]


def _safe_sources(value: Any) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, Mapping):
            continue
        title = str(item.get("title") or "").strip()
        href = _safe_href(item.get("url"), external_only=True)
        locator = str(item.get("locator") or "").strip()
        if title and href and locator:
            output.append({"title": title, "url": href, "locator": locator})
    return output[:3]


def _publish_builtin_assets(reader_dir: Path) -> tuple[list[dict[str, Any]], str]:
    package_root = resources.files("arc_companion").joinpath("web_assets")
    payloads: list[tuple[Path, bytes]] = []
    digest = hashlib.sha256()
    for relative, item in _resource_files(package_root):
        with item.open("rb") as handle:
            data = handle.read()
        payloads.append((relative, data))
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(data).digest())
    asset_root = f"assets/builtin-{digest.hexdigest()}"
    output: list[dict[str, Any]] = []
    for relative, data in payloads:
        destination = reader_dir / asset_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        _publish_fault_point(f"builtin-asset:{relative.as_posix()}")
        _write_bytes(destination, data)
        output.append(_file_record(destination, root=reader_dir.parent))
    return sorted(output, key=lambda record: str(record["path"])), asset_root


def _resource_files(root: Any, prefix: Path = Path()) -> list[tuple[Path, Any]]:
    """Walk the importlib Traversable API without assuming pathlib methods."""
    output: list[tuple[Path, Any]] = []
    for item in sorted(root.iterdir(), key=lambda value: value.name):
        relative = prefix / item.name
        if item.is_dir():
            output.extend(_resource_files(item, relative))
        elif item.is_file():
            output.append((relative, item))
    return output


def _publish_source_assets(
    root: Path,
    state: Mapping[str, Any],
    snapshot: dict[str, Any],
    *,
    final_overrides: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    checkpoint = _checkpoint(root, state)
    overrides = _deep_merge(_reader_final_overrides(checkpoint), final_overrides or {})
    envelope = _document_envelope(checkpoint, overrides)
    document = _document(envelope, overrides)
    assets = {
        str(item.get("asset_id") or item.get("id") or ""): dict(item)
        for item in document.get("assets") or []
        if isinstance(item, Mapping) and (item.get("asset_id") or item.get("id"))
    }
    records: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, str]] = {}
    source_groups = [
        segment.get("source") or []
        for chapter in snapshot.get("chapters") or []
        for segment in chapter.get("segments") or []
    ]
    source_groups.extend(
        appendix.get("source") or []
        for appendix in snapshot.get("appendices") or []
    )
    for group in source_groups:
        for source in group:
            identifiers = list(source.pop("asset_ids", []) or [])
            rendered = []
            for identifier in identifiers:
                asset = assets.get(str(identifier))
                path = asset_path(asset or {}) if asset else None
                if path is None or not path.is_file() or path.suffix.casefold() not in _WEB_IMAGE_SUFFIXES:
                    continue
                source_digest = sha256_file(path)
                expected = str(asset.get("sha256") or "")
                if expected and expected != source_digest:
                    raise WebReaderError(f"source asset hash mismatch: {path}")
                cached = by_id.get(str(identifier))
                if cached is None:
                    suffix = path.suffix.casefold()
                    if suffix == ".svg":
                        data = _safe_svg(path.read_text(encoding="utf-8", errors="replace")).encode("utf-8")
                    else:
                        data = path.read_bytes()
                    digest = hashlib.sha256(data).hexdigest()
                    destination = root / "reader" / "assets" / "source" / f"{digest}{suffix}"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    _publish_fault_point(f"source-asset:{digest}{suffix}")
                    _write_bytes(destination, data)
                    cached = {
                        "url": destination.relative_to(root / "reader").as_posix(),
                        "sha256": sha256_file(destination),
                    }
                    by_id[str(identifier)] = cached
                    records.append(_file_record(destination, root=root))
                rendered.append(dict(cached))
            if rendered:
                source["assets"] = rendered
    return sorted(records, key=lambda record: str(record["path"]))


def _index_html(*, data_script: str, asset_root: str, title: str) -> str:
    escaped_title = _escape_html(title)
    escaped_assets = _escape_html(asset_root.rstrip("/"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>{escaped_title}</title>
  <link rel="stylesheet" href="{escaped_assets}/katex/katex.min.css">
  <link rel="stylesheet" href="{escaped_assets}/reader.css">
  <script defer src="{escaped_assets}/katex/katex.min.js"></script>
  <script defer src="{_escape_html(data_script)}"></script>
  <script defer src="{escaped_assets}/reader.js"></script>
</head>
<body>
  <button id="sidebar-toggle" class="sidebar-toggle" type="button" aria-controls="chapter-sidebar" aria-expanded="true">☰</button>
  <div id="reader-app" class="reader-shell">
    <aside id="chapter-sidebar" class="sidebar" aria-label="Chapter navigation"></aside>
    <main id="reader-main" class="reader-main" tabindex="-1"></main>
  </div>
  <noscript>This companion reader requires JavaScript to mount its local snapshot.</noscript>
</body>
</html>
"""


def _safe_script_json(value: Mapping[str, Any]) -> str:
    return canonical_json(value).replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def _safe_svg(value: str) -> str:
    paired = re.compile(
        rf"<\s*(?P<tag>{_SVG_BLOCKED_ELEMENTS})\b[^>]*>.*?"
        rf"<\s*/\s*(?P=tag)\s*>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    rendered = value
    while paired.search(rendered):
        rendered = paired.sub("", rendered)
    rendered = re.sub(
        rf"<\s*/?\s*(?:{_SVG_BLOCKED_ELEMENTS})\b[^>]*>",
        "",
        rendered,
        flags=re.IGNORECASE | re.DOTALL,
    )
    rendered = re.sub(r"\s+on[a-zA-Z]+\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+)", "", rendered)
    rendered = re.sub(
        r"\s+style\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+)",
        "",
        rendered,
        flags=re.IGNORECASE,
    )

    def safe_href(match: re.Match[str]) -> str:
        literal = match.group("value")
        unquoted = literal[1:-1] if literal[:1] in {'"', "'"} else literal
        return match.group(0) if re.fullmatch(r"\s*#[A-Za-z_][A-Za-z0-9_.:-]*\s*", unquoted) else ""

    rendered = re.sub(
        r"\s+(?:href|xlink:href)\s*=\s*(?P<value>\"[^\"]*\"|'[^']*'|[^\s>]+)",
        safe_href,
        rendered,
        flags=re.IGNORECASE,
    )

    def safe_url_attribute(match: re.Match[str]) -> str:
        literal = match.group("value")
        unquoted = literal[1:-1] if literal[:1] in {'"', "'"} else literal
        references = re.findall(
            r"url\(\s*(['\"]?)(.*?)\1\s*\)", unquoted, flags=re.IGNORECASE
        )
        if not references:
            return match.group(0)
        return (
            match.group(0)
            if all(re.fullmatch(r"#[A-Za-z0-9_.:-]+", value.strip()) for _, value in references)
            else ""
        )

    rendered = re.sub(
        r"\s+[A-Za-z_:][A-Za-z0-9_.:-]*\s*=\s*(?P<value>\"[^\"]*\"|'[^']*'|[^\s>]+)",
        safe_url_attribute,
        rendered,
        flags=re.IGNORECASE,
    )
    return rendered


def _inside(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise WebReaderError(f"path escapes companion project: {path}") from exc
    return resolved


def _state_output_path(root: Path, state: Mapping[str, Any], key: str, fallback: Path) -> Path:
    raw = state.get(key)
    return _inside(root, Path(str(raw))) if raw else fallback.resolve()


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = read_json(path)
    except (OSError, ValueError) as exc:
        raise WebReaderError(f"could not read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise WebReaderError(f"{label} is not an object: {path}")
    return value


def _segment_checkpoint_name(segment_id: str) -> str:
    return hashlib.sha256(segment_id.encode("utf-8")).hexdigest()


def _mapping_of_objects(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): deepcopy(dict(item))
        for key, item in value.items()
        if isinstance(item, Mapping)
    }


def _deep_merge(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    return {**deepcopy(dict(first)), **deepcopy(dict(second))}


def _entity_indexes(document: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for plural in ("equations", "figures", "tables"):
        indexed: dict[str, dict[str, Any]] = {}
        singular = plural[:-1]
        for item in document.get(plural) or []:
            if not isinstance(item, Mapping):
                continue
            identifier = str(item.get("id") or item.get(f"{singular}_id") or item.get("block_id") or "")
            if identifier:
                indexed[identifier] = dict(item)
        output[plural] = indexed
    return output


def _entity_for(block: Mapping[str, Any], values: Mapping[str, dict[str, Any]]) -> dict[str, Any] | None:
    for key in ("entity_id", "ref_id", "equation_id", "figure_id", "table_id", "id", "block_id"):
        identifier = str(block.get(key) or "")
        if identifier and identifier in values:
            return values[identifier]
    return None


def _tex_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)] if value not in {None, ""} else []


def _strip_equation_identity(value: str) -> str:
    rendered = re.sub(r"\\label\s*\{(?:[^{}]|\{[^{}]*\})*\}", "", value)
    rendered = re.sub(r"\\tag\*?\s*\{(?:[^{}]|\{[^{}]*\})*\}", "", rendered)
    return rendered.strip()


def _table_rows(entity: Mapping[str, Any]) -> list[list[str]]:
    output: list[list[str]] = []
    for raw_row in entity.get("rows") or []:
        cells = raw_row.get("cells") if isinstance(raw_row, Mapping) else raw_row
        if not isinstance(cells, list):
            continue
        output.append([
            str(cell.get("text") or cell.get("content") or "") if isinstance(cell, Mapping) else str(cell)
            for cell in cells
        ])
    return output


def _list_items(value: Any) -> list[str]:
    return [
        str(item.get("text") or item.get("content") or item.get("title") or "") if isinstance(item, Mapping) else str(item)
        for item in value if isinstance(value, list)
    ]


def _safe_href(value: Any, *, external_only: bool = False) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if not external_only and text.startswith("#") and re.fullmatch(r"#[A-Za-z0-9_.:-]+", text):
        return text
    parsed = urlparse(text)
    return text if parsed.scheme in {"http", "https"} and parsed.netloc else ""


def _author_names(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    output: list[str] = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, Mapping):
            name = str(item.get("name") or item.get("full_name") or "").strip()
        else:
            name = str(item).strip()
        if name:
            output.append(name)
    return output


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _escape_html(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _file_record(path: Path, *, root: Path) -> dict[str, Any]:
    resolved = _inside(root.resolve(), path)
    return {
        "path": resolved.relative_to(root.resolve()).as_posix(),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _validate_file_record(root: Path, value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256", "bytes"}:
        raise WebReaderError("web manifest contains an invalid file record")
    relative = Path(str(value.get("path") or ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise WebReaderError("web manifest contains an unsafe path")
    path = _inside(root, relative)
    if not path.is_file():
        raise WebReaderError(f"web manifest file is missing: {relative}")
    if sha256_file(path) != str(value.get("sha256") or ""):
        raise WebReaderError(f"web manifest hash mismatch: {relative}")
    if path.stat().st_size != int(value.get("bytes") or -1):
        raise WebReaderError(f"web manifest byte size mismatch: {relative}")


def _write_bytes(path: Path, value: bytes) -> None:
    _atomic_write_bytes(path, value)


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
