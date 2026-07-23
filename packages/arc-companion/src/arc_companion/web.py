from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import resources
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import threading
import time
from typing import Any
import unicodedata
from urllib.parse import urlparse
import weakref

from .io import canonical_json, read_json, sha256_file, sha256_json, write_json, write_text
from .reader_publish import (
    PreparedReaderCandidate,
    ReaderObject,
    ReaderPublishCoordinator,
    READER_PUBLISH_INTERVAL_SECONDS,
    READER_PUBLISH_STATE_VERSION,
    parse_reader_commit_utc,
)
from .reader_text import clean_reader_annotation, clean_reader_translation
from .source import asset_path, block_id
from .source_credit import (
    SourceCreditError,
    source_credit_placement,
    source_credit_visible_projection,
    validate_source_credit,
)


READER_SNAPSHOT_VERSION = "arc.companion.reader-snapshot.v4"
READER_FINAL_VERSION = "arc.companion.reader-final.v4"
_LEGACY_READER_FINAL_VERSION = "arc.companion.reader-final.v3"
WEB_MANIFEST_VERSION = "arc.companion.web-manifest.v3"
WEB_RENDER_VERSION = "arc.companion.web-render.v5"
WEB_VALIDATION_VERSION = "arc.companion.web-validation.v3"
_LEGACY_READER_SNAPSHOT_VERSION = "arc.companion.reader-snapshot.v3"
_LEGACY_WEB_MANIFEST_VERSION = "arc.companion.web-manifest.v2"
_LEGACY_WEB_RENDER_VERSION = "arc.companion.web-render.v4"
_READER_IDENTITY_FIELDS = (
    "output_html",
    "output_html_sha256",
    "reader_snapshot_path",
    "reader_snapshot_sha256",
    "web_manifest_path",
    "web_manifest_sha256",
    "web_render_version",
)

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


class ReaderDependencyMissing(WebReaderError):
    """A committed manifest dependency is missing but the pointer is intact."""


_READER_COMMIT_LOCKS_GUARD = threading.Lock()
_READER_COMMIT_LOCKS: weakref.WeakValueDictionary[
    Path, threading.RLock
] = weakref.WeakValueDictionary()


@dataclass(frozen=True)
class PreparedWebReaderPublish:
    """One immutable Reader candidate shared by digest and publisher."""

    root: Path
    semantic: PreparedReaderCandidate
    objects: tuple[ReaderObject, ...]
    index: ReaderObject
    outputs: Mapping[str, Any]
    prior_reader_state: Mapping[str, Any]


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
    saved_overrides = _reader_final_overrides(
        checkpoint, replacement_overrides=final_overrides,
    )
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
    paper_title_block_ids = _paper_title_block_ids(document)
    entities = _entity_indexes(document)
    language = _normalize_language_tag(str(
        overrides.get("language")
        or current_state.get("annotation_language")
        or current_state.get("language")
        or "und"
    ))
    content = overrides.get("content") if isinstance(overrides.get("content"), Mapping) else {}
    source_language = _normalize_language_tag(str(
        overrides.get("source_language")
        or content.get("source_language")
        or current_state.get("source_language")
        or "und"
    ))
    title_translations = _title_translation_index(
        overrides.get("title_translations") or content.get("title_translations")
    )
    checkpoint_credit = (
        _read_object(checkpoint / "source-credit.json", label="source credit")
        if (
            checkpoint is not None
            and (checkpoint / "source-credit.json").is_file()
        )
        else None
    )
    try:
        supplied_source_credit = (
            overrides.get("source_credit")
            or content.get("source_credit")
            or checkpoint_credit
        )
        if supplied_source_credit is None:
            raise SourceCreditError("canonical source credit is required")
        source_credit = validate_source_credit(supplied_source_credit)
    except (SourceCreditError, TypeError) as exc:
        raise WebReaderError("reader source credit is invalid") from exc
    source_credit_front_matter_block_ids = sorted({
        str(value)
        for key, values in (
            ((document.get("front_matter") or {}).get("block_ids") or {})
        ).items()
        if key in {"title", "authors", "affiliations"}
        if isinstance(values, list)
        for value in values
        if str(value)
    })
    source_credit_order = source_credit_placement(
        source_credit,
        front_matter_block_ids=source_credit_front_matter_block_ids,
    )
    source_credit_visible_projection = _source_credit_visible_projection(
        source_credit, source_credit_order,
    )
    source_credit_records = {
        str(item["id"]): item
        for key in ("authors", "affiliations", "profiles")
        for item in source_credit[key]
    }
    source_block_text = {
        block_id(dict(item)): str(item.get("text") or "").strip()
        for item in document.get("blocks") or []
        if isinstance(item, Mapping) and block_id(dict(item))
    }
    source_credit_replaced_block_ids = [
        str(item["block_id"])
        for item in source_credit_order
        if item["slot"] == "source_block"
        and item["block_id"]
        and source_block_text.get(str(item["block_id"])) == str(
            source_credit_records[str(item["id"])].get("source_name")
            or source_credit_records[str(item["id"])].get("text")
            or ""
        ).strip()
    ]
    ambiguous_source_credit_blocks = [
        str(item["block_id"])
        for item in source_credit_order
        if item["slot"] == "source_block"
        and str(item["block_id"]) not in source_credit_replaced_block_ids
    ]
    if ambiguous_source_credit_blocks:
        raise WebReaderError(
            "source-credit block anchor does not identify an equivalent "
            "standalone source block: "
            + ", ".join(ambiguous_source_credit_blocks)
        )

    rendered_chapters: list[dict[str, Any]] = []
    ordered_segment_ids: list[str] = []
    translated_ids: list[str] = []
    annotated_ids: list[str] = []
    for chapter in chapters:
        chapter_id = str(chapter.get("chapter_id") or "")
        chapter_title_block_ids = {
            str(value) for value in chapter.get("title_block_ids") or [] if str(value)
        }
        rendered_segments: list[dict[str, Any]] = []
        for segment in segments_by_chapter.get(chapter_id, []):
            segment_id = str(segment.get("segment_id") or "")
            if not segment_id:
                continue
            if not bool(segment.get("structural_only")):
                ordered_segment_ids.append(segment_id)
            source_blocks = [
                blocks_by_id[value]
                for value in segment.get("block_ids") or []
                if value in blocks_by_id
                and value not in chapter_title_block_ids
                and value not in paper_title_block_ids
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
                    "structural_only": bool(segment.get("structural_only")),
                    "source": [
                        _source_block(
                            item,
                            entities=entities,
                            translated_title=_translated_block_title(
                                item, title_translations
                            ),
                            language=source_language,
                        )
                        for item in source_blocks
                    ],
                    "translation": _translation_view(
                        translation, source_blocks=source_blocks, entities=entities
                    ),
                    "companion": _annotation_view(annotation, language=language),
                    "lane_status": lane_states.get(segment_id, {}),
                }
            )
        source_chapter_title = str(chapter.get("title") or "")
        translated_chapter_title = _translated_chapter_title(
            chapter, title_translations
        )
        rendered_chapters.append(
            {
                "chapter_id": chapter_id,
                "title": translated_chapter_title or source_chapter_title,
                "source_title": source_chapter_title,
                "translated_title": translated_chapter_title,
                "page_start": _optional_int(chapter.get("page_start")),
                "page_end": _optional_int(chapter.get("page_end")),
                "guide": _guide_view(guides.get(chapter_id)),
                "structural_only": bool(chapter.get("structural_only")),
                "segments": rendered_segments,
            }
        )

    appendices = _source_only_appendices(
        document,
        overrides=overrides,
        entities=entities,
        translation_mode=translation_mode,
        source_language=source_language,
    )
    if translation_mode != "skipped":
        appendices.extend(_orphan_structural_appendices(
            document,
            chapters=chapters,
            title_translations=title_translations,
        ))
    if translation_mode != "skipped" and translation_mode == "pending" and translated_ids:
        translation_mode = "enabled"
    if translation_mode == "enabled" and glossary_view:
        _annotate_term_runs(rendered_chapters, glossary_view)
        _annotate_term_runs(appendices, glossary_view)

    source_title = str(
        overrides.get("title")
        or metadata.get("title")
        or (document.get("front_matter") or {}).get("title")
        or current_state.get("paper_id")
        or "Companion Reader"
    )
    translated_title = _translated_document_title(
        document, title_translations
    )
    title = translated_title or source_title
    snapshot: dict[str, Any] = {
        "schema_version": READER_SNAPSHOT_VERSION,
        "web_render_version": WEB_RENDER_VERSION,
        "status": str(overrides.get("status") or current_state.get("status") or "preparing"),
        # Publication timing belongs to state/manifest, not Reader-visible
        # snapshot bytes.  Keeping it out makes semantic no-op repair exact.
        "updated_at": "",
        "paper_id": str(current_state.get("paper_id") or ""),
        "title": title,
        "source_title": source_title,
        "translated_title": translated_title,
        # Compatibility only. The DOM renderer consumes source_credit_order.
        "authors": [item["source_name"] for item in source_credit["authors"]],
        "source_credit": source_credit,
        "source_credit_sha256": source_credit["canonical_sha256"],
        "source_credit_order": source_credit_order,
        "source_credit_visible_projection": source_credit_visible_projection,
        "source_credit_front_matter_block_ids": (
            source_credit_front_matter_block_ids
        ),
        "source_credit_replaced_block_ids": source_credit_replaced_block_ids,
        "language": language,
        "source_language": source_language,
        "direction": _language_direction(language),
        "source_direction": _language_direction(source_language),
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
    if overrides.get("translation_reference") is not None:
        try:
            from .translation_reference import (
                TranslationReferenceError,
                validate_translation_reference_provenance,
            )
            snapshot["translation_reference"] = (
                validate_translation_reference_provenance(
                    overrides.get("translation_reference"),
                    project_root=root,
                    expected_chapter_ids=[
                        str(item.get("chapter_id") or "")
                        for item in chapters
                    ],
                )
            )
        except (TranslationReferenceError, TypeError, ValueError) as exc:
            raise WebReaderError("reader translation reference is invalid") from exc
    snapshot["revision"] = sha256_json(snapshot)
    return snapshot


def publish_reader(
    project_dir: Path,
    *,
    snapshot: Mapping[str, Any] | None = None,
    state: Mapping[str, Any] | None = None,
    final_overrides: Mapping[str, Any] | None = None,
    prepared: PreparedWebReaderPublish | None = None,
) -> dict[str, Any]:
    """Prepare and atomically publish one exact static Reader candidate."""
    candidate = prepared or prepare_reader_publish(
        project_dir, snapshot=snapshot, state=state,
        final_overrides=final_overrides,
    )
    if candidate.root != Path(project_dir).resolve():
        raise WebReaderError("prepared Reader candidate belongs to another project")
    return publish_prepared_reader(candidate)


def prepare_reader_publish(
    project_dir: Path,
    *,
    snapshot: Mapping[str, Any] | None = None,
    state: Mapping[str, Any] | None = None,
    final_overrides: Mapping[str, Any] | None = None,
    created_at: datetime | None = None,
) -> PreparedWebReaderPublish:
    """Build fixed candidate bytes and semantic records without writing."""
    root = Path(project_dir).resolve()
    current_state = _state(root, state)
    checkpoint = _checkpoint(root, current_state)
    _reader_final_overrides(
        checkpoint, replacement_overrides=final_overrides,
    )
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

    builtin_objects, builtin_root = _prepare_builtin_assets()
    source_objects = _prepare_source_assets(
        root,
        current_state,
        value,
        final_overrides=final_overrides,
    )
    value["revision"] = sha256_json(
        {key: item for key, item in value.items() if key != "revision"}
    )

    snapshot_bytes = _json_file_bytes(value)
    snapshot_hash = hashlib.sha256(snapshot_bytes).hexdigest()
    snapshot_relative = f"reader/data/snapshot-{snapshot_hash}.json"
    snapshot_object = ReaderObject(
        snapshot_relative, snapshot_bytes, "snapshot",
    )
    data_text = "window.__ARC_COMPANION_SNAPSHOT__ = " + _safe_script_json(value) + ";\n"
    data_bytes = data_text.encode("utf-8")
    data_hash = hashlib.sha256(data_bytes).hexdigest()
    data_name = f"snapshot-{data_hash}.js"
    data_object = ReaderObject(
        f"reader/data/{data_name}", data_bytes, "data-script",
    )

    index_text = _index_html(
        data_script=f"data/{data_name}",
        asset_root=builtin_root,
        title=str(value.get("title") or "Companion Reader"),
        language=str(value.get("language") or "und"),
    )
    index_bytes = index_text.encode("utf-8")
    index_object = ReaderObject("reader/index.html", index_bytes, "index")
    semantic = PreparedReaderCandidate(
        snapshot=value,
        web_render_version=WEB_RENDER_VERSION,
        builtin_objects=tuple(builtin_objects),
        source_objects=tuple(source_objects),
    )
    manifest = {
        "schema_version": WEB_MANIFEST_VERSION,
        "web_render_version": WEB_RENDER_VERSION,
        "reader_semantic_sha256": semantic.semantic_sha256,
        "created_at": _aware_utc(
            created_at or datetime.now(timezone.utc)
        ).isoformat(),
        "snapshot": _object_record(snapshot_object),
        "data_script": _object_record(data_object),
        "index": _object_record(index_object),
        "assets": [
            *(_object_record(item) for item in builtin_objects),
            *(_object_record(item) for item in source_objects),
        ],
        "coverage": deepcopy(value.get("coverage") or {}),
        "source_credit": _source_credit_manifest(value),
        **(
            {"translation_reference": deepcopy(value["translation_reference"])}
            if value.get("translation_reference") is not None else {}
        ),
    }
    manifest_bytes = _json_file_bytes(manifest)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest_object = ReaderObject(
        f"reader/data/manifest-{manifest_hash}.json",
        manifest_bytes,
        "manifest",
    )
    outputs = {
        "output_html": str(root / index_object.relative_path),
        "output_html_sha256": index_object.sha256,
        "reader_snapshot_path": str(root / snapshot_object.relative_path),
        "reader_snapshot_sha256": snapshot_object.sha256,
        "web_manifest_path": str(root / manifest_object.relative_path),
        "web_manifest_sha256": manifest_object.sha256,
        "web_render_version": WEB_RENDER_VERSION,
        "source_credit_sha256": value["source_credit_sha256"],
        "source_credit_observation_sha256": sha256_json(
            value["source_credit_visible_projection"]
        ),
    }
    return PreparedWebReaderPublish(
        root=root,
        semantic=semantic,
        objects=(
            *builtin_objects,
            *source_objects,
            snapshot_object,
            data_object,
            manifest_object,
        ),
        index=index_object,
        outputs=outputs,
        prior_reader_state=_reader_identity_state(current_state),
    )


def publish_prepared_reader(
    candidate: PreparedWebReaderPublish,
) -> dict[str, Any]:
    with _reader_commit_lock(candidate.root):
        return _publish_prepared_reader_locked(candidate)


def _publish_prepared_reader_locked(
    candidate: PreparedWebReaderPublish,
) -> dict[str, Any]:
    """Publish immutable objects, validate, then switch the sole mutable index."""
    root = candidate.root
    index_path = root / candidate.index.relative_path
    previous_index: bytes | None = None
    if index_path.exists() or index_path.is_symlink():
        # The old pointer is authoritative for adoption.  Never overwrite an
        # invalid or unexplained pointer.
        try:
            inspect_reader_publish(root)
        except WebReaderError as strict_error:
            try:
                durable_identity = _reader_identity_state(_state(root, None))
                if durable_identity != dict(candidate.prior_reader_state):
                    raise WebReaderError(
                        "prepared legacy Reader identity is no longer current"
                    )
                _inspect_state_bound_legacy_reader(
                    root,
                    durable_identity,
                    expected_index_path=index_path,
                )
            except WebReaderError:
                if not isinstance(strict_error, ReaderDependencyMissing):
                    raise strict_error
                if index_path.is_symlink() or not index_path.is_file():
                    raise strict_error
                previous_index = index_path.read_bytes()
                if previous_index != candidate.index.data:
                    raise strict_error
            else:
                previous_index = index_path.read_bytes()
        else:
            previous_index = index_path.read_bytes()

    for item in candidate.objects:
        label = item.kind
        if item.kind == "builtin-asset":
            marker = item.relative_path.split("/builtin-", 1)[-1]
            label = "builtin-asset:" + marker.split("/", 1)[-1]
        elif item.kind == "source-asset":
            label = "source-asset:" + Path(item.relative_path).name
        _publish_fault_point(label)
        _create_or_adopt_immutable(
            root, root / item.relative_path, item.data,
        )

    manifest_path = Path(str(candidate.outputs["web_manifest_path"]))
    snapshot_path = Path(str(candidate.outputs["reader_snapshot_path"]))
    disk_manifest = _read_object(
        manifest_path, label="candidate web manifest",
    )
    _validate_reader_bundle(
        root,
        index_path=index_path,
        snapshot_path=snapshot_path,
        manifest=disk_manifest,
        index_text=candidate.index.data.decode("utf-8"),
    )

    published_state = {
        **dict(candidate.outputs),
    }
    index_commit_attempted = False
    try:
        _publish_fault_point("index")
        index_commit_attempted = True
        write_text(index_path, candidate.index.data.decode("utf-8"))
        _assert_file_identity(
            index_path,
            sha256=candidate.index.sha256,
            size=len(candidate.index.data),
        )
        _publish_fault_point("post-index-validation")
        report = validate_reader_project(root, state=published_state)
    except BaseException:
        if index_commit_attempted:
            _restore_index(index_path, previous_index)
        raise
    return {
        **dict(candidate.outputs),
        "web": report,
    }


def _reader_commit_lock(root: Path) -> threading.RLock:
    resolved = root.resolve()
    with _READER_COMMIT_LOCKS_GUARD:
        return _READER_COMMIT_LOCKS.setdefault(
            resolved, threading.RLock(),
        )


def inspect_reader_publish(project_dir: Path) -> dict[str, Any] | None:
    """Validate and describe the bundle committed by the actual index."""
    root = Path(project_dir).resolve()
    index_path = root / "reader" / "index.html"
    if not index_path.exists() and not index_path.is_symlink():
        return None
    if index_path.is_symlink() or not index_path.is_file():
        raise WebReaderError("committed reader index is not a regular file")
    manifest_path = _discover_manifest_for_index(root, index_path)
    manifest = _read_object(manifest_path, label="committed web manifest")
    snapshot_record = manifest.get("snapshot")
    if not isinstance(snapshot_record, Mapping):
        raise WebReaderError("web manifest snapshot record is invalid")
    snapshot_path = _inside(root, Path(str(snapshot_record.get("path") or "")))
    if not snapshot_path.exists():
        raise ReaderDependencyMissing(
            f"web manifest file is missing: {snapshot_record.get('path')}"
        )
    index_text = index_path.read_text(encoding="utf-8")
    snapshot, _ = _validate_reader_bundle(
        root,
        index_path=index_path,
        snapshot_path=snapshot_path,
        manifest=manifest,
        index_text=index_text,
    )
    builtin: list[ReaderObject] = []
    sources: list[ReaderObject] = []
    for record in manifest.get("assets") or []:
        if not isinstance(record, Mapping):
            continue
        relative = str(record.get("path") or "")
        kind = (
            "builtin-asset"
            if "/builtin-" in relative
            else "source-asset"
        )
        item = ReaderObject(
            relative,
            _inside(root, Path(relative)).read_bytes(),
            kind,
        )
        (builtin if kind == "builtin-asset" else sources).append(item)
    semantic = PreparedReaderCandidate(
        snapshot=snapshot,
        web_render_version=str(manifest.get("web_render_version") or ""),
        builtin_objects=tuple(builtin),
        source_objects=tuple(sources),
    ).semantic_sha256
    declared_semantic = str(
        manifest.get("reader_semantic_sha256") or ""
    )
    if declared_semantic and declared_semantic != semantic:
        raise WebReaderError(
            "web manifest Reader semantic digest is invalid"
        )
    return {
        "output_html": str(index_path),
        "output_html_sha256": sha256_file(index_path),
        "reader_snapshot_path": str(snapshot_path),
        "reader_snapshot_sha256": sha256_file(snapshot_path),
        "web_manifest_path": str(manifest_path),
        "web_manifest_sha256": sha256_file(manifest_path),
        "web_render_version": str(manifest.get("web_render_version") or ""),
        "source_credit_sha256": snapshot["source_credit_sha256"],
        "source_credit_observation_sha256": sha256_json(
            snapshot["source_credit_visible_projection"]
        ),
        "reader_committed_semantic_sha256": semantic,
    }


def _reader_identity_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Capture one complete Reader identity and reject split state selection."""
    direct = {
        key: deepcopy(state[key])
        for key in _READER_IDENTITY_FIELDS
        if state.get(key) not in (None, "")
    }
    published = state.get("published")
    web = published.get("web") if isinstance(published, Mapping) else None
    published_identity = {
        key: deepcopy(web[key])
        for key in _READER_IDENTITY_FIELDS
        if isinstance(web, Mapping) and web.get(key) not in (None, "")
    }
    required = set(_READER_IDENTITY_FIELDS)
    if direct and set(direct) != required:
        raise WebReaderError("top-level Reader identity is incomplete")
    if published_identity and set(published_identity) != required:
        raise WebReaderError("published Reader identity is incomplete")
    if direct and published_identity and published_identity != direct:
        raise WebReaderError(
            "published Reader identity disagrees with top-level state"
        )
    return published_identity or direct


def _inspect_state_bound_legacy_reader(
    root: Path,
    state: Mapping[str, Any],
    *,
    expected_index_path: Path | None = None,
) -> dict[str, Any]:
    """Validate only the immediately preceding, exactly state-bound bundle."""
    if set(state) != set(_READER_IDENTITY_FIELDS):
        raise WebReaderError(
            "legacy Reader upgrade requires a complete state binding"
        )
    if state.get("web_render_version") != _LEGACY_WEB_RENDER_VERSION:
        raise WebReaderError("state-bound Reader is not a supported legacy bundle")

    paths: dict[str, Path] = {}
    for key, hash_key in (
        ("output_html", "output_html_sha256"),
        ("reader_snapshot_path", "reader_snapshot_sha256"),
        ("web_manifest_path", "web_manifest_sha256"),
    ):
        raw = Path(str(state[key]))
        unresolved = raw if raw.is_absolute() else root / raw
        _reject_symlink_components(root, unresolved)
        path = _inside(root, raw)
        try:
            mode = unresolved.lstat().st_mode
        except FileNotFoundError as exc:
            raise WebReaderError(
                f"state-bound legacy Reader artifact is missing: {path}"
            ) from exc
        if (
            not stat.S_ISREG(mode)
            or path.stat().st_size == 0
            or sha256_file(path) != str(state[hash_key])
        ):
            raise WebReaderError(
                f"state-bound legacy Reader identity is invalid: {key}"
            )
        paths[key] = path

    index_path = paths["output_html"]
    required_index = (
        expected_index_path.resolve()
        if expected_index_path is not None
        else (root / "reader" / "index.html").resolve()
    )
    if index_path != required_index:
        raise WebReaderError(
            "state-bound legacy Reader does not select the committed index"
        )
    manifest_path = paths["web_manifest_path"]
    snapshot_path = paths["reader_snapshot_path"]
    manifest = _read_object(
        manifest_path, label="state-bound legacy web manifest",
    )
    index_text = index_path.read_text(encoding="utf-8")
    _validate_legacy_reader_bundle(
        root,
        index_path=index_path,
        snapshot_path=snapshot_path,
        manifest=manifest,
        index_text=index_text,
    )
    return dict(state)


def create_reader_publish_coordinator(
    project_dir: Path,
    *,
    state_loader: Any,
    state_merger: Any,
    utc_now: Any = lambda: datetime.now(timezone.utc),
    monotonic: Any = time.monotonic,
    interval_seconds: float = READER_PUBLISH_INTERVAL_SECONDS,
    lock: threading.RLock | None = None,
    prepared_publisher: Any = None,
    prepare_state: Mapping[str, Any] | None = None,
) -> ReaderPublishCoordinator:
    """Inspect/adopt the actual index and create one injectable coordinator."""
    root = Path(project_dir).resolve()
    state = dict(state_loader())
    try:
        inspected = inspect_reader_publish(root)
    except WebReaderError as strict_error:
        try:
            _inspect_state_bound_legacy_reader(
                root, _reader_identity_state(state),
            )
        except WebReaderError:
            if not isinstance(strict_error, ReaderDependencyMissing):
                raise strict_error
            # A matching prepared index may repair missing immutable
            # dependencies; publish_prepared_reader still refuses an
            # unexplained pointer.
            inspected = None
        else:
            inspected = None
    if inspected is None:
        state = dict(state_merger({
            "reader_publish_state_version": READER_PUBLISH_STATE_VERSION,
            "reader_dirty": True,
            "reader_committed_semantic_sha256": "",
            "reader_committed_at": "",
        }))
    else:
        repairs = {
            key: value
            for key, value in inspected.items()
            if state.get(key) != value
        }
        if (
            state.get("reader_publish_state_version")
            != READER_PUBLISH_STATE_VERSION
        ):
            repairs["reader_publish_state_version"] = (
                READER_PUBLISH_STATE_VERSION
            )
        if (
            repairs
            or parse_reader_commit_utc(
                state.get("reader_committed_at")
            ) is None
        ):
            repairs["reader_committed_at"] = _aware_utc(
                utc_now()
            ).isoformat()
        if repairs:
            state = dict(state_merger(repairs))

    def prepare(
        overrides: Mapping[str, Any] | None,
    ) -> PreparedReaderCandidate:
        web_candidate = prepare_reader_publish(
            root,
            state={**dict(state_loader()), **dict(prepare_state or {})},
            final_overrides=overrides,
            created_at=_aware_utc(utc_now()),
        )
        semantic = web_candidate.semantic
        return PreparedReaderCandidate(
            snapshot=semantic.snapshot,
            web_render_version=semantic.web_render_version,
            builtin_objects=semantic.builtin_objects,
            source_objects=semantic.source_objects,
            payload=web_candidate,
        )

    def publish(candidate: PreparedReaderCandidate) -> Mapping[str, Any]:
        if not isinstance(candidate.payload, PreparedWebReaderPublish):
            raise RuntimeError(
                "Reader coordinator lost its prepared candidate"
            )
        active_publisher = (
            prepared_publisher or publish_prepared_reader
        )
        return active_publisher(candidate.payload)

    return ReaderPublishCoordinator(
        state_loader=state_loader,
        state_merger=state_merger,
        preparer=prepare,
        publisher=publish,
        utc_now=utc_now,
        monotonic=monotonic,
        interval_seconds=interval_seconds,
        lock=lock,
    )


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
        "source_credit_sha256": snapshot["source_credit_sha256"],
        "source_credit_observation_sha256": sha256_json(
            snapshot["source_credit_visible_projection"]
        ),
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
    return _validate_reader_bundle_contract(
        root,
        index_path=index_path,
        snapshot_path=snapshot_path,
        manifest=manifest,
        index_text=index_text,
        legacy=False,
    )


def _validate_legacy_reader_bundle(
    root: Path,
    *,
    index_path: Path,
    snapshot_path: Path,
    manifest: Mapping[str, Any],
    index_text: str,
) -> tuple[dict[str, Any], list[str]]:
    """Validate the exact historical contract admitted only for replacement."""
    return _validate_reader_bundle_contract(
        root,
        index_path=index_path,
        snapshot_path=snapshot_path,
        manifest=manifest,
        index_text=index_text,
        legacy=True,
    )


def _validate_reader_bundle_contract(
    root: Path,
    *,
    index_path: Path,
    snapshot_path: Path,
    manifest: Mapping[str, Any],
    index_text: str,
    legacy: bool,
) -> tuple[dict[str, Any], list[str]]:
    snapshot = _read_object(snapshot_path, label="reader snapshot")
    snapshot_version = (
        _LEGACY_READER_SNAPSHOT_VERSION if legacy else READER_SNAPSHOT_VERSION
    )
    manifest_version = (
        _LEGACY_WEB_MANIFEST_VERSION if legacy else WEB_MANIFEST_VERSION
    )
    render_version = (
        _LEGACY_WEB_RENDER_VERSION if legacy else WEB_RENDER_VERSION
    )
    if snapshot.get("schema_version") != snapshot_version:
        raise WebReaderError("reader snapshot schema is invalid")
    if (
        legacy
        and snapshot.get("web_render_version")
        != _LEGACY_WEB_RENDER_VERSION
    ):
        raise WebReaderError("reader snapshot render version is stale")
    if manifest.get("schema_version") != manifest_version:
        raise WebReaderError("web manifest schema is invalid")
    if manifest.get("web_render_version") != render_version:
        raise WebReaderError("web manifest render version is stale")
    expected_revision = sha256_json(
        {key: item for key, item in snapshot.items() if key != "revision"}
    )
    if snapshot.get("revision") != expected_revision:
        raise WebReaderError("reader snapshot revision is invalid")
    if not legacy:
        _validate_current_reader_metadata(root, snapshot, manifest)

    assets = manifest.get("assets")
    if not isinstance(assets, list):
        raise WebReaderError("web manifest assets must be an array")
    for record in (manifest.get("snapshot"), manifest.get("data_script"), *assets):
        _validate_file_record(root, record)
    if not legacy:
        _validate_source_asset_bindings(root, snapshot, assets)
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
        if isinstance(segment, Mapping) and not bool(segment.get("structural_only"))
    ]
    if segment_ids != list(coverage.get("segment_ids") or []):
        raise WebReaderError("reader chapter content differs from declared segment coverage")
    _validate_snapshot_terms(snapshot)
    return snapshot, segment_ids


def _validate_current_reader_metadata(
    root: Path,
    snapshot: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    try:
        source_credit = validate_source_credit(snapshot.get("source_credit"))
    except (SourceCreditError, TypeError) as exc:
        raise WebReaderError("reader source credit is invalid") from exc
    if snapshot.get("source_credit_sha256") != source_credit["canonical_sha256"]:
        raise WebReaderError("reader source-credit hash is invalid")
    expected_credit_manifest = _source_credit_manifest(snapshot)
    if manifest.get("source_credit") != expected_credit_manifest:
        raise WebReaderError(
            "web manifest source credit differs from the reader snapshot"
        )
    front_ids = snapshot.get("source_credit_front_matter_block_ids")
    if not isinstance(front_ids, list) or not all(
        isinstance(item, str) for item in front_ids
    ):
        raise WebReaderError(
            "reader source-credit front-matter placement is invalid"
        )
    expected_order = source_credit_placement(
        source_credit, front_matter_block_ids=front_ids,
    )
    if snapshot.get("source_credit_order") != expected_order:
        raise WebReaderError("reader source-credit order is invalid")
    expected_visible = _source_credit_visible_projection(
        source_credit, expected_order,
    )
    if snapshot.get("source_credit_visible_projection") != expected_visible:
        raise WebReaderError(
            "reader source-credit visible projection is invalid"
        )
    expected_replaced = [
        str(item["block_id"])
        for item in expected_order
        if item["slot"] == "source_block" and item["block_id"]
    ]
    if snapshot.get("source_credit_replaced_block_ids") != expected_replaced:
        raise WebReaderError(
            "reader source-credit replacement blocks are invalid"
        )
    if snapshot.get("authors") != [
        item["source_name"] for item in source_credit["authors"]
    ]:
        raise WebReaderError("legacy reader authors differ from source credit")
    if snapshot.get("translation_reference") is not None:
        try:
            from .translation_reference import (
                TranslationReferenceError,
                validate_translation_reference_provenance,
            )
            compact_reference = validate_translation_reference_provenance(
                snapshot.get("translation_reference"),
                project_root=root,
                expected_chapter_ids=list(
                    (snapshot.get("coverage") or {}).get("chapter_ids") or []
                ),
            )
        except (TranslationReferenceError, TypeError, ValueError) as exc:
            raise WebReaderError(
                "reader translation reference is invalid"
            ) from exc
        if manifest.get("translation_reference") != compact_reference:
            raise WebReaderError(
                "web manifest translation reference differs from the reader snapshot"
            )
    elif "translation_reference" in manifest:
        raise WebReaderError(
            "web manifest contains translation reference absent from the snapshot"
        )


def _validate_source_asset_bindings(
    root: Path,
    snapshot: Mapping[str, Any],
    manifest_assets: Sequence[Any],
) -> None:
    expected: dict[str, str] = {}

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            url = value.get("url")
            digest = value.get("sha256")
            if (
                isinstance(url, str)
                and url.startswith("assets/source/")
                and isinstance(digest, str)
            ):
                relative = f"reader/{url}"
                existing = expected.get(relative)
                if existing is not None and existing != digest:
                    raise WebReaderError(
                        "reader snapshot source asset hashes conflict"
                    )
                expected[relative] = digest
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(snapshot)
    actual: dict[str, str] = {}
    for record in manifest_assets:
        if not isinstance(record, Mapping):
            continue
        relative = str(record.get("path") or "")
        if not relative.startswith("reader/assets/source/"):
            continue
        if relative in actual:
            raise WebReaderError(
                "web manifest repeats a source asset record"
            )
        actual[relative] = str(record.get("sha256") or "")
    if actual != expected:
        raise WebReaderError(
            "reader snapshot and manifest source assets differ"
        )
    for relative, digest in expected.items():
        path = root / relative
        _reject_symlink_components(root, path)
        if (
            not stat.S_ISREG(path.lstat().st_mode)
            or sha256_file(path) != digest
        ):
            raise WebReaderError(
                "reader source asset binding is invalid"
            )


def _source_credit_manifest(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    try:
        credit = validate_source_credit(snapshot.get("source_credit"))
    except (SourceCreditError, TypeError) as exc:
        raise WebReaderError("reader source credit is invalid") from exc
    front_ids = snapshot.get("source_credit_front_matter_block_ids")
    if not isinstance(front_ids, list) or not all(
        isinstance(item, str) for item in front_ids
    ):
        raise WebReaderError("reader source-credit front-matter placement is invalid")
    order = source_credit_placement(
        credit, front_matter_block_ids=front_ids,
    )
    visible = snapshot.get("source_credit_visible_projection")
    expected_visible = _source_credit_visible_projection(credit, order)
    if visible != expected_visible:
        raise WebReaderError("reader source-credit visible projection is invalid")
    anchors = {item["id"]: item for item in credit["anchors"]}
    return {
        "schema_version": credit["schema_version"],
        "canonical_sha256": credit["canonical_sha256"],
        "front_matter_block_ids": front_ids,
        "replaced_block_ids": list(
            snapshot.get("source_credit_replaced_block_ids") or []
        ),
        "ordered_items": order,
        "placements": [
            {
                "anchor_id": item["anchor_id"],
                "placement": anchors[item["anchor_id"]]["placement"],
                "block_id": anchors[item["anchor_id"]]["block_id"],
                "render_slot": item["slot"],
            }
            for item in order
        ],
        "visible_counts": {
            "authors": sum(item["kind"] == "author" for item in visible),
            "affiliations": sum(
                item["kind"] == "affiliation" for item in visible
            ),
            "profiles": sum(item["kind"] == "profile" for item in visible),
        },
    }


def _source_credit_visible_projection(
    credit: Mapping[str, Any], order: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build the exact, text-bearing sequence consumed by the DOM renderer."""
    front_ids = [
        str(item["block_id"])
        for item in order
        if item["slot"] == "front_matter" and item.get("block_id")
    ]
    return source_credit_visible_projection(
        credit, front_matter_block_ids=front_ids,
    )


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
    _reject_symlink_components(root, index_path)
    try:
        index_mode = index_path.lstat().st_mode
    except FileNotFoundError:
        raise WebReaderError(f"reader artifact is missing or empty: {index_path}")
    if not stat.S_ISREG(index_mode):
        raise WebReaderError("committed reader index is not a regular file")
    index_hash = sha256_file(index_path)
    index_size = index_path.stat().st_size
    legacy = root / "reader" / "manifest.json"
    data_dir = root / "reader" / "data"
    _reject_symlink_components(root, legacy)
    _reject_symlink_components(root, data_dir)
    candidates = ([legacy] if legacy.is_file() else []) + (
        sorted(
            data_dir.glob("manifest-*.json"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        if data_dir.is_dir() else []
    )
    expected_path = index_path.relative_to(root).as_posix()
    for candidate in candidates:
        _reject_symlink_components(root, candidate)
        if not stat.S_ISREG(candidate.lstat().st_mode):
            continue
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


def _object_record(value: ReaderObject) -> dict[str, Any]:
    return {
        "path": value.relative_path,
        "sha256": value.sha256,
        "bytes": len(value.data),
    }


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise WebReaderError("reader publish UTC clock must be aware")
    return value.astimezone(timezone.utc)


def _assert_file_identity(path: Path, *, sha256: str, size: int) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise WebReaderError(f"candidate reader file is missing: {path}") from exc
    if (
        not stat.S_ISREG(mode)
        or path.stat().st_size != size
        or sha256_file(path) != sha256
    ):
        raise WebReaderError(f"candidate reader write changed unexpectedly: {path}")


def _publish_fault_point(_label: str) -> None:
    """No-op seam used by tests to inject a failure at every publish write."""


def _restore_index(index_path: Path, previous: bytes | None) -> None:
    if previous is None:
        index_path.unlink(missing_ok=True)
        return
    _atomic_write_bytes(index_path, previous)


def _create_or_adopt_immutable(
    root: Path, path: Path, value: bytes,
) -> None:
    """Install one complete immutable target or adopt an exact winner."""
    _ensure_regular_directory_chain(root, path.parent)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.candidate-", dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            _adopt_exact_regular(path, value)
        else:
            _adopt_exact_regular(path, value)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except BaseException:
        raise
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _adopt_exact_regular(path: Path, value: bytes) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise WebReaderError(
            f"immutable Reader target cannot be adopted: {path}"
        ) from exc
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_size != len(value):
            raise WebReaderError(
                f"immutable Reader target conflicts: {path}"
            )
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        if digest.hexdigest() != hashlib.sha256(value).hexdigest():
            raise WebReaderError(
                f"immutable Reader target conflicts: {path}"
            )
    finally:
        os.close(descriptor)


def _ensure_regular_directory_chain(root: Path, parent: Path) -> None:
    try:
        relative = parent.relative_to(root)
    except ValueError as exc:
        raise WebReaderError("Reader target escapes the project") from exc
    current = root
    for component in relative.parts:
        current = current / component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            try:
                current.mkdir()
            except FileExistsError:
                pass
            mode = current.lstat().st_mode
        if not stat.S_ISDIR(mode):
            raise WebReaderError(
                f"Reader target parent is not a regular directory: {current}"
            )


def _reject_symlink_components(root: Path, path: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise WebReaderError("Reader path escapes the project") from exc
    current = root
    for component in relative.parts:
        current = current / component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            return
        if stat.S_ISLNK(mode):
            raise WebReaderError(
                f"Reader path contains a symbolic link: {current}"
            )


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


def _reader_final_overrides(
    checkpoint: Path | None,
    *,
    replacement_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if checkpoint is None:
        return {}
    path = checkpoint / "reader-final.json"
    if not path.is_file():
        return {}
    value = _read_object(path, label="reader final checkpoint")
    version = value.get("schema_version")
    if version not in {
        READER_FINAL_VERSION,
        _LEGACY_READER_FINAL_VERSION,
    }:
        raise WebReaderError("reader final checkpoint schema is invalid")
    overrides = value.get("final_overrides")
    if not isinstance(overrides, Mapping):
        raise WebReaderError("reader final checkpoint has no final_overrides object")
    if version == _LEGACY_READER_FINAL_VERSION:
        replacement_content = (
            replacement_overrides.get("content")
            if isinstance(replacement_overrides, Mapping)
            else None
        )
        replacement_credit = (
            replacement_overrides.get("source_credit")
            if isinstance(replacement_overrides, Mapping)
            else None
        ) or (
            replacement_content.get("source_credit")
            if isinstance(replacement_content, Mapping)
            else None
        )
        try:
            validate_source_credit(replacement_credit)
        except (SourceCreditError, TypeError) as exc:
            raise WebReaderError(
                "legacy reader final checkpoint requires current source credit"
            ) from exc
        # v3 predates source-credit-complete final payloads.  Validate its
        # envelope, but never merge stale optional fields into a current
        # replacement supplied by the caller.
        return {}
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


def _source_block(
    block: Mapping[str, Any],
    *,
    entities: Mapping[str, Mapping[str, dict[str, Any]]],
    translated_title: str = "",
    language: str = "und",
) -> dict[str, Any]:
    value = dict(block)
    kind = str(value.get("type") or value.get("kind") or "text").casefold()
    output: dict[str, Any] = {
        "block_id": block_id(dict(value)),
        "kind": kind,
        "title": str(value.get("title") or ""),
        "runs": _inline_runs(value),
        "language": language,
        "direction": _language_direction(language),
    }
    if translated_title and translated_title != _block_title(value):
        output["source_title"] = _block_title(value)
        output["translated_title"] = translated_title
        output["translated_title_runs"] = _translation_runs(
            translated_title, value
        )
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


def _paper_title_block_ids(document: Mapping[str, Any]) -> set[str]:
    """Return source blocks represented by the reader's single paper header."""
    output = {
        block_id(dict(item))
        for item in document.get("blocks") or []
        if isinstance(item, Mapping)
        and (
            str(item.get("source_role") or item.get("role") or "").casefold()
            == "front_matter_title"
            or "front_matter_title" in {
                str(value).casefold() for value in item.get("front_matter_roles") or []
            }
        )
    }
    block_ids = (document.get("front_matter") or {}).get("block_ids") or {}
    if isinstance(block_ids, Mapping):
        explicit = block_ids.get("title") or block_ids.get("titles") or []
        if not isinstance(explicit, Sequence) or isinstance(explicit, (str, bytes)):
            explicit = [explicit]
        output.update(str(value) for value in explicit if str(value))
    return {value for value in output if value}


def _title_translation_index(value: Any) -> dict[str, str]:
    """Normalize reviewed title translations while accepting simple test maps."""
    if not value:
        return {}
    records: Any = value
    if isinstance(value, Mapping):
        nested = value.get("titles") or value.get("translations")
        if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
            records = nested
        else:
            output: dict[str, str] = {}
            for key, raw in value.items():
                if key in {"schema_version", "source_language", "target_language", "source_sha256"}:
                    continue
                text = (
                    str(raw.get("text") or raw.get("translated_title") or raw.get("translation") or "").strip()
                    if isinstance(raw, Mapping)
                    else str(raw or "").strip()
                )
                if text:
                    output[str(key)] = text
            return output
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        return {}
    output: dict[str, str] = {}
    for raw in records:
        if not isinstance(raw, Mapping):
            continue
        text = str(
            raw.get("text")
            or raw.get("translated_title")
            or raw.get("translation")
            or ""
        ).strip()
        if not text:
            continue
        title_id = str(raw.get("title_id") or "").strip()
        block_value = str(raw.get("block_id") or "").strip()
        chapter_id = str(raw.get("chapter_id") or "").strip()
        role = str(raw.get("role") or "").casefold()
        if title_id:
            output[title_id] = text
        if block_value:
            output.setdefault(f"block:{block_value}", text)
            output.setdefault(block_value, text)
        if chapter_id and (
            role in {"chapter", "chapter_title"}
            or title_id == f"chapter:{chapter_id}"
        ):
            output.setdefault(f"chapter:{chapter_id}", text)
        if role in {"document", "document_title", "paper_title", "title"}:
            output.setdefault("document:title", text)
    return output


def _block_title(block: Mapping[str, Any]) -> str:
    heading = block.get("heading") if isinstance(block.get("heading"), Mapping) else {}
    return str(
        block.get("title")
        or heading.get("title")
        or heading.get("text")
        or (block.get("heading") if isinstance(block.get("heading"), str) else "")
        or block.get("text")
        or ""
    ).strip()


def _translated_block_title(
    block: Mapping[str, Any], translations: Mapping[str, str]
) -> str:
    identifier = block_id(dict(block))
    return str(
        translations.get(f"block:{identifier}")
        or translations.get(identifier)
        or ""
    ).strip()


def _translated_chapter_title(
    chapter: Mapping[str, Any], translations: Mapping[str, str]
) -> str:
    for identifier in chapter.get("title_block_ids") or []:
        value = translations.get(f"block:{identifier}") or translations.get(str(identifier))
        if value:
            return str(value).strip()
    chapter_id = str(chapter.get("chapter_id") or "")
    return str(translations.get(f"chapter:{chapter_id}") or "").strip()


def _translated_document_title(
    document: Mapping[str, Any], translations: Mapping[str, str]
) -> str:
    direct = str(translations.get("document:title") or "").strip()
    if direct:
        return direct
    for identifier in _paper_title_block_ids(document):
        value = translations.get(f"block:{identifier}") or translations.get(identifier)
        if value:
            return str(value).strip()
    return ""


def _normalize_language_tag(value: str) -> str:
    parts = [part for part in str(value or "und").strip().replace("_", "-").split("-") if part]
    if not parts:
        return "und"
    normalized = [parts[0].casefold()]
    for part in parts[1:]:
        normalized.append(
            part.title() if len(part) == 4 and part.isalpha()
            else part.upper() if len(part) in {2, 3} and part.isalpha()
            else part
        )
    return "-".join(normalized)


def _language_direction(language: str) -> str:
    base = _normalize_language_tag(language).split("-", 1)[0]
    if base in {"ar", "dv", "fa", "he", "ku", "ps", "sd", "ug", "ur", "yi"}:
        return "rtl"
    if base in {"mul", "und", "zxx"}:
        return "auto"
    return "ltr"


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
    fields = ("motivation", "main_content", "section_logic")
    output = [
        {"kind": key, "runs": _text_math_runs(str(value.get(key) or ""))}
        for key in fields
        if str(value.get(key) or "").strip()
    ]
    comparison = value.get("pedagogical_comparison")
    if isinstance(comparison, Mapping) and str(comparison.get("text") or "").strip():
        output.append({
            "kind": "pedagogical_comparison",
            "runs": _text_math_runs(str(comparison.get("text") or "")),
            "sources": _safe_sources(comparison.get("sources")),
        })
    if str(value.get("prerequisites") or "").strip():
        output.append({
            "kind": "prerequisites",
            "runs": _text_math_runs(str(value.get("prerequisites") or "")),
        })
    for item in value.get("historical_context") or []:
        if not isinstance(item, Mapping) or not str(item.get("text") or "").strip():
            continue
        output.append({
            "kind": "historical_context",
            "runs": _text_math_runs(str(item.get("text") or "")),
            "sources": _safe_sources(item.get("sources")),
        })
    for item in value.get("supplementary_reading") or []:
        if not isinstance(item, Mapping):
            continue
        title = str(item.get("title") or "").strip()
        reason = str(item.get("reason") or "").strip()
        if title and reason:
            output.append({
                "kind": "supplementary_reading",
                "runs": _text_math_runs(f"{title}: {reason}"),
            })
    return output


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
    source_language: str = "und",
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
                    _source_block(item, entities=entities, language=source_language)
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
        "source": [
            _source_block(item, entities=entities, language=source_language)
            for item in blocks
        ],
    }]


def _orphan_structural_appendices(
    document: Mapping[str, Any],
    *,
    chapters: Sequence[Mapping[str, Any]],
    title_translations: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Expose translated structural headings omitted from chapter body lanes."""
    represented = {
        str(identifier)
        for chapter in chapters
        for identifier in chapter.get("block_ids") or []
        if str(identifier)
    }
    represented.update(_paper_title_block_ids(document))
    output: list[dict[str, Any]] = []
    structural = {"part", "chapter", "heading", "section", "subsection", "subsubsection"}
    for raw in document.get("blocks") or []:
        if not isinstance(raw, Mapping):
            continue
        identifier = block_id(dict(raw))
        kind = str(raw.get("type") or raw.get("kind") or "").casefold()
        if not identifier or identifier in represented or kind not in structural:
            continue
        source_title = _block_title(raw)
        if not source_title:
            continue
        translated_title = _translated_block_title(raw, title_translations)
        output.append({
            "appendix_id": f"source-heading-{identifier}",
            "kind": "source_only_structural_heading",
            "title": translated_title or source_title,
            "source_title": source_title,
            "translated_title": translated_title,
            "source": [],
        })
    return output


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
            separator = str(raw.get("separator_before") or "")
            if separator:
                output.append({"type": "text", "text": separator})
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


def _prepare_builtin_assets() -> tuple[tuple[ReaderObject, ...], str]:
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
    output = tuple(
        ReaderObject(
            f"reader/{asset_root}/{relative.as_posix()}",
            data,
            "builtin-asset",
        )
        for relative, data in payloads
    )
    return output, asset_root


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


def _prepare_source_assets(
    root: Path,
    state: Mapping[str, Any],
    snapshot: dict[str, Any],
    *,
    final_overrides: Mapping[str, Any] | None,
) -> tuple[ReaderObject, ...]:
    checkpoint = _checkpoint(root, state)
    overrides = _deep_merge(
        _reader_final_overrides(
            checkpoint, replacement_overrides=final_overrides,
        ),
        final_overrides or {},
    )
    envelope = _document_envelope(checkpoint, overrides)
    document = _document(envelope, overrides)
    assets = {
        str(item.get("asset_id") or item.get("id") or ""): dict(item)
        for item in document.get("assets") or []
        if isinstance(item, Mapping) and (item.get("asset_id") or item.get("id"))
    }
    objects: list[ReaderObject] = []
    object_paths: set[str] = set()
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
                raw_data = path.read_bytes()
                source_digest = hashlib.sha256(raw_data).hexdigest()
                expected = str(asset.get("sha256") or "")
                if expected and expected != source_digest:
                    raise WebReaderError(f"source asset hash mismatch: {path}")
                cached = by_id.get(str(identifier))
                if cached is None:
                    suffix = path.suffix.casefold()
                    if suffix == ".svg":
                        data = _safe_svg(
                            raw_data.decode("utf-8", errors="replace")
                        ).encode("utf-8")
                    else:
                        data = raw_data
                    digest = hashlib.sha256(data).hexdigest()
                    relative = f"reader/assets/source/{digest}{suffix}"
                    cached = {
                        "url": Path(relative).relative_to("reader").as_posix(),
                        "sha256": digest,
                    }
                    by_id[str(identifier)] = cached
                    if relative not in object_paths:
                        objects.append(
                            ReaderObject(relative, data, "source-asset")
                        )
                        object_paths.add(relative)
                rendered.append(dict(cached))
            if rendered:
                source["assets"] = rendered
    return tuple(sorted(objects, key=lambda item: item.relative_path))


def _index_html(
    *, data_script: str, asset_root: str, title: str, language: str = "und"
) -> str:
    escaped_title = _escape_html(title)
    escaped_assets = _escape_html(asset_root.rstrip("/"))
    normalized_language = _normalize_language_tag(language)
    direction = _language_direction(normalized_language)
    return f"""<!doctype html>
<html lang="{_escape_html(normalized_language)}" dir="{direction}">
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
  <button id="sidebar-toggle" class="sidebar-toggle" type="button" aria-controls="chapter-sidebar" aria-expanded="true"></button>
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
    unresolved = root / relative
    _reject_symlink_components(root, unresolved)
    try:
        mode = unresolved.lstat().st_mode
    except FileNotFoundError:
        raise ReaderDependencyMissing(
            f"web manifest file is missing: {relative}"
        )
    if not stat.S_ISREG(mode):
        raise WebReaderError(f"web manifest file is not regular: {relative}")
    path = _inside(root, relative)
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
