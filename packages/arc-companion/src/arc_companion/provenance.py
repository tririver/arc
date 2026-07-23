from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Callable, Iterable, Mapping

from .artifact_ids import ArtifactIdError, resolve_artifact_dir
from .content import (
    READER_CONTENT_VERSION,
    content_object_path,
    load_reader_content,
)
from .io import canonical_json
from .pdf import (
    PDF_VALIDATION_RECEIPT_VERSION,
    match_validated_pdf_revision,
    normalize_run_root_pdf_state,
)
from .web import validate_reader_project


FINAL_PROVENANCE_VERSION = "arc.companion.final-provenance.v1"
RECEIPT_REF_VERSION = "arc.companion.provenance-receipt-ref.v1"
FINAL_COUNTS_VERSION = "arc.companion.final-counts.v1"
PROVENANCE_STATE_VERSION = "arc.companion.published-provenance.v1"
PROVENANCE_POLICY_VERSION = "arc.companion.provenance-policy.v1"

MAX_REF_BYTES = 16_777_216
MAX_REFS = 65_536
MAX_TOTAL_REF_BYTES = 536_870_912
MAX_PATH_BYTES = 512
MAX_BASIS_RECORDS = 262_144
MAX_JSON_DEPTH = 32

_SHA256 = re.compile(r"[0-9a-f]{64}")
_MODES = {
    "build",
    "render_pdf",
    "render_web",
    "render_all",
    "legacy_upgrade",
}
_OUTPUT_KEYS = (
    "pdf",
    "tex",
    "run_pdf",
    "source_manifest",
    "render_validation",
    "reader_index",
    "reader_manifest",
    "reader_snapshot",
)


class ProvenanceError(RuntimeError):
    """Stable local provenance validation or publication failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(str(message)[:512])
        self.code = code


@dataclass(frozen=True)
class ProvenancePlan:
    root: Path
    mode: str
    fingerprint: str
    checkpoint: Mapping[str, Any]
    reviewed_content: Mapping[str, Any]
    outputs: Mapping[str, Any]
    controls: tuple[Mapping[str, Any], ...]
    counts: Mapping[str, Any]
    recovery_bytes: bytes | None = None
    recovery_path: str | None = None


def plan_final_provenance(
    project_dir: Path,
    *,
    state: Mapping[str, Any],
    mode: str,
) -> ProvenancePlan:
    """Read and validate the current final roots without writing."""

    root = Path(project_dir).resolve()
    if mode not in _MODES:
        raise ProvenanceError("provenance_mode_invalid", "provenance mode is invalid")
    state_path = root / "state.json"
    state_bytes = _read_file(root, state_path, max_bytes=MAX_REF_BYTES)
    stored_state = _json_object(state_bytes, "provenance_state_invalid")
    if dict(stored_state) != dict(state):
        raise ProvenanceError(
            "provenance_state_changed", "supplied state is not the current state",
        )
    if state.get("status") != "complete":
        raise ProvenanceError(
            "provenance_state_invalid", "only complete state can publish provenance",
        )
    fingerprint = str(state.get("fingerprint") or "")
    if not fingerprint or len(fingerprint.encode("utf-8")) > 256:
        raise ProvenanceError(
            "provenance_state_invalid", "state fingerprint is unavailable",
        )
    checkpoint = _checkpoint_record(root, state)
    reviewed_content, envelope = _content_record(root, state)
    outputs = _output_records(root, state, mode=mode)
    controls = list(_control_records(
        root,
        state,
        checkpoint_path=root / str(checkpoint["path"]),
        envelope=envelope,
        outputs=outputs,
    ))
    recovery_bytes, recovery_path = _recovery_snapshot_plan(
        root, state, checkpoint=checkpoint,
    )
    if recovery_bytes is not None and recovery_path is not None:
        controls.append(_memory_ref(
            recovery_path,
            recovery_bytes,
            category="recovery_journal",
            receipt_schema=str(
                _json_object(
                    recovery_bytes, "provenance_recovery_invalid",
                ).get("schema_version") or ""
            ),
            status="complete",
            subject_sha256=str(checkpoint["identity_sha256"]),
        ))
    controls = list(_normalize_controls(controls))
    counts = _build_counts(root, envelope, controls)
    return ProvenancePlan(
        root=root,
        mode=mode,
        fingerprint=fingerprint,
        checkpoint=checkpoint,
        reviewed_content=reviewed_content,
        outputs=outputs,
        controls=tuple(controls),
        counts=counts,
        recovery_bytes=recovery_bytes,
        recovery_path=recovery_path,
    )


def commit_final_provenance(
    project_dir: Path,
    *,
    plan: ProvenancePlan,
    state: Mapping[str, Any],
    state_merger: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Revalidate and atomically publish immutable final provenance."""

    root = Path(project_dir).resolve()
    if root != plan.root:
        raise ProvenanceError(
            "provenance_plan_invalid", "provenance plan belongs to another project",
        )
    current_bytes = _read_file(root, root / "state.json", max_bytes=MAX_REF_BYTES)
    current = _json_object(current_bytes, "provenance_state_invalid")
    if dict(current) != dict(state):
        raise ProvenanceError(
            "provenance_state_changed", "state changed before provenance commit",
        )
    # GC state metadata is expected to change after planning. Rebuild the plan
    # from current state and require all non-GC semantic roots to remain exact.
    refreshed = plan_final_provenance(root, state=current, mode=plan.mode)
    if _plan_projection(refreshed) != _plan_projection(plan):
        raise ProvenanceError(
            "provenance_roots_changed", "publication roots changed after planning",
        )
    controls = list(refreshed.controls)
    gc_control = _gc_control(root, current)
    if gc_control is not None:
        controls.append(gc_control)
    controls = list(_normalize_controls(controls))
    counts_bytes = _json_bytes(refreshed.counts)
    counts_sha256 = hashlib.sha256(counts_bytes).hexdigest()
    counts_path = (
        root / ".arc-companion" / "provenance" / "count-bases"
        / f"{counts_sha256}.json"
    )
    _create_or_adopt(root, counts_path, counts_bytes)
    if refreshed.recovery_bytes is not None and refreshed.recovery_path is not None:
        history_path = root / refreshed.recovery_path
        _create_or_adopt(root, history_path, refreshed.recovery_bytes)
    counts_ref = {
        "path": _relative(root, counts_path),
        "sha256": counts_sha256,
        "bytes": len(counts_bytes),
        "schema_version": FINAL_COUNTS_VERSION,
    }
    semantic = _semantic_projection(
        fingerprint=refreshed.fingerprint,
        checkpoint=refreshed.checkpoint,
        reviewed_content=refreshed.reviewed_content,
        outputs=refreshed.outputs,
        controls=controls,
        counts=counts_ref,
    )
    final_id = _sha_json(semantic)
    document = {
        "schema_version": FINAL_PROVENANCE_VERSION,
        "status": "complete",
        "final_id": final_id,
        "fingerprint": refreshed.fingerprint,
        "checkpoint": dict(refreshed.checkpoint),
        "reviewed_content": dict(refreshed.reviewed_content),
        "outputs": {
            "mode": refreshed.mode,
            "content_sha256": refreshed.outputs["content_sha256"],
            **{key: refreshed.outputs.get(key) for key in _OUTPUT_KEYS},
        },
        "controls": controls,
        "counts": counts_ref,
        "attribution": {
            "status": refreshed.counts["attribution_status"],
            "segment_count": refreshed.counts["segment_counts"]["total"],
            "review_receipt_count": refreshed.counts[
                "review_counts"
            ]["receipts"],
            "review_calls": refreshed.counts["review_counts"]["calls"],
        },
    }
    provenance_bytes = _json_bytes(document)
    provenance_sha256 = hashlib.sha256(provenance_bytes).hexdigest()
    provenance_path = (
        root / ".arc-companion" / "provenance" / "objects"
        / f"{final_id}.json"
    )
    _create_or_adopt(root, provenance_path, provenance_bytes)
    published = {
        "schema_version": PROVENANCE_STATE_VERSION,
        "status": "complete",
        "final_id": final_id,
        "path": _relative(root, provenance_path),
        "sha256": provenance_sha256,
        "bytes": len(provenance_bytes),
        "counts_path": _relative(root, counts_path),
        "counts_sha256": counts_sha256,
        "counts_bytes": len(counts_bytes),
    }
    state_merger({"published_provenance": published})
    return published


def validate_published_provenance(
    project_dir: Path,
    state: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    root = Path(project_dir).resolve()
    published = state.get("published")
    published = published if isinstance(published, Mapping) else {}
    raw = published.get("provenance")
    if raw is None:
        if state.get("provenance_policy_version") == PROVENANCE_POLICY_VERSION:
            raise ProvenanceError(
                "provenance_state_partial",
                "required published provenance is unavailable",
            )
        return None
    mapping = _validate_state_mapping(raw)
    path = root / _safe_relative(str(mapping["path"]))
    value_bytes = _read_file(root, path, max_bytes=MAX_REF_BYTES)
    if (
        len(value_bytes) != mapping["bytes"]
        or hashlib.sha256(value_bytes).hexdigest() != mapping["sha256"]
    ):
        raise ProvenanceError(
            "provenance_object_invalid", "provenance object identity differs",
        )
    value = _json_object(value_bytes, "provenance_object_invalid")
    _validate_document_shape(value)
    if (
        value["final_id"] != mapping["final_id"]
        or path.name != f"{value['final_id']}.json"
    ):
        raise ProvenanceError(
            "provenance_object_invalid", "provenance final ID binding differs",
        )
    checkpoint = _checkpoint_record(root, state)
    reviewed_content, envelope = _content_record(root, state)
    if value["checkpoint"] != checkpoint:
        raise ProvenanceError(
            "provenance_checkpoint_invalid",
            "published checkpoint binding differs",
        )
    if value["reviewed_content"] != reviewed_content:
        raise ProvenanceError(
            "provenance_content_invalid",
            "published reviewed-content binding differs",
        )
    current_outputs = _output_records(
        root, state, mode=str(value["outputs"]["mode"]),
    )
    if value["outputs"] != {
        "mode": current_outputs["mode"],
        "content_sha256": current_outputs["content_sha256"],
        **{key: current_outputs.get(key) for key in _OUTPUT_KEYS},
    }:
        raise ProvenanceError(
            "provenance_output_invalid",
            "published output binding differs from current owner-validated state",
        )
    counts = _read_ref(root, value["counts"], expected_schema=FINAL_COUNTS_VERSION)
    _validate_counts(counts)
    if value["attribution"] != {
        "status": counts["attribution_status"],
        "segment_count": counts["segment_counts"]["total"],
        "review_receipt_count": counts["review_counts"]["receipts"],
        "review_calls": counts["review_counts"]["calls"],
    }:
        raise ProvenanceError(
            "provenance_counts_invalid",
            "provenance attribution summary differs from counts",
        )
    if (
        value["counts"]["path"] != mapping["counts_path"]
        or value["counts"]["sha256"] != mapping["counts_sha256"]
        or value["counts"]["bytes"] != mapping["counts_bytes"]
    ):
        raise ProvenanceError(
            "provenance_counts_invalid", "provenance counts state binding differs",
        )
    for output in value["outputs"].values():
        if isinstance(output, Mapping):
            _verify_file_record(root, output)
    for control in value["controls"]:
        loaded = _read_ref(
            root, control, expected_schema=str(control["receipt_schema"]),
        )
        if str(loaded.get("schema_version") or "") != control["receipt_schema"]:
            raise ProvenanceError(
                "provenance_control_invalid", "control receipt schema differs",
            )
        _validate_control_owner(
            root, control, loaded, state=state, envelope=envelope,
        )
    current_plan = plan_final_provenance(
        root,
        state=state,
        mode=str(value["outputs"]["mode"]),
    )
    non_gc_controls = [
        item for item in value["controls"]
        if item["category"] != "artifact_gc"
    ]
    gc_controls = [
        item for item in value["controls"]
        if item["category"] == "artifact_gc"
    ]
    current_gc = _gc_control(root, state)
    if (
        non_gc_controls != [dict(item) for item in current_plan.controls]
        or gc_controls != (
            [dict(current_gc)] if current_gc is not None else []
        )
    ):
        raise ProvenanceError(
            "provenance_control_invalid",
            "published controls differ from current owner-validated controls",
        )
    if counts != current_plan.counts:
        raise ProvenanceError(
            "provenance_counts_invalid",
            "final counts differ from the current provable basis",
        )
    semantic = _semantic_projection(
        fingerprint=str(value["fingerprint"]),
        checkpoint=value["checkpoint"],
        reviewed_content=value["reviewed_content"],
        outputs=value["outputs"],
        controls=value["controls"],
        counts=value["counts"],
    )
    if _sha_json(semantic) != value["final_id"]:
        raise ProvenanceError(
            "provenance_final_id_invalid", "provenance final ID differs",
        )
    return value


def provenance_paths(
    project_dir: Path,
    state: Mapping[str, Any],
) -> tuple[Path, ...]:
    value = validate_published_provenance(project_dir, state)
    if value is None:
        return ()
    root = Path(project_dir).resolve()
    paths = {
        root / _safe_relative(str(
            ((state.get("published") or {}).get("provenance") or {})["path"]
        )),
        *_document_paths(root, value),
    }
    return tuple(sorted(paths))


def retained_provenance_paths(
    project_dir: Path,
    state: Mapping[str, Any],
) -> tuple[Path, ...]:
    """Return the closed path graph of every retained immutable provenance."""

    root = Path(project_dir).resolve()
    paths = set(provenance_paths(root, state))
    objects = root / ".arc-companion" / "provenance" / "objects"
    if not objects.exists():
        return tuple(sorted(paths))
    candidates = sorted(objects.glob("*.json"))
    if len(candidates) > MAX_REFS:
        raise ProvenanceError(
            "provenance_limit_exceeded",
            "too many retained provenance objects",
        )
    for path in candidates:
        data = _read_file(root, path, max_bytes=MAX_REF_BYTES)
        value = _json_object(data, "provenance_object_invalid")
        _validate_document_shape(value)
        semantic = _semantic_projection(
            fingerprint=str(value["fingerprint"]),
            checkpoint=value["checkpoint"],
            reviewed_content=value["reviewed_content"],
            outputs=value["outputs"],
            controls=value["controls"],
            counts=value["counts"],
        )
        if (
            path.name != f"{value['final_id']}.json"
            or _sha_json(semantic) != value["final_id"]
        ):
            raise ProvenanceError(
                "provenance_final_id_invalid",
                "retained provenance final ID differs",
            )
        paths.add(path)
        paths.update(_document_paths(root, value))
    return tuple(sorted(paths))


def provenance_package_paths(
    project_dir: Path,
    state: Mapping[str, Any],
) -> tuple[Path, ...]:
    value = validate_published_provenance(project_dir, state)
    if value is None:
        return ()
    root = Path(project_dir).resolve()
    mapping = ((state.get("published") or {}).get("provenance") or {})
    return (
        root / _safe_relative(str(mapping["path"])),
        root / _safe_relative(str(mapping["counts_path"])),
    )


def _document_paths(
    root: Path, value: Mapping[str, Any],
) -> set[Path]:
    records = [
        value["checkpoint"],
        value["reviewed_content"],
        value["counts"],
        *value["controls"],
        *(
            record
            for record in value["outputs"].values()
            if isinstance(record, Mapping)
        ),
    ]
    return {
        root / _safe_relative(str(record["path"]))
        for record in records
    }


def _checkpoint_record(
    root: Path, state: Mapping[str, Any],
) -> Mapping[str, Any]:
    raw = state.get("checkpoint_dir")
    identity = str(
        state.get("checkpoint_identity") or state.get("fingerprint") or ""
    )
    if not raw or not _SHA256.fullmatch(identity):
        raise ProvenanceError(
            "provenance_checkpoint_invalid", "checkpoint identity is incomplete",
        )
    checkpoint = _inside(root, Path(str(raw)))
    checkpoint_root = root / ".arc-companion" / "checkpoints"
    try:
        allocation = resolve_artifact_dir(
            checkpoint_root,
            checkpoint,
            expected_identity=identity,
            kind="checkpoint",
        )
    except (ArtifactIdError, OSError, ValueError) as exc:
        raise ProvenanceError(
            "provenance_checkpoint_invalid", "checkpoint identity is invalid",
        ) from exc
    receipt_schema = "legacy-checkpoint"
    if allocation.receipt_path is not None:
        state_receipt_path = state.get("checkpoint_identity_receipt_path")
        state_receipt_sha256 = state.get(
            "checkpoint_identity_receipt_sha256"
        )
        if (
            not state_receipt_path
            or _inside(root, Path(str(state_receipt_path)))
            != allocation.receipt_path
            or state_receipt_sha256 != allocation.receipt_sha256
        ):
            raise ProvenanceError(
                "provenance_checkpoint_invalid",
                "checkpoint directory receipt binding differs",
            )
        receipt = _json_object(
            _read_file(root, allocation.receipt_path, max_bytes=MAX_REF_BYTES),
            "provenance_checkpoint_invalid",
        )
        receipt_schema = str(receipt.get("schema_version") or "")
    return {
        "identity_sha256": identity,
        "path": _relative(root, allocation.path),
        "schema_version": receipt_schema,
    }


def _content_record(
    root: Path, state: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    published = state.get("published")
    published = published if isinstance(published, Mapping) else {}
    semantic = str(published.get("content_sha256") or "")
    if not _SHA256.fullmatch(semantic):
        raise ProvenanceError(
            "provenance_content_invalid", "reviewed-content identity is invalid",
        )
    try:
        envelope = load_reader_content(root, semantic)
    except Exception as exc:
        raise ProvenanceError(
            "provenance_content_invalid", "reviewed-content object is invalid",
        ) from exc
    path = content_object_path(root, semantic)
    data = _read_file(root, path, max_bytes=MAX_TOTAL_REF_BYTES)
    return ({
        "semantic_sha256": semantic,
        "file_sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "path": _relative(root, path),
        "schema_version": READER_CONTENT_VERSION,
    }, envelope)


def _output_records(
    root: Path, state: Mapping[str, Any], *, mode: str,
) -> Mapping[str, Any]:
    published = state.get("published")
    published = published if isinstance(published, Mapping) else {}
    content_sha256 = str(published.get("content_sha256") or "")
    pdf = published.get("pdf")
    pdf = pdf if isinstance(pdf, Mapping) else {}
    web = published.get("web")
    web = web if isinstance(web, Mapping) else {}
    outputs: dict[str, Any] = {
        "mode": mode,
        "content_sha256": content_sha256,
        **{key: None for key in _OUTPUT_KEYS},
    }
    if pdf:
        decision = match_validated_pdf_revision(
            root, state, content_sha256=content_sha256,
        )
        if not decision.reusable:
            raise ProvenanceError(
                "provenance_output_invalid",
                f"current PDF is not reusable: {decision.reason}",
            )
        effective = normalize_run_root_pdf_state({**state, **dict(pdf)})
        for output, path_key, hash_key in (
            ("pdf", "output_pdf", "output_pdf_sha256"),
            ("tex", "output_tex", "output_tex_sha256"),
            ("run_pdf", "output_run_pdf", "output_run_pdf_sha256"),
            ("source_manifest", "source_manifest_path", "source_manifest_sha256"),
            ("render_validation", "validation_path", "validation_sha256"),
        ):
            if effective.get(path_key):
                outputs[output] = _file_record(
                    root, effective[path_key], str(effective.get(hash_key) or ""),
                )
    if web:
        effective_web = {**state, **dict(web)}
        try:
            validate_reader_project(root, state=effective_web)
        except Exception as exc:
            raise ProvenanceError(
                "provenance_output_invalid", "current Reader is invalid",
            ) from exc
        for output, path_key, hash_key in (
            ("reader_index", "output_html", "output_html_sha256"),
            ("reader_manifest", "web_manifest_path", "web_manifest_sha256"),
            ("reader_snapshot", "reader_snapshot_path", "reader_snapshot_sha256"),
        ):
            outputs[output] = _file_record(
                root, effective_web[path_key],
                str(effective_web.get(hash_key) or ""),
            )
    has_pdf = all(
        outputs[key] is not None
        for key in ("pdf", "tex", "source_manifest", "render_validation")
    )
    has_reader = all(
        outputs[key] is not None
        for key in ("reader_index", "reader_manifest", "reader_snapshot")
    )
    if (
        mode in {"build", "render_all"} and not (has_pdf and has_reader)
    ) or (mode == "render_pdf" and not has_pdf) or (
        mode == "render_web" and not has_reader
    ):
        raise ProvenanceError(
            "provenance_output_invalid",
            "provenance mode is missing its required current output lanes",
        )
    return outputs


def _control_records(
    root: Path,
    state: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    envelope: Mapping[str, Any],
    outputs: Mapping[str, Any],
) -> Iterable[Mapping[str, Any]]:
    source = checkpoint_path / "source-snapshot-receipt.json"
    source_value = _json_object(
        _read_file(root, source, max_bytes=MAX_REF_BYTES),
        "provenance_source_snapshot_invalid",
    )
    source_status = _validate_source_snapshot(
        root, checkpoint_path, source_value, state,
    )
    yield _control_ref(
        root,
        source,
        category="source_snapshot",
        allowed_schemas={
            "arc.companion.source-snapshot-receipt.v1",
            "arc.companion.source-snapshot-receipt.v2",
        },
        status=source_status,
        subject_sha256=_subject(str(state.get("fingerprint") or "")),
    )
    translation = checkpoint_path / "translation-reference.json"
    if translation.is_file():
        binding = _json_object(
            _read_file(root, translation, max_bytes=MAX_REF_BYTES),
            "provenance_translation_reference_invalid",
        )
        from .translation_reference import (
            TRANSLATION_REFERENCE_VALIDATION_VERSION,
            validate_translation_reference_provenance,
        )

        compact = (
            (envelope.get("content") or {}).get("translation_reference")
            if isinstance(envelope.get("content"), Mapping) else None
        )
        if (
            binding.get("schema_version")
            != TRANSLATION_REFERENCE_VALIDATION_VERSION
            or set(binding) != {
                "schema_version", "manifest_path", "manifest_sha256",
                "compact_provenance",
            }
            or binding.get("compact_provenance") != compact
            or not isinstance(compact, Mapping)
            or binding.get("manifest_path") != compact.get("manifest_path")
            or binding.get("manifest_sha256") != compact.get("manifest_sha256")
            or binding.get("manifest_path")
            != state.get("translation_reference_manifest_path")
            or binding.get("manifest_sha256")
            != state.get("translation_reference_manifest_sha256")
        ):
            raise ProvenanceError(
                "provenance_translation_reference_invalid",
                "translation-reference binding differs from reviewed content",
            )
        chapters = (envelope.get("content") or {}).get("chapters") or []
        chapter_ids = [
            str(item.get("chapter_id") or "")
            for item in chapters if isinstance(item, Mapping)
        ]
        validate_translation_reference_provenance(
            compact,
            project_root=root,
            expected_chapter_ids=chapter_ids,
        )
        yield _control_ref(
            root,
            translation,
            category="translation_reference",
            allowed_schemas={TRANSLATION_REFERENCE_VALIDATION_VERSION},
            status="validated",
            subject_sha256=None,
        )
    provenance = envelope.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    content_checkpoint = _inside(
        root, Path(str(provenance.get("checkpoint_dir") or checkpoint_path)),
    )
    review_receipts = envelope.get("review_receipts")
    review_receipts = (
        review_receipts if isinstance(review_receipts, Mapping) else {}
    )
    for name, identity in sorted(review_receipts.items()):
        if not isinstance(identity, Mapping):
            continue
        path = content_checkpoint / _safe_relative(str(name))
        data = _read_file(root, path, max_bytes=MAX_REF_BYTES)
        if (
            hashlib.sha256(data).hexdigest() != identity.get("sha256")
            or len(data) != identity.get("bytes")
        ):
            raise ProvenanceError(
                "provenance_review_invalid",
                "reviewed-content review receipt identity differs",
            )
        value = _json_object(
            data,
            "provenance_review_invalid",
        )
        saw_arbitration = False
        saw_reuse = False
        from .review_arbitration import REVIEW_ARBITRATION_RECEIPT_VERSION
        from .review_reuse import REVIEW_REUSE_RECEIPT_VERSION

        for key, category, statuses, receipt_schema in (
            (
                "review_arbitration_receipt",
                "review_arbitration",
                {"resolved", "no_conflicts"},
                REVIEW_ARBITRATION_RECEIPT_VERSION,
            ),
            (
                "review_reuse_receipt",
                "review_reuse",
                {"complete"},
                REVIEW_REUSE_RECEIPT_VERSION,
            ),
        ):
            reference = value.get(key)
            if not isinstance(reference, Mapping):
                continue
            saw_arbitration = saw_arbitration or category == "review_arbitration"
            saw_reuse = saw_reuse or category == "review_reuse"
            nested = content_checkpoint / _safe_relative(
                str(reference.get("path") or ""),
            )
            nested_ref = _control_ref(
                root,
                nested,
                category=category,
                allowed_schemas={receipt_schema},
                status="complete",
                subject_sha256=_subject(
                    str(
                        reference.get("merged_output_sha256")
                        or reference.get("final_review_sha256")
                        or ""
                    )
                ),
            )
            if nested_ref["sha256"] != reference.get("sha256"):
                raise ProvenanceError(
                    "provenance_review_invalid",
                    "review control reference hash differs",
                )
            nested_value = _json_object(
                _read_file(root, nested, max_bytes=MAX_REF_BYTES),
                "provenance_review_invalid",
            )
            actual_status = str(nested_value.get("status") or "")
            if category == "review_reuse":
                from .review_reuse import load_review_reuse_receipt

                nested_value = load_review_reuse_receipt(
                    root, nested.relative_to(root),
                )
                actual_status = "complete"
            if actual_status not in statuses:
                raise ProvenanceError(
                    "provenance_review_invalid",
                    "review control is not terminal",
                )
            yield {**nested_ref, "status": actual_status}
        if saw_arbitration and not saw_reuse:
            raise ProvenanceError(
                "provenance_review_invalid",
                "current review arbitration lacks its review-reuse proof",
            )
    validation = outputs.get("render_validation")
    if isinstance(validation, Mapping):
        yield _control_ref(
            root,
            root / str(validation["path"]),
            category="render_validation",
            allowed_schemas={PDF_VALIDATION_RECEIPT_VERSION},
            status="success",
            subject_sha256=str(
                (outputs.get("pdf") or {}).get("sha256") or ""
            ),
        )


def _validate_source_snapshot(
    root: Path,
    checkpoint_path: Path,
    receipt: Mapping[str, Any],
    state: Mapping[str, Any],
) -> str:
    schema = str(receipt.get("schema_version") or "")
    document = _checkpoint_json(root, checkpoint_path / "document.json")
    evidence = _optional_checkpoint_json(
        root, checkpoint_path / "evidence.json",
    )
    domain_context = _optional_checkpoint_json(
        root, checkpoint_path / "domain-context.json",
    )
    common_valid = (
        schema in {
            "arc.companion.source-snapshot-receipt.v1",
            "arc.companion.source-snapshot-receipt.v2",
        }
        and receipt.get("paper_id") == state.get("paper_id")
        and receipt.get("fingerprint") == state.get("fingerprint")
        and receipt.get("document_payload_sha256") == _sha_json(document)
        and _declared_optional_json_matches(
            receipt.get("evidence_sha256"), evidence,
        )
        and _declared_optional_json_matches(
            receipt.get("domain_context_sha256"), domain_context,
        )
    )
    if schema == "arc.companion.source-snapshot-receipt.v1":
        if not common_valid:
            raise ProvenanceError(
                "provenance_source_snapshot_invalid",
                "legacy source snapshot hashes differ",
            )
        return "legacy_validated"
    expected_keys = {
        "schema_version", "paper_id", "fingerprint", "checkpoint_identity",
        "build_instance_id", "build_request_sha256",
        "build_source_fingerprint", "document_payload_sha256",
        "chapters_pack_sha256", "evidence_sha256",
        "domain_context_sha256", "translation_reference_manifest_path",
        "translation_reference_manifest_sha256",
        "translation_reference_source_id",
        "translation_reference_source_hash",
    }
    chapters = _optional_checkpoint_json(
        root, checkpoint_path / "chapters.json",
    )
    reference_keys = (
        "translation_reference_manifest_path",
        "translation_reference_manifest_sha256",
        "translation_reference_source_id",
        "translation_reference_source_hash",
    )
    if not (
        common_valid
        and set(receipt) == expected_keys
        and receipt.get("checkpoint_identity")
        == (state.get("checkpoint_identity") or state.get("fingerprint"))
        and receipt.get("build_source_fingerprint") == state.get("fingerprint")
        and _SHA256.fullmatch(
            str(receipt.get("build_request_sha256") or "")
        )
        and isinstance(receipt.get("build_instance_id"), str)
        and bool(receipt["build_instance_id"])
        and _declared_optional_json_matches(
            receipt.get("chapters_pack_sha256"), chapters,
        )
        and all(receipt.get(key) == state.get(key) for key in reference_keys)
        and (
            all(receipt.get(key) is None for key in reference_keys)
            or all(receipt.get(key) not in {None, ""} for key in reference_keys)
        )
    ):
        raise ProvenanceError(
            "provenance_source_snapshot_invalid",
            "source snapshot owner binding differs",
        )
    return "validated"


def _checkpoint_json(root: Path, path: Path) -> Mapping[str, Any]:
    return _json_object(
        _read_file(root, path, max_bytes=MAX_TOTAL_REF_BYTES),
        "provenance_source_snapshot_invalid",
    )


def _optional_checkpoint_json(
    root: Path, path: Path,
) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    return _checkpoint_json(root, path)


def _declared_optional_json_matches(
    declared_sha256: object,
    value: Mapping[str, Any] | None,
) -> bool:
    if declared_sha256 is None:
        return True
    return (
        value is not None
        and _SHA256.fullmatch(str(declared_sha256))
        and declared_sha256 == _sha_json(value)
    )


def _gc_control(
    root: Path, state: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    gc = state.get("artifact_gc")
    if not isinstance(gc, Mapping):
        return None
    if gc.get("status") != "complete":
        raise ProvenanceError(
            "provenance_gc_invalid", "artifact-GC state is not terminal",
        )
    path = root / _safe_relative(str(gc.get("receipt_path") or ""))
    from .gc import load_gc_receipt

    receipt = load_gc_receipt(root, str(gc.get("receipt_path") or ""))
    record = _control_ref(
        root,
        path,
        category="artifact_gc",
        allowed_schemas={"arc.companion.gc-receipt.v1"},
        status="complete",
        subject_sha256=_subject(str(gc.get("candidate_set_sha256") or "")),
    )
    if record["sha256"] != gc.get("receipt_sha256"):
        raise ProvenanceError(
            "provenance_gc_invalid", "artifact-GC receipt identity differs",
        )
    if receipt.get("candidate_set_sha256") != gc.get(
        "candidate_set_sha256"
    ):
        raise ProvenanceError(
            "provenance_gc_invalid", "artifact-GC candidate identity differs",
        )
    return record


def _build_counts(
    root: Path,
    envelope: Mapping[str, Any],
    controls: Iterable[Mapping[str, Any]],
) -> Mapping[str, Any]:
    content = envelope.get("content")
    content = content if isinstance(content, Mapping) else {}
    segments = content.get("segments")
    segments = segments if isinstance(segments, list) else []
    basis = sorted({
        _sha_json({
            "segment_id": str(item.get("segment_id") or ""),
            "block_ids": list(item.get("block_ids") or []),
            "augmentation_block_ids": list(
                item.get("augmentation_block_ids") or []
            ),
            "structural_only": bool(item.get("structural_only")),
        })
        for item in segments if isinstance(item, Mapping)
    })
    review_controls = [
        item for item in controls
        if item["category"] in {
            "review_arbitration", "review_reuse",
        }
    ]
    review_calls: int | None = None
    reuse_controls = [
        item for item in review_controls
        if item["category"] == "review_reuse"
    ]
    if reuse_controls:
        from .review_reuse import load_review_reuse_receipt

        review_calls = sum(
            int(load_review_reuse_receipt(
                root, Path(str(item["path"])),
            )["actual_review_calls"])
            for item in reuse_controls
        )
    return {
        "schema_version": FINAL_COUNTS_VERSION,
        "status": "complete",
        "attribution_status": "partial",
        "segment_basis": [
            {"unit_sha256": value, "published": True}
            for value in basis
        ],
        "segment_counts": {
            "total": len(basis),
            "published": len(basis),
        },
        "review_counts": {
            "receipts": len(review_controls),
            "calls": review_calls,
        },
    }


def _recovery_snapshot_plan(
    root: Path,
    state: Mapping[str, Any],
    *,
    checkpoint: Mapping[str, Any],
) -> tuple[bytes | None, str | None]:
    path = root / ".arc-companion" / "resume-transaction.json"
    if not path.is_file():
        return None, None
    data = _read_file(root, path, max_bytes=MAX_REF_BYTES)
    value = _json_object(data, "provenance_recovery_invalid")
    if (
        value.get("schema_version") != "arc.companion.resume-transaction.v3"
        or value.get("status") != "complete"
        or not str(value.get("checkpoint_path") or "")
        or not str(value.get("checkpoint_fingerprint") or "")
        or not isinstance(value.get("entries"), list)
        or any(
            not isinstance(item, Mapping)
            or item.get("status") != "resolved"
            or not str(item.get("ledger_path") or "")
            or not str(item.get("session_key") or "")
            or not str(item.get("segment_id") or "")
            for item in value.get("entries") or []
        )
    ):
        raise ProvenanceError(
            "provenance_recovery_invalid",
            "resume transaction is not a terminal current-v3 journal",
        )
    checkpoint_path = str(value["checkpoint_path"])
    if _inside(root, Path(checkpoint_path)) != (
        root / str(checkpoint["path"])
    ):
        raise ProvenanceError(
            "provenance_recovery_invalid",
            "resume transaction checkpoint differs",
        )
    fingerprint = str(value["checkpoint_fingerprint"])
    if fingerprint != state.get("fingerprint"):
        raise ProvenanceError(
            "provenance_recovery_invalid",
            "resume transaction fingerprint differs",
        )
    digest = hashlib.sha256(data).hexdigest()
    return (
        data,
        f".arc-companion/resume-transactions/history/{digest}.json",
    )


def _control_ref(
    root: Path,
    path: Path,
    *,
    category: str,
    allowed_schemas: set[str] | None,
    status: str,
    subject_sha256: str | None,
) -> Mapping[str, Any]:
    data = _read_file(root, path, max_bytes=MAX_REF_BYTES)
    value = _json_object(data, "provenance_control_invalid")
    schema = str(value.get("schema_version") or "")
    if not schema or len(schema) > 128 or (
        allowed_schemas is not None and schema not in allowed_schemas
    ):
        raise ProvenanceError(
            "provenance_control_invalid", "control receipt schema is invalid",
        )
    return _memory_ref(
        _relative(root, path),
        data,
        category=category,
        receipt_schema=schema,
        status=status,
        subject_sha256=subject_sha256,
    )


def _memory_ref(
    path: str,
    data: bytes,
    *,
    category: str,
    receipt_schema: str,
    status: str,
    subject_sha256: str | None,
) -> Mapping[str, Any]:
    return {
        "schema_version": RECEIPT_REF_VERSION,
        "category": category,
        "path": _safe_relative(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "receipt_schema": receipt_schema,
        "status": status,
        "subject_sha256": subject_sha256,
    }


def _normalize_controls(
    controls: Iterable[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    records = sorted(
        (dict(item) for item in controls),
        key=lambda item: (
            str(item["category"]),
            str(item.get("subject_sha256") or ""),
            str(item["sha256"]),
            str(item["path"]),
        ),
    )
    if len(records) > MAX_REFS:
        raise ProvenanceError(
            "provenance_limit_exceeded", "too many provenance controls",
        )
    if sum(int(item["bytes"]) for item in records) > MAX_TOTAL_REF_BYTES:
        raise ProvenanceError(
            "provenance_limit_exceeded", "provenance controls exceed byte cap",
        )
    result: list[Mapping[str, Any]] = []
    seen_paths: dict[str, Mapping[str, Any]] = {}
    for record in records:
        _validate_control_shape(record)
        prior = seen_paths.get(str(record["path"]))
        if prior is not None:
            if prior != record:
                raise ProvenanceError(
                    "provenance_control_conflict",
                    "one control path has conflicting identities",
                )
            continue
        seen_paths[str(record["path"])] = record
        result.append(record)
    return tuple(result)


def _validate_state_mapping(value: object) -> Mapping[str, Any]:
    keys = {
        "schema_version", "status", "final_id", "path", "sha256", "bytes",
        "counts_path", "counts_sha256", "counts_bytes",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != keys
        or value.get("schema_version") != PROVENANCE_STATE_VERSION
        or value.get("status") != "complete"
        or not _SHA256.fullmatch(str(value.get("final_id") or ""))
        or not _SHA256.fullmatch(str(value.get("sha256") or ""))
        or not _SHA256.fullmatch(str(value.get("counts_sha256") or ""))
        or not _positive_int(value.get("bytes"))
        or not _positive_int(value.get("counts_bytes"))
    ):
        raise ProvenanceError(
            "provenance_state_partial", "published provenance state is invalid",
        )
    _safe_relative(str(value["path"]))
    _safe_relative(str(value["counts_path"]))
    return value


def _validate_document_shape(value: Mapping[str, Any]) -> None:
    if set(value) != {
        "schema_version", "status", "final_id", "fingerprint", "checkpoint",
        "reviewed_content", "outputs", "controls", "counts", "attribution",
    } or (
        value.get("schema_version") != FINAL_PROVENANCE_VERSION
        or value.get("status") != "complete"
        or not _SHA256.fullmatch(str(value.get("final_id") or ""))
        or not str(value.get("fingerprint") or "")
    ):
        raise ProvenanceError(
            "provenance_object_invalid", "provenance object shape is invalid",
        )
    if not isinstance(value.get("controls"), list):
        raise ProvenanceError(
            "provenance_object_invalid", "provenance controls are invalid",
        )
    if list(_normalize_controls(value["controls"])) != value["controls"]:
        raise ProvenanceError(
            "provenance_object_invalid",
            "provenance controls are not in canonical order",
        )
    if (
        not isinstance(value.get("checkpoint"), Mapping)
        or set(value["checkpoint"])
        != {"identity_sha256", "path", "schema_version"}
        or not _SHA256.fullmatch(
            str(value["checkpoint"].get("identity_sha256") or "")
        )
        or not str(value["checkpoint"].get("schema_version") or "")
        or not isinstance(value.get("reviewed_content"), Mapping)
        or set(value["reviewed_content"]) != {
            "semantic_sha256", "file_sha256", "bytes", "path",
            "schema_version",
        }
        or not _SHA256.fullmatch(
            str(value["reviewed_content"].get("semantic_sha256") or "")
        )
        or not _SHA256.fullmatch(
            str(value["reviewed_content"].get("file_sha256") or "")
        )
        or not _positive_int(value["reviewed_content"].get("bytes"))
        or value["reviewed_content"].get("schema_version")
        != READER_CONTENT_VERSION
        or not isinstance(value.get("outputs"), Mapping)
        or set(value["outputs"]) != {"mode", "content_sha256", *_OUTPUT_KEYS}
        or value["outputs"].get("mode") not in _MODES
        or not _SHA256.fullmatch(
            str(value["outputs"].get("content_sha256") or "")
        )
        or not isinstance(value.get("counts"), Mapping)
        or set(value["counts"])
        != {"path", "sha256", "bytes", "schema_version"}
        or not _SHA256.fullmatch(str(value["counts"].get("sha256") or ""))
        or not _positive_int(value["counts"].get("bytes"))
        or value["counts"].get("schema_version") != FINAL_COUNTS_VERSION
        or not isinstance(value.get("attribution"), Mapping)
        or set(value["attribution"]) != {
            "status", "segment_count", "review_receipt_count", "review_calls",
        }
    ):
        raise ProvenanceError(
            "provenance_object_invalid",
            "provenance nested object shape is invalid",
        )
    for key in _OUTPUT_KEYS:
        record = value["outputs"][key]
        if record is not None and (
            not isinstance(record, Mapping)
            or set(record) != {"path", "sha256", "bytes"}
        ):
            raise ProvenanceError(
                "provenance_object_invalid",
                "provenance output record shape is invalid",
            )
    _safe_relative(str(value["checkpoint"]["path"]))
    _safe_relative(str(value["reviewed_content"]["path"]))
    _safe_relative(str(value["counts"]["path"]))


def _validate_counts(value: Mapping[str, Any]) -> None:
    if (
        set(value) != {
            "schema_version", "status", "attribution_status", "segment_basis",
            "segment_counts", "review_counts",
        }
        or value.get("schema_version") != FINAL_COUNTS_VERSION
        or value.get("status") != "complete"
        or value.get("attribution_status") not in {"complete", "partial"}
        or not isinstance(value.get("segment_basis"), list)
    ):
        raise ProvenanceError(
            "provenance_counts_invalid", "final counts object is invalid",
        )
    basis = value["segment_basis"]
    if len(basis) > MAX_BASIS_RECORDS or any(
        not isinstance(item, Mapping)
        or set(item) != {"unit_sha256", "published"}
        or not _SHA256.fullmatch(str(item.get("unit_sha256") or ""))
        or item.get("published") is not True
        for item in basis
    ):
        raise ProvenanceError(
            "provenance_counts_invalid", "final segment basis is invalid",
        )
    units = [str(item["unit_sha256"]) for item in basis]
    review_counts = value.get("review_counts")
    if (
        units != sorted(set(units))
        or not isinstance(review_counts, Mapping)
        or set(review_counts) != {"receipts", "calls"}
        or type(review_counts.get("receipts")) is not int
        or int(review_counts["receipts"]) < 0
        or (
            review_counts.get("calls") is not None
            and (
                type(review_counts["calls"]) is not int
                or int(review_counts["calls"]) < 0
            )
        )
    ):
        raise ProvenanceError(
            "provenance_counts_invalid", "final review counts are invalid",
        )
    if value.get("segment_counts") != {
        "total": len(basis), "published": len(basis),
    }:
        raise ProvenanceError(
            "provenance_counts_invalid", "final segment counts differ",
        )


def _read_ref(
    root: Path,
    record: Mapping[str, Any],
    *,
    expected_schema: str,
) -> Mapping[str, Any]:
    path = root / _safe_relative(str(record.get("path") or ""))
    data = _read_file(root, path, max_bytes=MAX_REF_BYTES)
    if (
        hashlib.sha256(data).hexdigest() != record.get("sha256")
        or len(data) != record.get("bytes")
    ):
        raise ProvenanceError(
            "provenance_reference_invalid", "provenance reference identity differs",
        )
    value = _json_object(data, "provenance_reference_invalid")
    if value.get("schema_version") != expected_schema:
        raise ProvenanceError(
            "provenance_reference_invalid", "provenance reference schema differs",
        )
    return value


def _validate_control_owner(
    root: Path,
    record: Mapping[str, Any],
    value: Mapping[str, Any],
    *,
    state: Mapping[str, Any],
    envelope: Mapping[str, Any],
) -> None:
    category = str(record["category"])
    if category == "artifact_gc":
        from .gc import load_gc_receipt

        load_gc_receipt(root, str(record["path"]))
    elif category == "review_reuse":
        from .review_reuse import load_review_reuse_receipt

        load_review_reuse_receipt(root, Path(str(record["path"])))
    elif category == "render_validation":
        if (
            value.get("schema_version") != PDF_VALIDATION_RECEIPT_VERSION
            or value.get("result") != "success"
        ):
            raise ProvenanceError(
                "provenance_control_invalid",
                "render-validation control is not successful",
            )
    elif category == "source_snapshot":
        checkpoint = _inside(root, Path(str(state.get("checkpoint_dir") or "")))
        if _validate_source_snapshot(root, checkpoint, value, state) != record[
            "status"
        ]:
            raise ProvenanceError(
                "provenance_source_snapshot_invalid",
                "source snapshot validation status differs",
            )
    elif category == "translation_reference":
        from .translation_reference import (
            TRANSLATION_REFERENCE_VALIDATION_VERSION,
            validate_translation_reference_provenance,
        )

        compact = (
            (envelope.get("content") or {}).get("translation_reference")
            if isinstance(envelope.get("content"), Mapping) else None
        )
        if (
            value.get("schema_version")
            != TRANSLATION_REFERENCE_VALIDATION_VERSION
            or set(value) != {
                "schema_version", "manifest_path", "manifest_sha256",
                "compact_provenance",
            }
            or value.get("compact_provenance") != compact
            or not isinstance(compact, Mapping)
            or value.get("manifest_path") != compact.get("manifest_path")
            or value.get("manifest_sha256") != compact.get("manifest_sha256")
            or value.get("manifest_path")
            != state.get("translation_reference_manifest_path")
            or value.get("manifest_sha256")
            != state.get("translation_reference_manifest_sha256")
        ):
            raise ProvenanceError(
                "provenance_translation_reference_invalid",
                "translation-reference binding differs",
            )
        chapters = (envelope.get("content") or {}).get("chapters") or []
        validate_translation_reference_provenance(
            compact,
            project_root=root,
            expected_chapter_ids=[
                str(item.get("chapter_id") or "")
                for item in chapters if isinstance(item, Mapping)
            ],
        )
    elif category == "recovery_journal":
        if (
            value.get("schema_version")
            != "arc.companion.resume-transaction.v3"
            or value.get("status") != "complete"
            or not str(value.get("checkpoint_path") or "")
            or not str(value.get("checkpoint_fingerprint") or "")
            or not isinstance(value.get("entries"), list)
            or any(
                not isinstance(item, Mapping)
                or item.get("status") != "resolved"
                or not str(item.get("ledger_path") or "")
                or not str(item.get("session_key") or "")
                or not str(item.get("segment_id") or "")
                for item in value.get("entries") or []
            )
        ):
            raise ProvenanceError(
                "provenance_recovery_invalid",
                "recovery history is not terminal current-v3",
            )


def _verify_file_record(root: Path, record: Mapping[str, Any]) -> None:
    if set(record) != {"path", "sha256", "bytes"}:
        raise ProvenanceError(
            "provenance_output_invalid", "output record shape is invalid",
        )
    data = _read_file(root, root / _safe_relative(str(record["path"])),
                      max_bytes=MAX_TOTAL_REF_BYTES)
    if (
        hashlib.sha256(data).hexdigest() != record["sha256"]
        or len(data) != record["bytes"]
    ):
        raise ProvenanceError(
            "provenance_output_invalid", "output identity differs",
        )


def _file_record(root: Path, value: object, expected: str) -> Mapping[str, Any]:
    if not _SHA256.fullmatch(expected):
        raise ProvenanceError(
            "provenance_output_invalid", "output hash is invalid",
        )
    path = _inside(root, Path(str(value)))
    data = _read_file(root, path, max_bytes=MAX_TOTAL_REF_BYTES)
    if hashlib.sha256(data).hexdigest() != expected:
        raise ProvenanceError(
            "provenance_output_invalid", "output hash differs",
        )
    return {"path": _relative(root, path), "sha256": expected, "bytes": len(data)}


def _read_file(root: Path, path: Path, *, max_bytes: int) -> bytes:
    path = _inside(root, path)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ProvenanceError(
            "provenance_reference_invalid", "provenance file is unavailable",
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not os.path.isfile(path) or before.st_nlink != 1:
            raise ProvenanceError(
                "provenance_reference_invalid", "provenance file is not regular",
            )
        if before.st_size < 1 or before.st_size > max_bytes:
            raise ProvenanceError(
                "provenance_limit_exceeded", "provenance file exceeds byte cap",
            )
        data = b""
        while len(data) <= max_bytes:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - len(data)))
            if not chunk:
                break
            data += chunk
        after = os.fstat(descriptor)
        if (
            len(data) > max_bytes
            or (before.st_dev, before.st_ino, before.st_size, before.st_mode)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mode)
        ):
            raise ProvenanceError(
                "provenance_reference_invalid", "provenance file changed while read",
            )
        return data
    finally:
        os.close(descriptor)


def _create_or_adopt(root: Path, path: Path, data: bytes) -> None:
    _inside(root, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read_file(root, path, max_bytes=max(MAX_REF_BYTES, len(data))) != data:
            raise ProvenanceError(
                "provenance_immutable_conflict",
                "immutable provenance object conflicts",
            )
        return
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if _read_file(root, path, max_bytes=max(MAX_REF_BYTES, len(data))) != data:
                raise ProvenanceError(
                    "provenance_immutable_conflict",
                    "immutable provenance object conflicts",
                )
        _fsync(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _inside(root: Path, path: Path) -> Path:
    raw = path if path.is_absolute() else root / path
    try:
        relative = raw.absolute().relative_to(root)
    except ValueError as exc:
        raise ProvenanceError(
            "provenance_path_invalid", "provenance path escapes project",
        ) from exc
    safe = _safe_relative(relative.as_posix())
    current = root
    parts = PurePosixPath(safe).parts
    if len(parts) > MAX_JSON_DEPTH:
        raise ProvenanceError(
            "provenance_path_invalid", "provenance path is too deep",
        )
    for component in parts:
        current = current / component
        if current.is_symlink():
            raise ProvenanceError(
                "provenance_path_invalid", "provenance path contains symlink",
            )
    return root / safe


def _safe_relative(value: str) -> str:
    if (
        not value
        or value in {".", ".."}
        or "\\" in value
        or "\x00" in value
        or len(value.encode("utf-8")) > MAX_PATH_BYTES
    ):
        raise ProvenanceError(
            "provenance_path_invalid", "provenance path is invalid",
        )
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ProvenanceError(
            "provenance_path_invalid", "provenance path is not project-relative",
        )
    return path.as_posix()


def _json_object(data: bytes, code: str) -> Mapping[str, Any]:
    try:
        value = json.loads(data)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ProvenanceError(code, "provenance JSON is malformed") from exc
    if not isinstance(value, Mapping) or _json_depth(value) > MAX_JSON_DEPTH:
        raise ProvenanceError(code, "provenance JSON shape is invalid")
    return value


def _json_depth(value: object) -> int:
    if isinstance(value, Mapping):
        return 1 + max((_json_depth(item) for item in value.values()), default=0)
    if isinstance(value, list):
        return 1 + max((_json_depth(item) for item in value), default=0)
    return 0


def _validate_control_shape(record: Mapping[str, Any]) -> None:
    if (
        set(record) != {
            "schema_version", "category", "path", "sha256", "bytes",
            "receipt_schema", "status", "subject_sha256",
        }
        or record.get("schema_version") != RECEIPT_REF_VERSION
        or not str(record.get("category") or "")
        or not _SHA256.fullmatch(str(record.get("sha256") or ""))
        or not _positive_int(record.get("bytes"))
        or int(record["bytes"]) > MAX_REF_BYTES
        or not str(record.get("receipt_schema") or "")
        or not str(record.get("status") or "")
        or (
            record.get("subject_sha256") is not None
            and not _SHA256.fullmatch(str(record["subject_sha256"]))
        )
    ):
        raise ProvenanceError(
            "provenance_control_invalid", "control reference shape is invalid",
        )
    _safe_relative(str(record["path"]))


def _without_path(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return {key: item for key, item in value.items() if key != "path"}


def _subject(value: str) -> str | None:
    return value if _SHA256.fullmatch(value) else (
        hashlib.sha256(value.encode("utf-8")).hexdigest() if value else None
    )


def _relative(root: Path, path: Path) -> str:
    return _safe_relative(path.absolute().relative_to(root).as_posix())


def _json_bytes(value: object) -> bytes:
    return canonical_json(value).encode("utf-8") + b"\n"


def _sha_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _plan_projection(plan: ProvenancePlan) -> Mapping[str, Any]:
    return {
        "mode": plan.mode,
        "fingerprint": plan.fingerprint,
        "checkpoint": dict(plan.checkpoint),
        "reviewed_content": dict(plan.reviewed_content),
        "outputs": dict(plan.outputs),
        "controls": [dict(item) for item in plan.controls],
        "counts": dict(plan.counts),
        "recovery_path": plan.recovery_path,
        "recovery_sha256": (
            hashlib.sha256(plan.recovery_bytes).hexdigest()
            if plan.recovery_bytes is not None else None
        ),
    }


def _semantic_projection(
    *,
    fingerprint: str,
    checkpoint: Mapping[str, Any],
    reviewed_content: Mapping[str, Any],
    outputs: Mapping[str, Any],
    controls: Iterable[Mapping[str, Any]],
    counts: Mapping[str, Any],
) -> Mapping[str, Any]:
    return {
        "schema_version": FINAL_PROVENANCE_VERSION,
        "fingerprint": fingerprint,
        "checkpoint": _without_path(checkpoint),
        "reviewed_content": _without_path(reviewed_content),
        "outputs": {
            "mode": outputs["mode"],
            "content_sha256": outputs["content_sha256"],
            **{
                key: _without_path(outputs[key])
                for key in _OUTPUT_KEYS
                if outputs.get(key) is not None
            },
        },
        "controls": [_without_path(record) for record in controls],
        "counts": _without_path(counts),
    }


def _fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
