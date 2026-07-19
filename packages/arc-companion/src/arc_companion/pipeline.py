from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import uuid
from typing import Any, Callable

from bs4 import BeautifulSoup

from .glossary import generate_glossary
from .domain import load_domain_context
from .evidence import (
    arc_cache_descriptor,
    text_sha256,
    validate_annotation_citations,
    validate_cited_ids,
    validate_evidence_record,
)
from .evidence_requests import (
    EvidenceRequestController,
    EvidenceResolution,
    normalize_evidence_requests,
)
from .io import read_json, safe_name, sha256_file, sha256_json, write_json, write_text
from .latex import LatexError, render_companion_tex, validate_tex_fidelity
from .pdf import compile_latex, validate_pdf
from .prompts import (
    ANNOTATION_SCHEMA,
    TRANSLATION_COVERAGE_REPAIR_SCHEMA,
    TRANSLATION_SCHEMA,
    TRANSLATION_SLOT_REPAIR_SCHEMA,
    REVIEW_SCHEMA,
    SECTION_REVIEW_SCHEMA,
    PROMPT_VERSION,
    SCHEMA_VERSION,
    TRANSLATION_COVERAGE_REPAIR_PROMPT_VERSION,
    TRANSLATION_COVERAGE_REPAIR_SCHEMA_VERSION,
    TRANSLATION_RETRY_PROMPT_VERSION,
    annotation_prompt,
    review_prompt,
    section_review_prompt,
    translation_coverage_repair_prompt,
    translation_prompt,
    translation_retry_prompt,
)
from .projection import (
    annotation_input_block as _project_annotation_input_block,
    is_translatable as _project_is_translatable,
    opaque_inline_token as _project_opaque_inline_token,
    opaque_inline_tokens as _project_opaque_inline_tokens,
    prompt_safe_value as _project_prompt_safe_value,
    translation_input_block as _project_translation_input_block,
)
from .results import err, ok
from .segmentation import SegmentationError, segment_document, validate_exact_coverage
from .source import SourceBundle, SourceError, block_id, load_source_bundle


WORKFLOW_VERSION = "arc.companion.workflow.v5"
DEFAULT_LANGUAGE = "zh-CN"
DEFAULT_WORKERS = 24
DEFAULT_REVIEW_CONTEXT_CHARS = 140_000
LANGUAGE_NOTICE = "默认使用中文生成伴读；如需切换伴读语言，请直接指定目标语言。"
SEGMENTATION_TIER = "medium"
GLOSSARY_TIER = "medium"
TRANSLATION_TIER = "low"
TRANSLATION_RETRY_TIER = "medium"
TRANSLATION_COVERAGE_REPAIR_TIER = "medium"
ANNOTATION_TIER = "high"
REVIEW_TIER = "high"
REVIEW_VERSION = "arc.companion.review.v2"
FULL_PAPER_CONTEXT_VERSION = "arc.companion.full-paper-context.v1"
FULL_PAPER_CONTEXT_CHARS = 24_000
FIRST_WAVE_PREVIEW_VERSION = "arc.companion.first-wave-preview.v1"


class CompanionLaneError(RuntimeError):
    """Raised after a generation lane checkpoints all successes and aggregates failures."""

    def __init__(self, lane: str, failures: list[tuple[str, BaseException]]) -> None:
        self.lane = lane
        self.failures = list(failures)
        detail = "; ".join(
            f"{segment_id}: {type(exc).__name__}: {exc}" for segment_id, exc in failures
        )
        super().__init__(f"{lane} lane failed for {len(failures)} segment(s): {detail}")


class CompanionGenerationError(RuntimeError):
    """Raised after both generation lanes finish and one or both report failures."""

    def __init__(self, failures: dict[str, BaseException]) -> None:
        self.failures = dict(failures)
        detail = " | ".join(
            f"{lane}: {type(exc).__name__}: {exc}" for lane, exc in sorted(failures.items())
        )
        super().__init__(f"per-segment generation failed after draining both lanes: {detail}")


class TranslationOpaqueTokenError(RuntimeError):
    """A generated translation changed the controller-owned inline token stream."""

    def __init__(
        self,
        *,
        segment_id: str,
        block_id_value: str,
        expected_tokens: list[str],
        actual_tokens: list[str],
    ) -> None:
        self.segment_id = segment_id
        self.block_id = block_id_value
        self.expected_tokens = list(expected_tokens)
        self.actual_tokens = list(actual_tokens)
        super().__init__(
            f"translation {segment_id} changed, dropped, or reordered opaque inline tokens "
            f"in {block_id_value} (expected {len(expected_tokens)}, received {len(actual_tokens)})"
        )

    def prompt_payload(self) -> dict[str, Any]:
        return {
            "type": "opaque_inline_token_mismatch",
            "retry_prompt_version": TRANSLATION_RETRY_PROMPT_VERSION,
            "retry_model_tier": TRANSLATION_RETRY_TIER,
            "segment_id": self.segment_id,
            "block_id": self.block_id,
            "expected_token_count": len(self.expected_tokens),
            "actual_token_count": len(self.actual_tokens),
            "message": str(self),
        }


class TranslationCoverageError(RuntimeError):
    """A translation candidate does not map exactly once to every expected block."""


@dataclass(frozen=True)
class BuildOptions:
    paper_id: str
    project_dir: Path
    annotation_language: str = DEFAULT_LANGUAGE
    language_was_defaulted: bool = False
    provider: str = "auto"
    model: str | None = None
    workers: int = DEFAULT_WORKERS
    refresh: bool = False
    recache: bool = False
    force: bool = False
    review_context_chars: int = DEFAULT_REVIEW_CONTEXT_CHARS
    domain_id: str | None = None
    domain_manifest: Path | None = None

    def __post_init__(self) -> None:
        if not self.paper_id.strip():
            raise ValueError("paper_id is required")
        if self.workers < 1:
            raise ValueError("workers must be at least 1")
        if self.refresh and self.recache:
            raise ValueError("refresh and recache are mutually exclusive")
        if self.domain_id and self.domain_manifest is not None:
            raise ValueError("domain_id and domain_manifest are mutually exclusive")


def build_companion(
    options: BuildOptions,
    *,
    source_loader: Callable[..., SourceBundle] = load_source_bundle,
    llm: Callable[..., dict[str, Any]] | None = None,
    compiler: Callable[[Path, Path], None] = compile_latex,
    pdf_validator: Callable[[Path], dict[str, object]] = validate_pdf,
    evidence_controller: EvidenceRequestController | None = None,
) -> dict[str, Any]:
    """Build or resume one companion while keeping source and annotations separate."""
    if llm is None:
        from arc_llm import run_json

        llm = run_json
    project_dir = options.project_dir.resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    state_path = project_dir / "state.json"
    notice = LANGUAGE_NOTICE if options.language_was_defaulted else None
    previous_state = _read_optional_json(state_path)
    diagnostics: tuple[dict[str, str], ...] = ()
    _state(
        state_path,
        status="loading_source",
        paper_id=options.paper_id,
        notice=notice,
        diagnostics=[],
    )

    try:
        bundle = source_loader(options.paper_id, refresh=options.refresh, recache=options.recache)
        generation_document = _generation_document(bundle.document)
        diagnostics = bundle.diagnostics
        evidence = _evidence(bundle)
        domain_context = load_domain_context(
            domain_id=options.domain_id,
            domain_manifest=options.domain_manifest,
        )
        fingerprint = _fingerprint(bundle, options, evidence=evidence, domain_context=domain_context)
        checkpoint_dir = project_dir / ".arc-companion" / "checkpoints" / fingerprint
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if (
            not options.force
            and previous_state.get("status") == "complete"
            and previous_state.get("fingerprint") == fingerprint
            and _completion_outputs_match(previous_state)
            and _first_wave_preview_outputs_match(previous_state, workers=options.workers)
        ):
            resumed_state = {**previous_state, "diagnostics": list(diagnostics)}
            _state(state_path, **resumed_state)
            return ok(
                resumed_state,
                resumed=True,
                notice=notice,
                diagnostics=list(diagnostics),
            )

        write_json(checkpoint_dir / "document.json", bundle.parsed)
        write_json(checkpoint_dir / "evidence.json", evidence)
        if domain_context is not None:
            write_json(checkpoint_dir / "domain-context.json", domain_context)
        _state(
            state_path,
            status="segmenting",
            paper_id=bundle.paper_id,
            fingerprint=fingerprint,
            notice=notice,
            diagnostics=list(diagnostics),
        )

        initial_names = _protected_names(bundle)

        def segment() -> list[dict[str, Any]]:
            return segment_document(
                generation_document,
                checkpoint_dir=checkpoint_dir,
                workers=options.workers,
                force=options.force,
                call_model=lambda prompt, schema, artifact_dir, call_label: _llm_call(
                    llm, prompt, schema, options=options, artifact_dir=artifact_dir,
                    call_label=call_label, model_tier=SEGMENTATION_TIER,
                ),
            )

        def glossary_task() -> dict[str, Any]:
            return generate_glossary(
                generation_document,
                language=options.annotation_language,
                protected_names=initial_names,
                checkpoint_dir=checkpoint_dir,
                workers=options.workers,
                force=options.force,
                page_count=_page_count(bundle),
                call_model=lambda prompt, schema, artifact_dir, call_label: _llm_call(
                    llm, prompt, schema, options=options, artifact_dir=artifact_dir,
                    call_label=call_label, model_tier=GLOSSARY_TIER,
                ),
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            segmentation_future = executor.submit(segment)
            glossary_future = executor.submit(glossary_task)
            expanded = segmentation_future.result()
            glossary = glossary_future.result()
        protected_names = _protected_names(bundle, glossary=glossary)

        _state(
            state_path,
            status="generating",
            paper_id=bundle.paper_id,
            fingerprint=fingerprint,
            notice=notice,
            segment_count=len(expanded),
            checkpoint_dir=str(checkpoint_dir),
        )
        first_wave_count = min(options.workers, len(expanded))
        first_wave = expanded[:first_wave_count]
        remaining = expanded[first_wave_count:]
        first_wave_results = _generate_first_round_lanes(
            first_wave,
            options=options,
            bundle=bundle,
            evidence=evidence,
            domain_context=domain_context,
            glossary=glossary,
            protected_names=protected_names,
            checkpoint_dir=checkpoint_dir,
            llm=llm,
        )
        translations = first_wave_results["translation"]
        raw_annotations = first_wave_results["annotation"]
        write_json(checkpoint_dir / "translations.first-wave.v1.json", {
            "schema_version": "arc.companion.translations-first-wave.v1",
            "segment_ids": [str(item["segment_id"]) for item in first_wave],
            "translations": translations,
        })
        write_json(checkpoint_dir / "annotations.first-wave.v1.json", {
            "schema_version": "arc.companion.annotations-first-wave.v1",
            "segment_ids": [str(item["segment_id"]) for item in first_wave],
            "annotations": raw_annotations,
        })
        stem = f"{safe_name(bundle.paper_id)}_companion_{safe_name(options.annotation_language)}"
        preview = _publish_pdf_artifact(
            document=_first_wave_preview_document(bundle.document, first_wave),
            segments=first_wave,
            annotations=raw_annotations,
            translations=translations,
            glossary=glossary,
            metadata=bundle.metadata,
            language=options.annotation_language,
            output_dir=project_dir,
            stem=f"{stem}_first_round_preview",
            manifest_name="first-round-preview-source-manifest.json",
            validation_name="first-round-preview-validation.json",
            compiler=compiler,
            pdf_validator=pdf_validator,
        )
        _state(
            state_path,
            status="preview_ready",
            paper_id=bundle.paper_id,
            fingerprint=fingerprint,
            notice=notice,
            segment_count=len(expanded),
            preview_segment_count=first_wave_count,
            preview_segment_ids=[str(item["segment_id"]) for item in first_wave],
            first_wave_preview_version=FIRST_WAVE_PREVIEW_VERSION,
            preview_tex=preview["tex_path"],
            preview_pdf=preview["pdf_path"],
            preview_tex_sha256=preview["tex_sha256"],
            preview_pdf_sha256=preview["pdf_sha256"],
            preview_source_manifest_path=preview["manifest_path"],
            preview_source_manifest_sha256=preview["manifest_sha256"],
            preview_validation_path=preview["validation_path"],
            preview_validation_sha256=preview["validation_sha256"],
        )
        if remaining:
            remaining_results = _generate_first_round_lanes(
                remaining,
                options=options,
                bundle=bundle,
                evidence=evidence,
                domain_context=domain_context,
                glossary=glossary,
                protected_names=protected_names,
                checkpoint_dir=checkpoint_dir,
                llm=llm,
            )
            translations = {**translations, **remaining_results["translation"]}
            raw_annotations = {**raw_annotations, **remaining_results["annotation"]}

        write_json(checkpoint_dir / "annotations.first-round.v1.json", {
            "schema_version": "arc.companion.annotations-first-round.v1",
            "annotations": raw_annotations,
        })
        raw_annotations, evidence = _resolve_and_rerun_evidence_requests(
            expanded,
            raw_annotations,
            options=options,
            bundle=bundle,
            evidence=evidence,
            domain_context=domain_context,
            glossary=glossary,
            protected_names=protected_names,
            checkpoint_dir=checkpoint_dir,
            llm=llm,
            controller=evidence_controller or EvidenceRequestController(),
        )

        _state(state_path, status="reviewing", paper_id=bundle.paper_id, fingerprint=fingerprint, notice=notice,
               segment_count=len(expanded))
        reviewed_path = checkpoint_dir / "annotations.reviewed.v2.json"
        review_path = checkpoint_dir / "review.v2.json"
        if reviewed_path.is_file() and review_path.is_file() and not options.force:
            cached_reviewed = read_json(reviewed_path)
            review = read_json(review_path)
            if (
                not isinstance(cached_reviewed, dict)
                or cached_reviewed.get("schema_version") != REVIEW_VERSION
            ):
                raise RuntimeError("invalid review checkpoint")
            reviewed = cached_reviewed.get("annotations")
            reviewed_translations = cached_reviewed.get("translations")
            if (
                not isinstance(reviewed, dict)
                or not isinstance(reviewed_translations, dict)
                or set(reviewed) != {segment["segment_id"] for segment in expanded}
                or set(reviewed_translations) != set(reviewed)
            ):
                raise RuntimeError("review checkpoint does not match current segments")
        else:
            reviewed_translations, reviewed, review = _review(
                expanded,
                translations,
                raw_annotations,
                document=bundle.document,
                glossary=glossary,
                protected_names=protected_names,
                evidence=evidence,
                options=options,
                llm=llm,
                checkpoint_dir=checkpoint_dir,
            )
            write_json(reviewed_path, {
                "schema_version": REVIEW_VERSION,
                "translations": reviewed_translations,
                "annotations": reviewed,
            })
            write_json(review_path, review)

        _state(state_path, status="typesetting", paper_id=bundle.paper_id, fingerprint=fingerprint, notice=notice,
               segment_count=len(expanded))
        final_artifact = _publish_pdf_artifact(
            document=bundle.document,
            segments=expanded,
            annotations=reviewed,
            translations=reviewed_translations,
            glossary=glossary,
            metadata=bundle.metadata,
            language=options.annotation_language,
            output_dir=project_dir,
            stem=stem,
            manifest_name="source-manifest.json",
            validation_name="validation.json",
            compiler=compiler,
            pdf_validator=pdf_validator,
        )
        tex_path = Path(final_artifact["tex_path"])
        pdf_path = Path(final_artifact["pdf_path"])
        manifest_path = Path(final_artifact["manifest_path"])
        validation_path = Path(final_artifact["validation_path"])

        final_state = _state(
            state_path,
            status="complete",
            paper_id=bundle.paper_id,
            fingerprint=fingerprint,
            notice=notice,
            segment_count=len(expanded),
            output_tex=str(tex_path),
            output_pdf=str(pdf_path),
            output_tex_sha256=final_artifact["tex_sha256"],
            output_pdf_sha256=final_artifact["pdf_sha256"],
            source_manifest_sha256=final_artifact["manifest_sha256"],
            validation_sha256=final_artifact["validation_sha256"],
            source_manifest_path=str(manifest_path),
            validation_path=str(validation_path),
            checkpoint_dir=str(checkpoint_dir),
            diagnostics=list(diagnostics),
        )
        return ok(
            final_state,
            resumed=False,
            notice=notice,
            diagnostics=list(diagnostics),
        )
    except Exception as exc:
        failure_diagnostics: list[dict[str, Any]] = [dict(item) for item in diagnostics]
        if isinstance(exc, SegmentationError):
            failure_diagnostics.append(exc.diagnostic())
        failure = _state(
            state_path,
            status="failed",
            paper_id=options.paper_id,
            notice=notice,
            error=str(exc),
            diagnostics=failure_diagnostics,
        )
        return err(
            "companion_segmentation_failed" if isinstance(exc, SegmentationError) else "companion_build_failed",
            str(exc),
            state=failure,
            notice=notice,
            diagnostics=failure_diagnostics,
        )


def validate_and_expand_segments(
    segments: list[dict[str, Any]], blocks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not isinstance(segments, list) or not segments:
        raise ValueError("segmentation returned no segments")
    ordered_ids = [block_id(block) for block in blocks]
    positions = {value: index for index, value in enumerate(ordered_ids)}
    seen_segment_ids: set[str] = set()
    cursor = 0
    expanded: list[dict[str, Any]] = []
    for item in segments:
        segment_id = str(item.get("segment_id") or "")
        start = str(item.get("start_block_id") or "")
        end = str(item.get("end_block_id") or "")
        if not segment_id or segment_id in seen_segment_ids:
            raise ValueError(f"invalid or duplicate segment id: {segment_id}")
        if start not in positions or end not in positions:
            raise ValueError(f"segment {segment_id} references an unknown block")
        first, last = positions[start], positions[end]
        if first != cursor or last < first:
            raise ValueError(f"segment {segment_id} is not the next contiguous source range")
        block_ids = ordered_ids[first : last + 1]
        expanded.append({**item, "segment_id": segment_id, "block_ids": block_ids})
        seen_segment_ids.add(segment_id)
        cursor = last + 1
    if cursor != len(ordered_ids):
        raise ValueError("segmentation does not cover every source block exactly once")
    validate_exact_coverage(expanded, blocks)
    return expanded


def read_status(project_dir: Path) -> dict[str, Any]:
    path = project_dir.resolve() / "state.json"
    if not path.is_file():
        return err("companion_state_not_found", f"No companion state found in {project_dir}")
    return ok(read_json(path))


def _generate_first_round_lanes(
    segments: list[dict[str, Any]],
    *,
    options: BuildOptions,
    bundle: SourceBundle,
    evidence: dict[str, Any],
    domain_context: dict[str, Any] | None,
    glossary: dict[str, Any],
    protected_names: list[str],
    checkpoint_dir: Path,
    llm: Callable[..., dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Drain translation and annotation concurrently for one scheduled wave."""
    if not segments:
        return {"translation": {}, "annotation": {}}
    with ThreadPoolExecutor(max_workers=2) as executor:
        lane_futures = {
            "translation": executor.submit(
                _generate_translations,
                segments,
                options=options,
                bundle=bundle,
                glossary=glossary,
                protected_names=protected_names,
                checkpoint_dir=checkpoint_dir,
                llm=llm,
            ),
            "annotation": executor.submit(
                _generate_annotations,
                segments,
                options=options,
                bundle=bundle,
                evidence=evidence,
                domain_context=domain_context,
                glossary=glossary,
                protected_names=protected_names,
                checkpoint_dir=checkpoint_dir,
                llm=llm,
            ),
        }
        lane_results: dict[str, dict[str, dict[str, Any]]] = {}
        lane_failures: dict[str, BaseException] = {}
        for lane, future in lane_futures.items():
            try:
                lane_results[lane] = future.result()
            except Exception as exc:
                lane_failures[lane] = exc
        if lane_failures:
            raise CompanionGenerationError(lane_failures)
    return lane_results


def _first_wave_preview_document(
    document: dict[str, Any], segments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return a source-faithful prefix ending at the first wave's last block."""
    if not segments:
        raise ValueError("the first preview wave must contain at least one segment")
    blocks = list(document.get("blocks") or [])
    positions = {block_id(block): index for index, block in enumerate(blocks)}
    wave_block_ids = [
        str(value)
        for segment in segments
        for value in segment.get("block_ids") or []
    ]
    try:
        last_index = max(positions[value] for value in wave_block_ids)
    except (KeyError, ValueError) as exc:
        raise ValueError("first preview wave does not map to source blocks") from exc
    prefix_blocks = blocks[:last_index + 1]

    block_lookup_keys = (
        "entity_id", "source_id", "equation_id", "figure_id", "table_id", "id", "block_id",
    )
    entity_lookup_keys = (
        "id", "block_id", "source_id", "entity_id", "equation_id", "figure_id", "table_id",
    )

    def matching_entities(items: list[dict[str, Any]], kinds: set[str]) -> list[dict[str, Any]]:
        index: dict[str, int] = {}
        for index_number, item in enumerate(items):
            for key in entity_lookup_keys:
                if item.get(key):
                    index[str(item[key])] = index_number
        selected: set[int] = set()
        for block in prefix_blocks:
            kind = str(block.get("kind") or block.get("type") or "").casefold()
            if kind not in kinds:
                continue
            for key in block_lookup_keys:
                value = block.get(key)
                if value is not None and str(value) in index:
                    selected.add(index[str(value)])
                    break
        return [item for index_number, item in enumerate(items) if index_number in selected]

    equations = matching_entities(
        list(document.get("equations") or []), {"equation", "math", "display_math"},
    )
    figures = matching_entities(
        list(document.get("figures") or []), {"figure", "image"},
    )
    tables = matching_entities(list(document.get("tables") or []), {"table"})
    asset_ids: set[str] = set()
    for block in prefix_blocks:
        values = block.get("asset_ids") or (
            [block.get("asset_id")] if block.get("asset_id") else []
        )
        asset_ids.update(str(value) for value in values if value)
    for figure in figures:
        values = figure.get("asset_ids") or (
            [figure.get("asset_id")] if figure.get("asset_id") else []
        )
        asset_ids.update(str(value) for value in values if value)

    preview_document = dict(document)
    preview_document["blocks"] = prefix_blocks
    preview_document["equations"] = equations
    preview_document["figures"] = figures
    preview_document["tables"] = tables
    preview_document["assets"] = [
        item
        for item in document.get("assets") or []
        if str(item.get("asset_id") or item.get("id") or "") in asset_ids
    ]
    preview_document["bibliography"] = []
    prefix_links: list[dict[str, str]] = []
    for block in prefix_blocks:
        soup = BeautifulSoup(str(block.get("html") or ""), "html.parser")
        for anchor in soup.find_all("a"):
            href = str(anchor.get("href") or "").strip()
            if href:
                prefix_links.append({
                    "href": href,
                    "target_id": href[1:] if href.startswith("#") else "",
                    "text": " ".join(anchor.get_text(" ", strip=True).split()),
                })
    preview_document["links"] = prefix_links
    preview_document["preview_scope"] = {"kind": "source_prefix"}
    return preview_document


def _publish_pdf_artifact(
    *,
    document: dict[str, Any],
    segments: list[dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    translations: dict[str, dict[str, Any]],
    glossary: dict[str, Any],
    metadata: dict[str, Any],
    language: str,
    output_dir: Path,
    stem: str,
    manifest_name: str,
    validation_name: str,
    compiler: Callable[[Path, Path], None],
    pdf_validator: Callable[[Path], dict[str, object]],
) -> dict[str, Any]:
    """Render, validate, and atomically publish one preview or final PDF artifact."""
    tex, source_manifest = render_companion_tex(
        document,
        segments,
        annotations,
        output_dir=output_dir,
        language=language,
        metadata=metadata,
        translations=translations,
        glossary=glossary,
    )
    fidelity_errors = validate_tex_fidelity(tex, document, source_manifest)
    if fidelity_errors:
        raise LatexError("source fidelity validation failed: " + "; ".join(fidelity_errors))

    tex_path = output_dir / f"{stem}.tex"
    pdf_path = output_dir / f"{stem}.pdf"
    manifest_path = output_dir / manifest_name
    validation_path = output_dir / validation_name
    staging_stem = f"arc-companion-building-{safe_name(stem)}-{uuid.uuid4().hex[:12]}"
    building_tex = output_dir / f"{staging_stem}.tex"
    building_pdf = output_dir / f"{staging_stem}.pdf"
    building_manifest = output_dir / f"{staging_stem}-manifest.json"
    building_validation = output_dir / f"{staging_stem}-validation.json"
    staging_paths = (building_tex, building_pdf, building_manifest, building_validation)
    try:
        write_text(building_tex, tex)
        compiler(building_tex, building_pdf)
        pdf_report = pdf_validator(building_pdf)
        write_json(building_manifest, source_manifest)
        write_json(
            building_validation,
            {"ok": True, "pdf": pdf_report, "fidelity_errors": []},
        )
        os.replace(building_tex, tex_path)
        os.replace(building_pdf, pdf_path)
        os.replace(building_manifest, manifest_path)
        os.replace(building_validation, validation_path)
    except BaseException:
        for path in staging_paths:
            path.unlink(missing_ok=True)
        raise

    return {
        "tex_path": str(tex_path),
        "pdf_path": str(pdf_path),
        "manifest_path": str(manifest_path),
        "validation_path": str(validation_path),
        "tex_sha256": _sha256_existing_file(tex_path),
        "pdf_sha256": _sha256_existing_file(pdf_path),
        "manifest_sha256": _sha256_existing_file(manifest_path),
        "validation_sha256": _sha256_existing_file(validation_path),
        "pdf": pdf_report,
    }


def validate_project(project_dir: Path, *, pdf_validator: Callable[[Path], dict[str, object]] = validate_pdf) -> dict[str, Any]:
    status = read_status(project_dir)
    if not status.get("ok"):
        return status
    state = status["data"]
    pdf = Path(str(state.get("output_pdf") or ""))
    tex = Path(str(state.get("output_tex") or ""))
    manifest_path = project_dir.resolve() / "source-manifest.json"
    if not tex.is_file() or not manifest_path.is_file():
        return err("companion_validation_failed", "TeX or source manifest is missing")
    try:
        if not _completion_outputs_match(state):
            raise RuntimeError("completed companion outputs do not match their recorded hashes")
        checkpoint_dir = Path(str(state.get("checkpoint_dir") or ""))
        checkpoint_document = read_json(checkpoint_dir / "document.json")
        document = checkpoint_document.get("document") if isinstance(checkpoint_document, dict) else None
        if not isinstance(document, dict):
            raise RuntimeError("checkpoint source document is missing")
        manifest = read_json(manifest_path)
        fidelity_errors = validate_tex_fidelity(tex.read_text(encoding="utf-8"), document, manifest)
        if fidelity_errors:
            raise RuntimeError("source fidelity validation failed: " + "; ".join(fidelity_errors))
        report = pdf_validator(pdf)
    except (OSError, RuntimeError, ValueError) as exc:
        return err("companion_validation_failed", str(exc))
    return ok({"pdf": report, "manifest": manifest, "output_pdf": str(pdf)})


def _generate_annotations(
    segments: list[dict[str, Any]],
    *,
    options: BuildOptions,
    bundle: SourceBundle,
    evidence: dict[str, Any],
    domain_context: dict[str, Any] | None,
    glossary: dict[str, Any],
    protected_names: list[str],
    checkpoint_dir: Path,
    llm: Callable[..., dict[str, Any]],
    round_number: int = 1,
    first_drafts: dict[str, dict[str, Any]] | None = None,
    resolution_by_segment: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    by_id = {block_id(block): block for block in bundle.document["blocks"]}
    annotation_dir = checkpoint_dir / ("annotations" if round_number == 1 else "annotations-evidence-rerun")
    segment_evidence_dir = checkpoint_dir / "segment-evidence"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    segment_evidence_dir.mkdir(parents=True, exist_ok=True)
    output: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for segment in segments:
        path = annotation_dir / f"{_segment_checkpoint_name(segment['segment_id'])}.json"
        segment_evidence = _evidence_for_segment(segment, by_id, evidence)
        write_json(
            segment_evidence_dir / f"{_segment_checkpoint_name(segment['segment_id'])}.json",
            {
                "schema_version": "arc.companion.segment-evidence-checkpoint.v1",
                "segment_id": segment["segment_id"],
                "input_sha256": sha256_json(segment_evidence),
                "evidence": segment_evidence,
            },
        )
        if path.is_file() and not options.force:
            checkpoint = read_json(path)
            paper_context = _full_paper_context(bundle.document, segment, blocks_by_id=by_id)
            expected_hash = _segment_input_hash(
                segment, by_id, glossary=glossary,
                extra={
                    "evidence": segment_evidence,
                    "names": protected_names,
                    "paper_context": paper_context,
                    "runtime_access": _generation_runtime_policy(),
                    "domain_context": domain_context,
                    "round": round_number,
                    "first_draft": (first_drafts or {}).get(str(segment["segment_id"])),
                    "evidence_resolution": (resolution_by_segment or {}).get(str(segment["segment_id"])),
                },
            )
            if (
                isinstance(checkpoint, dict)
                and checkpoint.get("segment_id") == segment["segment_id"]
                and checkpoint.get("input_sha256") == expected_hash
                and isinstance(checkpoint.get("annotation"), dict)
                and checkpoint["annotation"].get("commentary")
            ):
                output[segment["segment_id"]] = checkpoint["annotation"]
                continue
        pending.append(segment)

    def generate(segment: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        selected = [_annotation_input_block(by_id[value], bundle.document) for value in segment["block_ids"]]
        segment_evidence = _evidence_for_segment(segment, by_id, evidence)
        paper_context = _full_paper_context(bundle.document, segment, blocks_by_id=by_id)
        value = _llm_call(
            llm,
            annotation_prompt(
                segment,
                selected,
                language=options.annotation_language,
                metadata=_annotation_metadata(bundle.metadata),
                evidence=segment_evidence,
                glossary=glossary,
                protected_names=protected_names,
                paper_context=paper_context,
                domain_context=domain_context,
                first_draft=(first_drafts or {}).get(str(segment["segment_id"])),
                evidence_resolution=(resolution_by_segment or {}).get(str(segment["segment_id"])),
            ),
            ANNOTATION_SCHEMA,
            options=options,
            artifact_dir=(
                checkpoint_dir / "llm"
                / ("annotations" if round_number == 1 else "annotations-evidence-rerun")
                / _segment_checkpoint_name(segment["segment_id"])
            ),
            call_label=(
                f"companion-annotation-{segment['segment_id']}"
                if round_number == 1
                else f"companion-annotation-evidence-rerun-{segment['segment_id']}"
            ),
            model_tier=ANNOTATION_TIER,
            allow_mcp=True,
            allow_internet=True,
        )
        if not str(value.get("commentary") or "").strip():
            raise RuntimeError(f"empty commentary for {segment['segment_id']}")
        normalized = {
            "commentary": str(value["commentary"]),
            "explanation": str(value.get("explanation") or value["commentary"]),
            "prior_work": str(value.get("prior_work") or ""),
            "later_work": str(value.get("later_work") or ""),
            "evidence_ids": _validated_evidence_ids(
                (
                    value.get("evidence_ids") or []
                    if round_number == 1
                    else _known_evidence_ids(value.get("evidence_ids") or [], segment_evidence["papers"])
                ),
                {"related_papers": segment_evidence["papers"]},
            ),
            "key_points": list(value.get("key_points") or []),
            "source_notes": list(value.get("source_notes") or []),
            "evidence_requests": (
                normalize_evidence_requests(segment["segment_id"], value.get("evidence_requests") or [])
                if round_number == 1 else []
            ),
        }
        if round_number > 1:
            _drop_unsupported_second_round_related_work(normalized, segment_evidence["papers"])
        _validate_annotation_evidence(normalized, segment_evidence["papers"])
        return segment["segment_id"], normalized

    with ThreadPoolExecutor(max_workers=min(options.workers, max(1, len(pending)))) as executor:
        futures = {executor.submit(generate, segment): segment for segment in pending}
        failures: list[tuple[str, BaseException]] = []
        for future in as_completed(futures):
            segment = futures[future]
            try:
                segment_id, value = future.result()
            except Exception as exc:
                failures.append((str(segment["segment_id"]), exc))
                continue
            else:
                output[segment_id] = value
                segment = next(item for item in segments if item["segment_id"] == segment_id)
                write_json(
                    annotation_dir / f"{_segment_checkpoint_name(segment_id)}.json",
                    {
                        "schema_version": "arc.companion.annotation-checkpoint.v3",
                        "round": round_number,
                        "segment_id": segment_id,
                        "input_sha256": _segment_input_hash(
                            segment, by_id, glossary=glossary,
                            extra={
                                "evidence": _evidence_for_segment(segment, by_id, evidence),
                                "names": protected_names,
                                "paper_context": _full_paper_context(
                                    bundle.document, segment, blocks_by_id=by_id
                                ),
                                "runtime_access": _generation_runtime_policy(),
                                "domain_context": domain_context,
                                "round": round_number,
                                "first_draft": (first_drafts or {}).get(str(segment_id)),
                                "evidence_resolution": (resolution_by_segment or {}).get(str(segment_id)),
                            },
                        ),
                        "annotation": value,
                    },
                )
        if failures:
            raise CompanionLaneError("annotation", failures)
    return {segment["segment_id"]: output[segment["segment_id"]] for segment in segments}


def _resolve_and_rerun_evidence_requests(
    segments: list[dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    *,
    options: BuildOptions,
    bundle: SourceBundle,
    evidence: dict[str, Any],
    domain_context: dict[str, Any] | None,
    glossary: dict[str, Any],
    protected_names: list[str],
    checkpoint_dir: Path,
    llm: Callable[..., dict[str, Any]],
    controller: EvidenceRequestController,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    requests = [
        dict(request)
        for segment in segments
        for request in annotations[str(segment["segment_id"])].get("evidence_requests") or []
    ]
    if not requests:
        write_json(checkpoint_dir / "evidence-resolution.v1.json", {
            "schema_version": "arc.companion.evidence-resolution.v1",
            "requests": [],
            "lanes": {},
            "accepted": [],
            "rejected": [],
            "rerun_segments": [],
        })
        return annotations, evidence

    resolution = controller.resolve(requests, existing_records=evidence.get("related_papers") or [])
    supported = set(resolution.supported_request_keys)
    audit = dict(resolution.audit)
    audit["round"] = 1
    audit["rerun_segments"] = sorted(resolution.evidence_ids_by_segment)
    write_json(checkpoint_dir / "evidence-resolution.v1.json", audit)

    merged_evidence = {
        **evidence,
        "related_papers": [*(evidence.get("related_papers") or []), *resolution.records],
        "required_evidence_ids_by_segment": {
            key: list(values) for key, values in resolution.evidence_ids_by_segment.items()
        },
    }
    segment_by_id = {str(item["segment_id"]): item for item in segments}
    rerun_ids = [value for value in segment_by_id if value in resolution.evidence_ids_by_segment]
    resolution_by_segment = {
        segment_id: {
            "round": 2,
            "registered_evidence_ids": list(resolution.evidence_ids_by_segment[segment_id]),
            "requests": [item for item in requests if item["segment_id"] == segment_id],
            "audit_path": str(checkpoint_dir / "evidence-resolution.v1.json"),
        }
        for segment_id in rerun_ids
    }
    if rerun_ids:
        rerun = _generate_annotations(
            [segment_by_id[value] for value in rerun_ids],
            options=options,
            bundle=bundle,
            evidence=merged_evidence,
            domain_context=domain_context,
            glossary=glossary,
            protected_names=protected_names,
            checkpoint_dir=checkpoint_dir,
            llm=llm,
            round_number=2,
            first_drafts={value: annotations[value] for value in rerun_ids},
            resolution_by_segment=resolution_by_segment,
        )
        annotations = {**annotations, **rerun}

    claim_bindings: dict[str, list[str]] = {}
    for segment_id, annotation in annotations.items():
        segment_requests = [item for item in requests if item["segment_id"] == segment_id]
        _clear_unresolved_requested_work(annotation, segment_requests, supported)
        annotation["evidence_requests"] = []
        claim_bindings[segment_id] = list(annotation.get("evidence_ids") or [])
    audit["round"] = 2
    audit["final_claim_evidence_ids"] = claim_bindings
    write_json(checkpoint_dir / "evidence-resolution.v1.json", audit)
    return annotations, merged_evidence


def _known_evidence_ids(values: Any, records: list[dict[str, Any]]) -> list[str]:
    valid = {str(item.get("evidence_id") or "") for item in records}
    return [str(value) for value in values if str(value) in valid]


def _drop_unsupported_second_round_related_work(
    annotation: dict[str, Any], records: list[dict[str, Any]],
) -> None:
    relation_by_id = {str(item.get("evidence_id") or ""): str(item.get("relation") or "") for item in records}
    used = list(annotation.get("evidence_ids") or [])
    if not any(relation_by_id.get(value) == "prior" for value in used):
        annotation["prior_work"] = ""
    if not any(relation_by_id.get(value) == "later" for value in used):
        annotation["later_work"] = ""
    used_relations = {
        relation for relation, field in (("prior", "prior_work"), ("later", "later_work"))
        if str(annotation.get(field) or "").strip()
    }
    annotation["evidence_ids"] = [
        value for value in used if relation_by_id.get(value) in used_relations
    ]


def _clear_unresolved_requested_work(
    annotation: dict[str, Any], requests: list[dict[str, Any]], supported: set[str],
) -> None:
    unresolved_relations = {
        str(item["relation"]) for item in requests if str(item["request_key"]) not in supported
    }
    if "prior" in unresolved_relations:
        annotation["prior_work"] = ""
    if "later" in unresolved_relations:
        annotation["later_work"] = ""


def _translation_draft_path(checkpoint_dir: Path, segment_id: str) -> Path:
    return checkpoint_dir / "translation-drafts" / f"{_segment_checkpoint_name(segment_id)}.json"


def _translation_coverage_attempt_path(checkpoint_dir: Path, segment_id: str) -> Path:
    return (
        checkpoint_dir
        / "translation-coverage-attempts"
        / f"{_segment_checkpoint_name(segment_id)}.json"
    )


def _matching_translation_coverage_attempt(
    checkpoint_dir: Path, segment_id: str, input_sha256: str,
) -> dict[str, Any] | None:
    path = _translation_coverage_attempt_path(checkpoint_dir, segment_id)
    value = read_json(path) if path.is_file() else None
    if (
        isinstance(value, dict)
        and value.get("segment_id") == segment_id
        and value.get("input_sha256") == input_sha256
    ):
        return value
    return None


def _translation_primary_draft_payload(
    segment: dict[str, Any],
    translation: dict[str, Any],
    *,
    input_sha256: str,
    origin: str,
) -> dict[str, Any]:
    model_generated = origin == "primary-model"
    return {
        "schema_version": "arc.companion.translation-primary-draft.v1",
        "segment_id": str(segment["segment_id"]),
        "input_sha256": input_sha256,
        "candidate_provenance": {
            "origin": origin,
            "prompt_version": PROMPT_VERSION if model_generated else None,
            "response_schema_version": SCHEMA_VERSION if model_generated else None,
            "model_tier": TRANSLATION_TIER if model_generated else None,
        },
        "translation": translation,
    }


def _seed_translation_coverage_draft(
    segment: dict[str, Any],
    *,
    bundle: SourceBundle,
    glossary: dict[str, Any],
    protected_names: list[str],
    checkpoint_dir: Path,
    translation: dict[str, Any] | None = None,
) -> Path:
    """Seed an auditable repair-only candidate without invoking the primary model."""
    by_id = {block_id(block): block for block in bundle.document["blocks"]}
    paper_context = _full_paper_context(bundle.document, segment, blocks_by_id=by_id)
    input_sha256 = _segment_input_hash(
        segment,
        by_id,
        glossary=glossary,
        extra={
            "names": protected_names,
            "paper_context": paper_context,
            "runtime_access": _generation_runtime_policy(),
        },
    )
    path = _translation_draft_path(checkpoint_dir, str(segment["segment_id"]))
    write_json(
        path,
        _translation_primary_draft_payload(
            segment,
            translation or {"blocks": []},
            input_sha256=input_sha256,
            origin="controller-seed",
        ),
    )
    return path


def _generate_translations(
    segments: list[dict[str, Any]],
    *,
    options: BuildOptions,
    bundle: SourceBundle,
    glossary: dict[str, Any],
    protected_names: list[str],
    checkpoint_dir: Path,
    llm: Callable[..., dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_id = {block_id(block): block for block in bundle.document["blocks"]}
    translation_dir = checkpoint_dir / "translations"
    translation_dir.mkdir(parents=True, exist_ok=True)
    output: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {}
    for segment in segments:
        path = translation_dir / f"{_segment_checkpoint_name(segment['segment_id'])}.json"
        paper_context = _full_paper_context(bundle.document, segment, blocks_by_id=by_id)
        expected_hash = _segment_input_hash(
            segment,
            by_id,
            glossary=glossary,
            extra={
                "names": protected_names,
                "paper_context": paper_context,
                "runtime_access": _generation_runtime_policy(),
            },
        )
        input_hashes[str(segment["segment_id"])] = expected_hash
        if path.is_file() and not options.force:
            checkpoint = read_json(path)
            if (
                isinstance(checkpoint, dict)
                and checkpoint.get("segment_id") == segment["segment_id"]
                and checkpoint.get("input_sha256") == expected_hash
                and isinstance(checkpoint.get("translation"), dict)
            ):
                _validate_translation(segment, checkpoint["translation"], by_id, protected_names)
                output[segment["segment_id"]] = checkpoint["translation"]
                continue
        pending.append(segment)

    def generate(
        segment: dict[str, Any],
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        segment_id = str(segment["segment_id"])
        selected = [by_id[value] for value in segment["block_ids"]]
        translatable = [_translation_input_block(block) for block in selected if _is_translatable(block)]
        paper_context = _full_paper_context(bundle.document, segment, blocks_by_id=by_id)
        artifact_dir = (
            checkpoint_dir / "llm" / "translations" / _segment_checkpoint_name(segment_id)
        )
        draft_path = _translation_draft_path(checkpoint_dir, segment_id)
        draft = read_json(draft_path) if draft_path.is_file() else None
        attempt = _matching_translation_coverage_attempt(
            checkpoint_dir, segment_id, input_hashes[segment_id]
        )
        candidate_provenance: dict[str, Any] | None = None
        translation: dict[str, Any] | None = None
        if (
            isinstance(draft, dict)
            and draft.get("schema_version") == "arc.companion.translation-primary-draft.v1"
            and draft.get("segment_id") == segment_id
            and draft.get("input_sha256") == input_hashes[segment_id]
            and isinstance(draft.get("translation"), dict)
        ):
            draft_translation = draft["translation"]
            try:
                _validate_translation(segment, draft_translation, by_id, protected_names)
            except TranslationCoverageError:
                translation = draft_translation
            except RuntimeError:
                pass
            else:
                if not options.force:
                    translation = draft_translation
            if translation is not None:
                candidate_provenance = dict(draft.get("candidate_provenance") or {})
        if attempt is not None and translation is None:
            raise RuntimeError(
                f"translation coverage repair attempt already consumed for {segment_id}"
            )
        if translatable:
            if translation is None:
                translation = _llm_call(
                    llm,
                    translation_prompt(
                        segment,
                        translatable,
                        language=options.annotation_language,
                        glossary=glossary,
                        protected_names=protected_names,
                        paper_context=paper_context,
                    ),
                    TRANSLATION_SCHEMA,
                    options=options,
                    artifact_dir=artifact_dir,
                    call_label=f"companion-translation-{segment_id}",
                    model_tier=TRANSLATION_TIER,
                    allow_mcp=True,
                    allow_internet=True,
                )
                draft = _translation_primary_draft_payload(
                    segment,
                    translation,
                    input_sha256=input_hashes[segment_id],
                    origin="primary-model",
                )
                write_json(draft_path, draft)
                candidate_provenance = dict(draft["candidate_provenance"])
        else:
            translation = {"blocks": []}
            candidate_provenance = {"origin": "controller-empty"}
        assert translation is not None
        repair_provenance: list[dict[str, Any]] = []
        try:
            _validate_translation(segment, translation, by_id, protected_names)
        except TranslationCoverageError:
            normalized, missing_blocks, diagnostics = _normalize_translation_coverage(
                segment, translation, by_id
            )
            if missing_blocks:
                if attempt is not None:
                    raise RuntimeError(
                        f"translation coverage repair attempt already consumed for {segment_id}"
                    )
                repair_contexts = [
                    _translation_coverage_repair_context(block) for block in missing_blocks
                ]
                attempt_path = _translation_coverage_attempt_path(checkpoint_dir, segment_id)
                write_json(attempt_path, {
                    "schema_version": "arc.companion.translation-coverage-attempt.v1",
                    "segment_id": segment_id,
                    "input_sha256": input_hashes[segment_id],
                    "status": "started",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "prompt_version": TRANSLATION_COVERAGE_REPAIR_PROMPT_VERSION,
                    "response_schema_version": TRANSLATION_COVERAGE_REPAIR_SCHEMA_VERSION,
                    "model_tier": TRANSLATION_COVERAGE_REPAIR_TIER,
                    "missing_block_ids": [block_id(block) for block in missing_blocks],
                })
                value = _llm_call(
                    llm,
                    translation_coverage_repair_prompt(
                        segment,
                        repair_contexts,
                        language=options.annotation_language,
                        glossary=glossary,
                        protected_names=protected_names,
                        paper_context={
                            **paper_context,
                            "access": {"allow_mcp": False, "allow_internet": False},
                        },
                        repair_model_tier=TRANSLATION_COVERAGE_REPAIR_TIER,
                    ),
                    TRANSLATION_COVERAGE_REPAIR_SCHEMA,
                    options=options,
                    artifact_dir=artifact_dir / "coverage-repair-1",
                    call_label=f"companion-translation-{segment_id}-coverage-repair-1",
                    model_tier=TRANSLATION_COVERAGE_REPAIR_TIER,
                    allow_mcp=False,
                    allow_internet=False,
                )
                translation = _apply_translation_coverage_repairs(
                    normalized, segment, missing_blocks, value, by_id
                )
                repair_provenance.append({
                    "kind": "coverage",
                    "attempt": 1,
                    "prompt_version": TRANSLATION_COVERAGE_REPAIR_PROMPT_VERSION,
                    "response_schema_version": TRANSLATION_COVERAGE_REPAIR_SCHEMA_VERSION,
                    "model_tier": TRANSLATION_COVERAGE_REPAIR_TIER,
                    "repaired_block_ids": [block_id(block) for block in missing_blocks],
                    "normalization": diagnostics,
                })
            else:
                translation = normalized
                repair_provenance.append({
                    "kind": "coverage-normalization",
                    "attempt": 0,
                    "normalization": diagnostics,
                })
            _validate_translation(segment, translation, by_id, protected_names)
        except TranslationOpaqueTokenError:
            token_errors = _translation_opaque_token_errors(segment, translation, by_id)
            source_blocks = [by_id[item.block_id] for item in token_errors]
            previous_by_id = {
                str(item.get("block_id") or ""): item for item in translation["blocks"]
            }
            repair_contexts = [
                _translation_slot_repair_context(
                    source_block,
                    str(previous_by_id[block_id(source_block)].get("text") or ""),
                    protected_names=protected_names,
                )
                for source_block in source_blocks
            ]
            value = _llm_call(
                llm,
                translation_retry_prompt(
                    segment,
                    repair_contexts,
                    validation_errors=[item.prompt_payload() for item in token_errors],
                    retry_model_tier=TRANSLATION_RETRY_TIER,
                ),
                TRANSLATION_SLOT_REPAIR_SCHEMA,
                options=options,
                artifact_dir=artifact_dir / "retry-1",
                call_label=f"companion-translation-{segment['segment_id']}-retry-1",
                model_tier=TRANSLATION_RETRY_TIER,
                allow_mcp=False,
                allow_internet=False,
            )
            translation = _apply_translation_slot_repairs(
                translation,
                source_blocks,
                value,
                protected_names=protected_names,
            )
            _validate_translation(segment, translation, by_id, protected_names)
            repair_provenance.append({
                "kind": "token-placement",
                "attempt": 1,
                "prompt_version": TRANSLATION_RETRY_PROMPT_VERSION,
                "model_tier": TRANSLATION_RETRY_TIER,
                "repaired_block_ids": [block_id(block) for block in source_blocks],
            })
        return segment_id, translation, {
            "candidate": candidate_provenance or {},
            "repairs": repair_provenance,
        }

    with ThreadPoolExecutor(max_workers=min(options.workers, max(1, len(pending)))) as executor:
        futures = {executor.submit(generate, segment): segment for segment in pending}
        failures: list[tuple[str, BaseException]] = []
        for future in as_completed(futures):
            segment = futures[future]
            try:
                segment_id, value, provenance = future.result()
            except Exception as exc:
                failures.append((str(segment["segment_id"]), exc))
                continue
            else:
                output[segment_id] = value
                segment = next(item for item in segments if item["segment_id"] == segment_id)
                write_json(
                    translation_dir / f"{_segment_checkpoint_name(segment_id)}.json",
                    {
                        "schema_version": "arc.companion.translation-checkpoint.v2",
                        "segment_id": segment_id,
                        "input_sha256": _segment_input_hash(
                            segment,
                            by_id,
                            glossary=glossary,
                            extra={
                                "names": protected_names,
                                "paper_context": _full_paper_context(
                                    bundle.document, segment, blocks_by_id=by_id
                                ),
                                "runtime_access": _generation_runtime_policy(),
                            },
                        ),
                        "generation_provenance": provenance,
                        "translation": value,
                    },
                )
        if failures:
            raise CompanionLaneError("translation", failures)
    return {segment["segment_id"]: output[segment["segment_id"]] for segment in segments}


def _review(
    segments: list[dict[str, Any]],
    translations: dict[str, dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    *,
    document: dict[str, Any],
    glossary: dict[str, Any],
    protected_names: list[str],
    evidence: dict[str, Any],
    options: BuildOptions,
    llm: Callable[..., dict[str, Any]],
    checkpoint_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    by_id = {block_id(block): block for block in document.get("blocks") or []}
    payload = {
        "segments": [
            {
                "segment": segment,
                "source_blocks": [_annotation_input_block(by_id[value], document) for value in segment["block_ids"]],
                "translation": translations[segment["segment_id"]],
                "annotation": annotations[segment["segment_id"]],
            }
            for segment in segments
        ]
    }
    findings: list[dict[str, Any]] = []
    hierarchical = len(json.dumps(payload, ensure_ascii=False)) > options.review_context_chars
    if hierarchical:
        chunks = _review_chunks(payload["segments"], options.review_context_chars // 2)

        def inspect(index: int, chunk: list[dict[str, Any]]) -> dict[str, Any]:
            return _llm_call(
                llm,
                section_review_prompt({"segments": chunk}, language=options.annotation_language),
                SECTION_REVIEW_SCHEMA,
                options=options,
                artifact_dir=checkpoint_dir / "llm" / "section-review" / str(index),
                call_label=f"companion-section-review-{index}",
                model_tier=REVIEW_TIER,
            )

        with ThreadPoolExecutor(max_workers=min(options.workers, len(chunks))) as executor:
            futures = {executor.submit(inspect, index, chunk): index for index, chunk in enumerate(chunks)}
            ordered: dict[int, dict[str, Any]] = {}
            for future in as_completed(futures):
                value = future.result()
                ordered[futures[future]] = value
        section_reviews: list[dict[str, Any]] = []
        review_coverage: set[str] = set()
        for index in sorted(ordered):
            value = ordered[index]
            chunk_ids = {str(item["segment"]["segment_id"]) for item in chunks[index]}
            reviewed_segments = [item for item in value.get("reviewed_segments") or [] if isinstance(item, dict)]
            reviewed_ids = {str(item.get("segment_id") or "") for item in reviewed_segments}
            if (
                reviewed_ids != chunk_ids
                or len(reviewed_segments) != len(chunk_ids)
                or any(
                    not isinstance(item.get("translation"), dict)
                    or not isinstance(item.get("annotation"), dict)
                    for item in reviewed_segments
                )
            ):
                raise RuntimeError(f"section review {index} did not cover every segment")
            chunk_findings = [item for item in value.get("findings") or [] if isinstance(item, dict)]
            invalid_finding_ids = {
                str(item.get("segment_id") or "") for item in chunk_findings
            } - chunk_ids
            if invalid_finding_ids:
                raise RuntimeError(f"section review {index} returned findings for unknown segments")
            review_coverage.update(reviewed_ids)
            findings.extend(chunk_findings)
            section_reviews.append({
                "section_index": index,
                "reviewed_segments": reviewed_segments,
                "reviewed_segment_ids": sorted(reviewed_ids),
                "findings": chunk_findings,
            })
        all_segment_ids = {str(item["segment_id"]) for item in segments}
        if review_coverage != all_segment_ids:
            missing = sorted(all_segment_ids - review_coverage)
            raise RuntimeError(f"hierarchical review did not cover every segment: {missing}")

    final_payload = {**payload, "glossary": glossary, "protected_names": protected_names}
    if hierarchical:
        final_payload = {
            "section_reviews": section_reviews,
            "reviewed_segment_ids": [str(item["segment_id"]) for item in segments],
            "source_anchors": _review_source_anchors(
                segments,
                blocks_by_id=by_id,
                document=document,
                total_chars=min(50_000, max(10_000, options.review_context_chars // 3)),
            ),
            "glossary": glossary,
            "protected_names": protected_names,
            "instruction": (
                "Consolidate section findings into non-conflicting translation and/or annotation patches. "
                "Use the bounded source anchor for every segment as a direct source-awareness check; "
                "section reviews contain the complete source-aware local review."
            ),
        }
    review = _llm_call(
        llm,
        review_prompt(final_payload, language=options.annotation_language, findings=findings),
        REVIEW_SCHEMA,
        options=options,
        artifact_dir=checkpoint_dir / "llm" / "final-review",
        call_label="companion-final-review",
        model_tier=REVIEW_TIER,
    )
    reviewed_translations = {key: {"blocks": [dict(item) for item in value.get("blocks") or []]} for key, value in translations.items()}
    reviewed = {key: dict(value) for key, value in annotations.items()}
    valid_ids = set(reviewed)
    patched: set[str] = set()
    for patch in review.get("patches") or []:
        segment_id = str(patch.get("segment_id") or "")
        if segment_id not in valid_ids or segment_id in patched:
            raise RuntimeError(f"review returned invalid or duplicate annotation patch: {segment_id}")
        translation_changed = patch.get("translation_blocks") is not None
        if translation_changed:
            replacement = {"blocks": list(patch.get("translation_blocks") or [])}
            segment = next(item for item in segments if item["segment_id"] == segment_id)
            _validate_translation(segment, replacement, by_id, protected_names)
            reviewed_translations[segment_id] = replacement
        annotation_fields = ("commentary", "explanation", "prior_work", "later_work", "evidence_ids")
        changed_annotation_fields = [field for field in annotation_fields if patch.get(field) is not None]
        if not changed_annotation_fields and not translation_changed:
            raise RuntimeError(f"review returned an empty patch: {segment_id}")
        for field in changed_annotation_fields:
            if field == "evidence_ids":
                segment = next(item for item in segments if item["segment_id"] == segment_id)
                segment_evidence = _evidence_for_segment(segment, by_id, evidence)
                reviewed[segment_id][field] = _validated_evidence_ids(
                    patch[field], {"related_papers": segment_evidence["papers"]}
                )
            else:
                text = str(patch[field])
                if field in {"commentary", "explanation"} and not text.strip():
                    raise RuntimeError(f"review returned empty {field} patch: {segment_id}")
                reviewed[segment_id][field] = text
        segment = next(item for item in segments if item["segment_id"] == segment_id)
        _validate_annotation_evidence(
            reviewed[segment_id], _evidence_for_segment(segment, by_id, evidence)["papers"]
        )
        patched.add(segment_id)
    return reviewed_translations, reviewed, {
        "hierarchical": hierarchical,
        "section_findings": findings,
        "reviewed_segment_ids": [str(item["segment_id"]) for item in segments],
        "issues": [str(item) for item in review.get("issues") or []],
        "patched_segment_ids": sorted(patched),
    }


def _llm_call(
    llm: Callable[..., dict[str, Any]],
    prompt: str,
    schema: dict[str, Any],
    *,
    options: BuildOptions,
    artifact_dir: Path,
    call_label: str,
    model_tier: str,
    allow_mcp: bool = False,
    allow_internet: bool = False,
) -> dict[str, Any]:
    runtime_env = _llm_runtime_env(allow_mcp=allow_mcp, allow_internet=allow_internet)
    return llm(
        prompt,
        schema=schema,
        provider=options.provider,
        model=options.model,
        model_tier=None if options.model else model_tier,
        env=runtime_env,
        session_policy="stateless",
        artifact_dir=artifact_dir,
        call_label=call_label,
    )


def _generation_runtime_policy() -> dict[str, bool | str]:
    return {"allow_mcp": True, "allow_internet": True, "mcp_mode": "arc-only"}


def _llm_runtime_env(*, allow_mcp: bool, allow_internet: bool) -> dict[str, str]:
    """Map portable access intent onto both supported host runtimes."""
    env = dict(os.environ)
    internet_value = "true" if allow_internet else "false"
    env["ARC_CODEX_ALLOW_INTERNET"] = internet_value
    env["ARC_CLAUDE_ALLOW_INTERNET"] = internet_value
    mcp_value = "true" if allow_mcp else "false"
    env["ARC_CODEX_ENABLE_MCP"] = mcp_value
    env["ARC_CLAUDE_ALLOW_MCP"] = mcp_value
    if allow_mcp:
        env["ARC_CODEX_MCP_MODE"] = "arc-only"
        env["ARC_CLAUDE_MCP_MODE"] = "arc-only"
    else:
        env.pop("ARC_CODEX_MCP_MODE", None)
        env.pop("ARC_CLAUDE_MCP_MODE", None)
    return env


def _fingerprint(
    bundle: SourceBundle,
    options: BuildOptions,
    *,
    evidence: dict[str, Any],
    domain_context: dict[str, Any] | None = None,
) -> str:
    integrity = bundle.document.get("integrity") or {}
    return sha256_json(
        {
            "workflow_version": WORKFLOW_VERSION,
            "prompt_version": PROMPT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "paper_id": bundle.paper_id,
            "document_hash": (
                integrity.get("document_hash")
                or bundle.document.get("document_hash")
                or bundle.parsed.get("document_hash")
                or sha256_json(bundle.document)
            ),
            "asset_manifest_hash": (
                integrity.get("asset_manifest_hash")
                or bundle.document.get("asset_manifest_hash")
                or bundle.parsed.get("asset_manifest_hash")
            ),
            "language": options.annotation_language,
            "provider": options.provider,
            "model": options.model,
            "model_tiers": {
                "segmentation": SEGMENTATION_TIER,
                "glossary": GLOSSARY_TIER,
                "translation": TRANSLATION_TIER,
                "annotation": ANNOTATION_TIER,
                "review": REVIEW_TIER,
            },
            "workers_per_lane": options.workers,
            "runtime_access": {
                "segmentation": {"allow_mcp": False, "allow_internet": False},
                "glossary": {"allow_mcp": False, "allow_internet": False},
                "translation": _generation_runtime_policy(),
                "annotation": _generation_runtime_policy(),
                "review": {"allow_mcp": False, "allow_internet": False},
            },
            "full_paper_context_version": FULL_PAPER_CONTEXT_VERSION,
            "metadata_hash": sha256_json(bundle.metadata),
            "evidence_hash": sha256_json(evidence),
            "domain_context_hash": sha256_json(domain_context) if domain_context is not None else None,
        }
    )


def _page_count(bundle: SourceBundle) -> int | None:
    """Read an explicit source page count when a provider supplied one; never estimate it."""
    candidates = [
        bundle.parsed.get("page_count"),
        bundle.document.get("page_count"),
        bundle.metadata.get("page_count"),
        bundle.metadata.get("number_of_pages"),
    ]
    for value in candidates:
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count > 0:
            return count
    return None


def _segment_checkpoint_name(segment_id: str) -> str:
    return hashlib.sha256(segment_id.encode("utf-8")).hexdigest()


def _segment_input_hash(
    segment: dict[str, Any],
    blocks_by_id: dict[str, dict[str, Any]],
    *,
    glossary: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    return sha256_json({
        "segment": segment,
        "blocks": [blocks_by_id[value] for value in segment.get("block_ids") or []],
        "glossary_hash": sha256_json(glossary) if glossary is not None else None,
        "extra": extra,
    })


def _sha256_existing_file(path: Path) -> str:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"completed companion output is missing or empty: {path}")
    return sha256_file(path)


def _completion_outputs_match(state: dict[str, Any]) -> bool:
    checks = (
        ("output_tex", "output_tex_sha256"),
        ("output_pdf", "output_pdf_sha256"),
        ("source_manifest_path", "source_manifest_sha256"),
        ("validation_path", "validation_sha256"),
    )
    for path_key, hash_key in checks:
        value = state.get(path_key)
        expected = str(state.get(hash_key) or "")
        if not value or not expected:
            return False
        path = Path(str(value))
        if not path.is_file() or path.stat().st_size == 0 or sha256_file(path) != expected:
            return False
    return True


def _first_wave_preview_outputs_match(state: dict[str, Any], *, workers: int) -> bool:
    if state.get("first_wave_preview_version") != FIRST_WAVE_PREVIEW_VERSION:
        return False
    try:
        segment_count = int(state.get("segment_count"))
        preview_segment_count = int(state.get("preview_segment_count"))
    except (TypeError, ValueError):
        return False
    segment_ids = state.get("preview_segment_ids")
    if (
        segment_count < 1
        or preview_segment_count != min(workers, segment_count)
        or not isinstance(segment_ids, list)
        or len(segment_ids) != preview_segment_count
        or any(not str(value) for value in segment_ids)
    ):
        return False
    checks = (
        ("preview_tex", "preview_tex_sha256"),
        ("preview_pdf", "preview_pdf_sha256"),
        ("preview_source_manifest_path", "preview_source_manifest_sha256"),
        ("preview_validation_path", "preview_validation_sha256"),
    )
    for path_key, hash_key in checks:
        value = state.get(path_key)
        expected = str(state.get(hash_key) or "")
        if not value or not expected:
            return False
        path = Path(str(value))
        if not path.is_file() or path.stat().st_size == 0 or sha256_file(path) != expected:
            return False
    return True


def _evidence(bundle: SourceBundle) -> dict[str, Any]:
    def compact(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        fields = ("paper_id", "arxiv_id", "inspire_id", "title", "authors", "year", "citation_count")
        return [{key: item[key] for key in fields if key in item} for item in items[:25]]

    return {
        "schema_version": "arc.companion.evidence.v2",
        "references": compact(bundle.references),
        "citers": compact(bundle.citers),
        "related_papers": list(bundle.related_evidence),
        "diagnostics": list(bundle.diagnostics),
    }


def _is_translatable(block: dict[str, Any]) -> bool:
    return _project_is_translatable(block)


def _translation_input_block(block: dict[str, Any]) -> dict[str, Any]:
    return _project_translation_input_block(block)


def _translation_coverage_slot_ids(source_block: dict[str, Any]) -> list[str]:
    return [
        f"{block_id(source_block)}.coverage-slot-{index:04d}"
        for index in range(len(_opaque_inline_tokens(source_block)) + 1)
    ]


def _translation_coverage_repair_context(source_block: dict[str, Any]) -> dict[str, Any]:
    """Describe N+1 natural-language slots without delegating opaque content."""
    projected_text = str(_translation_input_block(source_block).get("text") or "")
    expected_tokens = _opaque_inline_tokens(source_block)
    source_slots: list[str] = []
    cursor = 0
    for token in expected_tokens:
        index = projected_text.find(token, cursor)
        if index < 0:
            raise RuntimeError("coverage repair cannot locate a source opaque token")
        source_slots.append(projected_text[cursor:index])
        cursor = index + len(token)
    source_slots.append(projected_text[cursor:])
    opaque_boundaries = []
    for run in source_block.get("inline_runs") or []:
        if not isinstance(run, dict) or str(run.get("kind") or "") == "text":
            continue
        record = {
            "kind": str(run.get("kind") or ""),
            "source_content": str(run.get("tex") or run.get("content") or ""),
        }
        if run.get("href"):
            record["href"] = str(run["href"])
        opaque_boundaries.append(record)
    return {
        "block_id": block_id(source_block),
        "slot_ids": _translation_coverage_slot_ids(source_block),
        "source_natural_language_slots": source_slots,
        "opaque_boundaries": opaque_boundaries,
    }


def _normalize_translation_coverage(
    segment: dict[str, Any],
    translation: dict[str, Any],
    blocks_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Keep only uniquely mapped expected blocks and restore canonical source order."""
    expected_blocks = [
        blocks_by_id[value]
        for value in segment.get("block_ids") or []
        if _is_translatable(blocks_by_id[value])
    ]
    expected_ids = [block_id(block) for block in expected_blocks]
    raw_blocks = translation.get("blocks") if isinstance(translation, dict) else None
    raw_blocks = raw_blocks if isinstance(raw_blocks, list) else []
    candidates: dict[str, list[dict[str, Any]]] = {value: [] for value in expected_ids}
    unknown_ids: list[str] = []
    malformed_count = 0
    for item in raw_blocks:
        if not isinstance(item, dict):
            malformed_count += 1
            continue
        item_id = str(item.get("block_id") or "")
        if item_id in candidates:
            candidates[item_id].append(item)
        else:
            unknown_ids.append(item_id)
    unique = {
        value: items[0]
        for value, items in candidates.items()
        if len(items) == 1
    }
    missing_blocks = [
        block for block in expected_blocks if block_id(block) not in unique
    ]
    normalized = {
        **translation,
        "blocks": [unique[value] for value in expected_ids if value in unique],
    }
    diagnostics = {
        "expected_block_ids": expected_ids,
        "preserved_block_ids": [value for value in expected_ids if value in unique],
        "missing_block_ids": [block_id(block) for block in missing_blocks],
        "duplicate_block_ids": [
            value for value in expected_ids if len(candidates[value]) > 1
        ],
        "discarded_unknown_block_ids": unknown_ids,
        "discarded_malformed_count": malformed_count,
    }
    return normalized, missing_blocks, diagnostics


def _apply_translation_coverage_repairs(
    normalized_translation: dict[str, Any],
    segment: dict[str, Any],
    missing_blocks: list[dict[str, Any]],
    repair: dict[str, Any],
    blocks_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Interleave controller-owned tokens into newly translated missing blocks."""
    raw_repairs = repair.get("repairs") if isinstance(repair, dict) else None
    if not isinstance(raw_repairs, list):
        raise RuntimeError("translation coverage repair has no repairs list")
    expected_missing_ids = [block_id(block) for block in missing_blocks]
    actual_missing_ids = [
        str(item.get("block_id") or "") for item in raw_repairs if isinstance(item, dict)
    ]
    if actual_missing_ids != expected_missing_ids or len(raw_repairs) != len(actual_missing_ids):
        raise RuntimeError("translation coverage repair changed missing block coverage or order")
    additions: dict[str, dict[str, str]] = {}
    for source_block, block_repair in zip(missing_blocks, raw_repairs):
        raw_slots = block_repair.get("slots") if isinstance(block_repair, dict) else None
        if not isinstance(raw_slots, list):
            raise RuntimeError("translation coverage repair has no slot list")
        expected_slot_ids = _translation_coverage_slot_ids(source_block)
        actual_slot_ids = [
            str(item.get("slot_id") or "") for item in raw_slots if isinstance(item, dict)
        ]
        if actual_slot_ids != expected_slot_ids or len(raw_slots) != len(actual_slot_ids):
            raise RuntimeError("translation coverage repair changed slot coverage or order")
        slots = [str(item.get("text") or "") for item in raw_slots]
        if any("[[ARC_INLINE:" in text or _OPAQUE_INLINE_PATTERN.search(text) for text in slots):
            raise RuntimeError("translation coverage repair supplied controller-owned opaque content")
        expected_tokens = _opaque_inline_tokens(source_block)
        assembled = "".join(
            part
            for index, slot in enumerate(slots)
            for part in ([slot, expected_tokens[index]] if index < len(expected_tokens) else [slot])
        )
        additions[block_id(source_block)] = {
            "block_id": block_id(source_block),
            "text": assembled,
        }
    preserved = {
        str(item.get("block_id") or ""): item
        for item in normalized_translation.get("blocks") or []
        if isinstance(item, dict)
    }
    expected_ids = [
        block_id(blocks_by_id[value])
        for value in segment.get("block_ids") or []
        if _is_translatable(blocks_by_id[value])
    ]
    return {
        **normalized_translation,
        "blocks": [
            preserved[value] if value in preserved else additions[value]
            for value in expected_ids
        ],
    }


def _translation_slot_repair_context(
    source_block: dict[str, Any],
    previous_text: str,
    *,
    protected_names: list[str],
) -> dict[str, Any]:
    """Build an inert placement-only context without asking for retranslation."""
    expected_tokens = _opaque_inline_tokens(source_block)
    residue = _translation_natural_residue(previous_text)
    source_runs: list[dict[str, Any]] = []
    for index, run in enumerate(source_block.get("inline_runs") or [], start=1):
        if not isinstance(run, dict):
            continue
        kind = str(run.get("kind") or "")
        record: dict[str, Any] = {"order": index, "kind": kind}
        if kind == "text":
            record["source_text"] = str(run.get("content") or "")
        else:
            record["source_content"] = str(run.get("tex") or run.get("content") or "")
            if kind == "link" and run.get("href"):
                record["href"] = str(run["href"])
        source_runs.append(record)
    return {
        "block_id": block_id(source_block),
        "previous_invalid_text": previous_text,
        "prior_natural_language_residue": residue,
        "expected_slot_count": len(expected_tokens) + 1,
        "slot_ids": _translation_repair_slot_ids(source_block),
        "missing_protected_names": _minimal_missing_name_insertions(
            _missing_protected_names([source_block], residue, protected_names)
        ),
        "source_run_sequence": source_runs,
    }


def _translation_repair_slot_ids(source_block: dict[str, Any]) -> list[str]:
    return [
        f"{block_id(source_block)}.repair-slot-{index:04d}"
        for index in range(len(_opaque_inline_tokens(source_block)) + 1)
    ]


def _apply_translation_slot_repairs(
    previous_translation: dict[str, Any],
    source_blocks: list[dict[str, Any]],
    repair: dict[str, Any],
    *,
    protected_names: list[str],
) -> dict[str, Any]:
    """Patch every token-mismatched block in one bounded repair call."""
    raw_repairs = repair.get("repairs") if isinstance(repair, dict) else None
    if not isinstance(raw_repairs, list):
        raise RuntimeError("translation slot repair has no repairs list")
    expected_ids = [block_id(block) for block in source_blocks]
    actual_ids = [
        str(item.get("block_id") or "") for item in raw_repairs if isinstance(item, dict)
    ]
    if actual_ids != expected_ids or len(actual_ids) != len(raw_repairs):
        raise RuntimeError("translation slot repair changed failing block coverage or order")
    result = previous_translation
    for source_block, block_repair in zip(source_blocks, raw_repairs):
        result = _apply_translation_slot_repair_block(
            result, source_block, block_repair, protected_names=protected_names,
        )
    return result


def _apply_translation_slot_repair_block(
    previous_translation: dict[str, Any],
    source_block: dict[str, Any],
    repair: dict[str, Any],
    *,
    protected_names: list[str],
) -> dict[str, Any]:
    """Patch one failed block while preserving prior natural language and all other blocks."""
    block_id_value = block_id(source_block)
    if not isinstance(repair, dict) or str(repair.get("block_id") or "") != block_id_value:
        raise RuntimeError("translation slot repair changed the failing block_id")
    raw_slots = repair.get("slots")
    if not isinstance(raw_slots, list):
        raise RuntimeError("translation slot repair has no slot list")
    expected_slot_ids = _translation_repair_slot_ids(source_block)
    actual_slot_ids = [
        str(item.get("slot_id") or "") for item in raw_slots if isinstance(item, dict)
    ]
    if actual_slot_ids != expected_slot_ids or len(actual_slot_ids) != len(raw_slots):
        raise RuntimeError("translation slot repair changed slot coverage or order")
    slots = [str(item.get("text") or "") for item in raw_slots]
    if any("[[ARC_INLINE:" in text or _OPAQUE_INLINE_PATTERN.search(text) for text in slots):
        raise RuntimeError("translation slot repair supplied controller-owned opaque content")

    previous_blocks = previous_translation.get("blocks") if isinstance(previous_translation, dict) else None
    if not isinstance(previous_blocks, list):
        raise RuntimeError("translation slot repair has no prior block list")
    failed_index = next(
        (index for index, item in enumerate(previous_blocks)
         if isinstance(item, dict) and str(item.get("block_id") or "") == block_id_value),
        None,
    )
    if failed_index is None:
        raise RuntimeError("translation slot repair cannot find the prior failed block")
    previous_text = str(previous_blocks[failed_index].get("text") or "")
    residue = _translation_natural_residue(previous_text)
    missing_names = _minimal_missing_name_insertions(
        _missing_protected_names([source_block], residue, protected_names)
    )
    joined = "".join(slots)
    if missing_names:
        if not _is_exact_name_insertion_delta(joined, residue, missing_names):
            raise RuntimeError("translation slot repair changed prior natural language beyond name insertion")
    elif joined != residue:
        raise RuntimeError("translation slot repair retranslated or changed prior natural language")

    expected_tokens = _opaque_inline_tokens(source_block)
    assembled = "".join(
        part
        for index, slot in enumerate(slots)
        for part in ([slot, expected_tokens[index]] if index < len(expected_tokens) else [slot])
    )
    repaired_blocks = list(previous_blocks)
    repaired_blocks[failed_index] = {**previous_blocks[failed_index], "text": assembled}
    return {**previous_translation, "blocks": repaired_blocks}


def _translation_natural_residue(previous_text: str) -> str:
    """Remove any bounded controller-marker candidate, including mutated tokens."""
    return _OPAQUE_INLINE_CANDIDATE_PATTERN.sub("", previous_text)


def _opaque_inline_token(run: dict[str, Any]) -> str:
    return _project_opaque_inline_token(run)


_OPAQUE_INLINE_PATTERN = re.compile(r"\[\[ARC_INLINE:[^\]\s]+:[0-9a-f]{64}\]\]")
_OPAQUE_INLINE_CANDIDATE_PATTERN = re.compile(r"\[\[ARC_INLINE:[^\]\r\n]{0,512}\]\]")


def _opaque_inline_tokens(block: dict[str, Any]) -> list[str]:
    return _project_opaque_inline_tokens(block)


def _annotation_input_block(block: dict[str, Any], document: dict[str, Any]) -> dict[str, Any]:
    return _project_annotation_input_block(block, document)


def _full_paper_context(
    document: dict[str, Any],
    segment: dict[str, Any],
    *,
    blocks_by_id: dict[str, dict[str, Any]],
    max_chars: int = FULL_PAPER_CONTEXT_CHARS,
) -> dict[str, Any]:
    """Build bounded navigation and source anchors without preservation-only HTML."""
    excluded_generation_ids = _generation_excluded_block_ids(document)
    blocks = [
        block for block in document.get("blocks") or []
        if block_id(block) not in excluded_generation_ids
    ]
    positions = {block_id(block): index for index, block in enumerate(blocks)}
    member_ids = [str(value) for value in segment.get("block_ids") or []]
    member_positions = [positions[value] for value in member_ids if value in positions]
    first = min(member_positions) if member_positions else 0
    last = max(member_positions) if member_positions else first
    heading_kinds = {"heading", "section", "subsection", "subsubsection"}
    navigation: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        kind = str(block.get("type") or block.get("kind") or "").casefold()
        if kind not in heading_kinds:
            continue
        title = str(block.get("title") or block.get("text") or "").strip()
        anchor = ""
        for candidate in blocks[index + 1 : min(len(blocks), index + 8)]:
            candidate_kind = str(candidate.get("type") or candidate.get("kind") or "").casefold()
            if candidate_kind in heading_kinds:
                break
            anchor = str(candidate.get("text") or candidate.get("caption") or "").strip()
            if anchor:
                break
        navigation.append({
            "block_id": block_id(block),
            "level": block.get("level"),
            "title": title[:300],
            "anchor": anchor[:500],
        })

    neighbor_indices = [
        index for index in range(max(0, first - 2), min(len(blocks), last + 3))
        if index < first or index > last
    ]
    neighbors = [
        _bounded_projection(_annotation_input_block(blocks[index], document), 1_200)
        for index in neighbor_indices
    ]
    front = document.get("front_matter") or {}
    abstract = front.get("abstract") or ""
    if isinstance(abstract, dict):
        abstract = abstract.get("text") or ""
    context: dict[str, Any] = {
        "schema_version": FULL_PAPER_CONTEXT_VERSION,
        "paper_id": str((document.get("source") or {}).get("paper_id") or ""),
        "abstract": str(abstract)[:4_000],
        "current_segment": {
            "segment_id": str(segment.get("segment_id") or ""),
            "title": str(segment.get("title") or ""),
            "start_block_id": str(segment.get("start_block_id") or ""),
            "end_block_id": str(segment.get("end_block_id") or ""),
            "start_ordinal": first + 1,
            "end_ordinal": last + 1,
            "total_blocks": len(blocks),
        },
        "section_navigation": navigation,
        "neighboring_source_anchors": neighbors,
        "access": _generation_runtime_policy(),
    }
    _shrink_paper_context(context, max_chars=max_chars)
    return context


def _bounded_projection(value: dict[str, Any], limit: int) -> dict[str, Any]:
    output = _prompt_safe_value(value)
    for key in ("text", "title", "caption"):
        if key in output:
            output[key] = str(output[key])[:limit]
    if len(json.dumps(output, ensure_ascii=False)) > limit * 2:
        return {
            "block_id": str(output.get("block_id") or ""),
            "type": str(output.get("type") or ""),
            "text": str(output.get("text") or output.get("title") or output.get("caption") or "")[:limit],
        }
    return output


def _prompt_safe_value(value: Any) -> Any:
    return _project_prompt_safe_value(value)


def _shrink_paper_context(context: dict[str, Any], *, max_chars: int) -> None:
    def size() -> int:
        return len(json.dumps(context, ensure_ascii=False, separators=(",", ":")))

    navigation = context["section_navigation"]
    if size() > max_chars:
        for item in navigation:
            item["anchor"] = str(item.get("anchor") or "")[:160]
    if size() > max_chars:
        context["abstract"] = str(context.get("abstract") or "")[:1_200]
    if size() > max_chars:
        for item in navigation:
            item.pop("anchor", None)
            item["title"] = str(item.get("title") or "")[:160]
    if size() > max_chars:
        context["neighboring_source_anchors"] = [
            _bounded_projection(item, 400) for item in context["neighboring_source_anchors"]
        ]
    if size() > max_chars and navigation:
        original_count = len(navigation)
        keep = max(1, int(original_count * max_chars / size()))
        if keep < original_count:
            indices = sorted({round(i * (original_count - 1) / max(1, keep - 1)) for i in range(keep)})
            context["section_navigation"] = [navigation[index] for index in indices]
            context["navigation_omitted_count"] = original_count - len(indices)
    while size() > max_chars and len(context["section_navigation"]) > 1:
        current = context["section_navigation"]
        context["section_navigation"] = current[::2]
        context["navigation_omitted_count"] = len(navigation) - len(context["section_navigation"])
    if size() > max_chars:
        context["neighboring_source_anchors"] = []
        context["abstract"] = ""


def _validate_translation(
    segment: dict[str, Any],
    translation: dict[str, Any],
    blocks_by_id: dict[str, dict[str, Any]],
    protected_names: list[str],
) -> None:
    expected_blocks, raw_blocks = _translation_preflight(segment, translation, blocks_by_id)
    for source, translated in zip(expected_blocks, raw_blocks):
        text = str(translated.get("text") or "").strip()
        expected_tokens = _opaque_inline_tokens(source)
        actual_tokens = _OPAQUE_INLINE_PATTERN.findall(text)
        if actual_tokens != expected_tokens:
            raise TranslationOpaqueTokenError(
                segment_id=str(segment["segment_id"]),
                block_id_value=block_id(source),
                expected_tokens=expected_tokens,
                actual_tokens=actual_tokens,
            )
        _validate_names_in_generated([source], text, protected_names, label=block_id(source))


def _translation_preflight(
    segment: dict[str, Any],
    translation: dict[str, Any],
    blocks_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate segment-wide non-token invariants before any repair is allowed."""
    expected_blocks = [blocks_by_id[value] for value in segment.get("block_ids") or [] if _is_translatable(blocks_by_id[value])]
    expected_ids = [block_id(block) for block in expected_blocks]
    raw_blocks = translation.get("blocks") if isinstance(translation, dict) else None
    if not isinstance(raw_blocks, list):
        raise RuntimeError(f"translation {segment['segment_id']} has no block list")
    actual_ids = [str(item.get("block_id") or "") for item in raw_blocks if isinstance(item, dict)]
    if actual_ids != expected_ids or len(actual_ids) != len(raw_blocks):
        raise TranslationCoverageError(
            f"translation {segment['segment_id']} does not cover every translatable block exactly once"
        )
    for source, translated in zip(expected_blocks, raw_blocks):
        text = str(translated.get("text") or "").strip()
        source_has_natural_text = bool(_natural_text_for_name_validation(source).strip())
        if source_has_natural_text and not _translation_natural_residue(text).strip():
            raise RuntimeError(f"translation {segment['segment_id']} returned empty block {block_id(source)}")
    return expected_blocks, raw_blocks


def _translation_opaque_token_errors(
    segment: dict[str, Any],
    translation: dict[str, Any],
    blocks_by_id: dict[str, dict[str, Any]],
) -> list[TranslationOpaqueTokenError]:
    """Collect every token mismatch after ordinary coverage/non-empty checks passed."""
    expected_blocks, raw_blocks = _translation_preflight(segment, translation, blocks_by_id)
    errors: list[TranslationOpaqueTokenError] = []
    for source, translated in zip(expected_blocks, raw_blocks):
        if not isinstance(translated, dict):
            continue
        expected_tokens = _opaque_inline_tokens(source)
        actual_tokens = _OPAQUE_INLINE_PATTERN.findall(str(translated.get("text") or ""))
        if actual_tokens != expected_tokens:
            errors.append(TranslationOpaqueTokenError(
                segment_id=str(segment["segment_id"]),
                block_id_value=block_id(source),
                expected_tokens=expected_tokens,
                actual_tokens=actual_tokens,
            ))
    return errors


def _generation_document(document: dict[str, Any]) -> dict[str, Any]:
    """Remove source-only material while preserving it for final rendering."""
    excluded = _generation_excluded_block_ids(document)
    return {
        **document,
        "blocks": [
            block for block in document.get("blocks") or []
            if block_id(block) not in excluded
        ],
    }


def _generation_excluded_block_ids(document: dict[str, Any]) -> set[str]:
    return _front_matter_excluded_block_ids(document) | _source_only_block_ids(document)


def _source_only_block_ids(document: dict[str, Any]) -> set[str]:
    """Identify structurally preserved blocks that must never enter LLM generation."""
    blocks = list(document.get("blocks") or [])
    excluded: set[str] = set()
    source_only_sections: set[str] = set()
    for block in blocks:
        role = str(block.get("source_role") or "").casefold()
        kind = str(block.get("type") or block.get("kind") or "").casefold()
        inferred_role = role or _source_only_role_from_block(block)
        if inferred_role or kind in {"bibliography", "bibliography_item", "reference"}:
            excluded.add(block_id(block))
            section_id = str(block.get("section_id") or "")
            if section_id and inferred_role in {"acknowledgments", "references"}:
                source_only_sections.add(section_id)
    for block in blocks:
        if str(block.get("section_id") or "") in source_only_sections:
            excluded.add(block_id(block))
    return excluded


def _source_only_role_from_block(block: dict[str, Any]) -> str:
    html = str(block.get("html") or "").casefold()
    if re.search(r'(?:class|role)=["\'][^"\']*(?:ltx_toc|ltx_title_contents|doc-toc)', html):
        return "table_of_contents"
    if re.search(r'class=["\'][^"\']*acknowledg', html):
        return "acknowledgments"
    if re.search(r'class=["\'][^"\']*(?:bibliograph|reference)', html):
        return "references"
    kind = str(block.get("type") or block.get("kind") or "").casefold()
    if kind not in {"heading", "section", "subsection", "subsubsection"}:
        return ""
    title = re.sub(
        r"[^\w\u4e00-\u9fff]+",
        " ",
        str(block.get("title") or block.get("text") or "").casefold(),
    ).strip()
    if title in {"contents", "table of contents", "目录"}:
        return "table_of_contents"
    if title in {
        "acknowledgment", "acknowledgments", "acknowledgement", "acknowledgements", "致谢",
    }:
        return "acknowledgments"
    if title in {"references", "reference list", "bibliography", "literature cited", "参考文献"}:
        return "references"
    return ""


def _front_matter_excluded_block_ids(document: dict[str, Any]) -> set[str]:
    front = document.get("front_matter") or {}
    protected: set[str] = set()
    for key in ("title", "authors", "affiliations"):
        value = front.get(key)
        items = value if isinstance(value, list) else [value]
        for item in items:
            text = _normalized_front_text(item)
            if text:
                protected.add(text)
    return {
        block_id(block)
        for block in document.get("blocks") or []
        if _normalized_front_text(block.get("text") or block.get("title")) in protected
    }


def _normalized_front_text(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("text") or value.get("name") or ""
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _annotation_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in metadata.items()
        if key.casefold() not in {"title", "titles", "author", "authors", "affiliation", "affiliations"}
    }


def _protected_names(bundle: SourceBundle, *, glossary: dict[str, Any] | None = None) -> list[str]:
    """Protect seed/glossary names only; reference and citer authors stay evidence-local."""
    values: list[str] = []

    def collect(authors: Any) -> None:
        if isinstance(authors, str):
            values.append(authors)
        elif isinstance(authors, dict):
            name = authors.get("name") or authors.get("full_name")
            if name:
                values.append(str(name))
        elif isinstance(authors, list):
            for author in authors:
                collect(author)

    collect(bundle.metadata.get("authors"))
    collect((bundle.document.get("front_matter") or {}).get("authors"))
    if glossary:
        for entry in glossary.get("entries") or []:
            values.extend(str(value) for value in entry.get("protected_names") or [])
    unique: dict[str, str] = {}
    for value in values:
        text = re.sub(r"\s+", " ", str(value)).strip()
        if text:
            unique.setdefault(text.casefold(), text)
            # Also protect surname/name roots when a full author name is listed.
            for token in re.findall(r"[A-Za-z][A-Za-z'’-]{2,}", text):
                unique.setdefault(token.casefold(), token)
    return sorted(unique.values(), key=lambda value: (value.casefold(), value))


def _validate_names_in_generated(
    source_blocks: list[dict[str, Any]], generated: str, protected_names: list[str], *, label: str
) -> None:
    missing = _missing_protected_names(source_blocks, generated, protected_names)
    if missing:
        raise RuntimeError(f"generated text {label} translated or dropped protected names: {missing[:8]}")


def _missing_protected_names(
    source_blocks: list[dict[str, Any]], generated: str, protected_names: list[str],
) -> list[str]:
    source = "\n".join(_natural_text_for_name_validation(block) for block in source_blocks)
    return [
        name for name in protected_names
        if re.search(rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])", source, re.IGNORECASE)
        and not re.search(rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])", generated)
    ]


def _natural_text_for_name_validation(block: dict[str, Any]) -> str:
    """Exclude controller-owned math/citation/link content from name checks."""
    inline_runs = [item for item in block.get("inline_runs") or [] if isinstance(item, dict)]
    if inline_runs:
        return "\n".join(
            str(run.get("content") or "")
            for run in inline_runs
            if str(run.get("kind") or "") == "text"
        )
    return str(block.get("text") or "")


def _is_exact_name_insertion_delta(value: str, residue: str, names: list[str]) -> bool:
    """Accept only exact, boundary-delimited insertions of each requested Latin name."""
    def remove(current: str, remaining: tuple[str, ...]) -> bool:
        if not remaining:
            return current == residue
        name = remaining[0]
        matches = list(re.finditer(
            rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])", current,
        ))
        return any(
            remove(current[:match.start()] + current[match.end():], remaining[1:])
            for match in matches
        )

    return remove(value, tuple(names))


def _minimal_missing_name_insertions(missing_names: list[str]) -> list[str]:
    """Prefer a full missing name when its insertion also restores protected name roots."""
    return [
        name for name in missing_names
        if not any(
            other != name
            and re.search(rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])", other, re.IGNORECASE)
            for other in missing_names
        )
    ]


def _validated_evidence_ids(values: Any, evidence: dict[str, Any]) -> list[str]:
    records = list(evidence.get("related_papers") or [])
    if records and all(isinstance(item.get("source_descriptor"), dict) for item in records):
        return validate_cited_ids(values, records)
    valid = {str(item.get("evidence_id") or "") for item in records}
    ids = [str(value) for value in values]
    unknown = sorted(set(ids) - valid)
    if unknown:
        raise RuntimeError(f"annotation cited unknown evidence IDs: {unknown}")
    return list(dict.fromkeys(ids))


def _validate_annotation_evidence(annotation: dict[str, Any], papers: list[dict[str, Any]]) -> None:
    if papers and all(isinstance(item.get("source_descriptor"), dict) for item in papers):
        validate_annotation_citations(annotation, papers)
        return
    relation_by_id = {str(item.get("evidence_id") or ""): item.get("relation") for item in papers}
    used = set(str(value) for value in annotation.get("evidence_ids") or [])
    if str(annotation.get("prior_work") or "").strip() and not any(
        relation_by_id.get(value) == "prior" for value in used
    ):
        raise RuntimeError("prior-work commentary has no cited prior-work evidence")
    if str(annotation.get("later_work") or "").strip() and not any(
        relation_by_id.get(value) == "later" for value in used
    ):
        raise RuntimeError("later-work commentary has no cited later-work evidence")


def _evidence_for_segment(
    segment: dict[str, Any],
    blocks_by_id: dict[str, dict[str, Any]],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    source_text = " ".join(
        str(blocks_by_id[value].get("text") or blocks_by_id[value].get("title") or "")
        for value in segment.get("block_ids") or []
    )
    source_tokens = _terms(source_text)
    selected: list[dict[str, Any]] = []
    for relation in ("prior", "later"):
        candidates: list[tuple[int, int, dict[str, Any]]] = []
        for index, paper in enumerate(evidence.get("related_papers") or []):
            if paper.get("relation") != relation:
                continue
            search_text = " ".join([
                str(paper.get("title") or ""),
                str(paper.get("abstract") or ""),
                " ".join(str(block.get("text") or "") for block in paper.get("blocks") or []),
            ])
            score = len(source_tokens & _terms(search_text))
            candidates.append((-score, index, paper))
        for _, _, paper in sorted(candidates)[:2]:
            compact = {key: paper.get(key) for key in (
                "evidence_id", "relation", "paper_id", "title", "authors", "year",
                "citation_count", "evidence_level", "abstract",
            )}
            remaining = 4_000
            snippets: list[dict[str, str]] = []
            ranked_blocks = sorted(
                paper.get("blocks") or [],
                key=lambda block: -len(source_tokens & _terms(str(block.get("text") or ""))),
            )
            for block in ranked_blocks:
                text = str(block.get("text") or "")[:remaining]
                if not text:
                    continue
                snippets.append({
                    "block_id": str(block.get("block_id") or ""),
                    "text": text,
                    "sha256": text_sha256(text),
                })
                remaining -= len(text)
                if remaining <= 0:
                    break
            compact["snippets"] = snippets
            descriptor = paper.get("source_descriptor")
            if isinstance(descriptor, dict):
                if snippets:
                    locator = descriptor.get("locator") or {}
                    compact["source_descriptor"] = arc_cache_descriptor(
                        paper_id=str(paper.get("paper_id") or descriptor.get("canonical_locator") or ""),
                        title=str(paper.get("title") or ""),
                        authors=paper.get("authors") or [],
                        year=paper.get("year"),
                        evidence_level=str(paper.get("evidence_level") or "full_text"),
                        content=snippets,
                        document_hash=str(locator.get("document_hash") or ""),
                    )
                else:
                    compact["source_descriptor"] = descriptor
                validate_evidence_record(compact)
            selected.append(compact)
    required_ids = set(
        (evidence.get("required_evidence_ids_by_segment") or {}).get(str(segment.get("segment_id")), [])
    )
    selected_ids = {str(item.get("evidence_id") or "") for item in selected}
    for paper in evidence.get("related_papers") or []:
        evidence_id = str(paper.get("evidence_id") or "")
        if evidence_id in required_ids and evidence_id not in selected_ids:
            selected.append(paper)
            selected_ids.add(evidence_id)
    return {"schema_version": "arc.companion.segment-evidence.v2", "papers": selected}


def _terms(text: str) -> set[str]:
    return {token.casefold() for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", text)}


def _review_chunks(items: list[dict[str, Any]], target_chars: int) -> list[list[dict[str, Any]]]:
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    size = 0
    for item in items:
        item_size = len(json.dumps(item, ensure_ascii=False))
        if current and size + item_size > max(10_000, target_chars):
            chunks.append(current)
            boundary = current[-1]
            current = [boundary]
            size = len(json.dumps(boundary, ensure_ascii=False))
        current.append(item)
        size += item_size
    if current:
        chunks.append(current)
    return chunks


def _review_source_anchors(
    segments: list[dict[str, Any]],
    *,
    blocks_by_id: dict[str, dict[str, Any]],
    document: dict[str, Any],
    total_chars: int,
) -> list[dict[str, Any]]:
    """Give the hierarchical final reviewer bounded source context for every segment."""
    per_segment = max(160, min(1_200, total_chars // max(1, len(segments))))
    anchors: list[dict[str, Any]] = []
    for segment in segments:
        projection = [
            _annotation_input_block(blocks_by_id[value], document)
            for value in segment.get("block_ids") or []
        ]
        serialized = json.dumps(projection, ensure_ascii=False, separators=(",", ":"))
        if len(serialized) > per_segment:
            half = max(1, (per_segment - 3) // 2)
            excerpt = serialized[:half] + "..." + serialized[-half:]
        else:
            excerpt = serialized
        anchors.append({
            "segment_id": str(segment.get("segment_id") or ""),
            "title": str(segment.get("title") or ""),
            "start_block_id": str(segment.get("start_block_id") or ""),
            "end_block_id": str(segment.get("end_block_id") or ""),
            "source_sha256": sha256_json(projection),
            "source_excerpt": excerpt,
        })
    return anchors


def _state(path: Path, **values: Any) -> dict[str, Any]:
    state = {**_read_optional_json(path), **{key: value for key, value in values.items() if value is not None}}
    if values.get("status") and values.get("status") != "failed":
        state.pop("error", None)
    state["schema_version"] = "arc.companion.state.v1"
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(path, state)
    return state


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = read_json(path)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}
