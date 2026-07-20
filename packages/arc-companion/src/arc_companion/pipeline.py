from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import uuid
from typing import Any, Callable

from bs4 import BeautifulSoup

from .context_sources import load_context_evidence
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
    EVIDENCE_RESOLUTION_VERSION,
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
    TRANSLATION_SLOT_REPAIR_SCHEMA_VERSION,
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
from .reader_text import clean_reader_annotation, clean_reader_translation
from .results import err, ok
from .segmentation import SegmentationError, segment_document, validate_exact_coverage
from .source import SourceBundle, SourceError, block_id, load_source_bundle
from .substantive import non_substantive_block_ids


WORKFLOW_VERSION = "arc.companion.workflow.v9"
DEFAULT_LANGUAGE = "zh-CN"
DEFAULT_WORKERS = 24
DEFAULT_REVIEW_CONTEXT_CHARS = 140_000
LANGUAGE_NOTICE = "默认使用中文生成伴读；如需切换伴读语言，请直接指定目标语言。"
SEGMENTATION_TIER = "medium"
GLOSSARY_TIER = "medium"
TRANSLATION_TIER = "low"
TRANSLATION_RETRY_TIER = "medium"
TRANSLATION_COVERAGE_REPAIR_TIER = "medium"
TRANSLATION_CITATION_DELIMITER_NORMALIZER_VERSION = (
    "arc.companion.translation-citation-delimiters.v2"
)
TRANSLATION_PROTECTED_NAME_NORMALIZER_VERSION = (
    "arc.companion.translation-protected-names.v2"
)
TRANSLATION_TOKEN_REPAIR_VERSION = "arc.companion.translation-token-repair.v3"
ANNOTATION_TIER = "high"
REVIEW_TIER = "high"
REVIEW_VERSION = "arc.companion.review.v3"
ANNOTATION_CHECKPOINT_VERSION = "arc.companion.annotation-checkpoint.v5"
SECTION_REVIEW_CHECKPOINT_VERSION = "arc.companion.section-review-checkpoint.v1"
FULL_PAPER_CONTEXT_VERSION = "arc.companion.full-paper-context.v1"
FULL_PAPER_CONTEXT_CHARS = 24_000
FIRST_WAVE_PREVIEW_VERSION = "arc.companion.first-wave-preview.v2"
FINAL_RENDER_VERSION = "arc.companion.final-render.v3"
CONTEXT_SELECTION_VERSION = "arc.companion.context-selection.v2"
CONTEXT_SEGMENT_CHARS_PER_SOURCE = 3_000
CONTEXT_SEGMENT_CHARS_TOTAL = 12_000

_MCP_CONFIG_ENV_KEYS = (
    "ARC_CODEX_PROFILE",
    "ARC_CODEX_PROFILE_V2",
    "ARC_CODEX_CONFIG",
    "ARC_CODEX_CONFIG_JSON",
    "ARC_CODEX_MCP_MODE",
    "ARC_CODEX_ARC_MCP_COMMAND",
    "ARC_CODEX_ARC_MCP_ENV_JSON",
    "ARC_CLAUDE_MCP_MODE",
    "ARC_CLAUDE_MCP_CONFIG",
    "ARC_CLAUDE_MCP_CONFIG_JSON",
    "ARC_CLAUDE_STRICT_MCP_CONFIG",
    "ARC_CLAUDE_ARC_ONLY_ALLOW_EXTRA_CONFIGS",
    "ARC_CLAUDE_ARC_MCP_COMMAND",
    "ARC_CLAUDE_ARC_MCP_ARGS_JSON",
    "ARC_CLAUDE_ARC_MCP_ENV_JSON",
    "ARC_CLAUDE_ARC_MCP_CONFIG_PATH",
    "ARC_CLAUDE_TOOLS",
    "ARC_CLAUDE_ALLOWED_TOOLS",
)


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
    allow_mcp: bool = True
    allow_internet: bool = True
    context_paper_ids: tuple[str, ...] = ()
    stop_after_preview: bool = False

    def __post_init__(self) -> None:
        if not self.paper_id.strip():
            raise ValueError("paper_id is required")
        if self.workers < 1:
            raise ValueError("workers must be at least 1")
        if self.refresh and self.recache:
            raise ValueError("refresh and recache are mutually exclusive")
        if self.domain_id and self.domain_manifest is not None:
            raise ValueError("domain_id and domain_manifest are mutually exclusive")
        normalized_context_ids = tuple(
            dict.fromkeys(str(value).strip() for value in self.context_paper_ids if str(value).strip())
        )
        if len(normalized_context_ids) != len(self.context_paper_ids):
            raise ValueError("context_paper_ids must be non-empty and unique")
        if self.paper_id.strip() in normalized_context_ids:
            raise ValueError("the source paper cannot also be a context paper")
        object.__setattr__(self, "context_paper_ids", normalized_context_ids)


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
        context_evidence = (
            load_context_evidence(options.context_paper_ids) if options.context_paper_ids else []
        )
        evidence = _evidence(bundle, context_evidence=context_evidence)
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
            evidence=evidence,
            glossary=glossary,
            metadata=bundle.metadata,
            language=options.annotation_language,
            output_dir=project_dir,
            stem=f"{stem}_first_round_preview",
            manifest_name="first-round-preview-source-manifest.json",
            validation_name="first-round-preview-validation.json",
            compiler=compiler,
            pdf_validator=pdf_validator,
            augmentation_scope="substantive",
        )
        preview_state = _state(
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
        if options.stop_after_preview:
            return ok(
                preview_state,
                resumed=False,
                notice=notice,
                diagnostics=list(diagnostics),
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
            controller=evidence_controller or EvidenceRequestController(
                domain_paper_ids=(domain_context or {}).get("paper_ids") or (),
                seed_paper_ids=(bundle.paper_id,),
            ),
        )

        _state(state_path, status="reviewing", paper_id=bundle.paper_id, fingerprint=fingerprint, notice=notice,
               segment_count=len(expanded))
        reviewed_path = checkpoint_dir / "annotations.reviewed.v3.json"
        review_path = checkpoint_dir / "review.v3.json"
        if reviewed_path.is_file() and review_path.is_file() and not options.force:
            cached_reviewed = read_json(reviewed_path)
            review = read_json(review_path)
            if (
                not isinstance(cached_reviewed, dict)
                or cached_reviewed.get("schema_version") != REVIEW_VERSION
            ):
                raise RuntimeError("invalid review checkpoint")
            cached_reviewed, normalized_cached_segments = (
                _repair_reviewed_translation_checkpoint(
                    reviewed_path,
                    expanded,
                    {block_id(block): block for block in bundle.document.get("blocks") or []},
                    protected_names=protected_names,
                )
            )
            if normalized_cached_segments:
                review = {
                    **review,
                    "citation_delimiter_normalized_cached_segment_ids": (
                        normalized_cached_segments
                    ),
                }
                write_json(review_path, review)
            reviewed = cached_reviewed.get("annotations")
            reviewed_translations = cached_reviewed.get("translations")
            if (
                not isinstance(reviewed, dict)
                or not isinstance(reviewed_translations, dict)
                or set(reviewed) != {segment["segment_id"] for segment in expanded}
                or set(reviewed_translations) != set(reviewed)
            ):
                raise RuntimeError("review checkpoint does not match current segments")
            cached_reader_evidence = _reader_evidence_by_segment(
                expanded,
                document=bundle.document,
                evidence=evidence,
                annotations=reviewed,
            )
            reviewed = {
                str(segment_id): clean_reader_annotation(
                    annotation,
                    evidence_records=cached_reader_evidence.get(str(segment_id), []),
                    language=options.annotation_language,
                )
                for segment_id, annotation in reviewed.items()
            }
            reviewed_translations = {
                str(segment_id): clean_reader_translation(translation)
                for segment_id, translation in reviewed_translations.items()
            }
            if (
                reviewed != cached_reviewed.get("annotations")
                or reviewed_translations != cached_reviewed.get("translations")
            ):
                cached_reviewed["annotations"] = reviewed
                cached_reviewed["translations"] = reviewed_translations
                write_json(reviewed_path, cached_reviewed)
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
            evidence=evidence,
            glossary=glossary,
            metadata=bundle.metadata,
            language=options.annotation_language,
            output_dir=project_dir,
            stem=stem,
            manifest_name="source-manifest.json",
            validation_name="validation.json",
            compiler=compiler,
            pdf_validator=pdf_validator,
            augmentation_scope="substantive",
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
            final_render_version=FINAL_RENDER_VERSION,
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
    evidence: dict[str, Any] | None = None,
    augmentation_scope: str = "all",
) -> dict[str, Any]:
    """Render, validate, and atomically publish one preview or final PDF artifact."""
    evidence_by_segment = _reader_evidence_by_segment(
        segments,
        document=document,
        evidence=evidence or {},
        annotations=annotations,
    )
    tex, source_manifest = render_companion_tex(
        document,
        segments,
        annotations,
        output_dir=output_dir,
        language=language,
        metadata=metadata,
        translations=translations,
        glossary=glossary,
        evidence_by_segment=evidence_by_segment,
        augmentation_scope=augmentation_scope,
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


def _reader_evidence_by_segment(
    segments: list[dict[str, Any]],
    *,
    document: dict[str, Any],
    evidence: dict[str, Any],
    annotations: dict[str, dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Attach reader-facing source labels without changing evidence identity."""
    blocks_by_id = {
        block_id(block): block for block in document.get("blocks") or []
    }
    full_by_id = {
        str(record.get("evidence_id") or ""): record
        for record in evidence.get("related_papers") or []
        if isinstance(record, dict) and record.get("evidence_id")
    }
    result: dict[str, list[dict[str, Any]]] = {}
    for segment in segments:
        selected = _evidence_for_segment(segment, blocks_by_id, evidence).get("papers") or []
        records: list[dict[str, Any]] = []
        for compact in selected:
            evidence_id = str(compact.get("evidence_id") or "")
            full = full_by_id.get(evidence_id)
            if full is None:
                records.append(dict(compact))
                continue
            record = dict(full)
            record["selected_snippets"] = list(compact.get("snippets") or [])
            records.append(record)
        segment_id = str(segment.get("segment_id") or "")
        present = {str(record.get("evidence_id") or "") for record in records}
        cited = (
            (annotations or {}).get(segment_id, {}).get("evidence_ids") or []
        )
        for evidence_id in cited:
            full = full_by_id.get(str(evidence_id))
            if full is not None and str(evidence_id) not in present:
                records.append(dict(full))
                present.add(str(evidence_id))
        result[segment_id] = records
    return result


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
    usage_state: dict[str, Any] = {"counts": {}, "topics": []}
    segment_evidence_by_id = {
        str(segment["segment_id"]): _evidence_for_segment(
            segment, by_id, evidence, usage_state=usage_state,
        )
        for segment in segments
    }
    annotation_dir = checkpoint_dir / ("annotations" if round_number == 1 else "annotations-evidence-rerun")
    segment_evidence_dir = checkpoint_dir / "segment-evidence"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    segment_evidence_dir.mkdir(parents=True, exist_ok=True)
    output: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for segment in segments:
        path = annotation_dir / f"{_segment_checkpoint_name(segment['segment_id'])}.json"
        segment_evidence = segment_evidence_by_id[str(segment["segment_id"])]
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
            paper_context = _full_paper_context(
                bundle.document, segment, blocks_by_id=by_id, options=options
            )
            expected_hash = _segment_input_hash(
                segment, by_id, glossary=glossary,
                extra={
                    "evidence": segment_evidence,
                    "names": protected_names,
                    "paper_context": paper_context,
                    "runtime_access": _generation_runtime_policy(options),
                    "domain_context": domain_context,
                    "round": round_number,
                    "first_draft": (first_drafts or {}).get(str(segment["segment_id"])),
                    "evidence_resolution": (resolution_by_segment or {}).get(str(segment["segment_id"])),
                },
            )
            if (
                isinstance(checkpoint, dict)
                and checkpoint.get("schema_version") == ANNOTATION_CHECKPOINT_VERSION
                and checkpoint.get("segment_id") == segment["segment_id"]
                and checkpoint.get("input_sha256") == expected_hash
                and isinstance(checkpoint.get("annotation"), dict)
            ):
                output[segment["segment_id"]] = clean_reader_annotation(
                    checkpoint["annotation"],
                    evidence_records=segment_evidence["papers"],
                    language=options.annotation_language,
                )
                continue
        pending.append(segment)

    def generate(segment: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        selected = [_annotation_input_block(by_id[value], bundle.document) for value in segment["block_ids"]]
        segment_evidence = segment_evidence_by_id[str(segment["segment_id"])]
        paper_context = _full_paper_context(
            bundle.document, segment, blocks_by_id=by_id, options=options
        )
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
        normalized = {
            "commentary": str(value.get("commentary") or ""),
            "explanation": str(value.get("explanation") or ""),
            "prior_work": _normalize_related_work(
                value.get("prior_work"),
                known_evidence_ids=(
                    None if round_number == 1
                    else {str(item.get("evidence_id") or "") for item in segment_evidence["papers"]}
                ),
            ),
            "later_work": _normalize_related_work(
                value.get("later_work"),
                known_evidence_ids=(
                    None if round_number == 1
                    else {str(item.get("evidence_id") or "") for item in segment_evidence["papers"]}
                ),
            ),
            "context_claims": _normalize_related_work(
                value.get("context_claims") or [],
                known_evidence_ids=(
                    None if round_number == 1
                    else {str(item.get("evidence_id") or "") for item in segment_evidence["papers"]}
                ),
            ),
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
        normalized = clean_reader_annotation(
            normalized,
            evidence_records=segment_evidence["papers"],
            language=options.annotation_language,
        )
        _repair_unique_supplied_source_locators(normalized, segment_evidence["papers"])
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
                        "schema_version": ANNOTATION_CHECKPOINT_VERSION,
                        "round": round_number,
                        "segment_id": segment_id,
                        "input_sha256": _segment_input_hash(
                            segment, by_id, glossary=glossary,
                            extra={
                                "evidence": segment_evidence_by_id[str(segment_id)],
                                "names": protected_names,
                                "paper_context": _full_paper_context(
                                    bundle.document, segment, blocks_by_id=by_id, options=options
                                ),
                                "runtime_access": _generation_runtime_policy(options),
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
        write_json(checkpoint_dir / "evidence-resolution.v3.json", {
            "schema_version": EVIDENCE_RESOLUTION_VERSION,
            "requests": [],
            "lanes": {},
            "accepted": [],
            "rejected": [],
            "rerun_segments": [],
        })
        return annotations, evidence

    resolution = controller.resolve(requests, existing_records=evidence.get("related_papers") or [])
    supported = set(resolution.supported_request_keys)
    evidence_ids_by_request: dict[str, set[str]] = {}
    for accepted in resolution.audit.get("accepted") or []:
        if not isinstance(accepted, dict):
            continue
        request_key = str(accepted.get("request_key") or "")
        evidence_id = str(accepted.get("evidence_id") or "")
        if request_key in supported and evidence_id:
            evidence_ids_by_request.setdefault(request_key, set()).add(evidence_id)
    for segment_id, ids in resolution.evidence_ids_by_segment.items():
        segment_keys = [
            str(item["request_key"]) for item in requests
            if item["segment_id"] == segment_id and item["request_key"] in supported
        ]
        if len(segment_keys) == 1 and not evidence_ids_by_request.get(segment_keys[0]):
            evidence_ids_by_request[segment_keys[0]] = set(ids)
    audit = dict(resolution.audit)
    audit["round"] = 1
    audit["rerun_segments"] = sorted(resolution.evidence_ids_by_segment)
    write_json(checkpoint_dir / "evidence-resolution.v3.json", audit)

    merged_records = _merge_evidence_records(
        evidence.get("related_papers") or [], resolution.records,
    )
    merged_evidence = {
        **evidence,
        "related_papers": merged_records,
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
            "evidence_ids_by_request": {
                item["request_key"]: sorted(evidence_ids_by_request.get(item["request_key"], set()))
                for item in requests
                if item["segment_id"] == segment_id and item["request_key"] in supported
            },
            "requests": [item for item in requests if item["segment_id"] == segment_id],
            "audit_path": str(checkpoint_dir / "evidence-resolution.v3.json"),
        }
        for segment_id in rerun_ids
    }
    first_round_annotations = annotations
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
    claim_binding_records: dict[str, list[dict[str, Any]]] = {}
    for segment_id, annotation in annotations.items():
        segment_requests = [item for item in requests if item["segment_id"] == segment_id]
        _enforce_request_claim_bindings(
            annotation,
            segment_requests,
            supported=supported,
            evidence_ids_by_request=evidence_ids_by_request,
            first_draft=first_round_annotations.get(segment_id),
            records=merged_records,
        )
        _clear_unresolved_requested_work(annotation, segment_requests, supported)
        annotation["evidence_requests"] = []
        claim_bindings[segment_id] = list(annotation.get("evidence_ids") or [])
        claim_binding_records[segment_id] = [
            {
                "relation": relation,
                "text": str(claim.get("text") or ""),
                "request_key": claim.get("request_key"),
                "evidence_ids": list(claim.get("evidence_ids") or []),
                "source_locators": list(claim.get("source_locators") or []),
            }
            for field, relation in (
                ("prior_work", "prior"), ("later_work", "later"),
                ("context_claims", "context"),
            )
            for claim in annotation.get(field) or []
            if isinstance(annotation.get(field), list) and isinstance(claim, dict)
        ]
    audit["round"] = 2
    audit["final_claim_evidence_ids"] = claim_bindings
    audit["final_claim_bindings"] = claim_binding_records
    write_json(checkpoint_dir / "evidence-resolution.v3.json", audit)
    return annotations, merged_evidence


def _normalize_related_work(
    value: Any,
    *,
    known_evidence_ids: set[str] | None = None,
) -> str | list[dict[str, Any]]:
    """Preserve legacy strings while normalizing claim-level evidence bindings."""
    if not isinstance(value, list):
        raise RuntimeError("related work must use claim arrays in generated annotations")
    if len(value) > 3:
        raise RuntimeError("related work may contain at most three claims")
    claims: list[dict[str, Any]] = []
    for index, raw in enumerate(value, 1):
        if not isinstance(raw, dict) or not str(raw.get("text") or "").strip():
            raise RuntimeError(f"related-work claim {index} has no text")
        evidence_ids = [
            str(item) for item in raw.get("evidence_ids") or []
            if isinstance(item, str)
            and (known_evidence_ids is None or item in known_evidence_ids)
        ]
        locator_values = [
            {"evidence_id": str(item.get("evidence_id") or ""),
             "locator": str(item.get("locator") or "")}
            for item in raw.get("source_locators") or []
            if isinstance(item, dict)
            and str(item.get("evidence_id") or "") in evidence_ids
        ]
        request_key = raw.get("request_key")
        claims.append({
            "text": str(raw["text"]),
            "evidence_ids": list(dict.fromkeys(evidence_ids)),
            "source_locators": locator_values,
            "request_key": None if request_key is None else str(request_key),
        })
    return claims


def _repair_unique_supplied_source_locators(
    annotation: dict[str, Any], records: list[dict[str, Any]],
) -> None:
    """Correct an invented locator only when the registered choice is unique.

    Models occasionally copy the reader-facing section title into the
    structured provenance field. That value is not accepted as provenance.
    If the cited evidence record contains exactly one supplied source piece,
    the exact locator can instead be restored deterministically before strict
    validation. Missing locators, unknown evidence IDs, and records with
    multiple candidate locators remain errors.
    """
    allowed_by_id: dict[str, set[str]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        evidence_id = str(record.get("evidence_id") or "")
        pieces = record.get("snippets")
        if pieces is None:
            pieces = record.get("blocks")
        locators = {
            str(piece.get("locator") or piece.get("block_id") or "").strip()
            for piece in pieces or []
            if isinstance(piece, dict)
        }
        locators.discard("")
        if not locators and str(record.get("abstract") or "").strip():
            locators.add("abstract")
        if evidence_id:
            allowed_by_id[evidence_id] = locators

    for field in ("prior_work", "later_work", "context_claims"):
        claims = annotation.get(field)
        if not isinstance(claims, list):
            continue
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            for binding in claim.get("source_locators") or []:
                if not isinstance(binding, dict):
                    continue
                evidence_id = str(binding.get("evidence_id") or "")
                supplied = allowed_by_id.get(evidence_id, set())
                emitted = str(binding.get("locator") or "").strip()
                if emitted and emitted not in supplied and len(supplied) == 1:
                    binding["locator"] = next(iter(supplied))


def _merge_evidence_records(
    existing: list[dict[str, Any]], updates: tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    """Replace updated registry records in place and append genuinely new records."""
    output = [dict(item) for item in existing]
    index = {str(item.get("evidence_id") or ""): position for position, item in enumerate(output)}
    for raw in updates:
        item = dict(raw)
        evidence_id = str(item.get("evidence_id") or "")
        if evidence_id in index:
            output[index[evidence_id]] = item
        else:
            index[evidence_id] = len(output)
            output.append(item)
    return output


def _known_evidence_ids(values: Any, records: list[dict[str, Any]]) -> list[str]:
    valid = {str(item.get("evidence_id") or "") for item in records}
    return [str(value) for value in values if str(value) in valid]


def _drop_unsupported_second_round_related_work(
    annotation: dict[str, Any], records: list[dict[str, Any]],
) -> None:
    relation_by_id = {str(item.get("evidence_id") or ""): str(item.get("relation") or "") for item in records}
    used = list(annotation.get("evidence_ids") or [])
    for field, relation in (
        ("prior_work", "prior"), ("later_work", "later"),
        ("context_claims", "context"),
    ):
        value = annotation.get(field)
        if isinstance(value, list):
            annotation[field] = [
                claim for claim in value
                if isinstance(claim, dict)
                and claim.get("evidence_ids")
                and all(
                    relation_by_id.get(str(evidence_id)) == relation
                    for evidence_id in claim.get("evidence_ids") or []
                )
            ]
        elif not any(relation_by_id.get(evidence_id) == relation for evidence_id in used):
            annotation[field] = []
    if any(isinstance(annotation.get(field), list) for field in (
        "prior_work", "later_work", "context_claims",
    )):
        _sync_claim_evidence_ids(annotation)
        return
    used_relations = {
        relation for relation, field in (("prior", "prior_work"), ("later", "later_work"))
        if str(annotation.get(field) or "").strip()
    }
    if str(annotation.get("explanation") or annotation.get("commentary") or "").strip():
        used_relations.add("context")
    annotation["evidence_ids"] = [
        value for value in used if relation_by_id.get(value) in used_relations
    ]


def _clear_unresolved_requested_work(
    annotation: dict[str, Any], requests: list[dict[str, Any]], supported: set[str],
) -> None:
    unresolved = {
        str(item["request_key"]): str(item["relation"])
        for item in requests if str(item["request_key"]) not in supported
    }
    for field, relation in (
        ("prior_work", "prior"), ("later_work", "later"),
        ("context_claims", "context"),
    ):
        value = annotation.get(field)
        relation_keys = {key for key, item_relation in unresolved.items() if item_relation == relation}
        if not relation_keys:
            continue
        if isinstance(value, list):
            annotation[field] = [
                claim for claim in value
                if not isinstance(claim, dict)
                or str(claim.get("request_key") or "") not in relation_keys
            ]
        else:
            annotation[field] = []
    _sync_claim_evidence_ids(annotation)


def _enforce_request_claim_bindings(
    annotation: dict[str, Any],
    requests: list[dict[str, Any]],
    *,
    supported: set[str],
    evidence_ids_by_request: dict[str, set[str]],
    first_draft: dict[str, Any] | None,
    records: list[dict[str, Any]],
) -> None:
    """Drop only request-bound claims that ignored their specifically resolved evidence."""
    request_by_key = {str(item["request_key"]): item for item in requests}
    supported_keys_by_id = {
        str(item.get("evidence_id") or ""): {
            str(key) for key in item.get("supported_request_keys") or []
        }
        for item in records
    }
    for field, relation in (
        ("prior_work", "prior"), ("later_work", "later"),
        ("context_claims", "context"),
    ):
        value = annotation.get(field)
        resolved_keys = {
            key for key, request in request_by_key.items()
            if request["relation"] == relation and key in supported
        }
        if not resolved_keys:
            continue
        if not isinstance(value, list):
            annotation[field] = []
            continue
        prior_bindings = {
            _claim_binding_key(claim)
            for claim in (first_draft or {}).get(field) or []
            if isinstance(claim, dict)
        } if isinstance((first_draft or {}).get(field), list) else set()
        kept: list[dict[str, Any]] = []
        for claim in value:
            if not isinstance(claim, dict):
                continue
            request_key = str(claim.get("request_key") or "")
            if request_key not in resolved_keys:
                if _claim_binding_key(claim) in prior_bindings:
                    kept.append(claim)
                continue
            expected = evidence_ids_by_request.get(request_key, set())
            cited = {str(item) for item in claim.get("evidence_ids") or []}
            located = {
                str(item.get("evidence_id") or "")
                for item in claim.get("source_locators") or [] if isinstance(item, dict)
            }
            bound = cited.intersection(expected).intersection(located)
            if bound and any(request_key in supported_keys_by_id.get(item, set()) for item in bound):
                kept.append(claim)
        annotation[field] = kept
    _sync_claim_evidence_ids(annotation)


def _claim_binding_key(claim: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(claim.get("text") or "").strip(),
        claim.get("request_key"),
        tuple(str(item) for item in claim.get("evidence_ids") or []),
        tuple(
            (str(item.get("evidence_id") or ""), str(item.get("locator") or ""))
            for item in claim.get("source_locators") or [] if isinstance(item, dict)
        ),
    )


def _sync_claim_evidence_ids(annotation: dict[str, Any]) -> None:
    claim_fields = [
        annotation.get("prior_work"), annotation.get("later_work"),
        annotation.get("context_claims"),
    ]
    if not any(isinstance(value, list) for value in claim_fields):
        return
    claim_ids = list(dict.fromkeys(
        str(evidence_id)
        for value in claim_fields if isinstance(value, list)
        for claim in value if isinstance(claim, dict)
        for evidence_id in claim.get("evidence_ids") or []
    ))
    if any(
        not isinstance(value, list) and str(value or "").strip()
        for value in claim_fields
    ):
        claim_ids.extend(
            str(value) for value in annotation.get("evidence_ids") or []
            if str(value) not in claim_ids
        )
    annotation["evidence_ids"] = claim_ids


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


def _translation_token_attempt_path(checkpoint_dir: Path, segment_id: str) -> Path:
    return (
        checkpoint_dir
        / "translation-token-offset-attempts"
        / f"{_segment_checkpoint_name(segment_id)}.json"
    )


def _legacy_translation_token_attempt_path(
    checkpoint_dir: Path, segment_id: str,
) -> Path:
    return (
        checkpoint_dir
        / "translation-token-attempts"
        / f"{_segment_checkpoint_name(segment_id)}.json"
    )


def _read_checkpoint_json(path: Path) -> Any | None:
    """Read a checkpoint without letting a torn/corrupt file trigger fresh model work."""
    try:
        return read_json(path) if path.is_file() else None
    except (OSError, ValueError, TypeError):
        return None


def _matching_translation_token_attempt(
    checkpoint_dir: Path, segment_id: str, input_sha256: str,
) -> dict[str, Any] | None:
    path = _translation_token_attempt_path(checkpoint_dir, segment_id)
    value = _read_checkpoint_json(path)
    if (
        isinstance(value, dict)
        and value.get("schema_version") == "arc.companion.translation-token-attempt.v2"
        and value.get("prompt_version") == TRANSLATION_RETRY_PROMPT_VERSION
        and value.get("segment_id") == segment_id
        and value.get("input_sha256") == input_sha256
    ):
        return value
    return None


def _matching_superseded_translation_text_attempt(
    checkpoint_dir: Path, segment_id: str, input_sha256: str,
) -> dict[str, Any] | None:
    """Return a same-input legacy text repair for audit and low-rerun guards."""
    value = _read_checkpoint_json(
        _legacy_translation_token_attempt_path(checkpoint_dir, segment_id)
    )
    if (
        isinstance(value, dict)
        and value.get("segment_id") == segment_id
        and value.get("input_sha256") == input_sha256
        and str(value.get("prompt_version") or "") in {
            "arc.companion.translation-retry-prompt.v3",
            "arc.companion.translation-retry-prompt.v4",
        }
    ):
        return value
    return None


def _guard_translation_token_attempt_before_primary(
    checkpoint_dir: Path, segment_id: str, input_sha256: str,
) -> dict[str, Any] | None:
    """Fail closed on an unreadable or malformed current marker before low work."""
    path = _translation_token_attempt_path(checkpoint_dir, segment_id)
    if not path.is_file():
        return _matching_superseded_translation_text_attempt(
            checkpoint_dir, segment_id, input_sha256,
        )
    value = _read_checkpoint_json(path)
    if not isinstance(value, dict):
        raise RuntimeError(
            f"translation token repair marker is unreadable for {segment_id}; "
            "refusing a primary model call"
        )
    prompt_version = str(value.get("prompt_version") or "")
    if prompt_version != TRANSLATION_RETRY_PROMPT_VERSION:
        raise RuntimeError(
            f"translation token repair marker has unknown prompt identity for {segment_id}; "
            "refusing a primary model call"
        )
    if not (
        value.get("schema_version") == "arc.companion.translation-token-attempt.v2"
        and value.get("prompt_version") == TRANSLATION_RETRY_PROMPT_VERSION
        and value.get("segment_id") == segment_id
        and value.get("input_sha256") == input_sha256
    ):
        raise RuntimeError(
            f"translation token repair marker has malformed current identity for {segment_id}; "
            "refusing a primary model call"
        )
    return value


def _translation_token_repair_draft_path(
    checkpoint_dir: Path, segment_id: str,
) -> Path:
    return (
        checkpoint_dir
        / "translation-token-offset-repair-drafts"
        / f"{_segment_checkpoint_name(segment_id)}.json"
    )


def _matching_translation_token_repair_draft(
    checkpoint_dir: Path, segment_id: str, input_sha256: str,
) -> dict[str, Any] | None:
    path = _translation_token_repair_draft_path(checkpoint_dir, segment_id)
    value = _read_checkpoint_json(path)
    if (
        isinstance(value, dict)
        and value.get("schema_version") == "arc.companion.translation-token-repair-draft.v1"
        and value.get("prompt_version") == TRANSLATION_RETRY_PROMPT_VERSION
        and value.get("response_schema_version")
        == TRANSLATION_SLOT_REPAIR_SCHEMA_VERSION
        and value.get("segment_id") == segment_id
        and value.get("input_sha256") == input_sha256
        and isinstance(value.get("translation"), dict)
        and isinstance(value.get("repair_provenance"), dict)
        and value["repair_provenance"].get("repair_mode") == "offset-only"
        and isinstance(value.get("raw_response"), dict)
    ):
        return value
    return None


def _write_validated_translation_token_marker(
    checkpoint_dir: Path,
    segment_id: str,
    input_sha256: str,
    *,
    repaired: dict[str, Any],
    raw_response: dict[str, Any],
    repaired_block_ids: list[str],
    prior_marker: dict[str, Any] | None = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    write_json(_translation_token_attempt_path(checkpoint_dir, segment_id), {
        "schema_version": "arc.companion.translation-token-attempt.v2",
        "segment_id": segment_id,
        "input_sha256": input_sha256,
        "prompt_version": TRANSLATION_RETRY_PROMPT_VERSION,
        "response_schema_version": TRANSLATION_SLOT_REPAIR_SCHEMA_VERSION,
        "model_tier": TRANSLATION_RETRY_TIER,
        "block_ids": repaired_block_ids,
        "status": "validated",
        "started_at": str((prior_marker or {}).get("started_at") or now),
        "response_received_at": str(
            (prior_marker or {}).get("response_received_at") or now
        ),
        "validated_at": now,
        "validated_translation_sha256": sha256_json(repaired),
        "raw_response": raw_response,
        **(
            {"superseded_text_attempt": prior_marker["superseded_text_attempt"]}
            if isinstance((prior_marker or {}).get("superseded_text_attempt"), dict)
            else {}
        ),
    })


def _translation_checkpoint_requires_v4_upgrade(checkpoint: dict[str, Any]) -> bool:
    repairs = (checkpoint.get("generation_provenance") or {}).get("repairs") or []
    return any(
        isinstance(item, dict)
        and item.get("kind") == "token-placement"
        and (
            item.get("prompt_version") != TRANSLATION_RETRY_PROMPT_VERSION
            or item.get("repair_mode") != "offset-only"
        )
        for item in repairs
    )


def _translation_checkpoint_requires_protected_name_upgrade(
    checkpoint: dict[str, Any],
) -> bool:
    """Rebuild deterministic name repairs made under an older matching policy."""
    repairs = (checkpoint.get("generation_provenance") or {}).get("repairs") or []
    return any(
        isinstance(item, dict)
        and item.get("kind") == "protected-name-normalization"
        and item.get("normalizer_version")
        != TRANSLATION_PROTECTED_NAME_NORMALIZER_VERSION
        for item in repairs
    )


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
    options: BuildOptions,
    bundle: SourceBundle,
    glossary: dict[str, Any],
    protected_names: list[str],
    checkpoint_dir: Path,
    translation: dict[str, Any] | None = None,
) -> Path:
    """Seed an auditable repair-only candidate without invoking the primary model."""
    by_id = {block_id(block): block for block in bundle.document["blocks"]}
    paper_context = _full_paper_context(
        bundle.document, segment, blocks_by_id=by_id, options=options
    )
    input_sha256 = _segment_input_hash(
        segment,
        by_id,
        glossary=glossary,
        extra={
            "names": protected_names,
            "paper_context": paper_context,
            "runtime_access": _generation_runtime_policy(options),
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


def _repair_translation_token_placement(
    segment: dict[str, Any],
    translation: dict[str, Any],
    *,
    blocks_by_id: dict[str, dict[str, Any]],
    protected_names: list[str],
    options: BuildOptions,
    checkpoint_dir: Path,
    artifact_dir: Path,
    input_sha256: str,
    llm: Callable[..., dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the single lifetime-bounded token-placement repair for a segment input."""
    segment_id = str(segment["segment_id"])
    token_errors = _translation_opaque_token_errors(segment, translation, blocks_by_id)
    source_blocks = [blocks_by_id[item.block_id] for item in token_errors]
    if not source_blocks:
        raise RuntimeError(
            f"translation token repair has no structurally failing blocks for {segment_id}"
        )
    repair_draft_path = _translation_token_repair_draft_path(
        checkpoint_dir, segment_id,
    )
    persisted = _matching_translation_token_repair_draft(
        checkpoint_dir, segment_id, input_sha256,
    )
    invalid_persisted_draft = False
    if persisted is not None:
        try:
            repaired = _apply_translation_slot_repairs(
                translation,
                source_blocks,
                persisted["raw_response"],
                protected_names=protected_names,
                offset_only=True,
            )
            _validate_translation(segment, repaired, blocks_by_id, protected_names)
            if sha256_json(repaired) != sha256_json(persisted["translation"]):
                raise RuntimeError("persisted translation differs from replayed offsets")
        except RuntimeError:
            invalid_persisted_draft = True
        else:
            prior_marker = _read_checkpoint_json(
                _translation_token_attempt_path(checkpoint_dir, segment_id)
            )
            _write_validated_translation_token_marker(
                checkpoint_dir,
                segment_id,
                input_sha256,
                repaired=repaired,
                raw_response=persisted["raw_response"],
                repaired_block_ids=list(
                    persisted["repair_provenance"].get("repaired_block_ids") or []
                ),
                prior_marker=prior_marker if isinstance(prior_marker, dict) else None,
            )
            return repaired, dict(persisted["repair_provenance"])
    attempt_path = _translation_token_attempt_path(checkpoint_dir, segment_id)
    raw_attempt = _read_checkpoint_json(attempt_path)
    if attempt_path.is_file() and raw_attempt is None:
        raise RuntimeError(
            f"translation token repair marker is corrupt for {segment_id}; "
            "refusing a new model call"
        )
    attempt = _matching_translation_token_attempt(
        checkpoint_dir, segment_id, input_sha256,
    )
    if invalid_persisted_draft and not (
        isinstance(attempt, dict)
        and attempt.get("status") == "validated"
        and isinstance(attempt.get("raw_response"), dict)
    ):
        raise RuntimeError(
            f"translation token repair draft is invalid for {segment_id} and has no "
            "validated raw response; refusing a new model call"
        )
    raw_repair_draft = _read_checkpoint_json(repair_draft_path)
    if (
        repair_draft_path.is_file()
        and (
            raw_repair_draft is None
            or (
                isinstance(raw_repair_draft, dict)
                and raw_repair_draft.get("prompt_version")
                == TRANSLATION_RETRY_PROMPT_VERSION
                and raw_repair_draft.get("segment_id") == segment_id
                and raw_repair_draft.get("input_sha256") == input_sha256
            )
        )
        and attempt is None
    ):
        raise RuntimeError(
            f"translation token repair draft is corrupt for {segment_id}; refusing a new model call"
        )
    repair_input = translation
    previous_by_id = {
        str(item.get("block_id") or ""): item for item in repair_input["blocks"]
    }
    repair_contexts = [
        _translation_slot_repair_context(
            source_block,
            str(previous_by_id[block_id(source_block)].get("text") or ""),
            protected_names=protected_names,
            primary_text=None,
        )
        for source_block in source_blocks
    ]
    marker_base = {
        "schema_version": "arc.companion.translation-token-attempt.v2",
        "segment_id": segment_id,
        "input_sha256": input_sha256,
        "prompt_version": TRANSLATION_RETRY_PROMPT_VERSION,
        "response_schema_version": TRANSLATION_SLOT_REPAIR_SCHEMA_VERSION,
        "model_tier": TRANSLATION_RETRY_TIER,
        "block_ids": [block_id(block) for block in source_blocks],
    }
    superseded_text_attempt = _matching_superseded_translation_text_attempt(
        checkpoint_dir, segment_id, input_sha256,
    )
    if isinstance(superseded_text_attempt, dict):
        marker_base["superseded_text_attempt"] = {
            "path": str(_legacy_translation_token_attempt_path(checkpoint_dir, segment_id)),
            "sha256": sha256_json(superseded_text_attempt),
            "prompt_version": str(superseded_text_attempt.get("prompt_version") or ""),
            "status": str(superseded_text_attempt.get("status") or ""),
        }
    value: dict[str, Any] | None = None
    if attempt is not None:
        status = str(attempt.get("status") or "")
        if status in {"response_received", "validated"}:
            raw_response = attempt.get("raw_response")
            if not isinstance(raw_response, dict):
                raise RuntimeError(
                    f"translation token repair marker lacks its auditable response for {segment_id}"
                )
            value = raw_response
        elif status == "started":
            raise RuntimeError(
                f"translation token repair attempt already started for {segment_id}; "
                "refusing another model call"
            )
        else:
            raise RuntimeError(
                f"translation token repair marker has invalid status {status!r} for {segment_id}"
            )
    if value is None:
        started_at = (
            str(attempt.get("started_at") or "") if attempt is not None else ""
        ) or datetime.now(timezone.utc).isoformat()
        write_json(attempt_path, {
            **marker_base,
            "status": "started",
            "started_at": started_at,
        })
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
            artifact_dir=artifact_dir / "retry-offset-1",
            call_label=f"companion-translation-{segment_id}-retry-offset-1",
            model_tier=TRANSLATION_RETRY_TIER,
            allow_mcp=False,
            allow_internet=False,
        )
        write_json(attempt_path, {
            **marker_base,
            "status": "response_received",
            "started_at": started_at,
            "response_received_at": datetime.now(timezone.utc).isoformat(),
            "raw_response": value,
        })
    repaired = _apply_translation_slot_repairs(
        repair_input,
        source_blocks,
        value,
        protected_names=protected_names,
        allow_clause_rewrite=False,
        primary_translation=None,
        offset_only=True,
    )
    _validate_translation(segment, repaired, blocks_by_id, protected_names)
    provenance = {
        "kind": "token-placement",
        "attempt": 1,
        "repair_method": "model-offsets-controller-slices",
        "prompt_version": TRANSLATION_RETRY_PROMPT_VERSION,
        "repair_mode": "offset-only",
        "citation_delimiter_normalizer_version": (
            TRANSLATION_CITATION_DELIMITER_NORMALIZER_VERSION
        ),
        "model_tier": TRANSLATION_RETRY_TIER,
        "repaired_block_ids": [block_id(block) for block in source_blocks],
    }
    if isinstance(marker_base.get("superseded_text_attempt"), dict):
        provenance["superseded_text_attempt"] = marker_base["superseded_text_attempt"]
    write_json(repair_draft_path, {
        "schema_version": "arc.companion.translation-token-repair-draft.v1",
        "segment_id": segment_id,
        "input_sha256": input_sha256,
        "prompt_version": TRANSLATION_RETRY_PROMPT_VERSION,
        "response_schema_version": TRANSLATION_SLOT_REPAIR_SCHEMA_VERSION,
        "raw_response": value,
        "translation": repaired,
        "repair_provenance": provenance,
    })
    marker = _matching_translation_token_attempt(
        checkpoint_dir, segment_id, input_sha256,
    ) or marker_base
    _write_validated_translation_token_marker(
        checkpoint_dir,
        segment_id,
        input_sha256,
        repaired=repaired,
        raw_response=value,
        repaired_block_ids=[block_id(block) for block in source_blocks],
        prior_marker=marker,
    )
    return repaired, provenance


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
    v4_upgrade_ids: set[str] = set()
    protected_name_upgrade_ids: set[str] = set()
    for segment in segments:
        path = translation_dir / f"{_segment_checkpoint_name(segment['segment_id'])}.json"
        paper_context = _full_paper_context(
            bundle.document, segment, blocks_by_id=by_id, options=options
        )
        expected_hash = _segment_input_hash(
            segment,
            by_id,
            glossary=glossary,
            extra={
                "names": protected_names,
                "paper_context": paper_context,
                "runtime_access": _generation_runtime_policy(options),
            },
        )
        input_hashes[str(segment["segment_id"])] = expected_hash
        if path.is_file() and not options.force:
            checkpoint = _read_checkpoint_json(path)
            if (
                isinstance(checkpoint, dict)
                and checkpoint.get("segment_id") == segment["segment_id"]
                and checkpoint.get("input_sha256") == expected_hash
                and isinstance(checkpoint.get("translation"), dict)
            ):
                segment_id = str(segment["segment_id"])
                if _translation_checkpoint_requires_v4_upgrade(checkpoint):
                    v4_upgrade_ids.add(segment_id)
                    pending.append(segment)
                    continue
                if _translation_checkpoint_requires_protected_name_upgrade(checkpoint):
                    protected_name_upgrade_ids.add(segment_id)
                    pending.append(segment)
                    continue
                checkpoint = _repair_translation_checkpoint_citation_delimiters(
                    path, segment, by_id, protected_names=protected_names,
                )
                output[segment["segment_id"]] = checkpoint["translation"]
                continue
        pending.append(segment)

    def generate(
        segment: dict[str, Any],
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        segment_id = str(segment["segment_id"])
        selected = [by_id[value] for value in segment["block_ids"]]
        translatable = [_translation_input_block(block) for block in selected if _is_translatable(block)]
        paper_context = _full_paper_context(
            bundle.document, segment, blocks_by_id=by_id, options=options
        )
        artifact_dir = (
            checkpoint_dir / "llm" / "translations" / _segment_checkpoint_name(segment_id)
        )
        draft_path = _translation_draft_path(checkpoint_dir, segment_id)
        draft = _read_checkpoint_json(draft_path)
        attempt = _matching_translation_coverage_attempt(
            checkpoint_dir, segment_id, input_hashes[segment_id]
        )
        candidate_provenance: dict[str, Any] | None = None
        translation: dict[str, Any] | None = None
        restored_name_ids: list[str] = []
        if (
            isinstance(draft, dict)
            and draft.get("schema_version") == "arc.companion.translation-primary-draft.v1"
            and draft.get("segment_id") == segment_id
            and draft.get("input_sha256") == input_hashes[segment_id]
            and isinstance(draft.get("translation"), dict)
        ):
            draft_translation = draft["translation"]
            try:
                draft_translation, restored_name_ids = (
                    _restore_translation_protected_names(
                        segment, draft_translation, by_id, protected_names
                    )
                )
                _validate_translation(segment, draft_translation, by_id, protected_names)
            except TranslationCoverageError:
                translation = draft_translation
            except TranslationOpaqueTokenError:
                translation = draft_translation
            except RuntimeError:
                # Only coverage-invalid drafts have a bounded specialized resume path.
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
        if segment_id in v4_upgrade_ids and (
            translation is None
            or str((candidate_provenance or {}).get("origin") or "") != "primary-model"
        ):
            raise RuntimeError(
                f"v4 translation upgrade for {segment_id} requires its stored primary draft; "
                "refusing to rerun the low translation model"
            )
        if segment_id in protected_name_upgrade_ids and (
            translation is None
            or str((candidate_provenance or {}).get("origin") or "") != "primary-model"
        ):
            raise RuntimeError(
                f"protected-name translation upgrade for {segment_id} requires its stored "
                "primary draft; refusing to rerun the low translation model"
            )
        if translatable:
            if translation is None:
                guarded_attempt = _guard_translation_token_attempt_before_primary(
                    checkpoint_dir, segment_id, input_hashes[segment_id]
                )
                if guarded_attempt is not None:
                    raise RuntimeError(
                        f"translation token placement repair attempt already consumed for {segment_id}"
                    )
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
        try:
            translation, newly_restored_ids = _restore_translation_protected_names(
                segment, translation, by_id, protected_names
            )
        except TranslationCoverageError:
            newly_restored_ids = []
        restored_name_ids = list(dict.fromkeys([
            *restored_name_ids, *newly_restored_ids,
        ]))
        repair_provenance: list[dict[str, Any]] = []
        if restored_name_ids:
            repair_provenance.append({
                "kind": "protected-name-normalization",
                "attempt": 0,
                "normalizer_version": TRANSLATION_PROTECTED_NAME_NORMALIZER_VERSION,
                "repaired_block_ids": restored_name_ids,
            })
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
                    force_offline=True,
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
                try:
                    _validate_translation(segment, translation, by_id, protected_names)
                except TranslationOpaqueTokenError:
                    translation, provenance = _repair_translation_token_placement(
                        segment,
                        translation,
                        blocks_by_id=by_id,
                        protected_names=protected_names,
                        options=options,
                        checkpoint_dir=checkpoint_dir,
                        artifact_dir=artifact_dir,
                        input_sha256=input_hashes[segment_id],
                        llm=llm,
                    )
                    repair_provenance.append(provenance)
            if missing_blocks:
                # Model-backed coverage repair is self-contained; never chain another model repair.
                translation, coverage_name_ids = _restore_translation_protected_names(
                    segment, translation, by_id, protected_names
                )
                if coverage_name_ids:
                    repair_provenance.append({
                        "kind": "protected-name-normalization",
                        "attempt": 0,
                        "normalizer_version": (
                            TRANSLATION_PROTECTED_NAME_NORMALIZER_VERSION
                        ),
                        "repaired_block_ids": coverage_name_ids,
                    })
                    _validate_translation(segment, translation, by_id, protected_names)
        except TranslationOpaqueTokenError:
            translation, provenance = _repair_translation_token_placement(
                segment,
                translation,
                blocks_by_id=by_id,
                protected_names=protected_names,
                options=options,
                checkpoint_dir=checkpoint_dir,
                artifact_dir=artifact_dir,
                input_sha256=input_hashes[segment_id],
                llm=llm,
            )
            repair_provenance.append(provenance)
        translation, normalized_citation_ids = (
            _normalize_translation_citation_delimiters_for_segment(translation, by_id)
        )
        if normalized_citation_ids:
            repair_provenance.append(
                _citation_delimiter_normalization_provenance(normalized_citation_ids)
            )
        _validate_translation(segment, translation, by_id, protected_names)
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
                                    bundle.document, segment, blocks_by_id=by_id, options=options
                                ),
                                "runtime_access": _generation_runtime_policy(options),
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
    translations = {
        str(segment_id): clean_reader_translation(translation)
        for segment_id, translation in translations.items()
    }
    by_id = {block_id(block): block for block in document.get("blocks") or []}
    reader_evidence = _reader_evidence_by_segment(
        segments, document=document, evidence=evidence, annotations=annotations
    )
    annotations = {
        str(segment_id): clean_reader_annotation(
            annotation,
            evidence_records=reader_evidence.get(str(segment_id), []),
            language=options.annotation_language,
        )
        for segment_id, annotation in annotations.items()
    }
    payload = {
        "segments": [
            {
                "segment": segment,
                "source_blocks": [_annotation_input_block(by_id[value], document) for value in segment["block_ids"]],
                "translation": translations[segment["segment_id"]],
                "annotation": annotations[segment["segment_id"]],
                "context_evidence": _review_context_evidence(
                    segment, blocks_by_id=by_id, evidence=evidence
                ),
            }
            for segment in segments
        ]
    }
    findings: list[dict[str, Any]] = []
    hierarchical = len(json.dumps(payload, ensure_ascii=False)) > options.review_context_chars
    if hierarchical:
        chunks = _review_chunks(payload["segments"], options.review_context_chars // 2)
        recovered_reviews = (
            {} if options.force else _load_recovered_section_reviews(checkpoint_dir, chunks)
        )

        def inspect(index: int, chunk: list[dict[str, Any]]) -> dict[str, Any]:
            prompt = section_review_prompt(
                {"segments": chunk}, language=options.annotation_language
            )
            input_sha256 = sha256_json({
                "prompt": prompt,
                "schema": SECTION_REVIEW_SCHEMA,
                "model_tier": REVIEW_TIER,
            })
            path = checkpoint_dir / "section-reviews" / f"{index:04d}.json"
            if path.is_file() and not options.force:
                checkpoint = read_json(path)
                if (
                    isinstance(checkpoint, dict)
                    and checkpoint.get("schema_version") == SECTION_REVIEW_CHECKPOINT_VERSION
                    and checkpoint.get("input_sha256") == input_sha256
                    and isinstance(checkpoint.get("review"), dict)
                ):
                    return checkpoint["review"]
            value = recovered_reviews.get(index)
            if value is None:
                value = _llm_call(
                    llm,
                    prompt,
                    SECTION_REVIEW_SCHEMA,
                    options=options,
                    artifact_dir=checkpoint_dir / "llm" / "section-review" / str(index),
                    call_label=f"companion-section-review-{index}",
                    model_tier=REVIEW_TIER,
                )
            write_json(path, {
                "schema_version": SECTION_REVIEW_CHECKPOINT_VERSION,
                "section_index": index,
                "input_sha256": input_sha256,
                "reviewed_segment_ids": sorted(
                    str(item["segment"]["segment_id"]) for item in chunk
                ),
                "review": value,
            })
            return value

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
                "reviewed_segment_ids": sorted(reviewed_ids),
                "findings": chunk_findings,
                "patch_proposals": _section_review_patch_proposals(
                    chunks[index], reviewed_segments, chunk_findings
                ),
            })
        all_segment_ids = {str(item["segment_id"]) for item in segments}
        if review_coverage != all_segment_ids:
            missing = sorted(all_segment_ids - review_coverage)
            raise RuntimeError(f"hierarchical review did not cover every segment: {missing}")

    final_payload = {**payload, "glossary": glossary, "protected_names": protected_names}
    if hierarchical:
        final_context_budget = max(40_000, options.review_context_chars)
        source_anchors = _review_source_anchors(
            segments,
            blocks_by_id=by_id,
            document=document,
            total_chars=min(24_000, max(8_000, final_context_budget // 6)),
        )
        context_evidence_anchors = _review_context_evidence_anchors(
            payload["segments"],
            total_chars=min(24_000, max(8_000, final_context_budget // 6)),
        )
        bounded_glossary = _bounded_glossary_projection(
            glossary, total_chars=min(20_000, max(8_000, final_context_budget // 7))
        )
        projection_budget = max(
            10_000,
            final_context_budget
            - len(json.dumps(source_anchors, ensure_ascii=False))
            - len(json.dumps(context_evidence_anchors, ensure_ascii=False))
            - len(json.dumps(bounded_glossary, ensure_ascii=False))
            - 10_000,
        )
        final_payload = {
            "section_reviews": _bounded_section_review_projection(
                section_reviews, total_chars=projection_budget
            ),
            "reviewed_segment_ids": [str(item["segment_id"]) for item in segments],
            "source_anchors": source_anchors,
            "context_evidence_anchors": context_evidence_anchors,
            "glossary": bounded_glossary,
            "protected_names": protected_names,
            "instruction": (
                "Consolidate section findings into non-conflicting translation and/or annotation patches. "
                "Use the bounded source anchor for every segment as a direct source-awareness check; "
                "section reviews contain bounded findings and proposed corrections from the complete "
                "source-aware local review."
            ),
        }
    review = _llm_call(
        llm,
        review_prompt(
            final_payload,
            language=options.annotation_language,
            findings=None if hierarchical else findings,
        ),
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
    citation_normalized: set[str] = set()
    for patch in review.get("patches") or []:
        segment_id = str(patch.get("segment_id") or "")
        if segment_id not in valid_ids or segment_id in patched:
            raise RuntimeError(f"review returned invalid or duplicate annotation patch: {segment_id}")
        original_annotation = dict(reviewed[segment_id])
        translation_changed = patch.get("translation_blocks") is not None
        if translation_changed:
            replacement = clean_reader_translation(
                {"blocks": list(patch.get("translation_blocks") or [])}
            )
            segment = next(item for item in segments if item["segment_id"] == segment_id)
            replacement, changed_ids = _normalize_translation_citation_delimiters_for_segment(
                replacement, by_id
            )
            if changed_ids:
                citation_normalized.add(segment_id)
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
            elif field in {"prior_work", "later_work"}:
                reviewed[segment_id][field] = _normalize_related_work(patch[field])
            else:
                text = str(patch[field])
                reviewed[segment_id][field] = text
        if any(field in changed_annotation_fields for field in ("prior_work", "later_work")):
            _sync_claim_evidence_ids(reviewed[segment_id])
        segment = next(item for item in segments if item["segment_id"] == segment_id)
        _validate_annotation_evidence(
            clean_reader_annotation(
                reviewed[segment_id],
                evidence_records=reader_evidence.get(segment_id, []),
                language=options.annotation_language,
            ),
            _evidence_for_segment(segment, by_id, evidence)["papers"]
        )
        reviewed[segment_id] = clean_reader_annotation(
            reviewed[segment_id],
            evidence_records=reader_evidence.get(segment_id, []),
            language=options.annotation_language,
        )
        _assert_review_did_not_add_related_work(original_annotation, reviewed[segment_id])
        patched.add(segment_id)
    return reviewed_translations, reviewed, {
        "hierarchical": hierarchical,
        "section_findings": findings,
        "reviewed_segment_ids": [str(item["segment_id"]) for item in segments],
        "issues": [str(item) for item in review.get("issues") or []],
        "patched_segment_ids": sorted(patched),
        "citation_delimiter_normalized_segment_ids": sorted(citation_normalized),
    }


def _assert_review_did_not_add_related_work(
    before: dict[str, Any], after: dict[str, Any],
) -> None:
    """The non-research review pass may edit/drop claims but never create bindings."""
    for field in ("prior_work", "later_work"):
        old_value = before.get(field)
        new_value = after.get(field)
        if isinstance(new_value, list):
            old_bindings = {
                _claim_binding_key(claim)
                for claim in old_value or [] if isinstance(claim, dict)
            } if isinstance(old_value, list) else set()
            new_bindings = {
                _claim_binding_key(claim)
                for claim in new_value if isinstance(claim, dict)
            }
            if not new_bindings.issubset(old_bindings):
                raise RuntimeError("review added a related-work claim without prior claim evidence")
        elif str(new_value or "").strip() and not str(old_value or "").strip():
            raise RuntimeError("review added a related-work claim without prior claim evidence")
        elif isinstance(old_value, list) and str(new_value or "").strip():
            raise RuntimeError("review replaced claim bindings with unbound related-work text")


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
    force_offline: bool = False,
) -> dict[str, Any]:
    force_offline = force_offline or (not allow_mcp and not allow_internet)
    runtime_env = _llm_runtime_env(
        allow_mcp=not force_offline and allow_mcp and options.allow_mcp,
        allow_internet=not force_offline and allow_internet and options.allow_internet,
        force_disable_mcp=force_offline or not options.allow_mcp,
        force_disable_internet=force_offline or not options.allow_internet,
    )
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


def _generation_runtime_policy(options: BuildOptions | None = None) -> dict[str, bool | str]:
    allow_mcp = True if options is None else options.allow_mcp
    allow_internet = True if options is None else options.allow_internet
    policy: dict[str, bool | str] = {
        "allow_mcp": allow_mcp,
        "allow_internet": allow_internet,
    }
    if allow_mcp:
        policy["mcp_mode"] = "arc-only"
    return policy


def _llm_runtime_env(
    *,
    allow_mcp: bool,
    allow_internet: bool,
    force_disable_mcp: bool = False,
    force_disable_internet: bool = False,
) -> dict[str, str] | None:
    """Map portable access intent onto both supported host runtimes."""
    if not allow_mcp and not allow_internet and not force_disable_mcp and not force_disable_internet:
        return None
    env = dict(os.environ)
    if allow_internet or force_disable_internet:
        value = "true" if allow_internet else "false"
        env["ARC_CODEX_ALLOW_INTERNET"] = value
        env["ARC_CLAUDE_ALLOW_INTERNET"] = value
    if allow_mcp or force_disable_mcp:
        value = "true" if allow_mcp else "false"
        env["ARC_CODEX_ENABLE_MCP"] = value
        env["ARC_CLAUDE_ALLOW_MCP"] = value
    if allow_mcp:
        env["ARC_CODEX_MCP_MODE"] = "arc-only"
        env["ARC_CLAUDE_MCP_MODE"] = "arc-only"
    elif force_disable_mcp:
        for key in _MCP_CONFIG_ENV_KEYS:
            env.pop(key, None)
        # These controls map to provider-native isolation flags.  Merely setting
        # allow-MCP=false is insufficient when a parent profile/config injects
        # servers or tools before the provider evaluates that boolean.
        env["ARC_CODEX_IGNORE_USER_CONFIG"] = "true"
        env["ARC_CLAUDE_BARE"] = "true"
        claude_web_tools = "WebSearch,WebFetch" if allow_internet else ""
        env["ARC_CLAUDE_TOOLS"] = claude_web_tools
        env["ARC_CLAUDE_ALLOWED_TOOLS"] = claude_web_tools
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
            "rich_parser_version": bundle.document.get("parser_version"),
            "generation_projection_hash": sha256_json(_generation_document(bundle.document)),
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
                "translation": _generation_runtime_policy(options),
                "annotation": _generation_runtime_policy(options),
                "review": {"allow_mcp": False, "allow_internet": False},
            },
            "full_paper_context_version": FULL_PAPER_CONTEXT_VERSION,
            "context_selection": {
                "version": CONTEXT_SELECTION_VERSION,
                "chars_per_source": CONTEXT_SEGMENT_CHARS_PER_SOURCE,
                "chars_total": CONTEXT_SEGMENT_CHARS_TOTAL,
            },
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
    if state.get("final_render_version") != FINAL_RENDER_VERSION:
        return False
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


def _evidence(
    bundle: SourceBundle, *, context_evidence: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    def compact(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        fields = (
            "paper_id", "arxiv_id", "doi", "inspire_id", "title", "authors",
            "year", "citation_count", "abstract",
        )
        return [{key: item[key] for key in fields if key in item} for item in items]

    bibliography = [
        {
            key: item[key]
            for key in ("id", "label", "text", "doi", "arxiv_id", "links")
            if key in item
        }
        for item in bundle.document.get("bibliography") or []
        if isinstance(item, dict)
    ]

    return {
        "schema_version": "arc.companion.evidence.v3",
        "references": compact(bundle.references),
        "citers": compact(bundle.citers),
        "bibliography": bibliography,
        "related_papers": [*bundle.related_evidence, *(context_evidence or [])],
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
        "blocks": [preserved.get(value) or additions[value] for value in expected_ids],
    }


def _translation_slot_repair_context(
    source_block: dict[str, Any],
    previous_text: str,
    *,
    protected_names: list[str],
    primary_text: str | None = None,
) -> dict[str, Any]:
    """Build an inert placement-only context without asking for retranslation."""
    expected_tokens = _opaque_inline_tokens(source_block)
    residue = _translation_natural_residue(previous_text)
    affected_ordinals = _translation_repair_affected_ordinals(
        source_block, primary_text, previous_text,
    ) if primary_text is not None else set(range(1, len(expected_tokens) + 1))
    prior_tokens, prior_slots = _translation_token_delimited_slots(previous_text)
    if primary_text is not None and prior_tokens != expected_tokens:
        raise RuntimeError("legacy translation tokens do not match source occurrences")
    slot_regions = [
        _translation_slot_boundary_record(
            slot,
            slot_ordinal=index,
            token_count=len(expected_tokens),
            affected_ordinals=affected_ordinals,
        )
        for index, slot in enumerate(prior_slots)
    ] if prior_tokens == expected_tokens else []
    inline_runs = [
        run for run in source_block.get("inline_runs") or [] if isinstance(run, dict)
    ]
    source_runs: list[dict[str, Any]] = []
    expected_token_semantics: list[dict[str, Any]] = []
    opaque_index = 0
    for index, run in enumerate(inline_runs, start=1):
        kind = str(run.get("kind") or "")
        record: dict[str, Any] = {"order": index, "kind": kind}
        if kind == "text":
            record["source_text"] = str(run.get("content") or "")
        else:
            token = expected_tokens[opaque_index]
            opaque_index += 1
            record["source_content"] = str(run.get("tex") or run.get("content") or "")
            record["expected_token"] = token
            left_text = next((
                str(value.get("content") or "")
                for value in reversed(inline_runs[:index - 1])
                if str(value.get("kind") or "") == "text"
            ), "")
            right_text = next((
                str(value.get("content") or "")
                for value in inline_runs[index:]
                if str(value.get("kind") or "") == "text"
            ), "")
            expected_token_semantics.append({
                "token_ordinal": opaque_index,
                "token": token,
                "kind": kind,
                "source_content": record["source_content"],
                "left_source_text": left_text,
                "right_source_text": right_text,
            })
            if kind == "link" and run.get("href"):
                record["href"] = str(run["href"])
        source_runs.append(record)
    return {
        "block_id": block_id(source_block),
        "previous_invalid_text": previous_text,
        "prior_natural_language_residue": residue,
        "residue_length": len(residue),
        "indexed_residue": [
            {"offset": index, "character": character}
            for index, character in enumerate(residue)
        ],
        "token_delimited_slot_regions": slot_regions,
        "immutable_fragments": [
            fragment
            for item in slot_regions
            for fragment in (item["immutable_prefix"], item["immutable_suffix"])
            if fragment
        ],
        "affected_token_ordinals": sorted(affected_ordinals),
        "expected_tokens": expected_token_semantics,
        "expected_slot_count": len(expected_tokens) + 1,
        "slot_ids": _translation_repair_slot_ids(source_block),
        "missing_protected_names": _minimal_missing_name_insertions(
            _missing_protected_names([source_block], residue, protected_names)
        ),
        "source_run_sequence": source_runs,
}


def _translation_token_delimited_slots(text: str) -> tuple[list[str], list[str]]:
    """Return opaque occurrences and the exact natural slots around them."""
    matches = list(_OPAQUE_INLINE_CANDIDATE_PATTERN.finditer(text))
    tokens = [match.group(0) for match in matches]
    slots: list[str] = []
    cursor = 0
    for match in matches:
        slots.append(text[cursor:match.start()])
        cursor = match.end()
    slots.append(text[cursor:])
    return tokens, slots


def _translation_slot_boundary_record(
    text: str,
    *,
    slot_ordinal: int,
    token_count: int,
    affected_ordinals: set[int],
) -> dict[str, Any]:
    """Lock exact slot boundaries outside clauses touching affected occurrences."""
    left_affected = slot_ordinal > 0 and slot_ordinal in affected_ordinals
    right_affected = (
        slot_ordinal < token_count and slot_ordinal + 1 in affected_ordinals
    )
    immutable_prefix = text
    immutable_suffix = ""
    mutable_mode = "none"
    if left_affected and right_affected:
        immutable_prefix = ""
        mutable_mode = "full"
    elif left_affected:
        match = re.search(r"[。！？；;.!?：:，,]", text)
        boundary = match.end() if match is not None else len(text)
        immutable_prefix = ""
        immutable_suffix = text[boundary:]
        mutable_mode = "prefix"
    elif right_affected:
        matches = list(re.finditer(r"[。！？；;.!?：:，,]", text))
        boundary = matches[-1].end() if matches else 0
        immutable_prefix = text[:boundary]
        mutable_mode = "suffix"
    return {
        "slot_id": f"slot-{slot_ordinal:04d}",
        "slot_ordinal": slot_ordinal,
        "left_token_ordinal": slot_ordinal if slot_ordinal > 0 else None,
        "right_token_ordinal": slot_ordinal + 1 if slot_ordinal < token_count else None,
        "mutable_mode": mutable_mode,
        "prior_text": text,
        "immutable_prefix": immutable_prefix,
        "immutable_suffix": immutable_suffix,
    }


def _translation_repair_affected_ordinals(
    source_block: dict[str, Any], primary_text: str, v3_text: str,
) -> set[int]:
    """Identify affected source token occurrences with duplicate-safe LCS alignment."""
    expected = _opaque_inline_tokens(source_block)
    primary_tokens, _ = _translation_token_delimited_slots(primary_text)
    v3_tokens, _ = _translation_token_delimited_slots(v3_text)
    if v3_tokens != expected:
        raise RuntimeError("legacy v3 candidate has invalid opaque token occurrences")
    primary_offsets = _translation_token_residue_offsets_by_occurrence(primary_text)
    v3_offsets = _translation_token_residue_offsets_by_occurrence(v3_text)
    scores = [
        [(0, 0)] * (len(primary_tokens) + 1) for _ in range(len(expected) + 1)
    ]
    choices = [[""] * (len(primary_tokens) + 1) for _ in range(len(expected) + 1)]
    for expected_index in range(1, len(expected) + 1):
        for primary_index in range(1, len(primary_tokens) + 1):
            candidates = [
                (scores[expected_index - 1][primary_index], "up"),
                (scores[expected_index][primary_index - 1], "left"),
            ]
            if expected[expected_index - 1] == primary_tokens[primary_index - 1]:
                prior_count, prior_cost = scores[expected_index - 1][primary_index - 1]
                candidates.append((
                    (
                        prior_count + 1,
                        prior_cost + abs(
                            v3_offsets[expected_index - 1]
                            - primary_offsets[primary_index - 1]
                        ),
                    ),
                    "match",
                ))
            score, choice = max(
                candidates,
                key=lambda item: (item[0][0], -item[0][1], item[1] == "match"),
            )
            scores[expected_index][primary_index] = score
            choices[expected_index][primary_index] = choice
    stable_pairs: dict[int, int] = {}
    expected_index, primary_index = len(expected), len(primary_tokens)
    while expected_index and primary_index:
        choice = choices[expected_index][primary_index]
        if choice == "match":
            stable_pairs[expected_index] = primary_index
            expected_index -= 1
            primary_index -= 1
        elif choice == "up":
            expected_index -= 1
        else:
            primary_index -= 1
    return {
        ordinal for ordinal in range(1, len(expected) + 1)
        if ordinal not in stable_pairs
        or primary_offsets[stable_pairs[ordinal] - 1] != v3_offsets[ordinal - 1]
    }


def _translation_token_residue_offsets_by_occurrence(text: str) -> list[int]:
    offsets: list[int] = []
    cursor = 0
    residue_offset = 0
    for match in _OPAQUE_INLINE_CANDIDATE_PATTERN.finditer(text):
        residue_offset += len(text[cursor:match.start()])
        offsets.append(residue_offset)
        cursor = match.end()
    return offsets


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
    allow_clause_rewrite: bool = False,
    primary_translation: dict[str, Any] | None = None,
    offset_only: bool = False,
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
    primary_by_id = {
        str(item.get("block_id") or ""): item
        for item in (primary_translation or {}).get("blocks") or []
        if isinstance(item, dict)
    }
    for source_block, block_repair in zip(source_blocks, raw_repairs):
        result = _apply_translation_slot_repair_block(
            result, source_block, block_repair, protected_names=protected_names,
            allow_clause_rewrite=allow_clause_rewrite,
            offset_only=offset_only,
            primary_text=(
                str(primary_by_id.get(block_id(source_block), {}).get("text") or "")
                if primary_translation is not None else None
            ),
        )
    return result


def _apply_translation_slot_repair_block(
    previous_translation: dict[str, Any],
    source_block: dict[str, Any],
    repair: dict[str, Any],
    *,
    protected_names: list[str],
    allow_clause_rewrite: bool = False,
    primary_text: str | None = None,
    offset_only: bool = False,
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
    if all("start_offset" in item and "end_offset" in item for item in raw_slots):
        spans = [
            (item.get("start_offset"), item.get("end_offset")) for item in raw_slots
        ]
        if any(
            type(start) is not int or type(end) is not int
            for start, end in spans
        ):
            raise RuntimeError("translation slot repair returned non-integer residue offsets")
        cursor = 0
        slots: list[str] = []
        for start, end in spans:
            if start != cursor or end < start or end > len(residue):
                raise RuntimeError("translation slot repair offsets do not exactly partition prior residue")
            slots.append(residue[start:end])
            cursor = end
        if cursor != len(residue):
            raise RuntimeError("translation slot repair offsets do not cover prior residue")
    else:
        if offset_only:
            raise RuntimeError("translation offset repair returned prose instead of residue offsets")
        # Compatibility for direct callers with v1 payloads. New model calls
        # are schema-constrained to offsets and cannot enter this branch.
        slots = [str(item.get("text") or "") for item in raw_slots]
        if any("[[ARC_INLINE:" in text or _OPAQUE_INLINE_PATTERN.search(text) for text in slots):
            raise RuntimeError("translation slot repair supplied controller-owned opaque content")
        missing_names = _minimal_missing_name_insertions(
            _missing_protected_names([source_block], residue, protected_names)
        )
        joined = "".join(slots)
        if missing_names:
            if not _is_exact_name_insertion_delta(joined, residue, missing_names):
                raise RuntimeError(
                    "translation slot repair changed prior natural language beyond name insertion"
                )
        elif not allow_clause_rewrite and joined != residue:
            raise RuntimeError("translation slot repair retranslated or changed prior natural language")

    expected_tokens = _opaque_inline_tokens(source_block)
    assembled = "".join(
        part
        for index, slot in enumerate(slots)
        for part in ([slot, expected_tokens[index]] if index < len(expected_tokens) else [slot])
    )
    if allow_clause_rewrite:
        affected_ordinals = _translation_repair_affected_ordinals(
            source_block, primary_text, previous_text,
        ) if primary_text is not None else set(range(1, len(expected_tokens) + 1))
        _validate_translation_repair_slot_boundaries(
            previous_text,
            slots,
            expected_tokens=expected_tokens,
            affected_ordinals=affected_ordinals,
        )
    if all("start_offset" in item for item in raw_slots) and (
        _translation_natural_residue(assembled) != residue
    ):
        raise RuntimeError("translation slot repair changed prior natural language")
    repaired_blocks = list(previous_blocks)
    repaired_blocks[failed_index] = {**previous_blocks[failed_index], "text": assembled}
    return {**previous_translation, "blocks": repaired_blocks}


def _validate_translation_repair_slot_boundaries(
    previous_text: str,
    repaired_slots: list[str],
    *,
    expected_tokens: list[str],
    affected_ordinals: set[int],
) -> None:
    """Enforce byte-exact immutable boundaries per token-occurrence-anchored slot."""
    prior_tokens, prior_slots = _translation_token_delimited_slots(previous_text)
    if prior_tokens != expected_tokens or len(repaired_slots) != len(prior_slots):
        raise RuntimeError("translation structural repair changed token occurrence anchors")
    immutable_fragments: set[str] = set()
    for index, (prior_slot, repaired_slot) in enumerate(zip(prior_slots, repaired_slots)):
        boundary = _translation_slot_boundary_record(
            prior_slot,
            slot_ordinal=index,
            token_count=len(expected_tokens),
            affected_ordinals=affected_ordinals,
        )
        prefix = str(boundary["immutable_prefix"])
        suffix = str(boundary["immutable_suffix"])
        immutable_fragments.update(value for value in (prefix, suffix) if value)
        if prefix and not repaired_slot.startswith(prefix):
            raise RuntimeError(
                "translation structural repair changed text outside mutable clauses or token slots"
            )
        if suffix and not repaired_slot.endswith(suffix):
            raise RuntimeError(
                "translation structural repair changed text outside mutable clauses or token slots"
            )
        if boundary["mutable_mode"] == "none" and repaired_slot != prior_slot:
            raise RuntimeError(
                "translation structural repair changed text outside mutable clauses or token slots"
            )
    prior_natural = "".join(prior_slots)
    repaired_natural = "".join(repaired_slots)
    for fragment in immutable_fragments:
        if not fragment.strip(" \t\r\n。！？；;.!?：:，,"):
            continue
        if repaired_natural.count(fragment) != prior_natural.count(fragment):
            raise RuntimeError(
                "translation structural repair moved or copied immutable boundary text"
            )


def _normalize_translation_citation_delimiters(
    source_block: dict[str, Any], translated_text: str,
) -> str:
    """Canonicalize source-owned ASCII citation wrappers around immutable tokens."""
    source_text = str(_translation_input_block(source_block).get("text") or "")
    opaque_runs = [
        run
        for run in source_block.get("inline_runs") or []
        if isinstance(run, dict) and str(run.get("kind") or "") != "text"
    ]
    expected_tokens = _opaque_inline_tokens(source_block)
    source_matches = list(_OPAQUE_INLINE_PATTERN.finditer(source_text))
    if [match.group(0) for match in source_matches] != expected_tokens:
        raise RuntimeError("citation delimiter normalization cannot align source occurrences")
    normalized = translated_text
    for occurrence_index, (run, token) in enumerate(zip(opaque_runs, expected_tokens)):
        if str(run.get("kind") or "").casefold() != "citation":
            continue
        source_index = source_matches[occurrence_index].start()
        source_end = source_matches[occurrence_index].end()
        source_wraps_token = (
            source_index > 0
            and source_end < len(source_text)
            and source_text[source_index - 1] == "["
            and source_text[source_end] == "]"
        )
        if not source_wraps_token:
            continue
        opaque_matches = list(_OPAQUE_INLINE_PATTERN.finditer(normalized))
        if [match.group(0) for match in opaque_matches] != expected_tokens:
            raise RuntimeError(
                "citation delimiter normalization requires exact token occurrences"
            )
        target = opaque_matches[occurrence_index]
        index, end = target.start(), target.end()
        if (
            index > 0
            and end < len(normalized)
            and normalized[index - 1] == "["
            and normalized[end] == "]"
        ):
            continue
        prefix_pair = index >= 2 and normalized[index - 2:index] == "[]"
        suffix_pair = normalized[end:end + 2] == "[]"
        prefix_single = index > 0 and normalized[index - 1] == "["
        suffix_single = end < len(normalized) and normalized[end] == "]"
        adjacent_previous_token = (
            occurrence_index > 0
            and opaque_matches[occurrence_index - 1].end() == index
        )
        adjacent_next_token = (
            occurrence_index + 1 < len(opaque_matches)
            and opaque_matches[occurrence_index + 1].start() == end
        )
        bad_prefix = (
            index > 0 and normalized[index - 1] == "]" and not prefix_pair
            and not adjacent_previous_token
        )
        bad_suffix = (
            end < len(normalized) and normalized[end] == "[" and not suffix_pair
            and not adjacent_next_token
        )
        if bad_prefix or bad_suffix or (prefix_pair and suffix_pair):
            raise RuntimeError(
                "citation delimiter normalization found ambiguous adjacent brackets"
            )
        prefix_start = index - (2 if prefix_pair else 1 if prefix_single else 0)
        suffix_end = end + (2 if suffix_pair else 1 if suffix_single else 0)
        normalized = normalized[:prefix_start] + "[" + token + "]" + normalized[suffix_end:]
    return normalized


def _normalize_translation_citation_delimiters_for_segment(
    translation: dict[str, Any],
    blocks_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Normalize token-valid blocks in any candidate without masking token repair."""
    raw_blocks = translation.get("blocks") if isinstance(translation, dict) else None
    if not isinstance(raw_blocks, list):
        return translation, {}
    changed: dict[str, str] = {}
    normalized_blocks: list[Any] = []
    for item in raw_blocks:
        if not isinstance(item, dict):
            normalized_blocks.append(item)
            continue
        item_id = str(item.get("block_id") or "")
        source_block = blocks_by_id.get(item_id)
        previous_text = str(item.get("text") or "")
        if source_block is None or _OPAQUE_INLINE_PATTERN.findall(previous_text) != (
            _opaque_inline_tokens(source_block)
        ):
            normalized_blocks.append(item)
            continue
        method = _citation_delimiter_normalization_method(source_block, previous_text)
        normalized_text = _normalize_translation_citation_delimiters(
            source_block, previous_text
        )
        if normalized_text == previous_text:
            normalized_blocks.append(item)
            continue
        changed[item_id] = method
        normalized_blocks.append({**item, "text": normalized_text})
    if not changed:
        return translation, {}
    return {**translation, "blocks": normalized_blocks}, changed


def _citation_delimiter_normalization_method(
    source_block: dict[str, Any], translated_text: str,
) -> str:
    """Classify a source-owned wrapper repair for audit provenance."""
    source_text = str(_translation_input_block(source_block).get("text") or "")
    method = "relocated"
    opaque_runs = [
        run for run in source_block.get("inline_runs") or []
        if isinstance(run, dict) and str(run.get("kind") or "") != "text"
    ]
    for run, token in zip(opaque_runs, _opaque_inline_tokens(source_block)):
        if str(run.get("kind") or "").casefold() != "citation":
            continue
        source_index = source_text.find(token)
        source_end = source_index + len(token)
        if not (
            source_index > 0 and source_end < len(source_text)
            and source_text[source_index - 1] == "[" and source_text[source_end] == "]"
        ):
            continue
        index = translated_text.find(token)
        end = index + len(token)
        if index > 0 and end < len(translated_text) and (
            translated_text[index - 1] == "[" and translated_text[end] == "]"
        ):
            continue
        if not (
            (index >= 2 and translated_text[index - 2:index] == "[]")
            or translated_text[end:end + 2] == "[]"
        ):
            method = "synthesized"
    return method


def _citation_delimiter_normalization_provenance(
    changed: dict[str, str],
) -> dict[str, Any]:
    return {
        "kind": "citation-delimiter-normalization",
        "attempt": 0,
        "normalizer_version": TRANSLATION_CITATION_DELIMITER_NORMALIZER_VERSION,
        "repaired_block_ids": list(changed),
        "methods_by_block_id": changed,
    }


def _repair_reviewed_translation_checkpoint(
    checkpoint_path: Path,
    segments: list[dict[str, Any]],
    blocks_by_id: dict[str, dict[str, Any]],
    *,
    protected_names: list[str],
) -> tuple[dict[str, Any], list[str]]:
    """Validate and atomically migrate cached final-review translations."""
    checkpoint = read_json(checkpoint_path)
    translations = checkpoint.get("translations") if isinstance(checkpoint, dict) else None
    if not isinstance(translations, dict):
        raise RuntimeError("review checkpoint has no translation mapping")
    changed_segments: list[str] = []
    repaired_translations = dict(translations)
    for segment in segments:
        segment_id = str(segment["segment_id"])
        translation = translations.get(segment_id)
        if not isinstance(translation, dict):
            raise RuntimeError(f"review checkpoint has no translation for {segment_id}")
        repaired, changed_ids = _normalize_translation_citation_delimiters_for_segment(
            translation, blocks_by_id
        )
        _validate_translation(segment, repaired, blocks_by_id, protected_names)
        repaired_translations[segment_id] = repaired
        if changed_ids:
            changed_segments.append(segment_id)
    if not changed_segments:
        return checkpoint, []
    repaired_checkpoint = {**checkpoint, "translations": repaired_translations}
    write_json(checkpoint_path, repaired_checkpoint)
    return repaired_checkpoint, changed_segments


def _repair_translation_checkpoint_citation_delimiters(
    checkpoint_path: Path,
    segment: dict[str, Any],
    blocks_by_id: dict[str, dict[str, Any]],
    *,
    protected_names: list[str],
) -> dict[str, Any]:
    """Atomically normalize a validated checkpoint and record deterministic provenance."""
    checkpoint = read_json(checkpoint_path)
    translation = checkpoint.get("translation") if isinstance(checkpoint, dict) else None
    raw_blocks = translation.get("blocks") if isinstance(translation, dict) else None
    if not isinstance(raw_blocks, list):
        raise RuntimeError("translation checkpoint has no block list")
    for item in raw_blocks:
        if not isinstance(item, dict):
            raise RuntimeError("translation checkpoint contains a malformed block")
        item_id = str(item.get("block_id") or "")
        if blocks_by_id.get(item_id) is None:
            raise RuntimeError(f"translation checkpoint contains unknown block {item_id}")
    repaired_translation, changed_ids = _normalize_translation_citation_delimiters_for_segment(
        translation, blocks_by_id
    )
    _validate_translation(segment, repaired_translation, blocks_by_id, protected_names)
    if not changed_ids:
        return checkpoint
    generation_provenance = dict(checkpoint.get("generation_provenance") or {})
    repairs = list(generation_provenance.get("repairs") or [])
    repairs.append(_citation_delimiter_normalization_provenance(changed_ids))
    repaired_checkpoint = {
        **checkpoint,
        "generation_provenance": {**generation_provenance, "repairs": repairs},
        "translation": repaired_translation,
    }
    write_json(checkpoint_path, repaired_checkpoint)
    return repaired_checkpoint


def _translation_natural_residue(previous_text: str) -> str:
    """Remove any bounded controller-marker candidate, including mutated tokens."""
    return _OPAQUE_INLINE_CANDIDATE_PATTERN.sub("", previous_text)


def _opaque_inline_token(run: dict[str, Any]) -> str:
    return _project_opaque_inline_token(run)


_OPAQUE_INLINE_PATTERN = re.compile(r"\[\[ARC_INLINE:[^\]\s]+:[0-9a-f]{64}\]\]")
_OPAQUE_INLINE_CANDIDATE_PATTERN = re.compile(
    r"\[\s*\[ARC_INLINE:[^\]\r\n]{0,512}\]\s*\]"
)


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
    options: BuildOptions | None = None,
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
        "access": _generation_runtime_policy(options),
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


def _restore_translation_protected_names(
    segment: dict[str, Any],
    translation: dict[str, Any],
    blocks_by_id: dict[str, dict[str, Any]],
    protected_names: list[str],
) -> tuple[dict[str, Any], list[str]]:
    """Deterministically retain source eponyms without relaxing validation."""
    expected_blocks, raw_blocks = _translation_preflight(
        segment, translation, blocks_by_id
    )
    changed_ids: list[str] = []
    restored_blocks: list[dict[str, Any]] = []
    for source, translated in zip(expected_blocks, raw_blocks):
        text = str(translated.get("text") or "")
        missing = _minimal_missing_name_insertions(
            _missing_protected_names([source], text, protected_names)
        )
        if missing:
            text = _append_protected_name_annotation(text, missing)
            changed_ids.append(block_id(source))
        restored_blocks.append({**translated, "text": text})
    if not changed_ids:
        return translation, []
    return {**translation, "blocks": restored_blocks}, changed_ids


def _append_protected_name_annotation(text: str, names: list[str]) -> str:
    """Append one minimal Latin-name annotation before terminal punctuation."""
    if not names:
        return text
    annotation = f"（{'、'.join(names)}）"
    match = re.search(r"([。！？!?；;：:,，]+\s*)$", text)
    if match:
        return text[:match.start()] + annotation + text[match.start():]
    return text + annotation


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
    return non_substantive_block_ids(document)


def _annotation_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in metadata.items()
        if key.casefold() not in {"title", "titles", "author", "authors", "affiliation", "affiliations"}
    }


def _protected_names(bundle: SourceBundle, *, glossary: dict[str, Any] | None = None) -> list[str]:
    """Protect seed/glossary names only; reference and citer authors stay evidence-local."""
    metadata_names = _author_name_values(
        bundle.metadata.get("authors") or bundle.metadata.get("author")
    )
    if metadata_names:
        values = metadata_names
    else:
        front_names = _author_name_values(
            (bundle.document.get("front_matter") or {}).get("authors")
        )
        values = [
            name
            for value in front_names
            for name in _clean_front_matter_author_line(value)
        ]
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
                if token.casefold() not in _AUTHOR_CONNECTOR_TOKENS and token[:1].isupper():
                    unique.setdefault(token.casefold(), token)
    return sorted(unique.values(), key=lambda value: (value.casefold(), value))


_AUTHOR_CONNECTOR_TOKENS = frozenset({"al", "and", "et"})


def _author_name_values(authors: Any) -> list[str]:
    values: list[str] = []
    if isinstance(authors, str):
        if authors.strip():
            values.append(authors)
    elif isinstance(authors, dict):
        name = authors.get("name") or authors.get("full_name")
        if name and str(name).strip():
            values.append(str(name))
    elif isinstance(authors, list):
        for author in authors:
            values.extend(_author_name_values(author))
    return values


def _clean_front_matter_author_line(value: str) -> list[str]:
    """Recover names from a flattened legacy author line without protecting prose."""
    text = re.sub(r"[*†‡§]+", " ", value)
    text = re.sub(
        r"(?:(?<=\s)|(?<=[A-Za-z.]))\d+(?:\s*[,;]\s*\d+)*(?=\s|$)",
        " ",
        text,
    )
    text = re.sub(r"\bet\s+al\.?\b", " ", text, flags=re.IGNORECASE)
    parts = re.split(r"\s+(?:and|&)\s+", text, flags=re.IGNORECASE)
    return [
        cleaned
        for part in parts
        if (cleaned := re.sub(r"\s+", " ", part).strip(" ,;"))
    ]


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
        # A protected name is a canonical Latin spelling, not a case-folded
        # keyword.  Exact source case prevents roots such as ``May``, ``Young``,
        # or ``Lie`` from matching ordinary prose while retaining true eponyms.
        if re.search(rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])", source)
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
    if bool(annotation.get("prior_work")) and not any(
        relation_by_id.get(value) == "prior" for value in used
    ):
        raise RuntimeError("prior-work commentary has no cited prior-work evidence")
    if bool(annotation.get("later_work")) and not any(
        relation_by_id.get(value) == "later" for value in used
    ):
        raise RuntimeError("later-work commentary has no cited later-work evidence")


def _evidence_for_segment(
    segment: dict[str, Any],
    blocks_by_id: dict[str, dict[str, Any]],
    evidence: dict[str, Any],
    *,
    usage_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_text = " ".join(
        str(blocks_by_id[value].get("text") or blocks_by_id[value].get("title") or "")
        for value in segment.get("block_ids") or []
    )
    source_tokens = _terms(source_text)
    citation_targets = _segment_citation_targets(segment, blocks_by_id, evidence)
    related_papers = list(evidence.get("related_papers") or [])
    idf = _related_paper_idf(related_papers)
    required_ids = set(
        (evidence.get("required_evidence_ids_by_segment") or {}).get(str(segment.get("segment_id")), [])
    )
    selected: list[dict[str, Any]] = []
    for relation in ("prior", "later", "context"):
        candidates: list[tuple[int, int, float, int, int, dict[str, Any]]] = []
        for index, paper in enumerate(related_papers):
            if paper.get("relation") != relation:
                continue
            required = str(paper.get("evidence_id") or "") in required_ids
            if relation == "context" and not required:
                continue
            exact = relation == "prior" and _paper_matches_citation_targets(paper, citation_targets)
            local_relevance = _strongest_local_relevance(paper, source_tokens, idf)
            if not (required or exact or local_relevance is not None):
                continue
            relevance = (
                (16 if required else 0)
                + (12 if exact else 0)
                + (local_relevance or 0.0)
            )
            citation_prior = min(_citation_count_value(paper).bit_length(), 10)
            candidates.append((-int(exact), -int(required), -relevance, -citation_prior, index, paper))
        for _, _, _, _, _, paper in _select_related_candidates(
            candidates, usage_state=usage_state, limit=3,
        ):
            compact = {key: paper.get(key) for key in (
                "evidence_id", "relation", "paper_id", "arxiv_id", "doi", "inspire_id",
                "title", "authors", "year", "citation_count", "evidence_level", "abstract",
                "supported_request_keys",
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
                snippet = {
                    "block_id": str(block.get("block_id") or ""),
                    "text": text,
                    "sha256": text_sha256(text),
                }
                if str(block.get("section_title") or "").strip():
                    snippet["section_title"] = str(block["section_title"]).strip()
                snippets.append(snippet)
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
    context_papers = [
        paper for paper in evidence.get("related_papers") or []
        if paper.get("relation") == "context"
    ]
    if context_papers:
        per_source_budget = min(
            CONTEXT_SEGMENT_CHARS_PER_SOURCE,
            max(1, CONTEXT_SEGMENT_CHARS_TOTAL // len(context_papers)),
        )
        for paper in context_papers:
            compact = {key: paper.get(key) for key in (
                "evidence_id", "relation", "paper_id", "title", "authors", "year",
                "evidence_level", "abstract", "context_role", "context_index",
            )}
            ranked_blocks = sorted(
                enumerate(paper.get("blocks") or []),
                key=lambda pair: (
                    -len(source_tokens & _terms(str(pair[1].get("text") or ""))),
                    pair[0],
                ),
            )
            remaining = per_source_budget
            snippets: list[dict[str, str]] = []
            for _, block in ranked_blocks:
                text = str(block.get("text") or "")[:remaining]
                if not text:
                    continue
                snippet = {
                    "block_id": str(block.get("block_id") or ""),
                    "text": text,
                    "sha256": text_sha256(text),
                }
                if str(block.get("section_title") or "").strip():
                    snippet["section_title"] = str(block["section_title"]).strip()
                snippets.append(snippet)
                remaining -= len(text)
                if remaining <= 0:
                    break
            if not snippets:
                continue
            descriptor = paper.get("source_descriptor") or {}
            locator = descriptor.get("locator") or {}
            compact["snippets"] = snippets
            compact["context_selection"] = {
                "version": CONTEXT_SELECTION_VERSION,
                "query_sha256": text_sha256(source_text),
                "chars_per_source": CONTEXT_SEGMENT_CHARS_PER_SOURCE,
                "chars_total": CONTEXT_SEGMENT_CHARS_TOTAL,
                "selected_chars": sum(len(item["text"]) for item in snippets),
            }
            compact["source_descriptor"] = arc_cache_descriptor(
                paper_id=str(paper.get("paper_id") or descriptor.get("canonical_locator") or ""),
                title=str(paper.get("title") or ""),
                authors=paper.get("authors") or [],
                year=paper.get("year"),
                evidence_level="full_text",
                content=snippets,
                document_hash=str(locator.get("document_hash") or ""),
            )
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
    return {
        "schema_version": "arc.companion.segment-evidence.v3",
        "citation_targets": citation_targets,
        "reference_catalog": _segment_metadata_catalog(
            evidence.get("references") or [], source_tokens, citation_targets=citation_targets,
        ),
        "citer_catalog": _segment_metadata_catalog(
            evidence.get("citers") or [], source_tokens, citation_targets=[],
        ),
        "papers": selected,
    }


def _select_related_candidates(
    candidates: list[tuple[int, int, float, int, int, dict[str, Any]]],
    *,
    usage_state: dict[str, Any] | None,
    limit: int,
) -> list[tuple[int, int, float, int, int, dict[str, Any]]]:
    """Apply deterministic whole-paper soft reuse and topic-diversity penalties."""
    if usage_state is None:
        return sorted(candidates)[:limit]
    counts = usage_state.setdefault("counts", {})
    global_topics = usage_state.setdefault("topics", [])
    remaining = list(candidates)
    chosen: list[tuple[int, int, float, int, int, dict[str, Any]]] = []
    local_topics: list[set[str]] = []
    while remaining and len(chosen) < limit:
        ranked: list[tuple[float, int, tuple[int, int, float, int, int, dict[str, Any]], set[str]]] = []
        for candidate in remaining:
            exact = candidate[0] < 0
            required = candidate[1] < 0
            paper = candidate[-1]
            evidence_id = str(paper.get("evidence_id") or "")
            topic = _terms(f"{paper.get('title', '')} {paper.get('abstract', '')}")
            reuse_penalty = 0.0 if exact or required else 2.25 * int(counts.get(evidence_id, 0))
            comparisons = [*global_topics[-24:], *local_topics]
            diversity_penalty = 0.0
            if comparisons and topic and not exact and not required:
                diversity_penalty = 1.5 * max(
                    len(topic & other) / max(1, len(topic | other))
                    for other in comparisons if other
                )
            adjusted = -float(candidate[2]) - reuse_penalty - diversity_penalty
            ranked.append((-adjusted, int(candidate[4]), candidate, topic))
        _, _, best, topic = min(ranked)
        exact = best[0] < 0
        required = best[1] < 0
        evidence_id = str(best[-1].get("evidence_id") or "")
        adjusted = -min(ranked)[0]
        if adjusted <= 0.0 and not exact and not required:
            break
        chosen.append(best)
        remaining.remove(best)
        counts[evidence_id] = int(counts.get(evidence_id, 0)) + 1
        if topic:
            local_topics.append(topic)
            global_topics.append(topic)
    return chosen


def _terms(text: str) -> set[str]:
    generic = {
        "about", "absent", "after", "also", "although", "analysis", "and", "another",
        "approach", "are", "around", "been", "before", "being", "between", "both",
        "but", "cal", "can", "cannot", "cdots", "could", "delimited-", "different",
        "displaystyle", "does", "dot", "each", "either", "equiv", "equiv-", "field",
        "fields", "first", "for", "form", "frac",
        "from", "further", "general", "have", "here", "however", "into", "italic-",
        "greater-than", "its", "langle", "later", "left", "less-than", "may", "model",
        "models", "more", "most", "not", "one",
        "only", "other", "our", "over", "paper", "partial_", "perm", "physics", "prime",
        "proportional-to", "rangle", "rather", "result", "results",
        "right", "same", "second", "should", "sim", "similar-to", "since", "some",
        "study", "subscript",
        "such", "sum_", "superscript", "than", "that", "the", "their", "then", "theory",
        "there", "these", "they", "this", "those", "through", "times", "tilde",
        "two", "under", "using", "very", "was", "were", "where", "which", "while",
        "via", "with", "within", "without", "work", "would",
    }
    return {
        token.casefold()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", text)
        if token.casefold() not in generic
    }


def _segment_citation_targets(
    segment: dict[str, Any],
    blocks_by_id: dict[str, dict[str, Any]],
    evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    target_ids: list[str] = []
    for block_id_value in segment.get("block_ids") or []:
        for run in blocks_by_id.get(str(block_id_value), {}).get("inline_runs") or []:
            if not isinstance(run, dict) or str(run.get("kind") or "") != "citation":
                continue
            target_id = str(run.get("target_id") or "").lstrip("#").strip()
            if target_id and target_id not in target_ids:
                target_ids.append(target_id)
    bibliography = {
        str(item.get("id") or ""): item
        for item in evidence.get("bibliography") or []
        if isinstance(item, dict)
    }
    return [
        {key: bibliography.get(target_id, {}).get(key) for key in (
            "id", "label", "text", "doi", "arxiv_id", "links",
        ) if bibliography.get(target_id, {}).get(key) not in (None, "")}
        for target_id in target_ids
        if target_id in bibliography
    ]


def _related_paper_idf(papers: list[dict[str, Any]]) -> dict[str, float]:
    """Compute corpus rarity without joining a candidate's separate evidence pieces."""
    document_frequency: Counter[str] = Counter()
    for paper in papers:
        terms: set[str] = set()
        for _, piece in _paper_evidence_pieces(paper):
            terms.update(_terms(piece))
        document_frequency.update(terms)
    count = max(1, len(papers))
    return {
        term: 1.0 + math.log((count + 1) / (frequency + 1))
        for term, frequency in document_frequency.items()
    }


def _paper_evidence_pieces(paper: dict[str, Any]) -> list[tuple[str, str]]:
    pieces = [
        ("title", str(paper.get("title") or "")),
        ("abstract", str(paper.get("abstract") or "")),
    ]
    pieces.extend(
        ("block", str(block.get("text") or ""))
        for block in paper.get("blocks") or []
        if isinstance(block, dict)
    )
    return [(kind, text) for kind, text in pieces if text.strip()]


def _strongest_local_relevance(
    paper: dict[str, Any],
    source_terms: set[str],
    idf: dict[str, float],
) -> float | None:
    """Return strong relevance from one auditable piece, never an entire-paper union."""
    if not source_terms:
        return None
    unseen_weight = max(idf.values(), default=1.0) + 0.5

    def weight(terms: set[str]) -> float:
        return sum(idf.get(term, unseen_weight) for term in terms)

    source_weight = weight(source_terms)
    best: float | None = None
    for kind, text in _paper_evidence_pieces(paper):
        piece_terms = _terms(text)
        shared = source_terms & piece_terms
        if len(shared) < 2:
            continue
        shared_weight = weight(shared)
        piece_weight = weight(piece_terms)
        if shared_weight < 1.5 or source_weight <= 0 or piece_weight <= 0:
            continue
        source_coverage = shared_weight / source_weight
        piece_coverage = shared_weight / piece_weight
        cosine = shared_weight / math.sqrt(source_weight * piece_weight)
        if kind == "title":
            eligible = source_coverage >= 0.03 and piece_coverage >= 0.50
            score = 4.0 + 4.0 * piece_coverage + 2.0 * cosine
        else:
            eligible = (
                source_coverage >= 0.12
                and piece_coverage >= 0.10
                and cosine >= 0.15
            )
            score = 3.0 * cosine + source_coverage + piece_coverage
        if eligible and (best is None or score > best):
            best = score
    return best


def _paper_matches_citation_targets(
    paper: dict[str, Any], targets: list[dict[str, Any]],
) -> bool:
    paper_arxiv = _normalized_arxiv_id(paper.get("arxiv_id") or paper.get("paper_id"))
    paper_doi = _normalized_doi(paper.get("doi"))
    paper_title = _terms(str(paper.get("title") or ""))
    for target in targets:
        target_arxiv = _normalized_arxiv_id(target.get("arxiv_id"))
        target_doi = _normalized_doi(target.get("doi"))
        if paper_arxiv and target_arxiv and paper_arxiv == target_arxiv:
            return True
        if paper_doi and target_doi and paper_doi == target_doi:
            return True
        target_terms = _terms(str(target.get("text") or ""))
        if paper_title and len(paper_title & target_terms) >= max(2, (len(paper_title) + 1) // 2):
            return True
    return False


def _segment_metadata_catalog(
    records: list[dict[str, Any]],
    source_tokens: set[str],
    *,
    citation_targets: list[dict[str, Any]],
    limit: int = 40,
    max_chars: int = 12_000,
) -> list[dict[str, Any]]:
    fields = (
        "paper_id", "arxiv_id", "doi", "inspire_id", "title", "authors",
        "year", "citation_count", "abstract",
    )
    ranked = []
    for index, item in enumerate(records):
        exact = _paper_matches_citation_targets(item, citation_targets)
        search_text = f"{item.get('title', '')} {item.get('abstract', '')}"
        overlap = len(source_tokens & _terms(search_text))
        citation_prior = min(_citation_count_value(item).bit_length(), 10)
        ranked.append((-int(exact), -overlap, -citation_prior, index, item))
    output: list[dict[str, Any]] = []
    used = 2
    for _, _, _, _, item in sorted(ranked):
        compact = {key: item[key] for key in fields if key in item}
        if "abstract" in compact:
            compact["abstract"] = str(compact["abstract"])[:1_200]
        size = len(json.dumps(compact, ensure_ascii=False, separators=(",", ":"))) + 1
        if output and used + size > max_chars:
            break
        if size > max_chars:
            compact.pop("abstract", None)
            size = len(json.dumps(compact, ensure_ascii=False, separators=(",", ":"))) + 1
        if used + size > max_chars:
            continue
        output.append(compact)
        used += size
        if len(output) >= limit:
            break
    return output


def _citation_count_value(item: dict[str, Any]) -> int:
    try:
        return max(0, int(item.get("citation_count") or 0))
    except (TypeError, ValueError):
        return 0


def _normalized_arxiv_id(value: Any) -> str:
    if isinstance(value, list):
        value = value[0] if value else ""
    if isinstance(value, dict):
        value = value.get("value") or value.get("id") or ""
    text = str(value or "").casefold().strip()
    text = re.sub(r"^(?:arxiv:)?", "", text)
    return re.sub(r"v\d+$", "", text)


def _normalized_doi(value: Any) -> str:
    if isinstance(value, list):
        value = value[0] if value else ""
    if isinstance(value, dict):
        value = value.get("value") or value.get("id") or ""
    text = str(value or "").casefold().strip()
    return re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", text).rstrip(".,;)")


def _load_recovered_section_reviews(
    checkpoint_dir: Path,
    chunks: list[list[dict[str, Any]]],
) -> dict[int, dict[str, Any]]:
    """Import a controller-recovered failed-final payload within this fingerprint."""
    path = checkpoint_dir / "section-reviews.recovered-from-failed-final.v1.json"
    if not path.is_file():
        return {}
    recovered = read_json(path)
    if (
        not isinstance(recovered, dict)
        or recovered.get("schema_version") != "arc.companion.recovered-section-reviews.v1"
        or not isinstance(recovered.get("section_reviews"), list)
    ):
        raise RuntimeError("invalid recovered section-review checkpoint")
    expected_all = {str(item["segment"]["segment_id"]) for chunk in chunks for item in chunk}
    if set(str(value) for value in recovered.get("reviewed_segment_ids") or []) != expected_all:
        raise RuntimeError("recovered section reviews do not match current segment coverage")
    imported: dict[int, dict[str, Any]] = {}
    for item in recovered["section_reviews"]:
        if not isinstance(item, dict):
            raise RuntimeError("recovered section review is malformed")
        index = int(item.get("section_index", -1))
        if index < 0 or index >= len(chunks) or index in imported:
            raise RuntimeError("recovered section review has an invalid section index")
        expected_ids = {
            str(value["segment"]["segment_id"]) for value in chunks[index]
        }
        reviewed_segments = item.get("reviewed_segments")
        reviewed_ids = {
            str(value.get("segment_id") or "")
            for value in reviewed_segments or []
            if isinstance(value, dict)
        }
        if (
            not isinstance(reviewed_segments, list)
            or reviewed_ids != expected_ids
            or set(str(value) for value in item.get("reviewed_segment_ids") or [])
            != expected_ids
            or not isinstance(item.get("findings"), list)
        ):
            raise RuntimeError(f"recovered section review {index} does not match its chunk")
        imported[index] = {
            "findings": item["findings"],
            "reviewed_segments": reviewed_segments,
        }
    if set(imported) != set(range(len(chunks))):
        raise RuntimeError("recovered section reviews do not cover every section")
    return imported


def _section_review_patch_proposals(
    chunk: list[dict[str, Any]],
    reviewed_segments: list[dict[str, Any]],
    findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project full section-review echoes into final-review-compatible deltas."""
    original_by_id = {
        str(item["segment"]["segment_id"]): item for item in chunk
    }
    proposals: list[dict[str, Any]] = []
    patchable_annotation_fields = (
        "commentary", "explanation", "prior_work", "later_work", "evidence_ids"
    )
    finding_ids = {str(item.get("segment_id") or "") for item in findings}
    for reviewed in reviewed_segments:
        segment_id = str(reviewed.get("segment_id") or "")
        if segment_id not in finding_ids:
            continue
        original = original_by_id[segment_id]
        proposal: dict[str, Any] = {
            "segment_id": segment_id,
            "translation_blocks": None,
            "commentary": None,
            "explanation": None,
            "prior_work": None,
            "later_work": None,
            "evidence_ids": None,
            "reason": "section reviewer proposed a correction",
        }
        if reviewed.get("translation") != original.get("translation"):
            proposal["translation_blocks"] = list(
                (reviewed.get("translation") or {}).get("blocks") or []
            )
        reviewed_annotation = reviewed.get("annotation") or {}
        original_annotation = original.get("annotation") or {}
        for field in patchable_annotation_fields:
            if reviewed_annotation.get(field) != original_annotation.get(field):
                proposal[field] = reviewed_annotation.get(field)
        if any(
            proposal[field] is not None
            for field in ("translation_blocks", *patchable_annotation_fields)
        ):
            proposals.append(proposal)
    return proposals


def _bounded_section_review_projection(
    section_reviews: list[dict[str, Any]],
    *,
    total_chars: int,
) -> list[dict[str, Any]]:
    """Keep coverage, bounded findings, and as many complete proposals as fit."""
    projected: list[dict[str, Any]] = []
    for section in section_reviews:
        projected.append({
            "section_index": int(section["section_index"]),
            "reviewed_segment_ids": list(section.get("reviewed_segment_ids") or []),
            "findings": [],
            "omitted_finding_count": 0,
            "patch_proposals": [],
            "omitted_patch_proposal_count": 0,
        })
    budget = max(total_chars, len(json.dumps(projected, ensure_ascii=False)))
    for source, target in zip(section_reviews, projected):
        for finding in source.get("findings") or []:
            issue = str(finding.get("issue") or "")
            compact = {
                "segment_id": str(finding.get("segment_id") or ""),
                "issue": issue[:500],
                "issue_sha256": hashlib.sha256(issue.encode("utf-8")).hexdigest(),
                "truncated": len(issue) > 500,
            }
            target["findings"].append(compact)
            if len(json.dumps(projected, ensure_ascii=False)) <= budget:
                continue
            target["findings"].pop()
            target["omitted_finding_count"] += 1
    for source, target in zip(section_reviews, projected):
        for proposal in source.get("patch_proposals") or []:
            target["patch_proposals"].append(proposal)
            if len(json.dumps(projected, ensure_ascii=False)) <= budget:
                continue
            target["patch_proposals"].pop()
            target["omitted_patch_proposal_count"] += 1
    return projected


def _bounded_glossary_projection(
    glossary: dict[str, Any],
    *,
    total_chars: int,
) -> dict[str, Any]:
    """Keep complete glossary entries atomically within final-review context."""
    entries = list(glossary.get("entries") or []) if isinstance(glossary, dict) else []
    projected = {"entries": [], "omitted_entry_count": 0}
    for entry in entries:
        projected["entries"].append(entry)
        if len(json.dumps(projected, ensure_ascii=False)) <= total_chars:
            continue
        projected["entries"].pop()
        projected["omitted_entry_count"] += 1
    return projected




def _review_context_evidence(
    segment: dict[str, Any],
    *,
    blocks_by_id: dict[str, dict[str, Any]],
    evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return the exact bounded context records used for this segment."""
    return [
        item
        for item in _evidence_for_segment(segment, blocks_by_id, evidence)["papers"]
        if item.get("relation") == "context"
    ]


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


def _review_context_evidence_anchors(
    segment_payloads: list[dict[str, Any]], *, total_chars: int
) -> list[dict[str, Any]]:
    """Keep auditable per-segment context descriptors in the final review budget."""
    per_segment = max(1, total_chars // max(1, len(segment_payloads)))
    output: list[dict[str, Any]] = []
    for item in segment_payloads:
        segment = item.get("segment") or {}
        records: list[dict[str, Any]] = []
        remaining = per_segment
        for evidence_item in item.get("context_evidence") or []:
            snippets: list[dict[str, str]] = []
            for snippet in evidence_item.get("snippets") or []:
                if remaining <= 0:
                    break
                text = str(snippet.get("text") or "")[:remaining]
                if not text:
                    continue
                snippets.append({
                    "block_id": str(snippet.get("block_id") or ""),
                    "text": text,
                    "sha256": text_sha256(text),
                })
                remaining -= len(text)
            descriptor = evidence_item.get("source_descriptor")
            records.append({
                "evidence_id": str(evidence_item.get("evidence_id") or ""),
                "paper_id": str(evidence_item.get("paper_id") or ""),
                "source_descriptor": descriptor if isinstance(descriptor, dict) else {},
                "snippets": snippets,
                "selection": evidence_item.get("context_selection") or {},
            })
        output.append({
            "segment_id": str(segment.get("segment_id") or ""),
            "context_evidence": records,
            "excerpt_budget_chars": per_segment,
        })
    return output


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
