from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import wraps
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import shutil
import threading
import unicodedata
import uuid
from urllib.parse import urlparse
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

if TYPE_CHECKING:
    from arc_llm import EvidenceJournalContext

from bs4 import BeautifulSoup
from .context_sources import load_context_evidence
from .content import (
    checkpoint_receipts,
    reader_content_from_overrides,
    store_reader_content,
)
from .chapters import CHAPTERS_VERSION, build_chapters
from .artifact_store import AcceptedArtifactStore, artifact_id_for, canonical_sha256
from .chapter_glossary import (
    INDEX_GLOSSARY_BATCH_VERSION,
    generate_index_glossary,
    project_segment_glossary,
)
from .chapter_guide import (
    CHAPTER_GUIDE_VERSION,
    chapter_guide_artifact_valid,
    generate_chapter_guide,
)
from .chapter_scheduler import run_chapter_pipeline
from .ledger import (
    LaneLedgerError,
    accept_deferred_block,
    accept_controller_skipped_block,
    accept_reused_block,
    advance_block,
    clear_needs_supervision,
    initialize_control_ledger,
    initialize_lane_ledger,
    invalidate_suffix,
    lane_transition_guard,
    mark_needs_supervision,
    mark_response_received,
    mark_submitted,
)
from .ledger_registry import (
    LaneLedgerRegistryError,
    lane_ledger_is_registered,
    legacy_lane_ledger_paths,
    read_registered_lane_ledger,
    registered_lane_ledger_paths,
)
from .progress import CompanionProgress
from .recovery_units import (
    call_model_with_recovery_descriptor,
    recovery_unit_for_ledger,
    submission_descriptor,
    pipeline_lane_binding,
)
from .recovery_responses import (
    RecoveryResponseError,
    discover_submission_receipts,
    explicit_attempt_references,
    recover_complete_ledger_response,
    seal_submission_attempts,
    submission_receipt_reference,
    validate_ledger_submission_reference,
    write_ledger_submission_receipt,
    resolve_recovery_path,
)
from .migration import (
    MIGRATION_VERSION,
    NEVER_MIGRATED_ARTIFACTS,
    accepted_translation_projection_candidates,
    import_accepted_checkpoint_objects,
    legacy_translation_candidates,
    migrate_legacy_cuts,
    migrate_legacy_glossary,
    migrate_legacy_translations,
    read_legacy_checkpoint,
)
from .glossary import (
    GLOSSARY_VERSION,
    generate_glossary,
    glossary_entry_limit,
    validate_glossary_acceptance_checkpoint,
)
from .domain import load_domain_context
from .evidence import (
    arc_cache_descriptor,
    text_sha256,
    validate_evidence_record,
    validate_registry,
)
from .io import read_json, safe_name, sha256_file, sha256_json, write_json, write_text
from .intent_guidance import (
    build_intent_guidance,
    worker_guidance_payload,
    worker_guidance_prompt_prefix,
    worker_policy_descriptor,
    resolve_worker_evidence_requests,
)
from .language import base_language, contains_lexical_term, normalize_language_tag
from .latex import LatexError, render_companion_tex, validate_tex_fidelity
from .pdf import (
    compile_latex,
    managed_run_root_pdf_path,
    normalize_run_root_pdf_state,
    publish_run_root_pdf,
    validate_pdf,
)
from .prompts import (
    ANNOTATION_SCHEMA,
    TRANSLATION_COVERAGE_REPAIR_SCHEMA,
    TRANSLATION_SCHEMA,
    TRANSLATION_SLOT_REPAIR_SCHEMA,
    COMMENTARY_REVIEW_SCHEMA,
    REVIEW_SCHEMA,
    SECTION_REVIEW_SCHEMA,
    COMMENTARY_PROMPT_VERSION,
    PROMPT_VERSION,
    SCHEMA_VERSION,
    TRANSLATION_PROMPT_VERSION,
    TRANSLATION_COVERAGE_REPAIR_PROMPT_VERSION,
    TRANSLATION_COVERAGE_REPAIR_SCHEMA_VERSION,
    TRANSLATION_RETRY_PROMPT_VERSION,
    TRANSLATION_SLOT_REPAIR_SCHEMA_VERSION,
    annotation_prompt,
    commentary_review_prompt,
    review_prompt,
    section_review_prompt,
    translation_coverage_repair_prompt,
    translation_prompt,
    translation_retry_prompt,
)
from .projection import (
    annotation_input_block as _project_annotation_input_block,
    is_structural as _project_is_structural,
    is_translatable as _project_is_translatable,
    opaque_inline_token as _project_opaque_inline_token,
    opaque_inline_tokens as _project_opaque_inline_tokens,
    prompt_safe_value as _project_prompt_safe_value,
    translation_input_block as _project_translation_input_block,
)
from .reader_text import clean_reader_annotation, clean_reader_translation
from .regeneration import normalize_regeneration_lanes
from .response_normalization import (
    ResponseNormalizationError,
    normalize_complete_response_with_receipt,
)
from .secure_io import SecureReadError, read_bounded_file, read_bounded_json
from .reuse import REUSE_PLAN_VERSION, lane_recipe_sha256, lane_semantic_sha256
from .results import err, ok
from .run_lock import BuildInProgressError, ProjectBuildLock
from .segmentation import (
    SEGMENT_HARD_MAX_BLOCKS,
    SEGMENT_HARD_MAX_SOURCE_CHARS,
    SegmentationError,
    construct_segments_from_cuts,
    segment_document,
    validate_segmentation_acceptance_checkpoint,
    validate_exact_coverage,
)
from .source import SourceBundle, SourceError, block_id, load_source_bundle
from .substantive import non_substantive_block_ids
from .title_translation import (
    TITLE_TRANSLATION_PROMPT_VERSION,
    TITLE_TRANSLATION_SCHEMA,
    TITLE_TRANSLATION_VERSION,
    chunk_title_records,
    collect_title_records,
    merge_title_translation_chunks,
    title_translation_prompt,
    validate_title_translations,
)
from .stateful_pipeline import (
    ContextRolloverBudget,
    CorrectionBudget,
    LLMSubmissionLimiter,
    StatefulPromptStream,
    StatefulSessionError,
    continuity_capsule,
    read_stream_state,
    pin_lane_runtime_profile,
    resolve_lane_runtime_profile,
    write_stream_state,
)


WORKFLOW_VERSION = "arc.companion.workflow.v13"
CHAPTER_PROJECTION_VERSION = CHAPTERS_VERSION
AUGMENTATION_PROJECTION_VERSION = "arc.companion.augmentation-projection.v2"
DEFAULT_LANGUAGE = "zh-CN"
DEFAULT_WORKERS = 24
DEFAULT_REVIEW_CONTEXT_CHARS = 140_000
LANGUAGE_NOTICE = "默认使用中文生成伴读；如需切换伴读语言，请直接指定目标语言。"
SEGMENTATION_TIER = "medium"
GLOSSARY_TIER = "medium"
TRANSLATION_TIER = "medium"
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
INTENT_GUIDANCE_TIER = "high"
ANNOTATION_PROMPT_MAX_BYTES = 60 * 1024
ANNOTATION_GLOSSARY_MAX_BYTES = 8 * 1024
ANNOTATION_GLOSSARY_PROJECTION_VERSION = "arc.companion.annotation-glossary-projection.v1"
REVIEW_TIER = "medium"
TITLE_TRANSLATION_TIER = "medium"
REVIEW_VERSION = "arc.companion.review.v8"
ANNOTATION_CHECKPOINT_VERSION = "arc.companion.annotation-checkpoint.v7"
SECTION_REVIEW_CHECKPOINT_VERSION = "arc.companion.section-review-checkpoint.v3"
COMMENTARY_REVIEW_CHECKPOINT_VERSION = "arc.companion.commentary-review-checkpoint.v3"
SECTION_REVIEW_PROMPT_MAX_BYTES = 60 * 1024
REVIEW_PROMPT_MAX_BYTES = 60 * 1024
REVIEW_PROMPT_MIN_SOFT_BYTES = 32 * 1024
REVIEW_PROMPT_TARGET_NUMERATOR = 9
REVIEW_PROMPT_TARGET_DENOMINATOR = 10
REVIEW_PROMPT_BUDGET_AUDIT_VERSION = "arc.companion.review-prompt-budget-audit.v1"
FULL_PAPER_CONTEXT_VERSION = "arc.companion.full-paper-context.v3"
FULL_PAPER_CONTEXT_CHARS = 24_000
FIRST_WAVE_PREVIEW_VERSION = "arc.companion.first-wave-preview.v2"
FINAL_RENDER_VERSION = "arc.companion.final-render.v11"
READER_FINAL_CHECKPOINT_VERSION = "arc.companion.reader-final.v3"
CONTEXT_SELECTION_VERSION = "arc.companion.context-selection.v2"
CONTEXT_SEGMENT_CHARS_PER_SOURCE = 3_000
CONTEXT_SEGMENT_CHARS_TOTAL = 12_000
LEGACY_MIGRATION_VALIDATOR_VERSION = "arc.companion.legacy-validator.v1"

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


class TranslationRepairNeedsSupervision(RuntimeError):
    """A persisted, paid repair response is invalid and must not be resubmitted."""

    def __init__(self, *, segment_id: str, marker_path: Path, reason: str) -> None:
        self.segment_id = segment_id
        self.recovery_context = {
            "submission_state": "submitted",
            "resumable": False,
            "recovery_action": "operator-supervision",
            "blocked_reason": "persisted_repair_response_failed_local_validation",
            "repair_marker": str(marker_path),
        }
        super().__init__(
            f"persisted paid translation repair is invalid for {segment_id}; "
            f"refusing resubmission: {reason}"
        )


class CompanionLLMCircuitOpen(RuntimeError):
    """The build stopped submitting calls after a provider-wide fatal failure."""

    abort_batch = True


@dataclass(frozen=True)
class BuildOptions:
    paper_id: str
    project_dir: Path
    annotation_language: str = DEFAULT_LANGUAGE
    source_language: str | None = None
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
    allow_internet: bool = True
    inherit_host_tools: bool = False
    skip_translation: bool = False
    context_paper_ids: tuple[str, ...] = ()
    user_intent: str | None = None
    stop_after_first_chapter: bool = False
    document_kind: str = "auto"
    idle_timeout_seconds: float | None = None
    recovery_policy: str = "auto"
    max_auto_replacements: int = 3
    regenerate_lanes: tuple[str, ...] = ()
    regenerate_segments: tuple[str, ...] = ()
    confirm_expensive_regeneration: bool = False
    regenerate_commentary: bool = False
    supervised_native_resume_identities: tuple[
        tuple[str, str, str, int, str], ...
    ] = ()
    legacy_checkpoint: Path | None = None

    def __post_init__(self) -> None:
        if not self.paper_id.strip():
            raise ValueError("paper_id is required")
        if self.workers < 1:
            raise ValueError("workers must be at least 1")
        if self.refresh and self.recache:
            raise ValueError("refresh and recache are mutually exclusive")
        if self.domain_id and self.domain_manifest is not None:
            raise ValueError("domain_id and domain_manifest are mutually exclusive")
        if self.document_kind not in {"auto", "article", "book"}:
            raise ValueError("document_kind must be auto, article, or book")
        if self.idle_timeout_seconds is not None and self.idle_timeout_seconds <= 0:
            raise ValueError("idle_timeout_seconds must be positive")
        if self.recovery_policy not in {"auto", "manual"}:
            raise ValueError("recovery_policy must be auto or manual")
        if self.max_auto_replacements < 1:
            raise ValueError("max_auto_replacements must be at least 1")
        normalized_segments: list[str] = []
        for raw in self.regenerate_segments:
            lane, separator, segment_id = str(raw).partition(":")
            if separator != ":" or lane not in {"translation", "commentary"} or not segment_id:
                raise ValueError(
                    "regenerate_segments entries must be translation:SEGMENT_ID "
                    "or commentary:SEGMENT_ID"
                )
            value = f"{lane}:{segment_id}"
            if value not in normalized_segments:
                normalized_segments.append(value)
        object.__setattr__(self, "regenerate_segments", tuple(normalized_segments))
        source_language = str(self.source_language or "").strip() or None
        object.__setattr__(self, "source_language", source_language)
        requested_regeneration = tuple(self.regenerate_lanes) + (
            ("commentary",) if self.regenerate_commentary else ()
        )
        normalized_regeneration = normalize_regeneration_lanes(
            requested_regeneration,
            confirm_expensive_all=self.confirm_expensive_regeneration,
        )
        object.__setattr__(self, "regenerate_lanes", normalized_regeneration)
        object.__setattr__(self, "regenerate_commentary", "commentary" in normalized_regeneration)
        normalized_context_ids = tuple(
            dict.fromkeys(str(value).strip() for value in self.context_paper_ids if str(value).strip())
        )
        if len(normalized_context_ids) != len(self.context_paper_ids):
            raise ValueError("context_paper_ids must be non-empty and unique")
        if self.paper_id.strip() in normalized_context_ids:
            raise ValueError("the source paper cannot also be a context paper")
        object.__setattr__(self, "context_paper_ids", normalized_context_ids)
        object.__setattr__(
            self, "user_intent", str(self.user_intent or "").strip() or None,
        )
        normalized_resume_identities: list[tuple[str, str, str, int, str]] = []
        for raw in self.supervised_native_resume_identities:
            if not isinstance(raw, (tuple, list)) or len(raw) != 5:
                raise ValueError("supervised native resume identity must be a 5-tuple")
            ledger_path, session_key, logical_unit, generation, idempotency_key = raw
            try:
                normalized_generation = int(generation)
            except (TypeError, ValueError) as exc:
                raise ValueError("supervised native resume generation is invalid") from exc
            identity = (
                str(Path(str(ledger_path)).expanduser().resolve(strict=False)),
                str(session_key).strip(),
                str(logical_unit).strip(),
                normalized_generation,
                str(idempotency_key).strip(),
            )
            if (
                not all(identity[index] for index in (0, 1, 2, 4))
                or normalized_generation < 1
            ):
                raise ValueError(
                    "supervised native resume identities must be complete"
                )
            # Durable discovery can encounter the same authorization through
            # both a migrated transaction and a reconstructed receipt.  Fold
            # only an exact normalized five-field duplicate; a shared logical
            # key with any different ownership field remains distinct.
            if identity in normalized_resume_identities:
                continue
            normalized_resume_identities.append(identity)
        object.__setattr__(
            self,
            "supervised_native_resume_identities",
            tuple(normalized_resume_identities),
        )
        if self.legacy_checkpoint is not None:
            legacy_checkpoint = self.legacy_checkpoint.expanduser().resolve()
            if not legacy_checkpoint.exists():
                raise ValueError(f"legacy_checkpoint does not exist: {legacy_checkpoint}")
            if not (legacy_checkpoint.is_file() or legacy_checkpoint.is_dir()):
                raise ValueError("legacy_checkpoint must be a file or directory")
            object.__setattr__(self, "legacy_checkpoint", legacy_checkpoint)

    @property
    def supervised_native_resume_keys(self) -> tuple[str, ...]:
        """Compatibility/audit projection; provider routing never uses this."""

        return tuple(dict.fromkeys(
            identity[4] for identity in self.supervised_native_resume_identities
        ))


def build_companion(
    options: BuildOptions,
    *,
    source_loader: Callable[..., SourceBundle] = load_source_bundle,
    llm: Callable[..., dict[str, Any]] | None = None,
    compiler: Callable[[Path, Path], None] = compile_latex,
    pdf_validator: Callable[[Path], dict[str, object]] = validate_pdf,
    result_llm: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Build or resume one companion under an exclusive project lock."""
    project_dir = options.project_dir.resolve()
    lock = ProjectBuildLock(project_dir / ".arc-companion-build.lock")
    try:
        lock.acquire()
    except BuildInProgressError as exc:
        return err(
            "build_in_progress",
            str(exc),
            project_dir=str(project_dir),
            retryable=True,
        )
    try:
        from arc_llm.cancellation import install_signal_cancel_chain

        with install_signal_cancel_chain() as cancel_check:
            result = _build_companion_unlocked(
                options,
                source_loader=source_loader,
                llm=llm,
                compiler=compiler,
                pdf_validator=pdf_validator,
                result_llm=result_llm,
                cancel_check=cancel_check,
            )
            if (
                options.recovery_policy == "auto"
                and isinstance(result, dict)
                and result.get("status") == "needs_supervision"
            ):
                return _resume_companion_unlocked(
                    options.project_dir.resolve(),
                    action="auto",
                    cancel_check=cancel_check,
                    continuation=lambda recovered_options: _build_companion_unlocked(
                        recovered_options,
                        source_loader=source_loader,
                        llm=llm,
                        compiler=compiler,
                        pdf_validator=pdf_validator,
                        result_llm=result_llm,
                        cancel_check=cancel_check,
                    ),
                    source_preflight=lambda: _preflight_automatic_recovery_source(
                        options,
                    ),
                )
            return result
    finally:
        lock.release()


def _build_companion_unlocked(
    options: BuildOptions,
    *,
    source_loader: Callable[..., SourceBundle] = load_source_bundle,
    llm: Callable[..., dict[str, Any]] | None = None,
    compiler: Callable[[Path, Path], None] = compile_latex,
    pdf_validator: Callable[[Path], dict[str, object]] = validate_pdf,
    result_llm: Callable[..., Any] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Build or resume one companion while keeping source and annotations separate."""
    options = _with_effective_source_language(options)
    chapter_result_llm = result_llm
    if llm is None:
        from arc_llm import run_json
        from arc_llm import run_json_result

        llm = run_json
        chapter_result_llm = chapter_result_llm or run_json_result
    llm = _limit_llm_concurrency(llm, options.workers, cancel_check=cancel_check)
    project_dir = options.project_dir.resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    state_path = project_dir / "state.json"
    notice = LANGUAGE_NOTICE if options.language_was_defaulted else None
    previous_state = _read_optional_json(state_path)
    diagnostics: tuple[dict[str, str], ...] = ()
    fingerprint: str | None = None
    checkpoint_dir: Path | None = None
    _state(
        state_path,
        status="loading_source",
        paper_id=options.paper_id,
        source_language=options.source_language,
        translation_mode="skipped" if options.skip_translation else "enabled",
        notice=notice,
        diagnostics=[],
    )

    try:
        bundle = source_loader(
            options.paper_id,
            refresh=options.refresh,
            recache=options.recache,
            document_kind=options.document_kind,
        )
        generation_document = _generation_document(bundle.document)
        diagnostics = bundle.diagnostics
        def register_bootstrap_recovery_root(root: Path) -> None:
            nonlocal checkpoint_dir
            checkpoint_dir = root.resolve(strict=False)
            _state(
                state_path,
                status="building_intent_guidance",
                paper_id=options.paper_id,
                checkpoint_dir=str(checkpoint_dir),
                recovery_options=_recovery_options(options),
            )

        intent_guidance = build_intent_guidance(
            options.user_intent,
            source_language=str(options.source_language or "und"),
            target_language=options.annotation_language,
            document_type=(
                options.document_kind
                if options.document_kind != "auto"
                else str(
                    (bundle.parsed.get("structure") or {}).get("document_kind")
                    or "article"
                )
            ),
            context_paper_ids=options.context_paper_ids,
            project_dir=project_dir,
            call_model=lambda prompt, schema, artifact_dir, call_label, recovery_descriptor=None: _llm_call(
                llm, prompt, schema, options=options, artifact_dir=artifact_dir,
                call_label=call_label, model_tier=INTENT_GUIDANCE_TIER,
                force_offline=True, disable_paper_cli=True,
                recovery_descriptor=recovery_descriptor,
            ),
            accept_recovery=_accept_registered_pipeline_control,
            register_recovery_root=register_bootstrap_recovery_root,
        )
        intent_guidance_identity = _intent_guidance_identity(intent_guidance)
        # Legacy context papers retain their prior bounded-body behavior only
        # when no user intent was supplied. Intent-guided runs expose metadata
        # and compact TOCs to the guidance call, then read exact sections on demand.
        context_evidence = (
            load_context_evidence(options.context_paper_ids)
            if options.context_paper_ids and intent_guidance is None else []
        )
        evidence = _evidence(bundle, context_evidence=context_evidence)
        domain_context = load_domain_context(
            domain_id=options.domain_id,
            domain_manifest=options.domain_manifest,
        )
        fingerprint = _fingerprint(bundle, options, evidence=evidence, domain_context=domain_context)
        checkpoint_dir = _checkpoint_dir_with_legacy_worker_migration(
            project_dir,
            fingerprint=fingerprint,
            bundle=bundle,
            options=options,
            evidence=evidence,
            domain_context=domain_context,
            previous_state=previous_state,
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        previous_checkpoint = Path(str(previous_state.get("checkpoint_dir") or ""))
        if (
            not options.skip_translation
            and previous_checkpoint.is_dir()
            and previous_checkpoint != checkpoint_dir
            and not options.force
        ):
            for filename in ("glossary.json", "index-glossary.json"):
                source = previous_checkpoint / filename
                target = checkpoint_dir / filename
                if source.is_file() and not target.exists():
                    # The owning generator revalidates its content hash before
                    # reuse, so carrying the file across generation fingerprints
                    # cannot make a stale glossary authoritative.
                    shutil.copy2(source, target)
        if (
            not options.force
            and not options.regenerate_lanes
            and not options.regenerate_segments
            and previous_state.get("status") == "complete"
            and previous_state.get("fingerprint") == fingerprint
            and previous_state.get("translation_mode") == (
                "skipped" if options.skip_translation else "enabled"
            )
            and previous_state.get("intent_guidance_identity") == intent_guidance_identity
            and _read_optional_json(checkpoint_dir / "evidence.json") == evidence
            and _read_optional_json(checkpoint_dir / "domain-context.json") == (domain_context or {})
            and _completion_outputs_match(previous_state)
            and (
                isinstance(bundle.parsed.get("structure"), dict)
                or _first_wave_preview_outputs_match(previous_state)
            )
        ):
            plan_path = checkpoint_dir / "reuse-plan.json"
            if plan_path.is_file():
                plan = read_json(plan_path)
                current_recipes = {
                    lane: lane_recipe_sha256(
                        lane,
                        prompt=(
                            REVIEW_VERSION if lane == "review"
                            else CHAPTER_GUIDE_VERSION if lane == "guide"
                            else TITLE_TRANSLATION_PROMPT_VERSION if lane == "title_translation"
                            else _language_prompt_contract(
                                TRANSLATION_PROMPT_VERSION, options,
                            ) if lane == "translation"
                            else _language_prompt_contract(
                                COMMENTARY_PROMPT_VERSION, options,
                            ) if lane == "commentary"
                            else _language_prompt_contract(
                                PROMPT_VERSION, options,
                            ) if lane == "glossary"
                            else PROMPT_VERSION
                        ),
                        model=options.model,
                        tier={
                            "segmentation": SEGMENTATION_TIER, "glossary": GLOSSARY_TIER,
                            "title_translation": TITLE_TRANSLATION_TIER,
                            "guide": ANNOTATION_TIER, "translation": TRANSLATION_TIER,
                            "commentary": ANNOTATION_TIER, "review": REVIEW_TIER,
                        }[lane],
                        access_recipe=(
                            {
                                "provider": options.provider,
                                "allow_internet": options.allow_internet if lane == "commentary" else False,
                                "inherit_host_tools": (
                                    options.inherit_host_tools
                                    if intent_guidance is None else False
                                ),
                            }
                            if lane in {"translation", "commentary"}
                            else {
                                "provider": options.provider,
                                "allow_internet": options.allow_internet,
                                "inherit_host_tools": (
                                    options.inherit_host_tools
                                    if intent_guidance is None else False
                                ),
                            }
                            if lane == "guide"
                            else {"provider": options.provider, "allow_internet": False}
                        ),
                    )
                    for lane in (
                        "segmentation", "glossary", "title_translation", "guide",
                        "translation", "commentary", "review",
                    )
                }
                stale_lanes = {
                    lane for lane, recipe in current_recipes.items()
                    if any(item.get("recipe_sha256") != recipe for item in AcceptedArtifactStore(project_dir).iter_kind(lane))
                }
                for entry in plan.get("entries") or []:
                    if options.skip_translation and entry.get("lane") in {
                        "glossary", "title_translation",
                    }:
                        entry.update({
                            "status": "skipped", "artifact_id": None,
                            "reason": (
                                "glossary_disabled_for_same_language_source"
                                if entry.get("lane") == "glossary"
                                else "title_translation_disabled_for_same_language_source"
                            ),
                            "estimated_provider_calls": 0,
                        })
                        continue
                    if entry.get("lane") in stale_lanes:
                        entry.update({
                            "status": "recipe_stale",
                            "reason": "accepted artifact remains valid but its generation recipe changed",
                            "estimated_provider_calls": 0,
                        })
                plan["estimated_provider_calls"] = 0
                write_json(plan_path, plan)
            resumed_state = {**previous_state, "diagnostics": list(diagnostics)}
            run_pdf = publish_run_root_pdf(
                Path(str(previous_state["output_pdf"])), project_dir,
                managed_path=managed_run_root_pdf_path(previous_state),
            )
            resumed_state.update(run_pdf)
            if not _web_outputs_match(resumed_state):
                # The reviewed reader checkpoint is authoritative.  Rebuilding a
                # missing web bundle must never cause completed model work to run.
                published = _publish_reader_update(
                    project_dir, state_path, threading.RLock(), strict=True,
                )
                if published is not None:
                    resumed_state = {**resumed_state, **published}
            resumed_state = _state(state_path, **resumed_state)
            return ok(
                resumed_state,
                resumed=True,
                notice=notice,
                diagnostics=list(diagnostics),
            )

        write_json(checkpoint_dir / "document.json", bundle.parsed)
        _state(
            state_path,
            status="active",
            paper_id=bundle.paper_id,
            fingerprint=fingerprint,
            checkpoint_dir=str(checkpoint_dir),
            translation_mode="skipped" if options.skip_translation else "enabled",
            annotation_language=options.annotation_language,
            source_language=options.source_language,
            recovery_options=_recovery_options(options),
            intent_guidance_identity=intent_guidance_identity,
            notice=notice,
            diagnostics=list(diagnostics),
        )
        reader_publish_lock = threading.RLock()
        write_json(checkpoint_dir / "evidence.json", evidence)
        if domain_context is not None:
            write_json(checkpoint_dir / "domain-context.json", domain_context)
        write_json(checkpoint_dir / "source-snapshot-receipt.json", {
            "schema_version": "arc.companion.source-snapshot-receipt.v1",
            "paper_id": bundle.paper_id,
            "fingerprint": fingerprint,
            "document_payload_sha256": sha256_json(bundle.parsed),
            "evidence_sha256": sha256_json(evidence),
            "domain_context_sha256": (
                sha256_json(domain_context) if domain_context is not None else None
            ),
        })
        if isinstance(bundle.parsed.get("structure"), dict):
            return _build_chaptered_companion(
                options=options, bundle=bundle, evidence=evidence,
                domain_context=domain_context, checkpoint_dir=checkpoint_dir,
                fingerprint=fingerprint, notice=notice, diagnostics=diagnostics,
                llm=llm, compiler=compiler, pdf_validator=pdf_validator,
                result_llm=chapter_result_llm,
                intent_guidance=intent_guidance,
                cancel_check=cancel_check,
                require_first_chapter_freeze=(
                    previous_state.get("status") == "first_chapter_ready"
                    and previous_state.get("intent_guidance_identity")
                    == intent_guidance_identity
                ),
            )
        _state(
            state_path,
            status="segmenting",
            paper_id=bundle.paper_id,
            fingerprint=fingerprint,
            notice=notice,
            diagnostics=list(diagnostics),
        )
        _write_legacy_reuse_plan(checkpoint_dir, options)

        initial_names = _protected_names(bundle)

        def segment() -> list[dict[str, Any]]:
            return segment_document(
                generation_document,
                checkpoint_dir=checkpoint_dir,
                workers=options.workers,
                force=options.force,
                call_model=lambda prompt, schema, artifact_dir, call_label, recovery_descriptor=None: _llm_call(
                    llm, prompt, schema, options=options, artifact_dir=artifact_dir,
                    call_label=call_label, model_tier=SEGMENTATION_TIER,
                    recovery_descriptor=recovery_descriptor,
                ),
                accept_recovery=_accept_registered_pipeline_control,
            )

        def glossary_task() -> dict[str, Any]:
            glossary_document = _augmentation_document(generation_document)
            if not glossary_document.get("blocks"):
                return _empty_glossary(
                    language=options.annotation_language,
                    page_count=_page_count(bundle),
                )
            return generate_glossary(
                glossary_document,
                language=options.annotation_language,
                source_language=options.source_language,
                protected_names=initial_names,
                checkpoint_dir=checkpoint_dir,
                workers=options.workers,
                force=options.force,
                page_count=_page_count(bundle),
                intent_guidance_identity=intent_guidance_identity,
                call_model=lambda prompt, schema, artifact_dir, call_label, recovery_descriptor=None: _llm_call(
                    llm, _guided_prompt(prompt, intent_guidance, lane="glossary"), schema,
                    options=options, artifact_dir=artifact_dir,
                    call_label=call_label, model_tier=GLOSSARY_TIER,
                    paper_access_policy=_guidance_policy(intent_guidance, lane="glossary"),
                    intent_guidance=intent_guidance,
                    intent_guidance_lane="glossary",
                    recovery_descriptor=recovery_descriptor,
                ),
                accept_recovery=_accept_registered_pipeline_control,
            )

        # Segmentation is the structural preflight for every downstream lane.
        # Finish it before glossary submission so an invalid cut response keeps
        # its SegmentationError identity and cannot spend another provider call.
        expanded = segment()
        _state(
            state_path, status="active", paper_id=bundle.paper_id,
            fingerprint=fingerprint, checkpoint_dir=str(checkpoint_dir),
            segment_count=len(expanded), notice=notice,
        )
        _publish_reader_update(
            project_dir, state_path, reader_publish_lock,
        )
        glossary = {} if options.skip_translation else glossary_task()
        protected_names = _protected_names(bundle, glossary=glossary)
        title_translations = _generate_title_translations(
            options=options,
            bundle=bundle,
            document=bundle.document,
            chapters=[],
            glossary=glossary,
            protected_names=protected_names,
            checkpoint_dir=checkpoint_dir,
            intent_guidance_identity=intent_guidance_identity,
            call_model=lambda prompt, schema, artifact_dir, call_label, recovery_descriptor=None: _llm_call(
                llm,
                _guided_prompt(prompt, intent_guidance, lane="title_translation"),
                schema, options=options,
                artifact_dir=artifact_dir, call_label=call_label,
                model_tier=TITLE_TRANSLATION_TIER,
                paper_access_policy=_guidance_policy(
                    intent_guidance, lane="title_translation",
                ),
                intent_guidance=intent_guidance,
                intent_guidance_lane="title_translation",
                recovery_descriptor=recovery_descriptor,
            ),
        )

        accepted_translations: dict[str, dict[str, Any]] = {}
        accepted_annotations: dict[str, dict[str, Any]] = {}
        accepted_reader_lock = threading.RLock()

        def publish_accepted_reader(
            lane: str, segment_id: str, value: dict[str, Any],
        ) -> None:
            with accepted_reader_lock:
                target = (
                    accepted_translations if lane == "translation"
                    else accepted_annotations
                )
                target[str(segment_id)] = dict(value)
                overrides = {
                    "document": bundle.document,
                    "chapters": [],
                    "segments": expanded,
                    "chapter_guides": {},
                    "translations": (
                        None if options.skip_translation
                        else dict(accepted_translations)
                    ),
                    "annotations": dict(accepted_annotations),
                    "glossary": glossary,
                    "metadata": bundle.metadata,
                    "language": options.annotation_language,
                    "source_language": options.source_language,
                    "title_translations": title_translations,
                    "translation_mode": (
                        "skipped" if options.skip_translation else "enabled"
                    ),
                }
            _publish_reader_update(
                project_dir, state_path, reader_publish_lock,
                final_overrides=overrides,
            )

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
            accepted_callback=publish_accepted_reader,
            intent_guidance=intent_guidance,
        )
        translations = first_wave_results["translation"]
        raw_annotations = first_wave_results["annotation"]
        if not options.skip_translation:
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
            translations=None if options.skip_translation else translations,
            evidence=evidence,
            glossary=glossary,
            metadata=bundle.metadata,
            language=options.annotation_language,
            source_language=options.source_language,
            title_translations=title_translations,
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
        if options.stop_after_first_chapter:
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
                accepted_callback=publish_accepted_reader,
                intent_guidance=intent_guidance,
            )
            translations = {**translations, **remaining_results["translation"]}
            raw_annotations = {**raw_annotations, **remaining_results["annotation"]}

        write_json(checkpoint_dir / "annotations.first-round.v1.json", {
            "schema_version": "arc.companion.annotations-first-round.v1",
            "annotations": raw_annotations,
        })
        _state(state_path, status="reviewing", paper_id=bundle.paper_id, fingerprint=fingerprint, notice=notice,
               segment_count=len(expanded))
        review_identity_suffix = (
            "" if intent_guidance_identity is None
            else f".{str(intent_guidance_identity['output_sha256'])[:16]}"
        )
        reviewed_path = checkpoint_dir / f"annotations.reviewed.v5{review_identity_suffix}.json"
        review_path = checkpoint_dir / f"review.v5{review_identity_suffix}.json"
        if (
            reviewed_path.is_file()
            and review_path.is_file()
            and not options.force
            and not options.regenerate_commentary
        ):
            cached_reviewed = read_json(reviewed_path)
            review = read_json(review_path)
            if not isinstance(review, dict):
                raise RuntimeError("invalid review checkpoint")
            normalized_review = _with_historical_review_prompt_audit(review, options)
            if normalized_review != review:
                review = normalized_review
                write_json(review_path, review)
            if (
                not isinstance(cached_reviewed, dict)
                or cached_reviewed.get("schema_version") != REVIEW_VERSION
            ):
                raise RuntimeError("invalid review checkpoint")
            normalized_cached_segments: list[str] = []
            if not options.skip_translation:
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
                or set(reviewed) != {segment["segment_id"] for segment in expanded}
                or (
                    not options.skip_translation
                    and (
                        not isinstance(reviewed_translations, dict)
                        or set(reviewed_translations) != set(reviewed)
                    )
                )
            ):
                raise RuntimeError("review checkpoint does not match current segments")
            cached_reader_evidence = _reader_evidence_by_segment(
                expanded,
                document=bundle.document,
                evidence=evidence,
                annotations=reviewed,
            )
            reviewed = {
                str(segment_id): _validate_direct_annotation_sources(
                    annotation,
                )
                for segment_id, annotation in reviewed.items()
            }
            if not options.skip_translation:
                reviewed_translations = {
                    str(segment_id): clean_reader_translation(translation)
                    for segment_id, translation in reviewed_translations.items()
                }
            else:
                reviewed_translations = None
            if (
                reviewed != cached_reviewed.get("annotations")
                or (
                    not options.skip_translation
                    and reviewed_translations != cached_reviewed.get("translations")
                )
            ):
                cached_reviewed["annotations"] = reviewed
                if not options.skip_translation:
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
                intent_guidance=intent_guidance,
            )
            reviewed_payload = {
                "schema_version": REVIEW_VERSION,
                "translation_mode": "skipped" if options.skip_translation else "enabled",
                "annotations": reviewed,
            }
            if not options.skip_translation:
                reviewed_payload["translations"] = reviewed_translations
            write_json(reviewed_path, reviewed_payload)
            write_json(review_path, review)

        final_reader_overrides = {
            "status": "complete",
            "document": bundle.document,
            "chapters": [],
            "segments": expanded,
            "chapter_guides": {},
            "translations": reviewed_translations,
            "annotations": reviewed,
            "glossary": glossary,
            "metadata": bundle.metadata,
            "language": options.annotation_language,
            "source_language": options.source_language,
            "title_translations": title_translations,
            "translation_mode": "skipped" if options.skip_translation else "enabled",
        }
        content_object = _store_reviewed_content(
            project_dir, checkpoint_dir=checkpoint_dir,
            final_overrides=final_reader_overrides, evidence=evidence,
        )
        _write_reader_final_checkpoint(checkpoint_dir, final_reader_overrides)
        _state(
            state_path,
            content_sha256=content_object["content_sha256"],
            content_object_path=str(content_object["path"]),
        )

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
            source_language=options.source_language,
            title_translations=title_translations,
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
        _publish_reader_update(
            project_dir, state_path, reader_publish_lock,
            final_overrides=final_reader_overrides,
            strict=True,
        )
        managed_run_pdf = managed_run_root_pdf_path(
            _read_optional_json(state_path)
        )

        _accept_completed_pipeline_controls(checkpoint_dir)

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
            chapter_projection_version=CHAPTER_PROJECTION_VERSION,
            augmentation_projection_version=AUGMENTATION_PROJECTION_VERSION,
            chapter_guide_version=CHAPTER_GUIDE_VERSION,
            reader_final_checkpoint_version=READER_FINAL_CHECKPOINT_VERSION,
            translation_mode="skipped" if options.skip_translation else "enabled",
            checkpoint_dir=str(checkpoint_dir),
            diagnostics=list(diagnostics),
        )
        run_pdf = publish_run_root_pdf(
            Path(final_artifact["pdf_path"]), project_dir,
            managed_path=managed_run_pdf,
        )
        final_state = _state(state_path, **run_pdf)
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
        if checkpoint_dir is not None and _supervised_lane_ledger_paths(checkpoint_dir):
            supervised = _state(
                state_path,
                status="needs_supervision",
                paper_id=options.paper_id,
                fingerprint=fingerprint,
                checkpoint_dir=str(checkpoint_dir),
                recovery_options=_recovery_options(options),
                notice=notice,
                error=str(exc),
                diagnostics=failure_diagnostics,
            )
            return {
                "ok": False,
                "status": "needs_supervision",
                "data": supervised,
                "error": {
                    "code": "companion_needs_supervision",
                    "message": str(exc),
                },
                "errors": [],
                "meta": {"diagnostics": failure_diagnostics, "notice": notice},
            }
        failure = _state(
            state_path,
            status="failed",
            paper_id=options.paper_id,
            fingerprint=fingerprint,
            checkpoint_dir=str(checkpoint_dir) if checkpoint_dir is not None else None,
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


def _build_chaptered_companion(
    *, options: BuildOptions, bundle: SourceBundle, evidence: dict[str, Any],
    domain_context: dict[str, Any] | None, checkpoint_dir: Path,
    fingerprint: str, notice: str | None, diagnostics: tuple[dict[str, str], ...],
    llm: Callable[..., dict[str, Any]], compiler: Callable[[Path, Path], None],
    pdf_validator: Callable[[Path], dict[str, object]],
    result_llm: Callable[..., Any] | None = None,
    intent_guidance: Mapping[str, Any] | None = None,
    require_first_chapter_freeze: bool = False,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Execute the chapter contract while retaining legacy runs for old caches."""
    regeneration = set(options.regenerate_lanes)
    targeted_regeneration = set(options.regenerate_segments)
    targeted_suffix_reuse: set[str] = set()
    targeted_seen: set[str] = set()
    targeted_lock = threading.Lock()
    artifact_store = AcceptedArtifactStore(options.project_dir.resolve())
    guidance_identity = _intent_guidance_identity(intent_guidance)
    guidance_policy = _guidance_policy(intent_guidance)
    state_path = options.project_dir.resolve() / "state.json"
    document = bundle.document
    structure = bundle.parsed.get("structure") or {}
    index_pack = bundle.parsed.get("index_entries") or {}
    index_entries = (
        list(index_pack.get("entries") or []) if isinstance(index_pack, dict)
        else list(index_pack)
    )
    chapters_pack = build_chapters(document, structure=structure)
    write_json(checkpoint_dir / "chapters.json", chapters_pack)
    chapter_segments_ready = {
        str(item["chapter_id"]): threading.Event()
        for item in chapters_pack["chapters"]
    }
    migration_lock = threading.RLock()
    legacy = (
        read_legacy_checkpoint(options.legacy_checkpoint)
        if options.legacy_checkpoint is not None else None
    )
    migration_source_hash = _legacy_migration_source_hash(bundle)
    migration_prompt_hash = sha256_json({
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "workflow_version": WORKFLOW_VERSION,
    })
    migration_validator_hash = sha256_json({
        "validator_version": LEGACY_MIGRATION_VALIDATOR_VERSION,
        "translation_retry_prompt_version": TRANSLATION_RETRY_PROMPT_VERSION,
        "translation_token_repair_version": TRANSLATION_TOKEN_REPAIR_VERSION,
    })
    write_json(checkpoint_dir / "migration-metadata.json", {
        "schema_version": "arc.companion.migration-metadata.v1",
        "source_hash": migration_source_hash,
        "language": options.annotation_language,
        "translation_mode": "skipped" if options.skip_translation else "enabled",
        "prompt_hash": migration_prompt_hash,
        "validator_hash": migration_validator_hash,
    })
    object_migration = import_accepted_checkpoint_objects(
        options.project_dir.resolve(),
        validators={
            "guide": lambda value: chapter_guide_artifact_valid(
                value, evidence=evidence, allow_internet=options.allow_internet,
            ),
            "translation": lambda value: isinstance(value, dict) and isinstance(value.get("blocks"), list),
            "commentary": lambda value: isinstance(value, dict) and isinstance(value.get("commentary"), str),
            "review": lambda value: isinstance(value, dict),
        },
        contract_versions={
            "guide": CHAPTER_GUIDE_VERSION,
            "translation": SCHEMA_VERSION,
            "commentary": SCHEMA_VERSION,
            "review": REVIEW_VERSION,
        },
    )
    write_json(checkpoint_dir / "object-migration.json", object_migration)
    accepted_translation_candidates = (
        {}
        if "translation" in regeneration else
        accepted_translation_projection_candidates(
            artifact_store,
            source_hash=migration_source_hash,
            language=options.annotation_language,
        )
    )
    initial_reuse_entries = []
    for chapter in chapters_pack["chapters"]:
        chapter_id = str(chapter["chapter_id"])
        for lane in ("segmentation", "guide", "translation", "commentary"):
            if lane == "translation" and options.skip_translation:
                continue
            selected = lane in regeneration
            initial_reuse_entries.append({
                "chapter_id": chapter_id, "segment_id": None, "lane": lane,
                "status": "miss", "artifact_id": None,
                "reason": (
                    "explicitly selected for regeneration" if selected
                    else "detailed object lookup follows deterministic segmentation"
                ),
                "estimated_provider_calls": 1,
            })
    initial_reuse_entries.insert(0, {
        "chapter_id": "project", "segment_id": None, "lane": "glossary",
        "status": "skipped" if options.skip_translation else "miss",
        "artifact_id": None,
        "reason": (
            "glossary_disabled_for_same_language_source"
            if options.skip_translation
            else "explicitly selected for regeneration"
            if "glossary" in regeneration
            else "no validated project glossary object selected"
        ),
        "estimated_provider_calls": 0 if options.skip_translation else 1,
    })
    initial_reuse_entries.insert(1, {
        "chapter_id": "project", "segment_id": None,
        "lane": "title_translation",
        "status": "skipped" if options.skip_translation else "miss",
        "artifact_id": None,
        "reason": (
            "title_translation_disabled_for_same_language_source"
            if options.skip_translation
            else "explicitly selected for regeneration"
            if "translation" in regeneration
            else "title translation lookup waits for the exact title projection"
        ),
        "estimated_provider_calls": 0 if options.skip_translation else 1,
    })
    initial_reuse_entries.append({
        "chapter_id": "project", "segment_id": None, "lane": "review",
        "status": "miss", "artifact_id": None,
        "reason": (
            "explicitly selected for regeneration" if "review" in regeneration
            else "review lookup waits for exact base artifact hashes"
        ),
        "estimated_provider_calls": 1,
    })
    write_json(checkpoint_dir / "reuse-plan.json", {
        "schema_version": REUSE_PLAN_VERSION,
        "entries": initial_reuse_entries,
        "estimated_provider_calls": sum(item["estimated_provider_calls"] for item in initial_reuse_entries),
    })
    reuse_plan_header_lock = threading.Lock()

    def mark_plan_lane(
        lane: str, *, chapter_id: str | None, artifact: dict[str, Any] | None,
        reason: str,
    ) -> None:
        with reuse_plan_header_lock:
            path = checkpoint_dir / "reuse-plan.json"
            plan = read_json(path)
            for entry in plan.get("entries") or []:
                if entry.get("lane") != lane:
                    continue
                if chapter_id is not None and entry.get("chapter_id") != chapter_id:
                    continue
                status = str((artifact or {}).get("reuse_status") or "miss")
                entry.update({
                    "status": status,
                    "artifact_id": (artifact or {}).get("artifact_id"),
                    "reason": reason,
                    "estimated_provider_calls": 0 if artifact is not None else 1,
                })
            plan["estimated_provider_calls"] = sum(
                int(item.get("estimated_provider_calls") or 0)
                for item in plan.get("entries") or []
            )
            write_json(path, plan)
    cut_migration = {"reused": {}, "receipts": []}
    glossary_migration = {
        "accepted": False, "reason": "legacy_checkpoint_not_requested", "value": None,
    }
    migration_report: dict[str, Any] | None = None
    if legacy is not None:
        cut_migration = migrate_legacy_cuts(
            legacy.get("cuts") or (legacy.get("segmentation") or {}).get("cuts") or [],
            blocks=[dict(item) for item in document.get("blocks") or []],
            chapters=chapters_pack["chapters"],
            max_segment_blocks=SEGMENT_HARD_MAX_BLOCKS,
            max_segment_source_chars=SEGMENT_HARD_MAX_SOURCE_CHARS,
        )
        if options.skip_translation:
            glossary_migration = {
                "accepted": False,
                "reason": "glossary_disabled_for_same_language_source",
                "value": None,
            }
        else:
            glossary_migration = migrate_legacy_glossary(
                legacy.get("glossary"), metadata=_legacy_metadata_view(legacy),
                source_hash=migration_source_hash, language=options.annotation_language,
                prompt_hash=migration_prompt_hash, validator_hash=migration_validator_hash,
                index_entries=index_pack,
            )
        migration_report = {
            "schema_version": MIGRATION_VERSION,
            "source_checkpoint_sha256": sha256_json(legacy),
            "read_only_source": True,
            "cuts": cut_migration,
            "glossary": glossary_migration,
            "translations": {
                "ledgers": {},
                "receipts": ([{
                    "status": "skipped",
                    "reason": "translation_disabled_for_same_language_source",
                }] if options.skip_translation else []),
            },
            "never_migrated": list(NEVER_MIGRATED_ARTIFACTS),
        }
        write_json(checkpoint_dir / "legacy-migration.json", migration_report)
    elif accepted_translation_candidates and not options.skip_translation:
        migration_report = {
            "schema_version": MIGRATION_VERSION,
            "source_checkpoint_sha256": sha256_json(accepted_translation_candidates),
            "source_kind": "accepted_artifact_store",
            "read_only_source": True,
            "cuts": cut_migration,
            "glossary": glossary_migration,
            "translations": {"ledgers": {}, "receipts": []},
            "never_migrated": list(NEVER_MIGRATED_ARTIFACTS),
        }
        write_json(checkpoint_dir / "legacy-migration.json", migration_report)
    progress = CompanionProgress()
    supervision_event = threading.Event()

    def cancel_inflight_check() -> bool:
        """Only explicit caller cancellation may interrupt submitted calls."""
        return bool(cancel_check is not None and cancel_check())

    session_manager = None
    submission_limiter = LLMSubmissionLimiter(options.workers)
    if result_llm is not None:
        from arc_llm.sessions import LLMSessionManager
        session_manager = LLMSessionManager(checkpoint_dir / "sessions")
    initial_names = _protected_names(bundle)

    def model(
        prompt, schema, artifact_dir, call_label, tier, *, guided=False,
        prefix_guidance=True, guidance_lane=None, recovery_descriptor=None,
    ):
        with submission_limiter.permit():
            return _llm_call(llm, _guided_prompt(
                                 prompt, intent_guidance, lane=guidance_lane,
                             )
                             if guided and prefix_guidance else prompt,
                             schema, options=options,
                             artifact_dir=artifact_dir, call_label=call_label,
                             model_tier=tier,
                             paper_access_policy=(
                                 _guidance_policy(
                                     intent_guidance, lane=guidance_lane,
                                 ) if guided else None
                             ),
                             intent_guidance=intent_guidance if guided else None,
                             intent_guidance_lane=guidance_lane if guided else None,
                             recovery_descriptor=recovery_descriptor)

    glossary_input_hash = ""
    glossary_recipe_hash = ""
    glossary_object = None
    if options.skip_translation:
        glossary: dict[str, Any] = {}
    else:
        glossary_input_hash = lane_semantic_sha256("glossary", {
            "source": {"source_hash": migration_source_hash},
            "target_language": options.annotation_language,
            "index": index_pack,
            "protected_names": [],
            **({"intent_guidance": guidance_identity} if guidance_identity else {}),
        })
        glossary_recipe_hash = lane_recipe_sha256(
            "glossary", prompt=_language_prompt_contract(PROMPT_VERSION, options),
            model=options.model, tier=GLOSSARY_TIER,
            access_recipe={"provider": options.provider, "allow_internet": False},
        )
        glossary_object = None if "glossary" in regeneration else artifact_store.find(
            kind="glossary", semantic_input_sha256=glossary_input_hash,
            recipe_sha256=glossary_recipe_hash, contract_version="arc.companion.glossary.v1",
            predecessor_accepted_chain_sha256=hashlib.sha256(b"").hexdigest(),
            output_validator=lambda value: isinstance(value, dict) and isinstance(value.get("entries"), list),
        )
        if (
            glossary_object is None and "glossary" not in regeneration
            and guidance_identity is None
        ):
            compatible_glossaries = []
            for record in artifact_store.iter_kind("glossary"):
                value = record.get("output")
                provenance = record.get("provenance")
                if not isinstance(value, dict) or not isinstance(provenance, Mapping):
                    continue
                old_checkpoint = Path(str(provenance.get("checkpoint_dir") or ""))
                old_metadata = _read_checkpoint_json(
                    old_checkpoint / "migration-metadata.json", root=old_checkpoint,
                )
                if not isinstance(old_metadata, dict):
                    continue
                if (
                    old_metadata.get("source_hash") != migration_source_hash
                    or value.get("language") != options.annotation_language
                    or value.get("schema_version") != GLOSSARY_VERSION
                    or not isinstance(value.get("entries"), list)
                ):
                    continue
                try:
                    artifact_store.validate(record)
                except Exception:
                    continue
                compatible_glossaries.append(record)
            if compatible_glossaries:
                selected = max(
                    compatible_glossaries,
                    key=lambda item: float(item.get("created_at") or 0),
                )
                glossary_object = {**selected, "reuse_status": "warning_reuse"}
    if options.skip_translation:
        pass
    elif glossary_object is not None:
        mark_plan_lane(
            "glossary", chapter_id=None, artifact=glossary_object,
            reason="accepted glossary remains valid under the current contract",
        )
        glossary = dict(glossary_object["output"])
        write_json(checkpoint_dir / "glossary.json", glossary)
    elif glossary_migration.get("accepted") and guidance_identity is None:
        glossary = dict(glossary_migration["value"])
    elif index_entries:
        glossary = generate_index_glossary(
            index_entries, language=options.annotation_language,
            checkpoint_dir=checkpoint_dir, force="glossary" in regeneration,
            intent_guidance_identity=guidance_identity,
            call_model=lambda p, s, a, l, recovery_descriptor=None: model(
                p, s, a, l, GLOSSARY_TIER, guided=True,
                guidance_lane="glossary", recovery_descriptor=recovery_descriptor,
            ),
            accept_recovery=_accept_registered_pipeline_control,
        )
    else:
        glossary_document = _augmentation_document(document)
        glossary = (
            _empty_glossary(
                language=options.annotation_language,
                page_count=_page_count(bundle),
            )
            if not glossary_document.get("blocks")
            else generate_glossary(
                glossary_document, language=options.annotation_language,
                source_language=options.source_language,
                protected_names=initial_names, checkpoint_dir=checkpoint_dir,
                workers=options.workers, force="glossary" in regeneration,
                page_count=_page_count(bundle),
                intent_guidance_identity=guidance_identity,
                call_model=lambda p, s, a, l, recovery_descriptor=None: model(
                    p, s, a, l, GLOSSARY_TIER, guided=True,
                    guidance_lane="glossary", recovery_descriptor=recovery_descriptor,
                ),
                accept_recovery=_accept_registered_pipeline_control,
            )
        )
    if (
        not options.skip_translation
        and glossary_object is None
        and isinstance(glossary, dict)
    ):
        _store_validated_stateless_artifact(
            artifact_store, kind="glossary",
            semantic_input_sha256=glossary_input_hash,
            recipe_sha256=glossary_recipe_hash,
            contract_version="arc.companion.glossary.v1", output=glossary,
            segment_id="project:glossary", checkpoint_dir=checkpoint_dir,
            provider=options.provider, model=options.model,
        )
    protected_names = _protected_names(bundle, glossary=glossary)
    title_translations = _generate_title_translations(
        options=options,
        bundle=bundle,
        document=document,
        chapters=(
            list(chapters_pack["chapters"][:1])
            if options.stop_after_first_chapter
            else list(chapters_pack["chapters"])
        ),
        glossary=glossary,
        protected_names=protected_names,
        checkpoint_dir=checkpoint_dir,
        artifact_store=artifact_store,
        intent_guidance_identity=guidance_identity,
        call_model=lambda prompt, schema, artifact_dir, call_label, recovery_descriptor=None: model(
            prompt, schema, artifact_dir, call_label, TITLE_TRANSLATION_TIER,
            guided=True, guidance_lane="title_translation",
            recovery_descriptor=recovery_descriptor,
        ),
    )
    blocks_by_id = {block_id(item): item for item in document.get("blocks") or []}
    reader_publish_lock = threading.RLock()
    reader_data_lock = threading.RLock()
    reader_segments: dict[str, dict[str, Any]] = {}
    reader_guides: dict[str, dict[str, Any]] = {}
    reader_translations: dict[str, dict[str, Any]] = {}
    reader_annotations: dict[str, dict[str, Any]] = {}
    reader_chapter_order = {
        str(item.get("chapter_id") or ""): index
        for index, item in enumerate(chapters_pack["chapters"])
    }

    def chapter_reader_overrides() -> dict[str, Any]:
        with reader_data_lock:
            ordered_segments = sorted(
                reader_segments.values(),
                key=lambda item: (
                    reader_chapter_order.get(
                        str(item.get("chapter_id") or ""), len(reader_chapter_order)
                    ),
                    str(item.get("segment_id") or ""),
                ),
            )
            return {
                "document": document,
                "chapters": list(chapters_pack["chapters"]),
                "segments": ordered_segments,
                "chapter_guides": dict(reader_guides),
                "translations": (
                    None if options.skip_translation else dict(reader_translations)
                ),
                "annotations": dict(reader_annotations),
                "glossary": glossary,
                "metadata": bundle.metadata,
                "language": options.annotation_language,
                "source_language": options.source_language,
                "title_translations": title_translations,
                "translation_mode": (
                    "skipped" if options.skip_translation else "enabled"
                ),
            }

    def segment_glossary_for(segment: dict[str, Any]) -> dict[str, Any]:
        if options.skip_translation:
            return {}
        return project_segment_glossary(
            _augmentation_blocks(segment, blocks_by_id),
            glossary,
        )

    def prepare_segments(chapter):
        chapter_id = str(chapter["chapter_id"])
        chapter_document = {**_generation_document(document), "blocks": [
            blocks_by_id[value] for value in chapter["block_ids"] if value in blocks_by_id
        ]}
        segmentation_input_hash = lane_semantic_sha256("segmentation", {
            "source": {
                "chapter_id": chapter_id,
                "ordered_block_ids": [
                    block_id(item) for item in chapter_document["blocks"]
                ],
            },
            "chapter": {"chapter_id": chapter_id},
            "limits": {
                "max_blocks": SEGMENT_HARD_MAX_BLOCKS,
                "max_source_chars": SEGMENT_HARD_MAX_SOURCE_CHARS,
            },
        })
        segmentation_recipe_hash = lane_recipe_sha256(
            "segmentation", prompt=PROMPT_VERSION, model=options.model,
            tier=SEGMENTATION_TIER,
            access_recipe={"provider": options.provider, "allow_internet": False},
        )
        segmentation_object = None if "segmentation" in regeneration else artifact_store.find(
            kind="segmentation", semantic_input_sha256=segmentation_input_hash,
            recipe_sha256=segmentation_recipe_hash,
            contract_version="arc.companion.segmentation.v1",
            predecessor_accepted_chain_sha256=hashlib.sha256(b"").hexdigest(),
            output_validator=lambda value: isinstance(value, list) and bool(value),
        )
        if segmentation_object is None and "segmentation" not in regeneration:
            compatible = []
            expected_ids = [block_id(item) for item in chapter_document["blocks"]]
            for record in artifact_store.iter_kind("segmentation"):
                raw_output = record.get("output")
                if not isinstance(raw_output, list) or not raw_output:
                    continue
                actual_ids = [
                    str(identifier)
                    for item in raw_output if isinstance(item, Mapping)
                    for identifier in item.get("block_ids") or []
                ]
                if actual_ids != expected_ids:
                    continue
                cuts: list[int] = []
                cursor = 0
                for item in raw_output[:-1]:
                    cursor += len(item.get("block_ids") or [])
                    cuts.append(cursor)
                try:
                    augmented = construct_segments_from_cuts(cuts, chapter_document)
                except SegmentationError:
                    continue
                compatible.append((float(record.get("created_at") or 0), record, augmented))
            if compatible:
                _created, selected, augmented = max(compatible, key=lambda item: item[0])
                segmentation_object = {
                    **selected,
                    "output": augmented,
                    "reuse_status": "hit",
                }
        if segmentation_object is not None:
            mark_plan_lane(
                "segmentation", chapter_id=chapter_id, artifact=segmentation_object,
                reason="accepted segmentation remains valid under the current contract",
            )
            raw = [dict(item) for item in segmentation_object["output"]]
        else:
            raw = segment_document(
                chapter_document, checkpoint_dir=checkpoint_dir / "chapters" / chapter_id,
                workers=options.workers, force="segmentation" in regeneration,
                call_model=lambda p, s, a, l, recovery_descriptor=None: model(
                    p, s, a, l, SEGMENTATION_TIER,
                    recovery_descriptor=recovery_descriptor,
                ),
                accept_recovery=_accept_registered_pipeline_control,
                seed_cuts=(
                    list(cut_migration.get("reused", {}).get(chapter_id) or [])
                    if chapter_id in cut_migration.get("reused", {}) else None
                ),
            )
            validate_exact_coverage(raw, chapter_document["blocks"])
            _store_validated_stateless_artifact(
                artifact_store, kind="segmentation",
                semantic_input_sha256=segmentation_input_hash,
                recipe_sha256=segmentation_recipe_hash,
                contract_version="arc.companion.segmentation.v1", output=raw,
                segment_id=f"{chapter_id}:segmentation", checkpoint_dir=checkpoint_dir,
                provider=options.provider, model=options.model,
            )
        segments = [{**item, "chapter_id": chapter_id,
                     "segment_id": f"{chapter_id}.seg-{index:04d}"}
                    for index, item in enumerate(raw, 1)]
        for requested_lane, ledger_lane in (("translation", "translation"), ("commentary", "companion")):
            chapter_targets = [
                str(item["segment_id"]) for item in segments
                if f"{requested_lane}:{item['segment_id']}" in targeted_regeneration
            ]
            if (
                requested_lane not in regeneration and not chapter_targets
            ) or (requested_lane == "translation" and options.skip_translation):
                continue
            ledger_path = checkpoint_dir / "chapters" / chapter_id / f"{ledger_lane}-ledger.json"
            if not ledger_path.is_file():
                # A precise target on a fresh lane still has valid semantics:
                # run_lane will force that segment while ordinary cache rules
                # apply to the rest. Do not report the requested segment as
                # missing merely because there was no suffix ledger to stage.
                if chapter_targets:
                    with targeted_lock:
                        targeted_seen.update(
                            f"{requested_lane}:{value}"
                            for value in chapter_targets
                        )
                continue
            ledger = initialize_lane_ledger(
                ledger_path, chapter_id=chapter_id, lane=ledger_lane,
                segment_ids=[str(item["segment_id"]) for item in segments],
            )
            generation = int(ledger.get("generation") or 1) + 1
            start_segment_id = (
                str(segments[0]["segment_id"])
                if requested_lane in regeneration else chapter_targets[0]
            )
            staged_outputs: dict[str, dict[str, Any]] = {}
            if chapter_targets:
                start_index = next(
                    index for index, item in enumerate(ledger["blocks"])
                    if str(item["segment_id"]) == start_segment_id
                )
                for old_block in ledger["blocks"][start_index:]:
                    old_segment_id = str(old_block.get("segment_id") or "")
                    if (
                        old_segment_id in chapter_targets
                        or old_block.get("state") != "accepted"
                    ):
                        continue
                    output = old_block.get("translation")
                    if not isinstance(output, dict):
                        artifact_kind = (
                            "translations" if requested_lane == "translation"
                            else "annotations"
                        )
                        artifact_dir = _generation_segment_artifact_dir(
                            checkpoint_dir, artifact_kind, old_segment_id,
                            int(old_block.get("generation") or ledger.get("generation") or 1),
                        )
                        payload = _read_checkpoint_json(
                            artifact_dir
                            / f"{_segment_checkpoint_name(old_segment_id)}.json",
                            root=checkpoint_dir,
                        )
                        payload_key = (
                            "translation" if requested_lane == "translation"
                            else "annotation"
                        )
                        output = (
                            payload.get(payload_key)
                            if isinstance(payload, dict) else None
                        )
                    if (
                        isinstance(output, dict)
                        and sha256_json(output) == old_block.get("output_sha256")
                    ):
                        object_kind = (
                            "translation"
                            if requested_lane == "translation" else "commentary"
                        )
                        direct_artifact_id = artifact_id_for(
                            kind=object_kind,
                            semantic_input_sha256=str(
                                old_block.get("input_sha256") or ""
                            ),
                            output_sha256=str(
                                old_block.get("output_sha256") or ""
                            ),
                            contract_version=SCHEMA_VERSION,
                            predecessor_accepted_chain_sha256=str(
                                old_block.get(
                                    "predecessor_accepted_chain_sha256"
                                ) or ""
                            ),
                        )
                        source_artifact_id = None
                        if artifact_store.path_for(
                            object_kind, direct_artifact_id,
                        ).is_file():
                            source_artifact_id = direct_artifact_id
                        else:
                            logical_receipt = old_block.get("logical_receipt")
                            referenced_id = (
                                str(logical_receipt.get("artifact_id") or "")
                                if isinstance(logical_receipt, Mapping) else ""
                            )
                            if referenced_id:
                                try:
                                    referenced_path = artifact_store.path_for(
                                        object_kind, referenced_id,
                                    )
                                except Exception:
                                    referenced_path = None
                                if referenced_path is not None and referenced_path.is_file():
                                    source_artifact_id = referenced_id
                        staged_outputs[old_segment_id] = {
                            "output": output,
                            "output_sha256": old_block.get("output_sha256"),
                            "logical_receipt": {
                                "kind": "targeted_regeneration_suffix_stage",
                                "provider_calls": 0,
                                "source_generation": int(
                                    old_block.get("generation")
                                    or ledger.get("generation")
                                    or 1
                                ),
                                "source_logical_receipt": dict(
                                    old_block.get("logical_receipt") or {}
                                ),
                                **(
                                    {"source_artifact_id": source_artifact_id}
                                    if source_artifact_id else {}
                                ),
                            },
                            "validation_receipt": {
                                "staged_before_targeted_invalidation": True,
                                "source_validation_receipt": dict(
                                    old_block.get("validation_receipt") or {}
                                ),
                            },
                        }
            invalidate_suffix(
                ledger_path, from_segment_id=start_segment_id, generation=generation,
                staged_outputs=staged_outputs,
            )
            if chapter_targets:
                start = next(
                    index for index, item in enumerate(segments)
                    if str(item["segment_id"]) == start_segment_id
                )
                with targeted_lock:
                    targeted_seen.update(
                        f"{requested_lane}:{value}" for value in chapter_targets
                    )
                    targeted_suffix_reuse.update(
                        f"{requested_lane}:{item['segment_id']}"
                        for item in segments[start:]
                        if str(item["segment_id"]) not in chapter_targets
                    )
            if session_manager is not None:
                session_key = f"{chapter_id}:{ledger_lane}"
                ref = session_manager.get_existing(session_key)
                while ref is not None and ref.generation < generation:
                    ref = session_manager.rotate(
                        session_key,
                        reason=f"explicit-{requested_lane}-regeneration",
                    )
                if ref is not None and ref.generation != generation:
                    raise StatefulSessionError(
                        f"{session_key} generation changed outside explicit regeneration"
                    )
        with reader_data_lock:
            reader_segments.update({str(item["segment_id"]): dict(item) for item in segments})
            reader_segment_count = len(reader_segments)
        with reader_publish_lock:
            _state(
                state_path, status="active", paper_id=bundle.paper_id,
                fingerprint=fingerprint, checkpoint_dir=str(checkpoint_dir),
                segment_count=reader_segment_count, notice=notice,
            )
            _publish_reader_update(
                options.project_dir.resolve(), state_path, reader_publish_lock,
                final_overrides=chapter_reader_overrides(),
            )
        translation_candidates = (
            legacy_translation_candidates(legacy)
            if legacy is not None else accepted_translation_candidates
        )
        if (
            translation_candidates and not options.skip_translation
            and guidance_identity is None
        ):
            translation_migration = migrate_legacy_translations(
                translation_candidates,
                metadata=(
                    _legacy_metadata_view(legacy)
                    if legacy is not None else {
                        "source_hash": migration_source_hash,
                        "language": options.annotation_language,
                    }
                ),
                blocks=[dict(item) for item in document.get("blocks") or []],
                chapters=[chapter], segments=segments,
                source_hash=migration_source_hash, language=options.annotation_language,
                glossary=glossary, protected_names=protected_names,
                segment_input_hash=lambda item: _segment_input_hash(
                    dict(item), blocks_by_id, glossary=segment_glossary_for(dict(item)),
                ),
                migration_source=(
                    "legacy_checkpoint" if legacy is not None
                    else "accepted_artifact_store"
                ),
            )
            migrated_ledger = translation_migration["ledgers"].get(chapter_id)
            ledger_path = (
                checkpoint_dir / "chapters" / chapter_id / "translation-ledger.json"
            )
            with migration_lock:
                if migrated_ledger is not None and not ledger_path.exists():
                    write_json(ledger_path, migrated_ledger)
                assert migration_report is not None
                migration_report["translations"]["ledgers"][chapter_id] = (
                    migrated_ledger or {}
                )
                migration_report["translations"]["receipts"].extend(
                    translation_migration.get("receipts") or []
                )
                write_json(checkpoint_dir / "legacy-migration.json", migration_report)
        chapter_segments_ready[chapter_id].set()
        return segments

    def prepare_guide(chapter):
        chapter_id = str(chapter["chapter_id"])
        guide_segment_id = f"{chapter_id}:guide"
        guide_ledger_path = checkpoint_dir / "chapters" / chapter_id / "guide-ledger.json"
        if not chapter.get("content_block_ids"):
            guide = {
                "schema_version": CHAPTER_GUIDE_VERSION,
                "source_sha256": sha256_json({
                    "chapter_projection_version": CHAPTER_PROJECTION_VERSION,
                    "chapter_id": chapter_id,
                    "content_block_ids": [],
                    "target_language": options.annotation_language,
                }),
                "chapter_id": chapter_id,
                "motivation": None,
                "main_content": None,
                "section_logic": None,
                "prerequisites": None,
                "pedagogical_comparison": None,
                "historical_context": [],
                "supplementary_reading": [],
            }
            guide_ledger = initialize_lane_ledger(
                guide_ledger_path, chapter_id=chapter_id, lane="guide",
                segment_ids=[guide_segment_id],
            )
            if guide_ledger["blocks"][0]["state"] != "accepted":
                accept_controller_skipped_block(
                    guide_ledger_path, segment_id=guide_segment_id,
                    input_sha256=str(guide["source_sha256"]),
                    output_sha256=sha256_json(guide),
                    reason="chapter contains structural headings only",
                )
            mark_plan_lane(
                "guide", chapter_id=chapter_id, artifact={"reuse_status": "skipped"},
                reason="controller_skipped_structural_heading",
            )
            with reader_data_lock:
                reader_guides[chapter_id] = dict(guide)
            return guide
        guide_input_hash = lane_semantic_sha256("guide", {
            "chapter_source": {
                "chapter": _guide_chapter_descriptor(chapter),
                "blocks": [
                    blocks_by_id[value] for value in chapter.get("content_block_ids") or []
                    if value in blocks_by_id
                ],
            },
            "target_language": options.annotation_language,
            "verified_evidence": evidence,
            **({"intent_guidance": guidance_identity} if guidance_identity else {}),
        })
        guide_recipe_hash = lane_recipe_sha256(
            "guide", prompt=CHAPTER_GUIDE_VERSION, model=options.model,
            tier=ANNOTATION_TIER,
            access_recipe={
                "provider": options.provider,
                "allow_internet": options.allow_internet,
                "inherit_host_tools": (
                    options.inherit_host_tools if intent_guidance is None else False
                ),
            },
        )
        guide_ledger = None
        if result_llm is not None:
            guide_ledger = initialize_lane_ledger(
                guide_ledger_path, chapter_id=chapter_id, lane="guide",
                segment_ids=[guide_segment_id],
            )
            accepted_guide = guide_ledger["blocks"][0]
            accepted_validation = accepted_guide.get("validation_receipt")
            accepted_recipe_hash = str(
                accepted_validation.get("recipe_sha256")
                if isinstance(accepted_validation, Mapping) else ""
            )
            recipe_changed = (
                accepted_guide["state"] == "accepted"
                and accepted_recipe_hash != guide_recipe_hash
            )
            if (
                accepted_guide["state"] == "accepted"
                and (
                    accepted_guide.get("input_sha256") != guide_input_hash
                    or recipe_changed
                )
            ):
                guide_ledger = invalidate_suffix(
                    guide_ledger_path, from_segment_id=guide_segment_id,
                    generation=int(guide_ledger.get("generation") or 1) + 1,
                )
                if session_manager is not None:
                    session_key = f"{chapter_id}:guide"
                    if session_manager.get_existing(session_key) is not None:
                        session_manager.rotate(
                            session_key,
                            reason=(
                                "guide-recipe-change"
                                if recipe_changed else "guide-semantic-input-change"
                            ),
                        )
            if "guide" in regeneration and guide_ledger["blocks"][0]["state"] == "accepted":
                guide_generation = int(guide_ledger.get("generation") or 1) + 1
                guide_ledger = invalidate_suffix(
                    guide_ledger_path, from_segment_id=guide_segment_id,
                    generation=guide_generation,
                )
                if session_manager is not None:
                    session_key = f"{chapter_id}:guide"
                    if session_manager.get_existing(session_key) is not None:
                        session_manager.rotate(session_key, reason="explicit-guide-regeneration")
            if guide_ledger["blocks"][0]["state"] == "accepted":
                accepted_block = guide_ledger["blocks"][0]
                accepted_path = (
                    checkpoint_dir / "chapters" / chapter_id / "chapter-guide.json"
                )
                accepted_guide = _read_checkpoint_json(
                    accepted_path, root=checkpoint_dir,
                )
                if (
                    isinstance(accepted_guide, Mapping)
                    and sha256_json(accepted_guide)
                    == accepted_block.get("output_sha256")
                    and chapter_guide_artifact_valid(
                        accepted_guide,
                        evidence=evidence,
                        allow_internet=options.allow_internet,
                    )
                ):
                    guide = dict(accepted_guide)
                    mark_plan_lane(
                        "guide", chapter_id=chapter_id,
                        artifact={"reuse_status": "hit"},
                        reason=(
                            "accepted guide checkpoint passed current local validation"
                        ),
                    )
                    with reader_data_lock:
                        reader_guides[chapter_id] = dict(guide)
                    return guide
                raise RuntimeError(
                    "accepted guide checkpoint is missing or fails local validation"
                )
            if guide_ledger["blocks"][0]["state"] == "prepared" and "guide" not in regeneration:
                guide_object = artifact_store.find(
                    kind="guide", semantic_input_sha256=guide_input_hash,
                    recipe_sha256=guide_recipe_hash,
                    contract_version=CHAPTER_GUIDE_VERSION,
                    predecessor_accepted_chain_sha256=str(guide_ledger.get("accepted_chain_sha256") or ""),
                    output_validator=lambda value: (
                        chapter_guide_artifact_valid(
                            value,
                            evidence=evidence,
                            allow_internet=options.allow_internet,
                        )
                        and value.get("chapter_id") == chapter_id
                    ),
                )
                if guide_object is not None:
                    mark_plan_lane(
                        "guide", chapter_id=chapter_id, artifact=guide_object,
                        reason="accepted guide remains valid under the current contract",
                    )
                    guide = dict(guide_object["output"])
                    write_json(
                        checkpoint_dir / "chapters" / chapter_id / "chapter-guide.json",
                        guide,
                    )
                    accept_reused_block(
                        guide_ledger_path, segment_id=guide_segment_id,
                        input_sha256=guide_input_hash,
                        output_sha256=str(guide_object["output_sha256"]),
                        artifact_id=str(guide_object["artifact_id"]),
                        validation_receipt={
                            "local_validation": True, "object_store_revalidated": True,
                            "reuse_status": guide_object["reuse_status"],
                            "recipe_sha256": guide_recipe_hash,
                        },
                    )
                    with reader_data_lock:
                        reader_guides[chapter_id] = dict(guide)
                    return guide
        guide_receipt: dict[str, Any] = {}
        guide_provider_receipt: dict[str, Any] = {}
        last_guide_call: dict[str, str] = {}
        last_guide_submission_receipt: Path | None = None
        guide_policy = _guidance_policy(intent_guidance, lane="guide")
        guide_structured_policy = bool(
            guide_policy is not None
            and result_llm is not None
            and _accepts_explicit_keyword(result_llm, "paper_access_policy")
        )
        def guide_model(prompt, schema, artifact_dir, call_label):
            if result_llm is None or session_manager is None:
                return model(
                    prompt, schema, artifact_dir, call_label, ANNOTATION_TIER,
                    guided=True, prefix_guidance=False, guidance_lane="guide",
                    recovery_descriptor=submission_descriptor(
                        unit="guide",
                        logical_unit=guide_segment_id,
                        checkpoint_dir=checkpoint_dir,
                        artifact_root=artifact_dir,
                        acceptance_checkpoint=(
                            checkpoint_dir / "chapters" / chapter_id
                            / "chapter-guide.json"
                        ),
                        input_sha256=guide_input_hash,
                        ordered_siblings=[guide_segment_id],
                        suffix=[guide_segment_id],
                    ),
                )
            session_key = f"{chapter_id}:guide"
            existing_session = session_manager.get_existing(session_key)
            registered_guide, _registered_guide_digest = read_registered_lane_ledger(
                checkpoint_dir, guide_ledger_path,
            )
            registered_guide_block = next(
                item for item in registered_guide.get("blocks") or []
                if item.get("segment_id") == guide_segment_id
            )
            generation = int(
                registered_guide_block.get("generation")
                or registered_guide.get("generation")
                or 1
            )
            while existing_session is not None and existing_session.generation < generation:
                existing_session = session_manager.rotate(
                    session_key, reason="align guide ledger generation",
                )
            if existing_session is not None and existing_session.generation != generation:
                raise StatefulSessionError(
                    f"{session_key} generation changed outside guide control"
                )
            idempotency_key = f"{chapter_id}:guide:{call_label}:generation-{generation}"
            current_idempotency_key = idempotency_key
            last_guide_call.update({
                "idempotency_key": idempotency_key, "artifact_dir": str(artifact_dir),
                "generation": str(generation),
            })
            def invoke(
                active_prompt: str, active_schema: dict[str, Any], evidence_round: int,
            ) -> Any:
                nonlocal current_idempotency_key, last_guide_submission_receipt
                round_suffix = "" if evidence_round == 0 else f":evidence-{evidence_round:02d}"
                current_idempotency_key = idempotency_key + round_suffix
                active_label = (
                    call_label if evidence_round == 0
                    else f"{call_label}-evidence-{evidence_round:02d}"
                )
                active_artifact_dir = (
                    artifact_dir if evidence_round == 0
                    else artifact_dir / f"evidence-round-{evidence_round:02d}"
                )
                current_session = session_manager.get_existing(session_key)
                last_guide_call.update({
                    "idempotency_key": current_idempotency_key,
                    "artifact_dir": str(active_artifact_dir),
                })
                last_guide_submission_receipt = write_ledger_submission_receipt(
                    checkpoint_dir=checkpoint_dir,
                    artifact_dir=active_artifact_dir,
                    ledger_path=guide_ledger_path,
                    session_key=session_key,
                    logical_unit=guide_segment_id,
                    generation=generation,
                    idempotency_key=current_idempotency_key,
                    schema=active_schema,
                    prompt=active_prompt,
                    recovery_unit="guide",
                    input_sha256=guide_input_hash,
                    ordered_siblings=[guide_segment_id],
                    suffix=[guide_segment_id],
                    validator="chapter-guide-schema+invariants.v1",
                    application="normal-guide-pipeline-replay.v1",
                )
                native_resume_authorization = _supervised_native_resume_authorized(
                    options,
                    ledger_path=guide_ledger_path,
                    session_key=session_key,
                    logical_unit=guide_segment_id,
                    generation=generation,
                    idempotency_key=current_idempotency_key,
                )
                with submission_limiter.permit():
                    return result_llm(
                        active_prompt, schema=active_schema,
                        provider=options.provider, model=options.model,
                        model_tier=None if options.model else ANNOTATION_TIER,
                        env=_llm_runtime_env(
                            allow_internet=options.allow_internet,
                            force_disable_internet=not options.allow_internet,
                            inherit_host_tools=(
                                options.inherit_host_tools
                                if guide_policy is None else False
                            ),
                            paper_access_policy=guide_policy,
                            serialize_paper_access_policy=not guide_structured_policy,
                        ),
                        artifact_dir=active_artifact_dir, call_label=active_label,
                        idle_timeout_seconds=options.idle_timeout_seconds,
                        session_policy="stateful", session_manager=session_manager,
                        session_key=session_key,
                        idempotency_key=current_idempotency_key,
                        progress_contract_scope="session",
                        initial_native_authorization=(
                            str(guide_ledger_path.resolve(strict=False)),
                            session_key,
                            guide_segment_id,
                            generation,
                            current_idempotency_key,
                        ),
                        supervised_native_resume=native_resume_authorization,
                        validated_legacy_logical_identity=(
                            {
                                "provider": current_session.provider,
                                "model": current_session.model,
                                "session_key": session_key,
                                "generation": generation,
                                "idempotency_key": current_idempotency_key,
                            }
                            if native_resume_authorization is not None
                            and current_session is not None
                            else None
                        ),
                        validated_legacy_runtime_identity=(
                            {
                                "session_key": current_session.key,
                                "provider": current_session.provider,
                                "model": current_session.model,
                                "generation": current_session.generation,
                                "native_session_id": current_session.native_session_id,
                                "recorded_fp": current_session.runtime_fingerprint,
                            }
                            if native_resume_authorization is not None
                            and current_session is not None
                            else None
                        ),
                        cancel_check=cancel_inflight_check,
                        progress_callback=lambda event: (
                            _guarded_mark_transport_state(
                                guide_ledger_path,
                                checkpoint_dir=checkpoint_dir,
                                session_key=session_key,
                                logical_unit=guide_segment_id,
                                idempotency_key=idempotency_key,
                            )
                            if event.get("event") == "submitted"
                            else None,
                            progress.provider_event(event),
                        )[-1],
                        **(
                            {"paper_access_policy": guide_policy}
                            if guide_structured_policy else {}
                        ),
                    )
            try:
                outcome = invoke(
                    prompt, _intent_guidance_schema(schema, intent_guidance), 0,
                )
                evidence_source_generation, evidence_source_logical_task = (
                    _companion_evidence_source_identity(
                        project_dir=options.project_dir,
                        checkpoint_dir=checkpoint_dir,
                        run_id=fingerprint,
                        ledger_path=guide_ledger_path,
                        ledger=registered_guide,
                        session_key=session_key,
                        logical_unit=guide_segment_id,
                        target_generation=generation,
                        target_idempotency_key=idempotency_key,
                    )
                )
                outcome, final_value = _complete_stateful_reference_evidence(
                    outcome, intent_guidance=intent_guidance, lane="guide",
                    worker_id=call_label, schema=schema, call_round=invoke,
                    journal_context=_companion_evidence_journal_context(
                        checkpoint_dir=checkpoint_dir,
                        run_id=fingerprint,
                        lane="guide",
                        worker_id=call_label,
                        logical_task_id=evidence_source_logical_task,
                        source_generation=evidence_source_generation,
                        policy=guide_policy,
                        runtime={
                            "runtime_fingerprint": str(
                                getattr(outcome, "runtime_fingerprint", "")
                            ),
                            "provider": options.provider,
                            "model": options.model,
                            "model_tier": ANNOTATION_TIER,
                        },
                    ),
                    target_session=session_key,
                    target_generation=generation,
                    followup_id=idempotency_key,
                )
            except BaseException as exc:
                if _chapter_failure_requires_supervision(exc):
                    from arc_llm import read_recovery_context
                    recovery = read_recovery_context(
                        Path(last_guide_call["artifact_dir"]),
                        idempotency_key=current_idempotency_key,
                        session_manager=session_manager, session_key=session_key,
                    )
                    recovery_context = _recovery_context_json(
                        recovery, logical_unit=guide_segment_id,
                    )
                    if last_guide_submission_receipt is not None:
                        try:
                            seal_submission_attempts(
                                last_guide_submission_receipt,
                                checkpoint_dir=checkpoint_dir,
                                attempt_references=explicit_attempt_references(
                                    exc,
                                    checkpoint_dir=checkpoint_dir,
                                    artifact_dir=Path(last_guide_call["artifact_dir"]),
                                ),
                            )
                            recovery_context["submission_receipt"] = (
                                submission_receipt_reference(
                                    last_guide_submission_receipt,
                                    checkpoint_dir=checkpoint_dir,
                                )
                            )
                        except RecoveryResponseError as receipt_exc:
                            recovery_context["submission_receipt_error"] = str(receipt_exc)
                    mark_needs_supervision(
                        guide_ledger_path, segment_id=guide_segment_id, reason=str(exc),
                        recovery_context=recovery_context,
                    )
                    supervision_event.set()
                raise
            guide_receipt.clear()
            guide_receipt.update(dict(outcome.logical_receipt or {}))
            usage = getattr(outcome, "usage", {})
            usage_json = usage.to_json() if hasattr(usage, "to_json") else usage
            guide_provider_receipt.update({
                "provider": str(getattr(outcome, "provider", None) or options.provider),
                "model": str(getattr(outcome, "model", None) or options.model or "provider-default"),
                "call_id": str(guide_receipt.get("idempotency_key") or guide_receipt.get("call_id") or call_label),
                "usage": dict(usage_json) if isinstance(usage_json, dict) else {},
            })
            _guarded_mark_transport_state(
                guide_ledger_path,
                checkpoint_dir=checkpoint_dir,
                session_key=session_key,
                logical_unit=guide_segment_id,
                idempotency_key=idempotency_key,
                response_received=True,
            )
            return final_value
        try:
            guide = generate_chapter_guide(
                _guide_chapter_descriptor(chapter), [
                    blocks_by_id[value] for value in chapter.get("content_block_ids") or []
                    if value in blocks_by_id
                ],
                language=options.annotation_language, evidence=evidence,
                checkpoint_dir=checkpoint_dir / "chapters" / chapter_id,
                force="guide" in regeneration, call_model=guide_model,
                stateful=result_llm is not None,
                allow_internet=options.allow_internet,
                inherit_host_tools=(
                    options.inherit_host_tools if guide_policy is None else False
                ),
                recipe_identity=guide_recipe_hash,
                intent_guidance=intent_guidance,
            )
        except StatefulSessionError as exc:
            mark_needs_supervision(
                guide_ledger_path, segment_id=guide_segment_id, reason=str(exc),
                recovery_context={
                    **last_guide_call, "submission_state": "response_received",
                    "session_key": f"{chapter_id}:guide", "resumable": True,
                },
            )
            supervision_event.set()
            raise
        if result_llm is None and _pipeline_control_receipt_exists(
            checkpoint_dir, "guide", guide_segment_id,
        ):
            _accept_registered_pipeline_control(
                checkpoint_dir, "guide", guide_segment_id,
            )
        if result_llm is not None and guide_ledger is not None:
            current = initialize_lane_ledger(
                guide_ledger_path, chapter_id=chapter_id, lane="guide",
                segment_ids=[guide_segment_id],
            )["blocks"][0]["state"]
            if current != "accepted":
                advance_block(guide_ledger_path, segment_id=guide_segment_id, state="schema_valid")
                advance_block(guide_ledger_path, segment_id=guide_segment_id, state="invariant_valid")
                accepted_guide_ledger = advance_block(
                    guide_ledger_path, segment_id=guide_segment_id, state="accepted",
                    receipt=guide_receipt, input_sha256=guide_input_hash,
                    output_sha256=sha256_json(guide),
                    validation_receipt={
                        "local_validation": True,
                        "recipe_sha256": guide_recipe_hash,
                    },
                )
                accepted_guide_block = next(
                    item for item in accepted_guide_ledger["blocks"]
                    if item.get("segment_id") == guide_segment_id
                )
                if guide_provider_receipt and accepted_guide_block.get("logical_receipt"):
                    artifact_store.put_accepted(
                        kind="guide", semantic_input_sha256=guide_input_hash,
                        recipe_sha256=guide_recipe_hash,
                        contract_version=CHAPTER_GUIDE_VERSION, output=guide,
                        ledger_block=accepted_guide_block,
                        provider_receipt=guide_provider_receipt,
                        provenance={"checkpoint_dir": str(checkpoint_dir), "chapter_id": chapter_id},
                    )
        with reader_data_lock:
            reader_guides[chapter_id] = dict(guide)
        return guide

    ledger_paths: dict[tuple[str, str], Path] = {}
    logical_receipts: dict[str, dict[str, Any]] = {}
    provider_receipts: dict[str, dict[str, Any]] = {}
    reuse_plan_lock = reuse_plan_header_lock
    lane_identity_lock = threading.Lock()
    prepared_reported: set[str] = set()
    prepared_report_lock = threading.Lock()
    stream_lock = threading.Lock()
    prompt_streams: dict[tuple[str, str, int], StatefulPromptStream] = {}
    rollover_budgets: dict[tuple[str, str, int], ContextRolloverBudget] = {}
    lane_runtime_profiles: dict[tuple[str, str, int], dict[str, Any]] = {}
    correction_budget = CorrectionBudget()

    def record_segment_reuse(
        *, chapter_id: str, segment_id: str, lane: str,
        status: str, artifact_id: str | None, reason: str,
    ) -> None:
        with reuse_plan_lock:
            plan_path = checkpoint_dir / "reuse-plan.json"
            plan = read_json(plan_path)
            entries = list(plan.get("entries") or [])
            entries.append({
                "chapter_id": chapter_id, "segment_id": segment_id,
                "lane": "commentary" if lane == "companion" else lane,
                "status": status, "artifact_id": artifact_id, "reason": reason,
                "estimated_provider_calls": 0 if status in {
                    "hit", "composed_hit", "deferred_hit", "warning_reuse",
                    "recipe_stale", "skipped",
                } else 1,
            })
            plan["entries"] = entries
            plan["estimated_provider_calls"] = sum(
                int(item.get("estimated_provider_calls") or 0) for item in entries
                if item.get("segment_id") is not None
            )
            write_json(plan_path, plan)

    def lane_stream(prepared, lane: str, generation: int) -> StatefulPromptStream:
        key = (str(prepared.chapter["chapter_id"]), lane, generation)
        with stream_lock:
            session_key = f"{key[0]}:{lane}"
            ref = session_manager.get_existing(session_key)
            existing_generation = bool(
                ref is not None and ref.generation == generation and (
                    ref.native_session_id
                    or session_manager.turn_count(session_key, generation=generation) > 0
                    or ref.metadata.get("arc_runtime_started_generation") == generation
                )
            )
            lane_runtime_profiles[key] = resolve_lane_runtime_profile(
                checkpoint_dir / "chapters" / key[0]
                / f"{lane}-runtime-generation-{generation}.json",
                chapter_id=key[0], lane=lane, generation=generation,
                requested_allow_internet=options.allow_internet,
                inherit_host_tools=(
                    options.inherit_host_tools if guidance_policy is None else False
                ),
                existing_generation=existing_generation,
                # rotate() deliberately retains the previous generation's
                # provider identity as a selection hint, but its runtime
                # manifest does not belong to the new generation.
                recorded_runtime_fingerprint=(
                    ref.runtime_fingerprint if existing_generation else None
                ),
                provider=options.provider, model=options.model,
                model_tier=(TRANSLATION_TIER if lane == "translation" else ANNOTATION_TIER),
            )
            stream = prompt_streams.get(key)
            if stream is None:
                state_path = (
                    checkpoint_dir / "chapters" / key[0]
                    / f"{lane}-stream-generation-{generation}.json"
                )
                receipt_turns = session_manager.turn_count(
                    f"{key[0]}:{lane}", generation=generation,
                )
                restored = read_stream_state(
                    state_path, receipt_turn_count=receipt_turns,
                )
                if restored is not None:
                    stream, _persisted_budget = restored
                    if (stream.chapter_id, stream.lane, stream.generation) != key:
                        raise StatefulSessionError(
                            f"stateful stream identity changed for {key[0]}:{lane}:{generation}"
                        )
                    rollover_budgets[key] = ContextRolloverBudget.from_turn_records(
                        session_manager.turn_records(
                            f"{key[0]}:{lane}", generation=generation,
                        )
                    )
                    prompt_streams[key] = stream
                    return stream
                capsule = None
                if generation > 1:
                    ledger_path = (
                        checkpoint_dir / "chapters" / key[0] / f"{lane}-ledger.json"
                    )
                    if ledger_path.exists():
                        prior = [
                            block for block in (read_json(ledger_path).get("blocks") or [])
                            if block.get("state") == "accepted"
                            and int(block.get("generation") or 1) < generation
                        ]
                        if prior:
                            block = prior[-1]
                            capsule = continuity_capsule(
                                accepted_chain_sha256=str(block.get("accepted_chain_sha256") or ""),
                                segment_id=str(block.get("segment_id") or ""),
                                input_sha256=str(block.get("input_sha256") or ""),
                                output_sha256=str(block.get("output_sha256") or ""),
                            )
                stream = StatefulPromptStream(
                    chapter_id=key[0], lane=lane, generation=generation,
                    fixed_rules={
                        "source_is_immutable": True,
                        "advance_only_after_local_validation": True,
                        "preserve_equations_names_and_opaque_tokens": True,
                        "task_contract": (
                            "Translate every supplied natural-language block exactly once in source order; "
                            "use the glossary mapping, preserve protected names and opaque tokens byte-for-byte."
                            if lane == "translation" else
                            "Write rigorous optional reader commentary that adds reasoning rather than paraphrase; "
                            "use this native session's chapter history to avoid unnecessary repetition; search and "
                            "inspect sources in the same turn when useful, then return direct title/URL/locator citations."
                        ),
                        "target_language": options.annotation_language,
                        **({
                            "reference_access": (
                                "At bootstrap, read only reference_targets applicable to this lane. "
                                "Use arc-paper-worker policy-targets for a non-inline catalog, then "
                                "get-parsed-toc/get-parsed-section and artifact-read pagination when "
                                "a sandboxed shell is available. Otherwise return "
                                "arc_evidence_requests using list-reference-targets and the same "
                                "read-only operations; never parse, refresh, fetch, or use an "
                                "unauthorized source."
                            ),
                            "reference_authority": (
                                "The original source is authoritative for facts, coverage, and structure. "
                                "A reference translation may influence terminology, idiom, and style only; "
                                "never inherit its additions, omissions, or errors."
                            ),
                        } if intent_guidance is not None else {}),
                    },
                    static_context={
                        "chapter": _compact_chapter_descriptor(prepared.chapter),
                        "chapter_guide": prepared.guide,
                        "lane_instructions": (
                            translation_prompt(
                                {}, [], language=options.annotation_language,
                                glossary={}, protected_names=protected_names, paper_context={},
                                source_language=_multilingual_prompt_source(options),
                            )
                            if lane == "translation" else
                            annotation_prompt(
                                {}, [], language=options.annotation_language,
                                metadata=_annotation_metadata(bundle.metadata), evidence={},
                                glossary={}, protected_names=protected_names, paper_context={},
                                domain_context=domain_context,
                                source_language=_multilingual_prompt_source(options),
                            ).replace("\n\nGLOSSARY:\n{}\n\n", "\n\n")
                        ),
                        "paper": _static_paper_context(
                            _full_paper_context(
                                document, prepared.segments[0], blocks_by_id=blocks_by_id,
                                options=options,
                            )
                        ),
                        **(
                            {"intent_guidance": worker_guidance_payload(
                                intent_guidance,
                                lane=("commentary" if lane == "companion" else lane),
                            )}
                            if intent_guidance is not None else {}
                        ),
                    },
                    continuity_capsule=capsule,
                )
                stream.reconcile_turn_count(receipt_turns)
                prompt_streams[key] = stream
                rollover_budgets[key] = ContextRolloverBudget.from_turn_records(
                    session_manager.turn_records(
                        f"{key[0]}:{lane}", generation=generation,
                    )
                )
                write_stream_state(
                    state_path, stream=stream, budget=rollover_budgets[key],
                )
            return stream

    def run_lane(prepared, segment, lane):
        lane_binding = pipeline_lane_binding(str(lane))
        chapter_id, segment_id = prepared.chapter["chapter_id"], segment["segment_id"]
        # The scheduler may expose prepared segment objects while the chapter
        # preparation worker is still committing explicit invalidations.  A
        # paid lane must not snapshot its generation until that transaction is
        # complete.
        chapter_segments_ready[str(chapter_id)].wait()
        with prepared_report_lock:
            if chapter_id not in prepared_reported:
                progress.safe_boundary("chapter_prepared", chapter_id=chapter_id,
                                       segment_count=len(prepared.segments), substantive=True)
                prepared_reported.add(chapter_id)
        key = (chapter_id, lane)
        path = ledger_paths.setdefault(
            key, checkpoint_dir / "chapters" / chapter_id / f"{lane}-ledger.json")
        ledger = initialize_lane_ledger(
            path, chapter_id=chapter_id, lane=lane,
            segment_ids=[item["segment_id"] for item in prepared.segments])
        block_state = next(
            str(item.get("state") or "pending") for item in ledger["blocks"]
            if item["segment_id"] == segment_id
        )
        block_generation = next(
            int(item.get("generation") or ledger.get("generation") or 1)
            for item in ledger["blocks"] if item["segment_id"] == segment_id
        )
        if bool(segment.get("structural_only")):
            value = (
                {"blocks": []}
                if lane == "translation"
                else {
                    "explanation": "", "commentary": "", "commentary_sources": [],
                    "prior_work": [], "later_work": [],
                }
            )
            input_hash = lane_semantic_sha256(
                "commentary" if lane == "companion" else lane,
                {
                    "augmentation_projection_version": AUGMENTATION_PROJECTION_VERSION,
                    "segment_id": segment_id,
                    "augmentation_block_ids": [],
                    "structural_only": True,
                },
            )
            if block_state != "accepted":
                accept_controller_skipped_block(
                    path, segment_id=segment_id, input_sha256=input_hash,
                    output_sha256=sha256_json(value),
                    reason="segment contains structural headings only",
                )
            record_segment_reuse(
                chapter_id=chapter_id, segment_id=segment_id, lane=lane,
                status="skipped", artifact_id=None,
                reason="controller_skipped_structural_heading",
            )
            with reader_data_lock:
                target = reader_translations if lane == "translation" else reader_annotations
                target[segment_id] = dict(value)
            _publish_reader_update(
                options.project_dir.resolve(), state_path, reader_publish_lock,
                final_overrides=chapter_reader_overrides(),
            )
            return value
        newly_accepted = block_state != "accepted"
        migrated_translation = None
        if lane == "translation" and not newly_accepted:
            ledger_block = next(
                item for item in ledger["blocks"] if item["segment_id"] == segment_id
            )
            if isinstance(ledger_block.get("translation"), dict):
                migrated_translation = dict(ledger_block["translation"])
                migration_status = str(
                    (ledger_block.get("validation_receipt") or {}).get(
                        "reuse_status"
                    ) or "hit"
                )
                record_segment_reuse(
                    chapter_id=chapter_id, segment_id=segment_id, lane=lane,
                    status=migration_status, artifact_id=None,
                    reason=(
                        "accepted blocks were composed and locally revalidated"
                        if migration_status == "composed_hit"
                        else "accepted translation reused with terminology warnings"
                        if migration_status == "warning_reuse"
                        else "accepted translation passed current local validation"
                    ),
                )
        segment_glossary = segment_glossary_for(segment)
        object_lane = "commentary" if lane == "companion" else lane
        identity_block = next(
            item for item in ledger["blocks"] if item["segment_id"] == segment_id
        )
        predecessor_chain = str(
            identity_block.get("predecessor_accepted_chain_sha256")
            if block_state == "accepted"
            else ledger.get("accepted_chain_sha256")
            or ""
        )
        source_segment = {
            "segment": _semantic_segment_descriptor(segment),
            "blocks": [
                (_translation_input_block(blocks_by_id[value]) if lane == "translation"
                 else _annotation_input_block(blocks_by_id[value], document))
                for value in _augmentation_block_ids(segment, blocks_by_id)
            ],
        }
        if lane == "translation":
            semantic_context = {
                "source_segment": source_segment,
                "target_language": options.annotation_language,
                "glossary": segment_glossary,
                "protected_names": _protected_names_for_blocks(
                    protected_names,
                    _augmentation_blocks(segment, blocks_by_id),
                ),
                "guide": prepared.guide,
                "static_context": _full_paper_context(
                    document, segment, blocks_by_id=blocks_by_id, options=options,
                ),
                "predecessor_accepted_chain_sha256": predecessor_chain,
                **({"intent_guidance": guidance_identity} if guidance_identity else {}),
            }
        else:
            selected_evidence = _evidence_for_segment(
                segment, blocks_by_id, evidence, usage_state={"counts": {}, "topics": []},
            )
            semantic_context = {
                "source_segment": source_segment,
                "guide": prepared.guide,
                "metadata": _annotation_metadata(bundle.metadata),
                "selected_evidence": selected_evidence,
                "selected_domain_context": domain_context,
                "access_policy": _generation_runtime_policy(options),
                "predecessor_accepted_chain_sha256": predecessor_chain,
                **({"intent_guidance": guidance_identity} if guidance_identity else {}),
            }
        input_hash = lane_semantic_sha256(object_lane, semantic_context)
        local_deferred_value: dict[str, Any] | None = None
        local_deferred_artifact: dict[str, Any] | None = None
        deferred_translation = (
            identity_block.get("deferred_translation")
            if lane == "translation" else None
        )
        deferred_output = identity_block.get("deferred_output")
        if isinstance(deferred_translation, dict):
            deferred_output = deferred_translation
        if (
            newly_accepted
            and intent_guidance is None
            and isinstance(deferred_output, dict)
            and (
                identity_block.get("deferred_output_sha256") is None
                or identity_block.get("deferred_output_sha256")
                == sha256_json(deferred_output)
            )
            and object_lane not in regeneration
            and f"{('translation' if lane == 'translation' else 'commentary')}:{segment_id}"
            not in set(options.regenerate_segments)
            and _reusable_lane_output_valid(
                object_lane, segment, deferred_output,
                blocks_by_id=blocks_by_id, protected_names=protected_names,
            )
        ):
            value = dict(deferred_output)
            deferred_logical_receipt = dict(
                identity_block.get("deferred_logical_receipt") or {}
            )
            deferred_validation_receipt = dict(
                identity_block.get("deferred_validation_receipt") or {}
            )
            source_artifact_id = str(
                deferred_logical_receipt.get("source_artifact_id") or ""
            )
            source_artifact = None
            if lane == "companion":
                if source_artifact_id:
                    source_artifact = artifact_store.read(
                        object_lane, source_artifact_id,
                    )
                    artifact_store.validate(
                        source_artifact,
                        expected_kind=object_lane,
                        expected_id=source_artifact_id,
                        output_validator=lambda output: _reusable_lane_output_valid(
                            object_lane, segment, output,
                            blocks_by_id=blocks_by_id,
                            protected_names=protected_names,
                        ),
                    )
                    if (
                        source_artifact.get("contract_version") != SCHEMA_VERSION
                        or source_artifact.get("segment_id") != segment_id
                        or source_artifact.get("output_sha256")
                        != sha256_json(value)
                    ):
                        raise RuntimeError(
                            "deferred commentary source artifact does not match "
                            "the staged accepted output"
                        )
                else:
                    candidates = []
                    for record in artifact_store.iter_kind(object_lane):
                        if (
                            record.get("contract_version") == SCHEMA_VERSION
                            and record.get("segment_id") == segment_id
                            and record.get("output_sha256") == sha256_json(value)
                        ):
                            try:
                                artifact_store.validate(
                                    record,
                                    output_validator=lambda output: (
                                        _reusable_lane_output_valid(
                                            object_lane, segment, output,
                                            blocks_by_id=blocks_by_id,
                                            protected_names=protected_names,
                                        )
                                    ),
                                )
                            except Exception:
                                continue
                            candidates.append(record)
                    if candidates:
                        source_artifact = max(
                            candidates,
                            key=lambda item: float(item.get("created_at") or 0),
                        )
                        source_artifact_id = str(source_artifact["artifact_id"])
            accepted_logical_receipt = {
                **deferred_logical_receipt,
                "kind": (
                    "accepted_artifact_reuse"
                    if source_artifact is not None
                    else "deferred_accepted_artifact_reuse"
                ),
                "provider_calls": 0,
                **(
                    {"artifact_id": source_artifact_id}
                    if source_artifact is not None else {}
                ),
            }
            if lane == "companion" and source_artifact is None and result_llm is None:
                checkpoint_protected_names = _protected_names_for_blocks(
                    protected_names,
                    _augmentation_blocks(segment, blocks_by_id),
                )
                _write_annotation_checkpoint(
                    checkpoint_dir,
                    segment=segment,
                    generation=block_generation,
                    annotation=value,
                    input_sha256=_annotation_checkpoint_input_sha256(
                        segment,
                        bundle=bundle,
                        blocks_by_id=blocks_by_id,
                        options=options,
                        segment_glossary=segment_glossary,
                        segment_evidence=selected_evidence,
                        protected_names=checkpoint_protected_names,
                        domain_context=domain_context,
                        intent_guidance=intent_guidance,
                    ),
                )
            ledger = accept_deferred_block(
                path,
                segment_id=segment_id,
                input_sha256=input_hash,
                output_sha256=sha256_json(value),
                logical_receipt=accepted_logical_receipt,
                validation_receipt={
                    **deferred_validation_receipt,
                    "local_validation": True,
                    "reuse_status": "deferred_hit",
                    **(
                        {"object_store_revalidated": True}
                        if source_artifact is not None else {}
                    ),
                },
            )
            if source_artifact is not None:
                accepted_deferred_block = next(
                    item for item in ledger.get("blocks") or []
                    if item.get("segment_id") == segment_id
                )
                local_deferred_artifact = artifact_store.read_for_accepted_block(
                    kind=object_lane,
                    contract_version=SCHEMA_VERSION,
                    ledger_block=accepted_deferred_block,
                    output_validator=lambda output: _reusable_lane_output_valid(
                        object_lane, segment, output,
                        blocks_by_id=blocks_by_id,
                        protected_names=protected_names,
                    ),
                )
            if lane == "translation":
                migrated_translation = value
            local_deferred_value = value
            newly_accepted = False
            block_state = "accepted"
            record_segment_reuse(
                chapter_id=chapter_id, segment_id=segment_id, lane=lane,
                status="deferred_hit",
                artifact_id=(
                    str(local_deferred_artifact["artifact_id"])
                    if local_deferred_artifact is not None else None
                ),
                reason="staged accepted blocks passed current local validation",
            )
        semantic_invalidated = False
        if block_state == "accepted":
            accepted_block = next(
                item for item in ledger["blocks"] if item["segment_id"] == segment_id
            )
            if (
                accepted_block.get("input_sha256") != input_hash
                and not (
                    guidance_identity is None
                    and lane == "translation"
                    and isinstance(accepted_block.get("translation"), dict)
                )
            ):
                with lane_identity_lock:
                    current_ledger = initialize_lane_ledger(
                        path, chapter_id=chapter_id, lane=lane,
                        segment_ids=[item["segment_id"] for item in prepared.segments],
                    )
                    current_block = next(
                        item for item in current_ledger["blocks"]
                        if item["segment_id"] == segment_id
                    )
                    if current_block.get("state") == "accepted":
                        ledger = invalidate_suffix(
                            path, from_segment_id=segment_id,
                            generation=int(current_ledger.get("generation") or 1) + 1,
                        )
                        semantic_invalidated = True
                        block_state = "prepared"
                        block_generation = int(ledger.get("generation") or 1)
                        newly_accepted = True
                        predecessor_chain = str(ledger.get("accepted_chain_sha256") or "")
                        if session_manager is not None:
                            session_key = f"{chapter_id}:{lane}"
                            if session_manager.get_existing(session_key) is not None:
                                session_manager.rotate(
                                    session_key,
                                    reason=f"{lane}-semantic-input-change",
                                )
        contract_version = SCHEMA_VERSION
        recipe_hash = lane_recipe_sha256(
            object_lane,
            prompt=(
                _language_prompt_contract(TRANSLATION_PROMPT_VERSION, options)
                if lane == "translation"
                else _language_prompt_contract(COMMENTARY_PROMPT_VERSION, options)
            ),
            model=options.model,
            tier=(TRANSLATION_TIER if lane == "translation" else ANNOTATION_TIER),
            access_recipe={
                "provider": options.provider,
                "allow_internet": options.allow_internet if lane == "companion" else False,
                "inherit_host_tools": (
                    options.inherit_host_tools if intent_guidance is None else False
                ),
            },
        )
        reused_artifact = None
        accepted_commentary_artifact = None
        accepted_commentary_checkpoint_only = False
        requested_lane = "translation" if lane == "translation" else "commentary"
        targeted_key = f"{requested_lane}:{segment_id}"
        if (
            lane == "companion"
            and block_state == "accepted"
            and local_deferred_value is None
            and object_lane not in regeneration
            and targeted_key not in targeted_regeneration
        ):
            accepted_block = next(
                item for item in ledger["blocks"]
                if item["segment_id"] == segment_id
            )
            direct_artifact_id = artifact_id_for(
                kind=object_lane,
                semantic_input_sha256=str(accepted_block.get("input_sha256") or ""),
                output_sha256=str(accepted_block.get("output_sha256") or ""),
                contract_version=contract_version,
                predecessor_accepted_chain_sha256=str(
                    accepted_block.get("predecessor_accepted_chain_sha256") or ""
                ),
            )
            logical_receipt = accepted_block.get("logical_receipt")
            has_reuse_binding = bool(
                isinstance(logical_receipt, Mapping)
                and logical_receipt.get("kind") == "accepted_artifact_reuse"
                and logical_receipt.get("artifact_id")
            )
            expects_object = bool(
                result_llm is not None
                or artifact_store.path_for(
                    object_lane, direct_artifact_id,
                ).is_file()
                or has_reuse_binding
            )
            if expects_object:
                accepted_commentary_artifact = artifact_store.read_for_accepted_block(
                    kind=object_lane,
                    contract_version=contract_version,
                    ledger_block=accepted_block,
                    output_validator=lambda output: _reusable_lane_output_valid(
                        object_lane, segment, output,
                        blocks_by_id=blocks_by_id, protected_names=protected_names,
                    ),
                )
                record_segment_reuse(
                    chapter_id=chapter_id, segment_id=segment_id, lane=lane,
                    status="hit",
                    artifact_id=str(accepted_commentary_artifact["artifact_id"]),
                    reason=(
                        "accepted commentary object passed ledger binding and "
                        "current local validation"
                    ),
                )
            else:
                accepted_commentary_checkpoint_only = True
        if (
            newly_accepted and object_lane not in regeneration
            and targeted_key not in targeted_regeneration
        ):
            reused_artifact = artifact_store.find(
                kind=object_lane,
                semantic_input_sha256=input_hash,
                recipe_sha256=recipe_hash,
                contract_version=contract_version,
                predecessor_accepted_chain_sha256=(
                    None if lane == "translation" else predecessor_chain
                ),
                output_validator=lambda output: _reusable_lane_output_valid(
                    object_lane, segment, output,
                    blocks_by_id=blocks_by_id, protected_names=protected_names,
                ),
            )
            if reused_artifact is None and targeted_key in targeted_suffix_reuse:
                candidates = []
                for record in artifact_store.iter_kind(object_lane):
                    provenance = record.get("provenance")
                    if not isinstance(provenance, Mapping):
                        continue
                    if str(record.get("segment_id") or "") != segment_id:
                        continue
                    if Path(str(provenance.get("checkpoint_dir") or "")).resolve(
                        strict=False
                    ) != checkpoint_dir.resolve():
                        continue
                    try:
                        artifact_store.validate(
                            record,
                            output_validator=lambda output: _reusable_lane_output_valid(
                                object_lane, segment, output,
                                blocks_by_id=blocks_by_id,
                                protected_names=protected_names,
                            ),
                        )
                    except Exception:
                        continue
                    candidates.append(record)
                if candidates:
                    selected = max(
                        candidates, key=lambda item: float(item.get("created_at") or 0)
                    )
                    reused_artifact = {**selected, "reuse_status": "hit"}
            if reused_artifact is not None:
                value = dict(reused_artifact["output"])
                ledger = accept_reused_block(
                    path,
                    segment_id=segment_id,
                    input_sha256=input_hash,
                    output_sha256=str(reused_artifact["output_sha256"]),
                    artifact_id=str(reused_artifact["artifact_id"]),
                    validation_receipt={
                        "local_validation": True,
                        "object_store_revalidated": True,
                        "reuse_status": reused_artifact["reuse_status"],
                    },
                )
                reuse_status = str(reused_artifact["reuse_status"])
                source_artifact_id = str(reused_artifact["artifact_id"])
                accepted_reused_block = next(
                    item for item in ledger.get("blocks") or []
                    if item.get("segment_id") == segment_id
                )
                if str(reused_artifact.get("segment_id") or "") != segment_id:
                    raise RuntimeError(
                        "accepted artifact segment does not match the reused ledger block"
                    )
                rebound_artifact_id = artifact_id_for(
                    kind=object_lane,
                    semantic_input_sha256=input_hash,
                    output_sha256=str(reused_artifact["output_sha256"]),
                    contract_version=contract_version,
                    predecessor_accepted_chain_sha256=str(
                        accepted_reused_block[
                            "predecessor_accepted_chain_sha256"
                        ]
                    ),
                )
                rebound_artifact = (
                    reused_artifact
                    if rebound_artifact_id == source_artifact_id
                    else artifact_store.put_accepted(
                        kind=object_lane,
                        semantic_input_sha256=input_hash,
                        recipe_sha256=str(reused_artifact["recipe_sha256"]),
                        contract_version=contract_version,
                        output=value,
                        ledger_block=accepted_reused_block,
                        provider_receipt=dict(reused_artifact["provider_receipt"]),
                        provenance={
                            "checkpoint_dir": str(checkpoint_dir),
                            "chapter_id": chapter_id,
                            "segment_id": segment_id,
                            "derived_from_artifact_id": source_artifact_id,
                        },
                    )
                )
                reused_artifact = {
                    **rebound_artifact,
                    "reuse_status": reuse_status,
                }
                newly_accepted = False
                record_segment_reuse(
                    chapter_id=chapter_id, segment_id=segment_id, lane=lane,
                    status=str(reused_artifact["reuse_status"]),
                    artifact_id=str(reused_artifact["artifact_id"]),
                    reason=(
                        "accepted artifact matches semantic input and recipe"
                        if reused_artifact["reuse_status"] == "hit" else
                        "accepted artifact remains valid but its generation recipe changed"
                    ),
                )
        if reused_artifact is None and block_state != "accepted":
            record_segment_reuse(
                chapter_id=chapter_id, segment_id=segment_id, lane=lane,
                status="miss", artifact_id=None,
                reason=(
                    "explicitly selected for regeneration" if object_lane in regeneration
                    else "no accepted artifact matches semantic input and predecessor chain"
                ),
            )
        lane_llm = llm
        transport_control: dict[str, str | None] = {"idempotency_key": None}
        if result_llm is not None and session_manager is not None:
            stream = lane_stream(prepared, lane, block_generation)
            guidance_lane = "translation" if lane == "translation" else "commentary"
            lane_policy = _guidance_policy(intent_guidance, lane=guidance_lane)
            lane_structured_policy = bool(
                lane_policy is not None
                and _accepts_explicit_keyword(result_llm, "paper_access_policy")
            )
            def lane_llm(prompt: str, **kwargs: Any) -> dict[str, Any]:
                nonlocal block_generation, ledger, stream
                call_label = str(kwargs.get("call_label") or segment_id)
                artifact_dir = Path(kwargs["artifact_dir"])
                session_key = f"{chapter_id}:{lane}"
                # Chapter preparation and lane scheduling may overlap.  Bind
                # the paid call to the current registered generation at the
                # final controller boundary, never to a stale pre-invalidation
                # snapshot captured while the chapter was being prepared.
                current_ledger, _current_digest = read_registered_lane_ledger(
                    checkpoint_dir, path,
                )
                current_block = next(
                    item for item in current_ledger.get("blocks") or []
                    if item.get("segment_id") == segment_id
                )
                current_generation = int(
                    current_block.get("generation")
                    or current_ledger.get("generation")
                    or 1
                )
                if current_generation != block_generation:
                    ledger = current_ledger
                    block_generation = current_generation
                    stream = lane_stream(prepared, lane, block_generation)
                if "repair" in call_label.casefold() or "correction" in call_label.casefold():
                    correction_budget.consume(f"{chapter_id}:{lane}:{segment_id}")
                idempotency_key = (
                    f"{chapter_id}:{lane}:{call_label}:generation-{block_generation}"
                )
                if transport_control["idempotency_key"] is None:
                    transport_control["idempotency_key"] = idempotency_key
                correction = "repair" in call_label.casefold() or "correction" in call_label.casefold()
                paper_context = _full_paper_context(
                    document, segment, blocks_by_id=blocks_by_id, options=options,
                )
                segment_evidence = _evidence_for_segment(
                    segment, blocks_by_id, evidence, usage_state={"counts": {}, "topics": []},
                )
                current_payload = {
                    "segment": _semantic_segment_descriptor(segment),
                    "source_blocks": [
                        (_translation_input_block(blocks_by_id[value]) if lane == "translation"
                         else _annotation_input_block(blocks_by_id[value], document))
                        for value in _augmentation_block_ids(segment, blocks_by_id)
                    ],
                    "neighbor_context": _dynamic_paper_context(paper_context),
                    "bounded_sources": segment_evidence.get("bounded_sources") or [],
                    "protected_names": _protected_names_for_blocks(
                        protected_names,
                        [
                            blocks_by_id[value]
                            for value in _augmentation_block_ids(segment, blocks_by_id)
                            if value in blocks_by_id
                        ],
                    ),
                }
                current_payload["segment_glossary"] = segment_glossary
                runtime_profile = lane_runtime_profiles[
                    (chapter_id, lane, block_generation)
                ]
                stateful_prompt = stream.request(
                    (prompt if correction else (
                        "Translate the current segment payload under the generation rules."
                        if lane == "translation" else
                        "Research when useful and write the current segment annotation under the generation rules."
                    )), cursor=segment_id,
                    source_sha256=_segment_input_hash(
                        segment, blocks_by_id, glossary=segment_glossary
                    ),
                    current_payload=current_payload,
                )
                base_schema = kwargs.get("schema") or {}
                current_idempotency_key = idempotency_key
                current_artifact_dir = artifact_dir
                current_submission_receipt: Path | None = None
                def invoke(
                    active_prompt: str,
                    active_schema: dict[str, Any],
                    evidence_round: int,
                ) -> Any:
                    nonlocal current_idempotency_key
                    nonlocal current_artifact_dir, current_submission_receipt
                    round_suffix = (
                        "" if evidence_round == 0
                        else f":evidence-{evidence_round:02d}"
                    )
                    current_idempotency_key = idempotency_key + round_suffix
                    active_label = (
                        call_label if evidence_round == 0
                        else f"{call_label}-evidence-{evidence_round:02d}"
                    )
                    active_artifact_dir = (
                        artifact_dir if evidence_round == 0
                        else artifact_dir / f"evidence-round-{evidence_round:02d}"
                    )
                    current_artifact_dir = active_artifact_dir
                    ordered_segment_ids = [
                        str(item.get("segment_id") or "")
                        for item in ledger.get("blocks") or []
                    ]
                    try:
                        suffix_offset = ordered_segment_ids.index(segment_id)
                    except ValueError:
                        suffix_offset = len(ordered_segment_ids)
                    current_submission_receipt = write_ledger_submission_receipt(
                        checkpoint_dir=checkpoint_dir,
                        artifact_dir=active_artifact_dir,
                        ledger_path=path,
                        session_key=session_key,
                        logical_unit=segment_id,
                        generation=block_generation,
                        idempotency_key=current_idempotency_key,
                        schema=active_schema,
                        prompt=active_prompt,
                        recovery_unit=lane_binding.recovery_unit,
                        input_sha256=_segment_input_hash(
                            segment, blocks_by_id, glossary=segment_glossary,
                        ),
                        ordered_siblings=ordered_segment_ids,
                        suffix=ordered_segment_ids[suffix_offset:],
                        validator=lane_binding.validator,
                        application=lane_binding.application,
                    )
                    native_resume_authorization = _supervised_native_resume_authorized(
                        options,
                        ledger_path=path,
                        session_key=session_key,
                        logical_unit=segment_id,
                        generation=block_generation,
                        idempotency_key=current_idempotency_key,
                    )
                    with submission_limiter.permit():
                        existing_session = session_manager.get_existing(session_key)
                        return result_llm(
                            active_prompt, schema=active_schema,
                            provider=str(runtime_profile["provider"]),
                            model=runtime_profile.get("model"),
                            model_tier=(
                                None if runtime_profile.get("model")
                                else runtime_profile.get("model_tier")
                            ),
                            env=_llm_runtime_env(
                                allow_internet=bool(runtime_profile["allow_internet"]),
                                force_disable_internet=not bool(
                                    runtime_profile["allow_internet"]
                                ),
                                inherit_host_tools=(
                                    bool(runtime_profile["inherit_host_tools"])
                                    if lane_policy is None else False
                                ),
                                paper_access_policy=lane_policy,
                                serialize_paper_access_policy=not lane_structured_policy,
                            ),
                            artifact_dir=active_artifact_dir,
                            call_label=active_label,
                            idle_timeout_seconds=kwargs.get("idle_timeout_seconds"),
                            session_policy="stateful",
                            session_manager=session_manager,
                            session_key=session_key,
                            idempotency_key=current_idempotency_key,
                            progress_contract_scope="session",
                            schema_formatter_enabled=False,
                            initial_native_authorization=(
                                str(path.resolve(strict=False)),
                                session_key,
                                segment_id,
                                block_generation,
                                current_idempotency_key,
                            ),
                            supervised_native_resume=native_resume_authorization,
                            validated_legacy_logical_identity=(
                                {
                                    "provider": existing_session.provider,
                                    "model": existing_session.model,
                                    "session_key": session_key,
                                    "generation": block_generation,
                                    "idempotency_key": current_idempotency_key,
                                }
                                if native_resume_authorization is not None
                                and existing_session is not None else None
                            ),
                            validated_legacy_runtime_identity=(
                                {
                                    "session_key": existing_session.key,
                                    "provider": existing_session.provider,
                                    "model": existing_session.model,
                                    "generation": existing_session.generation,
                                    "native_session_id": existing_session.native_session_id,
                                    "recorded_fp": existing_session.runtime_fingerprint,
                                }
                                if native_resume_authorization is not None
                                and existing_session is not None else None
                            ),
                            cancel_check=cancel_inflight_check,
                            progress_callback=lambda event: (
                                _guarded_mark_transport_state(
                                    path,
                                    checkpoint_dir=checkpoint_dir,
                                    session_key=session_key,
                                    logical_unit=segment_id,
                                    idempotency_key=str(
                                        transport_control["idempotency_key"]
                                    ),
                                )
                                if event.get("event") == "submitted"
                                else None,
                                progress.provider_event(event),
                            )[-1],
                            **(
                                {"paper_access_policy": lane_policy}
                                if lane_structured_policy else {}
                            ),
                        )
                try:
                    outcome = invoke(
                        stateful_prompt,
                        _intent_guidance_schema(base_schema, intent_guidance),
                        0,
                    )
                    evidence_source_generation, evidence_source_logical_task = (
                        _companion_evidence_source_identity(
                            project_dir=options.project_dir,
                            checkpoint_dir=checkpoint_dir,
                            run_id=fingerprint,
                            ledger_path=path,
                            ledger=ledger,
                            session_key=session_key,
                            logical_unit=segment_id,
                            target_generation=block_generation,
                            target_idempotency_key=idempotency_key,
                        )
                    )
                    outcome, final_value = _complete_stateful_reference_evidence(
                        outcome, intent_guidance=intent_guidance,
                        lane=guidance_lane, worker_id=call_label,
                        schema=base_schema, call_round=invoke,
                        journal_context=_companion_evidence_journal_context(
                            checkpoint_dir=checkpoint_dir,
                            run_id=fingerprint,
                            lane=guidance_lane,
                            worker_id=call_label,
                            logical_task_id=evidence_source_logical_task,
                            source_generation=evidence_source_generation,
                            policy=lane_policy,
                            runtime=runtime_profile,
                        ),
                        target_session=session_key,
                        target_generation=block_generation,
                        followup_id=idempotency_key,
                    )
                except BaseException as exc:
                    if _chapter_failure_requires_supervision(exc):
                        from arc_llm import read_recovery_context
                        recovery = read_recovery_context(
                            current_artifact_dir,
                            idempotency_key=current_idempotency_key,
                            session_manager=session_manager, session_key=session_key,
                        )
                        recovery_context = _recovery_context_json(
                            recovery, logical_unit=segment_id,
                        )
                        if current_submission_receipt is not None:
                            try:
                                seal_submission_attempts(
                                    current_submission_receipt,
                                    checkpoint_dir=checkpoint_dir,
                                    attempt_references=explicit_attempt_references(
                                        exc,
                                        checkpoint_dir=checkpoint_dir,
                                        artifact_dir=current_artifact_dir,
                                    ),
                                )
                                recovery_context["submission_receipt"] = (
                                    submission_receipt_reference(
                                        current_submission_receipt,
                                        checkpoint_dir=checkpoint_dir,
                                    )
                                )
                            except RecoveryResponseError as receipt_exc:
                                recovery_context["submission_receipt_error"] = str(
                                    receipt_exc
                                )
                        mark_needs_supervision(
                            path, segment_id=segment_id, reason=str(exc),
                            recovery_context=recovery_context,
                        )
                        supervision_event.set()
                    raise
                logical_receipts[call_label] = dict(outcome.logical_receipt or {})
                if not logical_receipts[call_label]:
                    raise RuntimeError(f"stateful call {call_label} returned no logical receipt")
                _guarded_mark_transport_state(
                    path,
                    checkpoint_dir=checkpoint_dir,
                    session_key=session_key,
                    logical_unit=segment_id,
                    idempotency_key=str(transport_control["idempotency_key"]),
                    response_received=True,
                )
                usage = getattr(outcome, "usage", {})
                usage_json = usage.to_json() if hasattr(usage, "to_json") else usage
                logical = logical_receipts[call_label]
                provider_receipts[call_label] = {
                    "provider": str(getattr(outcome, "provider", None) or options.provider),
                    "model": str(getattr(outcome, "model", None) or options.model or "provider-default"),
                    "call_id": str(logical.get("idempotency_key") or logical.get("call_id") or call_label),
                    "usage": dict(usage_json) if isinstance(usage_json, dict) else {},
                }
                generation_key = (chapter_id, lane, block_generation)
                profile_path = (
                    checkpoint_dir / "chapters" / chapter_id
                    / f"{lane}-runtime-generation-{block_generation}.json"
                )
                refreshed_session = session_manager.get_existing(session_key)
                lane_runtime_profiles[generation_key] = pin_lane_runtime_profile(
                    profile_path, runtime_profile,
                    provider=(refreshed_session.provider if refreshed_session else options.provider),
                    model=(refreshed_session.model if refreshed_session else options.model),
                    runtime_fingerprint=str(getattr(outcome, "runtime_fingerprint", "")),
                    migrated_from_fingerprint=(
                        str(refreshed_session.metadata.get(
                            "arc_runtime_fingerprint_migrated_from"
                        ))
                        if refreshed_session is not None
                        and refreshed_session.metadata.get(
                            "arc_runtime_fingerprint_migrated_from"
                        ) else None
                    ),
                )
                rollover_budgets[generation_key].record(
                    getattr(outcome, "usage", {}),
                    prompt_bytes=getattr(outcome, "prompt_bytes", None),
                )
                stream.reconcile_turn_count(
                    session_manager.turn_count(
                        session_key, generation=block_generation,
                    )
                )
                write_stream_state(
                    checkpoint_dir / "chapters" / chapter_id
                    / f"{lane}-stream-generation-{block_generation}.json",
                    stream=stream, budget=rollover_budgets[generation_key],
                )
                return final_value
            # The stateful adapter writes the production receipt against the
            # real chapter lane immediately before result_llm.  The generic
            # stateless hook must not create a second receipt for that call.
            setattr(lane_llm, "_arc_owns_recovery_receipt", True)
        try:
            if accepted_commentary_artifact is not None:
                value = dict(accepted_commentary_artifact["output"])
            elif accepted_commentary_checkpoint_only:
                def reject_accepted_checkpoint_provider_call(
                    *_args: Any, **_kwargs: Any,
                ) -> dict[str, Any]:
                    raise RuntimeError(
                        "accepted commentary has no reusable object and its "
                        "local checkpoint is unavailable or invalid"
                    )

                value = _generate_annotations(
                    [segment], options=options, bundle=bundle, evidence=evidence,
                    domain_context=domain_context, glossary=segment_glossary,
                    protected_names=_protected_names_for_blocks(
                        protected_names, _augmentation_blocks(segment, blocks_by_id),
                    ), checkpoint_dir=checkpoint_dir,
                    llm=reject_accepted_checkpoint_provider_call,
                    generation=block_generation, force_generation=False,
                    intent_guidance=intent_guidance,
                )[segment_id]
                if not _reusable_lane_output_valid(
                    object_lane, segment, value,
                    blocks_by_id=blocks_by_id,
                    protected_names=protected_names,
                ):
                    raise RuntimeError(
                        "accepted commentary checkpoint fails current local validation"
                    )
                record_segment_reuse(
                    chapter_id=chapter_id, segment_id=segment_id, lane=lane,
                    status="hit", artifact_id=None,
                    reason=(
                        "accepted commentary checkpoint passed current local validation"
                    ),
                )
            elif reused_artifact is not None:
                pass
            elif local_deferred_value is not None:
                value = local_deferred_value
            elif migrated_translation is not None:
                value = migrated_translation
            elif lane == "translation":
                value = _generate_translations(
                    [segment], options=options, bundle=bundle,
                    glossary=segment_glossary,
                    protected_names=_protected_names_for_blocks(
                        protected_names, _augmentation_blocks(segment, blocks_by_id),
                    ),
                    checkpoint_dir=checkpoint_dir, llm=lane_llm,
                    generation=block_generation,
                    force_generation=(
                        "translation" in regeneration
                        or targeted_key in targeted_regeneration
                        or semantic_invalidated
                    ),
                    intent_guidance=intent_guidance,
                )[segment_id]
            else:
                value = _generate_annotations(
                    [segment], options=options, bundle=bundle, evidence=evidence,
                    domain_context=domain_context, glossary=segment_glossary,
                    # Only current-block explanations are repeated on delta turns; the
                    # complete source-to-target mapping lives in the generation bootstrap.
                    protected_names=_protected_names_for_blocks(
                        protected_names, _augmentation_blocks(segment, blocks_by_id),
                    ), checkpoint_dir=checkpoint_dir, llm=lane_llm,
                    generation=block_generation,
                    force_generation=(
                        "commentary" in regeneration
                        or targeted_key in targeted_regeneration
                        or semantic_invalidated
                    ),
                    intent_guidance=intent_guidance,
                )[segment_id]
        except BaseException as exc:
            _mark_translation_repair_supervision(
                path, segment_id=segment_id, exc=exc,
            )
            # A paid-repair supervision failure is local.  This path does not
            # set the shared stop/cancel event, so unrelated work may drain.
            raise
        if newly_accepted:
            # A successfully returned stateless call is itself a definitive
            # submission/response receipt; stateful calls normally crossed
            # these barriers earlier through provider progress.
            chapter_call_label = (
                f"companion-{'translation' if lane == 'translation' else 'annotation'}-"
                f"{segment_id}"
            )
            chapter_control_key = str(
                transport_control["idempotency_key"]
                or f"{chapter_id}:{lane}:{chapter_call_label}:generation-{block_generation}"
            )
            _guarded_mark_transport_state(
                path,
                checkpoint_dir=checkpoint_dir,
                session_key=f"{chapter_id}:{lane}",
                logical_unit=segment_id,
                idempotency_key=chapter_control_key,
                response_received=True,
            )
        accepted_ledger = None
        if newly_accepted:
            advance_block(path, segment_id=segment_id, state="schema_valid")
            advance_block(path, segment_id=segment_id, state="invariant_valid")
            accepted_ledger = advance_block(
                path, segment_id=segment_id, state="accepted",
                receipt=logical_receipts.get(
                    f"companion-{'translation' if lane == 'translation' else 'annotation'}-{segment_id}"
                ), input_sha256=input_hash, output_sha256=sha256_json(value),
                validation_receipt={"local_validation": True, "correction_turns_max": 1},
            )
            call_label = (
                f"companion-{'translation' if lane == 'translation' else 'annotation'}-{segment_id}"
            )
            provider_receipt = provider_receipts.get(call_label)
            accepted_block = next(
                item for item in accepted_ledger.get("blocks") or []
                if item.get("segment_id") == segment_id
            )
            if provider_receipt is not None and accepted_block.get("logical_receipt"):
                artifact_store.put_accepted(
                    kind=object_lane,
                    semantic_input_sha256=input_hash,
                    recipe_sha256=recipe_hash,
                    contract_version=contract_version,
                    output=value,
                    ledger_block=accepted_block,
                    provider_receipt=provider_receipt,
                    provenance={
                        "checkpoint_dir": str(checkpoint_dir),
                        "chapter_id": chapter_id,
                        "segment_id": segment_id,
                    },
                )
        if newly_accepted:
            progress.safe_boundary("block_accepted", chapter_id=chapter_id,
                                   segment_id=segment_id, lane=lane,
                                   block_status="accepted", substantive=True)
            current_index = next(
                index for index, item in enumerate(prepared.segments)
                if item["segment_id"] == segment_id
            )
            budget = rollover_budgets.get((chapter_id, lane, block_generation))
            if (
                result_llm is not None and session_manager is not None
                and budget is not None and budget.rollover_due()
                and current_index + 1 < len(prepared.segments)
            ):
                next_segment_id = str(prepared.segments[current_index + 1]["segment_id"])
                rotated = session_manager.rotate(
                    f"{chapter_id}:{lane}", reason="70-percent-context-boundary",
                )
                invalidate_suffix(
                    path, from_segment_id=next_segment_id, generation=rotated.generation,
                )
                accepted_block = next(
                    item for item in (accepted_ledger or {}).get("blocks") or []
                    if item.get("segment_id") == segment_id
                )
                with stream_lock:
                    prompt_streams[(chapter_id, lane, rotated.generation)] = StatefulPromptStream(
                        chapter_id=chapter_id, lane=lane, generation=rotated.generation,
                        fixed_rules=dict(stream.fixed_rules),
                        static_context=dict(stream.static_context),
                        continuity_capsule=continuity_capsule(
                            accepted_chain_sha256=str(accepted_block.get("accepted_chain_sha256") or ""),
                            segment_id=segment_id, input_sha256=input_hash,
                            output_sha256=sha256_json(value),
                        ),
                    )
                    rollover_budgets[(chapter_id, lane, rotated.generation)] = ContextRolloverBudget()
                    write_stream_state(
                        checkpoint_dir / "chapters" / chapter_id
                        / f"{lane}-stream-generation-{rotated.generation}.json",
                        stream=prompt_streams[(chapter_id, lane, rotated.generation)],
                        budget=rollover_budgets[(chapter_id, lane, rotated.generation)],
                    )
        with reader_data_lock:
            target = reader_translations if lane == "translation" else reader_annotations
            target[str(segment_id)] = dict(value)
        _publish_reader_update(
            options.project_dir.resolve(), state_path, reader_publish_lock,
            final_overrides=chapter_reader_overrides(),
        )
        return value

    try:
        scheduler_kwargs = {
            "workers": options.workers,
            "prepare_guide": prepare_guide,
            "prepare_segments": prepare_segments,
            "run_translation": (
                None if options.skip_translation
                else lambda prepared, segment: run_lane(prepared, segment, "translation")
            ),
            "run_companion": lambda prepared, segment: run_lane(prepared, segment, "companion"),
            # An uncertain submitted provider call stops new batch submissions
            # but never cancels calls already in flight. Deterministic local
            # failures remain contained to their lane/chapter.
            "stop_event": supervision_event,
            "cancel_check": cancel_inflight_check,
        }
        freeze_path = checkpoint_dir / (
            "first-chapter-freeze.json"
            if guidance_identity is None else
            f"first-chapter-freeze.{str(guidance_identity['output_sha256'])[:16]}.json"
        )
        existing_freeze = _load_first_chapter_freeze(
            freeze_path, chapters_pack["chapters"], required=require_first_chapter_freeze
        )
        if existing_freeze is not None and not options.stop_after_first_chapter:
            first_results = run_chapter_pipeline(
                chapters_pack["chapters"][:1], **scheduler_kwargs,
            )
            _verify_frozen_first_chapter_pre_review(existing_freeze, first_results)
            remaining_results = run_chapter_pipeline(
                chapters_pack["chapters"][1:], **scheduler_kwargs,
            )
            chapter_results = {**first_results, **remaining_results}
        else:
            chapter_results = run_chapter_pipeline(
                chapters_pack["chapters"],
                stop_after_first_chapter=options.stop_after_first_chapter,
                **scheduler_kwargs,
            )
    except Exception as exc:
        if not _chapter_failure_requires_supervision(exc):
            raise
        progress.safe_boundary("needs_supervision", reason=str(exc), substantive=True)
        supervised = _state(
            state_path, status="needs_supervision", paper_id=bundle.paper_id,
            fingerprint=fingerprint, checkpoint_dir=str(checkpoint_dir), error=str(exc),
            notice=notice, diagnostics=list(diagnostics),
            recovery_options=_recovery_options(options),
        )
        return {"ok": False, "status": "needs_supervision", "data": supervised,
                "error": {"code": "companion_needs_supervision", "message": str(exc)},
                "errors": [], "meta": {"diagnostics": list(diagnostics)}}
    segments = [item for chapter in chapter_results.values() for item in chapter["segments"]]
    missing_targets = targeted_regeneration.difference(targeted_seen)
    if missing_targets:
        raise ValueError(
            "regenerate-segment target not found: " + ", ".join(sorted(missing_targets))
        )
    translations = None if options.skip_translation else {
        key: value for chapter in chapter_results.values()
        for key, value in chapter["translation"].items()
    }
    annotations = {key: value for chapter in chapter_results.values()
                   for key, value in chapter["companion"].items()}
    review_segments = [item for item in segments if not item.get("structural_only")]
    review_segment_ids = {str(item["segment_id"]) for item in review_segments}
    review_input_hash = lane_semantic_sha256("review", {
        "translation_artifacts": (
            None if translations is None else {
                key: sha256_json(value) for key, value in sorted(translations.items())
                if key in review_segment_ids
            }
        ),
        "commentary_artifacts": {
            key: sha256_json(value) for key, value in sorted(annotations.items())
            if key in review_segment_ids
        },
        "review_contract": {
            "translation_mode": "skipped" if options.skip_translation else "enabled",
            "augmentation_projection_version": AUGMENTATION_PROJECTION_VERSION,
            "segment_ids": [str(item["segment_id"]) for item in review_segments],
        },
        **({"intent_guidance": guidance_identity} if guidance_identity else {}),
    })
    review_recipe_hash = lane_recipe_sha256(
        "review", prompt=REVIEW_VERSION, model=options.model, tier=REVIEW_TIER,
        access_recipe={"provider": options.provider, "allow_internet": False},
    )
    review_object = None if "review" in regeneration else artifact_store.find(
        kind="review", semantic_input_sha256=review_input_hash,
        recipe_sha256=review_recipe_hash, contract_version=REVIEW_VERSION,
        predecessor_accepted_chain_sha256=hashlib.sha256(b"").hexdigest(),
        output_validator=lambda value: (
            isinstance(value, dict)
            and isinstance(value.get("annotations"), dict)
            and isinstance(value.get("review"), dict)
        ),
    )
    if review_object is not None:
        mark_plan_lane(
            "review", chapter_id=None, artifact=review_object,
            reason="accepted review remains valid and remains bound to the same base artifacts",
        )
        review_output = dict(review_object["output"])
        translations = review_output.get("translations")
        annotations = dict(review_output["annotations"])
        chapter_review = _with_historical_review_prompt_audit(
            dict(review_output["review"]), options,
        )
    else:
        translations, annotations, chapter_review = _review(
            review_segments, translations, annotations, document=document, glossary=glossary,
            protected_names=protected_names, evidence=evidence, options=options,
            llm=llm, checkpoint_dir=checkpoint_dir,
            intent_guidance=intent_guidance,
        )
        review_output = {
            "translations": translations,
            "annotations": annotations,
            "review": chapter_review,
        }
        _store_validated_stateless_artifact(
            artifact_store, kind="review",
            semantic_input_sha256=review_input_hash,
            recipe_sha256=review_recipe_hash,
            contract_version=REVIEW_VERSION, output=review_output,
            segment_id="project:review", checkpoint_dir=checkpoint_dir,
            provider=options.provider, model=options.model,
        )
    write_json(checkpoint_dir / "chapter-review.json", chapter_review)
    _write_review_overlays(
        checkpoint_dir, chapter_results, translations=translations, annotations=annotations,
    )
    selected_chapters = [item for item in chapters_pack["chapters"]
                         if item["chapter_id"] in chapter_results]
    guides = {key: value["guide"] for key, value in chapter_results.items()}
    index_block_ids = {
        block_id(item) for item in document.get("blocks") or []
        if str(item.get("source_role") or item.get("role") or "").casefold() == "index"
    }
    index_block_ids.update(str(value) for value in structure.get("index_block_ids") or [])
    render_document = (
        {**document, "blocks": [item for item in document.get("blocks") or []
                                if block_id(item) not in index_block_ids
                                or _project_is_structural(item)]}
        if index_entries and not options.skip_translation else document
    )
    final_reader_overrides = {
        "status": (
            "first_chapter_ready" if options.stop_after_first_chapter else "complete"
        ),
        "document": render_document,
        "chapters": selected_chapters,
        "segments": segments,
        "chapter_guides": guides,
        "translations": translations,
        "annotations": annotations,
        "glossary": glossary,
        "metadata": bundle.metadata,
        "language": options.annotation_language,
        "source_language": options.source_language,
        "title_translations": title_translations,
        "translation_mode": "skipped" if options.skip_translation else "enabled",
    }
    content_object = _store_reviewed_content(
        options.project_dir.resolve(), checkpoint_dir=checkpoint_dir,
        final_overrides=final_reader_overrides, evidence=evidence,
    )
    _write_reader_final_checkpoint(checkpoint_dir, final_reader_overrides)
    _state(
        state_path,
        content_sha256=content_object["content_sha256"],
        content_object_path=str(content_object["path"]),
    )
    stem = f"{safe_name(bundle.paper_id)}_companion_{safe_name(options.annotation_language)}"
    artifact = _publish_pdf_artifact(
        document=render_document, segments=segments, annotations=annotations,
        translations=translations, evidence=evidence, glossary=glossary,
        metadata=bundle.metadata, language=options.annotation_language,
        source_language=options.source_language,
        title_translations=title_translations,
        output_dir=options.project_dir.resolve(), stem=stem,
        manifest_name="source-manifest.json", validation_name="validation.json",
        compiler=compiler, pdf_validator=pdf_validator, augmentation_scope="substantive",
        chapters=selected_chapters, chapter_guides=guides,
    )
    if existing_freeze is not None:
        _verify_frozen_first_chapter_final(
            existing_freeze, chapter_results, translations=translations, annotations=annotations,
        )
    if options.stop_after_first_chapter:
        freeze_path = checkpoint_dir / (
            "first-chapter-freeze.json"
            if guidance_identity is None else
            f"first-chapter-freeze.{str(guidance_identity['output_sha256'])[:16]}.json"
        )
        first_id = selected_chapters[0]["chapter_id"]
        first_segment_ids = [item["segment_id"] for item in chapter_results[first_id]["segments"]]
        freeze = {
            "schema_version": "arc.companion.first-chapter-freeze.v3",
            "translation_mode": "skipped" if options.skip_translation else "enabled",
            "chapter_id": first_id,
            "chapter_sha256": sha256_json(selected_chapters[0]),
            "guide_sha256": sha256_json(guides[first_id]),
            "pre_review_translation_sha256": (
                None if options.skip_translation
                else sha256_json(chapter_results[first_id]["translation"])
            ),
            "pre_review_annotation_sha256": sha256_json(chapter_results[first_id]["companion"]),
            "translation_sha256": (
                None if options.skip_translation
                else sha256_json({value: translations[value] for value in first_segment_ids})
            ),
            "annotation_sha256": sha256_json({value: annotations[value] for value in first_segment_ids}),
            "tex_sha256": artifact["tex_sha256"], "pdf_sha256": artifact["pdf_sha256"],
            "manifest_sha256": artifact["manifest_sha256"],
        }
        if freeze_path.is_file() and read_json(freeze_path) != freeze:
            raise RuntimeError("the confirmed first chapter freeze manifest changed")
        write_json(freeze_path, freeze)
    _publish_reader_update(
        options.project_dir.resolve(), state_path, reader_publish_lock,
        final_overrides=final_reader_overrides,
        strict=True,
    )
    # Provider success is only response_received. Promote stateless controls
    # after the owning business pipeline completed normalization, invariants,
    # and durable application checkpoints.
    _accept_completed_pipeline_controls(checkpoint_dir)
    managed_run_pdf = managed_run_root_pdf_path(
        _read_optional_json(state_path)
    )
    for chapter_id in chapter_results:
        progress.safe_boundary("chapter_complete", chapter_id=chapter_id,
                               artifact_paths=[artifact["pdf_path"]], substantive=True)
    status = "first_chapter_ready" if options.stop_after_first_chapter else "complete"
    final = _state(state_path, status=status, paper_id=bundle.paper_id,
                   fingerprint=fingerprint, checkpoint_dir=str(checkpoint_dir),
                   chapter_count=len(selected_chapters), segment_count=len(segments),
                   output_tex=artifact["tex_path"], output_pdf=artifact["pdf_path"],
                   output_tex_sha256=artifact["tex_sha256"],
                   output_pdf_sha256=artifact["pdf_sha256"],
                   source_manifest_path=artifact["manifest_path"],
                   source_manifest_sha256=artifact["manifest_sha256"],
                   validation_path=artifact["validation_path"],
                   validation_sha256=artifact["validation_sha256"],
                   final_render_version=FINAL_RENDER_VERSION,
                   chapter_projection_version=CHAPTER_PROJECTION_VERSION,
                   augmentation_projection_version=AUGMENTATION_PROJECTION_VERSION,
                   chapter_guide_version=CHAPTER_GUIDE_VERSION,
                   reader_final_checkpoint_version=READER_FINAL_CHECKPOINT_VERSION,
                   translation_mode="skipped" if options.skip_translation else "enabled",
                   notice=notice, diagnostics=list(diagnostics))
    run_pdf: dict[str, str] = {}
    if not options.stop_after_first_chapter:
        run_pdf = publish_run_root_pdf(
            Path(artifact["pdf_path"]), options.project_dir.resolve(),
            managed_path=managed_run_pdf,
        )
        final = _state(state_path, **run_pdf)
    progress.safe_boundary(
        status,
        artifact_paths=[
            artifact["pdf_path"],
            *([run_pdf["output_run_pdf"]] if run_pdf else []),
        ],
        substantive=True,
    )
    return {"ok": True, "status": status, "data": final, "errors": [],
            "meta": {"diagnostics": list(diagnostics), "notice": notice}}


def _supervised_lane_ledger_paths(checkpoint_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for path in _all_lane_ledger_paths(checkpoint_dir):
        try:
            ledger = _read_recovery_ledger(checkpoint_dir, path)
        except (OSError, ValueError, json.JSONDecodeError, SecureReadError):
            continue
        if isinstance(ledger, dict) and ledger.get("needs_supervision"):
            paths.append(path)
    return paths


def _active_supervision_entries(ledger: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return every active marker once, preserving earliest-first lane order."""

    raw = [ledger.get("needs_supervision"), *(ledger.get("supervision_entries") or [])]
    by_segment: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        segment_id = str(item.get("segment_id") or "")
        if segment_id and segment_id not in by_segment:
            by_segment[segment_id] = dict(item)
    positions = {
        str(item.get("segment_id") or ""): index
        for index, item in enumerate(ledger.get("blocks") or [])
        if isinstance(item, dict)
    }
    return sorted(
        by_segment.values(),
        key=lambda item: positions.get(str(item.get("segment_id") or ""), len(positions)),
    )


def _all_lane_ledger_paths(
    checkpoint_dir: Path, *, include_explicit_legacy: bool = False,
) -> list[Path]:
    """Read the bounded ARC-owned ledger registry for automatic recovery.

    The legacy scan is intentionally reachable only from explicit
    ``resume-native`` call sites.  It must never define automatic ownership.
    """

    paths = registered_lane_ledger_paths(checkpoint_dir)
    if include_explicit_legacy:
        known = {str(path) for path in paths}
        paths.extend(
            path for path in legacy_lane_ledger_paths(checkpoint_dir)
            if str(path) not in known
        )
    return sorted(paths)


def _chapter_failure_requires_supervision(exc: BaseException) -> bool:
    if isinstance(exc, TranslationRepairNeedsSupervision):
        return True
    if isinstance(exc, TimeoutError):
        return True
    try:
        from arc_llm import LLMWorkerError
        from arc_llm.providers.base import LLMSubmissionState, failure_disposition
        if isinstance(exc, LLMWorkerError):
            disposition = failure_disposition(exc)
            return bool(
                disposition
                and disposition.submission_state != LLMSubmissionState.NOT_SUBMITTED
            )
    except ImportError:
        pass
    name = type(exc).__name__.casefold()
    if any(token in name for token in ("cancel", "session", "provider")):
        return True
    failures = getattr(exc, "failures", None)
    if isinstance(failures, dict):
        return any(_chapter_failure_requires_supervision(value) for value in failures.values())
    if isinstance(failures, list):
        return any(
            _chapter_failure_requires_supervision(
                value[1] if isinstance(value, tuple) and len(value) == 2 else value
            )
            for value in failures
        )
    return bool(exc.__cause__ and _chapter_failure_requires_supervision(exc.__cause__))


def _translation_repair_supervision(
    exc: BaseException,
) -> TranslationRepairNeedsSupervision | None:
    """Find a paid-repair supervision signal through lane aggregation wrappers."""
    if isinstance(exc, TranslationRepairNeedsSupervision):
        return exc
    failures = getattr(exc, "failures", None)
    values: list[Any] = []
    if isinstance(failures, dict):
        values.extend(failures.values())
    elif isinstance(failures, list):
        values.extend(
            item[1] if isinstance(item, tuple) and len(item) == 2 else item
            for item in failures
        )
    for value in values:
        if isinstance(value, BaseException):
            found = _translation_repair_supervision(value)
            if found is not None:
                return found
    if exc.__cause__ is not None:
        return _translation_repair_supervision(exc.__cause__)
    return None


def _mark_translation_repair_supervision(
    path: Path, *, segment_id: str, exc: BaseException,
) -> bool:
    """Persist one local supervision marker even when lane errors are wrapped."""
    supervision = _translation_repair_supervision(exc)
    if supervision is None:
        return False
    mark_needs_supervision(
        path, segment_id=segment_id, reason=str(supervision),
        recovery_context=supervision.recovery_context,
    )
    return True


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


_BUILD_COMPANION_ENTRYPOINT = build_companion


def read_status(project_dir: Path) -> dict[str, Any]:
    path = project_dir.resolve() / "state.json"
    if not path.is_file():
        return err("companion_state_not_found", f"No companion state found in {project_dir}")
    from .observability import enrich_status

    state = normalize_run_root_pdf_state(read_json(path))
    return ok(enrich_status(project_dir, state))


def _finalize_paid_translation_repairs(
    checkpoint_dir: Path,
) -> list[dict[str, Any]]:
    """Classify every unaccepted paid token repair without provider calls."""
    document_payload = _read_checkpoint_json(
        checkpoint_dir / "document.json", root=checkpoint_dir,
    )
    document = (
        document_payload.get("document")
        if isinstance(document_payload, dict) else None
    )
    if not isinstance(document, dict):
        return []
    glossary = _read_checkpoint_json(
        checkpoint_dir / "glossary.json", root=checkpoint_dir,
    )
    protected_names = _protected_names(
        SourceBundle(
            paper_id=str(document_payload.get("paper_id") or "checkpoint"),
            parsed={}, document=document, metadata={}, references=[], citers=[],
        ),
        glossary=glossary if isinstance(glossary, dict) else None,
    )
    blocks_by_id = {
        block_id(item): item for item in document.get("blocks") or []
        if isinstance(item, dict)
    }
    ledgers: dict[str, tuple[Path, dict[str, Any], dict[str, Any]]] = {}
    for ledger_path in _all_lane_ledger_paths(checkpoint_dir):
        ledger = _read_recovery_ledger(checkpoint_dir, ledger_path)
        if ledger.get("lane") != "translation":
            continue
        for block in ledger.get("blocks") or []:
            if isinstance(block, dict) and block.get("state") != "accepted":
                ledgers[str(block.get("segment_id") or "")] = (
                    ledger_path, ledger, block,
                )
    repair_bindings: dict[tuple[str, str], dict[str, Any]] = {}
    ambiguous_bindings: set[tuple[str, str]] = set()
    for receipt_path, receipt in discover_submission_receipts(checkpoint_dir):
        unit = str(receipt.get("recovery_unit") or "")
        suffix = {
            "translation-token-repair": ":token-repair",
            "translation-coverage-repair": ":coverage-repair",
        }.get(unit)
        logical_unit = str(receipt.get("logical_unit") or "")
        if suffix is None or not logical_unit.endswith(suffix) or not receipt.get("sealed"):
            continue
        segment_id = logical_unit[:-len(suffix)]
        located = ledgers.get(segment_id)
        if located is None:
            continue
        _chapter_path, chapter_ledger, chapter_block = located
        generation = int(
            chapter_block.get("generation")
            or chapter_ledger.get("generation")
            or 1
        )
        if int(receipt.get("generation") or 0) != generation:
            continue
        try:
            control_path = resolve_recovery_path(
                checkpoint_dir, receipt.get("ledger_path"),
            )
            reference = submission_receipt_reference(
                receipt_path, checkpoint_dir=checkpoint_dir,
            )
            validated = _validate_pipeline_submission_reference(
                reference,
                checkpoint_dir=checkpoint_dir,
                ledger_path=control_path,
                session_key=str(receipt.get("session_key") or ""),
                logical_unit=logical_unit,
                generation=generation,
                idempotency_key=str(receipt.get("idempotency_key") or ""),
            )
            call_checkpoint_path = resolve_recovery_path(
                checkpoint_dir, validated.get("checkpoint_path"),
            )
            call_checkpoint = _read_recovery_json(
                checkpoint_dir, call_checkpoint_path,
            )
        except (RecoveryResponseError, SecureReadError, TypeError, ValueError):
            continue
        if not isinstance(call_checkpoint, Mapping):
            continue
        key = (unit, segment_id)
        if key in repair_bindings:
            ambiguous_bindings.add(key)
            repair_bindings.pop(key, None)
            continue
        if key not in ambiguous_bindings:
            repair_bindings[key] = {
                "receipt": validated,
                "receipt_reference": reference,
                "checkpoint_path": call_checkpoint_path,
                "checkpoint": dict(call_checkpoint),
            }
    entries: list[dict[str, Any]] = []
    token_marker_paths = [
        (segment_id, _translation_token_attempt_path(
            checkpoint_dir,
            segment_id,
            int(binding["receipt"].get("generation") or 1),
        ))
        for (unit, segment_id), binding in repair_bindings.items()
        if unit == "translation-token-repair"
    ]
    for segment_id, marker_path in sorted(token_marker_paths):
        binding = repair_bindings[("translation-token-repair", segment_id)]
        try:
            marker = _read_recovery_json(checkpoint_dir, marker_path)
        except SecureReadError:
            continue
        if not isinstance(marker, dict) or marker.get("status") not in {
            "response_received", "validated",
        }:
            continue
        if str(marker.get("segment_id") or "") != segment_id:
            continue
        try:
            marker_generation = _artifact_payload_generation(
                marker, checkpoint_dir, "translation-token-offset-attempts",
                segment_id,
            )
        except (TypeError, ValueError):
            continue
        located = ledgers.get(segment_id)
        if located is None:
            continue
        ledger_path, ledger, ledger_block = located
        block_generation = int(
            ledger_block.get("generation") or ledger.get("generation") or 1
        )
        if marker_generation != block_generation:
            continue
        draft = _read_checkpoint_json(
            _generation_segment_artifact_dir(
                checkpoint_dir, "translation-drafts", segment_id,
                marker_generation,
            ) / marker_path.name,
            root=checkpoint_dir,
        )
        source_ids = [str(value) for value in marker.get("block_ids") or []]
        raw_response = marker.get("raw_response")
        error: str | None = None
        receipt_path: Path | None = None
        if not (
            isinstance(draft, dict)
            and draft.get("segment_id") == segment_id
            and draft.get("input_sha256") == marker.get("input_sha256")
            and isinstance(draft.get("translation"), dict)
            and isinstance(raw_response, dict)
            and source_ids
            and all(value in blocks_by_id for value in source_ids)
        ) and error is None:
            error = "paid repair is missing its matching draft, response, or source blocks"
        elif error is None:
            try:
                draft_block_ids = [
                    str(item.get("block_id") or "")
                    for item in draft["translation"].get("blocks") or []
                    if isinstance(item, dict)
                    and str(item.get("block_id") or "") in blocks_by_id
                ]
                segmentation = _read_checkpoint_json(
                    ledger_path.parent / "segmentation.json", root=checkpoint_dir,
                )
                local_segment_id = segment_id.rsplit(".", 1)[-1]
                segment_record = next((
                    item for item in (
                        segmentation.get("segments")
                        if isinstance(segmentation, dict) else []
                    ) or []
                    if isinstance(item, dict)
                    and str(item.get("segment_id") or "") == local_segment_id
                ), None)
                validation_block_ids = (
                    [str(value) for value in segment_record.get("block_ids") or []]
                    if isinstance(segment_record, dict) else draft_block_ids
                )
                validation_segment = {
                    "segment_id": segment_id, "block_ids": validation_block_ids,
                }
                candidate, _ = _canonicalize_translation_opaque_candidates(
                    draft["translation"], blocks_by_id,
                )
                candidate, missing_blocks, _ = _normalize_translation_coverage(
                    validation_segment, candidate, blocks_by_id,
                )
                if missing_blocks:
                    raise RuntimeError(
                        "paid token repair cannot recover missing translation blocks"
                    )
                source_blocks = [blocks_by_id[value] for value in source_ids]
                normalization_artifact_dir = (
                    _generation_segment_artifact_dir(
                        checkpoint_dir, "llm/translations", segment_id,
                        marker_generation,
                    )
                    / marker_path.stem
                    / "retry-offset-1"
                )
                receipt_path = _repair_response_normalization_path(
                    normalization_artifact_dir,
                    marker_path,
                    checkpoint_dir=checkpoint_dir,
                    persisted_response=True,
                )

                def apply_and_validate_paid_response(
                    response: Mapping[str, Any],
                ) -> dict[str, Any]:
                    repaired_candidate = _apply_translation_slot_repairs(
                        candidate,
                        source_blocks,
                        dict(response),
                        protected_names=protected_names,
                        offset_only=True,
                    )
                    for source_id in source_ids:
                        source = blocks_by_id[source_id]
                        prior_text = next(
                            str(item.get("text") or "")
                            for item in candidate.get("blocks") or []
                            if isinstance(item, dict)
                            and item.get("block_id") == source_id
                        )
                        repaired_text = next(
                            str(item.get("text") or "")
                            for item in repaired_candidate.get("blocks") or []
                            if isinstance(item, dict)
                            and item.get("block_id") == source_id
                        )
                        if (
                            _translation_natural_residue(prior_text)
                            != _translation_natural_residue(repaired_text)
                            or _OPAQUE_INLINE_PATTERN.findall(repaired_text)
                            != _opaque_inline_tokens(source)
                        ):
                            raise RuntimeError(
                                "deterministic repair replay changed residue or token order"
                            )
                    _validate_translation(
                        validation_segment,
                        repaired_candidate,
                        blocks_by_id,
                        protected_names,
                    )
                    return repaired_candidate

                projected_response, _ = _normalize_translation_repair_response(
                    raw_response,
                    expected_ids=source_ids,
                    schema=TRANSLATION_SLOT_REPAIR_SCHEMA,
                    schema_version=TRANSLATION_SLOT_REPAIR_SCHEMA_VERSION,
                    invariant_validator=apply_and_validate_paid_response,
                    receipt_path=receipt_path,
                )
                repaired = apply_and_validate_paid_response(projected_response)
            except (RuntimeError, StopIteration) as exc:
                error = str(exc)
        call_checkpoint = binding["checkpoint"]
        if call_checkpoint.get("submission_state") not in {"submitted", "unknown"}:
            continue
        logical = dict(call_checkpoint.get("logical_identity") or {})
        context = {
            "submission_state": str(
                call_checkpoint.get("submission_state") or "submitted"
            ),
            "resumable": False,
            "recovery_action": (
                "operator-supervision" if error else "deterministic-replay"
            ),
            "repair_marker": str(marker_path),
            "checkpoint_path": str(binding["checkpoint_path"]),
            "idempotency_key": str(logical.get("idempotency_key") or ""),
            "submission_receipt": dict(binding["receipt_reference"]),
            "blocked_reason": (
                f"persisted_repair_response_failed_local_validation: {error}"
                if error else "paid_repair_ready_for_deterministic_replay"
            ),
            **(
                {
                    "response_normalization": _response_normalization_reference(
                        receipt_path, checkpoint_dir=checkpoint_dir,
                    )
                }
                if receipt_path is not None and receipt_path.is_file()
                else {}
            ),
        }
        if error:
            try:
                _current_ledger, current_digest = read_registered_lane_ledger(
                    checkpoint_dir, ledger_path,
                )
                mark_needs_supervision(
                    ledger_path, segment_id=segment_id,
                    reason=context["blocked_reason"], recovery_context=context,
                    expected_ledger_sha256=current_digest,
                    checkpoint_dir=checkpoint_dir,
                )
            except (LaneLedgerError, LaneLedgerRegistryError):
                continue
        entries.append({
            "ledger_path": str(ledger_path),
            "session_key": f"{ledger.get('chapter_id')}:{ledger.get('lane')}",
            "segment_id": segment_id,
            "idempotency_key": context["idempotency_key"],
            "initial_generation": block_generation,
            "target_generation": block_generation,
            "recovery_action": context["recovery_action"],
            "blocking_reason": context["blocked_reason"] if error else "",
            "recovery_context": context,
        })
    coverage_marker_paths = [
        (segment_id, _translation_coverage_attempt_path(
            checkpoint_dir,
            segment_id,
            int(binding["receipt"].get("generation") or 1),
        ))
        for (unit, segment_id), binding in repair_bindings.items()
        if unit == "translation-coverage-repair"
    ]
    for segment_id, marker_path in sorted(coverage_marker_paths):
        binding = repair_bindings[("translation-coverage-repair", segment_id)]
        try:
            marker = _read_recovery_json(checkpoint_dir, marker_path)
        except SecureReadError:
            continue
        if (
            not isinstance(marker, Mapping)
            or marker.get("status") not in {"response_received", "validated"}
            or not isinstance(marker.get("raw_response"), Mapping)
        ):
            continue
        if str(marker.get("segment_id") or "") != segment_id:
            continue
        located = ledgers.get(segment_id)
        if located is None:
            continue
        ledger_path, ledger, ledger_block = located
        block_generation = int(
            ledger_block.get("generation") or ledger.get("generation") or 1
        )
        try:
            marker_generation = _artifact_payload_generation(
                marker,
                checkpoint_dir,
                "translation-coverage-attempts",
                segment_id,
            )
        except (TypeError, ValueError):
            continue
        if marker_generation != block_generation:
            continue
        call_checkpoint = binding["checkpoint"]
        if call_checkpoint.get("submission_state") not in {"submitted", "unknown"}:
            continue
        logical = dict(call_checkpoint.get("logical_identity") or {})
        context = {
            "submission_state": str(
                call_checkpoint.get("submission_state") or "submitted"
            ),
            "resumable": False,
            "recovery_action": "deterministic-replay",
            "repair_marker": str(marker_path),
            "checkpoint_path": str(binding["checkpoint_path"]),
            "idempotency_key": str(logical.get("idempotency_key") or ""),
            "submission_receipt": dict(binding["receipt_reference"]),
            "blocked_reason": "paid_coverage_repair_ready_for_normal_validation",
        }
        entries.append({
            "ledger_path": str(ledger_path),
            "session_key": f"{ledger.get('chapter_id')}:{ledger.get('lane')}",
            "segment_id": segment_id,
            "idempotency_key": context["idempotency_key"],
            "initial_generation": block_generation,
            "target_generation": block_generation,
            "recovery_action": "deterministic-replay",
            "blocking_reason": "",
            "recovery_context": context,
        })
    return entries


def resume_companion(
    project_dir: Path,
    *,
    action: str = "auto",
    confirm_possible_duplicate_charge: bool = False,
) -> dict[str, Any]:
    """Resolve supervision and continue while retaining the project lock."""

    resolved = project_dir.resolve()
    lock = ProjectBuildLock(resolved / ".arc-companion-build.lock")
    try:
        lock.acquire()
    except BuildInProgressError as exc:
        return err(
            "build_in_progress",
            str(exc),
            project_dir=str(resolved),
            retryable=True,
        )
    try:
        from arc_llm.cancellation import install_signal_cancel_chain

        with install_signal_cancel_chain() as cancel_check:
            return _resume_companion_unlocked(
                resolved,
                action=action,
                confirm_possible_duplicate_charge=confirm_possible_duplicate_charge,
                cancel_check=cancel_check,
            )
    finally:
        lock.release()


def _resume_companion_unlocked(
    project_dir: Path,
    *,
    action: str,
    confirm_possible_duplicate_charge: bool = False,
    cancel_check: Callable[[], bool] | None = None,
    continuation: Callable[[BuildOptions], dict[str, Any]] | None = None,
    source_preflight: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Mutate supervised state and continue under the caller's project lock."""
    if action not in {"auto", "resume-native", "restart-generation"}:
        raise ValueError("action must be auto, resume-native, or restart-generation")
    native_mode = action in {"auto", "resume-native"}
    if action == "restart-generation" and not confirm_possible_duplicate_charge:
        return {
            "ok": False,
            "status": "needs_supervision",
            "data": None,
            "error": {
                "code": "duplicate_charge_confirmation_required",
                "message": (
                    "restart-generation may repeat a submitted paid call; pass "
                    "--confirm-possible-duplicate-charge to continue"
                ),
            },
            "errors": [],
            "meta": {"project_dir": str(project_dir.resolve()), "action": action},
        }
    state_path = project_dir.resolve() / "state.json"
    if not state_path.is_file():
        return err(
            "companion_state_not_found",
            f"No companion state found in {project_dir}",
            action=action,
        )
    state = read_json(state_path)
    if state.get("status") not in {"needs_supervision", "failed"}:
        return err("companion_not_supervised", "The companion build does not need supervision")
    active_run = state.get("active_run")
    active_run = active_run if isinstance(active_run, dict) else {}
    checkpoint_dir = _checkpoint_dir_from_recovery_state(project_dir, state)
    if not checkpoint_dir.is_dir():
        return err("companion_checkpoint_not_found", "The supervised checkpoint directory is missing")
    from .resume_transaction import (
        append_entries,
        begin_transaction,
        bind_transaction_checkpoint,
        load_transaction,
        mark_entry,
        mark_transaction,
    )

    checkpoint_binding = bind_transaction_checkpoint(
        project_dir,
        checkpoint_path=checkpoint_dir,
        checkpoint_fingerprint=str(state.get("fingerprint") or checkpoint_dir.name),
    )
    try:
        transaction = load_transaction(project_dir)
    except ValueError as exc:
        return err("resume_transaction_invalid", str(exc))
    if transaction and transaction.get("status") != "complete":
        if (
            action == "restart-generation"
            and transaction.get("action") == "auto"
        ):
            try:
                from .resume_transaction import authorize_manual_restart

                transaction = authorize_manual_restart(project_dir)
            except ValueError as exc:
                return err("resume_transaction_invalid", str(exc))
        if transaction.get("action") != action and not (
            action == "auto" and transaction.get("action") == "resume-native"
        ):
            return err(
                "resume_transaction_action_mismatch",
                "An incomplete resume transaction uses a different action",
            )
        if action == "auto" and transaction.get("action") == "resume-native":
            try:
                from .resume_transaction import upgrade_transaction_action

                transaction = upgrade_transaction_action(
                    project_dir, action="auto", policy="auto",
                    reason="automatic recovery resumed an incomplete native transaction",
                )
            except (ImportError, ValueError) as exc:
                return err("resume_transaction_invalid", str(exc))
        transaction_ledger_paths = {
            Path(str(item["ledger_path"])).resolve(strict=False)
            for item in transaction.get("entries") or []
            if item.get("ledger_path")
        }
        if action == "auto" and any(
            not lane_ledger_is_registered(checkpoint_dir, path)
            for path in transaction_ledger_paths
        ):
            return err(
                "automatic_recovery_ledger_unregistered",
                "Automatic recovery transaction contains a ledger outside the active control index",
                provider_calls=0,
            )
        ledger_paths = set(transaction_ledger_paths)
        # A continuation may itself be interrupted after another provider call
        # crossed the barrier. Discover it on every retry instead of freezing
        # the transaction's initial inventory.
        ledger_paths.update(
            path.resolve(strict=False) for path in _all_lane_ledger_paths(
                checkpoint_dir, include_explicit_legacy=(action == "resume-native"),
            )
        )
    else:
        transaction = None
        ledger_paths = {
            path.resolve(strict=False) for path in _all_lane_ledger_paths(
                checkpoint_dir, include_explicit_legacy=(action == "resume-native"),
            )
        }
    if transaction is not None:
        _backfill_legacy_generation_owners(checkpoint_dir, transaction)
    supervised_ledgers = []
    transaction_paths = {
        str(Path(str(item.get("ledger_path") or "")).resolve(strict=False))
        for item in (transaction or {}).get("entries") or []
    }
    for path in sorted(ledger_paths):
        try:
            ledger = _read_recovery_ledger(
                checkpoint_dir,
                path,
                allow_explicit_legacy=(action == "resume-native"),
            )
        except SecureReadError:
            continue
        if ledger.get("needs_supervision") or str(path) in transaction_paths:
            supervised_ledgers.append((path, ledger))
    saved = (
        transaction.get("recovery_options")
        if transaction else state.get("recovery_options") or active_run.get("recovery_options")
    )
    if not isinstance(saved, dict):
        return err(
            "companion_recovery_options_missing",
            "This supervised build predates resumable option checkpoints; rerun build explicitly",
        )

    from arc_llm.sessions import LLMSessionManager
    manager = LLMSessionManager(checkpoint_dir / "sessions")
    fresh_required_keys: set[str] = set()
    fresh_required_identities: set[tuple[str, str, str, int, str]] = set()
    durable_reconstruction_identities: set[
        tuple[str, str, str, int, str]
    ] = set()
    durable_replay_keys: set[str] = set()
    durable_replay_identities: set[tuple[str, str, str, int, str]] = set()
    if action == "auto":
        audit_candidates: list[dict[str, Any]] = [
            dict(item) for item in (transaction or {}).get("entries") or []
            if isinstance(item, Mapping)
        ]
        for ledger_path, ledger in supervised_ledgers:
            for supervision in _active_supervision_entries(ledger):
                context = dict(supervision.get("recovery_context") or {})
                audit_candidates.append({
                    "ledger_path": str(ledger_path),
                    "session_key": f"{ledger.get('chapter_id')}:{ledger.get('lane')}",
                    "segment_id": str(supervision.get("segment_id") or ""),
                    "idempotency_key": str(context.get("idempotency_key") or ""),
                    "initial_generation": int(ledger.get("generation") or 0),
                    "recovery_context": context,
                })
        for candidate in audit_candidates:
            if _entry_recovery_address_blocker(candidate, checkpoint_dir) is not None:
                continue
            try:
                _promote_entry_raw_response(candidate, checkpoint_dir)
            except BaseException as exc:
                from arc_llm import LLMWorkerError

                if isinstance(exc, (RecoveryResponseError, LLMWorkerError)):
                    return err(
                        "recovery_candidate_invalid",
                        str(exc),
                        provider_calls=0,
                        automatic_generation_restart=False,
                    )
                raise
            audit = _fresh_auto_recovery_audit(candidate, checkpoint_dir)
            key = str(candidate.get("idempotency_key") or "")
            identity = (
                str(Path(str(candidate.get("ledger_path") or "")).resolve(strict=False)),
                str(candidate.get("session_key") or ""),
                str(candidate.get("segment_id") or ""),
                int(
                    candidate.get("initial_generation")
                    or (candidate.get("recovery_context") or {}).get("generation")
                    or 0
                ),
                key,
            )
            if audit["fresh_generation_required"]:
                if key:
                    fresh_required_keys.add(key)
                fresh_required_identities.add(identity)
                checkpoint_path, checkpoint = _entry_call_checkpoint(
                    candidate, checkpoint_dir,
                )
                if checkpoint_path is not None and checkpoint is not None:
                    durable_reconstruction_identities.add(identity)
            elif bool(audit["complete_response_scan"]["complete"]):
                if key:
                    durable_replay_keys.add(key)
                durable_replay_identities.add(identity)
    native_resume_contexts: list[dict[str, Any]] = []
    for raw_context in (
        transaction.get("native_resume_contexts") or [] if transaction else []
    ):
        context = dict(raw_context)
        session_key = str(context.get("session_key") or "")
        ref = manager.get_existing(session_key) if session_key else None
        ledger_path = Path(str(context.get("ledger_path") or ""))
        ledger_generation = None
        try:
            ledger_generation = int(
                _read_recovery_ledger(
                    checkpoint_dir,
                    ledger_path,
                    allow_explicit_legacy=(action == "resume-native"),
                ).get("generation") or 0
            )
        except SecureReadError:
            ledger_generation = None
        if (
            ref is not None
            and int(context.get("generation") or 0) == ref.generation
            and (ledger_generation is None or ledger_generation == ref.generation)
            and _native_context_recovery_identity(context) not in (
                fresh_required_identities | durable_replay_identities
            )
        ):
            native_resume_contexts.append(context)
    native_resume_identities: list[tuple[str, str, str, int, str]] = [
        _native_context_recovery_identity(item) for item in native_resume_contexts
    ]
    reconstructed_fresh_entries: list[dict[str, Any]] = []
    blocked_native_entries: dict[tuple[str, str, str], dict[str, Any]] = {}
    if native_mode:
        for ledger_path, ledger in supervised_ledgers:
            for supervision in _active_supervision_entries(ledger):
                context = dict(supervision.get("recovery_context") or {})
                session_key = f"{ledger['chapter_id']}:{ledger['lane']}"
                segment_id = str(supervision.get("segment_id") or "")
                entry_identity = (
                    str(ledger_path.resolve(strict=False)), session_key, segment_id,
                )
                context_key = str(context.get("idempotency_key") or "")
                recovery_identity = (
                    str(ledger_path.resolve(strict=False)), session_key, segment_id,
                    int(ledger.get("generation") or 0),
                    context_key,
                )
                if action == "auto" and (
                    recovery_identity in fresh_required_identities
                    or recovery_identity in durable_replay_identities
                ):
                    continue
                if recovery_identity in {
                    _native_context_recovery_identity(item)
                    for item in native_resume_contexts
                }:
                    continue
                if not context.get("idempotency_key"):
                    blocked_native_entries[entry_identity] = {
                        "blocking_code": "native_resume_idempotency_key_missing",
                        "blocking_reason": (
                            f"The supervised call for {session_key} has no logical call key"
                        ),
                        "recovery_context": context,
                    }
                    continue
                if not context.get("resumable") or not (
                    context.get("native_session_id") or manager.has_native_session(session_key)
                ):
                    blocked_native_entries[entry_identity] = {
                        "blocking_code": "native_session_not_resumable",
                        "blocking_reason": (
                            f"The supervised call for {session_key} has no resumable native session"
                        ),
                        "recovery_context": context,
                    }
                    continue
                try:
                    validated = _validate_native_resume_context(
                        checkpoint_dir=checkpoint_dir,
                        ledger_path=ledger_path,
                        ledger=ledger,
                        session_manager=manager,
                        supervision=supervision,
                        allow_explicit_legacy=(action == "resume-native"),
                    )
                except ValueError as exc:
                    blocked_native_entries[entry_identity] = {
                        "blocking_code": "native_resume_context_invalid",
                        "blocking_reason": str(exc),
                        "recovery_context": context,
                    }
                    continue
                native_resume_contexts.append(validated)
                native_resume_identities.append(
                    _native_context_recovery_identity(validated)
                )
        reconstructed = _reconstruct_unresolved_native_resume_contexts(
            checkpoint_dir,
            session_manager=manager,
            excluded_keys=set(),
            excluded_identities={
                *{
                    _native_context_recovery_identity(item)
                    for item in native_resume_contexts
                },
                *fresh_required_identities,
                *durable_replay_identities,
            },
            suppress_typed_idle=(action == "auto"),
            fresh_required_entries=reconstructed_fresh_entries,
            allow_explicit_legacy=(action == "resume-native"),
        )
        for item in reconstructed_fresh_entries:
            key = str(item.get("idempotency_key") or "")
            if key:
                fresh_required_keys.add(key)
            fresh_required_identities.add((
                str(Path(str(item.get("ledger_path") or "")).resolve(strict=False)),
                str(item.get("session_key") or ""),
                str(item.get("segment_id") or ""),
                int(item.get("initial_generation") or 0),
                key,
            ))
        native_resume_contexts.extend(reconstructed)
        native_resume_identities.extend(
            _native_context_recovery_identity(item) for item in reconstructed
        )
    entries = []
    existing_entry_identities = {
        (
            str(Path(str(item.get("ledger_path") or "")).resolve(strict=False)),
            str(item.get("session_key") or ""),
            str(item.get("segment_id") or ""),
        )
        for item in (transaction or {}).get("entries") or []
    }
    for path, ledger in supervised_ledgers:
        for supervision in _active_supervision_entries(ledger):
            segment_id = str(supervision.get("segment_id") or "")
            if not segment_id:
                continue
            identity = (
                str(path.resolve(strict=False)),
                f"{ledger['chapter_id']}:{ledger['lane']}", segment_id,
            )
            if identity in existing_entry_identities:
                continue
            generation = int(ledger.get("generation") or 1)
            context = dict(supervision.get("recovery_context") or {})
            entry = {
                "ledger_path": str(path),
                "session_key": identity[1],
                "segment_id": segment_id,
                "idempotency_key": str(context.get("idempotency_key") or ""),
                "initial_generation": generation,
                "target_generation": generation + (
                    1 if action == "restart-generation" else 0
                ),
                "recovery_context": context,
                "recovery_action": str(
                    context.get("recovery_action") or ""
                ),
                "blocking_reason": str(
                    supervision.get("reason") or context.get("blocked_reason") or ""
                ),
                "supervision_reason": str(supervision.get("reason") or ""),
            }
            if action == "auto":
                entry.update(_fresh_auto_recovery_audit(entry, checkpoint_dir))
                recovery_identity = (
                    identity[0], identity[1], identity[2], generation,
                    str(entry.get("idempotency_key") or ""),
                )
                if recovery_identity in durable_reconstruction_identities:
                    entry["reconstructed_from_durable_state"] = True
            entry.update(blocked_native_entries.get(identity) or {})
            entries.append(entry)
            existing_entry_identities.add(identity)
    pending_entry_positions = {
        (
            str(Path(str(item.get("ledger_path") or "")).resolve(strict=False)),
            str(item.get("session_key") or ""),
            str(item.get("segment_id") or ""),
        ): index
        for index, item in enumerate(entries)
    }
    for reconstructed_entry in reconstructed_fresh_entries:
        identity = (
            str(Path(str(reconstructed_entry.get("ledger_path") or "")).resolve(strict=False)),
            str(reconstructed_entry.get("session_key") or ""),
            str(reconstructed_entry.get("segment_id") or ""),
        )
        if not all(identity):
            continue
        existing_position = pending_entry_positions.get(identity)
        if existing_position is not None:
            # Durable reconstruction may discover the exact same supervised
            # task that was seeded from the ledger a few lines above.  Merge
            # its stronger audit facts into that authorization entry instead
            # of either duplicating the task or dropping the reconstruction
            # evidence.
            entries[existing_position] = {
                **entries[existing_position],
                **dict(reconstructed_entry),
            }
        else:
            # Keep this enrichment even when the identity already exists in a
            # prior transaction.  append_entries() performs an idempotent
            # identity merge while preserving resolved transaction status.
            entries.append(dict(reconstructed_entry))
            pending_entry_positions[identity] = len(entries) - 1
        existing_entry_identities.add(identity)
    if native_mode:
        for context in native_resume_contexts:
            ledger_path = str(
                Path(str(context.get("ledger_path") or "")).resolve(strict=False)
            ) if context.get("ledger_path") else ""
            segment_id = str(context.get("segment_id") or "")
            session_key = str(context.get("session_key") or "")
            identity = (ledger_path, session_key, segment_id)
            if not all(identity) or identity in existing_entry_identities:
                continue
            generation = int(context.get("generation") or 1)
            entries.append({
                "ledger_path": ledger_path,
                "session_key": session_key,
                "segment_id": segment_id,
                "idempotency_key": str(context.get("idempotency_key") or ""),
                "initial_generation": generation,
                "target_generation": generation,
                "reconstructed_from_durable_state": bool(
                    context.get("reconstructed_from_durable_state")
                ),
            })
            existing_entry_identities.add(identity)
    if transaction is None and not entries:
        return err("companion_not_supervised", "The companion build does not need supervision")
    if transaction is None:
        try:
            transaction = begin_transaction(
                project_dir,
                action=action,
                recovery_options=saved,
                entries=entries,
                native_resume_contexts=native_resume_contexts,
                checkpoint_path=checkpoint_dir,
                checkpoint_fingerprint=str(
                    state.get("fingerprint") or checkpoint_dir.name
                ),
                authorization_source=(
                    "recovery_policy_auto" if action == "auto" else "operator"
                ),
            )
        except ValueError as exc:
            return err("resume_transaction_invalid", str(exc))
    elif entries or len(native_resume_contexts) != len(
        transaction.get("native_resume_contexts") or []
    ):
        try:
            transaction = append_entries(
                project_dir, entries,
                native_resume_contexts=native_resume_contexts,
            )
        except ValueError as exc:
            return err("resume_transaction_invalid", str(exc))
    if action == "auto":
        for index, raw in enumerate(transaction.get("entries") or []):
            entry = dict(raw)
            audit_entry = _entry_with_current_supervision(
                entry, checkpoint_dir=checkpoint_dir,
            )
            audit = _fresh_auto_recovery_audit(audit_entry, checkpoint_dir)
            if audit["fresh_generation_required"]:
                audit["fresh_task_start_segment_id"] = _first_nonaccepted_segment_id(
                    entry, checkpoint_dir=checkpoint_dir,
                )
            else:
                audit["fresh_task_start_segment_id"] = None
            if audit["complete_response_scan"].get("source") == "accepted_lane_block":
                ledger_value = _read_recovery_ledger(
                    checkpoint_dir, Path(str(entry.get("ledger_path") or "")),
                )
                accepted_block = next(
                    item for item in ledger_value.get("blocks") or []
                    if isinstance(item, Mapping)
                    and str(item.get("segment_id") or "")
                    == str(entry.get("segment_id") or "")
                )
                transaction = mark_entry(
                    project_dir, index, status="resolved", **audit,
                    accepted_chain_sha256=str(
                        accepted_block.get("accepted_chain_sha256") or ""
                    ),
                    output_sha256=str(accepted_block.get("output_sha256") or ""),
                )
                continue
            transaction = mark_entry(
                project_dir,
                index,
                status=str(entry.get("status") or "pending"),
                **audit,
            )
        fresh_required_keys.update(
            str(item.get("idempotency_key") or "")
            for item in transaction.get("entries") or []
            if item.get("fresh_generation_required")
            and item.get("idempotency_key")
        )
        durable_replay_keys.update(
            str(item.get("idempotency_key") or "")
            for item in transaction.get("entries") or []
            if item.get("durable_replay_required")
            and item.get("idempotency_key")
        )
        fresh_required_identities.update(
            _native_context_recovery_identity(item)
            for item in transaction.get("entries") or []
            if item.get("fresh_generation_required")
        )
        durable_replay_identities.update(
            _native_context_recovery_identity(item)
            for item in transaction.get("entries") or []
            if item.get("durable_replay_required")
        )
        native_excluded_identities = (
            fresh_required_identities | durable_replay_identities
        )
        native_resume_contexts = [
            item for item in native_resume_contexts
            if _native_context_recovery_identity(item) not in native_excluded_identities
        ]
        native_resume_identities = [
            _native_context_recovery_identity(item) for item in native_resume_contexts
        ]
        if native_excluded_identities:
            from .resume_transaction import suppress_native_resume_contexts

            transaction = suppress_native_resume_contexts(
                project_dir, idempotency_keys=set(),
                recovery_identities=native_excluded_identities,
                reason="automatic_recovery_uses_no_old_native_session",
            )
        if any(
            _native_context_recovery_identity(item) in native_excluded_identities
            for item in native_resume_contexts
        ):
            raise RuntimeError("fresh idle recovery leaked into native reconciliation")
    resumed: list[dict[str, Any]] = []
    reconcilable_native_entries = 0
    manual_restart_groups: dict[str, dict[str, Any]] = {}
    if action == "restart-generation":
        for raw in transaction.get("entries") or []:
            if raw.get("status") == "resolved":
                continue
            session_key = str(raw.get("session_key") or "")
            path = Path(str(raw.get("ledger_path") or ""))
            if not session_key:
                continue
            try:
                ledger = _read_recovery_ledger(
                    checkpoint_dir, path, allow_explicit_legacy=True,
                )
            except SecureReadError:
                continue
            ordered_ids = [
                str(item.get("segment_id") or "")
                for item in ledger.get("blocks") or []
            ]
            positions = {value: index for index, value in enumerate(ordered_ids)}
            segment_id = str(raw.get("segment_id") or "")
            candidate = {
                "ledger_path": path,
                "source_generation": int(raw.get("initial_generation") or ledger.get("generation") or 1),
                "target_generation": int(raw.get("target_generation") or (int(ledger.get("generation") or 1) + 1)),
                "suffix_start_segment_id": segment_id,
                "position": positions.get(segment_id, len(ordered_ids)),
            }
            current = manual_restart_groups.get(session_key)
            if current is None or candidate["position"] < current["position"]:
                manual_restart_groups[session_key] = candidate
    for index, entry in enumerate(transaction.get("entries") or []):
        path = Path(str(entry["ledger_path"]))
        ledger = _read_recovery_ledger(
            checkpoint_dir, path,
            allow_explicit_legacy=(action == "resume-native"),
        )
        segment_id = str(entry.get("segment_id") or "")
        session_key = str(entry["session_key"])
        if entry.get("status") == "resolved":
            resumed.append(dict(entry))
            continue
        if native_mode:
            recovery_action = str(entry.get("recovery_action") or "")
            if action == "auto" and entry.get("fresh_generation_required"):
                continue
            if action == "auto" and entry.get("durable_replay_required"):
                reconcilable_native_entries += 1
                if entry.get("status") in {"pending", "authorized"}:
                    mark_entry(
                        project_dir, index, status="reconciling",
                        recovery_action="durable-business-replay",
                    )
                resumed.append(dict(entry))
                continue
            if recovery_action == "deterministic-replay":
                reconcilable_native_entries += 1
                if entry.get("status") in {"pending", "authorized"}:
                    mark_entry(
                        project_dir, index, status="reconciling",
                        recovery_action=recovery_action,
                    )
                resumed.append(dict(entry))
                continue
            if recovery_action == "operator-supervision":
                # The response is already paid and locally proven invalid.
                # Retain its per-call authorization/context without attempting
                # native reconciliation or a normal resend.
                continue
            entry_key = str(entry.get("idempotency_key") or "")
            entry_recovery_identity = _native_context_recovery_identity(entry)
            validated = next((
                item for item in native_resume_contexts
                if _native_context_recovery_identity(item) == entry_recovery_identity
            ), None)
            if (
                validated is None
                and not entry_key
                and transaction.get("migrated_from")
                in {"arc.companion.resume-transaction.v1", "arc.companion.resume-transaction.v2"}
            ):
                legacy_matches = [
                    item for item in native_resume_contexts
                    if str(item.get("session_key") or "") == session_key
                    and str(item.get("logical_unit") or item.get("segment_id") or "")
                    == segment_id
                    and str(Path(str(item.get("ledger_path") or "")).resolve(strict=False))
                    == str(path.resolve(strict=False))
                ]
                if len(legacy_matches) == 1:
                    validated = legacy_matches[0]
            if validated is None:
                identity = (str(path.resolve(strict=False)), session_key, segment_id)
                blocked = blocked_native_entries.get(identity) or {}
                reason = str(
                    blocked.get("blocking_reason")
                    or entry.get("blocking_reason")
                    or f"No durable native authorization exists for {session_key}:{segment_id}"
                )
                mark_entry(
                    project_dir, index,
                    status=str(entry.get("status") or "pending"),
                    blocking_code=str(
                        blocked.get("blocking_code")
                        or entry.get("blocking_code") or ""
                    ),
                    blocking_reason=reason,
                    recovery_context=dict(
                        blocked.get("recovery_context")
                        or entry.get("recovery_context")
                        or {}
                    ),
                )
                continue
            reconcilable_native_entries += 1
            native_id = validated.get("native_session_id_to_restore")
            if native_id:
                try:
                    _restore_native_session_id(manager, validated, str(native_id))
                except ValueError as exc:
                    return err("native_resume_context_invalid", str(exc))
            updated = ledger
            if entry.get("status") == "pending":
                mark_entry(
                    project_dir, index, status="authorized",
                    idempotency_key=validated["idempotency_key"],
                    generation=validated["generation"],
                )
        else:
            group = manual_restart_groups[session_key]
            ref = manager.get_existing(session_key)
            if ref is None:
                return err(
                    "native_session_not_found",
                    f"No saved logical session exists for {session_key}",
                )
            initial_generation = int(group["source_generation"])
            target_generation = int(group["target_generation"])
            suffix_start = str(group["suffix_start_segment_id"])
            ordered_ids = [
                str(item.get("segment_id") or "")
                for item in ledger.get("blocks") or []
            ]
            suffix_position = ordered_ids.index(suffix_start)
            _record_legacy_generation_owners(
                checkpoint_dir,
                lane=session_key.rsplit(":", 1)[-1],
                segment_ids=ordered_ids[suffix_position:],
                generation=initial_generation,
            )
            if ref.generation == initial_generation:
                ref = manager.rotate(session_key, reason="supervised restart-generation")
            elif ref.generation != target_generation:
                return err(
                    "resume_transaction_generation_mismatch",
                    f"Session {session_key} changed outside the resume transaction",
                )
            updated = read_json(path)
            if not group.get("applied"):
                updated = (
                    invalidate_suffix(
                        path,
                        from_segment_id=str(group["suffix_start_segment_id"]),
                        generation=target_generation,
                    )
                    if int(updated.get("generation") or 0) != target_generation
                    or updated.get("needs_supervision")
                    else updated
                )
                group["applied"] = True
            mark_entry(
                project_dir, index, status="reconciling",
                ledger_path=str(path), session_key=session_key,
                segment_id=segment_id, generation=updated["generation"],
                target_generation=target_generation,
                suffix_start_segment_id=str(group["suffix_start_segment_id"]),
            )
            resumed.append(dict(entry))
            continue
        receipt = {
            "ledger_path": str(path), "session_key": session_key,
            "segment_id": segment_id, "generation": updated["generation"],
            "idempotency_key": validated["idempotency_key"],
        }
        if entry.get("status") in {"pending", "authorized"}:
            mark_entry(project_dir, index, status="reconciling", **receipt)
        resumed.append(receipt)

    if action == "resume-native" and reconcilable_native_entries == 0:
        mark_transaction(project_dir, "continuation_failed")
        return err(
            str(next(iter(blocked_native_entries.values()), {}).get(
                "blocking_code", "native_resume_authorization_missing"
            )),
            str(next(iter(blocked_native_entries.values()), {}).get(
                "blocking_reason", "No pending call has a durable native resume authorization"
            )),
            pending_calls=len(transaction.get("entries") or []),
        )

    mark_transaction(project_dir, "continuing")

    options = _options_from_recovery(project_dir.resolve(), saved)
    if native_mode:
        options = replace(
            options,
            supervised_native_resume_identities=tuple(
                identity
                for identity in dict.fromkeys(native_resume_identities)
                if _native_recovery_identity_complete(identity)
            ),
        )
    # Production continuation must not release and reacquire the project lock:
    # that would expose partially coordinated session/ledger mutations. Tests
    # may replace ``build_companion`` with a capture stub, which is safe to call
    # because it does not acquire an OS lock.
    should_continue_native = not (
        action == "auto" and reconcilable_native_entries == 0
    )
    if should_continue_native:
        result = _continue_supervised_build(
            options, cancel_check=cancel_check, continuation=continuation,
        )
    else:
        result = {
            "ok": False,
            "status": "needs_supervision",
            "data": state,
            "error": {
                "code": "native_resume_authorization_missing",
                "message": "No pending call has a durable native resume authorization",
            },
            "errors": [],
            "meta": {},
        }
    if action == "auto":
        _accept_completed_pipeline_controls(checkpoint_dir)
    if native_mode:
        # Calls submitted by the continuation can finish after an earlier
        # source-order block has failed.  Classify their paid repair responses
        # directly from durable state so none become orphan checkpoints.
        finalized_repairs = _finalize_paid_translation_repairs(checkpoint_dir)
        if finalized_repairs:
            append_entries(project_dir, finalized_repairs)
        latest = load_transaction(project_dir) or transaction
        known = {
            (
                str(Path(str(item.get("ledger_path") or "")).resolve(strict=False)),
                str(item.get("session_key") or ""),
                str(item.get("segment_id") or ""),
            )
            for item in latest.get("entries") or []
        }
        discovered_entries: list[dict[str, Any]] = []
        discovered_contexts: list[dict[str, Any]] = []
        for path in _all_lane_ledger_paths(
            checkpoint_dir, include_explicit_legacy=(action == "resume-native"),
        ):
            ledger = _read_recovery_ledger(
                checkpoint_dir, path,
                allow_explicit_legacy=(action == "resume-native"),
            )
            session_key = f"{ledger.get('chapter_id')}:{ledger.get('lane')}"
            for supervision in _active_supervision_entries(ledger):
                segment_id = str(supervision.get("segment_id") or "")
                identity = (str(path.resolve(strict=False)), session_key, segment_id)
                if not segment_id or identity in known:
                    continue
                context = dict(supervision.get("recovery_context") or {})
                entry = {
                    "ledger_path": str(path),
                    "session_key": session_key,
                    "segment_id": segment_id,
                    "idempotency_key": str(context.get("idempotency_key") or ""),
                    "initial_generation": int(ledger.get("generation") or 1),
                    "target_generation": int(ledger.get("generation") or 1),
                    "recovery_context": context,
                    "recovery_action": str(context.get("recovery_action") or ""),
                    "blocking_reason": str(
                        supervision.get("reason") or context.get("blocked_reason") or ""
                    ),
                    "supervision_reason": str(supervision.get("reason") or ""),
                }
                # Only the primary lane stop is eligible for exact native
                # validation; additional drained failures remain durable and
                # are grouped into the same suffix replacement if needed.
                if supervision == ledger.get("needs_supervision"):
                    try:
                        discovered_contexts.append(_validate_native_resume_context(
                            checkpoint_dir=checkpoint_dir,
                            ledger_path=path,
                            ledger=ledger,
                            session_manager=manager,
                            allow_explicit_legacy=(action == "resume-native"),
                        ))
                    except ValueError as exc:
                        entry["blocking_reason"] = str(exc)
                discovered_entries.append(entry)
                known.add(identity)
        if discovered_entries:
            append_entries(
                project_dir, discovered_entries,
                native_resume_contexts=discovered_contexts,
            )
    if native_mode or action == "restart-generation":
        # Supervision is cleared only after the recovered response has been
        # durably validated and accepted into the lane ledger.
        current_transaction = load_transaction(project_dir) or transaction
        for index, entry in enumerate(current_transaction.get("entries") or []):
            if entry.get("status") == "resolved":
                continue
            path = Path(str(entry["ledger_path"]))
            try:
                ledger, ledger_digest = read_registered_lane_ledger(
                    checkpoint_dir, path,
                )
            except LaneLedgerRegistryError:
                if action != "resume-native":
                    continue
                try:
                    ledger = _read_recovery_ledger(
                        checkpoint_dir, path, allow_explicit_legacy=True,
                    )
                except SecureReadError:
                    continue
                ledger_digest = None
            block = next((
                item for item in ledger.get("blocks") or []
                if str(item.get("segment_id") or "") == str(entry.get("segment_id") or "")
            ), None)
            if isinstance(block, dict) and block.get("state") == "accepted":
                if str(
                    (ledger.get("needs_supervision") or {}).get("segment_id") or ""
                ) == str(entry.get("segment_id") or ""):
                    clear_needs_supervision(
                        path,
                        expected_ledger_sha256=ledger_digest,
                        checkpoint_dir=(
                            checkpoint_dir if ledger_digest is not None else None
                        ),
                    )
                mark_entry(
                    project_dir, index, status="resolved",
                    accepted_chain_sha256=str(
                        block.get("accepted_chain_sha256") or ""
                    ),
                    output_sha256=str(block.get("output_sha256") or ""),
                )
    if action == "auto":
        recovery_journal = load_transaction(project_dir) or transaction
        prior_replacements = [
            dict(item) for item in recovery_journal.get("replacements") or []
            if str(item.get("status") or "") not in {"accepted"}
        ]
        prior_exhausted = _finalize_automatic_generation_restarts(
            project_dir,
            checkpoint_dir=checkpoint_dir,
            replacements=prior_replacements,
        )
        if prior_exhausted is not None:
            mark_transaction(project_dir, "continuation_failed")
            return prior_exhausted
        reconciled_journal = load_transaction(project_dir) or recovery_journal
        if (
            reconciled_journal.get("entries")
            and all(
                item.get("status") == "resolved"
                for item in reconciled_journal.get("entries") or []
            )
            and (not isinstance(result, dict) or not result.get("ok"))
        ):
            result = _continue_supervised_build(
                replace(options, supervised_native_resume_identities=()),
                cancel_check=cancel_check,
                continuation=continuation,
            )
        if source_preflight is not None or (
            continuation is None and build_companion is _BUILD_COMPANION_ENTRYPOINT
        ):
            try:
                if source_preflight is not None:
                    source_preflight()
                else:
                    _preflight_automatic_recovery_source(options)
            except (SourceError, OSError) as exc:
                mark_transaction(project_dir, "continuation_failed")
                return err(
                    "companion_source_unavailable",
                    str(exc),
                    project_dir=str(project_dir),
                    automatic_generation_restart=False,
                )
        restart = _prepare_automatic_generation_restarts(
            project_dir,
            checkpoint_dir=checkpoint_dir,
            session_manager=manager,
        )
        if restart.get("error") is not None:
            mark_transaction(project_dir, "continuation_failed")
            return restart["error"]
        replacements = list(restart.get("replacements") or [])
        if replacements:
            options = replace(options, supervised_native_resume_identities=())
            result = _continue_supervised_build(
                options, cancel_check=cancel_check, continuation=continuation,
            )
            _accept_completed_pipeline_controls(checkpoint_dir)
            exhausted = _finalize_automatic_generation_restarts(
                project_dir,
                checkpoint_dir=checkpoint_dir,
                replacements=replacements,
            )
            if exhausted is not None:
                mark_transaction(project_dir, "continuation_failed")
                return exhausted
            if (
                isinstance(result, dict)
                and result.get("status") == "needs_supervision"
            ):
                # A replacement continuation may discover a different logical
                # blocker. Re-enter the same durable transaction so that call
                # first receives replay/native reconciliation before spending
                # its own segment restart budget.
                return _resume_companion_unlocked(
                    project_dir, action="auto", cancel_check=cancel_check,
                    continuation=continuation,
                    source_preflight=source_preflight,
                )
    # Re-scan after continuation: new submitted/unknown failures join this
    # transaction on the next invocation and remain explicitly supervised.
    final_transaction = load_transaction(project_dir) or transaction
    all_resolved = all(
        item.get("status") == "resolved"
        for item in final_transaction.get("entries") or []
    )
    mark_transaction(
        project_dir,
        "complete" if all_resolved and (
            not isinstance(result, dict) or bool(result.get("ok"))
        ) else "continuation_failed",
    )
    return result


_AUTO_RESTART_EXCLUDED_CATEGORIES = {
    "authentication", "quota", "permission", "rate_limit", "cancelled",
    "local_io", "invalid_request",
}

_MAX_RECOVERY_CONTROL_BYTES = 16 * 1024 * 1024


def _read_recovery_json(
    checkpoint_dir: Path,
    path: Path,
    *,
    suffixes: tuple[str, ...] = (".json",),
) -> Any:
    """Read one exact recovery control through the shared no-follow reader."""

    root = checkpoint_dir.resolve(strict=False)
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    try:
        relative = lexical.relative_to(root)
    except ValueError as exc:
        raise SecureReadError("recovery control escapes its checkpoint") from exc
    return read_bounded_json(
        root,
        relative,
        max_bytes=_MAX_RECOVERY_CONTROL_BYTES,
        suffixes=suffixes,
    )


def _read_recovery_ledger(
    checkpoint_dir: Path,
    path: Path,
    *,
    allow_explicit_legacy: bool = False,
) -> dict[str, Any]:
    """Read a hash-bound registered ledger, except in explicit legacy mode."""

    root = checkpoint_dir.resolve(strict=False)
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    if not allow_explicit_legacy:
        try:
            value, _digest = read_registered_lane_ledger(root, lexical)
        except LaneLedgerRegistryError as exc:
            raise SecureReadError("recovery ledger is not registered or changed") from exc
        return value
    value = _read_recovery_json(root, lexical)
    if not isinstance(value, dict):
        raise SecureReadError("recovery ledger is not a JSON object")
    return value


def _entry_call_checkpoint(
    entry: Mapping[str, Any], checkpoint_dir: Path, *,
    allow_explicit_legacy: bool = False,
) -> tuple[Path | None, dict[str, Any] | None]:
    """Resolve one call checkpoint without trusting an arbitrary saved path."""

    root = checkpoint_dir.resolve(strict=False)
    context = entry.get("recovery_context")
    context = context if isinstance(context, Mapping) else {}
    key = str(entry.get("idempotency_key") or context.get("idempotency_key") or "")
    expected_name = (
        f"idempotency-{hashlib.sha256(key.encode('utf-8')).hexdigest()}.json"
        if key else ""
    )
    saved = str(context.get("checkpoint_path") or "")
    if saved:
        raw_saved_path = Path(saved).expanduser()
        saved_path = raw_saved_path.resolve(strict=False)
        try:
            relative = saved_path.relative_to(root)
        except ValueError:
            return None, None
        if not raw_saved_path.is_absolute() or raw_saved_path != saved_path:
            return None, None
        if (
            saved_path.parent.name == "call-checkpoints"
            and (not expected_name or saved_path.name == expected_name)
        ):
            try:
                value = _read_recovery_json(root, root / relative)
            except SecureReadError:
                return None, None
            if (
                isinstance(value, dict)
                and _checkpoint_matches_recovery_entry(value, entry, context)
                and _stateless_checkpoint_receipt_valid(
                    entry, context, checkpoint_dir=root,
                    checkpoint_path=saved_path, checkpoint=value,
                )
            ):
                return saved_path, value
            return None, None
        return None, None
    if not allow_explicit_legacy:
        return None, None
    candidates: list[Path] = []
    if expected_name:
        candidates.extend(path.resolve(strict=False) for path in root.rglob(expected_name))
    unique: dict[str, Path] = {str(path): path for path in candidates}
    valid: list[Path] = []
    for path in unique.values():
        try:
            path.relative_to(root)
        except ValueError:
            continue
        if path.parent.name != "call-checkpoints":
            continue
        if expected_name and path.name != expected_name:
            continue
        valid.append(path)
    if len(valid) != 1:
        return None, None
    try:
        value = _read_recovery_json(root, valid[0])
    except SecureReadError:
        return None, None
    if (
        isinstance(value, dict)
        and _checkpoint_matches_recovery_entry(value, entry, context)
        and _stateless_checkpoint_receipt_valid(
            entry, context, checkpoint_dir=root,
            checkpoint_path=valid[0], checkpoint=value,
        )
    ):
        return valid[0], value
    return None, None


def _checkpoint_matches_recovery_entry(
    checkpoint: Mapping[str, Any],
    entry: Mapping[str, Any],
    context: Mapping[str, Any],
) -> bool:
    logical = checkpoint.get("logical_identity")
    recipe = checkpoint.get("request_recipe")
    if not isinstance(logical, Mapping) or not isinstance(recipe, Mapping):
        return False
    generation = int(
        entry.get("initial_generation") or context.get("generation") or 0
    )
    stateless_control = bool(
        context.get("stateless_control") is True
        and logical.get("session_key") is None
        and logical.get("generation") is None
    )
    return bool(
        logical.get("idempotency_key")
        == str(entry.get("idempotency_key") or context.get("idempotency_key") or "")
        and (
            stateless_control
            or logical.get("session_key")
            == str(entry.get("session_key") or context.get("session_key") or "")
        )
        and (stateless_control or logical.get("generation") == generation)
        and (
            not context.get("provider")
            or logical.get("provider") == context.get("provider")
        )
        and (
            context.get("model") is None
            or logical.get("model") == context.get("model")
        )
        and (
            not context.get("runtime_fingerprint")
            or recipe.get("runtime_fingerprint") == context.get("runtime_fingerprint")
        )
    )


def _stateless_checkpoint_receipt_valid(
    entry: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    checkpoint_dir: Path,
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
) -> bool:
    logical = checkpoint.get("logical_identity")
    if not isinstance(logical, Mapping) or not (
        logical.get("session_key") is None and logical.get("generation") is None
    ):
        return True
    if context.get("stateless_control") is not True:
        return False
    reference = context.get("submission_receipt")
    if not isinstance(reference, Mapping):
        return False
    ledger_path = Path(str(entry.get("ledger_path") or ""))
    try:
        receipt = _validate_pipeline_submission_reference(
            reference,
            checkpoint_dir=checkpoint_dir,
            ledger_path=ledger_path,
            session_key=str(entry.get("session_key") or context.get("session_key") or ""),
            logical_unit=str(entry.get("segment_id") or context.get("logical_unit") or ""),
            generation=int(
                entry.get("initial_generation") or context.get("generation") or 0
            ),
            idempotency_key=str(
                entry.get("idempotency_key") or context.get("idempotency_key") or ""
            ),
        )
        receipt_checkpoint = resolve_recovery_path(
            checkpoint_dir, receipt.get("checkpoint_path"),
        )
    except (
        RecoveryResponseError, OSError, TypeError, ValueError,
        json.JSONDecodeError, SecureReadError,
    ):
        return False
    return bool(
        receipt_checkpoint == checkpoint_path.resolve(strict=False)
        and receipt.get("checkpoint_identity") == checkpoint.get("identity")
    )


def _validate_recovery_ledger_address(
    *,
    checkpoint_dir: Path,
    ledger_path: Path,
    ledger: Mapping[str, Any],
    session_key: str,
    segment_id: str,
    generation: int | None = None,
    allow_explicit_legacy: bool = False,
) -> str | None:
    """Return a stable blocker for any non-canonical recovery control path."""

    root = checkpoint_dir.resolve(strict=False)
    raw = ledger_path.expanduser()
    resolved = raw.resolve(strict=False)
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        return "recovery lane ledger is outside the active checkpoint"
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return "recovery lane ledger address contains a symlink"
    if not raw.is_absolute() or raw != resolved or not resolved.is_file():
        return "recovery lane ledger address is not a canonical regular file"
    registered = lane_ledger_is_registered(root, resolved)
    if not registered:
        legacy = {
            path.resolve(strict=False)
            for path in _all_lane_ledger_paths(root, include_explicit_legacy=True)
        } if allow_explicit_legacy else set()
        if resolved not in legacy:
            return "recovery lane ledger is not registered in the active control index"
    if ledger.get("schema_version") != "arc.companion.chapter-lane-ledger.v2":
        return "recovery lane ledger schema is unsupported"
    expected_session = f"{ledger.get('chapter_id')}:{ledger.get('lane')}"
    if not session_key or expected_session != session_key:
        return "recovery lane session identity does not match its ledger"
    matches = [
        item for item in ledger.get("blocks") or []
        if isinstance(item, Mapping)
        and str(item.get("segment_id") or "") == segment_id
    ]
    if len(matches) != 1:
        return "recovery logical unit is absent or ambiguous"
    ledger_generation = int(ledger.get("generation") or 0)
    block_generation = int(matches[0].get("generation") or 0)
    if ledger_generation < 1 or block_generation != ledger_generation:
        return "recovery logical unit generation does not match its ledger"
    if generation is not None and int(generation) != ledger_generation:
        return "recovery entry generation does not match its ledger"
    return None


def _entry_recovery_address_blocker(
    entry: Mapping[str, Any], checkpoint_dir: Path,
) -> str | None:
    path = Path(str(entry.get("ledger_path") or ""))
    try:
        ledger = _read_recovery_ledger(checkpoint_dir, path)
    except (OSError, ValueError, json.JSONDecodeError, SecureReadError):
        return "recovery lane ledger cannot be read"
    generation_value = entry.get("initial_generation")
    try:
        generation = int(generation_value) if generation_value is not None else None
    except (TypeError, ValueError):
        return "recovery entry generation is invalid"
    return _validate_recovery_ledger_address(
        checkpoint_dir=checkpoint_dir,
        ledger_path=path,
        ledger=ledger,
        session_key=str(entry.get("session_key") or ""),
        segment_id=str(entry.get("segment_id") or ""),
        generation=generation,
    )


def _last_call_progress_event(
    path: Path,
    *,
    checkpoint_dir: Path,
    idempotency_key: str,
    session_key: str | None,
    generation: int | None,
    checkpoint_identity: str,
    native_session_id: str | None,
) -> dict[str, Any] | None:
    try:
        root = checkpoint_dir.resolve(strict=False)
        lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
        relative = lexical.relative_to(root)
        raw = read_bounded_file(
            root,
            relative,
            max_bytes=_MAX_RECOVERY_CONTROL_BYTES,
            suffixes=(".jsonl",),
        )
        lines = raw.decode("utf-8").splitlines()
    except (OSError, UnicodeError, ValueError, SecureReadError):
        return None
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict):
            continue
        if value.get("idempotency_key") != idempotency_key:
            continue
        if value.get("session_key") != session_key:
            continue
        if value.get("generation") != generation:
            continue
        if value.get("checkpoint_identity") != checkpoint_identity:
            continue
        if str(value.get("native_session_id") or "") != str(native_session_id or ""):
            continue
        return value
    return None


def _recovery_trigger(entry: Mapping[str, Any], checkpoint_dir: Path) -> str | None:
    """Return a durable typed recovery trigger; never parse human reason text."""

    context = entry.get("recovery_context")
    context = context if isinstance(context, Mapping) else {}
    logical_unit = str(entry.get("segment_id") or "")
    if not logical_unit or context.get("logical_unit") != logical_unit:
        return None
    checkpoint_path, checkpoint = _entry_call_checkpoint(entry, checkpoint_dir)
    if checkpoint_path is None or checkpoint is None:
        return None
    checkpoint_identity = str(checkpoint.get("identity") or "")
    if not checkpoint_identity:
        return None
    response = checkpoint.get("response")
    checkpoint_native_id = (
        str(response.get("native_session_id") or "")
        if isinstance(response, Mapping) else ""
    )
    context_native_id = str(context.get("native_session_id") or "")
    if checkpoint_native_id and context_native_id and checkpoint_native_id != context_native_id:
        return None
    expected_native_id = context_native_id or checkpoint_native_id or None
    logical = checkpoint.get("logical_identity")
    logical = logical if isinstance(logical, Mapping) else {}
    stateless_control = bool(
        context.get("stateless_control") is True
        and logical.get("session_key") is None
        and logical.get("generation") is None
    )
    latest = context.get("latest_progress")
    if isinstance(latest, Mapping) and latest.get("event") == "idle_timeout":
        key = str(
            entry.get("idempotency_key") or context.get("idempotency_key") or ""
        )
        session_key = (
            None if stateless_control
            else str(entry.get("session_key") or context.get("session_key") or "")
        )
        generation_value = (
            None if stateless_control
            else context.get("generation")
            if context.get("generation") is not None
            else entry.get("initial_generation")
        )
        try:
            generation = None if generation_value is None else int(generation_value)
        except (TypeError, ValueError):
            generation = None
        if (
            key
            and logical_unit
            and (stateless_control or bool(session_key) and bool(generation and generation > 0))
            and context.get("logical_unit") == logical_unit
            and latest.get("idempotency_key") == key
            and latest.get("session_key") == session_key
            and latest.get("generation") == generation
            and latest.get("checkpoint_identity") == checkpoint_identity
            and str(latest.get("native_session_id") or "")
            == str(expected_native_id or "")
        ):
            return "idle_timeout"
    progress_value = str(checkpoint.get("progress_journal") or "")
    progress_path = (
        Path(progress_value).expanduser().resolve(strict=False)
        if progress_value else checkpoint_path.parent.parent / "progress.jsonl"
    )
    try:
        progress_path.relative_to(checkpoint_dir.resolve(strict=False))
    except ValueError:
        return None
    key = str(
        logical.get("idempotency_key")
        or entry.get("idempotency_key")
        or context.get("idempotency_key")
        or ""
    )
    generation_value = logical.get("generation")
    try:
        generation = int(generation_value) if generation_value is not None else None
    except (TypeError, ValueError):
        return None
    terminal = _last_call_progress_event(
        progress_path,
        checkpoint_dir=checkpoint_dir,
        idempotency_key=key,
        session_key=(
            None if stateless_control
            else str(logical.get("session_key") or entry.get("session_key") or "")
        ),
        generation=generation,
        checkpoint_identity=checkpoint_identity,
        native_session_id=expected_native_id,
    )
    return "idle_timeout" if terminal and terminal.get("event") == "idle_timeout" else None


def _complete_response_scan(
    entry: Mapping[str, Any], checkpoint_dir: Path,
) -> dict[str, Any]:
    """Describe complete durable work without interpreting partial provider output."""

    ledger_path = Path(str(entry.get("ledger_path") or ""))
    segment_id = str(entry.get("segment_id") or "")
    try:
        ledger = _read_recovery_ledger(checkpoint_dir, ledger_path)
        block = next((
            item for item in ledger.get("blocks") or []
            if isinstance(item, Mapping)
            and str(item.get("segment_id") or "") == segment_id
        ), None)
        if isinstance(block, Mapping) and block.get("state") == "accepted":
            return {
                "complete": True,
                "source": "accepted_lane_block",
                "validation_status": "accepted",
            }
    except (OSError, ValueError, json.JSONDecodeError, SecureReadError):
        pass
    _path, checkpoint = _entry_call_checkpoint(entry, checkpoint_dir)
    if isinstance(checkpoint, Mapping):
        response = checkpoint.get("response")
        value = response.get("value") if isinstance(response, Mapping) else None
        if (
            checkpoint.get("state") in {"response_received", "validated"}
            and isinstance(value, Mapping)
        ):
            return {
                "complete": True,
                "source": f"call_checkpoint_{checkpoint.get('state')}",
                "validation_status": "pending_normal_validation",
            }
    context = entry.get("recovery_context")
    context = context if isinstance(context, Mapping) else {}
    if str(entry.get("recovery_action") or context.get("recovery_action") or "") == "deterministic-replay":
        return {
            "complete": True,
            "source": "deterministic_replay_receipt",
            "validation_status": "pending_normal_validation",
        }
    return {
        "complete": False,
        "source": "no_complete_response",
        "validation_status": "none",
    }


def _native_context_recovery_identity(
    value: Mapping[str, Any],
) -> tuple[str, str, str, int, str]:
    context = value.get("recovery_context")
    context = context if isinstance(context, Mapping) else {}
    generation_value = (
        value.get("generation")
        or value.get("initial_generation")
        or context.get("generation")
        or 0
    )
    return (
        str(Path(str(value.get("ledger_path") or "")).resolve(strict=False)),
        str(value.get("session_key") or context.get("session_key") or ""),
        str(value.get("segment_id") or context.get("logical_unit") or ""),
        int(generation_value),
        str(value.get("idempotency_key") or context.get("idempotency_key") or ""),
    )


def _supervised_native_resume_authorized(
    options: BuildOptions,
    *,
    ledger_path: Path,
    session_key: str,
    logical_unit: str,
    generation: int,
    idempotency_key: str,
) -> tuple[str, str, str, int, str] | None:
    identity = (
        str(ledger_path.expanduser().resolve(strict=False)),
        str(session_key),
        str(logical_unit),
        int(generation),
        str(idempotency_key),
    )
    return identity if identity in options.supervised_native_resume_identities else None


def _native_recovery_identity_complete(
    identity: tuple[str, str, str, int, str],
) -> bool:
    return bool(
        identity[0] and identity[1] and identity[2]
        and identity[3] > 0 and identity[4]
    )


def _validate_pipeline_submission_reference(
    reference: Mapping[str, Any],
    *,
    checkpoint_dir: Path,
    ledger_path: Path,
    session_key: str,
    logical_unit: str,
    generation: int,
    idempotency_key: str,
) -> dict[str, Any]:
    """Validate one receipt against its exact current five-field control."""

    receipt_identity = str(reference.get("identity_sha256") or "")
    return validate_ledger_submission_reference(
        reference,
        checkpoint_dir=checkpoint_dir,
        expected_recovery_identity=(
            str(ledger_path.expanduser().resolve(strict=False)),
            session_key,
            logical_unit,
            generation,
            idempotency_key,
        ),
        expected_receipt_identity_sha256=receipt_identity,
    )


def _promote_entry_raw_response(
    entry: Mapping[str, Any], checkpoint_dir: Path,
) -> dict[str, Any] | None:
    """Promote only an exact, sealed production submission for business replay."""

    context = entry.get("recovery_context")
    context = context if isinstance(context, Mapping) else {}
    reference = context.get("submission_receipt")
    if not isinstance(reference, Mapping):
        return None
    return recover_complete_ledger_response(
        reference,
        checkpoint_dir=checkpoint_dir,
        ledger_path=Path(str(entry.get("ledger_path") or "")),
        session_key=str(entry.get("session_key") or ""),
        logical_unit=str(entry.get("segment_id") or ""),
        generation=int(
            entry.get("initial_generation")
            or context.get("generation")
            or 0
        ),
        idempotency_key=str(
            entry.get("idempotency_key") or context.get("idempotency_key") or ""
        ),
        expected_receipt_identity_sha256=str(
            reference.get("identity_sha256") or ""
        ),
    )


def _first_nonaccepted_segment_id(
    entry: Mapping[str, Any], *, checkpoint_dir: Path,
) -> str:
    path = Path(str(entry.get("ledger_path") or ""))
    try:
        ledger = _read_recovery_ledger(checkpoint_dir, path)
        for block in ledger.get("blocks") or []:
            if isinstance(block, Mapping) and block.get("state") != "accepted":
                return str(block.get("segment_id") or "")
    except SecureReadError:
        pass
    return str(entry.get("segment_id") or "")


def _entry_with_current_supervision(
    entry: Mapping[str, Any], *, checkpoint_dir: Path,
) -> dict[str, Any]:
    """Enrich an older transaction entry from its same-unit durable marker."""

    enriched = dict(entry)
    path = Path(str(entry.get("ledger_path") or ""))
    try:
        ledger = _read_recovery_ledger(checkpoint_dir, path)
    except SecureReadError:
        return enriched
    segment_id = str(entry.get("segment_id") or "")
    marker = next((
        item for item in _active_supervision_entries(ledger)
        if str(item.get("segment_id") or "") == segment_id
    ), None)
    if not isinstance(marker, Mapping):
        return enriched
    context = marker.get("recovery_context")
    if not isinstance(context, Mapping):
        return enriched
    prior = enriched.get("recovery_context")
    prior = prior if isinstance(prior, Mapping) else {}
    enriched["recovery_context"] = {**dict(prior), **dict(context)}
    if not enriched.get("idempotency_key"):
        enriched["idempotency_key"] = str(context.get("idempotency_key") or "")
    return enriched


def _fresh_auto_recovery_audit(
    entry: Mapping[str, Any], checkpoint_dir: Path,
) -> dict[str, Any]:
    trigger = _recovery_trigger(entry, checkpoint_dir)
    scan = _complete_response_scan(entry, checkpoint_dir)
    required = trigger == "idle_timeout" and not bool(scan["complete"])
    return {
        "recovery_trigger": trigger,
        "complete_response_scan": scan,
        "durable_replay_required": bool(scan["complete"]),
        "automatic_native_resume_suppressed": required,
        "fresh_generation_required": required,
    }


def _automatic_restart_blocker(
    entry: Mapping[str, Any], ledger: Mapping[str, Any] | None = None,
    checkpoint_dir: Path | None = None,
) -> str | None:
    """Return why an unresolved entry must remain under operator control."""

    session_key = str(entry.get("session_key") or "")
    lane = session_key.rsplit(":", 1)[-1]
    spec = recovery_unit_for_ledger(lane)
    if spec is None:
        return "automatic replacement lacks a registered recovery-unit handler"
    elif not isinstance(ledger, Mapping):
        return "automatic replacement lane ledger is unavailable"
    else:
        if checkpoint_dir is None:
            return "automatic replacement active checkpoint is unavailable"
        allowed_generations = {
            int(value) for value in (
                entry.get("initial_generation"), entry.get("target_generation"),
            ) if value is not None
        }
        ledger_generation = int(ledger.get("generation") or 0)
        if allowed_generations and ledger_generation not in allowed_generations:
            return "recovery entry generation does not match its ledger"
        address_blocker = _validate_recovery_ledger_address(
            checkpoint_dir=checkpoint_dir,
            ledger_path=Path(str(entry.get("ledger_path") or "")),
            ledger=ledger,
            session_key=session_key,
            segment_id=str(entry.get("segment_id") or ""),
            generation=ledger_generation,
        )
        if address_blocker is not None:
            return address_blocker
        if ledger.get("schema_version") != "arc.companion.chapter-lane-ledger.v2":
            return "automatic replacement lane ledger schema is unsupported"
        if str(ledger.get("lane") or "") != lane:
            return "automatic replacement lane identity does not match its ledger"
        chapter_id = session_key.rsplit(":", 1)[0]
        if str(ledger.get("chapter_id") or "") != chapter_id:
            return "automatic replacement chapter identity does not match its ledger"
        segment_id = str(entry.get("segment_id") or "")
        matches = [
            item for item in ledger.get("blocks") or []
            if isinstance(item, Mapping)
            and str(item.get("segment_id") or "") == segment_id
        ]
        if len(matches) != 1:
            return "automatic replacement logical unit is absent or ambiguous"
        if entry.get("fresh_generation_required"):
            context = entry.get("recovery_context")
            context = context if isinstance(context, Mapping) else {}
            reference = context.get("submission_receipt")
            if not isinstance(reference, Mapping):
                return "typed idle replacement lacks an exact production submission receipt"
            try:
                _validate_pipeline_submission_reference(
                    reference,
                    checkpoint_dir=checkpoint_dir,
                    ledger_path=Path(str(entry.get("ledger_path") or "")),
                    session_key=session_key,
                    logical_unit=segment_id,
                    generation=ledger_generation,
                    idempotency_key=str(
                        entry.get("idempotency_key")
                        or context.get("idempotency_key")
                        or ""
                    ),
                )
            except RecoveryResponseError as exc:
                return str(exc)
    context = entry.get("recovery_context")
    context = context if isinstance(context, dict) else {}
    category = str(context.get("failure_category") or "").casefold()
    if category in _AUTO_RESTART_EXCLUDED_CATEGORIES:
        return f"failure category {category} is not eligible for generation replacement"
    if category in {"", "unknown"}:
        supervision_reason = " ".join(
            str(entry.get("supervision_reason") or "").casefold().split()
        )
        if supervision_reason in {
            "submitted call cancelled", "submitted call canceled",
        }:
            return "cancelled calls are not eligible for generation replacement"
    return None


def _prepare_automatic_generation_restarts(
    project_dir: Path,
    *,
    checkpoint_dir: Path,
    session_manager: Any,
) -> dict[str, Any]:
    """Rotate each affected lane once and invalidate its earliest blocked suffix."""

    from .resume_transaction import (
        AutomaticRegenerationExhausted,
        claim_automatic_restart,
        load_transaction,
        mark_entry,
        mark_replacement,
    )

    transaction = load_transaction(project_dir)
    if transaction is None:
        return {"replacements": []}
    _backfill_legacy_generation_owners(checkpoint_dir, transaction)
    pending: list[tuple[int, dict[str, Any], Path, dict[str, Any], str]] = []
    blockers: list[str] = []
    for index, raw in enumerate(transaction.get("entries") or []):
        entry = dict(raw)
        if entry.get("status") == "resolved":
            continue
        path = Path(str(entry.get("ledger_path") or ""))
        try:
            ledger, ledger_digest = read_registered_lane_ledger(
                checkpoint_dir, path,
            )
        except LaneLedgerRegistryError:
            blockers.append(f"lane ledger is unregistered or changed: {path}")
            continue
        segment_id = str(entry.get("segment_id") or "")
        block = next((
            item for item in ledger.get("blocks") or []
            if str(item.get("segment_id") or "") == segment_id
        ), None)
        if isinstance(block, dict) and block.get("state") == "accepted":
            mark_entry(
                project_dir, index, status="resolved",
                accepted_chain_sha256=str(block.get("accepted_chain_sha256") or ""),
                output_sha256=str(block.get("output_sha256") or ""),
            )
            continue
        reason = _automatic_restart_blocker(entry, ledger, checkpoint_dir)
        if reason is not None:
            blockers.append(reason)
            continue
        pending.append((index, entry, path, ledger, ledger_digest))
    if blockers:
        return {"replacements": [], "blocked_reasons": blockers}
    groups: dict[
        str, list[tuple[int, dict[str, Any], Path, dict[str, Any], str]]
    ] = {}
    for item in pending:
        groups.setdefault(str(item[1].get("session_key") or ""), []).append(item)
    replacements: list[dict[str, Any]] = []
    max_auto_replacements = int(
        (transaction.get("recovery_options") or {}).get("max_auto_replacements") or 3
    )
    for session_key, items in sorted(groups.items()):
        ledger = items[0][3]
        path = items[0][2]
        ordered_ids = [str(item.get("segment_id") or "") for item in ledger.get("blocks") or []]
        positions = {value: index for index, value in enumerate(ordered_ids)}
        earliest = min(
            items, key=lambda item: positions.get(str(item[1].get("segment_id") or ""), len(ordered_ids))
        )
        earliest_id = str(earliest[1].get("segment_id") or "")
        restart_id = next((
            str(block.get("segment_id") or "")
            for block in ledger.get("blocks") or []
            if isinstance(block, Mapping) and block.get("state") != "accepted"
        ), earliest_id)
        if restart_id not in positions:
            return {
                "replacements": replacements,
                "error": err(
                    "resume_transaction_logical_unit_invalid",
                    f"Lane {session_key} has no structurally owned restart unit",
                ),
            }
        group_id = f"{session_key}:{restart_id}"
        active_record = next((
            dict(item) for item in reversed(transaction.get("replacements") or [])
            if str(item.get("group_id") or "") == group_id
            and str(item.get("status") or "") not in {"accepted", "failed"}
        ), None)
        prior_group_records = [
            dict(item) for item in transaction.get("replacements") or []
            if str(item.get("group_id") or "") == group_id
        ]
        reconstructed_inflight = bool(
            active_record is None
            and not prior_group_records
            and all(
                str(item[1].get("recovery_action") or "")
                == "generation_restart_required"
                for item in items
            )
            and any(
                int(item[1].get("target_generation") or 0)
                == int(ledger.get("generation") or 1)
                for item in items
            )
        )
        source_generation = int(
            active_record.get("source_generation")
            if active_record is not None else
            min(
                int(item[1].get("initial_generation") or ledger.get("generation") or 1)
                for item in items
            ) if reconstructed_inflight else
            ledger.get("generation") or 1
        )
        target_generation = int(
            active_record.get("target_generation")
            if active_record is not None else
            ledger.get("generation") if reconstructed_inflight else
            source_generation + 1
        )
        suffix_ids = ordered_ids[positions[restart_id]:]
        ref = session_manager.get_existing(session_key)
        while ref is not None and ref.generation < source_generation:
            ref = session_manager.rotate(
                session_key, reason="align fresh automatic replacement generation",
            )
        if (
            ref is not None
            and ref.generation not in {source_generation, target_generation}
        ):
            return {
                "replacements": replacements,
                "error": err(
                    "resume_transaction_generation_mismatch",
                    f"Session {session_key} changed outside automatic recovery",
                ),
            }
        group_records: list[dict[str, Any]] = []
        try:
            earliest_context = earliest[1].get("recovery_context")
            earliest_context = (
                earliest_context if isinstance(earliest_context, dict) else {}
            )
            record = active_record or claim_automatic_restart(
                    project_dir,
                    session_key=session_key,
                    segment_id=earliest_id,
                    group_id=group_id,
                    ledger_path=path,
                    source_generation=source_generation,
                    target_generation=target_generation,
                    suffix_start_segment_id=restart_id,
                    suffix_segment_ids=suffix_ids,
                    trigger_code=str(
                        earliest[1].get("recovery_trigger")
                        or earliest[1].get("blocking_code")
                        or earliest_context.get("blocked_reason")
                        or "generation_restart_required"
                    ),
                    trigger_reason=str(
                        earliest[1].get("blocking_reason")
                        or earliest_context.get("blocked_reason")
                        or "native reconciliation or persisted response could not be accepted"
                    ),
                    abandoned_logical_key=str(
                        earliest[1].get("idempotency_key") or ""
                    ),
                    possible_duplicate_charge=any(
                        str(((item[1].get("recovery_context") or {}).get(
                            "submission_state"
                        )) or "") in {"submitted", "unknown"}
                        for item in items
                    ),
                    max_auto_replacements=max_auto_replacements,
                )
            group_records.append(record)
            for index, entry, _entry_path, _entry_ledger, _entry_digest in items:
                mark_entry(
                    project_dir, index,
                    status=str(entry.get("status") or "pending"),
                    recovery_action="generation_restart_required",
                    recovery_trigger=entry.get("recovery_trigger"),
                    automatic_native_resume_suppressed=bool(
                        entry.get("automatic_native_resume_suppressed")
                    ),
                    fresh_generation_required=bool(
                        entry.get("fresh_generation_required")
                    ),
                    fresh_task_start_segment_id=restart_id,
                    target_generation=target_generation,
                    replacement_id=record["replacement_id"],
                    replacement_group_id=record["group_id"],
                    replacement_attempt=record["attempt"],
                )
        except AutomaticRegenerationExhausted as exc:
            return {
                "replacements": replacements,
                "error": err(
                    "automatic_regeneration_exhausted", str(exc),
                    session_key=session_key, segment_id=earliest_id,
                ),
            }
        _record_legacy_generation_owners(
            checkpoint_dir,
            lane=session_key.rsplit(":", 1)[-1],
            segment_ids=suffix_ids,
            generation=source_generation,
        )
        if ref is not None and ref.generation == source_generation:
            ref = session_manager.rotate(
                session_key, reason="automatic generation restart",
            )
        if ref is not None and ref.generation != target_generation:
            return {
                "replacements": replacements,
                "error": err(
                    "resume_transaction_generation_mismatch",
                    f"Session {session_key} changed outside automatic recovery",
                ),
            }
        for record in group_records:
            if str(record.get("status") or "claimed") == "claimed":
                record = mark_replacement(
                    project_dir, record["replacement_id"], status="rotated",
                    rotated_generation=target_generation,
                )
        try:
            current, current_digest = read_registered_lane_ledger(
                checkpoint_dir, path,
            )
        except LaneLedgerRegistryError:
            return {
                "replacements": replacements,
                "error": err(
                    "resume_transaction_generation_mismatch",
                    f"Lane ledger {path} changed outside automatic recovery",
                ),
            }
        if int(current.get("generation") or 0) == source_generation:
            try:
                current = invalidate_suffix(
                    path,
                    from_segment_id=restart_id,
                    generation=target_generation,
                    expected_ledger_sha256=current_digest,
                    checkpoint_dir=checkpoint_dir,
                )
            except LaneLedgerError as exc:
                return {
                    "replacements": replacements,
                    "error": err(
                        "resume_transaction_generation_mismatch",
                        f"Lane ledger {path} changed outside automatic recovery: {exc}",
                    ),
                }
        if int(current.get("generation") or 0) != target_generation:
            return {
                "replacements": replacements,
                "error": err(
                    "resume_transaction_generation_mismatch",
                    f"Lane ledger {path} changed outside automatic recovery",
                ),
            }
        for record in group_records:
            if str(record.get("status") or "claimed") in {"claimed", "rotated"}:
                updated = mark_replacement(
                    project_dir, record["replacement_id"], status="suffix_invalidated",
                    suffix_start_segment_id=restart_id,
                    suffix_segment_ids=suffix_ids,
                )
            else:
                updated = record
            replacements.append(updated)
    return {"replacements": replacements}


def _finalize_automatic_generation_restarts(
    project_dir: Path,
    *,
    checkpoint_dir: Path,
    replacements: list[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Resolve originals only after their replacement-generation blocks are accepted."""

    from .resume_transaction import load_transaction, mark_entry, mark_replacement

    transaction = load_transaction(project_dir)
    if transaction is None:
        return err("resume_transaction_invalid", "Resume transaction journal is missing")
    by_replacement: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, item in enumerate(transaction.get("entries") or []):
        if item.get("replacement_id"):
            by_replacement.setdefault(
                str(item.get("replacement_id") or ""), []
            ).append((index, dict(item)))
    exhausted_result: dict[str, Any] | None = None
    for raw in replacements:
        replacement_id = str(raw.get("replacement_id") or "")
        located = by_replacement.get(replacement_id) or []
        if not located:
            continue
        path = Path(str(raw.get("ledger_path") or located[0][1].get("ledger_path") or ""))
        try:
            ledger = _read_recovery_ledger(checkpoint_dir, path)
        except SecureReadError:
            continue
        replacement_segment_id = str(
            raw.get("segment_id") or located[0][1].get("segment_id") or ""
        )
        block = next((
            item for item in ledger.get("blocks") or []
            if str(item.get("segment_id") or "") == replacement_segment_id
        ), None)
        target_generation = int(raw.get("target_generation") or 0)
        if (
            isinstance(block, dict)
            and int(block.get("generation") or 0) == target_generation
            and block.get("state") == "accepted"
        ):
            for index, entry in located:
                entry_block = next((
                    item for item in ledger.get("blocks") or []
                    if str(item.get("segment_id") or "")
                    == str(entry.get("segment_id") or "")
                ), None)
                if isinstance(entry_block, dict) and entry_block.get("state") == "accepted":
                    mark_entry(
                        project_dir, index, status="resolved",
                        accepted_chain_sha256=str(
                            entry_block.get("accepted_chain_sha256") or ""
                        ),
                        output_sha256=str(entry_block.get("output_sha256") or ""),
                    )
            mark_replacement(
                project_dir, replacement_id, status="accepted",
                accepted_chain_sha256=str(block.get("accepted_chain_sha256") or ""),
                output_sha256=str(block.get("output_sha256") or ""),
            )
            continue
        if (
            isinstance(block, dict)
            and int(block.get("generation") or 0) == target_generation
            and block.get("state") in {
                "response_received", "schema_valid", "formula_valid", "token_valid",
            }
            and str(raw.get("status") or "") in {
                "claimed", "rotated", "suffix_invalidated",
            }
        ):
            mark_replacement(
                project_dir, replacement_id, status="response_persisted",
                block_state=str(block.get("state") or ""),
            )
        matching_supervision = next((
            item for item in _active_supervision_entries(ledger)
            if str(item.get("segment_id") or "")
            == replacement_segment_id
        ), None)
        if (
            matching_supervision is not None
            and int(ledger.get("generation") or 0) == target_generation
        ):
            mark_replacement(
                project_dir, replacement_id, status="failed",
                failure_reason=str(
                    matching_supervision.get("reason")
                    or "replacement failed validation"
                ),
            )
            budget_exhausted = int(raw.get("attempt") or 0) >= int(
                raw.get("max_auto_attempts") or 3
            )
            if exhausted_result is None and budget_exhausted:
                exhausted_result = err(
                    "automatic_regeneration_exhausted",
                    f"automatic replacement failed for {raw.get('session_key')}:{replacement_segment_id}",
                    session_key=raw.get("session_key"),
                    segment_id=replacement_segment_id,
                    target_generation=target_generation,
                )
    return exhausted_result


def _continue_supervised_build(
    options: BuildOptions,
    *,
    cancel_check: Callable[[], bool] | None,
    continuation: Callable[[BuildOptions], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Continue a build without releasing the already-held project lock."""

    if continuation is not None:
        return continuation(options)
    if build_companion is not _BUILD_COMPANION_ENTRYPOINT:
        return build_companion(options)
    return _build_companion_unlocked(
        options,
        source_loader=lambda paper_id, **kwargs: _load_recovery_source_bundle(
            options, paper_id=paper_id, load_kwargs=kwargs,
        ),
        llm=None,
        compiler=compile_latex,
        pdf_validator=validate_pdf,
        result_llm=None,
        cancel_check=cancel_check,
    )


def _load_recovery_source_bundle(
    options: BuildOptions,
    *,
    paper_id: str,
    load_kwargs: Mapping[str, Any],
) -> SourceBundle:
    """Load source normally, or verify and reuse the immutable build snapshot."""

    try:
        return load_source_bundle(paper_id, **dict(load_kwargs))
    except SourceError as original_error:
        state = _read_optional_json(options.project_dir.resolve() / "state.json")
        active_run = state.get("active_run")
        active_run = active_run if isinstance(active_run, dict) else {}
        checkpoint_dir = _checkpoint_dir_from_recovery_state(
            options.project_dir.resolve(), state,
        )
        payload = _read_checkpoint_json(
            checkpoint_dir / "document.json", root=checkpoint_dir,
        )
        evidence = _read_checkpoint_json(
            checkpoint_dir / "evidence.json", root=checkpoint_dir,
        )
        if not (
            isinstance(payload, dict)
            and str(payload.get("paper_id") or "") == paper_id
            and isinstance(payload.get("document"), dict)
            and isinstance(evidence, dict)
        ):
            raise original_error
        document = dict(payload["document"])
        integrity = document.get("integrity")
        if not isinstance(integrity, dict) or integrity.get("status") != "complete":
            raise original_error
        source = document.get("source")
        source = source if isinstance(source, dict) else {}
        front_matter = document.get("front_matter")
        front_matter = front_matter if isinstance(front_matter, dict) else {}
        metadata = {
            "title": str(
                front_matter.get("title") or source.get("title") or paper_id
            ),
            "_arc_companion_metadata_source": "checkpoint_snapshot",
        }
        bundle = SourceBundle(
            paper_id=paper_id,
            parsed=dict(payload),
            document=document,
            metadata=metadata,
            references=[
                dict(item) for item in evidence.get("references") or []
                if isinstance(item, dict)
            ],
            citers=[
                dict(item) for item in evidence.get("citers") or []
                if isinstance(item, dict)
            ],
            diagnostics=tuple(
                dict(item) for item in evidence.get("diagnostics") or []
                if isinstance(item, dict)
            ),
            related_evidence=tuple(
                dict(item) for item in evidence.get("related_papers") or []
                if isinstance(item, dict)
            ),
        )
        expected_fingerprint = str(
            state.get("fingerprint") or active_run.get("fingerprint") or checkpoint_dir.name
        )
        actual_fingerprint = _fingerprint(
            bundle, options, evidence=_evidence(bundle), domain_context=None,
        )
        if not expected_fingerprint or actual_fingerprint != expected_fingerprint:
            raise SourceError(
                "The cached source is unavailable and the checkpoint snapshot "
                "does not match the authoritative build fingerprint"
            ) from original_error
        receipt = _read_checkpoint_json(
            checkpoint_dir / "source-snapshot-receipt.json", root=checkpoint_dir,
        )
        if receipt is not None:
            domain_context = _read_checkpoint_json(
                checkpoint_dir / "domain-context.json", root=checkpoint_dir,
            )
            if not (
                isinstance(receipt, dict)
                and receipt.get("schema_version")
                == "arc.companion.source-snapshot-receipt.v1"
                and receipt.get("paper_id") == paper_id
                and receipt.get("fingerprint") == expected_fingerprint
                and receipt.get("document_payload_sha256") == sha256_json(payload)
                and receipt.get("evidence_sha256") == sha256_json(evidence)
                and receipt.get("domain_context_sha256") == (
                    sha256_json(domain_context) if domain_context is not None else None
                )
            ):
                raise SourceError(
                    "The checkpoint source snapshot receipt is invalid or changed"
                ) from original_error
        return bundle


def _preflight_automatic_recovery_source(options: BuildOptions) -> None:
    """Prove source authority before spending any replacement budget."""

    _load_recovery_source_bundle(
        options,
        paper_id=options.paper_id,
        load_kwargs={
            "refresh": options.refresh,
            "recache": options.recache,
            "document_kind": options.document_kind,
        },
    )


def _checkpoint_dir_from_recovery_state(
    project_dir: Path, state: Mapping[str, Any],
) -> Path:
    """Recover the checkpoint root even after a failed state write lost pointers."""

    active_run = state.get("active_run")
    active_run = active_run if isinstance(active_run, dict) else {}
    saved = str(
        state.get("checkpoint_dir") or active_run.get("checkpoint_dir") or ""
    )
    if saved:
        candidate = Path(saved)
        if candidate.is_dir():
            return candidate
    transaction = _read_checkpoint_json(
        project_dir / ".arc-companion" / "resume-transaction.json",
        root=project_dir,
    )
    if isinstance(transaction, dict):
        for entry in transaction.get("entries") or []:
            if not isinstance(entry, dict) or not entry.get("ledger_path"):
                continue
            ledger_path = Path(str(entry["ledger_path"])).resolve(strict=False)
            try:
                candidate = ledger_path.parents[2]
            except IndexError:
                continue
            if (
                candidate.is_dir()
                and (candidate / "document.json").is_file()
                and (candidate / "sessions").is_dir()
            ):
                return candidate
    return Path(saved)


def _validate_native_resume_context(
    *,
    checkpoint_dir: Path,
    ledger_path: Path,
    ledger: dict[str, Any],
    session_manager: Any,
    supervision: Mapping[str, Any] | None = None,
    allow_explicit_legacy: bool = False,
) -> dict[str, Any]:
    address_blocker = _validate_recovery_ledger_address(
        checkpoint_dir=checkpoint_dir,
        ledger_path=ledger_path,
        ledger=ledger,
        session_key=f"{ledger.get('chapter_id')}:{ledger.get('lane')}",
        segment_id=str(
            (supervision if supervision is not None else ledger.get("needs_supervision") or {}).get(
                "segment_id"
            )
            or ""
        ),
        generation=int(ledger.get("generation") or 0),
        allow_explicit_legacy=allow_explicit_legacy,
    )
    if address_blocker is not None:
        raise ValueError(address_blocker)
    active_supervision = dict(
        supervision if supervision is not None else ledger.get("needs_supervision") or {}
    )
    context = dict(active_supervision.get("recovery_context") or {})
    session_key = f"{ledger.get('chapter_id')}:{ledger.get('lane')}"
    ref = session_manager.get_existing(session_key)
    if ref is None:
        raise ValueError(f"No saved logical session exists for {session_key}")
    try:
        ledger_generation = int(ledger.get("generation"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"The supervised ledger for {session_key} has no valid generation") from exc
    if ref.generation != ledger_generation:
        raise ValueError(f"The supervised ledger generation does not match session {session_key}")
    context_session_key = context.get("session_key")
    if context_session_key is not None and str(context_session_key) != session_key:
        raise ValueError(f"The recovery context belongs to a different session than {session_key}")
    context_generation = context.get("generation")
    if context_generation is not None and int(context_generation) != ledger_generation:
        raise ValueError(f"The recovery context generation does not match session {session_key}")
    for field, expected in (
        ("provider", ref.provider),
        ("model", ref.model),
        ("runtime_fingerprint", ref.runtime_fingerprint),
    ):
        if field in context and context[field] != expected:
            raise ValueError(f"The recovery context {field} does not match session {session_key}")
    idempotency_key = str(context.get("idempotency_key") or "")
    if not idempotency_key:
        raise ValueError(f"The supervised call for {session_key} has no logical call key")
    if not (
        idempotency_key.startswith(f"{session_key}:")
        and idempotency_key.endswith(f":generation-{ledger_generation}")
    ):
        raise ValueError(f"The logical call key does not match session {session_key} generation")
    if not context.get("resumable"):
        raise ValueError(f"The supervised call for {session_key} is not resumable")
    context_native_id = str(context.get("native_session_id") or "")
    if ref.native_session_id:
        if context_native_id and context_native_id != ref.native_session_id:
            raise ValueError(f"The recovery native session id conflicts with session {session_key}")
        return {
            "session_key": session_key,
            "idempotency_key": idempotency_key,
            "provider": ref.provider,
            "model": ref.model,
            "runtime_fingerprint": ref.runtime_fingerprint,
            "generation": ref.generation,
            "native_session_id_to_restore": None,
            "ledger_path": str(ledger_path),
            "segment_id": str(active_supervision.get("segment_id") or ""),
        }
    if str(context_session_key or "") != session_key or context_generation is None:
        raise ValueError(f"The recovery context for {session_key} lacks exact session identity")
    checkpoint_value = str(context.get("checkpoint_path") or "")
    if not checkpoint_value:
        raise ValueError(f"The recovery context for {session_key} has no checkpoint path")
    checkpoint_path = Path(checkpoint_value).expanduser().resolve(strict=False)
    checkpoint_root = checkpoint_dir.resolve(strict=False)
    try:
        checkpoint_path.relative_to(checkpoint_root)
    except ValueError as exc:
        raise ValueError(f"The recovery checkpoint for {session_key} is outside the build") from exc
    expected_name = f"idempotency-{hashlib.sha256(idempotency_key.encode('utf-8')).hexdigest()}.json"
    if checkpoint_path.name != expected_name or checkpoint_path.parent.name != "call-checkpoints":
        raise ValueError(f"The recovery checkpoint does not match the logical call for {session_key}")
    artifact_dir = checkpoint_path.parent.parent
    from arc_llm import read_recovery_context
    fresh = read_recovery_context(
        artifact_dir,
        idempotency_key=idempotency_key,
        session_manager=session_manager,
        session_key=session_key,
    )
    if fresh.checkpoint_path is None or fresh.checkpoint_path.resolve(strict=False) != checkpoint_path:
        raise ValueError(f"The recovery checkpoint for {session_key} is missing or changed")
    if fresh.session_key != session_key or fresh.generation != ledger_generation:
        raise ValueError(f"The recovered call identity does not match session {session_key}")
    if (
        fresh.provider != ref.provider
        or fresh.model != ref.model
        or fresh.runtime_fingerprint != ref.runtime_fingerprint
    ):
        raise ValueError(f"The recovered provider/model runtime does not match session {session_key}")
    if context.get("checkpoint_state") != fresh.checkpoint_state:
        raise ValueError(f"The recovery checkpoint state changed for session {session_key}")
    if context.get("submission_state") != fresh.submission_state:
        raise ValueError(f"The recovery submission state changed for session {session_key}")
    if not fresh.resumable or not fresh.native_session_id:
        raise ValueError(f"The supervised call for {session_key} has no resumable native session")
    if not context_native_id or context_native_id != fresh.native_session_id:
        raise ValueError(f"The recovery native session id changed for session {session_key}")
    return {
        "session_key": session_key,
        "idempotency_key": idempotency_key,
        "provider": ref.provider,
        "model": ref.model,
        "runtime_fingerprint": ref.runtime_fingerprint,
        "generation": ref.generation,
        "native_session_id_to_restore": fresh.native_session_id,
        "ledger_path": str(ledger_path),
        "segment_id": str(active_supervision.get("segment_id") or ""),
    }


def _reconstruct_unresolved_native_resume_contexts(
    checkpoint_dir: Path,
    *,
    session_manager: Any,
    excluded_keys: set[str],
    excluded_identities: set[tuple[str, str, str, int, str]] | None = None,
    suppress_typed_idle: bool = False,
    fresh_required_entries: list[dict[str, Any]] | None = None,
    allow_explicit_legacy: bool = False,
) -> list[dict[str, Any]]:
    """Recover submitted calls even when the lane callback never ran."""
    from arc_llm import read_recovery_context

    # Retained as a private-call compatibility parameter only.  A bare
    # idempotency key is not an ownership identity and must never suppress a
    # different ledger/session/logical-unit/generation tuple.
    del excluded_keys
    recovered: list[dict[str, Any]] = []
    excluded_identities = set(excluded_identities or set())
    receipt_bindings: list[dict[str, Any]] = []
    for receipt_path, receipt in discover_submission_receipts(checkpoint_dir):
        if receipt.get("sealed") is not True:
            continue
        artifact = (
            checkpoint_dir.resolve(strict=False)
            / str(receipt.get("artifact_dir") or "")
        ).resolve(strict=False)
        receipt_bindings.append({
            **receipt,
            "resolved_artifact_dir": str(artifact),
            "resolved_ledger_path": str((
                checkpoint_dir.resolve(strict=False)
                / str(receipt.get("ledger_path") or "")
            ).resolve(strict=False)),
            "receipt_path": str(receipt_path),
            "receipt_reference": submission_receipt_reference(
                receipt_path, checkpoint_dir=checkpoint_dir,
            ),
        })
    registered_ledgers = {
        str(path.resolve(strict=False))
        for path in _all_lane_ledger_paths(
            checkpoint_dir, include_explicit_legacy=allow_explicit_legacy,
        )
    }
    # Stateless pipeline controls have no native session record.  Their sealed
    # receipt is the sole durable ownership proof after a callback crash.
    for binding in receipt_bindings:
        if binding.get("checkpoint_session_key") is not None:
            continue
        ledger_path = Path(str(binding.get("resolved_ledger_path") or ""))
        if str(ledger_path.resolve(strict=False)) not in registered_ledgers:
            continue
        try:
            if allow_explicit_legacy:
                ledger = _read_recovery_ledger(
                    checkpoint_dir, ledger_path, allow_explicit_legacy=True,
                )
                ledger_digest = None
            else:
                ledger, ledger_digest = read_registered_lane_ledger(
                    checkpoint_dir, ledger_path,
                )
        except (
            OSError, ValueError, json.JSONDecodeError, SecureReadError,
            LaneLedgerRegistryError,
        ):
            continue
        logical_unit = str(binding.get("logical_unit") or "")
        session_key = str(binding.get("session_key") or "")
        generation = int(binding.get("generation") or 0)
        key = str(binding.get("idempotency_key") or "")
        if not logical_unit or not session_key or generation < 1 or not key:
            continue
        unresolved_ids = {
            str(item.get("segment_id") or "")
            for item in ledger.get("blocks") or []
            if isinstance(item, Mapping) and item.get("state") != "accepted"
        }
        if logical_unit not in unresolved_ids:
            continue
        try:
            _validate_pipeline_submission_reference(
                binding["receipt_reference"],
                checkpoint_dir=checkpoint_dir,
                ledger_path=ledger_path,
                session_key=session_key,
                logical_unit=logical_unit,
                generation=generation,
                idempotency_key=key,
            )
            artifact_dir = Path(str(binding["resolved_artifact_dir"]))
            checkpoint_path = resolve_recovery_path(
                checkpoint_dir, binding.get("checkpoint_path"),
            )
        except (RecoveryResponseError, OSError, ValueError):
            continue
        try:
            checkpoint = _read_recovery_json(checkpoint_dir, checkpoint_path)
        except SecureReadError:
            continue
        logical = checkpoint.get("logical_identity") if isinstance(checkpoint, Mapping) else None
        recipe = checkpoint.get("request_recipe") if isinstance(checkpoint, Mapping) else None
        if (
            not isinstance(checkpoint, Mapping)
            or not isinstance(logical, Mapping)
            or not isinstance(recipe, Mapping)
            or logical.get("session_key") is not None
            or logical.get("generation") is not None
            or logical.get("idempotency_key") != key
            or checkpoint.get("state") not in {"submitted", "resuming", "failed"}
            or checkpoint.get("submission_state") not in {"submitted", "unknown"}
        ):
            continue
        fresh = read_recovery_context(artifact_dir, idempotency_key=key)
        if (
            fresh.checkpoint_path is None
            or fresh.checkpoint_path.resolve(strict=False)
            != checkpoint_path.resolve(strict=False)
        ):
            continue
        recovery_context = _recovery_context_json(
            fresh, logical_unit=logical_unit,
        )
        recovery_context.update({
            "submission_receipt": dict(binding["receipt_reference"]),
            "recovery_unit": str(binding.get("recovery_unit") or ""),
            "session_key": session_key,
            "generation": generation,
            "provider": logical.get("provider"),
            "model": logical.get("model"),
            "runtime_fingerprint": recipe.get("runtime_fingerprint"),
            "stateless_control": True,
        })
        candidate_entry = {
            "ledger_path": str(ledger_path),
            "session_key": session_key,
            "segment_id": logical_unit,
            "idempotency_key": key,
            "initial_generation": generation,
            "target_generation": generation + 1,
            "recovery_context": recovery_context,
            "reconstructed_from_durable_state": True,
        }
        recovery_identity = _native_context_recovery_identity(candidate_entry)
        if recovery_identity in excluded_identities:
            continue
        audit = _fresh_auto_recovery_audit(candidate_entry, checkpoint_dir)
        if suppress_typed_idle and audit["fresh_generation_required"]:
            candidate_entry.update(audit)
            candidate_entry["fresh_task_start_segment_id"] = next((
                str(item.get("segment_id") or "")
                for item in ledger.get("blocks") or []
                if isinstance(item, Mapping) and item.get("state") != "accepted"
            ), logical_unit)
            mark_needs_supervision(
                ledger_path,
                segment_id=logical_unit,
                reason="typed idle timeout requires a fresh generation",
                recovery_context=recovery_context,
                expected_ledger_sha256=ledger_digest,
                checkpoint_dir=checkpoint_dir if ledger_digest is not None else None,
            )
            if fresh_required_entries is not None:
                fresh_required_entries.append(candidate_entry)
        else:
            mark_needs_supervision(
                ledger_path,
                segment_id=logical_unit,
                reason="submitted stateless call discovered before control progress was persisted",
                recovery_context=recovery_context,
                expected_ledger_sha256=ledger_digest,
                checkpoint_dir=checkpoint_dir if ledger_digest is not None else None,
            )
        excluded_identities.add(recovery_identity)
    progress_paths = (
        sorted(checkpoint_dir.rglob("progress.jsonl"))
        if allow_explicit_legacy else []
    )
    durable_checkpoints: list[tuple[Path, dict[str, Any]]] = []
    if allow_explicit_legacy:
        checkpoint_paths = sorted(checkpoint_dir.rglob("call-checkpoints/*.json"))
    else:
        checkpoint_paths = []
        seen_checkpoint_paths: set[str] = set()
        for binding in receipt_bindings:
            try:
                path = resolve_recovery_path(
                    checkpoint_dir, binding.get("checkpoint_path"),
                )
            except RecoveryResponseError:
                continue
            address = str(path)
            if address not in seen_checkpoint_paths:
                seen_checkpoint_paths.add(address)
                checkpoint_paths.append(path)
    for path in checkpoint_paths:
        try:
            checkpoint = _read_recovery_json(checkpoint_dir, path)
        except SecureReadError:
            continue
        if not isinstance(checkpoint, dict):
            continue
        state = checkpoint.get("state")
        if (
            (
                state in {"submitted", "resuming"}
                or (state == "failed" and checkpoint.get("resumable"))
            )
            and checkpoint.get("submission_state") in {"submitted", "unknown"}
        ):
            durable_checkpoints.append((path, checkpoint))
    for ledger_path in _all_lane_ledger_paths(
        checkpoint_dir, include_explicit_legacy=allow_explicit_legacy,
    ):
        try:
            ledger = _read_recovery_ledger(
                checkpoint_dir,
                ledger_path,
                allow_explicit_legacy=allow_explicit_legacy,
            )
        except SecureReadError:
            continue
        unresolved_ids = {
            str(item.get("segment_id") or "")
            for item in ledger.get("blocks") or []
            if item.get("state") != "accepted"
        }
        if not unresolved_ids:
            continue
        session_key = f"{ledger.get('chapter_id')}:{ledger.get('lane')}"
        ref = session_manager.get_existing(session_key)
        if ref is None or not ref.native_session_id:
            continue
        generation = int(ledger.get("generation") or 0)
        if generation != ref.generation:
            continue
        # Candidate ownership is the complete recovery identity, never merely
        # an idempotency key.  A reused key in another lane/logical unit must
        # remain independently auditable, while two artifact addresses for the
        # exact same identity are ambiguous and therefore fail closed below.
        candidates: dict[
            tuple[str, str, str, int, str],
            dict[str, tuple[Path, dict[str, Any]]],
        ] = {}

        def add_candidate(
            artifact_dir: Path, key: str, event: Mapping[str, Any],
        ) -> None:
            matching_units = {
                str(item.get("logical_unit") or "")
                for item in receipt_bindings
                if item.get("resolved_artifact_dir")
                == str(artifact_dir.resolve(strict=False))
                and item.get("resolved_ledger_path")
                == str(ledger_path.resolve(strict=False))
                and item.get("session_key") == session_key
                and int(item.get("generation") or 0) == generation
                and item.get("idempotency_key") == key
                and str(item.get("logical_unit") or "") in unresolved_ids
            }
            if not matching_units:
                if not allow_explicit_legacy:
                    return
                matching_units = {""}
            artifact_address = str(artifact_dir.resolve(strict=False))
            for logical_unit in matching_units:
                identity = (
                    str(ledger_path.resolve(strict=False)), session_key,
                    logical_unit, generation, key,
                )
                candidates.setdefault(identity, {})[artifact_address] = (
                    artifact_dir, dict(event),
                )
        for checkpoint_path, checkpoint in durable_checkpoints:
            logical = checkpoint.get("logical_identity")
            recipe = checkpoint.get("request_recipe")
            if not isinstance(logical, dict):
                continue
            key = str(logical.get("idempotency_key") or "")
            if (
                logical.get("session_key") != session_key
                or logical.get("generation") != generation
                or logical.get("provider") != ref.provider
                or logical.get("model") != ref.model
                or not key
            ):
                continue
            artifact_dir = checkpoint_path.parent.parent
            add_candidate(artifact_dir, key, {
                "session_key": session_key,
                "idempotency_key": key,
                "generation": generation,
                "provider": ref.provider,
                "model": ref.model,
                "runtime_fingerprint": (
                    recipe.get("runtime_fingerprint")
                    if isinstance(recipe, dict) else None
                ),
            })
        for progress_path in progress_paths:
            try:
                relative = progress_path.resolve(strict=False).relative_to(
                    checkpoint_dir.resolve(strict=False)
                )
                raw_progress = read_bounded_file(
                    checkpoint_dir.resolve(strict=False),
                    relative,
                    max_bytes=_MAX_RECOVERY_CONTROL_BYTES,
                    suffixes=(".jsonl",),
                )
                lines = raw_progress.decode("utf-8").splitlines()
            except (OSError, UnicodeError, ValueError, SecureReadError):
                continue
            for line in lines:
                try:
                    event = json.loads(line)
                except (TypeError, json.JSONDecodeError):
                    continue
                if not isinstance(event, dict) or event.get("session_key") != session_key:
                    continue
                key = str(event.get("idempotency_key") or "")
                if (
                    not key
                    or event.get("generation") != generation
                ):
                    continue
                artifact_dir = progress_path.parent
                add_candidate(artifact_dir, key, event)
        for candidate_identity, artifact_candidates in sorted(candidates.items()):
            if len(artifact_candidates) != 1:
                continue
            (_artifact_address, (artifact_dir, event)), = artifact_candidates.items()
            (
                _candidate_ledger, _candidate_session, logical_hint,
                _candidate_generation, key,
            ) = candidate_identity
            checkpoint_path = (
                artifact_dir / "call-checkpoints"
                / f"idempotency-{hashlib.sha256(key.encode('utf-8')).hexdigest()}.json"
            )
            try:
                checkpoint = _read_recovery_json(checkpoint_dir, checkpoint_path)
            except SecureReadError:
                continue
            if not isinstance(checkpoint, dict):
                continue
            state = checkpoint.get("state")
            if not (
                state in {"submitted", "resuming"}
                or (state == "failed" and checkpoint.get("resumable"))
            ) or checkpoint.get("submission_state") not in {"submitted", "unknown"}:
                continue
            if any(
                event.get(field) is not None and event.get(field) != expected
                for field, expected in (
                    ("provider", ref.provider),
                    ("model", ref.model),
                    ("runtime_fingerprint", ref.runtime_fingerprint),
                )
            ):
                continue
            matching_bindings = [
                item for item in receipt_bindings
                if item.get("resolved_artifact_dir")
                == str(artifact_dir.resolve(strict=False))
                and item.get("resolved_ledger_path")
                == str(ledger_path.resolve(strict=False))
                and item.get("session_key") == session_key
                and int(item.get("generation") or 0) == generation
                and item.get("idempotency_key") == key
                and str(item.get("logical_unit") or "") in unresolved_ids
                and (
                    not logical_hint
                    or str(item.get("logical_unit") or "") == logical_hint
                )
            ]
            if len(matching_bindings) > 1:
                continue
            binding = matching_bindings[0] if matching_bindings else None
            if binding is not None:
                expected_ledger = (
                    checkpoint_dir.resolve(strict=False)
                    / str(binding.get("ledger_path") or "")
                ).resolve(strict=False)
                segment_id = str(binding.get("logical_unit") or "")
                if (
                    expected_ledger != ledger_path.resolve(strict=False)
                    or segment_id not in unresolved_ids
                ):
                    continue
                try:
                    _validate_pipeline_submission_reference(
                        binding["receipt_reference"],
                        checkpoint_dir=checkpoint_dir,
                        ledger_path=ledger_path,
                        session_key=session_key,
                        logical_unit=segment_id,
                        generation=generation,
                        idempotency_key=key,
                    )
                except RecoveryResponseError:
                    continue
            else:
                # Legacy records may still be offered for explicit native
                # reconciliation, but automatic typed-idle replacement needs
                # an exact production binding and never infers ownership from
                # a call-label substring.
                if not allow_explicit_legacy:
                    continue
                if logical_hint and logical_hint in unresolved_ids:
                    segment_id = logical_hint
                elif len(unresolved_ids) == 1:
                    segment_id = next(iter(unresolved_ids))
                else:
                    matching_ids = [
                        candidate_id for candidate_id in unresolved_ids
                        if candidate_id and candidate_id in key
                    ]
                    if len(matching_ids) != 1:
                        continue
                    segment_id = matching_ids[0]
            recovery_identity = (
                str(ledger_path.resolve(strict=False)),
                session_key,
                segment_id,
                generation,
                key,
            )
            if recovery_identity in excluded_identities:
                continue
            fresh = read_recovery_context(
                artifact_dir,
                idempotency_key=key,
                session_manager=session_manager,
                session_key=session_key,
            )
            if (
                fresh.checkpoint_path is None
                or fresh.checkpoint_path.resolve(strict=False)
                != checkpoint_path.resolve(strict=False)
                or fresh.generation != generation
                or fresh.provider != ref.provider
                or fresh.model != ref.model
                or fresh.runtime_fingerprint != ref.runtime_fingerprint
            ):
                continue
            try:
                if allow_explicit_legacy:
                    current_ledger = _read_recovery_ledger(
                        checkpoint_dir, ledger_path, allow_explicit_legacy=True,
                    )
                    current_ledger_digest = None
                else:
                    current_ledger, current_ledger_digest = (
                        read_registered_lane_ledger(checkpoint_dir, ledger_path)
                    )
            except (SecureReadError, LaneLedgerRegistryError):
                continue
            recovery_context = _recovery_context_json(
                fresh, logical_unit=segment_id,
            )
            if binding is not None:
                recovery_context.update({
                    "submission_receipt": dict(binding["receipt_reference"]),
                    "recovery_unit": str(binding.get("recovery_unit") or ""),
                    "session_key": session_key,
                    "generation": generation,
                })
            candidate_entry = {
                "ledger_path": str(ledger_path),
                "session_key": session_key,
                "segment_id": segment_id,
                "idempotency_key": key,
                "initial_generation": generation,
                "target_generation": generation + 1,
                "recovery_context": recovery_context,
                "reconstructed_from_durable_state": True,
            }
            audit = _fresh_auto_recovery_audit(candidate_entry, checkpoint_dir)
            if suppress_typed_idle and audit["fresh_generation_required"]:
                if binding is None:
                    continue
                candidate_entry.update(audit)
                candidate_entry["fresh_task_start_segment_id"] = next((
                    str(item.get("segment_id") or "")
                    for item in current_ledger.get("blocks") or []
                    if isinstance(item, Mapping) and item.get("state") != "accepted"
                ), segment_id)
                mark_needs_supervision(
                    ledger_path,
                    segment_id=segment_id,
                    reason="typed idle timeout requires a fresh generation",
                    recovery_context=recovery_context,
                    expected_ledger_sha256=current_ledger_digest,
                    checkpoint_dir=(
                        checkpoint_dir if current_ledger_digest is not None else None
                    ),
                )
                if fresh_required_entries is not None:
                    fresh_required_entries.append(candidate_entry)
                excluded_identities.add(recovery_identity)
                continue
            if not current_ledger.get("needs_supervision"):
                mark_needs_supervision(
                    ledger_path,
                    segment_id=segment_id,
                    reason="submitted call discovered before lane progress was persisted",
                    recovery_context=recovery_context,
                    expected_ledger_sha256=current_ledger_digest,
                    checkpoint_dir=(
                        checkpoint_dir if current_ledger_digest is not None else None
                    ),
                )
            recovered.append({
                "session_key": session_key,
                "idempotency_key": key,
                "provider": ref.provider,
                "model": ref.model,
                "runtime_fingerprint": ref.runtime_fingerprint,
                "generation": ref.generation,
                "native_session_id_to_restore": None,
                "ledger_path": str(ledger_path),
                "segment_id": segment_id,
                "reconstructed_from_durable_state": True,
            })
            excluded_identities.add(recovery_identity)
    return recovered


def _restore_native_session_id(session_manager: Any, validated: dict[str, Any], native_id: str) -> None:
    session_key = str(validated["session_key"])
    with session_manager.lock(session_key):
        ref = session_manager.get_existing(session_key)
        if ref is None or (
            ref.provider != validated["provider"]
            or ref.model != validated["model"]
            or ref.runtime_fingerprint != validated["runtime_fingerprint"]
            or ref.generation != validated["generation"]
        ):
            raise ValueError(f"Session {session_key} changed while restoring native recovery state")
        if ref.native_session_id and ref.native_session_id != native_id:
            raise ValueError(f"Session {session_key} already has a different native session id")
        if not ref.native_session_id:
            session_manager.update_native_session_id(session_key, native_id)


def _recovery_context_json(
    context: Any, *, logical_unit: str | None = None,
) -> dict[str, Any]:
    checkpoint = (
        _read_checkpoint_json(
            context.checkpoint_path,
            root=Path(context.checkpoint_path).parent.parent,
        )
        if context.checkpoint_path else None
    )
    return {
        "idempotency_key": context.idempotency_key,
        "checkpoint_path": str(context.checkpoint_path) if context.checkpoint_path else None,
        "checkpoint_state": context.checkpoint_state,
        "submission_state": context.submission_state,
        "native_session_id": context.native_session_id,
        "resumable": bool(context.resumable),
        "progress_journal": str(context.progress_journal) if context.progress_journal else None,
        "latest_progress": context.latest_progress,
        "session_key": context.session_key,
        "generation": context.generation,
        "provider": context.provider,
        "model": context.model,
        "runtime_fingerprint": context.runtime_fingerprint,
        "failure_category": (
            str(checkpoint.get("failure_category") or "")
            if isinstance(checkpoint, dict) else ""
        ),
        **({"logical_unit": logical_unit} if logical_unit else {}),
    }


def _recovery_options(options: BuildOptions) -> dict[str, Any]:
    return {
        "paper_id": options.paper_id,
        "annotation_language": options.annotation_language,
        "source_language": options.source_language,
        "language_was_defaulted": options.language_was_defaulted,
        "provider": options.provider,
        "model": options.model,
        "workers": options.workers,
        "review_context_chars": options.review_context_chars,
        "domain_id": options.domain_id,
        "domain_manifest": str(options.domain_manifest) if options.domain_manifest else None,
        "allow_internet": options.allow_internet,
        "inherit_host_tools": options.inherit_host_tools,
        "skip_translation": options.skip_translation,
        "context_paper_ids": list(options.context_paper_ids),
        "user_intent": options.user_intent,
        "stop_after_first_chapter": options.stop_after_first_chapter,
        "document_kind": options.document_kind,
        "idle_timeout_seconds": options.idle_timeout_seconds,
        "recovery_policy": options.recovery_policy,
        "max_auto_replacements": options.max_auto_replacements,
        "regenerate_lanes": list(options.regenerate_lanes),
        "regenerate_segments": list(options.regenerate_segments),
        "confirm_expensive_regeneration": options.confirm_expensive_regeneration,
        "regenerate_commentary": options.regenerate_commentary,
        "legacy_checkpoint": (
            str(options.legacy_checkpoint) if options.legacy_checkpoint else None
        ),
    }


def _with_effective_source_language(options: BuildOptions) -> BuildOptions:
    """Resolve persisted workflow language metadata without detecting prose."""
    requested = str(options.source_language or "").strip()
    if not requested:
        context = _read_optional_json(options.project_dir.resolve() / "context.json")
        requested = str(
            context.get("source_base_language")
            or context.get("source_language")
            or ""
        ).strip()
    if not requested and options.skip_translation:
        # The existing flag is an authoritative same-base-language assertion.
        # This preserves older direct callers that predate --source-language.
        requested = options.annotation_language
    normalized = normalize_language_tag(requested or "und")
    target = normalize_language_tag(options.annotation_language)
    if (
        options.skip_translation
        and base_language(normalized) not in {"und", "mul"}
        and base_language(normalized) != base_language(target)
    ):
        raise ValueError(
            "--skip-translation requires source and target languages with the same base"
        )
    if normalized == options.source_language:
        return options
    return replace(options, source_language=normalized)


def _language_prompt_contract(version: str, options: BuildOptions) -> str:
    """Preserve English recipe identities while versioning multilingual prompts."""
    source = normalize_language_tag(options.source_language or "und")
    if base_language(source) == "en":
        return version
    return f"{version}:multilingual-source-v1:{source}"


def _multilingual_prompt_source(options: BuildOptions) -> str | None:
    source = normalize_language_tag(options.source_language or "und")
    return None if base_language(source) == "en" else source


def _options_from_recovery(project_dir: Path, value: dict[str, Any]) -> BuildOptions:
    return BuildOptions(
        paper_id=str(value["paper_id"]), project_dir=project_dir,
        annotation_language=str(value.get("annotation_language") or DEFAULT_LANGUAGE),
        source_language=(
            str(value.get("source_language") or "").strip() or None
        ),
        language_was_defaulted=bool(value.get("language_was_defaulted")),
        provider=str(value.get("provider") or "auto"), model=value.get("model"),
        workers=int(value.get("workers") or DEFAULT_WORKERS),
        review_context_chars=int(value.get("review_context_chars") or DEFAULT_REVIEW_CONTEXT_CHARS),
        domain_id=value.get("domain_id"),
        domain_manifest=Path(value["domain_manifest"]) if value.get("domain_manifest") else None,
        allow_internet=bool(value.get("allow_internet", True)),
        inherit_host_tools=bool(value.get("inherit_host_tools", False)),
        skip_translation=bool(value.get("skip_translation", False)),
        context_paper_ids=tuple(value.get("context_paper_ids") or ()),
        user_intent=(str(value.get("user_intent") or "").strip() or None),
        stop_after_first_chapter=bool(value.get("stop_after_first_chapter")),
        document_kind=str(value.get("document_kind") or "auto"),
        idle_timeout_seconds=value.get("idle_timeout_seconds"),
        recovery_policy=str(value.get("recovery_policy") or "auto"),
        max_auto_replacements=int(value.get("max_auto_replacements") or 3),
        regenerate_lanes=tuple(value.get("regenerate_lanes") or ()),
        regenerate_segments=tuple(value.get("regenerate_segments") or ()),
        confirm_expensive_regeneration=bool(value.get("confirm_expensive_regeneration")),
        regenerate_commentary=bool(value.get("regenerate_commentary")),
        legacy_checkpoint=(
            Path(value["legacy_checkpoint"]) if value.get("legacy_checkpoint") else None
        ),
    )


def _load_first_chapter_freeze(
    path: Path, chapters: list[dict[str, Any]], *, required: bool,
) -> dict[str, Any] | None:
    if not path.is_file():
        if required:
            raise RuntimeError("the confirmed first chapter freeze manifest is missing")
        return None
    freeze = read_json(path)
    if (
        freeze.get("schema_version") != "arc.companion.first-chapter-freeze.v3"
        or not chapters or freeze.get("chapter_id") != chapters[0].get("chapter_id")
        or freeze.get("chapter_sha256") != sha256_json(chapters[0])
        or freeze.get("translation_mode") not in {"enabled", "skipped"}
    ):
        raise RuntimeError("the confirmed first chapter freeze manifest is invalid")
    return freeze


def _verify_frozen_first_chapter_pre_review(
    freeze: dict[str, Any], results: dict[str, dict[str, Any]],
) -> None:
    chapter_id = str(freeze["chapter_id"])
    result = results.get(chapter_id)
    if not isinstance(result, dict):
        raise RuntimeError("the confirmed first chapter result is missing before review")
    translation_hash = (
        None if freeze.get("translation_mode") == "skipped"
        else sha256_json(result["translation"])
    )
    if any((
        sha256_json(result["guide"]) != freeze.get("guide_sha256"),
        translation_hash != freeze.get("pre_review_translation_sha256"),
        sha256_json(result["companion"]) != freeze.get("pre_review_annotation_sha256"),
    )):
        raise RuntimeError("the confirmed first chapter changed before remaining chapters started")


def _verify_frozen_first_chapter_final(
    freeze: dict[str, Any], results: dict[str, dict[str, Any]], *,
    translations: dict[str, Any] | None, annotations: dict[str, Any],
) -> None:
    chapter_id = str(freeze["chapter_id"])
    result = results.get(chapter_id)
    if not isinstance(result, dict):
        raise RuntimeError("the confirmed first chapter result is missing after review")
    segment_ids = [item["segment_id"] for item in result["segments"]]
    translation_hash = (
        None if freeze.get("translation_mode") == "skipped"
        else sha256_json({value: translations[value] for value in segment_ids})
    )
    if (
        translation_hash != freeze.get("translation_sha256")
        or sha256_json({value: annotations[value] for value in segment_ids})
        != freeze.get("annotation_sha256")
    ):
        raise RuntimeError("final review attempted to rewrite the confirmed first chapter")


def _write_review_overlays(
    checkpoint_dir: Path, chapter_results: dict[str, dict[str, Any]], *,
    translations: dict[str, Any] | None, annotations: dict[str, Any],
) -> None:
    for chapter_id, result in chapter_results.items():
        segment_ids = [str(item["segment_id"]) for item in result["segments"]]
        lanes = [("companion", annotations)]
        if translations is not None:
            lanes.insert(0, ("translation", translations))
        for lane, values in lanes:
            ledger_path = checkpoint_dir / "chapters" / chapter_id / f"{lane}-ledger.json"
            ledger = read_json(ledger_path)
            reviewed = {value: values[value] for value in segment_ids}
            ledger_blocks = {str(item["segment_id"]): item for item in ledger["blocks"]}
            if any(ledger_blocks[value].get("state") != "accepted" for value in segment_ids):
                raise RuntimeError(f"review overlay requires an accepted {chapter_id}:{lane} ledger")
            write_json(
                checkpoint_dir / "chapters" / chapter_id / f"{lane}-review-overlay.json",
                {
                    "schema_version": "arc.companion.chapter-review-overlay.v1",
                    "chapter_id": chapter_id, "lane": lane,
                    "base_accepted_chain_sha256": ledger.get("accepted_chain_sha256"),
                    "reviewed_output_sha256": sha256_json(reviewed),
                    "blocks": [{
                        "segment_id": value,
                        "base_output_sha256": ledger_blocks[value].get("output_sha256"),
                        "accepted_chain_sha256": ledger_blocks[value].get("accepted_chain_sha256"),
                        "reviewed_output_sha256": sha256_json(values[value]),
                        "validation_receipt": {
                            "review_applied": True,
                            "reviewed_output_matches_sha256": True,
                        },
                    } for value in segment_ids],
                    "validation_receipt": {
                        "reviewed_segment_ids": segment_ids,
                        "review_output_matches_sha256": True,
                    },
                },
            )


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
    accepted_callback: Callable[[str, str, dict[str, Any]], None] | None = None,
    intent_guidance: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Drain translation and annotation concurrently for one scheduled wave."""
    if not segments:
        return {"translation": {}, "annotation": {}}
    with ThreadPoolExecutor(max_workers=1 if options.skip_translation else 2) as executor:
        lane_futures = {
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
                accepted_callback=accepted_callback,
                intent_guidance=intent_guidance,
            ),
        }
        if not options.skip_translation:
            lane_futures["translation"] = executor.submit(
                _generate_translations,
                segments,
                options=options,
                bundle=bundle,
                glossary=glossary,
                protected_names=protected_names,
                checkpoint_dir=checkpoint_dir,
                llm=llm,
                accepted_callback=accepted_callback,
                intent_guidance=intent_guidance,
            )
        lane_results: dict[str, dict[str, dict[str, Any]]] = {"translation": {}}
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
    translations: dict[str, dict[str, Any]] | None,
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
    chapters: list[dict[str, Any]] | None = None,
    chapter_guides: dict[str, dict[str, Any]] | None = None,
    source_language: str | None = None,
    title_translations: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Render, validate, and atomically publish one preview or final PDF artifact."""
    # Final builds never overwrite a path referenced by the published state.
    # A new revision directory makes every target immutable; the later state
    # write is the sole publication commit. Preview paths remain stable because
    # they are not last-good deliverables and existing CLI/tests expose them.
    artifact_dir = output_dir
    if manifest_name == "source-manifest.json":
        artifact_dir = (
            output_dir / ".arc-companion" / "renders" / "pdf"
            / f"{safe_name(stem)}-{uuid.uuid4().hex[:12]}"
        )
        artifact_dir.mkdir(parents=True, exist_ok=False)
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
        output_dir=artifact_dir,
        language=language,
        metadata=metadata,
        translations=translations,
        glossary=glossary,
        evidence_by_segment=evidence_by_segment,
        augmentation_scope=augmentation_scope,
        chapters=chapters,
        chapter_guides=chapter_guides,
        source_language=source_language,
        title_translations=title_translations,
    )
    fidelity_errors = validate_tex_fidelity(tex, document, source_manifest)
    if fidelity_errors:
        raise LatexError("source fidelity validation failed: " + "; ".join(fidelity_errors))

    tex_path = artifact_dir / f"{stem}.tex"
    pdf_path = artifact_dir / f"{stem}.pdf"
    manifest_path = artifact_dir / manifest_name
    validation_path = artifact_dir / validation_name
    staging_stem = f"arc-companion-building-{safe_name(stem)}-{uuid.uuid4().hex[:12]}"
    building_tex = artifact_dir / f"{staging_stem}.tex"
    building_pdf = artifact_dir / f"{staging_stem}.pdf"
    building_manifest = artifact_dir / f"{staging_stem}-manifest.json"
    building_validation = artifact_dir / f"{staging_stem}-validation.json"
    staging_paths = (building_tex, building_pdf, building_manifest, building_validation)
    try:
        write_text(building_tex, tex)
        compiler(building_tex, building_pdf)
        pdf_report = pdf_validator(building_pdf)
        write_json(building_manifest, source_manifest)
        write_json(
            building_validation,
            {
                "ok": True,
                "pdf": pdf_report,
                "fidelity_errors": [],
                "warnings": list(source_manifest.get("render_warnings") or []),
            },
        )
        _publish_artifact_replace(building_tex, tex_path)
        _publish_artifact_replace(building_pdf, pdf_path)
        _publish_artifact_replace(building_manifest, manifest_path)
        _publish_artifact_replace(building_validation, validation_path)
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


def _publish_artifact_replace(source: Path, target: Path) -> None:
    """Fault-injection seam for publishing files inside a new immutable revision."""
    os.replace(source, target)


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
        result[str(segment.get("segment_id") or "")] = records
    return result


def validate_project(project_dir: Path, *, pdf_validator: Callable[[Path], dict[str, object]] = validate_pdf) -> dict[str, Any]:
    status = read_status(project_dir)
    if not status.get("ok"):
        return status
    state = status["data"]
    published_value = state.get("published") or {}
    if not isinstance(published_value, dict):
        return err("companion_validation_failed", "Published companion state is invalid")
    published = dict(published_value)
    pdf_value = published.get("pdf") or {}
    web_value = published.get("web") or {}
    if not isinstance(pdf_value, dict) or not isinstance(web_value, dict):
        return err("companion_validation_failed", "Published companion outputs are invalid")
    effective = normalize_run_root_pdf_state({
        **state,
        **dict(pdf_value),
        **dict(web_value),
    })
    pdf = Path(str(effective.get("output_pdf") or ""))
    tex = Path(str(effective.get("output_tex") or ""))
    manifest_path = Path(str(effective.get("source_manifest_path") or ""))
    if not tex.is_file() or not manifest_path.is_file():
        return err("companion_validation_failed", "TeX or source manifest is missing")
    try:
        if not _published_pdf_outputs_match(effective):
            raise RuntimeError("completed companion outputs do not match their recorded hashes")
        if not _run_root_pdf_output_matches(effective, project_dir.resolve()):
            raise RuntimeError(
                "run-root delivery PDF is missing, outside the resolved "
                "--project-dir, or has a hash mismatch"
            )
        content_sha256 = str(published.get("content_sha256") or "")
        if content_sha256:
            from .content import load_reader_content

            document = load_reader_content(
                project_dir.resolve(), content_sha256,
            )["content"]["document"]
        else:
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
        from .web import validate_reader_project
        web_report = (
            validate_reader_project(project_dir.resolve(), state=effective)
            if effective.get("output_html") else None
        )
    except (OSError, RuntimeError, ValueError) as exc:
        return err("companion_validation_failed", str(exc))
    return ok({
        "pdf": report,
        "web": web_report,
        "manifest": manifest,
        "output_pdf": str(pdf),
        "output_run_pdf": effective.get("output_run_pdf"),
    })


def _published_pdf_outputs_match(state: dict[str, Any]) -> bool:
    """Validate a last-good PDF revision independently of the active run."""
    for path_key, hash_key in (
        ("output_tex", "output_tex_sha256"),
        ("output_pdf", "output_pdf_sha256"),
        ("source_manifest_path", "source_manifest_sha256"),
        ("validation_path", "validation_sha256"),
    ):
        value = state.get(path_key)
        expected = str(state.get(hash_key) or "")
        if not value or not expected:
            return False
        path = Path(str(value))
        if not path.is_file() or path.stat().st_size == 0 or sha256_file(path) != expected:
            return False
    return True


def _run_root_pdf_output_matches(
    state: Mapping[str, Any], run_root: Path,
) -> bool:
    """Validate an optional delivery in the exact resolved companion run root."""

    normalized = normalize_run_root_pdf_state(state)
    value = normalized.get("output_run_pdf")
    expected = normalized.get("output_run_pdf_sha256")
    if value is None and expected is None:
        return True
    if not value or not expected:
        return False
    path = Path(str(value))
    return (
        path.parent == run_root.resolve()
        and not path.is_symlink()
        and path.is_file()
        and path.stat().st_size > 0
        and sha256_file(path) == str(expected)
        and str(expected) == str(normalized.get("output_pdf_sha256") or "")
    )


def _annotation_checkpoint_input_sha256(
    segment: dict[str, Any],
    *,
    bundle: SourceBundle,
    blocks_by_id: dict[str, dict[str, Any]],
    options: BuildOptions,
    segment_glossary: dict[str, Any],
    segment_evidence: dict[str, Any],
    protected_names: list[str],
    domain_context: dict[str, Any] | None,
    intent_guidance: Mapping[str, Any] | None,
) -> str:
    """Bind an annotation checkpoint to its current deterministic inputs."""

    return _segment_input_hash(
        segment,
        blocks_by_id,
        glossary=segment_glossary,
        extra={
            "evidence": segment_evidence,
            "names": protected_names,
            "paper_context": _full_paper_context(
                bundle.document, segment,
                blocks_by_id=blocks_by_id, options=options,
            ),
            "runtime_access": _generation_runtime_policy(options),
            **(
                {"intent_guidance": _intent_guidance_identity(intent_guidance)}
                if intent_guidance is not None else {}
            ),
            "domain_context": domain_context,
        },
    )


def _write_annotation_checkpoint(
    checkpoint_dir: Path,
    *,
    segment: dict[str, Any],
    generation: int,
    annotation: dict[str, Any],
    input_sha256: str,
) -> None:
    """Persist one locally validated annotation for its owning generation."""

    segment_id = str(segment["segment_id"])
    write_json(
        _generation_segment_artifact_dir(
            checkpoint_dir, "annotations", segment_id, generation,
        ) / f"{_segment_checkpoint_name(segment_id)}.json",
        {
            "schema_version": ANNOTATION_CHECKPOINT_VERSION,
            "segment_id": segment_id,
            "generation": generation,
            "input_sha256": input_sha256,
            "annotation": annotation,
        },
    )


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
    generation: int = 1,
    accepted_callback: Callable[[str, str, dict[str, Any]], None] | None = None,
    force_generation: bool = False,
    intent_guidance: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    by_id = {block_id(block): block for block in bundle.document["blocks"]}
    usage_state: dict[str, Any] = {"counts": {}, "topics": []}
    segment_evidence_by_id = {
        str(segment["segment_id"]): _evidence_for_segment(
            segment, by_id, evidence, usage_state=usage_state,
        )
        for segment in segments
    }
    segment_evidence_dir = checkpoint_dir / "segment-evidence"
    segment_evidence_dir.mkdir(parents=True, exist_ok=True)
    output: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for segment in segments:
        annotation_dir = _generation_segment_artifact_dir(
            checkpoint_dir, "annotations", str(segment["segment_id"]), generation,
        )
        annotation_dir.mkdir(parents=True, exist_ok=True)
        path = annotation_dir / f"{_segment_checkpoint_name(segment['segment_id'])}.json"
        segment_evidence = segment_evidence_by_id[str(segment["segment_id"])]
        segment_glossary = (
            {}
            if options.skip_translation
            else project_segment_glossary(
                _augmentation_blocks(segment, by_id), glossary,
            )
        )
        write_json(
            segment_evidence_dir / f"{_segment_checkpoint_name(segment['segment_id'])}.json",
            {
                "schema_version": "arc.companion.segment-evidence-checkpoint.v1",
                "segment_id": segment["segment_id"],
                "input_sha256": sha256_json(segment_evidence),
                "evidence": segment_evidence,
            },
        )
        if bool(segment.get("structural_only")):
            empty_annotation = {
                "explanation": "", "commentary": "", "commentary_sources": [],
                "prior_work": [], "later_work": [],
            }
            output[str(segment["segment_id"])] = empty_annotation
            if accepted_callback is not None:
                accepted_callback(
                    "annotation", str(segment["segment_id"]), empty_annotation
                )
            continue
        if path.is_file() and not force_generation and not options.regenerate_commentary:
            checkpoint = read_json(path)
            expected_hash = _annotation_checkpoint_input_sha256(
                segment,
                bundle=bundle,
                blocks_by_id=by_id,
                options=options,
                segment_glossary=segment_glossary,
                segment_evidence=segment_evidence,
                protected_names=protected_names,
                domain_context=domain_context,
                intent_guidance=intent_guidance,
            )
            if (
                isinstance(checkpoint, dict)
                and checkpoint.get("schema_version") == ANNOTATION_CHECKPOINT_VERSION
                and checkpoint.get("segment_id") == segment["segment_id"]
                and _artifact_payload_generation(
                    checkpoint, checkpoint_dir, "annotations",
                    str(segment["segment_id"]),
                ) == generation
                and checkpoint.get("input_sha256") == expected_hash
                and isinstance(checkpoint.get("annotation"), dict)
            ):
                try:
                    cached_annotation = _validate_direct_annotation_sources(
                        checkpoint["annotation"],
                        allowed_urls=(
                            None if options.allow_internet
                            else _available_source_urls(segment_evidence)
                        ),
                    )
                except RuntimeError:
                    pass
                else:
                    output[segment["segment_id"]] = cached_annotation
                    if accepted_callback is not None:
                        accepted_callback("annotation", str(segment["segment_id"]), cached_annotation)
                    continue
        pending.append(segment)

    def generate(segment: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        selected = [
            _annotation_input_block(by_id[value], bundle.document)
            for value in _augmentation_block_ids(segment, by_id)
        ]
        segment_glossary = (
            {}
            if options.skip_translation
            else project_segment_glossary(
                _augmentation_blocks(segment, by_id), glossary,
            )
        )
        segment_evidence = segment_evidence_by_id[str(segment["segment_id"])]
        paper_context = _full_paper_context(
            bundle.document, segment, blocks_by_id=by_id, options=options
        )
        segment_id = str(segment["segment_id"])
        ordered_units = [str(item["segment_id"]) for item in pending]
        acceptance_path = (
            _generation_segment_artifact_dir(
                checkpoint_dir, "annotations", segment_id, generation,
            )
            / f"{_segment_checkpoint_name(segment_id)}.json"
        )
        annotation_input_sha256 = _annotation_checkpoint_input_sha256(
            segment,
            bundle=bundle,
            blocks_by_id=by_id,
            options=options,
            segment_glossary=segment_glossary,
            segment_evidence=segment_evidence,
            protected_names=protected_names,
            domain_context=domain_context,
            intent_guidance=intent_guidance,
        )
        value = _llm_call(
            llm,
            _guided_prompt(_bounded_annotation_prompt(
                _semantic_segment_descriptor(segment),
                selected,
                language=options.annotation_language,
                metadata=_annotation_metadata(bundle.metadata),
                evidence={"sources": segment_evidence.get("bounded_sources") or []},
                glossary=segment_glossary,
                protected_names=protected_names,
                paper_context=paper_context,
                domain_context=domain_context,
                source_language=_multilingual_prompt_source(options),
            ), intent_guidance, lane="commentary"),
            ANNOTATION_SCHEMA,
            options=options,
            artifact_dir=(
                _generation_segment_artifact_dir(
                    checkpoint_dir, "llm/annotations",
                    str(segment["segment_id"]), generation,
                )
                / _segment_checkpoint_name(segment["segment_id"])
            ),
            call_label=f"companion-annotation-{segment['segment_id']}",
            model_tier=ANNOTATION_TIER,
            allow_internet=True,
            paper_access_policy=_guidance_policy(intent_guidance, lane="commentary"),
            intent_guidance=intent_guidance,
            intent_guidance_lane="commentary",
            recovery_descriptor=submission_descriptor(
                unit="annotation",
                logical_unit=segment_id,
                checkpoint_dir=checkpoint_dir,
                artifact_root=(
                    _generation_segment_artifact_dir(
                        checkpoint_dir, "llm/annotations", segment_id, generation,
                    ) / _segment_checkpoint_name(segment_id)
                ),
                acceptance_checkpoint=acceptance_path,
                input_sha256=annotation_input_sha256,
                ordered_siblings=ordered_units,
                suffix=ordered_units[ordered_units.index(segment_id):],
            ),
        )
        normalized = _validate_direct_annotation_sources(
            value,
            allowed_urls=(
                None if options.allow_internet
                else _available_source_urls(segment_evidence)
            ),
        )
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
                _write_annotation_checkpoint(
                    checkpoint_dir,
                    segment=segment,
                    generation=generation,
                    annotation=value,
                    input_sha256=_annotation_checkpoint_input_sha256(
                        segment,
                        bundle=bundle,
                        blocks_by_id=by_id,
                        options=options,
                        segment_glossary=project_segment_glossary(
                            _augmentation_blocks(segment, by_id), glossary,
                        ),
                        segment_evidence=segment_evidence_by_id[str(segment_id)],
                        protected_names=protected_names,
                        domain_context=domain_context,
                        intent_guidance=intent_guidance,
                    ),
                )
                if _pipeline_control_receipt_exists(
                    checkpoint_dir, "annotation", str(segment_id),
                ):
                    _accept_registered_pipeline_control(
                        checkpoint_dir, "annotation", str(segment_id),
                    )
                if accepted_callback is not None:
                    accepted_callback("annotation", str(segment_id), value)
        if failures:
            raise CompanionLaneError("annotation", failures)
    return {segment["segment_id"]: output[segment["segment_id"]] for segment in segments}


def _bind_translation_repair_acceptance(
    checkpoint_dir: Path,
    *,
    segment_id: str,
    generation: int,
    input_sha256: str,
    translation_checkpoint: Path,
    provenance: Mapping[str, Any],
) -> None:
    """Bind each validated paid repair to the final business checkpoint."""

    final_reference = {
        "path": translation_checkpoint.resolve().relative_to(
            checkpoint_dir.resolve()
        ).as_posix(),
        "sha256": sha256_file(translation_checkpoint),
    }
    repairs = [
        item for item in provenance.get("repairs") or []
        if isinstance(item, Mapping)
    ]
    units = (
        (
            "token-placement",
            "translation-token-repair",
            f"{segment_id}:token-repair",
            _translation_token_attempt_path(checkpoint_dir, segment_id, generation),
        ),
        (
            "coverage",
            "translation-coverage-repair",
            f"{segment_id}:coverage-repair",
            _translation_coverage_attempt_path(checkpoint_dir, segment_id, generation),
        ),
    )
    for kind, _unit, logical_unit, marker_path in units:
        if not any(item.get("kind") == kind for item in repairs):
            continue
        marker = _read_checkpoint_json(marker_path, root=checkpoint_dir)
        if (
            not isinstance(marker, Mapping)
            or marker.get("status") != "validated"
            or marker.get("segment_id") != segment_id
            or int(marker.get("generation") or 0) != generation
            or marker.get("input_sha256") != input_sha256
        ):
            raise RuntimeError(f"validated {kind} repair marker is missing")
        write_json(marker_path, {
            **dict(marker),
            "final_translation_checkpoint": final_reference,
        })
        _accept_registered_pipeline_control(
            checkpoint_dir, _unit, logical_unit,
        )


def _validate_direct_annotation_sources(
    value: Any,
    *,
    allowed_urls: set[str] | None = None,
) -> dict[str, Any]:
    """Validate direct citations without interpreting or registering their claims."""
    if not isinstance(value, dict):
        raise RuntimeError("annotation must be an object")
    expected = {
        "explanation", "commentary", "commentary_sources", "prior_work", "later_work",
    }
    # Local development fakes and interrupted pre-v12 formatter responses can
    # still carry controller-era empty bookkeeping fields.  Collapse only that
    # lossless empty shape; never persist the removed fields.
    if "commentary_sources" not in value:
        legacy_only = set(value) - {"explanation", "commentary", "prior_work", "later_work"}
        if all(value.get(key) in (None, [], "") for key in legacy_only):
            value = {
                "explanation": str(value.get("explanation") or ""),
                "commentary": str(value.get("commentary") or ""),
                "commentary_sources": [],
                "prior_work": value.get("prior_work") or [],
                "later_work": value.get("later_work") or [],
            }
    if set(value) != expected:
        extra = sorted(set(value) - expected)
        missing = sorted(expected - set(value))
        raise RuntimeError(
            f"annotation fields do not match the direct-source contract "
            f"(missing={missing}, extra={extra})"
        )

    def sources(
        values: Any, *, owner: str, require_one: bool = False,
    ) -> list[dict[str, str]]:
        if (
            not isinstance(values, list)
            or len(values) > 3
            or (require_one and not values)
        ):
            raise RuntimeError(f"{owner} sources must be an array with at most three items")
        output: list[dict[str, str]] = []
        seen: set[str] = set()
        for index, raw in enumerate(values, 1):
            if not isinstance(raw, dict) or set(raw) != {"title", "url", "locator"}:
                raise RuntimeError(f"{owner} source {index} must contain title, url, and locator")
            title = raw.get("title")
            url = raw.get("url")
            locator = raw.get("locator")
            if not all(isinstance(item, str) and item.strip() for item in (title, url, locator)):
                raise RuntimeError(f"{owner} source {index} has an empty required field")
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise RuntimeError(f"{owner} source {index} URL must use HTTP(S)")
            duplicate_key = url.casefold()
            if duplicate_key in seen:
                raise RuntimeError(f"{owner} contains a duplicate source URL: {url}")
            if allowed_urls is not None and url not in allowed_urls:
                raise RuntimeError(
                    f"offline annotation cited a source not supplied by the prompt or ARC cache: {url}"
                )
            seen.add(duplicate_key)
            output.append(dict(raw))
        return output

    def claims(values: Any, *, owner: str) -> list[dict[str, Any]]:
        if not isinstance(values, list) or len(values) > 3:
            raise RuntimeError(f"{owner} must be an array with at most three claims")
        output: list[dict[str, Any]] = []
        for index, raw in enumerate(values, 1):
            if not isinstance(raw, dict) or set(raw) != {"text", "sources"}:
                raise RuntimeError(f"{owner} claim {index} must contain text and sources")
            if not isinstance(raw.get("text"), str) or not raw["text"].strip():
                raise RuntimeError(f"{owner} claim {index} has no text")
            output.append({
                "text": raw["text"],
                "sources": sources(
                    raw["sources"], owner=f"{owner} claim {index}", require_one=True
                ),
            })
        return output

    if not isinstance(value.get("explanation"), str) or not isinstance(value.get("commentary"), str):
        raise RuntimeError("annotation explanation and commentary must be strings")
    return {
        "explanation": value["explanation"],
        "commentary": value["commentary"],
        "commentary_sources": sources(
            value["commentary_sources"], owner="commentary"
        ),
        "prior_work": claims(value["prior_work"], owner="prior_work"),
        "later_work": claims(value["later_work"], owner="later_work"),
    }


def _available_source_urls(evidence: dict[str, Any]) -> set[str]:
    """Return URLs already present in bounded prompt/cache material for offline mode."""
    output: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key in (
                "url", "landing_url", "source_url", "html_url", "pdf_url",
                "canonical_locator",
            ):
                candidate = value.get(key)
                if isinstance(candidate, str):
                    parsed = urlparse(candidate)
                    if parsed.scheme in {"http", "https"} and parsed.netloc:
                        output.add(candidate)
            descriptor = value.get("source_descriptor")
            if isinstance(descriptor, dict):
                visit(descriptor)
            for child in value.values():
                if isinstance(child, (dict, list)):
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(evidence)
    return output


def _reusable_lane_output_valid(
    lane: str,
    segment: dict[str, Any],
    output: Any,
    *,
    blocks_by_id: dict[str, dict[str, Any]],
    protected_names: list[str],
) -> bool:
    """Re-run current deterministic output invariants before object reuse."""

    if not isinstance(output, dict):
        return False
    try:
        if lane == "translation":
            _validate_translation(segment, output, blocks_by_id, protected_names)
            return True
        if lane == "commentary":
            required_lists = ("prior_work", "later_work", "commentary_sources")
            if not isinstance(output.get("commentary"), str):
                return False
            if any(not isinstance(output.get(key), list) for key in required_lists):
                return False
            _validate_direct_annotation_sources(output, allowed_urls=None)
            return True
    except (RuntimeError, ValueError, TypeError):
        return False
    return False


def _annotation_source_urls(annotation: dict[str, Any]) -> set[str]:
    output: set[str] = set()
    for source in annotation.get("commentary_sources") or []:
        if isinstance(source, dict) and isinstance(source.get("url"), str):
            output.add(source["url"])
    for field in ("prior_work", "later_work"):
        for claim in annotation.get(field) or []:
            if not isinstance(claim, dict):
                continue
            for source in claim.get("sources") or []:
                if isinstance(source, dict) and isinstance(source.get("url"), str):
                    output.add(source["url"])
    return output


def _deduplicate_evidence_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep the first auditable record for each evidence identity."""
    output: list[dict[str, Any]] = []
    index: dict[str, int] = {}
    for record in records:
        evidence_id = str(record.get("evidence_id") or "")
        if not evidence_id or evidence_id not in index:
            if evidence_id:
                index[evidence_id] = len(output)
            output.append(record)
            continue
        position = index[evidence_id]
        first = output[position]
        validate_registry((first, record))
    return output


LEGACY_GENERATION_OWNERS_SCHEMA_VERSION = (
    "arc.companion.legacy-generation-owners.v1"
)


def _legacy_generation_owners_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "legacy-generation-owners.json"


def _legacy_generation_owner(
    checkpoint_dir: Path, artifact_name: str, segment_id: str,
) -> int | None:
    value = _read_checkpoint_json(
        _legacy_generation_owners_path(checkpoint_dir), root=checkpoint_dir,
    )
    if not isinstance(value, dict):
        return None
    owners = value.get("owners")
    if not isinstance(owners, dict):
        return None
    artifact_owners = owners.get(artifact_name)
    if not isinstance(artifact_owners, dict):
        return None
    try:
        owner = int(artifact_owners.get(segment_id) or 0)
    except (TypeError, ValueError):
        return None
    return owner if owner > 0 else None


def _artifact_payload_generation(
    value: Mapping[str, Any],
    checkpoint_dir: Path,
    artifact_name: str,
    segment_id: str,
) -> int:
    """Resolve missing legacy payload generations through durable ownership."""

    if value.get("generation") is not None:
        return int(value["generation"])
    return _legacy_generation_owner(
        checkpoint_dir, artifact_name, segment_id,
    ) or 1


def _record_legacy_generation_owners(
    checkpoint_dir: Path,
    *,
    lane: str,
    segment_ids: list[str],
    generation: int,
) -> dict[str, Any]:
    """Bind generationless artifacts before rotating their lane generation."""

    path = _legacy_generation_owners_path(checkpoint_dir)
    current = _read_checkpoint_json(path, root=checkpoint_dir)
    value = dict(current) if isinstance(current, dict) else {}
    owners = {
        str(kind): dict(items)
        for kind, items in (value.get("owners") or {}).items()
        if isinstance(items, dict)
    }
    artifact_names = (
        (
            "translations", "translation-drafts",
            "translation-coverage-attempts",
            "translation-token-offset-attempts", "translation-token-attempts",
            "translation-token-offset-repair-drafts", "llm/translations",
        )
        if lane == "translation"
        else ("annotations", "llm/annotations")
    )
    for artifact_name in artifact_names:
        artifact_owners = owners.setdefault(artifact_name, {})
        for segment_id in segment_ids:
            # The first durable attribution wins: a later crash replay or
            # manual override must not relabel an abandoned generation.
            artifact_owners.setdefault(str(segment_id), int(generation))
    value.update({
        "schema_version": LEGACY_GENERATION_OWNERS_SCHEMA_VERSION,
        "owners": owners,
    })
    write_json(path, value)
    return value


def _backfill_legacy_generation_owners(
    checkpoint_dir: Path, transaction: Mapping[str, Any],
) -> None:
    """Recover owner metadata after a crash that already persisted rotation."""

    for raw in transaction.get("replacements") or []:
        if not isinstance(raw, dict):
            continue
        session_key = str(raw.get("session_key") or "")
        suffix_ids = [str(value) for value in raw.get("suffix_segment_ids") or []]
        try:
            source_generation = int(raw.get("source_generation") or 0)
        except (TypeError, ValueError):
            continue
        if session_key and suffix_ids and source_generation > 0:
            _record_legacy_generation_owners(
                checkpoint_dir,
                lane=session_key.rsplit(":", 1)[-1],
                segment_ids=suffix_ids,
                generation=source_generation,
            )


def _generation_segment_artifact_dir(
    checkpoint_dir: Path,
    artifact_name: str,
    segment_id: str,
    generation: int,
) -> Path:
    """Resolve a segment artifact using its persisted generationless owner."""

    owner = _legacy_generation_owner(checkpoint_dir, artifact_name, segment_id)
    base = checkpoint_dir / artifact_name
    if owner == generation or (owner is None and generation == 1):
        return base
    return base / f"generation-{generation}"


def _translation_draft_path(
    checkpoint_dir: Path, segment_id: str, generation: int = 1,
) -> Path:
    return _generation_segment_artifact_dir(
        checkpoint_dir, "translation-drafts", segment_id, generation,
    ) / f"{_segment_checkpoint_name(segment_id)}.json"


def _translation_coverage_attempt_path(
    checkpoint_dir: Path, segment_id: str, generation: int = 1,
) -> Path:
    return (
        _generation_segment_artifact_dir(
            checkpoint_dir, "translation-coverage-attempts", segment_id, generation,
        )
        / f"{_segment_checkpoint_name(segment_id)}.json"
    )


TRANSLATION_COVERAGE_ATTEMPT_SCHEMA_VERSION = (
    "arc.companion.translation-coverage-attempt.v2"
)


def _matching_translation_coverage_attempt(
    checkpoint_dir: Path, segment_id: str, input_sha256: str,
    generation: int = 1,
) -> dict[str, Any] | None:
    path = _translation_coverage_attempt_path(checkpoint_dir, segment_id, generation)
    value = _read_checkpoint_json(path, root=checkpoint_dir)
    if (
        isinstance(value, dict)
        and value.get("schema_version") == TRANSLATION_COVERAGE_ATTEMPT_SCHEMA_VERSION
        and value.get("prompt_version") == TRANSLATION_COVERAGE_REPAIR_PROMPT_VERSION
        and value.get("response_schema_version")
        == TRANSLATION_COVERAGE_REPAIR_SCHEMA_VERSION
        and value.get("segment_id") == segment_id
        and value.get("input_sha256") == input_sha256
        and _artifact_payload_generation(
            value, checkpoint_dir, "translation-coverage-attempts", segment_id,
        ) == generation
    ):
        return value
    return None


def _matching_legacy_translation_coverage_attempt(
    checkpoint_dir: Path, segment_id: str, input_sha256: str,
    generation: int = 1,
) -> dict[str, Any] | None:
    """Return the v1 marker whose response could not be persisted for replay."""
    value = _read_checkpoint_json(
        _translation_coverage_attempt_path(checkpoint_dir, segment_id, generation),
        root=checkpoint_dir,
    )
    if (
        isinstance(value, dict)
        and value.get("schema_version") == "arc.companion.translation-coverage-attempt.v1"
        and value.get("prompt_version") == TRANSLATION_COVERAGE_REPAIR_PROMPT_VERSION
        and value.get("response_schema_version")
        == TRANSLATION_COVERAGE_REPAIR_SCHEMA_VERSION
        and value.get("segment_id") == segment_id
        and value.get("input_sha256") == input_sha256
        and _artifact_payload_generation(
            value, checkpoint_dir, "translation-coverage-attempts", segment_id,
        ) == generation
    ):
        return value
    return None


def _translation_token_attempt_path(
    checkpoint_dir: Path, segment_id: str, generation: int = 1,
) -> Path:
    return (
        _generation_segment_artifact_dir(
            checkpoint_dir, "translation-token-offset-attempts", segment_id, generation,
        )
        / f"{_segment_checkpoint_name(segment_id)}.json"
    )


TRANSLATION_REPAIR_NORMALIZATION_VALIDATOR_VERSION = (
    "arc.companion.translation-repair-normalization-validator.v1"
)


def _repair_response_normalization_path(
    artifact_dir: Path,
    marker_path: Path,
    *,
    checkpoint_dir: Path,
    persisted_response: bool,
) -> Path:
    """Use the attempt path, falling back beside historical marker-only work."""

    canonical = artifact_dir / "response-normalization.json"
    _require_owned_response_normalization_path(
        canonical,
        owner=artifact_dir,
        checkpoint_dir=checkpoint_dir,
    )
    if persisted_response and not artifact_dir.exists():
        selected = marker_path.with_name(
            f"{marker_path.stem}.response-normalization.json"
        )
        owner = marker_path.parent
    else:
        selected = canonical
        owner = artifact_dir
    _require_owned_response_normalization_path(
        selected,
        owner=owner,
        checkpoint_dir=checkpoint_dir,
    )
    return selected


def _require_owned_response_normalization_path(
    path: Path,
    *,
    owner: Path,
    checkpoint_dir: Path,
) -> None:
    """Reject lexical, resolved, or symlink escapes from the owning directory."""

    checkpoint_absolute = Path(os.path.abspath(checkpoint_dir))
    owner_absolute = Path(os.path.abspath(owner))
    path_absolute = Path(os.path.abspath(path))
    if path_absolute.parent != owner_absolute:
        raise RuntimeError(
            "response normalization receipt is outside its owning attempt"
        )
    try:
        owner_relative = owner_absolute.relative_to(checkpoint_absolute)
        path_absolute.relative_to(checkpoint_absolute)
    except ValueError as exc:
        raise RuntimeError(
            "response normalization receipt is outside the checkpoint root"
        ) from exc

    current = checkpoint_absolute
    for part in (*owner_relative.parts, path_absolute.name):
        current = current / part
        if current.is_symlink():
            raise RuntimeError(
                "response normalization receipt path contains a symlink"
            )

    checkpoint_resolved = checkpoint_absolute.resolve()
    owner_resolved = owner_absolute.resolve()
    path_resolved = path_absolute.resolve()
    try:
        owner_resolved.relative_to(checkpoint_resolved)
        path_resolved.relative_to(checkpoint_resolved)
    except ValueError as exc:
        raise RuntimeError(
            "response normalization receipt is outside the checkpoint root"
        ) from exc
    if path_resolved.parent != owner_resolved:
        raise RuntimeError(
            "response normalization receipt is outside its owning attempt"
        )


def _response_normalization_reference(
    receipt_path: Path,
    *,
    checkpoint_dir: Path,
) -> dict[str, str]:
    try:
        rendered_path = receipt_path.resolve().relative_to(
            checkpoint_dir.resolve()
        ).as_posix()
    except ValueError as exc:
        raise RuntimeError(
            "response normalization receipt is outside the checkpoint root"
        ) from exc
    return {
        "path": rendered_path,
        "sha256": sha256_file(receipt_path),
    }


def _validate_complete_response_schema(
    value: Mapping[str, Any], schema: Mapping[str, Any],
) -> None:
    from jsonschema.validators import validator_for

    validator = validator_for(schema)
    validator.check_schema(schema)
    validator(schema).validate(value)


def _normalize_translation_repair_response(
    raw_response: dict[str, Any],
    *,
    expected_ids: Sequence[str],
    schema: Mapping[str, Any],
    schema_version: str,
    invariant_validator: Callable[[Mapping[str, Any]], Any],
    receipt_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return normalize_complete_response_with_receipt(
        raw_response,
        "repairs",
        expected_ids,
        lambda item: item.get("block_id"),
        lambda candidate: _validate_complete_response_schema(candidate, schema),
        invariant_validator,
        (
            f"{TRANSLATION_REPAIR_NORMALIZATION_VALIDATOR_VERSION}:"
            f"{schema_version}"
        ),
        receipt_path=receipt_path,
    )


def _legacy_translation_token_attempt_path(
    checkpoint_dir: Path, segment_id: str, generation: int = 1,
) -> Path:
    return (
        _generation_segment_artifact_dir(
            checkpoint_dir, "translation-token-attempts", segment_id, generation,
        )
        / f"{_segment_checkpoint_name(segment_id)}.json"
    )


def _read_checkpoint_json(path: Path, *, root: Path) -> Any | None:
    """Read one root-bound checkpoint without following mutable addresses."""

    try:
        return _read_recovery_json(root, path)
    except (SecureReadError, OSError, ValueError, TypeError):
        return None


def _matching_translation_token_attempt(
    checkpoint_dir: Path, segment_id: str, input_sha256: str,
    generation: int = 1,
) -> dict[str, Any] | None:
    path = _translation_token_attempt_path(checkpoint_dir, segment_id, generation)
    value = _read_checkpoint_json(path, root=checkpoint_dir)
    if (
        isinstance(value, dict)
        and value.get("schema_version") == "arc.companion.translation-token-attempt.v2"
        and value.get("prompt_version") == TRANSLATION_RETRY_PROMPT_VERSION
        and value.get("segment_id") == segment_id
        and value.get("input_sha256") == input_sha256
        and _artifact_payload_generation(
            value, checkpoint_dir, "translation-token-offset-attempts", segment_id,
        ) == generation
    ):
        return value
    return None


def _matching_superseded_translation_text_attempt(
    checkpoint_dir: Path, segment_id: str, input_sha256: str,
    generation: int = 1,
) -> dict[str, Any] | None:
    """Return a same-input legacy text repair for audit and low-rerun guards."""
    value = _read_checkpoint_json(
        _legacy_translation_token_attempt_path(checkpoint_dir, segment_id, generation),
        root=checkpoint_dir,
    )
    if (
        isinstance(value, dict)
        and value.get("segment_id") == segment_id
        and value.get("input_sha256") == input_sha256
        and _artifact_payload_generation(
            value, checkpoint_dir, "translation-token-attempts", segment_id,
        ) == generation
        and str(value.get("prompt_version") or "") in {
            "arc.companion.translation-retry-prompt.v3",
            "arc.companion.translation-retry-prompt.v4",
        }
    ):
        return value
    return None


def _guard_translation_token_attempt_before_primary(
    checkpoint_dir: Path, segment_id: str, input_sha256: str,
    generation: int = 1,
) -> dict[str, Any] | None:
    """Fail closed on an unreadable or malformed current marker before low work."""
    path = _translation_token_attempt_path(checkpoint_dir, segment_id, generation)
    if not path.is_file():
        return _matching_superseded_translation_text_attempt(
            checkpoint_dir, segment_id, input_sha256,
            generation,
        )
    value = _read_checkpoint_json(path, root=checkpoint_dir)
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
        and _artifact_payload_generation(
            value, checkpoint_dir, "translation-token-offset-attempts", segment_id,
        ) == generation
    ):
        raise RuntimeError(
            f"translation token repair marker has malformed current identity for {segment_id}; "
            "refusing a primary model call"
        )
    return value


def _translation_token_repair_draft_path(
    checkpoint_dir: Path, segment_id: str, generation: int = 1,
) -> Path:
    return (
        _generation_segment_artifact_dir(
            checkpoint_dir, "translation-token-offset-repair-drafts", segment_id, generation,
        )
        / f"{_segment_checkpoint_name(segment_id)}.json"
    )


def _matching_translation_token_repair_draft(
    checkpoint_dir: Path, segment_id: str, input_sha256: str,
    generation: int = 1,
) -> dict[str, Any] | None:
    path = _translation_token_repair_draft_path(
        checkpoint_dir, segment_id, generation,
    )
    value = _read_checkpoint_json(path, root=checkpoint_dir)
    if (
        isinstance(value, dict)
        and value.get("schema_version") == "arc.companion.translation-token-repair-draft.v1"
        and value.get("prompt_version") == TRANSLATION_RETRY_PROMPT_VERSION
        and value.get("response_schema_version")
        == TRANSLATION_SLOT_REPAIR_SCHEMA_VERSION
        and value.get("segment_id") == segment_id
        and value.get("input_sha256") == input_sha256
        and _artifact_payload_generation(
            value, checkpoint_dir,
            "translation-token-offset-repair-drafts", segment_id,
        ) == generation
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
    response_normalization: Mapping[str, str] | None = None,
    generation: int = 1,
    prior_marker: dict[str, Any] | None = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    write_json(_translation_token_attempt_path(
        checkpoint_dir, segment_id, generation,
    ), {
        "schema_version": "arc.companion.translation-token-attempt.v2",
        "segment_id": segment_id,
        "generation": generation,
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
            {"response_normalization": dict(response_normalization)}
            if response_normalization is not None else {}
        ),
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
    generation: int = 1,
) -> dict[str, Any]:
    model_generated = origin == "primary-model"
    return {
        "schema_version": "arc.companion.translation-primary-draft.v1",
        "segment_id": str(segment["segment_id"]),
        "generation": generation,
        "input_sha256": input_sha256,
        "candidate_provenance": {
            "origin": origin,
            "prompt_version": TRANSLATION_PROMPT_VERSION if model_generated else None,
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
    generation: int = 1,
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
        glossary=project_segment_glossary(
            _augmentation_blocks(segment, by_id), glossary,
        ),
        extra={
            "names": protected_names,
            "paper_context": paper_context,
            "runtime_access": _generation_runtime_policy(options),
        },
    )
    path = _translation_draft_path(
        checkpoint_dir, str(segment["segment_id"]), generation,
    )
    write_json(
        path,
        _translation_primary_draft_payload(
            segment,
            translation or {"blocks": []},
            input_sha256=input_sha256,
            origin="controller-seed",
            generation=generation,
        ),
    )
    return path


def _repair_call_crossed_submission_barrier(artifact_dir: Path) -> bool:
    """Fail closed unless durable call state proves no provider submission."""
    checkpoint_dir = artifact_dir / "call-checkpoints"
    paths = sorted(checkpoint_dir.glob("*.json")) if checkpoint_dir.is_dir() else []
    if not paths:
        return False
    for path in paths:
        checkpoint = _read_checkpoint_json(path, root=artifact_dir)
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("submission_state") != "not_submitted"
        ):
            return True
    return False


def _completed_repair_checkpoint_response(
    artifact_dir: Path,
) -> dict[str, Any] | None:
    """Return one exact promoted repair response without invoking a provider."""

    checkpoint_root = artifact_dir / "call-checkpoints"
    paths = sorted(checkpoint_root.glob("*.json")) if checkpoint_root.is_dir() else []
    if len(paths) > 1:
        raise RuntimeError("repair artifact contains ambiguous call checkpoints")
    if not paths:
        return None
    checkpoint = _read_checkpoint_json(paths[0], root=artifact_dir)
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("repair call checkpoint is unreadable")
    if checkpoint.get("state") not in {"response_received", "validated"}:
        return None
    response = checkpoint.get("response")
    value = response.get("value") if isinstance(response, Mapping) else None
    if not isinstance(value, Mapping):
        raise RuntimeError("completed repair checkpoint has no object response")
    return dict(value)


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
    generation: int = 1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the single lifetime-bounded token-placement repair for a segment input."""
    segment_id = str(segment["segment_id"])
    token_errors = _translation_opaque_token_errors(segment, translation, blocks_by_id)
    source_blocks = [blocks_by_id[item.block_id] for item in token_errors]
    if not source_blocks:
        raise RuntimeError(
            f"translation token repair has no structurally failing blocks for {segment_id}"
        )
    expected_repair_ids = [block_id(block) for block in source_blocks]
    repair_draft_path = _translation_token_repair_draft_path(
        checkpoint_dir, segment_id, generation,
    )
    attempt_path = _translation_token_attempt_path(
        checkpoint_dir, segment_id, generation,
    )
    persisted = _matching_translation_token_repair_draft(
        checkpoint_dir, segment_id, input_sha256, generation,
    )
    invalid_persisted_draft = False
    if persisted is not None:
        persisted_marker = _matching_translation_token_attempt(
            checkpoint_dir, segment_id, input_sha256, generation,
        )
        persisted_provenance_ids = list(
            persisted["repair_provenance"].get("repaired_block_ids") or []
        )
        persisted_marker_ids = (
            list(persisted_marker.get("block_ids") or [])
            if isinstance(persisted_marker, dict) else []
        )
        if (
            persisted_provenance_ids != expected_repair_ids
            or (
                persisted_marker is not None
                and persisted_marker_ids != expected_repair_ids
            )
        ):
            raise TranslationRepairNeedsSupervision(
                segment_id=segment_id,
                marker_path=attempt_path,
                reason=(
                    "persisted token repair expected block IDs differ from its "
                    "owning marker or draft provenance"
                ),
            )
        try:
            receipt_path = _repair_response_normalization_path(
                artifact_dir / "retry-offset-1",
                attempt_path,
                checkpoint_dir=checkpoint_dir,
                persisted_response=True,
            )

            def validate_persisted_response(candidate: Mapping[str, Any]) -> None:
                repaired_candidate = _apply_translation_slot_repairs(
                    translation,
                    source_blocks,
                    dict(candidate),
                    protected_names=protected_names,
                    offset_only=True,
                )
                _validate_translation(
                    segment, repaired_candidate, blocks_by_id, protected_names,
                )

            projected_response, _ = _normalize_translation_repair_response(
                persisted["raw_response"],
                expected_ids=expected_repair_ids,
                schema=TRANSLATION_SLOT_REPAIR_SCHEMA,
                schema_version=TRANSLATION_SLOT_REPAIR_SCHEMA_VERSION,
                invariant_validator=validate_persisted_response,
                receipt_path=receipt_path,
            )
            repaired = _apply_translation_slot_repairs(
                translation,
                source_blocks,
                projected_response,
                protected_names=protected_names,
                offset_only=True,
            )
            _validate_translation(
                segment, repaired, blocks_by_id, protected_names,
            )
            if sha256_json(repaired) != sha256_json(persisted["translation"]):
                raise RuntimeError("persisted translation differs from replayed offsets")
        except RuntimeError:
            invalid_persisted_draft = True
        else:
            normalization_reference = _response_normalization_reference(
                receipt_path, checkpoint_dir=checkpoint_dir,
            )
            persisted_provenance = {
                **dict(persisted["repair_provenance"]),
                "response_normalization": normalization_reference,
            }
            if (
                persisted.get("response_normalization") != normalization_reference
                or persisted.get("repair_provenance") != persisted_provenance
            ):
                write_json(repair_draft_path, {
                    **persisted,
                    "response_normalization": normalization_reference,
                    "repair_provenance": persisted_provenance,
                })
            prior_marker = _read_checkpoint_json(
                _translation_token_attempt_path(checkpoint_dir, segment_id, generation),
                root=checkpoint_dir,
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
                response_normalization=normalization_reference,
                generation=generation,
                prior_marker=prior_marker if isinstance(prior_marker, dict) else None,
            )
            return repaired, persisted_provenance
    raw_attempt = _read_checkpoint_json(attempt_path, root=checkpoint_dir)
    if attempt_path.is_file() and raw_attempt is None:
        raise RuntimeError(
            f"translation token repair marker is corrupt for {segment_id}; "
            "refusing a new model call"
        )
    attempt = _matching_translation_token_attempt(
        checkpoint_dir, segment_id, input_sha256, generation,
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
    raw_repair_draft = _read_checkpoint_json(
        repair_draft_path, root=checkpoint_dir,
    )
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
        "generation": generation,
        "input_sha256": input_sha256,
        "prompt_version": TRANSLATION_RETRY_PROMPT_VERSION,
        "response_schema_version": TRANSLATION_SLOT_REPAIR_SCHEMA_VERSION,
        "model_tier": TRANSLATION_RETRY_TIER,
        "block_ids": [block_id(block) for block in source_blocks],
    }
    superseded_text_attempt = _matching_superseded_translation_text_attempt(
        checkpoint_dir, segment_id, input_sha256, generation,
    )
    if isinstance(superseded_text_attempt, dict):
        marker_base["superseded_text_attempt"] = {
            "path": str(_legacy_translation_token_attempt_path(
                checkpoint_dir, segment_id, generation,
            )),
            "sha256": sha256_json(superseded_text_attempt),
            "prompt_version": str(superseded_text_attempt.get("prompt_version") or ""),
            "status": str(superseded_text_attempt.get("status") or ""),
        }
    value: dict[str, Any] | None = None
    response_was_persisted = False
    if attempt is not None:
        status = str(attempt.get("status") or "")
        if status in {"response_received", "validated"}:
            if list(attempt.get("block_ids") or []) != expected_repair_ids:
                raise TranslationRepairNeedsSupervision(
                    segment_id=segment_id,
                    marker_path=attempt_path,
                    reason=(
                        "persisted token repair expected block IDs differ from its "
                        "owning marker"
                    ),
                )
            raw_response = attempt.get("raw_response")
            if not isinstance(raw_response, dict):
                raise RuntimeError(
                    f"translation token repair marker lacks its auditable response for {segment_id}"
                )
            value = raw_response
            response_was_persisted = True
        elif status == "started":
            value = _completed_repair_checkpoint_response(
                artifact_dir / "retry-offset-1"
            )
            if value is not None:
                response_was_persisted = True
            elif _repair_call_crossed_submission_barrier(
                artifact_dir / "retry-offset-1"
            ):
                raise RuntimeError(
                    f"translation token repair attempt already started for {segment_id}; "
                    "refusing another model call"
                )
            # The provider call checkpoint is durable before submission. Its
            # absence (or explicit not-submitted state) proves that local
            # preflight did not consume the bounded repair turn.
            else:
                attempt = None
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
                _semantic_segment_descriptor(segment),
                repair_contexts,
                validation_errors=[item.prompt_payload() for item in token_errors],
                retry_model_tier=TRANSLATION_RETRY_TIER,
            ),
            TRANSLATION_SLOT_REPAIR_SCHEMA,
            options=options,
            artifact_dir=artifact_dir / "retry-offset-1",
            call_label=f"companion-translation-{segment_id}-retry-offset-1",
            model_tier=TRANSLATION_RETRY_TIER,
            force_offline=True,
            recovery_descriptor=submission_descriptor(
                unit="translation-token-repair",
                logical_unit=f"{segment_id}:token-repair",
                checkpoint_dir=checkpoint_dir,
                artifact_root=artifact_dir / "retry-offset-1",
                acceptance_checkpoint=(
                    _generation_segment_artifact_dir(
                        checkpoint_dir, "translations", segment_id, generation,
                    ) / f"{_segment_checkpoint_name(segment_id)}.json"
                ),
                input_sha256=input_sha256,
                ordered_siblings=[f"{segment_id}:token-repair"],
                suffix=[f"{segment_id}:token-repair"],
            ),
        )
        write_json(attempt_path, {
            **marker_base,
            "status": "response_received",
            "started_at": started_at,
            "response_received_at": datetime.now(timezone.utc).isoformat(),
            "raw_response": value,
        })
    try:
        receipt_path = _repair_response_normalization_path(
            artifact_dir / "retry-offset-1",
            attempt_path,
            checkpoint_dir=checkpoint_dir,
            persisted_response=response_was_persisted,
        )

        def validate_response(candidate: Mapping[str, Any]) -> None:
            repaired_candidate = _apply_translation_slot_repairs(
                repair_input,
                source_blocks,
                dict(candidate),
                protected_names=protected_names,
                allow_clause_rewrite=False,
                primary_translation=None,
                offset_only=True,
            )
            _validate_translation(
                segment, repaired_candidate, blocks_by_id, protected_names,
            )

        try:
            projected_response, _ = _normalize_translation_repair_response(
                value,
                expected_ids=expected_repair_ids,
                schema=TRANSLATION_SLOT_REPAIR_SCHEMA,
                schema_version=TRANSLATION_SLOT_REPAIR_SCHEMA_VERSION,
                invariant_validator=validate_response,
                receipt_path=receipt_path,
            )
        except ResponseNormalizationError as exc:
            raise RuntimeError(f"translation slot repair {exc}") from exc
        repaired = _apply_translation_slot_repairs(
            repair_input,
            source_blocks,
            projected_response,
            protected_names=protected_names,
            allow_clause_rewrite=False,
            primary_translation=None,
            offset_only=True,
        )
        _validate_translation(
            segment, repaired, blocks_by_id, protected_names,
        )
    except RuntimeError as exc:
        # The raw response is durably stored before this local application
        # step.  It consumed the bounded provider turn, so invalid structure
        # is a single-call supervision case, never permission to resend.
        raise TranslationRepairNeedsSupervision(
            segment_id=segment_id, marker_path=attempt_path, reason=str(exc),
        ) from exc
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
        "response_normalization": _response_normalization_reference(
            receipt_path, checkpoint_dir=checkpoint_dir,
        ),
    }
    if isinstance(marker_base.get("superseded_text_attempt"), dict):
        provenance["superseded_text_attempt"] = marker_base["superseded_text_attempt"]
    write_json(repair_draft_path, {
        "schema_version": "arc.companion.translation-token-repair-draft.v1",
        "segment_id": segment_id,
        "generation": generation,
        "input_sha256": input_sha256,
        "prompt_version": TRANSLATION_RETRY_PROMPT_VERSION,
        "response_schema_version": TRANSLATION_SLOT_REPAIR_SCHEMA_VERSION,
        "raw_response": value,
        "response_normalization": _response_normalization_reference(
            receipt_path, checkpoint_dir=checkpoint_dir,
        ),
        "translation": repaired,
        "repair_provenance": provenance,
    })
    marker = _matching_translation_token_attempt(
        checkpoint_dir, segment_id, input_sha256, generation,
    ) or marker_base
    _write_validated_translation_token_marker(
        checkpoint_dir,
        segment_id,
        input_sha256,
        repaired=repaired,
        raw_response=value,
        repaired_block_ids=[block_id(block) for block in source_blocks],
        response_normalization=_response_normalization_reference(
            receipt_path, checkpoint_dir=checkpoint_dir,
        ),
        generation=generation,
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
    generation: int = 1,
    accepted_callback: Callable[[str, str, dict[str, Any]], None] | None = None,
    force_generation: bool = False,
    intent_guidance: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    by_id = {block_id(block): block for block in bundle.document["blocks"]}
    output: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {}
    v4_upgrade_ids: set[str] = set()
    protected_name_upgrade_ids: set[str] = set()
    for segment in segments:
        translation_dir = _generation_segment_artifact_dir(
            checkpoint_dir, "translations", str(segment["segment_id"]), generation,
        )
        translation_dir.mkdir(parents=True, exist_ok=True)
        path = translation_dir / f"{_segment_checkpoint_name(segment['segment_id'])}.json"
        paper_context = _full_paper_context(
            bundle.document, segment, blocks_by_id=by_id, options=options
        )
        segment_glossary = project_segment_glossary(
            _augmentation_blocks(segment, by_id), glossary,
        )
        expected_hash = _segment_input_hash(
            segment,
            by_id,
            glossary=segment_glossary,
            extra={
                "names": protected_names,
                "paper_context": paper_context,
                "runtime_access": _generation_runtime_policy(options),
                **(
                    {"intent_guidance": _intent_guidance_identity(intent_guidance)}
                    if intent_guidance is not None else {}
                ),
            },
        )
        input_hashes[str(segment["segment_id"])] = expected_hash
        if path.is_file() and not force_generation:
            checkpoint = _read_checkpoint_json(path, root=checkpoint_dir)
            if (
                isinstance(checkpoint, dict)
                and checkpoint.get("segment_id") == segment["segment_id"]
                and _artifact_payload_generation(
                    checkpoint, checkpoint_dir, "translations",
                    str(segment["segment_id"]),
                ) == generation
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
                if accepted_callback is not None:
                    accepted_callback(
                        "translation", str(segment["segment_id"]), checkpoint["translation"]
                    )
                continue
        pending.append(segment)

    def generate(
        segment: dict[str, Any],
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        segment_id = str(segment["segment_id"])
        selected = _augmentation_blocks(segment, by_id)
        segment_glossary = project_segment_glossary(selected, glossary)
        translatable = [_translation_input_block(block) for block in selected if _is_translatable(block)]
        paper_context = _full_paper_context(
            bundle.document, segment, blocks_by_id=by_id, options=options
        )
        artifact_dir = (
            _generation_segment_artifact_dir(
                checkpoint_dir, "llm/translations", segment_id, generation,
            ) / _segment_checkpoint_name(segment_id)
        )
        draft_path = _translation_draft_path(checkpoint_dir, segment_id, generation)
        draft = _read_checkpoint_json(draft_path, root=checkpoint_dir)
        attempt = _matching_translation_coverage_attempt(
            checkpoint_dir, segment_id, input_hashes[segment_id], generation
        )
        legacy_coverage_attempt = _matching_legacy_translation_coverage_attempt(
            checkpoint_dir, segment_id, input_hashes[segment_id], generation
        )
        raw_coverage_attempt = _read_checkpoint_json(
            _translation_coverage_attempt_path(checkpoint_dir, segment_id, generation),
            root=checkpoint_dir,
        )
        if (
            _translation_coverage_attempt_path(
                checkpoint_dir, segment_id, generation,
            ).is_file()
            and raw_coverage_attempt is None
        ):
            raise RuntimeError(
                f"translation coverage repair marker is unreadable for {segment_id}"
            )
        if (
            raw_coverage_attempt is not None
            and attempt is None
            and legacy_coverage_attempt is None
            and raw_coverage_attempt.get("input_sha256") == input_hashes[segment_id]
        ):
            raise RuntimeError(
                f"translation coverage repair marker has unknown identity for {segment_id}"
            )
        candidate_provenance: dict[str, Any] | None = None
        translation: dict[str, Any] | None = None
        restored_name_ids: list[str] = []
        canonicalized_token_ids: list[str] = []
        if (
            isinstance(draft, dict)
            and draft.get("schema_version") == "arc.companion.translation-primary-draft.v1"
            and draft.get("segment_id") == segment_id
            and _artifact_payload_generation(
                draft, checkpoint_dir, "translation-drafts", segment_id,
            ) == generation
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
                draft_translation, canonicalized_token_ids = (
                    _canonicalize_translation_opaque_candidates(
                        draft_translation, by_id
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
                if not force_generation:
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
                "refusing to rerun the medium translation model"
            )
        if segment_id in protected_name_upgrade_ids and (
            translation is None
            or str((candidate_provenance or {}).get("origin") or "") != "primary-model"
        ):
            raise RuntimeError(
                f"protected-name translation upgrade for {segment_id} requires its stored "
                "primary draft; refusing to rerun the medium translation model"
            )
        if translatable:
            if translation is None:
                guarded_attempt = _guard_translation_token_attempt_before_primary(
                    checkpoint_dir, segment_id, input_hashes[segment_id], generation
                )
                if guarded_attempt is not None:
                    raise RuntimeError(
                        f"translation token placement repair attempt already consumed for {segment_id}"
                    )
                translation = _llm_call(
                    llm,
                    _guided_prompt(translation_prompt(
                        _semantic_segment_descriptor(segment),
                        translatable,
                        language=options.annotation_language,
                        glossary=segment_glossary,
                        protected_names=protected_names,
                        paper_context=paper_context,
                        source_language=_multilingual_prompt_source(options),
                    ), intent_guidance, lane="translation"),
                    TRANSLATION_SCHEMA,
                    options=options,
                    artifact_dir=artifact_dir,
                    call_label=f"companion-translation-{segment_id}",
                    model_tier=TRANSLATION_TIER,
                    force_offline=True,
                    paper_access_policy=_guidance_policy(intent_guidance, lane="translation"),
                    intent_guidance=intent_guidance,
                    intent_guidance_lane="translation",
                    recovery_descriptor=submission_descriptor(
                        unit="translation",
                        logical_unit=segment_id,
                        checkpoint_dir=checkpoint_dir,
                        artifact_root=artifact_dir,
                        acceptance_checkpoint=(
                            _generation_segment_artifact_dir(
                                checkpoint_dir, "translations", segment_id, generation,
                            ) / f"{_segment_checkpoint_name(segment_id)}.json"
                        ),
                        input_sha256=input_hashes[segment_id],
                        ordered_siblings=[
                            str(item["segment_id"]) for item in pending
                        ],
                        suffix=[
                            str(item["segment_id"])
                            for item in pending[
                                [str(value["segment_id"]) for value in pending]
                                .index(segment_id):
                            ]
                        ],
                    ),
                )
                draft = _translation_primary_draft_payload(
                    segment,
                    translation,
                    input_sha256=input_hashes[segment_id],
                    origin="primary-model",
                    generation=generation,
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
        translation, newly_canonicalized_ids = (
            _canonicalize_translation_opaque_candidates(translation, by_id)
        )
        canonicalized_token_ids = list(dict.fromkeys([
            *canonicalized_token_ids, *newly_canonicalized_ids,
        ]))
        repair_provenance: list[dict[str, Any]] = []
        if restored_name_ids:
            repair_provenance.append({
                "kind": "protected-name-normalization",
                "attempt": 0,
                "normalizer_version": TRANSLATION_PROTECTED_NAME_NORMALIZER_VERSION,
                "repaired_block_ids": restored_name_ids,
            })
        if canonicalized_token_ids:
            repair_provenance.append({
                "kind": "opaque-token-canonicalization",
                "attempt": 0,
                "normalizer_version": TRANSLATION_OPAQUE_TOKEN_CANONICALIZER_VERSION,
                "repaired_block_ids": canonicalized_token_ids,
            })
        coverage_response: dict[str, Any] | None = None
        coverage_marker_base: dict[str, Any] | None = None
        coverage_response_was_persisted = False
        coverage_normalization_reference: dict[str, str] | None = None
        try:
            _validate_translation(segment, translation, by_id, protected_names)
        except TranslationCoverageError:
            normalized, missing_blocks, diagnostics = _normalize_translation_coverage(
                segment, translation, by_id
            )
            if missing_blocks:
                repair_contexts = [
                    _translation_coverage_repair_context(block) for block in missing_blocks
                ]
                attempt_path = _translation_coverage_attempt_path(
                    checkpoint_dir, segment_id, generation,
                )
                missing_block_ids = [block_id(block) for block in missing_blocks]
                coverage_marker_base = {
                    "schema_version": TRANSLATION_COVERAGE_ATTEMPT_SCHEMA_VERSION,
                    "segment_id": segment_id,
                    "generation": generation,
                    "input_sha256": input_hashes[segment_id],
                    "prompt_version": TRANSLATION_COVERAGE_REPAIR_PROMPT_VERSION,
                    "response_schema_version": TRANSLATION_COVERAGE_REPAIR_SCHEMA_VERSION,
                    "model_tier": TRANSLATION_COVERAGE_REPAIR_TIER,
                    "missing_block_ids": missing_block_ids,
                }
                if isinstance(legacy_coverage_attempt, dict):
                    coverage_marker_base["superseded_attempt"] = {
                        "schema_version": str(
                            legacy_coverage_attempt.get("schema_version") or ""
                        ),
                        "status": str(legacy_coverage_attempt.get("status") or ""),
                        "started_at": str(
                            legacy_coverage_attempt.get("started_at") or ""
                        ),
                        "sha256": sha256_json(legacy_coverage_attempt),
                    }
                if attempt is not None:
                    status = str(attempt.get("status") or "")
                    if list(attempt.get("missing_block_ids") or []) != missing_block_ids:
                        if status in {"response_received", "validated"}:
                            raise TranslationRepairNeedsSupervision(
                                segment_id=segment_id,
                                marker_path=attempt_path,
                                reason=(
                                    "persisted coverage repair expected block IDs "
                                    "differ from its owning marker"
                                ),
                            )
                        raise RuntimeError(
                            f"translation coverage repair marker changed missing blocks for {segment_id}"
                        )
                    if status in {"response_received", "validated"}:
                        raw_response = attempt.get("raw_response")
                        if not isinstance(raw_response, dict):
                            raise RuntimeError(
                                "translation coverage repair marker lacks its auditable "
                                f"response for {segment_id}"
                            )
                        coverage_response = raw_response
                        coverage_response_was_persisted = True
                    elif status == "started":
                        coverage_response = _completed_repair_checkpoint_response(
                            artifact_dir / "coverage-repair-1"
                        )
                        if coverage_response is not None:
                            coverage_response_was_persisted = True
                        elif _repair_call_crossed_submission_barrier(
                            artifact_dir / "coverage-repair-1"
                        ):
                            raise RuntimeError(
                                f"translation coverage repair attempt already started for {segment_id}; "
                                "refusing another model call"
                            )
                        # A local crash before a durable provider submission is
                        # safe to heal without consuming the bounded repair.
                        else:
                            attempt = None
                    else:
                        raise RuntimeError(
                            "translation coverage repair marker has invalid status "
                            f"{status!r} for {segment_id}"
                        )
                if coverage_response is None:
                    started_at = datetime.now(timezone.utc).isoformat()
                    write_json(attempt_path, {
                        **coverage_marker_base,
                        "status": "started",
                        "started_at": started_at,
                    })
                    try:
                        coverage_response = _llm_call(
                            llm,
                            translation_coverage_repair_prompt(
                                _semantic_segment_descriptor(segment),
                                repair_contexts,
                                language=options.annotation_language,
                                glossary=segment_glossary,
                                protected_names=protected_names,
                                paper_context={
                                    **paper_context,
                                    "access": {"allow_mcp": False, "allow_internet": False},
                                },
                                repair_model_tier=TRANSLATION_COVERAGE_REPAIR_TIER,
                                source_language=_multilingual_prompt_source(options),
                            ),
                            TRANSLATION_COVERAGE_REPAIR_SCHEMA,
                            options=options,
                            artifact_dir=artifact_dir / "coverage-repair-1",
                            call_label=f"companion-translation-{segment_id}-coverage-repair-1",
                            model_tier=TRANSLATION_COVERAGE_REPAIR_TIER,
                            force_offline=True,
                            recovery_descriptor=submission_descriptor(
                                unit="translation-coverage-repair",
                                logical_unit=f"{segment_id}:coverage-repair",
                                checkpoint_dir=checkpoint_dir,
                                artifact_root=artifact_dir / "coverage-repair-1",
                                acceptance_checkpoint=(
                                    _generation_segment_artifact_dir(
                                        checkpoint_dir, "translations", segment_id,
                                        generation,
                                    ) / f"{_segment_checkpoint_name(segment_id)}.json"
                                ),
                                input_sha256=input_hashes[segment_id],
                                ordered_siblings=[
                                    f"{segment_id}:coverage-repair"
                                ],
                                suffix=[f"{segment_id}:coverage-repair"],
                            ),
                        )
                    except CompanionLLMCircuitOpen:
                        # The shared limiter rejected this queued call before it
                        # entered any provider.  Do not consume its one lifetime
                        # repair attempt merely because another worker opened
                        # the build-wide circuit.
                        if isinstance(legacy_coverage_attempt, dict):
                            write_json(attempt_path, legacy_coverage_attempt)
                        else:
                            attempt_path.unlink(missing_ok=True)
                        raise
                    write_json(attempt_path, {
                        **coverage_marker_base,
                        "status": "response_received",
                        "started_at": started_at,
                        "response_received_at": datetime.now(timezone.utc).isoformat(),
                        "raw_response": coverage_response,
                    })
                try:
                    coverage_receipt_path = _repair_response_normalization_path(
                        artifact_dir / "coverage-repair-1",
                        attempt_path,
                        checkpoint_dir=checkpoint_dir,
                        persisted_response=coverage_response_was_persisted,
                    )

                    def validate_coverage_response(candidate: Mapping[str, Any]) -> None:
                        validation_translation = _apply_translation_coverage_repairs(
                            normalized,
                            segment,
                            missing_blocks,
                            dict(candidate),
                            by_id,
                        )
                        validation_translation, _ = _restore_translation_protected_names(
                            segment,
                            validation_translation,
                            by_id,
                            protected_names,
                        )
                        _validate_translation(
                            segment, validation_translation, by_id, protected_names,
                        )

                    projected_coverage_response, _ = (
                        _normalize_translation_repair_response(
                            coverage_response,
                            expected_ids=missing_block_ids,
                            schema=TRANSLATION_COVERAGE_REPAIR_SCHEMA,
                            schema_version=TRANSLATION_COVERAGE_REPAIR_SCHEMA_VERSION,
                            invariant_validator=validate_coverage_response,
                            receipt_path=coverage_receipt_path,
                        )
                    )
                    coverage_normalization_reference = (
                        _response_normalization_reference(
                            coverage_receipt_path, checkpoint_dir=checkpoint_dir,
                        )
                    )
                    translation = _apply_translation_coverage_repairs(
                        normalized,
                        segment,
                        missing_blocks,
                        projected_coverage_response,
                        by_id,
                    )
                except RuntimeError as exc:
                    raise TranslationRepairNeedsSupervision(
                        segment_id=segment_id,
                        marker_path=attempt_path,
                        reason=str(exc),
                    ) from exc
                repair_provenance.append({
                    "kind": "coverage",
                    "attempt": 1,
                    "prompt_version": TRANSLATION_COVERAGE_REPAIR_PROMPT_VERSION,
                    "response_schema_version": TRANSLATION_COVERAGE_REPAIR_SCHEMA_VERSION,
                    "model_tier": TRANSLATION_COVERAGE_REPAIR_TIER,
                    "repaired_block_ids": [block_id(block) for block in missing_blocks],
                    "normalization": diagnostics,
                    "response_normalization": coverage_normalization_reference,
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
                        generation=generation,
                    )
                    repair_provenance.append(provenance)
            if missing_blocks:
                # Model-backed coverage repair is self-contained; never chain another model repair.
                try:
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
                except RuntimeError as exc:
                    assert coverage_response is not None
                    raise TranslationRepairNeedsSupervision(
                        segment_id=segment_id,
                        marker_path=attempt_path,
                        reason=str(exc),
                    ) from exc
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
                generation=generation,
            )
            repair_provenance.append(provenance)
        try:
            translation, normalized_citation_ids = (
                _normalize_translation_citation_delimiters_for_segment(translation, by_id)
            )
            if normalized_citation_ids:
                repair_provenance.append(
                    _citation_delimiter_normalization_provenance(normalized_citation_ids)
                )
            _validate_translation(segment, translation, by_id, protected_names)
        except RuntimeError as exc:
            if coverage_response is None:
                raise
            raise TranslationRepairNeedsSupervision(
                segment_id=segment_id,
                marker_path=_translation_coverage_attempt_path(
                    checkpoint_dir, segment_id, generation,
                ),
                reason=str(exc),
            ) from exc
        if coverage_response is not None and coverage_marker_base is not None:
            prior_marker = _matching_translation_coverage_attempt(
                checkpoint_dir, segment_id, input_hashes[segment_id], generation
            )
            now = datetime.now(timezone.utc).isoformat()
            write_json(_translation_coverage_attempt_path(
                checkpoint_dir, segment_id, generation,
            ), {
                **coverage_marker_base,
                "status": "validated",
                "started_at": str((prior_marker or {}).get("started_at") or now),
                "response_received_at": str(
                    (prior_marker or {}).get("response_received_at") or now
                ),
                "validated_at": now,
                "validated_translation_sha256": sha256_json(translation),
                "raw_response": coverage_response,
                **(
                    {"response_normalization": coverage_normalization_reference}
                    if coverage_normalization_reference is not None else {}
                ),
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
                translation_checkpoint_path = (
                    _generation_segment_artifact_dir(
                        checkpoint_dir, "translations", segment_id, generation,
                    ) / f"{_segment_checkpoint_name(segment_id)}.json"
                )
                write_json(
                    translation_checkpoint_path,
                    {
                        "schema_version": "arc.companion.translation-checkpoint.v2",
                        "segment_id": segment_id,
                        "generation": generation,
                        "input_sha256": _segment_input_hash(
                            segment,
                            by_id,
                            glossary=project_segment_glossary(
                                _augmentation_blocks(segment, by_id), glossary,
                            ),
                            extra={
                                "names": protected_names,
                                "paper_context": _full_paper_context(
                                    bundle.document, segment, blocks_by_id=by_id, options=options
                                ),
                                "runtime_access": _generation_runtime_policy(options),
                                **(
                                    {"intent_guidance": _intent_guidance_identity(intent_guidance)}
                                    if intent_guidance is not None else {}
                                ),
                            },
                        ),
                        "generation_provenance": provenance,
                        "translation": value,
                    },
                )
                _bind_translation_repair_acceptance(
                    checkpoint_dir,
                    segment_id=str(segment_id),
                    generation=generation,
                    input_sha256=input_hashes[str(segment_id)],
                    translation_checkpoint=translation_checkpoint_path,
                    provenance=provenance,
                )
                if _pipeline_control_receipt_exists(
                    checkpoint_dir, "translation", str(segment_id),
                ):
                    _accept_registered_pipeline_control(
                        checkpoint_dir, "translation", str(segment_id),
                    )
                if accepted_callback is not None:
                    accepted_callback("translation", str(segment_id), value)
        if failures:
            raise CompanionLaneError("translation", failures)
    return {segment["segment_id"]: output[segment["segment_id"]] for segment in segments}


def _review(
    segments: list[dict[str, Any]],
    translations: dict[str, dict[str, Any]] | None,
    annotations: dict[str, dict[str, Any]],
    *,
    document: dict[str, Any],
    glossary: dict[str, Any],
    protected_names: list[str],
    evidence: dict[str, Any],
    options: BuildOptions,
    llm: Callable[..., dict[str, Any]],
    checkpoint_dir: Path,
    intent_guidance: Mapping[str, Any] | None = None,
) -> tuple[dict[str, dict[str, Any]] | None, dict[str, dict[str, Any]], dict[str, Any]]:
    active_segments = [item for item in segments if not item.get("structural_only")]
    if len(active_segments) != len(segments):
        active_ids = {str(item["segment_id"]) for item in active_segments}
        active_translations = (
            None if translations is None else {
                key: value for key, value in translations.items() if key in active_ids
            }
        )
        active_annotations = {
            key: value for key, value in annotations.items() if key in active_ids
        }
        reviewed_translations, reviewed_annotations, review = _review(
            active_segments,
            active_translations,
            active_annotations,
            document=document,
            glossary=glossary,
            protected_names=protected_names,
            evidence=evidence,
            options=options,
            llm=llm,
            checkpoint_dir=checkpoint_dir,
            intent_guidance=intent_guidance,
        )
        merged_annotations = dict(annotations)
        merged_annotations.update(reviewed_annotations)
        if translations is None:
            merged_translations = None
        else:
            merged_translations = dict(translations)
            merged_translations.update(reviewed_translations or {})
        return merged_translations, merged_annotations, review
    force_review = bool(
        {"translation", "commentary", "review"}.intersection(options.regenerate_lanes)
    )
    if not segments:
        return translations, annotations, {
            "hierarchical": False,
            "section_findings": [],
            "reviewed_segment_ids": [],
            "issues": [],
            "patched_segment_ids": [],
            "citation_delimiter_normalized_segment_ids": [],
            "logical_receipt": {
                "kind": "controller_skipped_structural_heading",
                "provider_calls": 0,
            },
        }
    if options.skip_translation:
        if translations not in (None, {}):
            raise RuntimeError("skip-translation review received translation content")
        reviewed, review = _review_commentary_only(
            segments,
            annotations,
            document=document,
            glossary={},
            evidence=evidence,
            options=options,
            llm=llm,
            checkpoint_dir=checkpoint_dir,
            intent_guidance=intent_guidance,
        )
        return None, reviewed, review
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
                "segment": _semantic_segment_descriptor(segment),
                "source_blocks": [
                    _annotation_input_block(by_id[value], document)
                    for value in _augmentation_block_ids(segment, by_id)
                ],
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
    direct_payload = {**payload, "glossary": glossary, "protected_names": protected_names}
    direct_prompt = _guided_prompt(review_prompt(
        direct_payload,
        language=options.annotation_language,
        findings=findings,
    ), intent_guidance, lane="review")
    # ``review_context_chars`` remains a user-controlled soft routing threshold.
    # Every actual prompt is measured in UTF-8 after all guidance and projections
    # are present.  Normal packing stops at 90% of the strict transport ceiling.
    prompt_budget = _review_prompt_budget(options)
    review_prompt_limit = int(prompt_budget["strict_limit_bytes"])
    review_prompt_target = int(prompt_budget["target_limit_bytes"])
    hierarchy_threshold = min(
        review_prompt_target,
        max(1, int(options.review_context_chars)),
    )
    hierarchical = _utf8_size(direct_prompt) > hierarchy_threshold
    review_call_audits: list[dict[str, Any]] = []
    if hierarchical:
        def render_section_prompt(chunk: list[dict[str, Any]]) -> str:
            chunk_text = json.dumps(
                [item.get("source_blocks") or [] for item in chunk], ensure_ascii=False
            )
            return _guided_prompt(section_review_prompt(
                {
                    "segments": chunk,
                    "glossary": _commentary_review_glossary_projection(
                        glossary, chunk, max_bytes=ANNOTATION_GLOSSARY_MAX_BYTES,
                    ),
                    "protected_names": [
                        name for name in protected_names if name and name in chunk_text
                    ],
                },
                language=options.annotation_language,
            ), intent_guidance, lane="review")

        section_calls = _pack_rendered_review_calls(
            payload["segments"],
            render_prompt=render_section_prompt,
            target_prompt_bytes=review_prompt_target,
            strict_prompt_bytes=review_prompt_limit,
            label="section review",
        )
        chunks = [list(call["items"]) for call in section_calls]
        section_logical_units = [
            _stable_recovery_chunk_id(
                "section-review",
                [str(value) for value in call.get("segment_ids") or []],
                payload={"prompt": call.get("prompt"), "items": call.get("items")},
            )
            for call in section_calls
        ]
        section_group_sha256 = sha256_json({
            "unit": "section-review",
            "segments": payload["segments"],
            "language": options.annotation_language,
        })
        # The final consolidation depends on section outputs, but its smallest
        # complete coverage projection does not. Validate that projection before
        # any parallel section call can incur cost.
        _bounded_hierarchical_review_prompt(
            [
                {
                    "section_index": index,
                    "reviewed_segment_ids": list(call["segment_ids"]),
                    "findings": [],
                    "patch_proposals": [],
                }
                for index, call in enumerate(section_calls)
            ],
            segments,
            blocks_by_id=by_id,
            document=document,
            segment_payloads=payload["segments"],
            glossary=glossary,
            protected_names=protected_names,
            language=options.annotation_language,
            max_prompt_bytes=review_prompt_target,
            strict_prompt_bytes=review_prompt_limit,
            intent_guidance=intent_guidance,
            essential_only=True,
        )
        recovered_reviews = (
            {} if force_review else _load_recovered_section_reviews(checkpoint_dir, chunks)
        )

        def inspect(
            index: int, rendered: dict[str, Any],
        ) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
            chunk = list(rendered["items"])
            prompt = str(rendered["prompt"])
            evidence_round_audits: list[dict[str, Any]] = []
            input_sha256 = sha256_json({
                "prompt": prompt,
                "schema": SECTION_REVIEW_SCHEMA,
                "model_tier": REVIEW_TIER,
            })
            logical_unit = section_logical_units[index]
            path = checkpoint_dir / "section-reviews" / f"{index:04d}.json"
            invalid_matching_checkpoint = False
            if path.is_file() and not force_review:
                checkpoint = read_json(path)
                try:
                    checkpoint_review = _normalize_sparse_review_patches(
                        checkpoint.get("review")
                        if isinstance(checkpoint, dict) else None,
                        block_order_by_segment=_review_patch_block_order(chunk),
                        scope="section review checkpoint",
                    )
                except RuntimeError:
                    # A conflicting or malformed cached patch is not trusted,
                    # but it also must not prevent a fresh provider review.
                    checkpoint_review = None
                checkpoint_matches = (
                    isinstance(checkpoint, dict)
                    and checkpoint.get("schema_version") == SECTION_REVIEW_CHECKPOINT_VERSION
                    and checkpoint.get("input_sha256") == input_sha256
                )
                checkpoint_validation_error = _section_review_validation_error(
                    checkpoint_review, chunk
                )
                if checkpoint_matches and checkpoint_validation_error is None:
                    if checkpoint_review != checkpoint.get("review"):
                        checkpoint = {**checkpoint, "review": checkpoint_review}
                        write_json(path, checkpoint)
                    _accept_completed_pipeline_controls(
                        checkpoint_dir,
                        caller_validated_units=frozenset({"section-review"}),
                        only_logical_unit=logical_unit,
                    )
                    return checkpoint_review, "checkpoint-reuse", []
                invalid_matching_checkpoint = bool(checkpoint_matches)
            value = (
                None
                if invalid_matching_checkpoint
                else recovered_reviews.get(index)
            )
            disposition = "recovered-reuse" if value is not None else "provider-call"
            if value is None:
                value = _llm_call(
                    llm,
                    prompt,
                    SECTION_REVIEW_SCHEMA,
                    options=options,
                    artifact_dir=checkpoint_dir / "llm" / "section-review" / logical_unit,
                    call_label=f"companion-section-review-{index}",
                    model_tier=REVIEW_TIER,
                    paper_access_policy=_guidance_policy(intent_guidance, lane="review"),
                    intent_guidance=intent_guidance,
                    intent_guidance_lane="review",
                    review_prompt_context=_review_prompt_context(
                        rendered,
                        stage="section",
                        audit_sink=evidence_round_audits,
                    ),
                    recovery_descriptor=submission_descriptor(
                        unit="section-review",
                        logical_unit=logical_unit,
                        checkpoint_dir=checkpoint_dir,
                        artifact_root=(
                            checkpoint_dir / "llm" / "section-review" / logical_unit
                        ),
                        acceptance_checkpoint=path,
                        input_sha256=input_sha256,
                        group_sha256=section_group_sha256,
                        ordered_siblings=section_logical_units,
                        suffix=section_logical_units[index:],
                    ),
                )
            if isinstance(value, dict) and "reviewed_segment_ids" not in value:
                legacy_reviewed = value.get("reviewed_segments")
                if isinstance(legacy_reviewed, list):
                    value = {
                        "reviewed_segment_ids": [
                            str(item.get("segment_id") or "")
                            for item in legacy_reviewed if isinstance(item, dict)
                        ],
                        "findings": list(value.get("findings") or []),
                        "patches": [],
                    }
            value = _normalize_sparse_review_patches(
                value,
                block_order_by_segment=_review_patch_block_order(chunk),
                scope="section review",
            )
            validation_error = _section_review_validation_error(value, chunk)
            if validation_error is not None:
                raise RuntimeError(f"section review {index} {validation_error}")
            write_json(path, {
                "schema_version": SECTION_REVIEW_CHECKPOINT_VERSION,
                "section_index": index,
                "input_sha256": input_sha256,
                "reviewed_segment_ids": sorted(
                    str(item["segment"]["segment_id"]) for item in chunk
                ),
                "prompt_budget_audit": _review_prompt_call_audit(
                    rendered,
                    stage="section",
                    call_label=f"companion-section-review-{index}",
                    disposition=disposition,
                ),
                "evidence_prompt_budget_audits": evidence_round_audits,
                "review": value,
            })
            _accept_registered_pipeline_control(
                checkpoint_dir, "section-review", logical_unit,
            )
            return value, disposition, evidence_round_audits

        with ThreadPoolExecutor(max_workers=min(options.workers, len(section_calls))) as executor:
            futures = {
                executor.submit(inspect, index, rendered): index
                for index, rendered in enumerate(section_calls)
            }
            ordered: dict[int, tuple[dict[str, Any], str, list[dict[str, Any]]]] = {}
            for future in as_completed(futures):
                ordered[futures[future]] = future.result()
        section_reviews: list[dict[str, Any]] = []
        review_coverage: set[str] = set()
        for index in sorted(ordered):
            value, disposition, evidence_round_audits = ordered[index]
            review_call_audits.append(_review_prompt_call_audit(
                section_calls[index],
                stage="section",
                call_label=f"companion-section-review-{index}",
                disposition=disposition,
            ))
            review_call_audits.extend(evidence_round_audits)
            validation_error = _section_review_validation_error(value, chunks[index])
            if validation_error is not None:
                raise RuntimeError(f"section review {index} {validation_error}")
            chunk_ids = {str(item["segment"]["segment_id"]) for item in chunks[index]}
            reviewed_ids = {str(item) for item in value["reviewed_segment_ids"]}
            chunk_findings = list(value["findings"])
            review_coverage.update(reviewed_ids)
            findings.extend(chunk_findings)
            section_reviews.append({
                "section_index": index,
                "reviewed_segment_ids": sorted(reviewed_ids),
                "findings": chunk_findings,
                "patch_proposals": _section_review_patch_proposals(
                    chunks[index], list(value["patches"]), chunk_findings
                ),
            })
        all_segment_ids = {str(item["segment_id"]) for item in segments}
        if review_coverage != all_segment_ids:
            missing = sorted(all_segment_ids - review_coverage)
            raise RuntimeError(f"hierarchical review did not cover every segment: {missing}")

    final_payload = direct_payload
    final_prompt = direct_prompt
    if hierarchical:
        final_payload, final_prompt = _bounded_hierarchical_review_prompt(
            section_reviews,
            segments,
            blocks_by_id=by_id,
            document=document,
            segment_payloads=payload["segments"],
            glossary=glossary,
            protected_names=protected_names,
            language=options.annotation_language,
            max_prompt_bytes=review_prompt_target,
            strict_prompt_bytes=review_prompt_limit,
            intent_guidance=intent_guidance,
        )
    _require_review_prompt_within_limit(
        final_prompt,
        label="final review",
        max_prompt_bytes=review_prompt_limit,
    )
    final_rendered = _rendered_review_call(
        payload["segments"],
        final_prompt,
        target_prompt_bytes=review_prompt_target,
        strict_prompt_bytes=review_prompt_limit,
        headroom_class="essential_final_headroom",
    )
    final_evidence_round_audits: list[dict[str, Any]] = []
    final_review_input_sha256 = sha256_json({
        "prompt": final_prompt,
        "schema": REVIEW_SCHEMA,
        "model_tier": REVIEW_TIER,
    })
    final_review_acceptance_path = checkpoint_dir / "final-review-accepted.json"
    review = _llm_call(
        llm,
        final_prompt,
        REVIEW_SCHEMA,
        options=options,
        artifact_dir=checkpoint_dir / "llm" / "final-review",
        call_label="companion-final-review",
        model_tier=REVIEW_TIER,
        paper_access_policy=_guidance_policy(intent_guidance, lane="review"),
        intent_guidance=intent_guidance,
        intent_guidance_lane="review",
        review_prompt_context=_review_prompt_context(
            final_rendered,
            stage="hierarchical-final" if hierarchical else "direct-final",
            audit_sink=final_evidence_round_audits,
        ),
        recovery_descriptor=submission_descriptor(
            unit="final-review",
            logical_unit="final-review",
            checkpoint_dir=checkpoint_dir,
            artifact_root=checkpoint_dir / "llm" / "final-review",
            acceptance_checkpoint=final_review_acceptance_path,
            input_sha256=final_review_input_sha256,
            ordered_siblings=["final-review"],
            suffix=["final-review"],
        ),
    )
    review = _normalize_sparse_review_patches(
        review,
        block_order_by_segment={
            str(segment["segment_id"]): _augmentation_block_ids(segment, by_id)
            for segment in segments
        },
        scope="final review",
    )
    review_call_audits.append(_review_prompt_call_audit(
        final_rendered,
        stage="hierarchical-final" if hierarchical else "direct-final",
        call_label="companion-final-review",
        disposition="provider-call",
    ))
    review_call_audits.extend(final_evidence_round_audits)
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
        annotation_fields = (
            "commentary", "explanation", "commentary_sources", "prior_work", "later_work",
        )
        changed_annotation_fields = [field for field in annotation_fields if patch.get(field) is not None]
        if not changed_annotation_fields and not translation_changed:
            raise RuntimeError(f"review returned an empty patch: {segment_id}")
        for field in changed_annotation_fields:
            if field in {"commentary", "explanation"}:
                text = str(patch[field])
                reviewed[segment_id][field] = text
            else:
                reviewed[segment_id][field] = patch[field]
        reviewed[segment_id] = _validate_direct_annotation_sources(
            reviewed[segment_id],
            allowed_urls=_annotation_source_urls(original_annotation),
        )
        reviewed[segment_id] = clean_reader_annotation(
            reviewed[segment_id],
            evidence_records=reader_evidence.get(segment_id, []),
            language=options.annotation_language,
        )
        patched.add(segment_id)
    final_review_audit = {
        "hierarchical": hierarchical,
        "section_findings": findings,
        "reviewed_segment_ids": [str(item["segment_id"]) for item in segments],
        "issues": [str(item) for item in review.get("issues") or []],
        "patched_segment_ids": sorted(patched),
        "citation_delimiter_normalized_segment_ids": sorted(citation_normalized),
        "prompt_budget_audit": {
            "schema_version": REVIEW_PROMPT_BUDGET_AUDIT_VERSION,
            "budget": prompt_budget,
            "routing": {
                "mode": "hierarchical" if hierarchical else "direct",
                "direct_prompt_bytes": _utf8_size(direct_prompt),
                "hierarchy_threshold_bytes": hierarchy_threshold,
            },
            "calls": review_call_audits,
            "historical_measurements_available": True,
        },
    }
    write_json(final_review_acceptance_path, {
        "schema_version": "arc.companion.final-review-acceptance.v1",
        "input_sha256": final_review_input_sha256,
        "response": review,
        "reviewed_translation_sha256": sha256_json(reviewed_translations),
        "reviewed_annotation_sha256": sha256_json(reviewed),
        "audit": final_review_audit,
    })
    _accept_registered_pipeline_control(
        checkpoint_dir, "final-review", "final-review",
    )
    return reviewed_translations, reviewed, final_review_audit


def _review_commentary_only(
    segments: list[dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    *,
    document: dict[str, Any],
    glossary: dict[str, Any],
    evidence: dict[str, Any],
    options: BuildOptions,
    llm: Callable[..., dict[str, Any]],
    checkpoint_dir: Path,
    intent_guidance: Mapping[str, Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Review commentary without exposing a translation field to the model."""
    by_id = {block_id(block): block for block in document.get("blocks") or []}
    reader_evidence = _reader_evidence_by_segment(
        segments, document=document, evidence=evidence, annotations=annotations
    )
    reviewed = {
        str(segment_id): clean_reader_annotation(
            annotation,
            evidence_records=reader_evidence.get(str(segment_id), []),
            language=options.annotation_language,
        )
        for segment_id, annotation in annotations.items()
    }
    segment_payloads = [{
        "segment": _semantic_segment_descriptor(segment),
        "source_blocks": [
            _annotation_input_block(by_id[value], document)
            for value in _augmentation_block_ids(segment, by_id)
        ],
        "annotation": reviewed[str(segment["segment_id"])],
        "context_evidence": _review_context_evidence(
            segment, blocks_by_id=by_id, evidence=evidence
        ),
    } for segment in segments]
    base = {"glossary": glossary}
    direct_prompt = _guided_prompt(commentary_review_prompt(
        {**base, "segments": segment_payloads}, language=options.annotation_language
    ), intent_guidance, lane="review")
    prompt_budget = _review_prompt_budget(options)
    limit = int(prompt_budget["strict_limit_bytes"])
    target = int(prompt_budget["target_limit_bytes"])
    hierarchy_threshold = min(target, max(1, int(options.review_context_chars)))
    hierarchical = _utf8_size(direct_prompt) > hierarchy_threshold

    # Freeze every final prompt before starting any worker.  A singleton may use
    # the ten-percent reserve, but never cross the strict ceiling.  The existing
    # empty-glossary fallback is used only when the relevant projection itself
    # cannot fit under that ceiling.
    def render_commentary_prompt(group: list[dict[str, Any]]) -> str:
        group_glossary = (
            glossary
            if not hierarchical else _commentary_review_glossary_projection(
                glossary, group, max_bytes=ANNOTATION_GLOSSARY_MAX_BYTES,
            )
        )
        prompt = _guided_prompt(commentary_review_prompt(
            {"glossary": group_glossary, "segments": group},
            language=options.annotation_language,
        ), intent_guidance, lane="review")
        if _utf8_size(prompt) > limit and len(group) == 1 and group_glossary.get("entries"):
            group_glossary = _empty_commentary_review_glossary(glossary)
            prompt = _guided_prompt(commentary_review_prompt(
                {"glossary": group_glossary, "segments": group},
                language=options.annotation_language,
            ), intent_guidance, lane="review")
        return prompt

    commentary_calls = (
        _pack_rendered_review_calls(
            segment_payloads,
            render_prompt=render_commentary_prompt,
            target_prompt_bytes=target,
            strict_prompt_bytes=limit,
            label="commentary-only review",
        )
        if hierarchical else [
            _rendered_review_call(
                segment_payloads,
                direct_prompt,
                target_prompt_bytes=target,
                strict_prompt_bytes=limit,
                headroom_class="singleton_headroom",
            )
        ]
    )
    payload_groups = [(list(call["items"]), {}) for call in commentary_calls]
    commentary_logical_units = [
        _stable_recovery_chunk_id(
            "commentary-review",
            [str(value) for value in call.get("segment_ids") or []],
            payload={"prompt": call.get("prompt"), "items": call.get("items")},
        )
        for call in commentary_calls
    ]
    commentary_group_sha256 = sha256_json({
        "unit": "commentary-review",
        "segments": segment_payloads,
        "language": options.annotation_language,
    })

    recovered_reviews = (
        {} if options.force or not hierarchical
        else _load_recovered_commentary_reviews(
            checkpoint_dir, [group for group, _ in payload_groups]
        )
    )

    def inspect(
        index: int, rendered: dict[str, Any],
    ) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
        group = list(rendered["items"])
        prompt = str(rendered["prompt"])
        evidence_round_audits: list[dict[str, Any]] = []
        input_sha256 = sha256_json({
            "prompt": prompt,
            "schema": COMMENTARY_REVIEW_SCHEMA,
            "model_tier": REVIEW_TIER,
        })
        logical_unit = commentary_logical_units[index]
        path = checkpoint_dir / "commentary-reviews" / f"{index:04d}.json"
        if path.is_file() and not options.force:
            checkpoint = read_json(path)
            if (
                isinstance(checkpoint, dict)
                and checkpoint.get("schema_version")
                == COMMENTARY_REVIEW_CHECKPOINT_VERSION
                and checkpoint.get("input_sha256") == input_sha256
                and _commentary_review_validation_error(
                    checkpoint.get("review"), group
                ) is None
            ):
                _accept_completed_pipeline_controls(
                    checkpoint_dir,
                    caller_validated_units=frozenset({"commentary-review"}),
                    only_logical_unit=logical_unit,
                )
                return checkpoint["review"], "checkpoint-reuse", []
        value = recovered_reviews.get(index)
        disposition = "recovered-reuse" if value is not None else "provider-call"
        if value is None:
            value = _llm_call(
                llm,
                prompt,
                COMMENTARY_REVIEW_SCHEMA,
                options=options,
                artifact_dir=checkpoint_dir / "llm" / "commentary-review" / logical_unit,
                call_label=f"companion-commentary-review-{index}",
                model_tier=REVIEW_TIER,
                paper_access_policy=_guidance_policy(intent_guidance, lane="review"),
                intent_guidance=intent_guidance,
                intent_guidance_lane="review",
                review_prompt_context=_review_prompt_context(
                    rendered,
                    stage="commentary",
                    audit_sink=evidence_round_audits,
                ),
                recovery_descriptor=submission_descriptor(
                    unit="commentary-review",
                    logical_unit=logical_unit,
                    checkpoint_dir=checkpoint_dir,
                    artifact_root=(
                        checkpoint_dir / "llm" / "commentary-review" / logical_unit
                    ),
                    acceptance_checkpoint=path,
                    input_sha256=input_sha256,
                    group_sha256=commentary_group_sha256,
                    ordered_siblings=commentary_logical_units,
                    suffix=commentary_logical_units[index:],
                ),
            )
        validation_error = _commentary_review_validation_error(value, group)
        if validation_error is not None:
            if validation_error == "attempted a translation patch":
                raise RuntimeError("commentary-only review attempted a translation patch")
            raise RuntimeError(f"commentary-only review {index} {validation_error}")
        write_json(path, {
            "schema_version": COMMENTARY_REVIEW_CHECKPOINT_VERSION,
            "group_index": index,
            "input_sha256": input_sha256,
            "reviewed_segment_ids": sorted(
                str(item["segment"]["segment_id"]) for item in group
            ),
            "prompt_budget_audit": _review_prompt_call_audit(
                rendered,
                stage="commentary",
                call_label=f"companion-commentary-review-{index}",
                disposition=disposition,
            ),
            "evidence_prompt_budget_audits": evidence_round_audits,
            "review": value,
        })
        _accept_registered_pipeline_control(
            checkpoint_dir, "commentary-review", logical_unit,
        )
        return value, disposition, evidence_round_audits

    responses: list[dict[str, Any]] = []
    dispositions: list[str] = []
    with ThreadPoolExecutor(max_workers=min(options.workers, len(payload_groups))) as executor:
        futures = [
            executor.submit(inspect, index, rendered)
            for index, rendered in enumerate(commentary_calls)
        ]
        completed = [future.result() for future in futures]
        responses = [value for value, _, _ in completed]
        dispositions = [disposition for _, disposition, _ in completed]

    valid_ids = set(reviewed)
    patched: set[str] = set()
    issues: list[str] = []
    for (group, _), response in zip(payload_groups, responses):
        group_ids = {
            str(item["segment"]["segment_id"]) for item in group
        }
        issues.extend(str(item) for item in response.get("issues") or [])
        for patch in response.get("patches") or []:
            if "translation" in patch or "translation_blocks" in patch:
                raise RuntimeError("commentary-only review attempted a translation patch")
            segment_id = str(patch.get("segment_id") or "")
            if segment_id not in group_ids or segment_id not in valid_ids or segment_id in patched:
                raise RuntimeError(
                    f"review returned out-of-group, invalid, or duplicate annotation patch: {segment_id}"
                )
            original = dict(reviewed[segment_id])
            fields = (
                "commentary", "explanation", "commentary_sources", "prior_work", "later_work",
            )
            changed = [field for field in fields if patch.get(field) is not None]
            if not changed:
                raise RuntimeError(f"review returned an empty patch: {segment_id}")
            for field in changed:
                if field in {"commentary", "explanation"}:
                    reviewed[segment_id][field] = str(patch[field])
                else:
                    reviewed[segment_id][field] = patch[field]
            reviewed[segment_id] = _validate_direct_annotation_sources(
                reviewed[segment_id], allowed_urls=_annotation_source_urls(original),
            )
            reviewed[segment_id] = clean_reader_annotation(
                reviewed[segment_id],
                evidence_records=reader_evidence.get(segment_id, []),
                language=options.annotation_language,
            )
            patched.add(segment_id)
    return reviewed, {
        "translation_mode": "skipped",
        "hierarchical": hierarchical,
        "review_group_count": len(payload_groups),
        "reviewed_segment_ids": [str(item["segment_id"]) for item in segments],
        "issues": issues,
        "patched_segment_ids": sorted(patched),
        "prompt_budget_audit": {
            "schema_version": REVIEW_PROMPT_BUDGET_AUDIT_VERSION,
            "budget": prompt_budget,
            "routing": {
                "mode": (
                    "commentary-hierarchical" if hierarchical else "commentary-direct"
                ),
                "direct_prompt_bytes": _utf8_size(direct_prompt),
                "hierarchy_threshold_bytes": hierarchy_threshold,
            },
            "calls": [
                audit
                for index, rendered in enumerate(commentary_calls)
                for audit in (
                    [_review_prompt_call_audit(
                        rendered,
                        stage="commentary",
                        call_label=f"companion-commentary-review-{index}",
                        disposition=dispositions[index],
                    )]
                    + completed[index][2]
                )
            ],
            "historical_measurements_available": True,
        },
    }


def _empty_commentary_review_glossary(glossary: dict[str, Any]) -> dict[str, Any]:
    entries = glossary.get("entries") or [] if isinstance(glossary, dict) else []
    return {
        "schema_version": "arc.companion.commentary-review-glossary-projection.v1",
        "source_glossary_schema_version": (
            glossary.get("schema_version") if isinstance(glossary, dict) else None
        ),
        "source_glossary_sha256": sha256_json(
            glossary if isinstance(glossary, dict) else {}
        ),
        "entries": [],
        "source_entry_count": len(entries),
        "selected_entry_count": 0,
        "omitted_entry_count": len(entries),
    }


def _commentary_review_glossary_projection(
    glossary: dict[str, Any],
    group: list[dict[str, Any]],
    *,
    max_bytes: int,
) -> dict[str, Any]:
    """Keep only complete glossary entries relevant to one local review group."""
    projection = _empty_commentary_review_glossary(glossary)
    entries = [
        dict(item) for item in (glossary.get("entries") or [])
        if isinstance(item, dict)
    ] if isinstance(glossary, dict) else []
    group_block_ids = {
        str(block_id_value)
        for item in group
        for block_id_value in (item.get("segment") or {}).get("block_ids") or []
    }
    group_text = _normalized_glossary_match_text(group)
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for index, entry in enumerate(entries):
        terms = [
            entry.get("source_term"), entry.get("target_term"),
            *(entry.get("aliases") or []),
        ]
        normalized_terms = [
            _normalized_glossary_match_text(value)
            for value in terms if str(value or "").strip()
        ]
        if str(entry.get("first_block_id") or "") in group_block_ids:
            priority = 0
        elif any(_glossary_term_in_text(term, group_text) for term in normalized_terms):
            priority = 1
        else:
            continue
        candidates.append((priority, index, entry))

    selected: list[tuple[int, dict[str, Any]]] = []
    for _, index, entry in sorted(candidates):
        proposed = [*selected, (index, entry)]
        ordered = [value for _, value in sorted(proposed)]
        candidate = {
            **projection,
            "entries": ordered,
            "selected_entry_count": len(ordered),
            "omitted_entry_count": len(entries) - len(ordered),
        }
        if _utf8_size(json.dumps(candidate, ensure_ascii=False)) <= max_bytes:
            selected = proposed
    ordered = [value for _, value in sorted(selected)]
    return {
        **projection,
        "entries": ordered,
        "selected_entry_count": len(ordered),
        "omitted_entry_count": len(entries) - len(ordered),
    }


def _commentary_review_validation_error(
    review: Any, group: list[dict[str, Any]],
) -> str | None:
    expected_ids = {
        str((item.get("segment") or {}).get("segment_id") or "") for item in group
    }
    if not expected_ids or "" in expected_ids:
        return "has invalid expected segment coverage"
    if not isinstance(review, dict):
        return "is malformed"
    patches = review.get("patches")
    issues = review.get("issues")
    if not isinstance(patches, list) or not isinstance(issues, list):
        return "is missing patches or issues"
    patch_ids: list[str] = []
    for patch in patches:
        if not isinstance(patch, dict):
            return "contains a malformed patch"
        if "translation" in patch or "translation_blocks" in patch:
            return "attempted a translation patch"
        segment_id = str(patch.get("segment_id") or "")
        if segment_id not in expected_ids:
            return "returned a patch outside its review group"
        patch_ids.append(segment_id)
    if len(patch_ids) != len(set(patch_ids)):
        return "returned duplicate patches"
    return None


def _load_recovered_commentary_reviews(
    checkpoint_dir: Path, groups: list[list[dict[str, Any]]],
) -> dict[int, dict[str, Any]]:
    """Import recovered local reviews only when exact disjoint groups still match."""
    path = checkpoint_dir / "commentary-reviews.recovered-from-failed-review.v1.json"
    if not path.is_file():
        return {}
    recovered = read_json(path)
    if (
        not isinstance(recovered, dict)
        or recovered.get("schema_version")
        != "arc.companion.recovered-commentary-reviews.v1"
        or not isinstance(recovered.get("commentary_reviews"), list)
    ):
        raise RuntimeError("invalid recovered commentary-review checkpoint")
    expected_all = {
        str(item["segment"]["segment_id"]) for group in groups for item in group
    }
    if set(str(value) for value in recovered.get("reviewed_segment_ids") or []) != expected_all:
        raise RuntimeError("recovered commentary reviews do not match current segment coverage")
    by_ids: dict[frozenset[str], dict[str, Any]] = {}
    covered: set[str] = set()
    for item in recovered["commentary_reviews"]:
        if not isinstance(item, dict):
            raise RuntimeError("recovered commentary review is malformed")
        declared_values = [str(value) for value in item.get("reviewed_segment_ids") or []]
        declared_ids = set(declared_values)
        review = item.get("review")
        if review is None:
            review = {"patches": item.get("patches"), "issues": item.get("issues")}
        synthetic_group = [
            {"segment": {"segment_id": segment_id}} for segment_id in declared_values
        ]
        if (
            not declared_ids
            or len(declared_values) != len(declared_ids)
            or _commentary_review_validation_error(review, synthetic_group) is not None
        ):
            raise RuntimeError("recovered commentary review does not match its group")
        frozen = frozenset(declared_ids)
        if frozen in by_ids or covered.intersection(declared_ids):
            raise RuntimeError("recovered commentary reviews have duplicate or overlapping groups")
        covered.update(declared_ids)
        by_ids[frozen] = review
    if covered != expected_all:
        raise RuntimeError("recovered commentary reviews do not cover every group")
    return {
        index: by_ids[ids]
        for index, group in enumerate(groups)
        if (ids := frozenset(str(item["segment"]["segment_id"]) for item in group)) in by_ids
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
                _related_work_claim_key(claim)
                for claim in old_value or [] if isinstance(claim, dict)
            } if isinstance(old_value, list) else set()
            new_bindings = {
                _related_work_claim_key(claim)
                for claim in new_value if isinstance(claim, dict)
            }
            if not new_bindings.issubset(old_bindings):
                raise RuntimeError("review added a related-work claim without prior claim evidence")
        elif str(new_value or "").strip() and not str(old_value or "").strip():
            raise RuntimeError("review added a related-work claim without prior claim evidence")
        elif isinstance(old_value, list) and str(new_value or "").strip():
            raise RuntimeError("review replaced claim bindings with unbound related-work text")


def _related_work_claim_key(claim: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(claim.get("text") or "").strip(),
        tuple(
            (
                str(item.get("title") or "").strip(),
                str(item.get("url") or "").strip(),
                str(item.get("locator") or "").strip(),
            )
            for item in claim.get("sources") or []
            if isinstance(item, dict)
        ),
    )


def _intent_guidance_schema(
    schema: dict[str, Any], intent_guidance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if intent_guidance is None:
        return schema
    from arc_llm import allow_evidence_requests

    return allow_evidence_requests(schema) or schema


def _controller_evidence_prompt(responses: Sequence[Any]) -> str:
    return (
        "CONTROLLER REFERENCE EVIDENCE ROUND:\n"
        + json.dumps(
            [
                {
                    "request_id": response.request_id,
                    "ok": response.ok,
                    "data": response.data,
                    "error": response.error,
                    "provenance": dict(response.provenance),
                }
                for response in responses
            ],
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
        + "\nUse these responses, request the next page only if necessary, and return "
        "the required result plus arc_evidence_requests ([] when complete)."
    )


def _companion_evidence_journal_context(
    *,
    checkpoint_dir: Path,
    run_id: str,
    lane: str,
    worker_id: str,
    logical_task_id: str,
    source_generation: int,
    policy: Any,
    runtime: Any,
) -> EvidenceJournalContext:
    """Build the explicit, provider-neutral address used by companion loops."""

    from arc_llm import EvidenceJournalContext, evidence_identity_hash

    return EvidenceJournalContext(
        journal_root=checkpoint_dir / "evidence-journal",
        run_id=str(run_id),
        lane_id=str(lane),
        worker_id=str(worker_id),
        logical_task_id=str(logical_task_id).split(":evidence-", 1)[0],
        source_generation=int(source_generation),
        policy_hash=evidence_identity_hash({
            "protocol": "arc.companion.controller-evidence.v1",
            "lane": str(lane),
            "policy": policy,
        }),
        runtime_hash=evidence_identity_hash({"lane": str(lane), "runtime": runtime}),
    )


def _companion_evidence_source_identity(
    *,
    project_dir: Path,
    checkpoint_dir: Path,
    run_id: str,
    ledger_path: Path,
    ledger: Mapping[str, Any],
    session_key: str,
    logical_unit: str,
    target_generation: int,
    target_idempotency_key: str,
) -> tuple[int, str]:
    """Recover a typed-idle source address from two durable authorities.

    The archived ledger marker proves which submitted source generation was
    abandoned.  The replacement transaction independently proves that the
    current generation was authorized as its replacement.  Any partial or
    conflicting evidence fails closed instead of silently spending a second
    resolver call under a fresh journal address.
    """

    current = (int(target_generation), _base_evidence_id(target_idempotency_key))
    if target_generation <= 1:
        return current
    from .resume_transaction import load_transaction

    transaction = load_transaction(project_dir)
    ledger_signals = [
        dict(item) for item in ledger.get("supervision_history") or []
        if isinstance(item, Mapping)
        and str(item.get("segment_id") or "") == logical_unit
        and int(item.get("target_generation") or 0) == target_generation
        and str(item.get("archive_reason") or "")
        == "generation_suffix_invalidated"
        and isinstance(item.get("recovery_context"), Mapping)
        and str(item["recovery_context"].get("session_key") or "") == session_key
        and ":evidence-" in str(
            item["recovery_context"].get("idempotency_key") or ""
        )
    ]
    transaction_signals = [
        dict(item) for item in (transaction or {}).get("replacements") or []
        if isinstance(item, Mapping)
        and str(item.get("session_key") or "") == session_key
        and int(item.get("target_generation") or 0) == target_generation
        and logical_unit in {
            str(value) for value in item.get("suffix_segment_ids") or []
        }
        and str(item.get("trigger_code") or "") == "idle_timeout"
        and ":evidence-" in str(item.get("abandoned_logical_key") or "")
    ]
    if not ledger_signals and not transaction_signals:
        return current
    if len(ledger_signals) != 1 or len(transaction_signals) != 1:
        raise RuntimeError(
            "evidence generation rotation has one-sided or ambiguous durable authority"
        )
    if transaction is None:
        raise RuntimeError("evidence generation rotation transaction is missing")
    checkpoint_path = str(transaction.get("checkpoint_path") or "")
    checkpoint_fingerprint = str(transaction.get("checkpoint_fingerprint") or "")
    replacement = transaction_signals[0]
    marker = ledger_signals[0]
    recovery = marker["recovery_context"]
    source_generation = int(replacement.get("source_generation") or 0)
    abandoned_key = str(replacement.get("abandoned_logical_key") or "")
    canonical_ledger = ledger_path.resolve(strict=False)
    exact = bool(
        checkpoint_path
        and Path(checkpoint_path).resolve(strict=False)
        == checkpoint_dir.resolve(strict=False)
        and checkpoint_fingerprint == str(run_id)
        and str(replacement.get("authorization_source") or "")
        == "recovery_policy_auto"
        and str(replacement.get("status") or "") in {
            "suffix_invalidated", "response_persisted", "accepted",
        }
        and Path(str(replacement.get("ledger_path") or "")).resolve(strict=False)
        == canonical_ledger
        and 0 < source_generation < target_generation
        and int(marker.get("source_generation") or 0) == source_generation
        and int(recovery.get("generation") or 0) == source_generation
        and str(recovery.get("idempotency_key") or "") == abandoned_key
        and bool(abandoned_key)
    )
    if not exact:
        raise RuntimeError("evidence generation rotation durable authorities conflict")
    return source_generation, _base_evidence_id(abandoned_key)


def _base_evidence_id(value: str) -> str:
    return str(value).split(":evidence-", 1)[0]


def _authoritative_build_fingerprint(checkpoint_dir: Path) -> str:
    """Read the build identity from its receipt, never from directory names."""

    receipt = read_json(checkpoint_dir / "source-snapshot-receipt.json")
    fingerprint = str(receipt.get("fingerprint") or "").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise RuntimeError("companion source snapshot has no authoritative build fingerprint")
    return fingerprint


def _complete_stateful_reference_evidence(
    initial_outcome: Any,
    *,
    intent_guidance: Mapping[str, Any] | None,
    lane: str,
    worker_id: str,
    schema: dict[str, Any],
    call_round: Callable[[str, dict[str, Any], int], Any],
    journal_context: EvidenceJournalContext | None = None,
    target_session: str | None = None,
    target_generation: int | None = None,
    followup_id: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Resolve controller reads as native-session delta turns for shell-less hosts."""
    outcome = initial_outcome
    active_schema = _intent_guidance_schema(schema, intent_guidance)
    for round_number in range(0, 4):
        value = dict(outcome.value)
        if intent_guidance is None:
            return outcome, value
        from arc_llm import (
            EvidenceJournal,
            EvidenceOperationPolicy,
            evidence_requests_from_output,
            resolve_evidence_round,
        )

        requests = evidence_requests_from_output(
            value, worker_id=worker_id, role="companion-content-worker",
        )
        if not requests:
            value.pop("arc_evidence_requests", None)
            return outcome, value
        if round_number >= 3:
            raise RuntimeError(
                "companion reference evidence exceeded three controller rounds"
            )
        resolver = lambda material, *, round_number: resolve_worker_evidence_requests(
            intent_guidance, material, round_number=round_number, lane=lane,
        )
        if journal_context is None:
            responses = resolve_evidence_round(
                requests, resolver, round_number=round_number + 1,
            )
        else:
            journal = EvidenceJournal(journal_context.journal_root)
            responses = journal.resolve_round(
                journal_context,
                requests,
                resolver,
                round_number=round_number + 1,
                operation_policies={
                    request.operation: EvidenceOperationPolicy(idempotent=True)
                    for request in requests
                },
            )
            journal.mark_delivered(
                journal_context,
                requests,
                round_number=round_number + 1,
                target_generation=(
                    target_generation or journal_context.source_generation
                ),
                target_session=target_session or journal_context.worker_id,
                followup_id=(
                    f"{followup_id or journal_context.logical_task_id}:"
                    f"evidence-{round_number + 1:02d}"
                ),
            )
        outcome = call_round(
            _controller_evidence_prompt(responses), active_schema, round_number + 1,
        )
    raise AssertionError("unreachable stateful evidence loop")


def _prepare_pipeline_recovery_control(
    descriptor: Mapping[str, Any], *, artifact_dir: Path,
) -> dict[str, Any]:
    if descriptor.get("schema_version") != "arc.companion.recovery-call-descriptor.v1":
        raise RecoveryResponseError("pipeline recovery descriptor schema is invalid")
    unit = str(descriptor.get("unit") or "")
    spec = recovery_unit_for_ledger(unit)
    if (
        spec is None
        or descriptor.get("validator") != spec.validator
        or descriptor.get("application") != spec.application
        or descriptor.get("side_effect_policy") != spec.side_effect_policy
        or descriptor.get("external_side_effects") is not False
    ):
        raise RecoveryResponseError("pipeline recovery descriptor handler is invalid")
    root = Path(str(descriptor.get("checkpoint_dir") or "")).resolve(strict=False)
    artifact_lexical = Path(str(descriptor.get("artifact_root") or ""))
    acceptance_lexical = Path(str(descriptor.get("acceptance_checkpoint") or ""))
    artifact_call_lexical = artifact_dir.absolute()
    safe_candidates: list[Path] = []
    for candidate in (artifact_lexical, acceptance_lexical, artifact_call_lexical):
        try:
            relative = candidate.relative_to(root)
            safe_candidates.append(resolve_recovery_path(root, relative.as_posix()))
        except (RecoveryResponseError, ValueError) as exc:
            raise RecoveryResponseError(
                "pipeline recovery descriptor escapes its active checkpoint"
            ) from exc
    artifact_root, acceptance, artifact_call = safe_candidates
    if artifact_root != artifact_call:
        raise RecoveryResponseError("pipeline recovery artifact identity changed")
    logical_unit = str(descriptor.get("logical_unit") or "")
    ordered = [str(item) for item in descriptor.get("ordered_siblings") or []]
    suffix = [str(item) for item in descriptor.get("suffix") or []]
    if (
        not logical_unit
        or logical_unit not in ordered
        or suffix != ordered[ordered.index(logical_unit):]
        or not re.fullmatch(r"[0-9a-f]{64}", str(descriptor.get("input_sha256") or ""))
        or not re.fullmatch(r"[0-9a-f]{64}", str(descriptor.get("group_sha256") or ""))
    ):
        raise RecoveryResponseError("pipeline recovery logical ownership is invalid")
    identity_sha = sha256_json({
        key: descriptor.get(key) for key in (
            "unit", "group_sha256", "validator", "application",
        )
    })
    chapter_id = f"pipeline-{unit}-{identity_sha[:12]}"
    ledger_path = (
        root / "recovery-controls" / unit / f"{identity_sha[:16]}-ledger.json"
    )
    # Concurrent section/glossary/review handlers may bind distinct logical
    # units to the same ordered control ledger.  Serialize the in-process
    # read/rebind/register transaction; the project build lock excludes a
    # second controller process.
    with _RECOVERY_CONTROL_INITIALIZE_LOCK:
        ledger = initialize_control_ledger(
            ledger_path,
            chapter_id=chapter_id,
            lane=unit,
            segment_ids=ordered,
            checkpoint_dir=root,
        )
        target = next(
            item for item in ledger.get("blocks") or []
            if item.get("segment_id") == logical_unit
        )
        if target.get("state") in {"accepted", "response_received"}:
            # Reaching this function means the normal business handler chose
            # to submit a new paid call.  An accepted control, or an
            # unsealed direct-adapter response that cannot be recovered,
            # belongs to the prior immutable attempt, so rotate the owned
            # suffix before the new receipt is created.  A sealed response is
            # recoverable and must be handled instead of resubmitted.
            if (
                target.get("state") == "response_received"
                and _pipeline_control_receipt_exists(root, unit, logical_unit)
            ):
                raise RecoveryResponseError(
                    "recoverable response must be applied before resubmission"
                )
            ledger = invalidate_suffix(
                ledger_path,
                from_segment_id=logical_unit,
                generation=int(ledger.get("generation") or 1) + 1,
            )
    generation = int(ledger.get("generation") or 1)
    session_key = f"{chapter_id}:{unit}"
    return {
        **dict(descriptor),
        "checkpoint_dir": str(root),
        "ledger_path": str(ledger_path.resolve(strict=False)),
        "acceptance_checkpoint": str(acceptance),
        "session_key": session_key,
        "generation": generation,
        "idempotency_key": (
            f"{session_key}:{logical_unit}:generation-{generation}"
        ),
        "ordered_siblings": ordered,
        "suffix": suffix,
    }


_RECOVERY_CONTROL_INITIALIZE_LOCK = threading.RLock()


def _guarded_mark_transport_state(
    ledger_path: Path,
    *,
    checkpoint_dir: Path,
    session_key: str,
    logical_unit: str,
    idempotency_key: str,
    response_received: bool = False,
) -> None:
    """CAS one production transport transition against its exact five-tuple."""
    try:
        guard = lane_transition_guard(
            ledger_path,
            segment_id=logical_unit,
            session_key=session_key,
            idempotency_key=idempotency_key,
            checkpoint_dir=checkpoint_dir,
        )
        mark_submitted(
            ledger_path,
            segment_id=logical_unit,
            expected_generation=guard.expected_generation,
            expected_ledger_sha256=guard.expected_ledger_sha256,
            authorization=guard.authorization,
            checkpoint_dir=checkpoint_dir,
        )
        if not response_received:
            return
        # The submitted transition changes the registered ledger digest.
        # Acquire a fresh guard instead of reusing stale CAS material.
        guard = lane_transition_guard(
            ledger_path,
            segment_id=logical_unit,
            session_key=session_key,
            idempotency_key=idempotency_key,
            checkpoint_dir=checkpoint_dir,
        )
        mark_response_received(
            ledger_path,
            segment_id=logical_unit,
            expected_generation=guard.expected_generation,
            expected_ledger_sha256=guard.expected_ledger_sha256,
            authorization=guard.authorization,
            checkpoint_dir=checkpoint_dir,
        )
    except (LaneLedgerError, LaneLedgerRegistryError) as exc:
        raise LaneLedgerError(
            "transport state rejected for "
            f"{ledger_path.name}:{logical_unit}:{idempotency_key}: {exc}"
        ) from exc


_CALLER_VALIDATED_RECOVERY_UNITS = frozenset({
    "intent-guidance", "guide", "section-review", "final-review",
    "commentary-review",
})


def _accept_completed_pipeline_controls(
    checkpoint_dir: Path, *, caller_validated_units: frozenset[str] = frozenset(),
    only_logical_unit: str | None = None,
    only_recovery_unit: str | None = None,
    strict: bool = False,
) -> int:
    """Accept only registered control units whose normal handler wrote its checkpoint."""

    root = checkpoint_dir.resolve(strict=False)
    accepted = 0
    for receipt_path, receipt in discover_submission_receipts(root):
        if not receipt.get("sealed"):
            continue
        unit = str(receipt.get("recovery_unit") or "")
        if only_recovery_unit is not None and unit != only_recovery_unit:
            continue
        if unit in _CALLER_VALIDATED_RECOVERY_UNITS and unit not in caller_validated_units:
            continue
        if (
            only_logical_unit is not None
            and receipt.get("logical_unit") != only_logical_unit
        ):
            continue
        spec = recovery_unit_for_ledger(unit)
        if spec is None:
            continue
        try:
            ledger_path = resolve_recovery_path(root, receipt.get("ledger_path"))
            ledger_path.relative_to(root / "recovery-controls")
        except (RecoveryResponseError, ValueError) as exc:
            if strict:
                raise RecoveryResponseError(
                    f"{unit}:{receipt.get('logical_unit')} ledger ownership is invalid"
                ) from exc
            continue
        logical_unit = str(receipt.get("logical_unit") or "")
        try:
            reference = submission_receipt_reference(
                receipt_path, checkpoint_dir=root,
            )
            validated_receipt = _validate_pipeline_submission_reference(
                reference,
                checkpoint_dir=root,
                ledger_path=ledger_path,
                session_key=str(receipt.get("session_key") or ""),
                logical_unit=logical_unit,
                generation=int(receipt.get("generation") or 0),
                idempotency_key=str(receipt.get("idempotency_key") or ""),
            )
            ledger, ledger_digest = read_registered_lane_ledger(root, ledger_path)
            if ledger_digest != validated_receipt.get(
                "current_registered_ledger_sha256"
            ):
                raise RecoveryResponseError(
                    "registered ledger changed after receipt validation"
                )
            receipt = validated_receipt
        except (LaneLedgerRegistryError, RecoveryResponseError) as exc:
            if strict:
                raise RecoveryResponseError(
                    f"{unit}:{receipt.get('logical_unit')} ledger registry is invalid"
                ) from exc
            continue
        block = next((
            item for item in ledger.get("blocks") or []
            if isinstance(item, Mapping)
            and str(item.get("segment_id") or "") == logical_unit
        ), None)
        if not isinstance(block, Mapping) or block.get("state") == "accepted":
            continue
        if block.get("state") != "response_received":
            continue
        try:
            acceptance_path = resolve_recovery_path(
                root, receipt.get("acceptance_checkpoint"),
            )
            acceptance_path.relative_to(root)
        except (RecoveryResponseError, ValueError) as exc:
            if strict:
                raise RecoveryResponseError(
                    f"{unit}:{logical_unit} acceptance ownership is invalid"
                ) from exc
            continue
        output_sha = _pipeline_acceptance_checkpoint_digest(
            unit,
            acceptance_path,
            receipt,
            checkpoint_dir=root,
            require_business_validation=unit not in caller_validated_units,
        )
        if output_sha is None:
            if strict:
                raise RecoveryResponseError(
                    f"{unit}:{logical_unit} business acceptance checkpoint is invalid"
                )
            continue
        ordered_blocks = [
            item for item in ledger.get("blocks") or []
            if isinstance(item, Mapping)
        ]
        block_index = next(
            index for index, item in enumerate(ordered_blocks)
            if str(item.get("segment_id") or "") == logical_unit
        )
        if (
            block_index
            and ordered_blocks[block_index - 1].get("state") != "accepted"
        ):
            # A later concurrent response may be durable before its prefix.
            # Keep it response_received until a subsequent ordered sweep.
            continue
        try:
            for state in ("schema_valid", "invariant_valid"):
                advance_block(
                    ledger_path,
                    segment_id=logical_unit,
                    state=state,
                    expected_ledger_sha256=ledger_digest,
                    checkpoint_dir=root,
                )
                ledger, ledger_digest = read_registered_lane_ledger(
                    root, ledger_path,
                )
            advance_block(
                ledger_path,
                segment_id=logical_unit,
                state="accepted",
                input_sha256=str(receipt.get("input_sha256") or ""),
                output_sha256=output_sha,
                receipt={
                    "kind": "normal_pipeline_acceptance_replay",
                    "validator": spec.validator,
                    "application": spec.application,
                    "recovery_identity": {
                        "control_address": str(ledger_path.resolve(strict=False)),
                        "session_key": str(receipt.get("session_key") or ""),
                        "logical_unit": logical_unit,
                        "generation": int(receipt.get("generation") or 0),
                        "idempotency_key": str(
                            receipt.get("idempotency_key") or ""
                        ),
                    },
                    "submission_receipt_identity_sha256": str(
                        reference.get("identity_sha256") or ""
                    ),
                },
                expected_ledger_sha256=ledger_digest,
                checkpoint_dir=root,
            )
            ledger, ledger_digest = read_registered_lane_ledger(root, ledger_path)
            if ledger.get("needs_supervision"):
                clear_needs_supervision(
                    ledger_path,
                    expected_ledger_sha256=ledger_digest,
                    checkpoint_dir=root,
                )
        except (LaneLedgerError, LaneLedgerRegistryError) as exc:
            if strict:
                raise RecoveryResponseError(
                    f"{unit}:{logical_unit} control acceptance failed"
                ) from exc
            continue
        accepted += 1
    return accepted


def _pipeline_control_receipt_exists(
    checkpoint_dir: Path,
    unit: str,
    logical_unit: str,
) -> bool:
    root = checkpoint_dir.resolve(strict=False)
    for receipt_path, receipt in discover_submission_receipts(root):
        if not (
            receipt.get("sealed")
            and receipt.get("recovery_unit") == unit
            and receipt.get("logical_unit") == logical_unit
        ):
            continue
        try:
            ledger_path = resolve_recovery_path(root, receipt.get("ledger_path"))
            _validate_pipeline_submission_reference(
                submission_receipt_reference(receipt_path, checkpoint_dir=root),
                checkpoint_dir=root,
                ledger_path=ledger_path,
                session_key=str(receipt.get("session_key") or ""),
                logical_unit=logical_unit,
                generation=int(receipt.get("generation") or 0),
                idempotency_key=str(receipt.get("idempotency_key") or ""),
            )
        except (RecoveryResponseError, TypeError, ValueError):
            continue
        return True
    return False


def _accept_registered_pipeline_control(
    checkpoint_dir: Path,
    unit: str,
    logical_unit: str,
) -> int:
    """Require acceptance when a sealed production submission is registered.

    The package also supports direct callback adapters used by embedders and
    tests. They receive the descriptor but may not implement ARC's durable
    receipt protocol; in that case there is no control ledger to promote.
    """

    if not _pipeline_control_receipt_exists(checkpoint_dir, unit, logical_unit):
        return 1
    return _require_exact_pipeline_control_acceptance(
        checkpoint_dir, unit, logical_unit,
    )


def _require_exact_pipeline_control_acceptance(
    checkpoint_dir: Path,
    unit: str,
    logical_unit: str,
) -> int:
    """Accept and verify the one exact control owned by a completed handler."""

    root = checkpoint_dir.resolve(strict=False)
    with _RECOVERY_CONTROL_INITIALIZE_LOCK:
        transitioned = _accept_completed_pipeline_controls(
            root,
            caller_validated_units=(
                frozenset({unit}) if unit in _CALLER_VALIDATED_RECOVERY_UNITS
                else frozenset()
            ),
            only_logical_unit=logical_unit,
            only_recovery_unit=unit,
            strict=True,
        )
    if transitioned > 1:
        raise RecoveryResponseError(
            f"{unit}:{logical_unit} matched more than one control acceptance"
        )
    verified_ledgers: set[Path] = set()
    for receipt_path, receipt in discover_submission_receipts(root):
        if (
            not receipt.get("sealed")
            or receipt.get("recovery_unit") != unit
            or receipt.get("logical_unit") != logical_unit
        ):
            continue
        try:
            ledger_path = resolve_recovery_path(root, receipt.get("ledger_path"))
            reference = submission_receipt_reference(
                receipt_path, checkpoint_dir=root,
            )
            ledger, _digest = read_registered_lane_ledger(root, ledger_path)
        except (RecoveryResponseError, LaneLedgerRegistryError, TypeError, ValueError):
            continue
        block = next((
            item for item in ledger.get("blocks") or []
            if isinstance(item, Mapping)
            and item.get("segment_id") == logical_unit
            and int(item.get("generation") or 0)
            == int(receipt.get("generation") or 0)
        ), None)
        logical_receipt = (
            block.get("logical_receipt") if isinstance(block, Mapping) else None
        )
        expected_identity = {
            "control_address": str(ledger_path.resolve(strict=False)),
            "session_key": str(receipt.get("session_key") or ""),
            "logical_unit": logical_unit,
            "generation": int(receipt.get("generation") or 0),
            "idempotency_key": str(receipt.get("idempotency_key") or ""),
        }
        if (
            isinstance(block, Mapping)
            and block.get("state") == "accepted"
            and isinstance(logical_receipt, Mapping)
            and logical_receipt.get("recovery_identity") == expected_identity
            and logical_receipt.get("submission_receipt_identity_sha256")
            == reference.get("identity_sha256")
        ):
            verified_ledgers.add(ledger_path.resolve(strict=False))
    if len(verified_ledgers) != 1:
        raise RecoveryResponseError(
            f"{unit}:{logical_unit} did not resolve to one accepted control ledger"
        )
    return 1


def _pipeline_acceptance_checkpoint_valid(
    unit: str,
    path: Path,
    receipt: Mapping[str, Any],
    *,
    checkpoint_dir: Path | None = None,
    _validated_value: Mapping[str, Any] | None = None,
    _validated_raw: bytes | None = None,
) -> bool:
    value: Any = _validated_value
    if value is None:
        try:
            value = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            return False
    if not isinstance(value, Mapping):
        return False
    input_sha = str(receipt.get("input_sha256") or "")
    if unit == "segmentation":
        return (
            value.get("window_sha256") == input_sha
            and isinstance(value.get("response"), Mapping)
            and _recovery_candidate_matches_schema(value["response"], receipt)
            and validate_segmentation_acceptance_checkpoint(dict(value))
        )
    if unit in {"glossary", "glossary-consolidation"}:
        result = value.get("result")
        return (
            value.get("input_sha256") == input_sha
            and isinstance(result, Mapping)
            and isinstance(result.get("entries"), list)
            and _recovery_candidate_matches_schema(result, receipt)
            and validate_glossary_acceptance_checkpoint(dict(value))
        )
    if unit == "intent-guidance":
        return (
            value.get("schema_version") == "arc.companion.intent-guidance.v2"
            and value.get("semantic_input_sha256") == input_sha
            and isinstance(value.get("guidance"), str)
            and value.get("resolution_status") in {"resolved", "ambiguous"}
            and _recovery_candidate_matches_schema({
                "guidance": value.get("guidance"),
                "resolution_status": value.get("resolution_status"),
                "reference_targets": value.get("reference_targets") or [],
            }, receipt)
        )
    if unit == "annotation":
        return (
            value.get("schema_version") == ANNOTATION_CHECKPOINT_VERSION
            and value.get("segment_id") == receipt.get("logical_unit")
            and value.get("generation") == receipt.get("generation")
            and value.get("input_sha256") == input_sha
            and isinstance(value.get("annotation"), Mapping)
            and _recovery_candidate_matches_schema(value["annotation"], receipt)
        )
    if unit == "translation":
        segment_id = str(receipt.get("logical_unit") or "").split(":", 1)[0]
        return (
            value.get("schema_version") == "arc.companion.translation-checkpoint.v2"
            and value.get("segment_id") == segment_id
            and value.get("generation") == receipt.get("generation")
            and value.get("input_sha256") == input_sha
            and isinstance(value.get("translation"), Mapping)
            and _recovery_candidate_matches_schema(value["translation"], receipt)
        )
    if unit in {
        "translation-token-repair", "translation-coverage-repair",
    }:
        if checkpoint_dir is None:
            return False
        return _translation_repair_acceptance_valid(
            unit,
            value,
            receipt,
            checkpoint_dir=checkpoint_dir,
            acceptance_path=path,
            acceptance_raw=_validated_raw,
        )
    if unit == "glossary-index":
        response = value.get("response")
        expected_ids = [
            str(item) for item in value.get("expected_entry_ids") or []
        ]
        batch_identity = {
            "source_sha256": value.get("source_sha256"),
            "language": value.get("language"),
            "expected_entry_ids": expected_ids,
            "prompt_sha256": value.get("prompt_sha256"),
            "schema_sha256": value.get("schema_sha256"),
        }
        expected_logical_suffix = sha256_json(expected_ids)[:16]
        return (
            value.get("schema_version") == INDEX_GLOSSARY_BATCH_VERSION
            and value.get("logical_unit") == receipt.get("logical_unit")
            and str(value.get("logical_unit") or "").endswith(
                f"-{expected_logical_suffix}"
            )
            and value.get("input_sha256") == input_sha
            and sha256_json(batch_identity) == input_sha
            and value.get("source_sha256") == receipt.get("group_sha256")
            and isinstance(value.get("language"), str)
            and bool(value.get("language"))
            and value.get("prompt_sha256") == receipt.get("prompt_sha256")
            and value.get("schema_sha256") == receipt.get("schema_sha256")
            and bool(expected_ids)
            and isinstance(response, Mapping)
            and set(response) == {"entries"}
            and all(
                isinstance(item, Mapping)
                and set(item) == {"entry_id", "target", "explanation"}
                and isinstance(item.get("entry_id"), str)
                and isinstance(item.get("target"), str)
                and isinstance(item.get("explanation"), str)
                for item in response.get("entries") or []
            )
            and [
                str(item.get("entry_id") or "")
                for item in response.get("entries") or []
                if isinstance(item, Mapping)
            ] == expected_ids
            and len(response.get("entries") or []) == len(expected_ids)
            and _recovery_candidate_matches_schema(response, receipt)
        )
    if unit == "title-translation":
        response = value.get("response")
        return (
            value.get("schema_version")
            == "arc.companion.title-translation-chunk.v1"
            and value.get("logical_unit") == receipt.get("logical_unit")
            and value.get("input_sha256") == input_sha
            and isinstance(value.get("title_ids"), list)
            and isinstance(response, Mapping)
            and [
                str(item.get("title_id") or "")
                for item in response.get("titles") or []
                if isinstance(item, Mapping)
            ] == [str(item) for item in value.get("title_ids") or []]
            and _recovery_candidate_matches_schema(response, receipt)
        )
    if unit == "section-review":
        return (
            value.get("schema_version") == SECTION_REVIEW_CHECKPOINT_VERSION
            and value.get("input_sha256") == input_sha
            and isinstance(value.get("reviewed_segment_ids"), list)
            and isinstance(value.get("review"), Mapping)
            and isinstance(value["review"].get("findings"), list)
            and isinstance(value["review"].get("patches"), list)
            and _recovery_candidate_matches_schema(value["review"], receipt)
        )
    if unit == "commentary-review":
        return (
            value.get("schema_version") == COMMENTARY_REVIEW_CHECKPOINT_VERSION
            and value.get("input_sha256") == input_sha
            and isinstance(value.get("reviewed_segment_ids"), list)
            and isinstance(value.get("review"), Mapping)
            and isinstance(value["review"].get("issues"), list)
            and isinstance(value["review"].get("patches"), list)
            and _recovery_candidate_matches_schema(value["review"], receipt)
        )
    if unit == "final-review":
        return (
            isinstance(value.get("section_findings"), list)
            and isinstance(value.get("issues"), list)
            and isinstance(value.get("reviewed_segment_ids"), list)
        )
    if unit == "guide":
        return (
            chapter_guide_artifact_valid(value)
            and value.get("chapter_id")
            == str(receipt.get("logical_unit") or "").split(":", 1)[0]
        )
    return False


def _pipeline_acceptance_checkpoint_digest(
    unit: str,
    path: Path,
    receipt: Mapping[str, Any],
    *,
    checkpoint_dir: Path,
    require_business_validation: bool,
) -> str | None:
    root = checkpoint_dir.resolve(strict=False)
    lexical = Path(os.path.abspath(os.fspath(path.expanduser())))
    try:
        relative = lexical.relative_to(root)
        raw = read_bounded_file(
            root,
            relative,
            max_bytes=_MAX_RECOVERY_CONTROL_BYTES,
            suffixes=(".json",),
        )
        value = json.loads(raw)
    except (
        OSError, UnicodeError, ValueError, json.JSONDecodeError,
        SecureReadError,
    ):
        return None
    if not isinstance(value, Mapping):
        return None
    if require_business_validation and not _pipeline_acceptance_checkpoint_valid(
        unit,
        lexical,
        receipt,
        checkpoint_dir=root,
        _validated_value=value,
        _validated_raw=raw,
    ):
        return None
    return hashlib.sha256(raw).hexdigest()


def _translation_repair_acceptance_valid(
    unit: str,
    checkpoint: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    checkpoint_dir: Path,
    acceptance_path: Path,
    acceptance_raw: bytes | None = None,
) -> bool:
    logical_unit = str(receipt.get("logical_unit") or "")
    suffix = ":token-repair" if unit == "translation-token-repair" else ":coverage-repair"
    if not logical_unit.endswith(suffix):
        return False
    segment_id = logical_unit[: -len(suffix)]
    generation = int(receipt.get("generation") or 0)
    if (
        checkpoint.get("schema_version") != "arc.companion.translation-checkpoint.v2"
        or checkpoint.get("segment_id") != segment_id
        or checkpoint.get("generation") != generation
        or checkpoint.get("input_sha256") != receipt.get("input_sha256")
        or not isinstance(checkpoint.get("translation"), Mapping)
    ):
        return False
    if unit == "translation-token-repair":
        marker_path = _translation_token_attempt_path(
            checkpoint_dir, segment_id, generation,
        )
        expected_prompt = TRANSLATION_RETRY_PROMPT_VERSION
        expected_schema = TRANSLATION_SLOT_REPAIR_SCHEMA_VERSION
        expected_kind = "token-placement"
        expected_ids_field = "block_ids"
    else:
        marker_path = _translation_coverage_attempt_path(
            checkpoint_dir, segment_id, generation,
        )
        expected_prompt = TRANSLATION_COVERAGE_REPAIR_PROMPT_VERSION
        expected_schema = TRANSLATION_COVERAGE_REPAIR_SCHEMA_VERSION
        expected_kind = "coverage"
        expected_ids_field = "missing_block_ids"
    try:
        marker = _read_recovery_json(checkpoint_dir, marker_path)
    except SecureReadError:
        return False
    if (
        not isinstance(marker, Mapping)
        or marker.get("status") != "validated"
        or marker.get("segment_id") != segment_id
        or int(marker.get("generation") or 0) != generation
        or marker.get("input_sha256") != receipt.get("input_sha256")
        or marker.get("prompt_version") != expected_prompt
        or marker.get("response_schema_version") != expected_schema
        or not isinstance(marker.get("raw_response"), Mapping)
        or marker.get("validated_translation_sha256")
        != sha256_json(checkpoint["translation"])
    ):
        return False
    final_reference = marker.get("final_translation_checkpoint")
    if not isinstance(final_reference, Mapping):
        return False
    try:
        final_path = resolve_recovery_path(
            checkpoint_dir, final_reference.get("path"),
        )
        root = checkpoint_dir.resolve(strict=False)
        lexical_acceptance = Path(os.path.abspath(os.fspath(
            acceptance_path.expanduser()
        )))
        if acceptance_raw is None:
            acceptance_raw = read_bounded_file(
                root,
                lexical_acceptance.relative_to(root),
                max_bytes=_MAX_RECOVERY_CONTROL_BYTES,
                suffixes=(".json",),
            )
    except (RecoveryResponseError, SecureReadError, OSError, ValueError):
        return False
    if (
        final_path != lexical_acceptance
        or hashlib.sha256(acceptance_raw).hexdigest()
        != final_reference.get("sha256")
    ):
        return False
    repairs = (checkpoint.get("generation_provenance") or {}).get("repairs")
    repair = next((
        item for item in repairs or []
        if isinstance(item, Mapping) and item.get("kind") == expected_kind
    ), None)
    expected_ids = [str(item) for item in marker.get(expected_ids_field) or []]
    if (
        not isinstance(repair, Mapping)
        or not expected_ids
        or [str(item) for item in repair.get("repaired_block_ids") or []]
        != expected_ids
        or repair.get("response_normalization") != marker.get("response_normalization")
    ):
        return False
    normalization = marker.get("response_normalization")
    if not isinstance(normalization, Mapping):
        return False
    try:
        normalization_path = resolve_recovery_path(
            checkpoint_dir, normalization.get("path"),
        )
        root = checkpoint_dir.resolve(strict=False)
        raw = read_bounded_file(
            root,
            normalization_path.relative_to(root),
            max_bytes=_MAX_RECOVERY_CONTROL_BYTES,
            suffixes=(".json",),
        )
        normalized_receipt = json.loads(raw)
    except (
        RecoveryResponseError, SecureReadError, OSError, UnicodeError,
        ValueError, json.JSONDecodeError,
    ):
        return False
    return bool(
        hashlib.sha256(raw).hexdigest() == normalization.get("sha256")
        and isinstance(normalized_receipt, Mapping)
        and normalized_receipt.get("schema_version")
        == "arc.companion.response-normalization-receipt.v1"
        and normalized_receipt.get("decision") == "accepted"
        and normalized_receipt.get("validator_version")
        == TRANSLATION_REPAIR_NORMALIZATION_VALIDATOR_VERSION
        and normalized_receipt.get("expected_ids") == expected_ids
        and normalized_receipt.get("original_response_sha256")
        == sha256_json(marker["raw_response"])
    )


def _recovery_candidate_matches_schema(
    candidate: Mapping[str, Any], receipt: Mapping[str, Any],
) -> bool:
    schema = receipt.get("schema")
    if not isinstance(schema, Mapping):
        return False
    try:
        from jsonschema.validators import validator_for

        validator_type = validator_for(schema)
        validator_type.check_schema(schema)
        return not any(validator_type(schema).iter_errors(dict(candidate)))
    except (TypeError, ValueError):
        return False


def _llm_call(
    llm: Callable[..., dict[str, Any]],
    prompt: str,
    schema: dict[str, Any],
    *,
    options: BuildOptions,
    artifact_dir: Path,
    call_label: str,
    model_tier: str,
    allow_internet: bool = False,
    force_offline: bool = False,
    paper_access_policy: Mapping[str, Any] | None = None,
    disable_paper_cli: bool = False,
    intent_guidance: Mapping[str, Any] | None = None,
    intent_guidance_lane: str | None = None,
    review_prompt_context: Mapping[str, Any] | None = None,
    recovery_descriptor: Mapping[str, Any] | None = None,
    evidence_journal_context: EvidenceJournalContext | None = None,
    evidence_target_generation: int | None = None,
) -> dict[str, Any]:
    force_offline = force_offline or not allow_internet
    structured_policy = bool(
        paper_access_policy is not None
        and _accepts_explicit_keyword(llm, "paper_access_policy")
    )
    runtime_env = _llm_runtime_env(
        allow_internet=not force_offline and allow_internet and options.allow_internet,
        force_disable_internet=force_offline or not options.allow_internet,
        inherit_host_tools=(
            options.inherit_host_tools
            if paper_access_policy is None and not disable_paper_cli else False
        ),
        paper_access_policy=paper_access_policy,
        serialize_paper_access_policy=not structured_policy,
        disable_paper_cli=disable_paper_cli,
    )
    active_prompt = prompt
    active_schema = _intent_guidance_schema(schema, intent_guidance)
    control: dict[str, Any] | None = None
    if recovery_descriptor is not None and not bool(
        getattr(llm, "_arc_owns_recovery_receipt", False)
    ):
        control = _prepare_pipeline_recovery_control(
            recovery_descriptor,
            artifact_dir=artifact_dir,
        )
    if intent_guidance is not None and evidence_journal_context is None and control is not None:
        checkpoint_root = Path(control["checkpoint_dir"])
        build_fingerprint = _authoritative_build_fingerprint(checkpoint_root)
        control_ledger, _control_digest = read_registered_lane_ledger(
            checkpoint_root, Path(control["ledger_path"]),
        )
        evidence_source_generation, evidence_source_logical_task = (
            _companion_evidence_source_identity(
                project_dir=options.project_dir,
                checkpoint_dir=checkpoint_root,
                run_id=build_fingerprint,
                ledger_path=Path(control["ledger_path"]),
                ledger=control_ledger,
                session_key=str(control["session_key"]),
                logical_unit=str(control["logical_unit"]),
                target_generation=int(control["generation"]),
                target_idempotency_key=str(control["idempotency_key"]),
            )
        )
        evidence_journal_context = _companion_evidence_journal_context(
            checkpoint_dir=checkpoint_root,
            run_id=build_fingerprint,
            lane=str(intent_guidance_lane or "general"),
            worker_id=call_label,
            logical_task_id=evidence_source_logical_task,
            source_generation=evidence_source_generation,
            policy=paper_access_policy,
            runtime={
                "provider": options.provider,
                "model": options.model,
                "model_tier": model_tier,
                "environment_hash": sha256_json(runtime_env),
            },
        )
    for round_number in range(0, 4):
        active_label = (
            call_label if round_number == 0
            else f"{call_label}-evidence-{round_number:02d}"
        )
        if review_prompt_context is not None:
            strict_limit = int(review_prompt_context["strict_limit_bytes"])
            _require_review_prompt_within_limit(
                active_prompt,
                label=active_label,
                max_prompt_bytes=strict_limit,
            )
            if round_number:
                audit_sink = review_prompt_context.get("audit_sink")
                if isinstance(audit_sink, list):
                    audit_sink.append(_review_evidence_round_audit(
                        active_prompt,
                        context=review_prompt_context,
                        call_label=active_label,
                    ))
        call_kwargs: dict[str, Any] = {}
        if structured_policy:
            call_kwargs["paper_access_policy"] = paper_access_policy
        base_artifact_dir = (
            artifact_dir if round_number == 0
            else artifact_dir / f"evidence-round-{round_number:02d}"
        )
        active_artifact_dir = base_artifact_dir
        idempotency_key = None
        submission_receipt_path: Path | None = None
        latest_progress: dict[str, Any] | None = None
        if control is not None:
            idempotency_key = str(control["idempotency_key"])
            if round_number:
                idempotency_key += f":evidence-{round_number:02d}"
            try:
                submission_receipt_path = write_ledger_submission_receipt(
                    checkpoint_dir=Path(control["checkpoint_dir"]),
                    artifact_dir=active_artifact_dir,
                    ledger_path=Path(control["ledger_path"]),
                    session_key=str(control["session_key"]),
                    logical_unit=str(control["logical_unit"]),
                    generation=int(control["generation"]),
                    idempotency_key=idempotency_key,
                    schema=active_schema,
                    prompt=active_prompt,
                    recovery_unit=str(control["unit"]),
                    input_sha256=str(control["input_sha256"]),
                    group_sha256=str(control["group_sha256"]),
                    ordered_siblings=list(control["ordered_siblings"]),
                    suffix=list(control["suffix"]),
                    validator=str(control["validator"]),
                    application=str(control["application"]),
                    acceptance_checkpoint=Path(control["acceptance_checkpoint"]),
                    stateful_checkpoint_identity=False,
                )
            except RecoveryResponseError as exc:
                raise RecoveryResponseError(
                    "pipeline recovery receipt rejected for "
                    f"{control['unit']}:{control['logical_unit']}:"
                    f"generation-{control['generation']}: {exc}"
                ) from exc

        def recovery_progress(event: Mapping[str, Any]) -> None:
            nonlocal latest_progress
            if control is None:
                return
            latest_progress = {
                **dict(event),
                "session_key": control["session_key"],
                "generation": control["generation"],
                "logical_unit": control["logical_unit"],
            }
            if event.get("event") == "submitted":
                with _RECOVERY_CONTROL_INITIALIZE_LOCK:
                    _guarded_mark_transport_state(
                        Path(control["ledger_path"]),
                        checkpoint_dir=Path(control["checkpoint_dir"]),
                        session_key=str(control["session_key"]),
                        logical_unit=str(control["logical_unit"]),
                        idempotency_key=str(control["idempotency_key"]),
                    )

        try:
            recovery_call_kwargs: dict[str, Any] = {}
            if control is not None:
                recovery_call_kwargs.update({
                    "idempotency_key": idempotency_key,
                    "progress_callback": recovery_progress,
                })
            value = llm(
                active_prompt,
                schema=active_schema,
                provider=options.provider,
                model=options.model,
                model_tier=None if options.model else model_tier,
                env=runtime_env,
                session_policy="stateless",
                artifact_dir=active_artifact_dir,
                call_label=active_label,
                idle_timeout_seconds=options.idle_timeout_seconds,
                **call_kwargs,
                **recovery_call_kwargs,
            )
        except BaseException as exc:
            if (
                control is not None
                and not isinstance(exc, CompanionLLMCircuitOpen)
                and _chapter_failure_requires_supervision(exc)
            ):
                from arc_llm import read_recovery_context

                recovery = read_recovery_context(
                    active_artifact_dir,
                    idempotency_key=str(idempotency_key),
                )
                recovery_context = _recovery_context_json(
                    recovery, logical_unit=str(control["logical_unit"]),
                )
                recovery_context.update({
                    "session_key": control["session_key"],
                    "generation": control["generation"],
                    "recovery_unit": control["unit"],
                    "stateless_control": True,
                    "latest_progress": latest_progress,
                })
                if submission_receipt_path is not None:
                    try:
                        seal_submission_attempts(
                            submission_receipt_path,
                            checkpoint_dir=Path(control["checkpoint_dir"]),
                            attempt_references=explicit_attempt_references(
                                exc,
                                checkpoint_dir=Path(control["checkpoint_dir"]),
                                artifact_dir=active_artifact_dir,
                            ),
                        )
                        recovery_context["submission_receipt"] = (
                            submission_receipt_reference(
                                submission_receipt_path,
                                checkpoint_dir=Path(control["checkpoint_dir"]),
                            )
                        )
                    except RecoveryResponseError as receipt_exc:
                        recovery_context["submission_receipt_error"] = str(receipt_exc)
                with _RECOVERY_CONTROL_INITIALIZE_LOCK:
                    mark_needs_supervision(
                        Path(control["ledger_path"]),
                        segment_id=str(control["logical_unit"]),
                        reason=str(exc),
                        recovery_context=recovery_context,
                    )
            raise
        if (
            control is not None
            and submission_receipt_path is not None
            and (active_artifact_dir / "call-checkpoints").is_dir()
        ):
            seal_submission_attempts(
                submission_receipt_path,
                checkpoint_dir=Path(control["checkpoint_dir"]),
                attempt_references=explicit_attempt_references(
                    value,
                    checkpoint_dir=Path(control["checkpoint_dir"]),
                    artifact_dir=active_artifact_dir,
                ),
            )
        if intent_guidance is None:
            if control is not None:
                with _RECOVERY_CONTROL_INITIALIZE_LOCK:
                    _guarded_mark_transport_state(
                        Path(control["ledger_path"]),
                        checkpoint_dir=Path(control["checkpoint_dir"]),
                        session_key=str(control["session_key"]),
                        logical_unit=str(control["logical_unit"]),
                        idempotency_key=str(control["idempotency_key"]),
                        response_received=True,
                    )
            from arc_llm import ARC_LLM_CALL_RECORD_FIELD, strip_arc_llm_call_records

            return (
                strip_arc_llm_call_records(value)
                if ARC_LLM_CALL_RECORD_FIELD in value else value
            )
        from arc_llm import (
            EvidenceJournal,
            EvidenceOperationPolicy,
            evidence_requests_from_output,
            resolve_evidence_round,
        )

        requests = evidence_requests_from_output(
            value, worker_id=call_label, role="companion-content-worker",
        )
        if not requests:
            if control is not None:
                with _RECOVERY_CONTROL_INITIALIZE_LOCK:
                    _guarded_mark_transport_state(
                        Path(control["ledger_path"]),
                        checkpoint_dir=Path(control["checkpoint_dir"]),
                        session_key=str(control["session_key"]),
                        logical_unit=str(control["logical_unit"]),
                        idempotency_key=str(control["idempotency_key"]),
                        response_received=True,
                    )
            from arc_llm import strip_arc_llm_call_records

            return strip_arc_llm_call_records({
                key: item for key, item in value.items()
                if key != "arc_evidence_requests"
            })
        if round_number >= 3:
            raise RuntimeError("companion reference evidence exceeded three controller rounds")
        resolver = lambda material, *, round_number: resolve_worker_evidence_requests(
                intent_guidance, material, round_number=round_number,
                lane=intent_guidance_lane,
            )
        journal = (
            EvidenceJournal(evidence_journal_context.journal_root)
            if evidence_journal_context is not None else None
        )
        def journaled_resolver(material, *, round_number):
            if journal is None:
                return resolver(material, round_number=round_number)
            return journal.resolve_round(
                evidence_journal_context,
                material,
                resolver,
                round_number=round_number,
                operation_policies={
                    request.operation: EvidenceOperationPolicy(idempotent=True)
                    for request in material
                },
            )
        if review_prompt_context is None:
            responses = resolve_evidence_round(
                requests, journaled_resolver, round_number=round_number + 1,
            )
            delivered_requests = requests
        else:
            delivered_requests = _bounded_review_evidence_requests(
                requests,
                active_prompt=active_prompt,
                strict_prompt_bytes=int(
                    review_prompt_context["strict_limit_bytes"]
                ),
            )
            responses = resolve_evidence_round(
                delivered_requests,
                journaled_resolver,
                round_number=round_number + 1,
            )
        followup = _controller_evidence_prompt(responses)
        if (
            review_prompt_context is not None
            and _utf8_size(active_prompt + "\n\n" + followup)
            > int(review_prompt_context["strict_limit_bytes"])
        ):
            raise RuntimeError(
                "review evidence response cannot fit under the strict prompt limit"
            )
        if journal is not None:
            target_followup_base = (
                str(control["idempotency_key"])
                if control is not None
                else evidence_journal_context.logical_task_id
            )
            journal.mark_delivered(
                evidence_journal_context,
                delivered_requests,
                round_number=round_number + 1,
                target_generation=(
                    int(control["generation"])
                    if control is not None
                    else evidence_target_generation
                    or evidence_journal_context.source_generation
                ),
                target_session=(
                    str(control["session_key"])
                    if control is not None else evidence_journal_context.worker_id
                ),
                followup_id=(
                    f"{target_followup_base}:evidence-{round_number + 1:02d}"
                ),
                operation_policies={
                    request.operation: EvidenceOperationPolicy(idempotent=True)
                    for request in delivered_requests
                },
            )
        active_prompt = active_prompt + "\n\n" + followup
    raise AssertionError("unreachable evidence loop")


def _bounded_review_evidence_followup(
    requests: Sequence[Any],
    *,
    resolver: Callable[..., Any],
    round_number: int,
    active_prompt: str,
    strict_prompt_bytes: int,
) -> tuple[tuple[Any, ...], str]:
    """Fit controller evidence into the remaining strict review prompt budget."""
    from arc_llm import resolve_evidence_round

    bounded_requests = _bounded_review_evidence_requests(
        requests,
        active_prompt=active_prompt,
        strict_prompt_bytes=strict_prompt_bytes,
    )
    responses = resolve_evidence_round(
        bounded_requests, resolver, round_number=round_number,
    )
    followup = _controller_evidence_prompt(responses)
    if _utf8_size(active_prompt + "\n\n" + followup) > strict_prompt_bytes:
        raise RuntimeError(
            "review evidence response cannot fit under the strict prompt limit"
        )
    return responses, followup


def _bounded_review_evidence_requests(
    requests: Sequence[Any],
    *,
    active_prompt: str,
    strict_prompt_bytes: int,
) -> tuple[Any, ...]:
    """Choose one conservative page limit before the addressed request executes."""

    empty_followup = _controller_evidence_prompt(())
    remaining = strict_prompt_bytes - _utf8_size(active_prompt + "\n\n" + empty_followup)
    if remaining < 1:
        raise RuntimeError(
            "review evidence response has no remaining space under the strict prompt limit"
        )
    page_limit = max(1, remaining // max(1, len(requests)) - 512)
    bounded_requests = []
    for request in requests:
        arguments = dict(getattr(request, "arguments", {}) or {})
        key = (
            "limit_bytes"
            if str(getattr(request, "operation", "")) == "list-reference-targets"
            else "limit"
        )
        try:
            requested = int(arguments.get(key, page_limit))
        except (TypeError, ValueError):
            requested = page_limit
        arguments[key] = max(1, min(requested, page_limit))
        bounded_requests.append(replace(request, arguments=arguments))
    return tuple(bounded_requests)


def _review_evidence_round_audit(
    prompt: str,
    *,
    context: Mapping[str, Any],
    call_label: str,
) -> dict[str, Any]:
    prompt_bytes = _utf8_size(prompt)
    target_bytes = int(context["target_limit_bytes"])
    strict_bytes = int(context["strict_limit_bytes"])
    return {
        "stage": f"{context['stage']}-evidence",
        "call_label": call_label,
        "segment_ids": list(context.get("segment_ids") or []),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_bytes": prompt_bytes,
        "target_headroom_bytes": max(0, target_bytes - prompt_bytes),
        "target_overage_bytes": max(0, prompt_bytes - target_bytes),
        "strict_headroom_bytes": strict_bytes - prompt_bytes,
        "budget_class": (
            "normal" if prompt_bytes <= target_bytes else "evidence_headroom"
        ),
        "disposition": "provider-call",
    }


def _limit_llm_concurrency(
    llm: Callable[..., dict[str, Any]], max_concurrent_calls: int,
    *,
    cancel_check: Callable[[], bool] | None = None,
) -> Callable[..., dict[str, Any]]:
    """Share one call budget without cancelling unrelated submitted calls.

    A local/preflight failure opens the submission barrier for queued work only.
    The provider-facing cancellation callback remains tied exclusively to the
    caller's explicit cancellation signal, so already submitted calls drain.
    """
    external_cancel_check = cancel_check
    permits = threading.BoundedSemaphore(max_concurrent_calls)
    tripped = threading.Event()
    state_lock = threading.Lock()
    abort_reason: BaseException | None = None

    def raise_if_tripped() -> None:
        if external_cancel_check is not None and external_cancel_check():
            tripped.set()
        if not tripped.is_set():
            return
        with state_lock:
            reason = abort_reason
        message = "companion LLM circuit is open after another call failed or cancellation was requested"
        if reason is not None and str(reason):
            message += f": {reason}"
        raise CompanionLLMCircuitOpen(message) from reason

    @wraps(llm)
    def limited(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal abort_reason
        raise_if_tripped()
        with permits:
            raise_if_tripped()
            call_kwargs = kwargs
            if _accepts_explicit_keyword(llm, "cancel_check"):
                parent_cancel_check = kwargs.get("cancel_check")

                def cancel_check() -> bool:
                    return bool(
                        callable(parent_cancel_check) and parent_cancel_check()
                    ) or bool(external_cancel_check is not None and external_cancel_check())

                call_kwargs = {**kwargs, "cancel_check": cancel_check}
            try:
                return llm(*args, **call_kwargs)
            except BaseException as exc:
                with state_lock:
                    if abort_reason is None:
                        abort_reason = exc
                tripped.set()
                raise

    return limited


def _accepts_explicit_keyword(call: Callable[..., Any], name: str) -> bool:
    """Return true only for a named keyword, not a permissive fake's **kwargs."""
    try:
        parameter = inspect.signature(call).parameters.get(name)
    except (TypeError, ValueError):
        return False
    return parameter is not None and parameter.kind in {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }


def _generation_runtime_policy(options: BuildOptions | None = None) -> dict[str, bool | str]:
    allow_internet = True if options is None else options.allow_internet
    return {
        "allow_mcp": False,
        "allow_internet": allow_internet,
        "arc_paper_cli_access": "full",
        "inherit_host_tools": False if options is None else options.inherit_host_tools,
    }


def _intent_guidance_identity(
    artifact: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if artifact is None:
        return None
    references = []
    targets = artifact.get("reference_targets") or []
    for source in artifact.get("reference_sources") or []:
        if not isinstance(source, Mapping):
            continue
        source_id = str(source.get("source_id") or "")
        references.append({
            "source_id": source_id,
            "document_hash": str(
                source.get("document_hash") or source.get("source_hash") or ""
            ),
            "locators": sorted({
                str(target.get("locator") or "")
                for target in targets
                if isinstance(target, Mapping)
                and str(target.get("source_id") or "") == source_id
            }),
        })
    return {
        "user_intent_sha256": artifact.get("user_intent_sha256"),
        "semantic_input_sha256": artifact.get("semantic_input_sha256"),
        "output_sha256": artifact.get("output_sha256"),
        "references": references,
    }


def _guided_prompt(
    prompt: str,
    intent_guidance: Mapping[str, Any] | None,
    *,
    lane: str | None = None,
) -> str:
    if intent_guidance is None:
        return prompt
    return (
        worker_guidance_prompt_prefix(intent_guidance, lane=lane)
        + "\nIf this host has no sandboxed shell, request the same exact cached reads through "
        "arc_evidence_requests. Use list-reference-targets to inspect a non-inline target "
        "catalog, then get-parsed-toc or get-parsed-section with arguments source_id, "
        "locator, and optional byte offset/limit; return [] when no controller read is "
        "needed.\n"
        + prompt
    )


def _guidance_policy(
    intent_guidance: Mapping[str, Any] | None,
    *,
    lane: str | None = None,
) -> dict[str, Any] | None:
    return (
        None if intent_guidance is None
        else worker_policy_descriptor(intent_guidance, lane=lane)
    )


def _llm_runtime_env(
    *,
    allow_internet: bool,
    force_disable_internet: bool = False,
    inherit_host_tools: bool = False,
    paper_access_policy: Mapping[str, Any] | None = None,
    serialize_paper_access_policy: bool = True,
    disable_paper_cli: bool = False,
) -> dict[str, str] | None:
    """Map portable access intent onto both supported host runtimes."""
    if disable_paper_cli:
        inherit_host_tools = False
    env = dict(os.environ)
    if allow_internet or force_disable_internet:
        value = "true" if allow_internet else "false"
        env["ARC_CODEX_ALLOW_INTERNET"] = value
        env["ARC_CLAUDE_ALLOW_INTERNET"] = value
    env["ARC_PAPER_CLI_ACCESS"] = "none" if disable_paper_cli else "full"
    for key in (
        "ARC_PAPER_WORKER_ALLOWED_OPERATIONS_JSON",
        "ARC_PAPER_WORKER_ALLOWED_TARGETS_JSON",
        "ARC_PAPER_WORKER_READ_POLICY_PATH",
        "ARC_PAPER_WORKER_READ_POLICY_SHA256",
        "ARC_PAPER_WORKER_READ_POLICY_SCHEMA",
    ):
        env.pop(key, None)
    if (
        paper_access_policy is not None
        and not disable_paper_cli
        and serialize_paper_access_policy
    ):
        operations = list(
            paper_access_policy.get("operations")
            or paper_access_policy.get("allowed_operations")
            or []
        )
        targets = paper_access_policy.get("targets")
        if not isinstance(targets, Mapping):
            targets = {
                str(source_id): {"sections": []}
                for source_id in paper_access_policy.get("authorized_source_ids") or []
            }
            for target in paper_access_policy.get("authorized_section_targets") or []:
                if not isinstance(target, Mapping):
                    continue
                source_id = str(target.get("source_id") or "")
                locator = str(target.get("locator") or "")
                if source_id in targets and locator:
                    targets[source_id]["sections"].append(locator)
        env["ARC_PAPER_WORKER_ALLOWED_OPERATIONS_JSON"] = json.dumps(
            operations,
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
        env["ARC_PAPER_WORKER_ALLOWED_TARGETS_JSON"] = json.dumps(
            dict(targets),
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
    env["ARC_LLM_INHERIT_HOST_TOOLS"] = "true" if inherit_host_tools else "false"
    if not inherit_host_tools:
        for key in _MCP_CONFIG_ENV_KEYS:
            env.pop(key, None)
        env["ARC_CODEX_ENABLE_MCP"] = "false"
        env["ARC_CLAUDE_ALLOW_MCP"] = "false"
        env["ARC_CODEX_IGNORE_USER_CONFIG"] = "true"
        env["ARC_CLAUDE_BARE"] = "true"
        claude_tools = ["WebSearch", "WebFetch"] if allow_internet else []
        claude_allowed = list(claude_tools)
        if paper_access_policy is not None and not disable_paper_cli:
            claude_tools.insert(0, "Bash")
            claude_allowed = [
                "Bash(arc-paper-worker get-parsed-toc:*)",
                "Bash(arc-paper-worker get-parsed-section:*)",
                "Bash(arc-paper-worker policy-targets:*)",
                "Bash(arc-paper-worker artifact-read:*)",
                *claude_allowed,
            ]
        env["ARC_CLAUDE_TOOLS"] = ",".join(claude_tools)
        env["ARC_CLAUDE_ALLOWED_TOOLS"] = ",".join(claude_allowed)
    return env


def _fingerprint(
    bundle: SourceBundle,
    options: BuildOptions,
    *,
    evidence: dict[str, Any],
    domain_context: dict[str, Any] | None = None,
) -> str:
    return sha256_json(
        _fingerprint_payload(
            bundle,
            options,
            evidence=evidence,
            domain_context=domain_context,
        )
    )


def _legacy_migration_source_hash(bundle: SourceBundle) -> str:
    """Return the strongest stable rich-source identity available to migration."""
    integrity = bundle.document.get("integrity") or {}
    return str(
        bundle.parsed.get("source_hash")
        or bundle.document.get("source_hash")
        or integrity.get("document_hash")
        or sha256_json(_generation_document(bundle.document))
    )


def _store_validated_stateless_artifact(
    store: AcceptedArtifactStore,
    *,
    kind: str,
    semantic_input_sha256: str,
    recipe_sha256: str,
    contract_version: str,
    output: Any,
    segment_id: str,
    checkpoint_dir: Path,
    provider: str,
    model: str | None,
) -> dict[str, Any]:
    """Persist locally validated stateless output with explicit unavailable usage."""

    predecessor = hashlib.sha256(b"").hexdigest()
    output_sha = sha256_json(output)
    generation = 1
    chain = hashlib.sha256(json.dumps({
        "predecessor": predecessor, "segment_id": segment_id,
        "input_sha256": semantic_input_sha256, "output_sha256": output_sha,
        "generation": generation,
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    block = {
        "segment_id": segment_id, "state": "accepted", "generation": generation,
        "input_sha256": semantic_input_sha256, "output_sha256": output_sha,
        "predecessor_accepted_chain_sha256": predecessor,
        "accepted_chain_sha256": chain,
        "logical_receipt": {
            "kind": "validated_stateless_checkpoint", "checkpoint_state": "validated",
        },
        "validation_receipt": {"local_validation": True},
    }
    return store.put_accepted(
        kind=kind, semantic_input_sha256=semantic_input_sha256,
        recipe_sha256=recipe_sha256, contract_version=contract_version,
        output=output, ledger_block=block,
        provider_receipt={
            "provider": provider, "model": model or "provider-default",
            "call_id": f"stateless:{kind}:{segment_id}:{output_sha}",
            "usage": {"availability": "not_exposed_by_stateless_host_adapter"},
        },
        provenance={"checkpoint_dir": str(checkpoint_dir), "segment_id": segment_id},
    )


def _stable_recovery_chunk_id(
    unit: str,
    member_ids: Sequence[str],
    *,
    payload: Any,
) -> str:
    """Address an independently submitted chunk by its exact stable content."""

    digest = sha256_json({
        "unit": unit,
        "member_ids": [str(value) for value in member_ids],
        "payload": payload,
    })[:16]
    return f"{unit}-{digest}"


def _generate_title_translations(
    *,
    options: BuildOptions,
    bundle: SourceBundle,
    document: dict[str, Any],
    chapters: list[dict[str, Any]],
    glossary: dict[str, Any],
    protected_names: list[str],
    checkpoint_dir: Path,
    call_model: Callable[[str, dict[str, Any], Path, str], dict[str, Any]],
    artifact_store: AcceptedArtifactStore | None = None,
    intent_guidance_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Translate every visible structural title without entering segment lanes."""
    if options.skip_translation:
        return None
    projection_document = {**document, "metadata": dict(bundle.metadata or {})}
    records = collect_title_records(projection_document, chapters)
    source_language = str(options.source_language or "und")
    source_sha256 = sha256_json(records)
    title_glossary = _title_glossary_projection(records, glossary)
    semantic_input_sha256 = lane_semantic_sha256("title_translation", {
        "source_titles": records,
        "source_language": source_language,
        "target_language": options.annotation_language,
        "glossary": title_glossary,
        "protected_names": protected_names,
        **(
            {"intent_guidance": dict(intent_guidance_identity)}
            if intent_guidance_identity else {}
        ),
    })
    recipe_sha256 = lane_recipe_sha256(
        "title_translation",
        prompt=TITLE_TRANSLATION_PROMPT_VERSION,
        model=options.model,
        tier=TITLE_TRANSLATION_TIER,
        access_recipe={"provider": options.provider, "allow_internet": False},
    )
    store = artifact_store or AcceptedArtifactStore(options.project_dir.resolve())

    def valid(value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        if (
            value.get("schema_version") != TITLE_TRANSLATION_VERSION
            or value.get("source_sha256") != source_sha256
            or value.get("source_language") != source_language
            or value.get("target_language") != options.annotation_language
        ):
            return False
        try:
            merge_title_translation_chunks(
                records, [value], protected_names=protected_names,
            )
        except (RuntimeError, TypeError, ValueError):
            return False
        return True

    accepted = None
    if "translation" not in options.regenerate_lanes:
        accepted = store.find(
            kind="title_translation",
            semantic_input_sha256=semantic_input_sha256,
            recipe_sha256=recipe_sha256,
            contract_version=TITLE_TRANSLATION_VERSION,
            predecessor_accepted_chain_sha256=hashlib.sha256(b"").hexdigest(),
            output_validator=valid,
        )
    if accepted is not None:
        output = dict(accepted["output"])
        write_json(checkpoint_dir / "title-translations.json", output)
        return output

    responses: list[dict[str, Any]] = []
    chunks = list(chunk_title_records(records))
    title_units = [
        _stable_recovery_chunk_id("title-translation", [
            str(item.get("title_id") or "") for item in chunk
        ], payload=chunk)
        for chunk in chunks
    ]
    for index, chunk in enumerate(chunks, 1):
        prompt = title_translation_prompt(
            chunk,
            source_language=source_language,
            target_language=options.annotation_language,
            glossary=title_glossary,
            protected_names=protected_names,
        )
        logical_unit = title_units[index - 1]
        artifact_dir = checkpoint_dir / "title-translation" / logical_unit
        chunk_input_sha256 = sha256_json({
            "records": chunk,
            "prompt": prompt,
            "schema": TITLE_TRANSLATION_SCHEMA,
        })
        chunk_checkpoint = (
            checkpoint_dir / "title-translation-chunks" / f"{logical_unit}.json"
        )
        response: dict[str, Any] | None = None
        if chunk_checkpoint.is_file() and "translation" not in options.regenerate_lanes:
            candidate = _read_checkpoint_json(
                chunk_checkpoint, root=checkpoint_dir,
            )
            if (
                isinstance(candidate, Mapping)
                and candidate.get("schema_version")
                == "arc.companion.title-translation-chunk.v1"
                and candidate.get("logical_unit") == logical_unit
                and candidate.get("input_sha256") == chunk_input_sha256
                and isinstance(candidate.get("response"), Mapping)
            ):
                try:
                    response = validate_title_translations(
                        chunk,
                        candidate["response"],
                        protected_names=protected_names,
                    )
                except (RuntimeError, TypeError, ValueError):
                    response = None
        descriptor = submission_descriptor(
                unit="title-translation",
                logical_unit=logical_unit,
                checkpoint_dir=checkpoint_dir,
                artifact_root=artifact_dir,
                acceptance_checkpoint=chunk_checkpoint,
                input_sha256=chunk_input_sha256,
                group_sha256=semantic_input_sha256,
                ordered_siblings=title_units,
                suffix=title_units[index - 1:],
            )
        submitted = response is None
        if submitted:
            raw_response = call_model_with_recovery_descriptor(
                call_model,
                prompt,
                TITLE_TRANSLATION_SCHEMA,
                artifact_dir,
                logical_unit,
                descriptor,
            )
            response = validate_title_translations(
                chunk, raw_response, protected_names=protected_names,
            )
            write_json(chunk_checkpoint, {
                "schema_version": "arc.companion.title-translation-chunk.v1",
                "logical_unit": logical_unit,
                "input_sha256": chunk_input_sha256,
                "title_ids": [str(item.get("title_id") or "") for item in chunk],
                "response": response,
            })
        if submitted:
            _accept_registered_pipeline_control(
                checkpoint_dir, "title-translation", logical_unit,
            )
        else:
            _accept_completed_pipeline_controls(
                checkpoint_dir,
                only_logical_unit=logical_unit,
                only_recovery_unit="title-translation",
            )
        responses.append(response)
    merged = merge_title_translation_chunks(
        records, responses, protected_names=protected_names,
    )
    translated_by_id = {
        str(item["title_id"]): str(item["text"])
        for item in merged["titles"]
    }
    output = {
        "schema_version": TITLE_TRANSLATION_VERSION,
        "source_sha256": source_sha256,
        "source_language": source_language,
        "target_language": options.annotation_language,
        "titles": [
            {
                "title_id": str(record["title_id"]),
                "role": str(record.get("role") or "heading"),
                "block_id": record.get("block_id"),
                "chapter_id": record.get("chapter_id"),
                "source_block_ids": list(record.get("source_block_ids") or []),
                "text": translated_by_id[str(record["title_id"])],
            }
            for record in records
        ],
    }
    if not valid(output):
        raise RuntimeError("generated title translation artifact failed local validation")
    _store_validated_stateless_artifact(
        store,
        kind="title_translation",
        semantic_input_sha256=semantic_input_sha256,
        recipe_sha256=recipe_sha256,
        contract_version=TITLE_TRANSLATION_VERSION,
        output=output,
        segment_id="project:title-translation",
        checkpoint_dir=checkpoint_dir,
        provider=options.provider,
        model=options.model,
    )
    write_json(checkpoint_dir / "title-translations.json", output)
    return output


def _title_glossary_projection(
    records: list[dict[str, Any]], glossary: dict[str, Any],
) -> dict[str, Any]:
    source_text = "\n".join(str(item.get("source_text") or "") for item in records)
    entries = []
    for entry in glossary.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        terms = [
            str(entry.get("source_term") or entry.get("source") or ""),
            *(str(value) for value in entry.get("aliases") or []),
        ]
        if any(term and term.casefold() in source_text.casefold() for term in terms):
            entries.append(dict(entry))
    return {"entries": entries}


def _legacy_metadata_view(legacy: dict[str, Any]) -> dict[str, Any]:
    metadata = (
        dict(legacy.get("metadata") or {})
        if isinstance(legacy.get("metadata"), dict) else {}
    )
    for key in ("source_hash", "language", "prompt_hash", "validator_hash"):
        if legacy.get(key) not in (None, ""):
            metadata[key] = legacy[key]
    return metadata


def _fingerprint_payload(
    bundle: SourceBundle,
    options: BuildOptions,
    *,
    evidence: dict[str, Any],
    domain_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    integrity = bundle.document.get("integrity") or {}
    # This identity owns only stable source partition storage. Generation
    # recipes, evidence, review, render, and runtime policy have lane-local keys.
    return {
        "paper_id": bundle.paper_id,
        "document_hash": (
            integrity.get("document_hash")
            or bundle.document.get("document_hash")
            or bundle.parsed.get("document_hash")
            or sha256_json(bundle.document)
        ),
        "rich_parser_version": bundle.document.get("parser_version"),
        "generation_projection_hash": sha256_json(_generation_document(bundle.document)),
        "chapter_projection_version": CHAPTER_PROJECTION_VERSION,
        "augmentation_projection_version": AUGMENTATION_PROJECTION_VERSION,
        "asset_manifest_hash": (
            integrity.get("asset_manifest_hash")
            or bundle.document.get("asset_manifest_hash")
            or bundle.parsed.get("asset_manifest_hash")
        ),
        "language": options.annotation_language,
    }


def _legacy_worker_fingerprint(
    bundle: SourceBundle,
    options: BuildOptions,
    *,
    evidence: dict[str, Any],
    domain_context: dict[str, Any] | None,
    workers_per_lane: int,
) -> str:
    payload = _fingerprint_payload(
        bundle,
        options,
        evidence=evidence,
        domain_context=domain_context,
    )
    return sha256_json({**payload, "workers_per_lane": workers_per_lane})


def _checkpoint_dir_with_legacy_worker_migration(
    project_dir: Path,
    *,
    fingerprint: str,
    bundle: SourceBundle,
    options: BuildOptions,
    evidence: dict[str, Any],
    domain_context: dict[str, Any] | None,
    previous_state: dict[str, Any],
) -> Path:
    """Move an exactly matched pre-total-budget checkpoint into its content identity."""
    checkpoint_root = project_dir / ".arc-companion" / "checkpoints"
    target = checkpoint_root / fingerprint
    if target.exists() or options.force:
        return target

    context_workers = _read_optional_json(project_dir / "context.json").get("workers")
    candidates: list[int] = []
    if (
        not isinstance(context_workers, bool)
        and isinstance(context_workers, int)
        and context_workers > 0
    ):
        candidates.append(context_workers)
    candidates.extend(value for value in range(1, 1025) if value not in candidates)

    recorded_fingerprint = previous_state.get("fingerprint")
    legacy_workers: int | None = None
    legacy_fingerprint: str | None = None
    legacy_payload = _fingerprint_payload(
        bundle,
        options,
        evidence=evidence,
        domain_context=domain_context,
    )
    for candidate in candidates:
        candidate_fingerprint = sha256_json(
            {**legacy_payload, "workers_per_lane": candidate}
        )
        if candidate_fingerprint == recorded_fingerprint:
            legacy_workers = candidate
            legacy_fingerprint = candidate_fingerprint
            break
    if legacy_workers is None or legacy_fingerprint is None:
        return target
    legacy = checkpoint_root / legacy_fingerprint
    recorded_checkpoint = previous_state.get("checkpoint_dir")
    if (
        not recorded_checkpoint
        or Path(str(recorded_checkpoint)).resolve() != legacy.resolve()
        or not legacy.is_dir()
    ):
        return target

    checkpoint_root.mkdir(parents=True, exist_ok=True)
    os.replace(legacy, target)
    write_json(target / "checkpoint-migration.v1.json", {
        "schema_version": "arc.companion.checkpoint-migration.v1",
        "kind": "workers-to-total-concurrency-budget",
        "legacy_fingerprint": legacy_fingerprint,
        "content_fingerprint": fingerprint,
        "legacy_workers_per_lane": legacy_workers,
        "migrated_at": datetime.now(timezone.utc).isoformat(),
    })
    # Preserve completion/preview recovery in this invocation. The migration
    # proved that only the legacy operational worker field changed.
    previous_state["fingerprint"] = fingerprint
    previous_state["checkpoint_dir"] = str(target)
    return target


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
    bound_glossary = glossary
    if (
        glossary is not None
        and glossary.get("schema_version") != "arc.companion.segment-glossary.v2"
    ):
        bound_glossary = project_segment_glossary(
            _augmentation_blocks(segment, blocks_by_id),
            glossary,
        )
    return sha256_json({
        "augmentation_projection_version": AUGMENTATION_PROJECTION_VERSION,
        "segment": _semantic_segment_descriptor(segment),
        "blocks": [
            _augmentation_semantic_block(block)
            for block in _augmentation_blocks(segment, blocks_by_id)
        ],
        "glossary_hash": sha256_json(bound_glossary) if bound_glossary is not None else None,
        "extra": extra,
    })


def _sha256_existing_file(path: Path) -> str:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"completed companion output is missing or empty: {path}")
    return sha256_file(path)


def _completion_outputs_match(state: dict[str, Any]) -> bool:
    if state.get("final_render_version") != FINAL_RENDER_VERSION:
        return False
    if state.get("chapter_projection_version") != CHAPTER_PROJECTION_VERSION:
        return False
    if state.get("augmentation_projection_version") != AUGMENTATION_PROJECTION_VERSION:
        return False
    if state.get("chapter_guide_version") != CHAPTER_GUIDE_VERSION:
        return False
    if state.get("reader_final_checkpoint_version") != READER_FINAL_CHECKPOINT_VERSION:
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


def _web_outputs_match(state: dict[str, Any]) -> bool:
    try:
        from .web import WEB_RENDER_VERSION
    except ImportError:
        return False
    if state.get("web_render_version") != WEB_RENDER_VERSION:
        return False
    for path_key, hash_key in (
        ("output_html", "output_html_sha256"),
        ("reader_snapshot_path", "reader_snapshot_sha256"),
        ("web_manifest_path", "web_manifest_sha256"),
    ):
        path_value = state.get(path_key)
        expected = str(state.get(hash_key) or "")
        if not path_value or not expected:
            return False
        path = Path(str(path_value))
        if not path.is_file() or path.stat().st_size == 0 or sha256_file(path) != expected:
            return False
    return True


def _first_wave_preview_outputs_match(state: dict[str, Any]) -> bool:
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
        or preview_segment_count < 1
        or preview_segment_count > segment_count
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
            for key in ("id", "label", "title", "text", "doi", "arxiv_id", "links")
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
        for value in _augmentation_block_ids(segment, blocks_by_id)
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
        and _translation_block_has_required_natural_text(
            blocks_by_id[value], items[0]
        )
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
        "empty_block_ids": [
            value
            for value in expected_ids
            if len(candidates[value]) == 1
            and not _translation_block_has_required_natural_text(
                blocks_by_id[value], candidates[value][0]
            )
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
        for value in _augmentation_block_ids(segment, blocks_by_id)
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


def _translation_token_slot_ids(source_block: dict[str, Any]) -> list[str]:
    """Return token occurrence IDs used by an early v5 response variant."""
    token_ids: list[str] = []
    for token in _opaque_inline_tokens(source_block):
        if not _OPAQUE_INLINE_PATTERN.fullmatch(token):
            raise RuntimeError("source block has an invalid opaque token")
        token_ids.append(token[len("[[ARC_INLINE:"):-2].rsplit(":", 1)[0])
    return token_ids


def _translation_slots_from_token_insertion_offsets(
    raw_slots: list[dict[str, Any]], residue: str,
) -> list[str]:
    """Convert paid token-position responses into the canonical N+1 spans.

    A short-lived response shape identified each source token and returned a
    zero-width insertion offset instead of identifying the surrounding N+1
    repair slots.  The conversion is deterministic and preserves every byte of
    the already translated natural-language residue.
    """
    positions: list[int] = []
    for item in raw_slots:
        start = item.get("start_offset")
        end = item.get("end_offset")
        if type(start) is not int or type(end) is not int:
            raise RuntimeError(
                "translation token insertion repair returned non-integer offsets"
            )
        if start != end or start < 0 or start > len(residue):
            raise RuntimeError(
                "translation token insertion repair returned an invalid insertion point"
            )
        if positions and start < positions[-1]:
            raise RuntimeError(
                "translation token insertion repair changed source token order"
            )
        positions.append(start)
    boundaries = [0, *positions, len(residue)]
    return [
        residue[boundaries[index]:boundaries[index + 1]]
        for index in range(len(boundaries) - 1)
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
    if len(actual_slot_ids) != len(raw_slots):
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
    token_insertion_shape = actual_slot_ids == _translation_token_slot_ids(source_block)
    if token_insertion_shape:
        slots = _translation_slots_from_token_insertion_offsets(raw_slots, residue)
    elif actual_slot_ids != expected_slot_ids:
        raise RuntimeError("translation slot repair changed slot coverage or order")
    elif all("start_offset" in item and "end_offset" in item for item in raw_slots):
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
    r"\[(?:[ \t\u200b]*\[)?ARC_INLINE:[^\]\r\n]{0,512}\](?:[ \t\u200b]*\])?"
)
_OPAQUE_INLINE_CANONICALIZABLE_PATTERN = re.compile(
    r"\[(?:[ \t\u200b]*\[)?ARC_INLINE:"
    r"(?P<token_id>[A-Za-z0-9_.-]+):(?P<digest>[0-9a-f]{12,64})"
    r"\](?:[ \t\u200b]*\])?"
)
TRANSLATION_OPAQUE_TOKEN_CANONICALIZER_VERSION = (
    "arc.companion.translation-opaque-token-canonicalizer.v1"
)


def _candidate_digest_matches(expected_digest: str, candidate_digest: str) -> bool:
    """Require the source digest or its identity-bearing twelve-character prefix."""
    return (
        expected_digest == candidate_digest
        or (
            len(candidate_digest) >= 12
            and expected_digest.startswith(candidate_digest[:12])
        )
    )


def _canonicalize_translation_opaque_candidates(
    translation: dict[str, Any],
    blocks_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Restore bounded marker mutations without moving or regenerating prose."""
    raw_blocks = translation.get("blocks") if isinstance(translation, dict) else None
    if not isinstance(raw_blocks, list):
        return translation, []
    changed_ids: list[str] = []
    repaired_blocks: list[Any] = []
    for item in raw_blocks:
        if not isinstance(item, dict):
            repaired_blocks.append(item)
            continue
        item_id = str(item.get("block_id") or "")
        source_block = blocks_by_id.get(item_id)
        text = str(item.get("text") or "")
        if source_block is None:
            repaired_blocks.append(item)
            continue
        expected_tokens = _opaque_inline_tokens(source_block)
        if _OPAQUE_INLINE_PATTERN.findall(text) == expected_tokens:
            repaired_blocks.append(item)
            continue
        candidates = list(_OPAQUE_INLINE_CANONICALIZABLE_PATTERN.finditer(text))
        if len(candidates) != len(expected_tokens):
            repaired_blocks.append(item)
            continue
        identities_match = True
        for candidate, expected in zip(candidates, expected_tokens):
            payload = expected.removeprefix("[[ARC_INLINE:").removesuffix("]]")
            expected_token_id, expected_digest = payload.rsplit(":", 1)
            candidate_digest = str(candidate.group("digest") or "")
            if (
                candidate.group("token_id") != expected_token_id
                or not _candidate_digest_matches(expected_digest, candidate_digest)
            ):
                identities_match = False
                break
        if not identities_match:
            repaired_blocks.append(item)
            continue
        pieces: list[str] = []
        cursor = 0
        for candidate, expected in zip(candidates, expected_tokens):
            pieces.extend([text[cursor:candidate.start()], expected])
            cursor = candidate.end()
        pieces.append(text[cursor:])
        repaired_text = "".join(pieces)
        if _translation_natural_residue(repaired_text) != _translation_natural_residue(text):
            raise RuntimeError(
                "opaque-token canonicalization changed prior natural language"
            )
        if _OPAQUE_INLINE_PATTERN.findall(repaired_text) != expected_tokens:
            raise RuntimeError("opaque-token canonicalization did not restore source order")
        changed_ids.append(item_id)
        repaired_blocks.append({**item, "text": repaired_text})
    if not changed_ids:
        return translation, []
    return {**translation, "blocks": repaired_blocks}, changed_ids


def _opaque_inline_tokens(block: dict[str, Any]) -> list[str]:
    return _project_opaque_inline_tokens(block)


def _annotation_input_block(block: dict[str, Any], document: dict[str, Any]) -> dict[str, Any]:
    return _project_annotation_input_block(block, document)


def _bounded_annotation_prompt(
    segment: dict[str, Any],
    blocks: list[dict[str, Any]],
    *,
    language: str,
    metadata: dict[str, Any],
    evidence: dict[str, Any],
    glossary: dict[str, Any],
    protected_names: list[str],
    paper_context: dict[str, Any],
    domain_context: dict[str, Any] | None,
    source_language: str | None = None,
    max_bytes: int = ANNOTATION_PROMPT_MAX_BYTES,
) -> str:
    """Project optional context so every submitted annotation prompt is safely bounded."""
    attempts = (
        (ANNOTATION_GLOSSARY_MAX_BYTES, None),
        (ANNOTATION_GLOSSARY_MAX_BYTES, 10_000),
        (ANNOTATION_GLOSSARY_MAX_BYTES, 6_000),
        (ANNOTATION_GLOSSARY_MAX_BYTES, 3_000),
        (4_000, 3_000),
        (2_000, 3_000),
        (1_000, 3_000),
    )
    last_size = 0
    for glossary_bytes, paper_context_chars in attempts:
        projected_glossary = (
            {}
            if not glossary
            else _annotation_glossary_projection(
                glossary,
                segment=segment,
                blocks=blocks,
                max_bytes=glossary_bytes,
            )
        )
        projected_paper_context = json.loads(json.dumps(paper_context, ensure_ascii=False))
        if paper_context_chars is not None:
            _shrink_paper_context(projected_paper_context, max_chars=paper_context_chars)
        prompt = annotation_prompt(
            segment,
            blocks,
            language=language,
            metadata=metadata,
            evidence=evidence,
            glossary=projected_glossary,
            protected_names=protected_names,
            paper_context=projected_paper_context,
            domain_context=domain_context,
            source_language=source_language,
        )
        if not glossary:
            prompt = prompt.replace("\n\nGLOSSARY:\n{}\n\n", "\n\n")
        last_size = len(prompt.encode("utf-8"))
        if last_size < max_bytes:
            return prompt
    raise RuntimeError(
        f"annotation prompt essential payload is {last_size} bytes and cannot be bounded "
        f"below the {max_bytes}-byte transport limit without dropping source or evidence"
    )


def _annotation_glossary_projection(
    glossary: dict[str, Any],
    *,
    segment: dict[str, Any],
    blocks: list[dict[str, Any]],
    max_bytes: int = ANNOTATION_GLOSSARY_MAX_BYTES,
) -> dict[str, Any]:
    """Return the already source-only segment projection unchanged."""
    if (
        isinstance(glossary, dict)
        and glossary.get("schema_version") == "arc.companion.segment-glossary.v2"
    ):
        return json.loads(json.dumps(glossary, ensure_ascii=False))
    # Compatibility for callers outside the chapter pipeline: project only
    # immutable source blocks and never inspect evidence or generated text.
    return project_segment_glossary(
        [item for item in blocks if isinstance(item, dict)], glossary,
    )

def _normalized_glossary_match_text(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    normalized = unicodedata.normalize("NFKC", text).casefold()
    separated = "".join(
        " " if unicodedata.category(char)[0] in {"P", "Z"} else char
        for char in normalized
    )
    return re.sub(r"\s+", " ", separated).strip()


def _glossary_term_in_text(term: str, text: str) -> bool:
    if not term:
        return False
    if re.fullmatch(r"[a-z0-9 ]+", term):
        return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None
    return term in text


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
    heading_kinds = {
        "heading", "section", "subsection", "subsubsection", "chapter", "part",
    }
    navigation: list[dict[str, Any]] = []
    equation_navigation: list[dict[str, Any]] = []
    equation_by_id: dict[str, dict[str, Any]] = {}
    for equation in document.get("equations") or []:
        if not isinstance(equation, dict):
            continue
        for key in ("id", "equation_id", "entity_id", "block_id", "source_id"):
            if equation.get(key) is not None:
                equation_by_id[str(equation[key])] = equation
    location_block_id = ""
    for index, block in enumerate(blocks):
        kind = str(block.get("type") or block.get("kind") or "").casefold()
        if kind in heading_kinds:
            location_block_id = block_id(block)
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
                "anchor": anchor[:500],
            })
            continue
        if kind not in {"equation", "math", "display_math"}:
            continue
        entity = next((
            equation_by_id[str(block[key])]
            for key in ("equation_id", "entity_id", "source_id", "id", "block_id")
            if block.get(key) is not None and str(block[key]) in equation_by_id
        ), {})
        printed_numbers = entity.get("printed_equation_numbers") or block.get("printed_equation_numbers")
        if isinstance(printed_numbers, list):
            number = ", ".join(str(value) for value in printed_numbers if str(value).strip())
        else:
            number = str(
                entity.get("number") or block.get("number") or
                entity.get("label") or block.get("label") or ""
            ).strip()
        formula = str(
            entity.get("tex") or entity.get("content") or entity.get("text") or
            block.get("tex") or block.get("content") or block.get("text") or ""
        ).strip()
        equation_navigation.append({
            "block_id": block_id(block),
            "number": number[:120],
            "location_block_id": location_block_id,
            "formula": formula[:600],
        })

    neighbor_indices = [
        index for index in range(max(0, first - 2), min(len(blocks), last + 3))
        if index < first or index > last
    ]
    neighbors = [
        _bounded_projection(_annotation_input_block(blocks[index], document), 1_200)
        for index in neighbor_indices if not _project_is_structural(blocks[index])
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
            "start_block_id": str(segment.get("start_block_id") or ""),
            "end_block_id": str(segment.get("end_block_id") or ""),
            "start_ordinal": first + 1,
            "end_ordinal": last + 1,
            "total_blocks": len(blocks),
        },
        "section_navigation": navigation,
        "equation_navigation": equation_navigation,
        "neighboring_source_anchors": neighbors,
        "access": _generation_runtime_policy(options),
    }
    _shrink_paper_context(context, max_chars=max_chars)
    return context


def _static_paper_context(context: dict[str, Any]) -> dict[str, Any]:
    """Paper context sent once per native generation."""
    return {
        key: context.get(key)
        for key in (
            "schema_version", "paper_id", "abstract", "section_navigation",
            "navigation_omitted_count", "equation_navigation",
            "equation_navigation_omitted_count", "access",
        )
        if key in context
    }


def _dynamic_paper_context(context: dict[str, Any]) -> dict[str, Any]:
    """Segment-local anchors refreshed on every stateful turn."""
    return {
        "current_segment": dict(context.get("current_segment") or {}),
        "neighboring_source_anchors": list(context.get("neighboring_source_anchors") or []),
    }


def _compact_chapter_descriptor(chapter: dict[str, Any]) -> dict[str, Any]:
    return {
        key: chapter.get(key)
        for key in (
            "chapter_id", "start_block_id", "end_block_id", "page_start", "page_end",
        )
        if key in chapter
    }


def _guide_chapter_descriptor(chapter: Mapping[str, Any]) -> dict[str, Any]:
    """Project chapter identity without exposing displayed structural titles."""
    return {
        key: chapter.get(key)
        for key in (
            "chapter_id", "content_block_ids", "page_start", "page_end",
        )
        if key in chapter
    }


def _compact_segment_descriptor(segment: dict[str, Any]) -> dict[str, Any]:
    return {
        key: segment.get(key)
        for key in (
            "segment_id", "chapter_id", "title", "start_block_id", "end_block_id",
            "start_ordinal", "end_ordinal", "page_start", "page_end",
        )
        if key in segment
    }


def _augmentation_block_ids(
    segment: Mapping[str, Any], blocks_by_id: Mapping[str, dict[str, Any]],
) -> list[str]:
    """Return the source blocks eligible for generated augmentation."""
    explicit = segment.get("augmentation_block_ids")
    if isinstance(explicit, list):
        return [str(value) for value in explicit if str(value) in blocks_by_id]
    return [
        str(value) for value in segment.get("block_ids") or []
        if str(value) in blocks_by_id
        and not _project_is_structural(blocks_by_id[str(value)])
    ]


def _augmentation_blocks(
    segment: Mapping[str, Any], blocks_by_id: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [blocks_by_id[value] for value in _augmentation_block_ids(segment, blocks_by_id)]


def _augmentation_semantic_block(block: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(_prompt_safe_value(dict(block)))
    for key in ("section_title", "chapter_title", "part_title", "heading_title"):
        value.pop(key, None)
    return value


def _semantic_segment_descriptor(segment: Mapping[str, Any]) -> dict[str, Any]:
    """Describe augmentation identity without controller or source heading text."""
    return {
        "segment_id": str(segment.get("segment_id") or ""),
        "chapter_id": str(segment.get("chapter_id") or ""),
        "augmentation_block_ids": [
            str(value) for value in segment.get("augmentation_block_ids") or []
        ],
        "structural_only": bool(segment.get("structural_only")),
    }


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
    equations = context.get("equation_navigation") or []
    if size() > max_chars:
        for item in navigation:
            item["anchor"] = str(item.get("anchor") or "")[:160]
        for item in equations:
            item["formula"] = str(item.get("formula") or "")[:240]
    if size() > max_chars:
        context["abstract"] = str(context.get("abstract") or "")[:1_200]
    if size() > max_chars:
        for item in navigation:
            item.pop("anchor", None)
        for item in equations:
            item["formula"] = str(item.get("formula") or "")[:120]
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
    if size() > max_chars and equations:
        original_equation_count = len(equations)
        keep = max(1, int(original_equation_count * max_chars / size()))
        if keep < original_equation_count:
            indices = sorted({
                round(index * (original_equation_count - 1) / max(1, keep - 1))
                for index in range(keep)
            })
            context["equation_navigation"] = [equations[index] for index in indices]
            context["equation_navigation_omitted_count"] = (
                original_equation_count - len(indices)
            )
    while size() > max_chars and len(context["section_navigation"]) > 1:
        current = context["section_navigation"]
        context["section_navigation"] = current[::2]
        context["navigation_omitted_count"] = len(navigation) - len(context["section_navigation"])
    while size() > max_chars and len(context.get("equation_navigation") or []) > 1:
        current_equations = context["equation_navigation"]
        context["equation_navigation"] = current_equations[::2]
        context["equation_navigation_omitted_count"] = (
            len(equations) - len(context["equation_navigation"])
        )
    if size() > max_chars:
        context["neighboring_source_anchors"] = []
        context["abstract"] = ""
    if size() > max_chars:
        for item in context.get("equation_navigation") or []:
            item.pop("formula", None)


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
    expected_blocks = [
        blocks_by_id[value]
        for value in _augmentation_block_ids(segment, blocks_by_id)
        if _is_translatable(blocks_by_id[value])
    ]
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
        if not _translation_block_has_required_natural_text(source, translated):
            raise TranslationCoverageError(
                f"translation {segment['segment_id']} returned empty block {block_id(source)}"
            )
    return expected_blocks, raw_blocks


def _translation_block_has_required_natural_text(
    source: dict[str, Any], translated: dict[str, Any],
) -> bool:
    """Return whether a mapped block contains prose when its source contains prose."""
    if not _natural_text_for_name_validation(source).strip():
        return True
    text = str(translated.get("text") or "").strip()
    return bool(_translation_natural_residue(text).strip())


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


def _augmentation_document(document: dict[str, Any]) -> dict[str, Any]:
    """Return generation-eligible source while preserving headings elsewhere."""
    projected = _generation_document(document)
    return {
        **projected,
        "blocks": [
            block for block in projected.get("blocks") or []
            if not _project_is_structural(block)
        ],
    }


def _empty_glossary(*, language: str, page_count: int | None) -> dict[str, Any]:
    return {
        "schema_version": GLOSSARY_VERSION,
        "source_sha256": sha256_json({
            "augmentation_projection_version": AUGMENTATION_PROJECTION_VERSION,
            "blocks": [], "language": language, "page_count": page_count,
        }),
        "language": language,
        "page_count": page_count,
        "entry_limit": glossary_entry_limit(page_count),
        "entries": [],
        "logical_receipt": {
            "kind": "controller_skipped_structural_heading",
            "provider_calls": 0,
        },
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
            # Also protect roots in cased writing systems.  Uncased compact
            # scripts retain the full metadata name without unsafe splitting.
            for token in re.findall(r"[^\W\d_][^\W_\d'’-]*(?:['’-][^\W\d_]+)*", text):
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
        if contains_lexical_term(source, name, case_sensitive=True)
        and not contains_lexical_term(generated, name, case_sensitive=True)
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
            and contains_lexical_term(other, name)
            for other in missing_names
        )
    ]


def _protected_names_for_blocks(
    protected_names: list[str], blocks: list[dict[str, Any]],
) -> list[str]:
    """Project protected names from immutable source only."""
    source = "\n".join(
        " ".join(str(block.get(key) or "") for key in ("title", "text", "markdown", "tex"))
        for block in blocks
    )
    return [
        name for name in protected_names
        if name and contains_lexical_term(source, name, case_sensitive=True)
    ]


def _evidence_for_segment(
    segment: dict[str, Any],
    blocks_by_id: dict[str, dict[str, Any]],
    evidence: dict[str, Any],
    *,
    usage_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_text = " ".join(
        str(blocks_by_id[value].get("text") or blocks_by_id[value].get("title") or "")
        for value in _augmentation_block_ids(segment, blocks_by_id)
    )
    source_tokens = _terms(source_text)
    citation_targets = _segment_citation_targets(segment, blocks_by_id, evidence)
    related_papers = _deduplicate_evidence_records(
        list(evidence.get("related_papers") or [])
    )
    idf = _related_paper_idf(related_papers)
    selected: list[dict[str, Any]] = []
    for relation in ("prior", "later"):
        candidates: list[tuple[int, int, float, int, int, dict[str, Any]]] = []
        for index, paper in enumerate(related_papers):
            if paper.get("relation") != relation:
                continue
            exact = relation == "prior" and _paper_matches_citation_targets(paper, citation_targets)
            local_relevance = _strongest_local_relevance(paper, source_tokens, idf)
            if not (exact or local_relevance is not None):
                continue
            relevance = (
                (12 if exact else 0)
                + (local_relevance or 0.0)
            )
            citation_prior = min(_citation_count_value(paper).bit_length(), 10)
            candidates.append((-int(exact), 0, -relevance, -citation_prior, index, paper))
        for _, _, _, _, _, paper in _select_related_candidates(
            candidates, usage_state=usage_state, limit=3,
        ):
            compact = {key: paper.get(key) for key in (
                "evidence_id", "relation", "paper_id", "arxiv_id", "doi", "inspire_id",
                "title", "authors", "year", "citation_count", "evidence_level", "abstract",
                "url", "landing_url", "source_url", "html_url",
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
    selected_ids = {str(item.get("evidence_id") or "") for item in selected}
    context_papers = [
        paper for paper in related_papers
        if paper.get("relation") == "context"
    ]
    if context_papers:
        per_source_budget = min(
            CONTEXT_SEGMENT_CHARS_PER_SOURCE,
            max(1, CONTEXT_SEGMENT_CHARS_TOTAL // len(context_papers)),
        )
        for paper in context_papers:
            evidence_id = str(paper.get("evidence_id") or "")
            if evidence_id in selected_ids:
                continue
            compact = {key: paper.get(key) for key in (
                "evidence_id", "relation", "paper_id", "title", "authors", "year",
                "evidence_level", "abstract", "context_role", "context_index",
                "url", "landing_url", "source_url", "html_url",
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
            selected_ids.add(evidence_id)
    bounded_sources: list[dict[str, str]] = []
    seen_source_urls: set[str] = set()
    for paper in selected:
        url = next((
            str(paper.get(key) or "") for key in (
                "url", "landing_url", "source_url", "html_url", "pdf_url",
            )
            if str(paper.get(key) or "").startswith(("http://", "https://"))
        ), "")
        title = str(paper.get("title") or "").strip()
        if not title or not url or url.casefold() in seen_source_urls:
            continue
        snippets = [item for item in paper.get("snippets") or [] if isinstance(item, dict)]
        locator = next((
            str(item.get("section_title") or item.get("block_id") or "").strip()
            for item in snippets
            if str(item.get("section_title") or item.get("block_id") or "").strip()
        ), "Abstract")
        bounded_sources.append({"title": title, "url": url, "locator": locator})
        seen_source_urls.add(url.casefold())
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
        "bounded_sources": bounded_sources,
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
    for block_id_value in _augmentation_block_ids(segment, blocks_by_id):
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


def _section_review_validation_error(
    review: Any,
    chunk: list[dict[str, Any]],
) -> str | None:
    """Return why a section review is unsafe to reuse or persist, if anything."""
    expected_ids = [
        str((item.get("segment") or {}).get("segment_id") or "")
        for item in chunk
    ]
    if (
        not expected_ids
        or any(not value for value in expected_ids)
        or len(expected_ids) != len(set(expected_ids))
    ):
        return "has invalid expected segment coverage"
    if not isinstance(review, dict):
        return "is malformed"
    reviewed_ids = review.get("reviewed_segment_ids")
    if not isinstance(reviewed_ids, list):
        return "is missing reviewed segment coverage"
    if any(not isinstance(item, str) or not item for item in reviewed_ids):
        return "contains a malformed reviewed segment id"
    if len(reviewed_ids) != len(set(reviewed_ids)):
        return "contains duplicate reviewed segments"
    if len(reviewed_ids) != len(expected_ids) or set(reviewed_ids) != set(expected_ids):
        return "did not cover every segment"
    findings = review.get("findings")
    if not isinstance(findings, list) or any(
        not isinstance(item, dict)
        or not isinstance(item.get("segment_id"), str)
        or not item["segment_id"]
        or not isinstance(item.get("issue"), str)
        for item in findings
    ):
        return "contains malformed findings"
    invalid_finding_ids = {
        str(item.get("segment_id") or "") for item in findings
    } - set(expected_ids)
    if invalid_finding_ids:
        return "returned findings for unknown segments"
    patches = review.get("patches")
    if not isinstance(patches, list):
        return "is missing sparse patches"
    patch_ids = [
        str(item.get("segment_id") or "") for item in patches if isinstance(item, dict)
    ]
    if len(patch_ids) != len(patches) or any(value not in expected_ids for value in patch_ids):
        return "returned a malformed or out-of-chunk patch"
    if len(patch_ids) != len(set(patch_ids)):
        return "returned duplicate patches"
    return None


def _load_recovered_section_reviews(
    checkpoint_dir: Path,
    chunks: list[list[dict[str, Any]]],
) -> dict[int, dict[str, Any]]:
    """Import only recovered sections whose exact segment set survives rechunking."""
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
    recovered_by_ids: dict[frozenset[str], dict[str, Any]] = {}
    recovered_indices: set[int] = set()
    covered_ids: set[str] = set()
    for item in recovered["section_reviews"]:
        if not isinstance(item, dict):
            raise RuntimeError("recovered section review is malformed")
        index = int(item.get("section_index", -1))
        if index < 0 or index in recovered_indices:
            raise RuntimeError("recovered section review has an invalid section index")
        recovered_indices.add(index)
        declared_id_values = [
            str(value) for value in item.get("reviewed_segment_ids") or []
        ]
        declared_ids = set(declared_id_values)
        review = {
            "reviewed_segment_ids": declared_id_values,
            "findings": item.get("findings"),
            "patches": item.get("patches") or item.get("patch_proposals") or [],
        }
        review = _normalize_sparse_review_patches(
            review, scope="recovered section review",
        )
        validation_error = _section_review_validation_error(
            review,
            [
                {"segment": {"segment_id": segment_id}}
                for segment_id in declared_id_values
            ],
        )
        if (
            not declared_ids
            or len(declared_id_values) != len(declared_ids)
            or validation_error is not None
        ):
            raise RuntimeError(f"recovered section review {index} does not match its chunk")
        frozen_ids = frozenset(declared_ids)
        if frozen_ids in recovered_by_ids:
            raise RuntimeError("recovered section reviews have duplicate chunks")
        overlap = covered_ids.intersection(declared_ids)
        if overlap:
            raise RuntimeError(
                "recovered section reviews have overlapping segment coverage: "
                f"{sorted(overlap)}"
            )
        covered_ids.update(declared_ids)
        recovered_by_ids[frozen_ids] = review
    if covered_ids != expected_all:
        raise RuntimeError("recovered section reviews do not cover every section")

    imported: dict[int, dict[str, Any]] = {}
    for index, chunk in enumerate(chunks):
        expected_ids = frozenset(
            str(value["segment"]["segment_id"]) for value in chunk
        )
        if value := recovered_by_ids.get(expected_ids):
            imported[index] = value
    return imported


def _section_review_patch_proposals(
    chunk: list[dict[str, Any]],
    patches: list[dict[str, Any]],
    findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Preserve already sparse, schema-validated section patches."""
    valid_ids = {str(item["segment"]["segment_id"]) for item in chunk}
    return [dict(item) for item in patches if str(item.get("segment_id") or "") in valid_ids]


_SPARSE_REVIEW_PATCH_FIELDS = (
    "commentary",
    "explanation",
    "commentary_sources",
    "prior_work",
    "later_work",
)


def _review_patch_block_order(
    chunk: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    return {
        str((item.get("segment") or {}).get("segment_id") or ""): [
            str(value)
            for value in (
                (item.get("segment") or {}).get("augmentation_block_ids") or []
            )
        ]
        for item in chunk
        if isinstance(item, Mapping) and isinstance(item.get("segment"), Mapping)
    }


def _normalize_sparse_review_patches(
    review: Any,
    *,
    block_order_by_segment: Mapping[str, Sequence[str]] | None = None,
    scope: str = "review",
) -> Any:
    """Merge compatible same-segment sparse patches without hiding conflicts.

    Patch reasons are audit metadata rather than replacement fields. Distinct
    reasons are retained in first-seen order on the merged patch.
    """

    if not isinstance(review, dict) or not isinstance(review.get("patches"), list):
        return review
    patches = review["patches"]
    if any(not isinstance(item, dict) for item in patches):
        return review

    block_order_by_segment = block_order_by_segment or {}
    merged_by_segment: dict[str, dict[str, Any]] = {}
    translation_blocks_by_segment: dict[str, dict[str, dict[str, Any]]] = {}
    translation_order_by_segment: dict[str, list[str]] = {}
    translation_present: set[str] = set()
    reasons_by_segment: dict[str, list[str]] = {}
    order: list[str] = []
    for raw_patch in patches:
        patch = dict(raw_patch)
        segment_id = str(patch.get("segment_id") or "")
        if segment_id not in merged_by_segment:
            merged_by_segment[segment_id] = {
                **patch,
                **{field: None for field in _SPARSE_REVIEW_PATCH_FIELDS},
                "translation_blocks": None,
            }
            translation_blocks_by_segment[segment_id] = {}
            translation_order_by_segment[segment_id] = []
            reasons_by_segment[segment_id] = []
            order.append(segment_id)
        merged = merged_by_segment[segment_id]
        for field in _SPARSE_REVIEW_PATCH_FIELDS:
            incoming = patch.get(field)
            if incoming is None:
                continue
            existing = merged.get(field)
            if existing is None:
                merged[field] = incoming
            elif canonical_sha256(existing) != canonical_sha256(incoming):
                raise RuntimeError(
                    f"{scope} returned conflicting patches for "
                    f"{segment_id} field {field}"
                )
        incoming_blocks = patch.get("translation_blocks")
        if incoming_blocks is not None:
            translation_present.add(segment_id)
            if not isinstance(incoming_blocks, list):
                raise RuntimeError(
                    f"{scope} returned malformed translation blocks for {segment_id}"
                )
            blocks_by_id = translation_blocks_by_segment[segment_id]
            first_seen = translation_order_by_segment[segment_id]
            for block in incoming_blocks:
                if not isinstance(block, dict) or not str(block.get("block_id") or ""):
                    raise RuntimeError(
                        f"{scope} returned malformed translation blocks for {segment_id}"
                    )
                current_block_id = str(block["block_id"])
                existing_block = blocks_by_id.get(current_block_id)
                if existing_block is None:
                    blocks_by_id[current_block_id] = dict(block)
                    first_seen.append(current_block_id)
                elif canonical_sha256(existing_block) != canonical_sha256(block):
                    raise RuntimeError(
                        f"{scope} returned conflicting patches for {segment_id} "
                        f"translation block {current_block_id}"
                    )
        reason = patch.get("reason")
        if isinstance(reason, str) and reason not in reasons_by_segment[segment_id]:
            reasons_by_segment[segment_id].append(reason)

    normalized: list[dict[str, Any]] = []
    for segment_id in order:
        patch = merged_by_segment[segment_id]
        translation_blocks = translation_blocks_by_segment[segment_id]
        if segment_id in translation_present:
            source_order = {
                str(block_id): index
                for index, block_id in enumerate(
                    block_order_by_segment.get(segment_id) or []
                )
            }
            first_seen_order = {
                block_id: index
                for index, block_id in enumerate(
                    translation_order_by_segment[segment_id]
                )
            }
            ordered_block_ids = sorted(
                translation_blocks,
                key=lambda block_id: (
                    source_order.get(block_id, len(source_order)),
                    first_seen_order[block_id],
                ),
            )
            patch["translation_blocks"] = [
                translation_blocks[block_id] for block_id in ordered_block_ids
            ]
        reasons = reasons_by_segment[segment_id]
        if reasons:
            patch["reason"] = "; ".join(reasons)
        normalized.append(patch)
    return {**review, "patches": normalized}


def _utf8_size(value: str) -> int:
    return len(value.encode("utf-8"))


def _review_prompt_byte_limit(options: BuildOptions) -> int:
    """Combine the user soft budget with ARC's hard transport ceiling."""
    return min(
        REVIEW_PROMPT_MAX_BYTES,
        max(REVIEW_PROMPT_MIN_SOFT_BYTES, int(options.review_context_chars)),
    )


def _review_prompt_target_limit(options: BuildOptions) -> int:
    """Return the packing target, leaving ten percent below the hard limit."""
    return (
        _review_prompt_byte_limit(options) * REVIEW_PROMPT_TARGET_NUMERATOR
        // REVIEW_PROMPT_TARGET_DENOMINATOR
    )


def _review_prompt_budget(options: BuildOptions) -> dict[str, Any]:
    strict = _review_prompt_byte_limit(options)
    return {
        "review_context_chars": int(options.review_context_chars),
        "strict_limit_bytes": strict,
        "target_limit_bytes": (
            strict * REVIEW_PROMPT_TARGET_NUMERATOR
            // REVIEW_PROMPT_TARGET_DENOMINATOR
        ),
        "target_ratio": {
            "numerator": REVIEW_PROMPT_TARGET_NUMERATOR,
            "denominator": REVIEW_PROMPT_TARGET_DENOMINATOR,
        },
    }


def _with_historical_review_prompt_audit(
    review: Mapping[str, Any], options: BuildOptions,
) -> dict[str, Any]:
    """Normalize pre-audit accepted reviews without triggering model work."""
    normalized = dict(review)
    if isinstance(normalized.get("prompt_budget_audit"), dict):
        return normalized
    target = _review_prompt_target_limit(options)
    normalized["prompt_budget_audit"] = {
        "schema_version": REVIEW_PROMPT_BUDGET_AUDIT_VERSION,
        "budget": _review_prompt_budget(options),
        "routing": {
            "mode": "artifact-reuse",
            "direct_prompt_bytes": 0,
            "hierarchy_threshold_bytes": min(
                target, max(1, int(options.review_context_chars))
            ),
        },
        "calls": [],
        "historical_measurements_available": False,
    }
    return normalized


def _review_prompt_call_audit(
    rendered: Mapping[str, Any],
    *,
    stage: str,
    call_label: str,
    disposition: str,
) -> dict[str, Any]:
    prompt_bytes = int(rendered["prompt_bytes"])
    target_bytes = int(rendered["target_limit_bytes"])
    strict_bytes = int(rendered["strict_limit_bytes"])
    return {
        "stage": stage,
        "call_label": call_label,
        "segment_ids": list(rendered["segment_ids"]),
        "prompt_sha256": str(rendered["prompt_sha256"]),
        "prompt_bytes": prompt_bytes,
        "target_headroom_bytes": max(0, target_bytes - prompt_bytes),
        "target_overage_bytes": max(0, prompt_bytes - target_bytes),
        "strict_headroom_bytes": strict_bytes - prompt_bytes,
        "budget_class": str(rendered["budget_class"]),
        "disposition": disposition,
    }


def _review_prompt_context(
    rendered: Mapping[str, Any],
    *,
    stage: str,
    audit_sink: list[dict[str, Any]],
) -> dict[str, Any]:
    """Describe one logical review call for bounded evidence follow-up turns."""
    return {
        "stage": stage,
        "segment_ids": list(rendered["segment_ids"]),
        "target_limit_bytes": int(rendered["target_limit_bytes"]),
        "strict_limit_bytes": int(rendered["strict_limit_bytes"]),
        "audit_sink": audit_sink,
    }


def _rendered_review_call(
    items: list[dict[str, Any]],
    prompt: str,
    *,
    target_prompt_bytes: int,
    strict_prompt_bytes: int,
    headroom_class: str,
) -> dict[str, Any]:
    size = _utf8_size(prompt)
    return {
        "items": items,
        "prompt": prompt,
        "prompt_bytes": size,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "segment_ids": [
            str((item.get("segment") or {}).get("segment_id") or "")
            for item in items
        ],
        "target_limit_bytes": target_prompt_bytes,
        "strict_limit_bytes": strict_prompt_bytes,
        "budget_class": "normal" if size <= target_prompt_bytes else headroom_class,
    }


def _pack_rendered_review_calls(
    items: list[dict[str, Any]],
    *,
    render_prompt: Callable[[list[dict[str, Any]]], str],
    target_prompt_bytes: int,
    strict_prompt_bytes: int,
    label: str,
    headroom_class: str = "singleton_headroom",
) -> list[dict[str, Any]]:
    """Greedily pack ordered items using the exact prompt sent to the provider."""
    calls: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_prompt = ""
    for item in items:
        candidate = [*current, item]
        candidate_prompt = render_prompt(candidate)
        if _utf8_size(candidate_prompt) <= target_prompt_bytes:
            current = candidate
            current_prompt = candidate_prompt
            continue
        if current:
            calls.append(_rendered_review_call(
                current,
                current_prompt,
                target_prompt_bytes=target_prompt_bytes,
                strict_prompt_bytes=strict_prompt_bytes,
                headroom_class=headroom_class,
            ))
        single_prompt = render_prompt([item])
        single_size = _utf8_size(single_prompt)
        if single_size > strict_prompt_bytes:
            segment_id = str((item.get("segment") or {}).get("segment_id") or "")
            raise RuntimeError(
                f"{label} segment {segment_id or '<unknown>'} requires a "
                f"{single_size}-byte prompt, exceeding the strict "
                f"{strict_prompt_bytes}-byte limit"
            )
        current = [item]
        current_prompt = single_prompt
    if current:
        calls.append(_rendered_review_call(
            current,
            current_prompt,
            target_prompt_bytes=target_prompt_bytes,
            strict_prompt_bytes=strict_prompt_bytes,
            headroom_class=headroom_class,
        ))
    return calls


def _require_review_prompt_within_limit(
    prompt: str,
    *,
    label: str,
    max_prompt_bytes: int,
) -> None:
    size = _utf8_size(prompt)
    if size > max_prompt_bytes:
        raise RuntimeError(
            f"{label} requires a {size}-byte prompt, exceeding the strict "
            f"{max_prompt_bytes}-byte limit"
        )


def _bounded_hierarchical_review_prompt(
    section_reviews: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    *,
    blocks_by_id: dict[str, dict[str, Any]],
    document: dict[str, Any],
    segment_payloads: list[dict[str, Any]],
    glossary: dict[str, Any],
    protected_names: list[str],
    language: str,
    max_prompt_bytes: int,
    strict_prompt_bytes: int | None = None,
    intent_guidance: Mapping[str, Any] | None = None,
    essential_only: bool = False,
) -> tuple[dict[str, Any], str]:
    """Render successively smaller projections and measure the exact guided prompt."""
    strict_prompt_bytes = (
        max_prompt_bytes if strict_prompt_bytes is None else strict_prompt_bytes
    )
    instruction = (
        "Consolidate section findings into non-conflicting translation and/or annotation patches. "
        "Use the bounded source anchor for every segment as a direct source-awareness check; "
        "section reviews contain bounded findings and proposed corrections from the complete "
        "source-aware local review."
    )
    # Optional material shrinks monotonically.  The final attempt still keeps
    # every section/segment identity, a source digest, and a source excerpt; if
    # that essential audit projection does not fit, fail before calling a model.
    attempts = (
        (0.24, 0.18, 0.12, 0.28, True, 160),
        (0.16, 0.10, 0.08, 0.18, True, 96),
        (0.10, 0.05, 0.04, 0.10, True, 64),
        (0.06, 0.00, 0.00, 0.00, False, 32),
    )
    if essential_only:
        attempts = attempts[-1:]
    last_prompt = ""
    last_payload: dict[str, Any] = {}
    for source_ratio, context_ratio, glossary_ratio, section_ratio, include_context, minimum_excerpt in attempts:
        source_anchors = _review_source_anchors(
            segments,
            blocks_by_id=blocks_by_id,
            document=document,
            total_chars=int(max_prompt_bytes * source_ratio),
            minimum_excerpt_chars=minimum_excerpt,
        )
        context_evidence_anchors = (
            _review_context_evidence_anchors(
                segment_payloads,
                total_chars=int(max_prompt_bytes * context_ratio),
            )
            if include_context
            else []
        )
        bounded_glossary = _bounded_glossary_projection(
            glossary,
            total_chars=int(max_prompt_bytes * glossary_ratio),
        )
        projected_sections = (
            [
                {
                    "section_index": int(section["section_index"]),
                    "reviewed_segment_ids": list(
                        section.get("reviewed_segment_ids") or []
                    ),
                }
                for section in section_reviews
            ]
            if section_ratio == 0.0 else _bounded_section_review_projection(
                section_reviews,
                total_chars=int(max_prompt_bytes * section_ratio),
            )
        )
        last_payload = {
            "section_reviews": projected_sections,
            "reviewed_segment_ids": [str(item["segment_id"]) for item in segments],
            "source_anchors": source_anchors,
            "context_evidence_anchors": context_evidence_anchors,
            "glossary": bounded_glossary,
            "protected_names": protected_names,
            "instruction": instruction,
        }
        last_prompt = _guided_prompt(
            review_prompt(last_payload, language=language, findings=None),
            intent_guidance,
            lane="review",
        )
        if _utf8_size(last_prompt) <= max_prompt_bytes:
            return last_payload, last_prompt
    _require_review_prompt_within_limit(
        last_prompt,
        label="hierarchical final review essential projection",
        max_prompt_bytes=strict_prompt_bytes,
    )
    return last_payload, last_prompt


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


def _review_chunks(
    items: list[dict[str, Any]],
    *,
    language: str,
    max_prompt_bytes: int = SECTION_REVIEW_PROMPT_MAX_BYTES,
    strict_prompt_bytes: int | None = None,
    render_prompt: Callable[[list[dict[str, Any]]], str] | None = None,
) -> list[list[dict[str, Any]]]:
    """Pack complete segments using a caller's exact final prompt renderer."""
    renderer = render_prompt or (
        lambda group: section_review_prompt({"segments": group}, language=language)
    )
    return [
        list(call["items"])
        for call in _pack_rendered_review_calls(
            items,
            render_prompt=renderer,
            target_prompt_bytes=max_prompt_bytes,
            strict_prompt_bytes=(
                max_prompt_bytes if strict_prompt_bytes is None else strict_prompt_bytes
            ),
            label="section review",
        )
    ]


def _review_source_anchors(
    segments: list[dict[str, Any]],
    *,
    blocks_by_id: dict[str, dict[str, Any]],
    document: dict[str, Any],
    total_chars: int,
    minimum_excerpt_chars: int = 160,
) -> list[dict[str, Any]]:
    """Give the hierarchical final reviewer bounded source context for every segment."""
    per_segment = max(
        1,
        min(
            1_200,
            max(minimum_excerpt_chars, total_chars // max(1, len(segments))),
        ),
    )
    anchors: list[dict[str, Any]] = []
    for segment in segments:
        projection = [
            _annotation_input_block(blocks_by_id[value], document)
            for value in _augmentation_block_ids(segment, blocks_by_id)
        ]
        serialized = json.dumps(projection, ensure_ascii=False, separators=(",", ":"))
        if len(serialized) > per_segment:
            half = max(1, (per_segment - 3) // 2)
            excerpt = serialized[:half] + "..." + serialized[-half:]
        else:
            excerpt = serialized
        anchors.append({
            "segment_id": str(segment.get("segment_id") or ""),
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


def _write_reader_final_checkpoint(
    checkpoint_dir: Path, final_overrides: dict[str, Any],
) -> Path:
    path = checkpoint_dir / "reader-final.json"
    write_json(path, {
        "schema_version": READER_FINAL_CHECKPOINT_VERSION,
        "final_overrides": final_overrides,
    })
    return path


def _write_legacy_reuse_plan(checkpoint_dir: Path, options: BuildOptions) -> Path:
    """Write the legacy-path lane plan before its first provider submission."""

    entries = []
    for lane in (
        "segmentation", "glossary", "title_translation", "translation",
        "commentary", "review",
    ):
        skipped = options.skip_translation and lane in {
            "glossary", "title_translation", "translation",
        }
        entries.append({
            "chapter_id": "project", "segment_id": None, "lane": lane,
            "status": "skipped" if skipped else "miss", "artifact_id": None,
            "reason": (
                "glossary_disabled_for_same_language_source"
                if skipped and lane == "glossary"
                else "title_translation_disabled_for_same_language_source"
                if skipped and lane == "title_translation"
                else "translation_disabled_for_same_language_source" if skipped
                else "explicitly selected for regeneration" if lane in options.regenerate_lanes
                else "legacy pipeline cache lookup follows deterministic preparation"
            ),
            "estimated_provider_calls": 0 if skipped else 1,
        })
    path = checkpoint_dir / "reuse-plan.json"
    write_json(path, {
        "schema_version": REUSE_PLAN_VERSION,
        "entries": entries,
        "estimated_provider_calls": sum(item["estimated_provider_calls"] for item in entries),
    })
    return path


def _store_reviewed_content(
    project_dir: Path,
    *,
    checkpoint_dir: Path,
    final_overrides: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Commit the reviewed render input before any fallible renderer runs."""
    segments = list(final_overrides.get("segments") or [])
    annotations = dict(final_overrides.get("annotations") or {})
    reader_evidence = _reader_evidence_by_segment(
        segments,
        document=dict(final_overrides.get("document") or {}),
        evidence=evidence,
        annotations=annotations,
    )
    chains, overlays = checkpoint_receipts(checkpoint_dir)
    content = reader_content_from_overrides(
        final_overrides,
        reader_evidence_by_segment=reader_evidence,
        accepted_ledger_chains=chains,
        review_overlay_hashes=overlays,
    )
    review_receipts: dict[str, Any] = {}
    for name in ("review.v5.json", "chapter-review.json", "annotations.reviewed.v5.json"):
        path = checkpoint_dir / name
        if path.is_file():
            review_receipts[name] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    return store_reader_content(
        project_dir,
        content=content,
        checkpoint_dir=checkpoint_dir,
        review_receipts=review_receipts,
    )


def _publish_reader_update(
    project_dir: Path,
    state_path: Path,
    lock: threading.RLock,
    *,
    final_overrides: dict[str, Any] | None = None,
    strict: bool = False,
) -> dict[str, Any] | None:
    """Serialize reader publication and preserve the last atomic bundle on failure."""
    try:
        from .web import publish_reader
    except ImportError:
        if strict:
            raise
        return None
    with lock:
        try:
            published = publish_reader(
                project_dir,
                state=_read_optional_json(state_path),
                final_overrides=final_overrides,
            )
        except Exception:
            if strict:
                raise
            return None
        return _state(state_path, **dict(published))


def _state(path: Path, **values: Any) -> dict[str, Any]:
    previous = normalize_run_root_pdf_state(_read_optional_json(path))
    values = normalize_run_root_pdf_state(values)
    previous_managed_run_pdf = managed_run_root_pdf_path(previous)
    if previous_managed_run_pdf is not None:
        previous["run_pdf_managed_path"] = str(previous_managed_run_pdf)
    incoming_fingerprint = values.get("fingerprint")
    previous_fingerprint = previous.get("fingerprint")
    fingerprint_changed = (
        incoming_fingerprint is not None
        and previous_fingerprint is not None
        and incoming_fingerprint != previous_fingerprint
    )
    fingerprint_unresolved = (
        values.get("status") in {"loading_source", "failed"}
        and incoming_fingerprint is None
    )
    if fingerprint_changed or fingerprint_unresolved:
        previous = {
            key: value
            for key, value in previous.items()
            if not _fingerprint_bound_state_key(key)
        }
    if "notice" in values and values["notice"] is None:
        previous.pop("notice", None)
    state = {
        **previous,
        **{key: value for key, value in values.items() if value is not None},
    }
    if (
        values.get("output_pdf")
        and values.get("output_pdf_sha256")
        and not values.get("output_run_pdf")
    ):
        state.pop("output_run_pdf", None)
        state.pop("output_run_pdf_sha256", None)
    if values.get("output_run_pdf"):
        state["run_pdf_managed_path"] = values["output_run_pdf"]
    published = dict(previous.get("published") or {})
    content_sha256 = values.get("content_sha256")
    if content_sha256:
        published["content_sha256"] = content_sha256
        if values.get("content_object_path"):
            published["content_object_path"] = values["content_object_path"]
    if values.get("output_pdf") and values.get("output_pdf_sha256"):
        published["pdf"] = {
            key: state.get(key)
            for key in (
                "output_tex", "output_pdf", "output_tex_sha256", "output_pdf_sha256",
                "output_run_pdf", "output_run_pdf_sha256",
                "source_manifest_path", "source_manifest_sha256", "validation_path",
                "validation_sha256", "final_render_version",
            )
            if state.get(key) is not None
        }
        if published.get("content_sha256"):
            published["pdf"]["content_sha256"] = published["content_sha256"]
    elif (
        values.get("output_run_pdf")
        and values.get("output_run_pdf_sha256")
        and isinstance(published.get("pdf"), dict)
    ):
        published["pdf"] = {
            **published["pdf"],
            "output_run_pdf": values["output_run_pdf"],
            "output_run_pdf_sha256": values["output_run_pdf_sha256"],
        }
    if values.get("output_html") and values.get("output_html_sha256"):
        published["web"] = {
            key: state.get(key)
            for key in (
                "output_html", "output_html_sha256", "reader_snapshot_path",
                "reader_snapshot_sha256", "web_manifest_path", "web_manifest_sha256",
                "web_render_version",
            )
            if state.get(key) is not None
        }
        if published.get("content_sha256"):
            published["web"]["content_sha256"] = published["content_sha256"]
    if published:
        state["published"] = published
    state["active_run"] = {
        key: value for key, value in state.items()
        if key not in {
            "schema_version", "active_run", "published", "revisions",
            "run_pdf_managed_path",
        }
        and not key.startswith("output_")
    }
    if values.get("status") and values.get("status") not in {"failed", "needs_supervision"}:
        state.pop("error", None)
    state["schema_version"] = "arc.companion.state.v3"
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(path, state)
    from .observability import append_state_event

    append_state_event(path, state)
    return state


def _fingerprint_bound_state_key(key: str) -> bool:
    return key in {
        "fingerprint",
        "checkpoint_dir",
        "segment_count",
        "annotation_language",
        "first_wave_preview_version",
        "chapter_projection_version",
        "augmentation_projection_version",
        "chapter_guide_version",
        "reader_final_checkpoint_version",
        "final_render_version",
        "web_render_version",
        "reader_snapshot_path",
        "reader_snapshot_sha256",
        "web_manifest_path",
        "web_manifest_sha256",
        "web",
    } or key.startswith(
        ("preview_", "output_", "source_manifest_", "validation_")
    )


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = read_json(path)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}
