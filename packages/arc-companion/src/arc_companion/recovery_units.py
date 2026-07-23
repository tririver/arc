from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path
from typing import Any, Callable, Mapping

from .review_arbitration import (
    REVIEW_ARBITRATION_OUTPUT_SCHEMA_VERSION,
)


@dataclass(frozen=True)
class RecoveryUnitSpec:
    unit: str
    owner: str
    validator: str
    ledger_lane: str | None
    application: str = "normal-pipeline-replay.v1"
    side_effect_policy: str = "no_unproven_external_side_effects"


@dataclass(frozen=True)
class PipelineLaneBinding:
    public_lane: str
    recovery_unit: str
    validator: str
    application: str


# This registry is deliberately explicit.  Adding a structured pipeline lane
# without deciding who validates it and whether repeating it can cause an
# external side effect must fail the registry-completeness test.
RECOVERY_UNIT_REGISTRY: dict[str, RecoveryUnitSpec] = {
    "intent-guidance": RecoveryUnitSpec(
        "intent-guidance", "pipeline", "intent-guidance-schema.v1", None,
        "normal-intent-guidance-pipeline-replay.v1",
    ),
    "segmentation": RecoveryUnitSpec(
        "segmentation", "pipeline", "segment-document.v1", None,
        "normal-segmentation-pipeline-replay.v1",
    ),
    "glossary": RecoveryUnitSpec(
        "glossary", "pipeline", "glossary-schema.v1", None,
        "normal-glossary-pipeline-replay.v1",
    ),
    "glossary-index": RecoveryUnitSpec(
        "glossary-index", "pipeline", "glossary-index-schema.v1", None,
        "normal-glossary-index-pipeline-replay.v1",
    ),
    "glossary-consolidation": RecoveryUnitSpec(
        "glossary-consolidation", "pipeline", "glossary-schema.v1", None,
        "normal-glossary-consolidation-pipeline-replay.v1",
    ),
    "title-translation": RecoveryUnitSpec(
        "title-translation", "pipeline", "title-translation-schema.v1", None,
        "normal-title-translation-pipeline-replay.v1",
    ),
    "guide": RecoveryUnitSpec(
        "guide", "chapter-lane-ledger", "chapter-guide-schema+invariants.v1", "guide",
        "normal-guide-pipeline-replay.v1",
    ),
    "translation": RecoveryUnitSpec(
        "translation", "chapter-lane-ledger", "translation-schema+invariants.v1", "translation",
        "normal-translation-pipeline-replay.v1",
    ),
    "translation-token-repair": RecoveryUnitSpec(
        "translation-token-repair", "pipeline",
        "translation-token-repair-schema+invariants.v1", None,
        "normal-translation-token-repair-pipeline-replay.v1",
    ),
    "translation-coverage-repair": RecoveryUnitSpec(
        "translation-coverage-repair", "pipeline",
        "translation-coverage-repair-schema+invariants.v1", None,
        "normal-translation-coverage-repair-pipeline-replay.v1",
    ),
    "companion": RecoveryUnitSpec(
        "companion", "chapter-lane-ledger", "annotation-schema+invariants.v1", "companion",
        "normal-companion-pipeline-replay.v1",
    ),
    "annotation": RecoveryUnitSpec(
        "annotation", "pipeline", "annotation-schema+invariants.v1", None,
        "normal-annotation-pipeline-replay.v1",
    ),
    "section-review": RecoveryUnitSpec(
        "section-review", "pipeline", "section-review-schema.v1", None,
        "normal-section-review-pipeline-replay.v1",
    ),
    "final-review": RecoveryUnitSpec(
        "final-review", "pipeline", "final-review-schema.v1", None,
        "normal-final-review-pipeline-replay.v1",
    ),
    "commentary-review": RecoveryUnitSpec(
        "commentary-review", "pipeline", "commentary-review-schema.v1", None,
        "normal-commentary-review-pipeline-replay.v1",
    ),
    "review-arbitration": RecoveryUnitSpec(
        "review-arbitration", "pipeline",
        "review-arbitration-decision+semantics.v1", None,
        "normal-review-arbitration-pipeline-replay.v1",
    ),
}

PIPELINE_LEDGER_LANES = frozenset({"translation", "companion", "guide"})

# This is the only source for values accepted by the dynamic chapter-lane
# adapter.  Call labels and free-form strings may never select a handler.
PIPELINE_LANE_REGISTRY: dict[str, PipelineLaneBinding] = {
    lane: PipelineLaneBinding(
        public_lane=lane,
        recovery_unit=lane,
        validator=RECOVERY_UNIT_REGISTRY[lane].validator,
        application=RECOVERY_UNIT_REGISTRY[lane].application,
    )
    for lane in sorted(PIPELINE_LEDGER_LANES)
}


def pipeline_lane_binding(lane: str) -> PipelineLaneBinding:
    try:
        return PIPELINE_LANE_REGISTRY[lane]
    except KeyError as exc:
        raise ValueError(f"Unknown dynamic pipeline lane: {lane}") from exc


def submission_descriptor(
    *,
    unit: str,
    logical_unit: str,
    checkpoint_dir: Path,
    artifact_root: Path,
    acceptance_checkpoint: Path,
    input_sha256: str,
    ordered_siblings: list[str],
    suffix: list[str],
    group_sha256: str | None = None,
) -> dict[str, Any]:
    spec = RECOVERY_UNIT_REGISTRY.get(unit)
    if spec is None:
        raise ValueError(f"Unknown recovery unit: {unit}")
    normalized_input_sha256 = str(input_sha256)
    if not (
        len(normalized_input_sha256) == 64
        and all(character in "0123456789abcdef" for character in normalized_input_sha256)
    ):
        normalized_input_sha256 = hashlib.sha256(
            normalized_input_sha256.encode("utf-8")
        ).hexdigest()
    normalized_group_sha256 = str(group_sha256 or normalized_input_sha256)
    if not (
        len(normalized_group_sha256) == 64
        and all(character in "0123456789abcdef" for character in normalized_group_sha256)
    ):
        normalized_group_sha256 = hashlib.sha256(
            normalized_group_sha256.encode("utf-8")
        ).hexdigest()
    return {
        "schema_version": "arc.companion.recovery-call-descriptor.v1",
        "unit": unit,
        "logical_unit": logical_unit,
        "checkpoint_dir": str(checkpoint_dir.absolute()),
        "artifact_root": str(artifact_root.absolute()),
        # Preserve the lexical address so later ownership checks can reject
        # any in-root symlink component instead of silently resolving it.
        "acceptance_checkpoint": str(acceptance_checkpoint.absolute()),
        "input_sha256": normalized_input_sha256,
        "group_sha256": normalized_group_sha256,
        "ordered_siblings": list(ordered_siblings),
        "suffix": list(suffix),
        "validator": spec.validator,
        "application": spec.application,
        "side_effect_policy": spec.side_effect_policy,
        "external_side_effects": False,
    }


def call_model_with_recovery_descriptor(
    call_model: Any,
    prompt: str,
    schema: dict[str, Any],
    artifact_dir: Path,
    call_label: str,
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    """Pass explicit production ownership and fail closed if it is dropped.

    Recovery descriptors are part of the paid-call contract, not optional
    metadata.  A callback that cannot receive the descriptor would create an
    unindexed provider submission which automatic recovery could neither
    identify nor safely replace.
    """

    try:
        parameters = inspect.signature(call_model).parameters.values()
    except (TypeError, ValueError):
        parameters = ()
    if any(
        item.kind == inspect.Parameter.VAR_KEYWORD
        or item.name == "recovery_descriptor"
        for item in parameters
    ):
        return call_model(
            prompt,
            schema,
            artifact_dir,
            call_label,
            recovery_descriptor=dict(descriptor),
        )
    raise TypeError(
        "structured model callback must accept the recovery_descriptor keyword"
    )


def require_control_acceptance(
    accept_recovery: Callable[[Path, str, str], int] | None,
    *,
    checkpoint_dir: Path,
    unit: str,
    logical_unit: str,
) -> None:
    """Require the owning controller to confirm one exact durable acceptance.

    Business modules call this immediately after their complete validator and
    durable application checkpoint.  Tests and standalone library callers may
    omit the callback, but the production pipeline always supplies it.
    """

    if accept_recovery is None:
        return
    accepted = accept_recovery(checkpoint_dir, unit, logical_unit)
    if isinstance(accepted, bool) or accepted != 1:
        raise RuntimeError(
            f"{unit}:{logical_unit} did not reach one exact control acceptance"
        )


def validate_review_arbitration_acceptance_checkpoint(
    value: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> bool:
    """Validate one arbitration decision against its recovery receipt.

    The pipeline remains responsible for applying the decision.  This validator
    owns only the body-light acceptance binding needed for deterministic replay.
    """

    if not isinstance(value, Mapping) or not isinstance(receipt, Mapping):
        return False
    if set(value) != {
        "schema_version",
        "semantic_input_sha256",
        "output_sha256",
        "decision_summaries",
        "validated_output_path",
        "validated_output_sha256",
    }:
        return False
    semantic_input_sha256 = value.get("semantic_input_sha256")
    receipt_input_sha256 = receipt.get("input_sha256")
    if (
        not isinstance(semantic_input_sha256, str)
        or len(semantic_input_sha256) != 64
        or any(character not in "0123456789abcdef"
               for character in semantic_input_sha256)
        or semantic_input_sha256 != receipt_input_sha256
    ):
        return False
    output_sha256 = value.get("output_sha256")
    validated_output_sha256 = value.get("validated_output_sha256")
    validated_output_path = value.get("validated_output_path")
    summaries = value.get("decision_summaries")
    if not (
        _is_sha256(output_sha256)
        and _is_sha256(validated_output_sha256)
        and isinstance(validated_output_path, str)
        and validated_output_path
        and not Path(validated_output_path).is_absolute()
        and ".." not in Path(validated_output_path).parts
        and isinstance(summaries, list)
    ):
        return False
    for item in summaries:
        if (
            not isinstance(item, Mapping)
            or set(item) != {
                "path",
                "action",
                "selected_candidate_hashes",
                "replacement_sha256",
                "origin_hashes",
                "reason",
            }
            or not isinstance(item.get("path"), str)
            or item.get("action") not in {
                "select_candidate",
                "merge_candidates",
                "keep_original",
                "unresolved",
            }
            or not isinstance(item.get("reason"), str)
            or not isinstance(item.get("selected_candidate_hashes"), list)
            or not all(
                _is_sha256(candidate_hash)
                for candidate_hash in item["selected_candidate_hashes"]
            )
            or len(item["selected_candidate_hashes"])
            != len(set(item["selected_candidate_hashes"]))
            or not isinstance(item.get("origin_hashes"), list)
            or not all(_is_sha256(value) for value in item["origin_hashes"])
            or (
                item.get("replacement_sha256") is not None
                and not _is_sha256(item.get("replacement_sha256"))
            )
        ):
            return False
    return (
        value.get("schema_version")
        == REVIEW_ARBITRATION_OUTPUT_SCHEMA_VERSION
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def recovery_unit_for_ledger(lane: str) -> RecoveryUnitSpec | None:
    direct = RECOVERY_UNIT_REGISTRY.get(lane)
    if direct is not None:
        return direct
    return next((
        spec for spec in RECOVERY_UNIT_REGISTRY.values()
        if spec.ledger_lane == lane
    ), None)
