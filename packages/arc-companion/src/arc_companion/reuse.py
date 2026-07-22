from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .artifact_store import AcceptedArtifactStore, canonical_sha256


REUSE_PLAN_VERSION = "arc.companion.reuse-plan.v1"
LANES = (
    "segmentation", "glossary", "title_translation", "guide", "translation",
    "commentary", "review"
)


_SEMANTIC_FIELDS = {
    "segmentation": ("source", "chapter", "limits"),
    "glossary": (
        "source", "target_language", "index", "protected_names", "intent_guidance",
    ),
    "title_translation": (
        "source_titles", "source_language", "target_language", "glossary",
        "protected_names", "intent_guidance",
    ),
    "guide": (
        "chapter_source", "target_language", "verified_evidence", "intent_guidance",
    ),
    "translation": (
        "source_segment", "target_language", "glossary", "protected_names",
        "intent_guidance",
    ),
    "commentary": (
        "source_segment", "guide", "metadata", "selected_evidence",
        "selected_domain_context", "access_policy", "predecessor_accepted_chain_sha256",
        "intent_guidance",
    ),
    "review": (
        "translation_artifacts", "commentary_artifacts", "review_contract",
        "intent_guidance",
    ),
}


def lane_semantic_input(lane: str, context: Mapping[str, Any]) -> dict[str, Any]:
    """Project only fields that can change accepted content in one lane."""

    try:
        fields = _SEMANTIC_FIELDS[lane]
    except KeyError as exc:
        raise ValueError(f"unsupported generation lane: {lane}") from exc
    projected = {field: context.get(field) for field in fields if field != "intent_guidance"}
    # Preserve every pre-guidance semantic identity byte-for-byte.  The new
    # field exists only for runs that actually generated guidance.
    if "intent_guidance" in fields and context.get("intent_guidance") is not None:
        projected["intent_guidance"] = context["intent_guidance"]
    return projected


def lane_semantic_sha256(lane: str, context: Mapping[str, Any]) -> str:
    return canonical_sha256({"lane": lane, "input": lane_semantic_input(lane, context)})


def lane_recipe_sha256(
    lane: str,
    *,
    prompt: Any,
    model: str | None,
    tier: str | None,
    access_recipe: Any = None,
) -> str:
    if lane not in LANES:
        raise ValueError(f"unsupported generation lane: {lane}")
    return canonical_sha256({
        "lane": lane,
        "prompt": prompt,
        "model": model,
        "tier": tier,
        "access_recipe": access_recipe,
    })


@dataclass(frozen=True)
class ReuseRequest:
    chapter_id: str
    lane: str
    semantic_input_sha256: str
    recipe_sha256: str
    contract_version: str
    segment_id: str | None = None
    reason: str = "no accepted artifact has this semantic identity"


def build_reuse_plan(
    store: AcceptedArtifactStore,
    requests: Iterable[ReuseRequest],
    *,
    regenerate: Iterable[str] = (),
    validators: Mapping[str, Callable[[Any], bool]] | None = None,
) -> dict[str, Any]:
    selected_regeneration = set(regenerate)
    invalid = selected_regeneration.difference(LANES)
    if invalid:
        raise ValueError(f"unsupported regeneration lanes: {', '.join(sorted(invalid))}")
    entries = []
    provider_calls = 0
    for request in requests:
        if request.lane in selected_regeneration:
            status = "miss"
            artifact = None
            reason = "explicitly selected for regeneration"
        else:
            artifact = store.find(
                kind=request.lane,
                semantic_input_sha256=request.semantic_input_sha256,
                recipe_sha256=request.recipe_sha256,
                contract_version=request.contract_version,
                output_validator=(validators or {}).get(request.lane),
            )
            status = str((artifact or {}).get("reuse_status") or "miss")
            reason = (
                "accepted artifact matches semantic input and recipe"
                if status == "hit" else
                "accepted artifact remains valid but its generation recipe changed"
                if status == "recipe_stale" else request.reason
            )
        if status == "miss":
            provider_calls += 1
        entries.append({
            "chapter_id": request.chapter_id,
            "segment_id": request.segment_id,
            "lane": request.lane,
            "status": status,
            "artifact_id": None if artifact is None else artifact["artifact_id"],
            "reason": reason,
            "estimated_provider_calls": 1 if status == "miss" else 0,
        })
    return {
        "schema_version": REUSE_PLAN_VERSION,
        "entries": entries,
        "estimated_provider_calls": provider_calls,
    }


def write_reuse_plan(path: Path, plan: Mapping[str, Any]) -> None:
    # Reuse plans are diagnostic snapshots rather than immutable accepted data.
    from .io import write_json

    write_json(path, dict(plan))
