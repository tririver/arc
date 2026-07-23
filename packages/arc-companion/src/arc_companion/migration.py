from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .artifact_ids import (
    ARTIFACT_ID_RECEIPT_NAME,
    ArtifactIdError,
    resolve_artifact_dir,
)
from .artifact_store import AcceptedArtifactStore, ArtifactStoreError, canonical_sha256
from .content import (
    ContentBundleError,
    checkpoint_receipts,
    reader_content_from_overrides,
    store_reader_content,
)
from .io import sha256_file, sha256_json
from .ledger import LANE_LEDGER_VERSION
from .projection import (
    is_structural, is_translatable, opaque_inline_tokens, translation_input_block,
)
from .source import block_id


MIGRATION_VERSION = "arc.companion.legacy-migration.v1"
NEVER_MIGRATED_ARTIFACTS = (
    "tex",
    "pdf",
)
MIGRATION_CANDIDATE_ARTIFACTS = ("guide", "translation", "commentary", "review")
MIGRATABLE_LEDGER_VERSIONS = (
    "arc.companion.chapter-lane-ledger.v1",
    LANE_LEDGER_VERSION,
)


class LegacyMigrationError(ValueError):
    """The legacy checkpoint is malformed, never a request to rerun work."""


def read_legacy_checkpoint(path: Path) -> dict[str, Any]:
    """Read a legacy checkpoint without modifying it or resolving linked files."""

    if path.is_dir():
        return _read_legacy_checkpoint_dir(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LegacyMigrationError(f"could not read legacy checkpoint {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LegacyMigrationError("legacy checkpoint must be a JSON object")
    return value


def _read_legacy_checkpoint_dir(path: Path) -> dict[str, Any]:
    """Load only migration-eligible artifacts from a legacy run directory."""

    def optional(name: str) -> dict[str, Any]:
        candidate = path / name
        if not candidate.is_file():
            return {}
        value = read_legacy_checkpoint(candidate)
        return value

    segmentation = optional("segmentation.json")
    glossary = optional("glossary.json")
    document = optional("document.json")
    metadata = optional("migration-metadata.json")
    translations: dict[str, Any] = {}
    translation_dir = path / "translations"
    if translation_dir.is_dir():
        segment_blocks = {
            str(item.get("segment_id") or ""): list(item.get("block_ids") or [])
            for item in segmentation.get("segments") or []
            if isinstance(item, Mapping)
        }
        for candidate in sorted(translation_dir.glob("*.json")):
            value = read_legacy_checkpoint(candidate)
            key = str(value.get("segment_id") or candidate.stem)
            if not value.get("block_ids") and segment_blocks.get(key):
                value["block_ids"] = segment_blocks[key]
            translations[key] = value
    return {
        "schema_version": "arc.companion.legacy-checkpoint-view.v1",
        "source_hash": str(document.get("source_hash") or metadata.get("source_hash") or ""),
        "language": str(metadata.get("language") or glossary.get("language") or ""),
        "prompt_hash": str(metadata.get("prompt_hash") or ""),
        "validator_hash": str(metadata.get("validator_hash") or ""),
        "segmentation": segmentation,
        "cuts": list(segmentation.get("cuts") or []),
        "glossary": glossary or None,
        "translations": translations,
        "metadata": metadata,
    }


def plan_legacy_migration(
    legacy: Mapping[str, Any],
    *,
    document: Mapping[str, Any],
    chapters: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    segments: Iterable[Mapping[str, Any]],
    source_hash: str,
    language: str,
    prompt_hash: str,
    validator_hash: str,
    glossary: Mapping[str, Any],
    protected_names: Iterable[str] = (),
    index_entries: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
    max_segment_blocks: int = 24,
    max_segment_source_chars: int = 60_000,
) -> dict[str, Any]:
    """Build a read-only migration plan for a new chaptered run.

    The return value contains candidate cuts, a reusable glossary or ``None``,
    and ready-to-write translation lane ledgers.  This function performs no
    filesystem writes and never invokes a model.
    """

    blocks = [dict(item) for item in document.get("blocks") or [] if isinstance(item, Mapping)]
    if not blocks:
        raise LegacyMigrationError("current rich document contains no blocks")
    chapter_list = list(chapters.get("chapters") or []) if isinstance(chapters, Mapping) else list(chapters)
    segment_list = [dict(item) for item in segments]
    metadata = _legacy_metadata(legacy)
    cut_plan = migrate_legacy_cuts(
        legacy.get("cuts") or (legacy.get("segmentation") or {}).get("cuts") or [],
        blocks=blocks,
        chapters=chapter_list,
        max_segment_blocks=max_segment_blocks,
        max_segment_source_chars=max_segment_source_chars,
    )
    glossary_result = migrate_legacy_glossary(
        legacy.get("glossary"),
        metadata=metadata,
        source_hash=source_hash,
        language=language,
        prompt_hash=prompt_hash,
        validator_hash=validator_hash,
        index_entries=index_entries,
    )
    translations = _translations_with_segment_blocks(
        legacy.get("translations") or {}, legacy.get("segmentation")
    )
    translation_plan = migrate_legacy_translations(
        translations,
        metadata=metadata,
        blocks=blocks,
        chapters=chapter_list,
        segments=segment_list,
        source_hash=source_hash,
        language=language,
        glossary=glossary,
        protected_names=list(protected_names),
    )
    return {
        "schema_version": MIGRATION_VERSION,
        "source_checkpoint_sha256": sha256_json(legacy),
        "read_only_source": True,
        "cuts": cut_plan,
        "glossary": glossary_result,
        "translations": translation_plan,
        "never_migrated": list(NEVER_MIGRATED_ARTIFACTS),
    }


def legacy_translation_candidates(legacy: Mapping[str, Any]) -> Any:
    """Return translation candidates enriched only with legacy segment ownership."""
    return _translations_with_segment_blocks(
        legacy.get("translations") or {}, legacy.get("segmentation")
    )


def import_accepted_checkpoint_objects(
    project_dir: Path,
    *,
    validators: Mapping[str, Callable[[Any], bool]],
    contract_versions: Mapping[str, str],
) -> dict[str, Any]:
    """Strictly import accepted lane artifacts from every old fingerprint.

    No legacy provider session or non-accepted result is considered. Every
    imported value must be tied to a valid accepted ledger chain, match its
    recorded output hash, and pass the caller's current deterministic contract.
    The function makes no provider calls and never modifies old checkpoints.
    """

    store = AcceptedArtifactStore(project_dir)
    checkpoint_root = project_dir.resolve() / ".arc-companion" / "checkpoints"
    receipts: list[dict[str, Any]] = []
    if not checkpoint_root.is_dir():
        return {
            "schema_version": "arc.companion.object-migration.v1",
            "provider_calls": 0,
            "receipts": receipts,
        }
    for checkpoint in sorted(
        path for path in checkpoint_root.iterdir()
        if path.is_dir() and not path.is_symlink() and path.name != "aliases"
    ):
        try:
            _checkpoint_full_identity(checkpoint_root, checkpoint)
        except ArtifactIdError:
            continue
        metadata = _optional_object(checkpoint / "migration-metadata.json")
        recipe_sha = canonical_sha256({
            "legacy_checkpoint_recipe": {
                key: metadata.get(key)
                for key in ("prompt_hash", "validator_hash", "provider", "model")
            }
        })
        for ledger_path in sorted((checkpoint / "chapters").glob("*/*-ledger.json")):
            receipts.extend(_import_accepted_ledger(
                checkpoint,
                ledger_path,
                store=store,
                validators=validators,
                contract_versions=contract_versions,
                recipe_sha256=recipe_sha,
                metadata=metadata,
            ))
        overlays = sorted((checkpoint / "chapters").glob("*/*-review-overlay.json"))
        receipts.extend(_review_overlay_receipt(checkpoint, path) for path in overlays)
        review_path = checkpoint / "chapter-review.json"
        if review_path.is_file() and not overlays:
            receipts.append({
                "checkpoint": _checkpoint_full_identity(
                    checkpoint_root, checkpoint,
                ),
                "lane": "review",
                "accepted": False,
                "reason": "review_response_unbound_to_base_artifacts",
                "source_path": str(review_path),
            })
        reader_path = checkpoint / "reader-final.json"
        if reader_path.is_file():
            receipts.append(_reader_final_receipt(
                project_dir.resolve(), checkpoint, reader_path,
            ))
    return {
        "schema_version": "arc.companion.object-migration.v1",
        "provider_calls": 0,
        "receipts": receipts,
        "imported_artifact_ids": [
            item["artifact_id"] for item in receipts
            if item.get("accepted") and item.get("artifact_id")
        ],
        "imported_content_sha256": [
            item["content_sha256"] for item in receipts
            if item.get("accepted") and item.get("content_sha256")
        ],
    }


def _import_accepted_ledger(
    checkpoint: Path,
    ledger_path: Path,
    *,
    store: AcceptedArtifactStore,
    validators: Mapping[str, Callable[[Any], bool]],
    contract_versions: Mapping[str, str],
    recipe_sha256: str,
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    ledger = _optional_object(ledger_path)
    legacy_lane = str(ledger.get("lane") or "")
    lane = "commentary" if legacy_lane == "companion" else legacy_lane
    base = {
        "checkpoint": _checkpoint_full_identity(
            checkpoint.parent, checkpoint,
        ),
        "chapter_id": str(ledger.get("chapter_id") or ledger_path.parent.name),
        "lane": lane,
        "ledger_path": str(ledger_path),
    }
    if (
        ledger.get("schema_version") not in MIGRATABLE_LEDGER_VERSIONS
        or lane not in MIGRATION_CANDIDATE_ARTIFACTS
    ):
        return [{**base, "accepted": False, "reason": "unsupported_ledger_identity"}]
    validator = validators.get(lane)
    contract = str(contract_versions.get(lane) or "")
    if validator is None or not contract:
        return [{**base, "accepted": False, "reason": "current_contract_unavailable"}]
    expected_predecessor = hashlib.sha256(b"").hexdigest()
    receipts = []
    accepted_prefix_open = True
    for block in ledger.get("blocks") or []:
        segment_id = str(block.get("segment_id") or "")
        item = {**base, "segment_id": segment_id}
        if not accepted_prefix_open or block.get("state") != "accepted":
            accepted_prefix_open = False
            receipts.append({**item, "accepted": False, "reason": "ledger_state_not_accepted"})
            continue
        reason = _validate_accepted_ledger_block(block, expected_predecessor)
        if reason:
            accepted_prefix_open = False
            receipts.append({**item, "accepted": False, "reason": reason})
            continue
        expected_predecessor = str(block["accepted_chain_sha256"])
        candidate = _legacy_lane_candidate(
            checkpoint, lane=lane, chapter_id=base["chapter_id"], segment_id=segment_id,
            output_sha256=str(block.get("output_sha256") or ""),
        )
        if candidate is None:
            receipts.append({**item, "accepted": False, "reason": "output_checkpoint_not_proven"})
            continue
        try:
            valid = validator(candidate["output"])
        except Exception:
            valid = False
        if not valid:
            receipts.append({**item, "accepted": False, "reason": "current_contract_rejected"})
            continue
        try:
            record = store.put_accepted(
                kind=lane,
                semantic_input_sha256=str(block["input_sha256"]),
                recipe_sha256=recipe_sha256,
                contract_version=contract,
                output=candidate["output"],
                ledger_block=block,
                provider_receipt={
                    "provider": str(metadata.get("provider") or "legacy-provider-not-recorded"),
                    "model": str(metadata.get("model") or "legacy-model-not-recorded"),
                    "call_id": str(
                        (block.get("logical_receipt") or {}).get("idempotency_key")
                        or (block.get("logical_receipt") or {}).get("call_id")
                        or (
                            "legacy:"
                            f"{_checkpoint_full_identity(checkpoint.parent, checkpoint)}"
                            f":{lane}:{segment_id}"
                        )
                    ),
                    "usage": {"availability": "not_recorded_in_legacy_checkpoint"},
                },
                provenance={
                    "migration_version": MIGRATION_VERSION,
                    "checkpoint_dir": str(checkpoint),
                    "ledger": str(ledger_path),
                    "output_checkpoint": candidate["source_path"],
                    "legacy_lane": legacy_lane,
                },
            )
        except ArtifactStoreError as exc:
            receipts.append({**item, "accepted": False, "reason": "object_store_rejected", "detail": str(exc)})
            continue
        receipts.append({
            **item,
            "accepted": True,
            "reason": "accepted_ledger_and_current_contract_valid",
            "artifact_id": record["artifact_id"],
            "source_path": candidate["source_path"],
        })
    return receipts


def _validate_accepted_ledger_block(block: Mapping[str, Any], predecessor: str) -> str | None:
    if str(block.get("predecessor_accepted_chain_sha256") or "") != predecessor:
        return "accepted_chain_predecessor_mismatch"
    input_sha = str(block.get("input_sha256") or "")
    output_sha = str(block.get("output_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", input_sha) or not re.fullmatch(r"[0-9a-f]{64}", output_sha):
        return "input_or_output_hash_invalid"
    validation = block.get("validation_receipt")
    logical = block.get("logical_receipt")
    if not isinstance(validation, Mapping) or not validation:
        return "validation_receipt_missing"
    if not isinstance(logical, Mapping) or not logical:
        return "logical_receipt_missing"
    expected_chain = hashlib.sha256(json.dumps({
        "predecessor": predecessor,
        "segment_id": block.get("segment_id"),
        "input_sha256": input_sha,
        "output_sha256": output_sha,
        "generation": block.get("generation"),
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    if block.get("accepted_chain_sha256") != expected_chain:
        return "accepted_chain_hash_mismatch"
    return None


def _legacy_lane_candidate(
    checkpoint: Path,
    *,
    lane: str,
    chapter_id: str,
    segment_id: str,
    output_sha256: str,
) -> dict[str, Any] | None:
    candidates: list[tuple[Path, str | None]] = []
    if lane == "commentary":
        candidates.extend((path, "annotation") for path in (checkpoint / "annotations").glob("*.json"))
    elif lane == "translation":
        candidates.extend((path, "translation") for path in (checkpoint / "translations").glob("*.json"))
        candidates.extend((path, "translation") for path in (checkpoint / "translation-drafts").glob("*.json"))
    elif lane == "guide":
        candidates.append((checkpoint / "chapters" / chapter_id / "chapter-guide.json", None))
    for path, output_key in candidates:
        value = _optional_object(path)
        if not value:
            continue
        if lane != "guide" and str(value.get("segment_id") or "") != segment_id:
            continue
        output = value.get(output_key) if output_key else value
        if canonical_sha256(output) == output_sha256:
            return {"output": output, "source_path": str(path)}
    return None


def _optional_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _checkpoint_full_identity(root: Path, checkpoint: Path) -> str:
    """Return a receipt-bound identity, with exact full-name legacy fallback."""

    receipt = checkpoint / ARTIFACT_ID_RECEIPT_NAME
    if receipt.exists() or receipt.is_symlink():
        return resolve_artifact_dir(
            root,
            checkpoint,
            kind="checkpoint",
            allow_legacy=False,
        ).identity
    if re.fullmatch(r"[0-9a-f]{64}", checkpoint.name):
        return resolve_artifact_dir(
            root,
            checkpoint,
            expected_identity=checkpoint.name,
            kind="checkpoint",
        ).identity
    raise ArtifactIdError("legacy checkpoint identity is unavailable")


def _review_overlay_receipt(checkpoint: Path, path: Path) -> dict[str, Any]:
    overlay = _optional_object(path)
    lane = str(overlay.get("lane") or "")
    ledger_path = path.with_name(f"{lane}-ledger.json")
    ledger = _optional_object(ledger_path)
    base = {
        "checkpoint": _checkpoint_full_identity(
            checkpoint.parent, checkpoint,
        ),
        "chapter_id": str(overlay.get("chapter_id") or path.parent.name),
        "lane": "review",
        "reviewed_lane": "commentary" if lane == "companion" else lane,
        "source_path": str(path),
    }
    if overlay.get("schema_version") != "arc.companion.chapter-review-overlay.v1":
        return {**base, "accepted": False, "reason": "review_overlay_schema_invalid"}
    blocks = (
        ledger.get("blocks")
        if ledger.get("schema_version") in MIGRATABLE_LEDGER_VERSIONS else None
    )
    if not isinstance(blocks, list) or not blocks:
        return {**base, "accepted": False, "reason": "review_base_ledger_missing"}
    by_segment = {str(item.get("segment_id") or ""): item for item in blocks}
    if overlay.get("base_accepted_chain_sha256") != ledger.get("accepted_chain_sha256"):
        return {**base, "accepted": False, "reason": "review_base_chain_mismatch"}
    reviewed_blocks = overlay.get("blocks")
    if not isinstance(reviewed_blocks, list) or set(by_segment) != {
        str(item.get("segment_id") or "") for item in reviewed_blocks if isinstance(item, Mapping)
    }:
        return {**base, "accepted": False, "reason": "review_segment_coverage_mismatch"}
    for reviewed in reviewed_blocks:
        source = by_segment[str(reviewed.get("segment_id") or "")]
        if (
            source.get("state") != "accepted"
            or reviewed.get("base_output_sha256") != source.get("output_sha256")
            or reviewed.get("accepted_chain_sha256") != source.get("accepted_chain_sha256")
            or not re.fullmatch(r"[0-9a-f]{64}", str(reviewed.get("reviewed_output_sha256") or ""))
        ):
            return {**base, "accepted": False, "reason": "review_base_artifact_binding_invalid"}
    # Old overlays prove the base binding but do not contain the applied output.
    # They are useful provenance, never reusable review content by themselves.
    return {
        **base,
        "accepted": False,
        "base_binding_valid": True,
        "reason": "reviewed_output_checkpoint_missing",
        "overlay_sha256": canonical_sha256(overlay),
    }


def _reader_final_receipt(project_dir: Path, checkpoint: Path, path: Path) -> dict[str, Any]:
    value = _optional_object(path)
    base = {
        "checkpoint": _checkpoint_full_identity(
            checkpoint.parent, checkpoint,
        ),
        "lane": "reader-content",
        "source_path": str(path),
    }
    overrides = value.get("final_overrides")
    if not isinstance(overrides, Mapping):
        return {**base, "accepted": False, "reason": "reader_final_payload_missing"}
    chains, overlays = checkpoint_receipts(checkpoint)
    if not chains or not overlays:
        return {
            **base, "accepted": False,
            "reason": "reader_final_accepted_receipts_missing",
        }
    reviewed_hashes: dict[tuple[str, str], str] = {}
    for overlay_path in sorted((checkpoint / "chapters").glob("*/*-review-overlay.json")):
        overlay = _optional_object(overlay_path)
        reviewed_lane = "commentary" if overlay.get("lane") == "companion" else str(overlay.get("lane") or "")
        receipt = _review_overlay_receipt(checkpoint, overlay_path)
        if not receipt.get("base_binding_valid"):
            return {**base, "accepted": False, "reason": "reader_final_review_binding_invalid"}
        for block in overlay.get("blocks") or []:
            if isinstance(block, Mapping):
                reviewed_hashes[(reviewed_lane, str(block.get("segment_id") or ""))] = str(
                    block.get("reviewed_output_sha256") or ""
                )
    annotations = overrides.get("annotations")
    if not isinstance(annotations, Mapping) or any(
        canonical_sha256(output) != reviewed_hashes.get(("commentary", str(segment_id)))
        for segment_id, output in annotations.items()
    ):
        return {**base, "accepted": False, "reason": "reader_final_commentary_not_review_bound"}
    translations = overrides.get("translations")
    if translations is not None and (
        not isinstance(translations, Mapping) or any(
            canonical_sha256(output) != reviewed_hashes.get(("translation", str(segment_id)))
            for segment_id, output in translations.items()
        )
    ):
        return {**base, "accepted": False, "reason": "reader_final_translation_not_review_bound"}
    segment_ids = [
        str(item.get("segment_id") or "")
        for item in overrides.get("segments") or []
        if isinstance(item, Mapping)
    ]
    reader_evidence = {
        segment_id: {
            "commentary_sources": list(
                ((annotations or {}).get(segment_id) or {}).get("commentary_sources") or []
            )
        }
        for segment_id in segment_ids
    }
    content = reader_content_from_overrides(
        overrides,
        reader_evidence_by_segment=reader_evidence,
        accepted_ledger_chains=chains,
        review_overlay_hashes=overlays,
    )
    try:
        stored = store_reader_content(
            project_dir,
            content=content,
            checkpoint_dir=checkpoint,
            review_receipts={
                "legacy_reader_final": {
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            },
        )
    except (ContentBundleError, OSError, ValueError) as exc:
        return {
            **base,
            "accepted": False,
            "reason": "reader_final_current_contract_rejected",
            "detail": str(exc),
        }
    return {
        **base,
        "accepted": True,
        "reason": "reader_final_deterministically_revalidated",
        "content_sha256": stored["content_sha256"],
    }


def migrate_legacy_cuts(
    cuts: Iterable[Any],
    *,
    blocks: list[Mapping[str, Any]],
    chapters: Iterable[Mapping[str, Any]],
    max_segment_blocks: int,
    max_segment_source_chars: int,
) -> dict[str, Any]:
    block_ids = [block_id(dict(item)) for item in blocks]
    values = list(cuts)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        return {"reused": {}, "receipts": [{"accepted": False, "reason": "cuts_not_integers"}]}
    if len(values) != len(set(values)) or any(value < 1 or value >= len(blocks) for value in values):
        return {"reused": {}, "receipts": [{"accepted": False, "reason": "cuts_invalid_or_duplicate"}]}
    boundaries = [0, *sorted(values), len(blocks)]
    old_ranges = [block_ids[start:end] for start, end in zip(boundaries, boundaries[1:])]
    by_id = {block_id(dict(item)): dict(item) for item in blocks}
    reused: dict[str, list[int]] = {}
    receipts: list[dict[str, Any]] = []
    for chapter in chapters:
        chapter_id = str(chapter.get("chapter_id") or "")
        expected = [str(value) for value in chapter.get("block_ids") or []]
        expected_set = set(expected)
        intersecting = [group for group in old_ranges if expected_set.intersection(group)]
        if not expected or any(not set(group).issubset(expected_set) for group in intersecting):
            receipts.append({"chapter_id": chapter_id, "accepted": False, "reason": "legacy_segment_crosses_chapter"})
            continue
        covered = [value for group in intersecting for value in group]
        if covered != expected:
            receipts.append({"chapter_id": chapter_id, "accepted": False, "reason": "chapter_coverage_mismatch"})
            continue
        too_large = next((group for group in intersecting if not _segment_size_ok(
            group,
            by_id,
            max_blocks=max_segment_blocks,
            max_chars=max_segment_source_chars,
        )), None)
        if too_large is not None:
            receipts.append({"chapter_id": chapter_id, "accepted": False, "reason": "legacy_segment_exceeds_limits"})
            continue
        relative = []
        cursor = 0
        for group in intersecting[:-1]:
            cursor += len(group)
            relative.append(cursor)
        reused[chapter_id] = relative
        receipts.append({
            "chapter_id": chapter_id,
            "accepted": True,
            "reason": "exact_chapter_coverage_and_size_valid",
            "cut_after_chapter_ordinals": relative,
        })
    return {"reused": reused, "receipts": receipts}


def migrate_legacy_glossary(
    glossary: Any,
    *,
    metadata: Mapping[str, Any],
    source_hash: str,
    language: str,
    prompt_hash: str,
    validator_hash: str,
    index_entries: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    if _has_real_index(index_entries):
        return {"accepted": False, "reason": "real_index_requires_complete_index_glossary", "value": None}
    if not isinstance(glossary, Mapping):
        return {"accepted": False, "reason": "legacy_glossary_missing", "value": None}
    expected = {
        "source_hash": source_hash,
        "language": language,
        "prompt_hash": prompt_hash,
        "validator_hash": validator_hash,
    }
    actual = {
        key: str(glossary.get(key) or metadata.get(key) or "")
        for key in expected
    }
    mismatches = [key for key, value in expected.items() if actual[key] != str(value)]
    if mismatches:
        return {
            "accepted": False,
            "reason": "glossary_identity_mismatch",
            "mismatched_fields": mismatches,
            "value": None,
        }
    return {"accepted": True, "reason": "all_glossary_hashes_match", "value": dict(glossary)}


def migrate_legacy_translations(
    translations: Any,
    *,
    metadata: Mapping[str, Any],
    blocks: list[Mapping[str, Any]],
    chapters: Iterable[Mapping[str, Any]],
    segments: list[Mapping[str, Any]],
    source_hash: str,
    language: str,
    glossary: Mapping[str, Any],
    protected_names: list[str],
    segment_input_hash: Callable[[Mapping[str, Any]], str] | None = None,
    migration_source: str = "legacy_checkpoint",
) -> dict[str, Any]:
    candidates = _translation_candidates(translations)
    blocks_by_id = {block_id(dict(item)): dict(item) for item in blocks}
    block_candidates, structural_outputs = _translation_block_candidate_index(
        candidates, metadata=metadata, blocks_by_id=blocks_by_id,
    )
    segments_by_chapter: dict[str, list[Mapping[str, Any]]] = {}
    for segment in segments:
        segments_by_chapter.setdefault(str(segment.get("chapter_id") or ""), []).append(segment)
    receipts: list[dict[str, Any]] = []
    ledgers: dict[str, dict[str, Any]] = {}
    for chapter in chapters:
        chapter_id = str(chapter.get("chapter_id") or "")
        ordered = segments_by_chapter.get(chapter_id, [])
        accepted_prefix = True
        ledger_blocks: list[dict[str, Any]] = []
        chain = _hash("")
        for segment in ordered:
            segment_id = str(segment.get("segment_id") or "")
            composed = _compose_translation_candidate(
                segment,
                source_hash=source_hash,
                language=language,
                blocks_by_id=blocks_by_id,
                block_candidates=block_candidates,
                structural_outputs=structural_outputs,
            )
            if composed is None:
                receipt = {
                    "segment_id": segment_id,
                    "accepted": False,
                    "reason": "translation_missing_or_ambiguous",
                }
            else:
                receipt = _validate_translation_candidate(
                    composed,
                    segment=segment,
                    blocks_by_id=blocks_by_id,
                    metadata=metadata,
                    source_hash=source_hash,
                    language=language,
                    glossary=glossary,
                    protected_names=protected_names,
                    segment_input_hash=segment_input_hash,
                )
            locally_valid = bool(receipt.get("accepted"))
            deferred = locally_valid and not accepted_prefix
            if deferred:
                receipt = {
                    **receipt,
                    "accepted": False,
                    "status": "deferred_hit",
                    "reason": "deferred_hit",
                }
            receipts.append({"chapter_id": chapter_id, **receipt})
            if locally_valid and accepted_prefix:
                translation = receipt["translation"]
                input_sha = str(receipt["input_sha256"])
                output_sha = sha256_json(translation)
                validation_receipt = _translation_migration_validation_receipt(receipt)
                block_record = {
                    "segment_id": segment_id,
                    "state": "accepted",
                    "submission_state": "not_submitted",
                    "generation": 1,
                    "input_sha256": input_sha,
                    "output_sha256": output_sha,
                    "logical_receipt": {"kind": "legacy_migration", "provider_calls": 0},
                    "validation_receipt": validation_receipt,
                    "predecessor_accepted_chain_sha256": chain,
                }
                chain = _hash(json.dumps({
                    "predecessor": chain,
                    "segment_id": segment_id,
                    "input_sha256": input_sha,
                    "output_sha256": output_sha,
                    "generation": 1,
                }, sort_keys=True, separators=(",", ":")))
                block_record["accepted_chain_sha256"] = chain
                block_record["translation"] = translation
                ledger_blocks.append(block_record)
            elif deferred:
                translation = receipt["translation"]
                ledger_blocks.append({
                    "segment_id": segment_id,
                    "state": "prepared",
                    "submission_state": "not_submitted",
                    "generation": 1,
                    "deferred_translation": translation,
                    "deferred_input_sha256": str(receipt["input_sha256"]),
                    "deferred_output_sha256": sha256_json(translation),
                    "deferred_logical_receipt": {
                        "kind": "legacy_migration_deferred",
                        "provider_calls": 0,
                    },
                    "deferred_validation_receipt": (
                        _translation_migration_validation_receipt(receipt)
                    ),
                })
            else:
                ledger_blocks.append({
                    "segment_id": segment_id,
                    "state": "prepared",
                    "submission_state": "not_submitted",
                    "generation": 1,
                })
            accepted_prefix = accepted_prefix and locally_valid
        ledgers[chapter_id] = {
            "schema_version": LANE_LEDGER_VERSION,
            "chapter_id": chapter_id,
            "lane": "translation",
            "generation": 1,
            "needs_supervision": None,
            "blocks": ledger_blocks,
            "accepted_chain_sha256": chain,
            "migration_source": migration_source,
            "updated_at": 0.0,
        }
    return {"ledgers": ledgers, "receipts": receipts}


def _translation_block_candidate_index(
    candidates: list[dict[str, Any]],
    *,
    metadata: Mapping[str, Any],
    blocks_by_id: Mapping[str, dict[str, Any]],
) -> tuple[
    dict[tuple[str, str, str], list[dict[str, Any]]],
    dict[tuple[str, str], set[str]],
]:
    """Index accepted body translations independently of their old segment cuts."""

    by_block: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    structural_outputs: dict[tuple[str, str], set[str]] = {}
    for ordinal, candidate in enumerate(candidates):
        candidate_source = str(
            candidate.get("source_hash") or metadata.get("source_hash") or ""
        )
        candidate_language = str(
            candidate.get("language") or metadata.get("language") or ""
        )
        translation = (
            candidate.get("translation")
            if isinstance(candidate.get("translation"), Mapping)
            else candidate
        )
        raw_blocks = translation.get("blocks") if isinstance(translation, Mapping) else None
        if not candidate_source or not candidate_language or not isinstance(raw_blocks, list):
            continue
        identity = (candidate_source, candidate_language)
        candidate_block_ids = [
            str(value) for value in candidate.get("block_ids") or []
        ]
        for raw in raw_blocks:
            if not isinstance(raw, Mapping):
                continue
            identifier = str(raw.get("block_id") or "")
            source_block = blocks_by_id.get(identifier)
            if not identifier or source_block is None:
                continue
            if is_structural(source_block):
                structural_outputs.setdefault(identity, set()).add(identifier)
                continue
            by_block.setdefault((*identity, identifier), []).append({
                "block": dict(raw),
                "candidate_ordinal": ordinal,
                "candidate_block_ids": candidate_block_ids,
                "legacy_segment_id": str(candidate.get("legacy_segment_id") or ""),
                "accepted_artifact_id": str(candidate.get("accepted_artifact_id") or ""),
                "created_at": candidate.get("created_at"),
            })
    return by_block, structural_outputs


def _compose_translation_candidate(
    segment: Mapping[str, Any],
    *,
    source_hash: str,
    language: str,
    blocks_by_id: Mapping[str, dict[str, Any]],
    block_candidates: Mapping[tuple[str, str, str], list[dict[str, Any]]],
    structural_outputs: Mapping[tuple[str, str], set[str]],
) -> dict[str, Any] | None:
    expected_ids = [
        str(value)
        for value in segment.get("block_ids") or []
        if str(value) in blocks_by_id and is_translatable(blocks_by_id[str(value)])
    ]
    selected: list[dict[str, Any]] = []
    for identifier in expected_ids:
        matches = list(block_candidates.get((source_hash, language, identifier), []))
        match = _select_translation_block_candidate(matches)
        if match is None:
            return None
        selected.append(match)

    selected_origins = {
        (
            item["candidate_ordinal"],
            item["legacy_segment_id"],
            item["accepted_artifact_id"],
        )
        for item in selected
    }
    exact_range = False
    if len(selected_origins) == 1 and selected:
        candidate_ids = selected[0]["candidate_block_ids"]
        projected_ids = [
            identifier for identifier in candidate_ids
            if identifier in blocks_by_id and is_translatable(blocks_by_id[identifier])
        ]
        exact_range = projected_ids == expected_ids
    dropped_structural_ids = [
        str(value) for value in segment.get("block_ids") or []
        if str(value) in structural_outputs.get((source_hash, language), set())
    ]
    return {
        "source_hash": source_hash,
        "language": language,
        "translation": {"blocks": [dict(item["block"]) for item in selected]},
        "reuse_status": "hit" if exact_range else "composed_hit",
        "dropped_structural_block_ids": dropped_structural_ids,
        "accepted_artifact_ids": sorted({
            item["accepted_artifact_id"] for item in selected
            if item["accepted_artifact_id"]
        }),
    }


def _select_translation_block_candidate(
    matches: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not matches:
        return None
    by_output: dict[str, list[dict[str, Any]]] = {}
    for item in matches:
        by_output.setdefault(sha256_json(item["block"]), []).append(item)
    if len(by_output) == 1:
        return max(matches, key=lambda item: int(item["candidate_ordinal"]))
    dated = [
        item for item in matches
        if item.get("accepted_artifact_id") and isinstance(item.get("created_at"), (int, float))
    ]
    if len(dated) == len(matches):
        newest = max(float(item["created_at"]) for item in dated)
        latest = [item for item in dated if float(item["created_at"]) == newest]
        if len(latest) == 1:
            return latest[0]
    return None


def _translation_migration_validation_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    terminology_warnings = list(receipt.get("terminology_warnings") or [])
    return {
        "schema_version": MIGRATION_VERSION,
        "source": True,
        "language": True,
        "terminology": not terminology_warnings,
        "terminology_warnings": terminology_warnings,
        "opaque_tokens": True,
        "protected_names": True,
        "reuse_status": str(receipt.get("status") or "hit"),
    }


def accepted_translation_projection_candidates(
    store: AcceptedArtifactStore,
    *,
    source_hash: str | None = None,
    language: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Recover the newest accepted translation for each preserved source range.

    Artifact semantic hashes can evolve when projection contracts change. This
    read-only bridge retains the accepted object and revalidates its body-only
    projection through ``migrate_legacy_translations`` under the current source,
    language, glossary, and protected-name contracts.
    """
    selected: dict[tuple[str, str, tuple[str, ...]], tuple[float, dict[str, Any]]] = {}
    for record in store.iter_kind("translation"):
        provenance = record.get("provenance")
        if not isinstance(provenance, Mapping):
            continue
        checkpoint_text = str(provenance.get("checkpoint_dir") or "")
        checkpoint = Path(checkpoint_text)
        if not checkpoint.is_dir():
            continue
        metadata = _optional_object(checkpoint / "migration-metadata.json")
        candidate_source_hash = str(metadata.get("source_hash") or "")
        candidate_language = str(metadata.get("language") or "")
        if not candidate_source_hash or not candidate_language:
            continue
        if source_hash is not None and candidate_source_hash != source_hash:
            continue
        if language is not None and candidate_language != language:
            continue
        segment_id = str(record.get("segment_id") or "")
        chapter_id = str(provenance.get("chapter_id") or "")
        ledger_text = str(provenance.get("ledger") or "")
        if not chapter_id and ledger_text:
            chapter_id = Path(ledger_text).parent.name
        if not chapter_id and ".seg-" in segment_id:
            chapter_id = segment_id.split(".seg-", 1)[0]
        block_ids: list[str] = []
        segmentation_paths = (
            [checkpoint / "chapters" / chapter_id / "segmentation.json"]
            if chapter_id else
            sorted((checkpoint / "chapters").glob("*/segmentation.json"))
        )
        for segmentation_path in segmentation_paths:
            owner = segmentation_path.parent.name
            segmentation = _optional_object(segmentation_path)
            for index, raw in enumerate(segmentation.get("segments") or [], 1):
                if not isinstance(raw, Mapping):
                    continue
                raw_id = str(raw.get("segment_id") or "")
                normalized_ids = {
                    raw_id, f"{owner}.seg-{index:04d}", f"seg-{index:04d}",
                }
                if segment_id in normalized_ids:
                    block_ids = [str(value) for value in raw.get("block_ids") or []]
                    chapter_id = owner
                    break
            if block_ids:
                break
        if not block_ids:
            continue
        candidate = {
            "block_ids": block_ids,
            "source_hash": candidate_source_hash,
            "language": candidate_language,
            "translation": record.get("output"),
            "accepted_artifact_id": str(record.get("artifact_id") or ""),
            "created_at": record.get("created_at"),
        }
        key = (candidate_source_hash, candidate_language, tuple(block_ids))
        created = float(record.get("created_at") or 0)
        if key not in selected or created > selected[key][0]:
            selected[key] = (created, candidate)
    return {
        f"accepted-{index:04d}": item[1]
        for index, item in enumerate(
            sorted(
                selected.values(),
                key=lambda pair: (
                    pair[1]["source_hash"], pair[1]["language"],
                    tuple(pair[1]["block_ids"]),
                ),
            ), 1
        )
    }


def _validate_translation_candidate(
    candidate: Mapping[str, Any],
    *,
    segment: Mapping[str, Any],
    blocks_by_id: Mapping[str, dict[str, Any]],
    metadata: Mapping[str, Any],
    source_hash: str,
    language: str,
    glossary: Mapping[str, Any],
    protected_names: list[str],
    segment_input_hash: Callable[[Mapping[str, Any]], str] | None,
) -> dict[str, Any]:
    segment_id = str(segment.get("segment_id") or "")
    candidate_source = str(candidate.get("source_hash") or metadata.get("source_hash") or "")
    if candidate_source != source_hash:
        return {"segment_id": segment_id, "accepted": False, "reason": "source_hash_mismatch"}
    candidate_language = str(candidate.get("language") or metadata.get("language") or "")
    if candidate_language != language:
        return {"segment_id": segment_id, "accepted": False, "reason": "language_mismatch"}
    translation = candidate.get("translation") if isinstance(candidate.get("translation"), Mapping) else candidate
    raw_blocks = translation.get("blocks") if isinstance(translation, Mapping) else None
    expected = [
        blocks_by_id[str(value)]
        for value in segment.get("block_ids") or []
        if str(value) in blocks_by_id and is_translatable(blocks_by_id[str(value)])
    ]
    if not isinstance(raw_blocks, list):
        return {"segment_id": segment_id, "accepted": False, "reason": "translation_blocks_missing"}
    projected_raw_blocks = [
        item for item in raw_blocks
        if isinstance(item, Mapping)
        and not (
            str(item.get("block_id") or "") in blocks_by_id
            and is_structural(blocks_by_id[str(item.get("block_id") or "")])
        )
    ]
    dropped_structural_ids = [
        str(item.get("block_id") or "") for item in raw_blocks
        if isinstance(item, Mapping) and item not in projected_raw_blocks
    ]
    actual_ids = [str(item.get("block_id") or "") for item in projected_raw_blocks]
    expected_ids = [block_id(item) for item in expected]
    if actual_ids != expected_ids or len(projected_raw_blocks) != len(actual_ids):
        return {"segment_id": segment_id, "accepted": False, "reason": "source_block_coverage_mismatch"}
    terminology_warnings: list[str] = []
    for source, translated in zip(expected, projected_raw_blocks):
        text = _translated_text(translated)
        if not text.strip():
            return {"segment_id": segment_id, "accepted": False, "reason": "empty_translation_block"}
        if _opaque_tokens_in_text(text) != opaque_inline_tokens(source):
            return {"segment_id": segment_id, "accepted": False, "reason": "opaque_token_mismatch"}
        source_text = str(translation_input_block(source).get("text") or "")
        for term in _missing_terminology(source_text, text, glossary):
            if term not in terminology_warnings:
                terminology_warnings.append(term)
        if _missing_protected_names(source_text, text, protected_names):
            return {"segment_id": segment_id, "accepted": False, "reason": "protected_name_mismatch"}
    normalized = {"blocks": [dict(item) for item in projected_raw_blocks]}
    input_sha = (
        segment_input_hash(segment) if segment_input_hash is not None else
        sha256_json({
            "source_hash": source_hash,
            "language": language,
            "segment": dict(segment),
            "blocks": [translation_input_block(item) for item in expected],
            "glossary_sha256": sha256_json(glossary),
        })
    )
    reuse_status = str(candidate.get("reuse_status") or "hit")
    if terminology_warnings:
        reuse_status = "warning_reuse"
    return {
        "segment_id": segment_id,
        "accepted": True,
        "status": reuse_status,
        "reason": (
            "translation_revalidated" if reuse_status == "hit" else reuse_status
        ),
        "translation": normalized,
        "input_sha256": input_sha,
        "terminology_warnings": terminology_warnings,
        "dropped_structural_block_ids": list(dict.fromkeys([
            *[str(value) for value in candidate.get("dropped_structural_block_ids") or []],
            *dropped_structural_ids,
        ])),
        "accepted_artifact_ids": list(candidate.get("accepted_artifact_ids") or []),
    }


def _translation_candidates(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        output = []
        for key, item in value.items():
            if not isinstance(item, Mapping):
                continue
            record = dict(item)
            record.setdefault("legacy_segment_id", str(key))
            payload = record.get("translation") if isinstance(record.get("translation"), Mapping) else record
            if not record.get("block_ids") and isinstance(payload, Mapping):
                record["block_ids"] = [
                    str(block.get("block_id") or "")
                    for block in payload.get("blocks") or []
                    if isinstance(block, Mapping)
                ]
            output.append(record)
        return output
    if isinstance(value, list):
        output = []
        for item in value:
            if not isinstance(item, Mapping):
                continue
            record = dict(item)
            payload = record.get("translation") if isinstance(record.get("translation"), Mapping) else record
            if not record.get("block_ids") and isinstance(payload, Mapping):
                record["block_ids"] = [
                    str(block.get("block_id") or "")
                    for block in payload.get("blocks") or []
                    if isinstance(block, Mapping)
                ]
            output.append(record)
        return output
    return []


def _legacy_metadata(legacy: Mapping[str, Any]) -> dict[str, Any]:
    metadata = legacy.get("metadata") if isinstance(legacy.get("metadata"), Mapping) else {}
    return {**dict(metadata), **{
        key: legacy[key]
        for key in ("source_hash", "language", "prompt_hash", "validator_hash")
        if key in legacy
    }}


def _translations_with_segment_blocks(value: Any, segmentation: Any) -> Any:
    if not isinstance(value, Mapping) or not isinstance(segmentation, Mapping):
        return value
    by_id = {
        str(item.get("segment_id") or ""): list(item.get("block_ids") or [])
        for item in segmentation.get("segments") or []
        if isinstance(item, Mapping)
    }
    output: dict[str, Any] = {}
    for key, raw in value.items():
        if not isinstance(raw, Mapping):
            output[str(key)] = raw
            continue
        record = dict(raw)
        segment_id = str(record.get("segment_id") or key)
        if not record.get("block_ids") and by_id.get(segment_id):
            record["block_ids"] = by_id[segment_id]
        output[str(key)] = record
    return output


def _segment_size_ok(
    ids: list[str], by_id: Mapping[str, dict[str, Any]], *, max_blocks: int, max_chars: int
) -> bool:
    if len(ids) > max_blocks:
        return False
    projection = [translation_input_block(by_id[value]) for value in ids if value in by_id]
    return len(json.dumps(projection, ensure_ascii=False, sort_keys=True)) <= max_chars


def _has_real_index(value: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None) -> bool:
    if isinstance(value, Mapping):
        return bool(value.get("entries"))
    return bool(list(value or []))


def _translated_text(value: Mapping[str, Any]) -> str:
    return str(value.get("text") or value.get("translated_text") or value.get("translation") or "")


def _opaque_tokens_in_text(value: str) -> list[str]:
    return re.findall(r"\[\[ARC_INLINE:[^:\]]+:[0-9a-f]{64}\]\]", value)


def _missing_terminology(source: str, translated: str, glossary: Mapping[str, Any]) -> list[str]:
    missing = []
    source_folded = source.casefold()
    translated_folded = translated.casefold()
    for entry in glossary.get("entries") or []:
        if not isinstance(entry, Mapping):
            continue
        term = str(entry.get("source_term") or entry.get("source") or entry.get("term") or "").strip()
        target = str(entry.get("target_term") or entry.get("target") or entry.get("translation") or "").strip()
        if term and target and term.casefold() in source_folded and target.casefold() not in translated_folded:
            missing.append(term)
    return missing


def _missing_protected_names(source: str, translated: str, names: Iterable[str]) -> list[str]:
    return [
        name
        for name in names
        if name
        and re.search(rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])", source, re.IGNORECASE)
        and not re.search(rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])", translated, re.IGNORECASE)
    ]


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
