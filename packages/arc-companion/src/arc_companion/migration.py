from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .io import sha256_json
from .ledger import LANE_LEDGER_VERSION
from .projection import is_translatable, opaque_inline_tokens, translation_input_block
from .source import block_id


MIGRATION_VERSION = "arc.companion.legacy-migration.v1"
NEVER_MIGRATED_ARTIFACTS = (
    "guides",
    "annotations",
    "reviews",
    "tex",
    "pdf",
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
) -> dict[str, Any]:
    candidates = _translation_candidates(translations)
    by_blocks: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for candidate in candidates:
        by_blocks.setdefault(tuple(str(value) for value in candidate.get("block_ids") or []), []).append(candidate)
    blocks_by_id = {block_id(dict(item)): dict(item) for item in blocks}
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
            key = tuple(str(value) for value in segment.get("block_ids") or [])
            matching = by_blocks.get(key, [])
            receipt: dict[str, Any]
            if len(matching) != 1:
                receipt = {
                    "segment_id": segment_id,
                    "accepted": False,
                    "reason": "translation_missing_or_ambiguous",
                }
            else:
                receipt = _validate_translation_candidate(
                    matching[0],
                    segment=segment,
                    blocks_by_id=blocks_by_id,
                    metadata=metadata,
                    source_hash=source_hash,
                    language=language,
                    glossary=glossary,
                    protected_names=protected_names,
                    segment_input_hash=segment_input_hash,
                )
            if receipt.get("accepted") and not accepted_prefix:
                receipt = {**receipt, "accepted": False, "reason": "not_in_contiguous_accepted_prefix"}
            accepted_prefix = accepted_prefix and bool(receipt.get("accepted"))
            receipts.append({"chapter_id": chapter_id, **receipt})
            if accepted_prefix:
                translation = receipt["translation"]
                input_sha = str(receipt["input_sha256"])
                output_sha = sha256_json(translation)
                block_record = {
                    "segment_id": segment_id,
                    "state": "accepted",
                    "generation": 1,
                    "input_sha256": input_sha,
                    "output_sha256": output_sha,
                    "logical_receipt": {"kind": "legacy_migration", "provider_calls": 0},
                    "validation_receipt": {
                        "schema_version": MIGRATION_VERSION,
                        "source": True,
                        "language": True,
                        "terminology": True,
                        "opaque_tokens": True,
                        "protected_names": True,
                    },
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
            else:
                ledger_blocks.append({"segment_id": segment_id, "state": "pending", "generation": 1})
        ledgers[chapter_id] = {
            "schema_version": LANE_LEDGER_VERSION,
            "chapter_id": chapter_id,
            "lane": "translation",
            "generation": 1,
            "needs_supervision": None,
            "blocks": ledger_blocks,
            "accepted_chain_sha256": chain,
            "migration_source": "legacy_checkpoint",
            "updated_at": 0.0,
        }
    return {"ledgers": ledgers, "receipts": receipts}


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
    actual_ids = [str(item.get("block_id") or "") for item in raw_blocks if isinstance(item, Mapping)]
    expected_ids = [block_id(item) for item in expected]
    if actual_ids != expected_ids or len(raw_blocks) != len(actual_ids):
        return {"segment_id": segment_id, "accepted": False, "reason": "source_block_coverage_mismatch"}
    for source, translated in zip(expected, raw_blocks):
        text = _translated_text(translated)
        if not text.strip():
            return {"segment_id": segment_id, "accepted": False, "reason": "empty_translation_block"}
        if _opaque_tokens_in_text(text) != opaque_inline_tokens(source):
            return {"segment_id": segment_id, "accepted": False, "reason": "opaque_token_mismatch"}
        source_text = str(translation_input_block(source).get("text") or "")
        if _missing_terminology(source_text, text, glossary):
            return {"segment_id": segment_id, "accepted": False, "reason": "terminology_mismatch"}
        if _missing_protected_names(source_text, text, protected_names):
            return {"segment_id": segment_id, "accepted": False, "reason": "protected_name_mismatch"}
    normalized = {"blocks": [dict(item) for item in raw_blocks]}
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
    return {
        "segment_id": segment_id,
        "accepted": True,
        "reason": "translation_revalidated",
        "translation": normalized,
        "input_sha256": input_sha,
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
