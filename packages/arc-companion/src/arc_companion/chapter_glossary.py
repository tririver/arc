from __future__ import annotations

import json
import hashlib
import unicodedata
from pathlib import Path
from typing import Any, Callable, Mapping

from .io import sha256_json, write_json
from .language import contains_lexical_term
from .recovery_units import (
    call_model_with_recovery_descriptor,
    require_control_acceptance,
    submission_descriptor,
)
from .secure_io import SecureReadError, read_bounded_json

SEGMENT_GLOSSARY_VERSION = "arc.companion.segment-glossary.v2"
INDEX_GLOSSARY_VERSION = "arc.companion.index-glossary.v1"
INDEX_BATCH_SIZE = 100
INDEX_GLOSSARY_BATCH_VERSION = "arc.companion.index-glossary-batch.v1"
_MAX_INDEX_GLOSSARY_BATCH_BYTES = 16 * 1024 * 1024


def generate_index_glossary(
    index_entries: list[Mapping[str, Any]],
    *,
    language: str,
    checkpoint_dir: Path,
    force: bool,
    call_model: Callable[[str, dict[str, Any], Path, str], dict[str, Any]],
    intent_guidance_identity: Mapping[str, Any] | None = None,
    accept_recovery: Callable[[Path, str, str], int] | None = None,
) -> dict[str, Any]:
    """Translate every real index entry without deduplication or truncation."""
    flattened = _flatten_index(index_entries)
    source_sha256 = sha256_json({
        "entries": flattened, "language": language,
        **(
            {"intent_guidance": dict(intent_guidance_identity)}
            if intent_guidance_identity is not None else {}
        ),
    })
    path = checkpoint_dir / "index-glossary.json"
    if not force:
        cached = _read_valid_index_glossary_cache(
            checkpoint_dir,
            path,
            source_sha256=source_sha256,
            language=language,
            flattened=flattened,
        )
        if cached is not None:
            return cached
    supplements: dict[str, dict[str, str]] = {}
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["entries"],
        "properties": {"entries": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["entry_id", "target", "explanation"],
            "properties": {
                "entry_id": {"type": "string"}, "target": {"type": "string"},
                "explanation": {"type": "string"},
            },
        }}},
    }
    batches = [
        flattened[start : start + INDEX_BATCH_SIZE]
        for start in range(0, len(flattened), INDEX_BATCH_SIZE)
    ]
    logical_units = [
        (
            f"index-glossary-batch-{batch_number:04d}-"
            f"{sha256_json([str(item['entry_id']) for item in batch])[:16]}"
        )
        for batch_number, batch in enumerate(batches, 1)
    ]
    for batch_number, batch in enumerate(batches, 1):
        prompt = (
            f"Supply a standard {language} translation and one short explanation for every index entry. "
            "Return every entry_id exactly once; do not merge, delete, or reorder entries.\n"
            + __import__("json").dumps(batch, ensure_ascii=False, sort_keys=True)
        )
        artifact_dir = (
            checkpoint_dir / "llm" / "index-glossary"
            / f"batch-{batch_number:04d}"
        )
        logical_unit = logical_units[batch_number - 1]
        expected_ids = [str(item["entry_id"]) for item in batch]
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        schema_sha256 = sha256_json(schema)
        batch_identity = {
            "source_sha256": source_sha256,
            "language": language,
            "expected_entry_ids": expected_ids,
            "prompt_sha256": prompt_sha256,
            "schema_sha256": schema_sha256,
        }
        batch_input_sha256 = sha256_json(batch_identity)
        batch_checkpoint = (
            checkpoint_dir / "index-glossary-batches" / f"{logical_unit}.json"
        )
        result = None if force else _read_valid_batch_checkpoint(
            checkpoint_dir,
            batch_checkpoint,
            logical_unit=logical_unit,
            input_sha256=batch_input_sha256,
            expected_ids=expected_ids,
        )
        if result is None:
            result = call_model_with_recovery_descriptor(
                call_model,
                prompt,
                schema,
                artifact_dir,
                f"companion-index-glossary-{batch_number:04d}",
                submission_descriptor(
                    unit="glossary-index",
                    logical_unit=logical_unit,
                    checkpoint_dir=checkpoint_dir,
                    artifact_root=artifact_dir,
                    acceptance_checkpoint=batch_checkpoint,
                    input_sha256=batch_input_sha256,
                    group_sha256=source_sha256,
                    ordered_siblings=logical_units,
                    suffix=logical_units[batch_number - 1 :],
                ),
            )
        if not _valid_index_glossary_response(result, expected_ids):
            raise ValueError("index glossary batch did not preserve every entry exactly once in order")
        returned = list(result["entries"])
        write_json(batch_checkpoint, {
            "schema_version": INDEX_GLOSSARY_BATCH_VERSION,
            "logical_unit": logical_unit,
            "input_sha256": batch_input_sha256,
            "source_sha256": source_sha256,
            "language": language,
            "expected_entry_ids": expected_ids,
            "prompt_sha256": prompt_sha256,
            "schema_sha256": schema_sha256,
            "response": dict(result),
        })
        require_control_acceptance(
            accept_recovery,
            checkpoint_dir=checkpoint_dir,
            unit="glossary-index",
            logical_unit=logical_unit,
        )
        supplements.update({str(item["entry_id"]): {"target": str(item["target"]), "explanation": str(item["explanation"])} for item in returned})
    output_entries = [{**item, **supplements[item["entry_id"]]} for item in flattened]
    output = {"schema_version": INDEX_GLOSSARY_VERSION, "source_sha256": source_sha256, "language": language, "entry_limit": None, "entries": output_entries}
    write_json(path, output)
    return output


def _read_valid_index_glossary_cache(
    checkpoint_dir: Path,
    path: Path,
    *,
    source_sha256: str,
    language: str,
    flattened: list[dict[str, Any]],
) -> dict[str, Any] | None:
    try:
        value = read_bounded_json(
            checkpoint_dir,
            path.relative_to(checkpoint_dir),
            max_bytes=_MAX_INDEX_GLOSSARY_BATCH_BYTES,
            suffixes=(".json",),
        )
    except (SecureReadError, ValueError):
        return None
    entries = value.get("entries") if isinstance(value, Mapping) else None
    if (
        not isinstance(value, Mapping)
        or set(value) != {
            "schema_version", "source_sha256", "language", "entry_limit", "entries",
        }
        or value.get("schema_version") != INDEX_GLOSSARY_VERSION
        or value.get("source_sha256") != source_sha256
        or value.get("language") != language
        or value.get("entry_limit") is not None
        or not isinstance(entries, list)
        or len(entries) != len(flattened)
    ):
        return None
    for expected, actual in zip(flattened, entries, strict=True):
        if (
            not isinstance(actual, Mapping)
            or set(actual) != {*expected, "target", "explanation"}
            or any(
                actual.get(key) != item
                for key, item in expected.items()
                if key not in {"target", "explanation"}
            )
            or not isinstance(actual.get("target"), str)
            or not isinstance(actual.get("explanation"), str)
        ):
            return None
    return dict(value)


def _read_valid_batch_checkpoint(
    checkpoint_dir: Path,
    path: Path,
    *,
    logical_unit: str,
    input_sha256: str,
    expected_ids: list[str],
) -> dict[str, Any] | None:
    try:
        value = read_bounded_json(
            checkpoint_dir,
            path.relative_to(checkpoint_dir),
            max_bytes=_MAX_INDEX_GLOSSARY_BATCH_BYTES,
            suffixes=(".json",),
        )
    except (SecureReadError, ValueError):
        return None
    response = value.get("response") if isinstance(value, Mapping) else None
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != INDEX_GLOSSARY_BATCH_VERSION
        or value.get("logical_unit") != logical_unit
        or value.get("input_sha256") != input_sha256
        or value.get("expected_entry_ids") != expected_ids
        or not _valid_index_glossary_response(response, expected_ids)
    ):
        return None
    return dict(response)


def _valid_index_glossary_response(
    response: Any, expected_ids: list[str],
) -> bool:
    if not isinstance(response, Mapping) or set(response) != {"entries"}:
        return False
    entries = response.get("entries")
    if not isinstance(entries, list) or len(entries) != len(expected_ids):
        return False
    for expected_id, item in zip(expected_ids, entries, strict=True):
        if (
            not isinstance(item, Mapping)
            or set(item) != {"entry_id", "target", "explanation"}
            or item.get("entry_id") != expected_id
            or not isinstance(item.get("target"), str)
            or not isinstance(item.get("explanation"), str)
        ):
            return False
    return True


def project_segment_glossary(
    source_blocks: list[Mapping[str, Any]],
    glossary: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the paper glossary onto immutable source blocks for one segment.

    Projection is deliberately lexical.  Generated text and evidence are not
    inputs, so the same source segment always receives the same ordered terms.
    """
    source = _fold("\n".join(_source_text(item) for item in source_blocks))
    entries = _normalized_entries(glossary)
    by_id = {item["entry_id"]: item for item in entries}
    selected: list[dict[str, Any]] = []
    for entry in entries:
        if not any(_contains_term(source, _fold(term)) for term in _entry_terms(entry)):
            continue
        projected = {
            "entry_id": entry["entry_id"],
            "source": entry["source"],
            "target": entry["target"],
        }
        for key in ("aliases", "explanation", "protected_names"):
            if entry.get(key):
                projected[key] = entry[key]
        lineage = _entry_lineage(entry, by_id)
        if lineage:
            projected["lineage"] = lineage
        selected.append(projected)
    return {
        "schema_version": SEGMENT_GLOSSARY_VERSION,
        "source_glossary_sha256": sha256_json(glossary),
        "counts": {"source_entries": len(entries), "matched_entries": len(selected)},
        "entries": selected,
    }


def _normalized_entries(glossary: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(
        (item for item in glossary.get("entries") or [] if isinstance(item, Mapping)), 1
    ):
        source = _primary_term(raw)
        if not source:
            continue
        aliases = _unique_strings(raw.get("aliases") or raw.get("source_aliases") or [])
        aliases = [value for value in aliases if _fold(value) != _fold(source)]
        entry: dict[str, Any] = {
            "entry_id": str(raw.get("entry_id") or f"term-{ordinal:04d}"),
            "source": source,
            # Empty targets are meaningful and must not fall through to an alias.
            "target": _target_term(raw),
            "aliases": aliases,
            "explanation": str(raw.get("explanation") or raw.get("brief_explanation") or "").strip(),
            "protected_names": _unique_strings(raw.get("protected_names") or []),
        }
        if raw.get("parent_id"):
            entry["parent_id"] = str(raw["parent_id"])
        entries.append(entry)
    return entries


def _entry_lineage(
    entry: Mapping[str, Any], by_id: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, str]]:
    lineage: list[dict[str, str]] = []
    seen: set[str] = set()
    parent_id = str(entry.get("parent_id") or "")
    while parent_id and parent_id not in seen:
        seen.add(parent_id)
        parent = by_id.get(parent_id)
        if parent is None:
            break
        lineage.append({
            "entry_id": str(parent["entry_id"]),
            "source": str(parent["source"]),
            "target": str(parent["target"]),
        })
        parent_id = str(parent.get("parent_id") or "")
    lineage.reverse()
    return lineage


def _entry_terms(entry: Mapping[str, Any]) -> list[str]:
    primary = _primary_term(entry)
    aliases = entry.get("aliases") or entry.get("source_aliases") or []
    return [primary, *[str(value) for value in aliases if str(value).strip()]]


def _primary_term(entry: Mapping[str, Any]) -> str:
    return str(entry.get("source_term") or entry.get("source") or entry.get("term") or "").strip()


def _target_term(entry: Mapping[str, Any]) -> str:
    for key in ("target_term", "target", "translation"):
        if key in entry and entry[key] is not None:
            return str(entry[key]).strip()
    return ""


def _unique_strings(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        key = _fold(text)
        if text and key not in seen:
            seen.add(key)
            output.append(text)
    return output


def _source_text(block: Mapping[str, Any]) -> str:
    return " ".join(str(block.get(key) or "") for key in ("title", "text", "markdown", "tex"))


def _fold(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _contains_term(source: str, term: str) -> bool:
    return contains_lexical_term(source, term)


def _flatten_index(entries: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    def visit(item: Mapping[str, Any], parent_id: str | None) -> None:
        entry_id = f"index-{len(output) + 1:05d}"
        record = {key: value for key, value in dict(item).items() if key != "children"}
        record["entry_id"] = str(item.get("entry_id") or entry_id)
        record["source"] = str(item.get("term") or item.get("label") or item.get("source") or "")
        if parent_id:
            record["parent_id"] = parent_id
        output.append(record)
        for child in item.get("children") or []:
            if isinstance(child, Mapping):
                visit(child, record["entry_id"])
    for entry in entries:
        visit(entry, None)
    return output
