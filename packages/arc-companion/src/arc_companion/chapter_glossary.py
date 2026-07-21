from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Any, Callable, Mapping

from .io import sha256_json, write_json

SEGMENT_GLOSSARY_VERSION = "arc.companion.segment-glossary.v2"
INDEX_GLOSSARY_VERSION = "arc.companion.index-glossary.v1"
INDEX_BATCH_SIZE = 100


def generate_index_glossary(
    index_entries: list[Mapping[str, Any]],
    *,
    language: str,
    checkpoint_dir: Path,
    force: bool,
    call_model: Callable[[str, dict[str, Any], Path, str], dict[str, Any]],
) -> dict[str, Any]:
    """Translate every real index entry without deduplication or truncation."""
    flattened = _flatten_index(index_entries)
    source_sha256 = sha256_json({"entries": flattened, "language": language})
    path = checkpoint_dir / "index-glossary.json"
    if path.is_file() and not force:
        try:
            import json
            cached = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = {}
        if cached.get("schema_version") == INDEX_GLOSSARY_VERSION and cached.get("source_sha256") == source_sha256:
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
    for batch_number, start in enumerate(range(0, len(flattened), INDEX_BATCH_SIZE), 1):
        batch = flattened[start : start + INDEX_BATCH_SIZE]
        prompt = (
            f"Supply a standard {language} translation and one short explanation for every index entry. "
            "Return every entry_id exactly once; do not merge, delete, or reorder entries.\n"
            + __import__("json").dumps(batch, ensure_ascii=False, sort_keys=True)
        )
        result = call_model(prompt, schema, checkpoint_dir / "llm" / "index-glossary" / f"batch-{batch_number:04d}", f"companion-index-glossary-{batch_number:04d}")
        returned = [item for item in result.get("entries") or [] if isinstance(item, Mapping)]
        expected_ids = [item["entry_id"] for item in batch]
        actual_ids = [str(item.get("entry_id") or "") for item in returned]
        if actual_ids != expected_ids:
            raise ValueError("index glossary batch did not preserve every entry exactly once in order")
        supplements.update({str(item["entry_id"]): {"target": str(item["target"]), "explanation": str(item["explanation"])} for item in returned})
    output_entries = [{**item, **supplements[item["entry_id"]]} for item in flattened]
    output = {"schema_version": INDEX_GLOSSARY_VERSION, "source_sha256": source_sha256, "language": language, "entry_limit": None, "entries": output_entries}
    write_json(path, output)
    return output


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
    if not term:
        return False
    # Only Latin-script letters and Unicode decimal digits participate in word
    # boundaries.  CJK may remain adjacent, while e.g. ``résumé`` cannot match
    # the plural ``résumés`` and an ASCII digit cannot match an adjacent Nd digit.
    start = 0
    while (offset := source.find(term, start)) >= 0:
        before = source[offset - 1] if offset else ""
        after_offset = offset + len(term)
        after = source[after_offset] if after_offset < len(source) else ""
        if (
            (not _is_latin_or_decimal(term[0]) or not _is_latin_or_decimal(before))
            and (not _is_latin_or_decimal(term[-1]) or not _is_latin_or_decimal(after))
        ):
            return True
        start = offset + 1
    return False


def _is_latin_or_decimal(value: str) -> bool:
    if not value:
        return False
    return unicodedata.category(value) == "Nd" or (
        unicodedata.category(value).startswith("L")
        and "LATIN" in unicodedata.name(value, "")
    )


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
