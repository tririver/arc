from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Callable, Mapping

from .io import sha256_json, write_json

from .source import block_id


CHAPTER_GLOSSARY_VERSION = "arc.companion.chapter-glossary.v1"
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


def project_chapter_glossary(
    chapter: Mapping[str, Any],
    document: Mapping[str, Any],
    glossary: Mapping[str, Any],
    *,
    index_entries: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Select terms using source blocks and index/page overlap only."""
    wanted = {str(value) for value in chapter.get("block_ids") or []}
    source = "\n".join(
        _source_text(item)
        for item in document.get("blocks") or []
        if isinstance(item, Mapping) and block_id(item) in wanted
    )
    folded_source = _fold(source)
    page_start = _int_or_none(chapter.get("page_start"))
    page_end = _int_or_none(chapter.get("page_end"))
    index_by_term = _index_lookup(index_entries or [])
    entries: list[dict[str, Any]] = []
    for ordinal, item in enumerate(
        (item for item in glossary.get("entries") or [] if isinstance(item, Mapping)), 1
    ):
        value = dict(item)
        value.setdefault("entry_id", f"term-{ordinal:04d}")
        entries.append(value)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for entry in entries:
        terms = _entry_terms(entry)
        index_match = any(
            _page_overlap(index_by_term.get(_fold(term), []), page_start, page_end) for term in terms
        )
        source_match = any(_contains_term(folded_source, _fold(term)) for term in terms if term.strip())
        if source_match or index_match:
            value = dict(entry)
            selected.append(value)
            selected_ids.add(str(value["entry_id"]))
    # A matched subentry is not meaningful without its complete visible path.
    by_id = {str(item.get("entry_id") or ""): dict(item) for item in entries}
    pending = [str(item.get("parent_id")) for item in selected if item.get("parent_id")]
    while pending:
        parent_id = pending.pop()
        if parent_id in selected_ids:
            continue
        parent = by_id.get(parent_id)
        if parent is None:
            continue
        selected_ids.add(parent_id)
        if parent.get("parent_id"):
            pending.append(str(parent["parent_id"]))
    # Preserve global glossary order after adding ancestors.  This also keeps a
    # grandparent before its parent and child without relying on entry ids.
    selected = [dict(item) for item in entries if str(item.get("entry_id") or "") in selected_ids]
    return {
        "schema_version": CHAPTER_GLOSSARY_VERSION,
        "chapter_id": str(chapter.get("chapter_id") or ""),
        "entries": selected,
        "compact_mapping": [
            {"source": _primary_term(item), "target": str(item.get("target") or item.get("translation") or "")}
            for item in selected
        ],
    }


def _entry_terms(entry: Mapping[str, Any]) -> list[str]:
    primary = _primary_term(entry)
    aliases = entry.get("aliases") or entry.get("source_aliases") or []
    return [primary, *[str(value) for value in aliases if str(value).strip()]]


def _primary_term(entry: Mapping[str, Any]) -> str:
    return str(entry.get("source") or entry.get("term") or entry.get("source_term") or "").strip()


def _source_text(block: Mapping[str, Any]) -> str:
    return " ".join(str(block.get(key) or "") for key in ("title", "text", "markdown", "tex"))


def _fold(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _contains_term(source: str, term: str) -> bool:
    if not term:
        return False
    if re.fullmatch(r"[\w -]+", term):
        return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", source) is not None
    return term in source


def _index_lookup(entries: list[Mapping[str, Any]]) -> dict[str, list[tuple[int, int]]]:
    output: dict[str, list[tuple[int, int]]] = {}
    def visit(item: Mapping[str, Any]) -> None:
        term = str(item.get("term") or item.get("label") or "").strip()
        ranges = item.get("page_ranges") or item.get("pages") or []
        normalized: list[tuple[int, int]] = []
        for value in ranges if isinstance(ranges, list) else []:
            if isinstance(value, int):
                normalized.append((value, value))
            elif isinstance(value, Mapping):
                start, end = _int_or_none(value.get("start")), _int_or_none(value.get("end"))
                if start is not None:
                    normalized.append((start, end if end is not None else start))
        if term:
            output.setdefault(_fold(term), []).extend(normalized)
        for child in item.get("children") or []:
            if isinstance(child, Mapping):
                visit(child)
    for entry in entries:
        visit(entry)
    return output


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


def _page_overlap(ranges: list[tuple[int, int]], start: int | None, end: int | None) -> bool:
    if start is None or end is None:
        return False
    return any(left <= end and right >= start for left, right in ranges)


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
