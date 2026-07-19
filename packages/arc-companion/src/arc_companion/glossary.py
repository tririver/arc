from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
from typing import Any, Callable

from .io import read_json, sha256_json, write_json
from .prompts import GLOSSARY_SCHEMA, glossary_consolidation_prompt, glossary_prompt
from .segmentation import build_block_inventory, build_segmentation_windows
from .source import block_id


GLOSSARY_VERSION = "arc.companion.glossary.v4"
ABSOLUTE_GLOSSARY_LIMIT = 200


def glossary_entry_limit(page_count: int | None) -> int:
    """Return the page-scaled glossary cap, with a conservative absolute fallback."""
    if page_count is None or page_count < 1:
        return ABSOLUTE_GLOSSARY_LIMIT
    if page_count <= 50:
        return 50
    if page_count <= 100:
        return 100
    return ABSOLUTE_GLOSSARY_LIMIT


def generate_glossary(
    document: dict[str, Any],
    *,
    language: str,
    protected_names: list[str],
    checkpoint_dir: Path,
    workers: int,
    force: bool,
    call_model: Callable[[str, dict[str, Any], Path, str], dict[str, Any]],
    page_count: int | None = None,
) -> dict[str, Any]:
    inventory = build_block_inventory(document)
    windows = build_segmentation_windows(inventory)
    blocks = list(document.get("blocks") or [])
    canonical_records = [_glossary_record(block, document) for block in blocks]
    source_hash = sha256_json({
        "canonical_records": canonical_records,
        "language": language,
        "names": protected_names,
        "page_count": page_count,
        "entry_limit": glossary_entry_limit(page_count),
    })
    final_path = checkpoint_dir / "glossary.json"
    if final_path.is_file() and not force:
        cached = _optional_json(final_path)
        if (
            cached.get("schema_version") == GLOSSARY_VERSION
            and cached.get("source_sha256") == source_hash
            and isinstance(cached.get("entries"), list)
        ):
            return cached

    candidate_dir = checkpoint_dir / "glossary-windows"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    entry_limit = glossary_entry_limit(page_count)

    def extract(index: int, window: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        path = candidate_dir / f"{index:04d}.json"
        start = int(window["start_ordinal"]) - 1
        end = int(window["end_ordinal"])
        selected = canonical_records[start:end]
        input_hash = sha256_json({"blocks": selected, "language": language, "names": protected_names})
        if path.is_file() and not force:
            cached = _optional_json(path)
            if cached.get("input_sha256") == input_hash and isinstance(cached.get("result"), dict):
                return index, cached["result"]
        result = call_model(
            glossary_prompt(
                selected,
                language=language,
                protected_names=protected_names,
                entry_limit=entry_limit,
            ),
            GLOSSARY_SCHEMA,
            checkpoint_dir / "llm" / "glossary" / f"window-{index:04d}",
            f"companion-glossary-window-{index:04d}",
        )
        value = {"entries": _normalize_entries(result.get("entries") or [], blocks)}
        write_json(path, {"input_sha256": input_hash, "result": value})
        return index, value

    candidates: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(windows)))) as executor:
        futures = {executor.submit(extract, index, window): index for index, window in enumerate(windows, 1)}
        for future in as_completed(futures):
            index, value = future.result()
            candidates[index] = value

    ordered = [candidates[index] for index in sorted(candidates)]
    consolidated = call_model(
        glossary_consolidation_prompt(
            ordered,
            language=language,
            protected_names=protected_names,
            entry_limit=entry_limit,
        ),
        GLOSSARY_SCHEMA,
        checkpoint_dir / "llm" / "glossary" / "consolidation",
        "companion-glossary-consolidation",
    )
    entries = _normalize_entries(consolidated.get("entries") or [], blocks)
    entries = _deduplicate(entries)
    entries = entries[:glossary_entry_limit(page_count)]
    _restore_protected_names(entries, protected_names)
    _validate_protected_names(entries, protected_names)
    output = {
        "schema_version": GLOSSARY_VERSION,
        "source_sha256": source_hash,
        "language": language,
        "page_count": page_count,
        "entry_limit": glossary_entry_limit(page_count),
        "entries": entries,
    }
    write_json(final_path, output)
    return output


def _normalize_entries(entries: list[Any], blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid_ids = {block_id(block) for block in blocks}
    positions = {block_id(block): index for index, block in enumerate(blocks)}
    normalized: list[dict[str, Any]] = []
    for raw in entries:
        if not isinstance(raw, dict):
            continue
        source = str(raw.get("source_term") or "").strip()
        target = str(raw.get("target_term") or "").strip()
        explanation = str(raw.get("brief_explanation") or "").strip()
        if not source or not target or not explanation:
            continue
        first = str(raw.get("first_block_id") or "")
        if first not in valid_ids:
            first = _first_occurrence(source, blocks)
        normalized.append({
            "source_term": source,
            "target_term": target,
            "brief_explanation": explanation,
            "aliases": _unique_strings(raw.get("aliases") or []),
            "protected_names": _unique_strings(raw.get("protected_names") or []),
            "first_block_id": first or None,
            "_position": positions.get(first, len(blocks)),
        })
    normalized.sort(key=lambda item: (int(item.pop("_position")), item["source_term"].casefold()))
    return normalized


def _deduplicate(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        key = entry["source_term"].casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(entry)
    return output


def _validate_protected_names(entries: list[dict[str, Any]], protected_names: list[str]) -> None:
    for entry in entries:
        source = entry["source_term"]
        target = entry["target_term"]
        names = {
            str(name)
            for name in entry.get("protected_names") or []
            if name and _contains_latin_name(source, str(name))
        }
        names.update(
            name for name in protected_names if name and _contains_latin_name(source, name)
        )
        missing = [name for name in names if not _contains_latin_name(target, name)]
        if missing:
            raise RuntimeError(
                f"glossary translated or dropped protected personal names in {source!r}: {missing}"
            )


def _restore_protected_names(
    entries: list[dict[str, Any]], protected_names: list[str]
) -> None:
    """Retain a translated term while restoring source-matched Latin names.

    Glossary models sometimes produce a perfectly standard translation but omit
    the Latin spelling that ARC protects (for example, ``Poisson bracket`` to
    ``泊松括号``).  Repair that deterministic presentation detail before the
    strict validator runs.  Names suggested by a model but absent from the
    source are discarded so substring coincidences cannot create annotations.
    """
    for entry in entries:
        source = str(entry.get("source_term") or "")
        candidates = _unique_strings([
            *(entry.get("protected_names") or []),
            *protected_names,
        ])
        matched = [
            name for name in candidates
            if name and _contains_latin_name(source, name)
        ]
        entry["protected_names"] = matched

        target = str(entry.get("target_term") or "").strip()
        missing = [name for name in matched if not _contains_latin_name(target, name)]
        if not missing:
            continue

        # Prefer a containing full name over separately repeating its parts.
        # Re-check against the growing annotation because one insertion may
        # satisfy several protected lexical units (e.g. ``Ada Lovelace``).
        annotations: list[str] = []
        augmented = target
        for name in sorted(missing, key=lambda value: (-len(value), value.casefold())):
            if _contains_latin_name(augmented, name):
                continue
            annotations.append(name)
            augmented = f"{augmented} {name}"
        if annotations:
            entry["target_term"] = f"{target}（{'、'.join(annotations)}）"


def _contains_latin_name(text: str, name: str) -> bool:
    """Match a protected Latin name as a complete lexical unit, not a substring."""
    return bool(
        re.search(
            rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])",
            text,
            flags=re.IGNORECASE,
        )
    )


def _first_occurrence(term: str, blocks: list[dict[str, Any]]) -> str:
    needle = term.casefold()
    for block in blocks:
        if needle in str(_canonical_block_language(block)).casefold():
            return block_id(block)
    return ""


def _glossary_record(
    block: dict[str, Any], document: dict[str, Any]
) -> dict[str, Any]:
    value = {"block_id": block_id(block), **_canonical_block_language(block)}
    kind = str(block.get("type") or block.get("kind") or "").casefold()
    plural = {"equation": "equations", "figure": "figures", "table": "tables"}.get(kind)
    if not plural:
        return value
    singular = plural[:-1]
    entity_id = str(block.get(f"{singular}_id") or block.get("entity_id") or block.get("ref_id") or "")
    for entity in document.get(plural) or []:
        candidate = str(entity.get("id") or entity.get(f"{singular}_id") or "")
        if entity_id and candidate == entity_id:
            if kind == "equation" and entity.get("tex"):
                value["math"] = entity["tex"]
            if kind in {"figure", "table"}:
                if entity.get("tag"):
                    value["tag"] = str(entity["tag"])
                if entity.get("caption"):
                    value["caption"] = str(entity["caption"])
            if kind == "table" and entity.get("rows"):
                value["cells"] = _canonical_table_cells(entity["rows"])
            break
    return value


def _canonical_block_language(block: dict[str, Any]) -> dict[str, Any]:
    """Project full canonical language while excluding preservation-only HTML/assets."""
    value: dict[str, Any] = {
        "type": str(block.get("type") or block.get("kind") or "text")
    }
    for key in ("text", "title", "caption"):
        content = block.get(key)
        if content not in (None, ""):
            value[key] = str(content)
    items = block.get("items") or block.get("list_items")
    if items:
        value["items"] = [_canonical_item(item) for item in items]
    for key in ("tex", "math"):
        if block.get(key):
            value[key] = block[key]
    return value


def _canonical_item(item: Any) -> Any:
    if isinstance(item, dict):
        return {
            key: _canonical_item(value)
            for key, value in item.items()
            if key in {"text", "title", "caption", "items", "math", "tex"}
        }
    if isinstance(item, list):
        return [_canonical_item(value) for value in item]
    return str(item)


def _canonical_table_cells(rows: Any) -> list[list[str]]:
    output: list[list[str]] = []
    for row in rows if isinstance(rows, list) else []:
        cells: list[str] = []
        for cell in row if isinstance(row, list) else []:
            if isinstance(cell, dict):
                cells.append(str(cell.get("text") or cell.get("value") or ""))
            else:
                cells.append(str(cell))
        output.append(cells)
    return output


def _unique_strings(values: list[Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            output.append(text)
    return output


def _optional_json(path: Path) -> dict[str, Any]:
    try:
        value = read_json(path)
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}
