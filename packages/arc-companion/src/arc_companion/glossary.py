from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from .io import read_json, sha256_json, write_json
from .language import base_language, contains_lexical_term
from .prompts import GLOSSARY_SCHEMA, glossary_consolidation_prompt, glossary_prompt
from .segmentation import build_block_inventory, build_segmentation_windows
from .source import block_id


GLOSSARY_VERSION = "arc.companion.glossary.v7"
ABSOLUTE_GLOSSARY_LIMIT = 200
CONSOLIDATION_MAX_ENTRIES = 100
CONSOLIDATION_PROMPT_MAX_BYTES = 60 * 1024
CONSOLIDATION_PROMPT_TARGET_BYTES = 48 * 1024


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
    source_language: str | None = None,
    intent_guidance_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt_source_language = (
        source_language if source_language and base_language(source_language) != "en" else None
    )
    inventory = build_block_inventory(document)
    windows = build_segmentation_windows(inventory)
    blocks = list(document.get("blocks") or [])
    canonical_records = [_glossary_record(block, document) for block in blocks]
    source_hash_input = {
        "canonical_records": canonical_records,
        "language": language,
        "names": protected_names,
        "page_count": page_count,
        "entry_limit": glossary_entry_limit(page_count),
        "consolidation_max_entries": CONSOLIDATION_MAX_ENTRIES,
        "consolidation_prompt_max_bytes": CONSOLIDATION_PROMPT_MAX_BYTES,
        "consolidation_prompt_target_bytes": CONSOLIDATION_PROMPT_TARGET_BYTES,
    }
    if prompt_source_language:
        source_hash_input["source_language"] = prompt_source_language
        source_hash_input["source_term_contract"] = "exact-source-spelling-v1"
    if intent_guidance_identity is not None:
        source_hash_input["intent_guidance"] = intent_guidance_identity
    source_hash = sha256_json(source_hash_input)
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
        input_hash = sha256_json({
            "glossary_version": GLOSSARY_VERSION,
            "blocks": selected,
            "language": language,
            "names": protected_names,
            "entry_limit": entry_limit,
            **({"source_language": prompt_source_language} if prompt_source_language else {}),
        })
        if path.is_file() and not force:
            cached = _optional_json(path)
            cached_result = cached.get("result")
            cached_hash = cached.get("input_sha256")
            legacy_hash = sha256_json({
                "blocks": selected,
                "language": language,
                "names": protected_names,
            })
            reusable_legacy = (
                not prompt_source_language
                and entry_limit == ABSOLUTE_GLOSSARY_LIMIT
                and cached_hash == legacy_hash
            )
            if (
                cached_hash == input_hash or reusable_legacy
            ) and isinstance(cached_result, dict):
                # v4 window prompts used the same extraction contract but did
                # not bind the page-scaled limit into their cache key. Only
                # migrate them at the unchanged absolute limit; lower limits
                # must regenerate rather than silently reuse an overlong list.
                if reusable_legacy:
                    write_json(path, {
                        "input_sha256": input_hash,
                        "result": cached_result,
                    })
                return index, cached_result
        result = call_model(
            glossary_prompt(
                selected,
                language=language,
                protected_names=protected_names,
                entry_limit=entry_limit,
                source_language=prompt_source_language,
            ),
            GLOSSARY_SCHEMA,
            checkpoint_dir / "llm" / "glossary" / f"window-{index:04d}",
            f"companion-glossary-window-{index:04d}",
        )
        value = {"entries": _normalize_entries(
            result.get("entries") or [], blocks,
            require_exact_source=bool(prompt_source_language),
        )}
        write_json(path, {"input_sha256": input_hash, "result": value})
        return index, value

    candidates: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(windows)))) as executor:
        futures = {executor.submit(extract, index, window): index for index, window in enumerate(windows, 1)}
        for future in as_completed(futures):
            index, value = future.result()
            candidates[index] = value

    ordered = [candidates[index] for index in sorted(candidates)]
    consolidated = _consolidate_candidates(
        ordered,
        blocks=blocks,
        language=language,
        protected_names=protected_names,
        entry_limit=entry_limit,
        checkpoint_dir=checkpoint_dir,
        workers=workers,
        force=force,
        call_model=call_model,
        source_language=prompt_source_language,
    )
    entries = _normalize_entries(
        consolidated.get("entries") or [], blocks,
        require_exact_source=bool(prompt_source_language),
    )
    entries = _deduplicate(entries)
    entries = entries[:glossary_entry_limit(page_count)]
    _restore_protected_names(entries, protected_names, language=language)
    _validate_protected_names(entries, protected_names)
    output = {
        "schema_version": GLOSSARY_VERSION,
        "source_sha256": source_hash,
        "language": language,
        **({"source_language": source_language} if source_language else {}),
        "page_count": page_count,
        "entry_limit": glossary_entry_limit(page_count),
        "entries": entries,
    }
    write_json(final_path, output)
    return output


def _consolidate_candidates(
    candidates: list[dict[str, Any]],
    *,
    blocks: list[dict[str, Any]],
    language: str,
    protected_names: list[str],
    source_language: str | None = None,
    entry_limit: int,
    checkpoint_dir: Path,
    workers: int,
    force: bool,
    call_model: Callable[[str, dict[str, Any], Path, str], dict[str, Any]],
) -> dict[str, Any]:
    """Merge window glossaries without constructing one unbounded model prompt."""
    entries = [
        entry
        for candidate in candidates
        for entry in candidate.get("entries") or []
        if isinstance(entry, dict)
    ]
    _validate_consolidation_entry_transport(
        entries,
        language=language,
        protected_names=protected_names,
        source_language=source_language,
    )
    if _consolidation_candidates_fit(
        candidates,
        language=language,
        protected_names=protected_names,
        source_language=source_language,
        output_limit=entry_limit,
    ):
        return _consolidate_node(
            candidates,
            blocks=blocks,
            language=language,
            protected_names=protected_names,
            source_language=source_language,
            entry_limit=entry_limit,
            output_limit=entry_limit,
            cache_path=checkpoint_dir / "glossary-consolidation" / "direct.json",
            artifact_dir=checkpoint_dir / "llm" / "glossary" / "consolidation",
            call_label="companion-glossary-consolidation",
            stage="final",
            force=force,
            call_model=call_model,
        )

    level = 1
    current = entries
    while True:
        if len(current) <= entry_limit:
            return {"entries": current}
        if _consolidation_input_fits(
            current,
            language=language,
            protected_names=protected_names,
            source_language=source_language,
            output_limit=entry_limit,
        ):
            return _consolidate_node(
                [{"entries": current}],
                blocks=blocks,
                language=language,
                protected_names=protected_names,
                source_language=source_language,
                entry_limit=entry_limit,
                output_limit=entry_limit,
                cache_path=(
                    checkpoint_dir / "glossary-consolidation"
                    / f"level-{level:04d}" / "0001.json"
                ),
                artifact_dir=(
                    checkpoint_dir / "llm" / "glossary" / "consolidation"
                    / f"level-{level:04d}" / "node-0001"
                ),
                call_label=f"companion-glossary-consolidation-l{level:04d}-n0001",
                stage="final",
                force=force,
                call_model=call_model,
            )

        batches = _consolidation_batches(
            current,
            language=language,
            protected_names=protected_names,
            source_language=source_language,
            entry_limit=entry_limit,
        )
        maximum_following_count = sum(
            min(entry_limit, max(1, len(batch) // 2)) for batch in batches
        )
        if maximum_following_count >= len(current):
            raise RuntimeError(
                "glossary consolidation cannot reduce this input within the prompt "
                "transport limits; refusing to spend calls on a no-progress level"
            )

        def merge_node(node: int, batch: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
            output_limit = min(entry_limit, max(1, len(batch) // 2))
            return node, _consolidate_node(
                [{"entries": batch}],
                blocks=blocks,
                language=language,
                protected_names=protected_names,
                source_language=source_language,
                entry_limit=entry_limit,
                output_limit=output_limit,
                cache_path=(
                    checkpoint_dir / "glossary-consolidation"
                    / f"level-{level:04d}" / f"{node:04d}.json"
                ),
                artifact_dir=(
                    checkpoint_dir / "llm" / "glossary" / "consolidation"
                    / f"level-{level:04d}" / f"node-{node:04d}"
                ),
                call_label=(
                    f"companion-glossary-consolidation-l{level:04d}-n{node:04d}"
                ),
                stage="intermediate",
                force=force,
                call_model=call_model,
            )

        merged: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=min(max(1, workers), len(batches))) as executor:
            futures = {
                executor.submit(merge_node, node, batch): node
                for node, batch in enumerate(batches, 1)
            }
            for future in as_completed(futures):
                node, value = future.result()
                merged[node] = value
        following = [
            entry
            for node in sorted(merged)
            for entry in merged[node].get("entries") or []
        ]
        if following and len(following) >= len(current):
            raise RuntimeError("hierarchical glossary consolidation did not reduce its input")
        current = following
        level += 1


def _consolidate_node(
    candidates: list[dict[str, Any]],
    *,
    blocks: list[dict[str, Any]],
    language: str,
    protected_names: list[str],
    source_language: str | None = None,
    entry_limit: int,
    output_limit: int,
    cache_path: Path | None,
    artifact_dir: Path,
    call_label: str,
    stage: str,
    force: bool,
    call_model: Callable[[str, dict[str, Any], Path, str], dict[str, Any]],
) -> dict[str, Any]:
    input_entry_count = sum(
        1
        for candidate in candidates
        for entry in candidate.get("entries") or []
        if isinstance(entry, dict)
    )
    cache_key = sha256_json({
        "glossary_version": GLOSSARY_VERSION,
        "stage": stage,
        "candidates": candidates,
        "language": language,
        "protected_names": protected_names,
        **({"source_language": source_language} if source_language else {}),
        "entry_limit": entry_limit,
        "output_limit": output_limit,
        "max_entries": CONSOLIDATION_MAX_ENTRIES,
        "prompt_max_bytes": CONSOLIDATION_PROMPT_MAX_BYTES,
        "prompt_target_bytes": CONSOLIDATION_PROMPT_TARGET_BYTES,
    })
    if cache_path is not None and cache_path.is_file() and not force:
        cached = _optional_json(cache_path)
        cached_result = cached.get("result")
        if cached.get("input_sha256") == cache_key and isinstance(cached_result, dict):
            raw_entries = cached_result.get("entries")
            if isinstance(raw_entries, list) and all(
                isinstance(entry, dict) for entry in raw_entries
            ):
                try:
                    normalized = _normalize_entries(
                        raw_entries, blocks,
                        require_exact_source=bool(source_language),
                    )
                    normalized = _deduplicate(normalized)[:output_limit]
                    _restore_protected_names(normalized, protected_names, language=language)
                    _validate_protected_names(normalized, protected_names)
                except (KeyError, TypeError, ValueError, RuntimeError):
                    pass
                else:
                    if input_entry_count == 0 or normalized:
                        return {"entries": normalized}

    prompt = glossary_consolidation_prompt(
        candidates,
        language=language,
        protected_names=protected_names,
        entry_limit=output_limit,
        source_language=source_language,
    )
    prompt_bytes = len(prompt.encode("utf-8"))
    if prompt_bytes >= CONSOLIDATION_PROMPT_MAX_BYTES:
        raise RuntimeError(
            f"glossary consolidation node {call_label} requires a {prompt_bytes}-byte "
            f"prompt, exceeding the strict {CONSOLIDATION_PROMPT_MAX_BYTES}-byte limit"
        )
    result = call_model(
        prompt,
        GLOSSARY_SCHEMA,
        artifact_dir,
        call_label,
    )
    entries = _normalize_entries(
        result.get("entries") or [], blocks,
        require_exact_source=bool(source_language),
    )
    entries = _deduplicate(entries)[:output_limit]
    _restore_protected_names(entries, protected_names, language=language)
    _validate_protected_names(entries, protected_names)
    if input_entry_count > 0 and not entries:
        raise RuntimeError(
            f"glossary consolidation node {call_label} returned no usable entries "
            f"for {input_entry_count} non-empty input entries"
        )
    value = {"entries": entries}
    if cache_path is not None:
        write_json(cache_path, {"input_sha256": cache_key, "result": value})
    return value


def _consolidation_input_fits(
    entries: list[dict[str, Any]],
    *,
    language: str,
    protected_names: list[str],
    source_language: str | None = None,
    output_limit: int,
) -> bool:
    return (
        len(entries) <= CONSOLIDATION_MAX_ENTRIES
        and _consolidation_prompt_bytes(
            [{"entries": entries}],
            language=language,
            protected_names=protected_names,
            source_language=source_language,
            output_limit=output_limit,
        ) < CONSOLIDATION_PROMPT_TARGET_BYTES
    )


def _consolidation_candidates_fit(
    candidates: list[dict[str, Any]],
    *,
    language: str,
    protected_names: list[str],
    source_language: str | None = None,
    output_limit: int,
) -> bool:
    entry_count = sum(
        1
        for candidate in candidates
        for entry in candidate.get("entries") or []
        if isinstance(entry, dict)
    )
    return (
        entry_count <= CONSOLIDATION_MAX_ENTRIES
        and _consolidation_prompt_bytes(
            candidates,
            language=language,
            protected_names=protected_names,
            source_language=source_language,
            output_limit=output_limit,
        ) < CONSOLIDATION_PROMPT_TARGET_BYTES
    )


def _consolidation_prompt_bytes(
    candidates: list[dict[str, Any]],
    *,
    language: str,
    protected_names: list[str],
    source_language: str | None = None,
    output_limit: int,
) -> int:
    prompt = glossary_consolidation_prompt(
        candidates,
        language=language,
        protected_names=protected_names,
        entry_limit=output_limit,
        source_language=source_language,
    )
    return len(prompt.encode("utf-8"))


def _consolidation_batches(
    entries: list[dict[str, Any]],
    *,
    language: str,
    protected_names: list[str],
    source_language: str | None = None,
    entry_limit: int,
) -> list[list[dict[str, Any]]]:
    _validate_consolidation_entry_transport(
        entries,
        language=language,
        protected_names=protected_names,
        source_language=source_language,
    )

    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for entry in entries:
        proposed = [*current, entry]
        proposed_output_limit = min(entry_limit, max(1, len(proposed) // 2))
        if current and (
            len(proposed) > CONSOLIDATION_MAX_ENTRIES
            or _consolidation_prompt_bytes(
                [{"entries": proposed}],
                language=language,
                protected_names=protected_names,
                source_language=source_language,
                output_limit=proposed_output_limit,
            ) >= CONSOLIDATION_PROMPT_TARGET_BYTES
        ):
            batches.append(current)
            current = [entry]
        else:
            current = proposed
    if current:
        batches.append(current)

    # Pair ordinary singleton nodes when the complete rendered prompt still
    # fits the target. A byte-heavy singleton may remain alone; the hard cap is
    # checked below and sibling nodes still provide count reduction.
    index = 0
    while len(batches) > 1 and index < len(batches):
        if len(batches[index]) != 1:
            index += 1
            continue
        if index + 1 < len(batches):
            proposed = [*batches[index], batches[index + 1][0]]
            if _consolidation_prompt_bytes(
                [{"entries": proposed}],
                language=language,
                protected_names=protected_names,
                source_language=source_language,
                output_limit=min(entry_limit, max(1, len(proposed) // 2)),
            ) < CONSOLIDATION_PROMPT_TARGET_BYTES:
                batches[index].append(batches[index + 1].pop(0))
                if not batches[index + 1]:
                    batches.pop(index + 1)
            index += 1
            continue
        proposed = [batches[index - 1][-1], *batches[index]]
        if _consolidation_prompt_bytes(
            [{"entries": proposed}],
            language=language,
            protected_names=protected_names,
            source_language=source_language,
            output_limit=min(entry_limit, max(1, len(proposed) // 2)),
        ) < CONSOLIDATION_PROMPT_TARGET_BYTES:
            batches[index].insert(0, batches[index - 1].pop())
            if not batches[index - 1]:
                batches.pop(index - 1)
        index += 1
    if any(len(batch) > CONSOLIDATION_MAX_ENTRIES for batch in batches):
        raise RuntimeError(
            "glossary consolidation batch exceeds the "
            f"{CONSOLIDATION_MAX_ENTRIES}-entry limit"
        )
    if any(
        _consolidation_prompt_bytes(
            [{"entries": batch}],
            language=language,
            protected_names=protected_names,
            source_language=source_language,
            output_limit=min(entry_limit, max(1, len(batch) // 2)),
        ) >= CONSOLIDATION_PROMPT_MAX_BYTES
        for batch in batches
    ):
        raise RuntimeError("glossary consolidation batch exceeds the strict prompt byte limit")
    return batches


def _validate_consolidation_entry_transport(
    entries: list[dict[str, Any]],
    *,
    language: str,
    protected_names: list[str],
    source_language: str | None = None,
) -> None:
    oversized = [
        index
        for index, entry in enumerate(entries, 1)
        if _consolidation_prompt_bytes(
            [{"entries": [entry]}],
            language=language,
            protected_names=protected_names,
            source_language=source_language,
            output_limit=1,
        ) >= CONSOLIDATION_PROMPT_MAX_BYTES
    ]
    if oversized:
        raise RuntimeError(
            "glossary consolidation contains an entry whose essential content exceeds "
            f"the strict {CONSOLIDATION_PROMPT_MAX_BYTES}-byte prompt limit: "
            f"entry {oversized[0]}"
        )


def _normalize_entries(
    entries: list[Any],
    blocks: list[dict[str, Any]],
    *,
    require_exact_source: bool = False,
) -> list[dict[str, Any]]:
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
        exact_first = _first_occurrence(source, blocks, case_sensitive=require_exact_source)
        if require_exact_source and not exact_first:
            continue
        first = str(raw.get("first_block_id") or "")
        if first not in valid_ids or require_exact_source:
            first = exact_first
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
            if name and _contains_protected_name(source, str(name))
        }
        names.update(
            name for name in protected_names if name and _contains_protected_name(source, name)
        )
        missing = [name for name in names if not _contains_protected_name(target, name)]
        if missing:
            raise RuntimeError(
                f"glossary translated or dropped protected personal names in {source!r}: {missing}"
            )


def _restore_protected_names(
    entries: list[dict[str, Any]],
    protected_names: list[str],
    *,
    language: str | None = None,
) -> None:
    """Retain a translated term while restoring source-matched names.

    Glossary models sometimes produce a perfectly standard translation but omit
    the source spelling that ARC protects (for example, ``Poisson bracket`` to
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
            if name and _contains_protected_name(source, name)
        ]
        entry["protected_names"] = matched

        target = str(entry.get("target_term") or "").strip()
        missing = [name for name in matched if not _contains_protected_name(target, name)]
        if not missing:
            continue

        # Prefer a containing full name over separately repeating its parts.
        # Re-check against the growing annotation because one insertion may
        # satisfy several protected lexical units (e.g. ``Ada Lovelace``).
        annotations: list[str] = []
        augmented = target
        for name in sorted(missing, key=lambda value: (-len(value), value.casefold())):
            if _contains_protected_name(augmented, name):
                continue
            annotations.append(name)
            augmented = f"{augmented} {name}"
        if annotations:
            if base_language(language) == "zh":
                entry["target_term"] = f"{target}（{'、'.join(annotations)}）"
            else:
                entry["target_term"] = f"{target} ({', '.join(annotations)})"


def _contains_protected_name(text: str, name: str) -> bool:
    """Match a protected source-form name as a complete Unicode lexical unit."""

    return contains_lexical_term(text, name)


def _first_occurrence(
    term: str,
    blocks: list[dict[str, Any]],
    *,
    case_sensitive: bool = False,
) -> str:
    for block in blocks:
        if contains_lexical_term(
            _canonical_block_text(block), term, case_sensitive=case_sensitive,
        ):
            return block_id(block)
    return ""


def _canonical_block_text(block: dict[str, Any]) -> str:
    values: list[str] = []

    def visit(value: Any, *, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                if child_key != "type":
                    visit(child, key=str(child_key))
        elif isinstance(value, list):
            for child in value:
                visit(child, key=key)
        elif value not in (None, ""):
            values.append(str(value))

    visit(_canonical_block_language(block))
    return "\n".join(values)


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
