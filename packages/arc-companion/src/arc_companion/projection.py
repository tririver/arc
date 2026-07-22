from __future__ import annotations

import re
from typing import Any

from .source import block_id


NON_TRANSLATABLE_TYPES = {
    "equation", "math", "display_math", "figure", "table",
    "bibliography", "bibliography_item", "reference",
}

STRUCTURAL_TYPES = {
    "heading", "section", "subsection", "subsubsection", "chapter", "part",
}


def is_structural(block: dict[str, Any]) -> bool:
    """Return whether a block is a source heading preserved only for navigation."""
    kind = str(block.get("type") or block.get("kind") or "").casefold()
    return kind in STRUCTURAL_TYPES


def is_translatable(block: dict[str, Any]) -> bool:
    kind = str(block.get("type") or block.get("kind") or "").casefold()
    if kind in NON_TRANSLATABLE_TYPES or is_structural(block):
        return False
    return bool(str(block.get("text") or block.get("title") or "").strip())


def translation_input_block(block: dict[str, Any]) -> dict[str, Any]:
    """Project one rich source block exactly as the translation prompt sees it."""
    inline_runs = list(block.get("inline_runs") or [])
    text = str(block.get("text") or block.get("title") or "")
    if inline_runs:
        text = "".join(
            str(run.get("separator_before") or "") + (
                str(run.get("content") or "")
                if str(run.get("kind") or "") == "text"
                else opaque_inline_token(run)
            )
            for run in inline_runs
            if isinstance(run, dict)
        )
    value = {
        "block_id": block_id(block),
        "type": str(block.get("type") or block.get("kind") or "text"),
        "text": text,
    }
    for key in ("section_id", "level", "items", "list_items"):
        if key in block:
            value[key] = block[key]
    return value


def opaque_inline_token(run: dict[str, Any]) -> str:
    token_id = str(run.get("token_id") or "")
    content_hash = str(run.get("content_hash") or "")
    if not token_id or not re.fullmatch(r"[0-9a-f]{64}", content_hash):
        raise RuntimeError("rich inline run lacks a stable token id or content hash")
    return f"[[ARC_INLINE:{token_id}:{content_hash}]]"


def opaque_inline_tokens(block: dict[str, Any]) -> list[str]:
    return [
        opaque_inline_token(run)
        for run in block.get("inline_runs") or []
        if isinstance(run, dict) and str(run.get("kind") or "") != "text"
    ]


def annotation_input_block(
    block: dict[str, Any], document: dict[str, Any]
) -> dict[str, Any]:
    """Project one rich source block exactly as the commentary prompt sees it."""
    kind = str(block.get("type") or block.get("kind") or "text")
    value: dict[str, Any] = {"block_id": block_id(block), "type": kind}
    for key in (
        "text", "title", "caption", "section_id", "level",
        "items", "list_items",
    ):
        if key in block:
            value[key] = prompt_safe_value(block[key])
    singular = {"equation": "equations", "figure": "figures", "table": "tables"}.get(
        kind.casefold()
    )
    if singular:
        entity_id = str(
            block.get(f"{singular[:-1]}_id")
            or block.get("entity_id")
            or block.get("ref_id")
            or block_id(block)
        )
        for entity in document.get(singular) or []:
            candidate_id = str(
                entity.get("id")
                or entity.get(f"{singular[:-1]}_id")
                or entity.get("block_id")
                or ""
            )
            if entity_id and candidate_id == entity_id:
                allowed = {
                    key: prompt_safe_value(entity[key])
                    for key in (
                        "id", "tex", "printed_equation_numbers", "tag", "caption",
                        "rows", "column_count",
                    )
                    if key in entity
                }
                value[singular[:-1]] = allowed
                break
    return value


def prompt_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: prompt_safe_value(item)
            for key, item in value.items()
            if "html" not in str(key).casefold()
            and str(key).casefold() not in {"cache_path", "local_path", "asset_bytes"}
        }
    if isinstance(value, list):
        return [prompt_safe_value(item) for item in value]
    return value
