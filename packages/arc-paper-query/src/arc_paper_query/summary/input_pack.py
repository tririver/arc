from __future__ import annotations

import hashlib
import json
from typing import Any

from ..ids import normalize_paper_id


def build_input_pack(
    paper_id: str,
    *,
    metadata: dict[str, Any],
    parsed: dict[str, Any],
    references: list[dict[str, Any]],
    max_section_chars: int = 12000,
) -> dict[str, Any]:
    normalized = normalize_paper_id(paper_id)
    sections = [
        {
            "section_id": section.get("section_id", ""),
            "title": section.get("title", ""),
            "text": _truncate_middle(str(section.get("text") or ""), max_section_chars),
        }
        for section in parsed.get("sections", [])
    ]
    pack = {
        "paper_id": normalized,
        "metadata": metadata,
        "toc": parsed.get("toc") or [],
        "sections": sections,
        "references": references,
    }
    pack["source_hash"] = _source_hash(pack)
    return pack


def _source_hash(pack: dict[str, Any]) -> str:
    payload = json.dumps(pack, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _truncate_middle(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    keep = max(1, (max_chars - len("\n[truncated]\n")) // 2)
    return text[:keep].rstrip() + "\n[truncated]\n" + text[-keep:].lstrip()
