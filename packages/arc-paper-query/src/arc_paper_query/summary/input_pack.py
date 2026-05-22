from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from ..ids import normalize_paper_id


def build_input_pack(
    paper_id: str,
    *,
    metadata: dict[str, Any],
    parsed: dict[str, Any],
    max_section_chars: int = 12000,
) -> dict[str, Any]:
    normalized = normalize_paper_id(paper_id)
    summary_sections = _summary_sections(parsed.get("sections", []))
    summary_toc = _summary_toc(parsed.get("toc") or [])
    sections = [
        {
            "section_id": section.get("section_id", ""),
            "title": section.get("title", ""),
            "level": section.get("level"),
            "text": _truncate_middle(str(section.get("text") or ""), max_section_chars),
        }
        for section in summary_sections
    ]
    pack = {
        "paper_id": normalized,
        "metadata": metadata,
        "toc": summary_toc,
        "sections": sections,
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


def _summary_sections(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        section
        for section in sections
        if _is_summary_level(section.get("level")) and not _is_non_content_section(section)
    ]


def _summary_toc(toc: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item
        for item in toc
        if _is_summary_level(item.get("level")) and not _is_non_content_section(item)
    ]


def _is_summary_level(level: Any) -> bool:
    try:
        return int(level) <= 2
    except (TypeError, ValueError):
        return True


def _is_non_content_section(section: dict[str, Any]) -> bool:
    section_id = str(section.get("section_id") or section.get("id") or "").lower()
    title = re.sub(r"^\s*(?:appendix\s+)?[a-z0-9.]+\s+", "", str(section.get("title") or "").strip(), flags=re.I)
    title = title.lower()
    return (
        section_id in {"bib", "references", "acknowledgments", "acknowledgements"}
        or title in {"references", "bibliography", "acknowledgments", "acknowledgements"}
    )
