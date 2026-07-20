from __future__ import annotations

from typing import Any
import re
import unicodedata

from bs4 import BeautifulSoup

from .source import block_id


def non_substantive_block_ids(document: dict[str, Any]) -> set[str]:
    """Return source blocks that must be preserved but not augmented.

    Explicit parser roles win.  The conservative fallback exists for local
    Markdown/OCR books whose front routes and contents entries often have no
    per-block role.  It identifies structure, not a particular language,
    title, author, or subject.
    """
    blocks = list(document.get("blocks") or [])
    if not blocks:
        return set()
    positions = {block_id(block): index for index, block in enumerate(blocks)}
    excluded: set[str] = set()

    excluded_roles = {
        "acknowledgement", "acknowledgements", "acknowledgment", "acknowledgments",
        "bibliography", "copyright", "front_matter_affiliations",
        "front_matter_authors", "front_matter_title", "publication",
        "publication_details", "references", "table_of_contents", "toc",
    }
    source_only_sections: set[str] = set()
    for block in blocks:
        role = str(block.get("source_role") or "").casefold()
        inferred_role = role or _structural_source_only_role(block)
        kind = _kind(block)
        if inferred_role in excluded_roles or kind in {
            "bibliography", "bibliography_item", "reference",
        }:
            excluded.add(block_id(block))
            section_id = str(block.get("section_id") or "")
            if section_id and inferred_role in {
                "acknowledgement", "acknowledgements", "acknowledgment",
                "acknowledgments", "bibliography", "copyright", "publication",
                "publication_details", "references",
            }:
                source_only_sections.add(section_id)
    excluded.update(
        block_id(block)
        for block in blocks
        if str(block.get("section_id") or "") in source_only_sections
    )

    front = document.get("front_matter") or {}
    structured_ids = front.get("block_ids") or {}
    if isinstance(structured_ids, dict):
        excluded_front_keys = {
            "affiliation", "affiliations", "author", "authors", "copyright",
            "institution", "institutions", "publication", "publisher", "title",
        }
        for key, values in structured_ids.items():
            if str(key).casefold() not in excluded_front_keys:
                continue
            if not isinstance(values, list):
                values = [values]
            excluded.update(str(value) for value in values if str(value) in positions)

    front_roles = _front_matter_block_roles(blocks, front)
    excluded.update(
        bid for bid, role in front_roles.items()
        if role in {"title", "author", "affiliation"}
    )

    toc_indices = [
        index for index, block in enumerate(blocks)
        if (
            str(block.get("source_role") or "").casefold() in {"table_of_contents", "toc"}
            or _structural_source_only_role(block) == "table_of_contents"
        )
    ]
    if toc_indices:
        toc_start = min(toc_indices)
        inferred_body_start = _body_start_after_contents(blocks, toc_indices)
        if inferred_body_start is not None:
            excluded.update(block_id(block) for block in blocks[toc_start:inferred_body_start])
        else:
            excluded.update(block_id(blocks[index]) for index in toc_indices)

        # Before a contents section, suppress metadata-like and list-oriented
        # route sections while retaining prose-rich chapters such as a preface.
        for group in _section_groups(blocks[:toc_start]):
            if any(front_roles.get(block_id(block)) == "abstract" for block in group):
                continue
            if _is_non_substantive_leading_group(group):
                excluded.update(block_id(block) for block in group)

    return excluded


def _structural_source_only_role(block: dict[str, Any]) -> str:
    html = str(block.get("html") or "").casefold()
    if re.search(r'(?:class|role)=["\'][^"\']*(?:ltx_toc|ltx_title_contents|doc-toc)', html):
        return "table_of_contents"
    if re.search(r'class=["\'][^"\']*acknowledg', html):
        return "acknowledgments"
    if re.search(r'class=["\'][^"\']*(?:bibliograph|reference)', html):
        return "references"
    if not _is_heading(block):
        return ""
    title = " ".join(re.findall(
        r"\w+", unicodedata.normalize(
            "NFKC", str(block.get("title") or block.get("text") or "")
        ).casefold(), flags=re.UNICODE,
    ))
    if title in {"contents", "table of contents"}:
        return "table_of_contents"
    if title in {"acknowledgment", "acknowledgments", "acknowledgement", "acknowledgements"}:
        return "acknowledgments"
    if title in {"references", "reference list", "bibliography", "literature cited"}:
        return "references"
    return ""


def _body_start_after_contents(
    blocks: list[dict[str, Any]], toc_indices: list[int]
) -> int | None:
    positions: dict[str, int] = {}
    for index, block in enumerate(blocks):
        for key in ("block_id", "source_id", "section_id", "id"):
            value = block.get(key)
            if value:
                positions.setdefault(str(value), index)

    linked_targets: list[int] = []
    for index in toc_indices:
        soup = BeautifulSoup(str(blocks[index].get("html") or ""), "html.parser")
        for anchor in soup.find_all("a"):
            href = str(anchor.get("href") or "")
            if href.startswith("#") and href[1:] in positions:
                target = positions[href[1:]]
                if target > index:
                    linked_targets.append(target)
    if linked_targets:
        return min(linked_targets)

    last_toc = max(toc_indices)
    headings = [
        (index, _heading_signature(block))
        for index, block in enumerate(blocks[last_toc + 1:], start=last_toc + 1)
        if _is_heading(block) and _heading_signature(block)
    ]
    # OCR/Markdown contents headings commonly reappear at the start of the
    # actual text.  The earliest repeated signature marks that boundary and
    # naturally keeps a substantive preface when it is the first real entry.
    for offset, (_, signature) in enumerate(headings):
        for later_index, later_signature in headings[offset + 1:]:
            if signature == later_signature:
                return later_index
    return None


def _front_matter_block_roles(
    blocks: list[dict[str, Any]], front: dict[str, Any]
) -> dict[str, str]:
    structural_role_map = {
        "front_matter_title": "title",
        "front_matter_authors": "author",
        "front_matter_affiliations": "affiliation",
        "front_matter_abstract": "abstract",
    }
    roles = {
        block_id(block): structural_role_map[str(block.get("source_role") or "").casefold()]
        for block in blocks
        if str(block.get("source_role") or "").casefold() in structural_role_map
    }
    for block in blocks:
        block_roles = {
            structural_role_map[value]
            for value in block.get("front_matter_roles") or []
            if value in structural_role_map
        }
        if "title" in block_roles:
            roles[block_id(block)] = "title"
        elif "author" in block_roles:
            roles[block_id(block)] = "author"
        elif "affiliation" in block_roles:
            roles[block_id(block)] = "affiliation"

    candidates: list[tuple[str, str]] = []
    title = _front_value(front.get("title")) if front.get("title") else ""
    if title:
        candidates.append(("title", title))
    authors = front.get("authors") or []
    if not isinstance(authors, list):
        authors = [authors]
    candidates.extend(("author", _author_name(value)) for value in authors)
    affiliations = front.get("affiliations") or front.get("institutions") or []
    if not isinstance(affiliations, list):
        affiliations = [affiliations]
    candidates.extend(("affiliation", _front_value(value)) for value in affiliations)
    abstract = _front_value(front.get("abstract")) if front.get("abstract") else ""
    if abstract:
        candidates.append(("abstract", abstract))

    used: set[tuple[str, str]] = set()
    for block in blocks:
        if block_id(block) in roles or block.get("section_id"):
            continue
        text = " ".join(str(block.get("text") or block.get("title") or "").split())
        for role, value in candidates:
            normalized = " ".join(value.split())
            key = (role, normalized)
            if key in used or not normalized:
                continue
            if text == normalized or (role == "abstract" and text and text in normalized):
                roles[block_id(block)] = role
                used.add(key)
                break
    return roles


def _heading_signature(block: dict[str, Any]) -> str:
    text = unicodedata.normalize(
        "NFKC", str(block.get("title") or block.get("text") or "")
    ).casefold()
    text = re.sub(r"\.{2,}\s*\d+\s*$", "", text)
    text = re.sub(r"\s+\d+\s*$", "", text)
    return " ".join(re.findall(r"\w+", text, flags=re.UNICODE))


def _is_heading(block: dict[str, Any]) -> bool:
    return bool(
        _kind(block) in {"heading", "section", "subsection", "subsubsection"}
        or block.get("heading_level")
        or isinstance(block.get("heading"), dict)
    )


def _section_groups(blocks: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_section = object()
    for block in blocks:
        section = block.get("section_id")
        if current and section != current_section:
            groups.append(current)
            current = []
        current.append(block)
        current_section = section
    if current:
        groups.append(current)
    return groups


def _is_non_substantive_leading_group(group: list[dict[str, Any]]) -> bool:
    if not group:
        return False
    if any(
        _kind(block) in {"equation", "math", "display_math", "figure", "image", "table"}
        for block in group
    ):
        return False
    prose = [block for block in group if not _is_heading(block)]
    if not prose:
        return True
    list_like = [
        block for block in prose
        if block.get("list_kind")
        or _kind(block) in {"list", "ordered_list", "unordered_list"}
        or re.match(r"^\s*(?:[-*+•]|\d+[.)])\s+", str(block.get("text") or ""))
    ]
    if len(list_like) >= 2:
        return True
    prose_text = [" ".join(str(block.get("text") or "").split()) for block in prose]
    # Contact/publication fragments are usually several short fields and often
    # contain a URI or e-mail address.  A short but continuous prose paragraph
    # is retained: it may be a substantive foreword or preface.
    has_route_token = any(
        re.search(r"(?:https?://|www\.|\b[^\s@]+@[^\s@]+\b)", text, flags=re.IGNORECASE)
        for text in prose_text
    )
    return bool(
        len(prose_text) >= 2
        and max((len(text) for text in prose_text), default=0) < 100
        and (
            has_route_token
            or (len(prose_text) >= 3 and max(len(text) for text in prose_text) < 45)
        )
    )


def _kind(block: dict[str, Any]) -> str:
    return str(block.get("type") or block.get("kind") or "text").casefold()


def _author_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("name") or value.get("full_name") or "")
    return str(value)


def _front_value(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("text", "name", "value", "title"):
            if value.get(key):
                return str(value[key])
        return "; ".join(
            str(item) for item in value.values() if item is not None and item != ""
        )
    return str(value)
