from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .projection import is_structural
from .source import block_id


CHAPTERS_VERSION = "arc.companion.chapters.v2"


class ChapterStructureError(ValueError):
    """The authoritative source structure cannot be mapped to rich blocks."""


@dataclass(frozen=True)
class ChapterRange:
    chapter_id: str
    title: str
    block_ids: tuple[str, ...]
    title_block_ids: tuple[str, ...] = ()
    structural_block_ids: tuple[str, ...] = ()
    content_block_ids: tuple[str, ...] = ()
    page_start: int | None = None
    page_end: int | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "chapter_id": self.chapter_id,
            "title": self.title,
            "block_ids": list(self.block_ids),
            "title_block_ids": list(self.title_block_ids),
            "structural_block_ids": list(self.structural_block_ids),
            "content_block_ids": list(self.content_block_ids),
            "start_block_id": self.block_ids[0],
            "end_block_id": self.block_ids[-1],
            "page_start": self.page_start,
            "page_end": self.page_end,
        }


def build_chapters(
    document: Mapping[str, Any],
    *,
    structure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project authoritative chapter boundaries onto a rich document.

    ARC-Paper owns boundary selection.  The fallback is intentionally modest:
    it uses real top-level headings for articles and otherwise keeps the
    substantive document as one chapter.  It never invents headings.
    """
    blocks = [dict(item) for item in document.get("blocks") or [] if isinstance(item, Mapping)]
    if not blocks:
        raise ChapterStructureError("source document contains no rich blocks")
    excluded = {
        block_id(item)
        for item in blocks
        if _source_role(item) in {"cover", "contents", "acknowledgements", "references", "index"}
    }
    excluded.update(
        str(value) for value in (structure or {}).get("excluded_block_ids") or []
        if str(value)
    )
    expected = [block_id(item) for item in blocks if block_id(item) not in excluded]
    if not expected:
        raise ChapterStructureError("source document contains no substantive blocks")

    authoritative = list((structure or {}).get("chapters") or [])
    if authoritative:
        section_blocks: dict[str, list[str]] = {}
        for item in blocks:
            identifier = str(item.get("section_id") or "")
            if identifier and block_id(item) in expected:
                section_blocks.setdefault(identifier, []).append(block_id(item))
        authoritative = [
            {
                **dict(item),
                "block_ids": (
                    list(item.get("block_ids") or [])
                    or [value for section_id in item.get("section_ids") or []
                        for value in section_blocks.get(str(section_id), [])]
                ),
                "page_start": item.get("page_start", item.get("pdf_page_start")),
                "page_end": item.get("page_end", item.get("pdf_page_end")),
            }
            for item in authoritative if isinstance(item, Mapping)
        ]
        ranges = _from_authoritative(authoritative, expected, blocks)
    else:
        ranges = _from_real_headings(blocks, expected)
    validate_chapter_coverage(ranges, expected)
    return {
        "schema_version": CHAPTERS_VERSION,
        "document_kind": str((structure or {}).get("document_kind") or "article"),
        "excluded_block_ids": sorted(excluded, key=lambda value: _position(value, blocks)),
        "chapters": [item.to_json() for item in ranges],
        "coverage": {
            "expected_block_count": len(expected),
            "covered_block_count": sum(len(item.block_ids) for item in ranges),
            "exactly_once": True,
        },
    }


def validate_chapter_coverage(
    chapters: Iterable[ChapterRange | Mapping[str, Any]], expected_block_ids: Iterable[str]
) -> None:
    expected = [str(value) for value in expected_block_ids]
    actual: list[str] = []
    chapter_ids: set[str] = set()
    for item in chapters:
        if isinstance(item, ChapterRange):
            chapter_id, ids = item.chapter_id, list(item.block_ids)
            title_ids = list(item.title_block_ids)
            structural_ids = list(item.structural_block_ids)
            content_ids = list(item.content_block_ids)
        else:
            chapter_id = str(item.get("chapter_id") or "")
            ids = [str(value) for value in item.get("block_ids") or []]
            title_ids = [str(value) for value in item.get("title_block_ids") or []]
            structural_ids = [str(value) for value in item.get("structural_block_ids") or []]
            content_ids = [str(value) for value in item.get("content_block_ids") or []]
        if not chapter_id or chapter_id in chapter_ids:
            raise ChapterStructureError(f"invalid or duplicate chapter id: {chapter_id!r}")
        if not ids:
            raise ChapterStructureError(f"chapter {chapter_id} contains no blocks")
        if title_ids or structural_ids or content_ids:
            if [*structural_ids, *content_ids] != [
                value for value in ids if value in set(structural_ids)
            ] + [value for value in ids if value in set(content_ids)]:
                # The simpler set checks below report invalid partitions; this
                # branch catches duplicate or out-of-order projection fields.
                raise ChapterStructureError(
                    f"chapter {chapter_id} structural/content projections are not ordered"
                )
            if (
                set(structural_ids) & set(content_ids)
                or set(structural_ids) | set(content_ids) != set(ids)
                or len(structural_ids) + len(content_ids) != len(ids)
            ):
                raise ChapterStructureError(
                    f"chapter {chapter_id} structural/content projections do not partition blocks"
                )
            if not set(title_ids).issubset(structural_ids):
                raise ChapterStructureError(
                    f"chapter {chapter_id} title blocks are not structural blocks"
                )
        chapter_ids.add(chapter_id)
        actual.extend(ids)
    if actual != expected:
        raise ChapterStructureError("chapters do not cover substantive source blocks exactly once in order")


def _from_authoritative(
    items: list[Any], expected: list[str], blocks: list[dict[str, Any]],
) -> list[ChapterRange]:
    positions = {value: index for index, value in enumerate(expected)}
    by_id = {block_id(item): item for item in blocks}
    output: list[ChapterRange] = []
    cursor = 0
    for ordinal, raw in enumerate(items, 1):
        if not isinstance(raw, Mapping):
            raise ChapterStructureError("authoritative chapter record is not an object")
        ids = [str(value) for value in raw.get("block_ids") or []]
        if not ids:
            start, end = str(raw.get("start_block_id") or ""), str(raw.get("end_block_id") or "")
            if start not in positions or end not in positions or positions[end] < positions[start]:
                raise ChapterStructureError("authoritative chapter has invalid block boundaries")
            ids = expected[positions[start] : positions[end] + 1]
        if ids[0] not in positions or positions[ids[0]] != cursor:
            raise ChapterStructureError("authoritative chapters are not contiguous")
        title = str(raw.get("title") or "").strip()
        structural_ids = tuple(value for value in ids if is_structural(by_id[value]))
        explicit_title_ids = tuple(str(value) for value in raw.get("title_block_ids") or [])
        if explicit_title_ids:
            if any(value not in structural_ids for value in explicit_title_ids):
                raise ChapterStructureError(
                    "authoritative chapter title blocks must be structural chapter members"
                )
            title_ids = explicit_title_ids
        else:
            title_ids = tuple(
                value for value in structural_ids
                if title and _heading_title(by_id[value]) == title
            )[:1]
        output.append(ChapterRange(
            chapter_id=f"ch-{ordinal:04d}",
            title=title,
            block_ids=tuple(ids),
            title_block_ids=title_ids,
            structural_block_ids=structural_ids,
            content_block_ids=tuple(value for value in ids if value not in structural_ids),
            page_start=_optional_int(raw.get("page_start")),
            page_end=_optional_int(raw.get("page_end")),
        ))
        cursor += len(ids)
    return output


def _from_real_headings(blocks: list[dict[str, Any]], expected: list[str]) -> list[ChapterRange]:
    selected = [item for item in blocks if block_id(item) in set(expected)]
    heading_levels = [
        _heading_level(item) for item in selected if _heading_level(item) is not None and _heading_title(item)
    ]
    if not heading_levels:
        return [_chapter_range("ch-0001", "", expected, selected)]
    level = min(heading_levels)
    starts = [index for index, item in enumerate(selected) if _heading_level(item) == level and _heading_title(item)]
    if len(starts) < 2:
        title = _heading_title(selected[starts[0]]) if starts else ""
        return [_chapter_range("ch-0001", title, expected, selected)]
    if starts[0] != 0:
        starts.insert(0, 0)
    starts.append(len(selected))
    output: list[ChapterRange] = []
    for ordinal, (start, end) in enumerate(zip(starts, starts[1:]), 1):
        group = selected[start:end]
        output.append(_chapter_range(
            f"ch-{ordinal:04d}", _heading_title(group[0]),
            [block_id(item) for item in group], group,
        ))
    return output


def _chapter_range(
    chapter_id: str, title: str, ids: list[str], blocks: list[dict[str, Any]],
) -> ChapterRange:
    structural_ids = tuple(block_id(item) for item in blocks if is_structural(item))
    title_ids = tuple(
        block_id(item) for item in blocks
        if is_structural(item) and title and _heading_title(item) == title
    )[:1]
    return ChapterRange(
        chapter_id, title, tuple(ids),
        title_block_ids=title_ids,
        structural_block_ids=structural_ids,
        content_block_ids=tuple(value for value in ids if value not in structural_ids),
    )


def _source_role(block: Mapping[str, Any]) -> str:
    return str(block.get("source_role") or block.get("role") or "").strip().casefold()


def _heading_level(block: Mapping[str, Any]) -> int | None:
    kind = str(block.get("type") or block.get("kind") or "").casefold()
    if kind not in {"heading", "section", "subsection", "subsubsection", "chapter", "part"}:
        return None
    value = block.get("level") or block.get("heading_level")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return {"part": 1, "chapter": 2, "section": 1, "subsection": 2, "subsubsection": 3}.get(kind, 1)


def _heading_title(block: Mapping[str, Any]) -> str:
    return str(block.get("title") or block.get("text") or "").strip()


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _position(value: str, blocks: list[dict[str, Any]]) -> int:
    return next((index for index, item in enumerate(blocks) if block_id(item) == value), len(blocks))
