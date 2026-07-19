from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup, NavigableString, Tag


DOCUMENT_SCHEMA_VERSION = "arc.paper.document.v2"
_HEADING_NAMES = {"h1", "h2", "h3", "h4", "h5", "h6"}
_ATOMIC_NAMES = _HEADING_NAMES | {"p", "ul", "ol", "pre", "blockquote"}
_ASSET_ATTRIBUTES = (("img", "src"), ("source", "src"), ("object", "data"))


def build_document(
    html: str,
    *,
    paper_id: str,
    source_url: str = "",
    assets: list[dict[str, Any]] | None = None,
    equations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the loss-aware ar5iv document contract.

    The contract deliberately keeps raw HTML fragments beside normalized text.
    Consumers can therefore render deterministically without asking an LLM to
    reconstruct source content that the normalized compatibility view omits.
    """

    soup = BeautifulSoup(html, "lxml")
    root = _academic_root(soup)
    asset_records = _asset_metadata(root, source_url=source_url, assets=assets or [])
    asset_by_url = {str(item.get("source_url") or ""): item for item in asset_records}
    enriched_equations = _equations(root, equations or [])
    figures = _figures(root, source_url=source_url, asset_by_url=asset_by_url)
    tables = _tables(root)
    bibliography = _bibliography(root)
    footnotes = _footnotes(root)
    links = _links(root, source_url=source_url)
    blocks = _blocks(root, figures=figures, tables=tables, bibliography=bibliography, equations=enriched_equations)
    integrity = _integrity(
        root,
        blocks=blocks,
        assets=asset_records,
        equations=enriched_equations,
        figures=figures,
        tables=tables,
        footnotes=footnotes,
        bibliography=bibliography,
        links=links,
    )
    document_hash = hashlib.sha256(
        (html + "\n" + "\n".join(str(item.get("sha256") or "") for item in asset_records)).encode("utf-8")
    ).hexdigest()
    asset_manifest_hash = hashlib.sha256(
        "\n".join(
            f"{item.get('source_url', '')}\t{item.get('sha256', '')}\t{item.get('status', '')}"
            for item in sorted(asset_records, key=lambda value: str(value.get("source_url") or ""))
        ).encode("utf-8")
    ).hexdigest()

    return {
        "schema_version": DOCUMENT_SCHEMA_VERSION,
        "source": {
            "paper_id": paper_id,
            "provider": "ar5iv" if "ar5iv.labs.arxiv.org" in source_url else "html",
            "url": source_url,
            "html_sha256": hashlib.sha256(html.encode("utf-8")).hexdigest(),
        },
        "front_matter": _front_matter(root),
        "blocks": blocks,
        "equations": enriched_equations,
        "figures": figures,
        "tables": tables,
        "footnotes": footnotes,
        "bibliography": bibliography,
        "links": links,
        "assets": asset_records,
        "integrity": integrity,
        "document_hash": document_hash,
        "asset_manifest_hash": asset_manifest_hash,
    }


def discover_asset_urls(html: str, *, source_url: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "lxml")
    root = _academic_root(soup)
    discovered: list[dict[str, str]] = []
    seen: set[str] = set()
    for tag_name, attribute in _ASSET_ATTRIBUTES:
        for element in root.find_all(tag_name):
            if not isinstance(element, Tag):
                continue
            raw_url = str(element.get(attribute) or "").strip()
            if not raw_url:
                continue
            resolved = urljoin(source_url, raw_url)
            if resolved in seen:
                continue
            seen.add(resolved)
            discovered.append({"source_url": resolved, "original_url": raw_url})
    return discovered


def _academic_root(soup: BeautifulSoup) -> Tag | BeautifulSoup:
    for selector in ("article.ltx_document", "article", "body"):
        found = soup.select_one(selector)
        if isinstance(found, Tag):
            return found
    return soup


def _front_matter(root: Tag | BeautifulSoup) -> dict[str, Any]:
    title = root.select_one(".ltx_title_document, h1.ltx_title, h1")
    abstract = root.select_one(".ltx_abstract")
    authors = [
        _text(item)
        for item in root.select(".ltx_creator_author, .ltx_personname, .ltx_author_name")
        if _text(item)
    ]
    affiliations = [_text(item) for item in root.select(".ltx_role_affiliation, .ltx_affiliation") if _text(item)]
    return {
        "title": _text(title),
        "authors": _dedupe(authors),
        "affiliations": _dedupe(affiliations),
        "abstract": _text(abstract),
    }


def _blocks(
    root: Tag | BeautifulSoup,
    *,
    figures: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    bibliography: list[dict[str, Any]],
    equations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    special_by_node_id: dict[int, tuple[str, str]] = {}
    for selector, kind, items in (
        ("__figures__", "figure", figures),
        (".ltx_table", "table", tables),
        (".ltx_bibitem", "bibliography", bibliography),
        (None, "equation", equations),
        (".ltx_note, .ltx_role_footnote", "footnote", []),
    ):
        if selector is None:
            nodes = _equation_nodes(root)
        elif selector == "__figures__":
            nodes = _figure_nodes(root)
        else:
            nodes = [node for node in root.select(selector) if isinstance(node, Tag)]
        items_by_id = {str(item.get("id") or ""): item for item in items}
        for index, node in enumerate(nodes):
            node_id = str(node.get("id") or "")
            indexed = items[index] if index < len(items) else {}
            item_id = str((items_by_id.get(node_id) or indexed).get("id") or node_id)
            special_by_node_id[id(node)] = (kind, item_id)

    selected: list[Tag] = []
    for node in root.find_all(True):
        if not isinstance(node, Tag) or _inside_site_chrome(node):
            continue
        special = special_by_node_id.get(id(node))
        is_atomic = node.name in _ATOMIC_NAMES or special is not None
        if not is_atomic:
            continue
        if node.name in {"ul", "ol"} and node.select_one(".ltx_bibitem") is not None:
            continue
        if any(id(parent) in special_by_node_id for parent in node.parents if isinstance(parent, Tag)):
            continue
        if node.name in _ATOMIC_NAMES and any(parent in selected for parent in node.parents):
            continue
        if _text(node) or special is not None:
            selected.append(node)

    blocks: list[dict[str, Any]] = []
    used_ids: Counter[str] = Counter()
    for order, node in enumerate(selected, start=1):
        special = special_by_node_id.get(id(node))
        kind = special[0] if special else (
            "heading" if node.name in _HEADING_NAMES else ("list" if node.name in {"ul", "ol"} else "prose")
        )
        preferred = (special[1] if special else "") or str(node.get("id") or f"block-{order:06d}")
        used_ids[preferred] += 1
        block_id = preferred if used_ids[preferred] == 1 else f"{preferred}--{used_ids[preferred]}"
        section = node.find_parent("section")
        blocks.append(
            _block_record(
                node,
                block_id=block_id,
                order=order,
                kind=kind,
                source_id=(special[1] if special else str(node.get("id") or "")),
                section_id=str(section.get("id") or "") if isinstance(section, Tag) else "",
            )
        )
    return blocks


def _block_record(
    node: Tag,
    *,
    block_id: str,
    order: int,
    kind: str,
    source_id: str,
    section_id: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "block_id": block_id,
        "order": order,
        "kind": kind,
        "source_id": source_id,
        "section_id": section_id,
        "text": _text(node),
        "html": str(node),
    }
    record["inline_runs"] = _inline_runs(node, block_id=block_id)
    if kind == "heading":
        tag = _text(node.select_one(":scope > .ltx_tag, .ltx_tag"))
        full_title = _text(node)
        record.update(
            {
                "level": int(node.name[1]) if node.name and node.name in _HEADING_NAMES else 1,
                "tag": tag,
                "title": full_title[len(tag) :].strip() if tag and full_title.startswith(tag) else full_title,
            }
        )
    elif kind == "list":
        items = _list_items(node, block_id=block_id)
        record.update(
            {
                "list_kind": "ordered" if node.name == "ol" else "unordered",
                "ordered": node.name == "ol",
                "start": _positive_int(node.get("start")) if node.name == "ol" and node.get("start") else None,
                "reversed": node.has_attr("reversed"),
                "items": items,
            }
        )
    return record


def _inline_runs(node: Tag, *, block_id: str) -> list[dict[str, Any]]:
    """Preserve translatable text and opaque inline material separately.

    Runs follow DOM order.  Math, citations, and links are controller-owned
    tokens: their token ids and hashes let downstream consumers prove that a
    translation neither dropped nor reordered source material.
    """

    values: list[tuple[str, str, dict[str, Any]]] = []

    def append(kind: str, content: str, **extra: Any) -> None:
        normalized = re.sub(r"\s+", " ", content).strip() if kind == "text" else content.strip()
        if not normalized:
            return
        if kind == "text" and values and values[-1][0] == "text":
            previous_kind, previous, previous_extra = values[-1]
            values[-1] = (previous_kind, f"{previous} {normalized}".strip(), previous_extra)
            return
        values.append((kind, normalized, extra))

    def visit(value: Tag | NavigableString) -> None:
        if isinstance(value, NavigableString):
            append("text", str(value))
            return
        if not isinstance(value, Tag):
            return
        if value.name == "math":
            annotation = value.find("annotation", attrs={"encoding": "application/x-tex"})
            tex = _text(annotation) if isinstance(annotation, Tag) else str(value.get("alttext") or "").strip()
            mathml = str(value)
            append("math", tex or mathml, tex=tex, mathml=mathml)
            return
        if value.name == "a":
            href = str(value.get("href") or "").strip()
            visible = _text(value)
            kind = "citation" if "bib" in href.casefold() or "ltx_ref" in set(value.get("class") or []) else "link"
            append(kind, visible or href, href=href, target_id=href[1:] if href.startswith("#") else "")
            return
        for child in value.children:
            if isinstance(child, (Tag, NavigableString)):
                visit(child)

    visit(node)
    runs: list[dict[str, Any]] = []
    for order, (kind, content, extra) in enumerate(values, start=1):
        material = {
            "kind": kind,
            "content": content,
            **extra,
        }
        content_hash = hashlib.sha256(
            repr(sorted(material.items())).encode("utf-8")
        ).hexdigest()
        runs.append(
            {
                "run_id": f"{block_id}.run-{order:04d}",
                "token_id": f"{block_id}.token-{order:04d}-{content_hash[:12]}",
                "order": order,
                "kind": kind,
                "content_hash": content_hash,
                **material,
            }
        )
    return runs


def _list_items(node: Tag, *, block_id: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item_order, item in enumerate(node.find_all("li", recursive=False), start=1):
        item_id = str(item.get("id") or f"{block_id}.item-{item_order}")
        nested = [
            child
            for child in item.find_all(("ul", "ol"), recursive=False)
            if isinstance(child, Tag)
        ]
        items.append(
            {
                "id": item_id,
                "order": item_order,
                "value": str(item.get("value") or ""),
                "text": _text(item),
                "html": str(item),
                "children": [
                    {
                        "list_kind": "ordered" if child.name == "ol" else "unordered",
                        "ordered": child.name == "ol",
                        "start": _positive_int(child.get("start"))
                        if child.name == "ol" and child.get("start")
                        else None,
                        "reversed": child.has_attr("reversed"),
                        "items": _list_items(child, block_id=f"{item_id}.list-{child_order}"),
                        "html": str(child),
                    }
                    for child_order, child in enumerate(nested, start=1)
                ],
            }
        )
    return items


def _equation_nodes(root: Tag | BeautifulSoup) -> list[Tag]:
    nodes: list[Tag] = []
    for node in root.find_all(("table", "div", "span", "tr")):
        if not isinstance(node, Tag):
            continue
        classes = set(node.get("class") or [])
        if node.name == "tr" and "ltx_equation" in classes:
            if node.find_parent("table", class_="ltx_equationgroup") is not None:
                nodes.append(node)
            continue
        if node.name in {"table", "div", "span"} and "ltx_equation" in classes:
            if node.find_parent(class_="ltx_equation") is None:
                nodes.append(node)
    return nodes


def _figure_nodes(root: Tag | BeautifulSoup) -> list[Tag]:
    return [
        node
        for node in root.select(".ltx_figure")
        if isinstance(node, Tag) and "ltx_table" not in set(node.get("class") or []) and node.select_one(".ltx_table") is None
    ]


def _equation_group_rows(group: Tag) -> list[Tag]:
    return [row for row in group.select("tr.ltx_equation") if isinstance(row, Tag)]


def _equation_node_id(node: Tag, *, group: Tag | None, row_index: int) -> str:
    if node.get("id"):
        return str(node["id"])
    tbody = node.find_parent("tbody")
    if isinstance(tbody, Tag) and tbody.get("id"):
        rows = tbody.find_all("tr", class_="ltx_equation", recursive=False)
        if len(rows) == 1:
            return str(tbody["id"])
    if isinstance(group, Tag) and group.get("id") and row_index >= 0:
        return f"{group['id']}.row-{row_index + 1}"
    if isinstance(tbody, Tag) and tbody.get("id") and row_index >= 0:
        return f"{tbody['id']}.row-{row_index + 1}"
    return ""


def _equations(root: Tag | BeautifulSoup, compatibility: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(item.get("id") or ""): dict(item) for item in compatibility}
    result: list[dict[str, Any]] = []
    nodes = _equation_nodes(root)
    for order, node in enumerate(nodes, start=1):
        if not isinstance(node, Tag):
            continue
        group = node.find_parent("table", class_="ltx_equationgroup")
        group_rows = _equation_group_rows(group) if isinstance(group, Tag) else []
        group_row_index = group_rows.index(node) if node in group_rows else -1
        equation_id = _equation_node_id(node, group=group, row_index=group_row_index) or f"equation-{order:06d}"
        item = by_id.get(equation_id, {"id": equation_id, "equation": _text(node)})
        math_nodes = [math for math in node.find_all("math") if isinstance(math, Tag)]
        tex_values = []
        for math in math_nodes:
            annotation = math.find("annotation", attrs={"encoding": "application/x-tex"})
            tex = _text(annotation) if isinstance(annotation, Tag) else str(math.get("alttext") or "").strip()
            if tex:
                tex_values.append(tex)
        numbers = [
            _text(tag).strip("() ")
            for tag in node.select(".ltx_tag_equation, .ltx_eqn_eqno, .ltx_tag")
            if _text(tag).strip("() ")
        ]
        labels = [str(tag.get("id") or "") for tag in node.select(".ltx_eqn_table, .ltx_eqn_row") if tag.get("id")]
        enriched = dict(item)
        enriched.update(
            {
                "id": equation_id,
                "order": order,
                "tex": tex_values,
                "mathml": [str(math) for math in math_nodes],
                "printed_equation_numbers": _dedupe(numbers),
                "printed_equation_number": numbers[0] if numbers else "",
                "labels": _dedupe(labels),
                "html": str(node),
                "layout": _equation_layout(node, equation_id=equation_id),
            }
        )
        if isinstance(group, Tag):
            enriched.update(
                {
                    "group_id": str(group.get("id") or ""),
                    "group_row": group_row_index + 1,
                    "group_row_count": len(group_rows),
                }
            )
        result.append(enriched)
    if not result:
        for order, item in enumerate(compatibility, start=1):
            enriched = dict(item)
            enriched.setdefault("order", order)
            enriched.setdefault("tex", [])
            enriched.setdefault("mathml", [])
            enriched.setdefault("printed_equation_numbers", [])
            enriched.setdefault("printed_equation_number", "")
            enriched.setdefault("labels", [])
            enriched.setdefault("html", "")
            result.append(enriched)
    return result


def _equation_layout(node: Tag, *, equation_id: str) -> dict[str, Any]:
    """Return a renderer-neutral display-equation row/cell contract."""

    rows = [node] if node.name == "tr" else [row for row in node.find_all("tr") if isinstance(row, Tag)]
    if not rows:
        rows = [node]
    layout_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows, start=1):
        cell_nodes = (
            [cell for cell in row.find_all(("td", "th"), recursive=False) if isinstance(cell, Tag)]
            if row.name == "tr"
            else [row]
        )
        cells: list[dict[str, Any]] = []
        number = ""
        for cell_index, cell in enumerate(cell_nodes, start=1):
            classes = set(cell.get("class") or [])
            if "ltx_eqn_eqno" in classes:
                number = _text(cell).strip("() ")
                continue
            math_nodes = [math for math in cell.find_all("math") if isinstance(math, Tag)]
            tex: list[str] = []
            for math in math_nodes:
                annotation = math.find("annotation", attrs={"encoding": "application/x-tex"})
                value = _text(annotation) if isinstance(annotation, Tag) else str(math.get("alttext") or "").strip()
                if value:
                    tex.append(value)
            alignment = str(cell.get("align") or "").strip().casefold()
            if not alignment:
                alignment = next(
                    (value.removeprefix("ltx_align_") for value in classes if value.startswith("ltx_align_")),
                    "center",
                )
            cells.append(
                {
                    "cell": cell_index,
                    "alignment": alignment,
                    "tex": tex,
                    "mathml": [str(math) for math in math_nodes],
                    "html": str(cell),
                }
            )
        layout_rows.append(
            {
                "row": row_index,
                "cells": cells,
                "row_break": row_index < len(rows),
                "number": number,
                "label": str(row.get("id") or ""),
            }
        )
    return {
        "group_id": str((node.find_parent("table", class_="ltx_equationgroup") or {}).get("id") or ""),
        "equation_id": equation_id,
        "rows": layout_rows,
    }


def _figures(
    root: Tag | BeautifulSoup,
    *,
    source_url: str,
    asset_by_url: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for order, node in enumerate(_figure_nodes(root), start=1):
        asset_ids = []
        for tag_name, attribute in _ASSET_ATTRIBUTES:
            for asset_node in node.find_all(tag_name):
                raw_url = str(asset_node.get(attribute) or "").strip()
                record = asset_by_url.get(urljoin(source_url, raw_url))
                if record and record.get("asset_id"):
                    asset_ids.append(str(record["asset_id"]))
        result.append(
            {
                "id": str(node.get("id") or f"figure-{order:06d}"),
                "order": order,
                "tag": _text(node.select_one(".ltx_tag_figure, .ltx_tag")),
                "caption": _text(node.select_one("figcaption, .ltx_caption")),
                "asset_ids": _dedupe(asset_ids),
                "html": str(node),
            }
        )
    return result


def _tables(root: Tag | BeautifulSoup) -> list[dict[str, Any]]:
    result = []
    for order, node in enumerate(root.select(".ltx_table"), start=1):
        if not isinstance(node, Tag):
            continue
        table = node if node.name == "table" else node.find("table")
        rows = []
        occupied: dict[tuple[int, int], dict[str, Any]] = {}
        if isinstance(table, Tag):
            for row_index, row in enumerate(table.find_all("tr")):
                cells = []
                column = 0
                for cell_index, cell in enumerate(row.find_all(("th", "td"), recursive=False)):
                    while (row_index, column) in occupied:
                        column += 1
                    rowspan = _positive_int(cell.get("rowspan"))
                    colspan = _positive_int(cell.get("colspan"))
                    item = {
                        "text": _text(cell),
                        "html": str(cell),
                        "rowspan": rowspan,
                        "colspan": colspan,
                        "row": row_index,
                        "column": column,
                        "cell_index": cell_index,
                    }
                    cells.append(item)
                    for row_offset in range(rowspan):
                        for column_offset in range(colspan):
                            occupied[(row_index + row_offset, column + column_offset)] = item
                    column += colspan
                rows.append(cells)
        max_row = max((row for row, _ in occupied), default=-1)
        max_column = max((column for _, column in occupied), default=-1)
        grid = [
            [
                {
                    "text": occupied[(row, column)]["text"],
                    "source_row": occupied[(row, column)]["row"],
                    "source_column": occupied[(row, column)]["column"],
                }
                if (row, column) in occupied
                else None
                for column in range(max_column + 1)
            ]
            for row in range(max_row + 1)
        ]
        result.append(
            {
                "id": str(node.get("id") or f"table-{order:06d}"),
                "order": order,
                "tag": _text(node.select_one(".ltx_tag_table, .ltx_tag")),
                "caption": _text(node.select_one("figcaption, .ltx_caption")),
                "rows": rows,
                "grid": grid,
                "html": str(node),
            }
        )
    return result


def _bibliography(root: Tag | BeautifulSoup) -> list[dict[str, Any]]:
    result = []
    for order, node in enumerate(root.select(".ltx_bibitem"), start=1):
        if not isinstance(node, Tag):
            continue
        label = node.select_one(".ltx_tag_bibitem, .ltx_tag")
        text = _text(node)
        urls = [str(link.get("href") or "") for link in node.find_all("a") if link.get("href")]
        doi_match = re.search(r"\b10\.\d{4,9}/[^\s<>]+", text, flags=re.IGNORECASE)
        arxiv_match = re.search(r"(?:arXiv:\s*)?([a-z-]+/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?", text, flags=re.IGNORECASE)
        result.append(
            {
                "id": str(node.get("id") or f"bib-{order:06d}"),
                "order": order,
                "label": _text(label),
                "text": text,
                "doi": doi_match.group(0).rstrip(".,;)") if doi_match else "",
                "arxiv_id": arxiv_match.group(1) if arxiv_match else "",
                "links": urls,
                "html": str(node),
            }
        )
    return result


def _footnotes(root: Tag | BeautifulSoup) -> list[dict[str, Any]]:
    result = []
    for order, node in enumerate(root.select(".ltx_note, .ltx_role_footnote"), start=1):
        if not isinstance(node, Tag):
            continue
        result.append(
            {
                "id": str(node.get("id") or f"footnote-{order:06d}"),
                "order": order,
                "text": _text(node),
                "html": str(node),
            }
        )
    return result


def _links(root: Tag | BeautifulSoup, *, source_url: str) -> list[dict[str, str]]:
    result = []
    for order, node in enumerate(root.find_all("a"), start=1):
        href = str(node.get("href") or "").strip()
        if not href:
            continue
        result.append(
            {
                "id": str(node.get("id") or f"link-{order:06d}"),
                "href": href,
                "resolved_url": urljoin(source_url, href),
                "target_id": href[1:] if href.startswith("#") else "",
                "kind": "citation" if "bib" in href.lower() else ("internal" if href.startswith("#") else "external"),
                "text": _text(node),
            }
        )
    return result


def _integrity(
    root: Tag | BeautifulSoup,
    *,
    blocks: list[dict[str, Any]],
    assets: list[dict[str, Any]],
    equations: list[dict[str, Any]],
    figures: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    footnotes: list[dict[str, Any]],
    bibliography: list[dict[str, Any]],
    links: list[dict[str, str]],
) -> dict[str, Any]:
    diagnostics: list[dict[str, Any]] = []
    for class_name in ("ltx_ERROR", "ltx_missing", "ltx_nounicode"):
        count = len(root.select(f".{class_name}"))
        if count:
            diagnostics.append({"code": class_name, "severity": "error", "count": count})
    missing_assets = [item for item in assets if item.get("status") != "cached"]
    if missing_assets:
        diagnostics.append({"code": "asset_missing", "severity": "error", "count": len(missing_assets)})
    cached_asset_ids = {
        str(item.get("asset_id") or "") for item in assets if item.get("status") == "cached" and item.get("asset_id")
    }
    unrenderable_figures = [
        str(item.get("id") or "")
        for item in figures
        if not item.get("asset_ids") or any(asset_id not in cached_asset_ids for asset_id in item.get("asset_ids") or [])
    ]
    if unrenderable_figures:
        diagnostics.append(
            {
                "code": "figure_asset_missing",
                "severity": "error",
                "ids": unrenderable_figures,
            }
        )
    unrenderable_tables = [str(item.get("id") or "") for item in tables if not item.get("grid")]
    if unrenderable_tables:
        diagnostics.append(
            {
                "code": "table_grid_missing",
                "severity": "error",
                "ids": unrenderable_tables,
            }
        )
    missing_tex = [item for item in equations if not item.get("tex") and not item.get("mathml")]
    if missing_tex:
        diagnostics.append({"code": "equation_source_missing", "severity": "error", "count": len(missing_tex)})
    group_gaps = []
    for group in root.select("table.ltx_equationgroup"):
        if not isinstance(group, Tag):
            continue
        group_id = str(group.get("id") or "")
        rows = _equation_group_rows(group)
        parsed_rows = [item for item in equations if item.get("group_id") == group_id]
        parsed_indexes = sorted(int(item.get("group_row") or 0) for item in parsed_rows)
        parsed_ids = [str(item.get("id") or "") for item in parsed_rows]
        if (
            len(parsed_rows) != len(rows)
            or parsed_indexes != list(range(1, len(rows) + 1))
            or len(set(parsed_ids)) != len(parsed_ids)
        ):
            group_gaps.append(
                {
                    "group_id": group_id,
                    "dom_rows": len(rows),
                    "parsed_rows": len(parsed_rows),
                    "parsed_indexes": parsed_indexes,
                    "parsed_ids": parsed_ids,
                }
            )
    if group_gaps:
        diagnostics.append({"code": "equation_group_gap", "severity": "error", "groups": group_gaps})
    ids = [str(node.get("id")) for node in root.find_all(id=True)]
    duplicate_ids = sorted(item for item, count in Counter(ids).items() if count > 1)
    if duplicate_ids:
        diagnostics.append({"code": "duplicate_dom_id", "severity": "error", "ids": duplicate_ids})
    known_ids = set(ids)
    broken_links = sorted(
        {item["href"] for item in links if item["href"].startswith("#") and item["href"][1:] not in known_ids}
    )
    if broken_links:
        diagnostics.append({"code": "broken_internal_link", "severity": "error", "hrefs": broken_links})
    block_source_ids = {str(item.get("source_id") or item.get("block_id") or "") for item in blocks}
    expected_entity_ids = _top_level_entity_ids(root)
    missing_entity_blocks = sorted(
        entity_id
        for entity_ids in expected_entity_ids.values()
        for entity_id in entity_ids
        if entity_id and entity_id not in block_source_ids
    )
    indexed_counts = {
        "equations": len(equations),
        "figures": len(figures),
        "tables": len(tables),
        "footnotes": len(footnotes),
        "bibliography": len(bibliography),
    }
    dom_counts = _top_level_block_counts(root)
    block_counts = Counter(str(item.get("kind") or "") for item in blocks)
    coverage_mismatch = {
        key: {"dom": dom_count, "blocks": block_counts.get(kind, 0)}
        for key, dom_count, kind in (
            ("headings", dom_counts["headings"], "heading"),
            ("lists", dom_counts["lists"], "list"),
            ("equations", dom_counts["equations"], "equation"),
            ("figures", dom_counts["figures"], "figure"),
            ("tables", dom_counts["tables"], "table"),
            ("footnotes", dom_counts["footnotes"], "footnote"),
            ("bibliography", dom_counts["bibliography"], "bibliography"),
        )
        if block_counts.get(kind, 0) != dom_count
    }
    if missing_entity_blocks or coverage_mismatch:
        diagnostics.append(
            {
                "code": "dom_block_coverage_gap",
                "severity": "error",
                "missing_entity_blocks": missing_entity_blocks,
                "counts": coverage_mismatch,
            }
        )
    status = "complete" if not any(item.get("severity") == "error" for item in diagnostics) else "partial"
    return {
        "status": status,
        "diagnostics": diagnostics,
        "block_count": len(blocks),
        "asset_count": len(assets),
        "equation_count": len(equations),
        "coverage": {"top_level_dom": dom_counts, "indexed": indexed_counts, "blocks": dict(block_counts)},
        "renderable": status == "complete",
    }


def _top_level_block_counts(root: Tag | BeautifulSoup) -> dict[str, int]:
    entity_ids = _top_level_entity_ids(root)
    headings = [
        node
        for node in root.find_all(tuple(_HEADING_NAMES))
        if isinstance(node, Tag) and not _has_special_ancestor(node)
    ]
    lists = [
        node
        for node in root.find_all(("ul", "ol"))
        if isinstance(node, Tag)
        and node.select_one(".ltx_bibitem") is None
        and node.find_parent(("ul", "ol")) is None
        and not _has_special_ancestor(node)
    ]
    return {
        "headings": len(headings),
        "lists": len(lists),
        "equations": len(entity_ids["equations"]),
        "figures": len(entity_ids["figures"]),
        "tables": len(entity_ids["tables"]),
        "footnotes": len(entity_ids["footnotes"]),
        "bibliography": len(entity_ids["bibliography"]),
    }


def _top_level_entity_ids(root: Tag | BeautifulSoup) -> dict[str, list[str]]:
    equations = []
    equation_nodes = _equation_nodes(root)
    for order, node in enumerate(equation_nodes, start=1):
        if _has_special_ancestor(node):
            continue
        group = node.find_parent("table", class_="ltx_equationgroup")
        rows = _equation_group_rows(group) if isinstance(group, Tag) else []
        row_index = rows.index(node) if node in rows else -1
        equations.append(_equation_node_id(node, group=group, row_index=row_index) or f"equation-{order:06d}")

    def ids_for(nodes: list[Tag], prefix: str) -> list[str]:
        return [
            str(node.get("id") or f"{prefix}-{order:06d}")
            for order, node in enumerate(nodes, start=1)
            if not _has_special_ancestor(node)
        ]

    figures = _figure_nodes(root)
    tables = [node for node in root.select(".ltx_table") if isinstance(node, Tag)]
    footnotes = [node for node in root.select(".ltx_note, .ltx_role_footnote") if isinstance(node, Tag)]
    bibliography = [node for node in root.select(".ltx_bibitem") if isinstance(node, Tag)]
    return {
        "equations": equations,
        "figures": ids_for(figures, "figure"),
        "tables": ids_for(tables, "table"),
        "footnotes": ids_for(footnotes, "footnote"),
        "bibliography": ids_for(bibliography, "bib"),
    }


def _has_special_ancestor(node: Tag) -> bool:
    for parent in node.parents:
        if not isinstance(parent, Tag):
            continue
        classes = set(parent.get("class") or [])
        is_figure_block = (
            "ltx_figure" in classes
            and "ltx_table" not in classes
            and parent.select_one(".ltx_table") is None
        )
        if is_figure_block or classes.intersection(
            {"ltx_table", "ltx_bibitem", "ltx_note", "ltx_role_footnote"}
        ):
            return True
    return False


def _asset_metadata(
    root: Tag | BeautifulSoup,
    *,
    source_url: str,
    assets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records = [dict(item) for item in assets]
    by_url = {str(item.get("source_url") or ""): item for item in records}
    for tag_name, attribute in _ASSET_ATTRIBUTES:
        for node in root.find_all(tag_name):
            raw_url = str(node.get(attribute) or "").strip()
            record = by_url.get(urljoin(source_url, raw_url))
            if not record:
                continue
            if node.get("width") is not None:
                record["declared_width"] = str(node.get("width"))
            if node.get("height") is not None:
                record["declared_height"] = str(node.get("height"))
            if node.get("alt") is not None:
                record["alt"] = str(node.get("alt"))
    return records


def _inside_site_chrome(node: Tag) -> bool:
    classes = set(node.get("class") or [])
    if classes.intersection({"ltx_page_navbar", "ltx_page_footer", "ltx_page_logo"}):
        return True
    return node.name in {"script", "style", "nav"}


def _text(node: Tag | NavigableString | None) -> str:
    if node is None:
        return ""
    if isinstance(node, NavigableString):
        return " ".join(str(node).split())
    return " ".join(node.get_text(" ", strip=True).split())


def _positive_int(value: Any) -> int:
    try:
        return max(1, int(value or 1))
    except (TypeError, ValueError):
        return 1


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
