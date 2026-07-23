from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any
import unicodedata

from bs4 import BeautifulSoup, NavigableString, Tag

from .io import sha256_file, sha256_json
from .reader_text import (
    clean_reader_annotation,
    clean_reader_text,
    is_machine_summary_label,
    strip_machine_details,
)
from .source import asset_path, block_id
from .substantive import non_substantive_block_ids
from .source_credit import (
    SourceCreditError,
    ordered_source_credit_items,
    source_credit_placement,
    source_credit_visible_projection,
    validate_source_credit,
)


class LatexError(RuntimeError):
    """Raised when a document cannot be rendered without losing source structure."""


def escape_tex(value: Any) -> str:
    # U+141F CANADIAN SYLLABICS FINAL ACUTE is visually confusable with a
    # fraction slash but has no useful mathematical semantics. Model output
    # occasionally substitutes it for an ordinary slash; normalize that one
    # confusable before escaping so the PDF remains searchable as plain text.
    text = str(value or "").replace("\u141f", "/")
    preserve_greek_letters = _looks_like_greek_prose(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
        "&": r"\&",
        "#": r"\#",
        "%": r"\%",
        "_": r"\_",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    rendered: list[str] = []
    math_atoms: list[str] = []

    def flush_math() -> None:
        if math_atoms:
            rendered.append("{\\rmfamily\\(" + "".join(math_atoms) + "\\)}")
            math_atoms.clear()

    for char in text:
        if ord(char) < 32 and char not in "\n\r\t":
            continue
        atom = _unicode_math_atom(char, preserve_greek_letters=preserve_greek_letters)
        if atom is not None:
            math_atoms.append(atom)
            continue
        flush_math()
        if char == "\u200b":
            continue
        if char in _UNICODE_SUBSCRIPTS:
            rendered.append(f"\\textsubscript{{{_UNICODE_SUBSCRIPTS[char]}}}")
            continue
        if char == "ᵢ":
            rendered.append("\\textsuperscript{i}")
            continue
        if char == "′":
            rendered.append("\\textsuperscript{{\\rmfamily\\(\\prime\\)}}")
            continue
        rendered.append(replacements.get(char, char))
    flush_math()
    return "".join(rendered)


_UNICODE_SUBSCRIPTS = {
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
}

_UNICODE_MATH_SYMBOLS = {
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta",
    "ε": r"\epsilon", "ϵ": r"\epsilon", "ζ": r"\zeta", "η": r"\eta",
    "θ": r"\theta", "ϑ": r"\vartheta", "ι": r"\iota", "κ": r"\kappa",
    "λ": r"\lambda", "μ": r"\mu", "ν": r"\nu", "ξ": r"\xi",
    "ο": r"o", "π": r"\pi", "ρ": r"\rho", "ϱ": r"\varrho",
    "σ": r"\sigma", "ς": r"\varsigma", "τ": r"\tau", "υ": r"\upsilon",
    "φ": r"\phi", "ϕ": r"\varphi", "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega",
    "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda",
    "Ξ": r"\Xi", "Π": r"\Pi", "Σ": r"\Sigma", "Υ": r"\Upsilon",
    "Φ": r"\Phi", "Ψ": r"\Psi", "Ω": r"\Omega",
    "∼": r"\sim", "≪": r"\ll", "≫": r"\gg",
    "≲": r"\lesssim", "∝": r"\propto", "≡": r"\equiv", "≈": r"\approx", "∑": r"\sum",
    "∫": r"\int", "ℓ": r"\ell",
    "⟨": r"\langle", "⟩": r"\rangle", "∙": r"\mathbin{\cdot}", "Ḣ": r"\dot{H}",
    "ℒ": r"\mathcal{L}", "ℋ": r"\mathcal{H}", "ℏ": r"\hbar",
    # Common mathematical symbols also occur in otherwise-natural-language
    # annotations. Render them as math atoms rather than asking the selected
    # sans/CJK text font to contain a partial mathematical alphabet.
    "−": "-", "√": r"\surd", "∂": r"\partial", "→": r"\to",
    "⊥": r"\perp", "∥": r"\parallel",
}

_UNICODE_MATH_MODIFIERS = {
    "⁰": "{}^{0}", "¹": "{}^{1}", "²": "{}^{2}", "³": "{}^{3}", "⁴": "{}^{4}",
    "⁵": "{}^{5}", "⁶": "{}^{6}", "⁷": "{}^{7}", "⁸": "{}^{8}", "⁹": "{}^{9}",
    "ⁿ": "{}^{n}",
    "⁺": "{}^{+}", "⁻": "{}^{-}", "⁼": "{}^{=}", "⁽": "{}^{(}", "⁾": "{}^{)}",
    "†": r"\dagger", "‡": r"\ddagger",
}

_GREEK_NAME_TO_TEX = {
    "ALPHA": r"\alpha", "BETA": r"\beta", "DELTA": r"\delta", "EPSILON": r"\epsilon",
    "ETA": r"\eta", "NU": r"\nu", "PHI": r"\phi", "PI": r"\pi", "SIGMA": r"\sigma",
    "TAU": r"\tau", "THETA": r"\theta", "ZETA": r"\zeta",
}


def _unicode_math_atom(
    char: str, *, preserve_greek_letters: bool = False
) -> str | None:
    modifier = _UNICODE_MATH_MODIFIERS.get(char)
    if modifier is not None:
        return modifier
    direct = _UNICODE_MATH_SYMBOLS.get(char)
    if direct is not None:
        if preserve_greek_letters and _is_greek_letter(char):
            return None
        return direct
    name = unicodedata.name(char, "")
    if not name.startswith("MATHEMATICAL "):
        return None
    descriptor = name.removeprefix("MATHEMATICAL ")
    greek = next((value for key, value in _GREEK_NAME_TO_TEX.items() if descriptor.endswith(f" {key}")), None)
    if greek is not None:
        base = greek
    else:
        match = re.search(r"(?:CAPITAL|SMALL) ([A-Z])$", descriptor)
        if match is None:
            return None
        base = match.group(1) if " CAPITAL " in f" {descriptor} " else match.group(1).lower()
    if "SCRIPT" in descriptor:
        if not base.startswith("\\"):
            base = f"\\mathcal{{{base}}}"
    elif "FRAKTUR" in descriptor:
        base = f"\\mathfrak{{{base}}}"
    elif "DOUBLE-STRUCK" in descriptor:
        base = f"\\mathbb{{{base}}}"
    elif "SANS-SERIF" in descriptor:
        base = f"\\mathsf{{{base}}}"
    elif "MONOSPACE" in descriptor:
        base = f"\\mathtt{{{base}}}"
    elif "BOLD" in descriptor and not base.startswith("\\"):
        base = f"\\mathbf{{{base}}}"
    if "BOLD" in descriptor and base.startswith("\\") and not base.startswith("\\mathbf"):
        base = f"\\boldsymbol{{{base}}}"
    return base


def _is_greek_letter(char: str) -> bool:
    return unicodedata.category(char).startswith("L") and "GREEK" in unicodedata.name(char, "")


def _looks_like_greek_prose(text: str) -> bool:
    run = 0
    for char in text:
        if _is_greek_letter(char):
            run += 1
            if run >= 2 or char not in _UNICODE_MATH_SYMBOLS:
                return True
        elif unicodedata.category(char).startswith("M") and run:
            continue
        else:
            run = 0
    return False


def _title_translation_index(value: Any) -> dict[str, str]:
    if not value:
        return {}
    records: Any = value
    if isinstance(value, Mapping):
        nested = value.get("titles") or value.get("translations")
        if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
            records = nested
        else:
            output: dict[str, str] = {}
            for key, raw in value.items():
                if key in {"schema_version", "source_language", "target_language", "source_sha256"}:
                    continue
                text = (
                    str(raw.get("text") or raw.get("translated_title") or raw.get("translation") or "").strip()
                    if isinstance(raw, Mapping)
                    else str(raw or "").strip()
                )
                if text:
                    output[str(key)] = text
            return output
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        return {}
    output: dict[str, str] = {}
    for raw in records:
        if not isinstance(raw, Mapping):
            continue
        text = str(
            raw.get("text")
            or raw.get("translated_title")
            or raw.get("translation")
            or ""
        ).strip()
        if not text:
            continue
        title_id = str(raw.get("title_id") or "").strip()
        block_value = str(raw.get("block_id") or "").strip()
        chapter_id = str(raw.get("chapter_id") or "").strip()
        role = str(raw.get("role") or "").casefold()
        if title_id:
            output[title_id] = text
        if block_value:
            output.setdefault(f"block:{block_value}", text)
            output.setdefault(block_value, text)
        if chapter_id and (
            role in {"chapter", "chapter_title"}
            or title_id == f"chapter:{chapter_id}"
        ):
            output.setdefault(f"chapter:{chapter_id}", text)
        if role in {"document", "document_title", "paper_title", "title"}:
            output.setdefault("document:title", text)
    return output


def _block_title(block: Mapping[str, Any]) -> str:
    heading = block.get("heading") if isinstance(block.get("heading"), Mapping) else {}
    return str(
        block.get("title")
        or heading.get("title")
        or heading.get("text")
        or (block.get("heading") if isinstance(block.get("heading"), str) else "")
        or block.get("text")
        or ""
    ).strip()


def _translated_block_title(
    block: Mapping[str, Any], translations: Mapping[str, str]
) -> str:
    identifier = block_id(dict(block))
    return str(
        translations.get(f"block:{identifier}")
        or translations.get(identifier)
        or ""
    ).strip()


def _translated_document_title(
    document: Mapping[str, Any], translations: Mapping[str, str]
) -> str:
    direct = str(translations.get("document:title") or "").strip()
    if direct:
        return direct
    front_roles = _front_matter_block_roles(
        list(document.get("blocks") or []), dict(document.get("front_matter") or {})
    )
    for identifier, role in front_roles.items():
        if role != "title":
            continue
        value = translations.get(f"block:{identifier}") or translations.get(identifier)
        if value:
            return str(value).strip()
    return ""


def _normalize_language_tag(value: str) -> str:
    parts = [part for part in str(value or "und").strip().replace("_", "-").split("-") if part]
    if not parts:
        return "und"
    normalized = [parts[0].casefold()]
    for part in parts[1:]:
        normalized.append(
            part.title() if len(part) == 4 and part.isalpha()
            else part.upper() if len(part) in {2, 3} and part.isalpha()
            else part
        )
    return "-".join(normalized)


def _language_base(value: str) -> str:
    return _normalize_language_tag(value).split("-", 1)[0]


def _language_direction(value: str) -> str:
    return "rtl" if _language_base(value) in {
        "ar", "dv", "fa", "he", "ku", "ps", "sd", "ug", "ur", "yi"
    } else "ltr"


def _cjk_variant(language: str) -> str | None:
    normalized = _normalize_language_tag(language)
    base = _language_base(normalized)
    if base == "ja":
        return "JP"
    if base == "ko":
        return "KR"
    if base != "zh":
        return None
    lowered = normalized.casefold()
    return "TC" if any(value in lowered for value in ("-hant", "-tw", "-hk", "-mo")) else "SC"


def _font_fallback(
    command: str,
    candidates: Sequence[str],
    *,
    required: bool,
    error_label: str = "No supported Unicode font found",
) -> str:
    fallback = (
        f"\\PackageError{{arc-companion}}{{{error_label}}}"
        "{Install Noto or Source Han fonts for the source and target languages.}"
        if required
        else "\\PackageWarning{arc-companion}{Preferred Unicode font unavailable}"
    )
    rendered = fallback
    for candidate in reversed(list(dict.fromkeys(candidates))):
        rendered = (
            f"\\IfFontExistsTF{{{candidate}}}{{\\{command}{{{candidate}}}}}"
            f"{{{rendered}}}"
        )
    return rendered


def _font_setup(source_language: str, target_language: str) -> str:
    setup = [
        _font_fallback("setsansfont", ["Noto Sans", "DejaVu Sans", "FreeSans"], required=False),
        _font_fallback("setmainfont", ["Noto Serif", "DejaVu Serif", "FreeSerif"], required=False),
    ]
    variants = [
        value for value in (
            _cjk_variant(target_language), _cjk_variant(source_language),
            "SC", "TC", "JP", "KR",
        ) if value
    ]
    serif: list[str] = []
    sans: list[str] = []
    for variant in variants:
        serif.extend([f"Noto Serif CJK {variant}", f"Source Han Serif {variant}"])
        sans.extend([f"Noto Sans CJK {variant}", f"Source Han Sans {variant}"])
    if any(_cjk_variant(value) for value in (source_language, target_language)):
        serif.extend(["Source Han Serif CN", "FandolSong-Regular"])
        sans.extend(["Source Han Sans CN", "FandolHei-Regular"])
        setup.append(_font_fallback(
            "setCJKmainfont", serif, required=True,
            error_label="No supported CJK serif font found",
        ))
        setup.append(_font_fallback(
            "setCJKsansfont", sans, required=True,
            error_label="No supported CJK sans font found",
        ))
    return "\n".join(setup)


def _render_warnings(source_language: str, target_language: str) -> list[dict[str, Any]]:
    rtl = [
        value for value in (source_language, target_language)
        if _language_direction(value) == "rtl"
    ]
    return ([{
        "code": "rtl_pdf_layout_not_guaranteed",
        "languages": rtl,
        "message": "PDF bidirectional layout is best-effort; use the HTML reader for authoritative RTL direction.",
    }] if rtl else [])


def render_companion_tex(
    document: dict[str, Any],
    segments: list[dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    *,
    output_dir: Path,
    language: str,
    metadata: dict[str, Any] | None = None,
    translations: dict[str, dict[str, Any]] | None = None,
    glossary: dict[str, Any] | list[dict[str, Any]] | None = None,
    evidence_by_segment: dict[str, list[dict[str, Any]]] | None = None,
    augmentation_scope: str = "all",
    chapters: list[dict[str, Any]] | None = None,
    chapter_guides: dict[str, dict[str, Any]] | None = None,
    source_language: str | None = None,
    title_translations: dict[str, Any] | list[dict[str, Any]] | None = None,
    source_credit: Mapping[str, Any] | None = None,
    translation_reference: Mapping[str, Any] | None = None,
    project_root: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    if augmentation_scope not in {"all", "substantive"}:
        raise ValueError(f"unsupported companion augmentation scope: {augmentation_scope}")
    blocks = document.get("blocks") or []
    figures = _index_entities(document.get("figures") or [])
    tables = _index_entities(document.get("tables") or [])
    equations = _index_entities(document.get("equations") or [])
    assets = _index_entities(document.get("assets") or [])
    translation_mode = translations is not None
    translations = translations or {}
    evidence_by_segment = evidence_by_segment or {}
    source_language = _normalize_language_tag(source_language or "und")
    language = _normalize_language_tag(language or "und")
    title_translation_index = _title_translation_index(title_translations)
    metadata = metadata or {}
    try:
        if source_credit is None:
            raise SourceCreditError("canonical source credit is required")
        canonical_source_credit = validate_source_credit(source_credit)
    except (SourceCreditError, TypeError) as exc:
        raise LatexError("source credit is invalid") from exc
    compact_translation_reference = None
    if translation_reference is not None:
        if project_root is None:
            raise LatexError(
                "translation-reference validation requires the project root"
            )
        try:
            from .translation_reference import (
                TranslationReferenceError,
                validate_translation_reference_provenance,
            )
            compact_translation_reference = (
                validate_translation_reference_provenance(
                    translation_reference,
                    project_root=project_root.resolve(),
                    expected_chapter_ids=[
                        str(item.get("chapter_id") or "")
                        for item in chapters or []
                    ],
                )
            )
        except (TranslationReferenceError, TypeError, ValueError) as exc:
            raise LatexError("translation reference is invalid") from exc

    copied_assets: list[dict[str, Any]] = []
    rendered_links: list[dict[str, str]] = []
    block_records: list[dict[str, str]] = []
    body: list[str] = []
    bibliography_blocks = {block_id(block) for block in blocks if _kind(block) in {"bibliography", "bibliography_item", "reference"}}
    bibliography_section_ids = {
        str(block.get("section_id") or "").strip()
        for block in blocks
        if block_id(block) in bibliography_blocks
        and str(block.get("section_id") or "").strip()
    }
    bibliography_heading_ids = {
        block_id(block)
        for block in blocks
        if _kind(block) in {
            "heading", "section", "subsection", "subsubsection", "chapter", "part",
        }
        and (
            str(block.get("source_role") or "").casefold() in {"references", "bibliography"}
            or (
                str(block.get("section_id") or "").strip()
                and str(block.get("section_id") or "").strip() in bibliography_section_ids
            )
        )
    }
    front_roles = _front_matter_block_roles(blocks, document.get("front_matter") or {})
    front_matter_block_ids = sorted({
        str(value)
        for key, values in (
            (document.get("front_matter") or {}).get("block_ids") or {}
        ).items()
        if key in {"title", "authors", "affiliations"}
        if isinstance(values, list)
        for value in values
        if str(value)
    })
    (
        source_credit_header_tex,
        source_credit_by_block,
        source_credit_manifest,
    ) = _render_source_credit(
        canonical_source_credit,
        front_matter_block_ids=front_matter_block_ids,
    )
    renderable_ids = {
        block_id(block)
        for block in blocks
        if block_id(block) not in bibliography_blocks
        and front_roles.get(block_id(block)) not in {
            "title", "author", "affiliation", "profile",
        }
    }
    source_credit_records = {
        str(item["id"]): item
        for key in ("authors", "affiliations", "profiles")
        for item in canonical_source_credit[key]
    }
    block_text_by_id = {
        block_id(block): str(block.get("text") or "").strip()
        for block in blocks
    }
    ambiguous_source_credit_blocks = [
        str(item["block_id"])
        for item in source_credit_manifest["ordered_items"]
        if item["slot"] == "source_block"
        and item["block_id"]
        and block_text_by_id.get(str(item["block_id"])) != str(
            source_credit_records[str(item["id"])].get("source_name")
            or source_credit_records[str(item["id"])].get("text")
            or ""
        ).strip()
    ]
    if ambiguous_source_credit_blocks:
        raise LatexError(
            "source-credit block anchor does not identify an equivalent "
            "standalone source block: "
            + ", ".join(ambiguous_source_credit_blocks)
        )
    source_credit_only_block_ids = {
        str(item["block_id"])
        for item in source_credit_manifest["ordered_items"]
        if item["slot"] == "source_block"
        and item["block_id"]
        and any(
            block_id(block) == str(item["block_id"])
            and str(block.get("text") or "").strip()
            == str(
                source_credit_records[str(item["id"])].get("source_name")
                or source_credit_records[str(item["id"])].get("text")
                or ""
            ).strip()
            for block in blocks
        )
    }
    renderable_ids.difference_update(source_credit_only_block_ids)
    excluded_augmentation_ids = (
        non_substantive_block_ids(document)
        if augmentation_scope == "substantive"
        else set()
    )
    segment_by_block: dict[str, str] = {}
    renderable_by_segment: dict[str, list[str]] = {}
    segment_records = {str(item["segment_id"]): item for item in segments}
    for segment in segments:
        segment_id = str(segment["segment_id"])
        member_ids = [str(value) for value in segment.get("block_ids") or []]
        if not member_ids:
            member_ids = _inclusive_block_ids(
                blocks,
                start=str(segment.get("start_block_id") or ""),
                end=str(segment.get("end_block_id") or ""),
            )
        visible_ids = [value for value in member_ids if value in renderable_ids]
        projected_ids = [
            str(value)
            for value in (
                segment.get("augmentation_block_ids")
                if "augmentation_block_ids" in segment
                else visible_ids
            ) or []
        ]
        augmentation_ids = [
            value for value in projected_ids
            if value in visible_ids and value not in excluded_augmentation_ids
        ]
        renderable_by_segment[segment_id] = augmentation_ids
        for value in augmentation_ids:
            previous = segment_by_block.setdefault(value, segment_id)
            if previous != segment_id:
                raise LatexError(f"source block {value} belongs to more than one segment")
    first_by_segment = {key: values[0] for key, values in renderable_by_segment.items() if values}
    last_by_segment = {key: values[-1] for key, values in renderable_by_segment.items() if values}
    semantic_segment_ids = [str(item["segment_id"]) for item in segments if renderable_by_segment.get(str(item["segment_id"]))]
    preservation_only_segment_ids = [
        str(item["segment_id"]) for item in segments if not renderable_by_segment.get(str(item["segment_id"]))
    ]
    rendered_translation_ids: list[str] = []
    rendered_annotation_ids: list[str] = []
    translation_audits: list[dict[str, Any]] = []
    source_box_open = False
    chapter_guides = chapter_guides or {}
    chapter_after_block = {
        str(item.get("block_ids", [""])[0]): str(item.get("chapter_id") or "")
        for item in chapters or [] if item.get("block_ids")
    }
    rendered_chapter_guides: list[str] = []
    for block in blocks:
        bid = block_id(block)
        source_hash = sha256_json(block)
        block_records.append({"block_id": bid, "sha256": source_hash})
        body.append(_source_block_marker(bid, source_hash))
        if bid in source_credit_by_block:
            body.append(source_credit_by_block[bid])
        segment_id = segment_by_block.get(bid)
        chapter_id = chapter_after_block.get(bid)
        if chapter_id:
            # Keep the source chapter heading, guide heading, and the opening
            # guide lines together while leaving long guides breakable across
            # subsequent pages.
            body.append("\\Needspace{10\\baselineskip}\n")
        if segment_id and first_by_segment.get(segment_id) == bid:
            body.append(_render_unit_heading())
        if bid in renderable_ids:
            is_table = _kind(block) == "table"
            if segment_id and not is_table and not source_box_open:
                body.append(_box_begin("arcsource"))
                source_box_open = True
            if is_table and source_box_open:
                body.append("\\end{arcsource}\n")
                source_box_open = False
            body.append(_render_block(
                block,
                equations=equations,
                figures=figures,
                tables=tables,
                assets=assets,
                output_dir=output_dir,
                copied_assets=copied_assets,
                rendered_links=rendered_links,
                translated_title=_translated_block_title(
                    block, title_translation_index
                ) or str(
                    title_translation_index.get(f"chapter:{chapter_id}") or ""
                ),
            ))
            if chapter_id:
                chapter_guide_tex = _render_chapter_guide(
                    chapter_guides.get(chapter_id) or {}, language=language
                )
                if chapter_guide_tex:
                    body.append(chapter_guide_tex)
                    rendered_chapter_guides.append(chapter_id)
            if segment_id and last_by_segment.get(segment_id) == bid:
                if source_box_open:
                    body.append("\\end{arcsource}\n")
                    source_box_open = False
                translation = translations.get(segment_id)
                if translation is not None:
                    translation_tex, translation_audit = _render_translation(
                        segment_id,
                        {
                            **segment_records[segment_id],
                            "block_ids": renderable_by_segment[segment_id],
                        },
                        translation,
                        document=document,
                        equations=equations,
                        language=language,
                    )
                    body.append(translation_tex)
                    rendered_translation_ids.append(segment_id)
                    translation_audits.append(translation_audit)
                annotation = annotations.get(segment_id)
                if not annotation:
                    raise LatexError(f"missing annotation for segment {segment_id}")
                body.append(_render_annotation(
                    segment_id,
                    annotation,
                    language=language,
                    evidence_records=evidence_by_segment.get(segment_id) or [],
                ))
                rendered_annotation_ids.append(segment_id)
        elif segment_id and last_by_segment.get(segment_id) == bid:
            # Defensive only: renderable segment endpoints are handled above.
            annotation = annotations.get(segment_id)
            if not annotation:
                raise LatexError(f"missing annotation for segment {segment_id}")
            body.append(_render_annotation(
                segment_id,
                annotation,
                language=language,
                evidence_records=evidence_by_segment.get(segment_id) or [],
            ))
            rendered_annotation_ids.append(segment_id)

    bibliography = document.get("bibliography") or []
    if bibliography:
        body.append(_render_bibliography(
            bibliography,
            rendered_links=rendered_links,
            include_heading=not bibliography_heading_ids,
        ))
    elif bibliography_blocks:
        body.extend(
            _render_plain_reference(block, rendered_links=rendered_links)
            for block in blocks
            if block_id(block) in bibliography_blocks
        )

    front = document.get("front_matter") or {}
    title = front.get("title") or metadata.get("title") or "Paper Companion"
    translated_title = _translated_document_title(
        document, title_translation_index
    )
    title_source_runs = [
        run
        for block in blocks
        if front_roles.get(block_id(block)) == "title"
        for run in block.get("inline_runs") or []
        if isinstance(run, dict)
    ]
    translated_title_tex = (
        _render_translated_inline_runs(
            translated_title, {"inline_runs": title_source_runs}
        )
        if translated_title and title_source_runs
        else escape_tex(translated_title)
    )
    front_body = _render_front_matter(
        front,
        represented_roles={*set(front_roles.values()), "affiliation"},
    )
    guide = "" if chapters else _render_reading_guide(
        language=language, include_translation=bool(translations)
    )
    # A glossary is a bilingual companion artifact.  Treat the absence of the
    # translation layer as authoritative even if a stale caller accidentally
    # passes glossary data from an earlier translated run.
    glossary_tex = _render_glossary(
        glossary if translation_mode else None,
        language=language,
    )
    glossary_back_matter = f"\n\\clearpage\n{glossary_tex}" if glossary_tex else ""
    annotation_sources_by_segment = {
        str(segment_id): source_manifest
        for segment_id, annotation in annotations.items()
        if (source_manifest := _annotation_source_manifest(annotation))
    }
    tex = (
        _preamble(
            title=title,
            translated_title=translated_title,
            translated_title_tex=translated_title_tex,
            authors="",
            language=language,
            source_language=source_language,
            include_translation=translation_mode,
        )
        + source_credit_header_tex
        + front_body
        + guide
        + "\n".join(body)
        + glossary_back_matter
        + "\n\\end{document}\n"
    )
    tex = _remove_disallowed_c0(tex)
    manifest = {
        "document_sha256": sha256_json(document),
        "document_hash": str(document.get("document_hash") or (document.get("integrity") or {}).get("document_hash") or ""),
        "blocks": block_records,
        "block_ids": [block_id(block) for block in blocks],
        "equation_numbers": _equation_numbers(document.get("equations") or []),
        "bibliography_labels": [str(item.get("label") or item.get("display_label") or "") for item in bibliography],
        "expected_links": _link_records(document.get("links") or []),
        "rendered_links": rendered_links,
        "tables": [_table_audit_record(item) for item in document.get("tables") or []],
        "assets": copied_assets,
        "source_language": source_language,
        "target_language": language,
        "render_warnings": _render_warnings(source_language, language),
        "source_credit": source_credit_manifest,
        **(
            {"translation_reference": compact_translation_reference}
            if compact_translation_reference is not None else {}
        ),
        "companion_layers": {
            "augmentation_scope": augmentation_scope,
            "excluded_augmentation_block_ids": [
                block_id(block)
                for block in blocks
                if block_id(block) in excluded_augmentation_ids
            ],
            "translation_mode": translation_mode,
            "semantic_segment_ids": semantic_segment_ids,
            "preservation_only_segment_ids": preservation_only_segment_ids,
            "provided_translation_segment_ids": sorted(str(value) for value in translations),
            "provided_annotation_segment_ids": sorted(str(value) for value in annotations),
            "rendered_translation_segment_ids": rendered_translation_ids,
            "rendered_annotation_segment_ids": rendered_annotation_ids,
            "annotation_sources_by_segment": annotation_sources_by_segment,
            "translations": translation_audits,
            "chapter_ids": [str(item.get("chapter_id") or "") for item in chapters or []],
            "expected_chapter_guide_ids": [
                str(item.get("chapter_id") or "")
                for item in chapters or []
                if _chapter_guide_has_content(
                    chapter_guides.get(str(item.get("chapter_id") or "")) or {}
                )
            ],
            "rendered_chapter_guide_ids": rendered_chapter_guides,
        },
    }
    return tex, manifest


def validate_tex_fidelity(tex: str, document: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_blocks = [
        {"block_id": block_id(block), "sha256": sha256_json(block)}
        for block in document.get("blocks") or []
    ]
    if manifest.get("blocks") != expected_blocks:
        errors.append("source block manifest does not match the current document")
    for record in expected_blocks:
        marker = _source_block_marker(record["block_id"], record["sha256"]).strip()
        if tex.count(marker) != 1:
            errors.append(f"source block {record['block_id']} is not covered exactly once")
    if manifest.get("document_sha256") != sha256_json(document):
        errors.append("source document hash mismatch")
    recorded_document_hash = str(
        document.get("document_hash") or (document.get("integrity") or {}).get("document_hash") or ""
    )
    if str(manifest.get("document_hash") or "") != recorded_document_hash:
        errors.append("arc-paper document hash mismatch")
    errors.extend(_validate_source_credit_tex(tex, manifest.get("source_credit")))
    for number in manifest.get("equation_numbers") or []:
        if f"\\tag{{{escape_tex(_clean_tag(number))}}}" not in tex:
            errors.append(f"missing equation number {number}")
    expected_labels = [value for value in manifest.get("bibliography_labels") or [] if value]
    for label in expected_labels:
        if escape_tex(label) not in tex:
            errors.append(f"missing bibliography label {label}")
    for asset in manifest.get("assets") or []:
        path = Path(str(asset.get("output_path") or ""))
        if not path.exists():
            errors.append(f"missing copied asset {path}")
        elif asset.get("output_sha256") and sha256_file(path) != asset["output_sha256"]:
            errors.append(f"asset hash mismatch {path}")
    expected_links = _link_records(document.get("links") or [])
    if manifest.get("expected_links") != expected_links:
        errors.append("source link manifest does not match the current document")
    rendered_links = [dict(item) for item in manifest.get("rendered_links") or []]
    allow_unresolved_targets = (
        (document.get("preview_scope") or {}).get("kind") == "source_prefix"
    )
    for expected in expected_links:
        if not _consume_matching_link(expected, rendered_links):
            errors.append(f"source link was not rendered: {expected.get('href')}")
        target = expected.get("target_id")
        if target and not allow_unresolved_targets and f"\\label{{{_safe_label(target)}}}" not in tex:
            errors.append(f"internal link target was not rendered: {target}")
    if rendered_links:
        errors.append(f"rendered {len(rendered_links)} unregistered source link occurrence(s)")
    expected_tables = [_table_audit_record(item) for item in document.get("tables") or []]
    if manifest.get("tables") != expected_tables:
        errors.append("source table manifest does not match the current document")
    for table in expected_tables:
        marker = _table_marker(table["id"], table["sha256"]).strip()
        if tex.count(marker) != 1:
            errors.append(f"table {table['id']} was not rendered exactly once")
    errors.extend(_validate_companion_layers(tex, manifest.get("companion_layers") or {}))
    return errors


def _preamble(
    *,
    title: Any,
    translated_title: Any = "",
    translated_title_tex: str = "",
    authors: str,
    language: str,
    source_language: str = "und",
    include_translation: bool = True,
) -> str:
    translation_definitions = rf"""
\definecolor{{ArcTranslationBackground}}{{HTML}}{{F0F7F3}}
\definecolor{{ArcTranslationRule}}{{HTML}}{{6E9B89}}
\newtcolorbox{{arctranslation}}[1][]{{enhanced,breakable,sharp corners,boxrule=0pt,
  leftrule=1.2pt,colback=ArcTranslationBackground,colframe=ArcTranslationRule,
  left=8pt,right=8pt,top=6pt,bottom=6pt,before skip=5pt,after skip=8pt,#1}}
""" if include_translation else ""
    font_setup = _font_setup(source_language, language)
    translated_title_tex = translated_title_tex or escape_tex(translated_title)
    bilingual_title = escape_tex(title)
    if translated_title_tex and str(translated_title).strip() != str(title).strip():
        bilingual_title += rf"\par\vspace{{.55em}}{{\Large {translated_title_tex}\par}}"
    return rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage{{amsmath,amssymb,mathtools}}
\usepackage{{graphicx,longtable,array,multirow,hyperref,xcolor,needspace}}
\usepackage[most]{{tcolorbox}}
\usepackage{{fontspec,xeCJK}}
{font_setup}
\usepackage[margin=25mm]{{geometry}}
\hypersetup{{hidelinks}}
\definecolor{{ArcCompanionBackground}}{{HTML}}{{FAF3E8}}
\definecolor{{ArcCompanionRule}}{{HTML}}{{A8735D}}
\newenvironment{{arcsource}}{{\par\begingroup}}{{\par\endgroup}}
{translation_definitions}
\tcbset{{arccompanionsurface/.style={{enhanced,breakable,sharp corners,boxrule=0pt,
  leftrule=1.2pt,colback=ArcCompanionBackground,colframe=ArcCompanionRule,
  left=8pt,right=8pt,top=6pt,bottom=6pt,before skip=5pt,after skip=12pt}}}}
\newtcolorbox{{arccompanion}}[1][]{{arccompanionsurface,#1}}
\newtcolorbox{{arcchapterguide}}[1][]{{arccompanionsurface,#1}}
\setlength{{\parindent}}{{1.5em}}
\setlength{{\parskip}}{{0.35em}}
\begin{{document}}
\sffamily
\begin{{titlepage}}
\centering
\vspace*{{\fill}}
{{\LARGE {bilingual_title}\par}}
\vspace{{2.5em}}
{{\large {escape_tex(authors)}\par}}
\vspace{{2.5em}}
{{\small {escape_tex(_labels(language)["language"])}: {escape_tex(language)}\par}}
\vspace*{{\fill}}
\end{{titlepage}}
"""


def _render_front_matter(front: dict[str, Any], *, represented_roles: set[str]) -> str:
    parts: list[str] = []
    affiliations = front.get("affiliations") or front.get("institutions") or []
    if affiliations and "affiliation" not in represented_roles:
        if not isinstance(affiliations, list):
            affiliations = [affiliations]
        values = [_front_value(item) for item in affiliations]
        parts.append("\\begin{center}\\small " + r" \\ ".join(escape_tex(value) for value in values if value) + "\\end{center}\n")
    abstract = front.get("abstract")
    if abstract and "abstract" not in represented_roles:
        parts.append(f"\\begin{{abstract}}\n{escape_tex(_front_value(abstract))}\n\\end{{abstract}}\n")
    keywords = front.get("keywords") or []
    if keywords and "keywords" not in represented_roles:
        if isinstance(keywords, list):
            keywords = ", ".join(_front_value(value) for value in keywords)
        parts.append(f"\\noindent\\textbf{{Keywords:}} {escape_tex(keywords)}\\par\n")
    return "".join(parts)


def _render_source_credit(
    source_credit: Mapping[str, Any],
    *,
    front_matter_block_ids: Sequence[str] = (),
) -> tuple[str, dict[str, str], dict[str, Any]]:
    credit = validate_source_credit(source_credit)
    ordered = ordered_source_credit_items(credit)
    placement = source_credit_placement(
        credit, front_matter_block_ids=front_matter_block_ids,
    )
    records = {
        (kind, str(record["id"])): record
        for kind, record, _anchor in ordered
    }
    header_items = [
        item for item in placement
        if item["slot"] in {"after_title", "front_matter"}
    ]
    block_items: dict[str, list[dict[str, Any]]] = {}
    for item in placement:
        if item["slot"] == "source_block" and item["block_id"]:
            block_items.setdefault(str(item["block_id"]), []).append(item)
    header_tex = _source_credit_rows(header_items, records=records)
    by_block = {
        block_id_value: _source_credit_rows(items, records=records)
        for block_id_value, items in block_items.items()
    }
    anchors = {item["id"]: item for item in credit["anchors"]}
    manifest_order: list[dict[str, Any]] = []
    for item in placement:
        anchor = anchors[str(item["anchor_id"])]
        manifest_order.append(dict(item))
    manifest = {
        "canonical": credit,
        "schema_version": credit["schema_version"],
        "canonical_sha256": credit["canonical_sha256"],
        "front_matter_block_ids": list(front_matter_block_ids),
        "replaced_block_ids": [
            str(item["block_id"])
            for item in manifest_order
            if item["slot"] == "source_block" and item["block_id"]
        ],
        "ordered_items": manifest_order,
        "placements": [
            {
                "anchor_id": item["anchor_id"],
                "placement": anchors[item["anchor_id"]]["placement"],
                "block_id": anchors[item["anchor_id"]]["block_id"],
                "render_slot": item["slot"],
            }
            for item in manifest_order
        ],
        "visible_counts": {
            "authors": len(credit["authors"]),
            "affiliations": len(credit["affiliations"]),
            "profiles": len(credit["profiles"]),
        },
    }
    return header_tex, by_block, manifest


def _source_credit_rows(
    items: Sequence[Mapping[str, Any]],
    *,
    records: Mapping[tuple[str, str], Mapping[str, Any]],
) -> str:
    if not items:
        return ""
    rows: list[str] = ["\\begin{center}\n"]
    for item in items:
        kind = str(item["kind"])
        record = records[(kind, str(item["id"]))]
        identifier = str(record["id"])
        content_hash = str(record["content_sha256"])
        rows.append(_source_credit_marker(kind, identifier, content_hash))
        if kind == "author":
            source_name = escape_tex(str(record["source_name"]))
            localized = record.get("localized_name")
            if localized:
                rows.append(
                    "{\\large "
                    + source_name
                    + " (\\textit{"
                    + escape_tex(str(localized))
                    + "})\\par}\n"
                )
            else:
                rows.append("{\\large " + source_name + "\\par}\n")
        elif kind == "affiliation":
            rows.append("{\\small " + escape_tex(str(record["text"])) + "\\par}\n")
        else:
            rows.append(
                "\\begin{minipage}{0.88\\textwidth}\\small "
                + escape_tex(str(record["text"]))
                + "\\end{minipage}\\par\n"
            )
    rows.append("\\end{center}\n")
    return "".join(rows)


def _validate_source_credit_tex(tex: str, raw_manifest: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(raw_manifest, Mapping):
        return ["source-credit manifest is missing"]
    try:
        credit = validate_source_credit(raw_manifest.get("canonical"))
    except (SourceCreditError, TypeError):
        return ["source-credit manifest canonical object is invalid"]
    ordered = ordered_source_credit_items(credit)
    anchors = {item["id"]: item for item in credit["anchors"]}
    front_ids = raw_manifest.get("front_matter_block_ids")
    if not isinstance(front_ids, list) or not all(
        isinstance(item, str) for item in front_ids
    ):
        return ["source-credit front-matter placement is invalid"]
    expected_order = source_credit_placement(
        credit, front_matter_block_ids=front_ids,
    )
    expected = {
        "canonical": credit,
        "schema_version": credit["schema_version"],
        "canonical_sha256": credit["canonical_sha256"],
        "front_matter_block_ids": front_ids,
        "replaced_block_ids": [
            str(item["block_id"])
            for item in expected_order
            if item["slot"] == "source_block" and item["block_id"]
        ],
        "ordered_items": expected_order,
        "placements": [
            {
                "anchor_id": item["anchor_id"],
                "placement": anchors[item["anchor_id"]]["placement"],
                "block_id": anchors[item["anchor_id"]]["block_id"],
                "render_slot": item["slot"],
            }
            for item in expected_order
        ],
        "visible_counts": {
            "authors": len(credit["authors"]),
            "affiliations": len(credit["affiliations"]),
            "profiles": len(credit["profiles"]),
        },
    }
    if dict(raw_manifest) != expected:
        errors.append("source-credit manifest facts do not match the canonical object")
    positions: list[int] = []
    for kind, record, _anchor in ordered:
        marker = _source_credit_marker(
            kind, str(record["id"]), str(record["content_sha256"])
        ).strip()
        if tex.count(marker) != 1:
            errors.append(
                f"source-credit {kind} {record['id']} is not rendered exactly once"
            )
            continue
        positions.append(tex.index(marker))
        if kind == "author":
            source_name = escape_tex(str(record["source_name"]))
            localized = record.get("localized_name")
            expected_text = (
                source_name + " (\\textit{" + escape_tex(str(localized)) + "})"
                if localized else source_name
            )
        else:
            expected_text = escape_tex(str(record["text"]))
        marker_position = tex.index(marker)
        if tex.find(expected_text, marker_position) < marker_position:
            errors.append(f"source-credit {kind} {record['id']} text is missing")
    if positions != sorted(positions):
        errors.append("source-credit TeX order differs from the canonical order")
    return errors


def validate_pdf_source_credit_text(
    pdf_path: Path,
    document: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the compiled searchable text against the canonical credit."""

    executable = shutil.which("pdftotext")
    if executable is None:
        raise LatexError("pdftotext is required for source-credit validation")
    completed = subprocess.run(
        [executable, str(pdf_path), "-"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise LatexError(
            "compiled PDF source-credit text could not be extracted: "
            + completed.stderr.strip()
        )
    searchable = _normalized_searchable_text(completed.stdout)
    credit_manifest = source_manifest.get("source_credit")
    credit_manifest = (
        credit_manifest if isinstance(credit_manifest, Mapping)
        else source_manifest
    )
    try:
        credit = validate_source_credit(credit_manifest.get("canonical"))
    except (SourceCreditError, TypeError) as exc:
        raise LatexError("compiled PDF source-credit manifest is invalid") from exc
    placements = credit_manifest.get("ordered_items")
    expected = source_credit_placement(
        credit,
        front_matter_block_ids=credit_manifest.get("front_matter_block_ids") or [],
    )
    if placements != expected:
        raise LatexError("compiled PDF source-credit placement is invalid")
    records = {
        (kind, str(item["id"])): item
        for key, kind in (
            ("authors", "author"),
            ("affiliations", "affiliation"),
            ("profiles", "profile"),
        )
        for item in credit[key]
    }
    sequence: list[dict[str, Any]] = []
    for item in expected:
        record = records[(str(item["kind"]), str(item["id"]))]
        source_text = _normalized_searchable_text(
            str(record.get("source_name") or record.get("text") or "")
        )
        if not source_text:
            raise LatexError("compiled PDF source-credit text is empty")
        sequence.append({
            "kind": item["kind"],
            "id": item["id"],
            "source_text": source_text,
            "localized_text": _normalized_searchable_text(
                str(record.get("localized_name") or "")
            ) or None,
            "slot": item["slot"],
            "block_id": item["block_id"],
        })
    observed_order: list[str] = []
    groups: dict[tuple[str, str | None], list[dict[str, Any]]] = {}
    for item in sequence:
        key = (
            "header" if item["slot"] in {"after_title", "front_matter"}
            else "source_block",
            item["block_id"],
        )
        groups.setdefault(key, []).append(item)
    replaced_ids = {
        str(item.get("block_id"))
        for item in sequence
        if item["slot"] == "source_block" and item.get("block_id")
    }
    scoped_regions = _pdf_source_credit_scopes(
        searchable,
        document,
        groups=list(groups.items()),
        replaced_block_ids=replaced_ids,
    )
    for (region_kind, region_block_id), items in groups.items():
        scoped = scoped_regions[(region_kind, region_block_id)]
        cursor = 0
        matched: list[dict[str, Any]] = []
        for item in items:
            source_span = _next_bounded_text_span(
                scoped, item["source_text"], cursor,
            )
            if source_span is None:
                raise LatexError("compiled PDF source-credit order is invalid")
            position, source_end = source_span
            localized = item["localized_text"]
            if localized:
                localized_span = _next_bounded_text_span(
                    scoped, localized, source_end,
                )
                if (
                    localized_span is None
                    or localized_span[0] - source_end > 8
                ):
                    raise LatexError(
                        "compiled PDF localized author is not adjacent to "
                        "its source name"
                    )
                cursor = localized_span[1]
            else:
                cursor = source_end
            matched.append({
                "source_text": item["source_text"],
                "source_span": source_span,
                "entry_span": (position, cursor),
            })
            observed_order.append(str(item["id"]))
        for text in {item["source_text"] for item in items}:
            expected_source_spans = {
                item["source_span"]
                for item in matched
                if item["source_text"] == text
            }
            visible_occurrences = []
            for occurrence in _bounded_text_spans(scoped, text):
                if occurrence in expected_source_spans:
                    visible_occurrences.append(occurrence)
                    continue
                if any(
                    entry["entry_span"][0] <= occurrence[0]
                    and occurrence[1] <= entry["entry_span"][1]
                    for entry in matched
                ):
                    continue
                visible_occurrences.append(occurrence)
            if len(visible_occurrences) != len(expected_source_spans):
                raise LatexError(
                    "compiled PDF source-credit occurrence count is invalid "
                    f"in its credit region: {text}"
                )
    return {
        "schema_version": "arc.companion.source-credit-pdf-observation.v1",
        "canonical_sha256": credit["canonical_sha256"],
        "searchable_text_sha256": hashlib.sha256(
            searchable.encode("utf-8")
        ).hexdigest(),
        "ordered_ids": observed_order,
        "visible_projection_sha256": sha256_json(
            source_credit_visible_projection(
                credit,
                front_matter_block_ids=(
                    credit_manifest.get("front_matter_block_ids") or []
                ),
            )
        ),
        "visible_counts": {
            "authors": sum(item["kind"] == "author" for item in sequence),
            "affiliations": sum(
                item["kind"] == "affiliation" for item in sequence
            ),
            "profiles": sum(item["kind"] == "profile" for item in sequence),
        },
    }


def _normalized_searchable_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _bounded_text_spans(value: str, text: str) -> list[tuple[int, int]]:
    if not text:
        return []
    prefix = r"(?<!\w)" if text[0].isalnum() else ""
    suffix = r"(?!\w)" if text[-1].isalnum() else ""
    return [
        match.span()
        for match in re.finditer(prefix + re.escape(text) + suffix, value)
    ]


def _next_bounded_text_span(
    value: str, text: str, cursor: int,
) -> tuple[int, int] | None:
    return next(
        (span for span in _bounded_text_spans(value, text) if span[0] >= cursor),
        None,
    )


def _pdf_source_credit_scopes(
    searchable: str,
    document: Mapping[str, Any],
    *,
    groups: Sequence[
        tuple[tuple[str, str | None], Sequence[Mapping[str, Any]]]
    ],
    replaced_block_ids: set[str],
) -> dict[tuple[str, str | None], str]:
    """Partition searchable text into disjoint source-credit regions."""

    blocks = [
        item for item in document.get("blocks") or []
        if isinstance(item, Mapping)
    ]
    front_group_ids = {
        str(value)
        for key, values in (
            (document.get("front_matter") or {}).get("block_ids") or {}
        ).items()
        if key in {"title", "authors", "affiliations"}
        for value in values
    }
    credit_block_ids = replaced_block_ids | front_group_ids
    landmarks: list[tuple[int, int]] = []
    cursor = 0
    for _key, items in groups:
        first_start: int | None = None
        last_end: int | None = None
        for item in items:
            source_span = _next_bounded_text_span(
                searchable, str(item["source_text"]), cursor,
            )
            if source_span is None:
                raise LatexError("compiled PDF source-credit order is invalid")
            if first_start is None:
                first_start = source_span[0]
            last_end = source_span[1]
            cursor = source_span[1]
            localized = str(item.get("localized_text") or "")
            if localized:
                localized_span = _next_bounded_text_span(
                    searchable, localized, cursor,
                )
                if (
                    localized_span is None
                    or localized_span[0] - cursor > 8
                ):
                    raise LatexError(
                        "compiled PDF localized author is not adjacent to "
                        "its source name"
                    )
                last_end = localized_span[1]
                cursor = localized_span[1]
        if first_start is None or last_end is None:
            raise LatexError("compiled PDF source-credit region is empty")
        landmarks.append((first_start, last_end))

    title = _normalized_searchable_text(
        str((document.get("front_matter") or {}).get("title") or "")
    )
    title_position = searchable.find(title) if title else -1
    block_indexes = {
        str(block.get("block_id") or ""): index
        for index, block in enumerate(blocks)
        if str(block.get("block_id") or "")
    }
    output: dict[tuple[str, str | None], str] = {}
    for index, ((key, _items), landmark) in enumerate(zip(groups, landmarks)):
        region_kind, block_id = key
        start = landmarks[index - 1][1] if index else 0
        end = landmarks[index + 1][0] if index + 1 < len(landmarks) else len(
            searchable
        )
        if region_kind == "header" and title_position >= 0:
            start = max(start, title_position + len(title))
        block_index = block_indexes.get(str(block_id or ""))
        if region_kind == "header":
            candidate_blocks = blocks
        elif block_index is None:
            raise LatexError("compiled PDF source-credit block scope is ambiguous")
        else:
            candidate_blocks = blocks[block_index + 1:]
            for block in reversed(blocks[:block_index]):
                candidate_id = str(block.get("block_id") or "")
                if candidate_id in credit_block_ids:
                    continue
                text = _normalized_searchable_text(
                    str(block.get("text") or "")
                )
                positions = _bounded_text_spans(searchable, text)
                previous = [
                    span for span in positions if span[1] <= landmark[0]
                ]
                if previous:
                    start = max(start, previous[-1][1])
                    break
        for block in candidate_blocks:
            candidate_id = str(block.get("block_id") or "")
            if candidate_id in credit_block_ids:
                continue
            text = _normalized_searchable_text(str(block.get("text") or ""))
            positions = _bounded_text_spans(searchable, text)
            following = [
                span for span in positions if span[0] >= landmark[1]
            ]
            if following:
                end = min(end, following[0][0])
                break
        if not (start <= landmark[0] < landmark[1] <= end):
            raise LatexError("compiled PDF source-credit region bounds are invalid")
        output[key] = searchable[start:end]
    return output


def _source_credit_marker(kind: str, identifier: str, content_hash: str) -> str:
    return (
        f"% ARC-SOURCE-CREDIT {kind} "
        f"{_safe_label(identifier)} {content_hash}\n"
    )


def _render_block(
    block: dict[str, Any],
    *,
    equations: dict[str, dict[str, Any]],
    figures: dict[str, dict[str, Any]],
    tables: dict[str, dict[str, Any]],
    assets: dict[str, dict[str, Any]],
    output_dir: Path,
    copied_assets: list[dict[str, Any]],
    rendered_links: list[dict[str, str]],
    translated_title: str = "",
) -> str:
    kind = _kind(block)
    heading = block.get("heading") if isinstance(block.get("heading"), dict) else {}
    if kind in {"part", "chapter", "section", "heading", "subsection", "subsubsection"} or block.get("heading_level") or heading:
        level = int(
            block.get("heading_level")
            or block.get("level")
            or heading.get("level")
            or {"section": 1, "subsection": 2, "subsubsection": 3}.get(kind, 1)
        )
        command = (
            "part" if kind == "part"
            else "section" if kind == "chapter"
            else {1: "section", 2: "subsection", 3: "subsubsection"}.get(level, "paragraph")
        )
        title = (
            block.get("title")
            or heading.get("title")
            or heading.get("text")
            or (block.get("heading") if isinstance(block.get("heading"), str) else None)
            or block.get("text")
        )
        rendered_title = _render_html_fragment(block.get("html"), rendered_links=rendered_links, contents_only=True)
        title_tex = rendered_title or escape_tex(title)
        translated_title_tex = _render_translated_inline_runs(translated_title, block)
        body_title_tex = title_tex
        if translated_title_tex and str(translated_title).strip() != str(title).strip():
            body_title_tex += rf"\protect\\{{\small {translated_title_tex}}}"
        navigation_title_tex = translated_title_tex or title_tex
        # Source headings already carry the paper's own section number (when
        # numbered).  A numbered LaTeX command would prepend a second number,
        # e.g. ``0.1 1 Introduction`` for an ar5iv h2 rendered as a subsection.
        # Keep the source text verbatim while registering the starred heading
        # explicitly so its hierarchy remains available to the TOC/bookmarks.
        return (
            _anchors(block)
            + f"\\{command}*{{{body_title_tex}}}\n"
            + f"\\addcontentsline{{toc}}{{{command}}}{{{navigation_title_tex}}}\n"
        )
    if kind in {"equation", "math", "display_math"}:
        entity = _entity_for(block, equations)
        return _anchors(block) + _render_equation(entity or block)
    if kind in {"figure", "image"}:
        entity = _entity_for(block, figures) or block
        return _anchors(block) + _render_figure(
            entity,
            assets=assets,
            output_dir=output_dir,
            copied_assets=copied_assets,
            rendered_links=rendered_links,
        )
    if kind == "table":
        return _anchors(block) + _render_table(
            _entity_for(block, tables) or block,
            rendered_links=rendered_links,
        )
    list_items = block.get("items") if isinstance(block.get("items"), list) else block.get("list_items")
    list_style = str(block.get("list_kind") or block.get("list_type") or "").lower()
    if isinstance(list_items, list) and (
        kind in {"list", "itemize", "enumerate", "ordered_list", "unordered_list"} or list_style
    ):
        ordered = bool(
            block.get("ordered")
            or kind in {"enumerate", "ordered_list"}
            or list_style in {"ordered", "ol", "enumerate", "numbered"}
        )
        if block.get("html"):
            return _anchors(block) + _render_html_fragment(block["html"], rendered_links=rendered_links)
        return _anchors(block) + _render_list(list_items, ordered=ordered)
    text = block.get("latex")
    if text and block.get("latex_trusted") is True:
        return _anchors(block) + str(text) + "\n\n"
    if block.get("html"):
        return _anchors(block) + _render_html_fragment(block["html"], rendered_links=rendered_links)
    visible_text = clean_reader_text(block.get("text") or block.get("title") or "")
    return _anchors(block) + escape_tex(visible_text) + "\n\n"


def _render_equation(
    entity: dict[str, Any],
    *,
    include_numbers: bool = True,
    include_labels: bool = True,
) -> str:
    raw_tex = entity.get("tex") or entity.get("latex") or entity.get("alttext") or entity.get("text")
    if not raw_tex:
        raise LatexError(f"equation {entity.get('id') or entity.get('equation_id')} has no TeX representation")
    tex_values = [str(value).strip() for value in raw_tex] if isinstance(raw_tex, list) else [str(raw_tex).strip()]
    if not include_numbers or not include_labels:
        tex_values = [
            _strip_equation_identity(
                value,
                strip_numbers=not include_numbers,
                strip_labels=not include_labels,
            )
            for value in tex_values
        ]
    raw_numbers = entity.get("printed_equation_numbers")
    if raw_numbers is None:
        raw_numbers = entity.get("printed_equation_number")
    if raw_numbers is None:
        raw_numbers = entity.get("number") or entity.get("equation_number") or entity.get("display_number") or entity.get("tag")
    numbers = [str(value) for value in raw_numbers] if isinstance(raw_numbers, list) else ([] if raw_numbers in {None, ""} else [str(raw_numbers)])
    raw_labels = entity.get("labels") or []
    labels = [str(value) for value in raw_labels] if isinstance(raw_labels, list) else [str(raw_labels)]
    primary_label = entity.get("label") or entity.get("tex_label")
    if len(tex_values) > 1 and len(numbers) != len(tex_values) and len(labels) != len(tex_values):
        aligned = r" \\ ".join(tex_values)
        number = numbers[0] if include_numbers and numbers else None
        label = (primary_label or (labels[0] if labels else None)) if include_labels else None
        return _equation_environment(f"\\begin{{aligned}}{aligned}\\end{{aligned}}", number=number, label=label)
    rendered = []
    for index, tex in enumerate(tex_values):
        label = (labels[index] if index < len(labels) else (primary_label if index == 0 else None)) if include_labels else None
        rendered.append(
            _equation_environment(
                tex,
                number=numbers[index] if include_numbers and index < len(numbers) else None,
                label=label,
            )
        )
    return "".join(rendered)


def _equation_environment(tex: str, *, number: Any, label: Any) -> str:
    tex = _disambiguate_math_row_starts(tex)
    if number not in {None, ""}:
        tex = _strip_equation_identity(tex, strip_numbers=True, strip_labels=False)
    tag = f"\n\\tag{{{escape_tex(_clean_tag(number))}}}" if number not in {None, ""} else ""
    label_tex = f"\n\\label{{{_safe_label(label)}}}" if label else ""
    return f"\\begingroup\\rmfamily\n\\begin{{equation*}}\n{tex}{tag}{label_tex}\n\\end{{equation*}}\n\\endgroup\n"


def _disambiguate_math_row_starts(tex: str) -> str:
    """Keep a bracketed row from being parsed as ``\\`` spacing.

    TeX ignores whitespace while looking for the optional argument to a math
    row terminator, so ``\\ [x]`` is parsed as a requested vertical length
    rather than as a new row beginning with ``[x]``.  An empty group is
    invisible but stops that optional-argument scan.  Compact, intentional
    spacing such as ``\\[2pt]`` is left unchanged.
    """
    return re.sub(r"\\\\(?P<space>\s+)(?=\[)", r"\\\\{}\g<space>", tex)


def _remove_disallowed_c0(value: str) -> str:
    """Drop Unicode control characters XeLaTeX cannot accept, preserving layout whitespace."""
    return "".join(
        char
        for char in value
        if char in "\n\r\t" or unicodedata.category(char) != "Cc"
    )


def _render_figure(
    entity: dict[str, Any],
    *,
    assets: dict[str, dict[str, Any]],
    output_dir: Path,
    copied_assets: list[dict[str, Any]],
    rendered_links: list[dict[str, str]],
) -> str:
    asset_ids = entity.get("asset_ids") or ([entity.get("asset_id")] if entity.get("asset_id") else [])
    if not asset_ids and entity.get("cache_path"):
        asset_ids = ["__inline__"]
        assets = {**assets, "__inline__": entity}
    if not asset_ids:
        raise LatexError(f"figure {entity.get('id') or entity.get('figure_id')} has no cached asset")
    rendered: list[str] = []
    for asset_id in asset_ids:
        asset = assets.get(str(asset_id))
        if not asset:
            raise LatexError(f"figure references unknown asset {asset_id}")
        source = asset_path(asset)
        if source is None or not source.is_file():
            raise LatexError(f"cached figure asset is missing: {source or asset_id}")
        expected = str(asset.get("sha256") or "")
        actual = sha256_file(source)
        if expected and expected != actual:
            raise LatexError(f"cached figure asset hash mismatch: {source}")
        extension = source.suffix.lower() or _media_extension(asset.get("media_type") or asset.get("content_type"))
        destination = _materialize_latex_asset(source, source_hash=actual, extension=extension, output_dir=output_dir)
        relative = destination.relative_to(output_dir).as_posix()
        copied_assets.append({
            "asset_id": str(asset_id),
            "source_sha256": actual,
            "output_sha256": sha256_file(destination),
            "output_path": str(destination),
        })
        rendered.append(f"\\includegraphics[width=0.95\\linewidth]{{\\detokenize{{{relative}}}}}")
    caption = _entity_caption(entity, rendered_links=rendered_links)
    number = entity.get("number") or entity.get("display_number") or entity.get("tag")
    label = _visible_tag(number, kind="Figure")
    return "\\begin{center}\n" + "\n".join(rendered) + f"\n\\par\\small {label}{caption}\n\\end{{center}}\n"


def _render_table(entity: dict[str, Any], *, rendered_links: list[dict[str, str]] | None = None) -> str:
    rendered_links = rendered_links if rendered_links is not None else []
    rows = entity.get("rows")
    grid = entity.get("grid")
    if not isinstance(rows, list) and not isinstance(grid, list):
        raise LatexError(f"table {entity.get('id') or entity.get('table_id')} has no canonical rows")
    column_count, layout_rows = _table_layout(entity)
    table_id = str(entity.get("id") or entity.get("table_id") or "table")
    table_hash = sha256_json(_table_source_shape(entity))
    lines = [_table_marker(table_id, table_hash).rstrip(), f"\\begin{{longtable}}{{{'l' * column_count}}}"]
    caption = entity.get("caption")
    number = entity.get("number") or entity.get("display_number") or entity.get("tag")
    if caption or number:
        prefix = _visible_tag(number, kind="Table")
        rendered_caption = _entity_caption(entity, rendered_links=rendered_links)
        lines.append(f"\\caption*{{{prefix}{rendered_caption}}}\\\\")
    for tokens in layout_rows:
        width = sum(int(cell.get("colspan") or 1) if isinstance(cell, dict) else 1 for cell in tokens)
        if width != column_count:
            raise LatexError(f"table row renders {width} columns, expected exactly {column_count}")
        lines.append(
            " & ".join(
                _render_cell(cell, rendered_links=rendered_links) if cell is not None else ""
                for cell in tokens
            ) + r" \\"
        )
    lines.append("\\end{longtable}")
    return "\n".join(lines) + "\n"


def _render_cell(cell: Any, *, rendered_links: list[dict[str, str]] | None = None) -> str:
    rendered_links = rendered_links if rendered_links is not None else []
    if not isinstance(cell, dict):
        return escape_tex(cell)
    content = _render_html_fragment(cell.get("html"), rendered_links=rendered_links, contents_only=True)
    if not content:
        content = escape_tex(cell.get("text") or cell.get("content") or "")
    rowspan = int(cell.get("rowspan") or 1)
    colspan = int(cell.get("colspan") or 1)
    if rowspan > 1:
        content = f"\\multirow{{{rowspan}}}{{*}}{{{content}}}"
    if colspan > 1:
        content = f"\\multicolumn{{{colspan}}}{{l}}{{{content}}}"
    return content


def _render_list(items: list[Any], *, ordered: bool) -> str:
    environment = "enumerate" if ordered else "itemize"
    lines = [f"\\begin{{{environment}}}"]
    for item in items:
        if isinstance(item, dict):
            text = item.get("text") or item.get("content") or item.get("title") or ""
            lines.append(f"\\item {escape_tex(text)}")
            children = item.get("items") or item.get("children")
            if isinstance(children, list) and children:
                lines.append(_render_list(children, ordered=bool(item.get("ordered"))))
        else:
            lines.append(f"\\item {escape_tex(item)}")
    lines.append(f"\\end{{{environment}}}")
    return "\n".join(lines) + "\n"


def _table_layout(entity: dict[str, Any]) -> tuple[int, list[list[dict[str, Any] | None]]]:
    """Return rows whose token widths add up to the table's exact column count."""
    raw_rows = entity.get("rows") or []
    grid = entity.get("grid") or []
    if grid:
        if not all(isinstance(row, list) for row in grid):
            raise LatexError("canonical table grid must be rectangular")
        grid_widths = {len(row) for row in grid if isinstance(row, list)}
        if len(grid_widths) != 1:
            raise LatexError("canonical table grid must be rectangular")
        grid_width = next(iter(grid_widths))
    else:
        grid_width = 0

    origins: dict[tuple[int, int], dict[str, Any]] = {}
    occupancy: dict[tuple[int, int], tuple[int, int]] = {}

    def add_origin(cell: dict[str, Any], row: int, column: int) -> None:
        rowspan = _positive_span(cell.get("rowspan"))
        colspan = _positive_span(cell.get("colspan"))
        normalized = {**cell, "row": row, "column": column, "rowspan": rowspan, "colspan": colspan}
        origin = (row, column)
        if origin in origins:
            return
        for row_offset in range(rowspan):
            for column_offset in range(colspan):
                coordinate = (row + row_offset, column + column_offset)
                if coordinate in occupancy:
                    raise LatexError(f"overlapping table cells at row {coordinate[0]}, column {coordinate[1]}")
                occupancy[coordinate] = origin
        origins[origin] = normalized

    for row_index, raw_row in enumerate(raw_rows):
        cells = raw_row.get("cells") if isinstance(raw_row, dict) else raw_row
        if not isinstance(cells, list):
            raise LatexError("table row has no cells")
        next_column = 0
        for cell in cells:
            if not isinstance(cell, dict):
                cell = {"text": cell}
            row = int(cell.get("row") if cell.get("row") is not None else row_index)
            explicit_column = cell.get("column")
            if explicit_column is not None:
                column = int(explicit_column)
            else:
                while (row, next_column) in occupancy:
                    next_column += 1
                column = next_column
            if row < 0 or column < 0:
                raise LatexError("table cell positions must be non-negative")
            add_origin(cell, row, column)
            next_column = column + _positive_span(cell.get("colspan"))

    grid_groups: dict[tuple[int, int], list[tuple[int, int, dict[str, Any]]]] = {}
    for row_index, grid_row in enumerate(grid):
        for column_index, cell in enumerate(grid_row):
            if not isinstance(cell, dict):
                continue
            origin_value = cell.get("origin")
            if isinstance(origin_value, dict):
                source_row = int(origin_value.get("row", row_index))
                source_column = int(origin_value.get("column", column_index))
            else:
                source_row = int(cell.get("source_row", cell.get("origin_row", row_index)))
                source_column = int(cell.get("source_column", cell.get("origin_column", column_index)))
            origin = (source_row, source_column)
            grid_groups.setdefault(origin, []).append((row_index, column_index, cell))

    for (source_row, source_column), members in grid_groups.items():
        if (source_row, source_column) in origins:
            continue
        origin_cell = next((cell for row, column, cell in members if (row, column) == (source_row, source_column)), members[0][2])
        max_row = max(row for row, _, _ in members)
        max_column = max(column for _, column, _ in members)
        inferred = dict(origin_cell)
        inferred.setdefault("rowspan", max_row - source_row + 1)
        inferred.setdefault("colspan", max_column - source_column + 1)
        add_origin(inferred, source_row, source_column)

    for row_index, grid_row in enumerate(grid):
        for column_index, cell in enumerate(grid_row):
            if not isinstance(cell, dict):
                if (row_index, column_index) in occupancy:
                    raise LatexError(f"canonical grid omits a spanned cell at row {row_index}, column {column_index}")
                continue
            origin_value = cell.get("origin")
            if isinstance(origin_value, dict):
                source_row = int(origin_value.get("row", row_index))
                source_column = int(origin_value.get("column", column_index))
            else:
                source_row = int(cell.get("source_row", cell.get("origin_row", row_index)))
                source_column = int(cell.get("source_column", cell.get("origin_column", column_index)))
            origin = (source_row, source_column)
            if origin not in origins and origin == (row_index, column_index):
                add_origin(cell, source_row, source_column)
            if occupancy.get((row_index, column_index)) != origin:
                raise LatexError(f"canonical grid disagrees with cell spans at row {row_index}, column {column_index}")

    inferred_width = max((column + _positive_span(cell.get("colspan")) for (_, column), cell in origins.items()), default=0)
    declared = int(entity.get("column_count") or 0)
    column_count = declared or grid_width or inferred_width
    if column_count < 1:
        raise LatexError("table has no columns")
    if grid_width and grid_width != column_count:
        raise LatexError(f"canonical grid has {grid_width} columns, expected exactly {column_count}")
    if inferred_width > column_count:
        raise LatexError(f"table cell spans {inferred_width} columns, expected exactly {column_count}")
    inferred_rows = max((row + _positive_span(cell.get("rowspan")) for (row, _), cell in origins.items()), default=0)
    row_count = len(grid) or max(len(raw_rows), inferred_rows)
    if len(grid) and inferred_rows > len(grid):
        raise LatexError("table cell rowspan exceeds the canonical grid")

    rendered_rows: list[list[dict[str, Any] | None]] = []
    for row in range(row_count):
        tokens: list[dict[str, Any] | None] = []
        column = 0
        while column < column_count:
            origin = occupancy.get((row, column))
            if origin == (row, column):
                cell = origins[origin]
                tokens.append(cell)
                column += _positive_span(cell.get("colspan"))
            else:
                tokens.append(None)
                column += 1
        rendered_rows.append(tokens)
    return column_count, rendered_rows


def _positive_span(value: Any) -> int:
    try:
        span = int(value or 1)
    except (TypeError, ValueError) as exc:
        raise LatexError(f"invalid table cell span: {value}") from exc
    if span < 1:
        raise LatexError(f"invalid table cell span: {value}")
    return span


def _render_annotation(
    segment_id: str,
    annotation: dict[str, Any],
    *,
    language: str,
    evidence_records: list[dict[str, Any]] | None = None,
) -> str:
    annotation = clean_reader_annotation(
        annotation,
        evidence_records=evidence_records or [],
        language=language,
    )
    labels = _labels(language)
    sections: list[str] = []
    explanation = str(annotation.get("explanation") or "").strip()
    commentary = str(annotation.get("commentary") or "").strip()
    prior = annotation.get("prior_work")
    later = annotation.get("later_work")
    combined_explanation = _merge_annotation_prose(explanation, commentary)
    if combined_explanation:
        sections.append(_annotation_section(
            labels["explanation"],
            combined_explanation,
            sources=annotation.get("commentary_sources"),
        ))
    if prior:
        sections.append(_annotation_section(labels["prior"], prior))
    if later:
        sections.append(_annotation_section(labels["later"], later))
    if not sections:
        return (
            _layer_marker("COMPANION", "BEGIN", segment_id)
            + _layer_marker("COMPANION", "END", segment_id)
        )
    return (
        _layer_marker("COMPANION", "BEGIN", segment_id)
        +
        f"\\Needspace{{6\\baselineskip}}\n"
        + _begin_guarded_box("arccompanion")
        + "\n".join(sections)
        + "\n\\end{arccompanion}\n"
        + _layer_marker("COMPANION", "END", segment_id)
    )


def _merge_annotation_prose(explanation: str, commentary: str) -> str:
    if not explanation:
        return commentary
    if not commentary:
        return explanation
    explanation_key = re.sub(r"\s+", " ", explanation).strip()
    commentary_key = re.sub(r"\s+", " ", commentary).strip()
    if explanation_key == commentary_key:
        return explanation
    if explanation_key in commentary_key:
        return commentary
    if commentary_key in explanation_key:
        return explanation
    return f"{explanation}\n\n{commentary}"


def _render_translation(
    segment_id: str,
    segment: dict[str, Any],
    translation: dict[str, Any],
    *,
    document: dict[str, Any],
    equations: dict[str, dict[str, Any]],
    language: str,
) -> tuple[str, dict[str, Any]]:
    """Render translated prose plus faithful, unnumbered copies of source equations."""
    labels = _labels(language)
    translated_blocks = {
        str(item.get("block_id") or ""): item
        for item in translation.get("blocks") or []
        if isinstance(item, dict) and item.get("block_id")
    }
    source_blocks = {block_id(item): item for item in document.get("blocks") or []}
    parts = [_layer_marker("TRANSLATION", "BEGIN", segment_id),
             _begin_guarded_box("arctranslation")]
    translated_block_ids: list[str] = []
    equation_block_ids: list[str] = []
    excluded_float_block_ids: list[str] = []
    front_roles = _front_matter_block_roles(document.get("blocks") or [], document.get("front_matter") or {})
    for bid in segment.get("block_ids") or []:
        source = source_blocks.get(str(bid))
        if source is None:
            continue
        kind = _kind(source)
        if front_roles.get(str(bid)) in {"title", "author"} or kind in {
            "figure", "image", "table", "bibliography", "bibliography_item", "reference"
        }:
            if kind in {"figure", "image", "table"}:
                excluded_float_block_ids.append(str(bid))
            continue
        if kind in {"equation", "math", "display_math"}:
            equation_block_ids.append(str(bid))
            parts.append(_translation_equation_marker(segment_id, str(bid)))
            parts.append(_render_equation(_entity_for(source, equations) or source, include_numbers=False, include_labels=False))
            continue
        item = translated_blocks.get(str(bid))
        if not item or item.get("translate") is False:
            continue
        text = item.get("text")
        if text in {None, ""}:
            text = item.get("translated_text") or item.get("translation") or ""
        if text:
            translated_block_ids.append(str(bid))
            parts.append(_render_translated_inline_runs(text, source) + "\n\n")
    parts.extend(["\\end{arctranslation}\n", _layer_marker("TRANSLATION", "END", segment_id)])
    return "".join(parts), {
        "segment_id": segment_id,
        "translated_block_ids": translated_block_ids,
        "equation_block_ids": equation_block_ids,
        "excluded_float_block_ids": excluded_float_block_ids,
    }


def _begin_guarded_box(environment: str) -> str:
    """Begin a renderer-owned box without letting leading ``[`` become options."""

    if environment not in {"arctranslation", "arccompanion", "arcchapterguide"}:
        raise LatexError(f"unknown guarded box environment: {environment}")
    return f"\\begin{{{environment}}}\\relax\n"


_OPAQUE_INLINE_PATTERN = re.compile(r"\[\[ARC_INLINE:([^\]\s]+):([0-9a-f]{64})\]\]")


def _render_translated_inline_runs(value: Any, source: dict[str, Any]) -> str:
    text = clean_reader_text(value)
    runs = {
        (str(run.get("token_id") or ""), str(run.get("content_hash") or "")): run
        for run in source.get("inline_runs") or []
        if isinstance(run, dict) and str(run.get("kind") or "") != "text"
    }
    if not runs:
        return _render_rich_text(text)
    output: list[str] = []
    position = 0
    for match in _OPAQUE_INLINE_PATTERN.finditer(text):
        output.append(_render_rich_text(text[position:match.start()]))
        run = runs.get((match.group(1), match.group(2)))
        if run is None:
            raise LatexError("translation contains an unknown or modified inline token")
        kind = str(run.get("kind") or "")
        content = str(run.get("content") or "")
        if kind == "math":
            tex = str(run.get("tex") or content)
            output.append(f"{{\\rmfamily\\({_strip_equation_identity(tex, strip_numbers=True, strip_labels=True)}\\)}}")
        elif kind == "link":
            href = str(run.get("href") or "")
            visible = escape_tex(content or href)
            if href.startswith("#"):
                output.append(f"\\hyperref[{_safe_label(href[1:])}]{{{visible}}}")
            elif href:
                output.append(f"\\href{{{_escape_url(href)}}}{{{visible}}}")
            else:
                output.append(visible)
        else:  # citation and other controller-owned visible tokens
            output.append(escape_tex(content))
        position = match.end()
    output.append(_render_rich_text(text[position:]))
    return "".join(output)


def _annotation_section(
    title: str,
    value: Any,
    *,
    sources: Any = None,
) -> str:
    if isinstance(value, list):
        rendered = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text") or item.get("summary") or item.get("claim") or item.get("title") or ""
                rendered.append(
                    _render_rich_text(text)
                    + _render_annotation_sources(item.get("sources"))
                )
            else:
                rendered.append(_render_rich_text(item))
        content = "\\begin{itemize}\n" + "\n".join(f"\\item {item}" for item in rendered) + "\n\\end{itemize}"
    else:
        content = _render_rich_text(value) + _render_annotation_sources(sources)
    return (
        "\\Needspace{4\\baselineskip}\n"
        f"\\medskip\\noindent\\textbf{{{escape_tex(title)}}}\\par\n{content}\n"
    )


def _render_annotation_sources(value: Any) -> str:
    sources = value if isinstance(value, list) else []
    rendered: list[str] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        title = str(source.get("title") or "").strip()
        url = str(source.get("url") or "").strip()
        locator = str(source.get("locator") or "").strip()
        if not (title and url and locator):
            continue
        rendered.append(
            f"\\href{{{_escape_url(url)}}}{{{escape_tex(title)}}}"
            f"\\,---\\,{escape_tex(locator)}"
        )
    if not rendered:
        return ""
    return (
        "\\par\\smallskip\\noindent{\\footnotesize "
        + "; ".join(rendered)
        + "}"
    )


def _annotation_source_manifest(annotation: dict[str, Any]) -> dict[str, Any]:
    """Preserve direct citations and their claim association without rewriting."""
    output: dict[str, Any] = {}
    if "commentary_sources" in annotation:
        output["commentary_sources"] = deepcopy(annotation["commentary_sources"])
    for field in ("prior_work", "later_work"):
        claims = annotation.get(field)
        if isinstance(claims, list) and any(
            isinstance(claim, dict) and "sources" in claim for claim in claims
        ):
            output[field] = deepcopy(claims)
    return output


def _render_rich_text(value: Any) -> str:
    """Escape prose while retaining explicit TeX math delimiters."""
    text = clean_reader_text(value)
    pattern = re.compile(r"(\\\[(?:.|\n)*?\\\]|\\\((?:.|\n)*?\\\)|\$\$(?:.|\n)*?\$\$|(?<!\\)\$(?:\\.|[^$\n])+?(?<!\\)\$)")
    result: list[str] = []
    position = 0
    for match in pattern.finditer(text):
        result.append(_render_undelimited_math_tokens(text[position:match.start()]))
        result.append("{\\rmfamily" + _strip_translation_equation_identity(match.group(0)) + "}")
        position = match.end()
    result.append(_render_undelimited_math_tokens(text[position:]))
    return "".join(result).replace("\n\n", "\\par\n")


def _render_undelimited_math_tokens(text: str) -> str:
    """Render conservative TeX-like tokens emitted without math delimiters."""
    rendered: list[str] = []
    plain_start = 0
    position = 0
    while position < len(text):
        if not _raw_math_start(text, position):
            position += 1
            continue
        end = _raw_math_end(text, position)
        token = text[position:end]
        if (
            end <= position
            or not _valid_raw_math_token(token)
            or not any(char in token for char in ("\\", "_", "^"))
        ):
            position += 1
            continue
        rendered.append(escape_tex(text[plain_start:position]))
        rendered.append(f"{{\\rmfamily\\({_strip_translation_equation_identity(token)}\\)}}")
        position = end
        plain_start = end
    rendered.append(escape_tex(text[plain_start:]))
    return "".join(rendered)


def _raw_math_start(text: str, position: int) -> bool:
    char = text[position]
    if char == "\\":
        return position + 1 < len(text) and text[position + 1].isalpha()
    if char == "{" and position + 1 < len(text) and text[position + 1] == "\\":
        return True
    if not char.isascii() or not char.isalpha():
        return False
    return position + 1 < len(text) and text[position + 1] in {"_", "^"}


def _raw_math_end(text: str, start: int) -> int:
    allowed = set("\\{}_^/().=+-*~[],:|")
    depth = 0
    position = start
    while position < len(text):
        char = text[position]
        if char == "{" and (position == 0 or text[position - 1] != "\\"):
            depth += 1
        elif char == "}" and (position == 0 or text[position - 1] != "\\"):
            depth -= 1
            if depth < 0:
                break
        elif char.isspace():
            if depth == 0:
                break
        elif not (char.isascii() and (char.isalnum() or char in allowed)):
            break
        position += 1
    return position


def _balanced_braces(value: str) -> bool:
    depth = 0
    for position, char in enumerate(value):
        if char == "{" and (position == 0 or value[position - 1] != "\\"):
            depth += 1
        elif char == "}" and (position == 0 or value[position - 1] != "\\"):
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _valid_raw_math_token(value: str) -> bool:
    if not _balanced_braces(value) or value.endswith(("\\", "_", "^")):
        return False
    for position, char in enumerate(value[:-1]):
        if char not in {"_", "^"}:
            continue
        following = value[position + 1]
        if following != "{" and following != "\\" and not (following.isascii() and following.isalnum()):
            return False
    return True


def _strip_translation_equation_identity(math: str) -> str:
    """Prevent model-returned math from duplicating source numbers or anchors."""
    return _strip_equation_identity(math, strip_numbers=True, strip_labels=True)


def _strip_equation_identity(
    tex: str,
    *,
    strip_numbers: bool,
    strip_labels: bool,
) -> str:
    commands: set[str] = set()
    if strip_numbers:
        commands.update({"tag", "tag*"})
    if strip_labels:
        commands.add("label")
    result: list[str] = []
    position = 0
    while position < len(tex):
        if tex[position] != "\\":
            result.append(tex[position])
            position += 1
            continue
        command_match = re.match(r"\\([A-Za-z]+\*?)\s*", tex[position:])
        if not command_match or command_match.group(1) not in commands:
            result.append(tex[position])
            position += 1
            continue
        group_start = position + command_match.end()
        if group_start >= len(tex) or tex[group_start] != "{":
            result.append(tex[position])
            position += 1
            continue
        group_end = _balanced_group_end(tex, group_start)
        if group_end is None:
            # Preserve malformed input so the final translation audit rejects it.
            result.append(tex[position])
            position += 1
            continue
        position = group_end
    return "".join(result)


def _balanced_group_end(value: str, start: int) -> int | None:
    depth = 0
    position = start
    while position < len(value):
        char = value[position]
        if char == "\\":
            position += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return position + 1
        position += 1
    return None


def _layer_marker(layer: str, boundary: str, segment_id: str) -> str:
    token = hashlib.sha256(str(segment_id).encode("utf-8")).hexdigest()[:16]
    return f"% ARC-{layer}-{boundary} {token}\n"


def _translation_equation_marker(segment_id: str, block_id_value: str) -> str:
    token = hashlib.sha256(f"{segment_id}\0{block_id_value}".encode("utf-8")).hexdigest()[:16]
    return f"% ARC-TRANSLATION-EQUATION {token}\n"


def _validate_companion_layers(tex: str, audit: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    semantic_ids = [str(value) for value in audit.get("semantic_segment_ids") or []]
    preservation_ids = {str(value) for value in audit.get("preservation_only_segment_ids") or []}
    rendered_annotations = [str(value) for value in audit.get("rendered_annotation_segment_ids") or []]
    rendered_translations = [str(value) for value in audit.get("rendered_translation_segment_ids") or []]
    provided_annotations = {str(value) for value in audit.get("provided_annotation_segment_ids") or []}
    provided_translations = {str(value) for value in audit.get("provided_translation_segment_ids") or []}
    known_ids = set(semantic_ids) | preservation_ids
    chapter_ids = [str(value) for value in audit.get("chapter_ids") or []]
    expected_guides = [
        str(value) for value in audit.get("expected_chapter_guide_ids") or []
    ]
    rendered_guides = [str(value) for value in audit.get("rendered_chapter_guide_ids") or []]
    if chapter_ids:
        if rendered_guides != expected_guides:
            errors.append("chapter guides do not match substantive chapter-guide content")
        if tex.count("% ARC-CHAPTER-GUIDE-BEGIN") != len(expected_guides):
            errors.append("chapter guide begin markers do not match substantive guides")
        if tex.count("% ARC-CHAPTER-GUIDE-END") != len(expected_guides):
            errors.append("chapter guide end markers do not match substantive guides")

    if rendered_annotations != semantic_ids:
        errors.append("companion commentary does not cover every semantic segment exactly once")
    if provided_annotations - known_ids:
        errors.append("companion commentary contains unknown segment ids")
    if audit.get("translation_mode"):
        if rendered_translations != semantic_ids:
            errors.append("translations do not cover every semantic segment exactly once")
        if provided_translations - known_ids:
            errors.append("translations contain unknown segment ids")
    else:
        if provided_translations:
            errors.append("translation-disabled manifest contains provided translation segment ids")
        if rendered_translations:
            errors.append("translation-disabled manifest contains rendered translation segment ids")
        if audit.get("translations"):
            errors.append("translation-disabled manifest contains translation audit records")
        if "ARC-TRANSLATION" in tex:
            errors.append("translation-disabled TeX contains translation markers")
        if re.search(
            r"\\(?:begin|end|newtcolorbox)\s*\{\s*arctranslation\s*\}", tex
        ):
            errors.append("translation-disabled TeX contains the arctranslation environment")

    for segment_id in semantic_ids:
        companion_region = _layer_region(tex, "COMPANION", segment_id)
        if companion_region is None:
            errors.append(f"companion layer for segment {segment_id} is not delimited exactly once")
        if not audit.get("translation_mode"):
            continue
        translation_region = _layer_region(tex, "TRANSLATION", segment_id)
        if translation_region is None:
            errors.append(f"translation layer for segment {segment_id} is not delimited exactly once")
            continue
        if re.search(r"\\(?:tag\*?|label)\s*\{", translation_region):
            errors.append(f"translation layer for segment {segment_id} contains an equation number or label")
        forbidden = (
            r"\\includegraphics",
            r"\\begin\s*\{(?:figure\*?|table\*?|longtable)\}",
            r"\\caption\*?\s*\{",
        )
        if any(re.search(pattern, translation_region) for pattern in forbidden):
            errors.append(f"translation layer for segment {segment_id} duplicates a figure or table")

    translation_records = audit.get("translations") or []
    if [str(item.get("segment_id")) for item in translation_records] != rendered_translations:
        errors.append("translation audit records do not match rendered translation segments")
    for record in translation_records:
        segment_id = str(record.get("segment_id") or "")
        for equation_block_id in record.get("equation_block_ids") or []:
            marker = _translation_equation_marker(segment_id, str(equation_block_id)).strip()
            if tex.count(marker) != 1:
                errors.append(
                    f"translated equation {equation_block_id} in segment {segment_id} is not covered exactly once"
                )
    return errors


def _layer_region(tex: str, layer: str, segment_id: str) -> str | None:
    begin = _layer_marker(layer, "BEGIN", segment_id).strip()
    end = _layer_marker(layer, "END", segment_id).strip()
    if tex.count(begin) != 1 or tex.count(end) != 1:
        return None
    start = tex.index(begin) + len(begin)
    finish = tex.index(end, start)
    return tex[start:finish]


def _render_unit_heading() -> str:
    """Separate semantic units visually without exposing controller labels."""
    return "\\par\\bigskip\\noindent\\rule{\\linewidth}{0.7pt}\\par\n"


def _box_begin(environment: str, label: str | None = None) -> str:
    heading = (
        f"\\noindent\\textbf{{{escape_tex(label)}}}\\par\n" if label else ""
    )
    return f"\\begin{{{environment}}}\n{heading}"


def _render_reading_guide(*, language: str, include_translation: bool) -> str:
    labels = _labels(language)
    if not include_translation:
        return ""
    if labels["is_chinese"]:
        text = "正文按“原文—译文—伴读”的顺序编排。译文复现公式但不重复公式编号，也不复制图表；伴读提供本段解释及有据可查的前人、后续工作。"
    else:
        text = "Each unit is ordered as Original, Translation, and Companion. Translations repeat equations without their numbers and do not duplicate figures or tables."
    return f"\\section*{{{escape_tex(labels['guide'])}}}\n{escape_tex(text)}\\par\n"


def _render_chapter_guide(guide: dict[str, Any], *, language: str) -> str:
    if not _chapter_guide_has_content(guide):
        return ""
    fields = (
        ("motivation", "学习动机"), ("main_content", "主要内容"),
        ("section_logic", "节间逻辑"),
    )
    parts = [
        "\\Needspace{4\\baselineskip}\n"
        f"\\medskip\\noindent\\textbf{{{escape_tex(title)}}}\\par\n{_render_rich_text(guide[key])}\n"
        for key, title in fields if str(guide.get(key) or "").strip()
    ]
    comparison = guide.get("pedagogical_comparison")
    if isinstance(comparison, dict) and str(comparison.get("text") or "").strip():
        parts.append(
            "\\Needspace{4\\baselineskip}\n"
            f"\\medskip\\noindent\\textbf{{教材顺序比较}}\\par\n"
            f"{_render_rich_text(comparison.get('text'))}"
            f"{_render_annotation_sources(comparison.get('sources'))}\n"
        )
    if str(guide.get("prerequisites") or "").strip():
        parts.append(
            "\\Needspace{4\\baselineskip}\n"
            f"\\medskip\\noindent\\textbf{{前置知识}}\\par\n"
            f"{_render_rich_text(guide.get('prerequisites'))}\n"
        )
    for item in guide.get("historical_context") or []:
        if not isinstance(item, dict) or not str(item.get("text") or "").strip():
            continue
        parts.append(
            "\\Needspace{4\\baselineskip}\n"
            f"\\medskip\\noindent\\textbf{{历史背景}}\\par\n"
            f"{_render_rich_text(item.get('text'))}"
            f"{_render_annotation_sources(item.get('sources'))}\n"
        )
    reading = guide.get("supplementary_reading") or []
    if reading:
        items = "\n".join(
            f"\\item {_render_rich_text(item.get('title'))}: {_render_rich_text(item.get('reason'))}"
            for item in reading if isinstance(item, dict)
        )
        if items:
            parts.append(
                "\\Needspace{4\\baselineskip}\n"
                f"\\medskip\\noindent\\textbf{{补充阅读}}\\par\n"
                f"\\begin{{itemize}}\n{items}\n\\end{{itemize}}\n"
            )
    heading = "章导读" if str(language).lower().startswith("zh") else "Chapter guide"
    return (
        "% ARC-CHAPTER-GUIDE-BEGIN\n"
        + _begin_guarded_box("arcchapterguide")
        + f"\\noindent\\textbf{{{heading}}}\\par\n"
        + "".join(parts)
        + "\\end{arcchapterguide}\n% ARC-CHAPTER-GUIDE-END\n"
    )


def _chapter_guide_has_content(guide: dict[str, Any]) -> bool:
    if any(
        str(guide.get(key) or "").strip()
        for key in ("motivation", "main_content", "section_logic", "prerequisites")
    ):
        return True
    comparison = guide.get("pedagogical_comparison")
    if isinstance(comparison, dict) and str(comparison.get("text") or "").strip():
        return True
    return any(
        isinstance(item, dict) and str(item.get("text") or "").strip()
        for item in guide.get("historical_context") or []
    ) or any(
        isinstance(item, dict)
        and str(item.get("title") or "").strip()
        and str(item.get("reason") or "").strip()
        for item in guide.get("supplementary_reading") or []
    )


def _render_glossary(
    glossary: dict[str, Any] | list[dict[str, Any]] | None,
    *,
    language: str,
) -> str:
    if isinstance(glossary, dict):
        entries = glossary.get("entries") or []
    else:
        entries = glossary or []
    if not entries:
        return ""
    labels = _labels(language)
    lines = [
        f"\\section*{{{escape_tex(labels['glossary'])}}}",
        "\\begingroup\\renewcommand{\\arraystretch}{1.25}",
        "\\begin{longtable}{>{\\raggedright\\arraybackslash}p{0.27\\linewidth}>{\\raggedright\\arraybackslash}p{0.25\\linewidth}>{\\raggedright\\arraybackslash}p{0.38\\linewidth}}",
        f"\\textbf{{{escape_tex(labels['source_term'])}}} & \\textbf{{{escape_tex(labels['target_term'])}}} & \\textbf{{{escape_tex(labels['glossary_explanation'])}}} \\\\ \\hline",
        "\\endfirsthead",
        f"\\textbf{{{escape_tex(labels['source_term'])}}} & \\textbf{{{escape_tex(labels['target_term'])}}} & \\textbf{{{escape_tex(labels['glossary_explanation'])}}} \\\\ \\hline",
        "\\endhead",
    ]
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        source = entry.get("source_term") or entry.get("english") or ""
        target = entry.get("target_term") or entry.get("chinese") or ""
        explanation = entry.get("brief_explanation") or entry.get("brief_explanation_zh") or entry.get("explanation") or ""
        lines.append(f"{escape_tex(source)} & {escape_tex(target)} & {escape_tex(explanation)} \\\\ ")
    lines.extend(["\\end{longtable}", "\\endgroup\n"])
    return "\n".join(lines)


def _labels(language: str) -> dict[str, Any]:
    chinese = str(language).lower().replace("_", "-").startswith("zh")
    if chinese:
        return {
            "is_chinese": True, "language": "阅读语言", "guide": "阅读导览", "glossary": "术语表",
            "source": "原文", "translation": "译文", "companion": "伴读", "unit": "伴读单元",
            "source_term": "英文术语", "target_term": "中文译法", "glossary_explanation": "简要解释",
            "explanation": "解释",
            "prior": "前人工作", "later": "后续工作",
        }
    return {
        "is_chinese": False, "language": "Reading language", "guide": "Reading guide", "glossary": "Glossary",
        "source": "Original", "translation": "Translation", "companion": "Companion", "unit": "Companion unit",
        "source_term": "Source term", "target_term": "Translation", "glossary_explanation": "Brief explanation",
        "explanation": "Explanation",
        "prior": "Prior work", "later": "Later work",
    }


def _inclusive_block_ids(blocks: list[dict[str, Any]], *, start: str, end: str) -> list[str]:
    ids = [block_id(item) for item in blocks]
    try:
        first = ids.index(start)
        last = ids.index(end, first)
    except ValueError:
        return []
    return ids[first:last + 1]


def _render_bibliography(
    items: list[dict[str, Any]], *, rendered_links: list[dict[str, str]],
    include_heading: bool = True,
) -> str:
    lines = []
    if include_heading:
        lines.append("\\section*{References}")
    lines.append("\\begin{thebibliography}{9999}")
    for index, item in enumerate(items, 1):
        label = str(item.get("label") or item.get("display_label") or index)
        key = _safe_label(item.get("id") or item.get("bib_id") or f"ref-{index}")
        text = _bibliography_text(item, rendered_links=rendered_links)
        lines.append(f"\\phantomsection\\label{{{key}}}\\bibitem[{{{escape_tex(label)}}}]{{{key}}} {text}")
    lines.append("\\end{thebibliography}")
    return "\n".join(lines) + "\n"


def _render_plain_reference(
    block: dict[str, Any], *, rendered_links: list[dict[str, str]]
) -> str:
    rendered = _bibliography_text(block, rendered_links=rendered_links)
    return _anchors(block) + rendered + "\n\n"


def _render_html_fragment(
    html: Any,
    *,
    rendered_links: list[dict[str, str]],
    contents_only: bool = False,
) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(strip_machine_details(str(html)), "html.parser")
    nodes: list[Any]
    first = next((item for item in soup.contents if isinstance(item, Tag)), None)
    if contents_only and isinstance(first, Tag):
        nodes = list(first.children)
    else:
        nodes = list(soup.contents)
    rendered = "".join(_render_html_node(node, rendered_links=rendered_links) for node in nodes)
    return re.sub(r"[ \t]+\n", "\n", rendered).strip() + ("\n\n" if not contents_only else "")


def _trim_outer_html_line_breaks(value: str) -> str:
    """Drop HTML ``br`` nodes that cannot carry layout at a block edge."""
    value = re.sub(r"^(?:\\\\[ \t]*(?:\n|$))+", "", value)
    value = re.sub(r"(?:\n?[ \t]*\\\\[ \t]*)+$", "", value)
    return value.strip()


def _render_html_node(
    node: Any,
    *,
    rendered_links: list[dict[str, str]],
    markerless_list_item: bool = False,
    toc_list_context: bool = False,
) -> str:
    if isinstance(node, NavigableString):
        text = re.sub(r"\s+", " ", str(node))
        # Markdown-derived rich blocks escape raw HTML containers into text
        # nodes (for example ``&lt;details&gt;``).  Apply the same reader
        # cleanup used by plain-text blocks here, while restoring the boundary
        # whitespace that ``clean_reader_text`` intentionally trims at the
        # whole-field level.  Math is handled by the parent ``math`` branch and
        # therefore never passes through this prose cleanup.
        leading_space = text.startswith(" ")
        trailing_space = text.endswith(" ")
        text = clean_reader_text(text)
        if not text:
            return ""
        if leading_space:
            text = " " + text
        if trailing_space:
            text += " "
        return escape_tex(text)
    if not isinstance(node, Tag):
        return ""
    name = str(node.name or "").lower()
    if name == "details":
        summary = node.find("summary", recursive=False)
        if isinstance(summary, Tag) and is_machine_summary_label(
            summary.get_text(" ", strip=True)
        ):
            return ""
    if name == "summary":
        if is_machine_summary_label(node.get_text(" ", strip=True)):
            return ""
        return "".join(
            _render_html_node(child, rendered_links=rendered_links)
            for child in node.children
        )
    if name in {"script", "style", "annotation", "semantics", "svg"}:
        return ""
    anchor = _anchor_for_id(node.get("id"))
    if name == "math":
        annotation = node.find("annotation", attrs={"encoding": "application/x-tex"})
        tex = annotation.get_text("", strip=True) if isinstance(annotation, Tag) else str(node.get("alttext") or "").strip()
        if not tex:
            raise LatexError("inline MathML has no TeX annotation or alttext")
        display = str(node.get("display") or "").lower() == "block"
        return anchor + (f"{{\\rmfamily\\[{tex}\\]}}" if display else f"{{\\rmfamily\\({tex}\\)}}")
    if name in {"ul", "ol"}:
        class_parts = {
            part.casefold()
            for token in node.get("class", [])
            for part in re.split(r"[-_]", str(token))
        }
        is_toc_list = toc_list_context or any(part.startswith("toc") for part in class_parts)
        markerless = name == "ol" and is_toc_list and _ordered_list_has_structural_labels(node)
        environment = "description" if markerless else ("enumerate" if name == "ol" else "itemize")
        items = "".join(
            _render_html_node(
                child,
                rendered_links=rendered_links,
                markerless_list_item=markerless,
                toc_list_context=is_toc_list,
            )
            for child in node.children
            if isinstance(child, Tag) and child.name == "li"
        )
        return anchor + f"\\begin{{{environment}}}\n{items}\\end{{{environment}}}\n"
    children = "".join(
        _render_html_node(
            child,
            rendered_links=rendered_links,
            toc_list_context=toc_list_context,
        )
        for child in node.children
    )
    if name == "a":
        href = str(node.get("href") or "").strip()
        if not href:
            return anchor + children
        record = {
            "href": href,
            "target_id": href[1:] if href.startswith("#") else "",
            "text": " ".join(node.get_text(" ", strip=True).split()),
        }
        rendered_links.append(record)
        visible = children or escape_tex(href)
        if href.startswith("#"):
            return anchor + f"\\hyperref[{_safe_label(href[1:])}]{{{visible}}}"
        return anchor + f"\\href{{{_escape_url(href)}}}{{{visible}}}"
    if name in {"strong", "b"}:
        return anchor + f"\\textbf{{{children}}}"
    if name in {"em", "i"}:
        return anchor + f"\\emph{{{children}}}"
    if name in {"code", "tt", "kbd", "samp"}:
        return anchor + f"\\texttt{{{children}}}"
    if name == "sup":
        return anchor + f"\\textsuperscript{{{children}}}"
    if name == "sub":
        return anchor + f"\\textsubscript{{{children}}}"
    if name == "br":
        return anchor + "\\\\\n"
    if name == "blockquote":
        return anchor + f"\\begin{{quote}}\n{children}\n\\end{{quote}}\n"
    if name == "pre":
        return anchor + f"\\begin{{quote}}\\ttfamily {children}\\end{{quote}}\n"
    if name == "li":
        item = "\\item[]" if markerless_list_item else "\\item"
        return anchor + f"{item} {children}\n"
    if name == "p":
        return anchor + _trim_outer_html_line_breaks(children) + "\n\n"
    if name in {"img", "source", "object"}:
        return anchor
    return anchor + children


_NUMERIC_STRUCTURAL_LIST_LABEL_RE = re.compile(r"^\d+(?:\.\d+)*\.?(?=\s+\S)")
_ALPHA_STRUCTURAL_LIST_LABEL_RE = re.compile(
    r"^(?:[A-Z](?:\.\d+)*|[IVXLCDM]+)\.?(?=\s+\S)"
)


def _ordered_list_has_structural_labels(node: Tag) -> bool:
    """Return whether every direct item carries its own section-like label."""
    labels: list[str] = []
    for item in node.find_all("li", recursive=False):
        parts: list[str] = []
        for child in item.children:
            if isinstance(child, Tag) and str(child.name).lower() in {"ol", "ul"}:
                break
            if isinstance(child, NavigableString):
                parts.append(str(child))
            elif isinstance(child, Tag):
                parts.append(child.get_text(" ", strip=True))
        labels.append(" ".join(" ".join(parts).split()))
    return bool(labels) and all(
        _NUMERIC_STRUCTURAL_LIST_LABEL_RE.match(label)
        or _ALPHA_STRUCTURAL_LIST_LABEL_RE.match(label)
        for label in labels
    )


def _bibliography_text(item: dict[str, Any], *, rendered_links: list[dict[str, str]]) -> str:
    html = item.get("html")
    if html:
        soup = BeautifulSoup(str(html), "html.parser")
        for tag in soup.select(".ltx_tag_bibitem"):
            tag.decompose()
        root = next((value for value in soup.contents if isinstance(value, Tag)), None)
        if isinstance(root, Tag):
            rendered = "".join(
                _render_html_node(child, rendered_links=rendered_links)
                for child in root.children
            ).strip()
            if rendered:
                return rendered
    label = str(item.get("label") or item.get("display_label") or "")
    text = str(item.get("text") or item.get("citation") or "")
    if label and text.startswith(label):
        text = text[len(label):].lstrip()
    return escape_tex(text)


def _entity_caption(entity: dict[str, Any], *, rendered_links: list[dict[str, str]]) -> str:
    html = entity.get("html")
    if html:
        soup = BeautifulSoup(str(html), "html.parser")
        caption = soup.select_one("figcaption, .ltx_caption")
        if isinstance(caption, Tag):
            for tag in caption.select(".ltx_tag_figure, .ltx_tag_table"):
                tag.decompose()
            rendered = "".join(
                _render_html_node(child, rendered_links=rendered_links)
                for child in caption.children
            ).strip()
            if rendered:
                return rendered
    caption = str(entity.get("caption") or "")
    tag = str(entity.get("number") or entity.get("display_number") or entity.get("tag") or "")
    if tag and caption.startswith(tag):
        caption = caption[len(tag):].lstrip()
    return escape_tex(caption)


def _front_matter_block_roles(
    blocks: list[dict[str, Any]], front: dict[str, Any]
) -> dict[str, str]:
    structural_role_map = {
        "front_matter_title": "title",
        "front_matter_authors": "author",
        "front_matter_affiliations": "affiliation",
        "front_matter_profiles": "profile",
        "front_matter_abstract": "abstract",
    }
    roles = {
        block_id(block): structural_role_map[str(block.get("source_role") or "").casefold()]
        for block in blocks
        if str(block.get("source_role") or "").casefold() in structural_role_map
    }
    for block in blocks:
        front_roles = {
            structural_role_map[value]
            for value in block.get("front_matter_roles") or []
            if value in structural_role_map
        }
        if "title" in front_roles:
            roles[block_id(block)] = "title"
        elif "author" in front_roles:
            # A combined author/affiliation source line is replaced by the
            # exact extracted author on the title page and affiliation below.
            roles[block_id(block)] = "author"
        elif "affiliation" in front_roles:
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
        if block_id(block) in roles:
            continue
        if block.get("section_id"):
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


def _anchors(item: dict[str, Any]) -> str:
    values = [item.get("block_id"), item.get("source_id"), item.get("section_id"), item.get("id")]
    unique = list(dict.fromkeys(str(value) for value in values if value))
    return "".join(_anchor_for_id(value) for value in unique)


def _entity_anchors(item: dict[str, Any]) -> str:
    values = [item.get("id"), item.get("equation_id"), item.get("figure_id"), item.get("table_id")]
    return "".join(_anchor_for_id(value) for value in dict.fromkeys(str(value) for value in values if value))


def _anchor_for_id(value: Any) -> str:
    return f"\\phantomsection\\label{{{_safe_label(value)}}}" if value else ""


def _source_block_marker(block_identifier: str, source_hash: str) -> str:
    identifier_hash = hashlib.sha256(block_identifier.encode("utf-8")).hexdigest()
    return f"% ARC-SOURCE-BLOCK {identifier_hash} {source_hash}\n"


def _table_source_shape(entity: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(entity.get("id") or entity.get("table_id") or "table"),
        "tag": entity.get("tag") or entity.get("number") or entity.get("display_number") or "",
        "caption": entity.get("caption") or "",
        "column_count": entity.get("column_count"),
        "rows": entity.get("rows") or [],
        "grid": entity.get("grid") or [],
    }


def _table_audit_record(entity: dict[str, Any]) -> dict[str, str]:
    shape = _table_source_shape(entity)
    return {"id": str(shape["id"]), "sha256": sha256_json(shape)}


def _table_marker(table_id: str, table_hash: str) -> str:
    identifier_hash = hashlib.sha256(table_id.encode("utf-8")).hexdigest()
    return f"% ARC-TABLE {identifier_hash} {table_hash}\n"


def _link_records(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "href": str(item.get("href") or ""),
            "target_id": str(item.get("target_id") or ""),
            "text": " ".join(str(item.get("text") or "").split()),
        }
        for item in items
        if item.get("href")
    ]


def _consume_matching_link(expected: dict[str, str], rendered: list[dict[str, str]]) -> bool:
    for index, item in enumerate(rendered):
        if all(str(item.get(key) or "") == str(expected.get(key) or "") for key in ("href", "target_id", "text")):
            rendered.pop(index)
            return True
    return False


def _escape_url(value: str) -> str:
    replacements = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "%": r"\%", "#": r"\#", "&": r"\&", "_": r"\_"}
    return "".join(replacements.get(char, char) for char in value)


def _index_entities(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for item in items:
        for key in ("id", "block_id", "equation_id", "figure_id", "table_id", "asset_id"):
            if item.get(key):
                output[str(item[key])] = item
    return output


def _entity_for(block: dict[str, Any], entities: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    for key in ("entity_id", "equation_id", "figure_id", "table_id", "id", "block_id"):
        if block.get(key) is not None and str(block[key]) in entities:
            return entities[str(block[key])]
    return None


def _kind(block: dict[str, Any]) -> str:
    return str(block.get("type") or block.get("kind") or "text").lower()


def _row_width(row: Any) -> int:
    cells = row.get("cells") if isinstance(row, dict) else row
    if not isinstance(cells, list):
        return 0
    return sum(int(cell.get("colspan") or 1) if isinstance(cell, dict) else 1 for cell in cells)


def _safe_label(value: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9:._-]+", "-", str(value or "item"))
    return cleaned or hashlib.sha256(str(value).encode()).hexdigest()[:12]


def _clean_tag(value: Any) -> str:
    text = str(value or "").strip()
    if text.startswith("(") and text.endswith(")"):
        return text[1:-1].strip()
    return text


def _equation_numbers(items: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for item in items:
        value = item.get("printed_equation_numbers")
        if value is None:
            value = item.get("printed_equation_number")
        if value is None:
            value = item.get("number") or item.get("equation_number") or item.get("display_number") or item.get("tag")
        candidates = value if isinstance(value, list) else [value]
        values.extend(str(candidate) for candidate in candidates if candidate not in {None, ""})
    return values


def _author_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("name") or value.get("full_name") or "")
    return str(value)


def _front_value(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("text", "name", "value", "title"):
            if value.get(key):
                return str(value[key])
        return "; ".join(str(item) for item in value.values() if item is not None and item != "")
    return str(value)


def _visible_tag(value: Any, *, kind: str) -> str:
    if value in {None, ""}:
        return ""
    text = str(value).strip()
    if text.lower().startswith(kind.lower()):
        return escape_tex(text) + (" " if text.endswith((":", ".")) else ". ")
    return f"{kind} {escape_tex(text)}. "


def _media_extension(value: Any) -> str:
    return {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/pdf": ".pdf",
        "application/pdf": ".pdf",
        "image/svg+xml": ".svg",
        "application/postscript": ".eps",
    }.get(str(value).split(";", 1)[0].lower(), ".bin")


def _materialize_latex_asset(source: Path, *, source_hash: str, extension: str, output_dir: Path) -> Path:
    asset_dir = output_dir / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    if extension == ".png" and _png_needs_latex_flattening(source):
        destination = asset_dir / f"{source_hash}-latex.png"
        if destination.exists() and destination.stat().st_size:
            return destination
        executable = shutil.which("magick") or shutil.which("convert")
        if executable is None:
            raise LatexError("16-bit PNG with alpha requires ImageMagick for reliable XeLaTeX rendering")
        if Path(executable).name == "magick":
            command = [executable, str(source), "-background", "white", "-alpha", "remove", "-alpha", "off", "-depth", "8", str(destination)]
        else:
            command = [executable, str(source), "-background", "white", "-alpha", "remove", "-alpha", "off", "-depth", "8", str(destination)]
        completed = subprocess.run(command, text=True, capture_output=True, timeout=120, check=False)
        if completed.returncode != 0 or not destination.is_file() or destination.stat().st_size == 0:
            raise LatexError(f"PNG conversion failed for {source}: {completed.stderr.strip()}")
        return destination
    if extension in {".pdf", ".png", ".jpg", ".jpeg"}:
        destination = asset_dir / f"{source_hash}{extension}"
        if not destination.exists():
            shutil.copy2(source, destination)
        return destination
    destination = asset_dir / f"{source_hash}.pdf"
    if destination.exists() and destination.stat().st_size:
        return destination
    if extension == ".svg":
        if executable := shutil.which("rsvg-convert"):
            command = [executable, "-f", "pdf", "-o", str(destination), str(source)]
        elif executable := shutil.which("inkscape"):
            command = [executable, str(source), "--export-type=pdf", f"--export-filename={destination}"]
        else:
            raise LatexError("SVG figure requires rsvg-convert or inkscape")
    elif extension in {".eps", ".ps"}:
        executable = shutil.which("epstopdf")
        if executable is None:
            raise LatexError("EPS/PS figure requires epstopdf")
        command = [executable, str(source), f"--outfile={destination}"]
    else:
        raise LatexError(f"unsupported cached figure format: {source}")
    completed = subprocess.run(command, text=True, capture_output=True, timeout=120, check=False)
    if completed.returncode != 0 or not destination.is_file() or destination.stat().st_size == 0:
        raise LatexError(f"figure conversion failed for {source}: {completed.stderr.strip()}")
    return destination


def _png_needs_latex_flattening(source: Path) -> bool:
    """Detect 16-bit alpha PNGs whose masks xdvipdfmx may render fully transparent."""
    try:
        header = source.read_bytes()[:26]
    except OSError:
        return False
    if len(header) < 26 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        return False
    bit_depth = header[24]
    color_type = header[25]
    return bit_depth == 16 and color_type in {4, 6}
