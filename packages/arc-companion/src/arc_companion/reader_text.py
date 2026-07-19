from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Iterable

_HTML_TAG = re.compile(
    r"</?(?:details|summary|div|span|p|br|hr|em|strong|b|i|u|small|sub|sup|"
    r"blockquote|pre|code|ul|ol|li|dl|dt|dd|table|thead|tbody|tfoot|tr|th|td|"
    r"figure|figcaption|section|article|header|footer|nav|aside|a|script|style|"
    r"annotation|semantics|svg)(?:\s[^>\n]*)?/?>",
    flags=re.IGNORECASE,
)
_BRACKETED_REFERENCE = re.compile(r"(?P<open>\[|【)(?P<body>[^\]】]+)(?P<close>\]|】)")
_MACHINE_SUMMARY_LABEL = re.compile(
    r"^(?:natural[\s_-]*image|ocr(?:[\s_-]*(?:metadata|text))?|image[\s_-]*metadata)$",
    flags=re.IGNORECASE,
)


def clean_reader_text(
    value: Any,
    *,
    evidence_ids: Iterable[str] = (),
    evidence_records: Iterable[dict[str, Any]] = (),
    language: str = "",
) -> str:
    """Remove controller-facing markup from prose shown to a reader.

    The immutable source document and structured evidence bindings remain
    untouched.  This function only normalizes a presentation copy.
    """
    text = str(value or "")
    text = _strip_html_containers(text)
    text = _replace_evidence_ids(
        text,
        evidence_ids=evidence_ids,
        evidence_records=evidence_records,
        language=language,
    )
    return re.sub(r"[ \t]+(?=\n)", "", text).strip()


def clean_reader_annotation(
    annotation: dict[str, Any],
    *,
    evidence_records: Iterable[dict[str, Any]] = (),
    language: str = "",
) -> dict[str, Any]:
    """Return a reader-clean annotation while retaining evidence metadata."""
    cleaned = deepcopy(annotation)
    evidence_records = list(evidence_records)
    citation_labels = _citation_labels(evidence_records, language=language)
    global_ids = _string_ids(cleaned.get("evidence_ids"))
    for field in ("commentary", "explanation", "prior_work", "later_work"):
        value = cleaned.get(field)
        if isinstance(value, list):
            normalized: list[Any] = []
            for item in value:
                if isinstance(item, dict):
                    entry = dict(item)
                    ids = [*global_ids, *_string_ids(entry.get("evidence_ids"))]
                    claim_ids = _string_ids(entry.get("evidence_ids"))
                    for text_field in ("text", "summary", "claim", "title"):
                        if text_field in entry:
                            rendered = clean_reader_text(
                                entry[text_field], evidence_ids=ids,
                                evidence_records=evidence_records, language=language,
                            )
                            if claim_ids and not any(
                                label in rendered
                                for label in citation_labels.values()
                            ):
                                rendered += _readable_citation(
                                    claim_ids,
                                    citation_labels,
                                    language=language,
                                )
                            entry[text_field] = rendered
                    normalized.append(entry)
                else:
                    normalized.append(clean_reader_text(
                        item, evidence_ids=global_ids,
                        evidence_records=evidence_records, language=language,
                    ))
            cleaned[field] = normalized
        elif value is not None:
            cleaned[field] = clean_reader_text(
                value, evidence_ids=global_ids,
                evidence_records=evidence_records, language=language,
            )
    return cleaned


def clean_reader_translation(translation: dict[str, Any]) -> dict[str, Any]:
    """Return a presentation copy of translated blocks without HTML wrappers."""
    cleaned = deepcopy(translation)
    for block in cleaned.get("blocks") or []:
        if not isinstance(block, dict):
            continue
        for field in ("text", "translated_text", "translation"):
            if field in block and block[field] is not None:
                block[field] = clean_reader_text(block[field])
    return cleaned


def _strip_html_containers(text: str) -> str:
    if not _HTML_TAG.search(text):
        return text
    rendered = text
    # These nodes carry machine/controller metadata rather than reader prose.
    # Container bodies remain available after their wrappers are removed.
    rendered = re.sub(
        r"<summary\b[^>]*>(?P<body>.*?)</summary\s*>",
        lambda match: "" if is_machine_summary_label(match.group("body")) else match.group("body"),
        rendered,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for name in ("script", "style", "annotation", "semantics", "svg"):
        rendered = re.sub(
            rf"<{name}\b[^>]*>.*?</{name}\s*>",
            "",
            rendered,
            flags=re.IGNORECASE | re.DOTALL,
        )
    rendered = re.sub(
        r"<\s*(?:br|hr)\b[^>]*/?>|</\s*(?:p|div|li|tr|section|article|"
        r"blockquote|pre|figure|figcaption)\s*>",
        "\n",
        rendered,
        flags=re.IGNORECASE,
    )
    rendered = _HTML_TAG.sub("", rendered)
    rendered = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", rendered)
    return rendered


def is_machine_summary_label(value: Any) -> bool:
    """Recognize conservative controller/media labels, not ordinary headings."""
    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    text = " ".join(text.split())
    return bool(text and _MACHINE_SUMMARY_LABEL.fullmatch(text))


def _replace_evidence_ids(
    text: str,
    *,
    evidence_ids: Iterable[str],
    evidence_records: Iterable[dict[str, Any]],
    language: str,
) -> str:
    known = set(_string_ids(evidence_ids))
    if not known:
        return text
    known_by_fold = {value.casefold(): value for value in known}
    labels = _citation_labels(evidence_records, language=language)

    id_patterns = {
        value: _soft_wrapped_id_pattern(value)
        for value in sorted(known, key=len, reverse=True)
    }
    id_pattern = "|".join(id_patterns.values())
    if id_pattern:
        evidence_note = re.compile(
            rf"(?:（|\()\s*(?:证据|evidence)\s*[:：]\s*"
            rf"(?P<ids>(?:{id_pattern})(?:\s*[,，;；]\s*(?:{id_pattern}))*)\s*(?:）|\))",
            flags=re.IGNORECASE,
        )

        def replace_note(match: re.Match[str]) -> str:
            tokens = [
                known_by_fold.get(
                    _normalize_soft_wrapped_id(part).casefold(),
                    _normalize_soft_wrapped_id(part),
                )
                for part in re.split(r"\s*[,，;；]\s*", match.group("ids"))
                if part
            ]
            return _readable_citation(tokens, labels, language=language)

        text = evidence_note.sub(replace_note, text)

        bare_evidence_note = re.compile(
            rf"(?:证据|evidence)\s*[:：]\s*"
            rf"(?P<ids>(?:{id_pattern})(?:\s*[,，;；]\s*(?:{id_pattern}))*)",
            flags=re.IGNORECASE,
        )
        text = bare_evidence_note.sub(replace_note, text)

    def replace(match: re.Match[str]) -> str:
        body = match.group("body")
        tokens = [
            known_by_fold.get(
                _normalize_soft_wrapped_id(part).casefold(),
                _normalize_soft_wrapped_id(part),
            )
            for part in re.split(r"\s*[,，;；]\s*|[ \t]+", body.strip())
            if part
        ]
        if tokens and all(token in known for token in tokens):
            return _readable_citation(tokens, labels, language=language)
        return match.group(0)

    cleaned = _BRACKETED_REFERENCE.sub(replace, text)
    # Final deterministic backstop: a model can emit a registered controller
    # ID without brackets, or a formatter can soft-wrap it immediately after a
    # hyphen.  Replace only IDs present in this annotation's structured
    # evidence list; ID-like prose that is not registered remains untouched.
    for evidence_id, pattern in id_patterns.items():
        cleaned = re.sub(
            pattern,
            lambda _match, value=evidence_id: _readable_citation(
                [value], labels, language=language
            ),
            cleaned,
            flags=re.IGNORECASE,
        )
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    if _is_chinese(language):
        cleaned = re.sub(r"[ \t]+(?=（)", "", cleaned)
    cleaned = re.sub(r"\s+([。！？；，,.!?;:])", r"\1", cleaned)
    return cleaned


def _soft_wrapped_id_pattern(evidence_id: str) -> str:
    """Match one registered ID with an optional soft line break after ``-``."""
    return re.escape(evidence_id).replace(r"\-", r"\-(?:[ \t]*\n[ \t]*)?")


def _normalize_soft_wrapped_id(value: str) -> str:
    return re.sub(r"(?<=-)[ \t]*\n[ \t]*", "", value)


def _readable_citation(
    evidence_ids: Iterable[str], labels: dict[str, str], *, language: str
) -> str:
    visible = [labels[value] for value in evidence_ids if labels.get(value)]
    if not visible:
        return ""
    joined = "；".join(dict.fromkeys(visible))
    return f"（{joined}）" if _is_chinese(language) else f" ({joined})"


def _citation_labels(
    records: Iterable[dict[str, Any]], *, language: str
) -> dict[str, str]:
    labels: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        evidence_id = str(record.get("evidence_id") or "")
        if not evidence_id:
            continue
        descriptor = record.get("source_descriptor") or {}
        title = clean_reader_text(
            record.get("title") or descriptor.get("title") or ""
        ).strip()
        if not title:
            continue
        section = _evidence_section_title(record)
        if _is_chinese(language):
            label = f"《{title}》"
            if section:
                label += f"，{section}"
        else:
            label = title
            if section:
                label += f", {section}"
        labels[evidence_id] = label
    return labels


def _evidence_section_title(record: dict[str, Any]) -> str:
    for owner in (
        record,
        *(record.get("selected_snippets") or []),
        *(record.get("snippets") or []),
    ):
        if not isinstance(owner, dict):
            continue
        value = owner.get("section_title") or owner.get("section")
        if value:
            return _short_label(value)

    for snippet in record.get("selected_snippets") or record.get("snippets") or []:
        if not isinstance(snippet, dict):
            continue
        inferred = _conservative_section_title(snippet.get("text") or "")
        if inferred:
            return inferred

    blocks = [item for item in record.get("blocks") or [] if isinstance(item, dict)]
    selected = [
        str(item.get("block_id") or "")
        for item in record.get("selected_snippets") or record.get("snippets") or []
        if isinstance(item, dict)
    ]
    positions = {
        str(item.get("block_id") or ""): index for index, item in enumerate(blocks)
    }
    selected_positions = [positions[value] for value in selected if value in positions]
    if selected_positions:
        for index in range(min(selected_positions), -1, -1):
            block = blocks[index]
            locator = str(block.get("block_id") or "").casefold()
            if "heading" in locator or locator.endswith(".title"):
                return _short_label(block.get("text") or "")
    return ""


def _conservative_section_title(value: Any) -> str:
    """Recover an explicit heading prefix without guessing prose boundaries."""
    text = " ".join(clean_reader_text(value).split())
    if not text:
        return ""
    if " ## " in text:
        prefix = text.split(" ## ", 1)[0].strip()
        if 1 < len(prefix) <= 140:
            return prefix
    punctuated = re.match(
        r"^((?:chapter\s+)?\d+(?:\.\d+)*\s+.{2,100}?[?!])(?:\s|$)",
        text,
        flags=re.IGNORECASE,
    )
    if punctuated:
        return punctuated.group(1).strip()
    page_delimited = re.match(
        r"^((?:(?i:chapter)\s+)?\d+(?:\.\d+)*\s+.{2,100}?)\s+"
        r"(?:[ivxlcdmIVXLCDM]+|\d+)\s+(?=[A-Z\u0391-\u03A9□])",
        text,
    )
    if page_delimited:
        return page_delimited.group(1).strip()
    return ""


def _short_label(value: Any) -> str:
    text = " ".join(clean_reader_text(value).split())
    return text if len(text) <= 140 else text[:137].rstrip() + "…"


def _is_chinese(language: str) -> bool:
    return str(language).lower().replace("_", "-").startswith("zh")


def _string_ids(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple, set)):
        return []
    return [str(value) for value in values if isinstance(value, str) and value]
