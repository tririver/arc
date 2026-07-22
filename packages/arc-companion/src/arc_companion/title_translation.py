from __future__ import annotations

from collections import Counter
import json
import re
from typing import Any, Iterable, Mapping, Sequence

from .projection import STRUCTURAL_TYPES, opaque_inline_tokens, translation_input_block
from .source import block_id


TITLE_TRANSLATION_VERSION = "arc.companion.title-translation.v1"
TITLE_TRANSLATION_PROMPT_VERSION = "arc.companion.title-translation-prompt.v1"
TITLE_TRANSLATION_MAX_BYTES = 48 * 1024

TITLE_TRANSLATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["titles"],
    "properties": {
        "titles": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["title_id", "text"],
                "properties": {
                    "title_id": {"type": "string", "minLength": 1},
                    "text": {"type": "string", "minLength": 1},
                },
            },
        }
    },
}


class TitleTranslationError(ValueError):
    """A title projection or translated-title response is not trustworthy."""


_OPAQUE_TOKEN_RE = re.compile(r"\[\[ARC_INLINE:[^\]\r\n]+\]\]")
_NUMBER_PREFIX_RE = re.compile(
    r"^(?P<prefix>\s*(?:"
    r"(?:\d+(?:\.\d+)*|[IVXLCDM]+)(?:[.)、:]|\s+)\s*"
    r"|\((?:\d+(?:\.\d+)*|[IVXLCDM]+)\)\s*"
    r"|\[(?:\d+(?:\.\d+)*|[IVXLCDM]+)\]\s*"
    r"))",
)


def collect_title_records(
    document: Mapping[str, Any],
    chapters: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Collect every visible document and structural title in stable source order.

    The document title owns any rich front-matter title blocks, so those blocks
    are not projected a second time. Chapter titles with no trustworthy source
    block receive a stable chapter fallback identity.
    """
    blocks = [
        dict(item)
        for item in document.get("blocks") or []
        if isinstance(item, Mapping)
    ]
    ids = [block_id(item) for item in blocks]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise TitleTranslationError("title projection requires unique non-empty block ids")
    by_id = dict(zip(ids, blocks))
    positions = {value: index for index, value in enumerate(ids)}
    chapter_list = _chapter_list(chapters)
    chapter_by_block: dict[str, str] = {}
    for chapter in chapter_list:
        chapter_id = str(chapter.get("chapter_id") or "").strip()
        if not chapter_id:
            raise TitleTranslationError("chapter title projection requires a chapter id")
        for value in chapter.get("block_ids") or []:
            candidate = str(value)
            if candidate:
                chapter_by_block.setdefault(candidate, chapter_id)

    records: list[tuple[float, dict[str, Any]]] = []
    document_record, owned_title_ids = _document_title_record(document, blocks)
    if document_record is not None:
        records.append((-2.0, document_record))

    for index, block in enumerate(blocks):
        identifier = block_id(block)
        if identifier in owned_title_ids or not _is_structural_title(block):
            continue
        source_text = _projected_text(block)
        if not source_text.strip():
            continue
        record = _record(
            title_id=f"block:{identifier}",
            source_text=source_text,
            role=_title_role(block),
            block_id_value=identifier,
            chapter_id=chapter_by_block.get(identifier),
            source_block_ids=[identifier],
            opaque_tokens=opaque_inline_tokens(block),
        )
        records.append((float(index), record))

    represented_block_ids = {
        str(record.get("block_id") or "") for _, record in records
    }
    represented_titles = {
        (str(record.get("chapter_id") or ""), _comparable_title(record["source_text"]))
        for _, record in records
        if record.get("chapter_id")
    }
    for chapter_index, chapter in enumerate(chapter_list):
        chapter_id = str(chapter.get("chapter_id") or "").strip()
        title = str(chapter.get("title") or "").strip()
        if not title:
            continue
        explicit_ids = [
            str(value) for value in chapter.get("title_block_ids") or [] if str(value)
        ]
        if any(value in represented_block_ids for value in explicit_ids):
            continue
        if (chapter_id, _comparable_title(title)) in represented_titles:
            continue
        chapter_ids = [str(value) for value in chapter.get("block_ids") or []]
        position = min(
            (positions[value] for value in chapter_ids if value in positions),
            default=len(blocks) + chapter_index,
        )
        records.append((position - 0.25, _record(
            title_id=f"chapter:{chapter_id}",
            source_text=title,
            role="chapter",
            block_id_value=None,
            chapter_id=chapter_id,
            source_block_ids=[],
            opaque_tokens=[],
        )))

    records.sort(key=lambda pair: pair[0])
    output = [record for _, record in records]
    title_ids = [record["title_id"] for record in output]
    if len(title_ids) != len(set(title_ids)):
        raise TitleTranslationError("title projection produced duplicate stable ids")
    return output


def chunk_title_records(
    records: Sequence[Mapping[str, Any]],
    *,
    max_bytes: int = TITLE_TRANSLATION_MAX_BYTES,
) -> list[list[dict[str, Any]]]:
    """Split title records without reordering or splitting an individual title."""
    if max_bytes < 1:
        raise TitleTranslationError("title chunk byte limit must be positive")
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for raw in records:
        record = dict(raw)
        candidate = [*current, record]
        if _transport_size(candidate) <= max_bytes:
            current = candidate
            continue
        if not current:
            raise TitleTranslationError(
                f"title {record.get('title_id')!r} exceeds the bounded prompt size"
            )
        chunks.append(current)
        current = [record]
        if _transport_size(current) > max_bytes:
            raise TitleTranslationError(
                f"title {record.get('title_id')!r} exceeds the bounded prompt size"
            )
    if current:
        chunks.append(current)
    return chunks


def title_translation_prompt(
    records: Sequence[Mapping[str, Any]],
    *,
    source_language: str,
    target_language: str,
    glossary: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    protected_names: Sequence[str] = (),
) -> str:
    """Build a deterministic, source-language-neutral title-only prompt."""
    payload = {
        "prompt_version": TITLE_TRANSLATION_PROMPT_VERSION,
        "source_language": str(source_language or "und"),
        "target_language": str(target_language or "und"),
        "titles": [_prompt_record(record) for record in records],
        "glossary": _glossary_entries(glossary),
        "protected_names": _unique_strings(protected_names),
    }
    instructions = (
        "Translate every supplied document or structural title into the target language. "
        "Return exactly one titles item for every title_id, in the supplied order, and no "
        "other content. This is title translation only: do not add explanations, annotations, "
        "commentary, summaries, or reading guidance. Preserve any leading number_prefix "
        "verbatim. Copy every opaque_token exactly once and unchanged. Preserve the exact "
        "source spelling of each protected personal name that occurs in a title. Apply the "
        "supplied glossary consistently. Translate structural words such as Part, Chapter, "
        "References, and Index; do not translate the protected numbering itself.\n"
    )
    return instructions + json.dumps(payload, ensure_ascii=False, sort_keys=True)


def normalize_title_translation_response(value: Any) -> dict[str, list[dict[str, str]]]:
    """Normalize the small set of response aliases used by host model adapters."""
    if isinstance(value, Mapping):
        candidates = value.get("titles")
        if candidates is None:
            candidates = value.get("translations")
    else:
        candidates = value
    if not isinstance(candidates, list):
        raise TitleTranslationError("title translation response must contain a titles array")
    normalized: list[dict[str, str]] = []
    for item in candidates:
        if not isinstance(item, Mapping):
            raise TitleTranslationError("each translated title must be an object")
        title_id = item.get("title_id", item.get("id"))
        text = item.get("text", item.get("translated_title", item.get("translation")))
        if not isinstance(title_id, str) or not isinstance(text, str):
            raise TitleTranslationError("each translated title requires string title_id and text")
        normalized.append({"title_id": title_id.strip(), "text": text.strip()})
    return {"titles": normalized}


def validate_title_translations(
    records: Sequence[Mapping[str, Any]],
    response: Any,
    *,
    protected_names: Sequence[str] = (),
) -> dict[str, list[dict[str, str]]]:
    """Validate title identity and restore only an entirely omitted source prefix."""
    normalized = normalize_title_translation_response(response)
    titles = normalized["titles"]
    expected_ids = [str(record.get("title_id") or "") for record in records]
    actual_ids = [item["title_id"] for item in titles]
    if actual_ids != expected_ids:
        raise TitleTranslationError(
            "translated titles must cover every projected title exactly once in source order"
        )
    if len(actual_ids) != len(set(actual_ids)):
        raise TitleTranslationError("translated title response contains duplicate ids")

    names = _unique_strings(protected_names)
    for record, translated in zip(records, titles):
        source_text = str(record.get("source_text") or "")
        target_text = translated["text"]
        if not source_text.strip() or not target_text:
            raise TitleTranslationError("source and translated titles must be non-empty")
        expected_prefix = str(record.get("number_prefix") or _number_prefix(source_text))
        if expected_prefix and not target_text.startswith(expected_prefix):
            actual_prefix = _number_prefix(target_text)
            if actual_prefix:
                raise TitleTranslationError(
                    f"translated title {translated['title_id']!r} changed its number prefix"
                )
            target_text = expected_prefix + target_text
            translated["text"] = target_text
        expected_tokens = Counter(
            str(value) for value in record.get("opaque_tokens") or _opaque_tokens(source_text)
        )
        actual_tokens = Counter(_opaque_tokens(target_text))
        if actual_tokens != expected_tokens:
            raise TitleTranslationError(
                f"translated title {translated['title_id']!r} changed opaque inline tokens"
            )
        missing_names = [
            name for name in names
            if _contains_protected_name(source_text, name)
            and not _contains_protected_name(target_text, name)
        ]
        if missing_names:
            raise TitleTranslationError(
                f"translated title {translated['title_id']!r} changed protected names: "
                f"{missing_names}"
            )
    return normalized


def merge_title_translation_chunks(
    records: Sequence[Mapping[str, Any]],
    responses: Iterable[Any],
    *,
    protected_names: Sequence[str] = (),
) -> dict[str, list[dict[str, str]]]:
    """Merge independently returned chunks and apply the full-document validator."""
    merged: list[dict[str, str]] = []
    for response in responses:
        merged.extend(normalize_title_translation_response(response)["titles"])
    return validate_title_translations(
        records, {"titles": merged}, protected_names=protected_names
    )


def _document_title_record(
    document: Mapping[str, Any], blocks: Sequence[dict[str, Any]]
) -> tuple[dict[str, Any] | None, set[str]]:
    front = document.get("front_matter")
    front = front if isinstance(front, Mapping) else {}
    configured = front.get("block_ids")
    configured = configured if isinstance(configured, Mapping) else {}
    explicit = configured.get("title", configured.get("titles", []))
    if isinstance(explicit, str):
        explicit = [explicit]
    explicit_ids = {str(value) for value in explicit or [] if str(value)}
    role_ids = {
        block_id(block)
        for block in blocks
        if _has_front_title_role(block)
    }
    owned_ids = explicit_ids | role_ids
    owned_blocks = [block for block in blocks if block_id(block) in owned_ids]

    # A front-matter role identifies ownership, not necessarily the precise
    # text projection.  In particular, older parser outputs could propagate a
    # title role across every block in the title heading's section.  Prefer
    # structural title blocks when they exist; for prose-only front matter,
    # use the canonical title to select the matching block(s).  Keep all owned
    # ids reserved so contaminated prose is not projected again as a heading.
    front_title = str(front.get("title") or "").strip()
    projected_blocks = _document_title_blocks(owned_blocks, front_title=front_title)

    source_text = ""
    opaque_tokens: list[str] = []
    if projected_blocks:
        source_text = "\n".join(
            value
            for value in (_projected_text(block).strip() for block in projected_blocks)
            if value
        )
        opaque_tokens = [
            token for block in projected_blocks for token in opaque_inline_tokens(block)
        ]
    if not source_text:
        metadata = document.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        source_text = str(
            front.get("title")
            or metadata.get("title")
            or document.get("title")
            or ""
        ).strip()
    if not source_text:
        return None, owned_ids
    return _record(
        title_id="document:title",
        source_text=source_text,
        role="document",
        block_id_value=None,
        chapter_id=None,
        source_block_ids=[block_id(block) for block in projected_blocks],
        opaque_tokens=opaque_tokens,
    ), owned_ids


def _document_title_blocks(
    owned_blocks: Sequence[dict[str, Any]], *, front_title: str
) -> list[dict[str, Any]]:
    structural = [block for block in owned_blocks if _is_structural_title(block)]
    if structural:
        return structural
    if not front_title:
        return list(owned_blocks)

    comparable = _comparable_title(front_title)
    exact = [
        block
        for block in owned_blocks
        if _comparable_title(_projected_text(block)) == comparable
    ]
    if exact:
        return exact
    if _comparable_title("\n".join(_projected_text(block) for block in owned_blocks)) == comparable:
        return list(owned_blocks)
    return []


def _record(
    *,
    title_id: str,
    source_text: str,
    role: str,
    block_id_value: str | None,
    chapter_id: str | None,
    source_block_ids: list[str],
    opaque_tokens: list[str],
) -> dict[str, Any]:
    return {
        "title_id": title_id,
        "source_text": source_text,
        "role": role,
        "block_id": block_id_value,
        "chapter_id": chapter_id,
        "source_block_ids": source_block_ids,
        "number_prefix": _number_prefix(source_text),
        "opaque_tokens": opaque_tokens,
    }


def _chapter_list(
    chapters: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    if chapters is None:
        return []
    values: Any = chapters.get("chapters") if isinstance(chapters, Mapping) else chapters
    if values is None:
        return []
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TitleTranslationError("chapters must be a sequence or a chapters object")
    if any(not isinstance(item, Mapping) for item in values):
        raise TitleTranslationError("every chapter must be an object")
    return [dict(item) for item in values]


def _is_structural_title(block: Mapping[str, Any]) -> bool:
    kind = str(block.get("type") or block.get("kind") or "").casefold()
    return kind in STRUCTURAL_TYPES


def _title_role(block: Mapping[str, Any]) -> str:
    kind = str(block.get("type") or block.get("kind") or "heading").casefold()
    return kind if kind in STRUCTURAL_TYPES else "heading"


def _has_front_title_role(block: Mapping[str, Any]) -> bool:
    if str(block.get("source_role") or "").casefold() == "front_matter_title":
        return True
    return "front_matter_title" in {
        str(value).casefold() for value in block.get("front_matter_roles") or []
    }


def _projected_text(block: dict[str, Any]) -> str:
    return str(translation_input_block(block).get("text") or "")


def _prompt_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "title_id": str(record.get("title_id") or ""),
        "source_text": str(record.get("source_text") or ""),
        "role": str(record.get("role") or "heading"),
        "number_prefix": str(record.get("number_prefix") or ""),
        "opaque_tokens": [str(value) for value in record.get("opaque_tokens") or []],
    }


def _glossary_entries(
    glossary: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    if glossary is None:
        return []
    values: Any = glossary.get("entries") if isinstance(glossary, Mapping) else glossary
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TitleTranslationError("title translation glossary must contain an entries array")
    return [dict(item) for item in values if isinstance(item, Mapping)]


def _number_prefix(text: str) -> str:
    match = _NUMBER_PREFIX_RE.match(text)
    return match.group("prefix") if match else ""


def _opaque_tokens(text: str) -> list[str]:
    return _OPAQUE_TOKEN_RE.findall(text)


def _contains_protected_name(text: str, name: str) -> bool:
    if not name:
        return False
    if any(_is_compact_script(character) for character in name):
        return name in text
    return bool(re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text))


def _is_compact_script(character: str) -> bool:
    value = ord(character)
    return (
        0x3400 <= value <= 0x9FFF
        or 0x3040 <= value <= 0x30FF
        or 0xAC00 <= value <= 0xD7AF
    )


def _comparable_title(value: str) -> str:
    return " ".join(value.split()).casefold()


def _unique_strings(values: Iterable[Any]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


def _transport_size(records: Sequence[Mapping[str, Any]]) -> int:
    return len(
        json.dumps({"titles": list(records)}, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )
