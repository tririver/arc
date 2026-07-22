"""Small, dependency-free helpers for language-aware companion artifacts.

The helpers intentionally implement only the BCP 47 canonicalization ARC
needs for stable metadata, direction selection, and font routing.  They do not
attempt language detection or registry validation.
"""

from __future__ import annotations

import re
import unicodedata


_RTL_LANGUAGES = frozenset({
    "ar", "arc", "ckb", "dv", "fa", "he", "khw", "ks", "nqo", "pa",
    "ps", "sd", "syr", "ug", "ur", "yi",
})
_AUTO_DIRECTION_LANGUAGES = frozenset({"mul", "und", "zxx"})
_TRADITIONAL_CHINESE_REGIONS = frozenset({"HK", "MO", "TW"})
_LEGACY_LANGUAGE_CODES = {"iw": "he", "in": "id", "ji": "yi"}
_LANGUAGE_RE = re.compile(r"^[A-Za-z]{2,8}$")
_SCRIPT_RE = re.compile(r"^[A-Za-z]{4}$")
_REGION_RE = re.compile(r"^(?:[A-Za-z]{2}|[0-9]{3})$")
_SUBTAG_RE = re.compile(r"^[A-Za-z0-9]{1,8}$")


def normalize_language_tag(value: str | None, *, default: str = "und") -> str:
    """Return a stable, conservatively canonicalized BCP 47 language tag.

    Empty or malformed values return ``default``.  The primary language is
    lowercase, a script subtag is title-case, a region is uppercase, and all
    remaining extension/variant subtags are lowercase.  Underscores are
    accepted as a common input spelling but are serialized as hyphens.
    """

    fallback = str(default or "und").strip() or "und"
    raw = str(value or "").strip().replace("_", "-")
    if not raw:
        return fallback
    parts = raw.split("-")
    if any(not part or not _SUBTAG_RE.fullmatch(part) for part in parts):
        return fallback
    if parts[0].lower() == "x" and len(parts) > 1:
        return "-".join(part.lower() for part in parts)
    if not _LANGUAGE_RE.fullmatch(parts[0]):
        return fallback

    primary = _LEGACY_LANGUAGE_CODES.get(parts[0].lower(), parts[0].lower())
    normalized = [primary]
    script_seen = False
    region_seen = False
    private_or_extension = False
    for part in parts[1:]:
        if private_or_extension:
            normalized.append(part.lower())
            continue
        if len(part) == 1:
            normalized.append(part.lower())
            private_or_extension = True
        elif not script_seen and _SCRIPT_RE.fullmatch(part):
            normalized.append(part.title())
            script_seen = True
        elif not region_seen and _REGION_RE.fullmatch(part):
            normalized.append(part.upper())
            region_seen = True
        else:
            normalized.append(part.lower())
    return "-".join(normalized)


def base_language(value: str | None) -> str:
    """Return the normalized primary language subtag."""

    return normalize_language_tag(value).split("-", 1)[0]


def is_same_base_language(left: str | None, right: str | None) -> bool:
    """Whether two known language tags share a primary language subtag."""

    left_base = base_language(left)
    right_base = base_language(right)
    unknown = _AUTO_DIRECTION_LANGUAGES
    return left_base not in unknown and left_base == right_base


def language_direction(value: str | None) -> str:
    """Return ``ltr``, ``rtl``, or ``auto`` for HTML language metadata."""

    language = base_language(value)
    if language in _AUTO_DIRECTION_LANGUAGES:
        return "auto"
    return "rtl" if language in _RTL_LANGUAGES else "ltr"


def cjk_font_region(value: str | None) -> str | None:
    """Return the Noto/Source-Han CJK region suffix for a language tag."""

    tag = normalize_language_tag(value)
    parts = tag.split("-")
    language = parts[0]
    if language == "ja":
        return "JP"
    if language == "ko":
        return "KR"
    if language not in {"zh", "yue"}:
        return None
    script = next((part for part in parts[1:] if _SCRIPT_RE.fullmatch(part)), "")
    region = next((part for part in parts[1:] if _REGION_RE.fullmatch(part)), "")
    if script == "Hant" or region in _TRADITIONAL_CHINESE_REGIONS or language == "yue":
        return "TC"
    return "SC"


def contains_lexical_term(text: str, term: str, *, case_sensitive: bool = False) -> bool:
    """Match a term using Unicode-aware lexical boundaries.

    NFKC normalization makes compatibility-width source text searchable.  At
    an alphabetic/numeric edge, a match cannot be embedded in another Unicode
    word.  CJK syllabic/ideographic edges remain searchable while adjacent so
    glossary terms such as ``量子`` still match ``量子场``.
    """

    haystack = unicodedata.normalize("NFKC", str(text))
    needle = unicodedata.normalize("NFKC", str(term)).strip()
    if not needle:
        return False
    if not case_sensitive:
        haystack = haystack.casefold()
        needle = needle.casefold()

    start = 0
    while (offset := haystack.find(needle, start)) >= 0:
        before = haystack[offset - 1] if offset else ""
        end = offset + len(needle)
        after = haystack[end] if end < len(haystack) else ""
        if (
            (not _requires_word_boundary(needle[0]) or not _is_word_character(before))
            and (not _requires_word_boundary(needle[-1]) or not _is_word_character(after))
        ):
            return True
        start = offset + 1
    return False


def _is_word_character(value: str) -> bool:
    return bool(value) and unicodedata.category(value)[0] in {"L", "M", "N"}


def _requires_word_boundary(value: str) -> bool:
    return _is_word_character(value) and not _is_cjk_character(value)


def _is_cjk_character(value: str) -> bool:
    if not value:
        return False
    codepoint = ord(value)
    return (
        0x2E80 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x323AF
        or 0x3040 <= codepoint <= 0x30FF
        or 0x31F0 <= codepoint <= 0x31FF
        or 0xAC00 <= codepoint <= 0xD7AF
        or 0x1100 <= codepoint <= 0x11FF
    )
