from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlparse

from .io import sha256_json, write_json
from .stateful_pipeline import StatefulSessionError


CHAPTER_GUIDE_VERSION = "arc.companion.chapter-guide.v3"
CHAPTER_GUIDE_SOURCE_WINDOW_BYTES = 48 * 1024

CHAPTER_GUIDE_SETUP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["window_received"],
    "properties": {"window_received": {"type": "integer", "minimum": 1}},
}

GUIDE_SOURCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "url", "locator"],
    "properties": {
        "title": {"type": "string", "minLength": 1},
        "url": {"type": "string", "minLength": 1, "pattern": "^https?://"},
        "locator": {"type": "string", "minLength": 1},
    },
}

GUIDE_SOURCED_TEXT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["text", "sources"],
    "properties": {
        "text": {"type": "string", "minLength": 1},
        "sources": {
            "type": "array", "minItems": 1, "maxItems": 3,
            "items": GUIDE_SOURCE_SCHEMA,
        },
    },
}

CHAPTER_GUIDE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "motivation", "main_content", "section_logic", "prerequisites",
        "pedagogical_comparison", "historical_context", "supplementary_reading",
    ],
    "properties": {
        "motivation": {"type": ["string", "null"]},
        "main_content": {"type": ["string", "null"]},
        "section_logic": {"type": ["string", "null"]},
        "prerequisites": {"type": ["string", "null"]},
        "pedagogical_comparison": {
            **GUIDE_SOURCED_TEXT_SCHEMA,
            "type": ["object", "null"],
        },
        "historical_context": {
            "type": "array", "maxItems": 3,
            "items": GUIDE_SOURCED_TEXT_SCHEMA,
        },
        "supplementary_reading": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["title", "identifier", "reason", "evidence_id"],
            "properties": {
                "title": {"type": "string"}, "identifier": {"type": ["string", "null"]},
                "reason": {"type": "string"}, "evidence_id": {"type": "string"},
            },
        }},
    },
}


def generate_chapter_guide(
    chapter: Mapping[str, Any],
    source_blocks: list[Mapping[str, Any]],
    *,
    language: str,
    evidence: Mapping[str, Any],
    checkpoint_dir: Path,
    force: bool,
    call_model: Callable[[str, dict[str, Any], Path, str], dict[str, Any]],
    stateful: bool = False,
    allow_internet: bool = False,
    inherit_host_tools: bool = False,
    recipe_identity: str = "",
    intent_guidance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate one source-bounded guide and validate every external claim."""
    source = [{key: item.get(key) for key in ("block_id", "type", "title", "text", "tex") if item.get(key) is not None} for item in source_blocks]
    verified = _verified_evidence(evidence)
    local_direct_sources = _local_direct_sources(evidence)
    # Offline claims may use only URLs actually projected into this prompt.
    allowed_urls = _available_source_urls({
        "verified_evidence": verified,
        "local_direct_sources": local_direct_sources,
    })
    source_sha256 = sha256_json({
        "chapter": dict(chapter), "source": source, "language": language,
        "verified_evidence": verified,
        "local_direct_sources": local_direct_sources,
        "access_policy": {
            "allow_internet": allow_internet,
            "inherit_host_tools": inherit_host_tools,
        },
        "recipe_identity": recipe_identity,
        **(
            {"intent_guidance_sha256": sha256_json(intent_guidance)}
            if intent_guidance is not None else {}
        ),
    })
    path = checkpoint_dir / "chapter-guide.json"
    if path.is_file() and not force:
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = {}
        if cached.get("schema_version") == CHAPTER_GUIDE_VERSION and cached.get("source_sha256") == source_sha256:
            return cached
    prompt_prefix = (
        f"Write a concise {language} guide for this chapter. Every field is optional in substance: use null "
        "for nullable fields and [] for arrays when the material would not help. Explain motivation, content, "
        "logical flow, and prerequisites from the supplied source. Prefer positive, direct statements. Use a "
        "corrective contrast such as 'not X but Y' only when the source explicitly raises that confusion, or an "
        "inspected reliable source establishes it as a common misunderstanding whose correction materially helps "
        "the reader. Never invent a reader's mistaken belief. pedagogical_comparison may compare this chapter's "
        "teaching order with a genuinely different textbook or reliable reference order; it must contain at least "
        "one inspected direct source. Append it conceptually to the section-logic discussion rather than repeating "
        "section_logic. historical_context may contain at most three directly relevant items about discovery order, "
        "a reliable historical story or anecdote, or a short attributed quotation or joke; every item needs one to "
        "three inspected direct sources. Each source must have its exact title, a direct HTTP(S) URL, and a "
        "reader-understandable locator. Omit folklore, weakly supported claims, and irrelevant color. When no "
        "reliable material exists, return null or [] silently: never explain missing material, empty fields, source "
        "limitations, or the generation process. Supplementary reading may cite only evidence_id values in "
        "verified_evidence; do not repeat the source bibliography and do not invent references. "
        + (
            "Use host internet search and arc-paper-worker when useful, and inspect every final cited page. "
            if allow_internet else
            "Internet access is disabled. Cite only direct HTTP(S) URLs present in verified_evidence or other local "
            "evidence supplied in this prompt; omit every external claim that cannot be supported by that whitelist. "
        )
        + "Also, never describe or predict pagination in the generated companion document.\n"
    )
    windows = _source_windows(source) if stateful else [source]
    guidance_prefix = ""
    if intent_guidance is not None:
        from .intent_guidance import worker_guidance_prompt_prefix

        guidance_prefix = (
            worker_guidance_prompt_prefix(intent_guidance, lane="guide")
            + "\nIf this host has no sandboxed shell, request exact cached reference reads "
            "through arc_evidence_requests. Use list-reference-targets to inspect a "
            "non-inline target catalog, then get-parsed-toc or get-parsed-section "
            "with source_id, locator, and optional byte offset/limit; return [] when "
            "no controller read is needed.\n"
        )
    if len(windows) == 1:
        prompt = guidance_prefix + prompt_prefix + json.dumps(
            {
                "chapter": dict(chapter), "source_blocks": source,
                "verified_evidence": verified, "local_direct_sources": local_direct_sources,
            },
            ensure_ascii=False, sort_keys=True,
        )
        result = call_model(prompt, CHAPTER_GUIDE_SCHEMA, checkpoint_dir / "llm", f"companion-guide-{chapter.get('chapter_id')}")
    else:
        for index, window in enumerate(windows, 1):
            setup_prompt = (
                (guidance_prefix if index == 1 else "")
                +
                "Prepare this bounded source window for the final chapter guide. Preserve its logical "
                "relationship to earlier windows in this same session. Do not draft the final guide yet.\n"
                + json.dumps({
                    "chapter": dict(chapter), "window": index, "window_count": len(windows),
                    "source_blocks": window,
                }, ensure_ascii=False, sort_keys=True)
            )
            acknowledgement = call_model(
                setup_prompt, CHAPTER_GUIDE_SETUP_SCHEMA,
                checkpoint_dir / "llm" / f"window-{index:04d}",
                f"companion-guide-{chapter.get('chapter_id')}-window-{index:04d}",
            )
            if int(acknowledgement.get("window_received") or 0) != index:
                raise StatefulSessionError(
                    "chapter guide source-window acknowledgement changed its ordinal"
                )
        final_prompt = prompt_prefix + json.dumps({
            "chapter": dict(chapter), "prepared_source_windows": len(windows),
            "verified_evidence": verified, "local_direct_sources": local_direct_sources,
        }, ensure_ascii=False, sort_keys=True)
        result = call_model(
            final_prompt, CHAPTER_GUIDE_SCHEMA, checkpoint_dir / "llm" / "final",
            f"companion-guide-{chapter.get('chapter_id')}-final",
        )
    normalized = _validate_guide_result(
        result, verified=verified,
        allowed_urls=None if allow_internet else allowed_urls,
    )
    output = {
        "schema_version": CHAPTER_GUIDE_VERSION,
        "source_sha256": source_sha256,
        "chapter_id": str(chapter.get("chapter_id") or ""),
        **normalized,
    }
    write_json(path, output)
    return output


def chapter_guide_artifact_valid(
    value: Any,
    *,
    evidence: Mapping[str, Any] | None = None,
    allow_internet: bool = True,
) -> bool:
    """Return whether an accepted object is a complete guide-v3 artifact.

    Artifact import cannot trust a legacy contract label: validate the stored
    payload itself before assigning the current contract version.
    """
    if not isinstance(value, Mapping):
        return False
    expected_metadata = {"schema_version", "source_sha256", "chapter_id"}
    expected_fields = set(CHAPTER_GUIDE_SCHEMA["properties"])
    if set(value) != expected_metadata | expected_fields:
        return False
    if value.get("schema_version") != CHAPTER_GUIDE_VERSION:
        return False
    if not isinstance(value.get("source_sha256"), str) or not value.get("source_sha256"):
        return False
    if not isinstance(value.get("chapter_id"), str) or not value.get("chapter_id"):
        return False
    reading = value.get("supplementary_reading")
    if not isinstance(reading, list):
        return False
    if evidence is None:
        verified = [
            {"evidence_id": str(item.get("evidence_id") or "")}
            for item in reading if isinstance(item, Mapping)
        ]
        allowed_urls = None
    else:
        verified = _verified_evidence(evidence)
        local_direct_sources = _local_direct_sources(evidence)
        allowed_urls = (
            None if allow_internet else _available_source_urls({
                "verified_evidence": verified,
                "local_direct_sources": local_direct_sources,
            })
        )
    try:
        normalized = _validate_guide_result(
            {field: value.get(field) for field in CHAPTER_GUIDE_SCHEMA["properties"]},
            verified=verified,
            allowed_urls=allowed_urls,
        )
    except (TypeError, ValueError):
        return False
    return all(normalized.get(field) == value.get(field) for field in expected_fields)


def _validate_guide_result(
    result: Mapping[str, Any],
    *,
    verified: list[dict[str, Any]],
    allowed_urls: set[str] | None,
) -> dict[str, Any]:
    expected = list(CHAPTER_GUIDE_SCHEMA["properties"])
    if set(result) != set(expected):
        missing = sorted(set(expected) - set(result))
        extra = sorted(set(result) - set(expected))
        raise ValueError(f"chapter guide fields do not match v3 schema (missing={missing}, extra={extra})")

    output: dict[str, Any] = {}
    for field in ("motivation", "main_content", "section_logic", "prerequisites"):
        value = result.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"chapter guide {field} must be a string or null")
        output[field] = value.strip() if isinstance(value, str) and value.strip() else None

    comparison = result.get("pedagogical_comparison")
    output["pedagogical_comparison"] = (
        None if comparison is None else
        _validate_sourced_text(comparison, owner="pedagogical_comparison", allowed_urls=allowed_urls)
    )
    history = result.get("historical_context")
    if not isinstance(history, list) or len(history) > 3:
        raise ValueError("chapter guide historical_context must contain at most three items")
    output["historical_context"] = [
        _validate_sourced_text(item, owner=f"historical_context[{index}]", allowed_urls=allowed_urls)
        for index, item in enumerate(history)
    ]

    reading = result.get("supplementary_reading")
    if not isinstance(reading, list):
        raise ValueError("chapter guide supplementary_reading must be an array")
    allowed_ids = {str(item["evidence_id"]) for item in verified}
    normalized_reading: list[dict[str, Any]] = []
    for index, item in enumerate(reading):
        if not isinstance(item, Mapping) or set(item) != {"title", "identifier", "reason", "evidence_id"}:
            raise ValueError(f"chapter guide supplementary_reading[{index}] has invalid fields")
        evidence_id = str(item.get("evidence_id") or "")
        if evidence_id not in allowed_ids:
            raise ValueError(
                f"chapter guide cited unverified supplementary evidence: {evidence_id}"
            )
        title = str(item.get("title") or "").strip()
        reason = str(item.get("reason") or "").strip()
        identifier = item.get("identifier")
        if not title or not reason or (identifier is not None and not isinstance(identifier, str)):
            raise ValueError(f"chapter guide supplementary_reading[{index}] is incomplete")
        normalized_reading.append({
            "title": title,
            "identifier": identifier.strip() if isinstance(identifier, str) else None,
            "reason": reason,
            "evidence_id": evidence_id,
        })
    output["supplementary_reading"] = normalized_reading
    return output


def _validate_sourced_text(
    value: Any, *, owner: str, allowed_urls: set[str] | None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"text", "sources"}:
        raise ValueError(f"chapter guide {owner} must contain only text and sources")
    text = str(value.get("text") or "").strip()
    sources = value.get("sources")
    if not text or not isinstance(sources, list) or not 1 <= len(sources) <= 3:
        raise ValueError(f"chapter guide {owner} requires text and one to three sources")
    normalized_sources: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping) or set(source) != {"title", "url", "locator"}:
            raise ValueError(f"chapter guide {owner} source {index} has invalid fields")
        title = str(source.get("title") or "").strip()
        url = str(source.get("url") or "").strip()
        locator = str(source.get("locator") or "").strip()
        parsed = urlparse(url)
        if not title or not locator or parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"chapter guide {owner} source {index} must have title, HTTP(S) URL, and locator")
        if url.casefold() in seen:
            raise ValueError(f"chapter guide {owner} contains a duplicate source URL")
        if allowed_urls is not None and url not in allowed_urls:
            raise ValueError(f"offline chapter guide cited a URL outside supplied local evidence: {url}")
        seen.add(url.casefold())
        normalized_sources.append({"title": title, "url": url, "locator": locator})
    return {"text": text, "sources": normalized_sources}


def _available_source_urls(evidence: Mapping[str, Any]) -> set[str]:
    output: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key in ("url", "landing_url", "source_url", "html_url", "pdf_url", "canonical_locator"):
                candidate = value.get(key)
                if isinstance(candidate, str):
                    parsed = urlparse(candidate)
                    if parsed.scheme in {"http", "https"} and parsed.netloc:
                        output.add(candidate)
            for child in value.values():
                if isinstance(child, (Mapping, list)):
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(evidence)
    return output


def _local_direct_sources(evidence: Mapping[str, Any]) -> list[dict[str, str]]:
    """Project bounded direct citations already available to an offline guide."""
    output: list[dict[str, str]] = []
    seen: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            title = str(value.get("title") or "").strip()
            url = next((
                str(value.get(key) or "").strip()
                for key in ("url", "landing_url", "source_url", "html_url", "pdf_url")
                if str(value.get(key) or "").startswith(("http://", "https://"))
            ), "")
            if title and url and url.casefold() not in seen:
                locator = str(
                    value.get("locator") or value.get("section_title") or "Abstract"
                ).strip()
                output.append({"title": title, "url": url, "locator": locator})
                seen.add(url.casefold())
            for child in value.values():
                if isinstance(child, (Mapping, list)):
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(evidence)
    return output


def _source_windows(source: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    windows: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for block in source:
        candidate = [*current, block]
        size = len(json.dumps(candidate, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        if current and size > CHAPTER_GUIDE_SOURCE_WINDOW_BYTES:
            windows.append(current)
            current = [block]
        else:
            current = candidate
    if current or not windows:
        windows.append(current)
    return windows


def _verified_evidence(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = evidence.get("related_papers") or evidence.get("papers") or []
    bibliography_ids: set[str] = set()
    for item in evidence.get("bibliography") or []:
        if isinstance(item, Mapping):
            bibliography_ids.update(_publication_identities(item))
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, Mapping) or not item.get("evidence_id"):
            continue
        identities = _publication_identities(item)
        if not identities or identities & bibliography_ids or identities & seen:
            continue
        seen.update(identities)
        output.append({
            key: item.get(key)
            for key in (
                "evidence_id", "title", "doi", "arxiv_id", "paper_id", "evidence_level",
                "url", "landing_url", "source_url", "html_url", "pdf_url",
            )
            if item.get(key) is not None
        })
    return output


def _publication_identities(item: Mapping[str, Any]) -> set[str]:
    """Return comparable DOI, arXiv, and title identities for one citation."""
    output: set[str] = set()
    doi = _normalized_doi(item.get("doi"))
    if doi:
        output.add(f"doi:{doi}")
    arxiv = _normalized_arxiv(item.get("arxiv_id") or item.get("arxiv") or item.get("paper_id"))
    if arxiv:
        output.add(f"arxiv:{arxiv}")
    title = _normalized_title(item.get("title"))
    if title:
        output.add(f"title:{title}")
    return output


def _normalized_doi(value: Any) -> str:
    text = str(value or "").strip().casefold()
    for prefix in ("https://doi.org/", "http://doi.org/", "http://dx.doi.org/", "doi:"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return text.rstrip(".,; ") if text.startswith("10.") and "/" in text else ""


def _normalized_arxiv(value: Any) -> str:
    text = str(value or "").strip().casefold()
    for prefix in ("https://arxiv.org/abs/", "http://arxiv.org/abs/", "arxiv:"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    if text.startswith("arxiv"):
        return ""
    # Version suffixes name the same bibliography item.
    import re
    text = re.sub(r"v\d+$", "", text)
    return text if re.fullmatch(r"(?:[a-z.-]+/\d{7}|\d{4}\.\d{4,5})", text) else ""


def _normalized_title(value: Any) -> str:
    import re
    import unicodedata

    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(re.findall(r"\w+", text, flags=re.UNICODE))
