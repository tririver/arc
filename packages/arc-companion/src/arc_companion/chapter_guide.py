from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from .io import sha256_json, write_json
from .stateful_pipeline import StatefulSessionError


CHAPTER_GUIDE_VERSION = "arc.companion.chapter-guide.v1"
CHAPTER_GUIDE_SOURCE_WINDOW_BYTES = 48 * 1024

CHAPTER_GUIDE_SETUP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["window_received"],
    "properties": {"window_received": {"type": "integer", "minimum": 1}},
}

CHAPTER_GUIDE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["motivation", "main_content", "section_logic", "book_position", "prerequisites", "supplementary_reading"],
    "properties": {
        "motivation": {"type": ["string", "null"]},
        "main_content": {"type": ["string", "null"]},
        "section_logic": {"type": ["string", "null"]},
        "book_position": {"type": ["string", "null"]},
        "prerequisites": {"type": ["string", "null"]},
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
) -> dict[str, Any]:
    """Generate one source-bounded guide and reject unverified reading claims."""
    source = [{key: item.get(key) for key in ("block_id", "type", "title", "text", "tex") if item.get(key) is not None} for item in source_blocks]
    source_sha256 = sha256_json({"chapter": dict(chapter), "source": source, "language": language})
    path = checkpoint_dir / "chapter-guide.json"
    if path.is_file() and not force:
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = {}
        if cached.get("schema_version") == CHAPTER_GUIDE_VERSION and cached.get("source_sha256") == source_sha256:
            return cached
    verified = _verified_evidence(evidence)
    prompt_prefix = (
        f"Write a concise {language} guide for this chapter. Every field is optional in substance: use null "
        "when it would not help. Explain motivation, content, logical flow, position, and prerequisites only "
        "from the supplied source. Supplementary reading may cite only evidence_id values in verified_evidence; "
        "do not repeat the source bibliography and do not invent references.\n"
    )
    windows = _source_windows(source) if stateful else [source]
    if len(windows) == 1:
        prompt = prompt_prefix + json.dumps(
            {"chapter": dict(chapter), "source_blocks": source, "verified_evidence": verified},
            ensure_ascii=False, sort_keys=True,
        )
        result = call_model(prompt, CHAPTER_GUIDE_SCHEMA, checkpoint_dir / "llm", f"companion-guide-{chapter.get('chapter_id')}")
    else:
        for index, window in enumerate(windows, 1):
            setup_prompt = (
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
            "verified_evidence": verified,
        }, ensure_ascii=False, sort_keys=True)
        result = call_model(
            final_prompt, CHAPTER_GUIDE_SCHEMA, checkpoint_dir / "llm" / "final",
            f"companion-guide-{chapter.get('chapter_id')}-final",
        )
    allowed = {str(item["evidence_id"]) for item in verified}
    reading = [dict(item) for item in result.get("supplementary_reading") or []]
    unknown = [str(item.get("evidence_id") or "") for item in reading if str(item.get("evidence_id") or "") not in allowed]
    if unknown:
        raise ValueError(f"chapter guide cited unverified supplementary evidence: {unknown}")
    output = {"schema_version": CHAPTER_GUIDE_VERSION, "source_sha256": source_sha256, "chapter_id": str(chapter.get("chapter_id") or ""), **result}
    write_json(path, output)
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
        output.append({key: item.get(key) for key in ("evidence_id", "title", "doi", "arxiv_id", "paper_id", "evidence_level")})
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
