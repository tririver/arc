from __future__ import annotations

import os
from typing import Any, Callable

from arc_llm.runner import resolve_llm_config, run_json

from .ids import arxiv_path_id, doi_value, extract_paper_ids, inspire_recid, normalize_paper_id


MetadataLookup = Callable[..., dict[str, Any]]


class ReferenceInferenceError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


REFERENCE_INFERENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["focus_scope", "candidates", "warnings"],
    "properties": {
        "focus_scope": {
            "type": "string",
            "enum": [
                "one_domain",
                "two_domains",
                "more_than_two_domains",
                "unclear",
                "not_a_research_request",
            ],
        },
        "candidates": {
            "type": "array",
            "maxItems": 2,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["domain", "paper_id", "title", "evidence_urls", "reasoning"],
                "properties": {
                    "domain": {"type": "string"},
                    "paper_id": {"type": "string"},
                    "title": {"type": "string"},
                    "evidence_urls": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 5,
                        "items": {"type": "string"},
                    },
                    "reasoning": {"type": "string"},
                },
            },
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
}


def infer_main_references(
    text: str,
    *,
    provider: str = "auto",
    model: str | None = None,
    refresh: bool = False,
    metadata_lookup: MetadataLookup,
) -> dict[str, Any]:
    request = (text or "").strip()
    if not request:
        raise ReferenceInferenceError("empty_reference_request", "Reference inference requires non-empty text.")

    env = _internet_enabled_env()
    config = resolve_llm_config(provider=provider, model=model, env=env)
    payload = run_json(
        _build_prompt(request),
        schema=REFERENCE_INFERENCE_SCHEMA,
        provider=config.provider,
        model=config.model,
        env=env,
    )
    verified = _verify_payload(payload, metadata_lookup=metadata_lookup, refresh=refresh)
    if not verified["paper_ids"]:
        raise ReferenceInferenceError(
            "reference_inference_unverified",
            "The LLM did not return any INSPIRE-verified paper identifiers.",
        )
    verified["provider"] = config.provider
    verified["model"] = config.model
    verified["raw_llm_response"] = payload
    return verified


def _internet_enabled_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "ARC_LLM_ALLOW_INTERNET": "true",
            "ARC_CODEX_ALLOW_INTERNET": "true",
            "ARC_CLAUDE_ALLOW_INTERNET": "true",
        }
    )
    return env


def _build_prompt(request: str) -> str:
    return f"""You identify main reference papers for theoretical-physics research requests.

Use live web search. Do not rely on memory alone.

Rules:
1. First decide whether the request focuses on one research domain, two domains,
   more than two domains, is unclear, or is not a research request.
2. If it focuses on one domain, return the single most relevant foundational or
   canonical reference paper for that domain.
3. If it is clearly interdisciplinary and spans two or more domains, return one
   strongest reference paper for each of the two most relevant domains.
4. Prefer arXiv identifiers, formatted as arXiv:0911.3380 or
   arXiv:hep-th/0601001. If no arXiv identifier is available, return a DOI as
   doi:10.xxxx/yyyy.
5. Forbid hallucination: return a candidate only if web search found a reliable
   source that verifies the identifier and title, such as arXiv, INSPIRE,
   publisher DOI pages, NASA ADS, or the paper itself.
6. If no verified paper identifier can be found, return no candidates and add a
   warning. Do not guess.

Return JSON matching the supplied schema only.

User request:
{request}
"""


def _verify_payload(
    payload: dict[str, Any],
    *,
    metadata_lookup: MetadataLookup,
    refresh: bool,
) -> dict[str, Any]:
    focus_scope = str(payload.get("focus_scope") or "unclear")
    limit = 1 if focus_scope == "one_domain" else 2
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        candidates = []

    paper_ids: list[str] = []
    verified_references: list[dict[str, Any]] = []
    rejected_candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    for candidate in candidates[:limit]:
        if not isinstance(candidate, dict):
            rejected_candidates.append({"candidate": candidate, "error": "Candidate was not an object."})
            continue
        candidate_id = _candidate_identifier(candidate.get("paper_id"))
        if not candidate_id:
            rejected_candidates.append({"candidate": candidate, "error": "Candidate did not contain a paper id."})
            continue
        evidence_urls = _evidence_urls(candidate)
        if not evidence_urls:
            rejected_candidates.append({"paper_id": candidate_id, "candidate": candidate, "error": "No evidence URL."})
            continue
        try:
            metadata = metadata_lookup(candidate_id, refresh=refresh)
        except Exception as exc:
            rejected_candidates.append({"paper_id": candidate_id, "candidate": candidate, "error": str(exc)})
            continue
        verified_id = _preferred_identifier(metadata, fallback=candidate_id)
        key = verified_id.lower()
        if not _supported_identifier(verified_id) or key in seen:
            continue
        seen.add(key)
        paper_ids.append(verified_id)
        verified_references.append(
            {
                "paper_id": verified_id,
                "input_paper_id": candidate_id,
                "domain": str(candidate.get("domain") or ""),
                "llm_title": str(candidate.get("title") or ""),
                "verified_title": str(metadata.get("title") or ""),
                "evidence_urls": evidence_urls,
                "reasoning": str(candidate.get("reasoning") or ""),
                "metadata": {
                    "paper_id": metadata.get("paper_id"),
                    "arxiv_id": metadata.get("arxiv_id"),
                    "doi": metadata.get("doi"),
                    "inspire_recid": metadata.get("inspire_recid"),
                    "citation_count": metadata.get("citation_count"),
                    "year": metadata.get("year"),
                },
            }
        )

    warnings = payload.get("warnings") if isinstance(payload.get("warnings"), list) else []
    return {
        "paper_ids": paper_ids,
        "focus_scope": focus_scope,
        "warnings": [str(item) for item in warnings],
        "verified_references": verified_references,
        "rejected_candidates": rejected_candidates,
    }


def _candidate_identifier(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if ids := extract_paper_ids(raw):
        return ids[0]
    normalized = normalize_paper_id(raw)
    if _supported_identifier(normalized):
        return normalized
    return ""


def _evidence_urls(candidate: dict[str, Any]) -> list[str]:
    raw_urls = candidate.get("evidence_urls")
    if not isinstance(raw_urls, list):
        return []
    return [url for item in raw_urls if (url := str(item).strip()) and url.startswith(("http://", "https://"))]


def _preferred_identifier(metadata: dict[str, Any], *, fallback: str) -> str:
    if arxiv_id := metadata.get("arxiv_id"):
        return normalize_paper_id(f"arXiv:{arxiv_id}")
    paper_id = normalize_paper_id(str(metadata.get("paper_id") or ""))
    if arxiv_path_id(paper_id):
        return paper_id
    if doi := metadata.get("doi"):
        return normalize_paper_id(f"doi:{doi}")
    if _supported_identifier(paper_id):
        return paper_id
    return normalize_paper_id(fallback)


def _supported_identifier(identifier: str) -> bool:
    normalized = normalize_paper_id(identifier)
    return bool(arxiv_path_id(normalized) or doi_value(normalized) or inspire_recid(normalized))
