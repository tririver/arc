from __future__ import annotations

from datetime import datetime
import hashlib
import json
import re
from typing import Any, Iterable
from urllib.parse import SplitResult, urlsplit, urlunsplit


SOURCE_DESCRIPTOR_VERSION = "arc.companion.source-descriptor.v1"
_EVIDENCE_LEVELS = {"full_text", "abstract_only", "web_excerpt"}
_RELATIONS = {"prior", "later", "context"}
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class EvidenceProvenanceError(ValueError):
    """Raised when an evidence record is not independently auditable."""


def text_sha256(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def json_sha256(value: Any) -> str:
    material = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return text_sha256(material)


def arc_cache_descriptor(
    *,
    paper_id: str,
    title: str,
    authors: Any,
    year: Any,
    evidence_level: str,
    content: Any,
    document_hash: str = "",
) -> dict[str, Any]:
    """Describe immutable evidence loaded through the public arc-paper cache."""
    descriptor = {
        "schema_version": SOURCE_DESCRIPTOR_VERSION,
        "source_type": "arc_cache",
        "provider": "arc-paper",
        "canonical_locator": str(paper_id).strip(),
        "title": str(title or "").strip(),
        "authors": _authors(authors),
        "year": year,
        "evidence_level": str(evidence_level),
        "content_sha256": json_sha256(content),
        "locator": {
            "paper_id": str(paper_id).strip(),
            "document_hash": str(document_hash or "").strip(),
            "field": "document_blocks" if evidence_level == "full_text" else "abstract",
        },
    }
    validate_source_descriptor(descriptor)
    return descriptor


def inspire_abstract_descriptor(
    *, paper_id: str, title: str, authors: Any, year: Any, abstract: str,
) -> dict[str, Any]:
    """Describe an abstract independently fetched from an INSPIRE record."""
    descriptor = {
        "schema_version": SOURCE_DESCRIPTOR_VERSION,
        "source_type": "inspire_record",
        "provider": "INSPIRE",
        "canonical_locator": str(paper_id).strip(),
        "title": str(title or "").strip(),
        "authors": _authors(authors),
        "year": year,
        "evidence_level": "abstract_only",
        "content_sha256": json_sha256(str(abstract or "").strip()),
        "locator": {
            "paper_id": str(paper_id).strip(),
            "field": "abstract",
        },
    }
    validate_source_descriptor(descriptor)
    return descriptor


def web_evidence_record(
    *,
    relation: str,
    url: str,
    title: str,
    excerpt: str,
    retrieved_at: str,
    authors: Any = None,
    year: Any = None,
    provider: str = "",
) -> dict[str, Any]:
    """Register a fetched web excerpt; callers must supply the bytes they observed."""
    canonical_url, fragment = canonical_web_url(url)
    text = str(excerpt or "").strip()
    if not text:
        raise EvidenceProvenanceError("web evidence excerpt is empty")
    retrieved = _retrieved_at(retrieved_at)
    content_hash = text_sha256(text)
    identity_hash = json_sha256({"url": canonical_url, "content_sha256": content_hash})
    hostname = str(urlsplit(canonical_url).hostname or "")
    evidence_id = f"web-{identity_hash[:20]}"
    snippet = {
        "locator": canonical_url + (f"#{fragment}" if fragment else ""),
        "text": text,
        "sha256": content_hash,
    }
    record = {
        "evidence_id": evidence_id,
        "relation": str(relation),
        "evidence_level": "web_excerpt",
        "title": str(title or "").strip(),
        "authors": _authors(authors),
        "year": year,
        "snippets": [snippet],
        "source_descriptor": {
            "schema_version": SOURCE_DESCRIPTOR_VERSION,
            "source_type": "web",
            "provider": str(provider or hostname).strip(),
            "canonical_locator": canonical_url,
            "title": str(title or "").strip(),
            "authors": _authors(authors),
            "year": year,
            "evidence_level": "web_excerpt",
            "retrieved_at": retrieved,
            "content_sha256": content_hash,
            "locator": {
                "url": canonical_url,
                "fragment": fragment,
            },
        },
    }
    validate_evidence_record(record)
    return record


def canonical_web_url(value: str) -> tuple[str, str]:
    parsed = urlsplit(str(value or "").strip())
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
        raise EvidenceProvenanceError("web evidence requires an HTTP(S) URL")
    if parsed.username or parsed.password:
        raise EvidenceProvenanceError("web evidence URL must not contain credentials")
    scheme = parsed.scheme.casefold()
    hostname = parsed.hostname.casefold()
    try:
        port = parsed.port
    except ValueError as exc:
        raise EvidenceProvenanceError("web evidence URL has an invalid port") from exc
    default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    host = f"[{hostname}]" if ":" in hostname else hostname
    netloc = host if port is None or default_port else f"{host}:{port}"
    path = parsed.path or "/"
    canonical = urlunsplit(SplitResult(scheme, netloc, path, parsed.query, ""))
    return canonical, parsed.fragment


def validate_source_descriptor(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceProvenanceError("source descriptor must be an object")
    if value.get("schema_version") != SOURCE_DESCRIPTOR_VERSION:
        raise EvidenceProvenanceError("unsupported source descriptor schema")
    source_type = str(value.get("source_type") or "")
    if source_type not in {"arc_cache", "inspire_record", "web"}:
        raise EvidenceProvenanceError(f"unsupported source type: {source_type}")
    level = str(value.get("evidence_level") or "")
    if level not in _EVIDENCE_LEVELS:
        raise EvidenceProvenanceError(f"unsupported evidence level: {level}")
    digest = str(value.get("content_sha256") or "").casefold()
    if not _SHA256.fullmatch(digest):
        raise EvidenceProvenanceError("source descriptor has no valid content SHA-256")
    canonical = str(value.get("canonical_locator") or "").strip()
    if not canonical:
        raise EvidenceProvenanceError("source descriptor has no canonical locator")
    if not str(value.get("provider") or "").strip():
        raise EvidenceProvenanceError("source descriptor has no provider")
    if source_type == "arc_cache":
        if value.get("provider") != "arc-paper":
            raise EvidenceProvenanceError("ARC cache evidence must be provided by arc-paper")
        locator = value.get("locator")
        if not isinstance(locator, dict) or str(locator.get("paper_id") or "").strip() != canonical:
            raise EvidenceProvenanceError("ARC cache locator does not match the canonical paper ID")
        if level not in {"full_text", "abstract_only"}:
            raise EvidenceProvenanceError("ARC cache evidence has an invalid evidence level")
        expected_field = "document_blocks" if level == "full_text" else "abstract"
        if str(locator.get("field") or "") != expected_field:
            raise EvidenceProvenanceError("ARC cache locator field disagrees with its evidence level")
        document_hash = str(locator.get("document_hash") or "").casefold()
        if document_hash and not _SHA256.fullmatch(document_hash):
            raise EvidenceProvenanceError("ARC cache locator has an invalid document hash")
    elif source_type == "inspire_record":
        if value.get("provider") != "INSPIRE":
            raise EvidenceProvenanceError("INSPIRE evidence must name the INSPIRE provider")
        locator = value.get("locator")
        if not isinstance(locator, dict) or str(locator.get("paper_id") or "").strip() != canonical:
            raise EvidenceProvenanceError("INSPIRE locator does not match the canonical paper ID")
        if str(locator.get("field") or "") != "abstract" or level != "abstract_only":
            raise EvidenceProvenanceError("INSPIRE evidence must identify a verified abstract")
    else:
        canonical_url, _ = canonical_web_url(canonical)
        if canonical_url != canonical:
            raise EvidenceProvenanceError("web evidence locator is not canonical")
        locator = value.get("locator")
        if not isinstance(locator, dict) or str(locator.get("url") or "") != canonical:
            raise EvidenceProvenanceError("web evidence locator does not match its canonical URL")
        if not isinstance(locator.get("fragment", ""), str):
            raise EvidenceProvenanceError("web evidence locator fragment must be a string")
        _retrieved_at(str(value.get("retrieved_at") or ""))
        if level != "web_excerpt":
            raise EvidenceProvenanceError("web evidence must use web_excerpt level")
    return value


def validate_evidence_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceProvenanceError("evidence record must be an object")
    evidence_id = str(value.get("evidence_id") or "")
    if not _SAFE_ID.fullmatch(evidence_id):
        raise EvidenceProvenanceError(f"invalid evidence ID: {evidence_id}")
    relation = str(value.get("relation") or "")
    if relation not in _RELATIONS:
        raise EvidenceProvenanceError(f"unsupported evidence relation: {relation}")
    descriptor = validate_source_descriptor(value.get("source_descriptor"))
    if str(value.get("evidence_level") or "") != descriptor["evidence_level"]:
        raise EvidenceProvenanceError("evidence level disagrees with its source descriptor")
    pieces = value.get("snippets")
    if pieces is None:
        pieces = value.get("blocks")
    if not isinstance(pieces, list):
        raise EvidenceProvenanceError(f"evidence {evidence_id} has no recorded source pieces")
    for piece in pieces:
        _validate_piece(piece, evidence_id=evidence_id)
    content: Any
    if pieces:
        content = pieces
    else:
        content = str(value.get("abstract") or "")
        if descriptor["evidence_level"] != "abstract_only" or not content.strip():
            raise EvidenceProvenanceError(f"evidence {evidence_id} has no usable content")
    if descriptor["source_type"] == "web":
        if len(pieces) != 1 or descriptor["content_sha256"] != pieces[0]["sha256"]:
            raise EvidenceProvenanceError("web descriptor hash does not match its recorded excerpt")
        locator = descriptor["locator"]
        expected_locator = descriptor["canonical_locator"] + (
            f"#{locator['fragment']}" if locator["fragment"] else ""
        )
        if pieces[0].get("locator") != expected_locator:
            raise EvidenceProvenanceError("web excerpt locator does not match its source descriptor")
        identity_hash = json_sha256({
            "url": descriptor["canonical_locator"],
            "content_sha256": descriptor["content_sha256"],
        })
        if evidence_id != f"web-{identity_hash[:20]}":
            raise EvidenceProvenanceError("web evidence ID is not derived from its recorded source")
    elif descriptor["content_sha256"] != json_sha256(content):
        raise EvidenceProvenanceError("paper descriptor hash does not match its recorded content")
    elif str(value.get("paper_id") or "").strip() != descriptor["canonical_locator"]:
        raise EvidenceProvenanceError("ARC evidence paper ID does not match its source descriptor")
    return value


def validate_registry(records: Iterable[Any]) -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    for raw in records:
        record = validate_evidence_record(raw)
        evidence_id = str(record["evidence_id"])
        if evidence_id in registry:
            raise EvidenceProvenanceError(f"duplicate evidence ID: {evidence_id}")
        registry[evidence_id] = record
    return registry


def validate_cited_ids(values: Any, records: Iterable[Any]) -> list[str]:
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise EvidenceProvenanceError("evidence_ids must be an array of strings")
    registry = validate_registry(records)
    ordered = list(dict.fromkeys(values))
    unknown = sorted(set(ordered) - set(registry))
    if unknown:
        raise EvidenceProvenanceError(f"unknown or unregistered evidence IDs: {unknown}")
    return ordered


def validate_annotation_citations(annotation: Any, records: Iterable[Any]) -> list[str]:
    """Enforce claim-level IDs and exact locators without trusting model descriptors."""
    if not isinstance(annotation, dict):
        raise EvidenceProvenanceError("annotation must be an object")
    material = list(records)
    ids = validate_cited_ids(annotation.get("evidence_ids"), material)
    registry = validate_registry(material)
    cited_relations = {str(registry[value]["relation"]) for value in ids}
    used_relations: set[str] = set()
    claim_bound_ids: list[str] = []
    has_claim_level_bindings = False
    for field, relation in (
        ("prior_work", "prior"), ("later_work", "later"),
        ("context_claims", "context"),
    ):
        value = annotation.get(field)
        if not _has_claims(value):
            continue
        used_relations.add(relation)
        if relation not in cited_relations:
            raise EvidenceProvenanceError(
                f"{relation}-work commentary has no registered {relation} evidence"
            )
        if isinstance(value, list):
            if len(value) > 3:
                raise EvidenceProvenanceError(f"{field} contains more than three claims")
            has_claim_level_bindings = True
            for index, claim in enumerate(value, 1):
                if not isinstance(claim, dict) or not str(claim.get("text") or "").strip():
                    raise EvidenceProvenanceError(f"{field} claim {index} has no text")
                claim_ids = validate_cited_ids(claim.get("evidence_ids"), material)
                if not claim_ids:
                    raise EvidenceProvenanceError(f"{field} claim {index} has no registered evidence")
                wrong = [
                    value for value in claim_ids
                    if str(registry[value]["relation"]) != relation
                ]
                if wrong:
                    raise EvidenceProvenanceError(
                        f"{field} claim {index} cites relation-mismatched evidence: {wrong}"
                    )
                if "request_key" not in claim:
                    raise EvidenceProvenanceError(
                        f"{field} claim {index} has no request_key field"
                    )
                request_key = claim.get("request_key")
                if request_key is not None and not str(request_key).strip():
                    raise EvidenceProvenanceError(
                        f"{field} claim {index} has an empty request_key"
                    )
                locators = claim.get("source_locators")
                if not isinstance(locators, list) or not locators:
                    raise EvidenceProvenanceError(
                        f"{field} claim {index} has no source locators"
                    )
                located_ids: set[str] = set()
                for locator_index, locator in enumerate(locators, 1):
                    if not isinstance(locator, dict):
                        raise EvidenceProvenanceError(
                            f"{field} claim {index} locator {locator_index} is not an object"
                        )
                    evidence_id = str(locator.get("evidence_id") or "")
                    location = str(locator.get("locator") or "").strip()
                    if evidence_id not in claim_ids or not location:
                        raise EvidenceProvenanceError(
                            f"{field} claim {index} has an unbound source locator"
                        )
                    if location not in _evidence_source_locators(registry[evidence_id]):
                        raise EvidenceProvenanceError(
                            f"{field} claim {index} cites an unknown source locator for {evidence_id}"
                        )
                    located_ids.add(evidence_id)
                if located_ids != set(claim_ids):
                    raise EvidenceProvenanceError(
                        f"{field} claim {index} must locate every cited evidence item"
                    )
                claim_bound_ids.extend(claim_ids)
    if _has_claims(annotation.get("explanation")) or _has_claims(annotation.get("commentary")):
        used_relations.add("context")
    unused = [value for value in ids if str(registry[value]["relation"]) not in used_relations]
    if unused:
        raise EvidenceProvenanceError(f"annotation cites evidence that supports no recorded claim: {unused}")
    context_ids = {
        value for value in ids if str(registry[value]["relation"]) == "context"
    }
    reader_context = _normalized_reader_context(annotation)
    missing_context_titles = [
        value
        for value in context_ids
        if not _normalized_title_present(registry[value].get("title"), reader_context)
    ]
    if missing_context_titles:
        raise EvidenceProvenanceError(
            "context evidence is not cited by exact source title in explanation/commentary: "
            f"{missing_context_titles}"
        )
    if has_claim_level_bindings and set(ids) != set(claim_bound_ids) | context_ids:
        raise EvidenceProvenanceError(
            "annotation evidence_ids must equal the union of claim-level evidence IDs"
        )
    return ids


def _normalized_reader_context(annotation: dict[str, Any]) -> str:
    values: list[str] = []
    for field in ("explanation", "commentary"):
        value = annotation.get(field)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    values.append(item)
                elif isinstance(item, dict):
                    values.extend(
                        str(item.get(key) or "")
                        for key in ("text", "summary", "claim", "title")
                    )
    return " ".join(" ".join(values).split()).casefold()


def _normalized_title_present(title: Any, reader_context: str) -> bool:
    normalized = " ".join(str(title or "").split()).casefold()
    return bool(normalized and normalized in reader_context)


def _evidence_source_locators(record: dict[str, Any]) -> set[str]:
    pieces = record.get("snippets")
    if pieces is None:
        pieces = record.get("blocks")
    locators = {
        str(piece.get("locator") or piece.get("block_id") or "").strip()
        for piece in pieces or []
        if isinstance(piece, dict)
    }
    if not locators and str(record.get("abstract") or "").strip():
        locators.add("abstract")
    return {value for value in locators if value}


def _validate_piece(value: Any, *, evidence_id: str) -> None:
    if not isinstance(value, dict):
        raise EvidenceProvenanceError(f"evidence {evidence_id} contains a non-object source piece")
    text = str(value.get("text") or "")
    if not text.strip():
        raise EvidenceProvenanceError(f"evidence {evidence_id} contains an empty source piece")
    digest = str(value.get("sha256") or "").casefold()
    if not _SHA256.fullmatch(digest) or digest != text_sha256(text):
        raise EvidenceProvenanceError(f"evidence {evidence_id} source-piece hash mismatch")
    locator = value.get("locator") or value.get("block_id")
    if not str(locator or "").strip():
        raise EvidenceProvenanceError(f"evidence {evidence_id} source piece has no locator")


def _retrieved_at(value: str) -> str:
    text = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvidenceProvenanceError("web evidence retrieved_at is not ISO-8601") from exc
    if parsed.tzinfo is None:
        raise EvidenceProvenanceError("web evidence retrieved_at must include a timezone")
    return parsed.isoformat()


def _authors(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    return [value]


def _has_claims(value: Any) -> bool:
    if isinstance(value, list):
        return any(
            str(item.get("text") or "").strip() if isinstance(item, dict) else str(item).strip()
            for item in value
        )
    return bool(str(value or "").strip())
