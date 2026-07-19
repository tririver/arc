from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Iterable, Mapping

from .evidence import (
    EvidenceProvenanceError,
    arc_cache_descriptor,
    text_sha256,
    validate_evidence_record,
)


EVIDENCE_REQUEST_VERSION = "arc.companion.evidence-request.v1"
EVIDENCE_RESOLUTION_VERSION = "arc.companion.evidence-resolution.v1"
LANE_NAMES = ("arc", "inspire", "web")
_RELATIONS = {"prior", "later", "context"}

EvidenceLane = Callable[[list[dict[str, Any]]], Iterable[dict[str, Any]]]


@dataclass(frozen=True)
class EvidenceResolution:
    records: tuple[dict[str, Any], ...]
    evidence_ids_by_segment: dict[str, tuple[str, ...]]
    supported_request_keys: tuple[str, ...]
    audit: dict[str, Any]


class EvidenceRequestController:
    """Run independent discovery/verification lanes and register audited evidence.

    A lane returns candidate envelopes with ``request_key`` and, for verified
    paper evidence, ``record``.  Web discovery results may instead return a
    URL/snippet envelope; those are audited but never registered as claim
    evidence.  Callers can inject host-specific lanes without coupling this
    controller to an agent UI or tool syntax.
    """

    def __init__(self, lanes: Mapping[str, EvidenceLane] | None = None) -> None:
        configured = dict(lanes or default_evidence_lanes())
        missing = set(LANE_NAMES) - set(configured)
        if missing:
            raise ValueError(f"evidence controller requires all lanes: {sorted(missing)}")
        self._lanes = {name: configured[name] for name in LANE_NAMES}

    def resolve(
        self,
        requests: Iterable[dict[str, Any]],
        *,
        existing_records: Iterable[dict[str, Any]] = (),
    ) -> EvidenceResolution:
        material = [dict(item) for item in requests]
        request_by_key = {str(item["request_key"]): item for item in material}
        audit_lanes: dict[str, Any] = {}
        lane_outputs: dict[str, list[dict[str, Any]]] = {}

        # All lanes are submitted before any result is inspected.  In
        # particular, an ARC/domain hit cannot suppress INSPIRE or web work.
        with ThreadPoolExecutor(max_workers=len(LANE_NAMES)) as executor:
            futures = {
                executor.submit(self._lanes[name], [dict(item) for item in material]): name
                for name in LANE_NAMES
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    output = [dict(item) for item in future.result()]
                except Exception as exc:  # one failed lane never cancels another
                    lane_outputs[name] = []
                    audit_lanes[name] = {
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "raw_results": [],
                    }
                else:
                    lane_outputs[name] = output
                    audit_lanes[name] = {"status": "complete", "raw_results": output}

        registry: list[dict[str, Any]] = []
        identity_to_id: dict[str, str] = {}
        for raw in existing_records:
            try:
                record = validate_evidence_record(raw)
            except EvidenceProvenanceError:
                continue
            registry.append(record)
            identity_to_id[_source_identity(record)] = str(record["evidence_id"])

        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        ids_by_segment: dict[str, list[str]] = {}
        supported_keys: set[str] = set()
        for lane_name in LANE_NAMES:
            for candidate in lane_outputs.get(lane_name, []):
                request_key = str(candidate.get("request_key") or "")
                request = request_by_key.get(request_key)
                base = {"lane": lane_name, "request_key": request_key}
                if request is None:
                    rejected.append({**base, "reason": "unknown_request"})
                    continue
                raw_record = candidate.get("record")
                if not isinstance(raw_record, dict):
                    rejected.append({**base, "reason": "discovery_only_not_claim_evidence"})
                    continue
                try:
                    record = validate_evidence_record(raw_record)
                except EvidenceProvenanceError as exc:
                    rejected.append({**base, "reason": f"invalid_record: {exc}"})
                    continue
                descriptor = record["source_descriptor"]
                if descriptor.get("source_type") == "web" and candidate.get("verified_source") is not True:
                    rejected.append({**base, "reason": "web_snippet_not_claim_evidence"})
                    continue
                if str(record.get("relation") or "") != request["relation"]:
                    rejected.append({**base, "reason": "relation_mismatch"})
                    continue
                identity = _source_identity(record)
                evidence_id = identity_to_id.get(identity)
                if evidence_id is None:
                    evidence_id = str(record["evidence_id"])
                    identity_to_id[identity] = evidence_id
                    registry.append(record)
                    accepted.append({**base, "evidence_id": evidence_id, "identity": identity})
                else:
                    accepted.append({
                        **base,
                        "evidence_id": evidence_id,
                        "identity": identity,
                        "deduplicated": True,
                    })
                segment_id = str(request["segment_id"])
                ids_by_segment.setdefault(segment_id, []).append(evidence_id)
                supported_keys.add(request_key)

        new_ids = {item["evidence_id"] for item in accepted if not item.get("deduplicated")}
        new_records = tuple(item for item in registry if item["evidence_id"] in new_ids)
        return EvidenceResolution(
            records=new_records,
            evidence_ids_by_segment={
                key: tuple(dict.fromkeys(values)) for key, values in ids_by_segment.items()
            },
            supported_request_keys=tuple(sorted(supported_keys)),
            audit={
                "schema_version": EVIDENCE_RESOLUTION_VERSION,
                "requests": material,
                "lanes": audit_lanes,
                "accepted": accepted,
                "rejected": rejected,
                "claim_evidence_policy": "verified_paper_or_fetched_web_sources",
            },
        )


def normalize_evidence_requests(segment_id: str, values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        raise ValueError("evidence_requests must be an array")
    if len(values) > 2:
        raise ValueError("each annotation may request at most two evidence items")
    output: list[dict[str, Any]] = []
    for index, raw in enumerate(values, 1):
        if not isinstance(raw, dict):
            raise ValueError("evidence request must be an object")
        relation = str(raw.get("relation") or "")
        needed_claim = str(raw.get("needed_claim") or "").strip()
        reason = str(raw.get("reason") or "").strip()
        if relation not in _RELATIONS or not needed_claim or not reason:
            raise ValueError("evidence request requires relation, needed_claim, and reason")
        request_key = f"{segment_id}:request-{index}"
        output.append({
            "schema_version": EVIDENCE_REQUEST_VERSION,
            "request_key": request_key,
            "segment_id": str(segment_id),
            "relation": relation,
            "needed_claim": needed_claim,
            "queries": _strings(raw.get("queries")),
            "candidate_paper_ids": _strings(raw.get("candidate_paper_ids")),
            "candidate_urls": _strings(raw.get("candidate_urls")),
            "reason": reason,
        })
    return output


def default_evidence_lanes() -> dict[str, EvidenceLane]:
    return {"arc": _arc_lane, "inspire": _inspire_lane, "web": _web_lane}


def _arc_lane(requests: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    from arc_paper import service

    for request in requests:
        for paper_id in request["candidate_paper_ids"]:
            parsed = service.get_parsed_source(paper_id)
            data = parsed.get("data") if isinstance(parsed, dict) and parsed.get("ok") else None
            if not isinstance(data, dict):
                continue
            sections = data.get("sections") or []
            blocks = []
            for index, section in enumerate(sections):
                text = str(section.get("text") or "").strip() if isinstance(section, dict) else ""
                if text:
                    blocks.append({
                        "block_id": str(section.get("section_id") or f"section-{index + 1}"),
                        "text": text,
                        "sha256": text_sha256(text),
                    })
            if not blocks:
                continue
            yield _candidate(request, _paper_record(
                paper_id=str(data.get("paper_id") or paper_id),
                relation=request["relation"],
                title="",
                authors=[],
                year=None,
                evidence_level="full_text",
                content=blocks,
                source_hash=str(data.get("source_hash") or ""),
            ))


def _inspire_lane(requests: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    from arc_paper import service

    for request in requests:
        for paper_id in request["candidate_paper_ids"]:
            result = service.get_metadata(paper_id)
            data = result.get("data") if isinstance(result, dict) and result.get("ok") else None
            if not isinstance(data, dict):
                continue
            abstract = str(data.get("abstract") or "").strip()
            canonical = str(data.get("paper_id") or paper_id)
            if not abstract:
                continue
            yield _candidate(request, _paper_record(
                paper_id=canonical,
                relation=request["relation"],
                title=str(data.get("title") or ""),
                authors=data.get("authors") or [],
                year=data.get("year"),
                evidence_level="abstract_only",
                content=abstract,
            ))


def _web_lane(requests: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    # URLs and snippets discovered by the writing agent remain discovery hints.
    # A host integration can inject a web lane that maps them to verified paper
    # records; raw snippets deliberately cannot enter the registry here.
    for request in requests:
        for url in request["candidate_urls"]:
            yield {"request_key": request["request_key"], "candidate_url": url, "discovery_only": True}


def _candidate(request: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    return {"request_key": request["request_key"], "record": record}


def _paper_record(
    *, paper_id: str, relation: str, title: str, authors: Any, year: Any,
    evidence_level: str, content: Any, source_hash: str = "",
) -> dict[str, Any]:
    identity = hashlib.sha256(
        f"{paper_id}\0{relation}\0{evidence_level}\0{content!r}".encode("utf-8")
    ).hexdigest()
    blocks = content if evidence_level == "full_text" else []
    abstract = content if evidence_level == "abstract_only" else ""
    record = {
        "evidence_id": f"evidence-{identity[:20]}",
        "relation": relation,
        "paper_id": paper_id,
        "title": title,
        "authors": authors if isinstance(authors, list) else [authors],
        "year": year,
        "evidence_level": evidence_level,
        "abstract": abstract,
        "blocks": blocks,
    }
    record["source_descriptor"] = arc_cache_descriptor(
        paper_id=paper_id,
        title=title,
        authors=record["authors"],
        year=year,
        evidence_level=evidence_level,
        content=blocks if blocks else abstract,
        document_hash=source_hash if len(source_hash) == 64 else "",
    )
    validate_evidence_record(record)
    return record


def _source_identity(record: dict[str, Any]) -> str:
    descriptor = record["source_descriptor"]
    return f"{descriptor['source_type']}:{descriptor['canonical_locator']}"


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("evidence request query and candidate fields must be arrays of strings")
    return list(dict.fromkeys(item.strip() for item in value if item.strip()))
