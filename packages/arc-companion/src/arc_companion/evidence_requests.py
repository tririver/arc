from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
from itertools import islice
import re
from typing import Any, Callable, Iterable, Mapping, Protocol

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
_MAX_GRAPH_RESULTS_PER_ANCHOR = 48
_MAX_DISCOVERED_PAPERS_PER_REQUEST = 96

EvidenceLane = Callable[[list[dict[str, Any]]], Iterable[dict[str, Any]]]


class EvidenceDiscoveryAdapter(Protocol):
    """Host-neutral discovery surface used by the three evidence lanes.

    ARC's built-in adapter implements cache/full-text and INSPIRE graph access.
    Hosts that provide scholarly or web search can inject those two search
    methods without coupling the companion pipeline to host-specific tools.
    """

    def search_arc_full_text(
        self, paper_ids: list[str], query: str,
    ) -> Iterable[dict[str, Any]]: ...

    def get_parsed_source(self, paper_id: str) -> dict[str, Any] | None: ...

    def get_metadata(self, paper_id: str) -> dict[str, Any] | None: ...

    def get_references(self, paper_id: str) -> Iterable[dict[str, Any]]: ...

    def get_citers(self, paper_id: str) -> Iterable[dict[str, Any]]: ...

    def search_inspire(self, query: str) -> Iterable[dict[str, Any]]: ...

    def search_web(self, query: str) -> Iterable[dict[str, Any]]: ...


class ArcPaperEvidenceAdapter:
    """Default adapter backed by public ``arc-paper`` service operations."""

    def search_arc_full_text(
        self, paper_ids: list[str], query: str,
    ) -> Iterable[dict[str, Any]]:
        from arc_paper import service

        result = service.search_full_text(paper_ids, query=query, limit=20)
        data = result.get("data") if isinstance(result, dict) and result.get("ok") else []
        return data if isinstance(data, list) else []

    def get_parsed_source(self, paper_id: str) -> dict[str, Any] | None:
        from arc_paper import service

        result = service.get_parsed_source(paper_id)
        data = result.get("data") if isinstance(result, dict) and result.get("ok") else None
        return data if isinstance(data, dict) else None

    def get_metadata(self, paper_id: str) -> dict[str, Any] | None:
        from arc_paper import service

        result = service.get_metadata(paper_id)
        data = result.get("data") if isinstance(result, dict) and result.get("ok") else None
        return data if isinstance(data, dict) else None

    def get_references(self, paper_id: str) -> Iterable[dict[str, Any]]:
        from arc_paper import service

        result = service.get_references(paper_id, enrich=True)
        data = result.get("data") if isinstance(result, dict) and result.get("ok") else []
        return data if isinstance(data, list) else []

    def get_citers(self, paper_id: str) -> Iterable[dict[str, Any]]:
        from arc_paper import service

        result = service.get_citers(paper_id, limit=_MAX_GRAPH_RESULTS_PER_ANCHOR)
        data = result.get("data") if isinstance(result, dict) and result.get("ok") else []
        return data if isinstance(data, list) else []

    def search_inspire(self, query: str) -> Iterable[dict[str, Any]]:
        # arc-paper does not yet expose free-form INSPIRE search.  Candidate
        # expansion through seed references/citers remains available; hosts
        # with scholarly search inject this method.
        return []

    def search_web(self, query: str) -> Iterable[dict[str, Any]]:
        # Web search is deliberately host-provided.  Raw snippets are discovery
        # hints only and still pass through paper verification below.
        return []


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

    def __init__(
        self,
        lanes: Mapping[str, EvidenceLane] | None = None,
        *,
        adapter: EvidenceDiscoveryAdapter | None = None,
        domain_paper_ids: Iterable[str] = (),
        seed_paper_ids: Iterable[str] = (),
    ) -> None:
        configured = dict(lanes or default_evidence_lanes(
            adapter=adapter,
            domain_paper_ids=domain_paper_ids,
            seed_paper_ids=seed_paper_ids,
        ))
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
        alias_to_id: dict[str, str] = {}
        registry_index: dict[str, int] = {}
        for raw in existing_records:
            try:
                record = validate_evidence_record(raw)
            except EvidenceProvenanceError:
                continue
            registry.append(record)
            registry_index[str(record["evidence_id"])] = len(registry) - 1
            for identity in _source_identities(record):
                alias_to_id[identity] = str(record["evidence_id"])

        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        ids_by_segment: dict[str, list[str]] = {}
        supported_keys: set[str] = set()
        created_ids: set[str] = set()
        ranked_candidates = sorted(
            (
                (lane_name, candidate)
                for lane_name in LANE_NAMES
                for candidate in lane_outputs.get(lane_name, [])
            ),
            key=lambda item: (-_score(item[1].get("relevance_score")), LANE_NAMES.index(item[0])),
        )
        for lane_name, candidate in ranked_candidates:
            request_key = str(candidate.get("request_key") or "")
            request = request_by_key.get(request_key)
            base = {
                "lane": lane_name,
                "request_key": request_key,
                "relevance_score": _score(candidate.get("relevance_score")),
                "discovery_source": str(candidate.get("discovery_source") or ""),
            }
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
            identities = _candidate_identities(record, candidate)
            evidence_id = next((alias_to_id[value] for value in identities if value in alias_to_id), None)
            identity = identities[0]
            if evidence_id is None:
                evidence_id = str(record["evidence_id"])
                registry.append(record)
                registry_index[evidence_id] = len(registry) - 1
                created_ids.add(evidence_id)
                accepted.append({**base, "evidence_id": evidence_id, "identity": identity})
            else:
                upgraded = False
                if evidence_id in created_ids:
                    index = registry_index[evidence_id]
                    if _record_quality(record) > _record_quality(registry[index]):
                        replacement = {**record, "evidence_id": evidence_id}
                        registry[index] = replacement
                        upgraded = True
                accepted.append({
                    **base,
                    "evidence_id": evidence_id,
                    "identity": identity,
                    "deduplicated": True,
                    "upgraded_to_preferred_source": upgraded,
                })
            for value in identities:
                alias_to_id[value] = evidence_id
            segment_id = str(request["segment_id"])
            ids_by_segment.setdefault(segment_id, []).append(evidence_id)
            supported_keys.add(request_key)

        new_records = tuple(item for item in registry if item["evidence_id"] in created_ids)
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


def default_evidence_lanes(
    *,
    adapter: EvidenceDiscoveryAdapter | None = None,
    domain_paper_ids: Iterable[str] = (),
    seed_paper_ids: Iterable[str] = (),
) -> dict[str, EvidenceLane]:
    backend = adapter or ArcPaperEvidenceAdapter()
    domain_ids = _canonical_ids(domain_paper_ids)
    seed_ids = _canonical_ids(seed_paper_ids)
    return {
        "arc": lambda requests: _arc_lane(requests, backend, domain_ids),
        "inspire": lambda requests: _inspire_lane(requests, backend, seed_ids),
        "web": lambda requests: _web_lane(requests, backend),
    }


def _arc_lane(
    requests: list[dict[str, Any]],
    adapter: EvidenceDiscoveryAdapter,
    domain_paper_ids: list[str],
) -> Iterable[dict[str, Any]]:
    for request in requests:
        explicit = _canonical_ids(request["candidate_paper_ids"])
        pool = _canonical_ids([*domain_paper_ids, *explicit])
        discovered: dict[str, tuple[float, str]] = {
            paper_id: (0.25 if paper_id in explicit else 0.05, "explicit_candidate")
            for paper_id in explicit
        }
        for query in request["queries"]:
            for hit in adapter.search_arc_full_text(pool, query):
                paper_id = _paper_id(hit)
                if not paper_id:
                    continue
                score = _relevance(request, hit, query=query)
                if paper_id in domain_paper_ids:
                    score += 0.05
                current = discovered.get(paper_id)
                if current is None or score > current[0]:
                    discovered[paper_id] = (score, "arc_full_text_query")
        for paper_id, (score, source) in sorted(discovered.items(), key=lambda item: -item[1][0]):
            data = adapter.get_parsed_source(paper_id)
            record = _full_text_record(request, data, paper_id)
            if record is not None:
                yield _candidate(
                    request,
                    record,
                    aliases=_paper_aliases(data or {}, paper_id),
                    relevance_score=score,
                    discovery_source=source,
                )


def _inspire_lane(
    requests: list[dict[str, Any]],
    adapter: EvidenceDiscoveryAdapter,
    seed_paper_ids: list[str],
) -> Iterable[dict[str, Any]]:
    for request in requests:
        explicit = _canonical_ids(request["candidate_paper_ids"])
        discovered: dict[str, tuple[float, str, dict[str, Any] | None]] = {
            value: (0.25, "explicit_candidate", None) for value in explicit
        }
        for query in request["queries"]:
            for hit in adapter.search_inspire(query):
                _remember_discovery(discovered, request, hit, query, "inspire_query")
        for anchor in _canonical_ids([*seed_paper_ids, *explicit]):
            for source, values in (
                ("inspire_reference", adapter.get_references(anchor)),
                ("inspire_citer", adapter.get_citers(anchor)),
            ):
                for hit in islice(values, _MAX_GRAPH_RESULTS_PER_ANCHOR):
                    _remember_discovery(discovered, request, hit, "", source)
                if len(discovered) >= _MAX_DISCOVERED_PAPERS_PER_REQUEST:
                    break
            if len(discovered) >= _MAX_DISCOVERED_PAPERS_PER_REQUEST:
                break
        for paper_id, (score, source, discovered_metadata) in sorted(
            discovered.items(), key=lambda item: -item[1][0]
        ):
            metadata = adapter.get_metadata(paper_id) or discovered_metadata
            record = _abstract_record(request, metadata, paper_id)
            if record is not None:
                yield _candidate(
                    request,
                    record,
                    aliases=_paper_aliases(metadata or {}, paper_id),
                    relevance_score=max(score, _relevance(request, metadata or {})),
                    discovery_source=source,
                )


def _web_lane(
    requests: list[dict[str, Any]], adapter: EvidenceDiscoveryAdapter,
) -> Iterable[dict[str, Any]]:
    for request in requests:
        discoveries = [
            {"url": url, "candidate_url": url}
            for url in request["candidate_urls"]
        ]
        for query in request["queries"]:
            discoveries.extend(dict(item) for item in adapter.search_web(query))
        for hit in discoveries:
            paper_ids = _canonical_ids([
                *_value_ids(hit.get("paper_id")),
                *_value_ids(hit.get("candidate_paper_ids")),
                *_ids_from_url(str(hit.get("url") or hit.get("candidate_url") or "")),
            ])
            verified = False
            for paper_id in paper_ids:
                parsed = adapter.get_parsed_source(paper_id)
                record = _full_text_record(request, parsed, paper_id)
                metadata = adapter.get_metadata(paper_id)
                if record is None:
                    record = _abstract_record(request, metadata, paper_id)
                if record is None:
                    continue
                verified = True
                yield _candidate(
                    request,
                    record,
                    aliases=_paper_aliases(metadata or parsed or {}, paper_id),
                    relevance_score=_relevance(request, {**hit, **(metadata or {})}),
                    discovery_source="web_mapped_verified_paper",
                    verified_source=True,
                )
            if not verified:
                yield {
                    "request_key": request["request_key"],
                    "candidate_url": str(hit.get("url") or hit.get("candidate_url") or ""),
                    "snippet": str(hit.get("snippet") or ""),
                    "discovery_source": "web_discovery_only",
                    "discovery_only": True,
                }


def _candidate(
    request: dict[str, Any],
    record: dict[str, Any],
    *,
    aliases: Iterable[str] = (),
    relevance_score: float = 0.0,
    discovery_source: str = "",
    verified_source: bool = False,
) -> dict[str, Any]:
    return {
        "request_key": request["request_key"],
        "record": record,
        "canonical_aliases": _canonical_ids(aliases),
        "relevance_score": relevance_score,
        "discovery_source": discovery_source,
        "verified_source": verified_source,
    }


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


def _full_text_record(
    request: dict[str, Any], data: dict[str, Any] | None, fallback_id: str,
) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    blocks: list[dict[str, str]] = []
    for index, section in enumerate(data.get("sections") or []):
        text = str(section.get("text") or "").strip() if isinstance(section, dict) else ""
        if text:
            blocks.append({
                "block_id": str(section.get("section_id") or f"section-{index + 1}"),
                "text": text,
                "sha256": text_sha256(text),
            })
    if not blocks:
        return None
    paper_id = _preferred_paper_id(data, fallback_id)
    record = _paper_record(
        paper_id=paper_id,
        relation=request["relation"],
        title=str(data.get("title") or ""),
        authors=data.get("authors") or [],
        year=data.get("year"),
        evidence_level="full_text",
        content=blocks,
        source_hash=str(data.get("source_hash") or ""),
    )
    record["canonical_aliases"] = _paper_aliases(data, paper_id)
    return record


def _abstract_record(
    request: dict[str, Any], metadata: dict[str, Any] | None, fallback_id: str,
) -> dict[str, Any] | None:
    if not isinstance(metadata, dict):
        return None
    abstract = str(metadata.get("abstract") or "").strip()
    if not abstract:
        return None
    paper_id = _preferred_paper_id(metadata, fallback_id)
    record = _paper_record(
        paper_id=paper_id,
        relation=request["relation"],
        title=str(metadata.get("title") or ""),
        authors=metadata.get("authors") or [],
        year=metadata.get("year"),
        evidence_level="abstract_only",
        content=abstract,
    )
    record["canonical_aliases"] = _paper_aliases(metadata, paper_id)
    return record


def _remember_discovery(
    discovered: dict[str, tuple[float, str, dict[str, Any] | None]],
    request: dict[str, Any],
    hit: dict[str, Any],
    query: str,
    source: str,
) -> None:
    paper_id = _paper_id(hit)
    if not paper_id:
        return
    score = _relevance(request, hit, query=query)
    if score <= 0.0 and source != "inspire_query":
        return
    if source == "inspire_query":
        score = max(0.1, score)
    current = discovered.get(paper_id)
    if current is None or score > current[0]:
        discovered[paper_id] = (score, source, hit)


def _paper_id(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    candidates: list[Any] = [value.get("paper_id"), value.get("arxiv_id")]
    identifiers = value.get("identifiers")
    if isinstance(identifiers, dict):
        candidates.extend(
            identifiers.get(key) for key in ("arxiv", "arxiv_id", "doi", "inspire", "inspire_recid")
        )
    candidates.extend([value.get("doi"), value.get("inspire_recid")])
    aliases = _canonical_ids(item for item in candidates if item not in (None, ""))
    return _preferred_alias(aliases)


def _paper_aliases(value: dict[str, Any], fallback_id: str) -> list[str]:
    candidates: list[Any] = [fallback_id, value.get("paper_id")]
    arxiv_id = value.get("arxiv_id")
    if arxiv_id:
        candidates.append(f"arXiv:{arxiv_id}")
    doi = value.get("doi")
    if doi:
        candidates.append(f"doi:{doi}")
    inspire_recid = value.get("inspire_recid")
    if inspire_recid:
        candidates.append(f"inspire:{inspire_recid}")
    identifiers = value.get("identifiers")
    if isinstance(identifiers, dict):
        candidates.extend(identifiers.values())
    return _canonical_ids(item for item in candidates if item not in (None, ""))


def _preferred_paper_id(value: dict[str, Any], fallback_id: str) -> str:
    return _preferred_alias(_paper_aliases(value, fallback_id)) or str(fallback_id)


def _preferred_alias(aliases: Iterable[str]) -> str:
    material = list(aliases)
    for prefix in ("arXiv:", "doi:", "inspire:"):
        if found := next((item for item in material if item.startswith(prefix)), None):
            return found
    return material[0] if material else ""


def _canonical_ids(values: Iterable[Any]) -> list[str]:
    from arc_paper.ids import normalize_paper_id

    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        normalized = normalize_paper_id(text)
        # INSPIRE result dictionaries often expose the numeric recid alone.
        if text.isdigit() and normalized == text:
            normalized = f"inspire:{text}"
        key = normalized.casefold()
        if key not in seen:
            seen.add(key)
            output.append(normalized)
    return output


def _value_ids(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)] if value not in (None, "") else []


def _ids_from_url(value: str) -> list[str]:
    from arc_paper.ids import extract_paper_ids

    return extract_paper_ids(value)


def _candidate_identities(record: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    relation = str(record["relation"])
    aliases = _canonical_ids([
        *_value_ids(record.get("paper_id")),
        *_value_ids(record.get("canonical_aliases")),
        *_value_ids(candidate.get("canonical_aliases")),
    ])
    if aliases:
        return [f"paper:{relation}:{value.casefold()}" for value in aliases]
    return _source_identities(record)


def _source_identities(record: dict[str, Any]) -> list[str]:
    relation = str(record["relation"])
    aliases = _canonical_ids([
        *_value_ids(record.get("paper_id")),
        *_value_ids(record.get("canonical_aliases")),
    ])
    if aliases:
        return [f"paper:{relation}:{value.casefold()}" for value in aliases]
    descriptor = record["source_descriptor"]
    return [
        f"{descriptor['source_type']}:{relation}:{descriptor['canonical_locator']}"
    ]


def _relevance(
    request: dict[str, Any], value: Mapping[str, Any], *, query: str = "",
) -> float:
    wanted = " ".join([
        str(request.get("needed_claim") or ""),
        " ".join(request.get("queries") or []),
        query,
    ])
    observed = " ".join(
        str(value.get(key) or "")
        for key in ("title", "abstract", "snippet", "text", "section_title")
    )
    wanted_terms = _terms(wanted)
    if not wanted_terms:
        return 0.0
    overlap = wanted_terms.intersection(_terms(observed))
    return min(1.0, len(overlap) / max(1, len(wanted_terms)))


def _terms(value: str) -> set[str]:
    return {
        term.casefold()
        for term in re.findall(r"[^\W_]{3,}", str(value), flags=re.UNICODE)
    }


def _score(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _record_quality(record: Mapping[str, Any]) -> int:
    return {"full_text": 2, "abstract_only": 1}.get(
        str(record.get("evidence_level") or ""), 0,
    )


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("evidence request query and candidate fields must be arrays of strings")
    return list(dict.fromkeys(item.strip() for item in value if item.strip()))
