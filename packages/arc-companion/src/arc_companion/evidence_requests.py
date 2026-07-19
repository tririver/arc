from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
from itertools import islice
import re
import threading
from typing import Any, Callable, Iterable, Mapping, Protocol

from .evidence import (
    EvidenceProvenanceError,
    arc_cache_descriptor,
    inspire_abstract_descriptor,
    json_sha256,
    text_sha256,
    validate_evidence_record,
)


EVIDENCE_REQUEST_VERSION = "arc.companion.evidence-request.v1"
EVIDENCE_RESOLUTION_VERSION = "arc.companion.evidence-resolution.v3"
SUPPORT_ASSESSMENT_VERSION = "arc.companion.evidence-support.v1"
LANE_NAMES = ("arc", "inspire", "web")
_RELATIONS = {"prior", "later", "context"}
_MAX_GRAPH_RESULTS_PER_ANCHOR = 48
_MAX_GRAPH_ANCHORS = 24
_MAX_DISCOVERED_PAPERS_PER_REQUEST = 96
_MAX_WEB_RESULTS_PER_QUERY = 24
_MAX_QUERIES_PER_REQUEST = 8
_MAX_LANE_RESULTS = 256
_MAX_TOTAL_CANDIDATES = 512

EvidenceLane = Callable[[list[dict[str, Any]]], Iterable[dict[str, Any]]]
WebSearch = Callable[[str], Iterable[dict[str, Any]]]


class EvidenceLaneUnavailable(RuntimeError):
    """Raised when the active host has no provider for a discovery lane."""


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

    def __init__(self, *, web_search: WebSearch | None = None) -> None:
        self._web_search = web_search

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
        from arc_paper import service

        result = service.search_inspire(query, limit=20)
        data = result.get("data") if isinstance(result, dict) and result.get("ok") else None
        if data is None:
            error = result.get("error") if isinstance(result, dict) else None
            raise RuntimeError(f"INSPIRE search failed: {error or 'unknown error'}")
        return data if isinstance(data, list) else []

    def search_web(self, query: str) -> Iterable[dict[str, Any]]:
        if self._web_search is None:
            raise EvidenceLaneUnavailable("host web search provider is not configured")
        return self._web_search(query)


class CachingEvidenceDiscoveryAdapter:
    """Share completed and in-flight provider calls across all requests/lanes."""

    def __init__(self, backend: EvidenceDiscoveryAdapter) -> None:
        self._backend = backend
        self._lock = threading.Lock()
        self._calls: dict[tuple[Any, ...], Future[tuple[dict[str, Any], ...] | dict[str, Any] | None]] = {}

    def search_arc_full_text(self, paper_ids: list[str], query: str) -> Iterable[dict[str, Any]]:
        return self._many(
            ("arc_search", tuple(paper_ids), query),
            lambda: self._backend.search_arc_full_text(paper_ids, query),
        )

    def get_parsed_source(self, paper_id: str) -> dict[str, Any] | None:
        return self._one(("parsed", paper_id), lambda: self._backend.get_parsed_source(paper_id))

    def get_metadata(self, paper_id: str) -> dict[str, Any] | None:
        return self._one(("metadata", paper_id), lambda: self._backend.get_metadata(paper_id))

    def get_references(self, paper_id: str) -> Iterable[dict[str, Any]]:
        return self._many(
            ("references", paper_id),
            lambda: islice(self._backend.get_references(paper_id), _MAX_GRAPH_RESULTS_PER_ANCHOR),
        )

    def get_citers(self, paper_id: str) -> Iterable[dict[str, Any]]:
        return self._many(
            ("citers", paper_id),
            lambda: islice(self._backend.get_citers(paper_id), _MAX_GRAPH_RESULTS_PER_ANCHOR),
        )

    def search_inspire(self, query: str) -> Iterable[dict[str, Any]]:
        return self._many(
            ("inspire_search", query),
            lambda: islice(self._backend.search_inspire(query), _MAX_DISCOVERED_PAPERS_PER_REQUEST),
        )

    def search_web(self, query: str) -> Iterable[dict[str, Any]]:
        return self._many(
            ("web_search", query),
            lambda: islice(self._backend.search_web(query), _MAX_WEB_RESULTS_PER_QUERY),
        )

    def _many(
        self, key: tuple[Any, ...], load: Callable[[], Iterable[dict[str, Any]]],
    ) -> tuple[dict[str, Any], ...]:
        value = self._memo(key, lambda: tuple(dict(item) for item in load()))
        return value if isinstance(value, tuple) else ()

    def _one(
        self, key: tuple[Any, ...], load: Callable[[], dict[str, Any] | None],
    ) -> dict[str, Any] | None:
        value = self._memo(key, lambda: dict(found) if (found := load()) is not None else None)
        return value if isinstance(value, dict) else None

    def _memo(
        self,
        key: tuple[Any, ...],
        load: Callable[[], tuple[dict[str, Any], ...] | dict[str, Any] | None],
    ) -> tuple[dict[str, Any], ...] | dict[str, Any] | None:
        owner = False
        with self._lock:
            future = self._calls.get(key)
            if future is None:
                future = Future()
                self._calls[key] = future
                owner = True
        if owner:
            try:
                future.set_result(load())
            except BaseException as exc:
                future.set_exception(exc)
        return future.result()


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
                    output = _consume_lane_fairly(
                        future.result(), list(request_by_key), limit=_MAX_LANE_RESULTS,
                    )
                except EvidenceLaneUnavailable as exc:
                    lane_outputs[name] = []
                    audit_lanes[name] = {
                        "status": "unavailable",
                        "reason": str(exc),
                        "raw_results": [],
                    }
                except Exception as exc:  # one failed lane never cancels another
                    lane_outputs[name] = []
                    audit_lanes[name] = {
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "raw_results": [],
                    }
                else:
                    unavailable = [item for item in output if item.get("lane_unavailable")]
                    usable = [item for item in output if not item.get("lane_unavailable")]
                    lane_outputs[name] = usable
                    audit_lanes[name] = {
                        "status": (
                            "unavailable" if unavailable else "complete"
                        ),
                        "raw_results": [_audit_candidate(item) for item in output],
                        "truncated": len(output) == _MAX_LANE_RESULTS,
                    }
                    if unavailable:
                        audit_lanes[name]["reason"] = str(unavailable[0].get("reason") or "")

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
        changed_ids: set[str] = set()
        ranked_candidates = _fair_ranked_candidates(
            material, lane_outputs, limit=_MAX_TOTAL_CANDIDATES,
        )
        for lane_name, candidate in ranked_candidates:
            request_key = str(candidate.get("request_key") or "")
            request = request_by_key.get(request_key)
            base = {
                "lane": lane_name,
                "request_key": request_key,
                "relevance_score": _score(candidate.get("relevance_score")),
                "discovery_source": str(candidate.get("discovery_source") or ""),
                "discovery_origin": str(candidate.get("discovery_origin") or ""),
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
            if lane_name == "web" and candidate.get("verified_source") is not True:
                rejected.append({**base, "reason": "web_snippet_not_claim_evidence"})
                continue
            if (
                lane_name == "inspire"
                and record.get("evidence_level") == "abstract_only"
                and descriptor.get("source_type") != "inspire_record"
            ):
                rejected.append({**base, "reason": "unverified_discovery_abstract"})
                continue
            if str(record.get("relation") or "") != request["relation"]:
                rejected.append({**base, "reason": "relation_mismatch"})
                continue
            support = _direct_support_assessment(request, record)
            if not support["supported"]:
                rejected.append({
                    **base,
                    "reason": "insufficient_direct_support",
                    "support_assessment": support,
                })
                continue
            record = {
                **record,
                "supported_request_keys": list(dict.fromkeys([
                    *(record.get("supported_request_keys") or []), request_key,
                ])),
            }
            identities = _candidate_identities(record, candidate)
            evidence_id = next((alias_to_id[value] for value in identities if value in alias_to_id), None)
            identity = identities[0]
            if evidence_id is None:
                evidence_id = str(record["evidence_id"])
                registry.append(record)
                registry_index[evidence_id] = len(registry) - 1
                created_ids.add(evidence_id)
                changed_ids.add(evidence_id)
                accepted.append({
                    **base, "evidence_id": evidence_id, "identity": identity,
                    "support_assessment": support,
                })
            else:
                upgraded = False
                index = registry_index[evidence_id]
                existing = registry[index]
                supported_request_keys = list(dict.fromkeys([
                    *(existing.get("supported_request_keys") or []), request_key,
                ]))
                if _record_quality(record) > _record_quality(existing):
                    replacement = {
                        **record, "evidence_id": evidence_id,
                        "supported_request_keys": supported_request_keys,
                    }
                    upgraded = True
                else:
                    replacement = {**existing, "supported_request_keys": supported_request_keys}
                registry[index] = replacement
                changed_ids.add(evidence_id)
                accepted.append({
                    **base,
                    "evidence_id": evidence_id,
                    "identity": identity,
                    "deduplicated": True,
                    "upgraded_to_preferred_source": upgraded,
                    "support_assessment": support,
                })
            for value in identities:
                alias_to_id[value] = evidence_id
            segment_id = str(request["segment_id"])
            ids_by_segment.setdefault(segment_id, []).append(evidence_id)
            supported_keys.add(request_key)

        new_records = tuple(item for item in registry if item["evidence_id"] in changed_ids)
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
                "claim_evidence_policy": "verified_arc_full_text_or_inspire_abstract",
                "candidate_limit": _MAX_TOTAL_CANDIDATES,
                "per_request": {
                    key: {
                        "considered": sum(
                            str(item.get("request_key") or "") == key
                            for _, item in ranked_candidates
                        ),
                        "accepted": sum(item.get("request_key") == key for item in accepted),
                        "rejected": sum(item.get("request_key") == key for item in rejected),
                    }
                    for key in request_by_key
                },
            },
        )


def _consume_lane_fairly(
    values: Iterable[dict[str, Any]], request_keys: list[str], *, limit: int,
) -> list[dict[str, Any]]:
    """Scan a bounded per-request envelope, then emit stable round-robin results."""
    if not request_keys:
        return []
    per_request = max(1, (limit + len(request_keys) - 1) // len(request_keys))
    buckets = {key: [] for key in request_keys}
    scan_limit = max(limit, _MAX_DISCOVERED_PAPERS_PER_REQUEST) * len(request_keys)
    for raw in islice(values, scan_limit):
        item = dict(raw)
        key = str(item.get("request_key") or "")
        bucket = buckets.setdefault(key, [])
        if len(bucket) < per_request:
            bucket.append(item)
    output: list[dict[str, Any]] = []
    while len(output) < limit:
        progressed = False
        for key in request_keys:
            if buckets.get(key):
                output.append(buckets[key].pop(0))
                progressed = True
                if len(output) >= limit:
                    break
        if not progressed:
            break
    return output


def _fair_ranked_candidates(
    requests: list[dict[str, Any]],
    lane_outputs: Mapping[str, list[dict[str, Any]]],
    *,
    limit: int,
) -> list[tuple[str, dict[str, Any]]]:
    buckets: dict[str, list[tuple[str, dict[str, Any]]]] = {
        str(request["request_key"]): [] for request in requests
    }
    for lane_name in LANE_NAMES:
        for candidate in lane_outputs.get(lane_name, []):
            key = str(candidate.get("request_key") or "")
            buckets.setdefault(key, []).append((lane_name, candidate))
    for values in buckets.values():
        values.sort(key=lambda item: (
            -_score(item[1].get("relevance_score")), LANE_NAMES.index(item[0]),
            str(item[1].get("discovery_source") or ""),
        ))
    order = [str(request["request_key"]) for request in requests]
    output: list[tuple[str, dict[str, Any]]] = []
    while len(output) < limit:
        progressed = False
        for key in order:
            if buckets.get(key):
                output.append(buckets[key].pop(0))
                progressed = True
                if len(output) >= limit:
                    break
        if not progressed:
            break
    return output


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
    backend = CachingEvidenceDiscoveryAdapter(adapter or ArcPaperEvidenceAdapter())
    domain_ids = _canonical_ids(domain_paper_ids)
    seed_ids = _canonical_ids(seed_paper_ids)
    return {
        "arc": lambda requests: _fair_request_results(
            requests, lambda request: _arc_lane([request], backend, domain_ids),
        ),
        "inspire": lambda requests: _fair_request_results(
            requests, lambda request: _inspire_lane([request], backend, seed_ids),
        ),
        "web": lambda requests: _fair_request_results(
            requests, lambda request: _web_lane([request], backend),
        ),
    }


def _fair_request_results(
    requests: list[dict[str, Any]],
    producer: Callable[[dict[str, Any]], Iterable[dict[str, Any]]],
) -> Iterable[dict[str, Any]]:
    """Round-robin bounded request iterators so early requests cannot exhaust a lane."""
    if not requests:
        return
    per_request = max(1, _MAX_LANE_RESULTS // len(requests))
    iterators = [(iter(producer(request)), 0) for request in requests]
    emitted = 0
    while iterators and emitted < _MAX_LANE_RESULTS:
        next_round: list[tuple[Iterable[dict[str, Any]], int]] = []
        for iterator, count in iterators:
            if count >= per_request or emitted >= _MAX_LANE_RESULTS:
                continue
            try:
                item = next(iterator)
            except StopIteration:
                continue
            yield item
            emitted += 1
            next_round.append((iterator, count + 1))
        iterators = next_round


def _arc_lane(
    requests: list[dict[str, Any]],
    adapter: EvidenceDiscoveryAdapter,
    domain_paper_ids: list[str],
) -> Iterable[dict[str, Any]]:
    for request in requests:
        explicit = _canonical_ids(request["candidate_paper_ids"])[
            :_MAX_DISCOVERED_PAPERS_PER_REQUEST
        ]
        pool = _canonical_ids([*domain_paper_ids, *explicit])
        discovered: dict[str, tuple[float, str]] = {
            paper_id: (0.25 if paper_id in explicit else 0.05, "explicit_candidate")
            for paper_id in explicit
        }
        for query in request["queries"][:_MAX_QUERIES_PER_REQUEST]:
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
                if len(discovered) >= _MAX_DISCOVERED_PAPERS_PER_REQUEST:
                    break
            if len(discovered) >= _MAX_DISCOVERED_PAPERS_PER_REQUEST:
                break
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
        explicit = _canonical_ids(request["candidate_paper_ids"])[
            :_MAX_DISCOVERED_PAPERS_PER_REQUEST
        ]
        discovered: dict[str, tuple[float, str, dict[str, Any] | None]] = {
            value: (0.25, "explicit_candidate", None) for value in explicit
        }
        for query in request["queries"][:_MAX_QUERIES_PER_REQUEST]:
            for hit in islice(adapter.search_inspire(query), _MAX_DISCOVERED_PAPERS_PER_REQUEST):
                _remember_discovery(discovered, request, hit, query, "inspire_query")
                if len(discovered) >= _MAX_DISCOVERED_PAPERS_PER_REQUEST:
                    break
            if len(discovered) >= _MAX_DISCOVERED_PAPERS_PER_REQUEST:
                break
        for anchor in _canonical_ids([*seed_paper_ids, *explicit])[:_MAX_GRAPH_ANCHORS]:
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
        for paper_id, (score, source, _discovery_metadata) in sorted(
            discovered.items(), key=lambda item: -item[1][0]
        ):
            # Search and graph payloads are discovery hints.  Registration is
            # allowed only after a fresh provider-level metadata verification.
            metadata = adapter.get_metadata(paper_id)
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
            {
                "url": url, "candidate_url": url,
                "discovery_origin": "annotation_agent",
            }
            for url in request["candidate_urls"][:_MAX_DISCOVERED_PAPERS_PER_REQUEST]
        ]
        search_unavailable: str | None = None
        for query in request["queries"][:_MAX_QUERIES_PER_REQUEST]:
            try:
                discoveries.extend(
                    dict(item)
                    for item in islice(adapter.search_web(query), _MAX_WEB_RESULTS_PER_QUERY)
                )
            except EvidenceLaneUnavailable as exc:
                search_unavailable = str(exc)
                break
            if len(discoveries) >= _MAX_DISCOVERED_PAPERS_PER_REQUEST:
                break
        for hit in discoveries[:_MAX_DISCOVERED_PAPERS_PER_REQUEST]:
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
                    discovery_origin=str(hit.get("discovery_origin") or "host_web_provider"),
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
        if search_unavailable is not None:
            yield {
                "request_key": request["request_key"],
                "lane_unavailable": True,
                "reason": search_unavailable,
                "discovery_source": "web_search_provider_unavailable",
            }


def _candidate(
    request: dict[str, Any],
    record: dict[str, Any],
    *,
    aliases: Iterable[str] = (),
    relevance_score: float = 0.0,
    discovery_source: str = "",
    discovery_origin: str = "",
    verified_source: bool = False,
) -> dict[str, Any]:
    return {
        "request_key": request["request_key"],
        "record": record,
        "canonical_aliases": _canonical_ids(aliases),
        "relevance_score": relevance_score,
        "discovery_source": discovery_source,
        "discovery_origin": discovery_origin,
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
    record["source_descriptor"] = inspire_abstract_descriptor(
        paper_id=paper_id,
        title=record["title"],
        authors=record["authors"],
        year=record["year"],
        abstract=abstract,
    )
    validate_evidence_record(record)
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


def _direct_support_assessment(
    request: Mapping[str, Any], record: Mapping[str, Any],
) -> dict[str, Any]:
    """Conservatively require concentrated overlap with one verified source piece."""
    wanted = _support_terms(" ".join([
        str(request.get("needed_claim") or ""),
        " ".join(str(item) for item in request.get("queries") or []),
    ]))
    pieces: list[tuple[str, str]] = []
    title = str(record.get("title") or "").strip()
    if title:
        pieces.append(("title", title))
    abstract = str(record.get("abstract") or "").strip()
    if abstract:
        pieces.append(("abstract", abstract))
    for index, block in enumerate(record.get("blocks") or [], 1):
        if not isinstance(block, Mapping):
            continue
        text = str(block.get("text") or "").strip()
        if not text:
            continue
        locator = str(block.get("locator") or block.get("block_id") or f"block-{index}")
        words = re.findall(r"\S+", text)
        for start in range(0, len(words), 64):
            excerpt = " ".join(words[start:start + 128])
            if excerpt:
                pieces.append((locator, excerpt))
            if start + 128 >= len(words):
                break
    best: dict[str, Any] | None = None
    for locator, text in pieces:
        observed = _support_terms(text)
        shared = wanted & observed
        if len(shared) < 2 or not wanted or not observed:
            continue
        wanted_coverage = len(shared) / len(wanted)
        piece_coverage = len(shared) / len(observed)
        cosine = len(shared) / ((len(wanted) * len(observed)) ** 0.5)
        supported = (
            wanted_coverage >= 0.25
            and piece_coverage >= 0.08
            and cosine >= 0.20
        )
        score = min(wanted_coverage, piece_coverage) + cosine
        candidate = {
            "version": SUPPORT_ASSESSMENT_VERSION,
            "supported": supported,
            "score": round(score, 6),
            "locator": locator,
            "matched_terms": sorted(shared)[:16],
            "wanted_coverage": round(wanted_coverage, 6),
            "piece_coverage": round(piece_coverage, 6),
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    return best or {
        "version": SUPPORT_ASSESSMENT_VERSION,
        "supported": False,
        "score": 0.0,
        "locator": "",
        "matched_terms": [],
        "wanted_coverage": 0.0,
        "piece_coverage": 0.0,
    }


def _support_terms(value: str) -> set[str]:
    generic = {
        "about", "after", "also", "and", "before", "claim",
        "context", "field", "fields", "for", "from", "general",
        "into", "model", "models", "paper", "result", "results",
        "study", "that", "the", "their", "theory", "these", "this", "those",
        "using", "with", "work",
    }
    return {term for term in _terms(value) if term not in generic}


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


def _audit_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Return an auditable summary without embedding evidence payloads."""
    record = candidate.get("record")
    summary = {
        "request_key": str(candidate.get("request_key") or ""),
        "discovery_source": str(candidate.get("discovery_source") or ""),
        "discovery_origin": str(candidate.get("discovery_origin") or ""),
        "relevance_score": _score(candidate.get("relevance_score")),
        "canonical_aliases": _canonical_ids(_value_ids(candidate.get("canonical_aliases"))),
    }
    if not isinstance(record, dict):
        snippet = str(candidate.get("snippet") or "")
        return {
            **summary,
            "candidate_url": str(candidate.get("candidate_url") or ""),
            "discovery_only": True,
            "excerpt": snippet[:240],
            "content_sha256": text_sha256(snippet) if snippet else "",
        }
    descriptor = record.get("source_descriptor")
    descriptor = descriptor if isinstance(descriptor, dict) else {}
    excerpt = str(record.get("abstract") or "")
    if not excerpt:
        blocks = record.get("blocks")
        if isinstance(blocks, list) and blocks and isinstance(blocks[0], dict):
            excerpt = str(blocks[0].get("text") or "")
    return {
        **summary,
        "paper_id": str(record.get("paper_id") or ""),
        "evidence_level": str(record.get("evidence_level") or ""),
        "source_type": str(descriptor.get("source_type") or ""),
        "canonical_locator": str(descriptor.get("canonical_locator") or ""),
        "content_sha256": str(descriptor.get("content_sha256") or json_sha256(record)),
        "excerpt": excerpt[:240],
    }


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("evidence request query and candidate fields must be arrays of strings")
    return list(dict.fromkeys(item.strip() for item in value if item.strip()))
