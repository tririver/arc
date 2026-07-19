from __future__ import annotations

from pathlib import Path
import threading

import jsonschema
import pytest

from arc_companion.evidence import arc_cache_descriptor, text_sha256, web_evidence_record
from arc_companion.evidence_requests import (
    EvidenceRequestController,
    EvidenceResolution,
    normalize_evidence_requests,
)
from arc_companion.pipeline import BuildOptions, _resolve_and_rerun_evidence_requests, _review
from arc_companion.prompts import ANNOTATION_SCHEMA
from arc_companion.source import SourceBundle


def _request(relation: str = "prior") -> dict:
    return {
        "relation": relation,
        "needed_claim": "A concrete historical claim.",
        "queries": ["specific mechanism"],
        "candidate_paper_ids": ["arXiv:1234.5678"],
        "candidate_urls": ["https://example.test/discovery"],
        "reason": "The source passage relies on this antecedent.",
    }


def _record(relation: str = "prior") -> dict:
    blocks = [{"block_id": "S1", "text": "Verified paper passage.", "sha256": text_sha256("Verified paper passage.")}]
    record = {
        "evidence_id": "verified-paper",
        "relation": relation,
        "paper_id": "arXiv:1234.5678",
        "title": "Verified paper",
        "authors": [],
        "year": 2020,
        "evidence_level": "full_text",
        "abstract": "",
        "blocks": blocks,
    }
    record["source_descriptor"] = arc_cache_descriptor(
        paper_id=record["paper_id"], title=record["title"], authors=[], year=2020,
        evidence_level="full_text", content=blocks, document_hash="d" * 64,
    )
    return record


def test_annotation_schema_and_controller_enforce_two_strict_requests() -> None:
    base = {
        "explanation": "Explanation", "prior_work": "", "later_work": "",
        "commentary": "Commentary", "evidence_ids": [], "key_points": [], "source_notes": [],
    }
    for count in (0, 1, 2):
        jsonschema.validate({**base, "evidence_requests": [_request() for _ in range(count)]}, ANNOTATION_SCHEMA)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**base, "evidence_requests": [_request() for _ in range(3)]}, ANNOTATION_SCHEMA)
    with pytest.raises(ValueError, match="at most two"):
        normalize_evidence_requests("seg", [_request(), _request(), _request()])
    invalid = _request()
    invalid["relation"] = "remembered"
    with pytest.raises(ValueError, match="requires relation"):
        normalize_evidence_requests("seg", [invalid])
    claim = {
        "text": "Located claim", "evidence_ids": ["paper-1"],
        "source_locators": [{"evidence_id": "paper-1", "locator": "S1"}],
        "request_key": None,
    }
    jsonschema.validate({
        **base, "prior_work": [claim], "evidence_ids": ["paper-1"],
        "evidence_requests": [],
    }, ANNOTATION_SCHEMA)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({
            **base, "prior_work": [claim] * 4, "evidence_ids": ["paper-1"],
            "evidence_requests": [],
        }, ANNOTATION_SCHEMA)


def test_controller_runs_all_lanes_deduplicates_and_never_registers_web_snippet() -> None:
    calls: list[str] = []
    lock = threading.Lock()
    normalized = normalize_evidence_requests("seg-1", [_request()])
    web_record = web_evidence_record(
        relation="prior", url="https://example.test/discovery", title="Snippet",
        excerpt="Search result snippet.", retrieved_at="2026-07-19T12:00:00Z",
    )

    def lane(name, output):
        def run(requests):
            with lock:
                calls.append(name)
            assert requests == normalized
            return output
        return run

    envelope = {"request_key": normalized[0]["request_key"], "record": _record()}
    controller = EvidenceRequestController({
        "arc": lane("arc", [envelope]),
        "inspire": lane("inspire", [envelope]),
        "web": lane("web", [{"request_key": normalized[0]["request_key"], "record": web_record}]),
    })
    result = controller.resolve(normalized)

    assert sorted(calls) == ["arc", "inspire", "web"]
    assert [item["evidence_id"] for item in result.records] == ["verified-paper"]
    assert result.evidence_ids_by_segment == {"seg-1": ("verified-paper",)}
    assert any(item.get("deduplicated") for item in result.audit["accepted"])
    assert any(item["reason"] == "web_snippet_not_claim_evidence" for item in result.audit["rejected"])
    assert all(result.audit["lanes"][name]["status"] == "complete" for name in ("arc", "inspire", "web"))


def test_lane_failure_does_not_cancel_other_lanes() -> None:
    normalized = normalize_evidence_requests("seg-1", [_request()])

    def failed(_requests):
        raise RuntimeError("offline")

    envelope = {"request_key": normalized[0]["request_key"], "record": _record()}
    result = EvidenceRequestController({
        "arc": lambda requests: [envelope],
        "inspire": failed,
        "web": lambda requests: [],
    }).resolve(normalized)

    assert result.records
    assert result.audit["lanes"]["inspire"]["status"] == "failed"
    assert result.audit["lanes"]["arc"]["status"] == "complete"


class _DiscoveryFixture:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.metadata = {
            "arXiv:1111.1111": {
                "paper_id": "arXiv:1111.1111", "arxiv_id": "1111.1111",
                "title": "Broad domain survey", "abstract": "A broad mechanism survey.",
                "authors": [], "year": 2011,
            },
            "arXiv:2222.2222": {
                "paper_id": "arXiv:2222.2222", "arxiv_id": "2222.2222",
                "doi": "10.1000/direct", "inspire_recid": "22",
                "title": "Specific mechanism calculation",
                "abstract": "A specific mechanism calculation proves the concrete historical claim.",
                "authors": [], "year": 2022,
            },
            "doi:10.1000/direct": {
                "paper_id": "arXiv:2222.2222", "arxiv_id": "2222.2222",
                "doi": "10.1000/direct", "inspire_recid": "22",
                "title": "Specific mechanism calculation",
                "abstract": "A specific mechanism calculation proves the concrete historical claim.",
                "authors": [], "year": 2022,
            },
            "inspire:22": {
                "paper_id": "arXiv:2222.2222", "arxiv_id": "2222.2222",
                "doi": "10.1000/direct", "inspire_recid": "22",
                "title": "Specific mechanism calculation",
                "abstract": "A specific mechanism calculation proves the concrete historical claim.",
                "authors": [], "year": 2022,
            },
        }

    def search_arc_full_text(self, paper_ids, query):
        self.calls.append(("arc_search", tuple(paper_ids), query))
        return [{
            "paper_id": "arXiv:1111.1111", "title": "Broad domain survey",
            "snippet": "broad mechanism",
        }]

    def get_parsed_source(self, paper_id):
        self.calls.append(("parsed", paper_id))
        metadata = self.metadata.get(paper_id)
        if not metadata:
            return None
        return {
            **metadata, "source_hash": "a" * 64,
            "sections": [{
                "section_id": "S1",
                "text": metadata["abstract"],
            }],
        }

    def get_metadata(self, paper_id):
        self.calls.append(("metadata", paper_id))
        return self.metadata.get(paper_id)

    def get_references(self, paper_id):
        self.calls.append(("references", paper_id))
        return [{"paper_id": "doi:10.1000/direct", **self.metadata["doi:10.1000/direct"]}]

    def get_citers(self, paper_id):
        self.calls.append(("citers", paper_id))
        return []

    def search_inspire(self, query):
        self.calls.append(("inspire_search", query))
        return [{"paper_id": "doi:10.1000/direct", **self.metadata["doi:10.1000/direct"]}]

    def search_web(self, query):
        self.calls.append(("web_search", query))
        return [{
            "url": "https://inspirehep.net/literature/22",
            "paper_id": "inspire:22",
            "snippet": "specific mechanism calculation",
        }]


def test_default_lanes_consume_queries_and_expand_beyond_domain_without_short_circuit() -> None:
    adapter = _DiscoveryFixture()
    normalized = normalize_evidence_requests("seg-1", [_request()])
    controller = EvidenceRequestController(
        adapter=adapter,
        domain_paper_ids=["arXiv:1111.1111"],
        seed_paper_ids=["arXiv:9999.9999"],
    )

    result = controller.resolve(normalized)

    assert ("arc_search", ("arXiv:1111.1111", "arXiv:1234.5678"), "specific mechanism") in adapter.calls
    assert ("inspire_search", "specific mechanism") in adapter.calls
    assert ("web_search", "specific mechanism") in adapter.calls
    assert ("references", "arXiv:9999.9999") in adapter.calls
    assert ("citers", "arXiv:9999.9999") in adapter.calls
    # The more claim-specific paper discovered outside the domain is ranked
    # ahead of the broad domain hit; domain membership is only a soft signal.
    assert result.records[0]["paper_id"] == "arXiv:2222.2222"
    assert {record["paper_id"] for record in result.records} == {
        "arXiv:1111.1111", "arXiv:2222.2222",
    }
    assert all(result.audit["lanes"][name]["status"] == "complete" for name in ("arc", "inspire", "web"))


def test_arxiv_doi_and_inspire_aliases_register_once_across_three_lanes() -> None:
    adapter = _DiscoveryFixture()
    request = _request()
    request["candidate_paper_ids"] = ["arXiv:2222.2222"]
    normalized = normalize_evidence_requests("seg-1", [request])

    result = EvidenceRequestController(
        adapter=adapter, seed_paper_ids=["arXiv:9999.9999"],
    ).resolve(normalized)

    matching = [record for record in result.records if record["paper_id"] == "arXiv:2222.2222"]
    assert len(matching) == 1
    assert matching[0]["evidence_level"] == "full_text"
    accepted = [item for item in result.audit["accepted"] if "2222.2222" in item["identity"]]
    assert {item["lane"] for item in accepted} == {"arc", "inspire", "web"}
    assert len({item["evidence_id"] for item in accepted}) == 1


def test_same_canonical_paper_keeps_prior_and_later_relation_bindings_separate() -> None:
    prior, later = normalize_evidence_requests("seg-1", [_request("prior"), _request("later")])
    prior_record = _record("prior")
    later_record = _record("later")
    later_record["evidence_id"] = "verified-paper-later"

    result = EvidenceRequestController({
        "arc": lambda requests: [
            {"request_key": prior["request_key"], "record": prior_record,
             "canonical_aliases": ["arXiv:1234.5678"]},
            {"request_key": later["request_key"], "record": later_record,
             "canonical_aliases": ["doi:10.1000/alias", "arXiv:1234.5678"]},
        ],
        "inspire": lambda requests: [],
        "web": lambda requests: [],
    }).resolve([prior, later])

    assert len(result.records) == 2
    assert result.evidence_ids_by_segment["seg-1"] == (
        "verified-paper", "verified-paper-later",
    )
    assert {record["relation"] for record in result.records} == {"prior", "later"}


def test_unmapped_web_snippet_is_discovery_only() -> None:
    adapter = _DiscoveryFixture()
    adapter.search_web = lambda query: [{
        "url": "https://example.test/result", "snippet": "unsupported technical assertion",
    }]
    normalized = normalize_evidence_requests("seg-1", [_request()])
    result = EvidenceRequestController(adapter=adapter).resolve(normalized)

    assert all(record["source_descriptor"]["source_type"] == "arc_cache" for record in result.records)
    assert any(
        item["lane"] == "web" and item["reason"] == "discovery_only_not_claim_evidence"
        for item in result.audit["rejected"]
    )


def test_resolution_reruns_only_segments_with_registered_evidence_once(monkeypatch, tmp_path: Path) -> None:
    segments = [
        {"segment_id": "s1", "block_ids": ["b1"]},
        {"segment_id": "s2", "block_ids": ["b2"]},
        {"segment_id": "s3", "block_ids": ["b3"]},
    ]
    requests = normalize_evidence_requests("s2", [_request()])
    unresolved = normalize_evidence_requests("s3", [_request("later")])
    annotations = {
        "s1": {"commentary": "one", "explanation": "one", "prior_work": "", "later_work": "", "evidence_ids": [], "key_points": [], "source_notes": [], "evidence_requests": []},
        "s2": {"commentary": "two", "explanation": "two", "prior_work": "", "later_work": "", "evidence_ids": [], "key_points": [], "source_notes": [], "evidence_requests": requests},
        "s3": {"commentary": "three", "explanation": "three", "prior_work": "", "later_work": "unverified", "evidence_ids": [], "key_points": [], "source_notes": [], "evidence_requests": unresolved},
    }
    record = _record()

    class Controller:
        def resolve(self, material, *, existing_records):
            assert {item["segment_id"] for item in material} == {"s2", "s3"}
            return EvidenceResolution(
                records=(record,), evidence_ids_by_segment={"s2": ("verified-paper",)},
                supported_request_keys=(requests[0]["request_key"],),
                audit={"schema_version": "arc.companion.evidence-resolution.v1", "requests": material, "lanes": {}, "accepted": [{
                    "request_key": requests[0]["request_key"], "evidence_id": "verified-paper",
                }], "rejected": []},
            )

    calls = []

    def rerun(selected, **kwargs):
        calls.append(([item["segment_id"] for item in selected], kwargs["round_number"]))
        return {"s2": {
            "commentary": "revised", "explanation": "revised", "prior_work": [{
                "text": "supported", "evidence_ids": ["verified-paper"],
                "source_locators": [{"evidence_id": "verified-paper", "locator": "S1"}],
                "request_key": requests[0]["request_key"],
            }],
            "later_work": "", "evidence_ids": ["verified-paper"], "key_points": [],
            "source_notes": [], "evidence_requests": [],
        }}

    monkeypatch.setattr("arc_companion.pipeline._generate_annotations", rerun)
    document = {"blocks": [{"block_id": f"b{i}", "type": "text", "text": str(i)} for i in range(1, 4)]}
    bundle = SourceBundle("arXiv:1", {"paper_id": "arXiv:1"}, document, {}, [], [])
    final, merged = _resolve_and_rerun_evidence_requests(
        segments, annotations, options=BuildOptions("arXiv:1", tmp_path), bundle=bundle,
        evidence={"related_papers": []}, domain_context=None, glossary={}, protected_names=[],
        checkpoint_dir=tmp_path, llm=lambda *args, **kwargs: {}, controller=Controller(),
    )

    assert calls == [(["s2"], 2)]
    assert final["s1"]["commentary"] == "one"
    assert final["s2"]["prior_work"][0]["text"] == "supported"
    assert final["s3"]["later_work"] == ""
    assert all(not item["evidence_requests"] for item in final.values())
    assert [item["evidence_id"] for item in merged["related_papers"]] == ["verified-paper"]
    audit = __import__("json").loads((tmp_path / "evidence-resolution.v1.json").read_text())
    assert audit["rerun_segments"] == ["s2"]
    assert audit["final_claim_evidence_ids"]["s2"] == ["verified-paper"]
    assert audit["final_claim_bindings"]["s2"][0]["source_locators"] == [
        {"evidence_id": "verified-paper", "locator": "S1"}
    ]


def test_rerun_ignoring_request_evidence_drops_only_that_claim(monkeypatch, tmp_path: Path) -> None:
    segment = {"segment_id": "s1", "block_ids": ["b1"]}
    request = normalize_evidence_requests("s1", [_request()])[0]
    generic = _record()
    generic["evidence_id"] = "generic-prior"
    resolved = _record()
    first_claim = {
        "text": "Existing supported context.", "evidence_ids": ["generic-prior"],
        "source_locators": [{"evidence_id": "generic-prior", "locator": "S1"}],
        "request_key": None,
    }
    annotations = {"s1": {
        "commentary": "one", "explanation": "one", "prior_work": [first_claim],
        "later_work": "", "evidence_ids": ["generic-prior"], "key_points": [],
        "source_notes": [], "evidence_requests": [request],
    }}

    class Controller:
        def resolve(self, material, *, existing_records):
            return EvidenceResolution(
                records=(resolved,), evidence_ids_by_segment={"s1": ("verified-paper",)},
                supported_request_keys=(request["request_key"],),
                audit={
                    "schema_version": "arc.companion.evidence-resolution.v1",
                    "requests": material, "lanes": {}, "rejected": [],
                    "accepted": [{
                        "request_key": request["request_key"],
                        "evidence_id": "verified-paper",
                    }],
                },
            )

    def rerun(_selected, **_kwargs):
        return {"s1": {
            "commentary": "revised", "explanation": "revised",
            "prior_work": [
                first_claim,
                {
                    "text": "Requested but bound to unrelated same-relation evidence.",
                    "evidence_ids": ["generic-prior"],
                    "source_locators": [{"evidence_id": "generic-prior", "locator": "S1"}],
                    "request_key": request["request_key"],
                },
            ],
            "later_work": "", "evidence_ids": ["generic-prior"], "key_points": [],
            "source_notes": [], "evidence_requests": [],
        }}

    monkeypatch.setattr("arc_companion.pipeline._generate_annotations", rerun)
    document = {"blocks": [{"block_id": "b1", "type": "text", "text": "Verified"}]}
    bundle = SourceBundle("arXiv:1", {"paper_id": "arXiv:1"}, document, {}, [], [])
    final, _ = _resolve_and_rerun_evidence_requests(
        [segment], annotations, options=BuildOptions("arXiv:1", tmp_path), bundle=bundle,
        evidence={"related_papers": [generic]}, domain_context=None, glossary={},
        protected_names=[], checkpoint_dir=tmp_path, llm=lambda *args, **kwargs: {},
        controller=Controller(),
    )

    assert final["s1"]["prior_work"] == [first_claim]
    assert final["s1"]["evidence_ids"] == ["generic-prior"]


def test_review_cannot_add_related_work_from_relation_level_id(tmp_path: Path) -> None:
    record = _record()
    segment = {"segment_id": "s1", "block_ids": ["b1"]}
    annotations = {"s1": {
        "commentary": "commentary", "explanation": "explanation",
        "prior_work": "", "later_work": "", "evidence_ids": [],
        "key_points": [], "source_notes": [], "evidence_requests": [],
    }}

    def reviewer(_prompt, **_kwargs):
        return {"patches": [{
            "segment_id": "s1", "translation_blocks": None,
            "commentary": None, "explanation": None,
            "prior_work": "Invented prior-work fact.", "later_work": None,
            "evidence_ids": ["verified-paper"], "reason": "invent",
        }], "issues": []}

    with pytest.raises(RuntimeError, match="review added a related-work claim"):
        _review(
            [segment], {"s1": {"blocks": [{"block_id": "b1", "text": "译文"}]}},
            annotations,
            document={"blocks": [{"block_id": "b1", "type": "text", "text": "Verified paper passage."}]},
            glossary={"entries": []}, protected_names=[],
            evidence={"related_papers": [record]},
            options=BuildOptions("arXiv:1", tmp_path, review_context_chars=100_000),
            llm=reviewer, checkpoint_dir=tmp_path,
        )
