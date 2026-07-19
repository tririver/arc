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
from arc_companion.pipeline import BuildOptions, _resolve_and_rerun_evidence_requests
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
                audit={"schema_version": "arc.companion.evidence-resolution.v1", "requests": material, "lanes": {}, "accepted": [], "rejected": []},
            )

    calls = []

    def rerun(selected, **kwargs):
        calls.append(([item["segment_id"] for item in selected], kwargs["round_number"]))
        return {"s2": {
            "commentary": "revised", "explanation": "revised", "prior_work": "supported",
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
    assert final["s2"]["prior_work"] == "supported"
    assert final["s3"]["later_work"] == ""
    assert all(not item["evidence_requests"] for item in final.values())
    assert [item["evidence_id"] for item in merged["related_papers"]] == ["verified-paper"]
    audit = __import__("json").loads((tmp_path / "evidence-resolution.v1.json").read_text())
    assert audit["rerun_segments"] == ["s2"]
    assert audit["final_claim_evidence_ids"]["s2"] == ["verified-paper"]
