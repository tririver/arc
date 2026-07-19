from __future__ import annotations

from copy import deepcopy

import pytest

from arc_companion.evidence import (
    EvidenceProvenanceError,
    arc_cache_descriptor,
    text_sha256,
    validate_annotation_citations,
    validate_cited_ids,
    validate_evidence_record,
    web_evidence_record,
)


def _arc_record(*, relation: str = "prior", evidence_id: str = "prior-001") -> dict:
    blocks = [{
        "block_id": "b-1",
        "type": "text",
        "text": "A recorded result.",
        "sha256": text_sha256("A recorded result."),
    }]
    return {
        "evidence_id": evidence_id,
        "relation": relation,
        "paper_id": "arXiv:1234.5678",
        "title": "Recorded paper",
        "authors": ["A. Author"],
        "year": 2024,
        "evidence_level": "full_text",
        "abstract": "",
        "blocks": blocks,
        "source_descriptor": arc_cache_descriptor(
            paper_id="arXiv:1234.5678",
            title="Recorded paper",
            authors=["A. Author"],
            year=2024,
            evidence_level="full_text",
            content=blocks,
            document_hash="d" * 64,
        ),
    }


def test_arc_cache_descriptor_preserves_existing_id_and_detects_tampering() -> None:
    record = _arc_record()

    assert validate_evidence_record(record)["evidence_id"] == "prior-001"
    assert record["source_descriptor"]["provider"] == "arc-paper"
    assert record["source_descriptor"]["locator"]["document_hash"] == "d" * 64

    changed = deepcopy(record)
    changed["blocks"][0]["text"] = "Unrecorded replacement."
    with pytest.raises(EvidenceProvenanceError, match="source-piece hash mismatch"):
        validate_evidence_record(changed)


def test_web_evidence_gets_stable_controller_id_and_auditable_descriptor() -> None:
    first = web_evidence_record(
        relation="later",
        url="HTTPS://Example.COM:443/paper?q=1#section-2",
        title="Web result",
        excerpt="Observed source passage.",
        retrieved_at="2026-07-19T12:00:00Z",
    )
    second = web_evidence_record(
        relation="later",
        url="https://example.com/paper?q=1#another-fragment",
        title="Web result",
        excerpt="Observed source passage.",
        retrieved_at="2026-07-19T12:00:00+00:00",
    )

    assert first["evidence_id"] == second["evidence_id"]
    assert first["evidence_id"].startswith("web-")
    assert first["source_descriptor"]["canonical_locator"] == "https://example.com/paper?q=1"
    assert first["snippets"][0]["locator"].endswith("#section-2")
    assert validate_evidence_record(first) is first

    forged_id = deepcopy(first)
    forged_id["evidence_id"] = "web-model-invented"
    with pytest.raises(EvidenceProvenanceError, match="not derived"):
        validate_evidence_record(forged_id)

    mismatched_locator = deepcopy(first)
    mismatched_locator["snippets"][0]["locator"] = "https://example.com/other"
    with pytest.raises(EvidenceProvenanceError, match="excerpt locator"):
        validate_evidence_record(mismatched_locator)


def test_web_evidence_rejects_unrecorded_or_unsafe_sources() -> None:
    with pytest.raises(EvidenceProvenanceError, match=r"HTTP\(S\)"):
        web_evidence_record(
            relation="prior",
            url="file:///tmp/claim.txt",
            title="Local",
            excerpt="Claim",
            retrieved_at="2026-07-19T12:00:00Z",
        )
    with pytest.raises(EvidenceProvenanceError, match="timezone"):
        web_evidence_record(
            relation="prior",
            url="https://example.com/source",
            title="Web",
            excerpt="Claim",
            retrieved_at="2026-07-19T12:00:00",
        )
    with pytest.raises(EvidenceProvenanceError, match="unknown or unregistered"):
        validate_cited_ids(["web-model-invented"], [])


def test_annotation_claims_require_registered_relation_matching_ids() -> None:
    prior = _arc_record()
    later = web_evidence_record(
        relation="later",
        url="https://example.com/later",
        title="Later work",
        excerpt="A later extension.",
        retrieved_at="2026-07-19T12:00:00Z",
    )

    assert validate_annotation_citations(
        {"prior_work": "Prior result.", "later_work": "", "evidence_ids": ["prior-001"]},
        [prior, later],
    ) == ["prior-001"]

    with pytest.raises(EvidenceProvenanceError, match="registered prior evidence"):
        validate_annotation_citations(
            {"prior_work": "Prior result.", "later_work": "", "evidence_ids": [later["evidence_id"]]},
            [prior, later],
        )
    with pytest.raises(EvidenceProvenanceError, match="supports no recorded claim"):
        validate_annotation_citations(
            {
                "prior_work": "Prior result.",
                "later_work": "",
                "evidence_ids": ["prior-001", later["evidence_id"]],
            },
            [prior, later],
        )


def test_claim_level_bindings_require_support_for_each_claim() -> None:
    prior = _arc_record()
    annotation = {
        "prior_work": [{
            "text": "Prior result.", "evidence_ids": ["prior-001"],
            "source_locators": [{"evidence_id": "prior-001", "locator": "b-1"}],
            "request_key": None,
        }],
        "later_work": [],
        "evidence_ids": ["prior-001"],
    }

    assert validate_annotation_citations(annotation, [prior]) == ["prior-001"]

    unrelated_locator = deepcopy(annotation)
    unrelated_locator["prior_work"][0]["source_locators"][0]["locator"] = "b-unrelated"
    with pytest.raises(EvidenceProvenanceError, match="unknown source locator"):
        validate_annotation_citations(unrelated_locator, [prior])

    missing = deepcopy(annotation)
    missing["prior_work"][0]["evidence_ids"] = []
    with pytest.raises(EvidenceProvenanceError, match="claim 1 has no registered evidence"):
        validate_annotation_citations(missing, [prior])

    unused = deepcopy(annotation)
    unused["evidence_ids"] = ["prior-001", "prior-002"]
    second = _arc_record(evidence_id="prior-002")
    with pytest.raises(EvidenceProvenanceError, match="must equal the union"):
        validate_annotation_citations(unused, [prior, second])
