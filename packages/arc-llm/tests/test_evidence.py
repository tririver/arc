from __future__ import annotations

import pytest

from arc_llm.evidence import EvidenceRequest, EvidenceResponse, resolve_evidence_round


def test_controller_round_preserves_request_order() -> None:
    requests = (
        EvidenceRequest("second", "paper.metadata", {"paper_id": "2"}),
        EvidenceRequest("first", "paper.metadata", {"paper_id": "1"}),
    )

    def controller(material, *, round_number):
        assert material == requests
        assert round_number == 2
        return (
            EvidenceResponse("first", True, {"title": "one"}),
            EvidenceResponse("second", True, {"title": "two"}),
        )

    responses = resolve_evidence_round(requests, controller, round_number=2)

    assert [item.request_id for item in responses] == ["second", "first"]
    assert responses[0].data == {"title": "two"}


def test_controller_round_is_bounded_to_three_rounds() -> None:
    request = EvidenceRequest("request", "paper.search")

    with pytest.raises(ValueError, match="exceeds max_rounds=3"):
        resolve_evidence_round((request,), lambda *_args, **_kwargs: (), round_number=4)


def test_controller_round_requires_exactly_one_response_per_request() -> None:
    request = EvidenceRequest("request", "paper.search")

    with pytest.raises(ValueError, match="do not match"):
        resolve_evidence_round((request,), lambda *_args, **_kwargs: (), round_number=1)


def test_controller_round_rejects_duplicate_request_ids() -> None:
    requests = (
        EvidenceRequest("request", "paper.search"),
        EvidenceRequest("request", "paper.metadata"),
    )

    with pytest.raises(ValueError, match="request IDs must be unique"):
        resolve_evidence_round(requests, lambda *_args, **_kwargs: (), round_number=1)


def test_controller_round_rejects_duplicate_response_ids() -> None:
    requests = (
        EvidenceRequest("first", "paper.search"),
        EvidenceRequest("second", "paper.metadata"),
    )

    with pytest.raises(ValueError, match="response IDs must be unique"):
        resolve_evidence_round(
            requests,
            lambda *_args, **_kwargs: (
                EvidenceResponse("first", True, {}),
                EvidenceResponse("first", True, {}),
            ),
            round_number=1,
        )


def test_controller_round_rejects_malformed_response_envelope() -> None:
    request = EvidenceRequest("request", "paper.search")

    with pytest.raises(ValueError, match="EvidenceResponse envelopes"):
        resolve_evidence_round(
            (request,), lambda *_args, **_kwargs: ({"request_id": "request"},),
            round_number=1,
        )


def test_failed_response_requires_error() -> None:
    with pytest.raises(ValueError, match="requires an error"):
        EvidenceResponse("request", False)


def test_evidence_contract_carries_reason_and_provenance() -> None:
    request = EvidenceRequest(
        "request",
        "paper.section",
        {"paper_id": "0911.3380", "section": "S2"},
        reason=" verify the normalization ",
    )
    response = EvidenceResponse(
        "request",
        True,
        {"text": "..."},
        provenance={"source": "arc-paper", "paper_id": "0911.3380"},
    )

    assert request.reason == "verify the normalization"
    assert response.provenance["source"] == "arc-paper"
