from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path

import pytest

from arc_companion.review_arbitration import (
    ReviewArbitrationError,
    ReviewPatchSource,
    canonical_sha256,
)
from arc_companion.review_reuse import (
    REVIEW_REUSE_INVALIDATION_CODES,
    REVIEW_REUSE_RECEIPT_VERSION,
    REVIEW_SEGMENT_ACCEPTANCE_VERSION,
    REVIEW_SEGMENT_IDENTITY_VERSION,
    REVIEW_SEGMENT_RULE_VERSION,
    REVIEW_SEGMENT_SOURCE_VERSION,
    AcceptedReviewSegment,
    ReviewReuseError,
    ReviewReusePlan,
    ReviewSegmentIdentity,
    ReviewSegmentSource,
    build_review_segment_acceptance,
    bind_review_reuse_plan_chunks,
    load_review_reuse_receipt,
    load_review_segment_acceptances,
    plan_review_reuse,
    publish_review_segment_acceptance,
    publish_review_segment_object,
    publish_reviewed_output,
    sanitize_segment_evidence,
    split_review_segment_response,
    validate_current_review_identities,
    validate_review_segment_source_set,
)
from arc_companion.prompts import SECTION_REVIEW_SCHEMA


def _identity(
    segment_id: str = "seg-1",
    *,
    mode: str = "translation_enabled",
    source: str = "source",
    translation: str = "translation",
    commentary: str = "commentary",
    glossary: str = "term",
    evidence: object = None,
    reference: str | None = None,
    intent: str | None = None,
    rule: str = REVIEW_SEGMENT_RULE_VERSION,
    schema_version: str = "section-review-output.v1",
) -> ReviewSegmentIdentity:
    return ReviewSegmentIdentity.build(
        segment_id=segment_id,
        mode=mode,
        semantic_segment={"source": source},
        augmentation_blocks=[{"block_id": f"{segment_id}-block", "text": source}],
        current_translation=(
            None if mode == "commentary_only"
            else {"blocks": [{"block_id": f"{segment_id}-block", "text": translation}]}
        ),
        current_annotation={
            "commentary": commentary,
            "explanation": "",
            "commentary_sources": [],
            "prior_work": [],
            "later_work": [],
        },
        local_glossary={"entries": [{"source": glossary, "target": glossary}]},
        protected_names=["Stable Name"],
        segment_evidence=(
            {"sources": [{"url": "https://example.test", "locator": "§1"}]}
            if evidence is None else evidence
        ),
        t14_reference_identity=(
            None if reference is None else {"reference": reference}
        ),
        t14_reference_artifact_sha256=(
            None if reference is None else canonical_sha256(reference)
        ),
        intent_guidance=(
            None if intent is None else {"guidance": intent}
        ),
        annotation_language="zh-CN",
        t15_contracts={
            "review": "v10",
            "arbitration": "v1",
        },
        provider_output_schema_version=schema_version,
        provider_output_schema=SECTION_REVIEW_SCHEMA,
        segment_rule=rule,
    )


def _patch(segment_id: str = "seg-1", text: object = "fixed") -> dict:
    return {
        "segment_id": segment_id,
        "translation_blocks": None,
        "commentary": text,
        "explanation": None,
        "commentary_sources": None,
        "prior_work": None,
        "later_work": None,
        "reason": "local fix",
    }


def _source(
    identity: ReviewSegmentIdentity | None = None,
    *,
    text: object = "fixed",
    audit: dict | None = None,
) -> ReviewSegmentSource:
    identity = identity or _identity()
    return ReviewSegmentSource.build(
        identity=identity,
        findings=[
            {"segment_id": identity.segment_id, "issue": "local issue"}
        ],
        patches=[_patch(identity.segment_id, text)],
        validation_receipt={
            "schema_version": (
                "arc.companion.review-segment-validation.v1"
            ),
            "schema_valid": True,
            "coverage_valid": True,
            "target_valid": True,
            "domain_valid": True,
            "supersession_valid": True,
            "candidate_count": 1,
            "owned_block_ids": [f"{identity.segment_id}-block"],
        },
        audit=audit,
    )


def _accept(
    project: Path,
    identity: ReviewSegmentIdentity,
    sources: list[ReviewSegmentSource],
    *,
    t15_supersession_edges: list[list[str]] | None = None,
    acceptance_supersession_edges: list[list[str]] | None = None,
) -> tuple[AcceptedReviewSegment, dict]:
    links = []
    for source in sources:
        path, object_sha = publish_review_segment_object(project, source)
        links.append({
            "path": str(path.relative_to(project)),
            "object_sha256": object_sha,
            "semantic_content_sha256": source.semantic_content_sha256,
        })
    reviewed_segments = {
        identity.segment_id: {
            "translation": {"blocks": []},
            "annotation": {"commentary": "accepted"},
        }
    }
    translation_hash = canonical_sha256({
        identity.segment_id: {"blocks": []},
    })
    annotation_hash = canonical_sha256({
        identity.segment_id: {"commentary": "accepted"},
    })
    merged_output_hash = canonical_sha256({
        "translations": {identity.segment_id: {"blocks": []}},
        "annotations": {
            identity.segment_id: {"commentary": "accepted"},
        },
    })
    output_path, output_sha = publish_reviewed_output(
        project,
        segments=reviewed_segments,
        merged_output_sha256=merged_output_hash,
        reviewed_translation_sha256=translation_hash,
        reviewed_annotation_sha256=annotation_hash,
    )
    merged_segment_sha = canonical_sha256(
        reviewed_segments[identity.segment_id]
    )
    t15_path = project / "checkpoints" / "t15-receipt.json"
    t15_path.parent.mkdir(parents=True, exist_ok=True)
    source_hashes = sorted(
        source.as_t15_source(stable_order=0).semantic_source_sha256
        for source in sources
    )
    t15_document = {
        "schema_version": "arc.companion.review-arbitration-receipt.v1",
        "status": "resolved",
        "unresolved_paths": [],
        "semantic_input_sha256": "c" * 64,
        "source_hashes": source_hashes,
        "merged_sha256": "d" * 64,
        "final_review_sha256": "d" * 64,
        "reviewed_translation_sha256": translation_hash,
        "reviewed_annotation_sha256": annotation_hash,
        "supersession_edges": t15_supersession_edges or [],
    }
    t15_path.write_text(
        json.dumps(t15_document, sort_keys=True), encoding="utf-8",
    )
    import hashlib
    t15_sha = hashlib.sha256(t15_path.read_bytes()).hexdigest()
    t15_link = {
        "path": str(t15_path.relative_to(project)),
        "sha256": t15_sha,
        "status": "resolved",
        **{
            key: t15_document[key]
            for key in (
                "schema_version",
                "semantic_input_sha256",
                "source_hashes",
                "merged_sha256",
                "final_review_sha256",
                "reviewed_translation_sha256",
                "reviewed_annotation_sha256",
                "supersession_edges",
            )
        },
    }
    document, acceptance_sha = build_review_segment_acceptance(
        identity=identity,
        object_links=links,
        validation={
            "schema_version": (
                "arc.companion.review-segment-validation.v1"
            ),
            "schema_valid": True,
            "domain_valid": True,
        },
        t15_receipt=t15_link,
        accepted_merged_segment_sha256=merged_segment_sha,
        reviewed_output={
            "path": str(output_path.relative_to(project)),
            "sha256": output_sha,
        },
        supersession_edges=acceptance_supersession_edges or [],
    )
    publish_review_segment_acceptance(
        project,
        identity=identity,
        document=document,
        acceptance_sha256=acceptance_sha,
    )
    return AcceptedReviewSegment(
        identity=identity,
        source_sha256s=tuple(
            source.semantic_content_sha256 for source in sources
        ),
        acceptance_sha256=acceptance_sha,
    ), dict(document)


def test_identity_is_exact_immutable_and_rejects_duplicate_current_ids() -> None:
    identity = _identity()
    assert identity.document["identity_version"] == REVIEW_SEGMENT_IDENTITY_VERSION
    assert identity.segment_id == "seg-1"
    assert len(identity.sha256) == 64
    mutated = dict(identity.document)
    mutated["segment_id"] = ""
    with pytest.raises(ReviewReuseError):
        ReviewSegmentIdentity.from_document(mutated)
    with pytest.raises(ReviewReuseError, match="duplicate"):
        validate_current_review_identities([identity, identity])


def test_identity_excludes_runtime_topology_and_provider_parameters() -> None:
    first = _identity()
    second = _identity()
    assert first.sha256 == second.sha256
    assert not {
        "chunk_index",
        "chunk_membership",
        "neighbors",
        "workers",
        "prompt_limit",
        "provider",
        "model",
        "tier",
        "runtime",
        "session",
        "path",
        "time",
    }.intersection(first.document)


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"source": "changed"}, "source_changed"),
        ({"translation": "changed"}, "translation_changed"),
        ({"commentary": "changed"}, "commentary_changed"),
        ({"glossary": "changed"}, "glossary_changed"),
        ({"evidence": {"source_hash": "changed"}}, "evidence_changed"),
        ({"reference": "changed"}, "reference_changed"),
        ({"intent": "changed"}, "intent_changed"),
        ({"rule": "changed-rule"}, "rule_changed"),
        ({"schema_version": "changed-schema"}, "schema_changed"),
    ],
)
def test_plan_reports_exact_local_invalidation(change: dict, expected: str) -> None:
    old = _identity(reference="old" if "reference" in change else None)
    current_args = dict(change)
    if "reference" not in change and old.document[
        "t14_reference_identity_sha256"
    ] is not None:
        current_args["reference"] = "old"
    current = _identity(**current_args)
    accepted = AcceptedReviewSegment(
        identity=old,
        source_sha256s=("a" * 64,),
        acceptance_sha256="b" * 64,
    )
    plan = plan_review_reuse(
        [current], {"seg-1": [accepted]}, False,
    )
    assert plan.entries[0].disposition == "invalidated"
    assert plan.entries[0].reason == expected
    assert expected in REVIEW_REUSE_INVALIDATION_CODES


def test_pure_identity_hashes_the_closed_evidence_projection_exactly() -> None:
    first = _identity(evidence={
        "url": "https://example.test",
        "locator": "p. 3",
        "content_sha256": "a" * 64,
        "path": "/run/one",
        "retrieved_at": "one",
        "cache_hit": False,
    })
    second = _identity(evidence={
        "url": "https://example.test",
        "locator": "p. 3",
        "content_sha256": "a" * 64,
        "path": "/run/two",
        "retrieved_at": "two",
        "cache_hit": True,
    })
    assert first.sha256 != second.sha256
    assert sanitize_segment_evidence(
        {"locator": "p. 3", "path": "/tmp"}
    ) == {"locator": "p. 3", "path": "/tmp"}


def test_source_is_scoped_and_new_reused_audit_does_not_change_t15_hash() -> None:
    identity = _identity()
    new = _source(identity, audit={"disposition": "provider-call"})
    reused = _source(identity, audit={"disposition": "reused"})
    assert new.semantic_content_sha256 == reused.semantic_content_sha256
    assert (
        new.as_t15_source(stable_order=0).semantic_source_sha256
        == reused.as_t15_source(stable_order=0).semantic_source_sha256
    )
    assert new.as_t15_source(stable_order=5).semantic_source_sha256 == (
        new.as_t15_source(stable_order=0).semantic_source_sha256
    )


def test_json_boolean_and_number_remain_distinct_identities() -> None:
    boolean = ReviewSegmentIdentity.build(
        **_identity_build_kwargs({"typed": True})
    )
    number = ReviewSegmentIdentity.build(
        **_identity_build_kwargs({"typed": 1})
    )
    assert boolean.sha256 != number.sha256


def _identity_build_kwargs(semantic_segment: dict) -> dict:
    return {
        "segment_id": "typed-segment",
        "mode": "commentary_only",
        "semantic_segment": semantic_segment,
        "augmentation_blocks": [],
        "current_translation": None,
        "current_annotation": {
            "commentary": "",
            "explanation": "",
            "commentary_sources": [],
            "prior_work": [],
            "later_work": [],
        },
        "local_glossary": {"entries": []},
        "protected_names": [],
        "segment_evidence": {},
        "t14_reference_identity": None,
        "t14_reference_artifact_sha256": None,
        "intent_guidance": None,
        "annotation_language": "en",
        "t15_contracts": {"review": "v10"},
        "provider_output_schema_version": "v1",
        "provider_output_schema": SECTION_REVIEW_SCHEMA,
    }


def test_segment_review_factory_requires_singleton_scoped_identity() -> None:
    review = {
        "reviewed_segment_ids": ["seg-1"],
        "findings": [],
        "patches": [],
    }
    with pytest.raises(ReviewArbitrationError, match="semantic identity"):
        ReviewPatchSource.from_review(
            source_kind="segment_review",
            stable_order=0,
            review=review,
            segment_set=["seg-1"],
        )
    with pytest.raises(ReviewArbitrationError, match="segment set"):
        ReviewPatchSource.from_review(
            source_kind="segment_review",
            stable_order=0,
            review={
                "reviewed_segment_ids": ["seg-1", "seg-2"],
                "findings": [],
                "patches": [],
            },
            segment_set=["seg-1", "seg-2"],
            source_semantic_identity={"identity_sha256": "a" * 64},
        )


def test_split_validates_complete_coverage_then_singletons() -> None:
    identities = {
        "seg-1": _identity("seg-1"),
        "seg-2": _identity("seg-2"),
    }
    validated = []

    def validate(identity, singleton):
        validated.append(identity.segment_id)
        assert singleton["reviewed_segment_ids"] == [identity.segment_id]
        return {
            "schema_version": (
                "arc.companion.review-segment-validation.v1"
            ),
            "schema_valid": True,
            "coverage_valid": True,
            "target_valid": True,
            "domain_valid": True,
            "supersession_valid": True,
            "candidate_count": len(singleton["patches"]),
            "owned_block_ids": [f"{identity.segment_id}-block"],
        }

    sources = split_review_segment_response(
        {
            "reviewed_segment_ids": ["seg-1", "seg-2"],
            "findings": [
                {"segment_id": "seg-2", "issue": "second only"}
            ],
            "patches": [_patch("seg-1")],
        },
        identities_by_segment=identities,
        schema=SECTION_REVIEW_SCHEMA,
        validate_singleton=validate,
    )
    assert validated == ["seg-1", "seg-2"]
    assert [item.identity.segment_id for item in sources] == ["seg-1", "seg-2"]
    assert sources[1].patches == ()
    assert sources[0].findings == ()


@pytest.mark.parametrize(
    "reviewed_ids",
    [["seg-1"], ["seg-1", "seg-1"], ["seg-1", "unknown"]],
)
def test_split_rejects_inexact_coverage(reviewed_ids: list[str]) -> None:
    with pytest.raises(ReviewReuseError, match="coverage"):
        split_review_segment_response(
            {
                "reviewed_segment_ids": reviewed_ids,
                "findings": [],
                "patches": [],
            },
            identities_by_segment={
                "seg-1": _identity("seg-1"),
                "seg-2": _identity("seg-2"),
            },
            schema=SECTION_REVIEW_SCHEMA,
            validate_singleton=lambda *_: {},
        )


def test_plan_is_body_free_ordered_and_regeneration_bypasses_hits() -> None:
    identities = [_identity("seg-1"), _identity("seg-2")]
    accepted = AcceptedReviewSegment(
        identity=identities[0],
        source_sha256s=("a" * 64,),
        acceptance_sha256="b" * 64,
    )
    normal = plan_review_reuse(
        identities, {"seg-1": [accepted]}, False,
    )
    assert [item.disposition for item in normal.entries] == [
        "reused", "uncovered",
    ]
    assert normal.entries[1].planned_miss_chunk_index is None
    assert "translation" not in json.dumps(normal.document)
    regenerated = plan_review_reuse(
        identities, {"seg-1": [accepted]}, True,
    )
    assert all(
        item.disposition == "explicit_regeneration"
        for item in regenerated.entries
    )


def test_plan_chunk_binding_is_sealed_hashed_and_strictly_replayable() -> None:
    identities = [_identity("seg-1"), _identity("seg-2")]
    base = plan_review_reuse(identities, {}, False)
    bound = bind_review_reuse_plan_chunks(
        base, [["seg-1", "seg-2"]],
    )
    assert bound.document["estimated_calls"] == 1
    chunk = bound.document["ordered_missing_chunks"][0]
    assert chunk["ordered_identity_sha256s"] == [
        item.sha256 for item in identities
    ]
    assert len(chunk["chunk_sha256"]) == 64
    assert all(
        item.planned_miss_chunk_sha256 == chunk["chunk_sha256"]
        for item in bound.entries
    )
    assert ReviewReusePlan.from_document(
        bound.document, expected_sha256=bound.sha256,
    ) == bound
    tampered = json.loads(json.dumps(bound.document))
    tampered["ordered_missing_chunks"][0]["logical_unit"] = "changed"
    with pytest.raises(ReviewReuseError):
        ReviewReusePlan.from_document(tampered)
    with pytest.raises(ReviewReuseError, match="not sealed"):
        ReviewReusePlan.from_document(
            base.document,
            require_sealed=True,
        )
    assert ReviewReusePlan.from_document(
        bound.document,
        require_sealed=True,
    ) == bound


def test_receipt_loader_rejects_unsealed_miss_plan(tmp_path: Path) -> None:
    identity = _identity()
    _accepted, acceptance = _accept(tmp_path, identity, [_source(identity)])
    plan = plan_review_reuse([identity], {}, False)
    root = tmp_path / ".arc-companion" / "review-segments"
    plan_path = root / "unsealed-plan.json"
    plan_path.write_text(
        json.dumps(plan.document, sort_keys=True),
        encoding="utf-8",
    )
    plan_path.chmod(0o600)
    receipt = {
        "schema_version": REVIEW_REUSE_RECEIPT_VERSION,
        "plan_path": str(plan_path.relative_to(tmp_path)),
        "plan_sha256": plan.sha256,
        "identity_sha256s": [identity.sha256],
        "source_sha256s": [],
        "acceptance_sha256s": [],
        "new_segment_count": 1,
        "reused_segment_count": 0,
        "actual_review_calls": 0,
        "t15_receipt": acceptance["t15_receipt"],
        "merged_output_sha256": "e" * 64,
        "merged_segment_sha256s": {identity.segment_id: "f" * 64},
        "schema_valid": True,
        "domain_valid": True,
    }
    receipt_path = root / "unsealed-receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True),
        encoding="utf-8",
    )
    receipt_path.chmod(0o600)
    with pytest.raises(ReviewReuseError, match="not sealed"):
        load_review_reuse_receipt(
            tmp_path,
            receipt_path.relative_to(tmp_path),
        )


def test_identity_requires_translation_t14_pair_and_matching_segment() -> None:
    kwargs = _identity_build_kwargs({"segment_id": "typed-segment"})
    kwargs["mode"] = "translation_enabled"
    with pytest.raises(ReviewReuseError, match="translation"):
        ReviewSegmentIdentity.build(**kwargs)
    kwargs = _identity_build_kwargs({"segment_id": "different"})
    with pytest.raises(ReviewReuseError, match="IDs"):
        ReviewSegmentIdentity.build(**kwargs)
    kwargs = _identity_build_kwargs({"segment_id": "typed-segment"})
    kwargs["t14_reference_identity"] = {"reference": "one"}
    with pytest.raises(ReviewReuseError, match="paired"):
        ReviewSegmentIdentity.build(**kwargs)


def test_source_rejects_unscoped_issues_bad_validation_and_unknown_blocks() -> None:
    identity = _identity()
    receipt = dict(_source(identity).validation_receipt)
    with pytest.raises(ReviewReuseError, match="unscoped"):
        ReviewSegmentSource.build(
            identity=identity,
            findings=[],
            patches=[],
            issues=["unscoped"],
            validation_receipt=receipt,
        )
    bad_receipt = dict(receipt)
    bad_receipt["domain_valid"] = False
    with pytest.raises(ReviewReuseError, match="validation"):
        ReviewSegmentSource.build(
            identity=identity,
            findings=[],
            patches=[],
            validation_receipt=bad_receipt,
        )
    translation_patch = _patch()
    translation_patch["commentary"] = None
    translation_patch["translation_blocks"] = [
        {"block_id": "unknown", "text": "changed"}
    ]
    with pytest.raises(ReviewReuseError, match="target"):
        ReviewSegmentSource.build(
            identity=identity,
            findings=[],
            patches=[translation_patch],
            validation_receipt=receipt,
        )


@pytest.mark.parametrize("kind", ["unknown", "self", "cross", "cycle"])
def test_source_rejects_every_supersession_error(kind: str) -> None:
    identity = _identity()
    receipt = dict(_source(identity).validation_receipt)
    first = _patch(text="first")
    second = _patch(text="second")
    second["reason"] = "second"
    explanation = _patch(text=None)
    explanation["explanation"] = "explanation"
    path = "/segments/seg-1/annotation/commentary"
    first_hash = canonical_sha256({"path": path, "replacement": "first"})
    second_hash = canonical_sha256({"path": path, "replacement": "second"})
    explanation_hash = canonical_sha256({
        "path": "/segments/seg-1/annotation/explanation",
        "replacement": "explanation",
    })
    if kind == "unknown":
        edges = [["0" * 64, first_hash]]
        patches = [first, second]
    elif kind == "self":
        edges = [[first_hash, first_hash]]
        patches = [first, second]
    elif kind == "cross":
        edges = [[first_hash, explanation_hash]]
        patches = [first, explanation]
    else:
        edges = [[first_hash, second_hash], [second_hash, first_hash]]
        patches = [first, second]
    receipt["candidate_count"] = 2
    if kind == "unknown":
        source = ReviewSegmentSource.build(
            identity=identity,
            findings=[],
            patches=patches,
            supersession_edges=edges,
            validation_receipt=receipt,
        )
        with pytest.raises(ReviewReuseError, match="supersession"):
            validate_review_segment_source_set([source])
        return
    with pytest.raises(ReviewReuseError, match="supersession"):
        ReviewSegmentSource.build(
            identity=identity,
            findings=[],
            patches=patches,
            supersession_edges=edges,
            validation_receipt=receipt,
        )


def test_objects_acceptances_are_project_local_immutable_and_reusable(
    tmp_path: Path,
) -> None:
    identity = _identity()
    source = _source(identity)
    _, document = _accept(tmp_path, identity, [source])
    accepted, sources = load_review_segment_acceptances(
        tmp_path, [identity],
    )
    assert accepted["seg-1"][0].identity.sha256 == identity.sha256
    assert sources["seg-1"][0].semantic_content_sha256 == (
        source.semantic_content_sha256
    )
    root = tmp_path / ".arc-companion" / "review-segments"
    assert stat_mode(root / "objects") == 0o700
    assert all(stat_mode(path) == 0o600 for path in root.rglob("*.json"))
    acceptance = next((root / "acceptances").glob("*.json"))
    acceptance.write_text('{"tampered":true}', encoding="utf-8")
    with pytest.raises(ReviewReuseError):
        load_review_segment_acceptances(
            tmp_path,
            [identity],
        )
    assert document["schema_version"] == REVIEW_SEGMENT_ACCEPTANCE_VERSION


def test_acceptance_supersession_must_be_applied_by_t15(
    tmp_path: Path,
) -> None:
    identity = _identity()
    first = _patch(text="first")
    second = _patch(text="second")
    second["reason"] = "second"
    path = "/segments/seg-1/annotation/commentary"
    first_hash = canonical_sha256({"path": path, "replacement": "first"})
    second_hash = canonical_sha256({"path": path, "replacement": "second"})
    source = ReviewSegmentSource.build(
        identity=identity,
        findings=[],
        patches=[first, second],
        supersession_edges=[[first_hash, second_hash]],
        validation_receipt={
            "schema_version": (
                "arc.companion.review-segment-validation.v1"
            ),
            "schema_valid": True,
            "coverage_valid": True,
            "target_valid": True,
            "domain_valid": True,
            "supersession_valid": True,
            "candidate_count": 2,
            "owned_block_ids": ["seg-1-block"],
        },
    )
    (tmp_path / "missing").mkdir()
    (tmp_path / "applied").mkdir()
    with pytest.raises(ReviewReuseError, match="not applied by T15"):
        _accept(
            tmp_path / "missing",
            identity,
            [source],
            acceptance_supersession_edges=[[first_hash, second_hash]],
        )
    _accept(
        tmp_path / "applied",
        identity,
        [source],
        t15_supersession_edges=[[first_hash, second_hash]],
        acceptance_supersession_edges=[[first_hash, second_hash]],
    )
    accepted, _sources = load_review_segment_acceptances(
        tmp_path / "applied", [identity],
    )
    assert accepted["seg-1"]


def stat_mode(path: Path) -> int:
    return os.stat(path, follow_symlinks=False).st_mode & 0o777


def test_different_nonsemantic_audits_publish_as_distinct_objects_and_collapse(
    tmp_path: Path,
) -> None:
    identity = _identity()
    first = _source(identity, audit={"reference": "one"})
    second = _source(identity, audit={"reference": "two"})
    first_path, first_hash = publish_review_segment_object(tmp_path, first)
    second_path, second_hash = publish_review_segment_object(tmp_path, second)
    assert first.semantic_content_sha256 == second.semantic_content_sha256
    assert first_path != second_path
    assert first_hash != second_hash
    _accept(tmp_path, identity, [first, second])
    _accepted, sources = load_review_segment_acceptances(
        tmp_path, [identity],
    )
    assert len(sources["seg-1"]) == 1


def test_acceptance_rejects_path_escape_symlink_and_hash_mismatch(
    tmp_path: Path,
) -> None:
    identity = _identity()
    source = _source(identity)
    _accepted, document = _accept(tmp_path, identity, [source])
    acceptance_path = next(
        (tmp_path / ".arc-companion" / "review-segments" / "acceptances")
        .glob("*.json")
    )
    original = json.loads(acceptance_path.read_text())
    original["object_links"][0]["path"] = "../escape.json"
    replacement_hash = canonical_sha256(original)
    replacement = acceptance_path.with_name(
        f"{identity.sha256}-{replacement_hash}.json"
    )
    replacement.write_text(json.dumps(original), encoding="utf-8")
    acceptance_path.unlink()
    with pytest.raises(ReviewReuseError, match="escapes"):
        load_review_segment_acceptances(
            tmp_path,
            [identity],
        )

    replacement.unlink()
    _accept(tmp_path, identity, [source])
    object_path = next(
        (tmp_path / ".arc-companion" / "review-segments" / "objects")
        .glob("*.json")
    )
    object_path.unlink()
    object_path.symlink_to(tmp_path / "outside.json")
    with pytest.raises(ReviewReuseError, match="unsafe|link"):
        load_review_segment_acceptances(
            tmp_path,
            [identity],
        )
    assert document["schema_version"] == REVIEW_SEGMENT_ACCEPTANCE_VERSION


def test_source_document_is_exact_and_body_hash_excludes_audit() -> None:
    source = _source()
    assert source.document["schema_version"] == REVIEW_SEGMENT_SOURCE_VERSION
    changed = dict(source.document)
    changed["extra"] = True
    with pytest.raises(ReviewReuseError, match="fields"):
        ReviewSegmentSource.from_document(changed)


def test_sanitized_fixture_models_37_legacy_calls_without_book_prose() -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "review_segment_reuse_v1.json"
        ).read_text()
    )
    assert fixture["legacy_call_count"] == 37
    assert len(fixture["legacy_calls"]) == 37
    import hashlib
    for call in fixture["legacy_calls"]:
        assert set(call) == {
            "call_id",
            "checkpoint_version",
            "reviewed_segment_ids",
            "findings",
            "patches",
            "response",
            "prompt",
            "prompt_sha256",
            "input_sha256",
        }
        assert call["response"] == {
            "reviewed_segment_ids": call["reviewed_segment_ids"],
            "findings": call["findings"],
            "patches": call["patches"],
        }
        prompt_sha = hashlib.sha256(
            call["prompt"].encode()
        ).hexdigest()
        assert call["prompt_sha256"] == prompt_sha
        version = call["checkpoint_version"].rsplit(".", 1)[-1]
        input_material = {
            "prompt_sha256": prompt_sha,
            "schema": f"section-review-{version}",
            "reviewed_segment_ids": call["reviewed_segment_ids"],
        }
        assert call["input_sha256"] == hashlib.sha256(
            json.dumps(
                input_material,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    assert sum(
        item["call_count"] for item in fixture["legacy_topologies"]
    ) == 37
    assert {
        item["checkpoint_version"]
        for item in fixture["legacy_topologies"]
    } == {
        "arc.companion.section-review-checkpoint.v3",
        "arc.companion.section-review-checkpoint.v4",
    }
    assert fixture["legacy_global_acceptance"]["review"] == {
        "patches": [],
        "issues": [],
    }
    text = json.dumps(fixture)
    assert "Barthes" not in text


def test_pipeline_reuses_unchanged_segment_with_zero_second_review_calls(
    tmp_path: Path,
) -> None:
    from arc_companion.pipeline import BuildOptions, _review

    segments = [{
        "segment_id": "seg-1",
        "chapter_id": "chapter-1",
        "block_ids": ["block-1"],
        "augmentation_block_ids": ["block-1"],
    }]
    document = {
        "blocks": [{
            "block_id": "block-1",
            "type": "paragraph",
            "text": "Local source.",
        }]
    }
    translations = {
        "seg-1": {
            "blocks": [{"block_id": "block-1", "text": "Local translation."}]
        }
    }
    annotations = {
        "seg-1": {
            "commentary": "",
            "explanation": "",
            "commentary_sources": [],
            "prior_work": [],
            "later_work": [],
        }
    }
    calls = 0

    def llm(_prompt: str, **kwargs):
        nonlocal calls
        assert kwargs["call_label"].startswith(
            "companion-review-segment-"
        )
        calls += 1
        return {
            "reviewed_segment_ids": ["seg-1"],
            "findings": [],
            "patches": [],
        }

    options = BuildOptions(
        paper_id="arXiv:0000.0000",
        project_dir=tmp_path,
        workers=1,
    )
    checkpoint_dir = tmp_path / "checkpoints"
    first = _review(
        segments,
        translations,
        annotations,
        document=document,
        glossary={"entries": []},
        protected_names=[],
        evidence={"related_papers": []},
        options=options,
        llm=llm,
        checkpoint_dir=checkpoint_dir,
    )
    assert calls == 1
    assert first[2]["review_reuse"]["actual_review_calls"] == 1

    second = _review(
        segments,
        translations,
        annotations,
        document=document,
        glossary={"entries": []},
        protected_names=[],
        evidence={"related_papers": []},
        options=options,
        llm=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unchanged segment must not call Review provider")
        ),
        checkpoint_dir=checkpoint_dir,
    )
    assert calls == 1
    assert second[2]["review_reuse"]["actual_review_calls"] == 0
    assert second[0] == first[0]
    assert second[1] == first[1]

    fresh = _review(
        segments,
        translations,
        annotations,
        document=document,
        glossary={"entries": []},
        protected_names=[],
        evidence={"related_papers": []},
        options=options,
        llm=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError(
                "fresh checkpoint must reuse project acceptance"
            )
        ),
        checkpoint_dir=tmp_path / "fresh-checkpoint",
    )
    assert fresh[2]["review_reuse"]["actual_review_calls"] == 0

    changed_document = {
        "blocks": [{
            "block_id": "block-1",
            "type": "paragraph",
            "text": "Changed local source.",
        }]
    }
    changed_calls = 0

    def changed_llm(_prompt: str, **_kwargs):
        nonlocal changed_calls
        changed_calls += 1
        return {
            "reviewed_segment_ids": ["seg-1"],
            "findings": [],
            "patches": [],
        }

    _review(
        segments,
        translations,
        annotations,
        document=changed_document,
        glossary={"entries": []},
        protected_names=[],
        evidence={"related_papers": []},
        options=options,
        llm=changed_llm,
        checkpoint_dir=tmp_path / "changed-checkpoint",
    )
    assert changed_calls == 1


@pytest.mark.parametrize(
    "stage",
    [
        "after_plan_before_submit",
        "after_response_before_objects",
        "after_subset_objects",
        "after_all_sources_before_t15",
        "after_t15_before_acceptances",
        "after_subset_acceptances",
        "after_acceptances_before_receipt",
    ],
)
def test_pipeline_review_reuse_crash_replay_never_repeats_review_call(
    stage: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import arc_companion.pipeline as pipeline_module
    from arc_companion.pipeline import BuildOptions, _review

    segments = [{
        "segment_id": "seg-1",
        "chapter_id": "chapter-1",
        "block_ids": ["block-1"],
        "augmentation_block_ids": ["block-1"],
    }]
    document = {
        "blocks": [{
            "block_id": "block-1",
            "type": "paragraph",
            "text": "Local source.",
        }]
    }
    translations = {
        "seg-1": {
            "blocks": [{"block_id": "block-1", "text": "Local translation."}]
        }
    }
    annotations = {
        "seg-1": {
            "commentary": "",
            "explanation": "",
            "commentary_sources": [],
            "prior_work": [],
            "later_work": [],
        }
    }
    calls = 0
    injected = False

    def llm(_prompt: str, **_kwargs):
        nonlocal calls
        calls += 1
        return {
            "reviewed_segment_ids": ["seg-1"],
            "findings": [],
            "patches": [],
        }

    def cutpoint(
        current: str, _checkpoint: Path, _detail: str | None = None,
    ) -> None:
        nonlocal injected
        if current == stage and not injected:
            injected = True
            raise RuntimeError(f"crash:{stage}")

    monkeypatch.setattr(
        pipeline_module,
        "_review_reuse_cutpoint",
        cutpoint,
    )
    kwargs = {
        "document": document,
        "glossary": {"entries": []},
        "protected_names": [],
        "evidence": {"related_papers": []},
        "options": BuildOptions(
            paper_id="arXiv:0000.0000",
            project_dir=tmp_path,
            workers=1,
        ),
        "llm": llm,
        "checkpoint_dir": tmp_path / "checkpoints",
    }
    with pytest.raises(RuntimeError, match=f"crash:{stage}"):
        _review(
            segments,
            translations,
            annotations,
            **kwargs,
        )
    result = _review(
        segments,
        translations,
        annotations,
        **kwargs,
    )
    assert injected
    assert calls == 1
    assert result[2]["review_reuse"]["actual_review_calls"] in {0, 1}
    assert (
        tmp_path / "checkpoints" / "review-reuse-receipt.json"
    ).is_file()


def test_pipeline_reviews_only_changed_middle_segment_and_ignores_topology(
    tmp_path: Path,
) -> None:
    from arc_companion.pipeline import BuildOptions, _review

    segments = [
        {
            "segment_id": f"seg-{index}",
            "chapter_id": "chapter-1",
            "block_ids": [f"block-{index}"],
            "augmentation_block_ids": [f"block-{index}"],
        }
        for index in range(3)
    ]
    document = {
        "blocks": [
            {
                "block_id": f"block-{index}",
                "type": "paragraph",
                "text": f"Local source {index}.",
            }
            for index in range(3)
        ]
    }
    translations = {
        f"seg-{index}": {
            "blocks": [{
                "block_id": f"block-{index}",
                "text": f"Local translation {index}.",
            }]
        }
        for index in range(3)
    }
    annotations = {
        f"seg-{index}": {
            "commentary": "",
            "explanation": "",
            "commentary_sources": [],
            "prior_work": [],
            "later_work": [],
        }
        for index in range(3)
    }
    reviewed_calls: list[list[str]] = []

    def llm(prompt: str, **kwargs):
        assert kwargs["call_label"].startswith(
            "companion-review-segment-"
        )
        payload = json.loads(prompt.split("PORTION:\n", 1)[1])
        segment_ids = [
            str(item["segment"]["segment_id"])
            for item in payload["segments"]
        ]
        reviewed_calls.append(segment_ids)
        return {
            "reviewed_segment_ids": segment_ids,
            "findings": [],
            "patches": [],
        }

    checkpoint_dir = tmp_path / "checkpoints"
    base_options = BuildOptions(
        paper_id="arXiv:0000.0000",
        project_dir=tmp_path,
        workers=1,
        review_context_chars=100_000,
    )
    kwargs = {
        "document": document,
        "glossary": {"entries": []},
        "protected_names": [],
        "evidence": {"related_papers": []},
        "llm": llm,
        "checkpoint_dir": checkpoint_dir,
    }
    _review(
        segments,
        translations,
        annotations,
        options=base_options,
        **kwargs,
    )
    assert reviewed_calls == [["seg-0", "seg-1", "seg-2"]]

    changed = json.loads(json.dumps(translations))
    changed["seg-1"]["blocks"][0]["text"] = "Changed translation."
    _review(
        segments,
        changed,
        annotations,
        options=base_options,
        **kwargs,
    )
    assert reviewed_calls[-1] == ["seg-1"]
    assert len(reviewed_calls) == 2

    topology_options = replace(
        base_options,
        workers=3,
        review_context_chars=1,
    )
    _review(
        segments,
        changed,
        annotations,
        options=topology_options,
        **kwargs,
    )
    assert len(reviewed_calls) == 2


def test_pipeline_commentary_reuse_and_explicit_regeneration(
    tmp_path: Path,
) -> None:
    from arc_companion.pipeline import BuildOptions, _review

    segments = [{
        "segment_id": "seg-1",
        "chapter_id": "chapter-1",
        "block_ids": ["block-1"],
        "augmentation_block_ids": ["block-1"],
    }]
    document = {
        "blocks": [{
            "block_id": "block-1",
            "type": "paragraph",
            "text": "本地原文。",
        }]
    }
    annotations = {
        "seg-1": {
            "commentary": "伴读",
            "explanation": "",
            "commentary_sources": [],
            "prior_work": [],
            "later_work": [],
        }
    }
    calls = 0

    def llm(prompt: str, **kwargs):
        nonlocal calls
        calls += 1
        assert "COMPANION:\n" in prompt
        assert kwargs["call_label"].startswith(
            "companion-review-segment-"
        )
        return {
            "reviewed_segment_ids": ["seg-1"],
            "findings": [],
            "patches": [],
        }

    base = BuildOptions(
        paper_id="local:commentary",
        project_dir=tmp_path,
        skip_translation=True,
        workers=1,
    )
    kwargs = {
        "document": document,
        "glossary": {"entries": []},
        "protected_names": [],
        "evidence": {"related_papers": []},
        "llm": llm,
        "checkpoint_dir": tmp_path / "checkpoints",
    }
    _review(segments, None, annotations, options=base, **kwargs)
    _review(segments, None, annotations, options=base, **kwargs)
    assert calls == 1

    regenerated = replace(base, regenerate_lanes=("review",))
    result = _review(
        segments,
        None,
        annotations,
        options=regenerated,
        **kwargs,
    )
    assert calls == 2
    assert result[2]["review_reuse"]["counts"][
        "explicit_regeneration"
    ] == 1


def test_pipeline_reuses_accepted_review_across_freeze_continuation(
    tmp_path: Path,
) -> None:
    from arc_companion.pipeline import BuildOptions, _review

    segments = [{
        "segment_id": "seg-1",
        "chapter_id": "chapter-1",
        "block_ids": ["block-1"],
        "augmentation_block_ids": ["block-1"],
    }]
    document = {
        "blocks": [{
            "block_id": "block-1",
            "type": "paragraph",
            "text": "Local source.",
        }]
    }
    translations = {
        "seg-1": {
            "blocks": [{"block_id": "block-1", "text": "Translation."}]
        }
    }
    annotations = {
        "seg-1": {
            "commentary": "baseline",
            "explanation": "",
            "commentary_sources": [],
            "prior_work": [],
            "later_work": [],
        }
    }
    calls = 0

    def llm(_prompt: str, **kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["call_label"].startswith(
            "companion-review-segment-"
        )
        return {
            "reviewed_segment_ids": ["seg-1"],
            "findings": [],
            "patches": [{
                "segment_id": "seg-1",
                "translation_blocks": None,
                "commentary": "accepted review",
                "explanation": None,
                "commentary_sources": None,
                "prior_work": None,
                "later_work": None,
                "reason": "precision",
            }],
        }

    options = BuildOptions(
        paper_id="local:freeze",
        project_dir=tmp_path,
        workers=1,
    )
    kwargs = {
        "document": document,
        "glossary": {"entries": []},
        "protected_names": [],
        "evidence": {"related_papers": []},
        "options": options,
        "llm": llm,
        "checkpoint_dir": tmp_path / "checkpoints",
    }
    first = _review(
        segments,
        translations,
        annotations,
        **kwargs,
    )
    assert first[1]["seg-1"]["commentary"] == "accepted review"
    freeze_binding = {
        "schema_version": "arc.companion.review-freeze-binding.v1",
        "freeze_sha256": "f" * 64,
        "segment_ids": ["seg-1"],
        "translation_mode": "enabled",
        "pre_review_translation_sha256": canonical_sha256(translations),
        "pre_review_annotation_sha256": canonical_sha256(annotations),
    }
    continued = _review(
        segments,
        translations,
        annotations,
        freeze_binding=freeze_binding,
        **kwargs,
    )
    assert calls == 1
    assert continued[1]["seg-1"]["commentary"] == "accepted review"


@pytest.mark.parametrize(
    ("legacy_version", "tamper_prompt_audit"),
    [("v3", False), ("v4", False), ("v4", True)],
)
def test_pipeline_strictly_imports_legacy_review_without_mutating_old_bytes(
    legacy_version: str,
    tamper_prompt_audit: bool,
    tmp_path: Path,
) -> None:
    import hashlib
    import arc_companion.pipeline as pipeline_module
    from arc_companion.pipeline import BuildOptions, _review
    from arc_companion.prompts import (
        REVIEW_SCHEMA,
        SECTION_REVIEW_SCHEMA,
        section_review_prompt,
    )

    segment = {
        "segment_id": "seg-1",
        "chapter_id": "chapter-1",
        "block_ids": ["block-1"],
        "augmentation_block_ids": ["block-1"],
    }
    segments = [segment]
    block = {
        "block_id": "block-1",
        "type": "paragraph",
        "text": "Local source.",
    }
    document = {"blocks": [block]}
    translations = {
        "seg-1": {
            "blocks": [{"block_id": "block-1", "text": "Translation."}]
        }
    }
    annotations = {
        "seg-1": {
            "commentary": "baseline",
            "explanation": "",
            "commentary_sources": [],
            "prior_work": [],
            "later_work": [],
        }
    }
    options = BuildOptions(
        paper_id="local:legacy",
        project_dir=tmp_path,
        workers=1,
        review_context_chars=1,
    )
    checkpoint_dir = tmp_path / "checkpoints"
    section_root = checkpoint_dir / "section-reviews"
    section_root.mkdir(parents=True)
    cleaned_annotation = pipeline_module.clean_reader_annotation(
        annotations["seg-1"],
        evidence_records=[],
        language=options.annotation_language,
    )
    payload = {
        "segment": pipeline_module._semantic_segment_descriptor(segment),
        "source_blocks": [
            pipeline_module._annotation_input_block(block, document)
        ],
        "translation": translations["seg-1"],
        "annotation": cleaned_annotation,
        "context_evidence": [],
    }
    section_prompt = section_review_prompt(
        {
            "segments": [payload],
            "glossary": (
                pipeline_module._commentary_review_glossary_projection(
                    {"entries": []},
                    [payload],
                    max_bytes=(
                        pipeline_module.ANNOTATION_GLOSSARY_MAX_BYTES
                    ),
                )
            ),
            "protected_names": [],
        },
        language=options.annotation_language,
    )
    prompt_budget = pipeline_module._review_prompt_budget(options)
    section_rendered = pipeline_module._rendered_review_call(
        [payload],
        section_prompt,
        target_prompt_bytes=prompt_budget["target_limit_bytes"],
        strict_prompt_bytes=prompt_budget["strict_limit_bytes"],
        headroom_class="segment_reuse_fixture",
    )
    section_audit = pipeline_module._review_prompt_call_audit(
        section_rendered,
        stage="section",
        call_label="companion-section-review-0",
        disposition="provider-call",
    )
    section_response = {
        "reviewed_segment_ids": ["seg-1"],
        "findings": [],
        "patches": [],
    }
    section_checkpoint = {
        "schema_version": (
            f"arc.companion.section-review-checkpoint.{legacy_version}"
        ),
        "section_index": 0,
        "input_sha256": pipeline_module.sha256_json({
            "prompt": section_prompt,
            "schema": SECTION_REVIEW_SCHEMA,
            "model_tier": pipeline_module.REVIEW_TIER,
        }),
        "reviewed_segment_ids": ["seg-1"],
        "review": section_response,
    }
    if legacy_version == "v4":
        section_checkpoint.update({
            "prompt_budget_audit": section_audit,
            "evidence_prompt_budget_audits": [],
        })
    section_path = section_root / "0000.json"
    section_path.write_text(
        json.dumps(section_checkpoint, sort_keys=True),
        encoding="utf-8",
    )
    _payload, final_prompt = (
        pipeline_module._bounded_hierarchical_review_prompt(
            [{
                "section_index": 0,
                "reviewed_segment_ids": ["seg-1"],
                "findings": [],
                "patch_proposals": [],
            }],
            segments,
            blocks_by_id={"block-1": block},
            document=document,
            segment_payloads=[payload],
            glossary={"entries": []},
            protected_names=[],
            language=options.annotation_language,
            max_prompt_bytes=(
                pipeline_module._review_prompt_target_limit(options)
            ),
            strict_prompt_bytes=(
                pipeline_module._review_prompt_byte_limit(options)
            ),
            intent_guidance=None,
        )
    )
    final_rendered = pipeline_module._rendered_review_call(
        [payload],
        final_prompt,
        target_prompt_bytes=prompt_budget["target_limit_bytes"],
        strict_prompt_bytes=prompt_budget["strict_limit_bytes"],
        headroom_class="essential_final_headroom",
    )
    final_audit = pipeline_module._review_prompt_call_audit(
        final_rendered,
        stage="hierarchical-final",
        call_label="companion-final-review",
        disposition="provider-call",
    )
    accepted_response = {"patches": [], "issues": []}
    receipt_path = (
        checkpoint_dir / "review-arbitration" / "legacy" / "receipt.json"
    )
    receipt_path.parent.mkdir(parents=True)
    receipt = {
        "schema_version": (
            "arc.companion.review-arbitration-receipt.v1"
        ),
        "status": "resolved",
        "unresolved_paths": [],
        "semantic_input_sha256": "a" * 64,
        "merged_sha256": canonical_sha256(accepted_response),
        "final_review_sha256": canonical_sha256(accepted_response),
        "reviewed_translation_sha256": canonical_sha256(translations),
        "reviewed_annotation_sha256": canonical_sha256({
            "seg-1": cleaned_annotation,
        }),
    }
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True),
        encoding="utf-8",
    )
    receipt_reference = {
        "path": str(receipt_path.relative_to(checkpoint_dir)),
        "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        **{
            key: receipt[key]
            for key in (
                "status",
                "semantic_input_sha256",
                "merged_sha256",
                "final_review_sha256",
                "reviewed_translation_sha256",
                "reviewed_annotation_sha256",
            )
        },
    }
    final_acceptance = {
        "schema_version": "arc.companion.final-review-acceptance.v1",
        "input_sha256": pipeline_module.sha256_json({
            "prompt": final_prompt,
            "schema": REVIEW_SCHEMA,
            "model_tier": pipeline_module.REVIEW_TIER,
        }),
        "response": accepted_response,
        "review_arbitration_receipt": receipt_reference,
        "reviewed_translation_sha256": canonical_sha256(translations),
        "reviewed_annotation_sha256": canonical_sha256({
            "seg-1": cleaned_annotation,
        }),
        "audit": {
            "prompt_budget_audit": {
                "calls": [section_audit, final_audit],
            }
        },
    }
    if tamper_prompt_audit:
        final_acceptance["audit"]["prompt_budget_audit"]["calls"][-1][
            "prompt_sha256"
        ] = "0" * 64
    acceptance_path = checkpoint_dir / "final-review-accepted.json"
    acceptance_path.write_text(
        json.dumps(final_acceptance, sort_keys=True),
        encoding="utf-8",
    )
    old_bytes = {
        path: path.read_bytes()
        for path in (section_path, acceptance_path, receipt_path)
    }
    calls = 0

    def llm(_prompt: str, **kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["call_label"].startswith(
            "companion-review-segment-"
        )
        return {
            "reviewed_segment_ids": ["seg-1"],
            "findings": [],
            "patches": [],
        }

    result = _review(
        segments,
        translations,
        annotations,
        document=document,
        glossary={"entries": []},
        protected_names=[],
        evidence={"related_papers": []},
        options=options,
        llm=llm,
        checkpoint_dir=checkpoint_dir,
    )
    assert calls == int(tamper_prompt_audit)
    assert result[2]["review_reuse"]["actual_review_calls"] == int(
        tamper_prompt_audit
    )
    assert result[2]["review_reuse"]["counts"][
        "invalidated" if tamper_prompt_audit else "reused"
    ] == 1
    assert all(path.read_bytes() == value for path, value in old_bytes.items())


@pytest.mark.parametrize("issues", [[], ["legacy unscoped issue"]])
def test_commentary_v4_legacy_import_requires_splittable_issues(
    issues: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import arc_companion.pipeline as pipeline_module
    from arc_companion.pipeline import BuildOptions
    from arc_companion.prompts import COMMENTARY_REVIEW_SCHEMA

    segment = {
        "segment_id": "seg-1",
        "chapter_id": "chapter-1",
        "block_ids": ["block-1"],
        "augmentation_block_ids": ["block-1"],
    }
    document = {"blocks": [{
        "block_id": "block-1",
        "type": "paragraph",
        "text": "Local source.",
    }]}
    annotations = {"seg-1": {
        "commentary": "",
        "explanation": "",
        "commentary_sources": [],
        "prior_work": [],
        "later_work": [],
    }}
    options = BuildOptions(
        paper_id="local:legacy-commentary",
        project_dir=tmp_path,
        skip_translation=True,
        workers=1,
    )
    prepared = pipeline_module._prepare_review_segment_reuse_inputs(
        [segment],
        None,
        annotations,
        document=document,
        glossary={"entries": []},
        protected_names=[],
        evidence={"related_papers": []},
        options=options,
        intent_guidance=None,
        t14_reference_by_segment=None,
    )
    payload = prepared["payload_by_segment"]["seg-1"]
    prompt = pipeline_module.commentary_review_prompt(
        {"glossary": {"entries": []}, "segments": [payload]},
        language=options.annotation_language,
    )
    rendered = pipeline_module._rendered_review_call(
        [payload],
        prompt,
        target_prompt_bytes=200_000,
        strict_prompt_bytes=240_000,
        headroom_class="legacy-fixture",
    )
    call_audit = pipeline_module._review_prompt_call_audit(
        rendered,
        stage="commentary",
        call_label="companion-commentary-review-0",
        disposition="provider-call",
    )
    checkpoint_dir = tmp_path / "checkpoints"
    review_root = checkpoint_dir / "commentary-reviews"
    review_root.mkdir(parents=True)
    (review_root / "0000.json").write_text(json.dumps({
        "schema_version": (
            "arc.companion.commentary-review-checkpoint.v4"
        ),
        "group_index": 0,
        "input_sha256": pipeline_module.sha256_json({
            "prompt": prompt,
            "schema": COMMENTARY_REVIEW_SCHEMA,
            "model_tier": pipeline_module.REVIEW_TIER,
        }),
        "reviewed_segment_ids": ["seg-1"],
        "review": {"issues": issues, "patches": []},
    }))
    (checkpoint_dir / "review.v5.json").write_text(json.dumps({
        "review_arbitration_receipt": {"path": "unused"},
        "prompt_budget_audit": {"calls": [call_audit]},
    }))
    (checkpoint_dir / "annotations.reviewed.v5.json").write_text(
        json.dumps({
            "schema_version": pipeline_module.REVIEW_VERSION,
            "annotations": prepared["annotations"],
        })
    )
    monkeypatch.setattr(
        pipeline_module,
        "_review_arbitration_reference_valid",
        lambda *_args, **_kwargs: True,
    )
    accepted, sources, proof = (
        pipeline_module._legacy_commentary_review_segment_sources(
            identities=prepared["identities"],
            payload_by_segment=prepared["payload_by_segment"],
            annotations=prepared["annotations"],
            glossary={"entries": []},
            options=options,
            checkpoint_dir=checkpoint_dir,
            intent_guidance=None,
            validate_singleton=lambda _identity, _singleton: {
                "schema_version": (
                    pipeline_module.REVIEW_SEGMENT_VALIDATION_VERSION
                ),
                "schema_valid": True,
                "coverage_valid": True,
                "target_valid": True,
                "domain_valid": True,
                "supersession_valid": True,
                "candidate_count": 0,
                "owned_block_ids": ["block-1"],
            },
        )
    )
    assert proof is not None
    if issues:
        assert not accepted["seg-1"][0].valid
        assert (
            accepted["seg-1"][0].invalidation_code
            == "legacy_unscoped_issue"
        )
        assert sources["seg-1"] == ()
    else:
        assert accepted["seg-1"][0].valid
        assert len(sources["seg-1"]) == 1
