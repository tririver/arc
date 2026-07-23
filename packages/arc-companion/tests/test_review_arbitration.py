from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import jsonschema
import pytest

from arc_companion.review_arbitration import (
    ANNOTATION_FIELDS,
    REVIEW_ARBITRATION_SCHEMA,
    CanonicalCandidate,
    ReviewArbitrationError,
    ReviewArbitrationNeedsSupervision,
    ReviewPatchSource,
    apply_non_conflicting_atoms,
    arbitration_payload,
    canonical_sha256,
    candidate_to_patch,
    json_pointer_escape,
    materialize_review_patches,
    plan_review_merge,
    trial_validate_candidates,
    validate_arbitration_output,
    validate_materialized_review,
)


def _patch(
    segment_id: str,
    *,
    translation_blocks=None,
    commentary=None,
    explanation=None,
    commentary_sources=None,
    prior_work=None,
    later_work=None,
    reason="reason",
):
    return {
        "segment_id": segment_id,
        "translation_blocks": translation_blocks,
        "commentary": commentary,
        "explanation": explanation,
        "commentary_sources": commentary_sources,
        "prior_work": prior_work,
        "later_work": later_work,
        "reason": reason,
    }


def _source(
    source_id: str,
    patches,
    *,
    order=0,
    findings=(),
    issues=(),
    call_record=False,
):
    review = {
        "reviewed_segment_ids": ["seg/1", "seg~2"],
        "patches": deepcopy(list(patches)),
        "findings": deepcopy(list(findings)),
    }
    if call_record:
        review["arc_llm_call_record"] = {"provider": "volatile"}
        review["patches"][0]["arc_llm_call_record"] = {"call_id": "volatile"}
    return ReviewPatchSource.from_review(
        source_kind="section",
        stable_order=order,
        review=review,
        segment_set=["seg/1", "seg~2"],
    )


def _plan(sources, *, skip=False, supersession=None):
    return plan_review_merge(
        sources,
        segment_order=["seg/1", "seg~2"],
        block_order_by_segment={
            "seg/1": ["b/1", "b~2"],
            "seg~2": ["b3"],
        },
        skip_translation=skip,
        contract_versions={"review": "v9", "arbitration": "v1"},
        provider="fake",
        model="fake-model",
        original_value_resolver=lambda path: {"original_for": path},
        invariant_context_resolver=lambda component: {
            "path_sha256": canonical_sha256(component.path)
        },
        controller_supersession_edges=supersession,
    )


def _decision(component, *, action, selected=(), patch=None, reason="decision"):
    return {
        "path": component.path,
        "action": action,
        "selected_candidate_hashes": list(selected),
        "replacement_patch": patch,
        "reason": reason,
    }


def test_schema_is_recursively_closed() -> None:
    def visit(node):
        if isinstance(node, list):
            for item in node:
                visit(item)
        elif isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
                assert set(node.get("required") or ()) == set(
                    (node.get("properties") or {}).keys()
                )
            for value in node.values():
                visit(value)

    visit(REVIEW_ARBITRATION_SCHEMA)
    jsonschema.Draft202012Validator.check_schema(REVIEW_ARBITRATION_SCHEMA)


def test_pointer_escaping_and_disjoint_atomization() -> None:
    plan = _plan([_source("s", [
        _patch(
            "seg/1",
            translation_blocks=[
                {"block_id": "b/1", "text": "one"},
                {"block_id": "b~2", "text": "two"},
            ],
            commentary="comment",
            prior_work=[],
        )
    ])])
    assert json_pointer_escape("seg/1") == "seg~11"
    assert [item.path for item in plan.canonical_groups] == [
        "/segments/seg~11/translation_blocks/b~11",
        "/segments/seg~11/translation_blocks/b~02",
        "/segments/seg~11/annotation/commentary",
        "/segments/seg~11/annotation/prior_work",
    ]
    assert not plan.components
    assert len(plan.non_conflicting_atoms) == 4


def test_canonical_duplicates_merge_origins_and_strip_call_records() -> None:
    patch = _patch("seg/1", commentary="same")
    plan = _plan([
        _source("a", [patch], order=0, call_record=True),
        _source("b", [patch], order=1),
    ])
    assert len(plan.atoms) == 2
    assert len(plan.canonical_groups) == 1
    assert len(plan.canonical_groups[0].origins) == 2
    assert not plan.components
    assert "volatile" not in json.dumps(plan.semantic_input)


def test_duplicate_reason_presentation_uses_declared_stable_order_only() -> None:
    later = _source(
        "later",
        [_patch("seg/1", commentary="same", reason="present second")],
        order=9,
    )
    earlier = _source(
        "earlier",
        [_patch("seg/1", commentary="same", reason="present first")],
        order=1,
    )
    plan = _plan([later, earlier])
    candidate = plan.non_conflicting_atoms[0]
    assert candidate.reasons == ("present first", "present second")
    semantic = candidate.semantic_payload()
    assert all("stable_order" not in item for item in semantic["origins"])
    reordered = _plan([
        ReviewPatchSource.from_review(
            source_kind="section",
            stable_order=1,
            review={
                "reviewed_segment_ids": ["seg/1", "seg~2"],
                "findings": [],
                "patches": [
                    _patch(
                        "seg/1",
                        commentary="same",
                        reason="present second",
                    )
                ],
            },
            segment_set=["seg/1", "seg~2"],
        ),
        ReviewPatchSource.from_review(
            source_kind="section",
            stable_order=9,
            review={
                "reviewed_segment_ids": ["seg/1", "seg~2"],
                "findings": [],
                "patches": [
                    _patch(
                        "seg/1",
                        commentary="same",
                        reason="present first",
                    )
                ],
            },
            segment_set=["seg/1", "seg~2"],
        ),
    ])
    assert reordered.semantic_input_sha256 == plan.semantic_input_sha256
    assert reordered.non_conflicting_atoms[0].reasons == (
        "present second", "present first",
    )


def test_source_identity_is_derived_complete_and_immutable() -> None:
    nested_finding = {"segment_id": "seg/1", "issue": "nested"}
    patch = _patch(
        "seg/1",
        translation_blocks=[{"block_id": "b/1", "text": "replacement"}],
    )
    review = {
        "reviewed_segment_ids": ["seg~2", "seg/1"],
        "findings": [nested_finding],
        "patches": [patch],
    }
    source = ReviewPatchSource.from_review(
        source_kind="section",
        stable_order=0,
        review=review,
        segment_set=["seg/1", "seg~2"],
    )
    assert source.source_id == (
        f"section:{source.semantic_source_sha256}"
    )
    review["patches"][0]["translation_blocks"][0]["text"] = "mutated"
    review["findings"][0]["issue"] = "mutated"
    assert source.patches[0]["translation_blocks"][0]["text"] == "replacement"
    assert source.findings[0]["issue"] == "nested"
    returned = source.patches[0]
    returned["translation_blocks"][0]["text"] = "also mutated"
    assert source.patches[0]["translation_blocks"][0]["text"] == "replacement"

    with pytest.raises(ReviewArbitrationError, match="segment set"):
        ReviewPatchSource.from_review(
            source_kind="section",
            stable_order=0,
            review=review,
            segment_set=["seg/1"],
        )
    with pytest.raises(ReviewArbitrationError, match="semantic identity"):
        ReviewPatchSource.from_review(
            source_kind="section",
            stable_order=0,
            review={
                **review,
                "reviewed_segment_ids": ["seg/1", "seg~2"],
            },
            segment_set=["seg/1", "seg~2"],
            source_id="caller-provenance",
        )
    stable_variant = ReviewPatchSource.from_review(
        source_kind="section",
        stable_order=7,
        review={
            **review,
            "reviewed_segment_ids": ["seg/1", "seg~2"],
        },
        segment_set=["seg~2", "seg/1"],
    )
    zero_variant = ReviewPatchSource.from_review(
        source_kind="section",
        stable_order=0,
        review={
            **review,
            "reviewed_segment_ids": ["seg/1", "seg~2"],
        },
        segment_set=["seg/1", "seg~2"],
    )
    assert stable_variant.semantic_source_sha256 == (
        zero_variant.semantic_source_sha256
    )
    for invalid_order in (True, 1.0, "1"):
        with pytest.raises(ReviewArbitrationError, match="stable order"):
            ReviewPatchSource.from_review(
                source_kind="section",
                stable_order=invalid_order,
                review={
                    **review,
                    "reviewed_segment_ids": ["seg/1", "seg~2"],
                },
                segment_set=["seg/1", "seg~2"],
            )


def test_commentary_source_uses_authoritative_translation_forbidden_schema() -> None:
    commentary_patch = {
        key: value
        for key, value in _patch("seg/1", commentary="valid").items()
        if key != "translation_blocks"
    }
    source = ReviewPatchSource.from_review(
        source_kind="commentary",
        stable_order=0,
        review={"patches": [commentary_patch], "issues": []},
        segment_set=["seg/1", "seg~2"],
    )
    plan = _plan([source], skip=True)
    assert [item.target_id for item in plan.non_conflicting_atoms] == [
        "commentary"
    ]
    with pytest.raises(ReviewArbitrationError, match="schema"):
        ReviewPatchSource.from_review(
            source_kind="commentary",
            stable_order=0,
            review={
                "patches": [_patch("seg/1", commentary="invalid")],
                "issues": [],
            },
            segment_set=["seg/1", "seg~2"],
        )
    with pytest.raises(ReviewArbitrationError, match="outside"):
        ReviewPatchSource.from_review(
            source_kind="commentary",
            stable_order=0,
            review={
                "patches": [{**commentary_patch, "segment_id": "seg~2"}],
                "issues": [],
            },
            segment_set=["seg/1"],
        )
    with pytest.raises(ReviewArbitrationError, match="must not"):
        ReviewPatchSource.from_review(
            source_kind="final",
            stable_order=0,
            review={"patches": [], "issues": []},
            segment_set=["seg/1"],
        )


def test_json_hash_distinguishes_boolean_from_number() -> None:
    assert canonical_sha256(True) != canonical_sha256(1)


@pytest.mark.parametrize(
    "patch,match",
    [
        ({"bad": "shape"}, "schema"),
        (_patch("", commentary="x"), "segment"),
        (_patch("unknown", commentary="x"), "segment"),
        (_patch("seg/1"), "no replacement"),
        (_patch("seg/1", translation_blocks=[]), "non-empty"),
        (
            _patch("seg/1", translation_blocks=[
                {"block_id": "", "text": "x"},
            ]),
            "schema",
        ),
        (
            _patch("seg/1", translation_blocks=[
                {"block_id": "b/1", "text": "x"},
                {"block_id": "b/1", "text": "y"},
            ]),
            "duplicate",
        ),
        (
            _patch("seg/1", translation_blocks=[
                {"block_id": "foreign", "text": "x"},
            ]),
            "not owned",
        ),
    ],
)
def test_malformed_patch_targets_fail_closed(patch, match) -> None:
    with pytest.raises(ReviewArbitrationError, match=match):
        _plan([_source("s", [patch])])


def test_skip_translation_rejects_before_arbitration() -> None:
    with pytest.raises(ReviewArbitrationError, match="skip-translation"):
        _plan([_source("s", [
            _patch(
                "seg/1",
                translation_blocks=[{"block_id": "b/1", "text": "x"}],
            )
        ])], skip=True)


def test_conflict_graph_connects_only_exact_targets_in_stable_order() -> None:
    plan = _plan([
        _source("late", [
            _patch("seg~2", explanation="second"),
            _patch("seg/1", commentary="candidate-b"),
        ], order=2),
        _source("early", [
            _patch("seg/1", commentary="candidate-a"),
            _patch("seg/1", explanation="only"),
        ], order=0),
    ])
    assert [item.path for item in plan.components] == [
        "/segments/seg~11/annotation/commentary",
    ]
    assert [item.path for item in plan.non_conflicting_atoms] == [
        "/segments/seg~11/annotation/explanation",
        "/segments/seg~02/annotation/explanation",
    ]


def test_valid_supersession_prunes_within_one_target() -> None:
    base = _plan([
        _source("a", [_patch("seg/1", commentary="a")]),
        _source("b", [_patch("seg/1", commentary="b")], order=1),
    ])
    winner, loser = [
        item.candidate_sha256 for item in base.components[0].candidates
    ]
    plan = _plan([
        _source("a", [_patch("seg/1", commentary="a")]),
        _source("b", [_patch("seg/1", commentary="b")], order=1),
    ], supersession=[(winner, loser)])
    assert not plan.components
    assert [item.candidate_sha256 for item in plan.non_conflicting_atoms] == [
        winner
    ]
    assert plan.pruned_candidate_hashes == (loser,)


@pytest.mark.parametrize("kind", ["unknown", "cross", "self", "cycle"])
def test_invalid_supersession_fails_closed(kind) -> None:
    base = _plan([
        _source("a", [
            _patch("seg/1", commentary="a"),
            _patch("seg/1", explanation="x"),
        ]),
        _source("b", [
            _patch("seg/1", commentary="b"),
            _patch("seg/1", explanation="y"),
        ], order=1),
    ])
    commentary = base.components[0].candidates
    explanation = base.components[1].candidates
    if kind == "unknown":
        edges = [(commentary[0].candidate_sha256, "0" * 64)]
    elif kind == "cross":
        edges = [(
            commentary[0].candidate_sha256,
            explanation[0].candidate_sha256,
        )]
    elif kind == "self":
        edges = [(
            commentary[0].candidate_sha256,
            commentary[0].candidate_sha256,
        )]
    else:
        edges = [
            (
                commentary[0].candidate_sha256,
                commentary[1].candidate_sha256,
            ),
            (
                commentary[1].candidate_sha256,
                commentary[0].candidate_sha256,
            ),
        ]
    with pytest.raises(ReviewArbitrationError):
        _plan([
            _source("a", [
                _patch("seg/1", commentary="a"),
                _patch("seg/1", explanation="x"),
            ]),
            _source("b", [
                _patch("seg/1", commentary="b"),
                _patch("seg/1", explanation="y"),
            ], order=1),
        ], supersession=edges)


def test_apply_nonconflicting_trial_preserves_invalid_as_unresolved() -> None:
    plan = _plan([_source("s", [
        _patch("seg/1", commentary="valid", explanation="invalid"),
    ])])
    baseline = {"annotation": {"commentary": "old", "explanation": "old"}}

    def apply(value, candidate):
        value["annotation"][candidate.target_id] = candidate.replacement

    def validate(_value, candidate):
        if candidate.target_id == "explanation":
            raise RuntimeError("invalid")

    partial, invalid = apply_non_conflicting_atoms(
        plan, baseline, apply_atom=apply, candidate_validator=validate,
    )
    assert partial["annotation"] == {
        "commentary": "valid",
        "explanation": "old",
    }
    assert [item.target_id for item in invalid] == ["explanation"]
    assert baseline["annotation"]["commentary"] == "old"


def test_each_nonconflicting_candidate_is_validated_on_original_baseline() -> None:
    plan = _plan([_source("s", [
        _patch("seg/1", commentary="first", explanation="second"),
    ])])
    baseline = {"annotation": {"commentary": "old", "explanation": "old"}}
    seen = []

    def apply(value, candidate):
        value["annotation"][candidate.target_id] = candidate.replacement

    def validate(value, candidate):
        seen.append((candidate.target_id, deepcopy(value["annotation"])))

    partial, invalid = apply_non_conflicting_atoms(
        plan, baseline, apply_atom=apply, candidate_validator=validate,
    )
    assert not invalid
    assert seen == [
        ("commentary", {"commentary": "first", "explanation": "old"}),
        ("explanation", {"commentary": "old", "explanation": "second"}),
    ]
    assert partial["annotation"] == {
        "commentary": "first",
        "explanation": "second",
    }


def test_semantic_input_binds_original_and_invariant_hashes() -> None:
    source = _source("a", [_patch("seg/1", commentary="a")])
    other = _source("b", [_patch("seg/1", commentary="b")], order=1)
    path = "/segments/seg~11/annotation/commentary"
    plan = plan_review_merge(
        [source, other],
        segment_order=["seg/1", "seg~2"],
        block_order_by_segment={"seg/1": ["b/1", "b~2"], "seg~2": ["b3"]},
        original_value_resolver=lambda value: {"path": value, "value": "old"},
        invariant_context_resolver=lambda component: {
            "allowed_source_hashes": [canonical_sha256(component.path)]
        },
    )
    assert plan.semantic_input["original_target_hashes"] == {
        path: canonical_sha256({"path": path, "value": "old"})
    }
    assert plan.semantic_input["invariant_hashes"] == {
        path: canonical_sha256({
            "allowed_source_hashes": [canonical_sha256(path)]
        })
    }
    assert set(plan.semantic_input["original_target_hashes"]) == {
        item.path for item in plan.canonical_groups
    }


def test_arbitration_payload_reads_frozen_snapshots() -> None:
    original = {"value": "old"}
    invariant = {"allowed_source_hashes": ["a" * 64]}
    plan = plan_review_merge(
        [
            _source("a", [_patch("seg/1", commentary="a")]),
            _source("b", [_patch("seg/1", commentary="b")], order=1),
        ],
        segment_order=["seg/1", "seg~2"],
        block_order_by_segment={"seg/1": ["b/1", "b~2"], "seg~2": ["b3"]},
        original_value_resolver=lambda _path: original,
        invariant_context_resolver=lambda _component: invariant,
    )
    original["value"] = "mutated"
    invariant["allowed_source_hashes"].append("b" * 64)
    payload = arbitration_payload(plan)
    assert payload["conflicts"][0]["original"] == {"value": "old"}
    assert payload["conflicts"][0]["invariant_context"] == {
        "allowed_source_hashes": ["a" * 64]
    }


def test_trial_validation_includes_conflicting_candidates() -> None:
    plan = _plan([
        _source("a", [_patch("seg/1", commentary="a")]),
        _source("b", [_patch("seg/1", commentary="invalid")], order=1),
    ])
    baseline = {"annotation": {"commentary": "old"}}

    def apply(value, candidate):
        value["annotation"]["commentary"] = candidate.replacement

    invalid = trial_validate_candidates(
        plan,
        baseline,
        apply_atom=apply,
        candidate_validator=lambda _value, candidate: (
            (_ for _ in ()).throw(RuntimeError("invalid"))
            if candidate.replacement == "invalid" else None
        ),
    )
    assert [item.replacement for item in invalid] == ["invalid"]


def test_materialize_translation_overlay_is_complete_and_reasons_stable() -> None:
    plan = _plan([_source("s", [
        _patch(
            "seg/1",
            translation_blocks=[{"block_id": "b~2", "text": "changed"}],
            commentary="comment",
            reason="first",
        ),
        _patch("seg/1", explanation="explain", reason="second"),
    ])])
    patches = materialize_review_patches(
        plan.non_conflicting_atoms,
        segment_order=["seg/1", "seg~2"],
        original_translation_blocks=lambda _segment: [
            {"block_id": "b/1", "text": "one"},
            {"block_id": "b~2", "text": "two"},
        ],
    )
    assert len(patches) == 1
    assert patches[0]["translation_blocks"] == [
        {"block_id": "b/1", "text": "one"},
        {"block_id": "b~2", "text": "changed"},
    ]
    assert patches[0]["commentary"] == "comment"
    assert patches[0]["explanation"] == "explain"
    assert patches[0]["reason"] == "first; second"


def test_materialize_rejects_invalid_baseline_and_duplicate_targets() -> None:
    plan = _plan([_source("s", [
        _patch("seg/1", commentary="comment"),
    ])])
    candidate = plan.non_conflicting_atoms[0]
    with pytest.raises(ReviewArbitrationError, match="duplicate materialized"):
        materialize_review_patches(
            [candidate, candidate],
            segment_order=["seg/1", "seg~2"],
            original_translation_blocks=lambda _segment: [],
        )
    with pytest.raises(ReviewArbitrationError, match="baseline translation"):
        materialize_review_patches(
            [candidate],
            segment_order=["seg/1", "seg~2"],
            original_translation_blocks=lambda _segment: [
                {"block_id": "dup", "text": "a"},
                {"block_id": "dup", "text": "b"},
            ],
        )


def test_findings_and_issues_preserve_order_and_duplicates() -> None:
    repeated = {"segment_id": "seg/1", "issue": "same"}
    plan = _plan([
        _source("a", [_patch("seg/1", commentary="a")],
                findings=[repeated]),
        _source("b", [_patch("seg/1", commentary="a")], order=1,
                findings=[repeated]),
        ReviewPatchSource.from_review(
            source_kind="final",
            stable_order=2,
            review={"patches": [], "issues": ["same"]},
        ),
        ReviewPatchSource.from_review(
            source_kind="final",
            stable_order=3,
            review={"patches": [], "issues": ["same"]},
        ),
    ])
    assert list(plan.findings) == [repeated, repeated]
    assert list(plan.issues) == ["same", "same"]
    assert len(plan.finding_hashes) == 2
    assert len(plan.issue_hashes) == 2


def test_arbitration_payload_contains_only_conflict_context() -> None:
    plan = _plan([
        _source("a", [_patch("seg/1", commentary="a")]),
        _source("b", [_patch("seg/1", commentary="b")], order=1),
    ])
    payload = arbitration_payload(plan)
    text = json.dumps(payload)
    assert len(payload["conflicts"]) == 1
    assert "unrelated sentinel" not in text
    assert payload["conflicts"][0]["original"] == {
        "original_for": plan.components[0].path
    }


def test_select_keep_and_unresolved_semantics() -> None:
    plan = _plan([
        _source("a", [
            _patch("seg/1", commentary="a"),
            _patch("seg/1", explanation="x"),
        ]),
        _source("b", [
            _patch("seg/1", commentary="b"),
            _patch("seg/1", explanation="y"),
        ], order=1),
    ])
    commentary, explanation = plan.components
    selected = commentary.candidates[0]
    output = {"decisions": [
        _decision(
            commentary,
            action="select_candidate",
            selected=[selected.candidate_sha256],
            patch=candidate_to_patch(selected),
        ),
        _decision(explanation, action="keep_original"),
    ]}
    resolution = validate_arbitration_output(output, plan)
    assert [item.candidate_sha256 for item in resolution.resolved_candidates] == [
        selected.candidate_sha256
    ]
    assert resolution.unresolved_paths == ()

    output["decisions"][1]["action"] = "unresolved"
    resolution = validate_arbitration_output(output, plan)
    assert resolution.unresolved_paths == (explanation.path,)


def test_merge_semantics_require_two_candidates_and_exact_target() -> None:
    plan = _plan([
        _source("a", [_patch("seg/1", commentary="first")]),
        _source("b", [_patch("seg/1", commentary="second")], order=1),
    ])
    component = plan.components[0]
    merged = {
        **candidate_to_patch(component.candidates[0]),
        "commentary": "synthesized",
    }
    output = {"decisions": [_decision(
        component,
        action="merge_candidates",
        selected=[
            item.candidate_sha256 for item in component.candidates
        ],
        patch=merged,
    )]}
    checked = []
    resolution = validate_arbitration_output(
        output,
        plan,
        synthesized_candidate_validator=lambda _component, value: checked.append(
            value
        ),
    )
    assert checked == ["synthesized"]
    assert resolution.resolved_candidates[0].replacement == "synthesized"

    output["decisions"][0]["selected_candidate_hashes"] = [
        component.candidates[0].candidate_sha256
    ]
    with pytest.raises(ReviewArbitrationError, match="at least two"):
        validate_arbitration_output(output, plan)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing", "paths"),
        ("extra", "paths"),
        ("duplicate", "duplicate"),
        ("foreign", "foreign"),
        ("wrong_target", "targets"),
        ("selection_mismatch", "differs"),
        ("keep_with_patch", "must not"),
    ],
)
def test_invalid_decisions_fail_closed(mutation, match) -> None:
    plan = _plan([
        _source("a", [_patch("seg/1", commentary="a")]),
        _source("b", [_patch("seg/1", commentary="b")], order=1),
    ])
    component = plan.components[0]
    candidate = component.candidates[0]
    decision = _decision(
        component,
        action="select_candidate",
        selected=[candidate.candidate_sha256],
        patch=candidate_to_patch(candidate),
    )
    decisions = [decision]
    if mutation == "missing":
        decisions = []
    elif mutation == "extra":
        decisions.append({**decision, "path": "/extra"})
    elif mutation == "duplicate":
        decisions.append(dict(decision))
    elif mutation == "foreign":
        decision["selected_candidate_hashes"] = ["f" * 64]
    elif mutation == "wrong_target":
        decision["replacement_patch"] = {
            **candidate_to_patch(candidate),
            "commentary": None,
            "explanation": "wrong",
        }
    elif mutation == "selection_mismatch":
        decision["replacement_patch"] = {
            **candidate_to_patch(candidate),
            "commentary": "not selected",
        }
    else:
        decision["action"] = "keep_original"
    with pytest.raises(ReviewArbitrationError, match=match):
        validate_arbitration_output({"decisions": decisions}, plan)


def test_sanitized_conflict_requires_synthesis_and_keeps_finding_counts() -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "barthes_review_conflicts.json"
        ).read_text(encoding="utf-8")
    )
    sources = [
        ReviewPatchSource.from_review(
            source_kind=item["source_kind"],
            stable_order=item["stable_order"],
            review=item["review"],
            segment_set=fixture["segment_order"],
        )
        for item in fixture["sources"]
    ]
    plan = plan_review_merge(
        sources,
        segment_order=fixture["segment_order"],
        block_order_by_segment=fixture["block_order_by_segment"],
        original_value_resolver=lambda path: path,
        invariant_context_resolver=lambda component: {"path": component.path},
    )
    assert len(plan.findings) == 2
    assert len(plan.components) == 2
    translation_component = next(
        item for item in plan.components
        if "/translation_blocks/" in item.path
    )
    explanation_component = next(
        item for item in plan.components
        if "/annotation/explanation" in item.path
    )
    merged_patch = candidate_to_patch(
        translation_component.candidates[0],
    )
    merged_patch["translation_blocks"] = [
        fixture["expected_synthesized_block"]
    ]
    output = {"decisions": [
        _decision(
            translation_component,
            action="merge_candidates",
            selected=[
                item.candidate_sha256
                for item in translation_component.candidates
            ],
            patch=merged_patch,
        ),
        _decision(explanation_component, action="unresolved"),
    ]}
    resolution = validate_arbitration_output(output, plan)
    assert resolution.resolved_candidates[0].replacement == (
        fixture["expected_synthesized_block"]
    )
    assert resolution.unresolved_paths == (explanation_component.path,)


def _minimal_full_review_schema():
    return {
        "type": "object",
        "required": ["patches", "issues"],
        "properties": {
            "patches": {"type": "array", "items": {"type": "object"}},
            "issues": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": False,
    }


def test_full_materialized_review_schema_and_domain_reapplication() -> None:
    path = "/segments/seg~11/annotation/commentary"
    review = {"patches": [_patch("seg/1", commentary="safe")], "issues": []}
    applied = []

    def project(value, paths):
        return (
            deepcopy(value)
            if path in paths else {"patches": [], "issues": list(value["issues"])}
        )

    def apply(state, value):
        for patch in value["patches"]:
            state["commentary"] = patch["commentary"]
            applied.append(patch["commentary"])

    state = validate_materialized_review(
        review,
        full_schema=_minimal_full_review_schema(),
        baseline={"commentary": "old"},
        changed_paths=[path],
        apply_review=apply,
        project_review=project,
    )
    assert state == {"commentary": "safe"}
    assert applied == ["safe"]

    with pytest.raises(ReviewArbitrationNeedsSupervision, match="schema"):
        validate_materialized_review(
            {**review, "extra": True},
            full_schema=_minimal_full_review_schema(),
            baseline={},
            changed_paths=[path],
            apply_review=lambda _state, _value: None,
            project_review=project,
        )


@pytest.mark.parametrize(
    "failure_value",
    ["new-url", "token-loss", "name-loss", "coverage-loss"],
)
def test_domain_callback_localizes_single_target_failures(failure_value) -> None:
    path = "/segments/seg~11/annotation/commentary"
    review = {"patches": [{"value": failure_value}], "issues": []}

    def project(value, paths):
        return deepcopy(value) if path in paths else {"patches": [], "issues": []}

    def apply(_state, value):
        if value["patches"]:
            raise RuntimeError(failure_value)

    with pytest.raises(ReviewArbitrationNeedsSupervision) as caught:
        validate_materialized_review(
            review,
            full_schema=_minimal_full_review_schema(),
            baseline={},
            changed_paths=[path],
            apply_review=apply,
            project_review=project,
        )
    assert caught.value.paths == (path,)


def test_cross_field_failure_isolates_smallest_combination() -> None:
    commentary = "/segments/seg~11/annotation/commentary"
    explanation = "/segments/seg~11/annotation/explanation"
    later = "/segments/seg~02/annotation/later_work"
    review = {"patches": [], "issues": []}

    def project(_value, paths):
        return {"patches": [{"path": path} for path in paths], "issues": []}

    def apply(_state, value):
        paths = {item["path"] for item in value["patches"]}
        if {commentary, explanation}.issubset(paths):
            raise RuntimeError("cross-field failure")

    with pytest.raises(ReviewArbitrationNeedsSupervision) as caught:
        validate_materialized_review(
            review,
            full_schema=_minimal_full_review_schema(),
            baseline={},
            changed_paths=[commentary, explanation, later],
            apply_review=apply,
            project_review=project,
        )
    assert caught.value.paths == (commentary, explanation)
