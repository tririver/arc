from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading

import jsonschema
import pytest

from arc_companion import pipeline as pipeline_module
from arc_companion.cli import _emit
from arc_companion.evidence import arc_cache_descriptor, text_sha256, validate_evidence_record
from arc_companion.pipeline import (
    BuildOptions,
    CompanionLLMCircuitOpen,
    CompanionLaneError,
    LANGUAGE_NOTICE,
    _evidence,
    _checkpoint_dir_with_legacy_worker_migration,
    _fingerprint,
    _first_wave_preview_outputs_match,
    _legacy_worker_fingerprint,
    _limit_llm_concurrency,
    _evidence_for_segment,
    _first_wave_preview_document,
    _full_paper_context,
    _generation_document,
    _generate_translations,
    _llm_call,
    _review,
    _translation_input_block,
    _validate_translation,
    _protected_names,
    _repair_reviewed_translation_checkpoint,
    _segment_checkpoint_name,
    _state,
    build_companion,
    validate_and_expand_segments,
    validate_project,
)
from arc_companion.prompts import ANNOTATION_SCHEMA
from arc_companion.source import SourceBundle
from arc_companion.run_lock import ProjectBuildLock


def _inline_run(kind: str, content: str, order: int, *, tex: str = "") -> dict[str, str | int]:
    digest = hashlib.sha256(f"{kind}:{content}".encode()).hexdigest()
    value: dict[str, str | int] = {
        "kind": kind,
        "content": content,
        "order": order,
        "token_id": f"b.token-{order:04d}-{digest[:12]}",
        "content_hash": digest,
    }
    if tex:
        value["tex"] = tex
    return value


def _offset_slots(
    block: dict[str, object], residue: str, cuts: list[int],
) -> list[dict[str, str | int]]:
    boundaries = [0, *cuts, len(residue)]
    slot_ids = pipeline_module._translation_repair_slot_ids(block)
    assert len(boundaries) == len(slot_ids) + 1
    return [
        {
            "slot_id": slot_id,
            "start_offset": boundaries[index],
            "end_offset": boundaries[index + 1],
        }
        for index, slot_id in enumerate(slot_ids)
    ]


def test_translation_uses_and_validates_ordered_opaque_inline_tokens() -> None:
    block = {
        "block_id": "b",
        "kind": "prose",
        "text": r"The t_NL and f_NL squared terms.",
        "inline_runs": [
            _inline_run("text", "The ", 1),
            _inline_run("math", r"t_{NL}", 2, tex=r"t_{NL}"),
            _inline_run("text", " and ", 3),
            _inline_run("math", r"f_{NL}^{2}", 4, tex=r"f_{NL}^{2}"),
            _inline_run("text", " squared terms.", 5),
        ],
    }
    projected = _translation_input_block(block)
    tokens = pipeline_module._OPAQUE_INLINE_PATTERN.findall(projected["text"])
    segment = {"segment_id": "s", "block_ids": ["b"]}

    assert len(tokens) == 2
    _validate_translation(segment, {"blocks": [{"block_id": "b", "text": f"译文 {tokens[0]} 与 {tokens[1]}。"}]}, {"b": block}, [])
    try:
        _validate_translation(segment, {"blocks": [{"block_id": "b", "text": f"译文 {tokens[1]} 与 {tokens[0]}。"}]}, {"b": block}, [])
    except RuntimeError as exc:
        assert "opaque inline tokens" in str(exc)
    else:
        raise AssertionError("reordered math tokens must be rejected")


def test_translation_rejects_an_extra_opaque_link_occurrence() -> None:
    link_run = _inline_run("link", "project page", 2)
    link_run["href"] = "https://example.test/project"
    block = {
        "block_id": "linked",
        "kind": "prose",
        "text": "See project page.",
        "inline_runs": [
            _inline_run("text", "See ", 1),
            link_run,
            _inline_run("text", ".", 3),
        ],
    }
    token = pipeline_module._opaque_inline_tokens(block)[0]
    segment = {"segment_id": "seg-link", "block_ids": ["linked"]}

    _validate_translation(
        segment, {"blocks": [{"block_id": "linked", "text": f"参见 {token}。"}]},
        {"linked": block}, [],
    )
    try:
        _validate_translation(
            segment,
            {"blocks": [{"block_id": "linked", "text": f"参见 {token} 和 {token}。"}]},
            {"linked": block},
            [],
        )
    except RuntimeError as exc:
        assert "opaque inline tokens" in str(exc)
    else:
        raise AssertionError("a duplicated rendered link token must be rejected")


def test_slot_repair_preserves_natural_text_and_assembles_twenty_two_mixed_tokens() -> None:
    inline_runs = [_inline_run("text", "Start ", 1)]
    for number in range(1, 23):
        kind = ("math", "citation", "link")[(number - 1) % 3]
        run = _inline_run(kind, f"opaque-{number}", number * 2, tex=f"x_{{{number}}}")
        if kind == "link":
            run["href"] = f"https://example.test/{number}"
        inline_runs.extend([run, _inline_run("text", f" text-{number} ", number * 2 + 1)])
    block = {
        "block_id": "dense", "type": "text", "text": "Dense mixed inline content.",
        "inline_runs": inline_runs,
    }
    residue_slots = [f"译文-{index}" for index in range(23)]
    previous = {
        "blocks": [
            {"block_id": "unchanged", "text": "必须逐字节保留。"},
            {"block_id": "dense", "text": "".join(residue_slots)},
        ]
    }
    slot_ids = pipeline_module._translation_repair_slot_ids(block)
    repaired = {
        "block_id": "dense",
        "slots": [
            {"slot_id": slot_id, "text": text}
            for slot_id, text in zip(slot_ids, residue_slots)
        ],
    }

    assembled = pipeline_module._apply_translation_slot_repairs(
        previous, [block], {"repairs": [repaired]}, protected_names=[],
    )
    assert assembled["blocks"][0] == previous["blocks"][0]
    text = assembled["blocks"][1]["text"]
    expected_tokens = pipeline_module._opaque_inline_tokens(block)
    assert len(expected_tokens) == 22
    assert pipeline_module._OPAQUE_INLINE_PATTERN.findall(text) == expected_tokens
    assert pipeline_module._OPAQUE_INLINE_PATTERN.sub("", text) == "".join(residue_slots)
    _validate_translation(
        {"segment_id": "seg-dense", "block_ids": ["dense"]},
        {"blocks": [assembled["blocks"][1]]},
        {"dense": block},
        [],
    )


def _bracketed_citation_block() -> dict:
    return {
        "block_id": "cited", "type": "text", "text": "Read as [9].",
        "inline_runs": [
            _inline_run("text", "Read as [", 1),
            _inline_run("citation", "9", 2),
            _inline_run("text", "].", 3),
        ],
    }


def test_citation_delimiter_normalizer_keeps_correct_and_relocates_empty_pair() -> None:
    block = _bracketed_citation_block()
    token = pipeline_module._opaque_inline_tokens(block)[0]

    assert pipeline_module._normalize_translation_citation_delimiters(
        block, f"参见[{token}]。",
    ) == f"参见[{token}]。"
    assert pipeline_module._normalize_translation_citation_delimiters(
        block, f"参见[]{token}。",
    ) == f"参见[{token}]。"
    assert pipeline_module._normalize_translation_citation_delimiters(
        block, f"参见{token}[]。",
    ) == f"参见[{token}]。"
    assert pipeline_module._normalize_translation_citation_delimiters(
        block, f"参见{token}]。",
    ) == f"参见[{token}]。"
    assert pipeline_module._normalize_translation_citation_delimiters(
        block, f"参见{token}。",
    ) == f"参见[{token}]。"


@pytest.mark.parametrize(
    "text_template",
    ("参见[额外]{}。", "参见{}[额外]。", "参见[]{}[]。"),
)
def test_citation_delimiter_normalizer_rejects_missing_ambiguous_or_nonempty_brackets(
    text_template: str,
) -> None:
    block = _bracketed_citation_block()
    token = pipeline_module._opaque_inline_tokens(block)[0]

    with pytest.raises(RuntimeError, match="citation delimiter normalization"):
        pipeline_module._normalize_translation_citation_delimiters(
            block, text_template.format(token),
        )


def test_slot_repair_relocates_citation_brackets_without_changing_residue() -> None:
    block = _bracketed_citation_block()
    token = pipeline_module._opaque_inline_tokens(block)[0]
    previous = {"blocks": [{"block_id": "cited", "text": "参见[]。"}]}
    slot_ids = pipeline_module._translation_repair_slot_ids(block)

    result = pipeline_module._apply_translation_slot_repairs(
        previous,
        [block],
        {"repairs": [{"block_id": "cited", "slots": [
            {"slot_id": slot_ids[0], "text": "参见"},
            {"slot_id": slot_ids[1], "text": "[]。"},
        ]}]},
        protected_names=[],
    )

    assembled = result["blocks"][0]["text"]
    assert assembled == f"参见{token}[]。"
    normalized = pipeline_module._normalize_translation_citation_delimiters(block, assembled)
    assert normalized == f"参见[{token}]。"
    assert pipeline_module._translation_natural_residue(normalized) == (
        "参见[]。"
    )


def test_offset_slot_repair_preserves_seg0016_residue_exactly() -> None:
    block = {
        "block_id": "S2.SS1.p9.18", "type": "text", "text": "A x B y C z.",
        "inline_runs": [
            _inline_run("text", "A ", 1), _inline_run("math", "x", 2, tex="x"),
            _inline_run("text", " B ", 3), _inline_run("math", "y", 4, tex="y"),
            _inline_run("text", " C ", 5), _inline_run("math", "z", 6, tex="z"),
            _inline_run("text", ".", 7),
        ],
    }
    tokens = pipeline_module._opaque_inline_tokens(block)
    prior_text = f"分别表{tokens[2]}示和{tokens[0]}，最后{tokens[1]}。"
    prior_residue = pipeline_module._translation_natural_residue(prior_text)
    previous = {"blocks": [{"block_id": block["block_id"], "text": prior_text}]}
    slot_ids = pipeline_module._translation_repair_slot_ids(block)
    boundaries = [3, 5, 8, len(prior_residue)]
    start = 0
    slots = []
    for slot_id, end in zip(slot_ids, boundaries):
        slots.append({"slot_id": slot_id, "start_offset": start, "end_offset": end})
        start = end
    repaired = pipeline_module._apply_translation_slot_repairs(
        previous, [block], {"repairs": [{"block_id": block["block_id"], "slots": slots}]},
        protected_names=[],
    )
    assert pipeline_module._translation_natural_residue(repaired["blocks"][0]["text"]) == prior_residue
    assert pipeline_module._OPAQUE_INLINE_PATTERN.findall(repaired["blocks"][0]["text"]) == tokens
    slots[1] = {**slots[1], "start_offset": slots[1]["start_offset"] + 1}
    with pytest.raises(RuntimeError, match="exactly partition prior residue"):
        pipeline_module._apply_translation_slot_repairs(
            previous, [block], {"repairs": [{"block_id": block["block_id"], "slots": slots}]},
            protected_names=[],
        )


def test_seg0007_slot_repair_then_synthesizes_source_owned_citation_brackets() -> None:
    block = {
        "block_id": "S1.p13.14", "type": "text", "text": "Value x [7].",
        "inline_runs": [
            _inline_run("text", "Value ", 1), _inline_run("math", "x", 2, tex="x"),
            _inline_run("text", " [", 3), _inline_run("citation", "7", 4),
            _inline_run("text", "].", 5),
        ],
    }
    previous = {"blocks": [{"block_id": block["block_id"], "text": "值并参见。"}]}
    slot_ids = pipeline_module._translation_repair_slot_ids(block)
    spans = [(0, 1), (1, 4), (4, 5)]
    repaired = pipeline_module._apply_translation_slot_repairs(
        previous, [block], {"repairs": [{"block_id": block["block_id"], "slots": [
            {"slot_id": slot_id, "start_offset": start, "end_offset": end}
            for slot_id, (start, end) in zip(slot_ids, spans)
        ]}]}, protected_names=[],
    )
    tokens = pipeline_module._opaque_inline_tokens(block)
    normalized, methods = pipeline_module._normalize_translation_citation_delimiters_for_segment(
        repaired, {block["block_id"]: block},
    )
    assert normalized["blocks"][0]["text"] == f"值{tokens[0]}并参见[{tokens[1]}]。"
    assert methods == {block["block_id"]: "synthesized"}


def test_seg0007_adjacent_math_marker_is_not_an_ambiguous_citation_bracket() -> None:
    block = {
        "block_id": "S1.p13.14", "type": "text", "text": "term f [7].",
        "inline_runs": [
            _inline_run("text", "term ", 1),
            _inline_run("math", "f", 2, tex="f_{NL}"),
            _inline_run("text", "[", 3),
            _inline_run("citation", "7", 4),
            _inline_run("text", "].", 5),
        ],
    }
    math_token, citation_token = pipeline_module._opaque_inline_tokens(block)
    repaired = {"blocks": [{
        "block_id": block["block_id"],
        "text": f"产生一个{math_token}{citation_token}项。",
    }]}

    normalized, methods = pipeline_module._normalize_translation_citation_delimiters_for_segment(
        repaired, {block["block_id"]: block},
    )

    assert normalized["blocks"][0]["text"] == (
        f"产生一个{math_token}[{citation_token}]项。"
    )
    assert methods == {block["block_id"]: "synthesized"}


def test_v5_offset_repair_preserves_v4_audit_once_and_persists_response(
    tmp_path: Path,
) -> None:
    block = {
        "block_id": "body", "type": "text", "text": "Value x.",
        "inline_runs": [
            _inline_run("text", "Value ", 1), _inline_run("math", "x", 2, tex="x"),
            _inline_run("text", ".", 3),
        ],
    }
    segment = {"segment_id": "seg-0016", "block_ids": ["body"]}
    translation = {"blocks": [{"block_id": "body", "text": "译文。"}]}
    checkpoint_dir = tmp_path / "checkpoints"
    input_sha256 = "fixture-input"
    old_marker = pipeline_module._legacy_translation_token_attempt_path(
        checkpoint_dir, segment["segment_id"],
    )
    old_marker.parent.mkdir(parents=True)
    old_marker.write_text(json.dumps({
        "schema_version": "arc.companion.translation-token-attempt.v1",
        "prompt_version": "arc.companion.translation-retry-prompt.v4",
        "segment_id": segment["segment_id"],
        "input_sha256": input_sha256,
        "status": "response_received",
        "raw_response": {"repairs": [{"block_id": "body", "slots": [
            {"slot_id": "legacy", "text": "改写过的译文。"},
        ]}]},
    }), encoding="utf-8")
    old_marker_bytes = old_marker.read_bytes()
    calls: list[str] = []
    def repair_llm(prompt: str, **kwargs):
        calls.append(str(kwargs["call_label"]))
        schema_text = json.dumps(kwargs["schema"])
        assert '"text"' not in schema_text
        assert '"start_offset"' in schema_text
        assert "PRIOR NATURAL LANGUAGE RESIDUE" in prompt
        assert "INDEXED RESIDUE" in prompt
        assert pipeline_module._opaque_inline_tokens(block)[0] in prompt
        return {"repairs": [{"block_id": "body", "slots": [
            *_offset_slots(block, "译文。", [2]),
        ]}]}
    repaired, provenance = pipeline_module._repair_translation_token_placement(
        segment, translation, blocks_by_id={"body": block}, protected_names=[],
        options=BuildOptions(paper_id="arXiv:0911.3380", project_dir=tmp_path),
        checkpoint_dir=checkpoint_dir, artifact_dir=checkpoint_dir / "llm",
        input_sha256=input_sha256, llm=repair_llm,
    )
    assert calls == ["companion-translation-seg-0016-retry-offset-1"]
    assert pipeline_module._translation_natural_residue(
        repaired["blocks"][0]["text"]
    ) == "译文。"
    assert provenance["attempt"] == 1
    persisted_path = pipeline_module._translation_token_repair_draft_path(
        checkpoint_dir, segment["segment_id"],
    )
    persisted = json.loads(persisted_path.read_text(encoding="utf-8"))
    assert persisted["raw_response"]["repairs"][0]["slots"][0]["end_offset"] == 2
    assert old_marker.read_bytes() == old_marker_bytes
    current_marker = pipeline_module._translation_token_attempt_path(
        checkpoint_dir, segment["segment_id"],
    )
    marker = json.loads(current_marker.read_text(encoding="utf-8"))
    assert marker["status"] == "validated"
    assert marker["superseded_text_attempt"]["prompt_version"].endswith(".v4")
    assert marker["superseded_text_attempt"]["sha256"] == pipeline_module.sha256_json(
        json.loads(old_marker_bytes)
    )

    def forbidden_llm(prompt: str, **kwargs):
        raise AssertionError("persisted structural repair must resume without another model call")

    tampered = json.loads(json.dumps(persisted))
    tampered["translation"]["blocks"][0]["text"] = (
        f"篡改{pipeline_module._opaque_inline_tokens(block)[0]}。"
    )
    persisted_path.write_text(json.dumps(tampered), encoding="utf-8")
    resumed, resumed_provenance = pipeline_module._repair_translation_token_placement(
        segment, translation, blocks_by_id={"body": block}, protected_names=[],
        options=BuildOptions(paper_id="arXiv:0911.3380", project_dir=tmp_path),
        checkpoint_dir=checkpoint_dir, artifact_dir=checkpoint_dir / "llm",
        input_sha256=input_sha256, llm=forbidden_llm,
    )
    assert resumed == repaired
    assert resumed_provenance == provenance
    recovered_marker = json.loads(current_marker.read_text(encoding="utf-8"))
    assert recovered_marker["status"] == "validated"
    assert recovered_marker["validated_translation_sha256"] == pipeline_module.sha256_json(
        repaired
    )
    assert recovered_marker["raw_response"] == persisted["raw_response"]
    assert old_marker.read_bytes() == old_marker_bytes


@pytest.mark.parametrize("legacy_version", ["v3", "v4"])
def test_legacy_final_checkpoint_requires_offset_upgrade(legacy_version: str) -> None:
    repair = {
        "kind": "token-placement",
        "prompt_version": f"arc.companion.translation-retry-prompt.{legacy_version}",
    }
    checkpoint = {
        "generation_provenance": {"repairs": [repair]},
        "translation": {"blocks": []},
    }
    assert pipeline_module._translation_checkpoint_requires_v4_upgrade(checkpoint)
    repair["prompt_version"] = pipeline_module.TRANSLATION_RETRY_PROMPT_VERSION
    repair["repair_mode"] = "offset-only"
    assert not pipeline_module._translation_checkpoint_requires_v4_upgrade(checkpoint)


def test_old_protected_name_checkpoint_is_rebuilt_from_clean_primary_draft(
    tmp_path: Path,
) -> None:
    block = {
        "block_id": "ordinary-lie",
        "type": "text",
        "text": "Choose vectors that lie in the transverse plane.",
        "inline_runs": [
            _inline_run("text", "Choose vectors that lie in the transverse plane.", 1),
        ],
    }
    document = {
        "front_matter": {}, "blocks": [block], "equations": [], "figures": [],
        "tables": [], "bibliography": [], "assets": [],
        "integrity": {"status": "complete"},
    }
    bundle = SourceBundle(
        paper_id="arXiv:1234.5678", parsed={"document": document}, document=document,
        metadata={}, references=[], citers=[],
    )
    segment = {"segment_id": "seg-ordinary-lie", "block_ids": ["ordinary-lie"]}
    clean = {
        "blocks": [{"block_id": "ordinary-lie", "text": "选择位于横向平面内的向量。"}]
    }
    polluted = {
        "blocks": [{"block_id": "ordinary-lie", "text": "选择位于横向平面内的向量（Lie）。"}]
    }
    checkpoint_dir = tmp_path / "checkpoints"
    options = BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path, workers=1)
    draft_path = pipeline_module._seed_translation_coverage_draft(
        segment, options=options, bundle=bundle, glossary={"entries": []},
        protected_names=["Lie"], checkpoint_dir=checkpoint_dir, translation=clean,
    )
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    draft["candidate_provenance"] = {
        "origin": "primary-model", "prompt_version": "fixture",
        "response_schema_version": "fixture", "model_tier": "low",
    }
    draft_path.write_text(json.dumps(draft), encoding="utf-8")
    translation_path = (
        checkpoint_dir / "translations"
        / f"{pipeline_module._segment_checkpoint_name(segment['segment_id'])}.json"
    )
    translation_path.parent.mkdir(parents=True)
    translation_path.write_text(json.dumps({
        "segment_id": segment["segment_id"],
        "input_sha256": draft["input_sha256"],
        "generation_provenance": {
            "candidate": draft["candidate_provenance"],
            "repairs": [{
                "kind": "protected-name-normalization", "attempt": 0,
                "normalizer_version": "arc.companion.translation-protected-names.v1",
                "repaired_block_ids": ["ordinary-lie"],
            }],
        },
        "translation": polluted,
    }), encoding="utf-8")

    def forbidden_llm(prompt: str, **kwargs):
        raise AssertionError("deterministic checkpoint migration must not call a model")

    result = _generate_translations(
        [segment], options=options, bundle=bundle, glossary={"entries": []},
        protected_names=["Lie"], checkpoint_dir=checkpoint_dir, llm=forbidden_llm,
    )

    assert result[segment["segment_id"]] == clean
    migrated = json.loads(translation_path.read_text(encoding="utf-8"))
    assert migrated["translation"] == clean
    assert migrated["generation_provenance"]["repairs"] == []


def test_v4_repairs_seg0007_and_seg0016_semantic_roles_in_mutable_clauses() -> None:
    seg7 = {
        "block_id": "S1.p13.14", "type": "text", "text": "contributes O to f [7].",
        "inline_runs": [
            _inline_run("text", "contributes ", 1),
            _inline_run("math", "O", 2, tex="{\\cal O}(\\epsilon^2)"),
            _inline_run("text", " to ", 3),
            _inline_run("math", "f", 4, tex="f_{NL}"),
            _inline_run("text", " [", 5),
            _inline_run("citation", "7", 6),
            _inline_run("text", "].", 7),
        ],
    }
    o_token, f_token, citation = pipeline_module._opaque_inline_tokens(seg7)
    prior7 = {"blocks": [{
        "block_id": seg7["block_id"],
        "text": f"这会对{o_token}产生一个{f_token}{citation}项。",
    }]}
    ids7 = pipeline_module._translation_repair_slot_ids(seg7)
    repaired7 = pipeline_module._apply_translation_slot_repairs(
        prior7, [seg7], {"repairs": [{"block_id": seg7["block_id"], "slots": [
            {"slot_id": ids7[0], "text": "这会产生一个"},
            {"slot_id": ids7[1], "text": "项，并贡献给"},
            {"slot_id": ids7[2], "text": ""},
            {"slot_id": ids7[3], "text": "。"},
        ]}]}, protected_names=[], allow_clause_rewrite=True,
    )
    normalized7, _ = pipeline_module._normalize_translation_citation_delimiters_for_segment(
        repaired7, {seg7["block_id"]: seg7},
    )
    assert normalized7["blocks"][0]["text"] == (
        f"这会产生一个{o_token}项，并贡献给{f_token}[{citation}]。"
    )

    seg16 = {
        "block_id": "S2.SS1.p9.18", "type": "text", "text": "replace p with q using r.",
        "inline_runs": [
            _inline_run("text", "replace ", 1), _inline_run("math", "p", 2, tex="p"),
            _inline_run("text", " with ", 3), _inline_run("math", "q", 4, tex="q"),
            _inline_run("text", " using ", 5), _inline_run("math", "r", 6, tex="r"),
            _inline_run("text", ".", 7),
        ],
    }
    p_token, q_token, relation = pipeline_module._opaque_inline_tokens(seg16)
    prior16 = {"blocks": [{
        "block_id": seg16["block_id"],
        "text": f"将{p_token}替换{q_token}为，利用关系{relation}。本段保持不变。",
    }]}
    ids16 = pipeline_module._translation_repair_slot_ids(seg16)
    repaired16 = pipeline_module._apply_translation_slot_repairs(
        prior16, [seg16], {"repairs": [{"block_id": seg16["block_id"], "slots": [
            {"slot_id": ids16[0], "text": "把"},
            {"slot_id": ids16[1], "text": "替换为"},
            {"slot_id": ids16[2], "text": "，所用关系为"},
            {"slot_id": ids16[3], "text": "。本段保持不变。"},
        ]}]}, protected_names=[], allow_clause_rewrite=True,
    )
    assert repaired16["blocks"][0]["text"] == (
        f"把{p_token}替换为{q_token}，所用关系为{relation}。本段保持不变。"
    )


def test_seg0017_offset_repair_keeps_prose_and_negation_byte_exact() -> None:
    block = {
        "block_id": "S2.SS1.p9.18", "type": "text",
        "text": "Do not replace p with q except by r.",
        "inline_runs": [
            _inline_run("text", "Do not replace ", 1),
            _inline_run("math", "p", 2, tex="p"),
            _inline_run("text", " with ", 3),
            _inline_run("math", "q", 4, tex="q"),
            _inline_run("text", " except by ", 5),
            _inline_run("math", "r", 6, tex="r"),
            _inline_run("text", ".", 7),
        ],
    }
    p_token, q_token, relation = pipeline_module._opaque_inline_tokens(block)
    primary_text = f"不应将{q_token}替换为{p_token}，且仅利用{relation}。"
    residue = pipeline_module._translation_natural_residue(primary_text)
    cuts = [residue.index("替换"), residue.index("，"), residue.index("。")]

    repaired = pipeline_module._apply_translation_slot_repairs(
        {"blocks": [{"block_id": block["block_id"], "text": primary_text}]},
        [block],
        {"repairs": [{"block_id": block["block_id"], "slots": [
            *_offset_slots(block, residue, cuts),
        ]}]},
        protected_names=[], offset_only=True,
    )
    repaired_text = repaired["blocks"][0]["text"]
    repaired_residue = pipeline_module._translation_natural_residue(repaired_text)

    assert repaired_text == (
        f"不应将{p_token}替换为{q_token}，且仅利用{relation}。"
    )
    assert repaired_residue.encode("utf-8") == residue.encode("utf-8")
    assert repaired_residue.count("不") == residue.count("不") == 1
    assert repaired_residue.count("且") == residue.count("且") == 1

    slot_ids = pipeline_module._translation_repair_slot_ids(block)
    with pytest.raises(RuntimeError, match="returned prose"):
        pipeline_module._apply_translation_slot_repairs(
            {"blocks": [{"block_id": block["block_id"], "text": primary_text}]},
            [block],
            {"repairs": [{"block_id": block["block_id"], "slots": [
                {"slot_id": slot_id, "text": "可以" if index == 0 else ""}
                for index, slot_id in enumerate(slot_ids)
            ]}]},
            protected_names=[], offset_only=True,
        )


def test_v4_rejects_changes_outside_token_bearing_clause() -> None:
    block = {
        "block_id": "body", "type": "text", "text": "Value x. Stable sentence.",
        "inline_runs": [
            _inline_run("text", "Value ", 1), _inline_run("math", "x", 2, tex="x"),
            _inline_run("text", ". Stable sentence.", 3),
        ],
    }
    token = pipeline_module._opaque_inline_tokens(block)[0]
    prior = {"blocks": [{"block_id": "body", "text": f"数值{token}。稳定句。"}]}
    slot_ids = pipeline_module._translation_repair_slot_ids(block)
    with pytest.raises(RuntimeError, match="outside mutable clauses"):
        pipeline_module._apply_translation_slot_repairs(
            prior, [block], {"repairs": [{"block_id": "body", "slots": [
                {"slot_id": slot_ids[0], "text": "值为"},
                {"slot_id": slot_ids[1], "text": "。稳定句被改。"},
            ]}]}, protected_names=[], allow_clause_rewrite=True,
        )


def test_v4_keeps_unaffected_token_bearing_clause_byte_exact() -> None:
    block = {
        "block_id": "body", "type": "text", "text": "A x. B y.",
        "inline_runs": [
            _inline_run("text", "A ", 1), _inline_run("math", "x", 2, tex="x"),
            _inline_run("text", ". B ", 3), _inline_run("math", "y", 4, tex="y"),
            _inline_run("text", ".", 5),
        ],
    }
    token1, token2 = pipeline_module._opaque_inline_tokens(block)
    primary = {"blocks": [{"block_id": "body", "text": f"稳定{token1}句。第二句缺失。"}]}
    v3 = {"blocks": [{"block_id": "body", "text": f"稳定{token1}句。第二{token2}句需修。"}]}
    slot_ids = pipeline_module._translation_repair_slot_ids(block)
    with pytest.raises(RuntimeError, match="outside mutable clauses"):
        pipeline_module._apply_translation_slot_repairs(
            v3, [block], {"repairs": [{"block_id": "body", "slots": [
                {"slot_id": slot_ids[0], "text": "稳定内容被改"},
                {"slot_id": slot_ids[1], "text": "句。第二"},
                {"slot_id": slot_ids[2], "text": "句已修。"},
            ]}]}, protected_names=[], allow_clause_rewrite=True,
            primary_translation=primary,
        )
    with pytest.raises(RuntimeError, match="outside mutable clauses"):
        pipeline_module._apply_translation_slot_repairs(
            v3, [block], {"repairs": [{"block_id": "body", "slots": [
                {"slot_id": slot_ids[0], "text": "稳"},
                {"slot_id": slot_ids[1], "text": "定句。第二"},
                {"slot_id": slot_ids[2], "text": "句需修。"},
            ]}]}, protected_names=[], allow_clause_rewrite=True,
            primary_translation=primary,
        )


def test_v4_duplicate_tokens_use_occurrence_anchored_immutable_slots() -> None:
    duplicate = _inline_run("math", "x", 2, tex="x")
    block = {
        "block_id": "body", "type": "text", "text": "A x. B x.",
        "inline_runs": [
            _inline_run("text", "A ", 1), duplicate,
            _inline_run("text", ". B ", 3), duplicate,
            _inline_run("text", ".", 5),
        ],
    }
    token1, token2 = pipeline_module._opaque_inline_tokens(block)
    assert token1 == token2
    primary_text = f"稳定{token1}句。第二句缺失。"
    v3_text = f"稳定{token1}句。第二{token2}句需修。"
    assert pipeline_module._translation_repair_affected_ordinals(
        block, primary_text, v3_text,
    ) == {2}
    prior = {"blocks": [{"block_id": "body", "text": v3_text}]}
    slot_ids = pipeline_module._translation_repair_slot_ids(block)
    with pytest.raises(RuntimeError, match="moved or copied"):
        pipeline_module._apply_translation_slot_repairs(
            prior, [block], {"repairs": [{"block_id": "body", "slots": [
                {"slot_id": slot_ids[0], "text": "稳定"},
                {"slot_id": slot_ids[1], "text": "句。稳定句。第二已修"},
                {"slot_id": slot_ids[2], "text": "句需修。"},
            ]}]}, protected_names=[], allow_clause_rewrite=True,
            primary_translation={"blocks": [{"block_id": "body", "text": primary_text}]},
        )


def test_token_repair_persists_response_before_apply_and_reuses_failed_response(
    tmp_path: Path,
) -> None:
    block = {
        "block_id": "body", "type": "text", "text": "Value x.",
        "inline_runs": [
            _inline_run("text", "Value ", 1),
            _inline_run("math", "x", 2, tex="x"),
            _inline_run("text", ".", 3),
        ],
    }
    segment = {"segment_id": "seg-audit", "block_ids": ["body"]}
    checkpoint_dir = tmp_path / "checkpoints"
    calls = 0

    def invalid_repair(prompt: str, **kwargs):
        nonlocal calls
        calls += 1
        slot_id = pipeline_module._translation_repair_slot_ids(block)[0]
        return {"repairs": [{"block_id": "body", "slots": [
            {"slot_id": slot_id, "text": "译文"},
        ]}]}

    arguments = dict(
        segment=segment,
        translation={"blocks": [{"block_id": "body", "text": "译文。"}]},
        blocks_by_id={"body": block}, protected_names=[],
        options=BuildOptions(paper_id="arXiv:0911.3380", project_dir=tmp_path),
        checkpoint_dir=checkpoint_dir, artifact_dir=checkpoint_dir / "llm",
        input_sha256="audit-input",
    )
    with pytest.raises(RuntimeError, match="slot repair"):
        pipeline_module._repair_translation_token_placement(
            **arguments, llm=invalid_repair,
        )
    marker_path = pipeline_module._translation_token_attempt_path(
        checkpoint_dir, segment["segment_id"],
    )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["status"] == "response_received"
    assert marker["raw_response"]["repairs"][0]["slots"][0]["text"] == "译文"

    def forbidden_llm(prompt: str, **kwargs):
        raise AssertionError("response_received must reuse the auditable raw response")

    with pytest.raises(RuntimeError, match="slot repair"):
        pipeline_module._repair_translation_token_placement(
            **arguments, llm=forbidden_llm,
        )
    assert calls == 1


def test_paid_token_repair_with_segment_wide_fake_slots_never_resubmits(
    tmp_path: Path,
) -> None:
    """The invalid response shape observed in production stays single-call supervised."""
    block = {
        "block_id": "body", "type": "text", "text": "Value x.",
        "inline_runs": [
            _inline_run("text", "Value ", 1),
            _inline_run("math", "x", 2, tex="x"),
            _inline_run("text", ".", 3),
        ],
    }
    segment = {"segment_id": "seg-invalid-paid", "block_ids": ["body"]}
    calls = 0

    def invalid_repair(prompt: str, **kwargs):
        nonlocal calls
        calls += 1
        return {"repairs": [
            {"block_id": "unrequested", "slots": [{
                "slot_id": "translation", "start_offset": 0, "end_offset": 0,
            }]},
            {"block_id": "body", "slots": [{
                "slot_id": "translation", "start_offset": 0, "end_offset": 0,
            }]},
        ]}

    arguments = dict(
        segment=segment,
        translation={"blocks": [{"block_id": "body", "text": "译文。"}]},
        blocks_by_id={"body": block}, protected_names=[],
        options=BuildOptions(paper_id="arXiv:0911.3380", project_dir=tmp_path),
        checkpoint_dir=tmp_path / "checkpoints",
        artifact_dir=tmp_path / "checkpoints" / "llm",
        input_sha256="invalid-paid-input",
    )
    with pytest.raises(
        pipeline_module.TranslationRepairNeedsSupervision,
        match="refusing resubmission",
    ) as first:
        pipeline_module._repair_translation_token_placement(
            **arguments, llm=invalid_repair,
        )
    assert first.value.recovery_context["submission_state"] == "submitted"
    assert first.value.recovery_context["resumable"] is False

    def forbidden_llm(prompt: str, **kwargs):
        raise AssertionError("invalid paid repair response must never be resubmitted")

    with pytest.raises(pipeline_module.TranslationRepairNeedsSupervision):
        pipeline_module._repair_translation_token_placement(
            **arguments, llm=forbidden_llm,
        )
    assert calls == 1


def test_wrapped_paid_repair_failure_is_persisted_on_lane_ledger(tmp_path: Path) -> None:
    ledger_path = tmp_path / "translation-ledger.json"
    pipeline_module.initialize_lane_ledger(
        ledger_path, chapter_id="ch-1", lane="translation", segment_ids=["seg-1"],
    )
    pipeline_module.mark_submitted(ledger_path, segment_id="seg-1")
    pipeline_module.mark_response_received(ledger_path, segment_id="seg-1")
    paid_failure = pipeline_module.TranslationRepairNeedsSupervision(
        segment_id="seg-1", marker_path=tmp_path / "paid-marker.json",
        reason="changed failing block coverage",
    )
    wrapped = pipeline_module.CompanionLaneError(
        "translation", [("seg-1", paid_failure)],
    )

    assert pipeline_module._mark_translation_repair_supervision(
        ledger_path, segment_id="seg-1", exc=wrapped,
    )
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    supervision = ledger["needs_supervision"]
    assert supervision["segment_id"] == "seg-1"
    assert supervision["recovery_context"]["submission_state"] == "submitted"
    assert supervision["recovery_context"]["resumable"] is False
    assert supervision["recovery_context"]["repair_marker"].endswith(
        "paid-marker.json"
    )


def test_paid_repair_finalizer_classifies_later_same_lane_responses(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    blocks = []
    for index in range(1, 4):
        blocks.append({
            "block_id": f"b{index}", "type": "text", "text": f"Value x{index}.",
            "inline_runs": [
                _inline_run("text", "Value ", index * 10 + 1),
                _inline_run("math", f"x{index}", index * 10 + 2, tex=f"x_{index}"),
                _inline_run("text", ".", index * 10 + 3),
            ],
        })
    pipeline_module.write_json(
        checkpoint / "document.json", {"document": {"blocks": blocks}},
    )
    ledger_path = checkpoint / "chapters" / "ch-1" / "translation-ledger.json"
    pipeline_module.initialize_lane_ledger(
        ledger_path, chapter_id="ch-1", lane="translation",
        segment_ids=["s1", "s2", "s3"],
    )
    # Source-order ledger progress stops at s1, while already-submitted worker
    # responses for s2/s3 can still land durably after that failure.
    pipeline_module.mark_submitted(ledger_path, segment_id="s1")
    pipeline_module.mark_response_received(ledger_path, segment_id="s1")

    for index, segment_id in enumerate(("s1", "s2", "s3"), start=1):
        key = f"repair-{index}"
        token_id = str(blocks[index - 1]["inline_runs"][1]["token_id"])
        valid = index == 3
        response = {"repairs": [{
            "block_id": f"b{index}" if valid else "unexpected",
            "slots": [{
                "slot_id": token_id if valid else "translation",
                "start_offset": 2 if valid else 0,
                "end_offset": 2 if valid else 0,
            }],
        }]}
        pipeline_module.write_json(
            checkpoint / "translation-drafts" / f"{key}.json", {
                "schema_version": "arc.companion.translation-primary-draft.v1",
                "segment_id": segment_id, "input_sha256": f"input-{index}",
                "translation": {"blocks": [{
                    "block_id": f"b{index}", "text": "译文。",
                }]},
            },
        )
        pipeline_module.write_json(
            checkpoint / "translation-token-offset-attempts" / f"{key}.json", {
                "segment_id": segment_id, "input_sha256": f"input-{index}",
                "status": "response_received", "block_ids": [f"b{index}"],
                "raw_response": response,
            },
        )
        pipeline_module.write_json(
            checkpoint / "llm" / "translations" / key / "retry-offset-1"
            / "call-checkpoints" / f"call-{index}.json", {
                "state": "validated", "submission_state": "submitted",
                "logical_identity": {"idempotency_key": f"idem-{index}"},
            },
        )

    entries = pipeline_module._finalize_paid_translation_repairs(checkpoint)

    assert {item["segment_id"] for item in entries} == {"s1", "s2", "s3"}
    actions = {item["segment_id"]: item["recovery_action"] for item in entries}
    assert actions == {
        "s1": "operator-supervision", "s2": "operator-supervision",
        "s3": "deterministic-replay",
    }
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert [item["segment_id"] for item in ledger["supervision_entries"]] == [
        "s1", "s2",
    ]
    assert ledger["needs_supervision"]["segment_id"] == "s1"


def test_paid_repair_finalizer_ignores_marker_from_abandoned_generation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    pipeline_module.write_json(
        checkpoint / "document.json", {"document": {"blocks": [{
            "block_id": "body", "type": "text", "text": "Source text.",
        }]}},
    )
    ledger_path = checkpoint / "chapters" / "ch-1" / "translation-ledger.json"
    pipeline_module.initialize_lane_ledger(
        ledger_path, chapter_id="ch-1", lane="translation", segment_ids=["s1"],
    )
    pipeline_module.invalidate_suffix(
        ledger_path, from_segment_id="s1", generation=2,
    )
    pipeline_module.write_json(
        checkpoint / "translation-token-offset-attempts" / "s1.json", {
            "segment_id": "s1", "generation": 1, "input_sha256": "old",
            "status": "response_received", "block_ids": ["body"],
            "raw_response": {"repairs": []},
        },
    )

    assert pipeline_module._finalize_paid_translation_repairs(checkpoint) == []
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["needs_supervision"] is None


def test_paid_repair_finalizer_runs_full_protected_name_validation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    block = {
        "block_id": "body", "type": "text", "text": "Hertz value x.",
        "inline_runs": [
            _inline_run("text", "Hertz value ", 1),
            _inline_run("math", "x", 2, tex="x"),
            _inline_run("text", ".", 3),
        ],
    }
    pipeline_module.write_json(
        checkpoint / "document.json", {"paper_id": "paper", "document": {"blocks": [block]}},
    )
    pipeline_module.write_json(checkpoint / "glossary.json", {"entries": [{
        "source_term": "Hertz equations", "protected_names": ["Hertz"],
    }]})
    ledger_path = checkpoint / "chapters" / "ch-1" / "translation-ledger.json"
    pipeline_module.initialize_lane_ledger(
        ledger_path, chapter_id="ch-1", lane="translation", segment_ids=["s1"],
    )
    pipeline_module.mark_submitted(ledger_path, segment_id="s1")
    pipeline_module.mark_response_received(ledger_path, segment_id="s1")
    key = "repair-name"
    pipeline_module.write_json(
        checkpoint / "translation-drafts" / f"{key}.json", {
            "segment_id": "s1", "input_sha256": "input",
            "translation": {"blocks": [{"block_id": "body", "text": "译文。"}]},
        },
    )
    token_id = str(block["inline_runs"][1]["token_id"])
    pipeline_module.write_json(
        checkpoint / "translation-token-offset-attempts" / f"{key}.json", {
            "segment_id": "s1", "input_sha256": "input",
            "status": "response_received", "block_ids": ["body"],
            "raw_response": {"repairs": [{"block_id": "body", "slots": [{
                "slot_id": token_id, "start_offset": 2, "end_offset": 2,
            }]}]},
        },
    )
    pipeline_module.write_json(
        checkpoint / "llm" / "translations" / key / "retry-offset-1"
        / "call-checkpoints" / "call.json", {
            "state": "validated", "submission_state": "submitted",
            "logical_identity": {"idempotency_key": "repair-idem"},
        },
    )

    entries = pipeline_module._finalize_paid_translation_repairs(checkpoint)

    assert len(entries) == 1
    assert entries[0]["recovery_action"] == "operator-supervision"
    assert "Hertz" in entries[0]["blocking_reason"]
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["needs_supervision"]["segment_id"] == "s1"
    assert ledger["needs_supervision"]["recovery_context"]["resumable"] is False


def test_dynamic_paid_repair_discovery_enriches_existing_transaction_entry(
    tmp_path: Path,
) -> None:
    from arc_companion.resume_transaction import (
        append_entries, begin_transaction, load_transaction, mark_entry,
    )

    identity = {
        "ledger_path": str(tmp_path / "ledger.json"),
        "session_key": "ch-1:translation", "segment_id": "s2",
        "idempotency_key": "primary-call", "initial_generation": 1,
        "target_generation": 1,
        "recovery_action": "deterministic-replay",
    }
    begin_transaction(
        tmp_path, action="resume-native", recovery_options={}, entries=[identity],
        native_resume_contexts=[{
            **identity, "native_session_id": "native-old", "resumable": True,
        }],
    )
    mark_entry(tmp_path, 0, status="reconciling")
    append_entries(tmp_path, [{
        **identity, "idempotency_key": "paid-repair-call",
        "recovery_action": "operator-supervision",
        "blocking_reason": "paid repair failed local validation",
        "recovery_context": {"repair_marker": "/checkpoint/marker.json"},
    }])

    transaction = load_transaction(tmp_path)
    assert transaction is not None
    assert len(transaction["entries"]) == 1
    entry = transaction["entries"][0]
    assert entry["status"] == "reconciling"
    assert entry["idempotency_key"] == "paid-repair-call"
    assert entry["recovery_action"] == "operator-supervision"
    assert entry["recovery_context"].get("resumable", False) is False
    assert entry["recovery_context"]["repair_marker"].endswith("marker.json")
    assert transaction["native_resume_contexts"] == []


def test_response_received_token_insertion_offsets_recover_without_resubmission(
    tmp_path: Path,
) -> None:
    """Replay the paid response shape found after the production crash."""
    block = {
        "block_id": "body", "type": "text", "text": "Value x.",
        "inline_runs": [
            _inline_run("text", "Value ", 1),
            _inline_run("math", "x", 2, tex="x"),
            _inline_run("text", ".", 3),
        ],
    }
    segment = {"segment_id": "seg-crash", "block_ids": ["body"]}
    checkpoint_dir = tmp_path / "checkpoints"
    attempt_path = pipeline_module._translation_token_attempt_path(
        checkpoint_dir, segment["segment_id"],
    )
    attempt_path.parent.mkdir(parents=True)
    token_id = str(block["inline_runs"][1]["token_id"])
    paid_response = {"repairs": [{"block_id": "body", "slots": [{
        "slot_id": token_id, "start_offset": 2, "end_offset": 2,
    }]}]}
    attempt_path.write_text(json.dumps({
        "schema_version": "arc.companion.translation-token-attempt.v2",
        "segment_id": segment["segment_id"], "input_sha256": "crash-input",
        "status": "response_received", "started_at": "2026-01-01T00:00:00+00:00",
        "response_received_at": "2026-01-01T00:01:00+00:00",
        "prompt_version": pipeline_module.TRANSLATION_RETRY_PROMPT_VERSION,
        "response_schema_version": pipeline_module.TRANSLATION_SLOT_REPAIR_SCHEMA_VERSION,
        "model_tier": pipeline_module.TRANSLATION_RETRY_TIER,
        "block_ids": ["body"], "raw_response": paid_response,
    }), encoding="utf-8")

    def forbidden_llm(prompt: str, **kwargs):
        raise AssertionError("a paid response_received repair must not be resubmitted")

    repaired, _ = pipeline_module._repair_translation_token_placement(
        segment,
        {"blocks": [{"block_id": "body", "text": "译文。"}]},
        blocks_by_id={"body": block}, protected_names=[],
        options=BuildOptions(paper_id="arXiv:0911.3380", project_dir=tmp_path),
        checkpoint_dir=checkpoint_dir, artifact_dir=checkpoint_dir / "llm",
        input_sha256="crash-input", llm=forbidden_llm,
    )

    token = pipeline_module._opaque_inline_tokens(block)[0]
    assert repaired["blocks"][0]["text"] == f"译文{token}。"
    marker = json.loads(attempt_path.read_text(encoding="utf-8"))
    assert marker["status"] == "validated"
    draft = pipeline_module._matching_translation_token_repair_draft(
        checkpoint_dir, segment["segment_id"], "crash-input",
    )
    assert draft is not None
    assert draft["raw_response"] == paid_response


def test_started_token_repair_without_call_checkpoint_is_safe_to_retry(
    tmp_path: Path,
) -> None:
    block = {
        "block_id": "body", "type": "text", "text": "Value x.",
        "inline_runs": [
            _inline_run("text", "Value ", 1),
            _inline_run("math", "x", 2, tex="x"),
            _inline_run("text", ".", 3),
        ],
    }
    segment = {"segment_id": "seg-resume", "block_ids": ["body"]}
    checkpoint_dir = tmp_path / "checkpoints"
    attempt_path = pipeline_module._translation_token_attempt_path(
        checkpoint_dir, segment["segment_id"],
    )
    attempt_path.parent.mkdir(parents=True)
    attempt_path.write_text(json.dumps({
        "schema_version": "arc.companion.translation-token-attempt.v2",
        "segment_id": segment["segment_id"], "input_sha256": "resume-input",
        "status": "started", "started_at": "2026-01-01T00:00:00+00:00",
        "prompt_version": pipeline_module.TRANSLATION_RETRY_PROMPT_VERSION,
        "response_schema_version": pipeline_module.TRANSLATION_SLOT_REPAIR_SCHEMA_VERSION,
        "model_tier": pipeline_module.TRANSLATION_RETRY_TIER,
        "block_ids": ["body"],
    }), encoding="utf-8")
    calls: list[str] = []

    def repair_llm(prompt: str, **kwargs):
        calls.append(str(kwargs["call_label"]))
        return {"repairs": [{"block_id": "body", "slots": [
            *_offset_slots(block, "译文。", [2]),
        ]}]}

    arguments = dict(
        segment=segment,
        translation={"blocks": [{"block_id": "body", "text": "译文。"}]},
        blocks_by_id={"body": block}, protected_names=[],
        options=BuildOptions(paper_id="arXiv:0911.3380", project_dir=tmp_path),
        checkpoint_dir=checkpoint_dir, artifact_dir=checkpoint_dir / "llm",
        input_sha256="resume-input",
    )
    repaired, _ = pipeline_module._repair_translation_token_placement(
        **arguments, llm=repair_llm,
    )
    assert calls == ["companion-translation-seg-resume-retry-offset-1"]
    assert repaired["blocks"][0]["text"] != "译文。"
    assert json.loads(attempt_path.read_text(encoding="utf-8"))["status"] == "validated"


@pytest.mark.parametrize("submission_state", ["submitted", "unknown"])
def test_started_token_repair_after_submission_barrier_fails_closed(
    tmp_path: Path, submission_state: str,
) -> None:
    block = {
        "block_id": "body", "type": "text", "text": "Value x.",
        "inline_runs": [
            _inline_run("text", "Value ", 1),
            _inline_run("math", "x", 2, tex="x"),
            _inline_run("text", ".", 3),
        ],
    }
    segment = {"segment_id": "seg-submitted", "block_ids": ["body"]}
    checkpoint_dir = tmp_path / "checkpoints"
    attempt_path = pipeline_module._translation_token_attempt_path(
        checkpoint_dir, segment["segment_id"],
    )
    attempt_path.parent.mkdir(parents=True)
    attempt_path.write_text(json.dumps({
        "schema_version": "arc.companion.translation-token-attempt.v2",
        "segment_id": segment["segment_id"], "input_sha256": "submitted-input",
        "status": "started", "started_at": "2026-01-01T00:00:00+00:00",
        "prompt_version": pipeline_module.TRANSLATION_RETRY_PROMPT_VERSION,
        "response_schema_version": pipeline_module.TRANSLATION_SLOT_REPAIR_SCHEMA_VERSION,
        "model_tier": pipeline_module.TRANSLATION_RETRY_TIER,
        "block_ids": ["body"],
    }), encoding="utf-8")
    call_dir = checkpoint_dir / "llm" / "retry-offset-1" / "call-checkpoints"
    call_dir.mkdir(parents=True)
    (call_dir / "call.json").write_text(json.dumps({
        "state": "submitted", "submission_state": submission_state,
    }), encoding="utf-8")

    def forbidden_llm(prompt: str, **kwargs):
        raise AssertionError("submitted or unknown repair must never be replayed")

    with pytest.raises(RuntimeError, match="already started"):
        pipeline_module._repair_translation_token_placement(
            segment,
            {"blocks": [{"block_id": "body", "text": "译文。"}]},
            blocks_by_id={"body": block}, protected_names=[],
            options=BuildOptions(paper_id="arXiv:0911.3380", project_dir=tmp_path),
            checkpoint_dir=checkpoint_dir, artifact_dir=checkpoint_dir / "llm",
            input_sha256="submitted-input", llm=forbidden_llm,
        )


def test_v4_upgrade_without_primary_draft_blocks_before_low_model(tmp_path: Path) -> None:
    block = {
        "block_id": "body", "type": "text", "text": "Value x.",
        "inline_runs": [
            _inline_run("text", "Value ", 1),
            _inline_run("math", "x", 2, tex="x"),
            _inline_run("text", ".", 3),
        ],
    }
    document = {
        "front_matter": {}, "blocks": [block], "equations": [], "figures": [],
        "tables": [], "bibliography": [], "assets": [],
        "integrity": {"status": "complete"},
    }
    bundle = SourceBundle(
        paper_id="arXiv:0911.3380", parsed={"document": document}, document=document,
        metadata={}, references=[], citers=[],
    )
    segment = {"segment_id": "seg-upgrade", "block_ids": ["body"]}
    options = BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path, workers=1)
    context = pipeline_module._full_paper_context(
        document, segment, blocks_by_id={"body": block},
    )
    input_sha256 = pipeline_module._segment_input_hash(
        segment, {"body": block}, glossary={"entries": []},
        extra={"names": [], "paper_context": context,
               "runtime_access": pipeline_module._generation_runtime_policy()},
    )
    checkpoint_dir = tmp_path / "checkpoints"
    final_path = (
        checkpoint_dir / "translations"
        / f"{pipeline_module._segment_checkpoint_name(segment['segment_id'])}.json"
    )
    final_path.parent.mkdir(parents=True)
    token = pipeline_module._opaque_inline_tokens(block)[0]
    final_path.write_text(json.dumps({
        "schema_version": "arc.companion.translation-checkpoint.v2",
        "segment_id": segment["segment_id"], "input_sha256": input_sha256,
        "generation_provenance": {"repairs": [{
            "kind": "token-placement",
            "prompt_version": "arc.companion.translation-retry-prompt.v4",
        }]},
        "translation": {"blocks": [{"block_id": "body", "text": f"译文{token}。"}]},
    }), encoding="utf-8")
    calls: list[str] = []

    def forbidden_llm(prompt: str, **kwargs):
        calls.append(str(kwargs.get("model_tier")))
        raise AssertionError("v4 upgrade must not fall back to low")

    with pytest.raises(CompanionLaneError) as exc_info:
        _generate_translations(
            [segment], options=options, bundle=bundle, glossary={"entries": []},
            protected_names=[], checkpoint_dir=checkpoint_dir, llm=forbidden_llm,
        )
    assert calls == []
    assert "requires its stored primary draft" in str(exc_info.value.failures[0][1])


def test_v3_attempt_v2_marker_migrates_to_v5_offsets_without_low(tmp_path: Path) -> None:
    block = {
        "block_id": "body", "type": "text", "text": "Value x.",
        "inline_runs": [
            _inline_run("text", "Value ", 1),
            _inline_run("math", "x", 2, tex="x"),
            _inline_run("text", ".", 3),
        ],
    }
    document = {
        "front_matter": {}, "blocks": [block], "equations": [], "figures": [],
        "tables": [], "bibliography": [], "assets": [],
        "integrity": {"status": "complete"},
    }
    bundle = SourceBundle(
        paper_id="arXiv:0911.3380", parsed={"document": document}, document=document,
        metadata={}, references=[], citers=[],
    )
    segment = {"segment_id": "seg-v3-marker", "block_ids": ["body"]}
    options = BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path, workers=1)
    context = pipeline_module._full_paper_context(
        document, segment, blocks_by_id={"body": block},
    )
    input_sha256 = pipeline_module._segment_input_hash(
        segment, {"body": block}, glossary={"entries": []},
        extra={"names": [], "paper_context": context,
               "runtime_access": pipeline_module._generation_runtime_policy()},
    )
    checkpoint_dir = tmp_path / "checkpoints"
    final_path = (
        checkpoint_dir / "translations"
        / f"{pipeline_module._segment_checkpoint_name(segment['segment_id'])}.json"
    )
    final_path.parent.mkdir(parents=True)
    token = pipeline_module._opaque_inline_tokens(block)[0]
    final_path.write_text(json.dumps({
        "schema_version": "arc.companion.translation-checkpoint.v2",
        "segment_id": segment["segment_id"], "input_sha256": input_sha256,
        "generation_provenance": {"repairs": [{
            "kind": "token-placement",
            "prompt_version": "arc.companion.translation-retry-prompt.v3",
        }]},
        "translation": {"blocks": [{
            "block_id": "body", "text": f"错误地把{token}作为对象。",
        }]},
    }), encoding="utf-8")
    primary_path = pipeline_module._translation_draft_path(
        checkpoint_dir, segment["segment_id"],
    )
    primary_path.parent.mkdir(parents=True)
    primary_path.write_text(json.dumps(
        pipeline_module._translation_primary_draft_payload(
            segment,
            {"blocks": [{"block_id": "body", "text": "缺少令牌。"}]},
            input_sha256=input_sha256, origin="primary-model",
        )
    ), encoding="utf-8")
    old_marker = pipeline_module._legacy_translation_token_attempt_path(
        checkpoint_dir, segment["segment_id"],
    )
    old_marker.parent.mkdir(parents=True)
    old_marker.write_text(json.dumps({
        "schema_version": "arc.companion.translation-token-attempt.v2",
        "prompt_version": "arc.companion.translation-retry-prompt.v3",
        "segment_id": segment["segment_id"], "input_sha256": input_sha256,
        "status": "validated",
    }), encoding="utf-8")
    old_marker_bytes = old_marker.read_bytes()
    calls: list[tuple[str, str]] = []

    def repair_llm(prompt: str, **kwargs):
        calls.append((str(kwargs["call_label"]), str(kwargs["model_tier"])))
        return {"repairs": [{"block_id": "body", "slots": [
            *_offset_slots(block, "缺少令牌。", [4]),
        ]}]}

    result = _generate_translations(
        [segment], options=options, bundle=bundle, glossary={"entries": []},
        protected_names=[], checkpoint_dir=checkpoint_dir, llm=repair_llm,
    )
    assert calls == [("companion-translation-seg-v3-marker-retry-offset-1", "medium")]
    assert result[segment["segment_id"]]["blocks"][0]["text"] == (
        f"缺少令牌{token}。"
    )
    assert old_marker.read_bytes() == old_marker_bytes
    migrated_marker = json.loads(pipeline_module._translation_token_attempt_path(
        checkpoint_dir, segment["segment_id"],
    ).read_text(encoding="utf-8"))
    assert migrated_marker["prompt_version"] == pipeline_module.TRANSLATION_RETRY_PROMPT_VERSION
    assert migrated_marker["status"] == "validated"
    assert migrated_marker["superseded_text_attempt"]["path"] == str(old_marker)


@pytest.mark.parametrize(
    "marker_kind",
    [
        "unreadable", "malformed-current", "current-missing-schema",
        "unknown-prompt", "legacy-v4",
    ],
)
def test_token_attempt_marker_fails_closed_before_low_model(
    tmp_path: Path, marker_kind: str,
) -> None:
    block = {
        "block_id": "body", "type": "text", "text": "Value x.",
        "inline_runs": [
            _inline_run("text", "Value ", 1),
            _inline_run("math", "x", 2, tex="x"),
            _inline_run("text", ".", 3),
        ],
    }
    document = {
        "front_matter": {}, "blocks": [block], "equations": [], "figures": [],
        "tables": [], "bibliography": [], "assets": [],
        "integrity": {"status": "complete"},
    }
    bundle = SourceBundle(
        paper_id="arXiv:0911.3380", parsed={"document": document}, document=document,
        metadata={}, references=[], citers=[],
    )
    segment = {"segment_id": "seg-marker", "block_ids": ["body"]}
    context = pipeline_module._full_paper_context(
        document, segment, blocks_by_id={"body": block},
    )
    input_sha256 = pipeline_module._segment_input_hash(
        segment, {"body": block}, glossary={"entries": []},
        extra={"names": [], "paper_context": context,
               "runtime_access": pipeline_module._generation_runtime_policy()},
    )
    checkpoint_dir = tmp_path / "checkpoints"
    marker_path = pipeline_module._translation_token_attempt_path(
        checkpoint_dir, segment["segment_id"],
    )
    if marker_kind == "legacy-v4":
        marker_path = pipeline_module._legacy_translation_token_attempt_path(
            checkpoint_dir, segment["segment_id"],
        )
    marker_path.parent.mkdir(parents=True)
    if marker_kind == "unreadable":
        marker_path.write_text("{broken", encoding="utf-8")
    elif marker_kind == "malformed-current":
        marker_path.write_text(json.dumps({
            "schema_version": "arc.companion.translation-token-attempt.v2",
            "prompt_version": pipeline_module.TRANSLATION_RETRY_PROMPT_VERSION,
            "segment_id": segment["segment_id"],
            "input_sha256": input_sha256 + "-wrong",
            "status": "started",
        }), encoding="utf-8")
    elif marker_kind == "current-missing-schema":
        marker_path.write_text(json.dumps({
            "prompt_version": pipeline_module.TRANSLATION_RETRY_PROMPT_VERSION,
            "segment_id": segment["segment_id"], "input_sha256": input_sha256,
            "status": "started",
        }), encoding="utf-8")
    elif marker_kind == "unknown-prompt":
        marker_path.write_text(json.dumps({
            "schema_version": "arc.companion.translation-token-attempt.v2",
            "prompt_version": "arc.companion.translation-retry-prompt.v99",
            "segment_id": segment["segment_id"], "input_sha256": input_sha256,
            "status": "started",
        }), encoding="utf-8")
    else:
        marker_path.write_text(json.dumps({
            "schema_version": "arc.companion.translation-token-attempt.v2",
            "prompt_version": "arc.companion.translation-retry-prompt.v4",
            "segment_id": segment["segment_id"], "input_sha256": input_sha256,
            "status": "response_received", "raw_response": {"repairs": []},
        }), encoding="utf-8")
    calls: list[str] = []

    def forbidden_llm(prompt: str, **kwargs):
        calls.append(str(kwargs.get("model_tier")))
        raise AssertionError("invalid token marker must fail before any model call")

    with pytest.raises(CompanionLaneError) as exc_info:
        _generate_translations(
            [segment],
            options=BuildOptions(
                paper_id=bundle.paper_id, project_dir=tmp_path, workers=1,
            ),
            bundle=bundle, glossary={"entries": []}, protected_names=[],
            checkpoint_dir=checkpoint_dir, llm=forbidden_llm,
        )
    assert calls == []
    message = str(exc_info.value.failures[0][1])
    if marker_kind == "legacy-v4":
        assert "attempt already consumed" in message
    else:
        assert "marker" in message
        assert "refusing a primary model call" in message


def test_damaged_current_repair_draft_without_validated_raw_fails_closed(
    tmp_path: Path,
) -> None:
    block = {
        "block_id": "body", "type": "text", "text": "Value x.",
        "inline_runs": [
            _inline_run("text", "Value ", 1),
            _inline_run("math", "x", 2, tex="x"),
            _inline_run("text", ".", 3),
        ],
    }
    segment = {"segment_id": "seg-damaged", "block_ids": ["body"]}
    checkpoint_dir = tmp_path / "checkpoints"
    repair_draft = pipeline_module._translation_token_repair_draft_path(
        checkpoint_dir, segment["segment_id"],
    )
    repair_draft.parent.mkdir(parents=True)
    repair_draft.write_text(json.dumps({
        "schema_version": "arc.companion.translation-token-repair-draft.v1",
        "segment_id": segment["segment_id"], "input_sha256": "damaged-input",
        "prompt_version": pipeline_module.TRANSLATION_RETRY_PROMPT_VERSION,
        "response_schema_version": pipeline_module.TRANSLATION_SLOT_REPAIR_SCHEMA_VERSION,
        "translation": {"blocks": []},
        "repair_provenance": {
            "kind": "token-placement", "repair_mode": "offset-only",
            "repaired_block_ids": ["body"],
        },
        "raw_response": {"repairs": []},
    }), encoding="utf-8")
    calls = 0

    def forbidden_llm(prompt: str, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("damaged current draft must not trigger a model")

    with pytest.raises(RuntimeError, match="no validated raw response"):
        pipeline_module._repair_translation_token_placement(
            segment,
            {"blocks": [{"block_id": "body", "text": "译文。"}]},
            blocks_by_id={"body": block}, protected_names=[],
            options=BuildOptions(paper_id="arXiv:0911.3380", project_dir=tmp_path),
            checkpoint_dir=checkpoint_dir, artifact_dir=checkpoint_dir / "llm",
            input_sha256="damaged-input", llm=forbidden_llm,
        )
    assert calls == 0


def test_citation_followed_by_identical_opaque_occurrence_uses_ordinals() -> None:
    duplicate_citation = _inline_run("citation", "7", 2)
    block = {
        "block_id": "cited", "type": "text", "text": "[7]7",
        "inline_runs": [
            _inline_run("text", "[", 1), duplicate_citation,
            _inline_run("text", "]", 3), duplicate_citation,
        ],
    }
    first, second = pipeline_module._opaque_inline_tokens(block)
    assert first == second
    normalized = pipeline_module._normalize_translation_citation_delimiters(
        block, f"{first}{second}",
    )
    assert normalized == f"[{first}]{second}"


def test_checkpoint_citation_delimiter_repair_revalidates_and_records_provenance(
    tmp_path: Path,
) -> None:
    block = _bracketed_citation_block()
    token = pipeline_module._opaque_inline_tokens(block)[0]
    segment = {"segment_id": "seg-0063", "block_ids": ["cited"]}
    checkpoint_path = tmp_path / "translation.json"
    checkpoint_path.write_text(json.dumps({
        "schema_version": "arc.companion.translation-checkpoint.v2",
        "segment_id": "seg-0063",
        "input_sha256": "fixture",
        "generation_provenance": {"candidate": {"origin": "primary-model"}, "repairs": []},
        "translation": {"blocks": [{"block_id": "cited", "text": f"参见{token}[]。"}]},
    }), encoding="utf-8")

    repaired = pipeline_module._repair_translation_checkpoint_citation_delimiters(
        checkpoint_path, segment, {"cited": block}, protected_names=[],
    )

    assert repaired["translation"]["blocks"][0]["text"] == f"参见[{token}]。"
    assert repaired["generation_provenance"]["candidate"] == {"origin": "primary-model"}
    assert repaired["generation_provenance"]["repairs"][-1] == {
        "kind": "citation-delimiter-normalization",
        "attempt": 0,
        "normalizer_version": pipeline_module.TRANSLATION_CITATION_DELIMITER_NORMALIZER_VERSION,
        "repaired_block_ids": ["cited"],
        "methods_by_block_id": {"cited": "relocated"},
    }
    assert json.loads(checkpoint_path.read_text(encoding="utf-8")) == repaired


def _citation_translation_bundle(blocks: list[dict]) -> SourceBundle:
    document = {
        "front_matter": {}, "blocks": blocks, "equations": [], "figures": [],
        "tables": [], "bibliography": [], "assets": [],
        "integrity": {"status": "complete"},
    }
    return SourceBundle(
        paper_id="arXiv:1234.5678", parsed={"document": document}, document=document,
        metadata={}, references=[], citers=[],
    )


def test_primary_translation_candidate_normalizes_citation_delimiters(
    tmp_path: Path,
) -> None:
    block = _bracketed_citation_block()
    token = pipeline_module._opaque_inline_tokens(block)[0]
    bundle = _citation_translation_bundle([block])
    calls: list[str] = []

    def llm(prompt: str, **kwargs):
        calls.append(str(kwargs["call_label"]))
        return {"blocks": [{"block_id": "cited", "text": f"参见{token}[]。"}]}

    checkpoint_dir = tmp_path / "checkpoints"
    result = _generate_translations(
        [{"segment_id": "seg-primary", "block_ids": ["cited"]}],
        options=BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path, workers=1),
        bundle=bundle, glossary={"entries": []}, protected_names=[],
        checkpoint_dir=checkpoint_dir, llm=llm,
    )

    assert calls == ["companion-translation-seg-primary"]
    assert result["seg-primary"]["blocks"][0]["text"] == f"参见[{token}]。"
    checkpoint = json.loads(next((checkpoint_dir / "translations").glob("*.json")).read_text())
    assert [item["kind"] for item in checkpoint["generation_provenance"]["repairs"]] == [
        "citation-delimiter-normalization"
    ]
    draft = json.loads(next((checkpoint_dir / "translation-drafts").glob("*.json")).read_text())
    assert draft["translation"]["blocks"][0]["text"] == f"参见{token}[]。"


def test_cached_translation_candidate_normalizes_and_rewrites_checkpoint(
    tmp_path: Path,
) -> None:
    block = _bracketed_citation_block()
    token = pipeline_module._opaque_inline_tokens(block)[0]
    bundle = _citation_translation_bundle([block])
    segment = {"segment_id": "seg-cached", "block_ids": ["cited"]}
    checkpoint_dir = tmp_path / "checkpoints"

    _generate_translations(
        [segment],
        options=BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path, workers=1),
        bundle=bundle, glossary={"entries": []}, protected_names=[],
        checkpoint_dir=checkpoint_dir,
        llm=lambda prompt, **kwargs: {
            "blocks": [{"block_id": "cited", "text": f"参见[{token}]。"}]
        },
    )
    checkpoint_path = next((checkpoint_dir / "translations").glob("*.json"))
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["translation"]["blocks"][0]["text"] = f"参见{token}[]。"
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    def forbidden_llm(prompt: str, **kwargs):
        raise AssertionError("a valid cached checkpoint must not call the model")

    result = _generate_translations(
        [segment],
        options=BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path, workers=1),
        bundle=bundle, glossary={"entries": []}, protected_names=[],
        checkpoint_dir=checkpoint_dir, llm=forbidden_llm,
    )

    assert result["seg-cached"]["blocks"][0]["text"] == f"参见[{token}]。"
    rewritten = json.loads(checkpoint_path.read_text())
    assert rewritten["translation"] == result["seg-cached"]
    assert rewritten["generation_provenance"]["repairs"][-1]["kind"] == (
        "citation-delimiter-normalization"
    )
    _generate_translations(
        [segment],
        options=BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path, workers=1),
        bundle=bundle, glossary={"entries": []}, protected_names=[],
        checkpoint_dir=checkpoint_dir, llm=forbidden_llm,
    )
    idempotent = json.loads(checkpoint_path.read_text())
    assert len(idempotent["generation_provenance"]["repairs"]) == 1


def test_coverage_repair_candidate_normalizes_citation_delimiters(
    tmp_path: Path,
) -> None:
    kept = {"block_id": "kept", "type": "text", "text": "Keep this."}
    cited = _bracketed_citation_block()
    token = pipeline_module._opaque_inline_tokens(cited)[0]
    bundle = _citation_translation_bundle([kept, cited])
    calls: list[str] = []

    def llm(prompt: str, **kwargs):
        label = str(kwargs["call_label"])
        calls.append(label)
        if label.endswith("coverage-repair-1"):
            slot_ids = pipeline_module._translation_coverage_slot_ids(cited)
            return {"repairs": [{"block_id": "cited", "slots": [
                {"slot_id": slot_ids[0], "text": "参见"},
                {"slot_id": slot_ids[1], "text": "[]。"},
            ]}]}
        return {"blocks": [{"block_id": "kept", "text": "保留。"}]}

    checkpoint_dir = tmp_path / "checkpoints"
    result = _generate_translations(
        [{"segment_id": "seg-coverage", "block_ids": ["kept", "cited"]}],
        options=BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path, workers=1),
        bundle=bundle, glossary={"entries": []}, protected_names=[],
        checkpoint_dir=checkpoint_dir, llm=llm,
    )

    assert calls == [
        "companion-translation-seg-coverage",
        "companion-translation-seg-coverage-coverage-repair-1",
    ]
    assert result["seg-coverage"]["blocks"][1]["text"] == f"参见[{token}]。"
    checkpoint = json.loads(next((checkpoint_dir / "translations").glob("*.json")).read_text())
    assert [item["kind"] for item in checkpoint["generation_provenance"]["repairs"]] == [
        "coverage", "citation-delimiter-normalization"
    ]


def test_final_review_translation_patch_normalizes_citation_delimiters(
    tmp_path: Path,
) -> None:
    block = _bracketed_citation_block()
    token = pipeline_module._opaque_inline_tokens(block)[0]
    segment = {"segment_id": "seg-review", "block_ids": ["cited"]}
    translations = {
        "seg-review": {"blocks": [{"block_id": "cited", "text": f"参见[{token}]。"}]}
    }
    annotations = {"seg-review": {
        "commentary": "伴读", "explanation": "解释", "prior_work": [],
        "later_work": [], "evidence_ids": [], "key_points": [], "source_notes": [],
    }}

    def llm(prompt: str, **kwargs):
        assert kwargs["call_label"] == "companion-final-review"
        return {"patches": [{
            "segment_id": "seg-review",
            "translation_blocks": [
                {"block_id": "cited", "text": f"审校后{token}[]。"}
            ],
            "commentary": None, "explanation": None, "prior_work": None,
            "later_work": None, "evidence_ids": None, "reason": "fix wording",
        }], "issues": []}

    reviewed, _, audit = _review(
        [segment], translations, annotations,
        document=_citation_translation_bundle([block]).document,
        glossary={"entries": []}, protected_names=[], evidence={"related_papers": []},
        options=BuildOptions(
            paper_id="arXiv:1234.5678", project_dir=tmp_path, workers=1,
            review_context_chars=100_000,
        ),
        llm=llm, checkpoint_dir=tmp_path / "checkpoints",
    )

    assert reviewed["seg-review"]["blocks"][0]["text"] == f"审校后[{token}]。"
    assert audit["citation_delimiter_normalized_segment_ids"] == ["seg-review"]


def test_final_review_may_remove_unhelpful_explanation(tmp_path: Path) -> None:
    block = _bracketed_citation_block()
    token = pipeline_module._opaque_inline_tokens(block)[0]
    segment = {"segment_id": "seg-review", "block_ids": ["cited"]}
    translations = {
        "seg-review": {"blocks": [{"block_id": "cited", "text": f"参见[{token}]。"}]}
    }
    annotations = {"seg-review": {
        "commentary": "重复原文。", "explanation": "重复原文。", "prior_work": "",
        "later_work": "", "evidence_ids": [], "key_points": [], "source_notes": [],
    }}

    def llm(prompt: str, **kwargs):
        assert kwargs["call_label"] == "companion-final-review"
        return {"patches": [{
            "segment_id": "seg-review", "translation_blocks": None,
            "commentary": "", "explanation": "", "prior_work": None,
            "later_work": None, "evidence_ids": None,
            "reason": "the passage is evident and the draft only repeats it",
        }], "issues": []}

    _, reviewed, _ = _review(
        [segment], translations, annotations,
        document=_citation_translation_bundle([block]).document,
        glossary={"entries": []}, protected_names=[], evidence={"related_papers": []},
        options=BuildOptions(
            paper_id="arXiv:1234.5678", project_dir=tmp_path, workers=1,
            review_context_chars=100_000,
        ),
        llm=llm, checkpoint_dir=tmp_path / "checkpoints",
    )

    assert reviewed["seg-review"]["explanation"] == ""
    assert reviewed["seg-review"]["commentary"] == ""


def test_cached_reviewed_translation_checkpoint_is_normalized_and_rewritten(
    tmp_path: Path,
) -> None:
    block = _bracketed_citation_block()
    token = pipeline_module._opaque_inline_tokens(block)[0]
    segment = {"segment_id": "seg-reviewed", "block_ids": ["cited"]}
    checkpoint_path = tmp_path / "annotations.reviewed.v2.json"
    checkpoint_path.write_text(json.dumps({
        "schema_version": pipeline_module.REVIEW_VERSION,
        "translations": {"seg-reviewed": {"blocks": [
            {"block_id": "cited", "text": f"缓存{token}[]。"}
        ]}},
        "annotations": {"seg-reviewed": {}},
    }), encoding="utf-8")

    repaired, changed = _repair_reviewed_translation_checkpoint(
        checkpoint_path, [segment], {"cited": block}, protected_names=[],
    )

    assert changed == ["seg-reviewed"]
    assert repaired["translations"]["seg-reviewed"]["blocks"][0]["text"] == (
        f"缓存[{token}]。"
    )
    assert json.loads(checkpoint_path.read_text()) == repaired
    idempotent, changed_again = _repair_reviewed_translation_checkpoint(
        checkpoint_path, [segment], {"cited": block}, protected_names=[],
    )
    assert changed_again == []
    assert idempotent == repaired


def test_duplicate_coverage_candidates_are_discarded_before_citation_normalization(
    tmp_path: Path,
) -> None:
    cited = _bracketed_citation_block()
    token = pipeline_module._opaque_inline_tokens(cited)[0]
    bundle = _citation_translation_bundle([cited])
    calls: list[str] = []

    def llm(prompt: str, **kwargs):
        label = str(kwargs["call_label"])
        calls.append(label)
        if label.endswith("coverage-repair-1"):
            slot_ids = pipeline_module._translation_coverage_slot_ids(cited)
            return {"repairs": [{"block_id": "cited", "slots": [
                {"slot_id": slot_ids[0], "text": "参见["},
                {"slot_id": slot_ids[1], "text": "]。"},
            ]}]}
        return {"blocks": [
            {"block_id": "cited", "text": f"将丢弃[额外]{token}。"},
            {"block_id": "cited", "text": f"也将丢弃{token}。"},
        ]}

    result = _generate_translations(
        [{"segment_id": "seg-duplicate", "block_ids": ["cited"]}],
        options=BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path, workers=1),
        bundle=bundle, glossary={"entries": []}, protected_names=[],
        checkpoint_dir=tmp_path / "checkpoints", llm=llm,
    )

    assert calls == [
        "companion-translation-seg-duplicate",
        "companion-translation-seg-duplicate-coverage-repair-1",
    ]
    assert result["seg-duplicate"]["blocks"][0]["text"] == f"参见[{token}]。"




def test_slot_repair_allows_only_exact_missing_name_insertion() -> None:
    block = {
        "block_id": "runs", "type": "text", "text": "Ada Lovelace uses x.",
        "inline_runs": [
            _inline_run("text", "Ada Lovelace uses ", 1),
            _inline_run("math", "x", 2, tex="x"),
            _inline_run("text", ".", 3),
        ],
    }
    previous = {"blocks": [{"block_id": "runs", "text": "艾达使用。"}]}
    slot_ids = pipeline_module._translation_repair_slot_ids(block)
    repaired = {
        "block_id": "runs",
        "slots": [
            {"slot_id": slot_ids[0], "text": "艾达Ada Lovelace使用"},
            {"slot_id": slot_ids[1], "text": "。"},
        ],
    }
    context = pipeline_module._translation_slot_repair_context(
        block, "艾达使用。", protected_names=["Ada", "Lovelace", "Ada Lovelace"],
    )
    assert context["missing_protected_names"] == ["Ada Lovelace"]
    result = pipeline_module._apply_translation_slot_repairs(
        previous, [block], {"repairs": [repaired]},
        protected_names=["Ada", "Lovelace", "Ada Lovelace"],
    )
    assert pipeline_module._OPAQUE_INLINE_PATTERN.sub("", result["blocks"][0]["text"]) == "艾达Ada Lovelace使用。"

    rephrased = {
        "block_id": "runs",
        "slots": [
            {"slot_id": slot_ids[0], "text": "Ada Lovelace重新翻译"},
            {"slot_id": slot_ids[1], "text": "。"},
        ],
    }
    try:
        pipeline_module._apply_translation_slot_repairs(
            previous, [block], {"repairs": [rephrased]},
            protected_names=["Ada", "Lovelace", "Ada Lovelace"],
        )
    except RuntimeError as exc:
        assert "beyond name insertion" in str(exc)
    else:
        raise AssertionError("protected-name repair must not permit retranslation")


@pytest.mark.parametrize(
    ("source_text", "translated_text", "protected_name"),
    [
        ("the Lorentz Lie algebra relations", "Lorentz 李代数关系", "Lie"),
        ("acts as a Lagrange multiplier", "充当拉格朗日乘子", "Lagrange"),
        ("Gordon equation", "该方程", "Gordon"),
    ],
)
def test_translation_restores_missing_eponym_as_minimal_latin_annotation(
    source_text: str, translated_text: str, protected_name: str,
) -> None:
    block = {
        "block_id": "tong-eponym",
        "type": "text",
        "text": source_text,
        "inline_runs": [_inline_run("text", source_text, 1)],
    }
    segment = {"segment_id": "seg-tong-eponym", "block_ids": ["tong-eponym"]}
    translation = {
        "blocks": [{"block_id": "tong-eponym", "text": translated_text + "。"}]
    }

    restored, changed = pipeline_module._restore_translation_protected_names(
        segment, translation, {"tong-eponym": block}, [protected_name],
    )

    assert changed == ["tong-eponym"]
    assert restored["blocks"][0]["text"] == (
        f"{translated_text}（{protected_name}）。"
    )
    _validate_translation(
        segment, restored, {"tong-eponym": block}, [protected_name],
    )


def test_single_word_protected_name_does_not_match_case_folded_ordinary_word() -> None:
    ordinary = {
        "block_id": "ordinary-lie",
        "type": "text",
        "text": "Choose vectors that lie in the transverse plane.",
        "inline_runs": [
            _inline_run("text", "Choose vectors that lie in the transverse plane.", 1),
        ],
    }
    segment = {"segment_id": "seg-ordinary-lie", "block_ids": ["ordinary-lie"]}
    translation = {
        "blocks": [{"block_id": "ordinary-lie", "text": "选择位于横向平面内的向量。"}]
    }

    assert pipeline_module._missing_protected_names([ordinary], "译文。", ["Lie"]) == []
    restored, changed = pipeline_module._restore_translation_protected_names(
        segment, translation, {"ordinary-lie": ordinary}, ["Lie"],
    )

    assert changed == []
    assert restored == translation


def test_single_word_protected_name_still_matches_canonical_source_case() -> None:
    eponym = {
        "block_id": "lie-algebra",
        "type": "text",
        "text": "The Lorentz Lie algebra closes.",
        "inline_runs": [_inline_run("text", "The Lorentz Lie algebra closes.", 1)],
    }

    assert pipeline_module._missing_protected_names(
        [eponym], "Lorentz 李代数封闭。", ["Lie"],
    ) == ["Lie"]


def test_old_token_attempt_version_does_not_consume_new_repair_lifetime(
    tmp_path: Path,
) -> None:
    segment_id = "seg-0043"
    input_hash = "tong-input-hash"
    path = pipeline_module._translation_token_attempt_path(tmp_path, segment_id)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "schema_version": "arc.companion.translation-token-attempt.v1",
        "segment_id": segment_id,
        "input_sha256": input_hash,
    }))

    assert pipeline_module._matching_translation_token_attempt(
        tmp_path, segment_id, input_hash,
    ) is None

    path.write_text(json.dumps({
        "schema_version": "arc.companion.translation-token-attempt.v2",
        "repair_version": pipeline_module.TRANSLATION_TOKEN_REPAIR_VERSION,
        "segment_id": segment_id,
        "input_sha256": input_hash,
    }))
    assert pipeline_module._matching_translation_token_attempt(
        tmp_path, segment_id, input_hash,
    ) is None

    path.write_text(json.dumps({
        "schema_version": "arc.companion.translation-token-attempt.v2",
        "prompt_version": pipeline_module.TRANSLATION_RETRY_PROMPT_VERSION,
        "segment_id": segment_id,
        "input_sha256": input_hash,
    }))
    assert pipeline_module._matching_translation_token_attempt(
        tmp_path, segment_id, input_hash,
    ) is not None


def test_slot_repair_rejects_bad_slot_coverage_opaque_content_and_rephrasing() -> None:
    block = {
        "block_id": "slots", "type": "text", "text": "Value x.",
        "inline_runs": [
            _inline_run("text", "Value ", 1),
            _inline_run("math", "x", 2, tex="x"),
            _inline_run("text", ".", 3),
        ],
    }
    token = pipeline_module._opaque_inline_tokens(block)[0]
    previous = {"blocks": [{"block_id": "slots", "text": "旧译文。"}]}
    slot_ids = pipeline_module._translation_repair_slot_ids(block)
    valid_slots = [
        {"slot_id": slot_ids[0], "text": "旧译文"},
        {"slot_id": slot_ids[1], "text": "。"},
    ]
    invalid_slots = {
        "missing": valid_slots[:-1],
        "duplicate": [valid_slots[0], valid_slots[0], valid_slots[1]],
        "out-of-order": list(reversed(valid_slots)),
        "opaque": [{"slot_id": slot_ids[0], "text": token}, valid_slots[1]],
        "rephrased": [
            {"slot_id": slot_ids[0], "text": "新译文"}, valid_slots[1],
        ],
    }
    for label, slots in invalid_slots.items():
        try:
            pipeline_module._apply_translation_slot_repairs(
                previous,
                [block],
                {"repairs": [{"block_id": "slots", "slots": slots}]},
                protected_names=[],
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"{label} slot repair must be rejected")


@pytest.mark.parametrize(
    "spans",
    [
        [(True, 2), (2, 4)],
        [(0, 2), (3, 4)],
        [(0, 3), (2, 4)],
        [(0, 2), (2, 3)],
        [(0, 2), (2, 5)],
    ],
)
def test_offset_repair_rejects_non_integer_and_non_partition_spans(
    spans: list[tuple[object, object]],
) -> None:
    block = {
        "block_id": "offsets", "type": "text", "text": "Value x.",
        "inline_runs": [
            _inline_run("text", "Value ", 1),
            _inline_run("math", "x", 2, tex="x"),
            _inline_run("text", ".", 3),
        ],
    }
    slot_ids = pipeline_module._translation_repair_slot_ids(block)
    slots = [
        {"slot_id": slot_id, "start_offset": start, "end_offset": end}
        for slot_id, (start, end) in zip(slot_ids, spans)
    ]
    with pytest.raises(RuntimeError, match="offset"):
        pipeline_module._apply_translation_slot_repairs(
            {"blocks": [{"block_id": "offsets", "text": "旧译文。"}]},
            [block], {"repairs": [{"block_id": "offsets", "slots": slots}]},
            protected_names=[], offset_only=True,
        )


def test_slot_repair_strips_a_bounded_mutated_marker_candidate() -> None:
    block = {
        "block_id": "mutated", "type": "text", "text": "A x B.",
        "inline_runs": [
            _inline_run("text", "A ", 1),
            _inline_run("math", "x", 2, tex="x"),
            _inline_run("text", " B.", 3),
        ],
    }
    previous = {
        "blocks": [{"block_id": "mutated", "text": "甲[[ARC_INLINE:broken token value]]乙"}]
    }
    context = pipeline_module._translation_slot_repair_context(
        block, previous["blocks"][0]["text"], protected_names=[],
    )
    assert context["prior_natural_language_residue"] == "甲乙"
    slots = [
        {"slot_id": context["slot_ids"][0], "text": "甲"},
        {"slot_id": context["slot_ids"][1], "text": "乙"},
    ]
    result = pipeline_module._apply_translation_slot_repairs(
        previous, [block], {"repairs": [{"block_id": "mutated", "slots": slots}]},
        protected_names=[],
    )
    assert pipeline_module._OPAQUE_INLINE_PATTERN.findall(result["blocks"][0]["text"]) == (
        pipeline_module._opaque_inline_tokens(block)
    )
    assert "broken token value" not in result["blocks"][0]["text"]


@pytest.mark.parametrize(
    "wrapper", ["single", "zero-width", "truncated-hash", "middle-truncated-hash"]
)
def test_controller_canonicalizes_identity_preserving_marker_mutations(
    wrapper: str,
) -> None:
    block = {
        "block_id": "canonical", "type": "text", "text": "A x B.",
        "inline_runs": [
            _inline_run("text", "A ", 1),
            _inline_run("math", "x", 2, tex="x"),
            _inline_run("text", " B.", 3),
        ],
    }
    expected = pipeline_module._opaque_inline_tokens(block)[0]
    if wrapper == "single":
        candidate = expected[1:-1]
    elif wrapper == "zero-width":
        candidate = "[\u200b" + expected[1:]
    elif wrapper == "truncated-hash":
        candidate = expected[:-6] + "]]"
    else:
        prefix, digest = expected[:-2].rsplit(":", 1)
        candidate = f"{prefix}:{digest[:12]}{digest[-24:]}]]"
    translation = {
        "blocks": [{"block_id": "canonical", "text": f"甲{candidate}乙。"}]
    }
    repaired, changed = pipeline_module._canonicalize_translation_opaque_candidates(
        translation, {"canonical": block},
    )
    assert changed == ["canonical"]
    assert repaired["blocks"][0]["text"] == f"甲{expected}乙。"
    assert pipeline_module._translation_natural_residue(
        repaired["blocks"][0]["text"]
    ) == "甲乙。"


@pytest.mark.parametrize("mutation", ["token-id", "short-hash", "non-hex"])
def test_controller_rejects_ambiguous_marker_candidates(mutation: str) -> None:
    block = {
        "block_id": "canonical", "type": "text", "text": "A x B.",
        "inline_runs": [
            _inline_run("text", "A ", 1),
            _inline_run("math", "x", 2, tex="x"),
            _inline_run("text", " B.", 3),
        ],
    }
    expected = pipeline_module._opaque_inline_tokens(block)[0]
    if mutation == "token-id":
        candidate = expected.replace("b.token-0002", "b.token-9999")
    elif mutation == "short-hash":
        candidate = expected.rsplit(":", 1)[0] + ":deadbeef]]"
    else:
        candidate = expected.rsplit(":", 1)[0] + ":not-a-digest]]"
    translation = {
        "blocks": [{"block_id": "canonical", "text": f"甲{candidate}乙。"}]
    }
    repaired, changed = pipeline_module._canonicalize_translation_opaque_candidates(
        translation, {"canonical": block},
    )
    assert changed == []
    assert repaired == translation


def test_protected_name_validation_ignores_opaque_runs_but_checks_text_runs() -> None:
    opaque_name = _inline_run("link", "Maldacena", 2)
    opaque_name["href"] = "https://example.test/maldacena"
    opaque_block = {
        "block_id": "opaque-name", "type": "text", "text": "See Maldacena.",
        "inline_runs": [
            _inline_run("text", "See ", 1), opaque_name, _inline_run("text", ".", 3),
        ],
    }
    token = pipeline_module._opaque_inline_tokens(opaque_block)[0]
    _validate_translation(
        {"segment_id": "opaque-name", "block_ids": ["opaque-name"]},
        {"blocks": [{"block_id": "opaque-name", "text": f"参见{token}。"}]},
        {"opaque-name": opaque_block},
        ["Maldacena"],
    )
    opaque_context = pipeline_module._translation_slot_repair_context(
        opaque_block, "参见。", protected_names=["Maldacena"],
    )
    assert opaque_context["missing_protected_names"] == []

    text_block = {
        "block_id": "text-name", "type": "text", "text": "Maldacena uses x.",
        "inline_runs": [
            _inline_run("text", "Maldacena uses ", 1),
            _inline_run("math", "x", 2, tex="x"),
            _inline_run("text", ".", 3),
        ],
    }
    text_token = pipeline_module._opaque_inline_tokens(text_block)[0]
    try:
        _validate_translation(
            {"segment_id": "text-name", "block_ids": ["text-name"]},
            {"blocks": [{"block_id": "text-name", "text": f"使用{text_token}。"}]},
            {"text-name": text_block},
            ["Maldacena"],
        )
    except RuntimeError as exc:
        assert "protected names" in str(exc)
    else:
        raise AssertionError("a protected name in natural text must remain present")

    wrong_case_context = pipeline_module._translation_slot_repair_context(
        text_block, f"maldacena 使用{text_token}。", protected_names=["Maldacena"],
    )
    assert wrong_case_context["missing_protected_names"] == ["Maldacena"]

    cross_run_block = {
        "block_id": "cross-run", "type": "text", "text": "Malda x cena.",
        "inline_runs": [
            _inline_run("text", "Malda", 1),
            _inline_run("math", "x", 2, tex="x"),
            _inline_run("text", "cena", 3),
        ],
    }
    assert pipeline_module._translation_slot_repair_context(
        cross_run_block, "译文", protected_names=["Maldacena"],
    )["missing_protected_names"] == []


def test_name_insertion_delta_ignores_existing_non_boundary_substrings() -> None:
    block = {
        "block_id": "name-substring", "type": "text", "text": "Ada uses x.",
        "inline_runs": [
            _inline_run("text", "Ada uses ", 1),
            _inline_run("math", "x", 2, tex="x"),
            _inline_run("text", ".", 3),
        ],
    }
    previous = {"blocks": [{"block_id": "name-substring", "text": "Adage。"}]}
    slot_ids = pipeline_module._translation_repair_slot_ids(block)
    result = pipeline_module._apply_translation_slot_repairs(
        previous,
        [block],
        {"repairs": [{
            "block_id": "name-substring",
            "slots": [
                {"slot_id": slot_ids[0], "text": "Adage。"},
                {"slot_id": slot_ids[1], "text": "Ada"},
            ],
        }]},
        protected_names=["Ada"],
    )
    assert "Adage" in result["blocks"][0]["text"]
    assert pipeline_module._natural_text_for_name_validation(block) == "Ada uses \n."
    _validate_translation(
        {"segment_id": "name-substring", "block_ids": ["name-substring"]},
        result, {"name-substring": block}, ["Ada"],
    )


def test_one_repair_call_handles_multiple_mismatched_blocks_and_preserves_valid_block(tmp_path: Path) -> None:
    blocks = []
    tokens = {}
    for number, kind in ((1, "math"), (2, "citation"), (3, "link")):
        run = _inline_run(kind, f"owned-{number}", 2, tex=f"x_{number}")
        if kind == "link":
            run["href"] = "https://example.test/owned"
        block = {
            "block_id": f"b{number}", "type": "text", "text": f"Text {number}.",
            "inline_runs": [
                _inline_run("text", f"Text {number} ", 1), run, _inline_run("text", ".", 3),
            ],
        }
        blocks.append(block)
        tokens[f"b{number}"] = pipeline_module._opaque_inline_tokens(block)[0]
    document = {
        "front_matter": {}, "blocks": blocks, "equations": [], "figures": [], "tables": [],
        "bibliography": [], "assets": [], "integrity": {"status": "complete"},
    }
    bundle = SourceBundle(
        paper_id="arXiv:1234.5678", parsed={"document": document}, document=document,
        metadata={}, references=[], citers=[],
    )
    calls: list[str] = []

    def llm(prompt: str, **kwargs):
        label = str(kwargs["call_label"])
        calls.append(label)
        if label.endswith("retry-offset-1"):
            assert "GLOSSARY:" not in prompt
            assert "Target language:" not in prompt
            return {"repairs": [
                {"block_id": "b1", "slots": [
                    *_offset_slots(blocks[0], "甲", [1]),
                ]},
                {"block_id": "b2", "slots": [
                    *_offset_slots(blocks[1], "乙", [0]),
                ]},
            ]}
        return {"blocks": [
            {"block_id": "b1", "text": "甲"},
            {"block_id": "b2", "text": "乙"},
            {"block_id": "b3", "text": f"保留{tokens['b3']}原文"},
        ]}

    result = _generate_translations(
        [{"segment_id": "seg-0001", "block_ids": ["b1", "b2", "b3"]}],
        options=BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path, workers=1),
        bundle=bundle, glossary={"entries": []}, protected_names=[],
        checkpoint_dir=tmp_path / "checkpoints", llm=llm,
    )
    assert calls == [
        "companion-translation-seg-0001",
        "companion-translation-seg-0001-retry-offset-1",
    ]
    assert result["seg-0001"]["blocks"][2]["text"] == f"保留{tokens['b3']}原文"
    assert pipeline_module._OPAQUE_INLINE_PATTERN.findall(
        result["seg-0001"]["blocks"][0]["text"]
    ) == [tokens["b1"]]
    assert pipeline_module._OPAQUE_INLINE_PATTERN.findall(
        result["seg-0001"]["blocks"][1]["text"]
    ) == [tokens["b2"]]


def test_coverage_normalization_preserves_unique_blocks_and_repairs_duplicates() -> None:
    blocks = [
        {"block_id": value, "type": "text", "text": f"Source {value}."}
        for value in ("b1", "b2", "b3")
    ]
    by_id = {str(block["block_id"]): block for block in blocks}
    segment = {"segment_id": "seg-0063", "block_ids": ["b1", "b2", "b3"]}
    kept_b1 = {"block_id": "b1", "text": "  已有一  ", "audit": {"raw": True}}
    kept_b3 = {"block_id": "b3", "text": "已有三"}
    candidate = {"blocks": [
        kept_b3,
        {"block_id": "unknown", "text": "discard"},
        kept_b1,
        {"block_id": "b2", "text": "ambiguous-a"},
        {"block_id": "b2", "text": "ambiguous-b"},
    ]}

    normalized, missing, diagnostics = pipeline_module._normalize_translation_coverage(
        segment, candidate, by_id,
    )
    assert normalized["blocks"] == [kept_b1, kept_b3]
    assert normalized["blocks"][0] is kept_b1
    assert normalized["blocks"][1] is kept_b3
    assert [block["block_id"] for block in missing] == ["b2"]
    assert diagnostics["duplicate_block_ids"] == ["b2"]
    assert diagnostics["discarded_unknown_block_ids"] == ["unknown"]

    repaired = pipeline_module._apply_translation_coverage_repairs(
        normalized,
        segment,
        missing,
        {"repairs": [{
            "block_id": "b2",
            "slots": [{
                "slot_id": pipeline_module._translation_coverage_slot_ids(blocks[1])[0],
                "text": "新增二",
            }],
        }]},
        by_id,
    )
    assert repaired["blocks"][0] is kept_b1
    assert repaired["blocks"][2] is kept_b3
    assert [item["block_id"] for item in repaired["blocks"]] == ["b1", "b2", "b3"]
    _validate_translation(segment, repaired, by_id, [])


def test_coverage_normalization_treats_empty_natural_language_as_missing() -> None:
    math_run = _inline_run("math", "x", 2, tex="x")
    blocks = [
        {"block_id": "kept", "type": "text", "text": "Keep this."},
        {
            "block_id": "empty", "type": "text", "text": "Value x.",
            "inline_runs": [
                _inline_run("text", "Value ", 1), math_run,
                _inline_run("text", ".", 3),
            ],
        },
    ]
    by_id = {str(block["block_id"]): block for block in blocks}
    segment = {"segment_id": "seg-empty", "block_ids": ["kept", "empty"]}
    token = pipeline_module._opaque_inline_tokens(blocks[1])[0]
    kept = {"block_id": "kept", "text": "保留此句。"}

    normalized, missing, diagnostics = pipeline_module._normalize_translation_coverage(
        segment,
        {"blocks": [kept, {"block_id": "empty", "text": token}]},
        by_id,
    )

    assert normalized["blocks"] == [kept]
    assert [block["block_id"] for block in missing] == ["empty"]
    assert diagnostics["preserved_block_ids"] == ["kept"]
    assert diagnostics["empty_block_ids"] == ["empty"]


def test_empty_primary_block_resumes_only_missing_segment_with_token_safe_repair(
    tmp_path: Path,
) -> None:
    math_run = _inline_run("math", "x", 2, tex="x")
    blocks = [
        {"block_id": "done", "type": "text", "text": "Already complete."},
        {"block_id": "kept", "type": "text", "text": "Keep this translation."},
        {
            "block_id": "empty", "type": "text", "text": "Translate x safely.",
            "inline_runs": [
                _inline_run("text", "Translate ", 1), math_run,
                _inline_run("text", " safely.", 3),
            ],
        },
    ]
    document = {
        "front_matter": {}, "blocks": blocks, "equations": [], "figures": [],
        "tables": [], "bibliography": [], "assets": [],
        "integrity": {"status": "complete"},
    }
    bundle = SourceBundle(
        paper_id="arXiv:1234.5678", parsed={"document": document}, document=document,
        metadata={}, references=[], citers=[],
    )
    completed = {"segment_id": "seg-done", "block_ids": ["done"]}
    missing = {"segment_id": "seg-missing", "block_ids": ["kept", "empty"]}
    options = BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path, workers=2)
    checkpoint_dir = tmp_path / "checkpoints"

    _generate_translations(
        [completed], options=options, bundle=bundle, glossary={"entries": []},
        protected_names=[], checkpoint_dir=checkpoint_dir,
        llm=lambda prompt, **kwargs: {
            "blocks": [{"block_id": "done", "text": "已经完成。"}],
        },
    )
    kept = {"block_id": "kept", "text": "逐字保留已有译文。", "audit": "keep"}
    token = pipeline_module._opaque_inline_tokens(blocks[2])[0]
    draft_path = pipeline_module._seed_translation_coverage_draft(
        missing,
        options=options,
        bundle=bundle,
        glossary={"entries": []},
        protected_names=[],
        checkpoint_dir=checkpoint_dir,
        translation={"blocks": [
            kept,
            {"block_id": "empty", "text": token},
        ]},
    )
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    draft["candidate_provenance"] = {
        "origin": "primary-model",
        "prompt_version": pipeline_module.PROMPT_VERSION,
        "response_schema_version": pipeline_module.SCHEMA_VERSION,
        "model_tier": pipeline_module.TRANSLATION_TIER,
    }
    draft_path.write_text(json.dumps(draft), encoding="utf-8")
    calls: list[str] = []

    def repair_llm(prompt: str, **kwargs):
        label = str(kwargs["call_label"])
        calls.append(label)
        assert label == "companion-translation-seg-missing-coverage-repair-1"
        assert kwargs["env"]["ARC_CODEX_ENABLE_MCP"] == "false"
        assert kwargs["env"]["ARC_CODEX_ALLOW_INTERNET"] == "false"
        assert token not in prompt
        return {"repairs": [{
            "block_id": "empty",
            "slots": [
                {"slot_id": slot_id, "text": text}
                for slot_id, text in zip(
                    pipeline_module._translation_coverage_slot_ids(blocks[2]),
                    ["安全翻译", "。"],
                )
            ],
        }]}

    result = _generate_translations(
        [completed, missing], options=options, bundle=bundle,
        glossary={"entries": []}, protected_names=[], checkpoint_dir=checkpoint_dir,
        llm=repair_llm,
    )

    assert calls == ["companion-translation-seg-missing-coverage-repair-1"]
    assert result["seg-done"]["blocks"][0]["text"] == "已经完成。"
    repaired = result["seg-missing"]["blocks"]
    assert repaired[0] == kept
    assert [block["block_id"] for block in repaired] == ["kept", "empty"]
    assert pipeline_module._OPAQUE_INLINE_PATTERN.findall(repaired[1]["text"]) == [token]
    assert pipeline_module._translation_natural_residue(repaired[1]["text"]).strip()


def test_coverage_repair_translates_all_omissions_with_controller_owned_dense_tokens(
    tmp_path: Path,
) -> None:
    math_run = _inline_run("math", "x", 2, tex="x")
    citation_run = _inline_run("citation", "[7]", 4)
    blocks = [
        {"block_id": "kept", "type": "text", "text": "Kept source."},
        {
            "block_id": "dense", "type": "text", "text": "Before x after [7] end.",
            "inline_runs": [
                _inline_run("text", "Before ", 1), math_run,
                _inline_run("text", " after ", 3), citation_run,
                _inline_run("text", " end.", 5),
            ],
        },
        {"block_id": "missing", "type": "text", "text": "Another omitted block."},
    ]
    document = {
        "front_matter": {}, "blocks": blocks, "equations": [], "figures": [],
        "tables": [], "bibliography": [], "assets": [], "integrity": {"status": "complete"},
    }
    bundle = SourceBundle(
        paper_id="arXiv:1234.5678", parsed={"document": document}, document=document,
        metadata={}, references=[], citers=[],
    )
    segment = {"segment_id": "seg-0063", "block_ids": ["kept", "dense", "missing"]}
    kept = {"block_id": "kept", "text": "  逐字节保留  "}
    calls: list[str] = []

    def llm(prompt: str, **kwargs):
        label = str(kwargs["call_label"])
        calls.append(label)
        if label.endswith("coverage-repair-1"):
            assert kwargs["model_tier"] == "medium"
            assert kwargs["env"]["ARC_CODEX_ENABLE_MCP"] == "false"
            assert kwargs["env"]["ARC_CLAUDE_ALLOW_MCP"] == "false"
            assert kwargs["env"]["ARC_CODEX_ALLOW_INTERNET"] == "false"
            assert kwargs["env"]["ARC_CLAUDE_ALLOW_INTERNET"] == "false"
            assert pipeline_module.TRANSLATION_COVERAGE_REPAIR_PROMPT_VERSION in prompt
            assert '"access": {"allow_mcp": false, "allow_internet": false}' in prompt
            assert all(token not in prompt for token in pipeline_module._opaque_inline_tokens(blocks[1]))
            return {"repairs": [
                {"block_id": "dense", "slots": [
                    {"slot_id": value, "text": text}
                    for value, text in zip(
                        pipeline_module._translation_coverage_slot_ids(blocks[1]),
                        ["之前", "之后", "结束。"],
                    )
                ]},
                {"block_id": "missing", "slots": [{
                    "slot_id": pipeline_module._translation_coverage_slot_ids(blocks[2])[0],
                    "text": "另一个遗漏块。",
                }]},
            ]}
        return {"blocks": [kept]}

    checkpoint_dir = tmp_path / "checkpoints"
    result = _generate_translations(
        [segment],
        options=BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path, workers=1),
        bundle=bundle, glossary={"entries": []}, protected_names=[],
        checkpoint_dir=checkpoint_dir, llm=llm,
    )["seg-0063"]

    assert calls == [
        "companion-translation-seg-0063",
        "companion-translation-seg-0063-coverage-repair-1",
    ]
    assert result["blocks"][0] is kept
    assert result["blocks"][0]["text"] == "  逐字节保留  "
    assert pipeline_module._OPAQUE_INLINE_PATTERN.findall(result["blocks"][1]["text"]) == (
        pipeline_module._opaque_inline_tokens(blocks[1])
    )
    final = json.loads(next((checkpoint_dir / "translations").glob("*.json")).read_text())
    assert final["generation_provenance"]["repairs"][0]["kind"] == "coverage"
    draft = json.loads(next((checkpoint_dir / "translation-drafts").glob("*.json")).read_text())
    assert draft["translation"] == {"blocks": [kept]}


def test_coverage_repair_rejects_a_preserved_token_invalid_block_without_second_model(
    tmp_path: Path,
) -> None:
    run = _inline_run("math", "x", 2, tex="x")
    token_block = {
        "block_id": "md-line-2507", "type": "text", "text": "factor x.",
        "inline_runs": [
            _inline_run("text", "factor ", 1), run, _inline_run("text", ".", 3),
        ],
    }
    missing_block = {"block_id": "missing", "type": "text", "text": "More text."}
    blocks = [token_block, missing_block]
    document = {
        "front_matter": {}, "blocks": blocks, "equations": [], "figures": [],
        "tables": [], "bibliography": [], "assets": [],
        "integrity": {"status": "complete"},
    }
    bundle = SourceBundle(
        paper_id="arXiv:1234.5678", parsed={"document": document}, document=document,
        metadata={}, references=[], citers=[],
    )
    calls: list[str] = []

    def llm(prompt: str, **kwargs):
        label = str(kwargs["call_label"])
        calls.append(label)
        if label.endswith("coverage-repair-1"):
            return {"repairs": [{
                "block_id": "missing",
                "slots": [{
                    "slot_id": pipeline_module._translation_coverage_slot_ids(
                        missing_block
                    )[0],
                    "text": "更多文字。",
                }],
            }]}
        return {"blocks": [{"block_id": "md-line-2507", "text": "写下该因子。"}]}

    with pytest.raises(CompanionLaneError):
        _generate_translations(
            [{"segment_id": "seg-0242", "block_ids": ["md-line-2507", "missing"]}],
            options=BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path, workers=1),
            bundle=bundle, glossary={"entries": []}, protected_names=[],
            checkpoint_dir=tmp_path / "checkpoints", llm=llm,
        )

    assert calls == [
        "companion-translation-seg-0242",
        "companion-translation-seg-0242-coverage-repair-1",
    ]


def test_coverage_draft_resumes_repair_after_interruption_before_attempt(
    tmp_path: Path, monkeypatch,
) -> None:
    block = {"block_id": "body", "type": "text", "text": "Missing source."}
    document = {
        "front_matter": {}, "blocks": [block], "equations": [], "figures": [],
        "tables": [], "bibliography": [], "assets": [], "integrity": {"status": "complete"},
    }
    bundle = SourceBundle(
        paper_id="arXiv:1234.5678", parsed={"document": document}, document=document,
        metadata={}, references=[], citers=[],
    )
    segment = {"segment_id": "seg-0063", "block_ids": ["body"]}
    checkpoint_dir = tmp_path / "checkpoints"
    original_write_json = pipeline_module.write_json

    def interrupted_write(path, value):
        if "translation-coverage-attempts" in Path(path).parts:
            raise RuntimeError("simulated interruption before attempt marker")
        original_write_json(path, value)

    monkeypatch.setattr(pipeline_module, "write_json", interrupted_write)
    first_calls: list[str] = []

    def primary_llm(prompt: str, **kwargs):
        first_calls.append(str(kwargs["call_label"]))
        return {"blocks": []}

    with pytest.raises(CompanionLaneError):
        _generate_translations(
            [segment],
            options=BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path, workers=1),
            bundle=bundle, glossary={"entries": []}, protected_names=[],
            checkpoint_dir=checkpoint_dir, llm=primary_llm,
        )
    assert first_calls == ["companion-translation-seg-0063"]
    assert list((checkpoint_dir / "translation-drafts").glob("*.json"))
    assert not list((checkpoint_dir / "translation-coverage-attempts").glob("*.json"))
    assert not list((checkpoint_dir / "translations").glob("*.json"))

    monkeypatch.setattr(pipeline_module, "write_json", original_write_json)
    resume_calls: list[str] = []

    def repair_llm(prompt: str, **kwargs):
        resume_calls.append(str(kwargs["call_label"]))
        return {"repairs": [{"block_id": "body", "slots": [{
            "slot_id": pipeline_module._translation_coverage_slot_ids(block)[0],
            "text": "补齐译文。",
        }]}]}

    result = _generate_translations(
        [segment],
        options=BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path, workers=1),
        bundle=bundle, glossary={"entries": []}, protected_names=[],
        checkpoint_dir=checkpoint_dir, llm=repair_llm,
    )
    assert resume_calls == ["companion-translation-seg-0063-coverage-repair-1"]
    assert result["seg-0063"]["blocks"][0]["text"] == "补齐译文。"


def test_invalid_coverage_response_is_checkpointed_and_replayed_without_model(
    tmp_path: Path,
) -> None:
    block = {"block_id": "body", "type": "text", "text": "Ada reports the result."}
    document = {
        "front_matter": {}, "blocks": [block], "equations": [], "figures": [],
        "tables": [], "bibliography": [], "assets": [], "integrity": {"status": "complete"},
    }
    bundle = SourceBundle(
        paper_id="arXiv:1234.5678", parsed={"document": document}, document=document,
        metadata={}, references=[], citers=[],
    )
    segment = {"segment_id": "seg-0063", "block_ids": ["body"]}
    checkpoint_dir = tmp_path / "checkpoints"
    calls: list[str] = []

    def invalid_llm(prompt: str, **kwargs):
        label = str(kwargs["call_label"])
        calls.append(label)
        if label.endswith("coverage-repair-1"):
            return {"repairs": [{"block_id": "body", "slots": [{
                "slot_id": "body.invalid-slot",
                "text": "省略姓名的译文。",
            }]}]}
        return {"blocks": []}

    with pytest.raises(CompanionLaneError):
        _generate_translations(
            [segment],
            options=BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path, workers=1),
            bundle=bundle, glossary={"entries": []}, protected_names=["Ada"],
            checkpoint_dir=checkpoint_dir, llm=invalid_llm,
        )
    assert calls == [
        "companion-translation-seg-0063",
        "companion-translation-seg-0063-coverage-repair-1",
    ]
    assert list((checkpoint_dir / "translation-coverage-attempts").glob("*.json"))
    assert not list((checkpoint_dir / "translations").glob("*.json"))
    marker_path = pipeline_module._translation_coverage_attempt_path(
        checkpoint_dir, segment["segment_id"],
    )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["schema_version"] == (
        pipeline_module.TRANSLATION_COVERAGE_ATTEMPT_SCHEMA_VERSION
    )
    assert marker["status"] == "response_received"
    assert marker["raw_response"]["repairs"][0]["slots"][0]["slot_id"] == (
        "body.invalid-slot"
    )

    def forbidden_llm(prompt: str, **kwargs):
        raise AssertionError("consumed coverage repair must not invoke any model")

    with pytest.raises(CompanionLaneError) as exc_info:
        _generate_translations(
            [segment],
            options=BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path, workers=1),
            bundle=bundle, glossary={"entries": []}, protected_names=["Ada"],
            checkpoint_dir=checkpoint_dir, llm=forbidden_llm,
        )
    assert "changed slot coverage" in str(exc_info.value)


def test_legacy_started_coverage_attempt_gets_one_v2_upgrade(tmp_path: Path) -> None:
    block = {"block_id": "body", "type": "text", "text": "Missing source."}
    document = {
        "front_matter": {}, "blocks": [block], "equations": [], "figures": [],
        "tables": [], "bibliography": [], "assets": [], "integrity": {"status": "complete"},
    }
    bundle = SourceBundle(
        paper_id="arXiv:1234.5678", parsed={"document": document}, document=document,
        metadata={}, references=[], citers=[],
    )
    segment = {"segment_id": "seg-legacy-coverage", "block_ids": ["body"]}
    options = BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path, workers=1)
    checkpoint_dir = tmp_path / "checkpoints"
    draft_path = pipeline_module._seed_translation_coverage_draft(
        segment, options=options, bundle=bundle, glossary={"entries": []},
        protected_names=[], checkpoint_dir=checkpoint_dir,
    )
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    attempt_path = pipeline_module._translation_coverage_attempt_path(
        checkpoint_dir, segment["segment_id"],
    )
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_path.write_text(json.dumps({
        "schema_version": "arc.companion.translation-coverage-attempt.v1",
        "segment_id": segment["segment_id"],
        "input_sha256": draft["input_sha256"],
        "status": "started",
        "started_at": "2026-01-01T00:00:00+00:00",
        "prompt_version": pipeline_module.TRANSLATION_COVERAGE_REPAIR_PROMPT_VERSION,
        "response_schema_version": (
            pipeline_module.TRANSLATION_COVERAGE_REPAIR_SCHEMA_VERSION
        ),
        "model_tier": pipeline_module.TRANSLATION_COVERAGE_REPAIR_TIER,
        "missing_block_ids": ["body"],
    }), encoding="utf-8")
    calls: list[str] = []

    def repair_llm(prompt: str, **kwargs):
        calls.append(str(kwargs["call_label"]))
        return {"repairs": [{"block_id": "body", "slots": [{
            "slot_id": pipeline_module._translation_coverage_slot_ids(block)[0],
            "text": "补齐译文。",
        }]}]}

    result = _generate_translations(
        [segment], options=options, bundle=bundle, glossary={"entries": []},
        protected_names=[], checkpoint_dir=checkpoint_dir, llm=repair_llm,
    )
    assert result[segment["segment_id"]]["blocks"][0]["text"] == "补齐译文。"
    assert calls == [f"companion-translation-{segment['segment_id']}-coverage-repair-1"]
    upgraded = json.loads(attempt_path.read_text(encoding="utf-8"))
    assert upgraded["status"] == "validated"
    assert upgraded["superseded_attempt"]["schema_version"].endswith(".v1")


def test_concurrent_legacy_coverage_upgrades_stop_at_build_circuit(
    tmp_path: Path,
) -> None:
    blocks = [
        {"block_id": f"body-{index}", "type": "text", "text": f"Missing source {index}."}
        for index in range(6)
    ]
    document = {
        "front_matter": {}, "blocks": blocks, "equations": [], "figures": [],
        "tables": [], "bibliography": [], "assets": [],
        "integrity": {"status": "complete"},
    }
    bundle = SourceBundle(
        paper_id="arXiv:1234.5678", parsed={"document": document}, document=document,
        metadata={}, references=[], citers=[],
    )
    segments = [
        {"segment_id": f"seg-legacy-{index}", "block_ids": [f"body-{index}"]}
        for index in range(6)
    ]
    options = BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path, workers=2)
    checkpoint_dir = tmp_path / "checkpoints"
    marker_paths: list[Path] = []
    for segment in segments:
        draft_path = pipeline_module._seed_translation_coverage_draft(
            segment,
            options=options,
            bundle=bundle,
            glossary={"entries": []},
            protected_names=[],
            checkpoint_dir=checkpoint_dir,
        )
        draft = json.loads(draft_path.read_text(encoding="utf-8"))
        marker_path = pipeline_module._translation_coverage_attempt_path(
            checkpoint_dir, segment["segment_id"],
        )
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(json.dumps({
            "schema_version": "arc.companion.translation-coverage-attempt.v1",
            "segment_id": segment["segment_id"],
            "input_sha256": draft["input_sha256"],
            "status": "started",
            "started_at": "2026-01-01T00:00:00+00:00",
            "prompt_version": pipeline_module.TRANSLATION_COVERAGE_REPAIR_PROMPT_VERSION,
            "response_schema_version": (
                pipeline_module.TRANSLATION_COVERAGE_REPAIR_SCHEMA_VERSION
            ),
            "model_tier": pipeline_module.TRANSLATION_COVERAGE_REPAIR_TIER,
            "missing_block_ids": list(segment["block_ids"]),
        }), encoding="utf-8")
        marker_paths.append(marker_path)

    class FatalProviderError(RuntimeError):
        abort_batch = True

    initial_budget = threading.Barrier(2)
    calls: list[str] = []
    calls_lock = threading.Lock()

    def fatal_provider(prompt: str, **kwargs):
        with calls_lock:
            calls.append(str(kwargs["call_label"]))
        call_dir = Path(kwargs["artifact_dir"]) / "call-checkpoints"
        call_dir.mkdir(parents=True, exist_ok=True)
        (call_dir / "submitted.json").write_text(json.dumps({
            "state": "submitted", "submission_state": "unknown",
        }), encoding="utf-8")
        initial_budget.wait(timeout=5)
        try:
            raise FatalProviderError("usage quota exhausted")
        except FatalProviderError as exc:
            raise RuntimeError("wrapped provider failure") from exc

    limited_llm = pipeline_module._limit_llm_concurrency(fatal_provider, 2)
    with pytest.raises(CompanionLaneError) as exc_info:
        _generate_translations(
            segments,
            options=options,
            bundle=bundle,
            glossary={"entries": []},
            protected_names=[],
            checkpoint_dir=checkpoint_dir,
            llm=limited_llm,
        )

    assert len(calls) == 2
    assert all(label.endswith("coverage-repair-1") for label in calls)
    failures = [failure for _, failure in exc_info.value.failures]
    assert sum(isinstance(failure, CompanionLLMCircuitOpen) for failure in failures) == 4
    markers = [json.loads(path.read_text(encoding="utf-8")) for path in marker_paths]
    assert sum(marker["schema_version"].endswith(".v2") for marker in markers) == 2
    assert sum(marker["schema_version"].endswith(".v1") for marker in markers) == 4
    assert all(
        marker["status"] == "started"
        and marker["superseded_attempt"]["schema_version"].endswith(".v1")
        for marker in markers if marker["schema_version"].endswith(".v2")
    )

    def forbidden_provider(prompt: str, **kwargs):
        raise AssertionError("an actually submitted coverage upgrade must remain consumed")

    submitted_segments = [
        segment for segment, marker in zip(segments, markers)
        if marker["schema_version"].endswith(".v2")
    ]
    with pytest.raises(CompanionLaneError, match="already started"):
        _generate_translations(
            submitted_segments,
            options=options,
            bundle=bundle,
            glossary={"entries": []},
            protected_names=[],
            checkpoint_dir=checkpoint_dir,
            llm=forbidden_provider,
        )
    assert [json.loads(path.read_text(encoding="utf-8")) for path in marker_paths] == markers


def test_current_started_coverage_attempt_fails_closed(tmp_path: Path) -> None:
    block = {"block_id": "body", "type": "text", "text": "Missing source."}
    document = {
        "front_matter": {}, "blocks": [block], "equations": [], "figures": [],
        "tables": [], "bibliography": [], "assets": [], "integrity": {"status": "complete"},
    }
    bundle = SourceBundle(
        paper_id="arXiv:1234.5678", parsed={"document": document}, document=document,
        metadata={}, references=[], citers=[],
    )
    segment = {"segment_id": "seg-current-coverage", "block_ids": ["body"]}
    options = BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path, workers=1)
    checkpoint_dir = tmp_path / "checkpoints"
    draft_path = pipeline_module._seed_translation_coverage_draft(
        segment, options=options, bundle=bundle, glossary={"entries": []},
        protected_names=[], checkpoint_dir=checkpoint_dir,
    )
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    attempt_path = pipeline_module._translation_coverage_attempt_path(
        checkpoint_dir, segment["segment_id"],
    )
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_path.write_text(json.dumps({
        "schema_version": pipeline_module.TRANSLATION_COVERAGE_ATTEMPT_SCHEMA_VERSION,
        "segment_id": segment["segment_id"],
        "input_sha256": draft["input_sha256"],
        "status": "started",
        "started_at": "2026-01-01T00:00:00+00:00",
        "prompt_version": pipeline_module.TRANSLATION_COVERAGE_REPAIR_PROMPT_VERSION,
        "response_schema_version": (
            pipeline_module.TRANSLATION_COVERAGE_REPAIR_SCHEMA_VERSION
        ),
        "model_tier": pipeline_module.TRANSLATION_COVERAGE_REPAIR_TIER,
        "missing_block_ids": ["body"],
    }), encoding="utf-8")
    call_dir = (
        checkpoint_dir / "llm" / "translations"
        / pipeline_module._segment_checkpoint_name(segment["segment_id"])
        / "coverage-repair-1" / "call-checkpoints"
    )
    call_dir.mkdir(parents=True)
    (call_dir / "submitted.json").write_text(json.dumps({
        "state": "submitted", "submission_state": "submitted",
    }), encoding="utf-8")

    def forbidden_llm(prompt: str, **kwargs):
        raise AssertionError("current started coverage attempt must not call the model")

    with pytest.raises(CompanionLaneError, match="already started"):
        _generate_translations(
            [segment], options=options, bundle=bundle, glossary={"entries": []},
            protected_names=[], checkpoint_dir=checkpoint_dir, llm=forbidden_llm,
        )


def test_controller_seeded_empty_draft_enters_repair_only(tmp_path: Path) -> None:
    block = {"block_id": "body", "type": "text", "text": "Seeded source."}
    document = {
        "front_matter": {}, "blocks": [block], "equations": [], "figures": [],
        "tables": [], "bibliography": [], "assets": [], "integrity": {"status": "complete"},
    }
    bundle = SourceBundle(
        paper_id="arXiv:1234.5678", parsed={"document": document}, document=document,
        metadata={}, references=[], citers=[],
    )
    segment = {"segment_id": "seg-0063", "block_ids": ["body"]}
    options = BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path, workers=1)
    checkpoint_dir = tmp_path / "checkpoints"
    draft_path = pipeline_module._seed_translation_coverage_draft(
        segment,
        options=options,
        bundle=bundle,
        glossary={"entries": []},
        protected_names=[],
        checkpoint_dir=checkpoint_dir,
    )
    assert json.loads(draft_path.read_text())["candidate_provenance"] == {
        "origin": "controller-seed",
        "prompt_version": None,
        "response_schema_version": None,
        "model_tier": None,
    }
    calls: list[str] = []

    def llm(prompt: str, **kwargs):
        calls.append(str(kwargs["call_label"]))
        return {"repairs": [{"block_id": "body", "slots": [{
            "slot_id": pipeline_module._translation_coverage_slot_ids(block)[0],
            "text": "种子补译。",
        }]}]}

    result = _generate_translations(
        [segment], options=options, bundle=bundle, glossary={"entries": []},
        protected_names=[], checkpoint_dir=checkpoint_dir, llm=llm,
    )
    assert calls == ["companion-translation-seg-0063-coverage-repair-1"]
    assert result["seg-0063"]["blocks"][0]["text"] == "种子补译。"


def test_offline_runtime_env_overrides_polluted_parent_access(monkeypatch) -> None:
    for key in (
        "ARC_CODEX_ALLOW_INTERNET",
        "ARC_CLAUDE_ALLOW_INTERNET",
        "ARC_CODEX_ENABLE_MCP",
        "ARC_CLAUDE_ALLOW_MCP",
    ):
        monkeypatch.setenv(key, "true")
    monkeypatch.setenv("ARC_CODEX_MCP_MODE", "unrestricted")
    monkeypatch.setenv("ARC_CLAUDE_MCP_MODE", "unrestricted")

    env = pipeline_module._llm_runtime_env(
        allow_internet=False,
        force_disable_internet=True,
    )

    assert env["ARC_CODEX_ALLOW_INTERNET"] == "false"
    assert env["ARC_CLAUDE_ALLOW_INTERNET"] == "false"
    assert env["ARC_CODEX_ENABLE_MCP"] == "false"
    assert env["ARC_CLAUDE_ALLOW_MCP"] == "false"
    assert env["ARC_PAPER_CLI_ACCESS"] == "full"
    assert env["ARC_LLM_INHERIT_HOST_TOOLS"] == "false"
    assert "ARC_CODEX_MCP_MODE" not in env
    assert "ARC_CLAUDE_MCP_MODE" not in env


@pytest.mark.parametrize("user_allows_internet", [False, True])
def test_translation_primary_and_repairs_share_offline_runtime_recipe(
    tmp_path: Path, user_allows_internet: bool,
) -> None:
    options = BuildOptions(
        paper_id="arXiv:1234.5678", project_dir=tmp_path,
        allow_internet=user_allows_internet,
    )
    environments: list[dict[str, str]] = []

    def capture(_prompt: str, **kwargs):
        environments.append(dict(kwargs["env"]))
        return {"ok": True}

    for label in ("primary", "token-repair", "coverage-repair"):
        pipeline_module._llm_call(
            capture, label, {"type": "object"}, options=options,
            artifact_dir=tmp_path / label, call_label=label,
            model_tier=pipeline_module.TRANSLATION_TIER,
            force_offline=True,
        )

    recipes = [{
        key: env.get(key) for key in (
            "ARC_CODEX_ALLOW_INTERNET", "ARC_CLAUDE_ALLOW_INTERNET",
            "ARC_CODEX_ENABLE_MCP", "ARC_CLAUDE_ALLOW_MCP",
            "ARC_LLM_INHERIT_HOST_TOOLS",
        )
    } for env in environments]
    assert recipes[0] == recipes[1] == recipes[2]
    assert recipes[0]["ARC_CODEX_ALLOW_INTERNET"] == "false"


def test_coverage_repair_uses_no_second_model_call_for_token_placement(tmp_path: Path) -> None:
    run = _inline_run("math", "x", 2, tex="x")
    blocks = [
        {
            "block_id": "bad-token", "type": "text", "text": "Value x.",
            "inline_runs": [_inline_run("text", "Value ", 1), run, _inline_run("text", ".", 3)],
        },
        {"block_id": "missing", "type": "text", "text": "Missing."},
    ]
    document = {
        "front_matter": {}, "blocks": blocks, "equations": [], "figures": [],
        "tables": [], "bibliography": [], "assets": [], "integrity": {"status": "complete"},
    }
    bundle = SourceBundle(
        paper_id="arXiv:1234.5678", parsed={"document": document}, document=document,
        metadata={}, references=[], citers=[],
    )
    calls: list[str] = []

    def llm(prompt: str, **kwargs):
        label = str(kwargs["call_label"])
        calls.append(label)
        if label.endswith("coverage-repair-1"):
            return {"repairs": [{"block_id": "missing", "slots": [{
                "slot_id": pipeline_module._translation_coverage_slot_ids(blocks[1])[0],
                "text": "补译。",
            }]}]}
        return {"blocks": [{"block_id": "bad-token", "text": "缺失 token。"}]}

    checkpoint_dir = tmp_path / "checkpoints"
    with pytest.raises(CompanionLaneError):
        _generate_translations(
            [{"segment_id": "seg-0063", "block_ids": ["bad-token", "missing"]}],
            options=BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path, workers=1),
            bundle=bundle, glossary={"entries": []}, protected_names=[],
            checkpoint_dir=checkpoint_dir, llm=llm,
        )
    assert calls == [
        "companion-translation-seg-0063",
        "companion-translation-seg-0063-coverage-repair-1",
    ]
    assert not list((checkpoint_dir / "translations").glob("*.json"))


def test_empty_block_uses_coverage_repair_before_preserved_token_validation(tmp_path: Path) -> None:
    blocks = []
    for number in (1, 2):
        blocks.append({
            "block_id": f"b{number}", "type": "text", "text": f"Text {number} x.",
            "inline_runs": [
                _inline_run("text", f"Text {number} ", 1),
                _inline_run("math", f"x_{number}", 2, tex=f"x_{number}"),
                _inline_run("text", ".", 3),
            ],
        })
    token_b2 = pipeline_module._opaque_inline_tokens(blocks[1])[0]
    document = {
        "front_matter": {}, "blocks": blocks, "equations": [], "figures": [], "tables": [],
        "bibliography": [], "assets": [], "integrity": {"status": "complete"},
    }
    bundle = SourceBundle(
        paper_id="arXiv:1234.5678", parsed={"document": document}, document=document,
        metadata={}, references=[], citers=[],
    )
    calls: list[str] = []

    def llm(prompt: str, **kwargs):
        label = str(kwargs["call_label"])
        calls.append(label)
        if label.endswith("coverage-repair-1"):
            return {"repairs": [{
                "block_id": "b2",
                "slots": [
                    {"slot_id": slot_id, "text": text}
                    for slot_id, text in zip(
                        pipeline_module._translation_coverage_slot_ids(blocks[1]),
                        ["第二段", "。"],
                    )
                ],
            }]}
        return {"blocks": [
            {"block_id": "b1", "text": "首块缺少 token。"},
            {"block_id": "b2", "text": token_b2},
        ]}

    try:
        _generate_translations(
            [{"segment_id": "seg-0001", "block_ids": ["b1", "b2"]}],
            options=BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path, workers=1),
            bundle=bundle, glossary={"entries": []}, protected_names=[],
            checkpoint_dir=tmp_path / "checkpoints", llm=llm,
        )
    except CompanionLaneError as exc:
        assert "opaque inline tokens" in str(exc)
    else:
        raise AssertionError("preserved token-invalid prose must fail final validation")
    assert calls == [
        "companion-translation-seg-0001",
        "companion-translation-seg-0001-coverage-repair-1",
    ]


def test_segment_preflight_allows_controller_owned_link_or_citation_only_blocks() -> None:
    for kind in ("link", "citation"):
        run = _inline_run(kind, "Maldacena" if kind == "link" else "17", 1)
        if kind == "link":
            run["href"] = "https://example.test/maldacena"
        block = {
            "block_id": f"{kind}-only", "type": "text", "text": str(run["content"]),
            "inline_runs": [run],
        }
        token = pipeline_module._opaque_inline_tokens(block)[0]
        _validate_translation(
            {"segment_id": f"seg-{kind}", "block_ids": [block["block_id"]]},
            {"blocks": [{"block_id": block["block_id"], "text": token}]},
            {block["block_id"]: block},
            ["Maldacena"],
        )


def test_extra_nontranslatable_block_normalizes_then_repairs_tokens_only(tmp_path: Path) -> None:
    math_run = _inline_run("math", "x", 2, tex="x")
    blocks = [
        {"block_id": "p1", "type": "text", "text": "First."},
        {
            "block_id": "p2", "type": "text", "text": "Value x.",
            "inline_runs": [
                _inline_run("text", "Value ", 1), math_run, _inline_run("text", ".", 3),
            ],
        },
        {"block_id": "S7.E1", "type": "equation", "text": "x"},
        {"block_id": "p3", "type": "text", "text": "Third."},
    ]
    document = {
        "front_matter": {}, "blocks": blocks, "equations": [], "figures": [],
        "tables": [], "bibliography": [], "assets": [], "integrity": {"status": "complete"},
    }
    bundle = SourceBundle(
        paper_id="arXiv:0911.3380", parsed={"document": document}, document=document,
        metadata={}, references=[], citers=[],
    )
    segment = {
        "segment_id": "seg-0063",
        "block_ids": ["p1", "p2", "S7.E1", "p3"],
    }
    p1 = {"block_id": "p1", "text": "  第一段原译  ", "audit": "preserve"}
    p3 = {"block_id": "p3", "text": "第三段原译"}
    raw = {"blocks": [
        p1,
        {"block_id": "p2", "text": "坏译文。"},
        {"block_id": "S7.E1", "text": "x"},
        p3,
    ]}
    checkpoint_dir = tmp_path / "checkpoints"
    options = BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path, workers=1)
    pipeline_module._seed_translation_coverage_draft(
        segment,
        options=options,
        bundle=bundle,
        glossary={"entries": []},
        protected_names=[],
        checkpoint_dir=checkpoint_dir,
        translation=raw,
    )
    calls: list[str] = []

    def llm(prompt: str, **kwargs):
        calls.append(str(kwargs["call_label"]))
        return {"repairs": [{"block_id": "p2", "slots": [
            *_offset_slots(blocks[1], "坏译文。", [3]),
        ]}]}

    result = _generate_translations(
        [segment], options=options, bundle=bundle, glossary={"entries": []},
        protected_names=[], checkpoint_dir=checkpoint_dir, llm=llm,
    )["seg-0063"]

    assert calls == ["companion-translation-seg-0063-retry-offset-1"]
    assert result["blocks"][0] == p1
    assert result["blocks"][2] == p3
    assert [item["block_id"] for item in result["blocks"]] == ["p1", "p2", "p3"]
    assert pipeline_module._OPAQUE_INLINE_PATTERN.findall(result["blocks"][1]["text"]) == (
        pipeline_module._opaque_inline_tokens(blocks[1])
    )
    checkpoint = json.loads(next((checkpoint_dir / "translations").glob("*.json")).read_text())
    assert [item["kind"] for item in checkpoint["generation_provenance"]["repairs"]] == [
        "coverage-normalization", "token-placement",
    ]
    assert not list((checkpoint_dir / "translation-coverage-attempts").glob("*.json"))
    assert list((checkpoint_dir / "translation-token-offset-attempts").glob("*.json"))


def test_token_invalid_draft_resumes_after_interruption_before_attempt(
    tmp_path: Path, monkeypatch,
) -> None:
    math_run = _inline_run("math", "x", 2, tex="x")
    block = {
        "block_id": "body", "type": "text", "text": "Value x.",
        "inline_runs": [
            _inline_run("text", "Value ", 1), math_run, _inline_run("text", ".", 3),
        ],
    }
    document = {
        "front_matter": {}, "blocks": [block], "equations": [], "figures": [],
        "tables": [], "bibliography": [], "assets": [], "integrity": {"status": "complete"},
    }
    bundle = SourceBundle(
        paper_id="arXiv:0911.3380", parsed={"document": document}, document=document,
        metadata={}, references=[], citers=[],
    )
    segment = {"segment_id": "seg-0063", "block_ids": ["body"]}
    checkpoint_dir = tmp_path / "checkpoints"
    original_write_json = pipeline_module.write_json

    def interrupted_write(path, value):
        if "translation-token-offset-attempts" in Path(path).parts:
            raise RuntimeError("simulated interruption before token attempt marker")
        original_write_json(path, value)

    monkeypatch.setattr(pipeline_module, "write_json", interrupted_write)
    first_calls: list[str] = []

    def primary_llm(prompt: str, **kwargs):
        first_calls.append(str(kwargs["call_label"]))
        return {"blocks": [{"block_id": "body", "text": "缺少令牌。"}]}

    options = BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path, workers=1)
    with pytest.raises(CompanionLaneError):
        _generate_translations(
            [segment], options=options, bundle=bundle, glossary={"entries": []},
            protected_names=[], checkpoint_dir=checkpoint_dir, llm=primary_llm,
        )
    assert first_calls == ["companion-translation-seg-0063"]
    assert list((checkpoint_dir / "translation-drafts").glob("*.json"))
    assert not list((checkpoint_dir / "translation-token-offset-attempts").glob("*.json"))

    monkeypatch.setattr(pipeline_module, "write_json", original_write_json)
    resume_calls: list[str] = []

    def repair_llm(prompt: str, **kwargs):
        resume_calls.append(str(kwargs["call_label"]))
        return {"repairs": [{"block_id": "body", "slots": [
            *_offset_slots(block, "缺少令牌。", [4]),
        ]}]}

    result = _generate_translations(
        [segment], options=options, bundle=bundle, glossary={"entries": []},
        protected_names=[], checkpoint_dir=checkpoint_dir, llm=repair_llm,
    )
    assert resume_calls == ["companion-translation-seg-0063-retry-offset-1"]
    assert pipeline_module._OPAQUE_INLINE_PATTERN.findall(
        result["seg-0063"]["blocks"][0]["text"]
    ) == pipeline_module._opaque_inline_tokens(block)


def test_translation_opaque_token_retry_is_bounded_and_checkpoints_successes(tmp_path: Path) -> None:
    blocks = []
    segments = []
    required_tokens: dict[str, str] = {}
    repair_slot_ids: dict[str, list[str]] = {}
    for number in range(1, 5):
        block_id_value = f"b{number}"
        math_run = _inline_run("math", f"x_{number}", 2, tex=f"x_{number}")
        block = {
            "block_id": block_id_value,
            "type": "text",
            "text": f"Value x_{number}.",
            "inline_runs": [
                _inline_run("text", "Value ", 1),
                math_run,
                _inline_run("text", ".", 3),
            ],
        }
        blocks.append(block)
        segment_id = f"seg-{number:04d}"
        segments.append({"segment_id": segment_id, "block_ids": [block_id_value]})
        required_tokens[segment_id] = pipeline_module._opaque_inline_tokens(block)[0]
        repair_slot_ids[segment_id] = pipeline_module._translation_repair_slot_ids(block)

    document = {
        "schema_version": "arc.paper.document.v1",
        "front_matter": {"title": "Retry fixture", "authors": [], "affiliations": []},
        "blocks": blocks,
        "equations": [], "figures": [], "tables": [], "bibliography": [], "assets": [],
        "integrity": {"status": "complete", "document_hash": "retry-fixture"},
    }
    bundle = SourceBundle(
        paper_id="arXiv:1234.5678",
        parsed={"paper_id": "arXiv:1234.5678", "document": document},
        document=document,
        metadata={"title": "Retry fixture"}, references=[], citers=[],
    )
    calls: dict[str, int] = {}
    calls_lock = threading.Lock()

    def retrying_llm(prompt: str, **kwargs):
        label = str(kwargs["call_label"])
        with calls_lock:
            calls[label] = calls.get(label, 0) + 1
        is_retry = label.endswith("-retry-offset-1")
        segment_id = label.removeprefix("companion-translation-").removesuffix(
            "-retry-offset-1"
        )
        block_id_value = f"b{int(segment_id[-4:])}"
        if is_retry:
            assert pipeline_module._translation_token_attempt_path(
                checkpoint_dir, segment_id,
            ).is_file()
            assert "VALIDATION ERROR" in prompt
            assert "opaque_inline_token_mismatch" in prompt
            assert required_tokens[segment_id] in prompt
            assert pipeline_module.TRANSLATION_RETRY_PROMPT_VERSION in prompt
            assert Path(kwargs["artifact_dir"]).name == "retry-offset-1"
            assert kwargs["env"]["ARC_CODEX_ENABLE_MCP"] == "false"
            assert kwargs["env"]["ARC_CLAUDE_ALLOW_MCP"] == "false"
            assert kwargs["env"]["ARC_CODEX_ALLOW_INTERNET"] == "false"
            assert kwargs["env"]["ARC_CLAUDE_ALLOW_INTERNET"] == "false"
            assert kwargs["model_tier"] == pipeline_module.TRANSLATION_RETRY_TIER == "medium"
            slots = _offset_slots(
                blocks[int(segment_id[-4:]) - 1], "缺少控制器令牌。", [7],
            )
            if segment_id == "seg-0003":
                slots = slots[:-1]
            elif segment_id == "seg-0004":
                slots[0]["start_offset"] = 1
            return {"repairs": [{"block_id": block_id_value, "slots": slots}]}
        else:
            assert kwargs["model_tier"] == pipeline_module.TRANSLATION_TIER == "medium"
            assert kwargs["env"]["ARC_CODEX_ALLOW_INTERNET"] == "false"
            assert kwargs["env"]["ARC_CLAUDE_ALLOW_INTERNET"] == "false"
            assert "perform this exact checklist" in prompt
            assert "every protected personal name" in prompt
        text = (
            f"译文 {required_tokens[segment_id]}。"
            if segment_id == "seg-0001" else "缺少控制器令牌。"
        )
        return {"blocks": [{"block_id": block_id_value, "text": text}]}

    checkpoint_dir = tmp_path / "checkpoints"
    try:
        _generate_translations(
            segments,
            options=BuildOptions(
                paper_id=bundle.paper_id, project_dir=tmp_path / "project", workers=4,
            ),
            bundle=bundle,
            glossary={"entries": []},
            protected_names=[],
            checkpoint_dir=checkpoint_dir,
            llm=retrying_llm,
        )
    except CompanionLaneError as exc:
        assert exc.lane == "translation"
        assert {segment_id for segment_id, _ in exc.failures} == {"seg-0003", "seg-0004"}
        assert all("slot repair" in str(error) for _, error in exc.failures)
    else:
        raise AssertionError("persistently invalid translations must fail the lane")

    assert calls == {
        "companion-translation-seg-0001": 1,
        "companion-translation-seg-0002": 1,
        "companion-translation-seg-0002-retry-offset-1": 1,
        "companion-translation-seg-0003": 1,
        "companion-translation-seg-0003-retry-offset-1": 1,
        "companion-translation-seg-0004": 1,
        "companion-translation-seg-0004-retry-offset-1": 1,
    }
    saved_ids = {
        json.loads(path.read_text(encoding="utf-8"))["segment_id"]
        for path in (checkpoint_dir / "translations").glob("*.json")
    }
    assert saved_ids == {"seg-0001", "seg-0002"}

    attempt_ids = {
        json.loads(path.read_text(encoding="utf-8"))["segment_id"]
        for path in (checkpoint_dir / "translation-token-offset-attempts").glob("*.json")
    }
    assert attempt_ids == {"seg-0002", "seg-0003", "seg-0004"}
    assert all(
        json.loads(path.read_text(encoding="utf-8"))["response_schema_version"]
        == pipeline_module.TRANSLATION_SLOT_REPAIR_SCHEMA_VERSION
        for path in (checkpoint_dir / "translation-token-offset-attempts").glob("*.json")
    )

    def forbidden_recovery_llm(prompt: str, **kwargs):
        raise AssertionError("consumed token repair must not invoke any model")

    with pytest.raises(CompanionLaneError) as exc_info:
        _generate_translations(
            segments,
            options=BuildOptions(
                paper_id=bundle.paper_id, project_dir=tmp_path / "project", workers=4,
            ),
            bundle=bundle,
            glossary={"entries": []},
            protected_names=[],
            checkpoint_dir=checkpoint_dir,
            llm=forbidden_recovery_llm,
        )
    assert {segment_id for segment_id, _ in exc_info.value.failures} == {
        "seg-0003", "seg-0004",
    }
    assert all("slot repair" in str(error) for _, error in exc_info.value.failures)


def test_offline_translation_session_keeps_same_runtime_for_token_repair(
    tmp_path: Path,
) -> None:
    from arc_llm.runner import _runtime_fp
    from arc_llm.sessions import LLMSessionManager

    block = {
        "block_id": "body", "type": "text", "text": "Value x.",
        "inline_runs": [
            _inline_run("text", "Value ", 1),
            _inline_run("math", "x", 2, tex="x"),
            _inline_run("text", ".", 3),
        ],
    }
    document = {
        "front_matter": {}, "blocks": [block], "equations": [], "figures": [],
        "tables": [], "bibliography": [], "assets": [],
        "integrity": {"status": "complete"},
    }
    bundle = SourceBundle(
        paper_id="local:runtime", parsed={"document": document}, document=document,
        metadata={}, references=[], citers=[],
    )
    options = BuildOptions(
        paper_id=bundle.paper_id, project_dir=tmp_path, workers=1,
        provider="codex-cli",
    )
    manager = LLMSessionManager(tmp_path / "sessions")
    offline_env = pipeline_module._llm_runtime_env(
        allow_internet=False, force_disable_internet=True, inherit_host_tools=False,
    )
    runtime_fingerprint = _runtime_fp(
        provider_used="codex-cli", model=None,
        model_tier=pipeline_module.TRANSLATION_TIER,
        env=offline_env, process_chain=None,
    )
    manager.get_or_create(
        key="ch:translation", provider="codex-cli", model=None,
        runtime_fingerprint=runtime_fingerprint,
    )
    labels: list[str] = []

    def stateful_llm(prompt: str, **kwargs):
        label = str(kwargs["call_label"])
        labels.append(label)
        fingerprint = _runtime_fp(
            provider_used="codex-cli", model=kwargs.get("model"),
            model_tier=kwargs.get("model_tier"), env=kwargs.get("env"),
            process_chain=None,
        )
        with manager.locked_turn(
            key="ch:translation", provider="codex-cli", model=None,
            runtime_fingerprint=fingerprint,
        ):
            pass
        if label.endswith("retry-offset-1"):
            return {"repairs": [{"block_id": "body", "slots": [
                *_offset_slots(block, "译文。", [2]),
            ]}]}
        return {"blocks": [{"block_id": "body", "text": "译文。"}]}

    result = _generate_translations(
        [{"segment_id": "seg-runtime", "block_ids": ["body"]}],
        options=options, bundle=bundle, glossary={"entries": []}, protected_names=[],
        checkpoint_dir=tmp_path / "checkpoints", llm=stateful_llm,
    )

    assert list(result) == ["seg-runtime"]
    assert labels == [
        "companion-translation-seg-runtime",
        "companion-translation-seg-runtime-retry-offset-1",
    ]

def test_empty_translation_uses_bounded_coverage_repair_and_names_remain_deterministic(
    tmp_path: Path,
) -> None:
    scenarios = {
        "empty": {"blocks": [{"block_id": "body", "text": ""}]},
        "protected-name": {"blocks": [{"block_id": "body", "text": "此处省略姓名。"}]},
    }
    for scenario, generated in scenarios.items():
        block = {
            "block_id": "body", "type": "text", "text": "Ada presents the result.",
            "inline_runs": [
                _inline_run("text", "Ada presents the result.", 1),
            ],
        }
        document = {
            "schema_version": "arc.paper.document.v1",
            "front_matter": {"title": "Validation fixture", "authors": [], "affiliations": []},
            "blocks": [block],
            "equations": [], "figures": [], "tables": [], "bibliography": [], "assets": [],
            "integrity": {"status": "complete", "document_hash": f"fixture-{scenario}"},
        }
        bundle = SourceBundle(
            paper_id="arXiv:1234.5678",
            parsed={"paper_id": "arXiv:1234.5678", "document": document},
            document=document,
            metadata={"title": "Validation fixture"}, references=[], citers=[],
        )
        calls: list[str] = []

        def invalid_llm(prompt: str, **kwargs):
            label = str(kwargs["call_label"])
            calls.append(label)
            if scenario == "empty" and label.endswith("coverage-repair-1"):
                return {"repairs": [{
                    "block_id": "body",
                    "slots": [{
                        "slot_id": pipeline_module._translation_coverage_slot_ids(block)[0],
                        "text": "艾达给出了结果。",
                    }],
                }]}
            return generated

        if scenario == "protected-name":
            result = _generate_translations(
                [{"segment_id": "seg-0001", "block_ids": ["body"]}],
                options=BuildOptions(
                    paper_id=bundle.paper_id, project_dir=tmp_path / scenario, workers=1,
                ),
                bundle=bundle,
                glossary={"entries": []},
                protected_names=["Ada"],
                checkpoint_dir=tmp_path / f"checkpoints-{scenario}",
                llm=invalid_llm,
            )
            assert result["seg-0001"]["blocks"][0]["text"] == "此处省略姓名（Ada）。"
        else:
            result = _generate_translations(
                [{"segment_id": "seg-0001", "block_ids": ["body"]}],
                options=BuildOptions(
                    paper_id=bundle.paper_id, project_dir=tmp_path / scenario, workers=1,
                ),
                bundle=bundle,
                glossary={"entries": []},
                protected_names=["Ada"],
                checkpoint_dir=tmp_path / f"checkpoints-{scenario}",
                llm=invalid_llm,
            )
            assert result["seg-0001"]["blocks"][0]["text"] == "艾达给出了结果（Ada）。"
        assert calls == ["companion-translation-seg-0001"] + (
            ["companion-translation-seg-0001-coverage-repair-1"]
            if scenario == "empty" else []
        )


def test_generation_document_excludes_front_matter_and_all_source_only_sections() -> None:
    document = {
        "front_matter": {
            "title": "Paper Title",
            "authors": ["Ada Author"],
            "affiliations": ["Theory Institute"],
            "abstract": "Abstract remains generative.",
        },
        "blocks": [
            {"block_id": "title", "text": "Paper Title"},
            {"block_id": "author", "text": "Ada Author"},
            {"block_id": "aff", "text": "Theory Institute"},
            {"block_id": "abstract", "text": "Abstract remains generative."},
            {
                "block_id": "toc-title", "kind": "heading", "text": "Contents",
                "html": '<h6 class="ltx_title_contents">Contents</h6>',
            },
            {
                "block_id": "toc-list", "kind": "list", "text": "1 Body",
                "html": '<ol class="ltx_toclist"><li><a href="#S1">1 Body</a></li></ol>',
            },
            {"block_id": "body", "kind": "prose", "section_id": "S1", "text": "Body."},
            {"block_id": "ack-title", "kind": "heading", "section_id": "Sx", "text": "Acknowledgments"},
            {"block_id": "ack-body", "kind": "prose", "section_id": "Sx", "text": "We thank Ada."},
            {"block_id": "refs-title", "kind": "heading", "section_id": "bib", "text": "References"},
            {"block_id": "ref-1", "kind": "bibliography", "section_id": "bib", "text": "[1] Work."},
        ],
    }

    assert [item["block_id"] for item in _generation_document(document)["blocks"]] == ["abstract", "body"]


def test_generation_document_prefers_structural_front_matter_ids_and_roles() -> None:
    document = {
        "parser_version": 4,
        "front_matter": {
            "title": "Metadata spelling may differ",
            "authors": ["Metadata author spelling"],
            "affiliations": ["Metadata affiliation spelling"],
            "block_ids": {
                "title": ["front-title"],
                "authors": ["front-authors"],
                "affiliations": ["front-affiliations"],
            },
        },
        "blocks": [
            {"block_id": "front-title", "kind": "prose", "text": "Source title"},
            {
                "block_id": "front-authors", "kind": "prose", "text": "Source authors",
                "source_role": "front_matter_authors",
            },
            {
                "block_id": "front-affiliations", "kind": "prose", "text": "Combined source affiliations",
                "source_role": "front_matter_affiliations",
            },
            {
                "block_id": "toc", "kind": "list", "text": "1 Body",
                "source_role": "table_of_contents",
            },
            {"block_id": "ack", "kind": "prose", "text": "Thanks", "source_role": "acknowledgments"},
            {"block_id": "ref", "kind": "bibliography", "text": "Reference", "source_role": "references"},
            {"block_id": "body", "kind": "prose", "text": "Generative body"},
        ],
    }

    projected = _generation_document(document)

    assert [block["block_id"] for block in projected["blocks"]] == ["body"]


def test_fingerprint_changes_when_generation_projection_changes_with_same_document_hash(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    options = BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path)
    evidence = {"references": [], "citers": []}
    first = _fingerprint(bundle, options, evidence=evidence)
    changed_document = {
        **bundle.document,
        "blocks": [
            {**block, "source_role": "front_matter_title"} if block["block_id"] == "b2" else block
            for block in bundle.document["blocks"]
        ],
    }
    changed = SourceBundle(
        paper_id=bundle.paper_id,
        parsed=bundle.parsed,
        document=changed_document,
        metadata=bundle.metadata,
        references=bundle.references,
        citers=bundle.citers,
    )

    assert _fingerprint(changed, options, evidence=evidence) != first


def _bundle(tmp_path: Path) -> SourceBundle:
    image = tmp_path / "cached.png"
    image.write_bytes(b"valid-png-fixture")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    document = {
        "schema_version": "arc.paper.document.v1",
        "front_matter": {
            "title": "A&B",
            "authors": [{"name": "Ada"}],
            "affiliations": [{"name": "Institute & Lab"}],
            "abstract": {"text": "An abstract with x & y."},
        },
        "blocks": [
            {"block_id": "b1", "type": "section", "title": "Setup"},
            {"block_id": "b2", "type": "text", "text": "Let x < y & y > 0."},
            {"block_id": "b3", "type": "equation", "equation_id": "eq1"},
            {"block_id": "b4", "type": "figure", "figure_id": "fig1"},
            {"block_id": "b5", "type": "table", "table_id": "tab1"},
            {"block_id": "b6", "type": "bibliography_item", "text": "Original visible reference"},
        ],
        "equations": [{
            "id": "eq1",
            "tex": ["E=mc^2", "p=mv"],
            "printed_equation_numbers": ["(7)", "(8)"],
            "label": "eq:energy",
        }],
        "figures": [{"id": "fig1", "asset_ids": ["a1"], "tag": "Figure 2:", "caption": "A plot"}],
        "tables": [{
            "id": "tab1",
            "tag": "Table 1:",
            "caption": "Values",
            "column_count": 2,
            "rows": [[{"text": "A", "rowspan": 2}, {"text": "B"}], [{"text": "C", "colspan": 1}]],
        }],
        "bibliography": [{"id": "ref1", "label": "[1]", "text": "Original visible reference"}],
        "assets": [{"asset_id": "a1", "cache_path": str(image), "sha256": digest}],
        "integrity": {"status": "complete", "document_hash": "document-fixture"},
    }
    return SourceBundle(
        paper_id="arXiv:1234.5678",
        parsed={"paper_id": "arXiv:1234.5678", "document": document},
        document=document,
        metadata={"title": "Fixture"},
        references=[{"paper_id": "arXiv:0001.0001", "title": "Prior"}],
        citers=[{"paper_id": "arXiv:9999.9999", "title": "Later"}],
    )


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.lock = threading.Lock()
        self.annotation_barrier = threading.Barrier(2)
        self.annotation_started = threading.Event()

    def __call__(self, prompt: str, **kwargs):
        with self.lock:
            self.calls.append({"prompt": prompt, **kwargs, "thread": threading.current_thread().name})
        label = kwargs["call_label"]
        if label.startswith("companion-segmentation-w-"):
            return {"cut_after_ordinals": [3]}
        if label.startswith("companion-glossary-window-"):
            return {"entries": [{
                "source_term": "energy", "target_term": "能量",
                "brief_explanation": "物理系统的能量", "aliases": [],
                "protected_names": [], "first_block_id": "b2",
            }]}
        if label == "companion-glossary-consolidation":
            return {"entries": [{
                "source_term": "energy", "target_term": "能量",
                "brief_explanation": "物理系统的能量", "aliases": [],
                "protected_names": [], "first_block_id": "b2",
            }]}
        if label.startswith("title-translation-"):
            payload = json.loads(prompt[prompt.index("{"):])
            return {"titles": [
                {"title_id": item["title_id"], "text": item["source_text"]}
                for item in payload["titles"]
            ]}
        if label.startswith("companion-translation-"):
            assert self.annotation_started.wait(timeout=5), "translation and commentary lanes did not overlap"
            return {"blocks": [
                {"block_id": "b1", "text": "设定"},
                {"block_id": "b2", "text": "令 x < y 且 y > 0。"},
            ]}
        if label.startswith("companion-annotation-"):
            self.annotation_started.set()
            self.annotation_barrier.wait(timeout=5)
            segment_id = label.rsplit("-", 1)[-1]
            return {
                "explanation": f"解释 {segment_id}", "prior_work": [], "later_work": [],
                "commentary": f"伴读 {segment_id}", "evidence_ids": [],
                "key_points": [], "source_notes": [],
            }
        if label.startswith("companion-section-review-"):
            return {
                "findings": [{"segment_id": "seg-0001", "issue": "check terminology"}],
                "reviewed_segments": [
                    {
                        "segment_id": "seg-0001",
                        "translation": {"blocks": [
                            {"block_id": "b1", "text": "设定"},
                            {"block_id": "b2", "text": "令 x < y 且 y > 0。"},
                        ]},
                        "annotation": {
                            "explanation": "解释 0001", "prior_work": [], "later_work": [],
                            "commentary": "伴读 0001", "evidence_ids": [],
                            "key_points": [], "source_notes": [],
                        },
                    },
                    {
                        "segment_id": "seg-0002", "translation": {"blocks": []},
                        "annotation": {
                            "explanation": "解释 0002", "prior_work": [], "later_work": [],
                            "commentary": "伴读 0002", "evidence_ids": [],
                            "key_points": [], "source_notes": [],
                        },
                    },
                ],
            }
        if label == "companion-final-review":
            return {"patches": [{
                "segment_id": "seg-0001",
                "translation_blocks": None,
                "commentary": "审校后的伴读",
                "explanation": "审校后的解释",
                "prior_work": None,
                "later_work": None,
                "evidence_ids": None,
                "reason": "precision",
            }], "issues": []}
        raise AssertionError(label)


def test_build_fails_typed_and_before_source_or_llm_when_project_is_locked(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "run"
    calls: list[str] = []

    def source_loader(*args, **kwargs):
        calls.append("source")
        raise AssertionError("source loading must not start while the project is locked")

    def llm(*args, **kwargs):
        calls.append("llm")
        raise AssertionError("LLM must not start while the project is locked")

    with ProjectBuildLock(project_dir / ".arc-companion-build.lock"):
        result = build_companion(
            BuildOptions(paper_id="arXiv:0000.00000", project_dir=project_dir),
            source_loader=source_loader,
            llm=llm,
        )

    assert result["ok"] is False
    assert result["error"]["code"] == "build_in_progress"
    assert result["meta"]["retryable"] is True
    assert calls == []


def test_build_uses_tiered_parallel_lanes_and_is_source_faithful_and_resumable(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    fake = FakeLLM()

    def source_loader(*args, **kwargs):
        return bundle

    def compiler(tex_path: Path, pdf_path: Path) -> None:
        assert tex_path.is_file()
        pdf_path.write_bytes(b"%PDF-1.7 fixture")

    result = build_companion(
        BuildOptions(
            paper_id=bundle.paper_id,
            project_dir=tmp_path / "run",
            language_was_defaulted=True,
            workers=12,
            review_context_chars=1,
        ),
        source_loader=source_loader,
        llm=fake,
        compiler=compiler,
        pdf_validator=lambda path: {"bytes": path.stat().st_size},
    )
    assert result["ok"], result
    assert result["meta"]["notice"].startswith("默认使用中文")
    tiers = {str(call["call_label"]): call["model_tier"] for call in fake.calls}
    assert all(tier == "medium" for label, tier in tiers.items() if "segmentation" in label)
    assert all(tier == "medium" for label, tier in tiers.items() if "glossary" in label)
    assert all(tier == "high" for label, tier in tiers.items() if "annotation" in label)
    assert all(tier == "medium" for label, tier in tiers.items() if "review" in label)
    assert all(tier == "medium" for label, tier in tiers.items() if "translation" in label)
    assert all(call["session_policy"] == "stateless" for call in fake.calls)
    externally_enabled = [
        call for call in fake.calls
        if str(call["call_label"]).startswith(("companion-translation-", "companion-annotation-"))
    ]
    assert externally_enabled
    for call in externally_enabled:
        env = call["env"]
        expected_internet = "false" if "translation" in str(call["call_label"]) else "true"
        assert env["ARC_CODEX_ALLOW_INTERNET"] == expected_internet
        assert env["ARC_CLAUDE_ALLOW_INTERNET"] == expected_internet
        assert env["ARC_CODEX_ENABLE_MCP"] == "false"
        assert env["ARC_CLAUDE_ALLOW_MCP"] == "false"
        assert "ARC_CODEX_MCP_MODE" not in env
        assert "ARC_CLAUDE_MCP_MODE" not in env
        assert "FULL-PAPER NAVIGATION CONTEXT" in str(call["prompt"])
        assert "Setup" not in str(call["prompt"])
    for call in fake.calls:
        if call in externally_enabled:
            continue
        env = call["env"]
        assert env["ARC_CODEX_ALLOW_INTERNET"] == "false"
        assert env["ARC_CLAUDE_ALLOW_INTERNET"] == "false"
        assert env["ARC_CODEX_ENABLE_MCP"] == "false"
        assert env["ARC_CLAUDE_ALLOW_MCP"] == "false"
    annotation_calls = [call for call in fake.calls if str(call["call_label"]).startswith("companion-annotation-")]
    assert len(annotation_calls) == 2
    assert len({call["thread"] for call in annotation_calls}) == 2
    assert any(str(call["call_label"]).startswith("companion-section-review-") for call in fake.calls)
    section_prompts = [str(call["prompt"]) for call in fake.calls if str(call["call_label"]).startswith("companion-section-review-")]
    assert section_prompts and all('"source_blocks"' in prompt for prompt in section_prompts)
    final_prompt = next(str(call["prompt"]) for call in fake.calls if call["call_label"] == "companion-final-review")
    assert '"section_reviews"' in final_prompt
    assert '"reviewed_segments"' not in final_prompt
    assert '"reviewed_segment_ids"' in final_prompt
    assert '"patch_proposals"' in final_prompt
    assert '"source_anchors"' in final_prompt
    assert '"source_excerpt"' in final_prompt
    assert '"segment_id": "seg-0001"' in final_prompt
    assert '"segment_id": "seg-0002"' in final_prompt

    data = result["data"]
    tex = Path(data["output_tex"]).read_text(encoding="utf-8")
    assert r"\tag{7}" in tex
    assert r"\tag{8}" in tex
    assert "E=mc^2" in tex
    assert "p=mv" in tex
    assert "['E=mc^2'" not in tex
    assert r"\multirow{2}{*}{A}" in tex
    assert "Figure 2: A plot" in tex
    assert "Figure Figure" not in tex
    assert "Institute \\& Lab" in tex
    assert "An abstract with x \\& y." in tex
    assert "Original visible reference" in tex
    assert "审校后的解释" in tex
    assert "解释 0002" in tex
    assert Path(data["output_pdf"]).is_file()
    run_pdf = Path(data["output_run_pdf"])
    assert run_pdf.parent == (tmp_path / "run").resolve()
    assert run_pdf.read_bytes() == Path(data["output_pdf"]).read_bytes()
    assert data["output_run_pdf_sha256"] == data["output_pdf_sha256"]
    assert not ((tmp_path / "run").parent / run_pdf.name).exists()
    assert data["web_render_version"] == "arc.companion.web-render.v4"
    assert Path(data["output_html"]).is_file()
    assert Path(data["reader_snapshot_path"]).is_file()
    assert Path(data["web_manifest_path"]).is_file()
    reader_snapshot = json.loads(Path(data["reader_snapshot_path"]).read_text())
    assert reader_snapshot["language"] == "zh-CN"
    validation = validate_project(
        tmp_path / "run", pdf_validator=lambda path: {"bytes": path.stat().st_size}
    )
    assert validation["ok"]
    assert validation["data"]["web"]["ok"] is True
    saved_evidence = list((Path(data["checkpoint_dir"]) / "segment-evidence").glob("*.json"))
    assert len(saved_evidence) == 2
    assert all("evidence" in json.loads(path.read_text(encoding="utf-8")) for path in saved_evidence)

    call_count = len(fake.calls)
    # Missing user-facing outputs are repaired without model work.
    Path(data["output_html"]).unlink()
    run_pdf.unlink()
    resumed = build_companion(
        BuildOptions(
            paper_id=bundle.paper_id,
            project_dir=tmp_path / "run",
            language_was_defaulted=True,
            workers=12,
            review_context_chars=1,
        ),
        source_loader=source_loader,
        llm=fake,
        compiler=compiler,
        pdf_validator=lambda path: {},
    )
    assert resumed["ok"] and resumed["meta"]["resumed"] is True
    assert len(fake.calls) == call_count
    assert Path(resumed["data"]["output_html"]).is_file()
    assert Path(resumed["data"]["output_run_pdf"]).read_bytes() == Path(
        resumed["data"]["output_pdf"]
    ).read_bytes()

    state_path = tmp_path / "run" / "state.json"
    stale_preview_state = json.loads(state_path.read_text(encoding="utf-8"))
    stale_preview_state.pop("first_wave_preview_version")
    state_path.write_text(json.dumps(stale_preview_state), encoding="utf-8")
    regenerated_preview = build_companion(
        BuildOptions(
            paper_id=bundle.paper_id,
            project_dir=tmp_path / "run",
            language_was_defaulted=True,
            workers=12,
            review_context_chars=1,
        ),
        source_loader=source_loader,
        llm=fake,
        compiler=compiler,
        pdf_validator=lambda path: {},
    )
    assert regenerated_preview["ok"] and regenerated_preview["meta"]["resumed"] is False
    assert regenerated_preview["data"]["first_wave_preview_version"]
    assert len(fake.calls) == call_count


def test_build_does_not_construct_an_evidence_controller(monkeypatch, tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    fake = FakeLLM()
    monkeypatch.setattr(pipeline_module, "load_domain_context", lambda **_kwargs: {
        "schema_version": "arc.companion.domain-context.v1",
        "paper_ids": ["arXiv:1111.1111", "arXiv:2222.2222"],
        "domains": [],
    })

    result = build_companion(
        BuildOptions(
            paper_id=bundle.paper_id, project_dir=tmp_path / "controller-context",
            domain_id="provided-domain",
        ),
        source_loader=lambda *args, **kwargs: bundle,
        llm=fake,
        compiler=lambda _tex, pdf: pdf.write_bytes(b"%PDF-1.7 fixture"),
        pdf_validator=lambda path: {"bytes": path.stat().st_size},
    )

    assert result["ok"], result
    assert not hasattr(pipeline_module, "EvidenceRequestController")


def test_legacy_string_annotation_checkpoint_is_rerun_and_upgraded(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    fake = FakeLLM()
    fake.annotation_barrier = threading.Barrier(1)
    segment = {
        "segment_id": "seg-0001", "block_ids": ["b1", "b2"],
        "start_block_id": "b1", "end_block_id": "b2",
    }
    checkpoint_dir = tmp_path / "legacy-checkpoint"
    kwargs = dict(
        options=BuildOptions(bundle.paper_id, tmp_path, workers=1), bundle=bundle,
        evidence={"related_papers": []}, domain_context=None, glossary={"entries": []},
        protected_names=[], checkpoint_dir=checkpoint_dir, llm=fake,
    )

    first = pipeline_module._generate_annotations([segment], **kwargs)
    checkpoint_path = (
        checkpoint_dir / "annotations"
        / f"{_segment_checkpoint_name('seg-0001')}.json"
    )
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["schema_version"] = "arc.companion.annotation-checkpoint.v3"
    checkpoint["annotation"]["prior_work"] = "legacy unbound prose"
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    calls_before = len([
        call for call in fake.calls
        if str(call["call_label"]).startswith("companion-annotation-")
    ])

    second = pipeline_module._generate_annotations([segment], **kwargs)

    calls_after = len([
        call for call in fake.calls
        if str(call["call_label"]).startswith("companion-annotation-")
    ])
    upgraded = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert first["seg-0001"]["prior_work"] == []
    assert second["seg-0001"]["prior_work"] == []
    assert calls_after == calls_before + 1
    assert upgraded["schema_version"] == pipeline_module.ANNOTATION_CHECKPOINT_VERSION


def test_replacement_generation_isolates_commentary_checkpoint_and_provider_artifacts(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    segment = {
        "segment_id": "seg-generation", "block_ids": ["b1", "b2"],
        "start_block_id": "b1", "end_block_id": "b2",
    }
    checkpoint_dir = tmp_path / "checkpoint"
    artifact_dirs: list[Path] = []

    def llm(_prompt: str, **kwargs):
        artifact_dirs.append(Path(kwargs["artifact_dir"]))
        return {
            "explanation": "解释", "prior_work": [], "later_work": [],
            "commentary": "伴读", "evidence_ids": [], "key_points": [],
            "source_notes": [],
        }

    kwargs = dict(
        options=BuildOptions(bundle.paper_id, tmp_path, workers=1), bundle=bundle,
        evidence={"related_papers": []}, domain_context=None, glossary={"entries": []},
        protected_names=[], checkpoint_dir=checkpoint_dir, llm=llm,
    )
    pipeline_module._generate_annotations([segment], generation=1, **kwargs)
    pipeline_module._generate_annotations([segment], generation=2, **kwargs)

    name = f"{_segment_checkpoint_name(segment['segment_id'])}.json"
    assert (checkpoint_dir / "annotations" / name).is_file()
    assert (checkpoint_dir / "annotations" / "generation-2" / name).is_file()
    assert artifact_dirs == [
        checkpoint_dir / "llm" / "annotations" / name.removesuffix(".json"),
        checkpoint_dir / "llm" / "annotations" / "generation-2"
        / name.removesuffix(".json"),
    ]


def test_replacement_generation_does_not_read_translation_generation_one_artifacts(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    segment = {
        "segment_id": "seg-generation", "block_ids": ["b1", "b2"],
        "start_block_id": "b1", "end_block_id": "b2",
    }
    checkpoint_dir = tmp_path / "checkpoint"
    calls: list[tuple[str, Path]] = []

    def llm(_prompt: str, **kwargs):
        calls.append((str(kwargs["call_label"]), Path(kwargs["artifact_dir"])))
        return {"blocks": [
            {"block_id": "b1", "text": "设定"},
            {"block_id": "b2", "text": "令 x 小于 y 且 y 大于零。"},
        ]}

    kwargs = dict(
        options=BuildOptions(bundle.paper_id, tmp_path, workers=1), bundle=bundle,
        glossary={"entries": []}, protected_names=[], checkpoint_dir=checkpoint_dir,
        llm=llm,
    )
    pipeline_module._generate_translations([segment], generation=1, **kwargs)
    pipeline_module._generate_translations([segment], generation=2, **kwargs)

    name = f"{_segment_checkpoint_name(segment['segment_id'])}.json"
    assert len(calls) == 2
    assert calls[1][1] == (
        checkpoint_dir / "llm" / "translations" / "generation-2"
        / name.removesuffix(".json")
    )
    assert (checkpoint_dir / "translation-drafts" / name).is_file()
    assert (checkpoint_dir / "translation-drafts" / "generation-2" / name).is_file()
    assert (checkpoint_dir / "translations" / name).is_file()
    assert (checkpoint_dir / "translations" / "generation-2" / name).is_file()


def test_generationless_artifacts_use_persisted_nonfirst_generation_owner(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    segment_id = "seg-owned"
    pipeline_module._record_legacy_generation_owners(
        checkpoint_dir,
        lane="translation",
        segment_ids=[segment_id],
        generation=3,
    )
    pipeline_module._record_legacy_generation_owners(
        checkpoint_dir,
        lane="companion",
        segment_ids=[segment_id],
        generation=3,
    )

    for artifact_name in (
        "translations", "translation-drafts",
        "translation-coverage-attempts",
        "translation-token-offset-attempts", "translation-token-attempts",
        "translation-token-offset-repair-drafts", "llm/translations",
        "annotations", "llm/annotations",
    ):
        assert pipeline_module._generation_segment_artifact_dir(
            checkpoint_dir, artifact_name, segment_id, 3,
        ) == checkpoint_dir / artifact_name
        assert pipeline_module._generation_segment_artifact_dir(
            checkpoint_dir, artifact_name, segment_id, 4,
        ) == checkpoint_dir / artifact_name / "generation-4"
        # Legacy payloads did not carry an explicit generation.  The persisted
        # path owner supplies it instead of silently treating them as gen 1.
        assert pipeline_module._artifact_payload_generation(
            {}, checkpoint_dir, artifact_name, segment_id,
        ) == 3

    owner = json.loads(
        (checkpoint_dir / "legacy-generation-owners.json").read_text(
            encoding="utf-8"
        )
    )
    assert owner["schema_version"] == (
        pipeline_module.LEGACY_GENERATION_OWNERS_SCHEMA_VERSION
    )

    coverage_path = pipeline_module._translation_coverage_attempt_path(
        checkpoint_dir, segment_id, 3,
    )
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    coverage_path.write_text(json.dumps({
        "schema_version": pipeline_module.TRANSLATION_COVERAGE_ATTEMPT_SCHEMA_VERSION,
        "prompt_version": pipeline_module.TRANSLATION_COVERAGE_REPAIR_PROMPT_VERSION,
        "response_schema_version": (
            pipeline_module.TRANSLATION_COVERAGE_REPAIR_SCHEMA_VERSION
        ),
        "segment_id": segment_id,
        "input_sha256": "input",
    }), encoding="utf-8")
    assert pipeline_module._matching_translation_coverage_attempt(
        checkpoint_dir, segment_id, "input", 3,
    ) is not None
    assert pipeline_module._matching_translation_coverage_attempt(
        checkpoint_dir, segment_id, "input", 1,
    ) is None


def test_generationless_generation_three_translation_and_commentary_replay(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    checkpoint_dir = tmp_path / "checkpoint"
    segment = {
        "segment_id": "seg-owned-replay", "block_ids": ["b1", "b2"],
        "start_block_id": "b1", "end_block_id": "b2",
    }
    segment_id = str(segment["segment_id"])
    pipeline_module._record_legacy_generation_owners(
        checkpoint_dir, lane="translation", segment_ids=[segment_id], generation=3,
    )
    pipeline_module._record_legacy_generation_owners(
        checkpoint_dir, lane="companion", segment_ids=[segment_id], generation=3,
    )
    translation_calls = 0
    commentary_calls = 0
    provider_artifact_dirs: list[Path] = []

    def translation_llm(_prompt: str, **kwargs):
        nonlocal translation_calls
        translation_calls += 1
        provider_artifact_dirs.append(Path(kwargs["artifact_dir"]))
        return {"blocks": [
            {"block_id": "b1", "text": "设定"},
            {"block_id": "b2", "text": "令 x 小于 y 且 y 大于零。"},
        ]}

    def commentary_llm(_prompt: str, **kwargs):
        nonlocal commentary_calls
        commentary_calls += 1
        provider_artifact_dirs.append(Path(kwargs["artifact_dir"]))
        return {
            "explanation": "解释", "prior_work": [], "later_work": [],
            "commentary": "伴读", "evidence_ids": [], "key_points": [],
            "source_notes": [],
        }

    options = BuildOptions(bundle.paper_id, tmp_path, workers=1)
    pipeline_module._generate_translations(
        [segment], options=options, bundle=bundle, glossary={"entries": []},
        protected_names=[], checkpoint_dir=checkpoint_dir,
        llm=translation_llm, generation=3,
    )
    pipeline_module._generate_annotations(
        [segment], options=options, bundle=bundle,
        evidence={"related_papers": []}, domain_context=None,
        glossary={"entries": []}, protected_names=[],
        checkpoint_dir=checkpoint_dir, llm=commentary_llm, generation=3,
    )
    name = f"{_segment_checkpoint_name(segment_id)}.json"
    translation_path = checkpoint_dir / "translations" / name
    annotation_path = checkpoint_dir / "annotations" / name
    for path in (translation_path, annotation_path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.pop("generation", None)
        path.write_text(json.dumps(payload), encoding="utf-8")

    pipeline_module._generate_translations(
        [segment], options=options, bundle=bundle, glossary={"entries": []},
        protected_names=[], checkpoint_dir=checkpoint_dir,
        llm=translation_llm, generation=3,
    )
    pipeline_module._generate_annotations(
        [segment], options=options, bundle=bundle,
        evidence={"related_papers": []}, domain_context=None,
        glossary={"entries": []}, protected_names=[],
        checkpoint_dir=checkpoint_dir, llm=commentary_llm, generation=3,
    )

    assert translation_calls == 1
    assert commentary_calls == 1
    assert provider_artifact_dirs == [
        checkpoint_dir / "llm" / "translations" / name.removesuffix(".json"),
        checkpoint_dir / "llm" / "annotations" / name.removesuffix(".json"),
    ]

    repair_path = pipeline_module._translation_token_repair_draft_path(
        checkpoint_dir, segment_id, 3,
    )
    repair_path.parent.mkdir(parents=True, exist_ok=True)
    repair_path.write_text(json.dumps({
        "schema_version": "arc.companion.translation-token-repair-draft.v1",
        "prompt_version": pipeline_module.TRANSLATION_RETRY_PROMPT_VERSION,
        "response_schema_version": pipeline_module.TRANSLATION_SLOT_REPAIR_SCHEMA_VERSION,
        "segment_id": segment_id, "input_sha256": "repair-input",
        "translation": {"blocks": []},
        "repair_provenance": {"repair_mode": "offset-only"},
        "raw_response": {},
    }), encoding="utf-8")
    assert pipeline_module._matching_translation_token_repair_draft(
        checkpoint_dir, segment_id, "repair-input", 3,
    ) is not None


@pytest.mark.parametrize(
    ("returned_url", "accepted", "cache_only"),
    [
        ("https://cache.example/paper", True, False),
        ("https://cache.example/descriptor", True, True),
        ("https://fabricated.example/paper", False, False),
    ],
)
def test_offline_annotation_accepts_only_bounded_cache_urls(
    tmp_path: Path, monkeypatch, returned_url: str, accepted: bool, cache_only: bool,
) -> None:
    bundle = _bundle(tmp_path)
    segment = {
        "segment_id": "seg-offline", "block_ids": ["b1", "b2"],
        "start_block_id": "b1", "end_block_id": "b2",
    }
    monkeypatch.setattr(
        pipeline_module, "_evidence_for_segment",
        lambda *args, **kwargs: {
            "schema_version": "arc.companion.segment-evidence.v3",
            "bounded_sources": [] if cache_only else [{
                "title": "Cached paper",
                "url": "https://cache.example/paper",
                "locator": "Abstract",
            }],
            "papers": ([{
                "source_descriptor": {
                    "provider": "arc-paper",
                    "canonical_locator": "https://cache.example/descriptor",
                },
            }] if cache_only else []),
        },
    )

    calls: list[str] = []

    def llm(_prompt: str, **_kwargs):
        calls.append(returned_url)
        return {
            "explanation": "Cached fact.", "commentary": "",
            "commentary_sources": [{
                "title": "Cached paper", "url": returned_url, "locator": "Abstract",
            }],
            "prior_work": [], "later_work": [],
        }

    call = lambda: pipeline_module._generate_annotations(
        [segment],
        options=BuildOptions(
            bundle.paper_id, tmp_path / returned_url.split("//", 1)[-1],
            workers=1, allow_internet=False,
        ),
        bundle=bundle, evidence={"related_papers": []}, domain_context=None,
        glossary={"entries": []}, protected_names=[],
        checkpoint_dir=tmp_path / ("accepted" if accepted else "rejected"), llm=llm,
    )
    if accepted:
        assert call()["seg-offline"]["commentary_sources"][0]["url"] == returned_url
        assert call()["seg-offline"]["commentary_sources"][0]["url"] == returned_url
        assert calls == [returned_url]
    else:
        with pytest.raises(CompanionLaneError, match="not supplied by the prompt or ARC cache"):
            call()


def test_annotation_prompt_projects_relevant_glossary_and_stays_below_transport_limit() -> None:
    segment = {
        "segment_id": "seg-0001", "block_ids": ["b1"], "title": "Gauge symmetry",
        "start_block_id": "b1", "end_block_id": "b1",
    }
    blocks = [{
        "block_id": "b1", "type": "text",
        "text": "Gauge symmetry constrains the vacuum while SOURCE-SENTINEL remains exact.",
    }]
    relevant_entries = [
        {
            "source_term": "gauge symmetry", "target_term": "规范对称性",
            "aliases": [], "brief_explanation": "relevant gauge entry",
            "first_block_id": "b1", "protected_names": [],
        },
        {
            "source_term": "vacuum", "target_term": "真空",
            "aliases": [], "brief_explanation": "relevant vacuum entry",
            "first_block_id": "elsewhere", "protected_names": [],
        },
    ]
    glossary = {
        "schema_version": "arc.companion.glossary.test",
        "entries": [
            *relevant_entries,
            *[
                {
                    "source_term": f"unrelated-term-{index}",
                    "target_term": f"无关术语{index}", "aliases": [],
                    "brief_explanation": "x" * 500,
                    "first_block_id": f"other-{index}", "protected_names": [],
                }
                for index in range(100)
            ],
        ],
    }
    paper_context = {
        "schema_version": "paper-context", "paper_id": "paper", "abstract": "a" * 4_000,
        "current_segment": {"segment_id": "seg-0001"},
        "section_navigation": [
            {"block_id": f"h{index}", "title": f"Heading {index}", "anchor": "n" * 500}
            for index in range(100)
        ],
        "neighboring_source_anchors": [], "access": {"allow_internet": False},
    }
    evidence = {
        "schema_version": "evidence", "papers": [{"evidence_id": "EVIDENCE-SENTINEL"}],
        "citation_targets": [], "reference_catalog": [], "citer_catalog": [],
    }
    prompt = pipeline_module._bounded_annotation_prompt(
        segment, blocks, language="zh-CN", metadata={}, evidence=evidence,
        glossary=glossary, protected_names=[], paper_context=paper_context,
        domain_context=None,
    )

    assert len(prompt.encode("utf-8")) < pipeline_module.ANNOTATION_PROMPT_MAX_BYTES
    assert "SOURCE-SENTINEL" in prompt
    assert "EVIDENCE-SENTINEL" in prompt
    assert "relevant gauge entry" in prompt
    assert "relevant vacuum entry" in prompt
    assert "unrelated-term-99" not in prompt


def test_annotation_prompt_fails_closed_when_required_source_exceeds_transport_limit() -> None:
    segment = {"segment_id": "seg-large", "block_ids": ["b-large"], "title": "Large"}
    blocks = [{"block_id": "b-large", "type": "text", "text": "x" * 70_000}]
    paper_context = {
        "section_navigation": [], "neighboring_source_anchors": [], "abstract": "",
        "current_segment": {"segment_id": "seg-large"},
    }

    with pytest.raises(RuntimeError, match="cannot be bounded"):
        pipeline_module._bounded_annotation_prompt(
            segment, blocks, language="zh-CN", metadata={},
            evidence={"papers": []}, glossary={"entries": []}, protected_names=[],
            paper_context=paper_context, domain_context=None,
        )


def test_annotation_transport_projection_does_not_invalidate_semantic_checkpoint(
    monkeypatch, tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    fake = FakeLLM()
    fake.annotation_barrier = threading.Barrier(1)
    segment = {
        "segment_id": "seg-0001", "block_ids": ["b1", "b2"],
        "start_block_id": "b1", "end_block_id": "b2",
    }
    kwargs = dict(
        options=BuildOptions(bundle.paper_id, tmp_path, workers=1), bundle=bundle,
        evidence={"related_papers": []}, domain_context=None,
        glossary={"entries": [{
            "source_term": "source", "target_term": "源", "aliases": [],
            "brief_explanation": "term", "first_block_id": "b1", "protected_names": [],
        }]},
        protected_names=[], checkpoint_dir=tmp_path / "projection-cache", llm=fake,
    )

    first = pipeline_module._generate_annotations([segment], **kwargs)
    annotation_calls = len([
        call for call in fake.calls
        if str(call["call_label"]).startswith("companion-annotation-")
    ])
    monkeypatch.setattr(
        pipeline_module, "_bounded_annotation_prompt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("cache was not reused")),
    )

    second = pipeline_module._generate_annotations([segment], **kwargs)

    assert second == first
    assert len([
        call for call in fake.calls
        if str(call["call_label"]).startswith("companion-annotation-")
    ]) == annotation_calls


def test_section_review_chunks_use_rendered_utf8_bytes_and_preserve_order() -> None:
    items = [
        {
            "segment": {"segment_id": f"seg-{index:04d}"},
            "source_blocks": [{"block_id": f"b{index}", "text": "源" * 200}],
            "translation": {"blocks": [{"block_id": f"b{index}", "text": "译" * 200}]},
            "annotation": {"commentary": "注" * 100},
            "context_evidence": [],
        }
        for index in range(4)
    ]
    exact_pair_bytes = len(pipeline_module.section_review_prompt(
        {"segments": items[:2]}, language="zh-CN",
    ).encode("utf-8"))

    strict_chunks = pipeline_module._review_chunks(
        items, language="zh-CN", max_prompt_bytes=exact_pair_bytes,
    )
    paired_chunks = pipeline_module._review_chunks(
        items, language="zh-CN", max_prompt_bytes=exact_pair_bytes + 1,
    )

    assert [len(chunk) for chunk in strict_chunks] == [2, 2]
    assert [len(chunk) for chunk in paired_chunks] == [2, 2]
    assert [item for chunk in paired_chunks for item in chunk] == items
    assert all(
        len(pipeline_module.section_review_prompt(
            {"segments": chunk}, language="zh-CN",
        ).encode("utf-8")) < exact_pair_bytes + 1
        for chunk in paired_chunks
    )


def test_section_review_chunks_fail_closed_when_one_segment_exceeds_limit() -> None:
    item = {
        "segment": {"segment_id": "seg-large"},
        "source_blocks": [{"block_id": "b-large", "text": "源" * 1_000}],
        "translation": {"blocks": []}, "annotation": {}, "context_evidence": [],
    }
    single_size = len(pipeline_module.section_review_prompt(
        {"segments": [item]}, language="zh-CN",
    ).encode("utf-8"))

    with pytest.raises(RuntimeError, match=r"seg-large.*exceeding"):
        pipeline_module._review_chunks(
            [item], language="zh-CN", max_prompt_bytes=single_size - 1,
        )


def test_review_prompt_budget_keeps_ten_percent_transport_headroom(tmp_path: Path) -> None:
    options = BuildOptions("local:budget", tmp_path, review_context_chars=140_000)
    budget = pipeline_module._review_prompt_budget(options)

    assert budget["strict_limit_bytes"] == 60 * 1024
    assert budget["target_limit_bytes"] == 55_296
    assert budget["target_ratio"] == {"numerator": 9, "denominator": 10}

    floor_budget = pipeline_module._review_prompt_budget(
        BuildOptions("local:budget-floor", tmp_path, review_context_chars=1)
    )
    assert floor_budget["strict_limit_bytes"] == 32 * 1024
    assert floor_budget["target_limit_bytes"] == 29_491


def test_rendered_review_packer_allows_only_singleton_to_use_headroom() -> None:
    items = [
        {"segment": {"segment_id": "s1"}},
        {"segment": {"segment_id": "s2"}},
    ]

    calls = pipeline_module._pack_rendered_review_calls(
        items,
        render_prompt=lambda group: "x" * (56 if len(group) == 1 else 90),
        target_prompt_bytes=50,
        strict_prompt_bytes=60,
        label="test review",
    )

    assert [call["segment_ids"] for call in calls] == [["s1"], ["s2"]]
    assert all(call["budget_class"] == "singleton_headroom" for call in calls)
    assert all(call["prompt_bytes"] == 56 for call in calls)

    with pytest.raises(RuntimeError, match=r"s1.*strict 55-byte limit"):
        pipeline_module._pack_rendered_review_calls(
            items[:1],
            render_prompt=lambda _group: "x" * 56,
            target_prompt_bytes=50,
            strict_prompt_bytes=55,
            label="test review",
        )


def test_section_review_preflights_guided_prompts_before_provider_calls(
    monkeypatch, tmp_path: Path,
) -> None:
    segments, translations, annotations, document = _minimal_section_review_inputs()
    provider_calls: list[str] = []

    def oversized_guidance(prompt: str, _guidance, *, lane=None) -> str:
        assert lane == "review"
        return prompt + ("G" * pipeline_module.REVIEW_PROMPT_MAX_BYTES)

    monkeypatch.setattr(pipeline_module, "_guided_prompt", oversized_guidance)

    with pytest.raises(RuntimeError, match=r"section review segment seg-0.*exceeding"):
        _review(
            segments,
            translations,
            annotations,
            document=document,
            glossary={"entries": []},
            protected_names=[],
            evidence={"related_papers": []},
            options=BuildOptions(
                paper_id="local:preflight", project_dir=tmp_path, workers=2,
                review_context_chars=1,
            ),
            llm=lambda _prompt, **kwargs: provider_calls.append(str(kwargs["call_label"])),
            checkpoint_dir=tmp_path / "checkpoints",
            intent_guidance={"guidance": "present"},
        )

    assert provider_calls == []


def _minimal_section_review_inputs() -> tuple[
    list[dict[str, object]],
    dict[str, dict[str, object]],
    dict[str, dict[str, object]],
    dict[str, object],
]:
    blocks = [
        {"block_id": f"b{index}", "type": "text", "text": f"source {index}"}
        for index in range(2)
    ]
    segments = [
        {"segment_id": f"seg-{index}", "block_ids": [f"b{index}"], "title": "Body"}
        for index in range(2)
    ]
    translations = {
        str(segment["segment_id"]): {
            "blocks": [{"block_id": segment["block_ids"][0], "text": "译文"}]
        }
        for segment in segments
    }
    annotations = {
        str(segment["segment_id"]): {
            "commentary": "伴读", "explanation": "解释", "prior_work": [],
            "later_work": [], "commentary_sources": [],
        }
        for segment in segments
    }
    document = {
        "front_matter": {}, "blocks": blocks, "equations": [], "figures": [],
        "tables": [], "bibliography": [], "assets": [],
    }
    return segments, translations, annotations, document


@pytest.mark.parametrize("invalid_mode", ["missing", "duplicate"])
def test_section_review_checkpoint_requires_exact_coverage_before_reuse(
    invalid_mode: str,
    tmp_path: Path,
) -> None:
    segments, translations, annotations, document = _minimal_section_review_inputs()
    section_calls: list[str] = []

    def llm(prompt: str, **kwargs):
        label = str(kwargs["call_label"])
        if label.startswith("companion-section-review-"):
            section_calls.append(label)
            portion = json.loads(prompt.split("PORTION:\n", 1)[1])
            return {
                "findings": [],
                "reviewed_segment_ids": [
                    item["segment"]["segment_id"] for item in portion["segments"]
                ],
                "patches": [],
            }
        return {"patches": [], "issues": []}

    checkpoint_dir = tmp_path / "checkpoints"
    kwargs = dict(
        document=document, glossary={"entries": []}, protected_names=[],
        evidence={"related_papers": []},
        options=BuildOptions(
            paper_id="arXiv:1234.5678", project_dir=tmp_path, workers=1,
            review_context_chars=1,
        ),
        llm=llm, checkpoint_dir=checkpoint_dir,
    )
    _, _, first_review = _review(segments, translations, annotations, **kwargs)
    assert len(section_calls) == 1
    prompt_audit = first_review["prompt_budget_audit"]
    assert prompt_audit["schema_version"] == (
        "arc.companion.review-prompt-budget-audit.v1"
    )
    assert prompt_audit["routing"]["mode"] == "hierarchical"
    assert [item["stage"] for item in prompt_audit["calls"]] == [
        "section", "hierarchical-final",
    ]
    assert all(
        item["prompt_bytes"] <= prompt_audit["budget"]["strict_limit_bytes"]
        for item in prompt_audit["calls"]
    )
    assert prompt_audit["calls"][0]["disposition"] == "provider-call"

    _, _, reused_review = _review(segments, translations, annotations, **kwargs)
    assert len(section_calls) == 1
    assert reused_review["prompt_budget_audit"]["calls"][0]["disposition"] == (
        "checkpoint-reuse"
    )

    checkpoint_path = checkpoint_dir / "section-reviews" / "0000.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    reviewed_segments = checkpoint["review"]["reviewed_segment_ids"]
    sparse = {
        "segment_id": reviewed_segments[0],
        "translation_blocks": None,
        "commentary": None,
        "explanation": None,
        "commentary_sources": None,
        "prior_work": None,
        "later_work": None,
    }
    checkpoint["review"]["patches"] = [
        {**sparse, "commentary": "局部伴读修订", "reason": "commentary"},
        {**sparse, "explanation": "局部解释修订", "reason": "explanation"},
    ]
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    _review(segments, translations, annotations, **kwargs)

    assert len(section_calls) == 1
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert len(checkpoint["review"]["patches"]) == 1
    assert checkpoint["review"]["patches"][0]["commentary"] == "局部伴读修订"
    assert checkpoint["review"]["patches"][0]["explanation"] == "局部解释修订"

    if invalid_mode == "missing":
        checkpoint["review"]["reviewed_segment_ids"] = reviewed_segments[:1]
    else:
        checkpoint["review"]["reviewed_segment_ids"] = [
            reviewed_segments[0], reviewed_segments[0],
        ]
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    _review(segments, translations, annotations, **kwargs)

    assert len(section_calls) == 2
    repaired = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert set(repaired["review"]["reviewed_segment_ids"]) == {"seg-0", "seg-1"}


def test_incomplete_new_section_review_is_not_cached(tmp_path: Path) -> None:
    segments, translations, annotations, document = _minimal_section_review_inputs()

    def llm(prompt: str, **kwargs):
        label = str(kwargs["call_label"])
        if label.startswith("companion-section-review-"):
            portion = json.loads(prompt.split("PORTION:\n", 1)[1])
            item = portion["segments"][0]
            return {
                "findings": [],
                "reviewed_segment_ids": [item["segment"]["segment_id"]],
                "patches": [],
            }
        raise AssertionError("final review must not run after incomplete section coverage")

    checkpoint_dir = tmp_path / "checkpoints"
    with pytest.raises(RuntimeError, match="did not cover every segment"):
        _review(
            segments, translations, annotations,
            document=document, glossary={"entries": []}, protected_names=[],
            evidence={"related_papers": []},
            options=BuildOptions(
                paper_id="arXiv:1234.5678", project_dir=tmp_path, workers=1,
                review_context_chars=1,
            ),
            llm=llm, checkpoint_dir=checkpoint_dir,
        )

    assert not (checkpoint_dir / "section-reviews" / "0000.json").exists()


@pytest.mark.parametrize("cached_failure", ["conflict", "malformed"])
def test_invalid_cached_section_review_patch_reruns_provider(
    cached_failure: str,
    tmp_path: Path,
) -> None:
    segments, translations, annotations, document = _minimal_section_review_inputs()
    section_calls: list[str] = []

    def llm(prompt: str, **kwargs):
        label = str(kwargs["call_label"])
        if label.startswith("companion-section-review-"):
            section_calls.append(label)
            portion = json.loads(prompt.split("PORTION:\n", 1)[1])
            return {
                "findings": [],
                "reviewed_segment_ids": [
                    item["segment"]["segment_id"] for item in portion["segments"]
                ],
                "patches": [],
            }
        return {"patches": [], "issues": []}

    checkpoint_dir = tmp_path / "checkpoints"
    kwargs = dict(
        document=document, glossary={"entries": []}, protected_names=[],
        evidence={"related_papers": []},
        options=BuildOptions(
            paper_id="arXiv:1234.5678", project_dir=tmp_path, workers=1,
            review_context_chars=1,
        ),
        llm=llm, checkpoint_dir=checkpoint_dir,
    )
    _review(segments, translations, annotations, **kwargs)
    assert len(section_calls) == 1

    checkpoint_path = checkpoint_dir / "section-reviews" / "0000.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    segment_id = checkpoint["review"]["reviewed_segment_ids"][0]
    empty = {
        "segment_id": segment_id,
        "translation_blocks": None,
        "commentary": None,
        "explanation": None,
        "commentary_sources": None,
        "prior_work": None,
        "later_work": None,
    }
    if cached_failure == "conflict":
        checkpoint["review"]["patches"] = [
            {**empty, "commentary": "版本一", "reason": "first"},
            {**empty, "commentary": "版本二", "reason": "second"},
        ]
    else:
        checkpoint["review"]["patches"] = [{
            **empty,
            "translation_blocks": "not-a-list",
            "reason": "malformed",
        }]
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    _review(segments, translations, annotations, **kwargs)

    assert len(section_calls) == 2
    repaired = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert repaired["review"]["patches"] == []


def test_new_section_review_patch_conflict_fails_closed(tmp_path: Path) -> None:
    segments, translations, annotations, document = _minimal_section_review_inputs()

    def llm(prompt: str, **kwargs):
        label = str(kwargs["call_label"])
        if not label.startswith("companion-section-review-"):
            raise AssertionError("final review must not run after a section conflict")
        portion = json.loads(prompt.split("PORTION:\n", 1)[1])
        segment_id = portion["segments"][0]["segment"]["segment_id"]
        empty = {
            "segment_id": segment_id,
            "translation_blocks": None,
            "commentary": None,
            "explanation": None,
            "commentary_sources": None,
            "prior_work": None,
            "later_work": None,
        }
        return {
            "findings": [],
            "reviewed_segment_ids": [
                item["segment"]["segment_id"] for item in portion["segments"]
            ],
            "patches": [
                {**empty, "commentary": "版本一", "reason": "first"},
                {**empty, "commentary": "版本二", "reason": "second"},
            ],
        }

    checkpoint_dir = tmp_path / "checkpoints"
    with pytest.raises(
        RuntimeError,
        match="section review returned conflicting patches.*field commentary",
    ):
        _review(
            segments, translations, annotations,
            document=document, glossary={"entries": []}, protected_names=[],
            evidence={"related_papers": []},
            options=BuildOptions(
                paper_id="arXiv:1234.5678", project_dir=tmp_path, workers=1,
                review_context_chars=1,
            ),
            llm=llm, checkpoint_dir=checkpoint_dir,
        )

    assert not (checkpoint_dir / "section-reviews" / "0000.json").exists()


def test_section_review_merges_compatible_sparse_patches_before_validation() -> None:
    chunk = [{"segment": {
        "segment_id": "seg-1",
        "augmentation_block_ids": ["b1", "b2"],
    }}]
    empty = {
        "segment_id": "seg-1",
        "translation_blocks": None,
        "commentary": None,
        "explanation": None,
        "commentary_sources": None,
        "prior_work": None,
        "later_work": None,
    }
    translation_blocks = [
        {"block_id": "b1", "text": "译文一"},
        {"block_id": "b2", "text": "译文二"},
    ]
    review = {
        "reviewed_segment_ids": ["seg-1"],
        "findings": [],
        "patches": [
            {
                **empty,
                "translation_blocks": [translation_blocks[1]],
                "reason": "translation b2",
            },
            {
                **empty,
                "translation_blocks": [translation_blocks[0]],
                "reason": "translation b1",
            },
            {
                **empty,
                "translation_blocks": [translation_blocks[1]],
                "reason": "translation b2",
            },
            {**empty, "commentary": "伴读", "reason": "commentary"},
            {**empty, "commentary": "伴读", "reason": "commentary"},
        ],
    }

    normalized = pipeline_module._normalize_sparse_review_patches(
        review,
        block_order_by_segment=pipeline_module._review_patch_block_order(chunk),
        scope="section review",
    )

    assert pipeline_module._section_review_validation_error(normalized, chunk) is None
    assert normalized["patches"] == [{
        **empty,
        "translation_blocks": translation_blocks,
        "commentary": "伴读",
        "reason": "translation b2; translation b1; commentary",
    }]


@pytest.mark.parametrize(
    ("field", "first", "second", "message"),
    [
        ("commentary", "版本一", "版本二", "seg-1 field commentary"),
        (
            "translation_blocks",
            [{"block_id": "b1", "text": "译文一"}],
            [{"block_id": "b1", "text": "译文二"}],
            "seg-1 translation block b1",
        ),
    ],
)
def test_section_review_sparse_patch_conflicts_fail_closed(
    field: str, first: object, second: object, message: str,
) -> None:
    empty = {
        "segment_id": "seg-1",
        "translation_blocks": None,
        "commentary": None,
        "explanation": None,
        "commentary_sources": None,
        "prior_work": None,
        "later_work": None,
    }
    review = {
        "reviewed_segment_ids": ["seg-1"],
        "findings": [],
        "patches": [
            {**empty, field: first, "reason": "first"},
            {**empty, field: second, "reason": "second"},
        ],
    }

    with pytest.raises(RuntimeError, match=message):
        pipeline_module._normalize_sparse_review_patches(
            review, scope="section review",
        )


def test_final_review_merges_disjoint_same_segment_patches(tmp_path: Path) -> None:
    segments, translations, annotations, document = _minimal_section_review_inputs()
    segment_id = segments[0]["segment_id"]
    empty = {
        "segment_id": segment_id,
        "translation_blocks": None,
        "commentary": None,
        "explanation": None,
        "commentary_sources": None,
        "prior_work": None,
        "later_work": None,
    }

    def llm(_prompt: str, **kwargs):
        assert kwargs["call_label"] == "companion-final-review"
        return {
            "patches": [
                {**empty, "commentary": "合并后的伴读", "reason": "commentary"},
                {**empty, "explanation": "合并后的解释", "reason": "explanation"},
            ],
            "issues": [],
        }

    _, reviewed_annotations, audit = _review(
        segments, translations, annotations,
        document=document, glossary={"entries": []}, protected_names=[],
        evidence={"related_papers": []},
        options=BuildOptions(
            paper_id="arXiv:1234.5678", project_dir=tmp_path, workers=1,
        ),
        llm=llm, checkpoint_dir=tmp_path / "checkpoints",
    )

    assert reviewed_annotations[segment_id]["commentary"] == "合并后的伴读"
    assert reviewed_annotations[segment_id]["explanation"] == "合并后的解释"
    assert audit["patched_segment_ids"] == [segment_id]


def test_direct_review_between_soft_default_and_hard_limit_becomes_hierarchical(
    tmp_path: Path,
) -> None:
    blocks = [
        {"block_id": f"b-{index}", "type": "text", "text": "源" * 8_000}
        for index in range(2)
    ]
    document = {
        "front_matter": {}, "blocks": blocks, "equations": [], "figures": [],
        "tables": [], "bibliography": [], "assets": [],
    }
    segments = [
        {"segment_id": f"seg-{index}", "block_ids": [f"b-{index}"]}
        for index in range(2)
    ]
    translations = {
        f"seg-{index}": {"blocks": [{
            "block_id": f"b-{index}", "text": "译" * 3_000,
        }]}
        for index in range(2)
    }
    annotations = {
        f"seg-{index}": {
            "commentary": "注" * 1_000, "explanation": "", "prior_work": [],
            "later_work": [], "evidence_ids": [], "key_points": [],
            "source_notes": [], "evidence_requests": [],
        }
        for index in range(2)
    }
    calls: list[tuple[str, int]] = []

    def llm(prompt: str, **kwargs):
        label = str(kwargs["call_label"])
        calls.append((label, len(prompt.encode("utf-8"))))
        assert len(prompt.encode("utf-8")) <= pipeline_module.REVIEW_PROMPT_MAX_BYTES
        if label.startswith("companion-section-review-"):
            portion = json.loads(prompt.split("PORTION:\n", 1)[1])
            return {
                "findings": [],
                "reviewed_segments": [{
                    "segment_id": item["segment"]["segment_id"],
                    "translation": item["translation"],
                    "annotation": item["annotation"],
                } for item in portion["segments"]],
            }
        return {"patches": [], "issues": []}

    _, _, audit = _review(
        segments,
        translations,
        annotations,
        document=document,
        glossary={"entries": []},
        protected_names=[],
        evidence={"related_papers": []},
        options=BuildOptions("arXiv:1234.5678", tmp_path),
        llm=llm,
        checkpoint_dir=tmp_path / "checkpoints",
    )

    assert audit["hierarchical"] is True
    assert any(label.startswith("companion-section-review-") for label, _ in calls)
    assert calls[-1][0] == "companion-final-review"


def test_hierarchical_final_review_fails_closed_when_essential_projection_is_too_large(
    tmp_path: Path,
) -> None:
    segment = {"segment_id": "seg-essential", "block_ids": ["b-essential"]}
    block = {"block_id": "b-essential", "type": "text", "text": "source"}

    with pytest.raises(RuntimeError, match="essential projection.*exceeding"):
        pipeline_module._bounded_hierarchical_review_prompt(
            [{
                "section_index": 0,
                "reviewed_segment_ids": ["seg-essential"],
                "findings": [],
                "patch_proposals": [],
            }],
            [segment],
            blocks_by_id={"b-essential": block},
            document={"blocks": [block]},
            segment_payloads=[{
                "segment": segment, "source_blocks": [block],
                "translation": {"blocks": []}, "annotation": {},
                "context_evidence": [],
            }],
            glossary={"entries": []},
            protected_names=["N" * pipeline_module.REVIEW_PROMPT_MAX_BYTES],
            language="zh-CN",
            max_prompt_bytes=pipeline_module.REVIEW_PROMPT_MAX_BYTES,
        )


def test_hierarchical_final_essential_preflight_is_output_independent() -> None:
    segment = {"segment_id": "seg-essential", "block_ids": ["b-essential"]}
    block = {"block_id": "b-essential", "type": "text", "text": "source"}
    common = dict(
        segments=[segment],
        blocks_by_id={"b-essential": block},
        document={"blocks": [block]},
        segment_payloads=[{
            "segment": segment,
            "source_blocks": [block],
            "translation": {"blocks": []},
            "annotation": {},
            "context_evidence": [],
        }],
        glossary={"entries": []},
        protected_names=[],
        language="zh-CN",
        max_prompt_bytes=55_296,
        strict_prompt_bytes=61_440,
        essential_only=True,
    )
    empty_payload, empty_prompt = pipeline_module._bounded_hierarchical_review_prompt(
        [{
            "section_index": 0,
            "reviewed_segment_ids": ["seg-essential"],
            "findings": [],
            "patch_proposals": [],
        }],
        **common,
    )
    large_payload, large_prompt = pipeline_module._bounded_hierarchical_review_prompt(
        [{
            "section_index": 0,
            "reviewed_segment_ids": ["seg-essential"],
            "findings": [
                {"segment_id": "seg-essential", "issue": "x" * 10_000}
                for _ in range(100)
            ],
            "patch_proposals": [{"segment_id": "seg-essential"}] * 100,
        }],
        **common,
    )

    assert large_payload == empty_payload
    assert large_prompt == empty_prompt


def test_recovered_section_reviews_reject_overlapping_old_chunk_topology(
    tmp_path: Path,
) -> None:
    def chunk_item(segment_id: str) -> dict[str, object]:
        return {"segment": {"segment_id": segment_id}}

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "section-reviews.recovered-from-failed-final.v1.json").write_text(
        json.dumps({
            "schema_version": "arc.companion.recovered-section-reviews.v1",
            "reviewed_segment_ids": ["seg-a", "seg-b", "seg-c"],
            "section_reviews": [
                {
                    "section_index": 0,
                    "reviewed_segment_ids": ["seg-a", "seg-b"],
                    "findings": [],
                    "reviewed_segments": [
                        {"segment_id": value, "translation": {}, "annotation": {}}
                        for value in ("seg-a", "seg-b")
                    ],
                },
                {
                    "section_index": 1,
                    "reviewed_segment_ids": ["seg-b", "seg-c"],
                    "findings": [],
                    "reviewed_segments": [
                        {"segment_id": value, "translation": {}, "annotation": {}}
                        for value in ("seg-b", "seg-c")
                    ],
                },
            ],
        }),
        encoding="utf-8",
    )
    new_chunks = [
        [chunk_item("seg-a")], [chunk_item("seg-b")], [chunk_item("seg-c")],
    ]

    with pytest.raises(RuntimeError, match="overlapping segment coverage"):
        pipeline_module._load_recovered_section_reviews(checkpoint_dir, new_chunks)


def test_hierarchical_review_bounds_final_prompt_and_reuses_section_checkpoints(
    tmp_path: Path,
) -> None:
    blocks = [
        {"block_id": f"b{index}", "type": "text", "text": "source " + ("x" * 5_000)}
        for index in range(12)
    ]
    document = {
        "front_matter": {}, "blocks": blocks, "equations": [], "figures": [],
        "tables": [], "bibliography": [], "assets": [],
    }
    segments = [
        {"segment_id": f"seg-{index:04d}", "block_ids": [f"b{index}"], "title": "Body"}
        for index in range(12)
    ]
    translations = {
        segment["segment_id"]: {"blocks": [{
            "block_id": segment["block_ids"][0], "text": "译文" + ("甲" * 5_000),
        }]}
        for segment in segments
    }
    annotations = {
        segment["segment_id"]: {
            "commentary": "伴读" + ("乙" * 2_000), "explanation": "解释",
            "prior_work": [], "later_work": [], "evidence_ids": [],
            "key_points": [], "source_notes": [], "evidence_requests": [],
        }
        for segment in segments
    }
    section_calls: list[str] = []
    final_prompt_lengths: list[int] = []

    def llm(prompt: str, **kwargs):
        label = str(kwargs["call_label"])
        if label.startswith("companion-section-review-"):
            section_calls.append(label)
            assert len(prompt.encode("utf-8")) <= 32 * 1024
            assert "Display equations" in prompt
            portion = json.loads(prompt.split("PORTION:\n", 1)[1])
            return {
                "findings": [{
                    "segment_id": item["segment"]["segment_id"],
                    "issue": "technical issue " + ("z" * 2_000),
                } for item in portion["segments"]],
                "reviewed_segments": [{
                    "segment_id": item["segment"]["segment_id"],
                    "translation": item["translation"],
                    "annotation": item["annotation"],
                } for item in portion["segments"]],
            }
        assert label == "companion-final-review"
        final_prompt_lengths.append(len(prompt.encode("utf-8")))
        assert '"reviewed_segments"' not in prompt
        assert '"patch_proposals"' in prompt
        assert "PRIOR SECTION FINDINGS" not in prompt
        assert "controller-owned or source-only blocks" in prompt
        return {"patches": [{
            "segment_id": "seg-0000", "translation_blocks": None,
            "commentary": "审校后的伴读", "explanation": None,
            "prior_work": None, "later_work": None, "evidence_ids": None,
            "reason": "technical precision",
        }], "issues": []}

    options = BuildOptions(
        paper_id="arXiv:1234.5678", project_dir=tmp_path, workers=4,
        review_context_chars=20_000,
    )
    reviewed_translations, reviewed_annotations, audit = _review(
        segments, translations, annotations,
        document=document, glossary={"entries": []}, protected_names=[],
        evidence={"related_papers": []}, options=options, llm=llm,
        checkpoint_dir=tmp_path / "checkpoints",
    )

    assert reviewed_translations == translations
    assert reviewed_annotations["seg-0000"]["commentary"] == "审校后的伴读"
    assert audit["reviewed_segment_ids"] == [item["segment_id"] for item in segments]
    assert final_prompt_lengths[0] <= 32 * 1024
    first_section_call_count = len(section_calls)
    assert first_section_call_count > 1
    assert len(list((tmp_path / "checkpoints" / "section-reviews").glob("*.json"))) == (
        first_section_call_count
    )

    section_checkpoint_paths = sorted(
        (tmp_path / "checkpoints" / "section-reviews").glob("*.json")
    )
    recovered_sections = []
    for path in section_checkpoint_paths:
        checkpoint = json.loads(path.read_text())
        recovered_sections.append({
            "section_index": checkpoint["section_index"],
            "reviewed_segment_ids": checkpoint["reviewed_segment_ids"],
            **checkpoint["review"],
        })
        path.unlink()
    recovered_path = (
        tmp_path / "checkpoints" /
        "section-reviews.recovered-from-failed-final.v1.json"
    )
    recovered_path.write_text(json.dumps({
        "schema_version": "arc.companion.recovered-section-reviews.v1",
        "reviewed_segment_ids": [item["segment_id"] for item in segments],
        "section_reviews": recovered_sections,
    }), encoding="utf-8")

    _review(
        segments, translations, annotations,
        document=document, glossary={"entries": []}, protected_names=[],
        evidence={"related_papers": []}, options=options, llm=llm,
        checkpoint_dir=tmp_path / "checkpoints",
    )
    assert len(section_calls) == first_section_call_count
    assert len(list((tmp_path / "checkpoints" / "section-reviews").glob("*.json"))) == (
        first_section_call_count
    )

    recovered_path.unlink()
    changed_translations = {**translations, "seg-0000": {"blocks": [
        {"block_id": "b0", "text": "不同译文"}
    ]}}
    _review(
        segments, changed_translations, annotations,
        document=document, glossary={"entries": []}, protected_names=[],
        evidence={"related_papers": []}, options=options, llm=llm,
        checkpoint_dir=tmp_path / "checkpoints",
    )
    assert len(section_calls) > first_section_call_count




def test_first_round_preview_is_published_before_review_without_evidence_rerun(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    fake = FakeLLM()
    fake.annotation_barrier = threading.Barrier(1)
    fake.annotation_started.set()  # A total budget of one intentionally serializes both lanes.
    project = tmp_path / "preview-order"
    compiler_calls: list[tuple[Path, str]] = []

    def llm(prompt: str, **kwargs):
        label = str(kwargs["call_label"])
        if label == "companion-final-review":
            assert (project / "arXiv-1234.5678_companion_zh-CN_first_round_preview.pdf").is_file()
            assert (project / "first-round-preview-source-manifest.json").is_file()
            assert (project / "first-round-preview-validation.json").is_file()
        return fake(prompt, **kwargs)

    def compiler(tex_path: Path, pdf_path: Path) -> None:
        if "first_round_preview" in tex_path.name:
            assert tex_path.parent == project
        else:
            assert tex_path.parent.parent.parent == project / ".arc-companion" / "renders"
        assert pdf_path.parent == tex_path.parent
        assert not tex_path.name.startswith(".")
        assert not pdf_path.name.startswith(".")
        assert tex_path.stem == pdf_path.stem
        compiler_calls.append((tex_path, tex_path.read_text(encoding="utf-8")))
        if len(compiler_calls) == 1:
            labels = [str(call["call_label"]) for call in fake.calls]
            assert "companion-translation-seg-0001" in labels
            assert "companion-annotation-seg-0001" in labels
            assert "companion-translation-seg-0002" not in labels
            assert "companion-annotation-seg-0002" not in labels
        pdf_path.write_bytes(b"%PDF-1.7 fixture")

    result = build_companion(
        BuildOptions(
            paper_id=bundle.paper_id,
            project_dir=project,
            review_context_chars=1,
            workers=1,
        ),
        source_loader=lambda *args, **kwargs: bundle,
        llm=llm,
        compiler=compiler,
        pdf_validator=lambda path: {"bytes": path.stat().st_size},
    )

    assert result["ok"], result
    assert len(compiler_calls) == 2
    assert not any("evidence-rerun" in str(call["call_label"]) for call in fake.calls)
    assert "解释 0001" in compiler_calls[0][1]
    assert "解释 0002" not in compiler_calls[0][1]
    assert r"\tag{7}" in compiler_calls[0][1]
    assert "Figure 2: A plot" not in compiler_calls[0][1]
    assert "审校后的解释" not in compiler_calls[0][1]
    assert "审校后的解释" in compiler_calls[1][1]
    state = result["data"]
    assert state["preview_segment_count"] == 1
    assert state["preview_segment_ids"] == ["seg-0001"]
    for key in (
        "preview_tex", "preview_pdf", "preview_source_manifest_path", "preview_validation_path",
    ):
        assert Path(state[key]).is_file()
    for key in (
        "preview_tex_sha256", "preview_pdf_sha256",
        "preview_source_manifest_sha256", "preview_validation_sha256",
    ):
        assert state[key]


def test_first_wave_uses_first_substantive_unit_and_preserves_leading_source(tmp_path: Path) -> None:
    substantive_preface = (
        "This preface explains the motivation and intended conceptual route in "
        "enough detail to be part of the book's substantive argument. "
    ) * 2
    blocks = [
        {
            "block_id": "title", "kind": "heading", "section_id": "title",
            "source_role": "front_matter_title", "text": "A General Book",
        },
        {
            "block_id": "authors", "kind": "prose", "section_id": "authors",
            "source_role": "front_matter_authors", "text": "A. Author",
        },
        {"block_id": "routes", "kind": "heading", "section_id": "routes", "text": "Reading routes"},
        {"block_id": "route-1", "kind": "prose", "section_id": "routes", "text": "• First resource"},
        {"block_id": "route-2", "kind": "prose", "section_id": "routes", "text": "• Second resource"},
        {
            "block_id": "toc", "kind": "heading", "section_id": "toc",
            "source_role": "table_of_contents", "text": "Contents",
        },
        {"block_id": "toc-preface", "kind": "heading", "section_id": "toc-p", "text": "Preface 3"},
        {"block_id": "toc-body", "kind": "heading", "section_id": "toc-b", "text": "1 Beginning 5"},
        {"block_id": "preface", "kind": "heading", "section_id": "preface", "text": "Preface"},
        {"block_id": "preface-text", "kind": "prose", "section_id": "preface", "text": substantive_preface},
        {"block_id": "body", "kind": "heading", "section_id": "body", "text": "1 Beginning"},
        {"block_id": "body-text", "kind": "prose", "section_id": "body", "text": "Physical body text."},
    ]
    document = {
        "schema_version": "arc.paper.document.v2",
        "front_matter": {
            "title": "A General Book", "authors": ["A. Author"],
            "block_ids": {"title": ["title"], "authors": ["authors"]},
        },
        "blocks": blocks, "equations": [], "figures": [], "tables": [],
        "assets": [], "bibliography": [], "links": [],
        "integrity": {"status": "complete", "document_hash": "substantive-fixture"},
    }
    bundle = SourceBundle(
        paper_id="local:substantive-fixture",
        parsed={"paper_id": "local:substantive-fixture", "document": document},
        document=document,
        metadata={"title": "A General Book", "authors": ["A. Author"]},
        references=[], citers=[],
    )
    calls: list[tuple[str, str]] = []

    def llm(prompt: str, **kwargs):
        label = str(kwargs["call_label"])
        calls.append((label, prompt))
        if label.startswith("title-translation-"):
            payload = json.loads(prompt[prompt.index("{"):])
            return {"titles": [
                {"title_id": item["title_id"], "text": item["source_text"]}
                for item in payload["titles"]
            ]}
        if label.startswith("companion-segmentation-w-"):
            return {"cut_after_ordinals": []}
        if label.startswith("companion-glossary-"):
            return {"entries": []}
        if label == "companion-translation-seg-0001":
            return {"blocks": [
                {"block_id": "preface", "text": "序言"},
                {"block_id": "preface-text", "text": "有实质内容的序言译文。"},
            ]}
        if label == "companion-annotation-seg-0001":
            return {
                "explanation": "序言说明全书动机。", "prior_work": [], "later_work": [],
                "commentary": "序言伴读。", "evidence_ids": [], "key_points": [],
                "source_notes": [], "evidence_requests": [],
            }
        raise AssertionError(f"preview gate submitted unexpected work: {label}")

    project = tmp_path / "substantive-first-wave"
    result = build_companion(
        BuildOptions(
            paper_id=bundle.paper_id, project_dir=project, workers=1,
            review_context_chars=1, stop_after_first_chapter=True,
        ),
        source_loader=lambda *args, **kwargs: bundle,
        llm=llm,
        compiler=lambda _tex, pdf: pdf.write_bytes(b"%PDF-1.7 fixture"),
        pdf_validator=lambda path: {"bytes": path.stat().st_size},
    )

    assert result["ok"], result
    assert [block["block_id"] for block in _generation_document(document)["blocks"]] == [
        "preface", "preface-text", "body", "body-text",
    ]
    assert result["data"]["preview_segment_ids"] == ["seg-0001"]
    assert result["data"]["preview_segment_count"] == 1
    assert not any(label.endswith("seg-0002") for label, _ in calls)
    for _label, prompt in calls:
        if _label.startswith("title-translation-"):
            continue
        assert "First resource" not in prompt
        assert "Second resource" not in prompt
        assert "Preface 3" not in prompt
    tex = Path(result["data"]["preview_tex"]).read_text(encoding="utf-8")
    assert "First resource" in tex and "Second resource" in tex and "Preface 3" in tex
    assert "Physical body text" not in tex
    assert "有实质内容的序言译文" in tex and "序言说明全书动机" in tex
    manifest = json.loads(Path(result["data"]["preview_source_manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["companion_layers"]["augmentation_scope"] == "substantive"
    assert manifest["companion_layers"]["semantic_segment_ids"] == ["seg-0001"]


def test_stop_after_first_chapter_returns_before_remaining_work_and_resumes(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    fake = FakeLLM()
    fake.annotation_barrier = threading.Barrier(1)
    fake.annotation_started.set()  # A total budget of one intentionally serializes both lanes.
    project = tmp_path / "preview-gate"
    compiler_calls: list[Path] = []
    validation_calls: list[Path] = []

    def compiler(tex_path: Path, pdf_path: Path) -> None:
        compiler_calls.append(tex_path)
        pdf_path.write_bytes(b"%PDF-1.7 fixture")

    def pdf_validator(path: Path) -> dict[str, object]:
        validation_calls.append(path)
        return {"bytes": path.stat().st_size}

    gated = build_companion(
        BuildOptions(
            paper_id=bundle.paper_id,
            project_dir=project,
            workers=1,
            review_context_chars=1,
            stop_after_first_chapter=True,
        ),
        source_loader=lambda *args, **kwargs: bundle,
        llm=fake,
        compiler=compiler,
        pdf_validator=pdf_validator,
    )

    assert gated["ok"], gated
    assert gated["data"]["status"] == "preview_ready"
    assert gated["data"]["preview_segment_ids"] == ["seg-0001"]
    assert Path(gated["data"]["preview_pdf"]).is_file()
    assert "output_run_pdf" not in gated["data"]
    assert len(compiler_calls) == len(validation_calls) == 1
    first_labels = [str(call["call_label"]) for call in fake.calls]
    assert "companion-translation-seg-0001" in first_labels
    assert "companion-annotation-seg-0001" in first_labels
    assert not any(label.endswith("seg-0002") for label in first_labels)
    assert not any("review" in label for label in first_labels)

    resumed = build_companion(
        BuildOptions(
            paper_id=bundle.paper_id,
            project_dir=project,
            workers=1,
            review_context_chars=1,
        ),
        source_loader=lambda *args, **kwargs: bundle,
        llm=fake,
        compiler=compiler,
        pdf_validator=pdf_validator,
    )

    assert resumed["ok"], resumed
    assert resumed["data"]["status"] == "complete"
    assert Path(resumed["data"]["output_pdf"]).is_file()
    assert Path(resumed["data"]["output_run_pdf"]).parent == project.resolve()
    all_labels = [str(call["call_label"]) for call in fake.calls]
    assert all_labels.count("companion-translation-seg-0001") == 1
    assert all_labels.count("companion-annotation-seg-0001") == 1
    assert "companion-annotation-seg-0002" in all_labels
    assert any("review" in label for label in all_labels)


def test_first_wave_preview_preserves_rich_entities_and_exact_link_occurrences() -> None:
    repeated_link = '<a href="https://example.test/paper">paper</a>'
    document = {
        "blocks": [
            {
                "block_id": "kept", "kind": "heading", "html": '<h2 id="kept">Kept</h2>',
            },
            {
                "block_id": "eq-block", "source_id": "eq-source", "kind": "equation",
                "html": f'<div>{repeated_link}<a href="#later">later</a></div>',
            },
            {
                "block_id": "fig-block", "entity_id": "fig-entity", "kind": "figure",
                "html": f"<figure>{repeated_link}</figure>",
            },
            {
                "block_id": "tab-block", "source_id": "tab-source", "kind": "table",
                "html": '<table><tr><td><a href="#kept">kept</a></td></tr></table>',
            },
            {
                "block_id": "later", "source_id": "later-equation", "kind": "equation",
                "html": f'<div>{repeated_link}<a href="#later">later</a></div>',
            },
        ],
        "equations": [
            {"id": "eq-source", "tex": ["x=1"]},
            {"id": "later-equation", "tex": ["y=2"]},
        ],
        "figures": [{"id": "fig-entity", "asset_ids": ["asset-kept"]}],
        "tables": [{"id": "tab-source", "rows": []}],
        "assets": [
            {"asset_id": "asset-kept", "cache_path": "/cache/kept"},
            {"asset_id": "asset-later", "cache_path": "/cache/later"},
        ],
        "bibliography": [{"id": "ref-later", "label": "[1]"}],
        "links": [
            {"id": "link-1", "href": "https://example.test/paper", "target_id": "", "text": "paper"},
            {"id": "link-2", "href": "https://example.test/paper", "target_id": "", "text": "paper"},
            {"id": "link-3", "href": "#kept", "target_id": "kept", "text": "kept"},
            {"id": "link-later-1", "href": "https://example.test/paper", "target_id": "", "text": "paper"},
            {"id": "link-later-2", "href": "#later", "target_id": "later", "text": "later"},
        ],
    }
    preview = _first_wave_preview_document(document, [{
        "segment_id": "first",
        "block_ids": ["kept", "eq-block", "fig-block", "tab-block"],
    }])

    assert [item["block_id"] for item in preview["blocks"]] == [
        "kept", "eq-block", "fig-block", "tab-block",
    ]
    assert [item["id"] for item in preview["equations"]] == ["eq-source"]
    assert [item["id"] for item in preview["figures"]] == ["fig-entity"]
    assert [item["id"] for item in preview["tables"]] == ["tab-source"]
    assert [item["asset_id"] for item in preview["assets"]] == ["asset-kept"]
    assert preview["links"] == [
        {"href": "https://example.test/paper", "target_id": "", "text": "paper"},
        {"href": "#later", "target_id": "later", "text": "later"},
        {"href": "https://example.test/paper", "target_id": "", "text": "paper"},
        {"href": "#kept", "target_id": "kept", "text": "kept"},
    ]
    assert preview["bibliography"] == []
    assert preview["preview_scope"] == {"kind": "source_prefix"}


def test_first_round_preview_failure_stops_before_evidence_and_review(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    fake = FakeLLM()
    fake.annotation_barrier = threading.Barrier(1)
    fake.annotation_started.set()  # A total budget of one intentionally serializes both lanes.

    result = build_companion(
        BuildOptions(
            paper_id=bundle.paper_id,
            project_dir=tmp_path / "preview-failure",
            workers=1,
        ),
        source_loader=lambda *args, **kwargs: bundle,
        llm=fake,
        compiler=lambda *_: (_ for _ in ()).throw(RuntimeError("preview compile failed")),
        pdf_validator=lambda _: (_ for _ in ()).throw(AssertionError("must not validate")),
    )

    assert not result["ok"]
    assert "preview compile failed" in result["error"]["message"]
    assert not any("review" in str(call["call_label"]) for call in fake.calls)
    assert not any(
        str(call["call_label"]).endswith("seg-0002") for call in fake.calls
    )
    assert json.loads(
        (tmp_path / "preview-failure" / "state.json").read_text(encoding="utf-8")
    )["status"] == "failed"


def test_outer_failure_preserves_supervised_lane_status(
    tmp_path: Path, monkeypatch,
) -> None:
    bundle = _bundle(tmp_path)
    bundle.parsed["structure"] = {"document_kind": "book", "chapters": []}

    def fail_after_lane_supervision(**kwargs):
        ledger_path = (
            kwargs["checkpoint_dir"] / "chapters" / "ch-0001" / "translation-ledger.json"
        )
        pipeline_module.initialize_lane_ledger(
            ledger_path, chapter_id="ch-0001", lane="translation", segment_ids=["s1"],
        )
        pipeline_module.mark_needs_supervision(
            ledger_path, segment_id="s1", reason="submitted call cancelled",
            recovery_context={"submission_state": "unknown", "resumable": True},
        )
        raise RuntimeError("aggregate lane cancellation")

    monkeypatch.setattr(
        pipeline_module, "_build_chaptered_companion", fail_after_lane_supervision,
    )
    result = build_companion(
        BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path / "supervised"),
        source_loader=lambda *args, **kwargs: bundle,
        llm=FakeLLM(),
    )

    assert result["status"] == "needs_supervision"
    state = json.loads((tmp_path / "supervised" / "state.json").read_text())
    assert state["status"] == "needs_supervision"
    assert state["recovery_options"]["paper_id"] == bundle.paper_id


def test_non_json_status_prefers_ready_preview_path(capsys) -> None:
    _emit(
        {"ok": True, "data": {"status": "preview_ready", "preview_pdf": "/run/preview.pdf"}, "meta": {}},
        json_output=False,
    )
    assert capsys.readouterr().out == "/run/preview.pdf\n"


def test_segment_ranges_must_cover_blocks_once_in_order() -> None:
    blocks = [{"block_id": value} for value in ("a", "b", "c")]
    valid = validate_and_expand_segments(
        [
            {"segment_id": "one", "title": "One", "start_block_id": "a", "end_block_id": "b"},
            {"segment_id": "two", "title": "Two", "start_block_id": "c", "end_block_id": "c"},
        ],
        blocks,
    )
    assert valid[0]["block_ids"] == ["a", "b"]

    invalid = [{"segment_id": "one", "title": "One", "start_block_id": "b", "end_block_id": "c"}]
    try:
        validate_and_expand_segments(invalid, blocks)
    except ValueError as exc:
        assert "contiguous" in str(exc)
    else:
        raise AssertionError("invalid segmentation was accepted")


def test_source_fingerprint_excludes_lane_metadata_and_evidence(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    options = BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path / "run")
    evidence = _evidence(bundle)
    first = _fingerprint(bundle, options, evidence=evidence)
    changed_metadata = SourceBundle(
        paper_id=bundle.paper_id,
        parsed=bundle.parsed,
        document=bundle.document,
        metadata={**bundle.metadata, "title": "Changed"},
        references=bundle.references,
        citers=bundle.citers,
    )
    assert _fingerprint(changed_metadata, options, evidence=evidence) == first
    assert _fingerprint(bundle, options, evidence={**evidence, "citers": []}) == first
    assert _segment_checkpoint_name("a/b") != _segment_checkpoint_name("a b")


def test_source_fingerprint_excludes_glossary_tier(tmp_path: Path, monkeypatch) -> None:
    bundle = _bundle(tmp_path)
    options = BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path / "run")
    evidence = _evidence(bundle)
    medium = _fingerprint(bundle, options, evidence=evidence)

    monkeypatch.setattr(pipeline_module, "GLOSSARY_TIER", "high")

    assert _fingerprint(bundle, options, evidence=evidence) == medium


def test_source_fingerprint_excludes_review_tier(tmp_path: Path, monkeypatch) -> None:
    bundle = _bundle(tmp_path)
    options = BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path / "run")
    evidence = _evidence(bundle)
    baseline = _fingerprint(bundle, options, evidence=evidence)

    monkeypatch.setattr(pipeline_module, "REVIEW_TIER", "high")

    assert _fingerprint(bundle, options, evidence=evidence) == baseline


def test_fingerprint_reuses_content_checkpoints_when_total_workers_change(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    evidence = _evidence(bundle)
    default = BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path / "run")
    old = BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path / "run", workers=12)

    assert default.workers == 24
    assert _fingerprint(bundle, default, evidence=evidence) == _fingerprint(bundle, old, evidence=evidence)


def test_shared_llm_limiter_bounds_aggregate_concurrency_across_lanes() -> None:
    lock = threading.Lock()
    release = threading.Event()
    budget_reached = threading.Event()
    active = 0
    peak = 0

    def model(prompt: str, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 2:
                budget_reached.set()
        assert release.wait(timeout=5)
        with lock:
            active -= 1
        return {"prompt": prompt}

    limited = _limit_llm_concurrency(model, 2)

    def lane(prefix: str) -> list[dict[str, str]]:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(limited, f"{prefix}-{index}") for index in range(4)]
            return [future.result() for future in futures]

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(lane, prefix) for prefix in ("translation", "annotation")]
        assert budget_reached.wait(timeout=5)
        assert peak == 2
        release.set()
        results = [future.result() for future in futures]

    assert sum(len(items) for items in results) == 8
    assert active == 0
    assert peak == 2


def test_shared_llm_limiter_stops_queued_work_but_drains_submitted_calls() -> None:
    lock = threading.Lock()
    both_started = threading.Event()
    release_abort = threading.Event()
    release_drain = threading.Event()
    calls: list[str] = []
    active = 0

    class FatalProviderError(RuntimeError):
        abort_batch = True

    def model(prompt: str, *, cancel_check=None) -> dict[str, str]:
        nonlocal active
        with lock:
            calls.append(prompt)
            active += 1
            if active == 2:
                both_started.set()
        assert both_started.wait(timeout=5)
        if prompt == "translation-active":
            assert release_abort.wait(timeout=5)
            try:
                raise FatalProviderError("usage quota exhausted")
            except FatalProviderError as exc:
                raise RuntimeError("wrapped provider failure") from exc
        assert cancel_check is not None
        assert release_drain.wait(timeout=5)
        assert cancel_check() is False
        return {"prompt": prompt}

    limited = _limit_llm_concurrency(model, 2)
    with ThreadPoolExecutor(max_workers=6) as executor:
        active_futures = [
            executor.submit(limited, "translation-active"),
            executor.submit(limited, "annotation-active"),
        ]
        assert both_started.wait(timeout=5)
        queued_futures = [
            executor.submit(limited, "translation-queued-1"),
            executor.submit(limited, "annotation-queued-1"),
            executor.submit(limited, "translation-queued-2"),
            executor.submit(limited, "annotation-queued-2"),
        ]
        release_abort.set()
        with pytest.raises(RuntimeError, match="wrapped provider failure"):
            active_futures[0].result(timeout=5)
        release_drain.set()
        assert active_futures[1].result(timeout=5) == {"prompt": "annotation-active"}
        for future in queued_futures:
            with pytest.raises(CompanionLLMCircuitOpen):
                future.result(timeout=5)

    assert set(calls) == {"translation-active", "annotation-active"}
    assert len(calls) == 2


def test_shared_llm_limiter_does_not_inject_cancel_check_into_simple_fake() -> None:
    calls: list[str] = []

    def model(prompt: str) -> dict[str, str]:
        calls.append(prompt)
        return {"prompt": prompt}

    assert _limit_llm_concurrency(model, 1)("plain") == {"prompt": "plain"}
    assert calls == ["plain"]


def test_shared_llm_limiter_propagates_external_cancel_and_stops_queued_work() -> None:
    requested = threading.Event()
    active_started = threading.Event()

    def model(prompt: str, *, cancel_check=None) -> dict[str, str]:
        active_started.set()
        assert cancel_check is not None
        while not cancel_check():
            threading.Event().wait(0.001)
        raise RuntimeError(f"cancelled {prompt}")

    limited = _limit_llm_concurrency(model, 1, cancel_check=requested.is_set)
    with ThreadPoolExecutor(max_workers=2) as executor:
        active = executor.submit(limited, "active")
        assert active_started.wait(timeout=5)
        queued = executor.submit(limited, "queued")
        requested.set()
        with pytest.raises(RuntimeError, match="cancelled active"):
            active.result(timeout=5)
        with pytest.raises(CompanionLLMCircuitOpen):
            queued.result(timeout=5)


def test_shared_llm_limiter_stops_successors_after_any_call_failure() -> None:
    calls: list[str] = []

    def model(prompt: str) -> dict[str, str]:
        calls.append(prompt)
        if prompt == "bad-unit":
            raise RuntimeError("ordinary validation failure")
        return {"prompt": prompt}

    limited = _limit_llm_concurrency(model, 1)
    with pytest.raises(RuntimeError, match="ordinary validation failure"):
        limited("bad-unit")
    with pytest.raises(CompanionLLMCircuitOpen):
        limited("independent-unit")
    assert calls == ["bad-unit"]


def test_24_worker_preflight_failure_drains_all_submitted_calls() -> None:
    """A local failure must not fan out cancellation to 23 provider calls."""
    all_calls_started = threading.Event()
    release_failure = threading.Event()
    release_submitted = threading.Event()
    calls: list[str] = []
    calls_lock = threading.Lock()

    def model(prompt: str, *, cancel_check=None) -> dict[str, str]:
        with calls_lock:
            calls.append(prompt)
            if len(calls) == 24:
                all_calls_started.set()
        assert all_calls_started.wait(timeout=5)
        if prompt == "local-preflight-failure":
            assert release_failure.wait(timeout=5)
            raise ValueError("local schema preflight failed")
        assert cancel_check is not None
        assert release_submitted.wait(timeout=5)
        assert cancel_check() is False
        return {"prompt": prompt}

    limited = _limit_llm_concurrency(model, 24)
    with ThreadPoolExecutor(max_workers=25) as executor:
        active = [
            executor.submit(limited, "local-preflight-failure"),
            *[
                executor.submit(limited, f"submitted-{index}")
                for index in range(23)
            ],
        ]
        assert all_calls_started.wait(timeout=5)
        queued = executor.submit(limited, "queued-after-failure")
        release_failure.set()
        with pytest.raises(ValueError, match="local schema preflight failed"):
            active[0].result(timeout=5)
        release_submitted.set()
        assert [future.result(timeout=5) for future in active[1:]] == [
            {"prompt": f"submitted-{index}"} for index in range(23)
        ]
        with pytest.raises(CompanionLLMCircuitOpen):
            queued.result(timeout=5)

    assert "queued-after-failure" not in calls
    assert len(calls) == 24


def test_legacy_worker_fingerprint_checkpoint_is_exactly_migrated(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    project = tmp_path / "run"
    options = BuildOptions(paper_id=bundle.paper_id, project_dir=project, workers=3)
    evidence = _evidence(bundle)
    fingerprint = _fingerprint(bundle, options, evidence=evidence)
    legacy_fingerprint = _legacy_worker_fingerprint(
        bundle,
        options,
        evidence=evidence,
        domain_context=None,
        workers_per_lane=24,
    )
    legacy = project / ".arc-companion" / "checkpoints" / legacy_fingerprint
    legacy.mkdir(parents=True)
    (legacy / "reusable.json").write_text("{}", encoding="utf-8")
    (project / "context.json").write_text(json.dumps({"workers": 24}), encoding="utf-8")

    previous_state = {
        "fingerprint": legacy_fingerprint,
        "checkpoint_dir": str(legacy),
    }
    target = _checkpoint_dir_with_legacy_worker_migration(
        project,
        fingerprint=fingerprint,
        bundle=bundle,
        options=options,
        evidence=evidence,
        domain_context=None,
        previous_state=previous_state,
    )

    assert target == project / ".arc-companion" / "checkpoints" / fingerprint
    assert not legacy.exists()
    assert (target / "reusable.json").is_file()
    migration = json.loads((target / "checkpoint-migration.v1.json").read_text())
    assert migration["legacy_fingerprint"] == legacy_fingerprint
    assert migration["content_fingerprint"] == fingerprint
    assert migration["legacy_workers_per_lane"] == 24
    assert previous_state["fingerprint"] == fingerprint
    assert previous_state["checkpoint_dir"] == str(target)


def test_legacy_worker_checkpoint_is_found_without_context_json(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    project = tmp_path / "run-without-context"
    options = BuildOptions(paper_id=bundle.paper_id, project_dir=project, workers=2)
    evidence = _evidence(bundle)
    fingerprint = _fingerprint(bundle, options, evidence=evidence)
    legacy_fingerprint = _legacy_worker_fingerprint(
        bundle,
        options,
        evidence=evidence,
        domain_context=None,
        workers_per_lane=317,
    )
    legacy = project / ".arc-companion" / "checkpoints" / legacy_fingerprint
    legacy.mkdir(parents=True)
    (legacy / "window-0152.json").write_text("{}", encoding="utf-8")
    previous_state = {
        "fingerprint": legacy_fingerprint,
        "checkpoint_dir": str(legacy.resolve()),
    }

    target = _checkpoint_dir_with_legacy_worker_migration(
        project,
        fingerprint=fingerprint,
        bundle=bundle,
        options=options,
        evidence=evidence,
        domain_context=None,
        previous_state=previous_state,
    )

    assert target.name == fingerprint
    assert not legacy.exists()
    assert (target / "window-0152.json").is_file()
    migration = json.loads((target / "checkpoint-migration.v1.json").read_text())
    assert migration["legacy_workers_per_lane"] == 317


def test_preview_completion_check_is_independent_of_current_worker_budget(tmp_path: Path) -> None:
    state: dict[str, object] = {
        "first_wave_preview_version": pipeline_module.FIRST_WAVE_PREVIEW_VERSION,
        "segment_count": 20,
        "preview_segment_count": 12,
        "preview_segment_ids": [f"seg-{index:04d}" for index in range(1, 13)],
    }
    for path_key, hash_key in (
        ("preview_tex", "preview_tex_sha256"),
        ("preview_pdf", "preview_pdf_sha256"),
        ("preview_source_manifest_path", "preview_source_manifest_sha256"),
        ("preview_validation_path", "preview_validation_sha256"),
    ):
        path = tmp_path / path_key
        path.write_bytes(path_key.encode("utf-8"))
        state[path_key] = str(path)
        state[hash_key] = hashlib.sha256(path.read_bytes()).hexdigest()

    assert _first_wave_preview_outputs_match(state)


def test_completed_fast_path_requires_current_projection_guide_and_reader_versions(
    tmp_path: Path,
) -> None:
    state: dict[str, object] = {
        "final_render_version": pipeline_module.FINAL_RENDER_VERSION,
        "chapter_projection_version": pipeline_module.CHAPTER_PROJECTION_VERSION,
        "augmentation_projection_version": pipeline_module.AUGMENTATION_PROJECTION_VERSION,
        "chapter_guide_version": pipeline_module.CHAPTER_GUIDE_VERSION,
        "reader_final_checkpoint_version": pipeline_module.READER_FINAL_CHECKPOINT_VERSION,
    }
    for path_key, hash_key in (
        ("output_tex", "output_tex_sha256"),
        ("output_pdf", "output_pdf_sha256"),
        ("source_manifest_path", "source_manifest_sha256"),
        ("validation_path", "validation_sha256"),
    ):
        path = tmp_path / path_key
        path.write_bytes(path_key.encode("utf-8"))
        state[path_key] = str(path)
        state[hash_key] = hashlib.sha256(path.read_bytes()).hexdigest()

    assert pipeline_module._completion_outputs_match(state)
    legacy = {**state, "augmentation_projection_version": "arc.companion.augmentation-projection.v1"}
    assert not pipeline_module._completion_outputs_match(legacy)
    assert all(path.is_file() for path in tmp_path.iterdir())


def test_run_root_pdf_validation_is_optional_for_legacy_state_and_strict_when_recorded(
    tmp_path: Path,
) -> None:
    project = (tmp_path / "project").resolve()
    project.mkdir()
    canonical = project / ".arc-companion" / "renders" / "rev" / "paper.pdf"
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"%PDF canonical")
    digest = hashlib.sha256(canonical.read_bytes()).hexdigest()
    delivery = project / "paper.pdf"
    delivery.write_bytes(canonical.read_bytes())
    state = {
        "output_pdf": str(canonical),
        "output_pdf_sha256": digest,
        "output_run_pdf": str(delivery),
        "output_run_pdf_sha256": digest,
    }

    assert pipeline_module._run_root_pdf_output_matches({}, project)
    assert pipeline_module._run_root_pdf_output_matches(state, project)
    assert pipeline_module._run_root_pdf_output_matches(
        {
            "output_pdf_sha256": digest,
            "output_project_pdf": str(delivery),
            "output_project_pdf_sha256": digest,
        },
        project,
    )
    assert not pipeline_module._run_root_pdf_output_matches(
        {**state, "output_run_pdf_sha256": None}, project,
    )
    assert not pipeline_module._run_root_pdf_output_matches(
        {**state, "output_run_pdf": str(tmp_path / "outside.pdf")}, project,
    )
    delivery.write_bytes(b"tampered")
    assert not pipeline_module._run_root_pdf_output_matches(state, project)
    delivery.unlink()
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(canonical.read_bytes())
    delivery.symlink_to(outside)
    assert not pipeline_module._run_root_pdf_output_matches(state, project)


def test_run_root_pdf_state_event_failure_keeps_already_committed_delivery(
    tmp_path: Path, monkeypatch,
) -> None:
    import arc_companion.observability as observability

    project = (tmp_path / "project").resolve()
    canonical = project / ".arc-companion" / "renders" / "rev" / "paper.pdf"
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"%PDF canonical")
    digest = hashlib.sha256(canonical.read_bytes()).hexdigest()
    state_path = project / "state.json"
    _state(
        state_path,
        status="complete",
        output_pdf=str(canonical),
        output_pdf_sha256=digest,
    )
    publication = pipeline_module.publish_run_root_pdf(canonical, project)
    monkeypatch.setattr(
        observability,
        "append_state_event",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("injected event append failure")
        ),
    )

    with pytest.raises(OSError, match="event append failure"):
        _state(state_path, **publication)

    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["output_run_pdf"] == publication["output_run_pdf"]
    assert (
        persisted["published"]["pdf"]["output_run_pdf"]
        == publication["output_run_pdf"]
    )
    assert Path(publication["output_run_pdf"]).is_file()


def test_run_root_pdf_precommit_failure_leaves_adoptable_delivery(
    tmp_path: Path, monkeypatch,
) -> None:
    project = (tmp_path / "project").resolve()
    canonical = project / ".arc-companion" / "renders" / "rev" / "paper.pdf"
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"%PDF canonical")
    state_path = project / "state.json"
    state_path.write_text('{"status":"complete"}', encoding="utf-8")
    publication = pipeline_module.publish_run_root_pdf(canonical, project)
    real_state = pipeline_module._state
    monkeypatch.setattr(
        pipeline_module,
        "_state",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("injected state write failure")
        ),
    )

    with pytest.raises(OSError, match="state write failure"):
        pipeline_module._state(state_path, **publication)

    assert Path(publication["output_run_pdf"]).is_file()
    monkeypatch.setattr(pipeline_module, "_state", real_state)
    adopted = pipeline_module.publish_run_root_pdf(canonical, project)
    persisted = pipeline_module._state(state_path, **adopted)
    assert persisted["output_run_pdf"] == publication["output_run_pdf"]


def test_fingerprint_change_preserves_delivery_ownership_until_replacement(
    tmp_path: Path,
) -> None:
    project = (tmp_path / "project").resolve()
    old_pdf = project / ".arc-companion" / "renders" / "old" / "paper.pdf"
    old_pdf.parent.mkdir(parents=True)
    old_pdf.write_bytes(b"%PDF old")
    state_path = project / "state.json"
    old_hash = hashlib.sha256(old_pdf.read_bytes()).hexdigest()
    old_publication = pipeline_module.publish_run_root_pdf(old_pdf, project)
    _state(
        state_path,
        status="complete",
        fingerprint="old-fingerprint",
        output_pdf=str(old_pdf),
        output_pdf_sha256=old_hash,
        **old_publication,
    )
    new_pdf = project / ".arc-companion" / "renders" / "new" / "paper.pdf"
    new_pdf.parent.mkdir(parents=True)
    new_pdf.write_bytes(b"%PDF new")
    new_hash = hashlib.sha256(new_pdf.read_bytes()).hexdigest()

    interrupted = _state(
        state_path,
        status="complete",
        fingerprint="new-fingerprint",
        output_pdf=str(new_pdf),
        output_pdf_sha256=new_hash,
    )

    assert "output_run_pdf" not in interrupted
    assert "output_run_pdf" not in interrupted["published"]["pdf"]
    assert (
        interrupted["run_pdf_managed_path"]
        == old_publication["output_run_pdf"]
    )
    assert pipeline_module._run_root_pdf_output_matches(interrupted, project)
    managed = pipeline_module.managed_run_root_pdf_path(interrupted)
    new_publication = pipeline_module.publish_run_root_pdf(
        new_pdf, project, managed_path=managed,
    )
    recovered = _state(state_path, **new_publication)
    assert Path(recovered["output_run_pdf"]).read_bytes() == b"%PDF new"
    assert recovered["output_run_pdf_sha256"] == new_hash


def test_controller_only_runtime_keeps_internet_enabled_and_scrubs_polluted_parent_env(
    tmp_path: Path, monkeypatch
) -> None:
    polluted = {
        "ARC_CODEX_PROFILE": "mcp-profile",
        "ARC_CODEX_PROFILE_V2": "mcp-profile-v2",
        "ARC_CODEX_CONFIG": "mcp_servers.bad.command='bad'",
        "ARC_CODEX_CONFIG_JSON": '{"mcp_servers.bad.command":"bad"}',
        "ARC_CODEX_MCP_MODE": "user-config",
        "ARC_CODEX_ARC_MCP_COMMAND": "bad-mcp",
        "ARC_CLAUDE_MCP_MODE": "user-config",
        "ARC_CLAUDE_MCP_CONFIG": "/tmp/bad-mcp.json",
        "ARC_CLAUDE_MCP_CONFIG_JSON": '["/tmp/bad-mcp.json"]',
        "ARC_CLAUDE_ARC_MCP_COMMAND": "bad-mcp",
        "ARC_CLAUDE_TOOLS": "default",
        "ARC_CLAUDE_ALLOWED_TOOLS": "mcp__bad__*",
    }
    for key, value in polluted.items():
        monkeypatch.setenv(key, value)
    captured = {}

    def fake_llm(prompt, **kwargs):
        captured.update(kwargs)
        return {"ok": True}

    options = BuildOptions(
        paper_id="local:book",
        project_dir=tmp_path,
        allow_internet=True,
    )
    _llm_call(
        fake_llm,
        "prompt",
        {"type": "object"},
        options=options,
        artifact_dir=tmp_path / "llm",
        call_label="test",
        model_tier="low",
        allow_internet=True,
    )

    env = captured["env"]
    assert env["ARC_CODEX_ENABLE_MCP"] == "false"
    assert env["ARC_CLAUDE_ALLOW_MCP"] == "false"
    assert env["ARC_CODEX_ALLOW_INTERNET"] == "true"
    assert env["ARC_CLAUDE_ALLOW_INTERNET"] == "true"
    assert env["ARC_CODEX_IGNORE_USER_CONFIG"] == "true"
    assert env["ARC_CLAUDE_BARE"] == "true"
    assert env["ARC_CLAUDE_TOOLS"] == "WebSearch,WebFetch"
    assert env["ARC_CLAUDE_ALLOWED_TOOLS"] == "WebSearch,WebFetch"
    for key in polluted:
        if key not in {"ARC_CLAUDE_TOOLS", "ARC_CLAUDE_ALLOWED_TOOLS"}:
            assert key not in env


def test_explicit_host_tool_inheritance_preserves_host_configuration(monkeypatch) -> None:
    monkeypatch.setenv("ARC_CODEX_PROFILE", "research-tools")
    monkeypatch.setenv("ARC_CLAUDE_MCP_CONFIG", "/tmp/research-mcp.json")

    env = pipeline_module._llm_runtime_env(
        allow_internet=True,
        inherit_host_tools=True,
    )

    assert env["ARC_PAPER_CLI_ACCESS"] == "full"
    assert env["ARC_LLM_INHERIT_HOST_TOOLS"] == "true"
    assert env["ARC_CODEX_PROFILE"] == "research-tools"
    assert env["ARC_CLAUDE_MCP_CONFIG"] == "/tmp/research-mcp.json"


def test_disabled_paper_cli_also_forces_bare_host_isolation(monkeypatch) -> None:
    monkeypatch.setenv("ARC_CODEX_PROFILE", "research-tools")
    monkeypatch.setenv("ARC_CLAUDE_MCP_CONFIG", "/tmp/research-mcp.json")

    env = pipeline_module._llm_runtime_env(
        allow_internet=False,
        inherit_host_tools=True,
        disable_paper_cli=True,
    )

    assert env["ARC_PAPER_CLI_ACCESS"] == "none"
    assert env["ARC_LLM_INHERIT_HOST_TOOLS"] == "false"
    assert env["ARC_CODEX_ENABLE_MCP"] == "false"
    assert env["ARC_CLAUDE_ALLOW_MCP"] == "false"
    assert "ARC_CODEX_PROFILE" not in env
    assert "ARC_CLAUDE_MCP_CONFIG" not in env


def test_guided_reference_policy_exposes_only_command_scoped_claude_bash() -> None:
    env = pipeline_module._llm_runtime_env(
        allow_internet=False,
        inherit_host_tools=False,
        paper_access_policy={
            "allowed_operations": [
                "artifact-read", "get-parsed-toc", "get-parsed-section",
            ],
            "authorized_source_ids": ["book"],
            "authorized_section_targets": [
                {"source_id": "book", "locator": "ch-2"},
            ],
        },
    )
    assert env["ARC_CLAUDE_TOOLS"] == "Bash"
    assert env["ARC_CLAUDE_ALLOWED_TOOLS"].split(",") == [
        "Bash(arc-paper-worker get-parsed-toc:*)",
        "Bash(arc-paper-worker get-parsed-section:*)",
        "Bash(arc-paper-worker policy-targets:*)",
        "Bash(arc-paper-worker artifact-read:*)",
    ]
    assert json.loads(env["ARC_PAPER_WORKER_ALLOWED_OPERATIONS_JSON"]) == [
        "artifact-read", "get-parsed-toc", "get-parsed-section",
    ]
    assert json.loads(env["ARC_PAPER_WORKER_ALLOWED_TARGETS_JSON"]) == {
        "book": {"sections": ["ch-2"]},
    }
    assert env["ARC_CLAUDE_ALLOW_MCP"] == "false"
    assert env["ARC_LLM_INHERIT_HOST_TOOLS"] == "false"


def test_guided_stateless_call_uses_controller_evidence_fallback(
    tmp_path: Path, monkeypatch,
) -> None:
    from arc_llm import EvidenceResponse
    from arc_companion.intent_guidance import (
        INTENT_GUIDANCE_VERSION,
        _target_catalog_metadata,
        _worker_payload_unvalidated,
    )

    artifact = {
        "schema_version": INTENT_GUIDANCE_VERSION,
        "semantic_input_sha256": "s" * 64,
        "user_intent_sha256": "u" * 64,
        "output_sha256": "o" * 64,
        "resolution_status": "resolved",
        "guidance": "Use the selected reference terminology.",
        "reference_targets": [{
            "source_id": "book", "locator": "ch-2", "purpose": "terms",
            "lanes": ["translation"],
        }],
        "reference_sources": [{
            "source_id": "book", "source_hash": "v1", "document_hash": "v1",
            "metadata": {}, "toc": [{"locator": "ch-2", "title": "Two"}],
        }],
    }
    artifact["target_catalog"] = _target_catalog_metadata(
        artifact["reference_targets"]
    )
    artifact["worker_payload"] = _worker_payload_unvalidated(artifact, lane=None)
    calls = []

    def fake_llm(prompt, **kwargs):
        calls.append((prompt, kwargs))
        if len(calls) == 1:
            return {
                "answer": "pending",
                "arc_evidence_requests": [{
                    "request_id": "r1", "operation": "get-parsed-section",
                    "arguments": {"source_id": "book", "locator": "ch-2"},
                    "reason": "terminology",
                }],
            }
        return {"answer": "done", "arc_evidence_requests": []}

    monkeypatch.setattr(
        pipeline_module, "resolve_worker_evidence_requests",
        lambda _artifact, requests, *, round_number, lane=None: tuple(
            EvidenceResponse(
                request.request_id, True, {
                    "content": "cached chapter "
                    + "x" * int(request.arguments.get("limit", 46 * 1024)),
                },
                provenance={"provider": "local-cache", "round": round_number},
            )
            for request in requests
        ),
    )
    round_audits = []
    result = _llm_call(
        fake_llm, "p" * 54_000, {
            "type": "object", "additionalProperties": False,
            "required": ["answer"], "properties": {"answer": {"type": "string"}},
        },
        options=BuildOptions(paper_id="local:x", project_dir=tmp_path),
        artifact_dir=tmp_path / "llm", call_label="guided", model_tier="medium",
        paper_access_policy=pipeline_module.worker_policy_descriptor(artifact),
        intent_guidance=artifact,
        intent_guidance_lane="translation",
        review_prompt_context={
            "stage": "section",
            "segment_ids": ["seg-1"],
            "target_limit_bytes": 55_296,
            "strict_limit_bytes": 61_440,
            "audit_sink": round_audits,
        },
    )
    assert result == {"answer": "done"}
    assert len(calls) == 2
    assert "CONTROLLER REFERENCE EVIDENCE ROUND" in calls[1][0]
    assert len(calls[1][0].encode("utf-8")) <= 61_440
    assert "arc_evidence_requests" in calls[0][1]["schema"]["properties"]
    assert round_audits[0]["budget_class"] == "evidence_headroom"
    assert round_audits[0]["strict_headroom_bytes"] >= 0


def test_guided_stateful_call_uses_controller_evidence_delta(
    monkeypatch,
) -> None:
    from arc_llm import EvidenceResponse

    class Outcome:
        def __init__(self, value):
            self.value = value

    artifact = {
        "schema_version": "arc.companion.intent-guidance.v1",
        "semantic_input_sha256": "s" * 64,
        "user_intent_sha256": "u" * 64,
        "output_sha256": "o" * 64,
        "resolution_status": "resolved",
        "guidance": "Use the selected terminology.",
        "reference_targets": [{
            "source_id": "book", "locator": "ch-2", "purpose": "terms",
            "lanes": ["translation"],
        }],
        "reference_sources": [{
            "source_id": "book", "source_hash": "v1", "document_hash": "v1",
            "metadata": {}, "toc": [{"locator": "ch-2", "title": "Two"}],
        }],
        "worker_payload": {
            "guidance": "Use the selected terminology.",
            "reference_targets": [{
                "source_id": "book", "locator": "ch-2", "purpose": "terms",
                "lanes": ["translation"],
            }],
        },
    }
    initial = Outcome({
        "answer": "pending",
        "arc_evidence_requests": [{
            "request_id": "r1", "operation": "get-parsed-section",
            "arguments": {"source_id": "book", "locator": "ch-2"},
            "reason": "terminology",
        }],
    })
    calls = []

    def call_round(prompt, schema, round_number):
        calls.append((prompt, schema, round_number))
        return Outcome({"answer": "done", "arc_evidence_requests": []})

    monkeypatch.setattr(
        pipeline_module, "resolve_worker_evidence_requests",
        lambda _artifact, requests, *, round_number, lane=None: tuple(
            EvidenceResponse(
                request.request_id, True, {"content": "cached chapter"},
                provenance={"provider": "local-cache", "lane": lane},
            )
            for request in requests
        ),
    )
    final_outcome, value = pipeline_module._complete_stateful_reference_evidence(
        initial, intent_guidance=artifact, lane="translation",
        worker_id="translation-1",
        schema={
            "type": "object", "additionalProperties": False,
            "required": ["answer"], "properties": {"answer": {"type": "string"}},
        },
        call_round=call_round,
    )

    assert final_outcome.value["answer"] == "done"
    assert value == {"answer": "done"}
    assert calls[0][2] == 1
    assert "CONTROLLER REFERENCE EVIDENCE ROUND" in calls[0][0]
    assert "arc_evidence_requests" in calls[0][1]["properties"]


def test_source_fingerprint_excludes_runtime_access_options(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    evidence = _evidence(bundle)
    enabled = BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path / "run")
    local_only = BuildOptions(
        paper_id=bundle.paper_id,
        project_dir=tmp_path / "run",
        allow_internet=False,
    )
    assert _fingerprint(bundle, enabled, evidence=evidence) == _fingerprint(
        bundle, local_only, evidence=evidence
    )
    inherited = BuildOptions(
        paper_id=bundle.paper_id,
        project_dir=tmp_path / "run",
        inherit_host_tools=True,
    )
    assert _fingerprint(bundle, enabled, evidence=evidence) == _fingerprint(
        bundle, inherited, evidence=evidence
    )


def test_source_fingerprint_excludes_context_recipe(tmp_path: Path, monkeypatch) -> None:
    bundle = _bundle(tmp_path)
    options = BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path / "run")
    evidence = _evidence(bundle)
    original = _fingerprint(bundle, options, evidence=evidence)

    monkeypatch.setattr(pipeline_module, "CONTEXT_SEGMENT_CHARS_PER_SOURCE", 2_999)
    assert _fingerprint(bundle, options, evidence=evidence) == original

    monkeypatch.setattr(pipeline_module, "CONTEXT_SEGMENT_CHARS_PER_SOURCE", 3_000)
    monkeypatch.setattr(pipeline_module, "CONTEXT_SELECTION_VERSION", "changed")
    assert _fingerprint(bundle, options, evidence=evidence) == original


def test_evidence_keeps_optional_source_diagnostics(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    warning = {
        "severity": "warning",
        "code": "citer_context_unavailable",
        "source": "arc-paper",
        "message": "Unable to load optional seed citers: offline",
    }
    bundle = SourceBundle(
        paper_id=bundle.paper_id,
        parsed=bundle.parsed,
        document=bundle.document,
        metadata=bundle.metadata,
        references=bundle.references,
        citers=[],
        diagnostics=(warning,),
    )

    evidence = _evidence(bundle)

    assert evidence["schema_version"] == "arc.companion.evidence.v3"
    assert evidence["citers"] == []
    assert evidence["diagnostics"] == [warning]


def test_global_protected_names_exclude_reference_and_citer_authors(tmp_path: Path) -> None:
    base = _bundle(tmp_path)
    bundle = SourceBundle(
        paper_id=base.paper_id,
        parsed=base.parsed,
        document=base.document,
        metadata={**base.metadata, "authors": [{"name": "Seed Author"}]},
        references=[{"title": "Prior", "authors": [{"name": "Tie Researcher"}]}],
        citers=[{"title": "Later", "authors": [{"name": "May Scholar"}]}],
    )
    glossary = {"entries": [{
        "source_term": "Feynman diagram",
        "target_term": "Feynman 图",
        "brief_explanation": "diagrammatic expansion",
        "protected_names": ["Feynman"],
    }]}

    names = _protected_names(bundle, glossary=glossary)

    assert "Seed Author" in names
    assert "Seed" in names
    assert "Author" in names
    assert "Feynman" in names
    assert "Tie Researcher" not in names
    assert "Tie" not in names
    assert "May Scholar" not in names
    assert "May" not in names


def test_protected_names_prefer_structured_metadata_over_combined_front_line(tmp_path: Path) -> None:
    base = _bundle(tmp_path)
    document = {
        **base.document,
        "front_matter": {
            **base.document["front_matter"],
            "authors": ["Xingang Chen 1,2 and Yi Wang 3,4"],
        },
    }
    bundle = SourceBundle(
        paper_id=base.paper_id,
        parsed=base.parsed,
        document=document,
        metadata={
            **base.metadata,
            "authors": [{"full_name": "Chen, Xingang"}, {"full_name": "Wang, Yi"}],
        },
        references=base.references,
        citers=base.citers,
    )

    names = _protected_names(bundle)

    assert "Chen, Xingang" in names
    assert "Wang, Yi" in names
    assert {"Chen", "Xingang", "Wang"} <= set(names)
    assert "Xingang Chen 1,2 and Yi Wang 3,4" not in names
    assert "and" not in names


def test_protected_names_clean_combined_front_line_when_metadata_authors_missing(tmp_path: Path) -> None:
    base = _bundle(tmp_path)
    document = {
        **base.document,
        "front_matter": {
            **base.document["front_matter"],
            "authors": ["Xingang Chen 1,2 and Yi Wang 3,4"],
        },
    }
    bundle = SourceBundle(
        paper_id=base.paper_id,
        parsed=base.parsed,
        document=document,
        metadata={"title": "No structured authors"},
        references=base.references,
        citers=base.citers,
    )

    names = _protected_names(bundle)

    assert "Xingang Chen" in names
    assert "Yi Wang" in names
    assert {"Xingang", "Chen", "Wang"} <= set(names)
    assert "and" not in names
    assert not any(any(character.isdigit() for character in name) for name in names)


def test_full_paper_context_is_bounded_navigable_and_excludes_raw_html(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    document = {
        **bundle.document,
        "blocks": [
            *bundle.document["blocks"],
            {
                "block_id": "b7", "type": "section", "title": "Distant physics",
                "raw_html": "<div>DO_NOT_EXPOSE_RAW_HTML</div>",
            },
            {
                "block_id": "b8", "type": "text", "text": "late-time transfer context",
                "preservation_html": "<span>DO_NOT_EXPOSE_PRESERVATION_HTML</span>",
            },
            {"block_id": "b9", "type": "equation", "equation_id": "eq-nine"},
        ],
        "equations": [
            *bundle.document.get("equations", []),
            {"id": "eq-nine", "tex": "E^2=p^2c^2+m^2c^4", "number": "(9)"},
        ],
    }
    by_id = {item["block_id"]: item for item in document["blocks"]}
    segment = {
        "segment_id": "seg-0001", "title": "Local", "start_block_id": "b1",
        "end_block_id": "b2", "block_ids": ["b1", "b2"],
    }

    context = _full_paper_context(document, segment, blocks_by_id=by_id, max_chars=4_000)
    serialized = json.dumps(context, ensure_ascii=False)

    assert context["schema_version"] == "arc.companion.full-paper-context.v3"
    assert any(item["block_id"] == "b7" for item in context["section_navigation"])
    equation = next(item for item in context["equation_navigation"] if item["block_id"] == "b9")
    assert equation == {
        "block_id": "b9", "number": "(9)", "location_block_id": "b7",
        "formula": "E^2=p^2c^2+m^2c^4",
    }
    assert "Distant physics" not in serialized
    assert "late-time transfer context" in serialized
    assert "DO_NOT_EXPOSE_RAW_HTML" not in serialized
    assert "DO_NOT_EXPOSE_PRESERVATION_HTML" not in serialized
    assert len(json.dumps(context, ensure_ascii=False, separators=(",", ":"))) <= 4_000


def test_segment_evidence_preserves_descriptor_and_hashes_selected_snippets() -> None:
    full_blocks = [{
        "block_id": "related-b1", "type": "text", "text": "transfer vertex field theory",
        "sha256": text_sha256("transfer vertex field theory"),
    }]
    paper = {
        "evidence_id": "prior-001", "relation": "prior", "paper_id": "arXiv:0001.0001",
        "title": "Transfer", "authors": ["A. Author"], "year": 2001,
        "citation_count": 10, "evidence_level": "full_text", "abstract": "",
        "blocks": full_blocks,
    }
    paper["source_descriptor"] = arc_cache_descriptor(
        paper_id=paper["paper_id"], title=paper["title"], authors=paper["authors"],
        year=paper["year"], evidence_level="full_text", content=full_blocks,
        document_hash="d" * 64,
    )
    segment = {"segment_id": "seg-0001", "block_ids": ["source-b1"]}
    by_id = {"source-b1": {"block_id": "source-b1", "type": "text", "text": "transfer vertex"}}

    selected = _evidence_for_segment(segment, by_id, {"related_papers": [paper]})

    assert selected["schema_version"] == "arc.companion.segment-evidence.v3"
    record = selected["papers"][0]
    assert record["source_descriptor"]["locator"]["document_hash"] == "d" * 64
    assert record["snippets"][0]["sha256"] == text_sha256(record["snippets"][0]["text"])
    assert validate_evidence_record(record) is record


def test_review_payload_contains_actual_bounded_context_evidence_and_descriptor(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    segment = {"segment_id": "seg-0001", "block_ids": ["b1", "b2"]}
    context_blocks = [{
        "block_id": "ctx-1", "text": "setup canonical field explanation",
        "sha256": text_sha256("setup canonical field explanation"),
    }]
    context = {
        "evidence_id": "context-001", "relation": "context", "paper_id": "isbn:one",
        "title": "Reference", "authors": [], "year": 2020, "evidence_level": "full_text",
        "abstract": "", "blocks": context_blocks,
        "context_role": "explanation_and_conceptual_connections_only",
    }
    context["source_descriptor"] = arc_cache_descriptor(
        paper_id="isbn:one", title="Reference", authors=[], year=2020,
        evidence_level="full_text", content=context_blocks, document_hash="e" * 64,
    )
    prompts = []

    def reviewer(prompt: str, **kwargs):
        prompts.append(prompt)
        return {"patches": [], "issues": []}

    _review(
        [segment],
        {"seg-0001": {"blocks": [
            {"block_id": "b1", "text": "设置"},
            {"block_id": "b2", "text": "令 x < y。"},
        ]}},
        {"seg-0001": {
            "explanation": "解释", "commentary": "伴读", "prior_work": [], "later_work": [],
            "context_claims": [], "evidence_ids": [], "key_points": [], "source_notes": [],
        }},
        document=bundle.document,
        glossary={"entries": []}, protected_names=[],
        evidence={"related_papers": [context]},
        options=BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path),
        llm=reviewer, checkpoint_dir=tmp_path / "checkpoint",
    )

    assert len(prompts) == 1
    assert '"context_evidence"' in prompts[0]
    assert '"source_descriptor"' in prompts[0]
    assert '"chars_per_source": 3000' in prompts[0]


def _related_work_record(
    evidence_id: str,
    *,
    relation: str,
    title: str,
    abstract: str,
    citation_count: int,
    paper_id: str,
    domain_role: str = "",
) -> dict:
    record = {
        "evidence_id": evidence_id,
        "relation": relation,
        "paper_id": paper_id,
        "title": title,
        "authors": [],
        "year": 2000,
        "citation_count": citation_count,
        "evidence_level": "abstract_only",
        "abstract": abstract,
        "blocks": [],
    }
    if domain_role:
        record["domain_role"] = domain_role
    record["source_descriptor"] = arc_cache_descriptor(
        paper_id=paper_id,
        title=title,
        authors=[],
        year=2000,
        evidence_level="abstract_only",
        content=abstract,
    )
    return record


def test_segment_related_work_is_empty_when_every_relevance_score_is_zero() -> None:
    segment = {"segment_id": "seg", "block_ids": ["source"]}
    by_id = {"source": {"block_id": "source", "text": "A localized conversion mechanism."}}
    papers = [
        _related_work_record(
            "prior-famous", relation="prior", title="WMAP observations",
            abstract="Microwave sky maps.", citation_count=100_000, paper_id="arXiv:0001.0001",
        ),
        _related_work_record(
            "later-famous", relation="later", title="Planck observations",
            abstract="Satellite data release.", citation_count=100_000, paper_id="arXiv:0001.0002",
        ),
        _related_work_record(
            "prior-icon", relation="prior", title="Maldacena correlators",
            abstract="Unrelated formal calculation.", citation_count=100_000, paper_id="arXiv:0001.0003",
        ),
    ]

    selected = _evidence_for_segment(segment, by_id, {"related_papers": papers})

    assert selected["papers"] == []


def test_long_generic_full_text_does_not_become_direct_related_work() -> None:
    segment = {"segment_id": "seg", "block_ids": ["source"]}
    by_id = {"source": {
        "block_id": "source",
        "text": (
            "We analyze a theoretical model and derive general observational results "
            "for parameters, fields, perturbations, correlations, and measurements."
        ),
        "inline_runs": [{
            "kind": "citation", "content": "Figure 1", "target_id": "S1.F1",
        }],
    }}
    generic_text = " ".join(
        "This paper analyzes a theoretical model with fields parameters perturbations "
        f"observational results correlations and measurements in topic{index}."
        for index in range(80)
    )
    paper = _related_work_record(
        "prior-famous", relation="prior", title="Broad overview and constraints",
        abstract="", citation_count=1_000_000, paper_id="arXiv:0001.0001",
    )
    paper["evidence_level"] = "full_text"
    paper["blocks"] = [{
        "block_id": "long-section", "text": generic_text,
        "sha256": text_sha256(generic_text),
    }]
    paper["source_descriptor"] = arc_cache_descriptor(
        paper_id=paper["paper_id"], title=paper["title"], authors=[], year=2000,
        evidence_level="full_text", content=paper["blocks"], document_hash="e" * 64,
    )

    selected = _evidence_for_segment(
        segment, by_id,
        {"bibliography": [{"id": "bib.real", "label": "[1]"}], "related_papers": [paper]},
    )

    assert selected["citation_targets"] == []
    assert selected["papers"] == []


def test_metadata_catalog_has_item_and_character_budgets() -> None:
    segment = {"segment_id": "seg", "block_ids": ["source"]}
    by_id = {"source": {"block_id": "source", "text": "specific conversion signal"}}
    references = [{
        "arxiv_id": f"0001.{index:04d}",
        "title": f"Candidate {index} specific conversion",
        "abstract": "x" * 5_000,
        "citation_count": index,
    } for index in range(100)]

    selected = _evidence_for_segment(
        segment, by_id, {"references": references, "related_papers": []},
    )

    catalog = selected["reference_catalog"]
    assert len(catalog) <= 40
    assert len(json.dumps(catalog, ensure_ascii=False, separators=(",", ":"))) <= 12_000


def test_terms_scattered_across_long_full_text_do_not_create_direct_relevance() -> None:
    segment = {"segment_id": "seg", "block_ids": ["source"]}
    by_id = {"source": {
        "block_id": "source",
        "text": "Spectator conversion bispectrum transfer vertex coupling.",
    }}
    paper = _related_work_record(
        "prior-scattered", relation="prior", title="A broad review",
        abstract="An unrelated overview.", citation_count=1, paper_id="arXiv:0001.0042",
    )
    paper["evidence_level"] = "full_text"
    paper["blocks"] = [
        {"block_id": "p1", "text": "Spectator telescope calibration and noise."},
        {"block_id": "p2", "text": "Conversion of detector units in a catalog."},
        {"block_id": "p3", "text": "Bispectrum appears in an unrelated appendix."},
        {"block_id": "p4", "text": "Transfer scheduling for archived observations."},
        {"block_id": "p5", "text": "Vertex indexing in a database table."},
    ]

    selected = _evidence_for_segment(segment, by_id, {"related_papers": [paper]})

    assert selected["papers"] == []


def test_direct_relevance_beats_citations_and_domain_membership() -> None:
    segment = {"segment_id": "seg", "block_ids": ["source"]}
    by_id = {"source": {"block_id": "source", "text": "Curvaton decay transfer produces isocurvature."}}
    papers = [
        _related_work_record(
            "prior-high", relation="prior", title="Precision satellite constraints",
            abstract="A broad observational catalog.", citation_count=500_000,
            paper_id="arXiv:0001.0001", domain_role="foundation",
        ),
        _related_work_record(
            "prior-direct", relation="prior", title="Curvaton decay transfer",
            abstract="Isocurvature transfer from curvaton decay.", citation_count=3,
            paper_id="arXiv:0001.0009",
        ),
    ]

    selected = _evidence_for_segment(segment, by_id, {"related_papers": papers})

    assert [item["evidence_id"] for item in selected["papers"]] == ["prior-direct"]


def test_selected_candidate_can_be_bound_in_production_claim_schema() -> None:
    segment = {"segment_id": "seg", "block_ids": ["source"]}
    by_id = {"source": {
        "block_id": "source", "text": "spectator conversion creates a bispectrum",
    }}
    text = "The spectator conversion creates a bispectrum."
    record = {
        "evidence_id": "prior-direct", "relation": "prior",
        "paper_id": "arXiv:0001.0009", "title": "Spectator conversion bispectrum",
        "url": "https://arxiv.org/abs/0001.0009",
        "authors": [], "year": 2000, "citation_count": 1,
        "evidence_level": "full_text", "abstract": "",
        "blocks": [{"block_id": "S2", "text": text, "sha256": text_sha256(text)}],
    }
    record["source_descriptor"] = arc_cache_descriptor(
        paper_id=record["paper_id"], title=record["title"], authors=[], year=2000,
        evidence_level="full_text", content=record["blocks"], document_hash="f" * 64,
    )

    selected = _evidence_for_segment(segment, by_id, {"related_papers": [record]})
    chosen = selected["papers"][0]
    annotation = {
        "explanation": "Explanation", "commentary": "Commentary",
        "commentary_sources": [],
        "prior_work": [{
            "text": "The prior paper studies this conversion.",
            "sources": [{
                "title": chosen["title"],
                "url": chosen["url"],
                "locator": chosen["snippets"][0]["block_id"],
            }],
        }],
        "later_work": [],
    }

    jsonschema.validate(annotation, ANNOTATION_SCHEMA)


def test_whole_paper_soft_reuse_can_leave_empty_but_exact_citation_is_exempt() -> None:
    segment = {"segment_id": "seg", "block_ids": ["source"]}
    by_id = {"source": {
        "block_id": "source", "text": "spectator conversion creates a bispectrum",
    }}
    record = _related_work_record(
        "prior-repeat", relation="prior", title="Spectator conversion bispectrum",
        abstract="Spectator conversion creates a bispectrum.", citation_count=1,
        paper_id="arXiv:0001.0009",
    )
    usage = {"counts": {}, "topics": []}
    outputs = [
        _evidence_for_segment(
            segment, by_id, {"related_papers": [record]}, usage_state=usage,
        )["papers"]
        for _ in range(8)
    ]

    assert outputs[0]
    assert outputs[-1] == []

    citation = _inline_run("citation", "[9]", 2)
    citation["target_id"] = "bib.bib9"
    by_id["source"]["inline_runs"] = [citation]
    exact = _evidence_for_segment(
        segment, by_id, {
            "bibliography": [{"id": "bib.bib9", "arxiv_id": "0001.0009"}],
            "related_papers": [record],
        }, usage_state=usage,
    )
    assert [item["evidence_id"] for item in exact["papers"]] == ["prior-repeat"]


def test_exact_bibliography_target_is_first_and_each_relation_is_capped_at_three() -> None:
    citation = _inline_run("citation", "[9]", 2)
    citation["target_id"] = "bib.bib9"
    citation["href"] = "#bib.bib9"
    internal = _inline_run("citation", "Figure 1", 3)
    internal["target_id"] = "S1.F1"
    internal["href"] = "#S1.F1"
    segment = {"segment_id": "seg", "block_ids": ["source"]}
    by_id = {"source": {
        "block_id": "source",
        "text": "Curvaton decay transfer isocurvature.",
        "inline_runs": [_inline_run("text", "See ", 1), citation, internal],
    }}
    papers = [
        _related_work_record(
            f"prior-{index}", relation="prior", title=f"Curvaton decay transfer {index}",
            abstract="Curvaton isocurvature transfer.", citation_count=100 - index,
            paper_id=f"arXiv:0001.000{index}",
        )
        for index in range(1, 5)
    ]
    papers.append(_related_work_record(
        "prior-exact", relation="prior", title="A specifically cited construction",
        abstract="The cited construction.", citation_count=0, paper_id="arXiv:0001.0009",
    ))
    papers.extend(
        _related_work_record(
            f"later-{index}", relation="later", title=f"Curvaton decay transfer extension {index}",
            abstract="A later curvaton isocurvature transfer extension.", citation_count=index,
            paper_id=f"arXiv:2501.000{index}",
        )
        for index in range(1, 5)
    )
    evidence = {
        "bibliography": [{"id": "bib.bib9", "label": "[9]", "arxiv_id": "0001.0009"}],
        "related_papers": papers,
    }

    selected = _evidence_for_segment(segment, by_id, evidence)

    assert selected["citation_targets"] == [{
        "id": "bib.bib9", "label": "[9]", "arxiv_id": "0001.0009",
    }]
    assert selected["papers"][0]["evidence_id"] == "prior-exact"
    assert sum(item["relation"] == "prior" for item in selected["papers"]) == 3
    assert sum(item["relation"] == "later" for item in selected["papers"]) == 3


def test_ninth_metadata_candidate_can_outrank_first_eight() -> None:
    segment = {"segment_id": "seg", "block_ids": ["source"]}
    by_id = {"source": {"block_id": "source", "text": "Spectator conversion bispectrum."}}
    records = [
        {
            "arxiv_id": f"0001.{index:04d}",
            "title": "Generic survey",
            "abstract": "Broad measurements.",
            "citation_count": 100_000 - index,
        }
        for index in range(1, 9)
    ] + [{
        "arxiv_id": "0001.0009",
        "title": "Spectator conversion bispectrum",
        "abstract": "A bispectrum from spectator conversion.",
        "citation_count": 1,
    }]
    papers = [
        _related_work_record(
            f"prior-{index:03d}", relation="prior", title=item["title"],
            abstract=item["abstract"], citation_count=item["citation_count"],
            paper_id=f"arXiv:{item['arxiv_id']}",
        )
        for index, item in enumerate(records, 1)
    ]

    selected = _evidence_for_segment(
        segment, by_id, {"references": records, "related_papers": papers},
    )

    assert selected["papers"][0]["evidence_id"] == "prior-009"
    assert selected["reference_catalog"][0]["arxiv_id"] == "0001.0009"


def test_invalid_review_patch_fails_without_publishing_pdf(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    fake = FakeLLM()
    original = fake.__call__
    final_prompts = []

    def bad_llm(prompt: str, **kwargs):
        if kwargs["call_label"] == "companion-final-review":
            final_prompts.append(prompt)
            return {"patches": [{"segment_id": "source-b1", "commentary": "tamper", "reason": "bad"}], "issues": []}
        return original(prompt, **kwargs)

    compiled: list[Path] = []

    def preview_compiler(tex_path: Path, pdf_path: Path) -> None:
        compiled.append(tex_path)
        pdf_path.write_bytes(b"%PDF-1.7 fixture")

    result = build_companion(
        BuildOptions(paper_id=bundle.paper_id, project_dir=tmp_path / "bad"),
        source_loader=lambda *args, **kwargs: bundle,
        llm=bad_llm,
        compiler=preview_compiler,
        pdf_validator=lambda _: {},
    )
    assert not result["ok"]
    assert "invalid or duplicate annotation patch" in result["error"]["message"]
    assert final_prompts and '"source_blocks"' in final_prompts[0]
    assert json.loads((tmp_path / "bad" / "state.json").read_text())["status"] == "failed"
    assert len(compiled) == 1
    assert (tmp_path / "bad" / "arXiv-1234.5678_companion_zh-CN_first_round_preview.pdf").is_file()
    assert not (tmp_path / "bad" / "arXiv-1234.5678_companion_zh-CN.pdf").exists()


def test_generation_failure_drains_both_lanes_and_retry_only_runs_missing_segments(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    base = FakeLLM()
    failed_generation_calls: list[str] = []

    def failing_llm(prompt: str, **kwargs):
        label = str(kwargs["call_label"])
        if label.startswith(("companion-translation-", "companion-annotation-")):
            failed_generation_calls.append(label)
            if label.endswith("seg-0001"):
                raise RuntimeError(f"intentional early failure: {label}")
            if label.startswith("companion-annotation-"):
                return {
                    "explanation": "later segment survived", "prior_work": [], "later_work": [],
                    "commentary": "later segment survived", "evidence_ids": [],
                    "key_points": [], "source_notes": [],
                }
            raise AssertionError(label)
        return base(prompt, **kwargs)

    project = tmp_path / "drained-lanes"
    first = build_companion(
        BuildOptions(paper_id=bundle.paper_id, project_dir=project, workers=2),
        source_loader=lambda *args, **kwargs: bundle,
        llm=failing_llm,
        compiler=lambda *_: (_ for _ in ()).throw(AssertionError("must not compile")),
        pdf_validator=lambda _: {},
    )

    assert not first["ok"]
    assert "translation lane failed" in first["error"]["message"]
    assert "annotation lane failed" in first["error"]["message"]
    checkpoint = Path(json.loads((project / "state.json").read_text())["checkpoint_dir"])
    completed_annotations = {
        json.loads(path.read_text(encoding="utf-8"))["segment_id"]
        for path in (checkpoint / "annotations").glob("*.json")
    }
    completed_translations = {
        json.loads(path.read_text(encoding="utf-8"))["segment_id"]
        for path in (checkpoint / "translations").glob("*.json")
    }
    # A terminal failure opens the shared generation circuit immediately, but
    # a successor that already crossed the provider boundary must drain and
    # remain reusable.  The per-lane workers race for the shared permits, so
    # either outcome is valid; the checkpoint must agree with the calls that
    # actually started.
    annotation_seg2_started = (
        "companion-annotation-seg-0002" in failed_generation_calls
    )
    assert completed_annotations == ({"seg-0002"} if annotation_seg2_started else set())
    # The non-translatable second segment is a deterministic empty artifact and
    # does not cross the provider circuit.
    assert completed_translations == {"seg-0002"}

    retry_generation_calls: list[str] = []

    def retry_llm(prompt: str, **kwargs):
        label = str(kwargs["call_label"])
        if label.startswith("companion-translation-"):
            retry_generation_calls.append(label)
            return {"blocks": [
                {"block_id": "b1", "text": "设定"},
                {"block_id": "b2", "text": "令 x < y 且 y > 0。"},
            ]}
        if label.startswith("companion-annotation-"):
            retry_generation_calls.append(label)
            return {
                "explanation": "retry explanation", "prior_work": [], "later_work": [],
                "commentary": "retry commentary", "evidence_ids": [],
                "key_points": [], "source_notes": [],
            }
        return base(prompt, **kwargs)

    def compiler(tex_path: Path, pdf_path: Path) -> None:
        pdf_path.write_bytes(b"%PDF-1.7 fixture")

    second = build_companion(
        BuildOptions(paper_id=bundle.paper_id, project_dir=project, workers=2),
        source_loader=lambda *args, **kwargs: bundle,
        llm=retry_llm,
        compiler=compiler,
        pdf_validator=lambda path: {"bytes": path.stat().st_size},
    )

    assert second["ok"], second
    expected_retry_calls = [
        "companion-annotation-seg-0001",
        "companion-translation-seg-0001",
    ]
    if not annotation_seg2_started:
        expected_retry_calls.append("companion-annotation-seg-0002")
    assert sorted(retry_generation_calls) == sorted(expected_retry_calls)


def test_invalid_segmentation_is_not_cached(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)

    def invalid_llm(prompt: str, **kwargs):
        assert kwargs["call_label"].startswith("companion-segmentation-w-")
        return {"cut_after_ordinals": [99]}

    project = tmp_path / "invalid-segments"
    result = build_companion(
        BuildOptions(paper_id=bundle.paper_id, project_dir=project),
        source_loader=lambda *args, **kwargs: bundle,
        llm=invalid_llm,
        compiler=lambda *_: (_ for _ in ()).throw(AssertionError("must not compile")),
        pdf_validator=lambda _: {},
    )
    assert not result["ok"]
    assert result["error"]["code"] == "companion_segmentation_failed"
    diagnostic = next(
        item
        for item in result["meta"]["diagnostics"]
        if item.get("code") == "companion_segmentation_failed"
    )
    assert diagnostic["source"] == "arc-companion"
    assert diagnostic["context"] == {
        "phase": "window",
        "window_id": "w-0001",
        "start_ordinal": 1,
            "end_ordinal": 5,
        "attempt": 3,
        "refinement": False,
    }
    state = json.loads((project / "state.json").read_text(encoding="utf-8"))
    assert diagnostic in state["diagnostics"]
    assert not list(project.rglob("segmentation.json"))


def test_json_emit_warns_about_segmentation_failure_on_stderr(capsys) -> None:
    result = {
        "ok": False,
        "data": None,
        "error": {
            "code": "companion_segmentation_failed",
            "message": "window w-0002 failed after 3 attempts",
        },
        "errors": [],
        "meta": {
            "diagnostics": [{
                "severity": "error",
                "code": "companion_segmentation_failed",
                "source": "arc-companion",
                "message": "window w-0002 failed after 3 attempts",
                "context": {"window_id": "w-0002", "attempt": 3},
            }],
        },
    }

    _emit(result, json_output=True)

    captured = capsys.readouterr()
    assert json.loads(captured.out) == result
    assert captured.err == "WARNING: window w-0002 failed after 3 attempts\n"


def test_complete_resume_requires_matching_output_hashes(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    fake = FakeLLM()
    compile_count = 0

    def compiler(tex_path: Path, pdf_path: Path) -> None:
        nonlocal compile_count
        compile_count += 1
        pdf_path.write_bytes(b"%PDF-1.7 fixture")

    project = tmp_path / "hashed-resume"
    first = build_companion(
        BuildOptions(paper_id=bundle.paper_id, project_dir=project),
        source_loader=lambda *args, **kwargs: bundle,
        llm=fake,
        compiler=compiler,
        pdf_validator=lambda path: {"bytes": path.stat().st_size},
    )
    assert first["ok"]
    state = json.loads((project / "state.json").read_text())
    assert state["output_pdf_sha256"] and state["output_tex_sha256"]
    Path(state["output_pdf"]).write_bytes(b"%PDF tampered")

    second = build_companion(
        BuildOptions(paper_id=bundle.paper_id, project_dir=project),
        source_loader=lambda *args, **kwargs: bundle,
        llm=fake,
        compiler=compiler,
        pdf_validator=lambda path: {"bytes": path.stat().st_size},
    )
    assert second["ok"]
    assert second["meta"]["resumed"] is False
    assert compile_count == 4


def test_pdf_failure_preserves_reviewed_content_before_typesetting(
    tmp_path: Path, monkeypatch,
) -> None:
    bundle = _bundle(tmp_path)
    project = tmp_path / "pdf-failure"
    publish_pdf = pipeline_module._publish_pdf_artifact
    publish_calls = 0

    def fail_final_pdf(*args, **kwargs):
        nonlocal publish_calls
        publish_calls += 1
        if publish_calls == 2:
            raise RuntimeError("final PDF failed")
        preview = publish_pdf(*args, **kwargs)
        state = json.loads((project / "state.json").read_text())
        stale = Path(state["checkpoint_dir"]) / "reader-final.json"
        stale.write_text('{"schema_version":"stale"}', encoding="utf-8")
        return preview

    monkeypatch.setattr(pipeline_module, "_publish_pdf_artifact", fail_final_pdf)

    result = build_companion(
        BuildOptions(paper_id=bundle.paper_id, project_dir=project),
        source_loader=lambda *args, **kwargs: bundle,
        llm=FakeLLM(),
        compiler=lambda _tex, pdf: pdf.write_bytes(b"%PDF fixture"),
        pdf_validator=lambda path: {"bytes": path.stat().st_size},
    )

    assert not result["ok"]
    assert publish_calls == 2
    checkpoint = Path(result["meta"]["state"]["checkpoint_dir"])
    assert (checkpoint / "reader-final.json").is_file()
    state = result["meta"]["state"]
    digest = state["published"]["content_sha256"]
    content_path = project / ".arc-companion" / "objects" / "reader-content" / f"{digest}.json"
    assert content_path.is_file()
    assert json.loads(content_path.read_text())["content_sha256"] == digest


def test_final_pdf_partial_replace_failure_preserves_published_revision(
    tmp_path: Path, monkeypatch,
) -> None:
    bundle = _bundle(tmp_path)
    project = tmp_path / "immutable-final-pdf"
    options = BuildOptions(paper_id=bundle.paper_id, project_dir=project)
    build_kwargs = {
        "source_loader": lambda *args, **kwargs: bundle,
        "compiler": lambda _tex, pdf: pdf.write_bytes(b"%PDF fixture"),
        "pdf_validator": lambda path: {"bytes": path.stat().st_size},
    }
    first = build_companion(options, llm=FakeLLM(), **build_kwargs)
    assert first["ok"] is True
    old_state = json.loads((project / "state.json").read_text())
    old_pdf = Path(old_state["published"]["pdf"]["output_pdf"])
    old_bytes = old_pdf.read_bytes()

    real_replace = pipeline_module._publish_artifact_replace
    final_replacements = 0

    def fail_second_final_replace(source: Path, target: Path) -> None:
        nonlocal final_replacements
        if ".arc-companion/renders/pdf/" in target.as_posix():
            final_replacements += 1
            if final_replacements == 2:
                raise OSError("injected final publish failure")
        real_replace(source, target)

    monkeypatch.setattr(
        pipeline_module, "_publish_artifact_replace", fail_second_final_replace,
    )
    rerender_state = dict(old_state)
    rerender_state["final_render_version"] = "stale-render-recipe"
    (project / "state.json").write_text(json.dumps(rerender_state), encoding="utf-8")
    failed = build_companion(
        BuildOptions(paper_id=bundle.paper_id, project_dir=project),
        llm=FakeLLM(), **build_kwargs,
    )

    assert failed["ok"] is False
    assert final_replacements == 2
    failed_state = json.loads((project / "state.json").read_text())
    assert failed_state["published"]["pdf"] == old_state["published"]["pdf"]
    assert old_pdf.read_bytes() == old_bytes


def test_non_failed_state_clears_stale_error(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    _state(path, status="failed", error="old failure")

    state = _state(path, status="segmenting")

    assert state["status"] == "segmenting"
    assert state["schema_version"] == "arc.companion.state.v3"
    assert "error" not in state


def test_state_change_of_fingerprint_clears_bound_artifacts(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    _state(
        path,
        status="complete",
        fingerprint="old-fingerprint",
        checkpoint_dir="/run/checkpoints/old-fingerprint",
        segment_count=12,
        preview_pdf="/run/old-preview.pdf",
        preview_validation_sha256="old-preview-validation",
        first_wave_preview_version="old-preview-version",
        output_pdf="/run/old-output.pdf",
        output_pdf_sha256="old-output-hash",
        source_manifest_path="/run/old-manifest.json",
        validation_path="/run/old-validation.json",
        final_render_version="old-render-version",
        output_html="/run/reader/index.html",
        output_html_sha256="old-html-hash",
        reader_snapshot_path="/run/reader/snapshot.json",
        reader_snapshot_sha256="old-snapshot-hash",
        web_manifest_path="/run/reader/manifest.json",
        web_manifest_sha256="old-web-manifest-hash",
        web_render_version="old-web-version",
        web={"ok": True, "snapshot_revision": "old"},
    )

    state = _state(
        path,
        status="failed",
        fingerprint="new-fingerprint",
        checkpoint_dir="/run/checkpoints/new-fingerprint",
        error="new build failed",
    )

    assert state["fingerprint"] == "new-fingerprint"
    assert state["checkpoint_dir"] == "/run/checkpoints/new-fingerprint"
    assert state["error"] == "new build failed"
    assert "segment_count" not in state
    assert not any(key.startswith("preview_") for key in state)
    assert "first_wave_preview_version" not in state
    assert not any(key.startswith("output_") for key in state)
    assert not any(key.startswith("source_manifest_") for key in state)
    assert not any(key.startswith("validation_") for key in state)
    assert "final_render_version" not in state
    assert "reader_snapshot_path" not in state
    assert "reader_snapshot_sha256" not in state
    assert "web_manifest_path" not in state
    assert "web_manifest_sha256" not in state
    assert "web_render_version" not in state
    assert "web" not in state


def test_failure_before_fingerprint_clears_prior_fingerprint_state(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    _state(
        path,
        status="preview_ready",
        fingerprint="old-fingerprint",
        checkpoint_dir="/run/checkpoints/old-fingerprint",
        segment_count=12,
        preview_pdf="/run/old-preview.pdf",
        preview_segment_ids=["seg-0001"],
        first_wave_preview_version="old-preview-version",
    )

    state = _state(path, status="failed", error="source loading failed")

    assert state["status"] == "failed"
    assert state["error"] == "source loading failed"
    assert "fingerprint" not in state
    assert "checkpoint_dir" not in state
    assert "segment_count" not in state
    assert not any(key.startswith("preview_") for key in state)
    assert "first_wave_preview_version" not in state


def test_loading_source_hides_prior_fingerprint_state(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    _state(
        path,
        status="complete",
        fingerprint="old-fingerprint",
        checkpoint_dir="/run/checkpoints/old-fingerprint",
        preview_pdf="/run/old-preview.pdf",
        output_pdf="/run/old-output.pdf",
    )

    state = _state(path, status="loading_source", paper_id="new-paper")

    assert state["status"] == "loading_source"
    assert state["paper_id"] == "new-paper"
    assert "fingerprint" not in state
    assert "checkpoint_dir" not in state
    assert "preview_pdf" not in state
    assert "output_pdf" not in state


def test_same_fingerprint_state_update_keeps_reusable_artifacts(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    _state(
        path,
        status="preview_ready",
        fingerprint="same-fingerprint",
        checkpoint_dir="/run/checkpoints/same-fingerprint",
        preview_pdf="/run/preview.pdf",
        first_wave_preview_version="preview-version",
    )

    state = _state(path, status="failed", fingerprint="same-fingerprint", error="retry")

    assert state["fingerprint"] == "same-fingerprint"
    assert state["checkpoint_dir"] == "/run/checkpoints/same-fingerprint"
    assert state["preview_pdf"] == "/run/preview.pdf"
    assert state["first_wave_preview_version"] == "preview-version"


def test_explicit_language_state_clears_default_language_notice(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    _state(
        path,
        status="preview_ready",
        fingerprint="same-fingerprint",
        notice=LANGUAGE_NOTICE,
    )

    state = _state(
        path,
        status="loading_source",
        paper_id="same-paper",
        notice=None,
    )

    assert state["status"] == "loading_source"
    assert state["paper_id"] == "same-paper"
    assert "notice" not in state
