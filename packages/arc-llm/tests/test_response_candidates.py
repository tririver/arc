from __future__ import annotations

import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from arc_llm.response_candidates import (
    LLMResponseCandidateConflict,
    LLMResponseCandidateReceiptError,
    material_from_claude,
    material_from_codex,
    material_from_kimi,
    persist_selection_receipt,
    select_response_candidate,
)
from arc_llm import response_candidates as candidate_module
from arc_llm.runner import run_json_result
from arc_llm.schema_cache import canonical_json, sha256_text
from arc_llm.usage import LLMProviderResponse, ResponseCandidateMaterial
from arc_llm.providers.base import LLMSubmissionState
from arc_llm.providers.base import LLMFailureCategory, LLMWorkerError
from arc_llm.call_checkpoint import LLMCallNeedsSupervision, LLMCallRetryExhausted


RESULT_SCHEMA = {
    "type": "object",
    "required": ["items"],
    "properties": {"items": {"type": "array", "items": {"type": "string"}}},
    "additionalProperties": False,
}


def _response(*material: ResponseCandidateMaterial) -> LLMProviderResponse[dict]:
    return LLMProviderResponse({"items": []}, candidate_material=material)


def _material(position: int, value=None, *, text=None, supersedes=()):
    return ResponseCandidateMaterial(
        source=f"test.{position}",
        protocol_position=position,
        value=value,
        text=text,
        event_id=f"event-{position}",
        supersedes=tuple(supersedes),
    )


def test_last_substantive_ignores_later_valid_empty_and_merges_equivalents():
    call_record = {"schema_version": "ignored"}
    selection = select_response_candidate(
        _response(
            _material(0, {"items": []}),
            _material(1, {"items": ["kept"], "arc_llm_call_record": call_record}),
            _material(2, {"items": ["kept"]}),
            _material(3, {"items": []}),
        ),
        schema=RESULT_SCHEMA,
        checkpoint_identity="identity",
    )

    assert selection.response.value == {"items": ["kept"]}
    assert selection.receipt["decision"] == "last_substantive"
    substantive = [item for item in selection.receipt["candidates"] if item["substantive"]]
    assert len(substantive) == 1
    assert [origin["protocol_position"] for origin in substantive[0]["origins"]] == [1, 2]


def test_barthes_section_review_regression_keeps_seven_findings_before_empty():
    fixture = Path(__file__).parent / "fixtures" / "barthes_section_review_11_candidates.jsonl"
    events = [json.loads(line) for line in fixture.read_text(encoding="utf-8").splitlines()]
    assert len(events) == 6
    assert all(
        event["value"] == {"findings": [], "patches": [], "reviewed_segment_ids": []}
        for event in events[:4]
    )
    substantive = events[4]["value"]
    assert sha256_text(canonical_json(substantive)) == (
        "6ebfd6d820ef660e1318ab1a3a5090b3c40b798f41e2d0b2c656598fcf1983cb"
    )
    assert len(substantive["findings"]) == len(substantive["patches"]) == 7
    assert len(substantive["reviewed_segment_ids"]) == 3
    assert events[-1]["source"] == "codex.output_last_message"
    assert events[-1]["value"] == {
        "findings": [], "patches": [], "reviewed_segment_ids": []
    }
    schema = {
        "type": "object",
        "required": ["findings", "patches", "reviewed_segment_ids"],
        "properties": {
            "findings": {"type": "array"},
            "patches": {"type": "array"},
            "reviewed_segment_ids": {"type": "array"},
        },
        "additionalProperties": False,
    }
    response = LLMProviderResponse(
        events[-1]["value"],
        candidate_material=tuple(
            ResponseCandidateMaterial(
                source=event["source"],
                protocol_position=index,
                value=event["value"],
                event_id=event["event_id"],
            )
            for index, event in enumerate(events)
        ),
    )
    selection = select_response_candidate(response, schema=schema, checkpoint_identity="barthes")
    assert selection.response.value == substantive
    assert selection.receipt["decision"] == "last_substantive"


def test_all_empty_selects_last_valid_and_invalid_rich_does_not_outrank_it():
    selection = select_response_candidate(
        _response(
            _material(0, {"items": [1, 2, 3]}),
            _material(1, {"items": []}),
            _material(2, {"items": []}),
        ),
        schema=RESULT_SCHEMA,
        checkpoint_identity=None,
    )

    assert selection.response.value == {"items": []}
    assert selection.receipt["decision"] == "last_valid_empty"
    assert selection.receipt["selected_ordinal"] is not None


def test_only_complete_objects_enter_selection_in_character_order():
    text = (
        'prefix {"items":["first"],"quoted":"} not a boundary"} '
        '```json\n{"items":["second"]}\n``` tail {"items":["truncated"]'
    )
    schema = {
        "type": "object",
        "required": ["items"],
        "properties": {
            "items": {"type": "array", "items": {"type": "string"}},
            "quoted": {"type": "string"},
        },
        "additionalProperties": False,
    }
    with pytest.raises(LLMResponseCandidateConflict) as caught:
        selection = select_response_candidate(
            _response(_material(0, text=text)), schema=schema, checkpoint_identity=None
        )
        if selection.conflict:
            raise selection.conflict

    assert len(caught.value.candidates) == 2


def test_malformed_open_object_does_not_hide_later_complete_object():
    selection = select_response_candidate(
        _response(_material(0, text='broken {"unfinished": 1 then {"items":["kept"]}')),
        schema=RESULT_SCHEMA,
        checkpoint_identity=None,
    )
    assert selection.response.value == {"items": ["kept"]}
    assert selection.receipt["decision"] == "last_substantive"


def test_braces_inside_quoted_prose_are_not_candidates():
    selection = select_response_candidate(
        _response(
            _material(
                0,
                text='example "{\\"items\\":[\\"fake\\"]}" then {"items":["real"]}',
            )
        ),
        schema=RESULT_SCHEMA,
        checkpoint_identity=None,
    )
    assert selection.response.value == {"items": ["real"]}
    assert len(selection.receipt["candidates"]) == 1


def test_valid_outer_object_suppresses_nested_object_candidate():
    schema = {
        "type": "object",
        "required": ["items"],
        "properties": {"items": {"type": "array"}},
    }
    outer = {"items": [{"items": ["nested"]}]}
    selection = select_response_candidate(
        _response(_material(0, text=json.dumps(outer))),
        schema=schema,
        checkpoint_identity=None,
    )
    assert selection.response.value == outer
    assert len(selection.receipt["candidates"]) == 1


def test_anyof_uses_maximum_mass_across_all_valid_branches():
    schema = {
        "type": "object",
        "required": ["result"],
        "properties": {
            "result": {
                "anyOf": [
                    {
                        "type": "object",
                        "required": ["empty"],
                        "properties": {"empty": {"type": "array"}},
                    },
                    {
                        "type": "object",
                        "required": ["payload"],
                        "properties": {"payload": {"type": "string"}},
                    },
                ]
            }
        },
    }
    value = {"result": {"empty": [], "payload": "x"}}
    selection = select_response_candidate(
        _response(_material(0, value)), schema=schema, checkpoint_identity=None
    )
    assert selection.response.value == value
    assert selection.receipt["decision"] == "last_substantive"


def test_nonfinite_json_constants_are_not_complete_candidates():
    selection = select_response_candidate(
        _response(_material(0, text='{"items":[NaN,Infinity]}')),
        schema={"type": "object"},
        checkpoint_identity=None,
    )
    assert selection.receipt["decision"] == "no_schema_valid_candidate"


def test_large_candidate_stream_fails_before_unbounded_json_parsing(monkeypatch):
    real_loads = candidate_module._loads_strict
    calls = 0

    def counted(text):
        nonlocal calls
        calls += 1
        return real_loads(text)

    monkeypatch.setattr(candidate_module, "_loads_strict", counted)
    with pytest.raises(LLMResponseCandidateReceiptError, match="exceed"):
        select_response_candidate(
            _response(_material(0, text="{}" * 10_000)),
            schema={"type": "object"},
            checkpoint_identity=None,
        )
    assert calls <= 258  # one whole-text probe plus the bounded range probes
    with pytest.raises(LLMResponseCandidateReceiptError, match="exceed"):
        select_response_candidate(
            LLMProviderResponse(
                {},
                candidate_material=tuple(
                    ResponseCandidateMaterial(
                        "generic.provider_value", index, value={"index": index}
                    )
                    for index in range(255)
                ) + (
                    ResponseCandidateMaterial(
                        "generic.provider_value", 255,
                        value={"index": 255}, text='{"index":256}'
                    ),
                ),
            ),
            schema={"type": "object"},
            checkpoint_identity=None,
        )


def test_non_equivalent_substantive_candidates_require_supervision_hashes_only():
    selection = select_response_candidate(
        _response(_material(0, {"items": ["a"]}), _material(1, {"items": ["b"]})),
        schema=RESULT_SCHEMA,
        checkpoint_identity="identity",
    )

    assert selection.receipt["decision"] == "ambiguous_substantive_conflict"
    assert selection.receipt["selected_sha256"] is None
    assert len(selection.receipt["conflict_hashes"]) == 2
    assert selection.conflict is not None
    message = str(selection.conflict)
    assert "\"items\"" not in message
    assert all(digest in message for digest in selection.receipt["conflict_hashes"])


def test_explicit_protocol_supersession_selects_terminal_value():
    selection = select_response_candidate(
        _response(
            _material(4, {"items": ["draft"]}),
            _material(7, {"items": ["final"]}, supersedes=(4,)),
        ),
        schema=RESULT_SCHEMA,
        checkpoint_identity=None,
    )

    assert selection.conflict is None
    assert selection.response.value == {"items": ["final"]}
    assert selection.receipt["decision"] == "protocol_supersession"


@pytest.mark.parametrize("atom", [False, 0])
def test_false_and_zero_are_substantive_atoms(atom):
    schema = {
        "type": "object",
        "required": ["result"],
        "properties": {"result": {"type": type(atom).__name__.replace("bool", "boolean").replace("int", "integer")}},
    }
    selection = select_response_candidate(
        _response(_material(0, {"result": atom})), schema=schema, checkpoint_identity=None
    )
    assert selection.receipt["decision"] == "last_substantive"


def test_nested_ref_and_oneof_use_validating_branch_for_semantic_mass():
    schema = {
        "$defs": {
            "payload": {
                "oneOf": [
                    {
                        "type": "object",
                        "required": ["values"],
                        "properties": {"values": {"type": "array", "items": {"type": "number"}}},
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "required": ["message"],
                        "properties": {"message": {"type": "string"}},
                        "additionalProperties": False,
                    },
                ]
            }
        },
        "type": "object",
        "required": ["result"],
        "properties": {"result": {"$ref": "#/$defs/payload"}},
    }
    selection = select_response_candidate(
        _response(
            _material(0, {"result": {"values": []}}),
            _material(1, {"result": {"values": [0]}}),
            _material(2, {"result": {"values": []}}),
        ),
        schema=schema,
        checkpoint_identity=None,
    )
    assert selection.response.value == {"result": {"values": [0]}}
    assert selection.receipt["decision"] == "last_substantive"


def test_ref_sibling_required_fields_contribute_semantic_mass():
    schema = {
        "$defs": {
            "base": {
                "type": "object",
                "required": ["empty"],
                "properties": {"empty": {"type": "array"}},
            }
        },
        "type": "object",
        "required": ["result"],
        "properties": {
            "result": {
                "$ref": "#/$defs/base",
                "required": ["payload"],
                "properties": {"payload": {"type": "string"}},
            }
        },
    }
    selection = select_response_candidate(
        _response(_material(0, {"result": {"empty": [], "payload": "kept"}})),
        schema=schema,
        checkpoint_identity=None,
    )
    assert selection.receipt["decision"] == "last_substantive"


def test_receipt_is_body_free_equivalent_and_tamper_fails_closed(tmp_path):
    selection = select_response_candidate(
        _response(_material(0, {"items": ["private body"]})),
        schema=RESULT_SCHEMA,
        checkpoint_identity="identity",
    )
    checkpoint = tmp_path / "call-checkpoints" / "call.json"
    name, digest = persist_selection_receipt(checkpoint, selection.receipt)
    assert name == "call.candidate-selection.json"
    assert len(digest) == 64
    receipt_path = checkpoint.parent / name
    assert "private body" not in receipt_path.read_text(encoding="utf-8")
    assert persist_selection_receipt(checkpoint, selection.receipt) == (name, digest)

    receipt_path.chmod(0o600)
    receipt_path.write_text(
        json.dumps(selection.receipt, ensure_ascii=False, sort_keys=False, indent=2) + "\n",
        encoding="utf-8",
    )
    actual_digest = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    assert persist_selection_receipt(checkpoint, selection.receipt) == (
        name, actual_digest
    )

    receipt_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(LLMResponseCandidateReceiptError, match="changed or is incompatible"):
        persist_selection_receipt(checkpoint, selection.receipt)

    duplicate = (
        '{"decision":"secret-body",'
        + canonical_json(selection.receipt)[1:]
        + "\n"
    )
    receipt_path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(LLMResponseCandidateReceiptError, match="duplicate receipt key"):
        persist_selection_receipt(checkpoint, selection.receipt)


def test_receipt_publish_is_concurrent_and_ignores_stale_partial_temp(tmp_path):
    selection = select_response_candidate(
        _response(_material(0, {"items": ["safe"]})),
        schema=RESULT_SCHEMA,
        checkpoint_identity="identity",
    )
    checkpoint = tmp_path / "call-checkpoints" / "concurrent.json"
    checkpoint.parent.mkdir(parents=True)
    stale = checkpoint.parent / ".concurrent.candidate-selection.json.dead.tmp"
    stale.write_text("half", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(
            pool.map(
                lambda _: persist_selection_receipt(checkpoint, selection.receipt),
                range(32),
            )
        )
    assert len(set(results)) == 1
    final = checkpoint.parent / results[0][0]
    assert json.loads(final.read_text(encoding="utf-8")) == selection.receipt
    assert stale.read_text(encoding="utf-8") == "half"


def test_receipt_publish_failure_never_exposes_final_or_partial_temp(tmp_path, monkeypatch):
    path = tmp_path / "call.candidate-selection.json"
    monkeypatch.setattr(candidate_module.os, "link", lambda *_args: (_ for _ in ()).throw(OSError("crash")))
    with pytest.raises(OSError, match="crash"):
        candidate_module._atomic_publish_exclusive(path, b'{"complete":true}\n')
    assert not path.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_receipt_metadata_is_sanitized_and_bounded():
    secret = "secret-body\nwith-control\x00"
    material = tuple(
        ResponseCandidateMaterial(
            source=secret,
            protocol_position=index,
            value={"items": ["same"]},
            event_id=secret,
            supersedes=tuple(range(64)),
        )
        for index in range(70)
    )
    selection = select_response_candidate(
        LLMProviderResponse({"items": []}, candidate_material=material),
        schema=RESULT_SCHEMA,
        checkpoint_identity="identity",
    )
    rendered = json.dumps(selection.receipt, ensure_ascii=False)
    assert secret not in rendered
    candidate = selection.receipt["candidates"][0]
    assert candidate["origin_count"] == 70
    assert len(candidate["origins"]) == 64
    assert candidate["origins_truncated"] is True
    assert candidate["supersedes_count"] == 64
    assert len(candidate["supersedes"]) == 64
    conflict = select_response_candidate(
        LLMProviderResponse(
            {"items": []},
            candidate_material=(
                ResponseCandidateMaterial(secret, 0, value={"items": ["a"]}, event_id=secret),
                ResponseCandidateMaterial(secret, 1, value={"items": ["b"]}, event_id=secret),
            ),
        ),
        schema=RESULT_SCHEMA,
        checkpoint_identity=None,
    )
    assert conflict.conflict is not None
    assert secret not in str(conflict.conflict)
    with pytest.raises(LLMResponseCandidateReceiptError, match="supersession metadata"):
        select_response_candidate(
            LLMProviderResponse(
                {"items": []},
                candidate_material=(
                    ResponseCandidateMaterial(
                        "generic.provider_value", 0,
                        value={"items": ["x"]}, supersedes=tuple(range(65))
                    ),
                ),
            ),
            schema=RESULT_SCHEMA,
            checkpoint_identity=None,
        )


def test_provider_material_protocol_rules():
    codex = material_from_codex(
        [
            {"type": "item.completed", "item": {"id": "i1", "type": "agent_message", "text": '{"items":["kept"]}'}},
        ],
        '{"items":[]}',
    )
    assert [item.source for item in codex] == [
        "codex.completed_message",
        "codex.output_last_message",
    ]
    assert codex[-1].supersedes == ()
    codex_selection = select_response_candidate(
        LLMProviderResponse({"items": []}, candidate_material=codex),
        schema=RESULT_SCHEMA,
        checkpoint_identity=None,
    )
    assert codex_selection.response.value == {"items": ["kept"]}

    claude_stdout = "\n".join(
        [
            json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": '{"items":["draft"]}'}]}}),
            json.dumps({"type": "result", "result": '{"items":["terminal"]}', "structured_output": {"items": ["final"]}}),
        ]
    )
    claude = material_from_claude(claude_stdout)
    assert [item.source for item in claude] == [
        "claude.completed_assistant_text",
        "claude.terminal_result",
        "claude.terminal_structured_output",
    ]
    assert claude[-1].supersedes == (1,)
    split_blocks = material_from_claude(
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": '{"items":'},
                        {"type": "text", "text": '["joined"]}'},
                    ]
                },
            }
        )
    )
    assert len(split_blocks) == 1
    assert split_blocks[0].text == '{"items":["joined"]}'

    superseding_claude = material_from_claude(
        "\n".join(
            [
                json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": '{"items":["draft"]}'}]}}),
                json.dumps({"type": "result", "result": '{"items":["draft"]}', "structured_output": {"items": ["final"]}}),
            ]
        )
    )
    claude_selection = select_response_candidate(
        LLMProviderResponse({"items": ["final"]}, candidate_material=superseding_claude),
        schema=RESULT_SCHEMA,
        checkpoint_identity=None,
    )
    assert claude_selection.response.value == {"items": ["final"]}
    assert claude_selection.receipt["decision"] == "protocol_supersession"

    kimi = material_from_kimi('{"items":["one"]}')
    assert len(kimi) == 1
    assert kimi[0].source == "kimi.session_prompt_message"
    kimi_selection = select_response_candidate(
        LLMProviderResponse({"items": ["one"]}, candidate_material=kimi),
        schema=RESULT_SCHEMA,
        checkpoint_identity=None,
    )
    assert kimi_selection.response.value == {"items": ["one"]}


class _ConflictProvider:
    calls = 0

    def generate_json_result(self, _prompt, **_kwargs):
        type(self).calls += 1
        return _response(
            _material(0, {"items": ["a"]}),
            _material(1, {"items": ["b"]}),
        )


class _StableProvider:
    calls = 0

    def generate_json_result(self, _prompt, **_kwargs):
        type(self).calls += 1
        return _response(_material(0, {"items": ["paid"]}))


class _MalformedMaterialProvider:
    calls = 0

    def generate_json_result(self, _prompt, **_kwargs):
        type(self).calls += 1
        return LLMProviderResponse(
            {"items": []},
            candidate_material=(
                ResponseCandidateMaterial(
                    source="generic.provider_value",
                    protocol_position=0,
                    value={"items": {"not-json-serializable"}},
                ),
            ),
        )


class _DeferredInvalidProvider:
    calls = 0

    def generate_json_result(self, _prompt, **_kwargs):
        type(self).calls += 1
        return LLMProviderResponse(
            {"items": []},
            candidate_material=(_material(0, {"wrong": "complete but invalid"}),),
            deferred_output_error=LLMWorkerError(
                "terminal strict parse error",
                category=LLMFailureCategory.OUTPUT_INVALID,
                submission_state=LLMSubmissionState.SUBMITTED,
            ),
        )


def test_conflict_checkpoint_replays_without_provider_and_remains_response_received(
    tmp_path, monkeypatch
):
    _ConflictProvider.calls = 0
    monkeypatch.setattr("arc_llm.runner.select_provider", lambda *_args, **_kwargs: _ConflictProvider())
    kwargs = {
        "provider": "manual",
        "schema": RESULT_SCHEMA,
        "artifact_dir": tmp_path,
        "call_label": "conflict",
        "idempotency_key": "conflict-key",
        "env": {},
    }

    with pytest.raises(LLMResponseCandidateConflict) as submitted:
        run_json_result("prompt", **kwargs)
    assert submitted.value.replayed is False
    assert submitted.value.submission_state == LLMSubmissionState.SUBMITTED
    assert _ConflictProvider.calls == 1
    checkpoints = [
        path
        for path in (tmp_path / "call-checkpoints").glob("idempotency-*.json")
        if ".candidate-selection." not in path.name
    ]
    assert len(checkpoints) == 1
    checkpoint = json.loads(checkpoints[0].read_text(encoding="utf-8"))
    assert checkpoint["schema_version"] == "arc.llm.call_checkpoint.v5"
    assert checkpoint["state"] == "response_received"
    assert len(checkpoint["response"]["candidate_material"]) == 2
    assert list((tmp_path / "call-checkpoints").glob("*.candidate-selection.json"))

    with pytest.raises(LLMResponseCandidateConflict) as replayed:
        run_json_result("prompt", **kwargs)
    assert replayed.value.replayed is True
    assert replayed.value.submission_state == LLMSubmissionState.NOT_SUBMITTED
    assert _ConflictProvider.calls == 1
    attempt_records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (tmp_path / "attempts").glob("*/record.json")
    ]
    assert {record["submission_state"] for record in attempt_records} == {
        "submitted", "not_submitted"
    }
    replay_record = next(
        record for record in attempt_records if record["submission_state"] == "not_submitted"
    )
    # Directory names include the attempt id as a suffix rather than equalling it.
    replay_dir = next(
        path for path in (tmp_path / "attempts").iterdir()
        if path.name.endswith(replay_record["attempt_id"][:12])
    )
    timeline = (replay_dir / replay_record["timeline"]["path"]).read_text(encoding="utf-8")
    assert '"event": "checkpoint_replayed"' in timeline
    submitted_record = next(
        record for record in attempt_records if record["submission_state"] == "submitted"
    )
    assert [
        item["sequence"] for item in submitted_record["parsed_response_candidates"]
    ] == [1, 2]
    assert {
        item["sha256"] for item in submitted_record["parsed_response_candidates"]
    } == set(checkpoint["response"]["candidate_selection"]["conflict_hashes"])
    receipt_path = next(
        (tmp_path / "call-checkpoints").glob("*.candidate-selection.json")
    )
    receipt_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(LLMResponseCandidateReceiptError) as tampered:
        run_json_result("prompt", **kwargs)
    assert tampered.value.submission_state == LLMSubmissionState.NOT_SUBMITTED
    assert _ConflictProvider.calls == 1


def test_sidecar_crash_keeps_atomic_response_for_zero_call_replay(tmp_path, monkeypatch):
    import arc_llm.runner as runner_module

    _StableProvider.calls = 0
    monkeypatch.setattr("arc_llm.runner.select_provider", lambda *_args, **_kwargs: _StableProvider())
    real_persist = runner_module.persist_selection_receipt
    failures = 0

    def crash_once(*args, **kwargs):
        nonlocal failures
        failures += 1
        if failures == 1:
            raise LLMResponseCandidateReceiptError("simulated sidecar crash")
        return real_persist(*args, **kwargs)

    monkeypatch.setattr(runner_module, "persist_selection_receipt", crash_once)
    kwargs = {
        "provider": "manual",
        "schema": RESULT_SCHEMA,
        "artifact_dir": tmp_path,
        "call_label": "sidecar-crash",
        "idempotency_key": "sidecar-crash-key",
        "env": {},
    }
    with pytest.raises(Exception, match="simulated sidecar crash"):
        run_json_result("prompt", **kwargs)
    assert _StableProvider.calls == 1
    checkpoint_paths = [
        path
        for path in (tmp_path / "call-checkpoints").glob("idempotency-*.json")
        if ".candidate-selection." not in path.name
    ]
    checkpoint = json.loads(checkpoint_paths[0].read_text(encoding="utf-8"))
    assert checkpoint["state"] == "response_received"
    assert checkpoint["response"]["value"] == {"items": ["paid"]}
    assert checkpoint["response"]["candidate_selection"]["decision"] == "last_substantive"

    replay = run_json_result("prompt", **kwargs)
    assert replay.value == {"items": ["paid"]}
    assert replay.logical_receipt["replayed"] is True
    assert replay.logical_receipt["candidate_selection_receipt"]["sha256"]
    assert _StableProvider.calls == 1


def test_post_provider_selection_failure_releases_checkpoint_without_resubmit(
    tmp_path, monkeypatch
):
    _MalformedMaterialProvider.calls = 0
    monkeypatch.setattr(
        "arc_llm.runner.select_provider",
        lambda *_args, **_kwargs: _MalformedMaterialProvider(),
    )
    kwargs = {
        "provider": "manual",
        "schema": RESULT_SCHEMA,
        "artifact_dir": tmp_path,
        "call_label": "malformed-material",
        "idempotency_key": "malformed-material-key",
        "env": {},
    }
    with pytest.raises(Exception):
        run_json_result("prompt", **kwargs)
    assert _MalformedMaterialProvider.calls == 1
    with pytest.raises(LLMCallNeedsSupervision):
        run_json_result("prompt", **kwargs)
    assert _MalformedMaterialProvider.calls == 1


def test_deferred_no_valid_candidate_keeps_strict_error_and_diagnostics(
    tmp_path, monkeypatch
):
    _DeferredInvalidProvider.calls = 0
    monkeypatch.setattr(
        "arc_llm.runner.select_provider",
        lambda *_args, **_kwargs: _DeferredInvalidProvider(),
    )
    kwargs = {
        "provider": "manual",
        "schema": RESULT_SCHEMA,
        "artifact_dir": tmp_path,
        "call_label": "deferred-invalid",
        "idempotency_key": "deferred-invalid-key",
        "env": {},
    }
    with pytest.raises(Exception, match="terminal strict parse error"):
        run_json_result("prompt", **kwargs)
    assert _DeferredInvalidProvider.calls == 1
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (tmp_path / "attempts").glob("*/record.json")
    ]
    assert len(records) == 1
    assert len(records[0]["parsed_response_candidates"]) == 1
    assert records[0]["parsed_response_candidates"][0]["sequence"] == 1
    with pytest.raises(LLMCallRetryExhausted):
        run_json_result("prompt", **kwargs)
    assert _DeferredInvalidProvider.calls == 1


def test_no_schema_valid_candidate_preserves_provider_value_for_existing_recovery():
    response = _response(_material(0, {"wrong": "shape"}))
    selection = select_response_candidate(
        response, schema=RESULT_SCHEMA, checkpoint_identity=None
    )
    assert selection.receipt["decision"] == "no_schema_valid_candidate"
    assert selection.response.value == {"items": []}
