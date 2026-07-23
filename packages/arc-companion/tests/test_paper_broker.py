from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time

import pytest

from arc_llm import EvidenceJournal, EvidenceJournalContext, EvidenceRequest
from arc_companion import paper_broker as broker_module
from arc_companion.pipeline import (
    BuildOptions,
    _llm_runtime_env,
    _paper_broker_for_call,
    _paper_broker_policy_for_call,
    _requested_paper_runtime_profile,
    _resolved_paper_runtime_profile,
)
from arc_companion.paper_broker import (
    ARTIFACT_READ_OPERATION,
    MAX_INLINE_BYTES,
    MAX_ROUND_RESPONSE_BYTES,
    PaperBroker,
    build_paper_broker_policy,
    paper_broker_prompt_prefix,
    paper_broker_schema,
)


def _request(operation: str, arguments: dict, *, request_id: str = "request-1"):
    return EvidenceRequest(
        request_id=request_id,
        operation=operation,
        arguments=arguments,
        reason="test",
        worker_id="worker-1",
        role="companion-content-worker",
    )


def _broker(tmp_path: Path, **policy_overrides) -> PaperBroker:
    return PaperBroker(
        checkpoint_root=tmp_path / "checkpoint",
        base_cache_root=tmp_path / "cache",
        policy=build_paper_broker_policy(**policy_overrides),
        run_id="run-1",
        generic_internet_allowed=False,
    )


def test_default_full_policy_is_catalog_driven_and_controller_only(tmp_path):
    policy = build_paper_broker_policy(nested_shell_capability={
        "nested_sandboxed_shell": False,
        "nested_shell_probe_id": "provider-contract-identity",
    })

    assert policy.access == "full"
    assert policy.allowed_operation_ids
    assert policy.paper_network_authorized is True
    assert policy.direct_shell_probe_id == "probe-not-requested"
    assert all("summary-batch.run" not in value for value in policy.allowed_operation_ids)
    prompt = paper_broker_prompt_prefix(policy)
    assert "arc_evidence_requests" in prompt
    assert "arc-paper-worker" not in prompt
    assert "generic_internet" not in prompt

    none_schema = paper_broker_schema(
        {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "arc_evidence_requests": {"type": "array"},
            },
            "required": ["answer", "arc_evidence_requests"],
        },
        access="none",
    )
    assert set(none_schema["properties"]) == {"answer"}
    assert none_schema["required"] == ["answer"]


def test_intent_policy_freezes_operations_while_general_full_uses_safe_default(
    monkeypatch, tmp_path,
):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    options = BuildOptions(paper_id="0911.3380", project_dir=tmp_path / "project")
    frozen_policy = _paper_broker_policy_for_call(
        options=options,
        paper_access_policy={
            "operations": ["get-title"],
            "authorized_source_ids": ["0911.3380"],
        },
        nested_shell_capability=None,
    )
    frozen = _paper_broker_for_call(
        options=options,
        intent_guidance={},
        lane="translation",
        policy=frozen_policy,
        journal_context=None,
        checkpoint_root=tmp_path / "checkpoint",
        broker_run_id="frozen",
    )
    general_policy = _paper_broker_policy_for_call(
        options=options,
        paper_access_policy=None,
        nested_shell_capability=None,
    )
    general = _paper_broker_for_call(
        options=options,
        intent_guidance={},
        lane="translation",
        policy=general_policy,
        journal_context=None,
        checkpoint_root=tmp_path / "checkpoint",
        broker_run_id="general",
    )

    assert frozen is not None and general is not None
    assert frozen.policy.allowed_operation_ids == ("arc-paper.get-title.v1",)
    assert len(general.policy.allowed_operation_ids) > 1
    assert set(frozen.policy.allowed_operation_ids) < set(
        general.policy.allowed_operation_ids
    )
    requested_profile = _requested_paper_runtime_profile(options)
    resolved_profile = _resolved_paper_runtime_profile(options, frozen)
    assert requested_profile["paper_direct_decision"] == "controller"
    assert resolved_profile == {
        "arc_paper_access": "full",
        "paper_policy_sha256": frozen.policy.policy_sha256,
        "paper_catalog_sha256": frozen.policy.catalog_sha256,
        "paper_network_authorized": True,
        "arc_paper_direct_shell": False,
        "paper_direct_decision": "controller",
        "direct_shell_probe_id": "probe-not-requested",
    }


def test_companion_runtime_env_rejects_process_alias_conflict_before_call(
    monkeypatch,
):
    monkeypatch.delenv("ARC_PAPER_ACCESS", raising=False)
    monkeypatch.setenv("ARC_PAPER_CLI_ACCESS", "none")

    with pytest.raises(ValueError, match="conflicts"):
        _llm_runtime_env(allow_internet=False, arc_paper_access="full")


def test_direct_policy_environment_contains_only_network_none_operations(monkeypatch):
    monkeypatch.delenv("ARC_PAPER_ACCESS", raising=False)
    monkeypatch.delenv("ARC_PAPER_CLI_ACCESS", raising=False)
    env = _llm_runtime_env(
        allow_internet=False,
        force_disable_internet=True,
        arc_paper_access="full",
        arc_paper_direct_shell=True,
        paper_access_policy={
            "operations": ["get-title", "extract-paper-ids"],
            "authorized_source_ids": [],
        },
    )

    assert json.loads(env["ARC_PAPER_WORKER_ALLOWED_OPERATIONS_JSON"]) == [
        "extract-paper-ids"
    ]
    assert env["ARC_CODEX_ALLOW_INTERNET"] == "false"


def test_explicit_direct_policy_requires_trusted_capability_and_keeps_controller_catalog():
    with pytest.raises(broker_module.PaperBrokerError) as unavailable:
        build_paper_broker_policy(
            direct_shell_requested=True,
            nested_shell_capability={
                "nested_sandboxed_shell": False,
                "nested_shell_probe_id": "probe-failed",
            },
        )
    assert unavailable.value.code == "paper_direct_shell_unavailable"

    policy = build_paper_broker_policy(
        allowed_operations=["get-title", "extract-paper-ids"],
        direct_shell_requested=True,
        nested_shell_capability={
            "nested_sandboxed_shell": True,
            "nested_shell_probe_id": "trusted-probe",
        },
    )
    prompt = paper_broker_prompt_prefix(policy)
    assert policy.direct_shell_available is True
    assert policy.direct_shell_probe_id == "trusted-probe"
    assert "{{ARC_NESTED_SHELL_CAPABILITY}}" in prompt
    assert "network=none" in prompt
    assert "arc-paper-worker" in prompt
    assert {item["name"] for item in broker_module.compact_catalog(
        policy.allowed_operation_ids
    )["operations"]} == {"get-title", "extract-paper-ids"}
    controls = broker_module.compact_catalog(policy.allowed_operation_ids)["controls"]
    assert {item["name"] for item in controls} == {
        "list-reference-targets", "artifact-read",
    }
    assert all(item["description"] for item in controls)
    assert all(
        item["parameters"]["additionalProperties"] is False for item in controls
    )


def test_alias_and_locator_are_canonical_before_controller_dispatch(monkeypatch, tmp_path):
    broker = _broker(
        tmp_path,
        allowed_operations=["get-parsed-section"],
        authorized_source_ids=["0911.3380"],
        authorized_sections=[
            {"source_id": "arXiv:0911.3380", "section": "Introduction"}
        ],
    )
    seen = []

    def dispatch(operation, arguments, **_kwargs):
        seen.append((operation, dict(arguments)))
        return {
            "ok": True,
            "data": {"text": "hello"},
            "errors": [],
            "meta": {"provider": "local-cache", "cache": "hit"},
        }

    monkeypatch.setattr(broker_module, "dispatch_operation", dispatch)
    alias = broker.canonicalize_request(_request("extract-ids", {"text": "0911.3380"}))
    assert alias.operation == "extract-paper-ids"

    response = broker.controller(
        (
            _request(
                "get-parsed-section",
                {"source_id": "0911.3380", "locator": "Introduction"},
            ),
        ),
        round_number=1,
    )[0]

    assert response.ok is True
    assert seen == [
        (
            "get-parsed-section",
            {"source_id": "arXiv:0911.3380", "section": "Introduction"},
        )
    ]
    assert response.provenance["route"] == "controller"
    assert response.provenance["generic_internet_allowed"] is False
    assert response.provenance["network_observed"] == "unknown"
    assert response.provenance["cache_observed"] == "hit"


def test_result_paths_are_scrubbed_and_large_results_page_by_handle(
    monkeypatch, tmp_path,
):
    broker = _broker(tmp_path, allowed_operations=["get-title"])
    owned = broker.session.base_root / "papers" / "owned.txt"
    owned.parent.mkdir(parents=True)
    owned.write_text("owned content", encoding="utf-8")
    external = tmp_path / "outside.txt"
    external.write_text("secret path", encoding="utf-8")
    large = "x" * (MAX_INLINE_BYTES + 1)

    monkeypatch.setattr(
        broker_module,
        "dispatch_operation",
        lambda *_args, **_kwargs: {
            "ok": True,
            "data": {
                "content": large,
                "owned_path": str(owned),
                "external_path": str(external),
                "cache_path": str(broker.session.base_root),
                "url": "https://reader:password@example.test/paper?token=secret#fragment",
            },
            "errors": [],
            "meta": {},
        },
    )
    response = broker.controller(
        (_request("get-title", {"paper_ids": ["0911.3380"]}),),
        round_number=1,
    )[0]

    assert response.ok is True
    assert set(response.data) == {"handle_id", "sha256", "size_bytes", "media_type"}
    receipt_text = next(broker.receipts_root.glob("*.json")).read_text(encoding="utf-8")
    assert str(external) not in receipt_text
    assert str(owned) not in receipt_text
    assert '"cache_path":' not in receipt_text
    handle_record = json.loads(
        (broker.handles_root / f"{response.data['handle_id']}.json").read_text(
            encoding="utf-8"
        )
    )
    object_value = json.loads(
        (broker.objects_root / handle_record["object_name"]).read_text(encoding="utf-8")
    )
    assert object_value["data"]["url"] == "https://example.test/paper"
    assert "secret" not in receipt_text

    page = broker.controller(
        (
            _request(
                ARTIFACT_READ_OPERATION,
                {"handle_id": response.data["handle_id"], "offset": 0, "limit": 1024},
                request_id="page-1",
            ),
        ),
        round_number=2,
    )[0]
    assert page.ok is True
    assert len(base64.b64decode(page.data["content_base64"])) == 1024
    assert page.data["next_offset"] == 1024
    assert page.data["sha256"] == response.data["sha256"]

    maximum_page = broker.controller(
        (
            _request(
                ARTIFACT_READ_OPERATION,
                {
                    "handle_id": response.data["handle_id"],
                    "offset": 0,
                    "limit": broker_module.MAX_PAGE_BYTES,
                },
                request_id="page-max",
            ),
        ),
        round_number=2,
    )[0]
    serialized = json.dumps({
        "request_id": maximum_page.request_id,
        "ok": maximum_page.ok,
        "data": maximum_page.data,
        "error": maximum_page.error,
        "provenance": dict(maximum_page.provenance),
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert len(serialized) <= MAX_ROUND_RESPONSE_BYTES


def test_none_policy_exposes_no_catalog_or_dispatch(monkeypatch, tmp_path):
    broker = _broker(tmp_path, access="none")
    monkeypatch.setattr(
        broker_module,
        "dispatch_operation",
        lambda *_args, **_kwargs: pytest.fail("disabled Broker must not dispatch"),
    )

    assert broker.catalog["operations"] == []
    assert broker.catalog["controls"] == []
    assert paper_broker_prompt_prefix(broker.policy) == ""
    response = broker.controller(
        (_request("get-title", {"paper_ids": ["0911.3380"]}),),
        round_number=1,
    )[0]
    assert response.ok is False
    assert response.provenance["error"] == {
        "code": "paper_access_disabled",
        "category": "local",
        "retryable": False,
    }


def test_round_budget_pages_one_of_two_individually_inline_results(
    monkeypatch, tmp_path,
):
    broker = _broker(tmp_path, allowed_operations=["get-title"])
    monkeypatch.setattr(
        broker_module,
        "dispatch_operation",
        lambda *_args, **_kwargs: {
            "ok": True, "data": {"content": "x" * (40 * 1024)},
            "errors": [], "meta": {},
        },
    )

    responses = broker.controller(
        (
            _request("get-title", {"paper_ids": ["0911.3380"]}, request_id="one"),
            _request("get-title", {"paper_ids": ["0911.3381"]}, request_id="two"),
        ),
        round_number=1,
    )
    encoded = json.dumps(
        [broker_module._serialize_response(item) for item in responses],
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")

    assert len(encoded) <= MAX_ROUND_RESPONSE_BYTES
    assert sum(item.provenance["result_inline"] is False for item in responses) == 1


def test_round_budget_accounts_for_mixed_page_and_ordinary_response(
    monkeypatch, tmp_path,
):
    broker = _broker(tmp_path, allowed_operations=["get-title"])
    monkeypatch.setattr(
        broker_module,
        "dispatch_operation",
        lambda *_args, **_kwargs: {
            "ok": True, "data": {"content": "x" * (40 * 1024)},
            "errors": [], "meta": {},
        },
    )
    initial = broker.controller(
        (_request("get-title", {"paper_ids": ["0911.3380"]}),),
        round_number=1,
    )[0]
    handle = initial.provenance["result_handle"]

    responses = broker.controller(
        (
            _request("get-title", {"paper_ids": ["0911.3381"]}, request_id="ordinary"),
            _request(
                ARTIFACT_READ_OPERATION,
                {"handle_id": handle["handle_id"], "offset": 0, "limit": 30 * 1024},
                request_id="page",
            ),
        ),
        round_number=2,
    )
    encoded = json.dumps(
        [broker_module._serialize_response(item) for item in responses],
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")

    assert len(encoded) <= MAX_ROUND_RESPONSE_BYTES
    assert responses[0].provenance["result_inline"] is False
    assert responses[1].ok is True


def test_controller_does_not_swallow_process_control_exceptions(
    monkeypatch, tmp_path,
):
    broker = _broker(tmp_path, allowed_operations=["get-title"])
    monkeypatch.setattr(
        broker_module,
        "dispatch_operation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(SystemExit(7)),
    )

    with pytest.raises(SystemExit):
        broker.controller(
            (_request("get-title", {"paper_ids": ["0911.3380"]}),),
            round_number=1,
        )


def test_llm_and_job_operations_defer_to_managed_job_route(monkeypatch, tmp_path):
    broker = _broker(tmp_path)
    monkeypatch.setattr(
        broker_module,
        "dispatch_operation",
        lambda *_args, **_kwargs: pytest.fail("managed job must not dispatch inline"),
    )

    response = broker.controller(
        (_request("get-llm-summary", {"paper_ids": ["0911.3380"]}),),
        round_number=1,
    )[0]

    assert response.ok is False
    assert response.provenance["error"]["code"] == "managed_job_required"


@pytest.mark.parametrize(
    ("failure", "category", "retryable"),
    [
        (type("RateLimit", (RuntimeError,), {"status_code": 429})("busy"), "rate_limit", False),
        (type("ServiceFailure", (RuntimeError,), {"status_code": 503})("down"), "transport", True),
        (TimeoutError("slow"), "timeout", True),
    ],
)
def test_broker_normalizes_transient_operation_failures(
    monkeypatch, tmp_path, failure, category, retryable,
):
    broker = _broker(tmp_path, allowed_operations=["get-title"])
    monkeypatch.setattr(
        broker_module,
        "dispatch_operation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
    )

    response = broker.controller(
        (_request("get-title", {"paper_ids": ["0911.3380"]}),),
        round_number=1,
    )[0]

    assert response.ok is False
    assert response.provenance["error"]["category"] == category
    assert response.provenance["error"]["retryable"] is retryable


def test_broker_result_receipt_recovers_t10_prepared_without_redispatch(
    monkeypatch, tmp_path,
):
    context = EvidenceJournalContext(
        journal_root=tmp_path / "journal",
        run_id="run-1",
        lane_id="translation",
        worker_id="worker-1",
        logical_task_id="segment-1",
        source_generation=1,
        policy_hash="embedding-policy",
        runtime_hash="embedding-runtime",
    )
    broker = PaperBroker(
        checkpoint_root=tmp_path / "checkpoint",
        base_cache_root=tmp_path / "cache",
        policy=build_paper_broker_policy(
            allowed_operations=["extract-paper-ids"],
        ),
        run_id="run-1",
        generic_internet_allowed=False,
        journal_context=context,
    )
    calls = 0

    def dispatch(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return {"ok": True, "data": ["arXiv:0911.3380"], "errors": [], "meta": {}}

    monkeypatch.setattr(broker_module, "dispatch_operation", dispatch)
    request = broker.canonicalize_request(
        _request("extract-ids", {"text": "0911.3380"})
    )
    journal = EvidenceJournal(broker.journal_context.journal_root)

    def crash_after_broker_result(state, _address, _receipt):
        if state == "executed":
            raise RuntimeError("simulated journal crash")

    with pytest.raises(RuntimeError, match="simulated journal crash"):
        journal.resolve_round(
            broker.journal_context,
            (request,),
            broker.controller,
            round_number=1,
            operation_policies={
                request.operation: broker._journal_policy(request, round_number=1)
            },
            transition_hook=crash_after_broker_result,
        )
    assert calls == 1
    assert next(broker.receipts_root.glob("*.json")).read_text().find(
        '"state":"result_persisted"'
    ) >= 0

    recovered = broker.resolve_round((request,), round_number=1)
    assert recovered[0].ok is True
    assert calls == 1


@pytest.mark.parametrize("preplayed", [1, 2])
def test_journal_replay_composition_reapplies_whole_round_budget(
    monkeypatch, tmp_path, preplayed,
):
    context = EvidenceJournalContext(
        journal_root=tmp_path / "journal",
        run_id="run-1",
        lane_id="translation",
        worker_id="worker-1",
        logical_task_id="segment-1",
        source_generation=1,
        policy_hash="embedding-policy",
        runtime_hash="embedding-runtime",
    )
    broker = PaperBroker(
        checkpoint_root=tmp_path / "checkpoint",
        base_cache_root=tmp_path / "cache",
        policy=build_paper_broker_policy(allowed_operations=["get-title"]),
        run_id="run-1",
        generic_internet_allowed=False,
        journal_context=context,
    )
    calls = 0

    def dispatch(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return {
            "ok": True, "data": {"content": "x" * (40 * 1024)},
            "errors": [], "meta": {},
        }

    monkeypatch.setattr(broker_module, "dispatch_operation", dispatch)
    requests = (
        _request("get-title", {"paper_ids": ["0911.3380"]}, request_id="one"),
        _request("get-title", {"paper_ids": ["0911.3381"]}, request_id="two"),
    )
    for request in requests[:preplayed]:
        assert broker.resolve_round((request,), round_number=1)[0].ok
    calls_before_composition = calls

    responses = broker.resolve_round(requests, round_number=1)
    encoded = json.dumps(
        [broker_module._serialize_response(item) for item in responses],
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")

    assert len(encoded) <= MAX_ROUND_RESPONSE_BYTES
    assert calls == calls_before_composition + (2 - preplayed)


@pytest.mark.parametrize(
    ("crash_state", "dispatches_before_recovery"),
    [
        ("prepared", 0),
        ("object_persisted", 1),
        ("promotion_persisted", 1),
        ("result_persisted", 1),
    ],
)
def test_broker_recovers_each_persisted_stage_without_duplicate_dispatch(
    monkeypatch, tmp_path, crash_state, dispatches_before_recovery,
):
    calls = 0

    def dispatch(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return {"ok": True, "data": ["arXiv:0911.3380"], "errors": [], "meta": {}}

    def crash(state, _receipt):
        if state == crash_state:
            raise SystemExit(f"crash after {state}")

    monkeypatch.setattr(broker_module, "dispatch_operation", dispatch)
    policy = build_paper_broker_policy(allowed_operations=["extract-paper-ids"])
    failed = PaperBroker(
        checkpoint_root=tmp_path / "checkpoint",
        base_cache_root=tmp_path / "cache",
        policy=policy,
        run_id="run-1",
        generic_internet_allowed=False,
        transition_hook=crash,
    )
    request = _request("extract-paper-ids", {"text": "0911.3380"})

    with pytest.raises(SystemExit, match=f"crash after {crash_state}"):
        failed.controller((request,), round_number=1)
    assert calls == dispatches_before_recovery

    recovered_broker = PaperBroker(
        checkpoint_root=tmp_path / "checkpoint",
        base_cache_root=tmp_path / "cache",
        policy=policy,
        run_id="run-1",
        generic_internet_allowed=False,
    )
    recovered = recovered_broker.controller((request,), round_number=1)[0]

    assert recovered.ok is True
    assert calls == 1
    assert next(recovered_broker.receipts_root.glob("*.json")).read_text().find(
        '"state":"result_persisted"'
    ) >= 0


def test_transactional_t10_recovery_finishes_persisted_object_without_redispatch(
    monkeypatch, tmp_path,
):
    calls = 0

    def dispatch(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return {"ok": True, "data": {"stored": True}, "errors": [], "meta": {}}

    def crash(state, _receipt):
        if state == "object_persisted":
            raise SystemExit("crash after object")

    monkeypatch.setattr(broker_module, "dispatch_operation", dispatch)
    context = EvidenceJournalContext(
        journal_root=tmp_path / "journal",
        run_id="run-1",
        lane_id="translation",
        worker_id="worker-1",
        logical_task_id="segment-1",
        source_generation=1,
        policy_hash="embedding-policy",
        runtime_hash="embedding-runtime",
    )
    policy = build_paper_broker_policy(allowed_operations=["mark-parsed-equation"])
    failed = PaperBroker(
        checkpoint_root=tmp_path / "checkpoint",
        base_cache_root=tmp_path / "cache",
        policy=policy,
        run_id="run-1",
        generic_internet_allowed=False,
        journal_context=context,
        transition_hook=crash,
    )
    request = _request("mark-parsed-equation", {"equation_id": "eq-1"})

    with pytest.raises(SystemExit, match="crash after object"):
        failed.resolve_round((request,), round_number=1)
    assert calls == 1

    recovered = PaperBroker(
        checkpoint_root=tmp_path / "checkpoint",
        base_cache_root=tmp_path / "cache",
        policy=policy,
        run_id="run-1",
        generic_internet_allowed=False,
        journal_context=context,
    ).resolve_round((request,), round_number=1)

    assert recovered[0].ok is True
    assert calls == 1


def test_handle_ownership_integrity_and_whole_round_budget(monkeypatch, tmp_path):
    broker = _broker(tmp_path, allowed_operations=["get-title"])
    monkeypatch.setattr(
        broker_module,
        "dispatch_operation",
        lambda *_args, **_kwargs: {
            "ok": True,
            "data": "x" * (MAX_INLINE_BYTES + 1),
            "errors": [],
            "meta": {},
        },
    )
    result = broker.controller(
        (_request("get-title", {"paper_ids": ["0911.3380"]}),),
        round_number=1,
    )[0]
    handle_id = result.data["handle_id"]

    other_run = PaperBroker(
        checkpoint_root=tmp_path / "checkpoint",
        base_cache_root=tmp_path / "cache",
        policy=broker.policy,
        run_id="run-2",
        generic_internet_allowed=False,
    )
    with pytest.raises(broker_module.PaperBrokerError) as ownership:
        other_run.read_page(handle_id, limit=1)
    assert ownership.value.code == "artifact_handle_forbidden"

    budget = broker.controller(
        (
            _request(
                ARTIFACT_READ_OPERATION,
                {"handle_id": handle_id, "offset": 0, "limit": 46 * 1024},
                request_id="page-a",
            ),
            _request(
                ARTIFACT_READ_OPERATION,
                {"handle_id": handle_id, "offset": 46 * 1024, "limit": 46 * 1024},
                request_id="page-b",
            ),
        ),
        round_number=2,
    )
    assert all(not response.ok for response in budget)
    assert {response.provenance["error"]["code"] for response in budget} == {
        "artifact_round_budget_exceeded"
    }

    record = json.loads(
        (broker.handles_root / f"{handle_id}.json").read_text(encoding="utf-8")
    )
    (broker.objects_root / record["object_name"]).write_bytes(b"corrupt")
    corrupted = broker.controller(
        (
            _request(
                ARTIFACT_READ_OPERATION,
                {"handle_id": handle_id, "limit": 1},
                request_id="page-corrupt",
            ),
        ),
        round_number=2,
    )[0]
    assert corrupted.ok is False
    assert corrupted.provenance["error"]["code"] == "artifact_integrity_failed"


def test_distinct_source_dispatches_overlap_without_call_scope_cross_talk(
    monkeypatch, tmp_path,
):
    broker = _broker(tmp_path, allowed_operations=["extract-paper-ids"])
    active = 0
    maximum = 0
    observed = {}
    guard = threading.Lock()

    def dispatch(_operation, arguments, **_kwargs):
        nonlocal active, maximum
        from arc_paper.cache import cache_root, write_json
        from arc_paper.runtime_context import current_worker_call_id

        with guard:
            active += 1
            maximum = max(maximum, active)
        try:
            time.sleep(0.02)
            token = str(arguments["text"])
            observed[token] = (cache_root(), current_worker_call_id())
            write_json(
                cache_root() / "queries" / f"{token}.json",
                {"schema_version": "arc.test.query.v1", "token": token},
            )
            return {"ok": True, "data": [token], "errors": [], "meta": {}}
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(broker_module, "dispatch_operation", dispatch)

    def resolve(token):
        return broker.controller(
            (
                _request(
                    "extract-paper-ids",
                    {"text": token},
                    request_id=f"request-{token}",
                ),
            ),
            round_number=1,
        )[0]

    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(resolve, ("a", "b")))

    assert all(response.ok for response in responses)
    assert maximum == 2
    assert {root for root, _call_id in observed.values()} == {
        broker.session.overlay_root,
    }
    assert observed["a"][1] != observed["b"][1]
    assert all(call_id and call_id.startswith("call-") for _root, call_id in observed.values())
    assert (broker.session.base_root / "queries/a.json").is_file()
    assert (broker.session.base_root / "queries/b.json").is_file()
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in broker.session.record_root.glob("*.json")
    ]
    assert len({record["writer_call_id"] for record in records}) == 2


def test_normalized_aliases_share_fetch_success_and_refresh_bypasses_it(
    monkeypatch, tmp_path,
):
    broker = _broker(tmp_path, allowed_operations=["get-metadata"])
    fetches = 0

    def dispatch(_operation, arguments, **_kwargs):
        nonlocal fetches
        from arc_paper.worker_session import worker_fetch_once

        paper_id = arguments["paper_ids"][0]

        def fetch():
            nonlocal fetches
            fetches += 1
            return {"paper_id": paper_id, "fetch": fetches}

        data = worker_fetch_once(
            paper_id,
            fetch,
            operation="metadata",
            replay_success=not bool(arguments.get("refresh", False)),
        )
        return {"ok": True, "data": data, "errors": [], "meta": {}}

    monkeypatch.setattr(broker_module, "dispatch_operation", dispatch)
    requests = (
        _request("get-metadata", {"paper_ids": ["0911.3380"]}, request_id="a"),
        _request(
            "get-metadata", {"paper_ids": ["arXiv:0911.3380"]}, request_id="b",
        ),
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, replay = list(pool.map(
            lambda request: broker.controller((request,), round_number=1)[0],
            requests,
        ))
    refreshed = broker.controller(
        (
            _request(
                "get-metadata",
                {"paper_ids": ["0911.3380"], "refresh": True},
                request_id="c",
            ),
        ),
        round_number=1,
    )[0]

    assert fetches == 2
    assert first.data["data"]["fetch"] == replay.data["data"]["fetch"] == 1
    assert refreshed.data["data"]["fetch"] == 2


def test_registered_artifacts_are_bound_to_operation_parameter_access_and_run(
    monkeypatch, tmp_path,
):
    from arc_paper import service

    broker = _broker(tmp_path, allowed_operations=["parse"])
    source = broker.register_input_bytes(
        b"# Paper\n",
        operation="parse",
        parameter="source_path",
        media_type="text/markdown",
    )
    seen = {}

    def parse_source(source_path, **options):
        path = Path(source_path)
        seen.update(content=path.read_text(encoding="utf-8"), options=options)
        return {"ok": True, "data": {"paper_id": "local:test"}, "errors": [], "meta": {}}

    monkeypatch.setattr(service, "parse_source", parse_source)
    response = broker.controller(
        (
            _request(
                "parse",
                {"source_path": {"handle_id": source["handle_id"]}},
            ),
        ),
        round_number=1,
    )[0]
    assert response.ok is True
    assert seen == {"content": "# Paper\n", "options": {}}

    wrong_parameter = broker.controller(
        (
            _request(
                "parse",
                {"html_path": {"handle_id": source["handle_id"]}},
                request_id="wrong-parameter",
            ),
        ),
        round_number=1,
    )[0]
    assert wrong_parameter.ok is False

    output = broker.register_output(
        operation="summary-batch.export", parameter="output",
    )
    output_path = broker._artifact_resolver(
        output["handle_id"],
        access="write",
        operation="summary-batch.export",
        parameter="output",
    )
    assert output_path.parent == broker.inputs_root

    other = PaperBroker(
        checkpoint_root=tmp_path / "checkpoint",
        base_cache_root=tmp_path / "cache",
        policy=broker.policy,
        run_id="other-run",
        generic_internet_allowed=False,
    )
    with pytest.raises(broker_module.PaperBrokerError) as wrong_run:
        other._artifact_resolver(
            source["handle_id"],
            access="read",
            operation="parse",
            parameter="source_path",
        )
    assert wrong_run.value.code == "artifact_handle_forbidden"


def test_broker_promotion_preserves_existing_base_conflict(monkeypatch, tmp_path):
    broker = _broker(tmp_path, allowed_operations=["extract-paper-ids"])
    base = broker.session.base_root / "queries" / "conflict.json"
    base.parent.mkdir(parents=True)
    base.write_text(
        '{"schema_version":"arc.test.query.v1","value":"base"}',
        encoding="utf-8",
    )

    def dispatch(*_args, **_kwargs):
        from arc_paper.cache import cache_root, write_json

        write_json(
            cache_root() / "queries" / "conflict.json",
            {"schema_version": "arc.test.query.v1", "value": "overlay"},
        )
        return {"ok": True, "data": [], "errors": [], "meta": {}}

    monkeypatch.setattr(broker_module, "dispatch_operation", dispatch)
    response = broker.controller(
        (_request("extract-paper-ids", {"text": "none"}),),
        round_number=1,
    )[0]

    assert response.ok is True
    assert json.loads(base.read_text(encoding="utf-8"))["value"] == "base"
    assert response.provenance["cache_observed"] == "conflict_preserved"
    assert response.provenance["promotion"]["conflicted"] == [
        "queries/conflict.json"
    ]


def test_unknown_operation_raw_paths_and_extra_command_parameters_fail_closed(tmp_path):
    broker = _broker(tmp_path, allowed_operations=["parse", "get-title"])

    unknown = broker.controller(
        (_request("shell", {"argv": ["arc-paper", "get-title"]}),),
        round_number=1,
    )[0]
    assert unknown.ok is False
    assert unknown.provenance["error"]["code"] == "paper_operation_unknown"

    raw_path = broker.controller(
        (_request("parse", {"source_path": "/etc/passwd"}),),
        round_number=1,
    )[0]
    raw_command = broker.controller(
        (
            _request(
                "get-title",
                {"paper_ids": ["0911.3380"], "argv": ["unexpected"]},
                request_id="argv",
            ),
        ),
        round_number=1,
    )[0]

    assert raw_path.ok is False
    assert raw_path.provenance["error"] == {
        "code": "paper_operation_parameters_invalid",
        "category": "local",
        "retryable": False,
    }
    assert "/etc/passwd" not in (raw_path.error or "")
    assert raw_command.provenance["error"]["code"] == (
        "paper_operation_parameters_invalid"
    )
