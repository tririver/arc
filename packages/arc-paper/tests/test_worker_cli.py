import base64
import json
import os
from pathlib import Path

import pytest

from arc_paper import worker_cli
from arc_paper.worker_session import WorkerCacheSession


def _output(capsys):
    return json.loads(capsys.readouterr().out)


@pytest.fixture(autouse=True)
def _enable_paper_cli(monkeypatch):
    monkeypatch.setenv("ARC_PAPER_CLI_ACCESS", "full")
    for name in (
        "ARC_PAPER_CACHE",
        "ARC_PAPER_WORKER_BASE_CACHE",
        "ARC_PAPER_WORKER_SESSION_DIR",
        "ARC_PAPER_WORKER_SESSION_ID",
        "ARC_LLM_WORKER_CONTEXT",
    ):
        monkeypatch.delenv(name, raising=False)


def test_worker_fails_closed_when_paper_cli_access_is_not_full(monkeypatch, capsys):
    monkeypatch.delenv("ARC_PAPER_CLI_ACCESS")
    monkeypatch.setattr(worker_cli.cli, "main", lambda _argv: (_ for _ in ()).throw(AssertionError()))

    assert worker_cli.main(["artifact-read", "sha256-" + "0" * 64 + ".json"]) == 1
    assert _output(capsys)["error"]["code"] == "paper_cli_disabled"


def test_raw_arc_paper_is_rejected_inside_worker_context(monkeypatch, capsys):
    monkeypatch.setenv("ARC_LLM_WORKER_CONTEXT", "true")
    assert worker_cli.cli.main(["extract-ids", "0911.3380", "--json"]) == 1
    assert _output(capsys)["error"]["code"] == "paper_worker_wrapper_required"


def test_direct_service_module_cannot_bypass_worker_wrapper(monkeypatch):
    monkeypatch.setenv("ARC_LLM_WORKER_CONTEXT", "true")
    result = worker_cli.cli.service.get_metadata("0911.3380")
    assert result["error"]["code"] == "paper_query_error"
    assert "paper_worker_wrapper_required" in result["error"]["message"]


def test_worker_context_uses_session_paths_without_local_bearer_guard(monkeypatch, tmp_path, capsys):
    base = tmp_path / "base"
    base.mkdir()
    session = WorkerCacheSession(base_root=base, run_root=tmp_path / "run", session_id="s1")
    for key, value in session.environment().items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("ARC_LLM_WORKER_CONTEXT", "true")

    assert worker_cli.main(["extract-ids", "0911.3380", "--json"]) == 0
    assert _output(capsys)["data"] == ["arXiv:0911.3380"]


def test_worker_rejects_every_nested_llm_entrypoint_before_dispatch(monkeypatch, capsys):
    monkeypatch.setattr(worker_cli.cli, "main", lambda _argv: (_ for _ in ()).throw(AssertionError()))
    commands = [
        ["llm-infer-main-references", "paper"],
        ["infer-main-references", "paper"],
        ["get-llm-summary", "0911.3380"],
        ["llm-summary", "0911.3380"],
        ["generate-llm-summary", "0911.3380"],
        ["llm-generate-summary", "0911.3380"],
        ["summary-batch", "run", "batch"],
    ]

    for command in commands:
        assert worker_cli.main(command) == 1
        assert _output(capsys)["error"]["code"] == "nested_llm_forbidden"


def test_worker_delegates_safe_command_and_preserves_result(monkeypatch, capsys):
    expected = {"ok": True, "data": {"title": "Example"}, "errors": [], "meta": {}}

    def fake_main(argv):
        assert argv == ["get-title", "0911.3380", "--json"]
        print(json.dumps(expected))
        return 0

    monkeypatch.setattr(worker_cli.cli, "main", fake_main)
    assert worker_cli.main(["get-title", "0911.3380", "--json"]) == 0
    assert _output(capsys) == expected


def test_worker_defers_unclassified_command_to_canonical_cli(monkeypatch, capsys):
    assert worker_cli.main(["future-command"]) == 1
    assert _output(capsys)["error"]["code"] == "worker_arguments_invalid"


def test_worker_capability_metadata_defaults_new_deterministic_command_to_allowed(monkeypatch, capsys):
    expected = {"ok": True, "data": "future", "errors": [], "meta": {}}

    def fake_main(argv):
        assert argv == ["future-deterministic-command"]
        print(json.dumps(expected))
        return 0

    monkeypatch.setattr(worker_cli.cli, "main", fake_main)
    assert worker_cli.main(["future-deterministic-command"]) == 0
    assert _output(capsys) == expected


def test_large_result_is_externalized_and_pageable(monkeypatch, tmp_path, capsys):
    expected = {"ok": True, "data": "é" * 100, "errors": [], "meta": {}}

    def fake_main(_argv):
        print(json.dumps(expected))
        return 0

    monkeypatch.setattr(worker_cli.cli, "main", fake_main)
    common = ["--session-dir", str(tmp_path), "--max-inline-bytes", "32"]
    assert worker_cli.main(common + ["get-title", "0911.3380"]) == 0
    result = _output(capsys)
    artifact = result["data"]["artifact"]
    assert artifact["size_bytes"] > 32
    assert (tmp_path / "artifacts" / artifact["handle"]).is_file()

    read_args = ["--session-dir", str(tmp_path), "artifact-read", artifact["handle"], "--limit", "17"]
    assert worker_cli.main(read_args) == 0
    page = _output(capsys)["data"]
    assert len(base64.b64decode(page["content"])) == 17
    assert page["next_offset"] == 17


def test_audit_records_ids_and_argument_shape_without_values(monkeypatch, tmp_path, capsys):
    def fake_main(_argv):
        print(json.dumps({"ok": True, "data": [], "errors": [], "meta": {"provider": "cache"}}))
        return 0

    monkeypatch.setattr(worker_cli.cli, "main", fake_main)
    secret = "private-search-phrase"
    argv = [
        "--session-dir",
        str(tmp_path),
        "--worker-id",
        "w1",
        "search-full-text",
        "0911.3380",
        "--query",
        secret,
    ]
    assert worker_cli.main(argv) == 0
    _output(capsys)
    raw_audit = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    assert secret not in raw_audit
    event = json.loads(raw_audit)
    assert event["paper_ids"] == ["arXiv:0911.3380"]
    assert event["parameters"]["flags"] == ["--query"]
    assert event["worker"] == "w1"


def test_artifact_handle_cannot_escape_session(tmp_path, capsys):
    assert worker_cli.main(["--session-dir", str(tmp_path), "artifact-read", "../secret"]) == 1
    assert _output(capsys)["error"]["code"] == "artifact_handle_invalid"


def test_artifact_read_detects_content_tampering(monkeypatch, tmp_path, capsys):
    expected = {"ok": True, "data": "x" * 100, "errors": [], "meta": {}}

    def fake_main(_argv):
        print(json.dumps(expected))
        return 0

    monkeypatch.setattr(worker_cli.cli, "main", fake_main)
    common = ["--session-dir", str(tmp_path), "--max-inline-bytes", "32"]
    assert worker_cli.main(common + ["get-title", "0911.3380"]) == 0
    artifact = _output(capsys)["data"]["artifact"]
    (tmp_path / "artifacts" / artifact["handle"]).write_text("tampered", encoding="utf-8")

    assert worker_cli.main(["--session-dir", str(tmp_path), "artifact-read", artifact["handle"]]) == 1
    assert _output(capsys)["error"]["code"] == "artifact_integrity_failed"


def test_worker_session_stages_and_audits_pending_after_success(monkeypatch, tmp_path, capsys):
    base = tmp_path / "base"
    base.mkdir()
    session = WorkerCacheSession(base_root=base, run_root=tmp_path / "run", session_id="s1")
    for key, value in session.environment().items():
        monkeypatch.setenv(key, value)
    # The controller passes paths before the wrapper activates ARC_PAPER_CACHE.
    monkeypatch.delenv("ARC_PAPER_CACHE")

    def fake_main(_argv):
        overlay = os.environ["ARC_PAPER_CACHE"]
        assert overlay == str(session.overlay_root)
        session.stage_bytes(
            "papers/example.json",
            b'{"schema_version":"test.v1"}',
            source={"operation": "test"},
        )
        print(json.dumps({"ok": True, "data": [], "errors": [], "meta": {}}))
        return 0

    monkeypatch.setattr(worker_cli.cli, "main", fake_main)
    assert worker_cli.main(["--worker-id", "w1", "get-title", "0911.3380"]) == 0
    result = _output(capsys)
    assert result["meta"]["overlay_promotion"] == {"status": "pending_controller"}
    assert not (base / "papers" / "example.json").exists()
    assert (session.overlay_root / "papers" / "example.json").is_file()
    event = json.loads(session.audit_path.read_text(encoding="utf-8"))
    assert event["status"] == "success"
    assert event["worker_id"] == "w1"
    assert event["paper_ids"] == ["arXiv:0911.3380"]
    assert event["promotion_status"] == "pending"

    promotion = session.promote()
    assert promotion.promoted == ("papers/example.json",)
    assert (base / "papers" / "example.json").is_file()


def test_worker_session_stages_and_audits_pending_after_command_failure(monkeypatch, tmp_path, capsys):
    base = tmp_path / "base"
    base.mkdir()
    session = WorkerCacheSession(base_root=base, run_root=tmp_path / "run", session_id="s1")
    for key, value in session.environment().items():
        monkeypatch.setenv(key, value)

    def failing_main(_argv):
        session.stage_bytes(
            "papers/from-failed-call.json",
            b'{"schema_version":"test.v1"}',
            source={"operation": "test"},
        )
        raise RuntimeError("expected failure")

    monkeypatch.setattr(worker_cli.cli, "main", failing_main)
    assert worker_cli.main(["get-title", "0911.3380"]) == 1
    result = _output(capsys)
    assert result["error"]["code"] == "worker_command_failed"
    assert result["meta"]["overlay_promotion"] == {"status": "pending_controller"}
    assert not (base / "papers" / "from-failed-call.json").exists()
    assert (session.overlay_root / "papers" / "from-failed-call.json").is_file()
    event = json.loads(session.audit_path.read_text(encoding="utf-8"))
    assert event["status"] == "failed"


def test_worker_session_finishes_after_keyboard_cancel(monkeypatch, tmp_path, capsys):
    base = tmp_path / "base"
    base.mkdir()
    session = WorkerCacheSession(base_root=base, run_root=tmp_path / "run", session_id="s1")
    for key, value in session.environment().items():
        monkeypatch.setenv(key, value)

    def cancelled_main(_argv):
        raise KeyboardInterrupt

    monkeypatch.setattr(worker_cli.cli, "main", cancelled_main)
    assert worker_cli.main(["get-title", "0911.3380"]) == 1
    result = _output(capsys)
    assert result["error"]["code"] == "worker_command_cancelled"
    event = json.loads(session.audit_path.read_text(encoding="utf-8"))
    assert event["status"] == "cancelled"


def test_summary_batch_export_rejects_output_outside_run_root(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(worker_cli.cli, "main", lambda _argv: (_ for _ in ()).throw(AssertionError()))
    run_root = tmp_path / "run"

    assert worker_cli.main(
        [
            "--session-dir",
            str(run_root),
            "summary-batch",
            "export",
            "batch",
            "--output",
            str(tmp_path / "escaped.jsonl"),
        ]
    ) == 1
    assert _output(capsys)["error"]["code"] == "worker_output_path_forbidden"


def test_summary_batch_export_rejects_symlink_escape(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(worker_cli.cli, "main", lambda _argv: (_ for _ in ()).throw(AssertionError()))
    run_root = tmp_path / "run"
    outside = tmp_path / "outside"
    run_root.mkdir()
    outside.mkdir()
    (run_root / "link").symlink_to(outside, target_is_directory=True)

    assert worker_cli.main(
        [
            "--session-dir",
            str(run_root),
            "summary-batch",
            "export",
            "batch",
            "--output",
            "link/escaped.jsonl",
        ]
    ) == 1
    assert _output(capsys)["error"]["code"] == "worker_output_path_forbidden"


def test_summary_batch_export_normalizes_relative_output_into_run_root(
    monkeypatch, tmp_path, capsys
):
    run_root = tmp_path / "run"
    run_root.mkdir()

    def fake_main(argv):
        output_index = argv.index("--output") + 1
        assert Path(argv[output_index]) == run_root / "exports" / "batch.jsonl"
        print(json.dumps({"ok": True, "data": {}, "errors": [], "meta": {}}))
        return 0

    monkeypatch.setattr(worker_cli.cli, "main", fake_main)
    assert worker_cli.main(
        [
            "--session-dir",
            str(run_root),
            "summary-batch",
            "export",
            "batch",
            "--output",
            "exports/batch.jsonl",
        ]
    ) == 0
    assert _output(capsys)["ok"] is True


def test_final_envelope_with_session_metadata_obeys_inline_boundary(
    monkeypatch, tmp_path, capsys
):
    base = tmp_path / "base"
    base.mkdir()
    session = WorkerCacheSession(base_root=base, run_root=tmp_path / "run", session_id="s1")
    for key, value in session.environment().items():
        monkeypatch.setenv(key, value)

    # Compact JSON is below 64 KiB while pretty-printed JSON is above it.  This
    # catches checks performed before promotion/audit metadata or on the wrong
    # serialization form.
    data = [{"key": "value", "n": index} for index in range(1800)]
    expected = {"ok": True, "data": data, "errors": [], "meta": {}}
    assert len(worker_cli._canonical_json(expected)) < worker_cli.MAX_INLINE_BYTES
    assert len(worker_cli._display_json(expected)) > worker_cli.MAX_INLINE_BYTES

    def fake_main(_argv):
        print(json.dumps(expected))
        return 0

    monkeypatch.setattr(worker_cli.cli, "main", fake_main)
    assert worker_cli.main(["get-title", "0911.3380"]) == 0
    captured = capsys.readouterr().out
    assert len(captured.encode("utf-8")) <= worker_cli.MAX_INLINE_BYTES
    result = json.loads(captured)
    assert result["meta"]["externalized"] is True
    assert result["meta"]["overlay_promotion"] == {"status": "pending_controller"}
    assert result["meta"]["worker_audit"]["status"] == "recorded"
    artifact = result["data"]["artifact"]
    stored = json.loads((session.run_root / "artifacts" / artifact["handle"]).read_text())
    assert stored["meta"]["overlay_promotion"] == {"status": "pending_controller"}
    assert stored["meta"]["worker_audit"]["status"] == "recorded"
