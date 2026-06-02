from __future__ import annotations

import time

import pytest

from arc_mcp import cli
from arc_mcp import worker
from arc_mcp.jobs import MCPJobCancelled, MCPJobManager, resolve_inline_wait_seconds


PROCESS_WORKER_TEST_TIMEOUT = 10


def test_job_manager_runs_and_records_progress(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_MCP_CACHE", str(tmp_path))
    manager = MCPJobManager(max_workers=1, worker_mode="thread")

    def runner(progress, cancel):
        progress({"event": "started", "step": 1})
        return {"ok": True, "data": {"value": 3}, "errors": [], "meta": {}}

    job_id = manager.start(job_type="test", payload={"paper_id": "arXiv:1"}, runner=runner)

    assert manager.wait(job_id, timeout=1) is True
    status = manager.status(job_id)
    assert status["status"] == "done"
    assert status["phase"] == "done"
    assert status["step"] == 1
    assert status["payload"] == {"paper_id": "arXiv:1"}
    assert manager.result(job_id)["result"]["data"]["value"] == 3


def test_job_manager_rejects_reserved_payload_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_MCP_CACHE", str(tmp_path))
    manager = MCPJobManager(max_workers=1, worker_mode="thread")

    with pytest.raises(ValueError, match="reserved job status keys"):
        manager.start(job_type="test", payload={"status": "done"}, runner=lambda progress, cancel: {})


def test_job_manager_returns_not_ready_before_deadline(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_MCP_CACHE", str(tmp_path))
    manager = MCPJobManager(max_workers=1, worker_mode="thread")

    def runner(progress, cancel):
        time.sleep(0.05)
        return {"ok": True, "data": {}, "errors": [], "meta": {}}

    job_id = manager.start(job_type="slow", payload={}, runner=runner)

    assert manager.wait(job_id, timeout=0.001) is False
    assert manager.status(job_id)["status"] in {"queued", "running"}
    assert manager.wait(job_id, timeout=1) is True


def test_job_manager_can_cancel_queued_job(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_MCP_CACHE", str(tmp_path))
    manager = MCPJobManager(max_workers=1, worker_mode="thread")

    def blocker(progress, cancel):
        time.sleep(0.1)
        return {"ok": True, "data": {}, "errors": [], "meta": {}}

    def queued(progress, cancel):
        raise AssertionError("queued job should not run")

    first = manager.start(job_type="blocker", payload={}, runner=blocker)
    second = manager.start(job_type="queued", payload={}, runner=queued)

    cancelled = manager.cancel(second)

    assert cancelled["status"] == "cancelled"
    assert manager.wait(first, timeout=1) is True


def test_job_manager_cooperative_cancel(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_MCP_CACHE", str(tmp_path))
    manager = MCPJobManager(max_workers=1, worker_mode="thread")

    def runner(progress, cancel):
        progress({"event": "started"})
        time.sleep(0.02)
        if cancel():
            raise MCPJobCancelled("stop")
        return {"ok": True, "data": {}, "errors": [], "meta": {}}

    job_id = manager.start(job_type="cancel", payload={}, runner=runner)
    time.sleep(0.005)
    manager.cancel(job_id)

    assert manager.wait(job_id, timeout=1) is True
    assert manager.status(job_id)["status"] == "cancelled"


def test_job_manager_eta_uses_persisted_history(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_MCP_CACHE", str(tmp_path))
    manager = MCPJobManager(max_workers=1, worker_mode="thread")

    def quick(progress, cancel):
        time.sleep(0.001)
        return {"ok": True, "data": {}, "errors": [], "meta": {}}

    for _ in range(3):
        job_id = manager.start(job_type="eta_test", payload={"provider": "manual"}, runner=quick)
        assert manager.wait(job_id, timeout=1) is True

    def slow(progress, cancel):
        time.sleep(0.05)
        return {"ok": True, "data": {}, "errors": [], "meta": {}}

    running = manager.start(job_type="eta_test", payload={"provider": "manual"}, runner=slow)
    time.sleep(0.005)
    status = manager.status(running)

    assert status["eta"]["available"] is True
    assert status["eta"]["samples"] == 3
    assert manager.wait(running, timeout=1) is True


def test_resolve_inline_wait_uses_env_timeout_minus_margin():
    env = {"ARC_MCP_TOOL_TIMEOUT_SEC": "120", "ARC_MCP_BACKGROUND_MARGIN_SEC": "10"}

    assert resolve_inline_wait_seconds(env=env) == 110


def test_resolve_inline_wait_explicit_override():
    env = {"ARC_MCP_INLINE_WAIT_SEC": "3", "ARC_MCP_TOOL_TIMEOUT_SEC": "120"}

    assert resolve_inline_wait_seconds(env=env) == 3


def test_worker_bool_arg_parses_string_booleans():
    assert worker._bool_arg("false") is False  # noqa: SLF001
    assert worker._bool_arg("0") is False  # noqa: SLF001
    assert worker._bool_arg("true") is True  # noqa: SLF001
    assert worker._bool_arg("on") is True  # noqa: SLF001
    assert worker._bool_arg(None, True) is True  # noqa: SLF001


def test_worker_bool_arg_rejects_invalid_values():
    with pytest.raises(ValueError, match="Expected boolean string"):
        worker._bool_arg("flase")  # noqa: SLF001
    with pytest.raises(ValueError, match="Expected boolean"):
        worker._bool_arg(2)  # noqa: SLF001


def test_worker_keeps_heavy_service_imports_lazy():
    assert not hasattr(worker, "domain_service")
    assert not hasattr(worker, "paper_service")
    assert not hasattr(worker, "run_batch")
    assert not hasattr(worker, "typeset_md2pdf")
    assert not hasattr(worker, "typeset_translate")


def test_resolve_inline_wait_reads_codex_config(tmp_path):
    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        """
[mcp_servers.arc]
command = "arc-mcp"
tool_timeout_sec = 240
""".strip(),
        encoding="utf-8",
    )

    assert resolve_inline_wait_seconds(env={"CODEX_HOME": str(codex_home)}, server_name="arc") == 230


def test_process_worker_persists_failed_status(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_MCP_CACHE", str(tmp_path))
    manager = MCPJobManager(max_workers=1)

    job_id = manager.start(job_type="unsupported_test_job", payload={})

    launch_status = manager.status(job_id)
    assert launch_status["status"] in {"queued", "running", "failed"}
    assert launch_status.get("phase") in {"queued", "worker_launching", "running", "failed"}
    assert manager.wait(job_id, timeout=PROCESS_WORKER_TEST_TIMEOUT) is True
    status = manager.status(job_id)
    assert status["status"] == "failed"
    assert status["error"]["code"] == "job_failed"


def test_process_worker_launch_failure_marks_job_failed(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_MCP_CACHE", str(tmp_path))

    def fail_popen(*args, **kwargs):
        raise OSError("spawn failed")

    monkeypatch.setattr("subprocess.Popen", fail_popen)
    manager = MCPJobManager(max_workers=1)

    job_id = manager.start(job_type="unsupported_test_job", payload={})
    status = manager.status(job_id)

    assert status["status"] == "failed"
    assert status["error"]["code"] == "job_worker_launch_failed"
    assert "spawn failed" in status["error"]["message"]


def test_cli_accepts_flat_job_commands(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ARC_MCP_CACHE", str(tmp_path))

    assert cli.main(["root", "--json"]) == 0
    flat = capsys.readouterr().out
    assert str(tmp_path) in flat

    assert cli.main(["jobs", "root", "--json"]) == 0
    nested = capsys.readouterr().out
    assert str(tmp_path) in nested


def test_worker_dispatches_md2pdf_job(monkeypatch, tmp_path):
    from arc_typeset import md2pdf as typeset_md2pdf

    source = tmp_path / "report.md"
    output = tmp_path / "report.pdf"
    source.write_text("# Report\n", encoding="utf-8")
    calls = {}
    events = []

    def convert_markdown_to_pdf(**kwargs):
        calls.update(kwargs)
        return {
            "ok": True,
            "data": {"input_path": str(source), "output_path": str(output), "pdf_size_bytes": 8},
            "errors": [],
            "meta": {},
        }

    monkeypatch.setattr(worker, "is_cancel_requested", lambda job_id: False)
    monkeypatch.setattr(worker, "record_progress", lambda job_id, event: events.append(event))
    monkeypatch.setattr(typeset_md2pdf, "convert_markdown_to_pdf", convert_markdown_to_pdf)

    result = worker._dispatch(
        "md2pdf",
        {
            "input": str(source),
            "output": str(output),
            "texlive_bin": "",
            "margin": "2cm",
            "mainfont": "Noto Sans",
            "cjk_mainfont": "Noto Sans CJK SC",
            "resource_path": [str(tmp_path)],
        },
        job_id="job-test",
    )

    assert result["ok"] is True
    assert result["data"]["output_path"] == str(output)
    assert calls["input_path"] == source
    assert calls["output_path"] == output
    assert calls["texlive_bin"] is None
    assert calls["resource_paths"] == [tmp_path]
    assert [event["event"] for event in events] == ["md2pdf_started", "md2pdf_completed"]


def test_worker_dispatches_translate_job(monkeypatch, tmp_path):
    from arc_typeset import translate as typeset_translate

    source = tmp_path / "report.md"
    output = tmp_path / "report.zh_CN.md"
    source.write_text("# Report\n", encoding="utf-8")
    calls = {}
    events = []

    def translate_markdown(**kwargs):
        calls.update(kwargs)
        return {
            "ok": True,
            "data": {
                "input_markdown_path": str(source),
                "output_markdown_path": str(output),
                "output_pdf_path": str(output.with_suffix(".pdf")),
            },
            "errors": [],
            "meta": {},
        }

    monkeypatch.setattr(worker, "is_cancel_requested", lambda job_id: False)
    monkeypatch.setattr(worker, "record_progress", lambda job_id, event: events.append(event))
    monkeypatch.setattr(typeset_translate, "translate_markdown", translate_markdown)

    result = worker._dispatch(
        "translate",
        {
            "input": str(source),
            "output": str(output),
            "target_language": "Chinese",
            "target_locale": "zh_CN",
            "provider": "manual",
            "model": "test-model",
            "model_tier": "low",
            "quality": False,
        },
        job_id="job-test",
    )

    assert result["ok"] is True
    assert result["data"]["output_markdown_path"] == str(output)
    assert calls["input_path"] == source
    assert calls["output_path"] == output
    assert calls["model_tier"] == "low"
    assert calls["convert_pdf"] is True
    assert [event["event"] for event in events] == ["translate_started", "translate_completed"]


def test_worker_dispatches_batch_translate_job(monkeypatch, tmp_path):
    from arc_typeset import translate as typeset_translate

    calls = {}
    events = []

    def batch_translate_project(**kwargs):
        calls.update(kwargs)
        return {
            "ok": True,
            "data": {"project_dir": str(tmp_path), "candidate_count": 1, "translated_count": 1},
            "errors": [],
            "meta": {},
        }

    monkeypatch.setattr(worker, "is_cancel_requested", lambda job_id: False)
    monkeypatch.setattr(worker, "record_progress", lambda job_id, event: events.append(event))
    monkeypatch.setattr(typeset_translate, "batch_translate_project", batch_translate_project)

    result = worker._dispatch(
        "batch_translate",
        {"project_dir": str(tmp_path), "target_language": "Chinese", "target_locale": "zh_CN", "model_tier": "low"},
        job_id="job-test",
    )

    assert result["ok"] is True
    assert result["data"]["translated_count"] == 1
    assert calls["project_dir"] == tmp_path
    assert calls["target_language"] == "Chinese"
    assert calls["model_tier"] == "low"
    assert [event["event"] for event in events] == ["batch_translate_started", "batch_translate_completed"]


def test_worker_dispatches_domain_build_model_tier(monkeypatch):
    from arc_domain import service as domain_service

    calls = {}
    events = []

    def build_domain(seed_paper, **kwargs):
        kwargs["seed_paper"] = seed_paper
        calls.update(kwargs)
        return {"ok": True, "data": {"domain_id": "domain-test"}, "errors": [], "meta": {}}

    monkeypatch.setattr(worker, "is_cancel_requested", lambda job_id: False)
    monkeypatch.setattr(worker, "record_progress", lambda job_id, event: events.append(event))
    monkeypatch.setattr(domain_service, "build_domain", build_domain)

    result = worker._dispatch(
        "domain_build",
        {"seed_paper": "arXiv:0911.3380", "intent": "inflation", "provider": "auto", "model_tier": "high"},
        job_id="job-test",
    )

    assert result["ok"] is True
    assert calls["seed_paper"] == "arXiv:0911.3380"
    assert calls["model_tier"] == "high"
    assert [event["event"] for event in events] == ["domain_started", "domain_completed"]
