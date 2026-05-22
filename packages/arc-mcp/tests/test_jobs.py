from __future__ import annotations

import time

from arc_mcp.jobs import MCPJobCancelled, MCPJobManager, resolve_inline_wait_seconds


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
    assert manager.result(job_id)["result"]["data"]["value"] == 3


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

    assert manager.wait(job_id, timeout=5) is True
    status = manager.status(job_id)
    assert status["status"] == "failed"
    assert status["error"]["code"] == "job_failed"
