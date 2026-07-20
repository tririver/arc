from __future__ import annotations

"""Tests for the thin arc-mcp adapter over arc-jobs."""

import json

from arc_jobs import JobCancelled, JobManager
from arc_jobs.jobs import cache_root
from arc_jobs.worker import run_job as generic_run_job

from arc_mcp import cli, worker
from arc_mcp.jobs import MCPJobCancelled, MCPJobManager, resolve_inline_wait_seconds


def test_arc_mcp_job_api_is_a_thin_arc_jobs_adapter():
    assert MCPJobManager is JobManager
    assert MCPJobCancelled is JobCancelled
    assert worker.run_job is generic_run_job


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


def test_generic_cache_does_not_read_arc_mcp_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("ARC_JOBS_CACHE", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("ARC_MCP_CACHE", str(tmp_path / "legacy"))
    monkeypatch.setattr("arc_jobs.jobs._project_root", lambda: None)
    monkeypatch.setattr("arc_jobs.jobs.Path.home", lambda: tmp_path / "home")

    assert cache_root() == tmp_path / "home" / ".cache" / "arc" / "arc-jobs"


def test_cli_md2pdf_starts_background_job_with_payload(monkeypatch, tmp_path, capsys):
    source = tmp_path / "report.md"
    output = tmp_path / "report.pdf"
    source.write_text("# Report\n", encoding="utf-8")
    calls = {}

    class FakeServer:
        def call_tool(self, name, args):
            calls["name"] = name
            calls["args"] = args
            return {
                "ok": False,
                "status": "job_running",
                "job_id": "job-123",
                "job_type": "md2pdf",
                "next": {"cli_command": "arc-jobs watch job-123 --json"},
                "errors": [],
                "meta": {},
            }

    monkeypatch.setattr(cli, "_server", lambda: FakeServer())

    exit_code = cli.main(
        [
            "md2pdf",
            str(source),
            "--output",
            str(output),
            "--texlive-bin",
            "",
            "--margin",
            "2cm",
            "--mainfont",
            "Noto Sans",
            "--cjk-mainfont",
            "Noto Sans CJK SC",
            "--resource-path",
            str(tmp_path),
            "--timeout-seconds",
            "12",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "job_running"
    assert calls["name"] == "md2pdf"
    assert calls["args"] == {
        "input": str(source),
        "output": str(output),
        "texlive_bin": "",
        "margin": "2cm",
        "mainfont": "Noto Sans",
        "cjk_mainfont": "Noto Sans CJK SC",
        "resource_path": [str(tmp_path)],
        "timeout_seconds": 12.0,
    }


def test_cli_md2pdf_returns_nonzero_when_launch_fails(monkeypatch, tmp_path, capsys):
    source = tmp_path / "report.md"
    source.write_text("# Report\n", encoding="utf-8")

    class FakeServer:
        def call_tool(self, name, args):
            return {
                "ok": False,
                "error": {"code": "invalid_timeout", "message": "timeout_seconds must be positive"},
                "errors": [],
                "meta": {},
            }

    monkeypatch.setattr(cli, "_server", lambda: FakeServer())

    assert cli.main(["md2pdf", str(source), "--timeout-seconds", "-1", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "invalid_timeout"


def test_arc_mcp_cli_job_view_reads_arc_jobs_cache(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path))

    assert cli.main(["root", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["data"]["cache_root"] == str(tmp_path)


def test_arc_mcp_cli_job_views_return_nonzero_for_unknown(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path))

    for command in ("status", "result", "watch"):
        assert cli.main([command, "missing", "--json"]) == 1
        assert json.loads(capsys.readouterr().out)["status"] == "job_unknown"


def test_arc_mcp_cli_job_views_return_nonzero_for_failed(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path))
    manager = JobManager(worker_mode="thread")
    job_id = manager.start(
        job_type="failed",
        payload={},
        runner=lambda progress, cancel: {"ok": False},
    )
    assert manager.wait(job_id, timeout=2)

    for command in ("status", "result", "watch"):
        assert cli.main([command, job_id, "--json"]) == 1
        assert json.loads(capsys.readouterr().out)["status"] == "failed"


def test_arc_mcp_cli_job_views_return_nonzero_for_cancelled(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path))
    manager = JobManager(worker_mode="process")
    monkeypatch.setattr(manager, "_launch_worker", lambda job_id: None)
    job_id = manager.submit(["arc-paper", "--json"])
    assert manager.cancel(job_id)["status"] == "cancelled"

    for command in ("status", "result", "watch"):
        assert cli.main([command, job_id, "--json"]) == 1
        assert json.loads(capsys.readouterr().out)["status"] == "cancelled"
