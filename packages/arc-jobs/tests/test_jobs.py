from __future__ import annotations

import json
import os
import stat
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from arc_jobs import cli, worker
from arc_jobs import jobs as jobs_module
from arc_jobs.jobs import (
    JobCancelled,
    JobManager,
    JobPaths,
    acquire_worker_lock,
    arc_jobs_cli_argv,
    read_json,
    release_worker_lock,
    tail_events,
    validate_arc_argv,
    write_json,
)


def _install_fake_cli(tmp_path: Path, monkeypatch, *, body: str) -> Path:
    scripts = tmp_path / "bin"
    scripts.mkdir(exist_ok=True)
    python = scripts / "python"
    python.write_text("", encoding="utf-8")
    command = scripts / "arc-paper"
    command.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    command.chmod(0o755)
    monkeypatch.setattr("arc_jobs.jobs.runtime_script_dirs", lambda: (scripts,))
    return command


def test_thread_job_compatibility_and_protocol_neutral_schema(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path))
    manager = JobManager(max_workers=1, worker_mode="thread")

    def runner(progress, cancel):
        progress({"event": "step_finished", "sections_completed": 1, "sections_total": 2})
        return {"ok": True, "data": {"value": 3}, "errors": [], "meta": {}}

    job_id = manager.start(job_type="test", payload={"paper_id": "1"}, runner=runner)

    assert manager.wait(job_id, timeout=2)
    status = manager.status(job_id)
    assert status["schema_version"] == "arc.job_status.v1"
    assert status["status"] == "done"
    assert status["progress"] == {"completed": 1, "total": 2}
    assert status["events"][0]["schema_version"] == "arc.job_event.v1"
    assert manager.result(job_id)["result"]["data"]["value"] == 3
    stored = json.loads(JobPaths.for_job(job_id).result.read_text(encoding="utf-8"))
    assert stored["schema_version"] == "arc.job_result.v1"


def test_thread_job_cooperative_cancel(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path))
    manager = JobManager(max_workers=1, worker_mode="thread")

    def runner(progress, cancel):
        progress({"event": "started"})
        time.sleep(0.03)
        if cancel():
            raise JobCancelled("stop")
        return {"ok": True}

    job_id = manager.start(job_type="test", payload={}, runner=runner)
    time.sleep(0.005)
    manager.cancel(job_id)

    assert manager.wait(job_id, timeout=2)
    assert manager.status(job_id)["status"] == "cancelled"


def test_process_job_requires_argv(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path))
    manager = JobManager(worker_mode="process")

    with pytest.raises(ValueError, match="require an allowlisted"):
        manager.start(job_type="old-dispatch", payload={})


@pytest.mark.parametrize("command", ["arc-mcp", "arc-jobs", "python", "sh"])
def test_command_allowlist_rejects_mcp_self_and_general_commands(tmp_path, monkeypatch, command):
    _install_fake_cli(tmp_path, monkeypatch, body="exit 0")

    with pytest.raises(ValueError, match="not allowed"):
        validate_arc_argv([command, "--json"])


def test_argv_rejects_shell_strings_and_nul(tmp_path, monkeypatch):
    _install_fake_cli(tmp_path, monkeypatch, body="exit 0")

    with pytest.raises(ValueError, match="not a shell command string"):
        validate_arc_argv("arc-paper --json")
    with pytest.raises(ValueError, match="NUL"):
        validate_arc_argv(["arc-paper", "bad\x00arg"])


def test_explicit_command_must_match_same_runtime(tmp_path, monkeypatch):
    _install_fake_cli(tmp_path, monkeypatch, body="exit 0")
    elsewhere = tmp_path / "elsewhere" / "arc-paper"
    elsewhere.parent.mkdir()
    elsewhere.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    elsewhere.chmod(0o755)

    with pytest.raises(ValueError, match="not the allowlisted executable"):
        validate_arc_argv([str(elsewhere), "--json"])


def test_worker_executes_argv_without_shell_and_persists_streams(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path / "cache"))
    _install_fake_cli(
        tmp_path,
        monkeypatch,
        body="printf '%s' '{\"ok\": true, \"data\": {\"value\": 7}}'; printf '%s' 'warning' >&2",
    )
    manager = JobManager(worker_mode="thread")
    # Persist a process-shaped job without launching a second interpreter so patched
    # runtime resolution remains active in this test process.
    job_id = manager.start(
        job_type="cli",
        payload={},
        argv=["arc-paper", "--json", "; touch", str(tmp_path / "pwned")],
        runner=lambda progress, cancel: {"unused": True},
    )
    assert manager.wait(job_id, timeout=2)
    # Re-run as a process worker against a fresh persisted process job.
    monkeypatch.setattr(manager, "_use_thread_worker", lambda: False)
    monkeypatch.setattr(manager, "_launch_worker", lambda job_id: None)
    process_job_id = manager.submit(["arc-paper", "--json", "; touch", str(tmp_path / "pwned")])

    assert worker.run_job(process_job_id) == 0
    result = manager.result(process_job_id)
    assert result["status"] == "done"
    assert result["result"]["output"]["data"]["value"] == 7
    paths = JobPaths.for_job(process_job_id)
    assert paths.stderr.read_text(encoding="utf-8") == "warning"
    assert not (tmp_path / "pwned").exists()


def test_process_job_runs_in_explicit_working_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path / "cache"))
    work = tmp_path / "project"
    work.mkdir()
    _install_fake_cli(
        tmp_path,
        monkeypatch,
        body="printf '{\"ok\": true, \"data\": {\"cwd\": \"%s\"}}' \"$PWD\"",
    )
    manager = JobManager(worker_mode="process")
    monkeypatch.setattr(manager, "_launch_worker", lambda job_id: None)
    job_id = manager.submit(["arc-paper", "--json"], cwd=work)

    assert worker.run_job(job_id) == 0
    result = manager.result(job_id)
    assert result["result"]["cwd"] == str(work.resolve())
    assert result["result"]["output"]["data"]["cwd"] == str(work.resolve())


def test_worker_persists_nonzero_exit_and_json_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path / "cache"))
    _install_fake_cli(tmp_path, monkeypatch, body="printf 'boom' >&2; exit 9")
    manager = JobManager(worker_mode="process")
    monkeypatch.setattr(manager, "_launch_worker", lambda job_id: None)
    job_id = manager.submit(["arc-paper", "--json"])

    assert worker.run_job(job_id) == 1
    result = manager.result(job_id)
    assert result["status"] == "failed"
    assert result["error"]["code"] == "job_command_failed"
    assert result["error"]["exit_code"] == 9
    assert result["result"]["exit_code"] == 9


def test_cancel_terminates_running_command(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path / "cache"))
    _install_fake_cli(tmp_path, monkeypatch, body="sleep 10")
    manager = JobManager(worker_mode="process")
    monkeypatch.setattr(manager, "_launch_worker", lambda job_id: None)
    job_id = manager.submit(["arc-paper"])

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(worker.run_job, job_id)
        deadline = time.monotonic() + 2
        while manager.status(job_id).get("phase") != "command_running":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        manager.cancel(job_id)
        assert future.result(timeout=2) == 0

    assert manager.status(job_id)["status"] == "cancelled"
    paths = JobPaths.for_job(job_id)
    assert paths.stdout.exists()
    assert paths.stderr.exists()


def test_cli_submit_and_all_json_commands(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path / "cache"))
    _install_fake_cli(tmp_path, monkeypatch, body="printf '%s' '{\"ok\": true}'")
    launched: list[str] = []
    monkeypatch.setattr(JobManager, "_launch_worker", lambda self, job_id: launched.append(job_id))

    assert cli.main(["submit", "--cwd", str(tmp_path), "--json", "--", "arc-paper", "--json"]) == 0
    submitted = json.loads(capsys.readouterr().out)
    job_id = submitted["job_id"]
    assert submitted["ok"] is True
    assert submitted["cwd"] == str(tmp_path)
    assert launched == [job_id]

    assert cli.main(["list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["data"]["jobs"][0]["job_id"] == job_id
    assert cli.main(["status", job_id, "--json"]) == 0
    queued = json.loads(capsys.readouterr().out)
    assert queued["status"] == "queued"
    assert queued["ok"] is True
    assert cli.main(["result", job_id, "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "queued"
    assert cli.main(["cancel", job_id, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] in {"cancel_requested", "cancelled"}


def test_cli_rejects_disallowed_submit_as_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path))

    assert cli.main(["submit", "--json", "--", "arc-mcp", "status", "x"]) == 1
    response = json.loads(capsys.readouterr().out)
    assert response["status"] == "invalid_request"
    assert response["error"]["code"] == "invalid_request"


def test_cli_catches_unexpected_errors_as_json(monkeypatch, capsys):
    monkeypatch.setattr(JobManager, "list_jobs", lambda self: (_ for _ in ()).throw(RuntimeError("boom")))

    assert cli.main(["list", "--json"]) == 1
    response = json.loads(capsys.readouterr().out)
    assert response["status"] == "internal_error"
    assert response["error"]["code"] == "internal_error"


def test_cli_status_is_nonzero_for_failed_job(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path / "cache"))
    manager = JobManager(worker_mode="thread")
    job_id = manager.start(
        job_type="failed",
        payload={},
        runner=lambda progress, cancel: {"ok": False},
    )
    assert manager.wait(job_id, timeout=2)

    assert cli.main(["status", job_id, "--json"]) == 1
    response = json.loads(capsys.readouterr().out)
    assert response["ok"] is True
    assert response["status"] == "failed"


def test_cli_watch_json_returns_terminal_result(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path))
    manager = JobManager(worker_mode="thread")
    job_id = manager.start(
        job_type="test",
        payload={},
        runner=lambda progress, cancel: {"ok": True, "data": {"finished": True}},
    )
    assert manager.wait(job_id, timeout=2)

    assert cli.main(["watch", job_id, "--interval", "0.01", "--json"]) == 0
    response = json.loads(capsys.readouterr().out)
    assert response["status"] == "done"
    assert response["result"]["data"]["finished"] is True


def test_cli_watch_progress_jsonl_emits_compact_events_and_terminal_result(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path))
    manager = JobManager(worker_mode="thread")
    job_id = manager.start(
        job_type="test",
        payload={},
        runner=lambda progress, cancel: {"ok": True, "data": {"finished": True}},
    )
    assert manager.wait(job_id, timeout=2)

    assert cli.main(["watch", job_id, "--progress-jsonl", "--json"]) == 0
    lines = capsys.readouterr().out.splitlines()
    records = [json.loads(line) for line in lines]
    assert all(
        line == json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        for line, record in zip(lines, records, strict=True)
    )
    assert all(record["job_id"] == job_id for record in records)
    assert all("event" in record for record in records[:-1])
    assert records[-1]["status"] == "done"
    assert records[-1]["result"]["data"]["finished"] is True


def test_cli_watch_progress_jsonl_emits_terminal_error_and_fails(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path))
    manager = JobManager(worker_mode="thread")

    def fail(progress, cancel):
        raise RuntimeError("boom")

    job_id = manager.start(job_type="test", payload={}, runner=fail)
    assert manager.wait(job_id, timeout=2)

    assert cli.main(["watch", job_id, "--progress-jsonl"]) == 1
    lines = capsys.readouterr().out.splitlines()
    records = [json.loads(line) for line in lines]
    assert all(
        line == json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        for line, record in zip(lines, records, strict=True)
    )
    assert records[-1]["ok"] is False
    assert records[-1]["status"] == "failed"
    assert records[-1]["error"]["code"] == "job_failed"


def test_cache_env_uses_protocol_neutral_name(tmp_path, monkeypatch):
    from arc_jobs.jobs import cache_root

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("ARC_JOBS_CACHE", raising=False)
    assert cache_root() == tmp_path / "xdg" / "arc" / "arc-jobs"


def test_stable_cli_invocation_preserves_virtualenv_symlink_path(tmp_path):
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    python.symlink_to(Path("/usr/bin/python3"))
    arc_jobs = bin_dir / "arc-jobs"
    arc_jobs.write_text("#!/bin/sh\n", encoding="utf-8")
    arc_jobs.chmod(0o755)

    assert arc_jobs_cli_argv("watch", "job", executable=python) == [
        str(arc_jobs),
        "watch",
        "job",
    ]


def test_worker_preserves_needs_llm_terminal_status(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path / "cache"))
    _install_fake_cli(
        tmp_path,
        monkeypatch,
        body="printf '%s' '{\"ok\": false, \"status\": \"needs_llm\"}'",
    )
    manager = JobManager(worker_mode="process")
    monkeypatch.setattr(manager, "_launch_worker", lambda job_id: None)
    job_id = manager.submit(["arc-paper", "--json"])

    assert worker.run_job(job_id) == 0
    assert manager.status(job_id)["status"] == "needs_llm"
    assert manager.result(job_id)["result"]["output"]["status"] == "needs_llm"


def test_cancel_terminal_job_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path))
    manager = JobManager(worker_mode="thread")
    job_id = manager.start(
        job_type="test",
        payload={},
        runner=lambda progress, cancel: {"ok": True},
    )
    assert manager.wait(job_id, timeout=2)

    assert manager.cancel(job_id)["status"] == "done"
    assert not JobPaths.for_job(job_id).cancel_request.exists()


def test_stale_worker_lock_is_reclaimed_by_process_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path / "cache"))
    manager = JobManager(worker_mode="process")
    monkeypatch.setattr(manager, "_launch_worker", lambda job_id: None)
    job_id = manager.submit(["arc-paper", "--json"])
    paths = JobPaths.for_job(job_id)
    write_json(paths.worker_lock, {"job_id": job_id, "pid": 99999999, "start_id": "old"})
    os.utime(paths.worker_lock, (time.time() - 10, time.time() - 10))

    assert acquire_worker_lock(job_id) is True
    assert read_json(paths.worker_lock)["pid"] == os.getpid()
    release_worker_lock(job_id)
    assert not paths.worker_lock.exists()


def test_status_restarts_pre_command_worker_after_crash(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path / "cache"))
    manager = JobManager(worker_mode="process")
    monkeypatch.setattr(manager, "_launch_worker", lambda job_id: None)
    job_id = manager.submit(["arc-paper", "--json"])
    paths = JobPaths.for_job(job_id)
    status = read_json(paths.status)
    status["updated_at"] = "2000-01-01T00:00:00+00:00"
    status["worker"] = {"pid": 99999999, "start_id": "old"}
    write_json(paths.status, status)
    write_json(paths.worker_lock, {"pid": 99999999, "start_id": "old"})
    os.utime(paths.worker_lock, (time.time() - 10, time.time() - 10))
    launched: list[str] = []

    def relaunch(recovered_job_id):
        launched.append(recovered_job_id)

    monkeypatch.setattr(manager, "_launch_worker", relaunch)

    recovered = manager.status(job_id)

    assert launched == [job_id]
    assert any(event["event"] == "job_worker_restarting" for event in recovered["events"])


def test_status_terminalizes_worker_that_repeatedly_fails_to_start(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path / "cache"))
    manager = JobManager(worker_mode="process")
    monkeypatch.setattr(manager, "_launch_worker", lambda job_id: None)
    job_id = manager.submit(["arc-paper", "--json"])
    paths = JobPaths.for_job(job_id)
    status = read_json(paths.status)
    status["worker_launch_attempts"] = jobs_module.MAX_WORKER_LAUNCH_ATTEMPTS
    status["worker"] = {"pid": 99999999, "start_id": "old"}
    write_json(paths.status, status)

    failed = manager.status(job_id)

    assert failed["status"] == "failed"
    assert failed["error"]["code"] == "job_worker_unavailable"


def test_status_terminates_orphaned_command_and_marks_terminal(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path / "cache"))
    manager = JobManager(worker_mode="process")
    monkeypatch.setattr(manager, "_launch_worker", lambda job_id: None)
    job_id = manager.submit(["arc-paper", "--json"])
    paths = JobPaths.for_job(job_id)
    status = read_json(paths.status)
    status["status"] = "running"
    status["worker"] = {"pid": 111, "start_id": "dead"}
    status["process"] = {"pid": 222, "start_id": "live", "argv": ["arc-paper"]}
    write_json(paths.status, status)
    monkeypatch.setattr(
        jobs_module,
        "_pid_record_alive",
        lambda record: record.get("pid") == 222,
    )
    terminated: list[dict] = []
    monkeypatch.setattr(
        jobs_module,
        "_terminate_recorded_process",
        lambda record: terminated.append(dict(record)) or True,
    )

    failed = manager.status(job_id)

    assert terminated == [status["process"]]
    assert failed["status"] == "failed"
    assert failed["error"]["code"] == "job_worker_lost"
    assert failed["error"]["termination_attempted"] is True


def test_job_storage_is_private(tmp_path, monkeypatch):
    root = tmp_path / "public-cache"
    root.mkdir(mode=0o755)
    monkeypatch.setenv("ARC_JOBS_CACHE", str(root))
    manager = JobManager(worker_mode="thread")
    job_id = manager.start(
        job_type="private",
        payload={"token": "secret"},
        runner=lambda progress, cancel: {"ok": True},
    )
    assert manager.wait(job_id, timeout=2)
    paths = JobPaths.for_job(job_id)

    for directory in (root, paths.job_dir.parent, paths.job_dir, root / "stats"):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    for path in (
        paths.job,
        paths.status,
        paths.events,
        paths.result,
        paths.heartbeat,
        root / "stats" / "jobs.sqlite",
    ):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_tail_events_uses_bounded_binary_tail(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    path.write_text(
        "".join(json.dumps({"event": f"event-{index}"}) + "\n" for index in range(200)),
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))

    events = tail_events(path, limit=3)

    assert [event["event"] for event in events] == ["event-197", "event-198", "event-199"]
