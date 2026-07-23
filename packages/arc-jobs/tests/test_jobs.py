from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

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
    snapshot_environment,
    restored_environment,
)


_TIER_MODEL_ENV_KEYS = tuple(
    f"ARC_LLM_{provider}_{tier}_MODEL"
    for provider in ("CODEX", "CLAUDE", "KIMI")
    for tier in ("LOW", "MEDIUM", "HIGH", "MAX")
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


@pytest.mark.skipif(os.name != "posix", reason="POSIX parent-death watchdog")
def test_command_watchdog_reaps_group_after_worker_sigkill(tmp_path):
    command_pid_path = tmp_path / "command.pid"
    command_script = (
        "import os,pathlib,signal,time; "
        f"pathlib.Path({str(command_pid_path)!r}).write_text(str(os.getpid())); "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    )
    worker_script = (
        "import subprocess,sys,time; "
        "from arc_jobs.worker import _start_process_watchdog; "
        f"p=subprocess.Popen([sys.executable,'-c',{command_script!r}],start_new_session=True); "
        "_start_process_watchdog(p); time.sleep(60)"
    )
    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    worker_process = subprocess.Popen([sys.executable, "-c", worker_script], env=child_env)
    deadline = time.monotonic() + 3
    while not command_pid_path.exists():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    command_pid = int(command_pid_path.read_text())

    os.kill(worker_process.pid, signal.SIGKILL)
    worker_process.wait(timeout=2)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and _test_pid_running(command_pid):
        time.sleep(0.02)
    assert not _test_pid_running(command_pid)


def _test_pid_running(pid: int) -> bool:
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return False
    return stat_text.rsplit(")", 1)[1].strip().split()[0] != "Z"


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


def test_deterministic_start_deduplicates_exact_concurrent_submission(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path))
    manager = JobManager(max_workers=1, worker_mode="thread")
    calls = 0

    def runner(progress, cancel):
        nonlocal calls
        calls += 1
        return {"ok": True}

    kwargs = {
        "job_type": "paper_operation",
        "payload": {"operation": "paper_summary", "paper_id": "x"},
        "runner": runner,
        "job_id": "paper-0123456789abcdef0123",
        "full_identity": {
            "operation": "paper_summary",
            "request_sha256": "a" * 64,
        },
    }
    with ThreadPoolExecutor(max_workers=8) as pool:
        job_ids = list(pool.map(lambda _: manager.start(**kwargs), range(8)))

    assert job_ids == ["paper-0123456789abcdef0123"] * 8
    assert manager.wait(job_ids[0], timeout=2)
    assert calls == 1
    stored = read_json(JobPaths.for_job(job_ids[0]).job)
    assert stored["full_identity"] == kwargs["full_identity"]
    assert len(stored["request_identity_sha256"]) == 64
    assert len(stored["payload_sha256"]) == 64
    assert stored["argv_sha256"] is None
    assert len(stored["environment_identity_sha256"]) == 64


def test_deterministic_start_rejects_identity_or_payload_mismatch(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path))
    manager = JobManager(max_workers=1, worker_mode="thread")
    identity = {"operation": "paper_summary", "request_sha256": "a" * 64}
    job_id = manager.start(
        job_type="paper_operation",
        payload={"paper_id": "x"},
        runner=lambda progress, cancel: {"ok": True},
        job_id="paper-0123456789abcdef0123",
        full_identity=identity,
    )
    assert job_id == "paper-0123456789abcdef0123"

    with pytest.raises(ValueError, match="collision or request mismatch"):
        manager.start(
            job_type="paper_operation",
            payload={"paper_id": "different"},
            runner=lambda progress, cancel: {"ok": True},
            job_id=job_id,
            full_identity=identity,
        )

    with pytest.raises(ValueError, match="collision or request mismatch"):
        manager.start(
            job_type="paper_operation",
            payload={"paper_id": "x"},
            runner=lambda progress, cancel: {"ok": True},
            job_id=job_id,
            full_identity={
                "operation": "paper_summary",
                "request_sha256": "b" * 64,
            },
        )


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


def test_worker_injects_internal_job_identity_after_environment_restore(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path / "cache"))
    _install_fake_cli(
        tmp_path,
        monkeypatch,
        body=(
            "printf '{\"ok\":true,\"data\":{\"job_id\":\"%s\","
            "\"job_type\":\"%s\"}}' \"$ARC_JOB_ID\" \"$ARC_JOB_TYPE\""
        ),
    )
    manager = JobManager(worker_mode="process")
    monkeypatch.setattr(manager, "_launch_worker", lambda job_id: None)
    job_id = manager.submit(["arc-paper", "--json"], job_type="paper_broker_operation")

    assert worker.run_job(job_id) == 0
    data = manager.result(job_id)["result"]["output"]["data"]
    assert data == {"job_id": job_id, "job_type": "paper_broker_operation"}
    persisted = read_json(JobPaths.for_job(job_id).job)["environment"]
    assert "ARC_JOB_ID" not in persisted
    assert "ARC_JOB_TYPE" not in persisted


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


def test_launch_maps_invalid_persisted_environment_before_process(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path / "cache"))
    _install_fake_cli(tmp_path, monkeypatch, body="printf '{\"ok\":true}'")
    manager = JobManager(worker_mode="process")
    monkeypatch.setattr(manager, "_launch_worker", lambda job_id: None)
    job_id = manager.submit(["arc-paper", "--json"])
    paths = JobPaths.for_job(job_id)
    job = read_json(paths.job)
    job["environment"]["ARC_LLM_TIMEOUT_SECONDS"] = "1"
    write_json(paths.job, job)

    JobManager._launch_worker(manager, job_id)

    assert manager.status(job_id)["error"]["code"] == "job_environment_invalid"


def test_launch_maps_process_creation_failure_before_command(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path / "cache"))
    _install_fake_cli(tmp_path, monkeypatch, body="printf '{\"ok\":true}'")
    manager = JobManager(worker_mode="process")
    monkeypatch.setattr(manager, "_launch_worker", lambda job_id: None)
    job_id = manager.submit(["arc-paper", "--json"])
    monkeypatch.setattr(
        jobs_module.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("test launch failure")
        ),
    )

    JobManager._launch_worker(manager, job_id)

    assert manager.status(job_id)["error"]["code"] == "job_worker_launch_failed"


def test_concurrent_completion_is_not_overwritten_by_cancel(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path / "cache"))
    _install_fake_cli(
        tmp_path, monkeypatch,
        body="printf '%s' '{\"ok\":true}'",
    )
    manager = JobManager(worker_mode="process")
    monkeypatch.setattr(manager, "_launch_worker", lambda job_id: None)
    job_id = manager.submit(["arc-paper", "--json"])
    barrier = Event()

    def complete():
        barrier.wait()
        jobs_module.finish_job(job_id, {"ok": True}, "done")

    def cancel():
        barrier.wait()
        manager.cancel(job_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        completion = pool.submit(complete)
        cancellation = pool.submit(cancel)
        barrier.set()
        completion.result()
        cancellation.result()

    status = manager.status(job_id)
    assert status["status"] == "done"
    assert manager.result(job_id)["ok"] is True


@pytest.mark.parametrize(
    ("terminal_status", "terminal_event"),
    [
        ("done", "job_done"),
        ("failed", "job_failed"),
        ("cancelled", "job_cancelled"),
    ],
)
def test_terminal_event_is_durable_before_terminal_status(
    tmp_path, monkeypatch, terminal_status, terminal_event,
) -> None:
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path / "cache"))
    _install_fake_cli(tmp_path, monkeypatch, body="printf '{\"ok\":true}'")
    manager = JobManager(worker_mode="process")
    monkeypatch.setattr(manager, "_launch_worker", lambda job_id: None)
    job_id = manager.submit(["arc-paper", "--json"])
    paths = JobPaths.for_job(job_id)
    observed_events: list[str] = []
    real_update_status = jobs_module.update_status

    def observe_terminal_update(job_id, *, paths=None, **fields):
        if fields.get("status") == terminal_status:
            observed_events.extend(
                event["event"]
                for event in tail_events(paths.events, limit=10)
            )
        return real_update_status(job_id, paths=paths, **fields)

    monkeypatch.setattr(jobs_module, "update_status", observe_terminal_update)
    if terminal_status == "done":
        jobs_module.finish_job(job_id, {"ok": True}, terminal_status)
    else:
        jobs_module.set_error(
            job_id,
            "job_cancelled" if terminal_status == "cancelled" else "job_failed",
            "terminal test",
            cancelled=terminal_status == "cancelled",
        )

    assert observed_events[-1] == terminal_event
    assert manager.status(job_id)["status"] == terminal_status


@pytest.mark.parametrize(
    ("reported", "expected_status", "expected_code"),
    [
        ("cancelled", "cancelled", "job_cancelled"),
        ("failed", "failed", "job_command_reported_failure"),
        ("completed", "failed", "job_command_failed"),
    ],
)
def test_nonzero_exit_preserves_failure_semantics_and_rejects_success(
    tmp_path, monkeypatch, reported, expected_status, expected_code
):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path / "cache"))
    _install_fake_cli(
        tmp_path,
        monkeypatch,
        body=f"printf '%s' '{{\"ok\":false,\"status\":\"{reported}\"}}'; exit 7",
    )
    manager = JobManager(worker_mode="process")
    monkeypatch.setattr(manager, "_launch_worker", lambda job_id: None)
    job_id = manager.submit(["arc-paper", "--json"])

    assert worker.run_job(job_id) == 1
    result = manager.result(job_id)

    assert result["status"] == expected_status
    assert result["error"]["code"] == expected_code
    assert result["result"]["exit_code"] == 7


@pytest.mark.parametrize("progress_kind", ["malformed", "oversized"])
def test_invalid_progress_terminates_and_reaps_running_command(
    tmp_path, monkeypatch, progress_kind
):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path / "cache"))
    if progress_kind == "malformed":
        body = "printf '%s\\n' 'not-json' > \"$ARC_JOB_PROGRESS_FILE\"; sleep 10"
    else:
        monkeypatch.setattr(worker, "MAX_PROGRESS_FILE_BYTES", 32)
        body = (
            "printf '%s' '123456789012345678901234567890123' "
            "> \"$ARC_JOB_PROGRESS_FILE\"; sleep 10"
        )
    _install_fake_cli(tmp_path, monkeypatch, body=body)
    manager = JobManager(worker_mode="process")
    monkeypatch.setattr(manager, "_launch_worker", lambda job_id: None)
    job_id = manager.submit(["arc-paper", "--json"])

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(worker.run_job, job_id)
        assert future.result(timeout=3) == 1

    status = manager.status(job_id)
    process = status["process"]
    assert status["status"] == "failed"
    assert status["error"]["code"] == "job_failed"
    assert jobs_module._pid_record_alive(process) is False


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


def test_cli_watch_until_review_returns_without_cancelling_job(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path))
    review_written = Event()
    request_second_review = Event()
    second_review_written = Event()
    release = Event()
    manager = JobManager(worker_mode="thread")

    def runner(progress, cancel):
        progress(
            {
                "schema_version": "arc.llm.progress.v1",
                "event": "review_due",
                "review_sequence": 2,
                "substantive": True,
                "summary": "Derived the bounded intermediate result.",
                "artifact_paths": ["steps/derived.json"],
            }
        )
        review_written.set()
        assert request_second_review.wait(timeout=2)
        progress(
            {
                "schema_version": "arc.llm.progress.v1",
                "event": "review_due",
                "review_sequence": 1,
                "substantive": False,
            }
        )
        second_review_written.set()
        assert release.wait(timeout=2)
        return {"ok": True}

    job_id = manager.start(job_type="test", payload={}, runner=runner)
    assert review_written.wait(timeout=2)

    assert cli.main(
        [
            "watch",
            job_id,
            "--interval",
            "0.01",
            "--until-review",
            "--after-review-sequence",
            "0",
            "--json",
        ]
    ) == 0
    response = json.loads(capsys.readouterr().out)
    assert response["status"] == "running"
    assert response["watch_status"] == "review_due"
    assert response["review_sequence"] == 1
    review_events = [event for event in response["events"] if event["event"] == "review_due"]
    assert review_events[-1]["provider_review_sequence"] == 2
    assert response["last_substantive_excerpt"] == "Derived the bounded intermediate result."
    assert manager.status(job_id)["cancel_requested"] is False

    request_second_review.set()
    assert second_review_written.wait(timeout=2)
    assert cli.main(
        [
            "watch",
            job_id,
            "--interval",
            "0.01",
            "--until-review",
            "--after-review-sequence",
            "1",
            "--json",
        ]
    ) == 0
    response = json.loads(capsys.readouterr().out)
    assert response["review_sequence"] == 2
    assert response["provider_review_sequence"] == 1
    assert manager.status(job_id)["cancel_requested"] is False

    release.set()
    assert manager.wait(job_id, timeout=2)


def test_concurrent_provider_review_sequences_become_monotonic_job_cursors(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path))
    manager = JobManager(worker_mode="thread")
    provider_sequences = [7, 1, 4, 1, 12, 2, 3, 1]

    def runner(progress, cancel):
        with ThreadPoolExecutor(max_workers=len(provider_sequences)) as executor:
            futures = [
                executor.submit(
                    progress,
                    {
                        "schema_version": "arc.llm.progress.v1",
                        "event": "review_due",
                        "review_sequence": sequence,
                        "substantive": False,
                    },
                )
                for sequence in provider_sequences
            ]
            for future in futures:
                future.result()
        return {"ok": True}

    job_id = manager.start(job_type="test", payload={}, runner=runner)
    assert manager.wait(job_id, timeout=2)
    status = manager.status(job_id)
    events = [event for event in status["events"] if event["event"] == "review_due"]

    assert status["review_sequence"] == len(provider_sequences)
    assert [event["review_sequence"] for event in events] == list(
        range(1, len(provider_sequences) + 1)
    )
    assert sorted(event["provider_review_sequence"] for event in events) == sorted(
        provider_sequences
    )


def test_watch_after_review_sequence_requires_until_review(monkeypatch, capsys):
    assert cli.main(["watch", "job", "--after-review-sequence", "1", "--json"]) == 1
    response = json.loads(capsys.readouterr().out)
    assert response["status"] == "invalid_request"
    assert "requires --until-review" in response["error"]["message"]


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


@pytest.mark.parametrize(
    ("reported", "ok_value", "exit_status"),
    [("first_chapter_ready", True, 0), ("needs_supervision", False, 1)],
)
def test_worker_preserves_companion_controlled_terminal_status(
    tmp_path, monkeypatch, reported, ok_value, exit_status
):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path / "cache"))
    _install_fake_cli(
        tmp_path,
        monkeypatch,
        body=(
            f"printf '%s' '{{\"ok\":{str(ok_value).lower()},"
            f"\"status\":\"{reported}\"}}'; exit {exit_status}"
        ),
    )
    manager = JobManager(worker_mode="process")
    monkeypatch.setattr(manager, "_launch_worker", lambda job_id: None)
    job_id = manager.submit(["arc-paper", "--json"])

    assert worker.run_job(job_id) == 0
    assert manager.status(job_id)["status"] == reported
    assert manager.result(job_id)["result"]["output"]["status"] == reported


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


def test_environment_snapshot_excludes_secrets_and_persists_host(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("ARC_AGENT_HOST", "kimi-code")
    monkeypatch.setenv("ARC_LLM_IDLE_TIMEOUT_SECONDS", "41")
    monkeypatch.setenv("OPENAI_API_KEY", "not-persisted")
    _install_fake_cli(tmp_path, monkeypatch, body="printf '%s' '{\"ok\":true}'")
    manager = JobManager(worker_mode="process")
    monkeypatch.setattr(manager, "_launch_worker", lambda job_id: None)
    job_id = manager.submit(["arc-paper", "--json"])
    persisted = read_json(JobPaths.for_job(job_id).job)["environment"]
    assert persisted["ARC_AGENT_HOST"] == "kimi-code"
    assert persisted["ARC_LLM_IDLE_TIMEOUT_SECONDS"] == "41"
    assert "ARC_LLM_TIMEOUT_SECONDS" not in persisted
    assert "OPENAI_API_KEY" not in persisted


@pytest.mark.parametrize(
    "env_key",
    [
        "ARC_LLM_TIMEOUT_SECONDS",
        "ARC_CODEX_TIMEOUT_SECONDS",
        "ARC_CLAUDE_TIMEOUT_SECONDS",
        "ARC_KIMI_TIMEOUT_SECONDS",
    ],
)
def test_submit_rejects_removed_total_timeout_from_caller_environment(
    tmp_path, monkeypatch, env_key
):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv(env_key, "60")
    manager = JobManager(worker_mode="process")
    monkeypatch.setattr(manager, "_launch_worker", lambda job_id: None)

    with pytest.raises(ValueError, match="total-timeout.*removed.*IDLE_TIMEOUT_SECONDS"):
        manager.submit(["arc-paper", "--json"])

    assert not list((tmp_path / "cache").glob("jobs/*/job.json"))


def test_environment_override_and_persisted_snapshot_reject_removed_total_timeout():
    with pytest.raises(ValueError, match="ARC_CODEX_IDLE_TIMEOUT_SECONDS"):
        snapshot_environment(overrides={"ARC_CODEX_TIMEOUT_SECONDS": "60"})
    with pytest.raises(ValueError, match="ARC_KIMI_IDLE_TIMEOUT_SECONDS"):
        restored_environment({"ARC_KIMI_TIMEOUT_SECONDS": "60"}, base={})


def test_environment_snapshot_preserves_explicit_provider_reasoning_effort(monkeypatch):
    monkeypatch.setenv("ARC_CODEX_REASONING_EFFORT", "medium")
    monkeypatch.setenv("ARC_CLAUDE_EFFORT", "high")

    persisted = snapshot_environment()
    restored = restored_environment(persisted, base={"PATH": os.environ["PATH"]})

    assert persisted["ARC_CODEX_REASONING_EFFORT"] == "medium"
    assert persisted["ARC_CLAUDE_EFFORT"] == "high"
    assert restored["ARC_CODEX_REASONING_EFFORT"] == "medium"
    assert restored["ARC_CLAUDE_EFFORT"] == "high"


def test_environment_snapshot_preserves_nonsecret_llm_concurrency_limits(monkeypatch):
    expected = {
        "ARC_LLM_MAX_CONCURRENCY": "24",
        "ARC_CODEX_MAX_CONCURRENCY": "8",
        "ARC_CLAUDE_MAX_CONCURRENCY": "6",
        "ARC_KIMI_MAX_CONCURRENCY": "4",
        "ARC_KIMI_ALLOW_INTERNAL_RETRIES": "1",
    }
    for key, value in expected.items():
        monkeypatch.setenv(key, value)

    persisted = snapshot_environment()
    restored = restored_environment(persisted, base={"PATH": os.environ["PATH"]})

    assert {key: persisted[key] for key in expected} == expected
    assert {key: restored[key] for key in expected} == expected


@pytest.mark.parametrize("env_key", _TIER_MODEL_ENV_KEYS)
def test_environment_snapshot_preserves_tier_model_override(monkeypatch, env_key):
    monkeypatch.setenv(env_key, "custom-tier-model")

    persisted = snapshot_environment()
    restored = restored_environment(persisted, base={"PATH": os.environ["PATH"]})

    assert persisted[env_key] == "custom-tier-model"
    assert restored[env_key] == "custom-tier-model"


@pytest.mark.parametrize("env_key", _TIER_MODEL_ENV_KEYS)
def test_submit_persists_tier_model_override(tmp_path, monkeypatch, env_key):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv(env_key, "custom-tier-model")
    _install_fake_cli(tmp_path, monkeypatch, body="printf '%s' '{\"ok\":true}'")
    manager = JobManager(worker_mode="process")
    monkeypatch.setattr(manager, "_launch_worker", lambda job_id: None)

    job_id = manager.submit(["arc-paper", "--json"])
    persisted = read_json(JobPaths.for_job(job_id).job)["environment"]

    assert persisted[env_key] == "custom-tier-model"


def test_detached_kimi_context_matches_foreground_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path / "cache"))
    kimi_context = {
        "ARC_AGENT_HOST": "kimi-code",
        "ARC_KIMI_BIN": "/opt/kimi/bin/kimi",
        "ARC_KIMI_WORK_DIR": str(tmp_path / "work"),
        "KIMI_CODE_HOME": str(tmp_path / "kimi-home"),
        "ARC_KIMI_IDLE_TIMEOUT_SECONDS": "73",
        "ARC_LLM_KIMI_LOW_MODEL": "kimi-low",
        "ARC_LLM_KIMI_MEDIUM_MODEL": "kimi-medium",
        "ARC_LLM_KIMI_HIGH_MODEL": "kimi-high",
        "ARC_LLM_KIMI_MAX_MODEL": "kimi-max",
    }
    for key, value in kimi_context.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("KIMI_API_KEY", "must-not-persist")
    _install_fake_cli(tmp_path, monkeypatch, body="printf '%s' '{\"ok\":true}'")
    manager = JobManager(worker_mode="process")
    monkeypatch.setattr(manager, "_launch_worker", lambda job_id: None)

    job_id = manager.submit(["arc-paper", "--json"])
    persisted = read_json(JobPaths.for_job(job_id).job)["environment"]
    restored = restored_environment(persisted, base={"PATH": os.environ["PATH"]})

    assert {key: persisted[key] for key in kimi_context} == kimi_context
    assert {key: restored[key] for key in kimi_context} == kimi_context
    assert "KIMI_API_KEY" not in persisted
    assert "KIMI_API_KEY" not in restored


def test_environment_overrides_reject_secrets_and_unknown_keys():
    with pytest.raises(ValueError, match="must not contain secrets"):
        snapshot_environment(overrides={"ARC_API_KEY": "secret"})
    with pytest.raises(ValueError, match="not allowlisted"):
        snapshot_environment(overrides={"ARC_UNKNOWN": "value"})


@pytest.mark.parametrize("host", ["codex", "claude-code", "kimi-code"])
def test_detached_job_persists_agent_host(tmp_path, monkeypatch, host):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("ARC_AGENT_HOST", host)
    _install_fake_cli(tmp_path, monkeypatch, body="printf '%s' '{\"ok\":true}'")
    manager = JobManager(worker_mode="process")
    monkeypatch.setattr(manager, "_launch_worker", lambda job_id: None)
    job_id = manager.submit(["arc-paper", "--json"])
    assert read_json(JobPaths.for_job(job_id).job)["environment"]["ARC_AGENT_HOST"] == host


@pytest.mark.parametrize("reported", ["completed", "degraded", "stopped"])
def test_worker_preserves_success_terminal_statuses(tmp_path, monkeypatch, reported):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path / "cache"))
    _install_fake_cli(
        tmp_path,
        monkeypatch,
        body=f"printf '%s' '{{\"ok\":true,\"status\":\"{reported}\",\"failure_count\":2}}'",
    )
    manager = JobManager(worker_mode="process")
    monkeypatch.setattr(manager, "_launch_worker", lambda job_id: None)
    job_id = manager.submit(["arc-paper", "--json"])
    assert worker.run_job(job_id) == 0
    result = manager.result(job_id)
    assert result["ok"] is True and result["status"] == reported
    assert result["result"]["output"]["failure_count"] == 2


def test_worker_forwards_progress_sidechannel(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path / "cache"))
    _install_fake_cli(
        tmp_path,
        monkeypatch,
        body=(
            "printf '%s\\n' '{\"schema_version\":\"arc.llm.progress.v1\","
            "\"event\":\"provider_progress\",\"phase\":\"proposers\",\"round_number\":2,"
            "\"role\":\"proposer\",\"completed_workers\":1,\"failed_workers\":1,"
            "\"substantive\":true,\"summary\":\"checked the first derivation\"}' "
            "> \"$ARC_JOB_PROGRESS_FILE\"; "
            "printf '%s' '{\"ok\":true,\"status\":\"degraded\"}'"
        ),
    )
    manager = JobManager(worker_mode="process")
    monkeypatch.setattr(manager, "_launch_worker", lambda job_id: None)
    job_id = manager.submit(["arc-paper", "--json"])
    assert worker.run_job(job_id) == 0
    status = manager.status(job_id)
    assert status["status"] == "degraded"
    assert status["round_number"] == 2 and status["failed_workers"] == 1
    assert status["role"] == "proposer"
    assert status["last_substantive_excerpt"] == "checked the first derivation"
    assert status["last_activity_at"] == status["last_substantive_at"]
    assert any(event["event"] == "provider_progress" for event in status["events"])


def test_worker_accepts_companion_progress_schema(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path / "cache"))
    _install_fake_cli(
        tmp_path,
        monkeypatch,
        body=(
            "printf '%s\\n' '{\"schema_version\":\"arc.companion.progress.v1\","
            "\"event\":\"block_accepted\",\"current\":2,\"chapter_id\":\"ch-0001\","
            "\"segment_id\":\"ch-0001.seg-0002\",\"lane\":\"companion\","
            "\"generation\":3,\"block_status\":\"accepted\",\"substantive\":true,"
            "\"summary\":\"accepted chapter block\"}' > \"$ARC_JOB_PROGRESS_FILE\"; "
            "printf '%s' '{\"ok\":true,\"status\":\"first_chapter_ready\"}'"
        ),
    )
    manager = JobManager(worker_mode="process")
    monkeypatch.setattr(manager, "_launch_worker", lambda job_id: None)
    job_id = manager.submit(["arc-paper", "--json"])

    assert worker.run_job(job_id) == 0
    status = manager.status(job_id)
    assert status["status"] == "first_chapter_ready"
    assert status["current"] == 2
    assert status["chapter_id"] == "ch-0001"
    assert status["segment_id"] == "ch-0001.seg-0002"
    assert status["lane"] == "companion"
    assert status["generation"] == 3
    assert status["block_status"] == "accepted"
    assert any(event["event"] == "block_accepted" for event in status["events"])


def test_non_substantive_progress_does_not_replace_last_excerpt(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path / "cache"))
    manager = JobManager(worker_mode="thread")

    def runner(progress, cancel):
        progress(
            {
                "schema_version": "arc.llm.progress.v1",
                "event": "provider_progress",
                "substantive": True,
                "summary": "Completed a source-backed calculation.",
            }
        )
        progress(
            {
                "schema_version": "arc.llm.progress.v1",
                "event": "provider_progress",
                "substantive": False,
                "summary": "still alive",
            }
        )
        return {"ok": True}

    job_id = manager.start(job_type="test", payload={}, runner=runner)
    assert manager.wait(job_id, timeout=2)
    assert manager.status(job_id)["last_substantive_excerpt"] == (
        "Completed a source-backed calculation."
    )


def test_live_status_projects_actual_proposer_phase_and_round(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_JOBS_CACHE", str(tmp_path / "cache"))
    progress_written = Event()
    release = Event()
    manager = JobManager(worker_mode="thread")

    def runner(progress, cancel):
        progress(
            {
                "schema_version": "arc.llm.proposers_reviewer.progress.v1",
                "event": "round_started",
                "phase": "proposers",
                "loop_id": "loop-1",
                "round_number": 3,
            }
        )
        progress_written.set()
        assert release.wait(timeout=2)
        return {"ok": True}

    job_id = manager.start(job_type="ideas", payload={}, runner=runner)
    assert progress_written.wait(timeout=2)

    status = manager.status(job_id)
    assert status["phase"] == "proposers"
    assert status["round_number"] == 3
    assert status["loop_id"] == "loop-1"

    release.set()
    assert manager.wait(job_id, timeout=2)


def test_arc_home_places_jobs_in_fixed_layout(tmp_path, monkeypatch):
    monkeypatch.delenv("ARC_JOBS_CACHE", raising=False)
    monkeypatch.delenv("ARC_JOBS_DIR", raising=False)
    monkeypatch.setenv("ARC_HOME", str(tmp_path / "arc-home"))
    assert jobs_module.jobs_root() == tmp_path / "arc-home" / "jobs"
    assert jobs_module.stats_db_path() == tmp_path / "arc-home" / "jobs" / ".stats" / "jobs.sqlite"
