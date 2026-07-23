from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from arc_llm import runner as runner_module
from arc_llm.call_record import ARC_LLM_CALL_RECORD_FIELD
from arc_llm.codex_binary import resolve_codex_binary
from arc_llm.nested_shell_capability import (
    NESTED_SHELL_CAPABILITY_SCHEMA_VERSION,
    NESTED_SHELL_PROBE_ID,
    NESTED_SHELL_PROMPT_MARKER,
    NestedShellCapability,
    _clear_process_cache_for_tests,
    build_codex_sandbox_probe_argv,
    classify_nested_shell_probe,
    nested_shell_warning_from_codex_events,
    render_nested_shell_prompt,
    resolve_nested_shell_capability,
    _run_probe,
)
from arc_llm.providers import codex_cli
from arc_llm.runner import run_json
from arc_llm.runtime_manifest import runtime_manifest, runtime_manifest_fingerprint
from arc_llm.usage import LLMProviderResponse


def _spawn_capability_resolver(env: dict[str, str], queue) -> None:
    from arc_llm.nested_shell_capability import resolve_nested_shell_capability

    result = resolve_nested_shell_capability(provider="codex-cli", env=env)
    queue.put(result.status)


@pytest.fixture(autouse=True)
def clear_process_cache() -> None:
    _clear_process_cache_for_tests()


def _binary(tmp_path: Path) -> Path:
    path = tmp_path / "custom-codex"
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o700)
    return path


def _env(tmp_path: Path, binary: Path, *, sandbox: str = "read-only") -> dict[str, str]:
    return {
        "ARC_CODEX_BIN": str(binary),
        "ARC_CODEX_SANDBOX": sandbox,
        "ARC_CODEX_WORK_DIR": str(tmp_path),
        "ARC_LLM_CACHE": str(tmp_path / "cache"),
        "PATH": os.defpath,
    }


def test_probe_argv_is_fixed_explicit_and_network_disabled(tmp_path: Path) -> None:
    argv = build_codex_sandbox_probe_argv(
        "/opt/custom/codex", sandbox_mode="workspace-write", cwd=tmp_path
    )
    assert argv == [
        "/opt/custom/codex",
        "sandbox",
        "--permission-profile",
        ":workspace",
        "--cd",
        str(tmp_path),
        "--sandbox-state-disable-network",
        "/bin/sh",
        "-c",
        "exit 0",
    ]
    assert "exec" not in argv
    assert "prompt" not in " ".join(argv).lower()


@pytest.mark.parametrize("kind", ["path", "absolute", "relative", "home", "symlink"])
def test_custom_codex_binary_matches_exec_help_and_probe(
    tmp_path: Path, monkeypatch, kind: str
) -> None:
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    binary_path = binary_dir / "custom-codex"
    binary_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary_path.chmod(0o700)
    env = {"PATH": str(binary_dir), "ARC_LLM_CACHE": str(tmp_path / "cache")}
    if kind == "path":
        env["ARC_CODEX_BIN"] = binary_path.name
    elif kind == "absolute":
        env["ARC_CODEX_BIN"] = str(binary_path)
    elif kind == "relative":
        monkeypatch.chdir(tmp_path)
        env["ARC_CODEX_BIN"] = "./bin/custom-codex"
    elif kind == "home":
        home = tmp_path / "home"
        home.mkdir()
        home_binary = home / "custom-codex"
        binary_path.replace(home_binary)
        binary_path = home_binary
        env.update({"HOME": str(home), "ARC_CODEX_BIN": "~/custom-codex"})
    else:
        link = tmp_path / "codex-link"
        link.symlink_to(binary_path)
        env["ARC_CODEX_BIN"] = str(link)
    expected = str(binary_path.resolve())
    captured: list[list[str]] = []

    def runner(argv, _env, _cwd, _timeout):
        captured.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    resolve_nested_shell_capability(
        provider="codex-cli", env=env, cwd=tmp_path, probe_runner=runner
    )
    exec_argv = codex_cli._base_cmd(env)  # noqa: SLF001
    help_calls: list[list[str]] = []

    def help_runner(argv, **_kwargs):
        help_calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="--json", stderr="")

    codex_cli._JSON_STREAM_SUPPORT_CACHE.clear()  # noqa: SLF001
    monkeypatch.setattr(codex_cli.subprocess, "run", help_runner)
    codex_cli._require_codex_json_stream_support(env)  # noqa: SLF001
    assert resolve_codex_binary(env, require_executable=True) == expected
    assert exec_argv[0] == help_calls[0][0] == captured[0][0] == expected


@pytest.mark.parametrize(
    ("stderr", "return_code", "expected"),
    [
        ("", 0, "available"),
        ("bwrap: No permissions to create a new namespace", 1, "namespace_denied"),
        ("kernel does not allow non-privileged user namespaces", 1, "namespace_denied"),
        ("bubblewrap: command not found", 127, "helper_missing"),
        ("unknown subcommand sandbox", 2, "helper_missing"),
        ("ordinary failure", 3, "probe_failed"),
    ],
)
def test_narrow_classifier(stderr: str, return_code: int, expected: str) -> None:
    assert classify_nested_shell_probe("", stderr, return_code=return_code) == expected


def test_success_probe_cache_and_receipt_are_bounded(tmp_path: Path) -> None:
    binary = _binary(tmp_path)
    calls: list[list[str]] = []

    def runner(argv, env, cwd, timeout):
        calls.append(list(argv))
        assert cwd == tmp_path
        assert timeout == 5.0
        assert env["ARC_CODEX_BIN"] == str(binary)
        return subprocess.CompletedProcess(argv, 0, b"x" * 9000, b"")

    first = resolve_nested_shell_capability(
        provider="codex-cli", env=_env(tmp_path, binary), probe_runner=runner
    )
    second = resolve_nested_shell_capability(
        provider="codex-cli", env=_env(tmp_path, binary), probe_runner=runner
    )
    assert first.status == "available" and first.cached is False
    assert second.status == "available" and second.cached is True
    assert len(calls) == 1
    receipt_path = next((tmp_path / "cache" / "nested-shell-capabilities").glob("*.json"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert "x" * 20 not in json.dumps(receipt)
    assert receipt["stdout_truncated"] is True
    assert receipt_path.stat().st_mode & 0o777 == 0o600
    assert receipt_path.parent.stat().st_mode & 0o777 == 0o700


def test_namespace_denial_is_cached_and_does_not_raise(tmp_path: Path) -> None:
    binary = _binary(tmp_path)

    def runner(argv, env, cwd, timeout):
        return subprocess.CompletedProcess(
            argv, 1, b"", b"bwrap: No permissions to create a new namespace"
        )

    result = resolve_nested_shell_capability(
        provider="codex-cli", env=_env(tmp_path, binary), probe_runner=runner
    )
    assert result.nested_sandboxed_shell is False
    assert result.status == "namespace_denied"
    assert result.warning == "nested_shell.namespace_denied"


def test_timeout_and_corrupt_receipt_reprobe_with_fake_clock(tmp_path: Path) -> None:
    binary = _binary(tmp_path)
    now = [100.0]
    calls = 0

    def runner(argv, env, cwd, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise subprocess.TimeoutExpired(argv, timeout)
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    result = resolve_nested_shell_capability(
        provider="codex-cli",
        env=_env(tmp_path, binary),
        probe_runner=runner,
        clock=lambda: now[0],
    )
    assert result.status == "timeout"
    receipt_path = next((tmp_path / "cache" / "nested-shell-capabilities").glob("*.json"))
    receipt_path.write_text("not-json", encoding="utf-8")
    _clear_process_cache_for_tests()
    result = resolve_nested_shell_capability(
        provider="codex-cli",
        env=_env(tmp_path, binary),
        probe_runner=runner,
        clock=lambda: now[0],
    )
    assert result.status == "available"
    assert calls == 2
    now[0] += 3601
    _clear_process_cache_for_tests()
    resolve_nested_shell_capability(
        provider="codex-cli",
        env=_env(tmp_path, binary),
        probe_runner=runner,
        clock=lambda: now[0],
    )
    assert calls == 3


def test_recipe_changes_for_cwd_sandbox_and_binary_stat(tmp_path: Path) -> None:
    binary = _binary(tmp_path)
    calls = 0

    def runner(argv, env, cwd, timeout):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    base = _env(tmp_path, binary)
    resolve_nested_shell_capability(
        provider="codex-cli", env=base, probe_runner=runner
    )
    other = tmp_path / "other"
    other.mkdir()
    resolve_nested_shell_capability(
        provider="codex-cli",
        env={**base, "ARC_CODEX_WORK_DIR": str(other)},
        probe_runner=runner,
    )
    resolve_nested_shell_capability(
        provider="codex-cli",
        env={**base, "ARC_CODEX_SANDBOX": "workspace-write"},
        probe_runner=runner,
    )
    binary.write_text("#!/bin/sh\n# changed\nexit 0\n", encoding="utf-8")
    resolve_nested_shell_capability(
        provider="codex-cli", env=base, probe_runner=runner
    )
    assert calls == 4


def test_sixteen_threads_share_one_probe(tmp_path: Path) -> None:
    binary = _binary(tmp_path)
    lock = threading.Lock()
    calls = 0

    def runner(argv, env, cwd, timeout):
        nonlocal calls
        with lock:
            calls += 1
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(
            lambda _index: resolve_nested_shell_capability(
                provider="codex-cli", env=_env(tmp_path, binary), probe_runner=runner
            ),
            range(16),
        ))
    assert calls == 1
    assert all(item.nested_sandboxed_shell for item in results)


def test_same_recipe_different_cache_roots_share_one_process_probe(tmp_path: Path) -> None:
    binary = _binary(tmp_path)
    barrier = threading.Barrier(2)
    calls = 0

    def runner(argv, env, cwd, timeout):
        nonlocal calls
        calls += 1
        time.sleep(0.05)
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    def resolve(cache_name: str) -> NestedShellCapability:
        barrier.wait()
        return resolve_nested_shell_capability(
            provider="codex-cli",
            env=_env(tmp_path, binary),
            cache_root=tmp_path / cache_name,
            probe_runner=runner,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(resolve, ("cache-a", "cache-b")))
    assert calls == 1
    assert [result.status for result in results] == ["available", "available"]
    assert sum(not result.cached for result in results) == 1


def test_two_spawn_processes_share_one_persistent_probe(tmp_path: Path) -> None:
    binary = tmp_path / "spawn-codex"
    counter = tmp_path / "probe-counter.txt"
    binary.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$ARC_TEST_PROBE_COUNTER\"\nexit 0\n",
        encoding="utf-8",
    )
    binary.chmod(0o700)
    env = {
        **_env(tmp_path, binary),
        "ARC_TEST_PROBE_COUNTER": str(counter),
    }
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(target=_spawn_capability_resolver, args=(env, queue))
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    assert sorted(queue.get(timeout=2) for _ in processes) == ["available", "available"]
    lines = counter.read_text(encoding="utf-8").splitlines()
    assert lines == [
        f"sandbox --permission-profile :read-only --cd {tmp_path} "
        "--sandbox-state-disable-network /bin/sh -c exit 0"
    ]
    receipts = list((tmp_path / "cache" / "nested-shell-capabilities").glob("*.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text(encoding="utf-8"))["recognized_code"] == "available"


def test_cache_lock_failure_runs_one_uncached_probe(tmp_path: Path, monkeypatch) -> None:
    from arc_llm import nested_shell_capability as capability_module

    binary = _binary(tmp_path)
    calls = 0

    def runner(argv, env, cwd, timeout):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    class BrokenLock:
        def __enter__(self):
            raise OSError("cache unavailable")

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(capability_module, "_address_lock", lambda path: BrokenLock())
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(
            lambda _index: resolve_nested_shell_capability(
                provider="codex-cli", env=_env(tmp_path, binary), probe_runner=runner
            ),
            range(16),
    ))
    assert all(result.status == "available" for result in results)
    assert sum(not result.cached for result in results) == 1
    assert calls == 1


def test_receipt_write_failure_keeps_probe_result(tmp_path: Path, monkeypatch) -> None:
    from arc_llm import nested_shell_capability as capability_module

    binary = _binary(tmp_path)
    calls = 0

    def runner(argv, env, cwd, timeout):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(
        capability_module,
        "_write_receipt",
        lambda path, receipt: (_ for _ in ()).throw(OSError("read-only cache")),
    )
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(
            lambda _index: resolve_nested_shell_capability(
                provider="codex-cli", env=_env(tmp_path, binary), probe_runner=runner
            ),
            range(16),
        ))
    assert all(result.status == "available" for result in results)
    assert calls == 1


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (FileNotFoundError("gone"), "helper_missing"),
        (OSError("local I/O"), "probe_failed"),
        (RuntimeError("malformed helper"), "probe_failed"),
    ],
)
def test_probe_start_and_local_failures_degrade_stably(
    tmp_path: Path, error: Exception, status: str
) -> None:
    binary = _binary(tmp_path)

    def runner(*args):
        raise error

    result = resolve_nested_shell_capability(
        provider="codex-cli", env=_env(tmp_path, binary), probe_runner=runner
    )
    assert result.status == status
    assert result.nested_sandboxed_shell is False
    assert result.warning == f"nested_shell.{status}"


def test_probe_does_not_swallow_keyboard_interrupt(tmp_path: Path) -> None:
    binary = _binary(tmp_path)

    def interrupted(*args):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        resolve_nested_shell_capability(
            provider="codex-cli",
            env=_env(tmp_path, binary),
            probe_runner=interrupted,
        )


@pytest.mark.skipif(
    sys.platform != "linux" or not Path("/proc").is_dir(),
    reason="residual child assertion requires Linux /proc",
)
def test_timeout_kills_child_after_leader_exits_on_term(tmp_path: Path) -> None:
    child_pid_path = tmp_path / "child.pid"
    child_code = (
        "import os,signal,sys,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "open(sys.argv[1], 'w').write(str(os.getpid())); "
        "time.sleep(30)"
    )
    leader_code = (
        "import os,subprocess,sys,time; "
        "subprocess.Popen([sys.executable, '-c', sys.argv[2], sys.argv[1]]); "
        "deadline=time.time()+2; "
        "\nwhile not os.path.exists(sys.argv[1]) and time.time()<deadline: time.sleep(0.01)"
        "\ntime.sleep(30)"
    )
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        _run_probe(
            [sys.executable, "-c", leader_code, str(child_pid_path), child_code],
            os.environ,
            tmp_path,
            0.3,
        )
    assert time.monotonic() - started < 2.5
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))

    def child_is_live() -> bool:
        try:
            status = Path(f"/proc/{child_pid}/stat").read_text(encoding="utf-8")
        except OSError:
            return False
        fields = status.split()
        return len(fields) > 2 and fields[2] != "Z"

    deadline = time.monotonic() + 1.0
    while child_is_live() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert child_is_live() is False


@pytest.mark.parametrize(
    ("provider", "sandbox", "status"),
    [
        ("claude-cli", "read-only", "provider_unsupported"),
        ("kimi-code-cli", "read-only", "provider_unsupported"),
        ("manual", "read-only", "provider_unsupported"),
        ("codex-cli", "danger-full-access", "unsafe_unsandboxed_mode"),
    ],
)
def test_unsupported_and_danger_modes_start_no_process(
    tmp_path: Path, provider: str, sandbox: str, status: str
) -> None:
    binary = _binary(tmp_path)

    def forbidden(*args):
        raise AssertionError("probe subprocess must not start")

    result = resolve_nested_shell_capability(
        provider=provider,
        env=_env(tmp_path, binary, sandbox=sandbox),
        probe_runner=forbidden,
    )
    assert result.status == status
    assert result.nested_sandboxed_shell is False


def test_runtime_fingerprint_uses_semantics_not_cache_metadata(tmp_path: Path) -> None:
    base = {
        "ARC_INTERNAL_NESTED_SANDBOXED_SHELL": "false",
        "ARC_INTERNAL_NESTED_SHELL_STATUS": "namespace_denied",
        "ARC_INTERNAL_NESTED_SHELL_PROBE_ID": "arc.llm.codex_sandbox_probe.v1",
    }
    first = runtime_manifest_fingerprint(runtime_manifest(
        provider="codex-cli", model="m", model_tier=None, env=base
    ))
    second = runtime_manifest_fingerprint(runtime_manifest(
        provider="codex-cli", model="m", model_tier=None, env={
            **base, "ARC_NESTED_SHELL_CHECKED_AT": "later", "ARC_NESTED_SHELL_CACHED": "true"
        }
    ))
    changed = runtime_manifest_fingerprint(runtime_manifest(
        provider="codex-cli", model="m", model_tier=None, env={
            **base,
            "ARC_INTERNAL_NESTED_SANDBOXED_SHELL": "true",
            "ARC_INTERNAL_NESTED_SHELL_STATUS": "available",
        }
    ))
    assert first == second
    assert changed != first


def test_prompt_renderer_removes_marker_and_direct_commands_when_false(tmp_path: Path) -> None:
    binary = _binary(tmp_path)
    false_capability = resolve_nested_shell_capability(
        provider="claude-cli", env=_env(tmp_path, binary)
    )
    prompt = f"before {NESTED_SHELL_PROMPT_MARKER} after"
    rendered = render_nested_shell_prompt(
        prompt, false_capability, controller_evidence_exposed=True
    )
    assert NESTED_SHELL_PROMPT_MARKER not in rendered
    assert "arc-paper-worker" not in rendered
    assert "arc_evidence_requests" in rendered


def test_prompt_renderer_true_defers_direct_allowlist_to_broker_catalog() -> None:
    capability = NestedShellCapability(
        schema_version=NESTED_SHELL_CAPABILITY_SCHEMA_VERSION,
        provider="codex-cli",
        nested_sandboxed_shell=True,
        status="available",
        probe_kind="codex_sandbox",
        probe_identity=NESTED_SHELL_PROBE_ID,
        warning=None,
    )
    rendered = render_nested_shell_prompt(
        NESTED_SHELL_PROMPT_MARKER,
        capability,
        controller_evidence_exposed=True,
    )
    assert NESTED_SHELL_PROMPT_MARKER not in rendered
    assert "catalog-authorized network=none operations" in rendered
    assert "Broker bootstrap" in rendered
    assert "Controller requests" in rendered
    assert "arc-paper-worker" not in rendered


def test_raw_event_warning_only_scans_typed_command_execution() -> None:
    diagnostic = "bwrap: No permissions to create a new namespace"
    assert nested_shell_warning_from_codex_events([
        {"type": "item.completed", "item": {
            "type": "command_execution", "aggregated_output": diagnostic,
            "exit_code": 1, "status": "failed",
        }},
        {"type": "turn.completed"},
    ]) == "nested_shell.namespace_denied"
    assert nested_shell_warning_from_codex_events([
        {"type": "item.completed", "item": {"type": "agent_message", "text": diagnostic}}
    ]) is None


def test_typed_namespace_failure_keeps_later_provider_result_valid(monkeypatch) -> None:
    calls = 0

    class Provider:
        def generate_json_result(self, prompt, **kwargs):
            nonlocal calls
            calls += 1
            return LLMProviderResponse(
                {"ok": True},
                raw_events=(
                    {"type": "thread.started", "thread_id": "session"},
                    {"type": "item.completed", "item": {
                        "type": "command_execution",
                        "aggregated_output": "bwrap: No permissions to create a new namespace",
                        "exit_code": 1,
                        "status": "failed",
                    }},
                    {"type": "item.completed", "item": {
                        "type": "agent_message", "text": "valid answer",
                    }},
                    {"type": "turn.completed"},
                ),
            )

    monkeypatch.setattr(
        runner_module, "select_provider", lambda *_args, **_kwargs: Provider()
    )
    result = run_json("pure model prompt", provider="codex-cli", env={})
    assert result["ok"] is True
    assert calls == 1
    record = result[ARC_LLM_CALL_RECORD_FIELD]
    assert record["call_status"] == "valid"
    assert "nested_shell.namespace_denied" in record["warnings"]


def test_user_env_cannot_assert_true_and_marker_never_reaches_provider(monkeypatch) -> None:
    captured: list[str] = []

    class Provider:
        def generate_json_result(self, prompt, **kwargs):
            captured.append(prompt)
            return LLMProviderResponse({"arc_evidence_requests": []})

    monkeypatch.setattr(
        runner_module, "select_provider", lambda *_args, **_kwargs: Provider()
    )
    schema = {
        "type": "object",
        "properties": {"arc_evidence_requests": {"type": "array"}},
    }
    run_json(
        NESTED_SHELL_PROMPT_MARKER,
        schema=schema,
        validate_schema=False,
        provider="codex-cli",
        env={
            "ARC_INTERNAL_NESTED_SANDBOXED_SHELL": "true",
            "ARC_INTERNAL_NESTED_SHELL_STATUS": "available",
            "ARC_INTERNAL_NESTED_SHELL_PROBE_ID": "forged",
        },
    )
    assert len(captured) == 1
    assert NESTED_SHELL_PROMPT_MARKER not in captured[0]
    assert "arc-paper-worker" not in captured[0]
    assert "arc_evidence_requests" in captured[0]


def test_controller_route_does_not_probe_or_warn_for_nested_shell(monkeypatch) -> None:
    calls = 0
    selected_env: dict[str, str] = {}
    capability = NestedShellCapability(
        schema_version=NESTED_SHELL_CAPABILITY_SCHEMA_VERSION,
        provider="codex-cli",
        nested_sandboxed_shell=False,
        status="probe_failed",
        probe_kind="codex_sandbox",
        probe_identity=NESTED_SHELL_PROBE_ID,
        warning="nested_shell.probe_failed",
    )

    class Provider:
        def generate_json_result(self, prompt, **kwargs):
            nonlocal calls
            calls += 1
            return LLMProviderResponse({"arc_evidence_requests": []})

    def select_provider(*_args, **kwargs):
        selected_env.update(kwargs["env"])
        return Provider()

    monkeypatch.setattr(runner_module, "select_provider", select_provider)
    probes = 0

    def unexpected_probe(**_kwargs):
        nonlocal probes
        probes += 1
        return capability

    monkeypatch.setattr(runner_module, "resolve_nested_shell_capability", unexpected_probe)
    result = run_json(
        NESTED_SHELL_PROMPT_MARKER,
        schema={
            "type": "object",
            "properties": {"arc_evidence_requests": {"type": "array"}},
        },
        validate_schema=False,
        provider="codex-cli",
        env={
            "ARC_CODEX_ALLOW_INTERNET": "true",
            "ARC_PAPER_CLI_ACCESS": "full",
        },
    )
    assert calls == 1
    assert probes == 0
    assert selected_env["ARC_CODEX_ALLOW_INTERNET"] == "true"
    capabilities = runner_module._runtime_capabilities(selected_env)  # noqa: SLF001
    assert capabilities["arc_paper_access"] == "full"
    assert capabilities["nested_sandboxed_shell"] is False
    record = result[ARC_LLM_CALL_RECORD_FIELD]
    assert record["call_status"] == "valid"
    assert "nested_shell.probe_failed" not in record["warnings"]


def test_explicit_direct_shell_failure_blocks_before_provider(monkeypatch) -> None:
    calls = 0
    capability = NestedShellCapability(
        schema_version=NESTED_SHELL_CAPABILITY_SCHEMA_VERSION,
        provider="codex-cli",
        nested_sandboxed_shell=False,
        status="probe_failed",
        probe_kind="codex_sandbox",
        probe_identity=NESTED_SHELL_PROBE_ID,
        warning="nested_shell.probe_failed",
    )

    class Provider:
        def generate_json_result(self, prompt, **kwargs):
            nonlocal calls
            calls += 1
            return LLMProviderResponse({"arc_evidence_requests": []})

    monkeypatch.setattr(runner_module, "select_provider", lambda *_a, **_k: Provider())
    monkeypatch.setattr(
        runner_module,
        "resolve_nested_shell_capability",
        lambda **_kwargs: capability,
    )

    with pytest.raises(runner_module.LLMConfigurationError):
        run_json(
            NESTED_SHELL_PROMPT_MARKER,
            schema={
                "type": "object",
                "properties": {"arc_evidence_requests": {"type": "array"}},
            },
            validate_schema=False,
            provider="codex-cli",
            env={
                "ARC_PAPER_ACCESS": "full",
                "ARC_PAPER_DIRECT_SHELL": "true",
            },
        )
    assert calls == 0
