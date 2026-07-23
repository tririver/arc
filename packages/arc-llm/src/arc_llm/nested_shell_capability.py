from __future__ import annotations

import contextlib
import hashlib
import json
import os
import signal
import stat
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .codex_binary import resolve_codex_binary
from .paths import llm_cache_root
from .schema_cache import canonical_json, sha256_text


NESTED_SHELL_CAPABILITY_SCHEMA_VERSION = "arc.llm.nested_shell_capability.v1"
NESTED_SHELL_PROBE_ID = "arc.llm.codex_sandbox_probe.v1"
NESTED_SHELL_PROMPT_MARKER = "{{ARC_NESTED_SHELL_CAPABILITY}}"
NESTED_SHELL_CACHE_TTL_SECONDS = 3600.0
NESTED_SHELL_PROBE_TIMEOUT_SECONDS = 5.0
NESTED_SHELL_OUTPUT_LIMIT_BYTES = 4096

_RUNTIME_BOOL = "ARC_INTERNAL_NESTED_SANDBOXED_SHELL"
_RUNTIME_STATUS = "ARC_INTERNAL_NESTED_SHELL_STATUS"
_RUNTIME_PROBE_ID = "ARC_INTERNAL_NESTED_SHELL_PROBE_ID"
_RUNTIME_KEYS = (_RUNTIME_BOOL, _RUNTIME_STATUS, _RUNTIME_PROBE_ID)

_STATUSES = {
    "available",
    "namespace_denied",
    "helper_missing",
    "timeout",
    "probe_failed",
    "unsafe_unsandboxed_mode",
    "provider_unsupported",
    "not_requested",
}
_PROCESS_CACHE: dict[str, tuple[float, "NestedShellCapability"]] = {}
_PROCESS_CACHE_LOCK = threading.Lock()
_RECIPE_THREAD_LOCKS: dict[str, threading.RLock] = {}
_RECIPE_THREAD_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True)
class NestedShellCapability:
    schema_version: str
    provider: str
    nested_sandboxed_shell: bool
    status: str
    probe_kind: str
    probe_identity: str
    warning: str | None = None
    cached: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != NESTED_SHELL_CAPABILITY_SCHEMA_VERSION:
            raise ValueError("unsupported nested shell capability schema")
        if self.status not in _STATUSES:
            raise ValueError(f"unsupported nested shell status: {self.status}")
        if self.nested_sandboxed_shell != (self.status == "available"):
            raise ValueError("nested shell boolean and status disagree")
        expected_warning = None if self.status == "available" else f"nested_shell.{self.status}"
        if self.warning != expected_warning:
            raise ValueError("nested shell warning is not canonical")

    def runtime_identity(self) -> dict[str, Any]:
        return {
            "nested_sandboxed_shell": self.nested_sandboxed_shell,
            "nested_shell_status": self.status,
            "nested_shell_probe_id": self.probe_identity,
        }

    def doctor_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider": self.provider,
            "nested_sandboxed_shell": self.nested_sandboxed_shell,
            "status": self.status,
            "probe_kind": self.probe_kind,
            "probe_identity": self.probe_identity,
            "warning": self.warning,
            "cached": self.cached,
        }


@dataclass(frozen=True)
class _ProbeReceipt:
    capability: NestedShellCapability
    recipe_sha256: str
    checked_at: float
    duration_ms: int
    return_code: int | None
    recognized_code: str
    stdout_sha256: str
    stderr_sha256: str
    stdout_truncated: bool
    stderr_truncated: bool


def clear_runtime_capability_values(env: dict[str, str]) -> None:
    """Remove internal values before accepting a caller-provided environment."""

    for key in _RUNTIME_KEYS:
        env.pop(key, None)


def apply_runtime_capability(env: dict[str, str], capability: NestedShellCapability) -> None:
    env[_RUNTIME_BOOL] = "true" if capability.nested_sandboxed_shell else "false"
    env[_RUNTIME_STATUS] = capability.status
    env[_RUNTIME_PROBE_ID] = capability.probe_identity


def capability_runtime_identity(env: Mapping[str, str] | None) -> dict[str, Any]:
    values = env or {}
    status = values.get(_RUNTIME_STATUS)
    probe_id = values.get(_RUNTIME_PROBE_ID)
    raw_bool = values.get(_RUNTIME_BOOL)
    if status not in _STATUSES or not probe_id or raw_bool not in {"true", "false"}:
        return {
            "nested_sandboxed_shell": False,
            "nested_shell_status": "not_requested",
            "nested_shell_probe_id": NESTED_SHELL_PROBE_ID,
        }
    available = raw_bool == "true"
    if available != (status == "available"):
        return {
            "nested_sandboxed_shell": False,
            "nested_shell_status": "not_requested",
            "nested_shell_probe_id": NESTED_SHELL_PROBE_ID,
        }
    return {
        "nested_sandboxed_shell": available,
        "nested_shell_status": status,
        "nested_shell_probe_id": probe_id,
    }


def resolve_nested_shell_capability(
    *,
    provider: str,
    env: Mapping[str, str] | None,
    cwd: Path | str | None = None,
    cache_root: Path | None = None,
    timeout_seconds: float = NESTED_SHELL_PROBE_TIMEOUT_SECONDS,
    clock: Callable[[], float] = time.time,
    probe_runner: Callable[[Sequence[str], Mapping[str, str], Path, float], subprocess.CompletedProcess[bytes]] | None = None,
) -> NestedShellCapability:
    """Resolve actual nested shell support without invoking a model.

    Only Codex sandboxed modes execute the fixed, harmless probe. Other
    providers and Codex full-access mode return a conservative result without
    starting a subprocess.
    """

    values = os.environ if env is None else env
    if provider != "codex-cli":
        return _capability(provider, "provider_unsupported", probe_kind="none")
    sandbox_mode = str(values.get("ARC_CODEX_SANDBOX") or "read-only").strip().lower()
    if sandbox_mode == "danger-full-access":
        return _capability(provider, "unsafe_unsandboxed_mode", probe_kind="none")
    if sandbox_mode not in {"read-only", "workspace-write"}:
        return _capability(provider, "probe_failed")

    binary = resolve_codex_binary(values, require_executable=True)
    if binary is None:
        return _capability(provider, "helper_missing")
    work_dir = Path(cwd or values.get("ARC_CODEX_WORK_DIR") or os.getcwd()).expanduser().resolve(strict=False)
    recipe = _probe_recipe(binary, sandbox_mode=sandbox_mode, cwd=work_dir, env=values)
    recipe_sha256 = sha256_text(canonical_json(recipe))
    root = cache_root or llm_cache_root(values)
    path = root / "nested-shell-capabilities" / f"{recipe_sha256}.json"
    with _recipe_thread_lock(recipe_sha256):
        observed_now = float(clock())
        with _PROCESS_CACHE_LOCK:
            memo = _PROCESS_CACHE.get(recipe_sha256)
            if (
                memo is not None
                and 0 <= observed_now - memo[0] <= NESTED_SHELL_CACHE_TTL_SECONDS
            ):
                return replace(memo[1], cached=True)

        result: NestedShellCapability | None = None
        try:
            with _address_lock(path):
                locked_now = float(clock())
                receipt = _read_receipt(path, recipe_sha256, now=locked_now)
                if receipt is not None:
                    result = replace(receipt.capability, cached=True)
                else:
                    result, receipt = _probe_once(
                        provider=provider,
                        binary=binary,
                        sandbox_mode=sandbox_mode,
                        work_dir=work_dir,
                        values=values,
                        timeout_seconds=timeout_seconds,
                        probe_runner=probe_runner,
                        recipe_sha256=recipe_sha256,
                        checked_at=locked_now,
                    )
                    try:
                        _write_receipt(path, receipt)
                    except OSError:
                        # Persistent caching is an optimization. A read-only
                        # cache must not block the provider call.
                        pass
        except OSError:
            # Directory creation and address locking can fail on an otherwise
            # usable local runtime. Probe once without persistent caching.
            if result is None:
                fallback_now = float(clock())
                result, _receipt = _probe_once(
                    provider=provider,
                    binary=binary,
                    sandbox_mode=sandbox_mode,
                    work_dir=work_dir,
                    values=values,
                    timeout_seconds=timeout_seconds,
                    probe_runner=probe_runner,
                    recipe_sha256=recipe_sha256,
                    checked_at=fallback_now,
                )

        assert result is not None
        memo_now = float(clock())
        with _PROCESS_CACHE_LOCK:
            _PROCESS_CACHE[recipe_sha256] = (memo_now, replace(result, cached=False))
        return result


def _probe_once(
    *,
    provider: str,
    binary: str,
    sandbox_mode: str,
    work_dir: Path,
    values: Mapping[str, str],
    timeout_seconds: float,
    probe_runner: Callable[[Sequence[str], Mapping[str, str], Path, float], subprocess.CompletedProcess[bytes]] | None,
    recipe_sha256: str,
    checked_at: float,
) -> tuple[NestedShellCapability, _ProbeReceipt]:
    argv = build_codex_sandbox_probe_argv(
        binary, sandbox_mode=sandbox_mode, cwd=work_dir
    )
    started = time.monotonic()
    runner = probe_runner or _run_probe
    stdout = b""
    stderr = b""
    return_code: int | None = None
    try:
        completed = runner(argv, values, work_dir, timeout_seconds)
        stdout = _probe_output_bytes(completed.stdout)
        stderr = _probe_output_bytes(completed.stderr)
        return_code = int(completed.returncode)
        status = classify_nested_shell_probe(
            stdout.decode("utf-8", "replace"),
            stderr.decode("utf-8", "replace"),
            return_code=return_code,
        )
    except FileNotFoundError:
        status = "helper_missing"
    except subprocess.TimeoutExpired:
        status = "timeout"
    except OSError:
        status = "probe_failed"
    except Exception:
        # Malformed local helper results and ordinary runner defects degrade
        # the optional capability. KeyboardInterrupt/SystemExit still escape.
        status = "probe_failed"
    duration_ms = max(0, int((time.monotonic() - started) * 1000))
    result = _capability(provider, status)
    return result, _ProbeReceipt(
        capability=result,
        recipe_sha256=recipe_sha256,
        checked_at=checked_at,
        duration_ms=duration_ms,
        return_code=return_code,
        recognized_code=status,
        stdout_sha256=hashlib.sha256(stdout).hexdigest(),
        stderr_sha256=hashlib.sha256(stderr).hexdigest(),
        stdout_truncated=len(stdout) > NESTED_SHELL_OUTPUT_LIMIT_BYTES,
        stderr_truncated=len(stderr) > NESTED_SHELL_OUTPUT_LIMIT_BYTES,
    )


def _probe_output_bytes(value: Any) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8", "replace")
    raise TypeError("probe output must be bytes or text")


def build_codex_sandbox_probe_argv(
    binary: str, *, sandbox_mode: str, cwd: Path
) -> list[str]:
    profile = ":read-only" if sandbox_mode == "read-only" else ":workspace"
    return [
        binary,
        "sandbox",
        "--permission-profile",
        profile,
        "--cd",
        str(cwd),
        "--sandbox-state-disable-network",
        "/bin/sh",
        "-c",
        "exit 0",
    ]


def classify_nested_shell_probe(stdout: str, stderr: str, *, return_code: int | None) -> str:
    """Classify only output from the dedicated probe or a typed command item."""

    material = f"{stdout}\n{stderr}".lower()
    if return_code == 0:
        return "available"
    if (
        "no permissions to create a new namespace" in material
        or "unprivileged user namespaces" in material
        or "non-privileged user namespaces" in material
        or "unprivileged_userns_clone" in material
        or ("bwrap" in material and "namespace" in material and "permission" in material)
    ):
        return "namespace_denied"
    if (
        return_code == 127
        or "command not found" in material
        or "no such file or directory" in material
        or "could not find bubblewrap" in material
        or "required arguments were not provided" in material
        or "unknown subcommand" in material
        or "unrecognized subcommand" in material
    ):
        return "helper_missing"
    return "probe_failed"


def nested_shell_warning_from_codex_events(events: Sequence[Mapping[str, Any]]) -> str | None:
    """Inspect typed command-execution diagnostics, never assistant prose."""

    for event in events:
        item = event.get("item")
        if not isinstance(item, Mapping) or item.get("type") != "command_execution":
            continue
        output = item.get("aggregated_output")
        if not isinstance(output, str):
            continue
        code = item.get("exit_code")
        return_code = code if isinstance(code, int) and not isinstance(code, bool) else 1
        if classify_nested_shell_probe("", output, return_code=return_code) == "namespace_denied":
            return "nested_shell.namespace_denied"
    return None


def render_nested_shell_prompt(
    prompt: str,
    capability: NestedShellCapability | Mapping[str, Any],
    *,
    controller_evidence_exposed: bool,
) -> str:
    if NESTED_SHELL_PROMPT_MARKER not in prompt:
        return prompt
    if isinstance(capability, NestedShellCapability):
        available = capability.nested_sandboxed_shell
    else:
        available = capability.get("nested_sandboxed_shell") is True
    if available:
        replacement = (
            "A verified sandboxed shell is available for this turn. Direct access is "
            "limited to these fixed deterministic reads: arc-paper-worker policy-targets "
            "with cursor-based, byte-bounded catalog pagination, "
            "arc-paper-worker get-parsed-toc, arc-paper-worker get-parsed-section, and "
            "arc-paper-worker artifact-read with offset-based, byte-bounded artifact "
            "pagination. Do not invoke "
            "other shell commands, raw arc-paper, Python arc_paper modules, arc-llm, "
            "arc-jobs, arc-domain, or nested model commands."
        )
    elif controller_evidence_exposed:
        replacement = (
            "No verified nested shell is available for this turn. Do not invoke shell "
            "commands or attempt a bypass. Request deterministic paper reads only through "
            "the addressed arc_evidence_requests Controller schema."
        )
    else:
        replacement = (
            "No verified nested shell is available for this turn. Do not invoke shell "
            "commands, command examples, or bypasses; use only evidence already supplied "
            "in the task context."
        )
    return prompt.replace(NESTED_SHELL_PROMPT_MARKER, replacement)


def _capability(provider: str, status: str, *, probe_kind: str = "codex_sandbox") -> NestedShellCapability:
    return NestedShellCapability(
        schema_version=NESTED_SHELL_CAPABILITY_SCHEMA_VERSION,
        provider=provider,
        nested_sandboxed_shell=status == "available",
        status=status,
        probe_kind=probe_kind,
        probe_identity=NESTED_SHELL_PROBE_ID if probe_kind != "none" else "none",
        warning=None if status == "available" else f"nested_shell.{status}",
    )


def _probe_recipe(binary: str, *, sandbox_mode: str, cwd: Path, env: Mapping[str, str]) -> dict[str, Any]:
    try:
        info = os.stat(binary)
        binary_identity: dict[str, Any] = {
            "path": str(Path(binary).resolve(strict=False)),
            "device": info.st_dev,
            "inode": info.st_ino,
            "size": info.st_size,
            "mtime_ns": info.st_mtime_ns,
        }
    except OSError:
        binary_identity = {"path": str(Path(binary).resolve(strict=False)), "missing": True}
    return {
        "schema_version": NESTED_SHELL_CAPABILITY_SCHEMA_VERSION,
        "provider": "codex-cli",
        "probe_identity": NESTED_SHELL_PROBE_ID,
        "binary": binary_identity,
        "sandbox_mode": sandbox_mode,
        "cwd": str(cwd),
        "sandbox_config": {
            "network_access": False,
            "ignore_user_config": str(env.get("ARC_CODEX_IGNORE_USER_CONFIG") or "").lower() == "true",
        },
        "boot_identity": _read_kernel_value(Path("/proc/sys/kernel/random/boot_id")),
        "userns": {
            "unprivileged_userns_clone": _read_kernel_value(Path("/proc/sys/kernel/unprivileged_userns_clone")),
            "max_user_namespaces": _read_kernel_value(Path("/proc/sys/user/max_user_namespaces")),
        },
    }


def _read_kernel_value(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()[:256] or None
    except OSError:
        return None


def _run_probe(
    argv: Sequence[str], env: Mapping[str, str], cwd: Path, timeout_seconds: float
) -> subprocess.CompletedProcess[bytes]:
    kwargs: dict[str, Any] = {"start_new_session": True} if os.name == "posix" else {
        "creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    }
    process = subprocess.Popen(
        list(argv),
        cwd=cwd,
        env=dict(env),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        **kwargs,
    )
    assert process.stdout is not None and process.stderr is not None
    captured: dict[str, bytearray] = {
        "stdout": bytearray(),
        "stderr": bytearray(),
    }

    def drain(name: str, stream: Any) -> None:
        while chunk := stream.read(4096):
            remaining = NESTED_SHELL_OUTPUT_LIMIT_BYTES + 1 - len(captured[name])
            if remaining > 0:
                captured[name].extend(chunk[:remaining])

    readers = [
        threading.Thread(target=drain, args=("stdout", process.stdout), daemon=True),
        threading.Thread(target=drain, args=("stderr", process.stderr), daemon=True),
    ]
    for reader in readers:
        reader.start()
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        grace_seconds = 0.5
        grace_deadline = time.monotonic() + grace_seconds
        _terminate_probe_group(process)
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            pass
        remaining = grace_deadline - time.monotonic()
        if remaining > 0:
            threading.Event().wait(remaining)
        # The leader may have exited after TERM while descendants kept the
        # original process group and inherited pipes alive. Always KILL the
        # original pgid after grace; ESRCH is harmless.
        _kill_probe_group(process)
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass
        for reader in readers:
            reader.join(timeout=0.5)
        raise subprocess.TimeoutExpired(argv, timeout_seconds) from exc
    for reader in readers:
        reader.join(timeout=0.5)
    return subprocess.CompletedProcess(
        list(argv),
        process.returncode,
        bytes(captured["stdout"]),
        bytes(captured["stderr"]),
    )


def _terminate_probe_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM) if os.name == "posix" else process.terminate()
    except OSError:
        pass


def _kill_probe_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL) if os.name == "posix" else process.kill()
    except OSError:
        pass


@contextlib.contextmanager
def _address_lock(path: Path):
    lock_path = path.with_suffix(".json.lock")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with contextlib.suppress(OSError):
        os.chmod(path.parent, 0o700)
    handle = lock_path.open("a+b")
    os.chmod(lock_path, 0o600)
    try:
        if os.name == "posix":
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if os.name == "posix":
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _recipe_thread_lock(recipe_sha256: str) -> threading.RLock:
    """Return the process-local single-flight lock for a canonical recipe."""

    with _RECIPE_THREAD_LOCKS_GUARD:
        return _RECIPE_THREAD_LOCKS.setdefault(recipe_sha256, threading.RLock())


def _read_receipt(path: Path, recipe_sha256: str, *, now: float) -> _ProbeReceipt | None:
    try:
        info = path.lstat()
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version")
            != "arc.llm.nested_shell_probe_receipt.v1"
            or payload.get("recipe_sha256") != recipe_sha256
        ):
            return None
        raw_checked_at = payload.get("checked_at")
        if (
            not isinstance(raw_checked_at, (int, float))
            or isinstance(raw_checked_at, bool)
        ):
            return None
        checked_at = float(raw_checked_at)
        if now - checked_at > NESTED_SHELL_CACHE_TTL_SECONDS or now < checked_at:
            return None
        raw_capability = payload["capability"]
        capability = NestedShellCapability(**raw_capability)
        recognized_code = str(payload["recognized_code"])
        if recognized_code != capability.status:
            return None
        for key in ("stdout_sha256", "stderr_sha256"):
            digest = payload.get(key)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                return None
        return_code = payload.get("return_code")
        if return_code is not None and (
            not isinstance(return_code, int) or isinstance(return_code, bool)
        ):
            return None
        duration_ms = payload.get("duration_ms")
        if (
            not isinstance(duration_ms, int)
            or isinstance(duration_ms, bool)
            or duration_ms < 0
        ):
            return None
        if not all(
            isinstance(payload.get(key), bool)
            for key in ("stdout_truncated", "stderr_truncated")
        ):
            return None
        return _ProbeReceipt(
            capability=replace(capability, cached=False),
            recipe_sha256=recipe_sha256,
            checked_at=checked_at,
            duration_ms=duration_ms,
            return_code=return_code,
            recognized_code=recognized_code,
            stdout_sha256=str(payload["stdout_sha256"]),
            stderr_sha256=str(payload["stderr_sha256"]),
            stdout_truncated=bool(payload["stdout_truncated"]),
            stderr_truncated=bool(payload["stderr_truncated"]),
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _write_receipt(path: Path, receipt: _ProbeReceipt) -> None:
    if path.exists() and path.is_symlink():
        return
    payload = {
        "schema_version": "arc.llm.nested_shell_probe_receipt.v1",
        "recipe_sha256": receipt.recipe_sha256,
        "checked_at": receipt.checked_at,
        "duration_ms": receipt.duration_ms,
        "return_code": receipt.return_code,
        "recognized_code": receipt.recognized_code,
        "stdout_sha256": receipt.stdout_sha256,
        "stderr_sha256": receipt.stderr_sha256,
        "stdout_truncated": receipt.stdout_truncated,
        "stderr_truncated": receipt.stderr_truncated,
        "capability": asdict(replace(receipt.capability, cached=False)),
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            os.chmod(temporary, 0o600)
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if hasattr(os, "O_DIRECTORY"):
            descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _clear_process_cache_for_tests() -> None:
    with _PROCESS_CACHE_LOCK:
        _PROCESS_CACHE.clear()
