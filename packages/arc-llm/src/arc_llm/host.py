from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class HostDetection:
    host: str
    confidence: float
    signals: list[str]


@dataclass(frozen=True)
class ProviderSelection:
    provider: str
    host: HostDetection
    signals: list[str]


def detect_host(
    *,
    env: Mapping[str, str] | None = None,
    process_chain: Sequence[str] | None = None,
) -> HostDetection:
    env = env if env is not None else os.environ
    if host := env.get("ARC_AGENT_HOST"):
        return HostDetection(host=host, confidence=1.0, signals=[f"env:ARC_AGENT_HOST={host}"])

    chain = list(process_chain) if process_chain is not None else _parent_process_chain()
    for item in chain:
        lowered = item.lower()
        if "@moonshot-ai/kimi-code" in lowered or _has_command_token(lowered, "kimi"):
            return HostDetection(host="kimi-code", confidence=0.9, signals=[f"parent:{item}"])
        if "@openai/codex" in lowered or _has_command_token(lowered, "codex"):
            return HostDetection(host="codex", confidence=0.9, signals=[f"parent:{item}"])
        if (
            "@anthropic-ai/claude-code" in lowered
            or ".claude/shell-snapshots" in lowered
            or _has_command_token(lowered, "claude")
        ):
            return HostDetection(host="claude-code", confidence=0.9, signals=[f"parent:{item}"])

    return HostDetection(host="unknown", confidence=0.0, signals=[])


def select_llm_provider(
    *,
    env: Mapping[str, str] | None = None,
    process_chain: Sequence[str] | None = None,
    explicit_provider: str | None = None,
) -> ProviderSelection:
    env = env if env is not None else os.environ
    host = detect_host(env=env, process_chain=process_chain)
    if explicit_provider:
        return ProviderSelection(provider=explicit_provider, host=host, signals=["explicit"])
    if native := _host_native_provider(host):
        return ProviderSelection(provider=native, host=host, signals=host.signals)
    return ProviderSelection(provider="manual", host=host, signals=host.signals)


def _host_native_provider(host: HostDetection) -> str | None:
    if host.host == "codex":
        return "codex-cli"
    if host.host == "claude-code":
        return "claude-cli"
    if host.host == "kimi-code":
        return "kimi-code-cli"
    return None


def _has_command_token(text: str, token: str) -> bool:
    parts = text.replace("\\", "/").split()
    return any(Path(part).name == token for part in parts)


def _parent_process_chain() -> list[str]:
    system = platform.system().lower()
    if system == "linux":
        return _linux_parent_process_chain()
    if system == "darwin":
        return _ps_parent_process_chain()
    if system == "windows":
        return _windows_parent_process_chain()
    return []


def _linux_parent_process_chain() -> list[str]:
    out: list[str] = []
    pid = os.getppid()
    seen: set[int] = set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        proc = Path("/proc") / str(pid)
        cmd = _read_nul_file(proc / "cmdline")
        if cmd:
            out.append(cmd)
        status = _read_text_file(proc / "status")
        next_pid = _parse_ppid(status)
        if next_pid is None:
            break
        pid = next_pid
    return out


def _ps_parent_process_chain() -> list[str]:
    out: list[str] = []
    pid = os.getppid()
    seen: set[int] = set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "ppid=,command="],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        line = result.stdout.strip()
        if not line:
            break
        first, _, command = line.partition(" ")
        out.append(command.strip())
        try:
            pid = int(first.strip())
        except ValueError:
            break
    return out


def _windows_parent_process_chain() -> list[str]:
    out: list[str] = []
    pid = os.getppid()
    seen: set[int] = set()
    while pid > 0 and pid not in seen:
        seen.add(pid)
        script = (
            "$p=Get-CimInstance Win32_Process -Filter \"ProcessId=%d\";"
            "if ($p) { Write-Output ($p.ParentProcessId.ToString() + \"`t\" + $p.CommandLine) }"
        ) % pid
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        line = result.stdout.strip()
        if not line:
            break
        parent, _, command = line.partition("\t")
        out.append(command.strip())
        try:
            pid = int(parent)
        except ValueError:
            break
    return out


def _read_nul_file(path: Path) -> str:
    try:
        return path.read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _parse_ppid(status: str) -> int | None:
    for line in status.splitlines():
        if line.startswith("PPid:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None
