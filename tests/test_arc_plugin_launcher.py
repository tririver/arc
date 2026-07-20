from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_PLUGIN = ROOT / "plugins/arc"
MCP_PLUGIN = ROOT / "plugins/arc-mcp"
SKILL = BASE_PLUGIN / "skills/arc"
CORE_LAUNCHER = SKILL / "scripts/arc-runtime"
MCP_LAUNCHER = MCP_PLUGIN / "scripts/arc-runtime"
CORE_BIN = BASE_PLUGIN / "bin"
MCP_BIN = MCP_PLUGIN / "bin"
CORE_TOOLS = (
    "arc-paper",
    "arc-domain",
    "arc-llm",
    "arc-typeset",
    "arc-companion",
    "arc-jobs",
)
ALL_TOOLS = (*CORE_TOOLS, "arc-mcp")


def _launcher_env(
    runtime_home: Path,
    *,
    uv: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "ARC_RUNTIME_HOME": str(runtime_home),
            "ARC_INSTALL_SOURCE": "git",
            "PATH": "/usr/bin:/bin",
        }
    )
    if uv is not None:
        env["ARC_INSTALL_UV"] = str(uv)
    if extra_env:
        env.update(extra_env)
    return env


def _run(
    runtime_home: Path,
    *args: str,
    launcher: Path = CORE_LAUNCHER,
    uv: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    runtime_home.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        [str(launcher), *args],
        cwd=runtime_home.parent,
        env=_launcher_env(runtime_home, uv=uv, extra_env=extra_env),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _runtime_dir(runtime_home: Path, launcher: Path = CORE_LAUNCHER) -> Path:
    result = _run(runtime_home, "doctor", launcher=launcher)
    assert result.returncode == 1, result.stderr
    values = dict(line.split("=", 1) for line in result.stdout.splitlines())
    return Path(values["runtime"])


def _write_fake_runtime_tool(bin_dir: Path, name: str, prefix: str = "cached") -> None:
    tool = bin_dir / name
    tool.write_text(
        "#!/bin/sh\nprintf '%s:%s\\n' '" + prefix + "' \"$1\"\n",
        encoding="utf-8",
    )
    tool.chmod(0o755)


def _prepare_ready_runtime(
    runtime_home: Path,
    *,
    launcher: Path,
    tools: tuple[str, ...],
    prefix: str,
) -> Path:
    doctor = _run(runtime_home, "doctor", launcher=launcher)
    assert doctor.returncode == 1, doctor.stderr
    values = dict(line.split("=", 1) for line in doctor.stdout.splitlines())
    runtime_dir = Path(values["runtime"])
    bin_dir = runtime_dir / "venv/bin"
    bin_dir.mkdir(parents=True)
    (runtime_dir / "install.ok").write_text(
        "\n".join(
            (
                f"profile={values['profile']}",
                f"runtime_fingerprint={values['fingerprint']}",
                f"constraints_sha256={values['constraints_sha256']}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    for tool in tools:
        _write_fake_runtime_tool(bin_dir, tool, prefix=f"{prefix}-{tool}")
    return runtime_dir


def _fake_uv(path: Path, prefix: str = "installed", *, fail: int = 0) -> Path:
    if fail:
        path.write_text(
            f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$UV_CALLS\"\n"
            f"printf 'fake installer exploded\\n' >&2\nexit {fail}\n",
            encoding="utf-8",
        )
        path.chmod(0o755)
        return path

    lines = [
        "#!/bin/sh",
        "printf '%s\\n' \"$*\" >> \"$UV_CALLS\"",
        "if [ \"$1\" = venv ]; then",
        "  mkdir -p \"$2/bin\"",
        "  printf '#!/bin/sh\\nexit 0\\n' > \"$2/bin/python\"",
    ]
    for tool in ALL_TOOLS:
        lines.append(
            f"  printf '#!/bin/sh\\nif [ \"$1\" = --help ]; then exit 0; fi\\necho {prefix}:{tool}:$1\\n' > \"$2/bin/{tool}\""
        )
    targets = " ".join(f'\"$2/bin/{tool}\"' for tool in ALL_TOOLS)
    lines.extend([f"  chmod +x \"$2/bin/python\" {targets}", "fi"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _fake_python3(path: Path, prefix: str = "pip") -> Path:
    inner = f"""\
#!/usr/bin/python3
import os
import pathlib
import sys

with open(os.environ["PY_CALLS"], "a", encoding="utf-8") as handle:
    handle.write("venv-python:" + " ".join(sys.argv[1:]) + "\\n")
if sys.argv[1:4] != ["-m", "pip", "install"]:
    sys.exit(99)
bin_dir = pathlib.Path(sys.argv[0]).parent
for tool in {list(ALL_TOOLS)!r}:
    target = bin_dir / tool
    target.write_text(
        "#!/bin/sh\\nif [ \\"$1\\" = --help ]; then exit 0; fi\\n"
        "echo {prefix}:" + tool + ":$1\\n",
        encoding="utf-8",
    )
    target.chmod(0o755)
"""
    outer = f"""\
#!/usr/bin/python3
import os
import pathlib
import sys

with open(os.environ["PY_CALLS"], "a", encoding="utf-8") as handle:
    handle.write("python3:" + " ".join(sys.argv[1:]) + "\\n")
if sys.argv[1:2] == ["-c"]:
    sys.exit(0)
if sys.argv[1:3] != ["-m", "venv"]:
    sys.exit(98)
venv = pathlib.Path(sys.argv[3])
bin_dir = venv / "bin"
bin_dir.mkdir(parents=True, exist_ok=True)
python = bin_dir / "python"
python.write_text({inner!r}, encoding="utf-8")
python.chmod(0o755)
"""
    path.write_text(textwrap.dedent(outer), encoding="utf-8")
    path.chmod(0o755)
    return path


def test_workflow_scripts_bootstrap_arc_llm_without_external_pythonpath() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = ""
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for script in ("ideas_runner.py", "calculate_runner.py"):
        result = subprocess.run(
            [sys.executable, str(SKILL / "workflows/scripts" / script), "--help"],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_base_plugin_is_cli_only_and_companion_alone_registers_mcp() -> None:
    base = json.loads((BASE_PLUGIN / ".codex-plugin/plugin.json").read_text())
    companion = json.loads((MCP_PLUGIN / ".codex-plugin/plugin.json").read_text())
    config = json.loads((MCP_PLUGIN / ".mcp.json").read_text())

    assert base["name"] == "arc"
    assert base["skills"] == "./skills/"
    assert "mcpServers" not in base
    assert "MCP" not in base["interface"]["capabilities"]
    assert not (BASE_PLUGIN / ".mcp.json").exists()
    assert not (BASE_PLUGIN / "bin/arc-mcp").exists()

    assert companion["name"] == "arc-mcp"
    assert companion["mcpServers"] == "./.mcp.json"
    assert companion["interface"]["capabilities"] == ["MCP"]
    assert config["mcpServers"]["arc"]["command"] == "./bin/arc-mcp"


def test_marketplaces_offer_arc_mcp_as_separate_optional_plugin() -> None:
    codex = json.loads((ROOT / ".agents/plugins/marketplace.json").read_text())
    claude = json.loads((ROOT / ".claude-plugin/marketplace.json").read_text())
    codex_entries = {entry["name"]: entry for entry in codex["plugins"]}
    claude_entries = {entry["name"]: entry for entry in claude["plugins"]}

    assert codex_entries["arc-mcp"]["source"]["path"] == "./plugins/arc-mcp"
    assert codex_entries["arc-mcp"]["policy"]["installation"] == "AVAILABLE"
    assert claude_entries["arc-mcp"]["source"] == "./plugins/arc-mcp"


def test_launchers_and_constraints_are_synchronized_but_profiles_are_isolated() -> None:
    assert CORE_LAUNCHER.read_bytes() == MCP_LAUNCHER.read_bytes()
    assert (SKILL / "scripts/runtime-constraints.txt").read_bytes() == (
        MCP_PLUGIN / "scripts/runtime-constraints.txt"
    ).read_bytes()
    assert (SKILL / "scripts/.arc-runtime-profile").read_text().strip() == "core"
    assert (MCP_PLUGIN / "scripts/.arc-runtime-profile").read_text().strip() == "mcp"
    constraints = (SKILL / "scripts/runtime-constraints.txt").read_text()
    for package in (
        "beautifulsoup4",
        "httpx",
        "json-repair",
        "jsonschema",
        "lxml",
        "mcp",
        "pydantic",
    ):
        assert re.search(rf"^{re.escape(package)}==[^=]+$", constraints, re.MULTILINE)


def test_plugin_bins_expose_runtime_jobs_and_profile_commands() -> None:
    for command in (*CORE_TOOLS, "arc-runtime"):
        assert (CORE_BIN / command).is_file()
        assert os.access(CORE_BIN / command, os.X_OK)
    for command in ("arc-mcp", "arc-runtime"):
        assert (MCP_BIN / command).is_file()
        assert os.access(MCP_BIN / command, os.X_OK)


def test_core_and_mcp_shims_reuse_separate_ready_runtimes(tmp_path: Path) -> None:
    runtime_home = tmp_path / "runtimes"
    core_dir = _prepare_ready_runtime(
        runtime_home, launcher=CORE_LAUNCHER, tools=CORE_TOOLS, prefix="core"
    )
    mcp_dir = _prepare_ready_runtime(
        runtime_home, launcher=MCP_LAUNCHER, tools=ALL_TOOLS, prefix="mcp"
    )
    assert core_dir != mcp_dir
    assert "/core/" in core_dir.as_posix()
    assert "/mcp/" in mcp_dir.as_posix()

    for tool in CORE_TOOLS:
        result = _run(runtime_home, "--probe", launcher=CORE_BIN / tool)
        assert result.returncode == 0, result.stderr
        assert result.stdout == f"core-{tool}:--probe\n"
    result = _run(runtime_home, "--probe", launcher=MCP_BIN / "arc-mcp")
    assert result.returncode == 0, result.stderr
    assert result.stdout == "mcp-arc-mcp:--probe\n"


def test_pinned_runtime_wins_over_same_named_path_command(tmp_path: Path) -> None:
    runtime_home = tmp_path / "runtimes"
    _prepare_ready_runtime(
        runtime_home, launcher=CORE_LAUNCHER, tools=CORE_TOOLS, prefix="pinned"
    )
    fake_bin = tmp_path / "path-bin"
    fake_bin.mkdir()
    _write_fake_runtime_tool(fake_bin, "arc-paper", prefix="path")

    result = _run(
        runtime_home,
        "arc-paper",
        "metadata",
        launcher=CORE_LAUNCHER,
        extra_env={"PATH": f"{fake_bin}:/usr/bin:/bin"},
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "pinned-arc-paper:metadata\n"


def test_first_core_use_lazy_installs_without_arc_mcp(tmp_path: Path) -> None:
    runtime_home = tmp_path / "runtimes"
    calls = tmp_path / "uv.log"
    uv = _fake_uv(tmp_path / "uv")
    first = _run(
        runtime_home,
        "arc-paper",
        "metadata",
        launcher=CORE_LAUNCHER,
        uv=uv,
        extra_env={"UV_CALLS": str(calls)},
    )
    second = _run(
        runtime_home,
        "arc-paper",
        "references",
        launcher=CORE_LAUNCHER,
        uv=uv,
        extra_env={"UV_CALLS": str(calls)},
    )

    assert first.returncode == second.returncode == 0
    assert first.stdout == "installed:arc-paper:metadata\n"
    assert second.stdout == "installed:arc-paper:references\n"
    install_calls = calls.read_text().splitlines()
    assert len(install_calls) == 2
    assert "arc-jobs @ git+https://github.com/tririver/arc.git@v1.0.0" in install_calls[1]
    assert "arc-mcp @" not in install_calls[1]
    assert "--constraint" in install_calls[1]


def test_companion_installs_complete_mcp_profile(tmp_path: Path) -> None:
    runtime_home = tmp_path / "runtimes"
    calls = tmp_path / "uv.log"
    uv = _fake_uv(tmp_path / "uv", prefix="mcp")
    result = _run(
        runtime_home,
        "arc-mcp",
        "root",
        launcher=MCP_LAUNCHER,
        uv=uv,
        extra_env={"UV_CALLS": str(calls)},
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "mcp:arc-mcp:root\n"
    install_call = calls.read_text().splitlines()[1]
    for package in ("arc-jobs", "arc-llm", "arc-paper", "arc-domain", "arc-typeset", "arc-companion", "arc-mcp"):
        assert f"{package} @ git+https://github.com/tririver/arc.git@v1.0.0" in install_call


def test_standalone_skill_launcher_is_complete(tmp_path: Path) -> None:
    skill = tmp_path / "installed-skills/arc"
    shutil.copytree(SKILL, skill)
    launcher = skill / "scripts/arc-runtime"
    runtime_home = tmp_path / "runtimes"
    calls = tmp_path / "uv.log"
    uv = _fake_uv(tmp_path / "uv", prefix="standalone")

    result = _run(
        runtime_home,
        "arc-domain",
        "root",
        launcher=launcher,
        uv=uv,
        extra_env={"UV_CALLS": str(calls)},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "standalone:arc-domain:root\n"


def test_runtime_fingerprint_covers_ref_constraints_python_and_local_content(
    tmp_path: Path,
) -> None:
    runtime_home = tmp_path / "runtimes"

    default_dir = _runtime_dir(runtime_home)
    changed_ref = _run(
        runtime_home,
        "doctor",
        extra_env={"ARC_INSTALL_REF": "v1.0.1"},
    )
    assert changed_ref.returncode == 1
    changed_ref_dir = Path(
        dict(line.split("=", 1) for line in changed_ref.stdout.splitlines())["runtime"]
    )

    skill = tmp_path / "changed-skill"
    shutil.copytree(SKILL, skill)
    with (skill / "scripts/runtime-constraints.txt").open("a", encoding="utf-8") as handle:
        handle.write("# distinct constraints payload\n")
    changed_constraints = _run(
        runtime_home,
        "doctor",
        launcher=skill / "scripts/arc-runtime",
    )
    assert changed_constraints.returncode == 1
    changed_constraints_dir = Path(
        dict(line.split("=", 1) for line in changed_constraints.stdout.splitlines())["runtime"]
    )

    python_a = tmp_path / "python-a"
    python_b = tmp_path / "python-b"
    for python, version in ((python_a, "3.11.9"), (python_b, "3.12.4")):
        python.write_text(
            f"#!/bin/sh\nif [ \"$1\" = -c ]; then echo {version}; exit 0; fi\nexit 1\n",
            encoding="utf-8",
        )
        python.chmod(0o755)
    python_results = [
        _run(
            runtime_home,
            "doctor",
            extra_env={"ARC_INSTALL_PYTHON_BIN": str(python)},
        )
        for python in (python_a, python_b)
    ]
    python_dirs = {
        Path(dict(line.split("=", 1) for line in result.stdout.splitlines())["runtime"])
        for result in python_results
    }

    checkout = tmp_path / "checkout"
    for package in ("arc-jobs", "arc-llm", "arc-paper", "arc-domain", "arc-typeset", "arc-companion"):
        package_dir = checkout / "packages" / package
        package_dir.mkdir(parents=True)
        (package_dir / "pyproject.toml").write_text("[project]\nname='example'\n")
    local_env = {
        "ARC_INSTALL_SOURCE": "local",
        "ARC_INSTALL_REPO_ROOT": str(checkout),
    }
    local_first = _run(runtime_home, "doctor", extra_env=local_env)
    local_first_dir = Path(
        dict(line.split("=", 1) for line in local_first.stdout.splitlines())["runtime"]
    )
    (checkout / "packages/arc-paper/pyproject.toml").write_text(
        "[project]\nname='changed'\n"
    )
    local_second = _run(runtime_home, "doctor", extra_env=local_env)
    local_second_dir = Path(
        dict(line.split("=", 1) for line in local_second.stdout.splitlines())["runtime"]
    )

    assert default_dir != changed_ref_dir
    assert default_dir != changed_constraints_dir
    assert len(python_dirs) == 2
    assert local_first_dir != local_second_dir


def test_setup_doctor_and_retry_are_idempotent(tmp_path: Path) -> None:
    runtime_home = tmp_path / "runtimes"
    calls = tmp_path / "uv.log"
    bad_uv = _fake_uv(tmp_path / "bad-uv", fail=42)
    good_uv = _fake_uv(tmp_path / "good-uv", prefix="retry")

    failed = _run(
        runtime_home,
        "setup",
        "--profile",
        "core",
        launcher=CORE_LAUNCHER,
        uv=bad_uv,
        extra_env={"UV_CALLS": str(calls)},
    )
    repeated = _run(
        runtime_home,
        "setup",
        launcher=CORE_LAUNCHER,
        uv=good_uv,
        extra_env={"UV_CALLS": str(calls)},
    )
    retried = _run(
        runtime_home,
        "setup",
        "--retry",
        launcher=CORE_LAUNCHER,
        uv=good_uv,
        extra_env={"UV_CALLS": str(calls)},
    )
    doctor = _run(runtime_home, "doctor", launcher=CORE_LAUNCHER)

    assert failed.returncode == 42
    assert "Install log tail:" in failed.stderr
    assert repeated.returncode == 42
    assert "Previous ARC core runtime install failed" in repeated.stderr
    assert retried.returncode == 0, retried.stderr
    assert "ARC core runtime ready:" in retried.stdout
    assert doctor.returncode == 0
    assert "status=ready" in doctor.stdout


def test_stale_install_lock_is_recovered_and_live_lock_is_preserved(tmp_path: Path) -> None:
    runtime_home = tmp_path / "runtimes"
    runtime_dir = _runtime_dir(runtime_home)
    lock_dir = runtime_dir / "install.lock"
    lock_dir.mkdir(parents=True)
    hostname = subprocess.run(
        ["hostname"], text=True, stdout=subprocess.PIPE, check=True
    ).stdout.strip()
    (lock_dir / "owner").write_text("stale-owner\n")
    (lock_dir / "pid").write_text("99999999\n")
    (lock_dir / "host").write_text(hostname + "\n")
    (lock_dir / "created").write_text(f"{int(time.time())}\n")
    (lock_dir / "process-start").write_text("1\n")
    calls = tmp_path / "uv.log"
    installed = _run(
        runtime_home,
        "setup",
        uv=_fake_uv(tmp_path / "uv"),
        extra_env={"UV_CALLS": str(calls)},
    )
    assert installed.returncode == 0, installed.stderr
    assert "Recovered stale ARC runtime install lock" in installed.stderr
    assert not lock_dir.exists()

    legacy_home = tmp_path / "legacy-runtimes"
    legacy_runtime = _runtime_dir(legacy_home)
    legacy_lock = legacy_runtime / "install.lock"
    legacy_lock.mkdir(parents=True)
    old = time.time() - 30
    os.utime(legacy_lock, (old, old))
    legacy_calls = tmp_path / "legacy-uv.log"
    legacy = _run(
        legacy_home,
        "setup",
        uv=_fake_uv(tmp_path / "legacy-uv"),
        extra_env={
            "ARC_INSTALL_LOCK_INIT_GRACE_SEC": "1",
            "UV_CALLS": str(legacy_calls),
        },
    )
    assert legacy.returncode == 0, legacy.stderr
    assert "Recovered stale ARC runtime install lock" in legacy.stderr
    assert not legacy_lock.exists()

    live_home = tmp_path / "live-runtimes"
    live_runtime = _runtime_dir(live_home)
    live_lock = live_runtime / "install.lock"
    live_lock.mkdir(parents=True)
    (live_lock / "owner").write_text("live-owner\n")
    (live_lock / "pid").write_text(f"{os.getpid()}\n")
    (live_lock / "host").write_text(hostname + "\n")
    (live_lock / "created").write_text(f"{int(time.time())}\n")
    waiting = _run(
        live_home,
        "setup",
        extra_env={"ARC_INSTALL_LOCK_TIMEOUT_SEC": "0"},
    )
    assert waiting.returncode == 75
    assert "Timed out waiting" in waiting.stderr
    assert live_lock.is_dir()


def test_doctor_exposes_deterministic_runtime_identity_and_ready_marker(tmp_path: Path) -> None:
    runtime_home = tmp_path / "runtimes"
    calls = tmp_path / "uv.log"
    setup = _run(
        runtime_home,
        "setup",
        uv=_fake_uv(tmp_path / "uv"),
        extra_env={"UV_CALLS": str(calls)},
    )
    assert setup.returncode == 0, setup.stderr

    doctor = _run(runtime_home, "doctor")
    assert doctor.returncode == 0, doctor.stderr
    values = dict(line.split("=", 1) for line in doctor.stdout.splitlines())
    assert values["status"] == "ready"
    assert values["profile"] == "core"
    assert values["source"] == "git"
    assert re.fullmatch(r"[0-9a-f]{64}", values["fingerprint"])
    assert re.fullmatch(r"[0-9a-f]{64}", values["constraints_sha256"])
    assert Path(values["runtime"]).name == values["fingerprint"]
    assert Path(values["venv"]) == Path(values["runtime"]) / "venv"
    ready_file = Path(values["ready_file"])
    assert ready_file == Path(values["runtime"]) / "install.ok"
    marker = dict(line.split("=", 1) for line in ready_file.read_text().splitlines())
    assert marker["runtime_fingerprint"] == values["fingerprint"]
    assert marker["constraints_sha256"] == values["constraints_sha256"]
    assert marker["profile"] == "core"
    for path in Path(values["runtime"]).rglob("*"):
        mode = path.stat().st_mode
        assert mode & 0o077 == 0, f"runtime state is not private: {path}"


def test_repository_credentials_are_redacted_from_runtime_state(tmp_path: Path) -> None:
    runtime_home = tmp_path / "runtimes"
    secret_repo = "https://private-user:top-secret@example.invalid/org/arc.git"
    secret_uv = tmp_path / "secret-uv"
    secret_uv.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = venv ]; then\n"
        "  mkdir -p \"$2/bin\"\n"
        "  printf '#!/bin/sh\\nexit 0\\n' > \"$2/bin/python\"\n"
        "  chmod 700 \"$2/bin/python\"\n"
        "  exit 0\n"
        "fi\n"
        "printf '%s\\n' \"$*\" >&2\n"
        "exit 42\n",
        encoding="utf-8",
    )
    secret_uv.chmod(0o755)
    env = {"ARC_INSTALL_REPO": secret_repo}

    doctor = _run(runtime_home, "doctor", extra_env=env)
    assert doctor.returncode == 1
    assert "private-user" not in doctor.stdout
    assert "top-secret" not in doctor.stdout
    assert "https://[REDACTED]@example.invalid/org/arc.git" in doctor.stdout
    values = dict(line.split("=", 1) for line in doctor.stdout.splitlines())
    runtime_dir = Path(values["runtime"])

    failed = _run(
        runtime_home,
        "setup",
        uv=secret_uv,
        extra_env=env,
    )
    assert failed.returncode == 42
    combined = failed.stdout + failed.stderr
    for path in runtime_dir.rglob("*"):
        if path.is_file():
            combined += path.read_text(encoding="utf-8", errors="ignore")
    assert "private-user" not in combined
    assert "top-secret" not in combined
    assert "[REDACTED]@example.invalid" in (runtime_dir / "install.log").read_text()


def test_python_venv_fallback_uses_constraints(tmp_path: Path) -> None:
    runtime_home = tmp_path / "runtimes"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _fake_python3(fake_bin / "python3")
    calls = tmp_path / "python.log"
    result = _run(
        runtime_home,
        "arc-paper",
        "metadata",
        launcher=CORE_LAUNCHER,
        extra_env={
            "ARC_INSTALL_UV": str(tmp_path / "missing-uv"),
            "ARC_INSTALL_PYTHON_BIN": str(fake_bin / "python3"),
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "PY_CALLS": str(calls),
        },
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "pip:arc-paper:metadata\n"
    text = calls.read_text()
    assert "python3:-m venv " in text
    assert "venv-python:-m pip install --constraint " in text


def test_installed_plugin_commit_wins_over_bundled_tag(tmp_path: Path) -> None:
    plugin_root = tmp_path / ".claude/plugins/cache/arc/arc-mcp/1.0.0"
    scripts = plugin_root / "scripts"
    scripts.mkdir(parents=True)
    shutil.copyfile(MCP_LAUNCHER, scripts / "arc-runtime")
    shutil.copyfile(MCP_PLUGIN / "scripts/.arc-runtime-profile", scripts / ".arc-runtime-profile")
    shutil.copyfile(MCP_PLUGIN / "scripts/runtime-constraints.txt", scripts / "runtime-constraints.txt")
    shutil.copyfile(MCP_PLUGIN / ".arc-install-ref", plugin_root / ".arc-install-ref")
    (plugin_root / ".claude-plugin").mkdir()
    (plugin_root / ".claude-plugin/plugin.json").write_text('{"name":"arc-mcp"}\n')
    launcher = scripts / "arc-runtime"
    launcher.chmod(0o755)

    home = tmp_path / "home"
    metadata = home / ".claude/plugins"
    metadata.mkdir(parents=True)
    commit = "15ce9ea6138ccf98f73bc64963964d7b0666ab73"
    (metadata / "installed_plugins.json").write_text(
        json.dumps(
            {
                "version": 2,
                "plugins": {
                    "arc-mcp@arc": [
                        {"installPath": str(plugin_root), "gitCommitSha": commit}
                    ]
                },
            }
        )
    )
    calls = tmp_path / "uv.log"
    uv = _fake_uv(tmp_path / "uv", prefix="pinned")
    result = _run(
        tmp_path / "runtimes",
        "arc-mcp",
        "root",
        launcher=launcher,
        uv=uv,
        extra_env={"HOME": str(home), "UV_CALLS": str(calls)},
    )
    assert result.returncode == 0, result.stderr
    assert f"@{commit}#subdirectory=packages/arc-mcp" in calls.read_text()
    assert "@v1.0.0#subdirectory=packages/arc-mcp" not in calls.read_text()


def test_mutable_install_refs_and_cross_profile_access_are_rejected(tmp_path: Path) -> None:
    mutable = _run(
        tmp_path / "mutable",
        "doctor",
        launcher=CORE_LAUNCHER,
        extra_env={"ARC_INSTALL_REF": "main"},
    )
    wrong_profile = _run(
        tmp_path / "profile",
        "setup",
        "--profile",
        "mcp",
        launcher=CORE_LAUNCHER,
    )
    direct_mcp = _run(
        tmp_path / "direct",
        "arc-mcp",
        "--help",
        launcher=CORE_LAUNCHER,
    )
    assert mutable.returncode == 78
    assert "full commit SHA or immutable" in mutable.stderr
    assert wrong_profile.returncode == 64
    assert direct_mcp.returncode == 64


def test_configured_local_checkout_installs_without_git_urls(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    for package in ("arc-jobs", "arc-llm", "arc-paper", "arc-domain", "arc-typeset", "arc-companion"):
        package_dir = checkout / "packages" / package
        package_dir.mkdir(parents=True)
        (package_dir / "pyproject.toml").write_text("[project]\n")
    calls = tmp_path / "uv.log"
    uv = _fake_uv(tmp_path / "uv", prefix="local")
    result = _run(
        tmp_path / "runtimes",
        "arc-paper",
        "root",
        launcher=CORE_LAUNCHER,
        uv=uv,
        extra_env={
            "ARC_INSTALL_SOURCE": "local",
            "ARC_INSTALL_REPO_ROOT": str(checkout),
            "UV_CALLS": str(calls),
        },
    )
    assert result.returncode == 0, result.stderr
    install_call = calls.read_text().splitlines()[1]
    assert str(checkout / "packages/arc-paper") in install_call
    assert "git+" not in install_call


def test_launcher_surface_has_no_legacy_mcp_or_dot_local_paths() -> None:
    files = [
        CORE_LAUNCHER,
        MCP_LAUNCHER,
        *(CORE_BIN / name for name in (*CORE_TOOLS, "arc-runtime")),
        MCP_BIN / "arc-mcp",
        MCP_BIN / "arc-runtime",
    ]
    combined = "\n".join(path.read_text() for path in files)
    assert "ARC_MCP_" not in combined
    assert "arc-mcp-runtime" not in combined
    assert ".local" not in combined
    assert "@main" not in combined
    assert "@stable" not in combined
