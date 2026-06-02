import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "plugins/arc/bin/arc-mcp"
PLUGIN_BIN = ROOT / "plugins/arc/bin"
ARC_TOOLS = ("arc-mcp", "arc-paper", "arc-domain", "arc-llm", "arc-typeset")


def _run_launcher(
    runtime_dir: Path,
    *args: str,
    launcher: Path = LAUNCHER,
    uv: Path | None = None,
    retry: bool = False,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "ARC_MCP_RUNTIME_DIR": str(runtime_dir),
            "ARC_MCP_LAUNCHER_NO_PATH": "1",
            "PATH": "/usr/bin:/bin",
        }
    )
    if uv is not None:
        env["ARC_MCP_INSTALLER_UV"] = str(uv)
    if retry:
        env["ARC_MCP_INSTALL_RETRY"] = "1"
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [str(launcher), *args],
        cwd=runtime_dir.parent,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _write_fake_runtime_tool(bin_dir: Path, name: str, prefix: str = "cached") -> None:
    tool = bin_dir / name
    tool.write_text(
        "#!/bin/sh\nprintf '%s:%s\\n' '" + prefix + "' \"$1\"\n",
        encoding="utf-8",
    )
    tool.chmod(0o755)


def _fake_uv_installs_all_tools(path: Path, prefix: str = "installed") -> Path:
    lines = [
        "#!/bin/sh",
        "printf '%s\\n' \"$*\" >> \"$UV_CALLS\"",
        "if [ \"$1\" = venv ]; then",
        "  mkdir -p \"$2/bin\"",
        "  printf '#!/bin/sh\\nexit 0\\n' > \"$2/bin/python\"",
    ]
    for tool in ARC_TOOLS:
        lines.append(
            f"  printf '#!/bin/sh\\nif [ \"$1\" = --help ]; then exit 0; fi\\necho {prefix}:{tool}:$1\\n' > \"$2/bin/{tool}\""
        )
    chmod_targets = " ".join(f'"$2/bin/{tool}"' for tool in ARC_TOOLS)
    lines.extend(
        [
            f"  chmod +x \"$2/bin/python\" {chmod_targets}",
            "fi",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_launcher_reuses_ready_cached_runtime(tmp_path):
    runtime_dir = tmp_path / "runtime"
    bin_dir = runtime_dir / "venv/bin"
    bin_dir.mkdir(parents=True)
    (runtime_dir / "install.ok").write_text("ready\n", encoding="utf-8")
    _write_fake_runtime_tool(bin_dir, "arc-mcp")

    result = _run_launcher(runtime_dir, "root", "--json")

    assert result.returncode == 0
    assert result.stdout == "cached:root\n"


def test_plugin_shims_reuse_ready_cached_runtime(tmp_path):
    runtime_dir = tmp_path / "runtime"
    bin_dir = runtime_dir / "venv/bin"
    bin_dir.mkdir(parents=True)
    (runtime_dir / "install.ok").write_text("ready\n", encoding="utf-8")
    for tool in ARC_TOOLS:
        _write_fake_runtime_tool(bin_dir, tool, prefix=f"cached-{tool}")

    for tool in ARC_TOOLS:
        result = _run_launcher(runtime_dir, "--probe", launcher=PLUGIN_BIN / tool)
        assert result.returncode == 0, result.stderr
        assert result.stdout == f"cached-{tool}:--probe\n"


def test_plugin_bin_exposes_all_arc_commands(tmp_path):
    runtime_dir = tmp_path / "runtime"
    bin_dir = runtime_dir / "venv/bin"
    bin_dir.mkdir(parents=True)
    (runtime_dir / "install.ok").write_text("ready\n", encoding="utf-8")
    for tool in ARC_TOOLS:
        _write_fake_runtime_tool(bin_dir, tool, prefix=f"plugin-{tool}")

    env = os.environ.copy()
    env.update(
        {
            "ARC_MCP_RUNTIME_DIR": str(runtime_dir),
            "PATH": f"{PLUGIN_BIN}:/usr/bin:/bin",
        }
    )
    for tool in ARC_TOOLS:
        result = subprocess.run(
            [tool, "--probe"],
            cwd=tmp_path,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout == f"plugin-{tool}:--probe\n"


def test_launcher_installs_once_outside_current_directory(tmp_path):
    runtime_dir = tmp_path / "runtime"
    calls = tmp_path / "uv-calls.log"
    fake_uv = tmp_path / "uv"
    _fake_uv_installs_all_tools(fake_uv)

    first = _run_launcher(runtime_dir, "status", uv=fake_uv, extra_env={"UV_CALLS": str(calls)})
    second = _run_launcher(runtime_dir, "result", uv=fake_uv, extra_env={"UV_CALLS": str(calls)})

    assert first.returncode == 0
    assert second.returncode == 0
    assert first.stdout == "installed:arc-mcp:status\n"
    assert second.stdout == "installed:arc-mcp:result\n"
    assert len(calls.read_text(encoding="utf-8").splitlines()) == 2
    assert not (tmp_path / ".venv").exists()
    assert ".venv" not in calls.read_text(encoding="utf-8")


def test_launcher_failed_install_records_failure_without_retrying(tmp_path):
    runtime_dir = tmp_path / "runtime"
    calls = tmp_path / "uv-calls.log"
    bad_uv = tmp_path / "bad-uv"
    bad_uv.write_text("#!/bin/sh\nprintf bad >> \"$UV_CALLS\"\nexit 42\n", encoding="utf-8")
    bad_uv.chmod(0o755)
    good_uv = tmp_path / "good-uv"
    _fake_uv_installs_all_tools(good_uv, prefix="ok")

    first = _run_launcher(runtime_dir, uv=bad_uv, extra_env={"UV_CALLS": str(calls)})
    second = _run_launcher(runtime_dir, uv=good_uv, extra_env={"UV_CALLS": str(calls)})
    assert calls.read_text(encoding="utf-8") == "bad"
    third = _run_launcher(runtime_dir, uv=good_uv, retry=True, extra_env={"UV_CALLS": str(calls)})

    assert first.returncode == 42
    assert second.returncode == 42
    assert "Previous ARC MCP runtime install failed" in second.stderr
    assert third.returncode == 0
    assert third.stdout == "ok:arc-mcp:\n"
    retry_calls = calls.read_text(encoding="utf-8")
    assert retry_calls.startswith("badvenv ")
    assert "\npip install --python " in retry_calls


def test_launcher_uses_configured_repo_root_for_local_packages(tmp_path):
    runtime_dir = tmp_path / "runtime"
    fake_plugin_root = tmp_path / "installed-plugin"
    fake_launcher = fake_plugin_root / "bin/arc-mcp"
    fake_launcher.parent.mkdir(parents=True)
    shutil.copyfile(LAUNCHER, fake_launcher)
    fake_launcher.chmod(0o755)

    local_repo = tmp_path / "local-repo"
    for package in ("arc-llm", "arc-paper", "arc-domain", "arc-typeset", "arc-mcp"):
        (local_repo / "packages" / package).mkdir(parents=True)

    calls = tmp_path / "uv-calls.log"
    fake_uv = tmp_path / "uv"
    _fake_uv_installs_all_tools(fake_uv, prefix="local")

    result = _run_launcher(
        runtime_dir,
        "root",
        launcher=fake_launcher,
        uv=fake_uv,
        extra_env={
            "ARC_MCP_REPO_ROOT": str(local_repo),
            "ARC_MCP_INSTALL_SOURCE": "local",
            "UV_CALLS": str(calls),
        },
    )

    assert result.returncode == 0
    assert result.stdout == "local:arc-mcp:root\n"
    install_call = calls.read_text(encoding="utf-8").splitlines()[1]
    assert str(local_repo / "packages/arc-mcp") in install_call
    assert "git+" not in install_call
