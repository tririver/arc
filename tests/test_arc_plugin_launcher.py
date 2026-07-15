import os
import shutil
import subprocess
import sys
import textwrap
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


def test_workflow_scripts_bootstrap_arc_llm_without_external_pythonpath():
    env = os.environ.copy()
    env["PYTHONPATH"] = ""
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    for script in ("ideas_runner.py", "calculate_runner.py"):
        result = subprocess.run(
            [sys.executable, str(ROOT / "plugins/arc/skills/arc/workflows/scripts" / script), "--help"],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        assert result.returncode == 0, result.stderr


def test_workflow_bootstrap_failure_lists_searched_roots_and_runtimes(tmp_path):
    scripts_dir = tmp_path / "installed-plugin/skills/arc/workflows/scripts"
    scripts_dir.mkdir(parents=True)
    shutil.copyfile(
        ROOT / "plugins/arc/skills/arc/workflows/scripts/_arc_script_bootstrap.py",
        scripts_dir / "_arc_script_bootstrap.py",
    )
    env = {
        "HOME": str(tmp_path / "empty-home"),
        "PYTHONPATH": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "ARC_REPO_ROOT": str(tmp_path / "missing-repo"),
        "ARC_MCP_REPO_ROOT": str(tmp_path / "missing-mcp-repo"),
        "ARC_MCP_RUNTIME_DIR": str(tmp_path / "missing-runtime"),
        "XDG_CACHE_HOME": str(tmp_path / "empty-cache"),
    }

    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0, {str(scripts_dir)!r}); "
                "from _arc_script_bootstrap import bootstrap_arc_pythonpath; "
                "bootstrap_arc_pythonpath()"
            ),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "Cannot import ARC internal module `arc_llm`" in result.stderr
    assert "Searched ARC roots:" in result.stderr
    assert "Searched ARC runtime site-packages:" in result.stderr


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


def _fake_python3_installs_all_tools(path: Path, prefix: str = "pip") -> Path:
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
for tool in {list(ARC_TOOLS)!r}:
    target = bin_dir / tool
    target.write_text(
        "#!/bin/sh\\n"
        "if [ \\"$1\\" = --help ]; then exit 0; fi\\n"
        "echo {prefix}:" + tool + ":$1\\n",
        encoding="utf-8",
    )
    target.chmod(0o755)
sys.exit(0)
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
sys.exit(0)
"""
    path.write_text(textwrap.dedent(outer), encoding="utf-8")
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


def test_ready_runtime_does_not_need_installer_tools(tmp_path):
    runtime_dir = tmp_path / "runtime"
    bin_dir = runtime_dir / "venv/bin"
    bin_dir.mkdir(parents=True)
    (runtime_dir / "install.ok").write_text("ready\n", encoding="utf-8")
    _write_fake_runtime_tool(bin_dir, "arc-mcp")

    result = _run_launcher(
        runtime_dir,
        "status",
        extra_env={"ARC_MCP_INSTALLER_UV": str(tmp_path / "missing-uv")},
    )

    assert result.returncode == 0
    assert result.stdout == "cached:status\n"
    assert result.stderr == ""


def test_launcher_falls_back_to_python_venv_when_uv_is_unavailable(tmp_path):
    runtime_dir = tmp_path / "runtime"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _fake_python3_installs_all_tools(fake_bin / "python3", prefix="fallback")
    calls = tmp_path / "python-calls.log"

    result = _run_launcher(
        runtime_dir,
        "status",
        extra_env={
            "ARC_MCP_INSTALLER_UV": str(tmp_path / "missing-uv"),
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "PY_CALLS": str(calls),
        },
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "fallback:arc-mcp:status\n"
    call_text = calls.read_text(encoding="utf-8")
    assert "python3:-m venv " in call_text
    assert "venv-python:-m pip install " in call_text


def test_launcher_uses_claude_installed_plugin_commit_for_git_runtime(tmp_path):
    runtime_dir = tmp_path / "runtime"
    fake_plugin_root = tmp_path / ".claude/plugins/cache/arc/arc/0.1.0"
    fake_launcher = fake_plugin_root / "bin/arc-mcp"
    fake_launcher.parent.mkdir(parents=True)
    shutil.copyfile(LAUNCHER, fake_launcher)
    fake_launcher.chmod(0o755)

    home = tmp_path / "home"
    metadata_dir = home / ".claude/plugins"
    metadata_dir.mkdir(parents=True)
    installed_commit = "15ce9ea6138ccf98f73bc64963964d7b0666ab73"
    (metadata_dir / "installed_plugins.json").write_text(
        (
            '{"version":2,"plugins":{"arc@arc":[{"scope":"user",'
            f'"installPath":"{fake_plugin_root}",'
            '"version":"0.1.0",'
            f'"gitCommitSha":"{installed_commit}"'
            "}]}}"
        ),
        encoding="utf-8",
    )

    calls = tmp_path / "uv-calls.log"
    fake_uv = tmp_path / "uv"
    _fake_uv_installs_all_tools(fake_uv, prefix="pinned")

    result = _run_launcher(
        runtime_dir,
        "status",
        launcher=fake_launcher,
        uv=fake_uv,
        extra_env={
            "HOME": str(home),
            "UV_CALLS": str(calls),
        },
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "pinned:arc-mcp:status\n"
    install_call = calls.read_text(encoding="utf-8").splitlines()[1]
    assert f"@{installed_commit}#subdirectory=packages/arc-mcp" in install_call
    assert "@main#subdirectory" not in install_call
    assert f"ref={installed_commit}" in (runtime_dir / "install.ok").read_text(
        encoding="utf-8"
    )


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


def test_launcher_failure_stderr_includes_install_log_tail(tmp_path):
    runtime_dir = tmp_path / "runtime"
    bad_uv = tmp_path / "bad-uv"
    bad_uv.write_text(
        "#!/bin/sh\nprintf 'fake installer exploded\\n' >&2\nexit 42\n",
        encoding="utf-8",
    )
    bad_uv.chmod(0o755)

    result = _run_launcher(runtime_dir, uv=bad_uv)

    assert result.returncode == 42
    assert "ARC MCP runtime install failed." in result.stderr
    assert "Install log tail:" in result.stderr
    assert "fake installer exploded" in result.stderr


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
