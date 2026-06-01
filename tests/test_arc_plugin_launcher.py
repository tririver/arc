import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "plugins/arc/bin/arc-mcp"


def _run_launcher(
    runtime_dir: Path,
    *args: str,
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
        [str(LAUNCHER), *args],
        cwd=runtime_dir.parent,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_launcher_reuses_ready_cached_runtime(tmp_path):
    runtime_dir = tmp_path / "runtime"
    bin_dir = runtime_dir / "venv/bin"
    bin_dir.mkdir(parents=True)
    (runtime_dir / "install.ok").write_text("ready\n", encoding="utf-8")
    arc_mcp = bin_dir / "arc-mcp"
    arc_mcp.write_text("#!/bin/sh\nprintf 'cached:%s\\n' \"$1\"\n", encoding="utf-8")
    arc_mcp.chmod(0o755)

    result = _run_launcher(runtime_dir, "root", "--json")

    assert result.returncode == 0
    assert result.stdout == "cached:root\n"


def test_launcher_installs_once_outside_current_directory(tmp_path):
    runtime_dir = tmp_path / "runtime"
    calls = tmp_path / "uv-calls.log"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        "\n".join(
            [
                "#!/bin/sh",
                "printf '%s\\n' \"$*\" >> \"$UV_CALLS\"",
                "if [ \"$1\" = venv ]; then",
                "  mkdir -p \"$2/bin\"",
                "  printf '#!/bin/sh\\nexit 0\\n' > \"$2/bin/python\"",
                "  printf '#!/bin/sh\\nif [ \"$1\" = --help ]; then exit 0; fi\\necho installed:$1\\n' > \"$2/bin/arc-mcp\"",
                "  chmod +x \"$2/bin/python\" \"$2/bin/arc-mcp\"",
                "fi",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    first = _run_launcher(runtime_dir, "status", uv=fake_uv, extra_env={"UV_CALLS": str(calls)})
    second = _run_launcher(runtime_dir, "result", uv=fake_uv, extra_env={"UV_CALLS": str(calls)})

    assert first.returncode == 0
    assert second.returncode == 0
    assert first.stdout == "installed:status\n"
    assert second.stdout == "installed:result\n"
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
    good_uv.write_text(
        "\n".join(
            [
                "#!/bin/sh",
                "printf good >> \"$UV_CALLS\"",
                "if [ \"$1\" = venv ]; then",
                "  mkdir -p \"$2/bin\"",
                "  printf '#!/bin/sh\\nexit 0\\n' > \"$2/bin/python\"",
                "  printf '#!/bin/sh\\nif [ \"$1\" = --help ]; then exit 0; fi\\necho ok\\n' > \"$2/bin/arc-mcp\"",
                "  chmod +x \"$2/bin/python\" \"$2/bin/arc-mcp\"",
                "fi",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    good_uv.chmod(0o755)

    first = _run_launcher(runtime_dir, uv=bad_uv, extra_env={"UV_CALLS": str(calls)})
    second = _run_launcher(runtime_dir, uv=good_uv, extra_env={"UV_CALLS": str(calls)})
    assert calls.read_text(encoding="utf-8") == "bad"
    third = _run_launcher(runtime_dir, uv=good_uv, retry=True, extra_env={"UV_CALLS": str(calls)})

    assert first.returncode == 42
    assert second.returncode == 42
    assert "Previous ARC MCP runtime install failed" in second.stderr
    assert third.returncode == 0
    assert calls.read_text(encoding="utf-8") == "badgoodgood"
