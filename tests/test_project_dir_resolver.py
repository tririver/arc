from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "plugins/arc/skills/arc/workflows/scripts/resolve-project-dir.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_resolver_places_generated_project_dir_directly_under_run_root(tmp_path: Path) -> None:
    result = _run("--name", "0911.3380", "--run-root", str(tmp_path), "--json")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["data"]["run_root"] == str(tmp_path.resolve())
    assert payload["data"]["project_dir_name"] == "0911.3380"
    assert payload["data"]["project_dir"] == str(tmp_path.resolve() / "0911.3380")


def test_resolver_rejects_arc_output_wrapped_names(tmp_path: Path) -> None:
    result = _run("--name", "arc-output/0911.3380", "--run-root", str(tmp_path), "--json")

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["errors"][0]["code"] == "invalid_project_dir_name"
    assert "arc-output" in payload["errors"][0]["message"]


def test_resolver_rejects_path_like_names(tmp_path: Path) -> None:
    for name in ["/tmp/0911.3380", "../0911.3380", "nested/0911.3380", "nested\\0911.3380"]:
        result = _run("--name", name, "--run-root", str(tmp_path), "--json")

        assert result.returncode == 2
        payload = json.loads(result.stdout)
        assert payload["ok"] is False
        assert payload["errors"][0]["code"] == "invalid_project_dir_name"


def test_resolver_rejects_claude_internal_run_root(tmp_path: Path) -> None:
    host_root = tmp_path / ".claude" / "projects" / "arc-deepseek-test"
    result = _run("--name", "0911.3380", "--run-root", str(host_root), "--json")

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["errors"][0]["code"] == "invalid_run_root"
    assert ".claude" in payload["errors"][0]["message"]
