from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/release-arc.sh"


def _run(cmd: list[str], cwd: Path, **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        **kwargs,
    )


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    result = _run(["git", *args], cwd)
    assert result.returncode == 0, result.stderr
    return result


def _write_minimal_arc_repo(work: Path) -> None:
    (work / "plugins/arc/.codex-plugin").mkdir(parents=True)
    (work / "plugins/arc/.claude-plugin").mkdir(parents=True)
    (work / "packages/arc-mcp/src/arc_mcp").mkdir(parents=True)
    (work / "packages/arc-paper/src/arc_paper").mkdir(parents=True)
    (work / "packages/arc-paper/tests").mkdir(parents=True)

    for host in ("codex", "claude"):
        (work / f"plugins/arc/.{host}-plugin/plugin.json").write_text(
            json.dumps(
                {
                    "name": "arc",
                    "version": "0.1.0",
                    "description": "ARC",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    package_dependencies = {
        "arc-llm": [],
        "arc-paper": ['"arc-llm>=0.1,<0.2"'],
        "arc-domain": ['"arc-llm>=0.1,<0.2"', '"arc-paper>=0.1,<0.2"'],
        "arc-typeset": ['"arc-llm>=0.1,<0.2"'],
        "arc-mcp": [
            '"arc-domain>=0.1,<0.2"',
            '"arc-llm>=0.1,<0.2"',
            '"arc-paper>=0.1,<0.2"',
            '"arc-typeset>=0.1,<0.2"',
        ],
    }
    for package, dependencies in package_dependencies.items():
        package_dir = work / "packages" / package
        package_dir.mkdir(parents=True, exist_ok=True)
        deps = ",\n  ".join(dependencies)
        deps_block = f"dependencies = [\n  {deps}\n]\n" if dependencies else ""
        (package_dir / "pyproject.toml").write_text(
            "\n".join(
                [
                    "[build-system]",
                    'requires = ["hatchling>=1.25"]',
                    'build-backend = "hatchling.build"',
                    "",
                    "[project]",
                    f'name = "{package}"',
                    'version = "0.1.0"',
                    'requires-python = ">=3.11"',
                    deps_block.rstrip(),
                    "",
                ]
            ),
            encoding="utf-8",
        )

    (work / "packages/arc-mcp/src/arc_mcp/__init__.py").write_text(
        '__version__ = "0.1.0"\n',
        encoding="utf-8",
    )
    (work / "packages/arc-paper/src/arc_paper/__init__.py").write_text(
        '__version__ = "0.1.0"\n',
        encoding="utf-8",
    )
    (work / "packages/arc-paper/tests/test_import.py").write_text(
        'from arc_paper import __version__\n\n\ndef test_version():\n    assert __version__ == "0.1.0"\n',
        encoding="utf-8",
    )
    (work / "packages/arc-paper/tests/test_package_metadata.py").write_text(
        'EXPECTED = ["arc-llm>=0.1,<0.2", "arc-paper>=0.1,<0.2"]\n',
        encoding="utf-8",
    )
    (work / "README.md").write_text("initial\n", encoding="utf-8")


def _init_release_repo(tmp_path: Path, *, push_feature: bool = True) -> tuple[Path, Path]:
    origin = tmp_path / "origin.git"
    work = tmp_path / "work"
    _git(tmp_path, "init", "--bare", str(origin))
    _git(tmp_path, "init", "-b", "main", str(work))
    _git(work, "config", "user.name", "Test User")
    _git(work, "config", "user.email", "test@example.com")
    _write_minimal_arc_repo(work)
    _git(work, "add", ".")
    _git(work, "commit", "-m", "initial")
    _git(work, "tag", "-a", "v0.1.0", "-m", "v0.1.0")
    _git(work, "remote", "add", "origin", str(origin))
    _git(work, "push", "-u", "origin", "main", "v0.1.0")
    _git(origin, "symbolic-ref", "HEAD", "refs/heads/main")

    (work / "README.md").write_text("initial\nfeature\n", encoding="utf-8")
    _git(work, "add", "README.md")
    _git(work, "commit", "-m", "feat: add release-worthy change")
    if push_feature:
        _git(work, "push", "origin", "main")
    return work, origin


def _replace(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new), encoding="utf-8")


def _commit_release_bump(work: Path, version: str = "0.2.0") -> None:
    _apply_release_bump(work, version)
    _git(work, "add", ".")
    _git(work, "commit", "-m", f"chore: release v{version}")


def _apply_release_bump(work: Path, version: str = "0.2.0") -> None:
    _replace(work / "plugins/arc/.codex-plugin/plugin.json", '"version": "0.1.0"', f'"version": "{version}"')
    _replace(work / "plugins/arc/.claude-plugin/plugin.json", '"version": "0.1.0"', f'"version": "{version}"')
    for pyproject in (work / "packages").glob("arc-*/pyproject.toml"):
        _replace(pyproject, 'version = "0.1.0"', f'version = "{version}"')
        text = pyproject.read_text(encoding="utf-8")
        text = text.replace(">=0.1,<0.2", ">=0.2,<0.3")
        pyproject.write_text(text, encoding="utf-8")
    _replace(work / "packages/arc-mcp/src/arc_mcp/__init__.py", '0.1.0', version)
    _replace(work / "packages/arc-paper/src/arc_paper/__init__.py", '0.1.0', version)
    _replace(work / "packages/arc-paper/tests/test_import.py", '0.1.0', version)
    _replace(work / "packages/arc-paper/tests/test_package_metadata.py", ">=0.1,<0.2", ">=0.2,<0.3")


def _run_script(
    work: Path,
    version: str = "0.2.0",
    *,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PATH"] = "/usr/bin:/bin"
    if extra_env:
        env.update(extra_env)
    return _run(
        [str(SCRIPT), version],
        work,
        input="\n" * 20,
        env=env,
    )


def test_release_script_bumps_versions_creates_one_tag_and_pushes_stable(tmp_path: Path) -> None:
    work, origin = _init_release_repo(tmp_path)

    result = _run_script(work)

    assert result.returncode == 0, result.stderr
    assert 'version = "0.2.0"' in (work / "packages/arc-mcp/pyproject.toml").read_text(encoding="utf-8")
    assert '"arc-llm>=0.2,<0.3"' in (work / "packages/arc-paper/pyproject.toml").read_text(encoding="utf-8")
    assert '__version__ = "0.2.0"' in (work / "packages/arc-mcp/src/arc_mcp/__init__.py").read_text(encoding="utf-8")
    assert 'assert __version__ == "0.2.0"' in (work / "packages/arc-paper/tests/test_import.py").read_text(encoding="utf-8")
    assert "arc-llm>=0.2,<0.3" in (work / "packages/arc-paper/tests/test_package_metadata.py").read_text(encoding="utf-8")
    assert json.loads((work / "plugins/arc/.codex-plugin/plugin.json").read_text(encoding="utf-8"))["version"] == "0.2.0"
    assert json.loads((work / "plugins/arc/.claude-plugin/plugin.json").read_text(encoding="utf-8"))["version"] == "0.2.0"

    refs = _git(origin, "show-ref").stdout
    assert "refs/heads/main" in refs
    assert "refs/heads/stable" in refs
    assert "refs/tags/v0.2.0" in refs
    assert "refs/tags/arc--v0.2.0" not in refs
    assert _git(work, "log", "-1", "--pretty=%s").stdout.strip() == "chore: release v0.2.0"

    dry_run_index = result.stdout.index("DRY RUN: git push --dry-run origin HEAD:main v0.2.0")
    push_index = result.stdout.index("RUN: git push origin HEAD:main v0.2.0")
    assert dry_run_index < push_index
    assert "arc--v0.2.0" not in result.stdout


def test_release_script_validates_claude_manifest_without_claude_tag(tmp_path: Path) -> None:
    work, _origin = _init_release_repo(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "claude-calls.log"
    fake_claude = fake_bin / "claude"
    fake_claude.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "printf '%s\\n' \"$*\" >> \"$CLAUDE_CALLS\"",
                "if [ \"$1 $2\" = 'plugin validate' ]; then exit 0; fi",
                "if [ \"$1 $2\" = 'plugin tag' ]; then exit 42; fi",
                "exit 64",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_claude.chmod(0o755)

    result = _run_script(
        work,
        extra_env={
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "CLAUDE_CALLS": str(calls),
        },
    )

    assert result.returncode == 0, result.stderr
    call_text = calls.read_text(encoding="utf-8")
    assert "plugin validate plugins/arc" in call_text
    assert "plugin tag" not in call_text


def test_release_script_resumes_after_committed_version_bump(tmp_path: Path) -> None:
    work, origin = _init_release_repo(tmp_path)
    _commit_release_bump(work)
    _git(work, "push", "origin", "main")

    result = _run_script(work)

    assert result.returncode == 0, result.stderr
    refs = _git(origin, "show-ref").stdout
    assert "refs/tags/v0.2.0" in refs
    assert "refs/heads/stable" in refs
    assert "Version files already at 0.2.0; continuing without a bump commit." in result.stdout
    assert "RUN: git commit -m chore: release v0.2.0" not in result.stdout


def test_release_script_resumes_after_uncommitted_version_bump(tmp_path: Path) -> None:
    work, origin = _init_release_repo(tmp_path)
    _apply_release_bump(work)

    result = _run_script(work)

    assert result.returncode == 0, result.stderr
    refs = _git(origin, "show-ref").stdout
    assert "refs/tags/v0.2.0" in refs
    assert "refs/heads/stable" in refs
    assert "Worktree has only release version-file changes; continuing resume." in result.stdout
    assert _git(work, "log", "-1", "--pretty=%s").stdout.strip() == "chore: release v0.2.0"


def test_release_script_rejects_tracked_generated_python_cache(tmp_path: Path) -> None:
    work, _origin = _init_release_repo(tmp_path)
    cache_dir = work / "packages/arc-llm/src/arc_llm/__pycache__"
    cache_dir.mkdir(parents=True)
    (cache_dir / "bad.cpython-311.pyc").write_bytes(b"cache")
    _git(work, "add", "-f", str(cache_dir / "bad.cpython-311.pyc"))
    _git(work, "commit", "-m", "chore: accidentally track pyc")

    result = _run_script(work)

    assert result.returncode != 0
    assert "Generated Python cache files are tracked" in result.stderr


def test_release_script_resumes_after_local_tag_at_head(tmp_path: Path) -> None:
    work, origin = _init_release_repo(tmp_path)
    _commit_release_bump(work)
    _git(work, "push", "origin", "main")
    _git(work, "tag", "-a", "v0.2.0", "-m", "v0.2.0")

    result = _run_script(work)

    assert result.returncode == 0, result.stderr
    refs = _git(origin, "show-ref").stdout
    assert "refs/tags/v0.2.0" in refs
    assert "refs/heads/stable" in refs
    assert "Reusing existing local tag v0.2.0 at HEAD." in result.stdout
    assert "RUN: git tag -a v0.2.0 -m v0.2.0" not in result.stdout


def test_release_script_rejects_dirty_worktree(tmp_path: Path) -> None:
    work, _origin = _init_release_repo(tmp_path)
    (work / "README.md").write_text("dirty\n", encoding="utf-8")

    result = _run_script(work)

    assert result.returncode != 0
    assert "Worktree is dirty" in result.stderr


def test_release_script_rejects_no_commits_since_latest_release_tag(tmp_path: Path) -> None:
    work, _origin = _init_release_repo(tmp_path)
    _git(work, "tag", "-a", "v0.1.1", "-m", "v0.1.1")

    result = _run_script(work)

    assert result.returncode != 0
    assert "No committed changes since v0.1.1" in result.stderr


def test_release_script_rejects_branch_behind_upstream(tmp_path: Path) -> None:
    work, origin = _init_release_repo(tmp_path)
    other = tmp_path / "other"
    _git(tmp_path, "clone", str(origin), str(other))
    _git(other, "config", "user.name", "Test User")
    _git(other, "config", "user.email", "test@example.com")
    (other / "README.md").write_text("remote change\n", encoding="utf-8")
    _git(other, "add", "README.md")
    _git(other, "commit", "-m", "feat: remote change")
    _git(other, "push", "origin", "main")

    result = _run_script(work)

    assert result.returncode != 0
    assert "Branch is behind upstream" in result.stderr


def test_release_script_rejects_invalid_version(tmp_path: Path) -> None:
    work, _origin = _init_release_repo(tmp_path)

    result = _run_script(work, "v0.2")

    assert result.returncode != 0
    assert "Usage: release-arc.sh VERSION" in result.stderr
