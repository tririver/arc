import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "plugins/arc/skills/arc/workflows/scripts"
BOOTSTRAP = SCRIPTS / "_arc_script_bootstrap.py"
VERIFIER = SCRIPTS / "verify-source-runtime.py"


def _make_fake_repo(tmp_path: Path, label: str) -> Path:
    root = tmp_path / label
    scripts = root / "plugins/arc/skills/arc/workflows/scripts"
    scripts.mkdir(parents=True)
    shutil.copyfile(BOOTSTRAP, scripts / BOOTSTRAP.name)
    for package, module in (
        ("arc-llm", "arc_llm"),
        ("arc-paper", "arc_paper"),
        ("arc-domain", "arc_domain"),
        ("arc-typeset", "arc_typeset"),
        ("arc-companion", "arc_companion"),
        ("arc-mcp", "arc_mcp"),
    ):
        module_dir = root / "packages" / package / "src" / module
        module_dir.mkdir(parents=True)
        (module_dir / "__init__.py").write_text(
            f"ORIGIN = {label!r}\n", encoding="utf-8"
        )
    return root


def _strict_bootstrap_code(scripts: Path) -> str:
    return (
        "import sys; "
        f"sys.path.insert(0, {str(scripts)!r}); "
        "from _arc_script_bootstrap import bootstrap_arc_pythonpath; "
        "bootstrap_arc_pythonpath(); "
        "import arc_llm; "
        "print(arc_llm.__file__)"
    )


def test_strict_bootstrap_prefers_required_checkout_over_installed_arc(tmp_path):
    repo = _make_fake_repo(tmp_path, "required")
    installed = tmp_path / "site-packages/arc_llm"
    installed.mkdir(parents=True)
    (installed / "__init__.py").write_text("ORIGIN = 'installed'\n", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "ARC_REQUIRE_REPO_ROOT": str(repo),
            "PYTHONPATH": str(installed.parent),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _strict_bootstrap_code(
                repo / "plugins/arc/skills/arc/workflows/scripts"
            ),
        ],
        env=env,
        cwd=tmp_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert str(repo / "packages/arc-llm/src/arc_llm") in result.stdout
    assert str(installed) not in result.stdout


def test_strict_bootstrap_rejects_preloaded_arc_from_outside_checkout(tmp_path):
    repo = _make_fake_repo(tmp_path, "required")
    installed = tmp_path / "site-packages/arc_llm"
    installed.mkdir(parents=True)
    (installed / "__init__.py").write_text("ORIGIN = 'installed'\n", encoding="utf-8")
    scripts = repo / "plugins/arc/skills/arc/workflows/scripts"
    code = (
        "import arc_llm, sys; "
        f"sys.path.insert(0, {str(scripts)!r}); "
        "from _arc_script_bootstrap import bootstrap_arc_pythonpath; "
        "bootstrap_arc_pythonpath()"
    )
    env = os.environ.copy()
    env.update(
        {
            "ARC_REQUIRE_REPO_ROOT": str(repo),
            "PYTHONPATH": str(installed.parent),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=tmp_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "already loaded outside the required checkout" in result.stderr
    assert str(installed) in result.stderr


def test_strict_bootstrap_rejects_bootstrap_from_another_checkout(tmp_path):
    required = _make_fake_repo(tmp_path, "required")
    other = _make_fake_repo(tmp_path, "other")
    env = os.environ.copy()
    env.update(
        {
            "ARC_REQUIRE_REPO_ROOT": str(required),
            "PYTHONPATH": "",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _strict_bootstrap_code(
                other / "plugins/arc/skills/arc/workflows/scripts"
            ),
        ],
        env=env,
        cwd=tmp_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "bootstrap loaded outside the required checkout" in result.stderr


def test_source_runtime_verifier_records_current_checkout(tmp_path):
    output = tmp_path / "source-provenance.json"
    python = ROOT / "packages/arc-paper/.venv/bin/python"
    result = subprocess.run(
        [
            str(python),
            str(VERIFIER),
            "--repo-root",
            str(ROOT),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert json.loads(result.stdout) == record
    assert record["schema_version"] == "arc.workflow.source_provenance.v1"
    assert Path(record["repo_root"]) == ROOT
    assert record["git"]["available"] is True
    assert len(record["git"]["head"]) == 40
    assert set(record["modules"]) == {
        "arc_llm",
        "arc_paper",
        "arc_domain",
        "arc_typeset",
        "arc_mcp",
    }
    for details in record["modules"].values():
        assert Path(details["file"]).is_relative_to(ROOT / "packages")
    hashed_paths = {item["path"] for item in record["workflow_files"]}
    assert "plugins/arc/skills/arc/workflows/ideas.md" in hashed_paths
    assert (
        "plugins/arc/skills/arc/workflows/scripts/verify-source-runtime.py"
        in hashed_paths
    )
