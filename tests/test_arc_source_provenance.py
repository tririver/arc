import json
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "plugins/arc/skills/arc/workflows/scripts"
BOOTSTRAP = SCRIPTS / "_arc_script_bootstrap.py"
VERIFIER = SCRIPTS / "verify-source-runtime.py"


def _load_verifier_module():
    old_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location("arc_source_verifier_test", VERIFIER)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))
        sys.dont_write_bytecode = old_dont_write_bytecode


def _make_fake_repo(tmp_path: Path, label: str) -> Path:
    root = tmp_path / label
    scripts = root / "plugins/arc/skills/arc/workflows/scripts"
    scripts.mkdir(parents=True)
    shutil.copyfile(BOOTSTRAP, scripts / BOOTSTRAP.name)
    for package, module in (
        ("arc-jobs", "arc_jobs"),
        ("arc-llm", "arc_llm"),
        ("arc-paper", "arc_paper"),
        ("arc-domain", "arc_domain"),
        ("arc-typeset", "arc_typeset"),
        ("arc-companion", "arc_companion"),
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


def _make_installed_skill_and_runtime(tmp_path: Path) -> tuple[Path, Path, Path]:
    skill = tmp_path / "installed-skill"
    scripts = skill / "workflows" / "scripts"
    scripts.mkdir(parents=True)
    shutil.copyfile(BOOTSTRAP, scripts / BOOTSTRAP.name)
    launcher = skill / "scripts" / "arc-runtime"
    launcher.parent.mkdir(parents=True)
    launcher.write_text(
        "#!/bin/sh\n"
        "printf 'profile=core\\nfingerprint=test\\nconstraints_sha256=constraints\\nruntime=%s\\nvenv=%s/venv\\nready_file=%s/install.ok\\nstatus=ready\\n' "
        "\"$FAKE_ARC_RUNTIME\" \"$FAKE_ARC_RUNTIME\" \"$FAKE_ARC_RUNTIME\"\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    runtime = tmp_path / "runtime"
    venv = runtime / "venv"
    module = (
        venv
        / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/arc_llm"
    )
    module.mkdir(parents=True)
    runtime_python = venv / "bin/python"
    runtime_python.parent.mkdir(parents=True, exist_ok=True)
    runtime_python.symlink_to(Path(sys.executable).resolve())
    (runtime / "install.ok").write_text(
        "profile=core\nruntime_fingerprint=test\nconstraints_sha256=constraints\n",
        encoding="utf-8",
    )
    (module / "__init__.py").write_text("ORIGIN = 'pinned-runtime'\n", encoding="utf-8")
    return scripts, runtime, module


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


def test_installed_skill_bootstrap_uses_only_its_doctor_selected_runtime(tmp_path):
    scripts, runtime, module = _make_installed_skill_and_runtime(tmp_path)
    unrelated = (
        tmp_path
        / "other-runtimes/v4/core/wrong/venv/lib/python3.11/site-packages/arc_llm"
    )
    unrelated.mkdir(parents=True)
    (unrelated / "__init__.py").write_text("ORIGIN = 'unrelated'\n", encoding="utf-8")
    env = {
        **os.environ,
        "ARC_INSTALL_SOURCE": "git",
        "ARC_RUNTIME_HOME": str(tmp_path / "other-runtimes"),
        "FAKE_ARC_RUNTIME": str(runtime),
        "HOME": str(tmp_path / "home"),
        "PYTHONPATH": str(scripts),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    code = (
        "from _arc_script_bootstrap import bootstrap_arc_pythonpath; "
        "bootstrap_arc_pythonpath(); import arc_llm; print(arc_llm.__file__)"
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

    assert result.returncode == 0, result.stderr
    assert str(module) in result.stdout
    assert str(unrelated) not in result.stdout


def test_installed_skill_bootstrap_rejects_preloaded_unrelated_runtime(tmp_path):
    scripts, runtime, _module = _make_installed_skill_and_runtime(tmp_path)
    unrelated = tmp_path / "unrelated/site-packages/arc_llm"
    unrelated.mkdir(parents=True)
    (unrelated / "__init__.py").write_text("ORIGIN = 'unrelated'\n", encoding="utf-8")
    env = {
        **os.environ,
        "ARC_INSTALL_SOURCE": "git",
        "FAKE_ARC_RUNTIME": str(runtime),
        "HOME": str(tmp_path / "home"),
        "PYTHONPATH": os.pathsep.join([str(unrelated.parent), str(scripts)]),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    code = (
        "import arc_llm; from _arc_script_bootstrap import bootstrap_arc_pythonpath; "
        "bootstrap_arc_pythonpath()"
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
    assert "already loaded outside the active ARC source/runtime" in result.stderr
    assert str(unrelated) in result.stderr


def test_installed_skill_ignores_unrelated_home_marketplace_checkout(tmp_path):
    scripts, runtime, module = _make_installed_skill_and_runtime(tmp_path)
    home = tmp_path / "home"
    unrelated_repo = _make_fake_repo(
        home / ".claude/plugins/marketplaces", "arc"
    )
    env = {
        **os.environ,
        "ARC_INSTALL_SOURCE": "git",
        "FAKE_ARC_RUNTIME": str(runtime),
        "HOME": str(home),
        "PYTHONPATH": str(scripts),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    code = (
        "from _arc_script_bootstrap import bootstrap_arc_pythonpath; "
        "bootstrap_arc_pythonpath(); import arc_llm; print(arc_llm.__file__)"
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

    assert result.returncode == 0, result.stderr
    assert str(module) in result.stdout
    assert str(unrelated_repo / "packages/arc-llm/src/arc_llm") not in result.stdout


def test_installed_skill_rejects_any_preloaded_foreign_arc_package(tmp_path):
    scripts, runtime, _module = _make_installed_skill_and_runtime(tmp_path)
    unrelated = tmp_path / "unrelated/site-packages/arc_paper"
    unrelated.mkdir(parents=True)
    (unrelated / "__init__.py").write_text("ORIGIN = 'unrelated'\n", encoding="utf-8")
    env = {
        **os.environ,
        "FAKE_ARC_RUNTIME": str(runtime),
        "HOME": str(tmp_path / "home"),
        "PYTHONPATH": os.pathsep.join([str(unrelated.parent), str(scripts)]),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    code = (
        "import arc_paper; from _arc_script_bootstrap import bootstrap_arc_pythonpath; "
        "bootstrap_arc_pythonpath()"
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
    assert "`arc_paper` already loaded outside" in result.stderr
    assert str(unrelated) in result.stderr


def test_installed_skill_rejects_incompatible_runtime_python(tmp_path):
    scripts, runtime, _module = _make_installed_skill_and_runtime(tmp_path)
    runtime_python = runtime / "venv/bin/python"
    runtime_python.unlink()
    runtime_python.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' "
        "'{\"implementation\":\"cpython\",\"major\":9,\"minor\":9,"
        "\"cache_tag\":\"cpython-99\",\"soabi\":\"cpython-99-test\"}'\n",
        encoding="utf-8",
    )
    runtime_python.chmod(0o755)
    env = {
        **os.environ,
        "FAKE_ARC_RUNTIME": str(runtime),
        "HOME": str(tmp_path / "home"),
        "PYTHONPATH": str(scripts),
        "PYTHONDWRITEBYTECODE": "1",
    }

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from _arc_script_bootstrap import bootstrap_arc_pythonpath; "
            "bootstrap_arc_pythonpath()",
        ],
        env=env,
        cwd=tmp_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "not ABI-compatible" in result.stderr
    assert str(runtime_python) in result.stderr


def test_installed_skill_rejects_site_packages_symlink_outside_venv(tmp_path):
    scripts, runtime, module = _make_installed_skill_and_runtime(tmp_path)
    site_packages = module.parent
    shutil.rmtree(site_packages)
    outside = tmp_path / "outside-site-packages"
    outside.mkdir()
    (outside / "arc_llm").mkdir()
    (outside / "arc_llm/__init__.py").write_text("ORIGIN = 'outside'\n")
    site_packages.symlink_to(outside, target_is_directory=True)
    env = {
        **os.environ,
        "FAKE_ARC_RUNTIME": str(runtime),
        "HOME": str(tmp_path / "home"),
        "PYTHONPATH": str(scripts),
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from _arc_script_bootstrap import bootstrap_arc_pythonpath; "
            "bootstrap_arc_pythonpath()",
        ],
        env=env,
        cwd=tmp_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "site-packages resolves outside" in result.stderr
    assert str(outside) in result.stderr


def test_source_runtime_verifier_records_current_checkout(tmp_path):
    output = tmp_path / "source-provenance.json"
    verifier = _load_verifier_module()
    try:
        runtime_python = verifier._runtime_python()
    except RuntimeError as exc:
        pytest.skip(f"ARC core runtime is not installed: {exc}")
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
    assert Path(record["runtime"]["python_executable"]) == runtime_python.absolute()
    assert Path(record["runtime"]["sys_prefix"]) == runtime_python.parent.parent.absolute()
    assert Path(record["repo_root"]) == ROOT
    assert record["git"]["available"] is True
    assert len(record["git"]["head"]) == 40
    assert set(record["modules"]) == {
        "arc_llm",
        "arc_jobs",
        "arc_paper",
        "arc_domain",
        "arc_typeset",
        "arc_companion",
    }
    for details in record["modules"].values():
        assert Path(details["file"]).is_relative_to(ROOT / "packages")
    hashed_paths = {item["path"] for item in record["workflow_files"]}
    assert "plugins/arc/skills/arc/workflows/ideas.md" in hashed_paths
    assert (
        "plugins/arc/skills/arc/workflows/scripts/verify-source-runtime.py"
        in hashed_paths
    )


def test_source_verifier_reexecs_incompatible_system_python_into_runtime(monkeypatch, tmp_path):
    module = _load_verifier_module()
    runtime = tmp_path / "venv/bin/python"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("", encoding="utf-8")
    monkeypatch.setattr(module.sys, "prefix", "/usr")
    monkeypatch.setattr(module.sys, "base_prefix", "/usr")
    monkeypatch.setattr(module.sys, "version_info", (9, 9, 0))
    monkeypatch.setattr(module, "_runtime_python", lambda: runtime)
    called = {}

    def fake_execve(executable, argv, env):
        called.update(executable=executable, argv=argv, env=env)
        raise RuntimeError("exec intercepted")

    monkeypatch.setattr(module.os, "execve", fake_execve)
    with pytest.raises(RuntimeError, match="exec intercepted"):
        module._reexec_runtime(["--repo-root", str(ROOT)])
    assert called["executable"] == str(runtime)
    assert called["env"][module.RUNTIME_REEXEC_ENV] == "1"
    assert called["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    source = VERIFIER.read_text(encoding="utf-8")
    assert source.index("sys.dont_write_bytecode = True") < source.index("from _arc_script_bootstrap import")


def test_source_verifier_resolves_doctor_python_without_current_abi_probe(monkeypatch, tmp_path):
    module = _load_verifier_module()
    scripts, runtime, _runtime_module = _make_installed_skill_and_runtime(tmp_path)
    launcher = scripts.parents[1] / "scripts" / "arc-runtime"
    monkeypatch.setenv("FAKE_ARC_RUNTIME", str(runtime))
    monkeypatch.setattr(module.sys, "version_info", (9, 9, 0))

    selected = module._runtime_python(launcher=launcher)

    assert selected == (runtime / "venv/bin/python").absolute()
