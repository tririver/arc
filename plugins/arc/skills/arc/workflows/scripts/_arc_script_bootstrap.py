from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path


ARC_PACKAGE_MODULES = (
    ("arc-jobs", "arc_jobs"),
    ("arc-llm", "arc_llm"),
    ("arc-paper", "arc_paper"),
    ("arc-domain", "arc_domain"),
    ("arc-typeset", "arc_typeset"),
    ("arc-companion", "arc_companion"),
)
ARC_PACKAGES = tuple(package for package, _module in ARC_PACKAGE_MODULES)
ARC_REQUIRE_REPO_ROOT = "ARC_REQUIRE_REPO_ROOT"


def bootstrap_arc_pythonpath() -> None:
    if required_root := os.environ.get(ARC_REQUIRE_REPO_ROOT):
        _bootstrap_required_repo_root(Path(required_root).expanduser())
        return
    for root in _candidate_roots():
        if _bootstrap_checkout(root):
            return
    runtime_site_packages = _active_runtime_site_packages()
    for _package, module_name in ARC_PACKAGE_MODULES:
        _reject_loaded_module_outside(module_name, runtime_site_packages)
    site_string = str(runtime_site_packages)
    sys.path[:] = [entry for entry in sys.path if _resolved_path(entry) != site_string]
    sys.path.insert(0, site_string)
    importlib.invalidate_caches()
    try:
        arc_llm = importlib.import_module("arc_llm")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"The active ARC core runtime does not contain `arc_llm`: {runtime_site_packages}"
        ) from exc
    _assert_module_in_source(arc_llm, runtime_site_packages)


def _bootstrap_checkout(root: Path) -> bool:
    package_sources = [root / "packages" / package / "src" for package in ARC_PACKAGES]
    if not all(source.is_dir() for source in package_sources):
        return False
    for package, module_name in ARC_PACKAGE_MODULES:
        _reject_loaded_module_outside(module_name, root / "packages" / package / "src")
    source_strings = [str(source.resolve()) for source in package_sources]
    source_set = set(source_strings)
    sys.path[:] = [entry for entry in sys.path if _resolved_path(entry) not in source_set]
    sys.path[:0] = source_strings
    importlib.invalidate_caches()
    try:
        arc_llm = importlib.import_module("arc_llm")
    except ModuleNotFoundError:
        return False
    _assert_module_in_source(arc_llm, root / "packages" / "arc-llm" / "src")
    return True


def _reject_loaded_module_outside(module_name: str, expected_root: Path) -> None:
    module = sys.modules.get(module_name)
    if module is None:
        return
    try:
        _assert_module_in_source(module, expected_root)
    except RuntimeError as exc:
        raise RuntimeError(
            f"ARC bootstrap rejected `{module_name}` already loaded outside the active "
            f"ARC source/runtime rooted at {expected_root}. Restart the process without "
            "the unrelated installed ARC package."
        ) from exc


def _active_runtime_site_packages() -> Path:
    skill_root = Path(__file__).resolve().parents[2]
    launcher = skill_root / "scripts" / "arc-runtime"
    if not launcher.is_file():
        raise RuntimeError(f"Cannot locate the ARC Skill runtime launcher: {launcher}")
    completed = subprocess.run(
        [str(launcher), "doctor", "--profile", "core"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    fields: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            fields[key.strip()] = value.strip()
    if (
        completed.returncode != 0
        or fields.get("profile") != "core"
        or fields.get("status") != "ready"
        or not fields.get("runtime")
        or not fields.get("venv")
        or not fields.get("fingerprint")
        or not fields.get("constraints_sha256")
        or not fields.get("ready_file")
    ):
        detail = completed.stderr.strip() or completed.stdout.strip() or "runtime not installed"
        raise RuntimeError(
            "The runtime pinned by this ARC Skill is not ready. Run "
            f"`{launcher} setup --profile core`, then retry. Doctor reported: {detail}"
        )
    runtime = Path(fields["runtime"]).expanduser()
    try:
        runtime = runtime.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"ARC doctor returned an inaccessible runtime: {runtime}") from exc
    venv = Path(fields["venv"]).expanduser()
    ready_file = Path(fields["ready_file"]).expanduser()
    try:
        venv = venv.resolve(strict=True)
        ready_file = ready_file.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("ARC doctor returned inaccessible runtime metadata paths") from exc
    if not _is_relative_to(venv, runtime) or not _is_relative_to(ready_file, runtime):
        raise RuntimeError("ARC doctor returned runtime metadata outside its runtime directory")
    _assert_runtime_python_compatible(venv, launcher=launcher)
    marker_fields: dict[str, str] = {}
    for line in ready_file.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            marker_fields[key.strip()] = value.strip()
    if (
        marker_fields.get("profile") != "core"
        or marker_fields.get("runtime_fingerprint") != fields["fingerprint"]
        or marker_fields.get("constraints_sha256") != fields["constraints_sha256"]
    ):
        raise RuntimeError("ARC runtime ready marker does not match this Skill's identity")
    candidates = [venv / "Lib" / "site-packages"]
    candidates.extend((venv / "lib").glob("python*/site-packages"))
    existing: list[Path] = []
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        resolved_candidate = candidate.resolve()
        if not _is_relative_to(resolved_candidate, venv):
            raise RuntimeError(
                "ARC runtime site-packages resolves outside its selected venv: "
                f"{candidate} -> {resolved_candidate}"
            )
        existing.append(resolved_candidate)
    if len(existing) != 1:
        formatted = ", ".join(str(candidate) for candidate in existing) or "none"
        raise RuntimeError(
            f"ARC runtime must contain exactly one site-packages directory; found {formatted}"
        )
    return existing[0]


def _assert_runtime_python_compatible(venv: Path, *, launcher: Path) -> None:
    candidates = (venv / "Scripts" / "python.exe", venv / "bin" / "python")
    runtime_python = next(
        (candidate for candidate in candidates if candidate.is_file() and os.access(candidate, os.X_OK)),
        None,
    )
    if runtime_python is None:
        raise RuntimeError(f"ARC runtime has no executable Python interpreter under {venv}")
    probe = (
        "import json,sys,sysconfig;"
        "print(json.dumps({'implementation':sys.implementation.name,"
        "'major':sys.version_info.major,'minor':sys.version_info.minor,"
        "'cache_tag':sys.implementation.cache_tag,"
        "'soabi':sysconfig.get_config_var('SOABI')}))"
    )
    completed = subprocess.run(
        [str(runtime_python), "-I", "-c", probe],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    try:
        runtime_abi = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
        raise RuntimeError(
            f"Cannot inspect ARC runtime Python {runtime_python}: {detail}"
        ) from exc
    if completed.returncode != 0 or not isinstance(runtime_abi, dict):
        detail = completed.stderr.strip() or completed.stdout.strip() or "probe failed"
        raise RuntimeError(f"Cannot inspect ARC runtime Python {runtime_python}: {detail}")

    current_abi = {
        "implementation": sys.implementation.name,
        "major": sys.version_info.major,
        "minor": sys.version_info.minor,
        "cache_tag": sys.implementation.cache_tag,
        "soabi": sysconfig.get_config_var("SOABI"),
    }
    required_fields = ("implementation", "major", "minor", "cache_tag")
    compatible = all(runtime_abi.get(key) == current_abi[key] for key in required_fields)
    if runtime_abi.get("soabi") and current_abi["soabi"]:
        compatible = compatible and runtime_abi["soabi"] == current_abi["soabi"]
    if compatible:
        return

    current_label = _format_python_abi(current_abi)
    runtime_label = _format_python_abi(runtime_abi)
    raise RuntimeError(
        "The ARC runtime Python is not ABI-compatible with the interpreter running "
        f"this workflow (current: {current_label}; runtime: {runtime_label}). Run the "
        f"corresponding ARC command through `{launcher}`, or invoke this workflow with "
        f"`{runtime_python}`, instead of importing the runtime into the current process."
    )


def _format_python_abi(identity: dict[str, object]) -> str:
    return (
        f"{identity.get('implementation')} "
        f"{identity.get('major')}.{identity.get('minor')} "
        f"cache_tag={identity.get('cache_tag')} soabi={identity.get('soabi')}"
    )


def _bootstrap_required_repo_root(root: Path) -> None:
    try:
        root = root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(
            f"{ARC_REQUIRE_REPO_ROOT} does not identify an accessible ARC checkout: "
            f"{root}"
        ) from exc
    if not root.is_dir():
        raise RuntimeError(f"{ARC_REQUIRE_REPO_ROOT} is not a directory: {root}")

    expected_bootstrap = (
        root
        / "plugins"
        / "arc"
        / "skills"
        / "arc"
        / "workflows"
        / "scripts"
        / "_arc_script_bootstrap.py"
    )
    try:
        expected_bootstrap = expected_bootstrap.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(
            f"{ARC_REQUIRE_REPO_ROOT} is not a complete ARC checkout; missing "
            f"{expected_bootstrap}"
        ) from exc
    if Path(__file__).resolve() != expected_bootstrap:
        raise RuntimeError(
            "Strict ARC source mode rejected a bootstrap loaded outside the required "
            f"checkout. Loaded {Path(__file__).resolve()}, expected {expected_bootstrap}."
        )

    package_sources: list[tuple[str, str, Path]] = []
    missing_sources: list[Path] = []
    for package, module_name in ARC_PACKAGE_MODULES:
        src = root / "packages" / package / "src"
        module_dir = src / module_name
        if not module_dir.is_dir():
            missing_sources.append(module_dir)
        package_sources.append((package, module_name, src))
    if missing_sources:
        formatted = "\n".join(f"  - {path}" for path in missing_sources)
        raise RuntimeError(
            f"{ARC_REQUIRE_REPO_ROOT} is not a complete ARC source checkout. "
            f"Missing package sources:\n{formatted}"
        )

    _reject_arc_modules_loaded_outside(package_sources)

    # Insert the checkout as one ordered block. Remove equivalent entries first so
    # site-packages and plugin runtimes can never take precedence in strict mode.
    source_strings = [str(src.resolve()) for _package, _module, src in package_sources]
    source_set = set(source_strings)
    sys.path[:] = [entry for entry in sys.path if _resolved_path(entry) not in source_set]
    sys.path[:0] = source_strings
    importlib.invalidate_caches()

    try:
        arc_llm = importlib.import_module("arc_llm")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Cannot import `arc_llm` from required ARC checkout {root}."
        ) from exc
    _assert_module_in_source(arc_llm, root / "packages" / "arc-llm" / "src")


def _reject_arc_modules_loaded_outside(
    package_sources: list[tuple[str, str, Path]],
) -> None:
    for _package, module_name, src in package_sources:
        module = sys.modules.get(module_name)
        if module is not None:
            _assert_module_in_source(module, src)


def _assert_module_in_source(module: object, source_root: Path) -> None:
    module_name = getattr(module, "__name__", repr(module))
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError(
            f"Strict ARC source mode cannot verify `{module_name}` because it has no "
            "filesystem origin."
        )
    resolved_file = Path(module_file).resolve()
    resolved_source = source_root.resolve()
    if not _is_relative_to(resolved_file, resolved_source):
        raise RuntimeError(
            "Strict ARC source mode rejected an ARC module already loaded outside "
            f"the required checkout: `{module_name}` came from {resolved_file}; "
            f"expected a path under {resolved_source}. Restart the process without "
            "the installed or cached ARC module."
        )


def _resolved_path(value: str) -> str:
    if not value:
        return value
    try:
        return str(Path(value).resolve())
    except (OSError, RuntimeError):
        return value


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _candidate_roots() -> list[Path]:
    roots: list[Path] = []
    if value := os.environ.get("ARC_INSTALL_REPO_ROOT"):
        roots.append(Path(value).expanduser())
    if containing_checkout := _checkout_containing_bootstrap():
        roots.append(containing_checkout)
    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            resolved = str(root.resolve())
        except OSError:
            resolved = str(root)
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(root)
    return deduped


def _checkout_containing_bootstrap() -> Path | None:
    here = Path(__file__).resolve()
    relative_bootstrap = Path(
        "plugins/arc/skills/arc/workflows/scripts/_arc_script_bootstrap.py"
    )
    for candidate in here.parents:
        expected = candidate / relative_bootstrap
        try:
            if expected.resolve(strict=True) == here:
                return candidate
        except (OSError, RuntimeError):
            continue
    return None
