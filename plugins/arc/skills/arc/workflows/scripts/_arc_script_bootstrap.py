from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


ARC_PACKAGE_MODULES = (
    ("arc-llm", "arc_llm"),
    ("arc-paper", "arc_paper"),
    ("arc-domain", "arc_domain"),
    ("arc-typeset", "arc_typeset"),
    ("arc-mcp", "arc_mcp"),
)
ARC_PACKAGES = tuple(package for package, _module in ARC_PACKAGE_MODULES)
ARC_REQUIRE_REPO_ROOT = "ARC_REQUIRE_REPO_ROOT"


def bootstrap_arc_pythonpath() -> None:
    if required_root := os.environ.get(ARC_REQUIRE_REPO_ROOT):
        _bootstrap_required_repo_root(Path(required_root).expanduser())
        return
    if _can_import_arc_llm():
        return
    for root in _candidate_roots():
        added = False
        for package in ARC_PACKAGES:
            src = root / "packages" / package / "src"
            if src.is_dir() and str(src) not in sys.path:
                sys.path.insert(0, str(src))
                added = True
        if added and _can_import_arc_llm():
            return
    for site_packages in _candidate_runtime_site_packages():
        if site_packages.is_dir() and str(site_packages) not in sys.path:
            sys.path.insert(0, str(site_packages))
            if _can_import_arc_llm():
                return
    searched_roots = "\n".join(f"  - {root}" for root in _candidate_roots()) or "  - (none)"
    searched_runtimes = "\n".join(f"  - {path}" for path in _candidate_runtime_site_packages()) or "  - (none)"
    raise RuntimeError(
        "Cannot import ARC internal module `arc_llm`. This does NOT mean `arc-llm` "
        "should be installed from PyPI. ARC tools are provided by the ARC MCP/plugin "
        "launcher and its bundled runtime. Run this workflow with the ARC runtime, "
        "or set ARC_MCP_REPO_ROOT/ARC_REPO_ROOT/PYTHONPATH to a checkout containing "
        "packages/arc-llm/src.\n"
        f"Searched ARC roots:\n{searched_roots}\n"
        f"Searched ARC runtime site-packages:\n{searched_runtimes}"
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


def _can_import_arc_llm() -> bool:
    try:
        __import__("arc_llm")
        return True
    except ModuleNotFoundError:
        return False


def _candidate_roots() -> list[Path]:
    roots: list[Path] = []
    for key in ("ARC_REPO_ROOT", "ARC_MCP_REPO_ROOT", "ARC_PLUGIN_ROOT"):
        value = os.environ.get(key)
        if value:
            roots.append(Path(value).expanduser())
    here = Path(__file__).resolve()
    roots.extend(here.parents)
    home = Path.home()
    roots.extend(
        [
            home / ".claude" / "plugins" / "marketplaces" / "arc",
            home / ".codex" / "plugins" / "marketplaces" / "arc",
        ]
    )
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


def _candidate_runtime_site_packages() -> list[Path]:
    roots: list[Path] = []
    if value := os.environ.get("ARC_MCP_RUNTIME_DIR"):
        roots.append(Path(value).expanduser())
    cache_home = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
    roots.extend((cache_home / "arc" / "arc-mcp-runtime").glob("v*/**/venv"))
    site_packages: list[Path] = []
    for root in roots:
        site_packages.extend(root.glob("lib/python*/site-packages"))
    return site_packages
