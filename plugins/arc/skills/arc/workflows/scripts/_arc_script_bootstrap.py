from __future__ import annotations

import os
import sys
from pathlib import Path


ARC_PACKAGES = ("arc-llm", "arc-paper", "arc-domain", "arc-typeset", "arc-mcp")


def bootstrap_arc_pythonpath() -> None:
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
    raise RuntimeError(
        "Cannot import ARC internal module `arc_llm`. This does NOT mean `arc-llm` "
        "should be installed from PyPI. ARC tools are provided by the ARC MCP/plugin "
        "launcher and its bundled runtime. Run this workflow with the ARC runtime, "
        "or set ARC_MCP_REPO_ROOT/ARC_REPO_ROOT/PYTHONPATH to a checkout containing "
        "packages/arc-llm/src."
    )


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
