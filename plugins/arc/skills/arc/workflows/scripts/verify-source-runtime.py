#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from _arc_script_bootstrap import (
    ARC_PACKAGE_MODULES,
    ARC_REQUIRE_REPO_ROOT,
    bootstrap_arc_pythonpath,
)


SCHEMA_VERSION = "arc.workflow.source_provenance.v1"
SKILL_RELATIVE_PATH = Path("plugins/arc/skills/arc")
HASHED_SUFFIXES = {".json", ".md", ".py"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that ARC packages and workflow files come from one required "
            "source checkout, then emit machine-readable provenance."
        )
    )
    parser.add_argument(
        "--repo-root",
        help=f"ARC checkout root (defaults to ${ARC_REQUIRE_REPO_ROOT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the JSON record to this path as well as stdout.",
    )
    parser.add_argument(
        "--file",
        dest="extra_files",
        action="append",
        default=[],
        help="Additional file under the checkout to include by SHA256; repeatable.",
    )
    return parser


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _git_provenance(root: Path) -> dict[str, Any]:
    head = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    diff = _git(root, "diff", "--binary", "HEAD", "--")
    if head.returncode != 0:
        return {
            "available": False,
            "error": head.stderr.decode("utf-8", errors="replace").strip(),
        }
    if status.returncode != 0 or diff.returncode != 0:
        error = (status.stderr or diff.stderr).decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Cannot capture ARC checkout working-tree state: {error}")
    status_text = status.stdout.decode("utf-8", errors="surrogateescape")
    return {
        "available": True,
        "head": head.stdout.decode("ascii", errors="replace").strip(),
        "dirty": bool(status_text),
        "status_porcelain": status_text.splitlines(),
        "status_sha256": _sha256_bytes(status.stdout),
        "diff_sha256": _sha256_bytes(diff.stdout),
    }


def _module_provenance(root: Path) -> dict[str, dict[str, str]]:
    modules: dict[str, dict[str, str]] = {}
    for package, module_name in ARC_PACKAGE_MODULES:
        source_root = (root / "packages" / package / "src").resolve()
        module = importlib.import_module(module_name)
        module_file_value = getattr(module, "__file__", None)
        if not module_file_value:
            raise RuntimeError(f"Cannot verify `{module_name}`: module has no __file__.")
        module_file = Path(module_file_value).resolve()
        if not _is_relative_to(module_file, source_root):
            raise RuntimeError(
                f"Strict ARC source verification failed: `{module_name}` came from "
                f"{module_file}, expected a path under {source_root}."
            )
        modules[module_name] = {
            "distribution": package,
            "file": str(module_file),
            "source_root": str(source_root),
        }
    return modules


def _workflow_files(root: Path, extra_files: Sequence[str]) -> list[dict[str, Any]]:
    skill_dir = (root / SKILL_RELATIVE_PATH).resolve()
    expected_script = (
        skill_dir / "workflows" / "scripts" / "verify-source-runtime.py"
    ).resolve()
    if Path(__file__).resolve() != expected_script:
        raise RuntimeError(
            "Source verifier itself is not running from the required ARC checkout: "
            f"loaded {Path(__file__).resolve()}, expected {expected_script}."
        )
    if not skill_dir.is_dir():
        raise RuntimeError(f"Required ARC skill directory is missing: {skill_dir}")

    files = {
        path.resolve()
        for path in skill_dir.rglob("*")
        if path.is_file() and path.suffix in HASHED_SUFFIXES
    }
    for value in extra_files:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve(strict=True)
        if not _is_relative_to(candidate, root):
            raise RuntimeError(f"Additional provenance file is outside repo root: {candidate}")
        if not candidate.is_file():
            raise RuntimeError(f"Additional provenance path is not a file: {candidate}")
        files.add(candidate)

    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(files)
    ]


def build_provenance(root: Path, extra_files: Sequence[str]) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=True)
    os.environ[ARC_REQUIRE_REPO_ROOT] = str(root)
    bootstrap_arc_pythonpath()
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(root),
        "git": _git_provenance(root),
        "runtime": {
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": sys.version,
            "require_repo_root_env": os.environ[ARC_REQUIRE_REPO_ROOT],
        },
        "modules": _module_provenance(root),
        "workflow_files": _workflow_files(root, extra_files),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root_value = args.repo_root or os.environ.get(ARC_REQUIRE_REPO_ROOT)
    if not root_value:
        raise SystemExit(
            f"error: --repo-root or {ARC_REQUIRE_REPO_ROOT} is required"
        )
    record = build_provenance(Path(root_value), args.extra_files)
    payload = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
