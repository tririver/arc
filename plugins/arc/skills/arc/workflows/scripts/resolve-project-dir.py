#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


HOST_INTERNAL_PARTS = {".claude", ".codex"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve an ARC workflow project directory")
    parser.add_argument("--name", required=True, help="safe project directory stem")
    parser.add_argument("--run-root", default=".", help="workflow launch directory, usually pwd -P")
    parser.add_argument("--json", action="store_true", help="emit a JSON result envelope")
    args = parser.parse_args(argv)

    try:
        payload = resolve_project_dir(name=args.name, run_root=args.run_root)
    except ProjectDirError as exc:
        return _emit(
            {"ok": False, "data": None, "errors": [{"code": exc.code, "message": str(exc)}], "meta": {}},
            json_output=args.json,
            status=2,
        )

    return _emit({"ok": True, "data": payload, "errors": [], "meta": {}}, json_output=args.json, status=0)


def resolve_project_dir(*, name: str, run_root: str | Path) -> dict[str, Any]:
    project_dir_name = _validate_name(name)
    root = Path(run_root).expanduser().resolve()
    _validate_run_root(root)
    return {
        "run_root": str(root),
        "project_dir_name": project_dir_name,
        "project_dir": str(root / project_dir_name),
    }


def _validate_name(raw: str) -> str:
    name = (raw or "").strip()
    if not name:
        raise ProjectDirError("invalid_project_dir_name", "Project directory name is empty.")
    if name in {".", ".."} or Path(name).is_absolute() or "/" in name or "\\" in name:
        detail = " Nested `arc-output/<name>` directories are not allowed." if "arc-output" in name.lower() else ""
        raise ProjectDirError(
            "invalid_project_dir_name",
            f"Project directory name must be a single safe stem, got: {raw!r}.{detail}",
        )
    if name.lower() == "arc-output":
        raise ProjectDirError(
            "invalid_project_dir_name",
            "`arc-output` is not a project directory name; use the ARC safe-dir stem directly.",
        )
    return name


def _validate_run_root(root: Path) -> None:
    parts = set(root.parts)
    if parts.intersection(HOST_INTERNAL_PARTS):
        raise ProjectDirError(
            "invalid_run_root",
            f"Run root must be the user's launch directory, not host-internal storage such as .claude or .codex: {root}",
        )
    if root.name == "arc-output":
        raise ProjectDirError("invalid_run_root", f"Run root must not be an inserted arc-output directory: {root}")


def _emit(payload: dict[str, Any], *, json_output: bool, status: int) -> int:
    if json_output:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    elif payload["ok"]:
        print(payload["data"]["project_dir"])
    else:
        print(payload["errors"][0]["message"], file=sys.stderr)
    return status


class ProjectDirError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


if __name__ == "__main__":
    raise SystemExit(main())
