from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Mapping


def resolve_codex_binary(
    env: Mapping[str, str], *, require_executable: bool = False
) -> str | None:
    """Return one canonical argv[0] for Codex exec, help, and sandbox calls."""

    requested = str(env.get("ARC_CODEX_BIN") or "codex").strip() or "codex"
    if requested == "~" or requested.startswith(("~/", "~\\")):
        home = env.get("HOME") or env.get("USERPROFILE")
        expanded = (Path(home) / requested[2:]) if home else Path(requested).expanduser()
    else:
        expanded = Path(requested)
    path_like = (
        expanded.is_absolute()
        or os.sep in requested
        or bool(os.altsep and os.altsep in requested)
        or requested.startswith("~")
    )
    if path_like:
        candidate = expanded
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        resolved = str(candidate.resolve(strict=False))
    else:
        located = shutil.which(requested, path=env.get("PATH", os.defpath))
        resolved = str(Path(located).resolve(strict=False)) if located else requested
    if require_executable:
        path = Path(resolved)
        if not path.is_file() or not os.access(path, os.X_OK):
            return None
    return resolved
