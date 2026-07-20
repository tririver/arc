from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


def arc_home(env: Mapping[str, str] | None = None) -> Path:
    """Resolve ARC's shared state root without writing into a source checkout."""
    material = os.environ if env is None else env
    if value := _text(material, "ARC_HOME"):
        return Path(value).expanduser()
    if value := _text(material, "XDG_DATA_HOME"):
        return Path(value).expanduser() / "arc"
    home = Path(material["HOME"]).expanduser() if material.get("HOME") else Path.home()
    return home / ".local" / "share" / "arc"


def llm_cache_root(env: Mapping[str, str] | None = None) -> Path:
    material = os.environ if env is None else env
    if value := _text(material, "ARC_LLM_CACHE"):
        return Path(value).expanduser()
    return arc_home(material) / "cache" / "arc-llm"


def llm_tmp_root(env: Mapping[str, str] | None = None) -> Path:
    material = os.environ if env is None else env
    if value := _text(material, "ARC_LLM_TMP_DIR"):
        return Path(value).expanduser()
    return arc_home(material) / "tmp" / "arc-llm"


def schema_cache_root(env: Mapping[str, str] | None = None) -> Path:
    material = os.environ if env is None else env
    if value := _text(material, "ARC_LLM_SCHEMA_CACHE_DIR"):
        return Path(value).expanduser()
    return llm_cache_root(material) / "schemas"


def _text(env: Mapping[str, str], key: str) -> str | None:
    value = env.get(key)
    return value.strip() if value is not None and value.strip() else None
