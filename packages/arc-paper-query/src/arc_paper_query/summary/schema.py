from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


PROMPT_VERSION = "paper-summary-v1"


@lru_cache(maxsize=4)
def load_summary_schema(prompt_version: str = PROMPT_VERSION) -> dict[str, Any]:
    path = _repo_root() / "schemas" / f"{prompt_version}.schema.json"
    return json.loads(path.read_text(encoding="utf-8"))


def validate_summary(summary: dict[str, Any], prompt_version: str = PROMPT_VERSION) -> None:
    validator = Draft202012Validator(load_summary_schema(prompt_version))
    validator.validate(summary)


def load_summary_prompt(prompt_version: str = PROMPT_VERSION) -> str:
    path = _repo_root() / "prompts" / f"{prompt_version}.md"
    return path.read_text(encoding="utf-8")


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "schemas").is_dir() and (parent / "prompts").is_dir():
            return parent
    raise FileNotFoundError("Could not locate ARC repository root with schemas/ and prompts/")
