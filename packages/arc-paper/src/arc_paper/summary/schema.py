from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Any

from jsonschema import Draft202012Validator


PROMPT_VERSION = "paper-summary-v1"


@lru_cache(maxsize=4)
def load_summary_schema(prompt_version: str = PROMPT_VERSION) -> dict[str, Any]:
    return json.loads(_summary_resource("schemas", f"{prompt_version}.schema.json").read_text(encoding="utf-8"))


def validate_summary(summary: dict[str, Any], prompt_version: str = PROMPT_VERSION) -> None:
    validator = Draft202012Validator(load_summary_schema(prompt_version))
    validator.validate(summary)


def load_summary_prompt(prompt_version: str = PROMPT_VERSION) -> str:
    return _summary_resource("prompts", f"{prompt_version}.md").read_text(encoding="utf-8")


def _summary_resource(resource_type: str, filename: str):
    return files("arc_paper.summary").joinpath(resource_type, filename)
