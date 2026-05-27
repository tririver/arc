#!/usr/bin/env python3
"""Filter foundation JSON for a single foundation-check step."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Mapping


def filter_foundation_context(
    payload: Mapping[str, Any],
    *,
    target_equation_id: str = "",
) -> dict[str, Any]:
    equations = payload.get("equations", [])
    if not isinstance(equations, list):
        equations = []

    allowed_equations: list[dict[str, Any]] = []
    omitted_equation_ids: list[str] = []
    target_equation: dict[str, Any] | None = None
    target_equation_id = target_equation_id.strip()
    if not target_equation_id:
        raise ValueError("target_equation_id is required")
    for item in equations:
        if not isinstance(item, dict):
            continue
        equation_id = str(item.get("id", ""))
        if target_equation_id and equation_id == target_equation_id:
            target_equation = sanitize_foundation_context_item(item)
            continue
        if equation_is_axiom_or_checked(item):
            allowed_equations.append(sanitize_foundation_context_item(item))
        elif equation_id:
            omitted_equation_ids.append(equation_id)
    if target_equation is None:
        raise ValueError(f"target_equation_id {target_equation_id} was not found in foundation equations")

    conventions = payload.get("conventions", [])
    if not isinstance(conventions, list):
        conventions = []
    allowed_conventions = [
        sanitize_foundation_context_item(item)
        for item in conventions
        if isinstance(item, dict) and convention_is_checked(item)
    ]

    return {
        "schema_version": "arc.foundation_context.v1",
        "target_equation_id": target_equation_id,
        "target_equation": target_equation,
        "allowed_equations": allowed_equations,
        "allowed_conventions": allowed_conventions,
        "omitted_equation_ids": omitted_equation_ids,
        "filter_rule": "Only the target equation plus axiom or checked foundation items are provided.",
    }


def equation_is_axiom_or_checked(item: Mapping[str, Any]) -> bool:
    if item.get("axiom_status") == "axiom":
        return True
    check_status = str(item.get("check_status", "")).strip().lower()
    return check_status == "checked" or check_status.startswith("checked_")


def convention_is_checked(item: Mapping[str, Any]) -> bool:
    check_status = str(item.get("check_status", "")).strip().lower()
    return check_status == "checked" or check_status.startswith("checked_")


FOUNDATION_CONTEXT_OMIT_KEYS = {"sources", "mcp", "cli", "cache_path", "source_path"}


def sanitize_foundation_context_item(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): sanitize_foundation_context_item(item)
            for key, item in value.items()
            if str(key) not in FOUNDATION_CONTEXT_OMIT_KEYS
        }
    if isinstance(value, list):
        return [sanitize_foundation_context_item(item) for item in value]
    return copy.deepcopy(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("foundation_json", type=Path)
    parser.add_argument("--target-equation-id", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    payload = json.loads(args.foundation_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("foundation_json must contain a JSON object")

    filtered = filter_foundation_context(
        payload,
        target_equation_id=args.target_equation_id,
    )
    text = json.dumps(filtered, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
