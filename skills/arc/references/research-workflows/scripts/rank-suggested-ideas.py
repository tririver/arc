#!/usr/bin/env python3
"""Rank the best scored round from each ARC suggest-ideas loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MARK_FIELDS = [
    "evidence_of_novelty",
    "user_intent_relevance",
    "scientific_value",
    "feasibility",
    "first_calculation_clarity",
    "total_score",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select each loop's highest-marked round and rank the selected ideas."
    )
    parser.add_argument("run_root", type=Path, help="suggest-ideas run artifact root")
    parser.add_argument("--format", choices=["json", "markdown"], default="markdown")
    args = parser.parse_args()

    payload = rank_run(args.run_root)
    if args.format == "json":
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(markdown_table(payload))


def rank_run(run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    loops_root = run_root / "loops"
    if not loops_root.is_dir():
        raise SystemExit(f"missing loops directory: {loops_root}")

    selected = []
    for loop_root in sorted(path for path in loops_root.iterdir() if path.is_dir()):
        loop_rounds = [_round_entry(loop_root, round_root) for round_root in _round_dirs(loop_root)]
        loop_rounds = [entry for entry in loop_rounds if entry is not None]
        if not loop_rounds:
            continue
        selected.append(max(loop_rounds, key=_rank_key))

    ranking = sorted(selected, key=_rank_key, reverse=True)
    for index, entry in enumerate(ranking, start=1):
        entry["rank"] = index

    return {
        "schema_version": "arc.suggest_ideas.selected_rounds.v1",
        "run_root": str(run_root),
        "ranking": ranking,
    }


def _round_dirs(loop_root: Path) -> list[Path]:
    rounds_root = loop_root / "rounds"
    if not rounds_root.is_dir():
        return []
    return sorted(path for path in rounds_root.iterdir() if path.is_dir() and path.name.startswith("round_"))


def _round_entry(loop_root: Path, round_root: Path) -> dict[str, Any] | None:
    proposer_output_path = _first_json(round_root / "proposer_outputs")
    review_path = _first_json(round_root / "reviews")
    if proposer_output_path is None or review_path is None:
        return None

    proposer_output = _read_json(proposer_output_path)
    review = _read_json(review_path)
    marks = review.get("review_payload", {}).get("marks", {})
    if "total_score" not in marks:
        return None

    relative = lambda path: str(path.relative_to(loop_root.parents[1]))
    return {
        "loop_id": loop_root.name,
        "round": _round_number(round_root),
        "title": str(proposer_output.get("title", "")),
        "marks": _normalized_marks(marks),
        "proposer_output_path": relative(proposer_output_path),
        "review_path": relative(review_path),
    }


def _normalized_marks(marks: dict[str, Any]) -> dict[str, Any]:
    normalized = {field: marks.get(field) for field in MARK_FIELDS}
    if normalized.get("user_intent_relevance") is None:
        normalized["user_intent_relevance"] = marks.get("user_intent_fit")
    return normalized


def _first_json(root: Path) -> Path | None:
    if not root.is_dir():
        return None
    return next(iter(sorted(root.glob("*.json"))), None)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return data


def _round_number(round_root: Path) -> int:
    try:
        return int(round_root.name.split("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def _rank_key(entry: dict[str, Any]) -> tuple[float, ...]:
    marks = entry["marks"]
    tie_break_order = [
        "total_score",
        "evidence_of_novelty",
        "user_intent_relevance",
        "scientific_value",
        "feasibility",
        "first_calculation_clarity",
    ]
    return tuple(float(marks.get(field) or 0.0) for field in tie_break_order) + (float(entry["round"]),)


def markdown_table(payload: dict[str, Any]) -> str:
    lines = [
        "| Rank | Loop | Round | Total | Novelty | Intent Relevance | Value | Feasibility | Clarity | Title |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for entry in payload["ranking"]:
        marks = entry["marks"]
        title = entry["title"].replace("|", "/")
        lines.append(
            "| {rank} | {loop_id} | {round} | {total_score:g} | {evidence_of_novelty:g} | "
            "{user_intent_relevance:g} | {scientific_value:g} | {feasibility:g} | "
            "{first_calculation_clarity:g} | {title} |".format(
                rank=entry["rank"],
                loop_id=entry["loop_id"],
                round=entry["round"],
                title=title,
                **marks,
            )
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
