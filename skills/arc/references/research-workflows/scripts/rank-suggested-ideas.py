#!/usr/bin/env python3
"""Rank the best scored round from each ARC suggest-ideas loop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


WORKFLOW_DIR = Path(__file__).resolve().parents[1]
if str(WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_DIR))

from research_ideas_marking import normalized_marks, rank_key_from_marks, report_columns  # noqa: E402


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
        best = dict(max(loop_rounds, key=_rank_key))
        best["rounds"] = loop_rounds
        selected.append(best)

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
    proposer_output_text = proposer_output_path.read_text(encoding="utf-8")
    review = _read_json(review_path)
    marks = review.get("review_payload", {}).get("marks", {})
    if "total_score" not in marks:
        return None

    relative = lambda path: str(path.relative_to(loop_root.parents[1]))
    return {
        "loop_id": loop_root.name,
        "round": _round_number(round_root),
        "title": str(proposer_output.get("title", "")),
        "marks": normalized_marks(marks),
        "proposer_output": proposer_output,
        "proposer_output_text": proposer_output_text,
        "proposer_output_path": relative(proposer_output_path),
        "review_path": relative(review_path),
    }

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
    return rank_key_from_marks(entry["marks"], round_number=entry["round"])


def markdown_table(payload: dict[str, Any]) -> str:
    lines = [_summary_table(payload), "", "## Appendix: Idea Details"]
    for entry in payload["ranking"]:
        lines.extend(["", *_appendix_section(entry)])
    return "\n".join(lines)


def _summary_table(payload: dict[str, Any]) -> str:
    lines = ["Suggested ideas:", ""]
    for entry in payload["ranking"]:
        lines.append(
            "{title} (Mark: {total})".format(
                title=_table_text(entry["title"]),
                total=_format_mark(entry["marks"].get("total_score")),
            )
        )
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _appendix_section(entry: dict[str, Any]) -> list[str]:
    return [
        f"### {entry['rank']}. {_heading_text(entry['title'])}",
        "",
        f"- Loop: `{entry['loop_id']}`",
        f"- Selected round: `{entry['round']}`",
        f"- Proposer output: `{entry['proposer_output_path']}`",
        f"- Review output: `{entry['review_path']}`",
        "",
        "#### Referee Marks by Round",
        "",
        _round_marks_table(entry),
        "",
        "#### Full Idea Verbatim",
        "",
        *_fenced_text_block(_handoff_text(entry.get("proposer_output", {}))),
    ]


def _round_marks_table(entry: dict[str, Any]) -> str:
    columns = report_columns()
    mark_headers = " | ".join(column["label"] for column in columns)
    mark_separator = "|".join("---:" for _ in columns)
    lines = [
        f"| Loop | Round | {mark_headers} |",
        f"|---|---:|{mark_separator}|",
    ]
    for round_entry in entry.get("rounds", []):
        marks = round_entry["marks"]
        mark_values = " | ".join(_format_mark(marks.get(column["field"])) for column in columns)
        lines.append(
            "| {loop_id} | {round} | {mark_values} |".format(
                loop_id=round_entry["loop_id"],
                round=round_entry["round"],
                mark_values=mark_values,
            )
        )
    return "\n".join(lines)


def _format_mark(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:g}"
    return ""


def _table_text(value: Any, *, max_width: int | None = None) -> str:
    text = str(value).replace("|", "/").replace("\n", " ").strip()
    if max_width:
        text = "<br>".join(_wrap_words(text, max_width=max_width))
    return text


def _heading_text(value: Any) -> str:
    text = str(value).replace("\n", " ").strip()
    return text or "Untitled Idea"


def _handoff_text(value: Any) -> str:
    data = value if isinstance(value, dict) else {}
    fields = [
        ("Title", data.get("title", "")),
        ("Idea Summary", data.get("idea_summary", "")),
        ("Calculation Plan", data.get("calculation_plan", "")),
    ]
    lines: list[str] = []
    for label, item in fields:
        text = str(item or "").strip()
        wrapped = _wrap_words(text, max_width=76) if text else [""]
        lines.append(f"{label}: {wrapped[0]}")
        lines.extend(f"  {line}" for line in wrapped[1:])
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _fenced_text_block(text: str) -> list[str]:
    return ["```text", text.rstrip(), "```"]


def _wrap_words(text: str, *, max_width: int) -> list[str]:
    text = str(text)
    if len(text) <= max_width:
        return [text]
    words = []
    for word in text.split(" "):
        words.extend(_split_long_token(word, max_width=max_width))
    lines: list[str] = []
    current = ""
    for word in words:
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= max_width:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def _split_long_token(token: str, *, max_width: int) -> list[str]:
    if len(token) <= max_width:
        return [token]
    return [token[index : index + max_width] for index in range(0, len(token), max_width)]


if __name__ == "__main__":
    main()
