#!/usr/bin/env python3
"""Rank the best scored round from each ARC ideas loop."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


WORKFLOW_DIR = Path(__file__).resolve().parents[1]
if str(WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_DIR))

from ideas_marking import normalized_marks, rank_key_from_marks, report_columns  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select each loop's highest-marked round and rank task-to-be-planned candidates."
    )
    parser.add_argument("run_root", type=Path, help="ideas run artifact root")
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
        "schema_version": "arc.ideas.selected_rounds.v1",
        "run_root": str(run_root),
        "user_intent": _run_user_intent(run_root),
        "summary_order": selected,
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


def _run_user_intent(run_root: Path) -> str:
    candidates = [
        run_root.parent.parent / f"{run_root.parent.name}.config.json",
        run_root.parent / "config.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            intent = _read_json(path).get("user_intent", "")
        except (OSError, json.JSONDecodeError, SystemExit):
            continue
        if isinstance(intent, str) and intent.strip():
            return intent.strip()
    return ""


def _round_number(round_root: Path) -> int:
    try:
        return int(round_root.name.split("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def _rank_key(entry: dict[str, Any]) -> tuple[float, ...]:
    return rank_key_from_marks(entry["marks"], round_number=entry["round"])


def markdown_table(payload: dict[str, Any]) -> str:
    lines = [
        _summary_table(payload),
        "",
        "# Appendix: Idea Details",
    ]
    for entry in payload["ranking"]:
        lines.extend(["", *_appendix_section(entry)])
    return "\n".join(lines)


def _summary_table(payload: dict[str, Any]) -> str:
    lines = [
        "# Ideas",
        "",
        "Abbreviations:",
        "",
        "IR=intent relevance, N=novelty, CN=confidence of novelty, SV=scientific value, "
        "PL=planning, WD=well-definedness, T=total.",
    ]
    for entry in payload.get("ranking", payload.get("summary_order", [])):
        lines.extend(["", *_round_marks_summary_section(entry)])
    if lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _round_marks_summary_section(entry: dict[str, Any]) -> list[str]:
    return [
        f"## `{entry['loop_id']}`",
        "",
        _heading_text(entry["title"]),
        "",
        _compact_round_marks_table(entry),
    ]


def _compact_round_marks_table(entry: dict[str, Any]) -> str:
    columns = [
        ("IR", "user_intent_relevance"),
        ("N", "novelty"),
        ("CN", "confidence_of_novelty"),
        ("SV", "scientific_value"),
        ("PL", "planning"),
        ("WD", "problem_well_definedness"),
        ("T", "total_score"),
    ]
    lines = [
        "| Round | IR | N | CN | SV | PL | WD | T |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for round_entry in entry.get("rounds", []):
        marks = round_entry["marks"]
        mark_values = " | ".join(_format_mark(marks.get(field)) for _, field in columns)
        lines.append(f"| {round_entry['round']} | {mark_values} |")
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
        _handoff_text(entry.get("proposer_output", {})),
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
        text = _math_markdown_text(str(item or "").strip())
        lines.append(f"{label}: {text}")
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _math_markdown_text(text: str) -> str:
    text = re.sub(r"`([^`]+)`", _math_markdown_span, text)
    text = _display_math_lines(text)
    return _inline_raw_math_tokens(text)


def _math_markdown_span(match: re.Match[str]) -> str:
    content = match.group(1)
    if _looks_like_math(content):
        return f"${_format_math(content)}$"
    return match.group(0)


def _looks_like_math(text: str) -> bool:
    return bool(re.search(r"[=<>^_∫⟨⟩δΔκγρτλπℓεαβηθΦΣ{}|≈≤≥]", text))


def _inline_raw_math_tokens(text: str) -> str:
    parts = re.split(r"(\$\$.*?\$\$|\$.*?\$)", text, flags=re.DOTALL)
    for index in range(0, len(parts), 2):
        parts[index] = re.sub(
            r"(?<![\w$])([A-Za-z]+\^[A-Za-z0-9]+_[A-Za-z0-9+-]+)(?![\w])",
            lambda m: f"${_format_math(m.group(1))}$",
            parts[index],
        )
        parts[index] = re.sub(
            r"(?<![\w$])([A-Za-zαβγδεηθκλρτΦΣΔπℓ]+_[A-Za-z0-9+-]+)(?![\w])",
            lambda m: f"${_format_math(m.group(1))}$",
            parts[index],
        )
    return "".join(parts)


def _display_math_lines(text: str) -> str:
    lines: list[str] = []
    in_display_math = False
    for line in text.splitlines():
        stripped = line.strip().rstrip(",")
        if stripped == "$$":
            lines.append(line)
            in_display_math = not in_display_math
            continue
        if in_display_math:
            lines.append(line)
            continue
        math_span = re.fullmatch(r"\$(.+)\$", stripped)
        if math_span and _looks_like_display_equation(math_span.group(1)):
            lines.extend(["$$", math_span.group(1), "$$"])
        elif _looks_like_display_equation(stripped):
            lines.extend(["$$", _format_math(stripped), "$$"])
        else:
            lines.append(line)
    return "\n".join(lines)


def _looks_like_display_equation(text: str) -> bool:
    if not text or ":" in text[:24]:
        return False
    return bool(re.match(r"^([A-Za-zαβγδεηθκλρτΦΣΔπℓ]+[A-Za-z0-9_]*\(|∫|\\int)", text))


def _format_math(text: str) -> str:
    text = str(text).strip()
    text = re.sub(
        r"\b([A-Za-zαβγδεηθκλρτΦΣΔπℓ]+(?:\^[A-Za-z0-9]+)?)_([A-Za-z0-9+-]+)(?![\w])",
        lambda m: f"{m.group(1)}_{{{m.group(2)}}}",
        text,
    )
    return re.sub(r"\s+", " ", text).strip()


if __name__ == "__main__":
    main()
