from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping


MARKING_SCHEME_SCHEMA = "arc.workflow.ideas.marking_scheme.v1"
DEFAULT_MARKING_SCHEME_FILENAME = "ideas-marking-scheme.json"


def load_marking_scheme(workflow_dir: Path | str | None = None) -> dict[str, Any]:
    path = _scheme_path(workflow_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"marking scheme must be a JSON object: {path}")
    _validate_scheme(payload, path=path)
    return payload


def marking_scheme_for_context(scheme: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return copy.deepcopy(dict(scheme or load_marking_scheme()))


def mark_fields(scheme: Mapping[str, Any] | None = None) -> list[str]:
    data = scheme or load_marking_scheme()
    return [str(item["field"]) for item in data["marks"]]


def total_score_field(scheme: Mapping[str, Any] | None = None) -> str:
    data = scheme or load_marking_scheme()
    return str(data["total_score"]["field"])


def score_fields(scheme: Mapping[str, Any] | None = None) -> list[str]:
    data = scheme or load_marking_scheme()
    return [*mark_fields(data), total_score_field(data)]


def marks_schema(scheme: Mapping[str, Any] | None = None) -> dict[str, Any]:
    data = scheme or load_marking_scheme()
    properties = {
        str(item["field"]): _number_schema(item)
        for item in data["marks"]
    }
    total = data["total_score"]
    properties[str(total["field"])] = _number_schema(total)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": score_fields(data),
        "properties": properties,
    }


def render_marking_instructions(scheme: Mapping[str, Any] | None = None) -> str:
    data = scheme or load_marking_scheme()
    lines = [
        "Use caller_context.marking_scheme as the sole scoring source of truth.",
        "Apply every mark on its listed numeric range and return every required field in marks.",
    ]
    for item in data["marks"]:
        lines.append(
            f"- {item['field']} ({item['minimum']}-{item['maximum']}): {item['guidance']}"
        )
    total = data["total_score"]
    lines.append(f"- {total['field']} ({total['minimum']}-{total['maximum']}): {total['guidance']}")
    return "\n".join(lines)


def normalized_marks(raw_marks: Any, scheme: Mapping[str, Any] | None = None) -> dict[str, Any]:
    marks = raw_marks if isinstance(raw_marks, Mapping) else {}
    return {field: marks.get(field) for field in score_fields(scheme)}


def rank_key_from_marks(
    raw_marks: Mapping[str, Any],
    *,
    round_number: int | float = 0,
    scheme: Mapping[str, Any] | None = None,
) -> tuple[float, ...]:
    data = scheme or load_marking_scheme()
    order = data.get("tie_break_order") or score_fields(data)
    return tuple(_float(raw_marks.get(str(field))) for field in order) + (float(round_number),)


def report_columns(scheme: Mapping[str, Any] | None = None) -> list[dict[str, str]]:
    data = scheme or load_marking_scheme()
    total = data["total_score"]
    return [
        {"field": str(total["field"]), "label": str(total.get("label") or total["field"])},
        *[
            {"field": str(item["field"]), "label": str(item.get("label") or item["field"])}
            for item in data["marks"]
        ],
    ]


def _scheme_path(workflow_dir: Path | str | None) -> Path:
    root = Path(workflow_dir).expanduser() if workflow_dir is not None else Path(__file__).resolve().parent
    if root.name == "scripts":
        root = root.parent / "json"
    return root / DEFAULT_MARKING_SCHEME_FILENAME


def _validate_scheme(payload: Mapping[str, Any], *, path: Path) -> None:
    if payload.get("schema_version") != MARKING_SCHEME_SCHEMA:
        raise ValueError(f"{path}.schema_version must be {MARKING_SCHEME_SCHEMA}")
    marks = payload.get("marks")
    if not isinstance(marks, list) or not marks:
        raise ValueError(f"{path}.marks must be a non-empty list")
    seen: set[str] = set()
    total_maximum = 0.0
    for item in marks:
        if not isinstance(item, Mapping):
            raise ValueError(f"{path}.marks items must be objects")
        field = _required_text(item, "field", path)
        if field in seen:
            raise ValueError(f"{path}.marks duplicate field: {field}")
        seen.add(field)
        _required_text(item, "label", path)
        _required_text(item, "guidance", path)
        minimum = _number(item, "minimum", path)
        maximum = _number(item, "maximum", path)
        if minimum != 0:
            raise ValueError(f"{path}.{field}.minimum must be 0")
        if maximum <= minimum:
            raise ValueError(f"{path}.{field}.maximum must be greater than minimum")
        total_maximum += maximum
    total = payload.get("total_score")
    if not isinstance(total, Mapping):
        raise ValueError(f"{path}.total_score must be an object")
    if _required_text(total, "field", path) != "total_score":
        raise ValueError(f"{path}.total_score.field must be total_score")
    if _number(total, "minimum", path) != 0:
        raise ValueError(f"{path}.total_score.minimum must be 0")
    if _number(total, "maximum", path) != total_maximum:
        raise ValueError(f"{path}.total_score.maximum must equal summed mark maxima")
    tie_break_order = payload.get("tie_break_order", [])
    if not isinstance(tie_break_order, list) or not tie_break_order:
        raise ValueError(f"{path}.tie_break_order must be a non-empty list")
    allowed = {*seen, "total_score"}
    unknown = [str(field) for field in tie_break_order if str(field) not in allowed]
    if unknown:
        raise ValueError(f"{path}.tie_break_order has unknown fields: {', '.join(unknown)}")


def _number_schema(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "number",
        "minimum": item["minimum"],
        "maximum": item["maximum"],
        "description": item.get("guidance", ""),
    }


def _required_text(item: Mapping[str, Any], field_name: str, path: Path) -> str:
    text = str(item.get(field_name, "")).strip()
    if not text:
        raise ValueError(f"{path}.{field_name} is required")
    return text


def _number(item: Mapping[str, Any], field_name: str, path: Path) -> float:
    value = item.get(field_name)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{path}.{field_name} must be a number")
    return float(value)


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
