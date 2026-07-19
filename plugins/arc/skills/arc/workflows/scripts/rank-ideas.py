#!/usr/bin/env python3
"""Rank the best scored round from each ARC ideas loop."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping


WORKFLOW_DIR = Path(__file__).resolve().parents[1]
if str(WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_DIR))

from ideas_marking import (  # noqa: E402
    load_marking_scheme,
    normalized_marks,
    rank_key_from_marks,
    report_columns,
    score_fields,
)


CROSS_MARKING_SCHEME = WORKFLOW_DIR / "json" / "ideas-cross-domain-marking-scheme.json"
CROSS_REPORT_COLUMNS = [
    ("IR", "user_intent_relevance"),
    ("TR", "cross_domain_transfer_quality"),
    ("TC", "substantive_target_contribution"),
    ("N", "novelty"),
    ("CN", "confidence_of_novelty"),
    ("SV", "scientific_value"),
    ("F", "calculation_feasibility"),
    ("WD", "problem_well_definedness"),
    ("T", "total_score"),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select each loop's highest-marked round and rank task-to-be-planned candidates."
    )
    parser.add_argument("run_root", type=Path, help="ideas run artifact root")
    parser.add_argument("--format", choices=["json", "markdown"], default="markdown")
    args = parser.parse_args()

    payload = rank_run(args.run_root)
    if payload.get("cross_domain"):
        diagnostics_path = args.run_root.resolve().parent / "cross-domain-diagnostics.json"
        diagnostics_path.write_text(
            json.dumps(payload["diagnostics"], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    elif payload.get("single_domain_qualification"):
        diagnostics_path = args.run_root.resolve().parent / "single-domain-diagnostics.json"
        diagnostics_path.write_text(
            json.dumps(payload["diagnostics"], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if args.format == "json":
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(markdown_table(payload))


def rank_run(run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    loops_root = run_root / "loops"
    if not loops_root.is_dir():
        raise SystemExit(f"missing loops directory: {loops_root}")

    cross_contexts = _cross_domain_contexts(run_root)
    cross_domain = bool(cross_contexts)
    single_contexts = {} if cross_domain else _single_domain_contexts(run_root)
    single_domain_qualification = any(
        context.get("requires_idea_assessment") is True for context in single_contexts.values()
    )
    legacy_single_domain = bool(single_contexts) and not single_domain_qualification
    scheme = load_marking_scheme(CROSS_MARKING_SCHEME) if cross_domain else None
    selected = []
    unqualified = []
    for loop_root in sorted(path for path in loops_root.iterdir() if path.is_dir()):
        loop_context = cross_contexts.get(loop_root.name, {})
        single_context = single_contexts.get(loop_root.name, {})
        loop_rounds = [
            _round_entry(
                loop_root,
                round_root,
                scheme=scheme,
                cross_context=loop_context if cross_domain else None,
                single_context=single_context if single_contexts else None,
            )
            for round_root in _round_dirs(loop_root)
        ]
        loop_rounds = [entry for entry in loop_rounds if entry is not None]
        if not loop_rounds:
            continue
        if cross_domain or single_domain_qualification:
            qualified_rounds = [entry for entry in loop_rounds if entry.get("qualified")]
            if not qualified_rounds:
                best_failed = dict(max(loop_rounds, key=lambda item: _rank_key(item, scheme=scheme)))
                best_failed["rounds"] = loop_rounds
                unqualified.append(best_failed)
                continue
            best = dict(max(qualified_rounds, key=lambda item: _rank_key(item, scheme=scheme)))
        else:
            best = dict(max(loop_rounds, key=_rank_key))
        best["rounds"] = loop_rounds
        selected.append(best)

    ranking = sorted(selected, key=lambda item: _rank_key(item, scheme=scheme), reverse=True)
    warnings: list[str] = []
    top_three: list[dict[str, Any]] = []
    portfolio_excluded: list[dict[str, Any]] = []
    if cross_domain:
        mechanism_counts: dict[str, int] = {}
        portfolio_ranking: list[dict[str, Any]] = []
        for entry in ranking:
            mechanism = str(entry.get("normalized_central_mechanism", ""))
            if mechanism and mechanism_counts.get(mechanism, 0) >= 2:
                excluded = dict(entry)
                excluded["portfolio_exclusion_reason"] = "central_mechanism_cap_2"
                portfolio_excluded.append(excluded)
                continue
            portfolio_ranking.append(entry)
            if mechanism:
                mechanism_counts[mechanism] = mechanism_counts.get(mechanism, 0) + 1
        ranking = portfolio_ranking
        used_signatures: set[str] = set()
        for entry in ranking:
            signature = str(entry.get("normalized_transfer_signature", ""))
            if not signature or signature in used_signatures:
                continue
            top_three.append(entry)
            used_signatures.add(signature)
            if len(top_three) == 3:
                break
        if len(top_three) < 3:
            warnings.append(
                f"WARNING: only {len(top_three)} qualified, transfer-distinct cross-domain candidates are available; "
                "the top three were not padded with unqualified or duplicate candidates."
            )
        top_ids = {(entry["loop_id"], entry["round"]) for entry in top_three}
        ranking = [*top_three, *[entry for entry in ranking if (entry["loop_id"], entry["round"]) not in top_ids]]
    elif single_domain_qualification:
        top_three = ranking[:3]
        if len(top_three) < 3:
            warnings.append(
                f"WARNING: only {len(top_three)} qualified single-domain candidates are available; "
                "the top three were not padded with infeasible candidates."
            )
    elif legacy_single_domain:
        warnings.append(
            "WARNING: legacy single-domain reviews do not contain idea_assessment; "
            "ranking used the legacy_no_feasibility_gate policy."
        )
    for index, entry in enumerate(ranking, start=1):
        entry["rank"] = index
    payload = {
        "schema_version": "arc.ideas.selected_rounds.v1",
        "run_root": str(run_root),
        "user_intent": _run_user_intent(run_root),
        "summary_order": ranking if cross_domain else selected,
        "ranking": ranking,
    }
    if cross_domain:
        payload.update(
            {
                "schema_version": "arc.ideas.selected_rounds.v2",
                "cross_domain": True,
                "top_three": top_three,
                "unqualified": unqualified,
                "portfolio_excluded": portfolio_excluded,
                "warnings": warnings,
                "diagnostics": _cross_diagnostics(
                    run_root,
                    ranking=ranking,
                    top_three=top_three,
                    unqualified=unqualified,
                    portfolio_excluded=portfolio_excluded,
                    warnings=warnings,
                ),
            }
        )
    elif single_domain_qualification:
        payload.update(
            {
                "schema_version": "arc.ideas.selected_rounds.v3",
                "single_domain_qualification": True,
                "summary_order": ranking,
                "top_three": top_three,
                "unqualified": unqualified,
                "warnings": warnings,
                "diagnostics": _single_domain_diagnostics(
                    run_root,
                    ranking=ranking,
                    top_three=top_three,
                    unqualified=unqualified,
                    warnings=warnings,
                ),
            }
        )
    elif warnings:
        payload["warnings"] = warnings
    return payload


def _round_dirs(loop_root: Path) -> list[Path]:
    rounds_root = loop_root / "rounds"
    if not rounds_root.is_dir():
        return []
    return sorted(path for path in rounds_root.iterdir() if path.is_dir() and path.name.startswith("round_"))


def _round_entry(
    loop_root: Path,
    round_root: Path,
    *,
    scheme: Mapping[str, Any] | None = None,
    cross_context: Mapping[str, Any] | None = None,
    single_context: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    proposer_output_path = _first_json(round_root / "proposer_outputs")
    review_path = _first_json(round_root / "reviews")
    if proposer_output_path is None or review_path is None:
        return None

    proposer_output = _read_json(proposer_output_path)
    proposer_output_text = proposer_output_path.read_text(encoding="utf-8")
    review = _read_json(review_path)
    marks = review.get("review_payload", {}).get("marks", {})
    recovered = _major_recovered(review) or _major_recovered(proposer_output)
    if "total_score" not in marks:
        marks = {field: 0 for field in score_fields(scheme)}
        marks["total_score"] = 0
    if recovered:
        marks = {field: 0 for field in score_fields(scheme)}

    relative = lambda path: str(path.relative_to(loop_root.parents[1]))
    entry = {
        "loop_id": loop_root.name,
        "round": _round_number(round_root),
        "title": str(proposer_output.get("title") or proposer_output.get("warning") or "Recovered / unstructured idea"),
        "marks": normalized_marks(marks, scheme),
        "proposer_output": proposer_output,
        "proposer_output_text": proposer_output_text,
        "proposer_output_path": relative(proposer_output_path),
        "review_path": relative(review_path),
    }
    if cross_context is not None:
        assessment = review.get("review_payload", {}).get("cross_domain_assessment", {})
        qualified, reasons, signature, compatibility = _cross_qualification(
            proposer_output,
            assessment,
            entry["marks"],
            cross_context=cross_context,
            recovered=recovered,
        )
        entry.update(
            {
                "qualified": qualified,
                "qualification_reasons": reasons,
                "cross_domain_assessment": assessment if isinstance(assessment, dict) else {},
                "compatibility_classification": compatibility,
                "normalized_transfer_signature": signature,
                "normalized_central_mechanism": _normalized_central_mechanism(
                    assessment.get("transfer_signature") if isinstance(assessment, Mapping) else None
                ),
            }
        )
    elif single_context is not None:
        assessment = review.get("review_payload", {}).get("idea_assessment")
        if single_context.get("requires_idea_assessment") is True or isinstance(assessment, Mapping):
            qualified, reasons, feasibility = _single_domain_qualification(assessment, recovered=recovered)
            entry.update(
                {
                    "qualified": qualified,
                    "qualification_policy": "single_domain_feasibility_gate_v1",
                    "qualification_reasons": reasons,
                    "idea_assessment": assessment if isinstance(assessment, dict) else {},
                    "feasibility_classification": feasibility,
                }
            )
        else:
            entry["qualification_policy"] = "legacy_no_feasibility_gate"
    return entry

def _first_json(root: Path) -> Path | None:
    if not root.is_dir():
        return None
    return next(iter(sorted(root.glob("*.json"))), None)


def _major_recovered(payload: dict[str, Any]) -> bool:
    record = payload.get("arc_llm_call_record")
    if not isinstance(record, dict):
        return False
    structured = record.get("structured_output")
    if not isinstance(structured, dict):
        return False
    return structured.get("mode") == "recovered" and structured.get("severity") in {"major", "fatal"}


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


def _cross_domain_contexts(run_root: Path) -> dict[str, dict[str, Any]]:
    candidates = [run_root / "config.json", run_root.parent / "ideas_batch_config.json"]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError, SystemExit):
            continue
        loops = payload.get("loops")
        if not isinstance(loops, list):
            continue
        contexts: dict[str, dict[str, Any]] = {}
        for loop in loops:
            if not isinstance(loop, dict):
                continue
            context = loop.get("caller_context")
            if not isinstance(context, dict):
                continue
            if context.get("generation_mode") != "cross_domain" and context.get("variant_id") != "cross_domain":
                continue
            loop_id = str(loop.get("loop_id", "")).strip()
            if loop_id:
                contexts[loop_id] = context
        if contexts:
            return contexts
    return {}


def _single_domain_contexts(run_root: Path) -> dict[str, dict[str, Any]]:
    candidates = [run_root / "config.json", run_root.parent / "ideas_batch_config.json"]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError, SystemExit):
            continue
        loops = payload.get("loops")
        if not isinstance(loops, list):
            continue
        contexts: dict[str, dict[str, Any]] = {}
        for loop in loops:
            if not isinstance(loop, dict):
                continue
            context = loop.get("caller_context")
            if not isinstance(context, dict) or context.get("variant_id") != "domain":
                continue
            loop_id = str(loop.get("loop_id", "")).strip()
            if not loop_id:
                continue
            contexts[loop_id] = {
                **context,
                "requires_idea_assessment": _loop_requires_idea_assessment(loop),
            }
        if contexts:
            return contexts
    return {}


def _loop_requires_idea_assessment(loop: Mapping[str, Any]) -> bool:
    reviewers = loop.get("reviewers")
    if not isinstance(reviewers, list) or not reviewers or not isinstance(reviewers[0], Mapping):
        return False
    schema = reviewers[0].get("output_schema")
    if not isinstance(schema, Mapping):
        return False
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return False
    review_payload = properties.get("review_payload", {})
    required = review_payload.get("required") if isinstance(review_payload, Mapping) else None
    return isinstance(required, list) and "idea_assessment" in required


def _single_domain_qualification(
    assessment: Any,
    *,
    recovered: bool,
) -> tuple[bool, list[str], dict[str, Any]]:
    reasons: list[str] = []
    if recovered:
        reasons.append("proposer_or_reviewer_has_major_or_fatal_structured_recovery")
    if not isinstance(assessment, Mapping):
        return False, [*reasons, "missing_idea_assessment"], _empty_single_feasibility_classification()

    feasibility_status = str(assessment.get("feasibility_status", ""))
    well_definedness = str(assessment.get("mathematical_well_definedness", ""))
    external_method_status = str(assessment.get("external_method_status", ""))
    blocking_failures = _string_list(assessment.get("blocking_feasibility_failures"))
    manageable_risks = _string_list(assessment.get("manageable_feasibility_risks"))

    if feasibility_status not in {"feasible", "feasible_with_named_risk"}:
        reasons.append("first_calculation_is_not_feasible")
    if assessment.get("bounded_first_calculation_ready") is not True:
        reasons.append("bounded_first_calculation_is_not_ready")
    if blocking_failures:
        reasons.append("blocking_feasibility_failures")
    if feasibility_status == "feasible_with_named_risk" and not manageable_risks:
        reasons.append("feasible_with_named_risk_requires_named_manageable_risk")
    if well_definedness == "not_well_defined" or well_definedness not in {
        "well_defined",
        "partially_defined",
    }:
        reasons.append("mathematical_problem_is_not_well_defined")
    if external_method_status not in {"not_used", "valid"}:
        reasons.append("external_method_must_be_not_used_or_valid")

    return (
        not reasons,
        reasons,
        {
            "policy": "explicit_blocking_and_manageable_v1",
            "feasibility_status": feasibility_status,
            "well_definedness": well_definedness,
            "bounded_first_calculation_ready": assessment.get("bounded_first_calculation_ready") is True,
            "blocking_failures": blocking_failures,
            "manageable_risks": manageable_risks,
            "external_method_status": external_method_status,
        },
    )


def _empty_single_feasibility_classification() -> dict[str, Any]:
    return {
        "policy": "missing_assessment",
        "feasibility_status": "",
        "well_definedness": "",
        "bounded_first_calculation_ready": False,
        "blocking_failures": [],
        "manageable_risks": [],
        "external_method_status": "",
    }


def _cross_qualification(
    proposer: Mapping[str, Any],
    assessment: Any,
    marks: Mapping[str, Any],
    *,
    cross_context: Mapping[str, Any],
    recovered: bool,
) -> tuple[bool, list[str], str, dict[str, Any]]:
    reasons: list[str] = []
    if recovered:
        reasons.append("proposer_or_reviewer_has_major_or_fatal_structured_recovery")
    if not isinstance(assessment, Mapping):
        return False, [*reasons, "missing_cross_domain_assessment"], "", _empty_compatibility_classification()

    cards = cross_context.get("domain_cards", [])
    known_domain_ids = {
        str(card.get("domain_id", "")).strip()
        for card in cards
        if isinstance(card, Mapping) and str(card.get("domain_id", "")).strip()
    }
    source = str(assessment.get("source_domain_id", "")).strip()
    target = str(assessment.get("target_domain_id", "")).strip()
    if not source or not target or source == target:
        reasons.append("source_and_target_must_be_distinct")
    if source not in known_domain_ids or target not in known_domain_ids:
        reasons.append("source_or_target_is_not_a_manifest_domain")

    roles = proposer.get("domain_roles")
    if not isinstance(roles, Mapping):
        reasons.append("missing_proposer_domain_roles")
    elif str(roles.get("source_domain_id", "")).strip() != source or str(
        roles.get("target_domain_id", "")
    ).strip() != target:
        reasons.append("proposer_and_reviewer_domain_roles_disagree")

    required_values = {
        "transfer_status": "genuine",
        "source_ingredient_validity": "valid",
        "target_adaptation_validity": "valid",
    }
    for field, required in required_values.items():
        if assessment.get(field) != required:
            reasons.append(f"{field}_must_be_{required}")
    if assessment.get("target_contribution_status") not in {"substantial", "transformative"}:
        reasons.append("target_contribution_must_be_substantial_or_transformative")
    if assessment.get("feasibility_status") not in {"feasible", "feasible_with_named_risk"}:
        reasons.append("first_calculation_is_not_feasible")
    compatibility = _compatibility_classification(assessment)
    if compatibility["blocking_failures"]:
        reasons.append("blocking_compatibility_failures")
    if (
        assessment.get("feasibility_status") == "feasible_with_named_risk"
        and not compatibility["manageable_risks"]
    ):
        reasons.append("feasible_with_named_risk_requires_named_manageable_risk")
    if assessment.get("disqualifying_reasons"):
        reasons.append("reviewer_reported_disqualifying_reasons")
    novelty = assessment.get("novelty_coverage")
    if not isinstance(novelty, Mapping) or not all(
        novelty.get(scope) is True for scope in ("source_domain", "target_domain", "intersection")
    ):
        reasons.append("source_target_and_intersection_novelty_checks_are_required")

    thresholds = {
        "cross_domain_transfer_quality": 10,
        "substantive_target_contribution": 14,
        "scientific_value": 6,
        "calculation_feasibility": 6,
        "problem_well_definedness": 6,
    }
    for field, minimum in thresholds.items():
        try:
            value = float(marks.get(field, 0))
        except (TypeError, ValueError):
            value = 0
        if value < minimum:
            reasons.append(f"{field}_below_{minimum}")

    signature = _normalized_transfer_signature(assessment.get("transfer_signature"))
    if not signature:
        reasons.append("complete_transfer_signature_is_required")
    return not reasons, reasons, signature, compatibility


def _compatibility_classification(assessment: Mapping[str, Any]) -> dict[str, Any]:
    if "blocking_compatibility_failures" in assessment or "manageable_compatibility_risks" in assessment:
        return {
            "policy": "explicit_blocking_and_manageable_v2",
            "blocking_failures": _string_list(assessment.get("blocking_compatibility_failures")),
            "manageable_risks": _string_list(assessment.get("manageable_compatibility_risks")),
        }

    legacy = _string_list(assessment.get("compatibility_failures"))
    if assessment.get("feasibility_status") == "feasible_with_named_risk":
        return {
            "policy": "legacy_compatibility_failures_as_named_risks",
            "blocking_failures": [],
            "manageable_risks": legacy,
        }
    return {
        "policy": "legacy_compatibility_failures_as_blocking",
        "blocking_failures": legacy,
        "manageable_risks": [],
    }


def _empty_compatibility_classification() -> dict[str, Any]:
    return {"policy": "missing_assessment", "blocking_failures": [], "manageable_risks": []}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _normalized_transfer_signature(raw: Any) -> str:
    if not isinstance(raw, Mapping):
        return ""
    fields = ("direction", "transferred_ingredient", "target_result", "first_calculation")
    values = [re.sub(r"\s+", " ", str(raw.get(field, "")).strip().lower()) for field in fields]
    if any(not value for value in values):
        return ""
    return " | ".join(values)


def _normalized_central_mechanism(raw: Any) -> str:
    if not isinstance(raw, Mapping):
        return ""
    values = [
        re.sub(r"\s+", " ", str(raw.get(field, "")).strip().lower())
        for field in ("direction", "transferred_ingredient")
    ]
    if any(not value for value in values):
        return ""
    return " | ".join(values)


def _cross_diagnostics(
    run_root: Path,
    *,
    ranking: list[dict[str, Any]],
    top_three: list[dict[str, Any]],
    unqualified: list[dict[str, Any]],
    portfolio_excluded: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    top_keys = {(entry["loop_id"], entry["round"]) for entry in top_three}
    candidates = []
    for qualified, entries in ((True, ranking), (False, unqualified)):
        for entry in entries:
            candidates.append(
                {
                    "loop_id": entry["loop_id"],
                    "round": entry["round"],
                    "title": entry["title"],
                    "qualified": qualified,
                    "qualification_reasons": entry.get("qualification_reasons", []),
                    "compatibility_classification": entry.get("compatibility_classification", {}),
                    "transfer_signature": entry.get("normalized_transfer_signature", ""),
                    "central_mechanism": entry.get("normalized_central_mechanism", ""),
                    "top_three": (entry["loop_id"], entry["round"]) in top_keys,
                    "marks": entry["marks"],
                }
            )
    for entry in portfolio_excluded:
        candidates.append(
            {
                "loop_id": entry["loop_id"],
                "round": entry["round"],
                "title": entry["title"],
                "qualified": True,
                "portfolio_excluded": True,
                "portfolio_exclusion_reason": entry["portfolio_exclusion_reason"],
                "qualification_reasons": entry.get("qualification_reasons", []),
                "transfer_signature": entry.get("normalized_transfer_signature", ""),
                "central_mechanism": entry.get("normalized_central_mechanism", ""),
                "top_three": False,
                "marks": entry["marks"],
            }
        )
    return {
        "schema_version": "arc.ideas.cross_domain_diagnostics.v1",
        "run_root": str(run_root),
        "qualified_count": len(ranking),
        "unqualified_count": len(unqualified),
        "portfolio_excluded_count": len(portfolio_excluded),
        "top_three_count": len(top_three),
        "distinct_qualified_transfer_signatures": len(
            {entry.get("normalized_transfer_signature", "") for entry in ranking}
        ),
        "warnings": warnings,
        "candidates": candidates,
    }


def _single_domain_diagnostics(
    run_root: Path,
    *,
    ranking: list[dict[str, Any]],
    top_three: list[dict[str, Any]],
    unqualified: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    top_keys = {(entry["loop_id"], entry["round"]) for entry in top_three}
    candidates = []
    for qualified, entries in ((True, ranking), (False, unqualified)):
        for entry in entries:
            assessment = entry.get("idea_assessment", {})
            candidates.append(
                {
                    "loop_id": entry["loop_id"],
                    "round": entry["round"],
                    "title": entry["title"],
                    "qualified": qualified,
                    "qualification_policy": entry.get("qualification_policy", ""),
                    "qualification_reasons": entry.get("qualification_reasons", []),
                    "problem_importance": (
                        assessment.get("problem_importance", "") if isinstance(assessment, Mapping) else ""
                    ),
                    "importance_rationale": (
                        assessment.get("importance_rationale", "") if isinstance(assessment, Mapping) else ""
                    ),
                    "feasibility_classification": entry.get("feasibility_classification", {}),
                    "top_three": (entry["loop_id"], entry["round"]) in top_keys,
                    "marks": entry["marks"],
                }
            )
    return {
        "schema_version": "arc.ideas.single_domain_diagnostics.v1",
        "run_root": str(run_root),
        "qualified_count": len(ranking),
        "unqualified_count": len(unqualified),
        "top_three_count": len(top_three),
        "warnings": warnings,
        "candidates": candidates,
    }


def _round_number(round_root: Path) -> int:
    try:
        return int(round_root.name.split("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def _rank_key(
    entry: dict[str, Any],
    *,
    scheme: Mapping[str, Any] | None = None,
) -> tuple[float, ...]:
    return rank_key_from_marks(entry["marks"], round_number=entry["round"], scheme=scheme)


def markdown_table(payload: dict[str, Any]) -> str:
    lines = [
        _summary_table(payload),
        "",
        "# Appendix: Idea Details",
    ]
    for entry in payload["ranking"]:
        lines.extend(["", *_appendix_section(entry)])
    if payload.get("cross_domain"):
        lines.extend(["", "# Appendix: Unqualified Cross-Domain Candidates"])
        if not payload.get("unqualified"):
            lines.extend(["", "None."])
        for entry in payload.get("unqualified", []):
            lines.extend(
                [
                    "",
                    f"## `{entry['loop_id']}` — {_heading_text(entry['title'])}",
                    "",
                    f"- Best observed round: `{entry['round']}`",
                    "- Qualification failures:",
                    *[f"  - {reason}" for reason in entry.get("qualification_reasons", [])],
                ]
            )
        lines.extend(["", "# Appendix: Portfolio-Excluded Cross-Domain Candidates"])
        if not payload.get("portfolio_excluded"):
            lines.extend(["", "None."])
        for entry in payload.get("portfolio_excluded", []):
            lines.extend(
                [
                    "",
                    f"## `{entry['loop_id']}` — {_heading_text(entry['title'])}",
                    "",
                    f"- Selected round: `{entry['round']}`",
                    f"- Exclusion: `{entry['portfolio_exclusion_reason']}`",
                ]
            )
    elif payload.get("single_domain_qualification"):
        lines.extend(["", "# Appendix: Unqualified Single-Domain Candidates"])
        if not payload.get("unqualified"):
            lines.extend(["", "None."])
        for entry in payload.get("unqualified", []):
            lines.extend(
                [
                    "",
                    f"## `{entry['loop_id']}` — {_heading_text(entry['title'])}",
                    "",
                    f"- Best observed round: `{entry['round']}`",
                    "- Qualification failures:",
                    *[f"  - {reason}" for reason in entry.get("qualification_reasons", [])],
                ]
            )
    return "\n".join(lines)


def _summary_table(payload: dict[str, Any]) -> str:
    if payload.get("cross_domain"):
        return _cross_summary_table(payload)
    lines = [
        "# Ideas",
        "",
        "Abbreviations:",
        "",
        "IR=intent relevance, N=novelty, CN=confidence of novelty, SV=scientific value, "
        "PL=planning, WD=well-definedness, T=total.",
    ]
    for warning in payload.get("warnings", []):
        lines.extend(["", str(warning)])
    for entry in payload.get("summary_order", payload.get("ranking", [])):
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


def _cross_summary_table(payload: dict[str, Any]) -> str:
    lines = [
        "# Ideas",
        "",
        "Abbreviations:",
        "",
        "IR=intent relevance, TR=transfer quality, TC=target contribution, N=novelty, "
        "CN=confidence of novelty, SV=scientific value, F=feasibility, WD=well-definedness, T=total.",
    ]
    for warning in payload.get("warnings", []):
        lines.extend(["", str(warning)])
    for entry in payload.get("summary_order", payload.get("ranking", [])):
        lines.extend(["", *_round_marks_summary_section_cross(entry)])
    return "\n".join(lines)


def _round_marks_summary_section_cross(entry: dict[str, Any]) -> list[str]:
    return [
        f"## `{entry['loop_id']}`",
        "",
        _heading_text(entry["title"]),
        "",
        _compact_cross_marks_table(entry),
    ]


def _compact_cross_marks_table(entry: dict[str, Any]) -> str:
    headers = " | ".join(label for label, _field in CROSS_REPORT_COLUMNS)
    separators = "|".join("---:" for _ in CROSS_REPORT_COLUMNS)
    lines = [f"| Round | {headers} |", f"|---:|{separators}|"]
    for round_entry in entry.get("rounds", []):
        marks = round_entry["marks"]
        values = " | ".join(_format_mark(marks.get(field)) for _label, field in CROSS_REPORT_COLUMNS)
        lines.append(f"| {round_entry['round']} | {values} |")
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
    if "cross_domain_assessment" in entry:
        columns = [{"label": label, "field": field} for label, field in CROSS_REPORT_COLUMNS]
    else:
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
