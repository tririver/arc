#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

from _arc_script_bootstrap import ARC_REQUIRE_REPO_ROOT, bootstrap_arc_pythonpath


JsonRunner = Callable[..., dict[str, Any]]
MetadataFetcher = Callable[[str], Mapping[str, Any]]
HISTORY_PATH_PARTS = (("0_ref",), ("arc-tests", "prev"))


class PartnerSelectionError(ValueError):
    pass


def select_partner(
    anchor_summary_path: Path,
    *,
    user_intent: str,
    anchor_field_id: str | None = None,
    provider: str = "auto",
    model: str | None = None,
    model_tier: str = "high",
    json_runner: JsonRunner | None = None,
    metadata_fetcher: MetadataFetcher | None = None,
) -> dict[str, Any]:
    anchor_path = anchor_summary_path.expanduser().resolve()
    _reject_historical_path(anchor_path)
    if json_runner is None or metadata_fetcher is None:
        _require_strict_source_mode()
        bootstrap_arc_pythonpath()
    call_json = json_runner or _default_json_runner()
    anchor_source = _read_object(anchor_path)
    anchor = _anchor_field(anchor_source, source_path=anchor_path, requested_field_id=anchor_field_id)
    selector_schema = _workflow_json("cross-domain-partner-selector.schema.json")
    critic_schema = _workflow_json("cross-domain-partner-critic.schema.json")
    selector_prompt = _selector_prompt(anchor, user_intent=user_intent)
    selector = call_json(
        selector_prompt,
        schema=selector_schema,
        provider=provider,
        model=model,
        model_tier=model_tier,
        role_hint="cross-domain partner selector",
    )
    raw_candidates = selector.get("candidates", [])
    if not isinstance(raw_candidates, list) or len(raw_candidates) != 6:
        raise PartnerSelectionError("selector must return exactly six candidates")

    fetch = metadata_fetcher or _fetch_metadata
    verified: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    seen_seeds: set[str] = set()
    for index, raw in enumerate(raw_candidates):
        if not isinstance(raw, dict):
            rejected.append(
                {
                    "representative_seed": f"<candidate_{index + 1}_missing>",
                    "reason": "candidate_is_not_an_object",
                }
            )
            continue
        candidate = dict(raw)
        seed = str(candidate.get("representative_seed", "")).strip()
        if not seed or seed in seen_seeds:
            rejected.append({"representative_seed": seed, "reason": "missing_or_duplicate_seed"})
            continue
        seen_seeds.add(seed)
        try:
            metadata = dict(fetch(seed))
        except Exception as exc:
            rejected.append({"representative_seed": seed, "reason": f"metadata_error: {exc}"})
            continue
        title = str(metadata.get("title", "")).strip()
        if not title:
            rejected.append({"representative_seed": seed, "reason": "metadata_has_no_title"})
            continue
        candidate["verified_metadata"] = {
            "paper_id": str(metadata.get("paper_id") or seed),
            "title": title,
            "abstract": str(metadata.get("abstract", "")).strip(),
        }
        verified.append(candidate)
    if len(verified) < 3:
        raise PartnerSelectionError(
            "fewer than three proposed representative seeds passed ARC paper metadata verification "
            f"({len(verified)}/6 verified); rejected seeds: {_format_rejections(rejected)}"
        )

    critic_prompt = _critic_prompt(anchor, verified, user_intent=user_intent)
    critic = call_json(
        critic_prompt,
        schema=critic_schema,
        provider=provider,
        model=model,
        model_tier=model_tier,
        role_hint="independent cross-domain partner critic",
    )
    ranked = _validated_ranking(critic, verified)
    eligible = [item for item in ranked if item["hard_gate_passed"]]
    if not eligible:
        raise PartnerSelectionError("critic found no candidate that passed the partner-selection hard gates")

    top_three = eligible[:3]
    return {
        "schema_version": "arc.workflow.cross_domain_partner_selection.v2",
        "anchor": anchor,
        "user_intent": user_intent,
        "context_files": [str(anchor_path)],
        "selection_policy": {
            "candidate_count": 6,
            "history_blind": True,
            "cache_blind": True,
            "score_weights": {
                "bridge_physical_feasibility": 35,
                "transferred_ingredient_specificity": 25,
                "substantive_target_opportunity": 25,
                "semantic_distinctness": 15,
            },
            "semantic_distance_is_diagnostic_only": True,
        },
        "selector_output": selector,
        "verified_candidates": verified,
        "metadata_rejections": rejected,
        "critic_output": critic,
        "ranking": ranked,
        "selected_candidate": top_three[0],
        "fallback_candidates": top_three[1:],
    }


def _selector_prompt(anchor: Mapping[str, Any], *, user_intent: str) -> str:
    return (
        "You are selecting a genuinely distinct theoretical-physics domain to pair with an anchor domain. "
        "Propose exactly six open-world candidates; do not select from a supplied list. Do not formulate complete "
        "research ideas and do not rank the candidates. For each candidate, identify a canonical representative "
        "paper using one exact identifier that ARC paper tools can resolve and verify. Prefer an arXiv identifier "
        "written exactly as arXiv:YYMM.NNNN or arXiv:YYMM.NNNNN for new-style papers, or "
        "arXiv:archive/YYMMNNN for old-style papers. An exact doi:10.... identifier or inspire:<numeric-recid> "
        "is also acceptable. Do not return a paper title, author name, URL, journal citation, placeholder, guessed "
        "identifier, or field label in representative_seed. If the identifier is uncertain, choose another "
        "canonical paper whose exact ID is known. Use six distinct representative_seed values. For each candidate, "
        "identify a "
        "specific transferable ingredient, the target capability gap, the required translation, a bounded first "
        "calculation, and physical compatibility risks. Either transfer direction is allowed. Only the target must "
        "receive a substantive contribution; the source may supply a mature method. Reject same-subfield relabeling. "
        "Semantic distance is diagnostic and is not valuable by itself.\n\n"
        f"User intent:\n{user_intent}\n\nAnchor domain card:\n{json.dumps(anchor, ensure_ascii=False)}"
    )


def _anchor_field(
    payload: Mapping[str, Any],
    *,
    source_path: Path,
    requested_field_id: str | None,
) -> dict[str, Any]:
    if payload.get("schema_version") == "arc.workflow.domain_manifest.v2":
        groups = payload.get("field_groups")
        if not isinstance(groups, list) or not groups:
            raise PartnerSelectionError("domain manifest v2 requires non-empty field_groups")
        requested = str(requested_field_id or "").strip()
        if not requested:
            if len(groups) != 1:
                raise PartnerSelectionError("--anchor-field-id is required when the domain manifest has multiple fields")
            requested = str(groups[0].get("field_id", "")) if isinstance(groups[0], Mapping) else ""
        group = next(
            (item for item in groups if isinstance(item, Mapping) and str(item.get("field_id", "")) == requested),
            None,
        )
        if group is None:
            raise PartnerSelectionError(f"anchor field {requested!r} is absent from the domain manifest")
        card = group.get("field_card")
        members = group.get("domain_package_ids")
        if not isinstance(card, Mapping) or not isinstance(members, list) or not members:
            raise PartnerSelectionError(f"anchor field {requested!r} lacks its field card or package provenance")
        return {
            "field_id": requested,
            "domain_package_ids": [str(item) for item in members],
            "field_card": dict(card),
            "source_path": str(source_path),
        }
    if payload.get("field_id") and isinstance(payload.get("field_card"), Mapping):
        members = payload.get("domain_package_ids", [])
        if not isinstance(members, list):
            raise PartnerSelectionError("anchor domain_package_ids must be an array")
        return {
            "field_id": str(payload["field_id"]),
            "domain_package_ids": [str(item) for item in members],
            "field_card": dict(payload["field_card"]),
            "source_path": str(source_path),
        }
    # Read-only compatibility for a single legacy summary. New artifacts still use
    # the v2 field-card contract so the pair writer can consume them uniformly.
    legacy_id = str(payload.get("domain_id", "")).strip()
    if not legacy_id:
        raise PartnerSelectionError("anchor input must be a v2 domain manifest, field card, or domain summary")
    return {
        "field_id": str(requested_field_id or legacy_id),
        "domain_package_ids": [legacy_id],
        "field_card": {
            "titles": [str(payload.get("domain_title", ""))],
            "overviews": [str(payload.get("overview") or payload.get("brief_introduction") or "")],
            "task_focus": [payload.get("task_focus", {})],
            "methodology": payload.get("methodology", []),
            "summary_json_paths": [str(source_path)],
        },
        "source_path": str(source_path),
        "compatibility_source": "legacy_domain_summary",
    }


def _critic_prompt(anchor: Mapping[str, Any], candidates: list[dict[str, Any]], *, user_intent: str) -> str:
    return (
        "Independently audit the candidate partner domains below. You did not generate them and must not infer cache "
        "availability or prior ARC runs. Score physical bridge feasibility out of 35, transferred-ingredient "
        "specificity out of 25, potential for a substantive contribution in one target domain out of 25, and "
        "semantic distinctness out of 15. Semantic distance alone is not a benefit. A hard-gate pass requires a valid "
        "representative paper, a non-decorative translation map, a bounded first calculation, and a domain genuinely "
        "outside the anchor's own subfield. Rank all candidates by total score after applying the hard gates.\n\n"
        f"User intent:\n{user_intent}\n\nAnchor domain card:\n{json.dumps(anchor, ensure_ascii=False)}\n\n"
        f"Verified candidates:\n{json.dumps(candidates, ensure_ascii=False)}"
    )


def _validated_ranking(critic: Mapping[str, Any], verified: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_seed = {str(item["representative_seed"]): item for item in verified}
    raw_ranking = critic.get("ranked_candidates", [])
    if not isinstance(raw_ranking, list):
        raise PartnerSelectionError("critic ranked_candidates must be an array")
    ranking: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_ranking:
        if not isinstance(raw, dict):
            continue
        seed = str(raw.get("representative_seed", "")).strip()
        if seed not in by_seed or seed in seen:
            continue
        seen.add(seed)
        entry = dict(raw)
        entry["total_score"] = sum(
            int(entry[field])
            for field in (
                "bridge_physical_feasibility",
                "transferred_ingredient_specificity",
                "substantive_target_opportunity",
                "semantic_distinctness",
            )
        )
        entry["candidate"] = by_seed[seed]
        ranking.append(entry)
    ranking.sort(key=lambda item: (bool(item["hard_gate_passed"]), item["total_score"]), reverse=True)
    if not ranking:
        raise PartnerSelectionError("critic did not rank any verified candidate")
    if len(ranking) != len(verified):
        raise PartnerSelectionError("critic must rank every metadata-verified candidate exactly once")
    return ranking


def _fetch_metadata(seed: str) -> Mapping[str, Any]:
    from arc_paper import service as paper_service

    result = paper_service.get_metadata(seed)
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        error = result.get("error", {}) if isinstance(result, Mapping) else {}
        raise PartnerSelectionError(str(error.get("message") or f"metadata lookup failed for {seed}"))
    data = result.get("data")
    if not isinstance(data, Mapping):
        raise PartnerSelectionError(f"metadata lookup returned no data for {seed}")
    return data


def _default_json_runner() -> JsonRunner:
    from arc_llm import run_json

    return run_json


def _format_rejections(rejected: list[dict[str, str]]) -> str:
    if not rejected:
        return "none recorded"
    return "; ".join(
        f"{item.get('representative_seed', '<missing>')!r}: {item.get('reason', 'unknown_reason')}"
        for item in rejected
    )


def _require_strict_source_mode() -> None:
    if not str(os.environ.get(ARC_REQUIRE_REPO_ROOT, "")).strip():
        raise PartnerSelectionError(
            f"{ARC_REQUIRE_REPO_ROOT} is required for cross-domain partner selection"
        )


def _reject_historical_path(path: Path) -> None:
    parts = path.parts
    for forbidden in HISTORY_PATH_PARTS:
        if any(tuple(parts[index : index + len(forbidden)]) == forbidden for index in range(len(parts))):
            raise PartnerSelectionError(f"historical ARC artifact is forbidden as selector input: {path}")


def _workflow_json(name: str) -> dict[str, Any]:
    return _read_object(Path(__file__).resolve().parents[1] / "json" / name)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PartnerSelectionError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PartnerSelectionError(f"JSON root must be an object: {path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Select a history-blind cross-domain partner for an ARC domain.")
    parser.add_argument("--anchor-summary", required=True)
    parser.add_argument("--user-intent", required=True)
    parser.add_argument("--anchor-field-id")
    parser.add_argument("--output", required=True)
    parser.add_argument("--provider", default="auto")
    parser.add_argument("--model")
    parser.add_argument("--model-tier", choices=["low", "medium", "high", "max"], default="high")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = select_partner(
            Path(args.anchor_summary),
            user_intent=args.user_intent,
            anchor_field_id=args.anchor_field_id,
            provider=args.provider,
            model=args.model,
            model_tier=args.model_tier,
        )
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        summary = {
            "status": "completed",
            "output": str(output),
            "selected_seed": result["selected_candidate"]["representative_seed"],
            "fallback_seeds": [item["representative_seed"] for item in result["fallback_candidates"]],
        }
        print(json.dumps(summary, ensure_ascii=False) if args.json else str(output))
        return 0
    except PartnerSelectionError as exc:
        result = {"status": "failed", "error": str(exc)}
        print(json.dumps(result, ensure_ascii=False) if args.json else f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
