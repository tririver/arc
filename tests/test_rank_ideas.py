from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "plugins/arc/skills/arc/workflows/scripts/rank-ideas.py"


def _load_rank_module() -> Any:
    spec = importlib.util.spec_from_file_location("rank_ideas", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    old_dont_write_bytecode = sys.dont_write_bytecode
    old_path = list(sys.path)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old_dont_write_bytecode
        sys.path[:] = old_path
    return module


def test_markdown_summary_uses_round_marks_by_idea_format(tmp_path: Path) -> None:
    ranker = _load_rank_module()
    run_root = tmp_path / "decoherence_bell" / "idea_loops"
    (tmp_path / "decoherence_bell.config.json").write_text(
        json.dumps({"user_intent": "suggest research directions about decoherence and Bell inequality test in cosmology"}),
        encoding="utf-8",
    )
    _write_round(
        run_root,
        "domain_idea_001",
        1,
        title="Lower scoring idea",
        marks={
            "user_intent_relevance": 20,
            "novelty": 8,
            "confidence_of_novelty": 7,
            "scientific_value": 9,
            "planning": 10,
            "problem_well_definedness": 10,
            "total_score": 64,
        },
    )
    _write_round(
        run_root,
        "domain_idea_002",
        1,
        title="Higher scoring idea",
        marks={
            "user_intent_relevance": 24,
            "novelty": 10,
            "confidence_of_novelty": 8,
            "scientific_value": 12,
            "planning": 14,
            "problem_well_definedness": 14,
            "total_score": 82,
        },
    )
    _write_round(
        run_root,
        "domain_idea_002",
        2,
        title="Higher scoring idea",
        marks={
            "user_intent_relevance": 25,
            "novelty": 10,
            "confidence_of_novelty": 8,
            "scientific_value": 12,
            "planning": 14,
            "problem_well_definedness": 14,
            "total_score": 83,
        },
    )
    _write_round(
        run_root,
        "domain_idea_003",
        1,
        title="Intent-bearing idea",
        marks={
            "user_intent_relevance": 25,
            "novelty": 10,
            "confidence_of_novelty": 8,
            "scientific_value": 12,
            "planning": 14,
            "problem_well_definedness": 14,
            "total_score": 84,
        },
    )

    markdown = ranker.markdown_table(ranker.rank_run(run_root))
    first_section = markdown.split("# Appendix: Idea Details", 1)[0]

    assert first_section.startswith(
        "# Ideas\n\n"
        "Abbreviations:\n\n"
        "IR=intent relevance, N=novelty, CN=confidence of novelty, SV=scientific value, "
        "PL=planning, WD=well-definedness, T=total.\n\n"
        "## `domain_idea_001`\n\n"
        "Lower scoring idea\n\n"
        "| Round | IR | N | CN | SV | PL | WD | T |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        "| 1 | 20 | 8 | 7 | 9 | 10 | 10 | 64 |\n\n"
        "## `domain_idea_002`\n\n"
        "Higher scoring idea\n\n"
        "| Round | IR | N | CN | SV | PL | WD | T |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        "| 1 | 24 | 10 | 8 | 12 | 14 | 14 | 82 |\n"
        "| 2 | 25 | 10 | 8 | 12 | 14 | 14 | 83 |\n\n"
        "## `domain_idea_003`\n\n"
        "Intent-bearing idea\n\n"
        "| Round | IR | N | CN | SV | PL | WD | T |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        "| 1 | 25 | 10 | 8 | 12 | 14 | 14 | 84 |"
    )
    assert markdown.startswith("# Ideas\n")
    assert "Higher scoring idea (Mark: 83)" not in first_section
    assert "\n# Ranked Ideas and Details\n" not in markdown
    assert "\n# Appendix: Idea Details\n" in markdown
    details_section = markdown.split("# Appendix: Idea Details", 1)[1]
    assert details_section.index("### 1. Intent-bearing idea") < details_section.index("### 2. Higher scoring idea")
    assert details_section.index("### 2. Higher scoring idea") < details_section.index("### 3. Lower scoring idea")
    assert "#### Full Idea Verbatim" in markdown
    assert "```text" not in markdown
    assert "Title: Higher scoring idea" in markdown
    assert "Idea Summary: summary" in markdown
    assert "Calculation Plan: Compute $ρ_{E}=⟨T_{ab}⟩u^a u^b$." in markdown
    assert "Raw $T_{ab}$, $η_{SL}$, and $G^a_{b}$." in markdown
    assert "Geometry $δ_{ij}$ and $α ∈ {0,0.3}$." in markdown
    assert "Explicit commands $\\rho_{E}$ and $\\partial^a T_{ab}=0$ stay unchanged." in markdown
    assert "$$\nT_{kk}(t,ρ,z)=A q(t/τ) sech(z/L)\n$$" in markdown
    assert "$$\nΔT(0,b_{ref}) = -4 G ∫ d^4x T_{kk}(x)\n$$" in markdown
    assert "$$\nE(α,β;N)=E_diag(N)+2\\,\\Re[z]\n$$" in markdown
    assert "$$\n$$\nE(α,β;N)" not in markdown
    assert "\\operatorname" not in markdown


def test_rank_run_zeroes_major_recovered_reviewer_marks(tmp_path: Path) -> None:
    ranker = _load_rank_module()
    run_root = tmp_path / "ideas" / "run_001"
    _write_round(
        run_root,
        "domain_idea_001",
        1,
        title="Clean lower round",
        marks={
            "user_intent_relevance": 20,
            "novelty": 8,
            "confidence_of_novelty": 7,
            "scientific_value": 9,
            "planning": 10,
            "problem_well_definedness": 10,
            "total_score": 64,
        },
    )
    _write_round(
        run_root,
        "domain_idea_001",
        2,
        title="Recovered high round",
        marks={
            "user_intent_relevance": 25,
            "novelty": 10,
            "confidence_of_novelty": 10,
            "scientific_value": 15,
            "planning": 15,
            "problem_well_definedness": 15,
            "total_score": 90,
        },
        review_extra={
            "arc_llm_call_record": {
                "structured_output": {"mode": "recovered", "severity": "major"}
            }
        },
    )

    entry = ranker.rank_run(run_root)["ranking"][0]
    recovered_round = next(item for item in entry["rounds"] if item["round"] == 2)

    assert entry["round"] == 1
    assert recovered_round["marks"]["total_score"] == 0
    assert all(value == 0 for value in recovered_round["marks"].values())


def test_rank_run_zeroes_major_recovered_proposer_marks(tmp_path: Path) -> None:
    ranker = _load_rank_module()
    run_root = tmp_path / "ideas" / "run_001"
    _write_round(
        run_root,
        "domain_idea_001",
        1,
        title="Clean lower round",
        marks={
            "user_intent_relevance": 20,
            "novelty": 8,
            "confidence_of_novelty": 7,
            "scientific_value": 9,
            "planning": 10,
            "problem_well_definedness": 10,
            "total_score": 64,
        },
    )
    _write_round(
        run_root,
        "domain_idea_001",
        2,
        title="Recovered proposer high round",
        marks={
            "user_intent_relevance": 25,
            "novelty": 10,
            "confidence_of_novelty": 10,
            "scientific_value": 15,
            "planning": 15,
            "problem_well_definedness": 15,
            "total_score": 90,
        },
        proposer_extra={
            "arc_llm_call_record": {
                "structured_output": {"mode": "recovered", "severity": "major"}
            }
        },
    )

    entry = ranker.rank_run(run_root)["ranking"][0]
    recovered_round = next(item for item in entry["rounds"] if item["round"] == 2)

    assert entry["round"] == 1
    assert recovered_round["marks"]["total_score"] == 0
    assert all(value == 0 for value in recovered_round["marks"].values())


def test_rank_run_includes_unstructured_round_without_marks(tmp_path: Path) -> None:
    ranker = _load_rank_module()
    run_root = tmp_path / "ideas" / "run_001"
    round_root = run_root / "loops/domain_idea_001/rounds/round_001"
    proposer_dir = round_root / "proposer_outputs"
    review_dir = round_root / "reviews"
    proposer_dir.mkdir(parents=True)
    review_dir.mkdir(parents=True)
    (proposer_dir / "proposer_001.json").write_text(
        json.dumps(
            {
                "schema_version": "arc.llm.unstructured_output.v1",
                "warning": "Output did not satisfy the requested JSON schema.",
                "raw_text": "Recovered natural-language idea",
            }
        ),
        encoding="utf-8",
    )
    (review_dir / "reviewer_001.json").write_text(
        json.dumps({"review_payload": {"comments": "marks missing"}}),
        encoding="utf-8",
    )

    entry = ranker.rank_run(run_root)["ranking"][0]

    assert entry["title"] == "Output did not satisfy the requested JSON schema."
    assert entry["marks"]["total_score"] == 0
    assert all(value == 0 for value in entry["marks"].values())


def test_cross_domain_rank_uses_qualification_before_score(tmp_path: Path) -> None:
    ranker = _load_rank_module()
    run_root = tmp_path / "ideas" / "run_001" / "idea_loops"
    _write_cross_config(run_root, ["cross_domain_idea_001", "cross_domain_idea_002"])
    _write_cross_round(
        run_root,
        "cross_domain_idea_001",
        1,
        title="Decorative high score",
        total=96,
        assessment=_cross_assessment(transfer_status="decorative"),
    )
    _write_cross_round(
        run_root,
        "cross_domain_idea_001",
        2,
        title="Genuine lower score",
        total=75,
        assessment=_cross_assessment(signature_suffix="one"),
    )
    _write_cross_round(
        run_root,
        "cross_domain_idea_002",
        1,
        title="Incremental target",
        total=99,
        assessment=_cross_assessment(target_contribution_status="incremental", signature_suffix="two"),
    )

    payload = ranker.rank_run(run_root)

    assert payload["schema_version"] == "arc.ideas.selected_rounds.v2"
    assert payload["ranking"][0]["title"] == "Genuine lower score"
    assert payload["ranking"][0]["round"] == 2
    assert payload["unqualified"][0]["title"] == "Incremental target"
    assert "target_contribution_must_be_substantial_or_transformative" in payload["unqualified"][0][
        "qualification_reasons"
    ]
    assert payload["warnings"]
    markdown = ranker.markdown_table(payload)
    assert "# Appendix: Unqualified Cross-Domain Candidates" in markdown
    assert "Decorative high score" not in markdown.split("# Appendix: Idea Details", 1)[0]


def test_cross_domain_top_three_requires_distinct_transfer_signatures(tmp_path: Path) -> None:
    ranker = _load_rank_module()
    run_root = tmp_path / "ideas" / "run_001" / "idea_loops"
    loop_ids = [f"cross_domain_idea_{index:03d}" for index in range(1, 5)]
    _write_cross_config(run_root, loop_ids)
    signatures = ["same", "same", "different-a", "different-b"]
    for index, (loop_id, signature) in enumerate(zip(loop_ids, signatures), start=1):
        _write_cross_round(
            run_root,
            loop_id,
            1,
            title=f"Idea {index}",
            total=90 - index,
            assessment=_cross_assessment(signature_suffix=signature),
        )

    payload = ranker.rank_run(run_root)

    assert [entry["title"] for entry in payload["top_three"]] == ["Idea 1", "Idea 3", "Idea 4"]
    assert [entry["title"] for entry in payload["ranking"][:3]] == ["Idea 1", "Idea 3", "Idea 4"]
    assert payload["diagnostics"]["distinct_qualified_transfer_signatures"] == 3


def test_cross_domain_cli_writes_diagnostics(tmp_path: Path) -> None:
    run_root = tmp_path / "ideas" / "run_001" / "idea_loops"
    _write_cross_config(run_root, ["cross_domain_idea_001"])
    _write_cross_round(
        run_root,
        "cross_domain_idea_001",
        1,
        title="Qualified",
        total=80,
        assessment=_cross_assessment(),
    )

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(run_root), "--format", "json"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    diagnostics_path = run_root.parent / "cross-domain-diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["schema_version"] == "arc.ideas.cross_domain_diagnostics.v1"
    assert diagnostics["qualified_count"] == 1


def test_cross_domain_portfolio_caps_one_central_mechanism_at_two(tmp_path: Path) -> None:
    ranker = _load_rank_module()
    run_root = tmp_path / "ideas" / "run_001" / "idea_loops"
    loop_ids = [f"cross_domain_idea_{index:03d}" for index in range(1, 4)]
    _write_cross_config(run_root, loop_ids)
    for index, loop_id in enumerate(loop_ids, start=1):
        assessment = _cross_assessment(signature_suffix=f"result-{index}")
        assessment["transfer_signature"]["transferred_ingredient"] = "same central ingredient"
        _write_cross_round(
            run_root,
            loop_id,
            1,
            title=f"Mechanism idea {index}",
            total=91 - index,
            assessment=assessment,
        )

    payload = ranker.rank_run(run_root)

    assert [entry["title"] for entry in payload["ranking"]] == ["Mechanism idea 1", "Mechanism idea 2"]
    assert [entry["title"] for entry in payload["portfolio_excluded"]] == ["Mechanism idea 3"]
    assert payload["diagnostics"]["portfolio_excluded_count"] == 1


def test_cross_domain_manageable_compatibility_risk_does_not_block_qualification(tmp_path: Path) -> None:
    ranker = _load_rank_module()
    run_root = tmp_path / "ideas" / "run_001" / "idea_loops"
    _write_cross_config(run_root, ["cross_domain_idea_001"])
    _write_cross_round(
        run_root,
        "cross_domain_idea_001",
        1,
        title="Feasible with a bounded risk",
        total=82,
        assessment=_cross_assessment(
            feasibility_status="feasible_with_named_risk",
            manageable_compatibility_risks=["The first calculation must test the translation regime."],
        ),
    )

    payload = ranker.rank_run(run_root)

    assert [entry["title"] for entry in payload["ranking"]] == ["Feasible with a bounded risk"]
    assert payload["ranking"][0]["compatibility_classification"] == {
        "policy": "explicit_blocking_and_manageable_v2",
        "blocking_failures": [],
        "manageable_risks": ["The first calculation must test the translation regime."],
    }


def test_cross_domain_blocking_compatibility_failure_remains_a_hard_gate(tmp_path: Path) -> None:
    ranker = _load_rank_module()
    run_root = tmp_path / "ideas" / "run_001" / "idea_loops"
    _write_cross_config(run_root, ["cross_domain_idea_001"])
    _write_cross_round(
        run_root,
        "cross_domain_idea_001",
        1,
        title="Physically incompatible",
        total=95,
        assessment=_cross_assessment(
            feasibility_status="feasible_with_named_risk",
            blocking_compatibility_failures=["The source and target regimes contradict a conservation law."],
            manageable_compatibility_risks=["A numerical coefficient remains uncertain."],
        ),
    )

    payload = ranker.rank_run(run_root)

    assert payload["ranking"] == []
    assert payload["unqualified"][0]["title"] == "Physically incompatible"
    assert "blocking_compatibility_failures" in payload["unqualified"][0]["qualification_reasons"]


def test_cross_domain_legacy_compatibility_field_uses_feasibility_status(tmp_path: Path) -> None:
    ranker = _load_rank_module()
    run_root = tmp_path / "ideas" / "run_001" / "idea_loops"
    loop_ids = ["cross_domain_idea_001", "cross_domain_idea_002"]
    _write_cross_config(run_root, loop_ids)
    legacy_risk = ["Legacy reviewer named a calculation-bounded compatibility risk."]
    _write_cross_round(
        run_root,
        loop_ids[0],
        1,
        title="Legacy named risk",
        total=82,
        assessment=_cross_assessment(
            feasibility_status="feasible_with_named_risk",
            legacy_compatibility_failures=legacy_risk,
            signature_suffix="legacy-risk",
        ),
    )
    _write_cross_round(
        run_root,
        loop_ids[1],
        1,
        title="Legacy blocking failure",
        total=90,
        assessment=_cross_assessment(
            feasibility_status="feasible",
            legacy_compatibility_failures=["Legacy unresolved incompatibility."],
            signature_suffix="legacy-blocking",
        ),
    )

    payload = ranker.rank_run(run_root)

    assert [entry["title"] for entry in payload["ranking"]] == ["Legacy named risk"]
    assert payload["ranking"][0]["compatibility_classification"] == {
        "policy": "legacy_compatibility_failures_as_named_risks",
        "blocking_failures": [],
        "manageable_risks": legacy_risk,
    }
    failed = payload["unqualified"][0]
    assert failed["title"] == "Legacy blocking failure"
    assert failed["compatibility_classification"]["policy"] == "legacy_compatibility_failures_as_blocking"
    assert "blocking_compatibility_failures" in failed["qualification_reasons"]


def _write_cross_config(run_root: Path, loop_ids: list[str]) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    cards = [{"domain_id": "domain-a"}, {"domain_id": "domain-b"}]
    (run_root / "config.json").write_text(
        json.dumps(
            {
                "loops": [
                    {
                        "loop_id": loop_id,
                        "caller_context": {
                            "variant_id": "cross_domain",
                            "generation_mode": "cross_domain",
                            "domain_cards": cards,
                        },
                    }
                    for loop_id in loop_ids
                ]
            }
        ),
        encoding="utf-8",
    )


def _cross_assessment(
    *,
    transfer_status: str = "genuine",
    target_contribution_status: str = "substantial",
    feasibility_status: str = "feasible",
    signature_suffix: str = "default",
    blocking_compatibility_failures: list[str] | None = None,
    manageable_compatibility_risks: list[str] | None = None,
    legacy_compatibility_failures: list[str] | None = None,
) -> dict[str, Any]:
    assessment = {
        "source_domain_id": "domain-a",
        "target_domain_id": "domain-b",
        "transfer_status": transfer_status,
        "target_contribution_status": target_contribution_status,
        "source_ingredient_validity": "valid",
        "target_adaptation_validity": "valid",
        "resulting_new_capability": "new capability",
        "feasibility_status": feasibility_status,
        "blocking_compatibility_failures": blocking_compatibility_failures or [],
        "manageable_compatibility_risks": manageable_compatibility_risks or [],
        "novelty_coverage": {"source_domain": True, "target_domain": True, "intersection": True},
        "disqualifying_reasons": [],
        "recommended_action": "refine_current",
        "transfer_signature": {
            "direction": "domain-a to domain-b",
            "transferred_ingredient": f"ingredient {signature_suffix}",
            "target_result": f"result {signature_suffix}",
            "first_calculation": f"calculation {signature_suffix}",
        },
    }
    if legacy_compatibility_failures is not None:
        assessment.pop("blocking_compatibility_failures")
        assessment.pop("manageable_compatibility_risks")
        assessment["compatibility_failures"] = legacy_compatibility_failures
    return assessment


def _write_cross_round(
    run_root: Path,
    loop_id: str,
    round_number: int,
    *,
    title: str,
    total: int,
    assessment: dict[str, Any],
) -> None:
    marks = {
        "user_intent_relevance": 12,
        "cross_domain_transfer_quality": 12,
        "substantive_target_contribution": 16,
        "novelty": 8,
        "confidence_of_novelty": 7,
        "scientific_value": 8,
        "calculation_feasibility": 7,
        "problem_well_definedness": 7,
        "total_score": total,
    }
    _write_round(
        run_root,
        loop_id,
        round_number,
        title=title,
        marks=marks,
        proposer_extra={
            "domain_roles": {
                "source_domain_id": "domain-a",
                "target_domain_id": "domain-b",
                "supporting_domain_ids": [],
            }
        },
        review_extra={"review_payload": {"marks": marks, "cross_domain_assessment": assessment}},
    )


def _write_round(
    run_root: Path,
    loop_id: str,
    round_number: int,
    *,
    title: str,
    marks: dict[str, int],
    proposer_extra: dict[str, Any] | None = None,
    review_extra: dict[str, Any] | None = None,
) -> None:
    round_root = run_root / "loops" / loop_id / "rounds" / f"round_{round_number:03d}"
    proposer_dir = round_root / "proposer_outputs"
    review_dir = round_root / "reviews"
    proposer_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    proposer_payload = {
        "title": title,
        "idea_summary": "summary",
        "calculation_plan": (
            "Compute `ρ_E=⟨T_ab⟩u^a u^b`. Raw T_ab, η_SL, and G^a_b. "
            "Geometry `δ_ij` and `α ∈ {0,0.3}`. "
            "Explicit commands `\\rho_E` and `\\partial^a T_ab=0` stay unchanged.\n\n"
            "T_kk(t,ρ,z)=A q(t/τ) sech(z/L)\n\n"
            "ΔT(0,b_ref) = -4 G ∫ d^4x T_kk(x),\n\n"
            "$$\nE(α,β;N)=E_diag(N)+2\\,\\Re[z]\n$$"
        ),
    }
    if proposer_extra:
        proposer_payload.update(proposer_extra)
    (proposer_dir / "proposer_001.json").write_text(json.dumps(proposer_payload), encoding="utf-8")
    review_payload = {"review_payload": {"marks": marks}}
    if review_extra:
        review_payload.update(review_extra)
    (review_dir / "reviewer_001.json").write_text(json.dumps(review_payload), encoding="utf-8")
