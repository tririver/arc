from __future__ import annotations

import importlib.util
import json
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
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old_dont_write_bytecode
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


def _write_round(run_root: Path, loop_id: str, round_number: int, *, title: str, marks: dict[str, int]) -> None:
    round_root = run_root / "loops" / loop_id / "rounds" / f"round_{round_number:03d}"
    proposer_dir = round_root / "proposer_outputs"
    review_dir = round_root / "reviews"
    proposer_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    (proposer_dir / "proposer_001.json").write_text(
        json.dumps(
            {
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
        ),
        encoding="utf-8",
    )
    (review_dir / "reviewer_001.json").write_text(
        json.dumps({"review_payload": {"marks": marks}}),
        encoding="utf-8",
    )
