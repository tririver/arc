from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import jsonschema


ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "skills/arc/references/research-workflows"


def test_research_calculation_workflow_files_exist() -> None:
    for name in [
        "research-plan.md",
        "research-foundation.md",
        "research-execute.md",
        "research-plan.schema.json",
        "research-foundation.schema.json",
        "research-execute.schema.json",
    ]:
        assert (WF / name).is_file()


def test_arc_skill_routes_case_3_to_three_research_workflows() -> None:
    text = (ROOT / "skills/arc/SKILL.md").read_text(encoding="utf-8")

    assert "references/research-workflows/calculate.md" not in text
    assert "research-plan.md" in text
    assert "research-foundation.md" in text
    assert "research-execute.md" in text


def test_research_workflow_docs_stay_human_readable() -> None:
    for name in ["research-plan.md", "research-foundation.md", "research-execute.md"]:
        text = (WF / name).read_text(encoding="utf-8")
        assert len(text.splitlines()) <= 220
        assert "0_ref/" not in text
        assert "/scripts/" not in text


def test_research_plan_requires_review_after_drafting() -> None:
    text = (WF / "research-plan.md").read_text(encoding="utf-8")

    assert "review the plan" in text.lower()
    assert "independent reviewer" in text.lower()
    assert "main agent" in text.lower()


def test_research_plan_requires_explicit_step_quantity_contracts() -> None:
    text = (WF / "research-plan.md").read_text(encoding="utf-8").lower()

    assert "calculate which quantity" in text
    assert "in terms of which quantity" in text
    assert "end of every step" in text


def test_research_foundation_requires_convention_alignment_checks() -> None:
    text = (WF / "research-foundation.md").read_text(encoding="utf-8").lower()

    assert "consistent convention" in text
    assert "multiple papers" in text
    assert "convention_check" in text
    assert "check loop" in text


def test_research_execute_requires_solid_symbolic_and_filtered_checks() -> None:
    text = (WF / "research-execute.md").read_text(encoding="utf-8").lower()

    assert "integrity.md" in text
    assert "expand" in text
    assert "simplify" in text
    assert "substitutions" in text
    assert "10 randomly selected data points" in text
    assert "relative error" in text
    assert "check history" in text
    assert "axiom and checked" in text
    assert "unchecked" in text
    assert "internet" in text
    assert "paper tools" in text
    assert "wolfram" in text


def test_research_workflow_filter_script_exists() -> None:
    script = WF / "scripts/filter-foundation-context.py"

    assert script.is_file()
    text = script.read_text(encoding="utf-8")
    assert "target_equation_id" in text
    assert "omitted_equation_ids" in text


def test_research_workflow_filter_script_omits_unchecked_context(tmp_path) -> None:
    foundation = {
        "schema_version": "arc.research_foundation.v1",
        "run_id": "run_001",
        "version": 1,
        "conventions": [
            {"id": "conv_checked", "check_status": "checked", "consistency_status": "normalized"},
            {"id": "conv_unchecked", "check_status": "not_checked", "consistency_status": "normalized"},
        ],
        "equations": [
            {
                "id": "eq_axiom",
                "axiom_status": "axiom",
                "check_status": "not_checked",
                "sources": [{"paper_id": "arXiv:1", "mcp": "get_section(...)", "cli": "arc-paper ..."}],
            },
            {
                "id": "eq_target",
                "axiom_status": "not_axiom",
                "check_status": "not_checked",
                "sources": [{"paper_id": "arXiv:2", "mcp": "get_section(...)", "cli": "arc-paper ..."}],
            },
            {"id": "eq_unchecked", "axiom_status": "not_axiom", "check_status": "not_checked"},
        ],
    }
    foundation_path = tmp_path / "foundation.json"
    foundation_path.write_text(json.dumps(foundation), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(WF / "scripts/filter-foundation-context.py"),
            str(foundation_path),
            "--target-equation-id",
            "eq_target",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    filtered = json.loads(result.stdout)
    assert "source_path" not in filtered
    assert "sources" not in result.stdout
    assert "arc-paper" not in result.stdout
    assert [item["id"] for item in filtered["allowed_conventions"]] == ["conv_checked"]
    assert [item["id"] for item in filtered["allowed_equations"]] == ["eq_axiom"]
    assert filtered["target_equation"]["id"] == "eq_target"
    assert filtered["omitted_equation_ids"] == ["eq_unchecked"]


def test_research_workflow_schemas_are_valid_json_and_referenced() -> None:
    expected = {
        "research-plan": "arc.research_plan.v1",
        "research-foundation": "arc.research_foundation.v1",
        "research-execute": "arc.research_execute.v1",
    }

    for stem, schema_version in expected.items():
        schema = json.loads((WF / f"{stem}.schema.json").read_text(encoding="utf-8"))
        markdown = (WF / f"{stem}.md").read_text(encoding="utf-8")
        assert schema["properties"]["schema_version"]["const"] == schema_version
        assert schema_version in markdown


def test_research_foundation_schema_requires_evidence_for_checked_equations() -> None:
    schema = json.loads((WF / "research-foundation.schema.json").read_text(encoding="utf-8"))
    equation = {
        "id": "eq_001",
        "label": "checked result",
        "latex": "x=y",
        "role": "useful_result",
        "axiom_status": "not_axiom",
        "publication_status": "published_low",
        "citation_count": 1,
        "check_status": "checked_numerical",
        "judgment": "reasonable",
        "sources": [{"paper_id": "arXiv:1", "section": "S1", "mcp": "get_section(...)", "cli": "arc-paper ..."}],
    }
    document = {
        "schema_version": "arc.research_foundation.v1",
        "run_id": "run_001",
        "version": 2,
        "created_from_plan": "plan.json",
        "conventions": [],
        "equations": [equation],
    }

    validator = jsonschema.Draft202012Validator(schema)
    assert list(validator.iter_errors(document))

    equation.update(
        {
            "check_method": "numerical",
            "check_history": ["expanded first; analytic check failed; sampled 10 points"],
            "numerical_relative_error": 1e-8,
            "consensus_artifact": "execute/run/state.json",
        }
    )
    assert list(validator.iter_errors(document)) == []


def test_packaged_workflow_copies_match_source() -> None:
    for host in ["codex", "claude"]:
        packaged = ROOT / f"packaging/{host}/arc/skills/arc/references/research-workflows"
        for path in WF.glob("research-*"):
            assert (packaged / path.name).read_text(encoding="utf-8") == path.read_text(
                encoding="utf-8"
            )
        script = Path("scripts/filter-foundation-context.py")
        assert (packaged / script).read_text(encoding="utf-8") == (WF / script).read_text(
            encoding="utf-8"
        )
