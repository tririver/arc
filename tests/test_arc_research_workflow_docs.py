from __future__ import annotations

import json
from pathlib import Path


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


def test_packaged_workflow_copies_match_source() -> None:
    for host in ["codex", "claude"]:
        packaged = ROOT / f"packaging/{host}/arc/skills/arc/references/research-workflows"
        for path in WF.glob("research-*"):
            assert (packaged / path.name).read_text(encoding="utf-8") == path.read_text(
                encoding="utf-8"
            )
