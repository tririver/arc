from __future__ import annotations

import json
import importlib
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "skills/arc/workflows"
WJ = WF / "json"
WS = WF / "scripts"
SKILL = ROOT / "skills/arc"


def test_calculation_workflow_files_exist() -> None:
    for name in ["plan.md", "calculate.md", "check.md"]:
        assert (WF / name).is_file()
    assert not (WF / "foundation.md").exists()
    for name in ["plan.schema.json", "foundation.schema.json", "calculate.schema.json"]:
        assert not (WJ / name).exists()
    assert not (WS / "filter-foundation-context.py").exists()


def test_arc_skill_routes_check_and_calculation_workflows() -> None:
    text = (ROOT / "skills/arc/SKILL.md").read_text(encoding="utf-8")

    assert "references/" not in text
    assert "classify" in text.lower()
    assert "four cases" in text.lower()
    assert "check.md" in text
    assert "plan.md" in text
    assert "calculate.md" in text
    assert "foundation.md" not in text
    assert "work-note.md" in text


def test_arc_skill_requires_nonblocking_pdf_export_for_project_markdown_reports() -> None:
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8").lower()

    assert "md2pdf" in text
    assert "background" in text
    assert "do not wait" in text
    assert "markdown report" in text


def test_workflows_start_pdf_export_for_user_facing_markdown() -> None:
    for name in [
        "check.md",
        "domain.md",
        "ideas.md",
        "plan.md",
        "calculate.md",
    ]:
        text = (WF / name).read_text(encoding="utf-8").lower()
        assert "md2pdf" in text
        assert "background" in text
        assert "do not wait" in text


def test_ideas_phase_5_uses_clean_selection_prompt() -> None:
    text = (WF / "ideas.md").read_text(encoding="utf-8")
    lower = text.lower()

    assert "### Phase 5: Select Next Action" in text
    assert "use the host's discrete" in lower
    assert "option labels must be the raw labels" in lower
    assert "`1`" in text
    assert "`2`" in text
    assert "`3`" in text
    assert "`other`" in text
    assert "or quit" not in lower
    assert "`Let's discuss`" in text
    assert "Do not render numbered-list prefixes inside option labels" in text
    assert "If no discrete selection tool is available, ask only for the idea number" in text
    assert "`other`. Do not print `quit` or `Let's discuss` in the typed fallback" in text


def test_interaction_rules_name_codex_discrete_choice_tool() -> None:
    text = (SKILL / "rules/interaction.md").read_text(encoding="utf-8")
    lower = text.lower()

    assert "`request_user_input`" in text
    assert "codex" in lower
    assert "collaboration mode" in lower
    assert "before printing a typed fallback" in lower
    assert "do not include list numbering inside option labels" in lower


def test_workflow_docs_stay_human_readable() -> None:
    for name in [
        "check.md",
        "plan.md",
        "calculate.md",
    ]:
        text = (WF / name).read_text(encoding="utf-8")
        assert len(text.splitlines()) <= 220
        assert "0_ref/" not in text
        assert "/scripts/" not in text


def test_check_workflow_keeps_notes_out_of_proposer_context() -> None:
    text = (WF / "check.md").read_text(encoding="utf-8").lower()

    assert "markdown or pdf research notes" in text
    assert "full note body" in text
    assert "proposer agents" in text
    assert "claims to check" in text
    assert "blind reference check" in text
    assert "reviewer_reference_claim" in text
    assert "user-specified" in text
    assert "inferred" in text


def test_plan_requires_review_after_drafting() -> None:
    text = (WF / "plan.md").read_text(encoding="utf-8")

    assert "review the plan" in text.lower()
    assert "independent reviewer" in text.lower()
    assert "main agent" in text.lower()
    assert "<project-dir>/calculate/<run-id>/work-notes/work-note-v001.md" in text
    assert "<project-dir>/work-note.md" in text
    assert 'md2pdf(input="<project-dir>/work-note.md")' in text
    assert "<project-dir>/plan.md" not in text


def test_plan_requires_explicit_step_quantity_contracts() -> None:
    text = (WF / "plan.md").read_text(encoding="utf-8").lower()

    assert "calculate which quantity" in text
    assert "in terms of which quantity" in text
    assert "end of every step" in text
    assert "largest coherent chunks" in text
    assert "do not split by raw equation count" in text
    assert "at least 20 steps" not in text
    assert "do not disclose the exact expected expression" in text
    assert "derive the target quantity in terms of named dependencies" in text
    assert "expected final formula" in text


def test_plan_routes_reference_equations_to_blind_checks() -> None:
    text = (WF / "plan.md").read_text(encoding="utf-8").lower()

    assert "do not include the target equation or later text" in text
    assert "blind reference check" in text
    assert "reviewer-only reference claim" in text


def test_plan_workflow_writes_work_note_versions() -> None:
    plan = (WF / "plan.md").read_text(encoding="utf-8")
    plan_lower = plan.lower()

    assert "<project-dir>/work-note.md" in plan
    assert "<project-dir>/calculate/<run-id>/work-notes/work-note-v001.md" in plan
    assert "write immutable version first" in plan_lower
    assert "mirror" in plan_lower
    assert "version" in plan_lower


def test_calculate_workflow_uses_work_note_runtime_artifacts() -> None:
    calculate = (WF / "calculate.md").read_text(encoding="utf-8")

    assert "<project-dir>/work-note.md" in calculate
    assert "<project-dir>/calculate/<run-id>/execute/consensus.config.json" in calculate
    assert "<project-dir>/calculate/<run-id>/execute/<consensus-run-id>/" in calculate
    assert "calculation-report.md" not in calculate
    assert "foundation/latest.json" not in calculate
    assert "latest-plan.md" not in calculate
    assert "note-check-triage.json" not in calculate
    assert "validate-note-check" not in calculate


def test_check_workflow_hands_off_to_work_note() -> None:
    check = (WF / "check.md").read_text(encoding="utf-8")
    check_lower = check.lower()

    assert "planning-request" in check_lower
    assert "calculation-report.md" not in check
    assert "foundation/latest.json" not in check
    assert "latest-plan.md" not in check
    assert "note-check-triage.json" not in check
    assert "validate-note-check" not in check


def test_work_note_declares_required_sections() -> None:
    text = (WF / "plan.md").read_text(encoding="utf-8")
    archive_index = text.find("<project-dir>/calculate/<run-id>/work-notes/work-note-v001.md")
    assert archive_index != -1

    expected_headings = [
        "# Work Note",
        "## Task",
        "## Physics Background And Logic Flow",
        "## Notation And Conventions",
        "## Axioms And Starting Points",
        "## Accepted Derived Results",
        "## Validation-Only References",
        "## Detailed Steps Ready To Calculate",
        "## Rough Steps For Later Planning",
        "## Reviewer-Only Targets",
        "## Calculation Status",
        "## Open Questions",
        "## Revision History",
        "## Journal",
        "## Source Audit Trail",
    ]
    template = text[archive_index:]
    work_note_match = re.search(r"(?m)^# Work Note$", template)
    assert work_note_match is not None

    template_body = template[work_note_match.start():]
    template_end = template_body.find("Each equation-heavy section")
    if template_end == -1:
        template_end = template_body.find("```", len("# Work Note"))
    assert template_end != -1

    headings = [
        line
        for line in template_body[:template_end].splitlines()
        if line == "# Work Note" or line.startswith("## ")
    ]
    assert headings == expected_headings


def test_work_note_requires_physics_prose_and_logic_flow() -> None:
    text = (WF / "plan.md").read_text(encoding="utf-8").lower()

    assert "physics background" in text
    assert "logic flow" in text
    assert "use f1 and f2 to derive s3" in text
    assert "not only equations" in text
    assert "at least as clear" in text
    assert "journal" in text
    assert "main text explains physics" in text
    assert "verbatim" in text


def test_plan_workflow_owns_work_note_planning_only() -> None:
    plan = (WF / "plan.md").read_text(encoding="utf-8").lower()

    assert "plan.md owns work-note structure" in plan
    assert "initial foundations" in plan
    assert "accepted-premise promotion" in plan
    assert "ready-step boundaries" in plan
    assert "rough-step planning" in plan
    assert "plan.md owns consensus execution" not in plan
    assert "refer to the owning workflow" in plan


def test_calculate_workflow_owns_consensus_results_only() -> None:
    calculate = (WF / "calculate.md").read_text(encoding="utf-8").lower()

    assert "calculate.md owns consensus execution" in calculate
    assert "current-step result-status" in calculate
    assert "candidate reusable result" in calculate
    assert "write a planning request" in calculate
    assert "does not change ready-step boundaries" in calculate
    assert "does not change rough steps" in calculate
    assert "does not change future plan structure" in calculate
    assert "calculate.md owns note parsing" not in calculate
    assert "refer to the owning workflow" in calculate


def test_check_workflow_owns_note_parsing_only() -> None:
    check = (WF / "check.md").read_text(encoding="utf-8").lower()

    assert "check.md owns note parsing" in check
    assert "planning handoff" in check
    assert "check.md owns work-note structure" not in check
    assert "check.md owns consensus execution" not in check
    assert "consensus behavior" not in check
    assert "refer to the owning workflow" in check


def test_calculate_uses_phase_specific_source_defaults() -> None:
    text = (WF / "calculate.md").read_text(encoding="utf-8").lower()

    assert "blind reference check" in text
    assert "reviewer_reference_claim" in text
    assert "proposer_runtime" in text
    assert '"allow_internet": false' in text
    assert '"allow_mcp": false' in text
    assert '"allow_internet": true' in text
    assert '"allow_mcp": true' in text
    assert "reference_disagrees" in text
    assert "post-check new calculation" in text


def test_calculate_uses_three_total_consensus_attempts() -> None:
    text = (WF / "calculate.md").read_text(encoding="utf-8").lower()

    assert '"max_recalculations": 2' in text
    assert "3 total attempts" in text
    assert "1 initial attempt + 2 recalculations" in text
    assert "4 attempts" not in text


def test_calculate_uses_reviewer_judgment_not_mandatory_sympy_gate() -> None:
    text = (WF / "calculate.md").read_text(encoding="utf-8").lower()

    assert "reviewer judgment" in text
    assert "sympy" in text
    assert "wolfram" in text
    assert "optional" in text
    assert "mandatory a-b" not in text
    assert "before `all_agree`, at least two" not in text
    assert "special limits are sanity checks" in text
    assert "not proof of full agreement" in text


def test_ideas_ranking_script_selects_best_round_per_loop(tmp_path) -> None:
    run_root = tmp_path / "ideas" / "run_001"
    _write_idea_round(run_root, "idea_001", 1, "first", total=10, novelty=4)
    _write_idea_round(run_root, "idea_001", 2, "better", total=15, novelty=3)
    _write_idea_round(run_root, "idea_002", 1, "high novelty", total=15, novelty=8)
    _write_idea_round(run_root, "idea_002", 2, "lower", total=12, novelty=9)

    result = subprocess.run(
        [
            sys.executable,
            str(WS / "rank-ideas.py"),
            str(run_root),
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    ranking = json.loads(result.stdout)["ranking"]
    assert [(item["loop_id"], item["round"], item["title"]) for item in ranking] == [
        ("idea_002", 1, "high novelty"),
        ("idea_001", 2, "better"),
    ]
    assert ranking[0]["marks"]["user_intent_relevance"] == 6
    assert ranking[0]["marks"]["confidence_of_novelty"] == 7
    assert "user_intent_fit" not in ranking[0]["marks"]

    markdown = subprocess.run(
        [
            sys.executable,
            str(WS / "rank-ideas.py"),
            str(run_root),
            "--format",
            "markdown",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "# Ideas\n\n" in markdown
    assert "Abbreviations:\n\nIR=intent relevance" in markdown
    summary = markdown.split("# Ranked Ideas and Details", 1)[0]
    details = markdown.split("# Ranked Ideas and Details", 1)[1]
    assert summary.index("## `idea_001`\n\nbetter") < summary.index("## `idea_002`\n\nhigh novelty")
    assert "| Round | IR | N | CN | SV | PL | WD | T |" in markdown
    assert "| Title | Total Mark | Rank |" not in markdown
    assert "| Loop | Round | Total | Intent Relevance | Novelty | Confidence | Value | Planning | Well-definedness |" in markdown
    assert "# Ranked Ideas and Details" in markdown
    assert "# Appendix: Idea Details" not in markdown
    assert details.index("### 1. high novelty") < details.index("### 2. better")
    assert "#### Referee Marks by Round" in markdown
    assert "#### Full Idea Verbatim" in markdown
    assert "```text" not in markdown
    assert "Title: high novelty" in markdown
    assert "Idea Summary:" in markdown
    assert "Calculation Plan:" in markdown
    assert "novelty_checks:" not in markdown
    assert "motivation:" not in markdown
    assert "```json" not in markdown
    assert "user_intent_fit" not in markdown


def test_ideas_marking_scheme_is_centralized() -> None:
    scheme = json.loads((WJ / "ideas-marking-scheme.json").read_text(encoding="utf-8"))
    reviewer_schema_text = (WJ / "ideas-reviewer-output.schema.json").read_text(encoding="utf-8")
    reviewer = json.loads((WJ / "ideas-reviewer.template.json").read_text(encoding="utf-8"))

    fields = [item["field"] for item in scheme["marks"]]
    maxima = {item["field"]: item["maximum"] for item in scheme["marks"]}

    assert fields == [
        "user_intent_relevance",
        "novelty",
        "confidence_of_novelty",
        "scientific_value",
        "planning",
        "problem_well_definedness",
    ]
    assert maxima == {
        "user_intent_relevance": 25,
        "novelty": 15,
        "confidence_of_novelty": 15,
        "scientific_value": 15,
        "planning": 15,
        "problem_well_definedness": 15,
    }
    assert sum(maxima.values()) == scheme["total_score"]["maximum"] == 100
    assert "evidence_of_novelty" not in reviewer_schema_text
    assert "0-30 scale" not in reviewer["prompt"]["template"]
    assert "marking_scheme" in reviewer["prompt"]["template"]


def test_ideas_marking_scheme_has_discriminating_score_anchors() -> None:
    scheme = json.loads((WJ / "ideas-marking-scheme.json").read_text(encoding="utf-8"))
    guidance = {item["field"]: item["guidance"] for item in scheme["marks"]}

    assert "Use the full numeric range" in scheme["calibration_guidance"]
    assert "A total score above 90 should be rare" in scheme["calibration_guidance"]
    assert "merely reasonable idea with unclear novelty or weak execution plan should usually fall around 55-75" in scheme["calibration_guidance"]
    assert "15: confidently publishable in a top journal" in guidance["novelty"]
    assert "10: marginally publishable in a top journal" in guidance["novelty"]
    assert "5: marginally publishable in a second-tier or specialized journal" in guidance["novelty"]
    assert "0: not publishable" in guidance["novelty"]
    assert "10: clear plan; each major step can be done by an AI agent" in guidance["planning"]
    assert "5: has a plan, but some steps are too broad or difficult for an AI agent" in guidance["planning"]
    assert "0: most steps cannot be done by an AI agent" in guidance["planning"]


def test_ideas_reviewer_comments_turn_marks_into_scientific_guidance() -> None:
    reviewer = json.loads((WJ / "ideas-reviewer.template.json").read_text(encoding="utf-8"))
    template = reviewer["prompt"]["template"]

    assert "Use caller_context.marking_scheme as the organizing checklist for reviewer feedback" in template
    assert "interpret the assigned mark scientifically" in template
    assert "what is already working" in template
    assert "weak or middling" in template
    assert "Do not restate the marking scheme, discuss score optimization, or tell the proposer how to chase rubric points" in template
    assert "Add any other scientifically important comments" in template


def test_ideas_workflow_points_to_active_runner_without_global_review() -> None:
    text = (WF / "ideas.md").read_text(encoding="utf-8")

    assert "ideas_runner.py" in text
    assert "global reviewer" not in text
    assert "global_review" not in text
    assert "five reviewer reports per loop" in text
    assert "<project-dir>/ideas/<run-id>/idea_loops/loops/" in text
    assert "scripts/rank-ideas.py" in text
    assert "<project-dir>/ideas/<run-id>/ideas.md" not in text
    assert "<project-dir>/ideas.md" not in text


def test_ideas_workflow_has_deterministic_ranked_report_deliverable() -> None:
    text = (WF / "ideas.md").read_text(encoding="utf-8")

    assert "<project-dir>/ideas/<run-id>/ranked-ideas.md" in text
    assert "<project-dir>/ranked-ideas.md" in text
    assert 'md2pdf(input="<project-dir>/ranked-ideas.md")' in text
    assert "ranked_ideas.md" not in text
    assert "<project-dir>/suggested-ideas.md" not in text


def test_ideas_loop_reviewer_template_has_arc_only_access() -> None:
    reviewer = json.loads((WJ / "ideas-reviewer.template.json").read_text(encoding="utf-8"))

    assert reviewer["id"] == "reviewer_001"
    assert reviewer["runtime"]["allow_mcp"] is True
    assert reviewer["runtime"]["mcp_mode"] == "arc-only"


def test_ideas_reviewer_uses_hundred_point_marking_scheme() -> None:
    sys.path.insert(0, str(WS))
    try:
        config_module = importlib.import_module("ideas_config")
        runner_module = importlib.import_module("ideas_runner")
    finally:
        sys.path.remove(str(WS))

    config = config_module.load_ideas_config(
        {
            "schema_version": "arc.workflow.ideas.config.v1",
            "run_id": "test",
            "run_dir": "/tmp/arc-test",
            "project_dir": "/tmp/arc-test-project",
            "user_intent": "intent",
            "variant_config_dir": str(WJ),
        }
    )
    reviewer_payload = runner_module._loop_reviewer_payload(config.variants[0])
    marks = reviewer_payload["output_schema"]["properties"]["review_payload"]["properties"]["marks"]
    mark_properties = marks["properties"]
    reviewer = json.loads((WJ / "ideas-reviewer.template.json").read_text(encoding="utf-8"))

    assert marks["required"] == [
        "user_intent_relevance",
        "novelty",
        "confidence_of_novelty",
        "scientific_value",
        "planning",
        "problem_well_definedness",
        "total_score",
    ]
    assert mark_properties["user_intent_relevance"]["minimum"] == 0
    assert mark_properties["user_intent_relevance"]["maximum"] == 25
    assert mark_properties["novelty"]["maximum"] == 15
    assert mark_properties["confidence_of_novelty"]["maximum"] == 15
    assert mark_properties["scientific_value"]["minimum"] == 0
    assert mark_properties["scientific_value"]["maximum"] == 15
    assert mark_properties["planning"]["minimum"] == 0
    assert mark_properties["planning"]["maximum"] == 15
    assert mark_properties["problem_well_definedness"]["minimum"] == 0
    assert mark_properties["problem_well_definedness"]["maximum"] == 15
    assert mark_properties["total_score"]["minimum"] == 0
    assert mark_properties["total_score"]["maximum"] == 100
    assert "marking_scheme" in reviewer["prompt"]["template"]
    assert "confidence_of_novelty" not in reviewer["prompt"]["template"]
    assert "evidence_of_novelty" not in reviewer["prompt"]["template"]
    assert "user_intent_fit" not in reviewer["prompt"]["template"]


def test_ideas_config_template_has_no_global_reviewer() -> None:
    config = json.loads((WJ / "ideas.config.template.json").read_text(encoding="utf-8"))

    assert "reviewer" not in config
    assert config["loops_per_variant"] == 5


def test_ideas_full_info_template_includes_domain_context_and_arc_tools() -> None:
    batch = json.loads((WJ / "ideas-batch.template.json").read_text(encoding="utf-8"))
    variant = json.loads((WJ / "ideas-domain.variant.json").read_text(encoding="utf-8"))
    loop = json.loads((WJ / "ideas-loop.template.json").read_text(encoding="utf-8"))
    proposer = json.loads((WJ / "ideas-proposer.template.json").read_text(encoding="utf-8"))

    assert variant["loop_template"] == "ideas-loop.template.json"
    assert variant["proposer_template"] == "ideas-proposer.template.json"
    assert batch["schema_version"] == "arc.llm.proposers_reviewer_batch.config.v1"
    assert batch["max_concurrent_loops"] == 10
    assert "domain_markdown_files" in loop["caller_context"]
    assert "arc_paper_tool_notes" in loop["caller_context"]
    assert proposer["runtime"]["allow_internet"] is True
    assert proposer["runtime"]["allow_mcp"] is True
    assert proposer["runtime"]["mcp_mode"] == "arc-only"


def test_ideas_no_info_description_mentions_shared_marking_scheme() -> None:
    variant = json.loads((WJ / "ideas-no-info.variant.json").read_text(encoding="utf-8"))
    description = variant["description"]

    assert "common marking scheme" in description
    assert "no ARC domain Markdown" in description
    assert "no ARC paper-tool guidance" in description
    assert "no MCP access" in description


def test_ideas_proposer_templates_emphasize_marking_scheme_quality_checklist() -> None:
    for name in ["ideas-proposer.template.json", "ideas-no-info-proposer.template.json"]:
        proposer = json.loads((WJ / name).read_text(encoding="utf-8"))
        template = proposer["prompt"]["template"]

        assert "caller_context.marking_scheme" in template
        assert "**Very Important**: Before finalizing, use caller_context.marking_scheme" in template
        assert "scientific quality checklist" in template
        assert "without writing to optimize marks" in template


def test_ideas_proposer_templates_request_report_ready_math() -> None:
    for name in ["ideas-proposer.template.json", "ideas-no-info-proposer.template.json"]:
        proposer = json.loads((WJ / name).read_text(encoding="utf-8"))
        template = proposer["prompt"]["template"]

        assert "$ρ_E$" in template
        assert "$T_{ab}$" in template
        assert "$η_{SL}$" in template
        assert "$$ΔT(0,b_{ref}) = ...$$" in template
        assert "do not write ASCII placeholders" in template


def test_ideas_proposer_schemas_are_codex_strict() -> None:
    proposer = json.loads((WJ / "ideas-proposer.template.json").read_text(encoding="utf-8"))
    schema = proposer["output_schema"]

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert "title" in schema["required"]
    assert "calculation_plan" in schema["required"]


def test_packaged_workflow_copies_match_source() -> None:
    for host in ["codex", "claude"]:
        packaged = ROOT / f"packaging/{host}/arc/skills/arc/workflows"
        assert (packaged / "check.md").read_text(encoding="utf-8") == (
            WF / "check.md"
        ).read_text(encoding="utf-8")
        for path in [*WF.glob("*.md"), *WJ.glob("*.json"), *WS.glob("*.py")]:
            assert (packaged / path.relative_to(WF)).read_text(encoding="utf-8") == path.read_text(
                encoding="utf-8"
            )


def test_build_domain_report_instructions_include_task_focus_solved_cases_and_open_axes() -> None:
    text = (WF / "domain.md").read_text(encoding="utf-8")

    assert "Task Focus for Idea Generation" in text
    assert "Key Papers" in text
    assert "foundation_paper" in text
    assert "best_reference_paper" in text
    assert "Known Solved Cases" in text
    assert "Open Axes for New Work" in text
    assert "these axes are examples" in text
    assert "not a complete" in text
    assert "discover additional axes" in text
    assert "Frequently Asked" in text
    assert "Do not render separate" in text
    assert "llm_get_summary" not in text
    assert "foundation_<foundation-safe>.md" not in text
    assert "Summarize Best-Reference Papers" not in text
    assert "paper_json_pack" in text
    assert "arc-paper" in text


def test_arc_skill_preserves_seed_domain_anchor_in_user_intent() -> None:
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")

    assert "scientific domain anchors" in text
    assert "field started by arXiv" in text
    assert "Remove operational instructions" in text
    assert "user_intent" in text
    assert "seed_paper_list" in text


def test_ideas_requires_domain_markdown_not_single_paper_summaries() -> None:
    variant = json.loads((WJ / "ideas-domain.variant.json").read_text(encoding="utf-8"))
    loop = json.loads((WJ / "ideas-loop.template.json").read_text(encoding="utf-8"))

    assert variant["context_policy"]["require_domain_markdown"] is True
    assert variant["context_policy"]["attach_domain_markdown"] is True
    assert "domain_markdown_files" in loop["caller_context"]
    assert "best-reference paper summaries" not in json.dumps(loop)
    assert "single-paper LLM summaries" not in json.dumps(loop)


def test_packaged_skill_references_include_required_workflow_inputs() -> None:
    required = [
        Path("workflows/check.md"),
        Path("workflows/domain.md"),
        Path("workflows/ideas.md"),
        Path("workflows/json/ideas.config.template.json"),
        Path("workflows/scripts/ideas_config.py"),
        Path("workflows/scripts/ideas_marking.py"),
        Path("workflows/scripts/ideas_runner.py"),
        Path("workflows/json/ideas-batch.template.json"),
        Path("workflows/json/ideas-domain.variant.json"),
        Path("workflows/json/ideas-loop.template.json"),
        Path("workflows/json/ideas-marking-scheme.json"),
        Path("workflows/json/ideas-no-info-loop.template.json"),
        Path("workflows/json/ideas-no-info-proposer.template.json"),
        Path("workflows/json/ideas-no-info.variant.json"),
        Path("workflows/json/ideas-proposer.template.json"),
        Path("workflows/json/ideas-reviewer.template.json"),
        Path("workflows/json/ideas-reviewer-output.schema.json"),
        Path("workflows/scripts/rank-ideas.py"),
        Path("manuals/arc-domain.md"),
        Path("manuals/arc-llm.md"),
        Path("manuals/arc-mcp.md"),
        Path("manuals/arc-paper.md"),
    ]

    for host in ["codex", "claude"]:
        packaged_skill = ROOT / f"packaging/{host}/arc/skills/arc"
        for relative in required:
            assert (packaged_skill / relative).is_file()


def test_packaged_workflows_do_not_include_stale_calculation_artifacts() -> None:
    stale_paths = [
        Path("foundation.md"),
        Path("json/plan.schema.json"),
        Path("json/foundation.schema.json"),
        Path("json/calculate.schema.json"),
        Path("scripts/filter-foundation-context.py"),
    ]

    for host in ["codex", "claude"]:
        packaged = ROOT / f"packaging/{host}/arc/skills/arc/workflows"
        for relative in stale_paths:
            assert not (packaged / relative).exists()


def test_packaged_skill_references_stay_synced_with_source() -> None:
    synced_roots = [
        Path("SKILL.md"),
        Path("manuals"),
        Path("workflows/check.md"),
        Path("workflows/domain.md"),
        Path("workflows/calculate.md"),
        Path("workflows/ideas.md"),
        Path("workflows/json/ideas.config.template.json"),
        Path("workflows/plan.md"),
        Path("workflows/scripts/ideas_config.py"),
        Path("workflows/scripts/ideas_marking.py"),
        Path("workflows/scripts/ideas_runner.py"),
        Path("workflows/json/ideas-batch.template.json"),
        Path("workflows/json/ideas-domain.variant.json"),
        Path("workflows/json/ideas-loop.template.json"),
        Path("workflows/json/ideas-marking-scheme.json"),
        Path("workflows/json/ideas-no-info-loop.template.json"),
        Path("workflows/json/ideas-no-info-proposer.template.json"),
        Path("workflows/json/ideas-no-info.variant.json"),
        Path("workflows/json/ideas-proposer.template.json"),
        Path("workflows/json/ideas-reviewer.template.json"),
        Path("workflows/json/ideas-reviewer-output.schema.json"),
        Path("workflows/scripts/rank-ideas.py"),
    ]

    expected_files: list[Path] = []
    for relative in synced_roots:
        source = SKILL / relative
        if source.is_dir():
            expected_files.extend(path.relative_to(SKILL) for path in source.glob("*"))
        else:
            expected_files.append(relative)

    for host in ["codex", "claude"]:
        packaged_skill = ROOT / f"packaging/{host}/arc/skills/arc"
        for relative in expected_files:
            assert (packaged_skill / relative).read_text(encoding="utf-8") == (
                SKILL / relative
            ).read_text(encoding="utf-8")


def test_adapter_scripts_use_installed_arc_mcp_without_repo_local_defaults() -> None:
    for host in ["codex", "claude"]:
        script = ROOT / f"packaging/{host}/arc/scripts/arc-mcp-{host}"
        text = script.read_text(encoding="utf-8")

        assert "/arc-dev" not in text
        assert ".venv/bin/arc-mcp" not in text
        assert "ARC_PAPER_CACHE" not in text
        assert "ARC_DOMAIN_CACHE" not in text
        assert 'exec arc-mcp "$@"' in text
        assert "ARC_LLM_MODEL_TIER" not in text
        assert f"ARC_{host.upper()}_MODEL=" not in text
        assert f"ARC_{host.upper()}_MODEL_TIER=" not in text


def test_interaction_reference_allows_portable_typed_fallback() -> None:
    text = (ROOT / "skills/arc/rules/interaction.md").read_text(encoding="utf-8").lower()

    assert "typed fallback" in text
    assert "when no discrete selection" in text or "if no discrete selection" in text
    assert "enter the exact option label" in text
    assert "cannot present the required selection ui" not in text


def _write_idea_round(
    run_root: Path,
    loop_id: str,
    round_number: int,
    title: str,
    *,
    total: float,
    novelty: float,
) -> None:
    round_root = run_root / "loops" / loop_id / "rounds" / f"round_{round_number:03d}"
    proposer_dir = round_root / "proposer_outputs"
    review_dir = round_root / "reviews"
    proposer_dir.mkdir(parents=True)
    review_dir.mkdir(parents=True)
    (proposer_dir / "proposer_001.json").write_text(json.dumps({"title": title}), encoding="utf-8")
    marks = {
        "novelty": novelty,
        "confidence_of_novelty": 7,
        "planning": 3,
        "scientific_value": 3,
        "user_intent_relevance": 6,
        "problem_well_definedness": 3,
        "total_score": total,
    }
    (review_dir / "reviewer_001.json").write_text(
        json.dumps({"review_payload": {"marks": marks}}),
        encoding="utf-8",
    )
