from __future__ import annotations

import json
import importlib
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins/arc"
MCP_PLUGIN = ROOT / "plugins/arc-mcp"
SKILL = PLUGIN / "skills/arc"
RULES = SKILL / "rules"
WF = SKILL / "workflows"
WJ = WF / "json"
WS = WF / "scripts"


def test_calculation_workflow_files_exist() -> None:
    for name in ["plan.md", "calculate.md", "check.md"]:
        assert (WF / name).is_file()
    assert not (WF / "foundation.md").exists()
    for name in ["plan.schema.json", "foundation.schema.json", "calculate.schema.json"]:
        assert not (WJ / name).exists()
    assert not (WS / "filter-foundation-context.py").exists()


def test_arc_skill_routes_check_and_calculation_workflows() -> None:
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")

    assert "references/" not in text
    assert "classify" in text.lower()
    assert "five cases" in text.lower()
    assert "check.md" in text
    assert "plan.md" in text
    assert "calculate.md" in text
    assert "companion.md" in text
    assert "foundation.md" not in text
    assert "work-note.md" in text


def test_arc_skill_has_preflight_gate_for_managed_workflows() -> None:
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    text_flat = " ".join(text.split())

    assert "## Preflight Gate" in text
    assert "managed ARC workflow run" in text
    assert "workflow artifacts" in text
    assert "domain references" in text
    assert "ranked ideas" in text_flat
    assert "recommendations, research directions" in text_flat
    assert "There is no \"lightweight recommendation\" exception for a mode-eligible managed workflow." in text_flat
    assert "Direct ARC tool tasks are exempt from the automation mode gate" in text_flat
    assert "collecting citers or references" in text_flat
    assert "generating paper summaries or summary batches" in text_flat
    assert "non-evaluative paper-data output" in text_flat
    assert "must not produce recommendations, research directions, scientific rankings" in text_flat
    assert "ARC reports, or project-local workflow artifacts" in text_flat
    assert "download papers that cited 0911.3380 since 2024" in text_flat
    assert "direct ARC tool orchestration" in text_flat
    assert "get_metadata" in text
    assert "domain_get_summary" in text
    assert text.index("## Preflight Gate") < text.index("## Required References")


def test_arc_skill_frontloads_workflow_references_before_route_selection() -> None:
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    required = text[text.index("## Required References") : text.index("## Workflow")]
    required_flat = " ".join(required.split())

    assert "Note checking, verification, or audit requests" in required
    assert "`workflows/check.md` before any parse, section read, or equation extraction call" in required_flat
    assert "When the user intent triggers a workflow-specific file" in required
    for name in ["check.md", "domain.md", "ideas.md", "plan.md", "calculate.md"]:
        assert f"`workflows/{name}`" in required
    assert "blocking requirement before any workflow CLI call" in required_flat


def test_arc_skill_case3_requires_full_check_workflow_phases() -> None:
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    case3 = text[text.index("Case 3:") : text.index("Case 4:")]
    case3_flat = " ".join(case3.split())

    assert "`workflows/check.md` was already loaded in Required References" in case3
    assert (
        "Parse -> Preflight -> Write Planning Handoff -> Execute `plan.md` and "
        "`calculate.md` -> Record Note-Check Status"
    ) in case3_flat
    assert "Do not skip directly to parsing results" in case3
    assert "mandatory" in case3


def test_check_plan_calculate_workflows_treat_heavy_workload_as_nonoptional() -> None:
    for name in ["check.md", "plan.md", "calculate.md"]:
        text = " ".join((WF / name).read_text(encoding="utf-8").lower().split())
        assert "heavy workload" in text
        assert "workload size is not a stop condition" in text
        assert "must not skip mandatory phases" in text
        assert "user explicitly stops" in text


def test_arc_skill_references_pdf_export_manuals() -> None:
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8").lower()
    manual = (SKILL / "manuals/arc-jobs.md").read_text(encoding="utf-8").lower()
    manual_flat = " ".join(manual.split())

    assert "markdown report export" in text
    assert "`rules/math_typeset.md`" in text
    assert "`manuals/arc-jobs.md`" in text
    assert "md2pdf" in manual
    assert "background cli job" in manual
    assert "do not wait" in manual
    assert "markdown report" in manual
    assert "report-export gate is satisfied after the job is accepted" in manual_flat
    assert "arc-jobs submit --job-type md2pdf" in manual
    assert "print `warning:`" in manual_flat
    assert "do not debug pandoc or tex" in manual


def test_math_typeset_rules_define_markdown_math_hygiene() -> None:
    text = (RULES / "math_typeset.md").read_text(encoding="utf-8")

    assert "ARC Math Typesetting Reference" in text
    assert "Use `$...$` for inline math" in text
    assert "Do not use Markdown code spans for TeX or math snippets" in text
    assert r"`\partial_{x_0}^2`" in text
    assert r"$\partial_{x_0}^2$" in text
    assert r"`\hat{\mathcal K}_+ - \hat{\mathcal K}_-`" in text
    assert r"$\hat{\mathcal K}_+ - \hat{\mathcal K}_-$" in text
    assert "stable IDs such as `eq_00009`" in text


def test_report_workflows_reference_math_typeset_rules() -> None:
    for name in ["check.md", "domain.md", "ideas.md", "plan.md", "calculate.md"]:
        text = (WF / name).read_text(encoding="utf-8")

        assert "`rules/math_typeset.md`" in text


def test_arc_skill_lists_math_typeset_reference() -> None:
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")

    assert "`rules/math_typeset.md`" in text
    assert "Markdown math" in text


def test_workflows_start_pdf_export_for_user_facing_markdown() -> None:
    expected_pdf_guard_counts = {
        "check.md": 1,
        "domain.md": 1,
        "ideas.md": 1,
        "plan.md": 2,
        "calculate.md": 1,
    }
    for name, guard_count in expected_pdf_guard_counts.items():
        text = (WF / name).read_text(encoding="utf-8").lower()
        text_flat = " ".join(text.split())
        assert "`manuals/arc-jobs.md` markdown report export" in text
        assert "md2pdf" in text
        assert "report-export gate" in text
        assert "warning:" in text
        assert "do not wait" in text
        assert "arc-mcp md2pdf" not in text
        assert text_flat.count("do not debug or fix pdf generation") == guard_count


def test_ideas_phase_4_uses_clean_selection_prompt_without_dry_run() -> None:
    text = (WF / "ideas.md").read_text(encoding="utf-8")
    manual = (SKILL / "manuals/arc-llm.md").read_text(encoding="utf-8")
    lower = text.lower()

    assert "--dry-run" not in text
    assert "Check Planned Calls" not in text
    assert "idea workflow dry run" not in manual.lower()
    assert "### Phase 4: Select Next Action" in text
    assert "use the host's selection/menu" in " ".join(lower.split())
    assert "`Proceed with ranked idea #1 (Recommended)`" in text
    assert "`Proceed with ranked idea #2`" in text
    assert "`Proceed with ranked idea #3`" in text
    assert "`other`" not in text
    assert "or quit" not in lower
    assert "`Let's discuss`" not in text
    assert "The option labels must be the raw labels" not in text
    assert "with the same three options" in text


def test_arc_context_json_defines_run_identity_and_skill_paths() -> None:
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")

    assert "`run_id`" in text
    assert "`created_at`" in text
    assert "`arc_run_root`" in text
    assert "`project_dir_name`" in text
    assert "`skill_version`" in text
    assert "`skill_dir`" in text
    assert "`skill_workflow_json_dir`" in text


def test_arc_skill_resolves_generated_project_dir_under_launch_cwd() -> None:
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    setup = text[text.index("Step 4: Resolve `<project-dir>`.") : text.index("Step 5: Write `<project-dir>/context.json`.")]
    setup_flat = " ".join(setup.split())

    assert "Capture `<arc-run-root>` by running `pwd -P`" in setup
    assert "resolve-project-dir.py" in setup
    assert "<arc-run-root>/<project_dir_name>" in setup_flat
    assert "direct child" in setup
    assert "Do not create `arc-output/<project_dir_name>`" in setup_flat
    assert ".claude" in setup
    assert ".codex" in setup


def test_readme_documents_project_dirs_as_launch_cwd_children() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    workflow = text[text.index("## End-To-End Research Workflows") :]
    workflow_flat = " ".join(workflow.split())

    assert "<launch-cwd>/<safe-dir-name>/context.json" in workflow
    assert "direct child of the directory where the agent command was launched" in workflow_flat
    assert "not under host-internal directories such as `.claude/projects`" in workflow
    assert "not wrapped in `arc-output/`" in workflow


def test_workflow_script_commands_use_skill_dir_placeholder() -> None:
    ideas = (WF / "ideas.md").read_text(encoding="utf-8")
    calculate = (WF / "calculate.md").read_text(encoding="utf-8")
    manual = (SKILL / "manuals/arc-llm.md").read_text(encoding="utf-8")
    skill = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "python3 <skill-dir>/workflows/scripts/ideas_runner.py" in ideas
    assert "python3 <skill-dir>/workflows/scripts/rank-ideas.py" in ideas
    assert "python3 <skill-dir>/workflows/scripts/calculate_runner.py" in calculate
    assert "python3 workflows/scripts/" not in ideas
    assert "python3 workflows/scripts/" not in calculate
    assert "Do not diagnose `arc-llm` by running `pip show arc-llm`" in manual
    assert "wrong Python path/runtime" in manual
    for text in (ideas, calculate, skill, readme):
        assert "Do not diagnose `arc-llm` by running `pip show arc-llm`" not in text
        assert "wrong Python path/runtime" not in text


def test_domain_summary_warnings_are_visible_and_recorded() -> None:
    domain = (WF / "domain.md").read_text(encoding="utf-8")
    manual = (SKILL / "manuals/arc-domain.md").read_text(encoding="utf-8")

    assert "print `WARNING:` immediately" in domain
    assert "`<project-dir>/context/domain/warnings.md`" in domain
    assert "summary warnings to project `self-reflect.md` and `context/domain/warnings.md`" in manual


def test_manuals_do_not_hardcode_checkout_cache_paths() -> None:
    for manual in ["arc-paper.md", "arc-domain.md", "arc-mcp.md"]:
        text = (SKILL / "manuals" / manual).read_text(encoding="utf-8")
        assert "/arc-dev/cache/" not in text
    assert "doctor-cache" in (SKILL / "manuals/arc-paper.md").read_text(encoding="utf-8")
    assert "ARC_JOBS_CACHE" in (SKILL / "manuals/arc-jobs.md").read_text(encoding="utf-8")


def test_self_reflection_allows_missing_git_metadata() -> None:
    text = (SKILL / "rules/self-reflection.md").read_text(encoding="utf-8")

    assert "Git: unavailable" in text
    assert "Archive:" in text
    assert "Run: <run_id>" in text


def test_interaction_rules_define_portable_selection_menu() -> None:
    text = (SKILL / "rules/interaction.md").read_text(encoding="utf-8")
    lower = text.lower()

    assert "`request_user_input`" not in text
    assert "codex" not in lower
    assert "collaboration mode" not in lower
    assert "selection/menu tool" in lower
    assert "two or three real, bounded options" in lower
    assert "end that label with" in lower
    assert "`(Recommended)`" in text


def test_interaction_rules_define_automation_mode_gate_examples() -> None:
    text = (SKILL / "rules/interaction.md").read_text(encoding="utf-8")
    lower = text.lower()
    lower_flat = " ".join(lower.split())

    assert "## Automation Mode Gate" in text
    text_flat = " ".join(text.split())
    assert "Do not gather \"just context\"" in text_flat
    assert "managed ARC workflow run" in text_flat
    assert "Direct ARC tool tasks do not need an automation mode" in text_flat
    assert "recommendations, research directions, scientific rankings, or ARC reports" in text_flat
    assert "non-evaluative paper-data outputs" in text_flat
    assert "Direct tasks must not produce" in text_flat
    assert "recommend research directions" in text
    assert "suggest ideas" in text
    assert "what is the title and abstract" in text
    assert "direct paper lookup allowed" in text
    assert "download papers that cited 0911.3380 since 2024" in text_flat
    assert "direct tool orchestration allowed" in text_flat
    assert "do not include list numbering inside option labels" in lower_flat
    assert "Run automatically (Recommended)" in text
    assert "Confirm major steps" in text
    assert "Discuss before running" in text


def test_automation_mode_gate_depends_on_direct_human_arc_invocation() -> None:
    skill = " ".join((SKILL / "SKILL.md").read_text(encoding="utf-8").split())
    interaction = " ".join(
        (SKILL / "rules/interaction.md").read_text(encoding="utf-8").split()
    )

    assert "directly by a human whose current prompt explicitly names ARC" in skill
    assert "Ask for an automation mode only when both conditions are true" in interaction
    assert "directly from a human whose prompt explicitly names ARC" in interaction
    assert "Quoted, forwarded, or delegated text does not count" in interaction
    assert "does not expose reliable provenance" in interaction
    assert "provenance is unavailable or ambiguous" in skill
    assert "For an agent-invoked managed workflow" in interaction
    assert "a human prompt that does not explicitly name ARC" in interaction
    assert "do not ask for a mode" in interaction
    assert "set the execution mode to `auto`" in interaction
    assert "Agent-delegated request" in interaction
    assert "Direct human prompt" in interaction
    assert "without naming ARC: do not ask for mode" in interaction


def test_automatic_workflows_preserve_requested_scope() -> None:
    skill = " ".join((SKILL / "SKILL.md").read_text(encoding="utf-8").split())
    interaction = " ".join(
        (SKILL / "rules/interaction.md").read_text(encoding="utf-8").split()
    )
    domain = " ".join((WF / "domain.md").read_text(encoding="utf-8").split())
    ideas = " ".join((WF / "ideas.md").read_text(encoding="utf-8").split())

    assert "perform exactly the workflow scope requested by the caller" in skill
    assert "never opts the caller into downstream workflows" in skill
    assert "it does not authorize a downstream workflow" in interaction
    assert "stop after domain construction" in interaction
    assert "stop after ranked ideas" in interaction
    assert "`auto` does not authorize idea generation" in domain
    assert "`auto` does not authorize either a selection question or a move to calculation" in ideas
    assert "proceed with ranked idea #1 in `auto` mode without asking" in ideas


def test_workflow_docs_stay_human_readable() -> None:
    for name in [
        "check.md",
        "plan.md",
        "calculate.md",
    ]:
        text = (WF / name).read_text(encoding="utf-8")
        assert len(text.splitlines()) <= 220
        assert "0_ref/" not in text
        if name != "calculate.md":
            assert "/scripts/" not in text
        else:
            assert "workflows/scripts/calculate_runner.py" in text


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
    assert "`manuals/arc-jobs.md` Markdown Report Export" in text
    assert "`<project-dir>/work-note.md`" in text
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


def test_plan_requires_equation_coverage_ledger() -> None:
    text = " ".join((WF / "plan.md").read_text(encoding="utf-8").lower().split())

    assert "equation coverage ledger" in text
    assert "every parsed equation id" in text
    assert "ready step, rough step, or skipped-with-reason" in text
    assert "steps may cover multiple equations" in text
    assert "source_anchor alone is not enough" in text


def test_check_workflow_requires_equation_coverage_handoff() -> None:
    text = " ".join((WF / "check.md").read_text(encoding="utf-8").lower().split())

    assert "equation coverage ledger" in text
    assert "parsed equation inventory" in text
    assert "equation id or equation-id range" in text
    assert "source_excerpt" in text
    assert "source tools are disabled" in text


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
    assert "<project-dir>/calculate/<run-id>/execute/calculate.config.json" in calculate
    assert "<project-dir>/calculate/<run-id>/execute/<calculate-run-id>/" in calculate
    assert "calculate.config.template.json" in calculate
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
        "## Equation Coverage Ledger",
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


def test_plan_workflow_orders_blocks_by_dependency_then_source_anchor() -> None:
    plan = (WF / "plan.md").read_text(encoding="utf-8").lower()

    assert "dependency/topological order" in plan
    assert "same dependency priority" in plan
    assert "earliest source anchor" in plan
    assert "source line number" in plan
    assert "accepted results" in plan
    assert "journal and revision history" in plan
    assert "chronological" in plan


def test_plan_workflow_removes_promoted_steps_from_rough_list() -> None:
    plan = (WF / "plan.md").read_text(encoding="utf-8").lower()

    assert "remove that step from" in plan
    assert "accepted," in plan
    assert "ready, or blocked detailed steps must not remain" in plan
    assert "no accepted/ready/blocked step is duplicated" in plan


def test_accepted_steps_leave_detailed_ready_section() -> None:
    calculate = " ".join((WF / "calculate.md").read_text(encoding="utf-8").lower().split())
    plan = " ".join((WF / "plan.md").read_text(encoding="utf-8").lower().split())

    assert "accepted step result goes to `## accepted derived results`" in calculate
    assert "remove the accepted step block from `## detailed steps ready to calculate`" in calculate
    assert "no `status: accepted` step block may remain" in calculate
    assert "`## detailed steps ready to calculate` is the executable backlog" in plan
    assert "accepted steps must live in `## accepted derived results`" in plan
    assert "not in the ready-step section" in plan


def test_accepted_steps_keep_trace_outside_ready_section() -> None:
    calculate = " ".join((WF / "calculate.md").read_text(encoding="utf-8").lower().split())

    assert "calculation status" in calculate
    assert "revision history" in calculate
    assert "journal" in calculate
    assert "step id" in calculate
    assert "reviewer status" in calculate
    assert "source discrepancy status" in calculate
    assert "artifact paths" in calculate


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
    assert '"allow_mcp": true' not in text
    assert "controller-mediated" in text
    assert "reference_disagrees" in text
    assert "post-check new calculation" in text


def test_calculate_uses_two_total_consensus_attempts() -> None:
    text = (WF / "calculate.md").read_text(encoding="utf-8").lower()

    assert '"max_recalculations": 1' in text
    assert "2 total attempts" in text
    assert "1 initial attempt + 1 recalculation" in text
    assert "3 total attempts" not in text


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


def test_calculate_pause_requires_explicit_human_expert_question() -> None:
    text = (WF / "calculate.md").read_text(encoding="utf-8").lower()

    assert "human expert question:" in text
    assert "do not merely say that the workflow paused" in text
    assert "name the step" in text
    assert "unresolved equation or claim" in text
    assert "user-facing response" in text


def test_calculate_human_resolution_continues_until_stop_condition() -> None:
    text = (WF / "calculate.md").read_text(encoding="utf-8").lower()

    assert "human expert later resolves" in text
    assert "unblocks the workflow" in text
    assert "continue with the next ready detailed step" in text
    assert "return to `plan.md`" in text
    assert "the user explicitly asks to pause or stop" in text


def test_check_workflow_repeats_until_requested_coverage_complete() -> None:
    text = " ".join((WF / "check.md").read_text(encoding="utf-8").lower().split())

    assert "repeat steps 1 and 2" in text
    assert "requested note-check coverage is complete" in text
    assert "do not stop only because one ready step was accepted" in text
    assert "rough or pending coverage remains" in text
    assert "return to `plan.md`" in text


def test_arc_skill_case4_repeats_until_requested_calculation_complete() -> None:
    text = " ".join((SKILL / "SKILL.md").read_text(encoding="utf-8").lower().split())

    assert "before leaving case 4" in text
    assert "ready detailed step exists" in text
    assert "rough or pending coverage remains from the original calculation request" in text
    assert "return to `workflows/plan.md`" in text
    assert "requested calculation coverage is complete" in text


def test_work_note_color_marking_only_colors_literal_markers() -> None:
    calculate = " ".join((WF / "calculate.md").read_text(encoding="utf-8").lower().split())
    plan = " ".join((WF / "plan.md").read_text(encoding="utf-8").lower().split())
    check = " ".join((WF / "check.md").read_text(encoding="utf-8").lower().split())

    assert "specific human" in calculate
    assert "expert answer" in calculate
    assert "unresolved scientific acceptance" in calculate
    assert "question" in calculate
    assert "ordinary user task" in calculate
    assert "only color" in calculate
    assert "literal marker" in calculate
    assert "`[confirmed source issue]`" in calculate
    assert "`human-resolved`" in calculate
    assert "do not color the surrounding prose" in calculate
    assert "do not color the surrounding equations" in calculate
    assert "color is stripped" in calculate or "color is unavailable" in calculate
    assert "marker remains authoritative" in calculate
    assert "exactly one of the below two cases" in calculate
    assert "source_discrepancies" in calculate
    assert "whole affected prose/equation block" not in calculate
    assert "affected visible block in red" not in calculate

    assert "`calculate.md` source-discrepancy" in plan
    assert "do not define new marker semantics in `plan.md`" in plan
    assert "`[confirmed source issue]`" not in plan
    assert "`human-resolved` marker's background" not in plan

    assert "marker-only color rule" in check


def test_calculate_workflow_uses_pdf_marker_colorbox_templates() -> None:
    text = (WF / "calculate.md").read_text(encoding="utf-8")

    assert r"\definecolor{arcsourceissue}{HTML}{8B0000}" in text
    assert r"\definecolor{archumanresolved}{HTML}{003F8C}" in text
    assert r"\colorbox{arcsourceissue}{\textcolor{white}{[confirmed source issue]}}" in text
    assert r"\colorbox{archumanresolved}{\textcolor{white}{[human-resolved]}}" in text
    assert "Do not use custom no-argument marker macros" in text


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
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
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
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
    ).stdout
    assert "# Ideas\n\n" in markdown
    assert "Abbreviations:\n\nIR=intent relevance" in markdown
    summary = markdown.split("# Appendix: Idea Details", 1)[0]
    details = markdown.split("# Appendix: Idea Details", 1)[1]
    assert summary.index("## `idea_001`\n\nbetter") < summary.index("## `idea_002`\n\nhigh novelty")
    assert "| Round | IR | N | CN | SV | PL | WD | T |" in markdown
    assert "| Title | Total Mark | Rank |" not in markdown
    assert "| Loop | Round | Total | Intent Relevance | Novelty | Confidence | Value | Planning | Well-definedness |" in markdown
    assert "# Ranked Ideas and Details" not in markdown
    assert "# Appendix: Idea Details" in markdown
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
    assert "three reviewer reports per loop" in text
    assert "<project-dir>/ideas/<run-id>/idea_loops/loops/" in text
    assert "scripts/rank-ideas.py" in text
    assert "<project-dir>/ideas/<run-id>/ideas.md" not in text
    assert "<project-dir>/ideas.md" not in text


def test_ideas_workflow_requires_context_and_runner_artifacts() -> None:
    text = (WF / "ideas.md").read_text(encoding="utf-8")

    assert "If `<project-dir>/context.json` is missing" in text
    assert "explicit `automation_level`" in text
    assert "return to `SKILL.md` Phase 1 Step 1" in text
    assert "Do not synthesize ideas manually" in text
    assert "Final ranked ideas must come from `ideas_runner.py` artifacts" in text


def test_ideas_workflow_has_deterministic_ranked_report_deliverable() -> None:
    text = (WF / "ideas.md").read_text(encoding="utf-8")

    assert "<project-dir>/ideas/<run-id>/ranked-ideas.md" in text
    assert "<project-dir>/ranked-ideas.md" in text
    assert "`<project-dir>/ranked-ideas.md`" in text
    assert "manuals/arc-jobs.md" in text
    assert "ranked_ideas.md" not in text
    assert "<project-dir>/suggested-ideas.md" not in text


def test_ideas_loop_reviewer_template_uses_controller_evidence_without_mcp() -> None:
    reviewer = json.loads((WJ / "ideas-reviewer.template.json").read_text(encoding="utf-8"))

    assert reviewer["id"] == "reviewer_001"
    assert reviewer["runtime"]["allow_mcp"] is False
    assert "mcp_mode" not in reviewer["runtime"]
    assert "controller-supplied" in reviewer["prompt"]["template"]


def test_ideas_reviewer_uses_hundred_point_marking_scheme() -> None:
    old_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(WS))
    try:
        config_module = importlib.import_module("ideas_config")
        runner_module = importlib.import_module("ideas_runner")
    finally:
        sys.dont_write_bytecode = old_dont_write_bytecode
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
    assert config["domain_manifest_path"] == "<project-dir>/domain/domain-manifest.json"


def test_domain_and_ideas_workflows_use_explicit_domain_manifest() -> None:
    domain = (WF / "domain.md").read_text(encoding="utf-8")
    ideas = (WF / "ideas.md").read_text(encoding="utf-8")

    assert "write-domain-manifest.py" in domain
    assert "arc.workflow.domain_manifest.v2" in domain
    assert "field_count" in domain
    assert "field_id" in domain
    assert "domain_manifest_path" in ideas
    assert "two or more fields use cross-domain prompts" in ideas
    assert "source domain may contribute a mature method" in ideas


def test_ideas_worker_templates_default_to_high_model_tier() -> None:
    domain_variant = json.loads((WJ / "ideas-domain.variant.json").read_text(encoding="utf-8"))
    no_info_variant = json.loads((WJ / "ideas-no-info.variant.json").read_text(encoding="utf-8"))
    reviewer = json.loads((WJ / "ideas-reviewer.template.json").read_text(encoding="utf-8"))

    assert domain_variant["proposer"]["model_tier"] == "high"
    assert no_info_variant["proposer"]["model_tier"] == "high"
    assert reviewer["model_tier"] == "high"


def test_max_model_tier_requires_an_explicit_user_request() -> None:
    skill = " ".join((SKILL / "SKILL.md").read_text(encoding="utf-8").split())
    manual = " ".join((SKILL / "manuals/arc-llm.md").read_text(encoding="utf-8").split())

    assert "Never select the `max` model tier automatically" in skill
    assert "only when the user explicitly requests the `max` model tier" in skill
    assert "Never select the `max` model tier automatically" in manual
    assert "no workflow default or automatic task mapping may select it" in manual


def test_readme_and_llm_manual_document_experimental_kimi_provider() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    manual = (SKILL / "manuals/arc-llm.md").read_text(encoding="utf-8")
    warning = (
        "kimi-code-cli is experimental and inherits Kimi Code configuration, instructions, "
        "skills, hooks, plugins, MCP, tool permissions, and persistent sessions; it may access "
        "the network, run commands, and modify files."
    )

    for text in (readme, manual):
        compact = " ".join(text.split())
        assert "Kimi Code CLI `>=0.28.0`" in text
        assert "`kimi login`" in text
        assert "ARC_AGENT_HOST=kimi-code" in text
        assert "ARC_KIMI_BIN" in text
        assert "ARC_KIMI_WORK_DIR" in text
        assert "ARC_KIMI_TIMEOUT_SECONDS" in text
        assert "ARC_LLM_KIMI_LOW_MODEL" in text
        assert "provider-side" in text
        assert "not a sandbox" in compact
        assert warning in compact

    assert "usage fields remain null" in readme
    assert "all fields in its usage object are null" in manual
    assert "does not copy, migrate, or delete Kimi sessions" in readme
    assert "does not copy, migrate, or delete the user's Kimi" in manual


def test_ideas_full_info_template_includes_domain_and_controller_context() -> None:
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
    assert proposer["runtime"]["allow_mcp"] is False
    assert "mcp_mode" not in proposer["runtime"]
    assert "controller-supplied" in proposer["prompt"]["template"]


def test_ideas_no_info_description_mentions_shared_marking_scheme() -> None:
    variant = json.loads((WJ / "ideas-no-info.variant.json").read_text(encoding="utf-8"))
    description = variant["description"]

    assert variant["enabled"] is False
    assert "common marking scheme" in description
    assert "no ARC domain Markdown" in description
    assert "no ARC paper-tool guidance" in description
    assert "no MCP access" in description


def test_readme_documents_domain_only_ideas_release_default() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    ideas_section = text.split("### 2. Ideas", 1)[1].split("### 3.", 1)[0]

    assert "release idea workflow feeds ARC-built domain Markdown" in ideas_section
    assert "no-info variant is disabled by default" in ideas_section
    assert "opt-in test fixture" in ideas_section
    assert "comparison" not in ideas_section


def test_ideas_workflow_documents_enabled_variants_not_file_renaming() -> None:
    text = (WF / "ideas.md").read_text(encoding="utf-8")

    assert "runs only enabled variants" in text
    assert "domain variant" in text
    assert "variant_inactivated" not in text
    assert "rename" not in text.lower()


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


def test_build_domain_report_instructions_include_mathematical_opportunities() -> None:
    text = (WF / "domain.md").read_text(encoding="utf-8")

    assert "Task Focus for Idea Generation" in text
    assert "Key Papers" in text
    assert "foundation_paper" in text
    assert "best_reference_paper" in text
    assert "Mathematical Opportunities" in text
    assert "mathematical_opportunities.well_defined_problems" in text
    assert "important" in text
    assert "feasible" in text
    assert "external_search_lead" in text
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


def test_ideas_attaches_optional_domain_markdown_not_single_paper_summaries() -> None:
    variant = json.loads((WJ / "ideas-domain.variant.json").read_text(encoding="utf-8"))
    loop = json.loads((WJ / "ideas-loop.template.json").read_text(encoding="utf-8"))

    assert variant["context_policy"]["require_domain_markdown"] is False
    assert variant["context_policy"]["attach_domain_markdown"] is True
    assert "domain_markdown_files" in loop["caller_context"]
    assert "best-reference paper summaries" not in json.dumps(loop)
    assert "single-paper LLM summaries" not in json.dumps(loop)


def test_root_plugin_manifests_use_canonical_arc_skill_tree() -> None:
    codex_manifest = json.loads((PLUGIN / ".codex-plugin/plugin.json").read_text(encoding="utf-8"))
    claude_manifest = json.loads((PLUGIN / ".claude-plugin/plugin.json").read_text(encoding="utf-8"))

    assert codex_manifest["name"] == "arc"
    assert codex_manifest["skills"] == "./skills/"
    assert "mcpServers" not in codex_manifest
    assert "mcpServers" not in claude_manifest
    assert not (PLUGIN / ".mcp.json").exists()
    assert not (PLUGIN / "bin/arc-mcp").exists()
    assert claude_manifest["name"] == "arc"
    assert (SKILL / "SKILL.md").is_file()
    legacy_skill = ROOT / "skills/arc"
    assert not legacy_skill.exists()
    assert not legacy_skill.is_symlink()


def test_arc_runtime_and_job_docs_cover_unified_context_and_lifecycle() -> None:
    skill = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    jobs = (SKILL / "manuals/arc-jobs.md").read_text(encoding="utf-8")

    for value in (
        "ARC_HOME",
        "cache/arc-paper/",
        "cache/arc-domain/",
        "cache/arc-llm/",
        "tmp/arc-llm/",
        "migration-conflicts/",
        "ARC_AGENT_HOST",
    ):
        assert value in skill
    assert "migration status" in skill
    assert "detected host" in skill
    assert "provider" in skill
    assert "1800-second" in jobs
    assert "worker_call_timeout_seconds" in jobs
    assert "--progress-jsonl" in jobs
    assert "`degraded`" in jobs
    assert "full provider process group" in jobs
    assert "tokens, API keys" in jobs


def test_domain_and_ideas_docs_route_by_semantic_fields_and_frozen_recency() -> None:
    skill = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    domain = (WF / "domain.md").read_text(encoding="utf-8")
    manual = (SKILL / "manuals/arc-domain.md").read_text(encoding="utf-8")
    ideas = (WF / "ideas.md").read_text(encoding="utf-8")

    for text in (skill, domain, manual):
        assert "recent_window_days" in text
        assert "as_of_date" in text
        assert "corresponding" in text and "two" in text
    assert "arc.workflow.domain_manifest.v2" in domain
    assert "arc.workflow.domain_manifest.v2" in manual
    assert "field_count" in ideas
    assert "multiple seed-specific packages" in ideas
    assert "field_id" in ideas
    assert "status is `completed` or `degraded`" in ideas
    assert "rank only usable loops" in ideas


def test_core_skill_docs_keep_mcp_optional_and_external() -> None:
    operating = (SKILL / "rules/operating.md").read_text(encoding="utf-8")
    mcp = (SKILL / "manuals/arc-mcp.md").read_text(encoding="utf-8")
    codex_manifest = json.loads((PLUGIN / ".codex-plugin/plugin.json").read_text(encoding="utf-8"))
    claude_manifest = json.loads((PLUGIN / ".claude-plugin/plugin.json").read_text(encoding="utf-8"))

    assert "CLI-only" in operating
    assert "separately installed optional `arc-mcp`" in operating
    assert "base `arc` plugin does not contain an MCP manifest" in mcp
    assert "mcpServers" not in codex_manifest
    assert "mcpServers" not in claude_manifest
    assert not (PLUGIN / ".mcp.json").exists()


def test_arc_plugin_has_no_packaged_skill_copies() -> None:
    legacy_skill = ROOT / "skills/arc"
    assert not legacy_skill.exists()
    assert not legacy_skill.is_symlink()
    assert not (ROOT / "packaging/codex/arc/skills/arc").exists()
    assert not (ROOT / "packaging/claude/arc/skills/arc").exists()


def test_arc_skill_tree_contains_no_python_bytecode() -> None:
    bad_paths = [
        str(path.relative_to(ROOT))
        for root in [PLUGIN, ROOT / ".claude-plugin"]
        if root.exists()
        for path in root.rglob("*")
        if Path(path).name == "__pycache__" or Path(path).suffix == ".pyc"
    ]
    assert bad_paths == []


def test_generated_python_caches_are_ignored_for_release_artifacts() -> None:
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")

    for pattern in ("__pycache__/", "*.py[cod]", ".pytest_cache/", ".mypy_cache/", ".ruff_cache/"):
        assert pattern in text


def test_optional_mcp_plugin_uses_bundled_arc_mcp_launcher() -> None:
    manifest = json.loads((MCP_PLUGIN / ".codex-plugin/plugin.json").read_text(encoding="utf-8"))
    mcp_config = json.loads((MCP_PLUGIN / ".mcp.json").read_text(encoding="utf-8"))
    arc_server = mcp_config["mcpServers"]["arc"]

    assert manifest["name"] == "arc-mcp"
    assert manifest["mcpServers"] == "./.mcp.json"
    assert arc_server["command"] == "./bin/arc-mcp"
    assert arc_server["args"] == []
    assert arc_server["cwd"] == "."


def test_readme_documents_marketplace_first_install() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "plugins/arc/bin/arc-runtime setup --profile core" in text
    assert "codex plugin add arc-mcp@arc" in text
    assert "separate `arc-mcp` plugin" in text
    assert "ARC_MCP_INSTALL_RETRY=1" not in text
    assert "arc-runtime setup --profile core --retry" in text
    assert "Python `venv` + `pip` is the" in text
    assert "host-recorded full commit SHA" in text
    assert "codex plugin marketplace add tririver/arc --ref stable" in text
    assert "codex plugin add arc@arc" in text
    assert "/plugin marketplace add tririver/arc@stable" in text
    assert "/plugin install arc" in text
    assert "/path/to/arc/packages/arc-paper/.venv/bin/arc-mcp" not in text
    assert "packaging/codex/arc" not in text
    assert "packaging/claude/arc" not in text


def test_readme_does_not_reference_non_repository_local_artifacts() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    unavailable_artifacts = [
        "0_ref/",
        "arc-tests/",
    ]

    for artifact in unavailable_artifacts:
        assert artifact not in text


def test_interaction_reference_allows_portable_typed_fallback() -> None:
    text = (SKILL / "rules/interaction.md").read_text(encoding="utf-8").lower()

    assert "typed fallback" in text
    assert "when no selection/menu tool" in text or "if no selection/menu tool" in text
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
    (run_root / "loops" / loop_id / "state.json").write_text(
        json.dumps({"status": "completed", "loop_id": loop_id, "rounds_completed": round_number}),
        encoding="utf-8",
    )
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
