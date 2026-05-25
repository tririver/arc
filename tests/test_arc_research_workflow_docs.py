from __future__ import annotations

import json
import importlib
import subprocess
import sys
from pathlib import Path

import jsonschema


ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "skills/arc/references/research-workflows"
SKILL = ROOT / "skills/arc"


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
    assert "classify" in text.lower()
    assert "three cases" in text.lower()
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


def test_suggest_ideas_ranking_script_selects_best_round_per_loop(tmp_path) -> None:
    run_root = tmp_path / "suggest-ideas" / "run_001"
    _write_idea_round(run_root, "idea_001", 1, "first", total=10, novelty=4)
    _write_idea_round(run_root, "idea_001", 2, "better", total=15, novelty=3)
    _write_idea_round(run_root, "idea_002", 1, "high novelty", total=15, novelty=8)
    _write_idea_round(run_root, "idea_002", 2, "lower", total=12, novelty=9)

    result = subprocess.run(
        [
            sys.executable,
            str(WF / "scripts/rank-suggested-ideas.py"),
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
            str(WF / "scripts/rank-suggested-ideas.py"),
            str(run_root),
            "--format",
            "markdown",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "| Rank | Loop | Round | Total | Intent Relevance | Novelty | Confidence | Value | Planning | Well-definedness | Title |" in markdown
    assert "user_intent_fit" not in markdown


def test_suggest_ideas_marking_scheme_is_centralized() -> None:
    scheme = json.loads((WF / "suggest-ideas-marking-scheme.json").read_text(encoding="utf-8"))
    reviewer_schema_text = (WF / "suggest-ideas-reviewer-output.schema.json").read_text(encoding="utf-8")
    reviewer = json.loads((WF / "suggest-ideas-reviewer.template.json").read_text(encoding="utf-8"))

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


def test_suggest_ideas_reviewer_comments_turn_marks_into_scientific_guidance() -> None:
    reviewer = json.loads((WF / "suggest-ideas-reviewer.template.json").read_text(encoding="utf-8"))
    template = reviewer["prompt"]["template"]

    assert "Use caller_context.marking_scheme as the organizing checklist for reviewer feedback" in template
    assert "interpret the assigned mark scientifically" in template
    assert "what is already working" in template
    assert "weak or middling" in template
    assert "Do not restate the marking scheme, discuss score optimization, or tell the proposer how to chase rubric points" in template
    assert "Add any other scientifically important comments" in template


def test_research_ideas_workflow_points_to_active_runner_without_global_review() -> None:
    text = (WF / "research-ideas.md").read_text(encoding="utf-8")

    assert "research_ideas_runner.py" in text
    assert "global reviewer" not in text
    assert "global_review" not in text
    assert "five reviewer reports per loop" in text
    assert "<project-dir>/research-ideas/<run-id>/research-ideas.md" in text
    assert "<project-dir>/research-ideas.md" in text


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


def test_suggest_ideas_loop_reviewer_template_has_arc_only_access() -> None:
    reviewer = json.loads((WF / "suggest-ideas-reviewer.template.json").read_text(encoding="utf-8"))

    assert reviewer["id"] == "reviewer_001"
    assert reviewer["runtime"]["allow_mcp"] is True
    assert reviewer["runtime"]["mcp_mode"] == "arc-only"


def test_suggest_ideas_reviewer_uses_hundred_point_marking_scheme() -> None:
    sys.path.insert(0, str(WF))
    try:
        config_module = importlib.import_module("research_ideas_config")
        runner_module = importlib.import_module("research_ideas_runner")
    finally:
        sys.path.remove(str(WF))

    config = config_module.load_research_ideas_config(
        {
            "schema_version": "arc.workflow.research_ideas.config.v1",
            "run_id": "test",
            "run_dir": "/tmp/arc-test",
            "project_dir": "/tmp/arc-test-project",
            "user_intent": "intent",
            "variant_config_dir": str(WF),
        }
    )
    reviewer_payload = runner_module._loop_reviewer_payload(config.variants[0])
    marks = reviewer_payload["output_schema"]["properties"]["review_payload"]["properties"]["marks"]
    mark_properties = marks["properties"]
    reviewer = json.loads((WF / "suggest-ideas-reviewer.template.json").read_text(encoding="utf-8"))

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


def test_research_ideas_config_template_has_no_global_reviewer() -> None:
    config = json.loads((WF / "research-ideas.config.template.json").read_text(encoding="utf-8"))

    assert "reviewer" not in config
    assert config["loops_per_variant"] == 5


def test_suggest_ideas_full_info_template_includes_domain_context_and_arc_tools() -> None:
    batch = json.loads((WF / "suggest-ideas-batch.template.json").read_text(encoding="utf-8"))
    variant = json.loads((WF / "suggest-ideas-domain.variant.json").read_text(encoding="utf-8"))
    loop = json.loads((WF / "suggest-ideas-loop.template.json").read_text(encoding="utf-8"))
    proposer = json.loads((WF / "suggest-ideas-proposer.template.json").read_text(encoding="utf-8"))

    assert variant["loop_template"] == "suggest-ideas-loop.template.json"
    assert variant["proposer_template"] == "suggest-ideas-proposer.template.json"
    assert batch["schema_version"] == "arc.llm.proposers_reviewer_batch.config.v1"
    assert batch["max_concurrent_loops"] == 10
    assert "domain_markdown_files" in loop["caller_context"]
    assert "arc_paper_tool_notes" in loop["caller_context"]
    assert proposer["runtime"]["allow_internet"] is True
    assert proposer["runtime"]["allow_mcp"] is True
    assert proposer["runtime"]["mcp_mode"] == "arc-only"


def test_suggest_ideas_proposer_schemas_are_codex_strict() -> None:
    proposer = json.loads((WF / "suggest-ideas-proposer.template.json").read_text(encoding="utf-8"))
    schema = proposer["output_schema"]

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert "title" in schema["required"]
    assert "calculation_plan" in schema["required"]


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
        script = Path("scripts/rank-suggested-ideas.py")
        assert (packaged / script).read_text(encoding="utf-8") == (WF / script).read_text(
            encoding="utf-8"
        )


def test_build_domain_report_instructions_include_task_focus_solved_cases_and_open_axes() -> None:
    text = (WF / "build-domain.md").read_text(encoding="utf-8")

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


def test_suggest_ideas_requires_domain_markdown_not_single_paper_summaries() -> None:
    variant = json.loads((WF / "suggest-ideas-domain.variant.json").read_text(encoding="utf-8"))
    loop = json.loads((WF / "suggest-ideas-loop.template.json").read_text(encoding="utf-8"))

    assert variant["context_policy"]["require_domain_markdown"] is True
    assert variant["context_policy"]["attach_domain_markdown"] is True
    assert "domain_markdown_files" in loop["caller_context"]
    assert "best-reference paper summaries" not in json.dumps(loop)
    assert "single-paper LLM summaries" not in json.dumps(loop)


def test_packaged_skill_references_include_required_workflow_inputs() -> None:
    required = [
        Path("references/research-workflows/build-domain.md"),
        Path("references/research-workflows/research-ideas.md"),
        Path("references/research-workflows/research-ideas.config.template.json"),
        Path("references/research-workflows/research_ideas_config.py"),
        Path("references/research-workflows/research_ideas_marking.py"),
        Path("references/research-workflows/research_ideas_runner.py"),
        Path("references/research-workflows/suggest-ideas-batch.template.json"),
        Path("references/research-workflows/suggest-ideas-domain.variant.json"),
        Path("references/research-workflows/suggest-ideas-loop.template.json"),
        Path("references/research-workflows/suggest-ideas-marking-scheme.json"),
        Path("references/research-workflows/suggest-ideas-no-info-loop.template.json"),
        Path("references/research-workflows/suggest-ideas-no-info-proposer.template.json"),
        Path("references/research-workflows/suggest-ideas-no-info.variant.json"),
        Path("references/research-workflows/suggest-ideas-proposer.template.json"),
        Path("references/research-workflows/suggest-ideas-reviewer.template.json"),
        Path("references/research-workflows/suggest-ideas-reviewer-output.schema.json"),
        Path("references/research-workflows/scripts/rank-suggested-ideas.py"),
        Path("references/package-manuals/arc-domain.md"),
        Path("references/package-manuals/arc-llm.md"),
        Path("references/package-manuals/arc-mcp.md"),
        Path("references/package-manuals/arc-paper.md"),
    ]

    for host in ["codex", "claude"]:
        packaged_skill = ROOT / f"packaging/{host}/arc/skills/arc"
        for relative in required:
            assert (packaged_skill / relative).is_file()


def test_packaged_skill_references_stay_synced_with_source() -> None:
    synced_roots = [
        Path("SKILL.md"),
        Path("references/package-manuals"),
        Path("references/research-workflows/build-domain.md"),
        Path("references/research-workflows/research-ideas.md"),
        Path("references/research-workflows/research-ideas.config.template.json"),
        Path("references/research-workflows/research_ideas_config.py"),
        Path("references/research-workflows/research_ideas_marking.py"),
        Path("references/research-workflows/research_ideas_runner.py"),
        Path("references/research-workflows/suggest-ideas-batch.template.json"),
        Path("references/research-workflows/suggest-ideas-domain.variant.json"),
        Path("references/research-workflows/suggest-ideas-loop.template.json"),
        Path("references/research-workflows/suggest-ideas-marking-scheme.json"),
        Path("references/research-workflows/suggest-ideas-no-info-loop.template.json"),
        Path("references/research-workflows/suggest-ideas-no-info-proposer.template.json"),
        Path("references/research-workflows/suggest-ideas-no-info.variant.json"),
        Path("references/research-workflows/suggest-ideas-proposer.template.json"),
        Path("references/research-workflows/suggest-ideas-reviewer.template.json"),
        Path("references/research-workflows/suggest-ideas-reviewer-output.schema.json"),
        Path("references/research-workflows/scripts/rank-suggested-ideas.py"),
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


def test_interaction_reference_allows_portable_typed_fallback() -> None:
    text = (ROOT / "skills/arc/references/rules/interaction.md").read_text(encoding="utf-8").lower()

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
