from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "skills/arc/references/research-workflows"


def _load_runner_module():
    sys.path.insert(0, str(WF))
    try:
        return importlib.import_module("research_ideas_runner")
    finally:
        sys.path.remove(str(WF))


def test_research_ideas_runs_five_round_loops_then_global_review(tmp_path: Path) -> None:
    runner = _load_runner_module()
    project_dir = tmp_path / "project"
    (project_dir / "domain").mkdir(parents=True)
    (project_dir / "domain" / "domain_summary.md").write_text("# Domain\n", encoding="utf-8")
    config = {
        "schema_version": "arc.workflow.research_ideas.config.v1",
        "run_id": "ideas_test",
        "run_dir": str(project_dir / "research-ideas"),
        "project_dir": str(project_dir),
        "user_intent": "intent",
        "variant_config_dir": str(WF),
        "variant_glob": "suggest-ideas-*.variant.json",
        "loops_per_variant": 1,
        "artifact_options": {"save_prompts": True},
        "reviewer": {"provider": "manual", "model": None, "model_tier": "high", "allow_tools": False},
    }
    seen_batch_configs: list[dict[str, Any]] = []
    seen_global_proposals: list[dict[str, Any]] = []
    seen_global_contexts: list[dict[str, Any]] = []

    def fake_batch_runner(
        batch_config: dict[str, Any],
        *,
        json_runner: Any,
        base_env: dict[str, str] | None,
        process_chain: list[str] | None,
        dry_run: bool = False,
        max_concurrent_loops: int | None = None,
    ) -> dict[str, Any]:
        assert dry_run is False
        seen_batch_configs.append(batch_config)
        run_root = Path(batch_config["run_dir"]) / batch_config["run_id"]
        loops = []
        for loop in batch_config["loops"]:
            loop_id = loop["loop_id"]
            loop_root = run_root / "loops" / loop_id
            for round_number in range(1, 6):
                round_root = loop_root / "rounds" / f"round_{round_number:03d}"
                proposal = {
                    "title": f"{loop_id} round {round_number}",
                    "idea_summary": "summary",
                    "motivation": "motivation",
                    "novelty_checks": [f"check {round_number}"],
                    "calculation_plan": "plan",
                    "validation_checks": ["validation"],
                    "risks": ["risk"],
                }
                review = {
                    "schema_version": "arc.llm.review_envelope.v1",
                    "controller": {"message": "continue", "stop_requested": False},
                    "proposer_messages": {
                        "proposer_001": {"message": f"revise after round {round_number}"}
                    },
                    "review_payload": {
                        "marks": {
                            "user_intent_relevance": 20,
                            "novelty": 10,
                            "confidence_of_novelty": 10 + round_number,
                            "scientific_value": 10,
                            "planning": 10,
                            "problem_well_definedness": 10,
                            "total_score": 70 + round_number,
                        },
                        "reviewer_benchmark": {
                            "same_direction_alternative": "benchmark",
                            "preserves_proposer_direction": True,
                            "comparison": "comparison",
                        },
                        "improvement_comments": [f"comment {round_number}"],
                        "evidence_checked": ["evidence"],
                        "tool_queries_used": ["query"],
                    },
                }
                _write_json(round_root / "proposer_outputs" / "proposer_001.json", proposal)
                _write_json(round_root / "reviews" / "reviewer_001.json", review)
            loops.append(
                {
                    "loop_id": loop_id,
                    "status": "completed",
                    "rounds_completed": loop["max_rounds"],
                    "loop_root": str(loop_root),
                }
            )
        return {
            "schema_version": "arc.llm.proposers_reviewer_batch.result.v1",
            "status": "completed",
            "run_id": batch_config["run_id"],
            "run_root": str(run_root),
            "loops": loops,
        }

    def fake_global_runner(
        prompt: str,
        *,
        schema: dict[str, Any] | None,
        provider: str,
        model: str | None,
        model_tier: str | None,
        env: dict[str, str],
    ) -> dict[str, Any]:
        context = _context_from_prompt(prompt)
        seen_global_contexts.append(context)
        ideas = context["caller_context"]["ideas"]
        seen_global_proposals.extend(ideas)
        reviews = {
            idea["idea_id"]: {
                "marks": {
                    "user_intent_relevance": 25,
                    "novelty": 15,
                    "confidence_of_novelty": 15,
                    "scientific_value": 15,
                    "planning": 15,
                    "problem_well_definedness": 15,
                    "total_score": 100,
                },
                "main_concerns": [],
                "evidence_checked": [],
                "selected_for_next_phase": True,
                "next_phase_prompt": "next",
            }
            for idea in ideas
        }
        return {
            "schema_version": "arc.workflow.research_ideas.global_review.v1",
            "reviews": reviews,
            "ranking": [
                {"rank": index, "idea_id": idea["idea_id"], "total_score": 100}
                for index, idea in enumerate(ideas, start=1)
            ],
            "cross_variant_observations": [],
        }

    result = runner.run_research_ideas(
        config,
        json_runner=fake_global_runner,
        batch_runner=fake_batch_runner,
        base_env={},
    )

    assert result["status"] == "completed"
    assert result["proposal_count"] == 2
    assert result["reviewer_call_count"] == 1
    assert result["loop_reviewer_call_count"] == 10
    batch_config = seen_batch_configs[0]
    assert batch_config["max_concurrent_loops"] == 2
    assert {loop["loop_id"] for loop in batch_config["loops"]} == {
        "domain_idea_001",
        "no_info_idea_001",
    }
    assert {loop["max_rounds"] for loop in batch_config["loops"]} == {5}
    assert all(loop["early_stop"]["enabled"] is False for loop in batch_config["loops"])
    assert all(loop["reviewers"][0]["output_schema"]["properties"]["schema_version"]["const"] == "arc.llm.review_envelope.v1" for loop in batch_config["loops"])
    assert all("marking_scheme" in loop["caller_context"] for loop in batch_config["loops"])
    mark_schema = batch_config["loops"][0]["reviewers"][0]["output_schema"]["properties"]["review_payload"]["properties"]["marks"]
    assert "confidence_of_novelty" in mark_schema["required"]
    assert "evidence_of_novelty" not in mark_schema["required"]
    assert mark_schema["properties"]["user_intent_relevance"]["maximum"] == 25
    assert mark_schema["properties"]["problem_well_definedness"]["maximum"] == 15
    assert seen_global_contexts[0]["caller_context"]["marking_scheme"]["total_score"]["maximum"] == 100
    assert {idea["proposal"]["title"] for idea in seen_global_proposals} == {
        "domain_idea_001 round 5",
        "no_info_idea_001 round 5",
    }
    for idea in result["ideas"]:
        assert idea["selected_round"] == 5
        assert idea["rounds_completed"] == 5
        assert idea["output"]["title"].endswith("round 5")
        assert Path(idea["selected_review_path"]).is_file()


def _context_from_prompt(prompt: str) -> dict[str, Any]:
    marker = "## ARC Worker Context\n"
    assert marker in prompt
    return json.loads(prompt.split(marker, 1)[1])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
