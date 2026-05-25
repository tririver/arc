from __future__ import annotations

import importlib
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


def test_research_ideas_launches_five_report_loops_without_postprocessing(tmp_path: Path) -> None:
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
    }
    seen_batch_configs: list[dict[str, Any]] = []

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
        return {
            "schema_version": "arc.llm.proposers_reviewer_batch.result.v1",
            "status": "completed",
            "run_id": batch_config["run_id"],
            "run_root": str(run_root),
            "loops": [
                {
                    "loop_id": loop["loop_id"],
                    "status": "completed",
                    "rounds_completed": loop["max_rounds"],
                    "loop_root": str(run_root / "loops" / loop["loop_id"]),
                }
                for loop in batch_config["loops"]
            ],
        }

    def forbidden_global_runner(
        prompt: str,
        *,
        schema: dict[str, Any] | None,
        provider: str,
        model: str | None,
        model_tier: str | None,
        env: dict[str, str],
    ) -> dict[str, Any]:
        raise AssertionError("research ideas workflow must not call a global reviewer")

    result = runner.run_research_ideas(
        config,
        json_runner=forbidden_global_runner,
        batch_runner=fake_batch_runner,
        base_env={},
    )

    assert result["status"] == "completed"
    assert result["proposal_count"] == 2
    assert result["reviewer_call_count"] == 0
    assert result["loop_reviewer_call_count"] == 10
    assert "global_review" not in result
    assert "ideas" not in result
    assert "report" not in result
    assert Path(result["batch_config_path"]).is_file()

    batch_config = seen_batch_configs[0]
    assert batch_config["run_dir"] == str(project_dir / "research-ideas" / "ideas_test")
    assert batch_config["run_id"] == "idea_loops"
    assert batch_config["max_concurrent_loops"] == 2
    assert {loop["loop_id"] for loop in batch_config["loops"]} == {
        "domain_idea_001",
        "no_info_idea_001",
    }
    assert {loop["max_rounds"] for loop in batch_config["loops"]} == {5}
    assert all(loop["early_stop"]["enabled"] is False for loop in batch_config["loops"])
    assert all(
        loop["reviewers"][0]["output_schema"]["properties"]["schema_version"]["const"]
        == "arc.llm.review_envelope.v1"
        for loop in batch_config["loops"]
    )
    assert all("marking_scheme" in loop["caller_context"] for loop in batch_config["loops"])
    mark_schema = batch_config["loops"][0]["reviewers"][0]["output_schema"]["properties"]["review_payload"][
        "properties"
    ]["marks"]
    assert "confidence_of_novelty" in mark_schema["required"]
    assert "evidence_of_novelty" not in mark_schema["required"]
    assert mark_schema["properties"]["user_intent_relevance"]["maximum"] == 25
    assert mark_schema["properties"]["problem_well_definedness"]["maximum"] == 15
    assert {loop["loop_id"] for loop in result["loops"]} == {
        "domain_idea_001",
        "no_info_idea_001",
    }
    assert result["batch_result"]["run_root"] == str(project_dir / "research-ideas" / "ideas_test" / "idea_loops")
