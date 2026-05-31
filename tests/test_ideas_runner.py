from __future__ import annotations

import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "plugins/arc/skills/arc/workflows"
WJ = WF / "json"
WS = WF / "scripts"


def _load_runner_module():
    old_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(WS))
    try:
        return importlib.import_module("ideas_runner")
    finally:
        sys.dont_write_bytecode = old_dont_write_bytecode
        sys.path.remove(str(WS))


def test_ideas_launches_five_report_loops_without_postprocessing(tmp_path: Path) -> None:
    runner = _load_runner_module()
    project_dir = tmp_path / "project"
    (project_dir / "domain").mkdir(parents=True)
    (project_dir / "domain" / "domain_summary.md").write_text("# Domain\n", encoding="utf-8")
    config = {
        "schema_version": "arc.workflow.ideas.config.v1",
        "run_id": "ideas_test",
        "run_dir": str(project_dir / "ideas"),
        "project_dir": str(project_dir),
        "user_intent": "intent",
        "variant_config_dir": str(WJ),
        "variant_glob": "ideas-*.variant.json",
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
        raise AssertionError("ideas workflow must not call a global reviewer")

    result = runner.run_ideas(
        config,
        json_runner=forbidden_global_runner,
        batch_runner=fake_batch_runner,
        base_env={},
    )

    assert result["status"] == "completed"
    assert result["proposal_count"] == 2
    assert result["reviewer_call_count"] == 10
    assert result["loop_reviewer_call_count"] == 10
    assert "global_review" not in result
    assert "ideas" not in result
    assert "report" not in result
    assert Path(result["batch_config_path"]).is_file()

    batch_config = seen_batch_configs[0]
    assert batch_config["run_dir"] == str(project_dir / "ideas" / "ideas_test")
    assert batch_config["run_id"] == "idea_loops"
    assert batch_config["max_concurrent_loops"] == 2
    assert batch_config["session"]["policy"] == "stateful"
    assert batch_config["session"]["history_mode"] == "delta"
    assert batch_config["session"]["max_concurrent_same_prefix"] == 12
    assert {loop["loop_id"] for loop in batch_config["loops"]} == {
        "domain_idea_001",
        "no_info_idea_001",
    }
    assert {loop["max_rounds"] for loop in batch_config["loops"]} == {5}
    assert all(loop["early_stop"]["enabled"] is False for loop in batch_config["loops"])
    assert all(loop["proposers"][0]["model_tier"] == "medium" for loop in batch_config["loops"])
    assert all(loop["reviewers"][0]["model_tier"] == "medium" for loop in batch_config["loops"])
    assert all(
        loop["reviewers"][0]["output_schema"]["properties"]["schema_version"]["const"]
        == "arc.llm.review_envelope.v1"
        for loop in batch_config["loops"]
    )
    assert all(loop["cache_context"]["volatile_caller_context_keys"] == ["idea_id", "variant_id"] for loop in batch_config["loops"])
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
    assert result["batch_result"]["run_root"] == str(project_dir / "ideas" / "ideas_test" / "idea_loops")


def test_ideas_caps_concurrency_for_many_loops(tmp_path: Path) -> None:
    runner = _load_runner_module()
    project_dir = tmp_path / "project"
    (project_dir / "domain").mkdir(parents=True)
    (project_dir / "domain" / "domain_summary.md").write_text("# Domain\n", encoding="utf-8")
    config = {
        "schema_version": "arc.workflow.ideas.config.v1",
        "run_id": "ideas_test",
        "run_dir": str(project_dir / "ideas"),
        "project_dir": str(project_dir),
        "user_intent": "intent",
        "variant_config_dir": str(WJ),
        "variant_glob": "ideas-*.variant.json",
        "loops_per_variant": 10,
    }
    seen_max_concurrent: list[int | None] = []

    def fake_batch_runner(
        batch_config: dict[str, Any],
        *,
        json_runner: Any,
        base_env: dict[str, str] | None,
        process_chain: list[str] | None,
        dry_run: bool = False,
        max_concurrent_loops: int | None = None,
    ) -> dict[str, Any]:
        seen_max_concurrent.append(max_concurrent_loops)
        run_root = Path(batch_config["run_dir"]) / batch_config["run_id"]
        return {
            "schema_version": "arc.llm.proposers_reviewer_batch.result.v1",
            "status": "completed",
            "run_id": batch_config["run_id"],
            "run_root": str(run_root),
            "loops": [],
        }

    result = runner.run_ideas(config, batch_runner=fake_batch_runner, base_env={})

    assert result["proposal_count"] == 20
    assert seen_max_concurrent == [12]
    assert "loop concurrency capped at 12" in "\n".join(result["warnings"])
    assert "unlimited loop concurrency" not in "\n".join(result["warnings"])


def test_ideas_warning_uses_env_concurrency_cap(tmp_path: Path, monkeypatch: Any) -> None:
    runner = _load_runner_module()
    monkeypatch.setenv("ARC_IDEAS_MAX_CONCURRENT_LOOPS", "3")
    project_dir = tmp_path / "project"
    (project_dir / "domain").mkdir(parents=True)
    (project_dir / "domain" / "domain_summary.md").write_text("# Domain\n", encoding="utf-8")
    config = {
        "schema_version": "arc.workflow.ideas.config.v1",
        "run_id": "ideas_test",
        "run_dir": str(project_dir / "ideas"),
        "project_dir": str(project_dir),
        "user_intent": "intent",
        "variant_config_dir": str(WJ),
        "variant_glob": "ideas-*.variant.json",
        "loops_per_variant": 10,
    }

    def fake_batch_runner(
        batch_config: dict[str, Any],
        *,
        json_runner: Any,
        base_env: dict[str, str] | None,
        process_chain: list[str] | None,
        dry_run: bool = False,
        max_concurrent_loops: int | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": "arc.llm.proposers_reviewer_batch.result.v1",
            "status": "completed",
            "run_id": batch_config["run_id"],
            "run_root": str(Path(batch_config["run_dir"]) / batch_config["run_id"]),
            "loops": [],
        }

    result = runner.run_ideas(config, batch_runner=fake_batch_runner, base_env={})
    warning = "\n".join(result["warnings"])

    assert result["max_concurrent_loops"] == 3
    assert "loop concurrency capped at 3" in warning
    assert "unlimited loop concurrency" not in warning


def test_ideas_save_prompts_string_false(tmp_path: Path) -> None:
    runner = _load_runner_module()
    project_dir = tmp_path / "project"
    config = {
        "schema_version": "arc.workflow.ideas.config.v1",
        "run_id": "ideas_test",
        "run_dir": str(project_dir / "ideas"),
        "project_dir": str(project_dir),
        "user_intent": "intent",
        "variant_config_dir": str(WJ),
        "variant_glob": "ideas-*.variant.json",
        "loops_per_variant": 1,
        "artifact_options": {"save_prompts": "false"},
    }

    parsed = runner.load_ideas_config(config)

    assert parsed.save_prompts is False


def test_ideas_rejects_invalid_save_prompts_string(tmp_path: Path) -> None:
    runner = _load_runner_module()
    project_dir = tmp_path / "project"
    config = {
        "schema_version": "arc.workflow.ideas.config.v1",
        "run_id": "ideas_test",
        "run_dir": str(project_dir / "ideas"),
        "project_dir": str(project_dir),
        "user_intent": "intent",
        "variant_config_dir": str(WJ),
        "variant_glob": "ideas-*.variant.json",
        "loops_per_variant": 1,
        "artifact_options": {"save_prompts": "maybe"},
    }

    with pytest.raises(runner.ConfigError, match="must be a boolean"):
        runner.load_ideas_config(config)


@pytest.mark.parametrize("value", ["0", "-5", "abc"])
def test_max_concurrent_loops_rejects_invalid_env(monkeypatch: Any, value: str) -> None:
    runner = _load_runner_module()
    monkeypatch.setenv("ARC_IDEAS_MAX_CONCURRENT_LOOPS", value)

    with pytest.raises(runner.ConfigError, match="positive integer"):
        runner._max_concurrent_loops(10)  # noqa: SLF001


def test_domain_variant_attaches_all_domain_markdown_files_recursively(tmp_path: Path) -> None:
    runner = _load_runner_module()
    project_dir = tmp_path / "project"
    domain_dir = project_dir / "domain"
    (domain_dir / "nested").mkdir(parents=True)
    (domain_dir / "overview.md").write_text("# Overview\n", encoding="utf-8")
    (domain_dir / "nested" / "details.md").write_text("# Details\n", encoding="utf-8")
    (domain_dir / "notes.txt").write_text("not attached\n", encoding="utf-8")
    config = {
        "schema_version": "arc.workflow.ideas.config.v1",
        "run_id": "ideas_test",
        "run_dir": str(project_dir / "ideas"),
        "project_dir": str(project_dir),
        "user_intent": "intent",
        "variant_config_dir": str(WJ),
        "variant_glob": "ideas-domain.variant.json",
        "loops_per_variant": 1,
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
        seen_batch_configs.append(batch_config)
        run_root = Path(batch_config["run_dir"]) / batch_config["run_id"]
        return {
            "schema_version": "arc.llm.proposers_reviewer_batch.result.v1",
            "status": "completed",
            "run_id": batch_config["run_id"],
            "run_root": str(run_root),
            "loops": [
                {
                    "loop_id": batch_config["loops"][0]["loop_id"],
                    "status": "completed",
                    "rounds_completed": 5,
                    "loop_root": str(run_root / "loops" / batch_config["loops"][0]["loop_id"]),
                }
            ],
        }

    runner.run_ideas(config, batch_runner=fake_batch_runner, base_env={})

    domain_files = seen_batch_configs[0]["loops"][0]["caller_context"]["domain_markdown_files"]
    assert domain_files == [
        {"path": "domain/nested/details.md", "content": "# Details\n"},
        {"path": "domain/overview.md", "content": "# Overview\n"},
    ]


def test_ideas_warns_when_reviewer_tier_is_below_proposer(tmp_path: Path) -> None:
    runner = _load_runner_module()
    workflow_dir = _workflow_dir_with_reviewer_tier(tmp_path, "low")
    project_dir = tmp_path / "project"
    config = {
        "schema_version": "arc.workflow.ideas.config.v1",
        "run_id": "ideas_test",
        "run_dir": str(project_dir / "ideas"),
        "project_dir": str(project_dir),
        "user_intent": "intent",
        "variant_config_dir": str(workflow_dir),
        "variant_glob": "ideas-no-info.variant.json",
        "loops_per_variant": 1,
    }

    def fake_batch_runner(
        batch_config: dict[str, Any],
        *,
        json_runner: Any,
        base_env: dict[str, str] | None,
        process_chain: list[str] | None,
        dry_run: bool = False,
        max_concurrent_loops: int | None = None,
    ) -> dict[str, Any]:
        run_root = Path(batch_config["run_dir"]) / batch_config["run_id"]
        return {
            "schema_version": "arc.llm.proposers_reviewer_batch.result.v1",
            "status": "completed",
            "run_id": batch_config["run_id"],
            "run_root": str(run_root),
            "loops": [],
        }

    result = runner.run_ideas(config, batch_runner=fake_batch_runner, base_env={})

    warning = "\n".join(result["warnings"])
    assert "WARNING: REVIEWER MODEL TIER BELOW PROPOSER" in warning
    assert "no_info_idea_001" in warning
    assert "proposer_001=medium" in warning
    assert "reviewer_001=low" in warning
    warnings_path = Path(result["warnings_path"])
    assert warnings_path.is_file()
    warnings_file = warnings_path.read_text(encoding="utf-8")
    assert "WARNING: Running 1 variants x 1 proposer-reviewer loops" in warnings_file
    assert "WARNING: REVIEWER MODEL TIER BELOW PROPOSER" in warnings_file


def test_ideas_cli_prints_warnings(tmp_path: Path, capsys: Any) -> None:
    runner = _load_runner_module()
    workflow_dir = _workflow_dir_with_reviewer_tier(tmp_path, "low")
    project_dir = tmp_path / "project"
    config_path = tmp_path / "ideas.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "arc.workflow.ideas.config.v1",
                "run_id": "ideas_test",
                "run_dir": str(project_dir / "ideas"),
                "project_dir": str(project_dir),
                "user_intent": "intent",
                "variant_config_dir": str(workflow_dir),
                "variant_glob": "ideas-no-info.variant.json",
                "loops_per_variant": 1,
            }
        ),
        encoding="utf-8",
    )

    exit_code = runner.main(["--config", str(config_path), "--dry-run"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "WARNING: Running 1 variants x 1 proposer-reviewer loops" in captured.out
    assert "WARNING: REVIEWER MODEL TIER BELOW PROPOSER" in captured.out
    assert "no_info_idea_001" in captured.out


def test_ideas_result_includes_round_score_table_from_loop_transcripts(tmp_path: Path) -> None:
    runner = _load_runner_module()
    project_dir = tmp_path / "project"
    (project_dir / "domain").mkdir(parents=True)
    (project_dir / "domain" / "domain_summary.md").write_text("# Domain\n", encoding="utf-8")
    config = {
        "schema_version": "arc.workflow.ideas.config.v1",
        "run_id": "ideas_test",
        "run_dir": str(project_dir / "ideas"),
        "project_dir": str(project_dir),
        "user_intent": "intent",
        "variant_config_dir": str(WJ),
        "variant_glob": "ideas-*.variant.json",
        "loops_per_variant": 1,
    }

    def fake_batch_runner(
        batch_config: dict[str, Any],
        *,
        json_runner: Any,
        base_env: dict[str, str] | None,
        process_chain: list[str] | None,
        dry_run: bool = False,
        max_concurrent_loops: int | None = None,
    ) -> dict[str, Any]:
        run_root = Path(batch_config["run_dir"]) / batch_config["run_id"]
        scores_by_loop = {
            "domain_idea_001": [77, 78, 78, 78, 78],
            "no_info_idea_001": [75, 80, 80, 80, 80],
        }
        for loop in batch_config["loops"]:
            loop_id = loop["loop_id"]
            loop_root = run_root / "loops" / loop_id
            loop_root.mkdir(parents=True)
            title = f"{loop_id} final title"
            with (loop_root / "transcript.jsonl").open("w", encoding="utf-8") as handle:
                for round_number, total_score in enumerate(scores_by_loop[loop_id], start=1):
                    _write_jsonl(
                        handle,
                        {
                            "type": "proposer_output",
                            "round_number": round_number,
                            "output": {"title": title},
                        },
                    )
                    _write_jsonl(
                        handle,
                        {
                            "type": "review",
                            "round_number": round_number,
                            "output": {
                                "review_payload": {
                                    "marks": {
                                        "user_intent_relevance": 20,
                                        "novelty": 10,
                                        "confidence_of_novelty": 10,
                                        "scientific_value": 10,
                                        "planning": 10,
                                        "problem_well_definedness": 10,
                                        "total_score": total_score,
                                    }
                                }
                            },
                        },
                    )
        return {
            "schema_version": "arc.llm.proposers_reviewer_batch.result.v1",
            "status": "completed",
            "run_id": batch_config["run_id"],
            "run_root": str(run_root),
            "loops": [
                {
                    "loop_id": loop["loop_id"],
                    "status": "completed",
                    "rounds_completed": 5,
                    "loop_root": str(run_root / "loops" / loop["loop_id"]),
                }
                for loop in batch_config["loops"]
            ],
        }

    result = runner.run_ideas(config, batch_runner=fake_batch_runner, base_env={})

    table = result["round_score_table"]
    assert table["schema_version"] == "arc.workflow.ideas.round_score_table.v1"
    assert table["columns"] == [
        "Idea",
        "Group",
        "Final Title",
        "R1",
        "R2",
        "R3",
        "R4",
        "R5",
        "Δ R1→R5",
        "Best",
    ]
    assert [row["loop_id"] for row in table["rows"]] == ["domain_idea_001", "no_info_idea_001"]
    assert table["rows"][0]["total_scores_by_round"] == {"1": 77, "2": 78, "3": 78, "4": 78, "5": 78}
    assert table["rows"][0]["delta_total"] == 1
    assert table["rows"][0]["best_total"] == 78
    assert table["rows"][1]["total_scores_by_round"] == {"1": 75, "2": 80, "3": 80, "4": 80, "5": 80}
    assert table["rows"][1]["delta_total"] == 5
    assert table["rows"][1]["best_total"] == 80
    assert "| domain_idea_001 | domain | domain_idea_001 final title | 77 | 78 | 78 | 78 | 78 | +1 | 78 |" in table["markdown"]
    assert "| no_info_idea_001 | no_info | no_info_idea_001 final title | 75 | 80 | 80 | 80 | 80 | +5 | 80 |" in table["markdown"]


def _write_jsonl(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload) + "\n")


def _workflow_dir_with_reviewer_tier(tmp_path: Path, tier: str) -> Path:
    workflow_dir = tmp_path / "workflow"
    shutil.copytree(WJ, workflow_dir)
    reviewer_path = workflow_dir / "ideas-reviewer.template.json"
    reviewer = json.loads(reviewer_path.read_text(encoding="utf-8"))
    reviewer["model_tier"] = tier
    reviewer_path.write_text(json.dumps(reviewer, indent=2), encoding="utf-8")
    return workflow_dir
