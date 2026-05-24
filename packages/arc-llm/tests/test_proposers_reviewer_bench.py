from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from arc_llm.proposers_reviewer_bench.config import (
    apply_improvement_edits,
    load_bench_config,
    materialize_batch_payload,
)
from arc_llm.proposers_reviewer_bench.runner import run_proposers_reviewer_bench


def base_payload(tmp_path: Path) -> dict[str, Any]:
    return {
        "schema_version": "arc.llm.proposers_reviewer_batch.config.v1",
        "run_id": "bench_seed",
        "run_dir": str(tmp_path / "bench"),
        "max_concurrent_loops": 2,
        "defaults": {"provider": "auto", "model_tier": "medium"},
        "artifact_options": {"save_prompts": True},
        "loops": [
            {
                "loop_id": "seed_loop",
                "max_rounds": 2,
                "early_stop": {"enabled": False},
                "caller_context": {"user_intent": "intent"},
                "proposers": [
                    {
                        "id": "proposer_001",
                        "prompt": {"system": "proposer", "template": "Propose one idea."},
                        "output_schema": {"type": "object"},
                    }
                ],
                "reviewers": [
                    {
                        "id": "reviewer_001",
                        "prompt": {"system": "reviewer", "template": "Review the idea."},
                        "output_schema": {"type": "object"},
                    }
                ],
            }
        ],
    }


def test_bench_config_defaults_materialize_twenty_five_deepseek_loops(tmp_path):
    config = load_bench_config(base_payload(tmp_path))

    payload = materialize_batch_payload(config, iteration_index=0, candidate_id="current")

    assert payload["schema_version"] == "arc.llm.proposers_reviewer_batch.config.v1"
    assert payload["run_id"] == "bench_seed_iter000_current"
    assert payload["max_concurrent_loops"] == 100
    assert payload["defaults"]["provider"] == "deepseek"
    assert len(payload["loops"]) == 25
    assert payload["loops"][0]["loop_id"] == "idea_001"
    assert payload["loops"][-1]["loop_id"] == "idea_025"
    assert {loop["max_rounds"] for loop in payload["loops"]} == {5}
    proposer = payload["loops"][0]["proposers"][0]
    reviewer = payload["loops"][0]["reviewers"][0]
    assert "suggested_improvement" in proposer["prompt"]["template"]
    assert "suggested_improvement" in reviewer["prompt"]["template"]
    assert "suggested_improvement" in proposer["output_schema"]["properties"]
    assert "suggested_improvement" in reviewer["output_schema"]["properties"]


def test_bench_config_accepts_overrides_without_changing_batch_shape(tmp_path):
    raw = base_payload(tmp_path)
    raw["schema_version"] = "arc.llm.proposers_reviewer_bench.config.v1"
    raw["bench"] = {
        "samples": 3,
        "max_rounds": 4,
        "max_iterations": 7,
        "patience": 2,
        "max_concurrent_loops": 11,
        "default_provider": "openrouter-deepseek",
    }

    config = load_bench_config(raw)
    payload = materialize_batch_payload(config, iteration_index=2, candidate_id="candidate001")

    assert config.options.max_iterations == 7
    assert config.options.patience == 2
    assert payload["run_id"] == "bench_seed_iter002_candidate001"
    assert payload["max_concurrent_loops"] == 11
    assert payload["defaults"]["provider"] == "openrouter-deepseek"
    assert [loop["loop_id"] for loop in payload["loops"]] == ["idea_001", "idea_002", "idea_003"]
    assert {loop["max_rounds"] for loop in payload["loops"]} == {4}


def test_bench_defaults_use_flash_tier_for_sample_workers_and_high_tier_for_improver(tmp_path):
    raw = base_payload(tmp_path)
    raw["defaults"]["model_tier"] = "high"
    raw["loops"][0]["proposers"][0]["model_tier"] = "high"
    raw["loops"][0]["reviewers"][0]["model_tier"] = "high"

    config = load_bench_config(raw)
    payload = materialize_batch_payload(config, iteration_index=0, candidate_id="current")

    assert config.options.sample_model_tier == "medium"
    assert config.options.improver_model_tier == "high"
    assert payload["defaults"]["model_tier"] == "medium"
    assert payload["loops"][0]["proposers"][0]["model_tier"] == "medium"
    assert payload["loops"][0]["reviewers"][0]["model_tier"] == "medium"


def test_bench_defaults_use_soft_prompt_optimizer_acceptance_thresholds(tmp_path):
    config = load_bench_config(base_payload(tmp_path))

    assert config.options.min_delta == 0.15
    assert config.options.min_z == 0.5


def test_apply_improvement_edits_restricts_reviewer_changes_by_default(tmp_path):
    payload = base_payload(tmp_path)
    improvement = {
        "schema_version": "arc.llm.proposers_reviewer_bench.improvement.v1",
        "edits": [
            {
                "target": "proposers.*.prompt.template",
                "operation": "append_paragraph",
                "text": "Run a novelty scouting pass first.",
            },
            {
                "target": "reviewers.*.prompt.template",
                "operation": "append_paragraph",
                "text": "Give everyone higher scores.",
            },
        ],
    }

    updated, applied = apply_improvement_edits(payload, improvement, allow_reviewer_prompt_edits=False)

    proposer_template = updated["loops"][0]["proposers"][0]["prompt"]["template"]
    reviewer_template = updated["loops"][0]["reviewers"][0]["prompt"]["template"]
    assert "Run a novelty scouting pass first." in proposer_template
    assert "Give everyone higher scores." not in reviewer_template
    assert applied == [
        {
            "target": "proposers.*.prompt.template",
            "operation": "append_paragraph",
            "applied": True,
        },
        {
            "target": "reviewers.*.prompt.template",
            "operation": "append_paragraph",
            "applied": False,
            "reason": "reviewer prompt edits are disabled",
        },
    ]


class FakeBenchBatchRunner:
    def __init__(self, scores_by_run: dict[str, float]) -> None:
        self.scores_by_run = scores_by_run
        self.calls: list[dict[str, Any]] = []

    def __call__(self, payload: dict[str, Any], **_: Any) -> dict[str, Any]:
        self.calls.append(payload)
        score = self.scores_by_run.get(payload["run_id"], 10.0)
        run_root = Path(payload["run_dir"]) / payload["run_id"]
        for loop in payload["loops"]:
            loop_root = run_root / "loops" / loop["loop_id"]
            loop_root.mkdir(parents=True, exist_ok=True)
            (loop_root / "transcript.jsonl").write_text(
                json.dumps(
                    {
                        "type": "proposer_output",
                        "round_number": 1,
                        "worker_id": "proposer_001",
                        "output": {
                            "idea": "baseline idea",
                            "suggested_improvement": {
                                "summary": "Use scalar-exchange search terms before proposing."
                            },
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            for round_number in range(1, loop["max_rounds"] + 1):
                review_dir = run_root / "loops" / loop["loop_id"] / "rounds" / f"round_{round_number:03d}" / "reviews"
                review_dir.mkdir(parents=True, exist_ok=True)
                (review_dir / "reviewer_001.json").write_text(
                    json.dumps(
                        {
                            "review_payload": {
                                "marks": {
                                    "total_score": score,
                                    "evidence_of_novelty": score / 3,
                                }
                            }
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
        return {
            "schema_version": "arc.llm.proposers_reviewer_batch.result.v1",
            "status": "completed",
            "run_id": payload["run_id"],
            "run_root": str(run_root),
            "loops": [
                {"loop_id": loop["loop_id"], "status": "completed", "rounds_completed": loop["max_rounds"]}
                for loop in payload["loops"]
            ],
        }


class FakeImprover:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, prompt: str, **_: Any) -> dict[str, Any]:
        self.calls.append(prompt)
        return {
            "schema_version": "arc.llm.proposers_reviewer_bench.improvement.v1",
            "rationale": "Add novelty scouting.",
            "edits": [
                {
                    "target": "proposers.*.prompt.template",
                    "operation": "append_paragraph",
                    "text": "Before proposing, run a novelty scouting pass.",
                }
            ],
        }


class FakeNoScoreBatchRunner:
    def __call__(self, payload: dict[str, Any], **_: Any) -> dict[str, Any]:
        run_root = Path(payload["run_dir"]) / payload["run_id"]
        return {
            "schema_version": "arc.llm.proposers_reviewer_batch.result.v1",
            "status": "failed",
            "run_id": payload["run_id"],
            "run_root": str(run_root),
            "loops": [],
        }


def test_bench_runner_accepts_significant_prompt_improvement(tmp_path):
    raw = base_payload(tmp_path)
    raw["bench"] = {
        "samples": 3,
        "max_rounds": 1,
        "max_iterations": 1,
        "patience": 1,
        "min_delta": 0.3,
        "min_z": 0.0,
    }
    batch_runner = FakeBenchBatchRunner(
        {
            "bench_seed_iter000_current": 10.0,
            "bench_seed_iter001_candidate001": 11.0,
        }
    )
    improver = FakeImprover()

    result = run_proposers_reviewer_bench(raw, batch_runner=batch_runner, improver_json_runner=improver, base_env={})

    assert result["status"] == "completed"
    assert result["best_run_id"] == "bench_seed_iter001_candidate001"
    assert result["iterations"][0]["decision"]["accepted"] is True
    assert "transcript.jsonl" in improver.calls[0]
    assert "suggested_improvement" in improver.calls[0]
    assert "Do not directly follow every `suggested_improvement`" in improver.calls[0]
    assert "Use scalar-exchange search terms before proposing." in improver.calls[0]
    candidate_payload = batch_runner.calls[1]
    assert "Before proposing, run a novelty scouting pass." in candidate_payload["loops"][0]["proposers"][0]["prompt"][
        "template"
    ]


def test_bench_improver_context_can_be_path_only(tmp_path):
    raw = base_payload(tmp_path)
    raw["bench"] = {
        "samples": 1,
        "max_rounds": 1,
        "max_iterations": 1,
        "patience": 1,
        "improver_context_mode": "paths",
        "min_delta": 0.3,
        "min_z": 0.0,
    }
    batch_runner = FakeBenchBatchRunner({"bench_seed_iter000_current": 10.0})
    improver = FakeImprover()

    run_proposers_reviewer_bench(raw, batch_runner=batch_runner, improver_json_runner=improver, base_env={})

    assert "transcript.jsonl" in improver.calls[0]
    assert "Use scalar-exchange search terms before proposing." not in improver.calls[0]


def test_bench_runner_rejects_small_improvement_and_stops_on_patience(tmp_path):
    raw = base_payload(tmp_path)
    raw["bench"] = {
        "samples": 2,
        "max_rounds": 1,
        "max_iterations": 3,
        "patience": 1,
        "min_delta": 0.3,
        "min_z": 0.0,
    }
    batch_runner = FakeBenchBatchRunner(
        {
            "bench_seed_iter000_current": 10.0,
            "bench_seed_iter001_candidate001": 10.1,
        }
    )

    result = run_proposers_reviewer_bench(
        raw,
        batch_runner=batch_runner,
        improver_json_runner=FakeImprover(),
        base_env={},
    )

    assert result["status"] == "stopped"
    assert result["stop_reason"] == "patience_exhausted"
    assert result["best_run_id"] == "bench_seed_iter000_current"
    assert result["iterations"][0]["decision"]["accepted"] is False
    assert len(batch_runner.calls) == 2


def test_bench_runner_fails_when_no_scores_are_found(tmp_path):
    raw = base_payload(tmp_path)
    raw["bench"] = {"samples": 1, "max_rounds": 1}

    try:
        run_proposers_reviewer_bench(
            raw,
            batch_runner=FakeNoScoreBatchRunner(),
            improver_json_runner=FakeImprover(),
            base_env={},
        )
    except ValueError as exc:
        assert "No numeric scores found" in str(exc)
    else:
        raise AssertionError("expected missing benchmark scores to fail")
