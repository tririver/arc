from __future__ import annotations

import json

from arc_llm import cli


def minimal_config() -> dict:
    return {
        "schema_version": "arc.llm.proposers_reviewer_batch.config.v1",
        "run_id": "run_001",
        "run_dir": "project/ideas",
        "loops": [
            {
                "loop_id": "loop_001",
                "max_rounds": 1,
                "proposers": [
                    {
                        "id": "proposer_001",
                        "prompt": {"template": "propose"},
                        "output_schema": {"type": "object"},
                    }
                ],
                "reviewers": [
                    {
                        "id": "reviewer_001",
                        "prompt": {"template": "review"},
                        "output_schema": {"type": "object"},
                    }
                ],
            }
        ],
    }


def test_cli_runs_proposers_reviewer_loop_from_config(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(minimal_config()), encoding="utf-8")
    captured = {}

    def fake_run(config, **kwargs):
        captured["config"] = config
        captured["kwargs"] = kwargs
        return {"status": "completed"}

    monkeypatch.setattr(cli, "run_proposers_reviewer_batch", fake_run, raising=False)

    args = cli._build_parser().parse_args(["proposers-reviewer-loop", "--config", str(config_path), "--json"])
    result = cli._dispatch(args)

    assert result == {"status": "completed"}
    assert captured["config"]["run_id"] == "run_001"
    assert captured["kwargs"]["dry_run"] is False
    assert captured["kwargs"]["max_concurrent_loops"] is None


def test_cli_dry_run_validates_without_llm_calls(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(minimal_config()), encoding="utf-8")
    captured = {}

    def fake_run(config, **kwargs):
        captured["kwargs"] = kwargs
        return {"status": "dry_run"}

    monkeypatch.setattr(cli, "run_proposers_reviewer_batch", fake_run, raising=False)

    args = cli._build_parser().parse_args(["proposers-reviewer-loop", "--config", str(config_path), "--dry-run"])
    result = cli._dispatch(args)

    assert result == {"status": "dry_run"}
    assert captured["kwargs"]["dry_run"] is True


def test_cli_max_concurrent_loops_overrides_operational_concurrency(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(minimal_config()), encoding="utf-8")
    captured = {}

    def fake_run(config, **kwargs):
        captured["kwargs"] = kwargs
        return {"status": "completed"}

    monkeypatch.setattr(cli, "run_proposers_reviewer_batch", fake_run, raising=False)

    args = cli._build_parser().parse_args(
        ["proposers-reviewer-loop", "--config", str(config_path), "--max-concurrent-loops", "4"]
    )
    cli._dispatch(args)

    assert captured["kwargs"]["max_concurrent_loops"] == 4


def test_cli_runs_proposers_reviewer_bench_from_config(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config = minimal_config()
    config["bench"] = {"samples": 2, "max_rounds": 1}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    captured = {}

    def fake_run(config, **kwargs):
        captured["config"] = config
        captured["kwargs"] = kwargs
        return {"status": "completed", "best_run_id": "run_001_iter000_current"}

    monkeypatch.setattr(cli, "run_proposers_reviewer_bench", fake_run, raising=False)

    args = cli._build_parser().parse_args(["proposers-reviewer-bench", "--config", str(config_path), "--json"])
    result = cli._dispatch(args)

    assert result["status"] == "completed"
    assert captured["config"]["run_id"] == "run_001"
    assert captured["kwargs"]["dry_run"] is False


def test_cli_runs_proposers_reviewer_consensus_from_config(tmp_path, monkeypatch):
    config_path = tmp_path / "consensus.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "arc.llm.proposers_reviewer_consensus.config.v1",
                "run_id": "calc_001",
                "run_dir": str(tmp_path / "execute"),
                "steps": [{"step_id": "step_001", "prompt": "derive x"}],
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_run(config, **kwargs):
        captured["config"] = config
        captured["kwargs"] = kwargs
        return {"status": "dry_run"}

    monkeypatch.setattr(cli, "run_proposers_reviewer_consensus", fake_run, raising=False)

    args = cli._build_parser().parse_args(
        ["proposers-reviewer-consensus", "--config", str(config_path), "--dry-run", "--json"]
    )
    result = cli._dispatch(args)

    assert result == {"status": "dry_run"}
    assert captured["config"]["run_id"] == "calc_001"
    assert captured["kwargs"]["dry_run"] is True
