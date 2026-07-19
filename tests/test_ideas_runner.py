from __future__ import annotations

import importlib
import hashlib
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
SINGLE_DOMAIN_GOLDEN_SHA256 = {
    "ideas-domain.variant.json": "5bff07665fdca9a9b9983106bba4954f090cbc2add7ab1edc324e5bd4792ef81",
    "ideas-loop.template.json": "4880f3cafc76163dc4c348c7716167e0fcdfbf82a975c24fc1e2b5f4fc2c3e99",
    "ideas-proposer.template.json": "e6efaa6c250231b0d75c861728f5ffb95cde202f0aaad4900cf115bc9daad70e",
    "ideas-reviewer.template.json": "5d778af29cad459211315c3f4b1e4a8bfbecc3732e310d7d7b72be36e9050cbd",
    "ideas-reviewer-output.schema.json": "be8504c44d47bb471488d522720cffaa610d1633b473349e65907b64010dcf0d",
    "ideas-marking-scheme.json": "a126c2add3c15d13b4911e72687e53528e2374f6ee724ab8d53adca50beaecc1",
}


def _load_runner_module():
    old_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(WS))
    try:
        return importlib.import_module("ideas_runner")
    finally:
        sys.dont_write_bytecode = old_dont_write_bytecode
        sys.path.remove(str(WS))


def test_single_domain_worker_contract_files_match_expected_golden() -> None:
    actual = {
        name: hashlib.sha256((WJ / name).read_bytes()).hexdigest()
        for name in SINGLE_DOMAIN_GOLDEN_SHA256
    }

    assert actual == SINGLE_DOMAIN_GOLDEN_SHA256


def test_all_ideas_loop_templates_default_to_three_rounds() -> None:
    template_names = [
        "ideas-loop.template.json",
        "ideas-cross-domain-loop.template.json",
        "ideas-no-info-loop.template.json",
    ]

    assert {
        json.loads((WJ / name).read_text(encoding="utf-8"))["max_rounds"]
        for name in template_names
    } == {3}


def test_ideas_launches_three_round_loop_without_postprocessing(tmp_path: Path) -> None:
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
            "warnings_summary": {
                "structured_output_warning_count": 2,
                "structured_output_warnings_path": str(run_root / "structured_output_warnings.jsonl"),
                "cache_warning_count": 1,
                "cache_warnings_path": str(run_root / "cache_warnings.jsonl"),
            },
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
    assert result["proposal_count"] == 1
    assert result["reviewer_call_count"] == 3
    assert result["loop_reviewer_call_count"] == 3
    assert "global_review" not in result
    assert "ideas" not in result
    assert "report" not in result
    assert Path(result["batch_config_path"]).is_file()

    batch_config = seen_batch_configs[0]
    assert batch_config["run_dir"] == str(project_dir / "ideas" / "ideas_test")
    assert batch_config["run_id"] == "idea_loops"
    assert batch_config["max_concurrent_loops"] == 1
    assert batch_config["session"]["policy"] == "stateful"
    assert batch_config["session"]["history_mode"] == "delta"
    assert batch_config["session"]["max_concurrent_same_prefix"] == 12
    assert batch_config["output_recovery"]["schema_violation_policy"] == "peer_visible"
    assert batch_config["output_recovery"]["reviewer_validation_retries"] == 0
    assert {loop["loop_id"] for loop in batch_config["loops"]} == {"domain_idea_001"}
    assert {loop["max_rounds"] for loop in batch_config["loops"]} == {3}
    assert all(loop["early_stop"]["enabled"] is False for loop in batch_config["loops"])
    assert all(loop["proposers"][0]["model_tier"] == "high" for loop in batch_config["loops"])
    assert all(loop["reviewers"][0]["model_tier"] == "high" for loop in batch_config["loops"])
    assert all(
        loop["reviewers"][0]["output_schema"]["properties"]["schema_version"]["const"]
        == "arc.llm.review_envelope.v1"
        for loop in batch_config["loops"]
    )
    assert all(loop["cache_context"]["volatile_caller_context_keys"] == ["idea_id", "variant_id"] for loop in batch_config["loops"])
    assert all("marking_scheme" in loop["caller_context"] for loop in batch_config["loops"])
    assert all("generation_mode" not in loop["caller_context"] for loop in batch_config["loops"])
    assert all("domain_cards" not in loop["caller_context"] for loop in batch_config["loops"])
    mark_schema = batch_config["loops"][0]["reviewers"][0]["output_schema"]["properties"]["review_payload"][
        "properties"
    ]["marks"]
    assert "confidence_of_novelty" in mark_schema["required"]
    assert "evidence_of_novelty" not in mark_schema["required"]
    assert mark_schema["properties"]["user_intent_relevance"]["maximum"] == 25
    assert mark_schema["properties"]["problem_well_definedness"]["maximum"] == 15
    assert {loop["loop_id"] for loop in result["loops"]} == {"domain_idea_001"}
    assert result["batch_result"]["run_root"] == str(project_dir / "ideas" / "ideas_test" / "idea_loops")
    assert result["warnings_summary"]["structured_output_warning_count"] == 2
    assert result["warnings_summary"]["cache_warning_count"] == 1


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
        "loops_per_variant": 20,
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
    assert "Running 1 variants x 10 proposer-reviewer loops" in warning
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
                    "rounds_completed": batch_config["loops"][0]["max_rounds"],
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


def test_domain_variant_warns_and_continues_without_domain_markdown_when_optional(tmp_path: Path) -> None:
    runner = _load_runner_module()
    workflow_dir = tmp_path / "workflow"
    shutil.copytree(WJ, workflow_dir)
    variant_path = workflow_dir / "ideas-domain.variant.json"
    variant = json.loads(variant_path.read_text(encoding="utf-8"))
    variant["context_policy"]["require_domain_markdown"] = False
    variant_path.write_text(json.dumps(variant, indent=2), encoding="utf-8")
    project_dir = tmp_path / "project"
    config = {
        "schema_version": "arc.workflow.ideas.config.v1",
        "run_id": "ideas_test",
        "run_dir": str(project_dir / "ideas"),
        "project_dir": str(project_dir),
        "user_intent": "intent",
        "variant_config_dir": str(workflow_dir),
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
            "loops": [],
        }

    result = runner.run_ideas(config, batch_runner=fake_batch_runner, base_env={})

    assert result["status"] == "completed"
    assert "domain_markdown_unavailable" in "\n".join(result["warnings"])
    caller_context = seen_batch_configs[0]["loops"][0]["caller_context"]
    assert "domain_markdown_files" not in caller_context
    assert "Domain markdown was unavailable" in "\n".join(caller_context["warnings"])


def test_missing_manifest_preserves_legacy_single_domain_route(tmp_path: Path) -> None:
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
    }

    parsed = runner.load_ideas_config(config)

    assert parsed.research_scope == "single_domain"
    assert [variant.variant_id for variant in parsed.variants] == ["domain"]
    assert "domain_manifest_unavailable" in "\n".join(parsed.routing_warnings)


def test_strict_source_mode_rejects_external_variant_directory(tmp_path: Path, monkeypatch: Any) -> None:
    runner = _load_runner_module()
    external = tmp_path / "installed-plugin-json"
    external.mkdir()
    monkeypatch.setenv("ARC_REQUIRE_REPO_ROOT", str(ROOT))

    with pytest.raises(runner.ConfigError, match="requires variant_config_dir from the required checkout"):
        runner.load_ideas_config(
            {
                "schema_version": "arc.workflow.ideas.config.v1",
                "run_id": "ideas_test",
                "run_dir": str(tmp_path / "ideas"),
                "project_dir": str(tmp_path / "project"),
                "user_intent": "intent",
                "variant_config_dir": str(external),
            }
        )


def test_explicit_missing_manifest_fails_instead_of_silently_using_single_domain(tmp_path: Path) -> None:
    runner = _load_runner_module()
    project_dir = tmp_path / "project"

    with pytest.raises(runner.ConfigError, match="domain_manifest_path does not exist"):
        runner.load_ideas_config(
            {
                "schema_version": "arc.workflow.ideas.config.v1",
                "run_id": "ideas_test",
                "run_dir": str(project_dir / "ideas"),
                "project_dir": str(project_dir),
                "user_intent": "intent",
                "domain_manifest_path": "domain/missing.json",
                "variant_config_dir": str(WJ),
            }
        )


def test_duplicate_manifest_domain_ids_use_single_domain_route(tmp_path: Path) -> None:
    runner = _load_runner_module()
    project_dir = tmp_path / "project"
    domain_dir = project_dir / "domain"
    domain_dir.mkdir(parents=True)
    manifest = {
        "schema_version": "arc.workflow.domain_manifest.v1",
        "domains": [{"domain_id": "same"}, {"domain_id": "same"}],
    }
    (domain_dir / "domain-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    parsed = runner.load_ideas_config(
        {
            "schema_version": "arc.workflow.ideas.config.v1",
            "run_id": "ideas_test",
            "run_dir": str(project_dir / "ideas"),
            "project_dir": str(project_dir),
            "user_intent": "intent",
            "variant_config_dir": str(WJ),
            "variant_glob": "ideas-*.variant.json",
            "loops_per_variant": 1,
        }
    )

    assert parsed.research_scope == "single_domain"
    assert [variant.variant_id for variant in parsed.variants] == ["domain"]


def test_invalid_domain_manifest_fails_before_materializing_ideas(tmp_path: Path) -> None:
    runner = _load_runner_module()
    project_dir = tmp_path / "project"
    domain_dir = project_dir / "domain"
    domain_dir.mkdir(parents=True)
    (domain_dir / "domain-manifest.json").write_text(
        json.dumps({"schema_version": "wrong", "domains": [{"domain_id": "a"}]}),
        encoding="utf-8",
    )

    with pytest.raises(runner.ConfigError, match="schema_version must be arc.workflow.domain_manifest.v1"):
        runner.load_ideas_config(
            {
                "schema_version": "arc.workflow.ideas.config.v1",
                "run_id": "ideas_test",
                "run_dir": str(project_dir / "ideas"),
                "project_dir": str(project_dir),
                "user_intent": "intent",
                "variant_config_dir": str(WJ),
            }
        )


def test_cross_domain_manifest_routes_to_structured_profiles_and_contract(tmp_path: Path) -> None:
    runner = _load_runner_module()
    project_dir = tmp_path / "project"
    _write_cross_domain_manifest(project_dir)
    config = {
        "schema_version": "arc.workflow.ideas.config.v1",
        "run_id": "ideas_test",
        "run_dir": str(project_dir / "ideas"),
        "project_dir": str(project_dir),
        "user_intent": "transfer a useful method",
        "variant_config_dir": str(WJ),
        "variant_glob": "ideas-*.variant.json",
        "loops_per_variant": 5,
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
                    "loop_id": loop["loop_id"],
                    "status": "completed",
                    "rounds_completed": loop["max_rounds"],
                    "loop_root": str(run_root / "loops" / loop["loop_id"]),
                }
                for loop in batch_config["loops"]
            ],
        }

    result = runner.run_ideas(config, batch_runner=fake_batch_runner, base_env={})

    assert result["research_scope"] == "cross_domain"
    assert result["proposal_count"] == 5
    assert result["reviewer_call_count"] == 15
    assert result["loop_reviewer_call_count"] == 15
    loops = seen_batch_configs[0]["loops"]
    assert {loop["max_rounds"] for loop in loops} == {3}
    assert {loop["loop_id"] for loop in loops} == {
        "cross_domain_idea_001",
        "cross_domain_idea_002",
        "cross_domain_idea_003",
        "cross_domain_idea_004",
        "cross_domain_idea_005",
    }
    profile_ids = [loop["caller_context"]["exploration_profile"]["profile_id"] for loop in loops]
    assert profile_ids == [
        "forward_transfer",
        "reverse_transfer",
        "method_transfer",
        "observable_or_constraint_transfer",
        "high_upside_wildcard",
    ]
    cards = loops[0]["caller_context"]["domain_cards"]
    assert loops[0]["caller_context"]["generation_mode"] == "cross_domain"
    assert [card["domain_id"] for card in cards] == ["domain-a", "domain-b"]
    assert cards[0]["task_focus"]["research_scope"] == "scope a"
    assert cards[1]["methodology"] == [{"name": "method b"}]
    assert "domain_markdown_files" not in loops[0]["caller_context"]
    assert loops[0]["cache_context"]["volatile_caller_context_keys"] == [
        "idea_id",
        "variant_id",
        "exploration_profile",
    ]
    proposer_schema = loops[0]["proposers"][0]["output_schema"]
    assert "domain_roles" in proposer_schema["required"]
    assert "transfer_map" in proposer_schema["required"]
    review_payload = loops[0]["reviewers"][0]["output_schema"]["properties"]["review_payload"]
    assert "cross_domain_assessment" in review_payload["required"]
    marks = review_payload["properties"]["marks"]
    assert marks["properties"]["cross_domain_transfer_quality"]["maximum"] == 15
    assert marks["properties"]["substantive_target_contribution"]["maximum"] == 20
    assessment = review_payload["properties"]["cross_domain_assessment"]
    assert assessment["properties"]["recommended_action"]["enum"] == [
        "refine_current",
        "rebuild_bridge",
        "replace_idea",
    ]
    assert assessment["properties"]["transfer_status"]["enum"] == [
        "genuine",
        "partial",
        "decorative",
        "single_domain",
    ]


def test_cross_domain_nondefault_loop_count_requires_explicit_profiles(tmp_path: Path) -> None:
    runner = _load_runner_module()
    project_dir = tmp_path / "project"
    _write_cross_domain_manifest(project_dir)
    config = {
        "schema_version": "arc.workflow.ideas.config.v1",
        "run_id": "ideas_test",
        "run_dir": str(project_dir / "ideas"),
        "project_dir": str(project_dir),
        "user_intent": "intent",
        "variant_config_dir": str(WJ),
        "loops_per_variant": 2,
    }

    with pytest.raises(runner.ConfigError, match="five default exploration profiles"):
        runner.load_ideas_config(config)

    config["exploration_profiles"] = [
        {"profile_id": "custom_forward", "mission": "Transfer A to B."},
        {"profile_id": "custom_reverse", "mission": "Transfer B to A."},
    ]
    parsed = runner.load_ideas_config(config)
    ideas = runner._materialize_ideas(parsed)  # noqa: SLF001

    assert [idea.caller_context["exploration_profile"]["profile_id"] for idea in ideas] == [
        "custom_forward",
        "custom_reverse",
    ]


def test_cross_domain_rejects_manifest_summary_domain_mismatch(tmp_path: Path) -> None:
    runner = _load_runner_module()
    project_dir = tmp_path / "project"
    _write_cross_domain_manifest(project_dir)
    manifest_path = project_dir / "domain" / "domain-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["domains"][0]["summary_json_path"] = manifest["domains"][1]["summary_json_path"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    parsed = runner.load_ideas_config(
        {
            "schema_version": "arc.workflow.ideas.config.v1",
            "run_id": "ideas_test",
            "run_dir": str(project_dir / "ideas"),
            "project_dir": str(project_dir),
            "user_intent": "intent",
            "variant_config_dir": str(WJ),
        }
    )

    with pytest.raises(runner.ConfigError, match="points to summary"):
        runner._materialize_ideas(parsed)  # noqa: SLF001


def test_cross_domain_worker_schemas_are_strict_and_source_to_target(tmp_path: Path) -> None:
    runner = _load_runner_module()
    project_dir = tmp_path / "project"
    _write_cross_domain_manifest(project_dir)
    parsed = runner.load_ideas_config(
        {
            "schema_version": "arc.workflow.ideas.config.v1",
            "run_id": "ideas_test",
            "run_dir": str(project_dir / "ideas"),
            "project_dir": str(project_dir),
            "user_intent": "intent",
            "variant_config_dir": str(WJ),
        }
    )
    proposer = json.loads((WJ / "ideas-cross-domain-proposer.template.json").read_text(encoding="utf-8"))
    reviewer = runner._loop_reviewer_payload(parsed.variants[0])["output_schema"]  # noqa: SLF001
    marking = json.loads((WJ / "ideas-cross-domain-marking-scheme.json").read_text(encoding="utf-8"))

    _assert_strict_objects(proposer["output_schema"])
    _assert_strict_objects(reviewer)
    roles = proposer["output_schema"]["properties"]["domain_roles"]
    assert roles["required"] == ["source_domain_id", "target_domain_id", "supporting_domain_ids"]
    assessment = reviewer["properties"]["review_payload"]["properties"]["cross_domain_assessment"]
    assert "transfer_signature" in assessment["required"]
    assert assessment["properties"]["target_contribution_status"]["enum"] == [
        "incremental",
        "substantial",
        "transformative",
    ]
    assert "blocking_compatibility_failures" in assessment["required"]
    assert "manageable_compatibility_risks" in assessment["required"]
    assert "compatibility_failures" not in assessment["properties"]
    assert sum(item["maximum"] for item in marking["marks"]) == 100
    prompt = proposer["prompt"]["template"]
    assert "The source domain does not need a new contribution" in prompt
    assert "Do not demand bidirectional innovation" in prompt


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
    assert "proposer_001=high" in warning
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
        scores_by_loop = {"domain_idea_001": [77, 78, 78]}
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
                    "rounds_completed": loop["max_rounds"],
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
        "Δ R1→R3",
        "Best",
    ]
    assert [row["loop_id"] for row in table["rows"]] == ["domain_idea_001"]
    assert table["rows"][0]["total_scores_by_round"] == {"1": 77, "2": 78, "3": 78}
    assert table["rows"][0]["delta_total"] == 1
    assert table["rows"][0]["best_total"] == 78
    assert "| domain_idea_001 | domain | domain_idea_001 final title | 77 | 78 | 78 | +1 | 78 |" in table["markdown"]
    assert "no_info_idea_001" not in table["markdown"]


def test_major_recovered_reviewer_marks_are_zero_in_round_scores(tmp_path: Path) -> None:
    runner = _load_runner_module()
    scheme = runner.load_marking_scheme(WJ)
    loop_root = tmp_path / "idea_loops" / "loops" / "domain_idea_001"
    loop_root.mkdir(parents=True)
    with (loop_root / "transcript.jsonl").open("w", encoding="utf-8") as handle:
        _write_jsonl(
            handle,
            {
                "type": "review",
                "round_number": 1,
                "output": {
                    "arc_llm_call_record": {
                        "structured_output": {"mode": "recovered", "severity": "major"}
                    },
                    "review_payload": {
                        "marks": {
                            "user_intent_relevance": 25,
                            "novelty": 10,
                            "confidence_of_novelty": 10,
                            "scientific_value": 15,
                            "planning": 15,
                            "problem_well_definedness": 15,
                            "total_score": 90,
                        }
                    },
                },
            },
        )

    rounds, _title = runner._loop_round_scores_from_transcript(loop_root, scheme=scheme)  # noqa: SLF001

    assert rounds[1]["total_score"] == 0
    assert all(value == 0 for value in rounds[1].values())


def test_major_recovered_proposer_marks_are_zero_in_round_scores(tmp_path: Path) -> None:
    runner = _load_runner_module()
    scheme = runner.load_marking_scheme(WJ)
    loop_root = tmp_path / "idea_loops" / "loops" / "domain_idea_001"
    loop_root.mkdir(parents=True)
    with (loop_root / "transcript.jsonl").open("w", encoding="utf-8") as handle:
        _write_jsonl(
            handle,
            {
                "type": "proposer_output",
                "round_number": 1,
                "output": {
                    "title": "Recovered idea",
                    "arc_llm_call_record": {
                        "structured_output": {"mode": "recovered", "severity": "major"}
                    },
                },
            },
        )
        _write_jsonl(
            handle,
            {
                "type": "review",
                "round_number": 1,
                "output": {
                    "review_payload": {
                        "marks": {
                            "user_intent_relevance": 25,
                            "novelty": 10,
                            "confidence_of_novelty": 10,
                            "scientific_value": 15,
                            "planning": 15,
                            "problem_well_definedness": 15,
                            "total_score": 90,
                        }
                    },
                },
            },
        )

    rounds, title = runner._loop_round_scores_from_transcript(loop_root, scheme=scheme)  # noqa: SLF001

    assert title == "Recovered idea"
    assert rounds[1]["total_score"] == 0
    assert all(value == 0 for value in rounds[1].values())


def _write_jsonl(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload) + "\n")


def _workflow_dir_with_reviewer_tier(tmp_path: Path, tier: str) -> Path:
    workflow_dir = tmp_path / "workflow"
    shutil.copytree(WJ, workflow_dir)
    variant_path = workflow_dir / "ideas-no-info.variant.json"
    variant = json.loads(variant_path.read_text(encoding="utf-8"))
    variant["enabled"] = True
    variant_path.write_text(json.dumps(variant, indent=2), encoding="utf-8")
    reviewer_path = workflow_dir / "ideas-reviewer.template.json"
    reviewer = json.loads(reviewer_path.read_text(encoding="utf-8"))
    reviewer["model_tier"] = tier
    reviewer_path.write_text(json.dumps(reviewer, indent=2), encoding="utf-8")
    return workflow_dir


def _write_cross_domain_manifest(project_dir: Path) -> None:
    domain_dir = project_dir / "domain"
    domain_dir.mkdir(parents=True)
    domains = []
    for suffix in ("a", "b"):
        summary_path = domain_dir / f"{suffix}_domain_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "schema_version": "arc.domain_summary.v4",
                    "domain_id": f"domain-{suffix}",
                    "domain_title": f"Domain {suffix.upper()}",
                    "overview": f"overview {suffix}",
                    "task_focus": {"research_scope": f"scope {suffix}"},
                    "methodology": [{"name": f"method {suffix}"}],
                    "known_solved_cases": [{"case": f"case {suffix}"}],
                    "open_axes_for_new_work": [{"axis": f"axis {suffix}"}],
                }
            ),
            encoding="utf-8",
        )
        domains.append(
            {
                "domain_id": f"domain-{suffix}",
                "seed_paper": f"seed:{suffix}",
                "title": f"Domain {suffix.upper()}",
                "summary_json_path": f"domain/{summary_path.name}",
            }
        )
    (domain_dir / "domain-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "arc.workflow.domain_manifest.v1",
                "domain_count": 2,
                "domains": domains,
            }
        ),
        encoding="utf-8",
    )


def _assert_strict_objects(schema: Any) -> None:
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            assert schema.get("additionalProperties") is False
            properties = schema.get("properties", {})
            assert set(schema.get("required", [])) == set(properties)
        for value in schema.values():
            _assert_strict_objects(value)
    elif isinstance(schema, list):
        for item in schema:
            _assert_strict_objects(item)
