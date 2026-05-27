from __future__ import annotations

import copy
import json

import pytest

from arc_llm.proposers_reviewer.config import ConfigError, load_batch_config, worker_env
from arc_llm.proposers_reviewer.prompts import render_prompt, reviewer_context


def minimal_config() -> dict:
    return {
        "schema_version": "arc.llm.proposers_reviewer_batch.config.v1",
        "run_id": "idea-test",
        "run_dir": "project/ideas",
        "max_concurrent_loops": 2,
        "defaults": {
            "provider": "auto",
            "model_tier": "high",
            "runtime": {
                "allow_internet": False,
                "allow_mcp": False,
            },
        },
        "loops": [
            {
                "loop_id": "idea_001",
                "max_rounds": 5,
                "early_stop": {"enabled": False},
                "proposers": [
                    {
                        "id": "proposer_001",
                        "prompt": {"system": "sys", "template": "prop {round_number}"},
                        "output_schema": {"type": "object"},
                        "runtime": {"allow_internet": True},
                    }
                ],
                "reviewers": [
                    {
                        "id": "reviewer_001",
                        "prompt": {"system": "sys", "template": "review {round_number}"},
                        "output_schema": {"type": "object"},
                        "runtime": {"allow_mcp": True, "claude_effort": "high"},
                    }
                ],
                "caller_context": {"user_intent": "test intent"},
            }
        ],
    }


def test_valid_config_parses_and_merges_defaults():
    config = load_batch_config(minimal_config())

    assert config.schema_version == "arc.llm.proposers_reviewer_batch.config.v1"
    assert str(config.run_dir) == "project/ideas"
    assert config.run_id == "idea-test"
    assert config.max_concurrent_loops == 2
    assert config.artifact_options.save_prompts is True
    assert len(config.loops) == 1

    loop = config.loops[0]
    assert loop.loop_id == "idea_001"
    assert loop.max_rounds == 5
    assert loop.early_stop_enabled is False
    assert loop.caller_context == {"user_intent": "test intent"}

    proposer = loop.proposers[0]
    reviewer = loop.reviewers[0]
    assert proposer.provider == "auto"
    assert proposer.model is None
    assert proposer.model_tier == "high"
    assert proposer.runtime["allow_internet"] is True
    assert proposer.runtime["allow_mcp"] is False
    assert reviewer.runtime["allow_mcp"] is True
    assert reviewer.runtime["claude_effort"] == "high"


def test_duplicate_loop_ids_fail():
    payload = minimal_config()
    payload["loops"].append(copy.deepcopy(payload["loops"][0]))

    with pytest.raises(ConfigError, match="duplicate loop_id"):
        load_batch_config(payload)


def test_multiple_reviewers_fail_in_v1():
    payload = minimal_config()
    payload["loops"][0]["reviewers"].append(
        {
            "id": "reviewer_002",
            "prompt": {"system": "sys", "template": "review"},
            "output_schema": {"type": "object"},
        }
    )

    with pytest.raises(ConfigError, match="exactly one reviewer"):
        load_batch_config(payload)


def test_duplicate_proposer_ids_fail():
    payload = minimal_config()
    payload["loops"][0]["proposers"].append(copy.deepcopy(payload["loops"][0]["proposers"][0]))

    with pytest.raises(ConfigError, match="duplicate proposer id"):
        load_batch_config(payload)


def test_runtime_merge_does_not_mutate_input():
    payload = minimal_config()
    original = copy.deepcopy(payload)

    config = load_batch_config(payload)

    assert payload == original
    assert config.loops[0].proposers[0].runtime is not payload["loops"][0]["proposers"][0]["runtime"]


def test_worker_env_maps_runtime_without_mutating_base_env():
    config = load_batch_config(minimal_config())
    proposer = config.loops[0].proposers[0]
    base_env = {"ARC_AGENT_HOST": "codex", "KEEP": "value"}

    env = worker_env(proposer, base_env=base_env)

    assert base_env == {"ARC_AGENT_HOST": "codex", "KEEP": "value"}
    assert env["ARC_AGENT_HOST"] == "codex"
    assert env["KEEP"] == "value"
    assert env["ARC_CODEX_ALLOW_INTERNET"] == "true"
    assert env["ARC_CLAUDE_ALLOW_INTERNET"] == "true"
    assert "ARC_CODEX_ENABLE_MCP" not in env
    assert env["ARC_LLM_MODEL_TIER"] == "high"
    assert env["ARC_CODEX_REASONING_EFFORT"] == "xhigh"


def test_worker_env_maps_mcp_model_and_provider_options():
    config = load_batch_config(minimal_config())
    reviewer = config.loops[0].reviewers[0]

    env = worker_env(reviewer, base_env={})

    assert env["ARC_CODEX_ENABLE_MCP"] == "true"
    assert env["ARC_CLAUDE_ALLOW_MCP"] == "true"
    assert env["ARC_LLM_MODEL_TIER"] == "high"
    assert env["ARC_CODEX_REASONING_EFFORT"] == "xhigh"
    assert env["ARC_CLAUDE_EFFORT"] == "high"


def test_worker_env_maps_arc_only_mcp_and_codex_filesystem_options():
    payload = minimal_config()
    runtime = payload["loops"][0]["reviewers"][0]["runtime"]
    runtime.update(
        {
            "mcp_mode": "arc-only",
            "arc_mcp_command": "/tmp/arc-mcp",
            "arc_mcp_env": {"ARC_PAPER_CACHE": "/tmp/arc-paper"},
            "codex_work_dir": "/tmp/project",
            "codex_add_dirs": ["/tmp/project/skills", "/tmp/arc-skills"],
        }
    )
    config = load_batch_config(payload)

    env = worker_env(config.loops[0].reviewers[0], base_env={})

    assert env["ARC_CODEX_ENABLE_MCP"] == "true"
    assert env["ARC_CODEX_MCP_MODE"] == "arc-only"
    assert env["ARC_CODEX_ARC_MCP_COMMAND"] == "/tmp/arc-mcp"
    assert json.loads(env["ARC_CODEX_ARC_MCP_ENV_JSON"]) == {"ARC_PAPER_CACHE": "/tmp/arc-paper"}
    assert env["ARC_CODEX_WORK_DIR"] == "/tmp/project"
    assert json.loads(env["ARC_CODEX_ADD_DIRS"]) == ["/tmp/project/skills", "/tmp/arc-skills"]


def test_worker_env_rejects_invalid_mcp_mode():
    payload = minimal_config()
    payload["loops"][0]["reviewers"][0]["runtime"]["mcp_mode"] = "broad"
    config = load_batch_config(payload)

    with pytest.raises(ConfigError, match="mcp_mode"):
        worker_env(config.loops[0].reviewers[0], base_env={})


def test_worker_specific_model_tier_overrides_default_model_tier():
    payload = minimal_config()
    payload["loops"][0]["proposers"][0]["model_tier"] = "low"

    config = load_batch_config(payload)

    assert config.loops[0].proposers[0].model_tier == "low"


def test_artifact_options_can_disable_prompt_saving():
    payload = minimal_config()
    payload["artifact_options"] = {"save_prompts": False}

    config = load_batch_config(payload)

    assert config.artifact_options.save_prompts is False


def test_invalid_model_tier_fails():
    payload = minimal_config()
    payload["defaults"]["model_tier"] = "strong"

    with pytest.raises(ConfigError, match="model_tier must be one of"):
        load_batch_config(payload)


def test_worker_runtime_can_disable_appended_full_context():
    payload = minimal_config()
    payload["loops"][0]["caller_context"] = {
        "user_intent": "test intent",
        "domain_markdown_files": [{"path": "domain/report.md", "content": "domain text"}],
    }
    reviewer_payload = payload["loops"][0]["reviewers"][0]
    reviewer_payload["runtime"]["append_context"] = False
    reviewer_payload["prompt"]["template"] = (
        "review only this\n{current_proposer_outputs_json}\n{correspondence_json}"
    )
    config = load_batch_config(payload)
    loop = config.loops[0]
    reviewer = loop.reviewers[0]
    context = reviewer_context(
        loop=loop,
        worker=reviewer,
        round_number=1,
        correspondence=[],
        current_proposer_outputs={"proposer_001": {"title": "visible idea"}},
    )

    prompt = render_prompt(reviewer, context)

    assert "visible idea" in prompt
    assert "domain_markdown_files" not in prompt
    assert "domain text" not in prompt
    assert "## ARC Worker Context" not in prompt
