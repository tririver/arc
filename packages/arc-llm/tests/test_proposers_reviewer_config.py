from __future__ import annotations

import copy
import json

import pytest

from arc_llm.call_record import ARC_LLM_CALL_RECORD_FIELD
from arc_llm.evidence import EVIDENCE_REQUESTS_FIELD
from arc_llm.proposers_reviewer.config import ConfigError, load_batch_config, worker_env
from arc_llm.proposers_reviewer.prompts import proposer_context, render_prompt, reviewer_context


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
    assert config.worker_idle_timeout_seconds is None
    assert config.session.policy == "stateful"
    assert config.session.history_mode == "delta"
    assert config.session.max_concurrent_same_prefix == 12
    assert len(config.loops) == 1

    loop = config.loops[0]
    assert loop.loop_id == "idea_001"
    assert loop.max_rounds == 5
    assert loop.early_stop_enabled is False
    assert loop.evidence_enabled is True
    assert loop.caller_context == {"user_intent": "test intent"}
    assert loop.session.policy == "stateful"

    proposer = loop.proposers[0]
    reviewer = loop.reviewers[0]
    assert proposer.provider == "auto"
    assert proposer.model is None
    assert proposer.model_tier == "high"
    assert proposer.runtime["allow_internet"] is True
    assert proposer.runtime["allow_mcp"] is False
    assert proposer.evidence_enabled is True
    assert EVIDENCE_REQUESTS_FIELD in proposer.output_schema.get("properties", {})
    assert ARC_LLM_CALL_RECORD_FIELD not in proposer.output_schema.get("properties", {})
    assert reviewer.runtime["allow_mcp"] is True
    assert reviewer.evidence_enabled is True
    assert reviewer.runtime["claude_effort"] == "high"


def test_idle_timeout_config_preserves_unspecified_and_parses_batch_and_worker_overrides():
    payload = minimal_config()
    assert load_batch_config(payload).worker_idle_timeout_seconds is None

    payload["worker_idle_timeout_seconds"] = 12
    payload["loops"][0]["proposers"][0]["worker_idle_timeout_seconds"] = 4
    config = load_batch_config(payload)

    assert config.worker_idle_timeout_seconds == 12
    assert config.loops[0].proposers[0].worker_idle_timeout_seconds == 4
    assert config.loops[0].reviewers[0].worker_idle_timeout_seconds is None


@pytest.mark.parametrize("location", ["batch", "worker"])
def test_removed_total_timeout_config_fails_fast_with_migration_hint(location):
    payload = minimal_config()
    if location == "batch":
        payload["worker_call_timeout_seconds"] = 12
    else:
        payload["loops"][0]["proposers"][0]["worker_call_timeout_seconds"] = 4

    with pytest.raises(ConfigError, match="removed; use worker_idle_timeout_seconds"):
        load_batch_config(payload)


def test_session_options_parse_and_loop_overrides_parent():
    payload = minimal_config()
    payload["session"] = {
        "policy": "stateless",
        "history_mode": "full",
        "scope_id": "ideas/run",
        "reuse_across_batch_calls": True,
        "max_concurrent_same_prefix": 4,
        "root": "project/ideas/shared-sessions",
        "cache_guard": {"enabled": True, "mode": "warn", "warmup_calls": 2, "min_cached_input_ratio": 0.5},
    }
    payload["loops"][0]["session"] = {"policy": "stateful", "history_mode": "delta", "scope_id": "ideas/run/loop"}
    payload["loops"][0]["cache_context"] = {
        "static_caller_context_keys": ["user_intent"],
        "volatile_caller_context_keys": ["idea_id"],
    }

    config = load_batch_config(payload)
    loop = config.loops[0]

    assert config.session.policy == "stateless"
    assert config.session.history_mode == "full"
    assert config.session.reuse_across_batch_calls is True
    assert config.session.max_concurrent_same_prefix == 4
    assert str(config.session.root) == "project/ideas/shared-sessions"
    assert loop.session.policy == "stateful"
    assert loop.session.history_mode == "delta"
    assert loop.session.scope_id == "ideas/run/loop"
    assert loop.session.root == config.session.root
    assert loop.cache_context.static_caller_context_keys == ["user_intent"]
    assert loop.cache_context.volatile_caller_context_keys == ["idea_id"]


def test_session_scope_rejects_absolute_or_parent_paths():
    payload = minimal_config()
    payload["session"] = {"scope_id": "../bad"}

    with pytest.raises(ConfigError, match="scope_id"):
        load_batch_config(payload)


def test_reuse_across_batch_calls_requires_scope_id():
    payload = minimal_config()
    payload["session"] = {"reuse_across_batch_calls": True}

    with pytest.raises(ConfigError, match="reuse_across_batch_calls requires session.scope_id"):
        load_batch_config(payload)


def test_string_booleans_parse_for_fail_fast_and_early_stop():
    payload = minimal_config()
    payload["fail_fast"] = "false"
    payload["loops"][0]["early_stop"]["enabled"] = "false"

    config = load_batch_config(payload)

    assert config.fail_fast is False
    assert config.loops[0].early_stop_enabled is False


def test_invalid_string_boolean_fails():
    payload = minimal_config()
    payload["loops"][0]["early_stop"]["enabled"] = "nope"

    with pytest.raises(ConfigError, match="early_stop.enabled"):
        load_batch_config(payload)


def test_evidence_capability_cascades_from_batch_loop_and_worker():
    batch_disabled = minimal_config()
    batch_disabled["evidence"] = {"enabled": False}
    batch_disabled["loops"][0]["evidence"] = {"enabled": True}
    disabled = load_batch_config(batch_disabled)

    assert disabled.evidence.enabled is False
    assert disabled.loops[0].evidence_enabled is False
    assert all(not worker.evidence_enabled for worker in disabled.loops[0].proposers + disabled.loops[0].reviewers)
    assert EVIDENCE_REQUESTS_FIELD not in disabled.loops[0].proposers[0].output_schema.get("properties", {})

    worker_disabled = minimal_config()
    worker_disabled["loops"][0]["proposers"][0]["evidence"] = {"enabled": False}
    enabled = load_batch_config(worker_disabled)

    assert enabled.loops[0].evidence_enabled is True
    assert enabled.loops[0].proposers[0].evidence_enabled is False
    assert enabled.loops[0].reviewers[0].evidence_enabled is True
    assert EVIDENCE_REQUESTS_FIELD not in enabled.loops[0].proposers[0].output_schema.get("properties", {})
    assert EVIDENCE_REQUESTS_FIELD in enabled.loops[0].reviewers[0].output_schema.get("properties", {})


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
    assert env["ARC_CODEX_ENABLE_MCP"] == "false"
    assert env["ARC_CLAUDE_ALLOW_MCP"] == "false"
    assert "ARC_LLM_MODEL_TIER" not in env
    assert env["ARC_CODEX_REASONING_EFFORT"] == "high"


def test_worker_env_false_runtime_clears_inherited_permission_flags():
    payload = minimal_config()
    payload["loops"][0]["proposers"][0]["runtime"] = {"allow_internet": False, "allow_mcp": False}
    config = load_batch_config(payload)
    worker = config.loops[0].proposers[0]
    base_env = {
        "ARC_CODEX_ALLOW_INTERNET": "true",
        "ARC_CLAUDE_ALLOW_INTERNET": "true",
        "ARC_CODEX_ENABLE_MCP": "true",
        "ARC_CLAUDE_ALLOW_MCP": "true",
        "ARC_CODEX_MCP_MODE": "arc-only",
        "ARC_CLAUDE_MCP_MODE": "arc-only",
        "ARC_CODEX_PROFILE": "mcp-profile",
        "ARC_CODEX_CONFIG": 'mcp_servers.arc.command="arc-mcp"',
        "ARC_CLAUDE_MCP_CONFIG": "/tmp/arc-mcp.json",
        "ARC_CLAUDE_TOOLS": "default",
        "ARC_CLAUDE_ALLOWED_TOOLS": "mcp__arc__*",
    }

    env = worker_env(worker, base_env=base_env)

    assert env["ARC_CODEX_ALLOW_INTERNET"] == "false"
    assert env["ARC_CLAUDE_ALLOW_INTERNET"] == "false"
    assert env["ARC_CODEX_ENABLE_MCP"] == "false"
    assert env["ARC_CLAUDE_ALLOW_MCP"] == "false"
    assert "ARC_CODEX_MCP_MODE" not in env
    assert "ARC_CLAUDE_MCP_MODE" not in env
    assert "ARC_CODEX_PROFILE" not in env
    assert "ARC_CODEX_CONFIG" not in env
    assert "ARC_CLAUDE_MCP_CONFIG" not in env
    assert "ARC_CLAUDE_TOOLS" not in env
    assert "ARC_CLAUDE_ALLOWED_TOOLS" not in env
    assert env["ARC_CODEX_IGNORE_USER_CONFIG"] == "true"
    assert env["ARC_CLAUDE_BARE"] == "true"


def test_worker_env_maps_mcp_model_and_provider_options():
    config = load_batch_config(minimal_config())
    reviewer = config.loops[0].reviewers[0]

    env = worker_env(reviewer, base_env={})

    assert env["ARC_CODEX_ENABLE_MCP"] == "true"
    assert env["ARC_CLAUDE_ALLOW_MCP"] == "true"
    assert "ARC_LLM_MODEL_TIER" not in env
    assert env["ARC_CODEX_REASONING_EFFORT"] == "high"
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
            "claude_allowed_tools": "mcp__arc__get_title",
        }
    )
    config = load_batch_config(payload)

    env = worker_env(config.loops[0].reviewers[0], base_env={})

    assert env["ARC_CODEX_ENABLE_MCP"] == "true"
    assert env["ARC_CODEX_MCP_MODE"] == "arc-only"
    assert env["ARC_CLAUDE_MCP_MODE"] == "arc-only"
    assert env["ARC_CODEX_ARC_MCP_COMMAND"] == "/tmp/arc-mcp"
    assert env["ARC_CLAUDE_ARC_MCP_COMMAND"] == "/tmp/arc-mcp"
    assert json.loads(env["ARC_CODEX_ARC_MCP_ENV_JSON"]) == {"ARC_PAPER_CACHE": "/tmp/arc-paper"}
    assert json.loads(env["ARC_CLAUDE_ARC_MCP_ENV_JSON"]) == {"ARC_PAPER_CACHE": "/tmp/arc-paper"}
    assert env["ARC_CLAUDE_ALLOWED_TOOLS"] == "mcp__arc__get_title"
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


def test_default_exact_model_requires_explicit_provider():
    payload = minimal_config()
    payload["defaults"]["provider"] = "auto"
    payload["defaults"]["model"] = "gpt-5.5"

    with pytest.raises(ConfigError, match="defaults.model requires explicit provider"):
        load_batch_config(payload)


def test_worker_exact_model_requires_explicit_provider():
    payload = minimal_config()
    payload["defaults"].pop("model_tier")
    payload["loops"][0]["proposers"][0]["model"] = "gpt-5.5"

    with pytest.raises(ConfigError, match=r"idea_001\.proposers\.proposer_001\.model requires explicit provider"):
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


def test_worker_prompt_context_strips_arc_llm_call_records():
    payload = minimal_config()
    payload["loops"][0]["caller_context"] = {
        "accepted": {
            "arc_llm_call_record": {"provider_used": "codex-cli"},
            "result": "keep",
        }
    }
    config = load_batch_config(payload)
    loop = config.loops[0]
    call_record = {"provider_used": "codex-cli"}
    correspondence = [
        {
            "type": "proposer_output",
            "output": {
                "title": "visible",
                "arc_llm_call_record": call_record,
            },
        }
    ]

    proposer = proposer_context(
        loop=loop,
        worker=loop.proposers[0],
        round_number=2,
        correspondence=correspondence,
    )
    reviewer = reviewer_context(
        loop=loop,
        worker=loop.reviewers[0],
        round_number=1,
        correspondence=correspondence,
        current_proposer_outputs={
            "proposer_001": {
                "title": "visible",
                "arc_llm_call_record": call_record,
            }
        },
    )

    rendered = render_prompt(loop.reviewers[0], reviewer)

    assert "arc_llm_call_record" not in json.dumps(proposer, ensure_ascii=False)
    assert "arc_llm_call_record" not in rendered
    assert "visible" in rendered
    assert "keep" in json.dumps(proposer, ensure_ascii=False)
