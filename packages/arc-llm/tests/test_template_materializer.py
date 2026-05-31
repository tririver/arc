from __future__ import annotations

from arc_llm.proposers_reviewer.template_materializer import (
    deep_merge,
    materialize_batch,
    materialize_loop,
    materialize_worker,
    replace_placeholders,
)


def test_deep_merge_merges_dicts_and_replaces_lists():
    result = deep_merge(
        {"runtime": {"allow_mcp": False, "allow_internet": False}, "items": [1]},
        {"runtime": {"allow_mcp": True}, "items": [2]},
    )

    assert result == {"runtime": {"allow_mcp": True, "allow_internet": False}, "items": [2]}


def test_replace_placeholders_recurses_through_json_like_values():
    result = replace_placeholders({"a": "<id>", "b": ["x-<id>"]}, {"<id>": "001"})

    assert result == {"a": "001", "b": ["x-001"]}


def test_materialize_worker_loop_and_batch_payload():
    proposer = materialize_worker(
        {"id": "template", "prompt": {"system": "s", "template": "hello <name>"}, "runtime": {"allow_mcp": False}},
        worker_id="proposer_001",
        overrides={"runtime": {"allow_mcp": True}},
        replacements={"<name>": "world"},
        output_schema={"type": "object"},
    )
    loop = materialize_loop(
        {"loop_id": "template", "max_rounds": 2, "early_stop": {"enabled": False}},
        loop_id="loop_001",
        caller_context={"user_intent": "intent"},
        proposers=[proposer],
        reviewers=[{"id": "reviewer_001", "prompt": {"system": "s", "template": "review"}}],
        session={"scope_id": "scope"},
        cache_context={"static_caller_context_keys": ["user_intent"]},
    )
    batch = materialize_batch(
        run_id="run_001",
        run_dir="/tmp/run",
        loops=[loop],
        session={"policy": "stateful"},
        max_concurrent_loops=1,
    )

    assert proposer["id"] == "proposer_001"
    assert proposer["prompt"]["template"] == "hello world"
    assert proposer["runtime"]["allow_mcp"] is True
    assert proposer["output_schema"] == {"type": "object"}
    assert loop["loop_id"] == "loop_001"
    assert loop["session"]["scope_id"] == "scope"
    assert batch["session"]["policy"] == "stateful"
