from __future__ import annotations

import json

from arc_llm.cache_audit import audit_run, first_difference


def test_cache_audit_summarizes_usage_and_duplicate_context(tmp_path):
    run_root = tmp_path / "run"
    sessions = run_root / "sessions"
    sessions.mkdir(parents=True)
    (sessions / "calls.jsonl").write_text(
        json.dumps(
            {
                "provider_used": "codex-cli",
                "model_used": "gpt-5.5",
                "runtime_fingerprint": "fp",
                "usage": {"input_tokens": 100, "cached_input_tokens": 70, "output_tokens": 5},
                "prompt_sha256": "p1",
                "static_prefix_sha256": "s1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    prompt_dir = run_root / "loops" / "loop_001" / "rounds" / "round_001" / "prompts"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "reviewer_001.md").write_text(
        "caller_context\n" + ("x" * 1200) + "\n## ARC Worker Context\n{\"caller_context\": \""
        + ("x" * 1200)
        + "\"}\n",
        encoding="utf-8",
    )

    result = audit_run(run_root)

    assert result["total_calls"] == 1
    assert result["total_input_tokens"] == 100
    assert result["total_cached_input_tokens"] == 70
    assert result["overall_cached_input_ratio"] == 0.7
    assert result["duplicate_context_warnings"]


def test_cache_audit_reads_shared_session_root_from_config(tmp_path):
    run_root = tmp_path / "runs" / "run_001"
    shared_sessions = tmp_path / "shared_sessions"
    shared_sessions.mkdir(parents=True)
    run_root.mkdir(parents=True)
    (run_root / "config.json").write_text(
        json.dumps(
            {
                "run_id": "run_001",
                "run_dir": str(tmp_path / "runs"),
                "session": {"root": str(shared_sessions), "reuse_across_batch_calls": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (shared_sessions / "calls.jsonl").write_text(
        json.dumps({"usage": {"input_tokens": 10, "cached_input_tokens": 5, "output_tokens": 2}}) + "\n",
        encoding="utf-8",
    )

    result = audit_run(run_root)

    assert result["total_calls"] == 1
    assert result["overall_cached_input_ratio"] == 0.5
    assert str(shared_sessions / "calls.jsonl") in result["session_call_paths"]


def test_cache_audit_reads_loop_session_root_from_config(tmp_path):
    run_root = tmp_path / "runs" / "run_001"
    shared_sessions = tmp_path / "loop_sessions"
    shared_sessions.mkdir(parents=True)
    run_root.mkdir(parents=True)
    (run_root / "config.json").write_text(
        json.dumps(
            {
                "run_id": "run_001",
                "run_dir": str(tmp_path / "runs"),
                "loops": [
                    {
                        "loop_id": "loop_001",
                        "session": {"root": str(shared_sessions), "reuse_across_batch_calls": True},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (shared_sessions / "calls.jsonl").write_text(
        json.dumps({"usage": {"input_tokens": 20, "cached_input_tokens": 10}}) + "\n",
        encoding="utf-8",
    )

    result = audit_run(run_root)

    assert result["total_calls"] == 1
    assert str(shared_sessions / "calls.jsonl") in result["session_call_paths"]


def test_cache_audit_reads_workflow_session_roots(tmp_path):
    run_root = tmp_path / "run"
    calc_sessions = run_root / "llm_sessions"
    idea_sessions = run_root / "idea_loops" / "sessions"
    calc_sessions.mkdir(parents=True)
    idea_sessions.mkdir(parents=True)
    (calc_sessions / "calls.jsonl").write_text(
        json.dumps({"usage": {"input_tokens": 10, "cached_input_tokens": 4}}) + "\n",
        encoding="utf-8",
    )
    (idea_sessions / "calls.jsonl").write_text(
        json.dumps({"usage": {"input_tokens": 20, "cached_input_tokens": 10}}) + "\n",
        encoding="utf-8",
    )

    result = audit_run(run_root)

    assert result["total_calls"] == 2
    assert result["total_input_tokens"] == 30
    assert str(calc_sessions / "calls.jsonl") in result["session_call_paths"]
    assert str(idea_sessions / "calls.jsonl") in result["session_call_paths"]


def test_cache_audit_counts_claude_cache_read_tokens(tmp_path):
    run_root = tmp_path / "run"
    sessions = run_root / "sessions"
    sessions.mkdir(parents=True)
    (sessions / "calls.jsonl").write_text(
        json.dumps(
            {
                "provider_used": "claude-cli",
                "usage": {
                    "cache_creation_input_tokens": 100,
                    "cache_read_input_tokens": 900,
                    "output_tokens": 5,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = audit_run(run_root)

    assert result["total_input_tokens"] == 1000
    assert result["total_cached_input_tokens"] == 900
    assert result["overall_cached_input_ratio"] == 0.9


def test_cache_audit_reports_schema_changes_inside_session(tmp_path):
    run_root = tmp_path / "run"
    sessions = run_root / "sessions"
    sessions.mkdir(parents=True)
    (sessions / "calls.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"session_key": "s", "schema_sha256": "schema-a", "usage": {}}),
                json.dumps({"session_key": "s", "schema_sha256": "schema-b", "usage": {}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = audit_run(run_root)

    assert result["schema_change_warnings"] == [
        {
            "session_key": "s",
            "distinct_schema_sha256_count": 2,
            "schema_sha256_values": ["schema-a", "schema-b"],
        }
    ]


def test_first_difference_reports_line_and_column():
    diff = first_difference("same\nabc\n", "same\naxc\n")

    assert diff["line"] == 2
    assert diff["column"] == 2
    assert diff["left_snippet"] == "abc"
    assert diff["right_snippet"] == "axc"
