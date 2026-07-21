from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

from arc_llm.evidence import EVIDENCE_REQUESTS_FIELD, allow_evidence_requests


pytestmark = pytest.mark.skipif(
    os.environ.get("ARC_RUN_LLM_TESTS") != "1" or os.environ.get("ARC_RUN_NET_TESTS") != "1",
    reason="true LLM CLI smoke test; set ARC_RUN_LLM_TESTS=1 and ARC_RUN_NET_TESTS=1 to run",
)


@pytest.mark.parametrize("provider,binary", [("codex-cli", "codex"), ("claude-cli", "claude")])
def test_stateful_run_json_cli_smoke(provider, binary, tmp_path):
    if shutil.which(binary) is None:
        pytest.skip(f"{binary} CLI is not installed")
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {"ok": {"type": "boolean"}, "round": {"type": "integer"}},
                "required": ["ok", "round"],
            }
        ),
        encoding="utf-8",
    )
    session_root = tmp_path / "sessions"
    base = [
        sys.executable,
        "-m",
        "arc_llm.cli",
        "run-json",
        "--provider",
        provider,
        "--session-policy",
        "stateful",
        "--session-root",
        str(session_root),
        "--session-key",
        f"smoke/{provider}",
        "--schema",
        str(schema_path),
        "--idle-timeout-seconds",
        "180",
    ]

    first = subprocess.run(
        [*base, "--prompt-text", 'return {"ok": true, "round": 1}'],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=180,
        check=False,
    )
    assert first.returncode == 0, first.stderr or first.stdout
    second = subprocess.run(
        [*base, "--prompt-text", 'now return {"ok": true, "round": 2}'],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=180,
        check=False,
    )
    assert second.returncode == 0, second.stderr or second.stdout
    assert (session_root / "calls.jsonl").is_file()


def test_codex_evidence_schema_strict_structured_output_smoke(tmp_path):
    if shutil.which("codex") is None:
        pytest.skip("codex CLI is not installed")
    schema = allow_evidence_requests(
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        }
    )
    assert schema is not None
    schema_path = tmp_path / "evidence-schema.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "arc_llm.cli",
            "run-json",
            "--provider",
            "codex-cli",
            "--schema",
            str(schema_path),
            "--prompt-text",
            "Return ok=true and arc_evidence_requests as an empty array.",
            "--json",
            "--idle-timeout-seconds",
            "180",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "HTTP 400" not in result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload[EVIDENCE_REQUESTS_FIELD] == []
