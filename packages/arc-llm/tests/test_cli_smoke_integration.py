from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest


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
                "required": ["ok"],
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
    ]

    first = subprocess.run(
        [*base, "--prompt", 'return {"ok": true}'],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=180,
        check=False,
    )
    assert first.returncode == 0, first.stderr or first.stdout
    second = subprocess.run(
        [*base, "--prompt", 'now return {"ok": true, "round": 2}'],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=180,
        check=False,
    )
    assert second.returncode == 0, second.stderr or second.stdout
    assert (session_root / "calls.jsonl").is_file()
