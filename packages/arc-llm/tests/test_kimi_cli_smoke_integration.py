from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
ARC_TESTS_ROOT = REPO_ROOT / "arc-tests"
ARC_LLM_CALL_RECORD_FIELD = "arc_llm_call_record"

pytestmark = pytest.mark.skipif(
    os.environ.get("ARC_RUN_LLM_TESTS") != "1" or os.environ.get("ARC_RUN_NET_TESTS") != "1",
    reason="true Kimi CLI smoke test; set ARC_RUN_LLM_TESTS=1 and ARC_RUN_NET_TESTS=1 to run",
)


def test_kimi_cli_text_json_stateful_resume_and_auto_detection():
    kimi_binary = os.environ.get("ARC_KIMI_BIN", "kimi")
    if shutil.which(kimi_binary) is None:
        pytest.skip(f"{kimi_binary} CLI is not installed")

    _assert_arc_tests_root_is_ignored()
    ARC_TESTS_ROOT.mkdir(parents=True, exist_ok=True)
    run_root = Path(tempfile.mkdtemp(prefix="kimi-cli-smoke-", dir=ARC_TESTS_ROOT))
    work_dir = run_root / "work"
    session_root = run_root / "sessions"
    work_dir.mkdir()
    session_root.mkdir()

    schema_path = run_root / "schema.json"
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
    text_prompt = _write_prompt(run_root, "text-prompt.txt", "Reply with a short plain-text greeting.")
    first_prompt = _write_prompt(
        run_root,
        "json-first-prompt.txt",
        'Return the JSON object {"ok": true, "round": 1}.',
    )
    second_prompt = _write_prompt(
        run_root,
        "json-second-prompt.txt",
        'Continue this session and return the JSON object {"ok": true, "round": 2}.',
    )
    auto_prompt = _write_prompt(
        run_root,
        "auto-prompt.txt",
        'Return the JSON object {"ok": true}.',
    )

    env = _smoke_env(work_dir)
    explicit_text = _run_cli(
        [
            "run-text",
            "--provider",
            "kimi-code-cli",
            "--prompt",
            str(text_prompt),
            "--session-root",
            str(run_root / "text-call"),
        ],
        env=env,
        cwd=work_dir,
    )
    assert explicit_text.stdout.strip()

    stateful_base = [
        "run-json",
        "--provider",
        "kimi-code-cli",
        "--schema",
        str(schema_path),
        "--session-policy",
        "stateful",
        "--session-root",
        str(session_root),
        "--session-key",
        "smoke/kimi-code-cli",
        "--json",
    ]
    first = _json_stdout(
        _run_cli([*stateful_base, "--prompt", str(first_prompt)], env=env, cwd=work_dir)
    )
    second = _json_stdout(
        _run_cli([*stateful_base, "--prompt", str(second_prompt)], env=env, cwd=work_dir)
    )

    assert first["ok"] is True
    assert first["round"] == 1
    assert second["ok"] is True
    assert second["round"] == 2
    first_native_id = first[ARC_LLM_CALL_RECORD_FIELD]["native_session_id"]
    second_native_id = second[ARC_LLM_CALL_RECORD_FIELD]["native_session_id"]
    assert first_native_id
    assert second_native_id == first_native_id
    assert (session_root / "calls.jsonl").is_file()

    auto_env = dict(env)
    auto_env["ARC_AGENT_HOST"] = "kimi-code"
    auto = _json_stdout(
        _run_cli(
            [
                "run-json",
                "--provider",
                "auto",
                "--schema",
                str(schema_path),
                "--prompt",
                str(auto_prompt),
                "--session-root",
                str(run_root / "auto-call"),
                "--json",
            ],
            env=auto_env,
            cwd=work_dir,
        )
    )
    assert auto["ok"] is True
    assert auto[ARC_LLM_CALL_RECORD_FIELD]["provider_requested"] == "auto"
    assert auto[ARC_LLM_CALL_RECORD_FIELD]["provider_used"] == "kimi-code-cli"


def _assert_arc_tests_root_is_ignored() -> None:
    probe = ARC_TESTS_ROOT / "kimi-cli-smoke-ignore-probe"
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", str(probe.relative_to(REPO_ROOT))],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert result.returncode == 0, "Refusing real Kimi smoke test because arc-tests/ is not git-ignored"


def _write_prompt(run_root: Path, name: str, text: str) -> Path:
    path = run_root / name
    path.write_text(text, encoding="utf-8")
    return path


def _smoke_env(work_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["ARC_KIMI_WORK_DIR"] = str(work_dir)
    source_root = str(REPO_ROOT / "packages" / "arc-llm" / "src")
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (source_root, current_pythonpath)))
    return env


def _run_cli(args: list[str], *, env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, "-m", "arc_llm.cli", *args, "--timeout-seconds", "300"],
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result


def _json_stdout(result: subprocess.CompletedProcess[str]) -> dict:
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict)
    return payload
