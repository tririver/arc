from __future__ import annotations

import json
import subprocess

from ..schema import load_summary_schema, validate_summary
from .base import LLMProviderError
from .codex_cli import _task_prompt


class ClaudeCliProvider:
    name = "claude-cli"

    def generate_summary(self, task: dict, *, model: str | None = None) -> dict:
        schema = task.get("output_schema") or load_summary_schema()
        cmd = [
            "claude",
            "-p",
            "--bare",
            "--tools",
            "",
            "--no-session-persistence",
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(schema, ensure_ascii=False),
        ]
        if model:
            cmd.extend(["--model", model])
        cmd.append(_task_prompt(task))

        result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise LLMProviderError(result.stderr or result.stdout or "claude -p failed")
        summary = _extract_summary(result.stdout)
        validate_summary(summary)
        return summary


def _extract_summary(stdout: str) -> dict:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise LLMProviderError(f"Claude output was not JSON: {exc}") from exc
    if isinstance(payload, dict) and payload.get("schema_version") == "arc.paper_llm_summary.v1":
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("result"), str):
        try:
            nested = json.loads(payload["result"])
        except json.JSONDecodeError as exc:
            raise LLMProviderError(f"Claude result field was not summary JSON: {exc}") from exc
        return nested
    raise LLMProviderError("Claude output did not contain a paper summary")
