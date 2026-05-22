from __future__ import annotations

import json
import subprocess
from typing import Any

from .base import LLMWorkerError


class ClaudeCliProvider:
    name = "claude-cli"

    def generate_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        schema = schema or {"type": "object"}
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

        result = subprocess.run(cmd, input=prompt, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise LLMWorkerError(result.stderr or result.stdout or "claude -p failed")
        return _extract_json(result.stdout)

    def generate_text(self, prompt: str, *, model: str | None = None) -> str:
        cmd = [
            "claude",
            "-p",
            "--bare",
            "--tools",
            "",
            "--no-session-persistence",
        ]
        if model:
            cmd.extend(["--model", model])

        result = subprocess.run(cmd, input=prompt, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise LLMWorkerError(result.stderr or result.stdout or "claude -p failed")
        return result.stdout


def _extract_json(stdout: str) -> dict[str, Any]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise LLMWorkerError(f"Claude output was not JSON: {exc}") from exc
    if isinstance(payload, dict) and isinstance(payload.get("result"), str):
        try:
            nested = json.loads(payload["result"])
        except json.JSONDecodeError as exc:
            raise LLMWorkerError(f"Claude result field was not JSON: {exc}") from exc
        if not isinstance(nested, dict):
            raise LLMWorkerError("Claude result JSON was not an object")
        return nested
    if isinstance(payload, dict) and isinstance(payload.get("result"), dict):
        return payload["result"]
    if isinstance(payload, dict):
        return payload
    raise LLMWorkerError("Claude JSON output was not an object")
