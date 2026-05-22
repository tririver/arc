from __future__ import annotations

import json
import subprocess
from typing import Any, Callable

from ..model import resolve_summary_model
from ..schema import load_summary_schema, validate_summary
from .base import LLMProviderError
from .pipeline import apply_provider_provenance, generate_summary_with_section_pipeline


class ClaudeCliProvider:
    name = "claude-cli"

    def generate_summary(
        self,
        task: dict,
        *,
        model: str | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict:
        resolved_model = resolve_summary_model(self.name, model)
        summary = generate_summary_with_section_pipeline(
            task,
            model=resolved_model,
            run_json=self._run_json,
            progress_callback=progress_callback,
        )
        summary = apply_provider_provenance(summary, task, method=self.name, model=resolved_model)
        validate_summary(summary)
        return summary

    def _run_json(self, prompt: str, schema: dict, model: str | None) -> dict:
        schema = schema or load_summary_schema()
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
            raise LLMProviderError(result.stderr or result.stdout or "claude -p failed")
        return _extract_summary(result.stdout)


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
