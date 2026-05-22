from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

from ..model import resolve_summary_model
from ..schema import load_summary_schema, validate_summary
from .base import LLMProviderError
from .pipeline import apply_provider_provenance, generate_summary_with_section_pipeline


class CodexCliProvider:
    name = "codex-cli"

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
        with tempfile.TemporaryDirectory(prefix="arc-paper-summary-") as tmp:
            tmpdir = Path(tmp)
            schema_path = tmpdir / "summary.schema.json"
            output_path = tmpdir / "summary.output.json"
            prompt_path = tmpdir / "prompt.txt"
            schema = schema or load_summary_schema()
            schema_path.write_text(json.dumps(schema, ensure_ascii=False), encoding="utf-8")
            prompt_path.write_text(prompt, encoding="utf-8")

            cmd = [
                "codex",
                "exec",
                "--skip-git-repo-check",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(output_path),
            ]
            if model:
                cmd.extend(["-m", model])
            cmd.append("-")

            result = subprocess.run(cmd, input=prompt, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                raise LLMProviderError(result.stderr or result.stdout or "codex exec failed")
            try:
                payload = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise LLMProviderError(f"Could not read Codex summary output: {exc}") from exc
            return payload
