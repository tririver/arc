from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from ..schema import load_summary_schema, validate_summary
from .base import LLMProviderError


class CodexCliProvider:
    name = "codex-cli"

    def generate_summary(self, task: dict, *, model: str | None = None) -> dict:
        with tempfile.TemporaryDirectory(prefix="arc-paper-summary-") as tmp:
            tmpdir = Path(tmp)
            schema_path = tmpdir / "summary.schema.json"
            output_path = tmpdir / "summary.output.json"
            prompt_path = tmpdir / "prompt.txt"
            schema = task.get("output_schema") or load_summary_schema()
            schema_path.write_text(json.dumps(schema, ensure_ascii=False), encoding="utf-8")
            prompt = _task_prompt(task)
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
            cmd.append(prompt)

            result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                raise LLMProviderError(result.stderr or result.stdout or "codex exec failed")
            try:
                summary = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise LLMProviderError(f"Could not read Codex summary output: {exc}") from exc
            validate_summary(summary)
            return summary


def _task_prompt(task: dict) -> str:
    return "\n\n".join(
        part
        for part in [
            task.get("system_prompt", ""),
            task.get("user_prompt", ""),
            "Input pack:",
            json.dumps(task.get("input_pack", {}), ensure_ascii=False, indent=2),
            "Return JSON only.",
        ]
        if part
    )
