from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .base import LLMWorkerError


class CodexCliProvider:
    name = "codex-cli"

    def generate_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        schema = schema or {"type": "object"}
        with tempfile.TemporaryDirectory(prefix="arc-llm-worker-") as tmp:
            tmpdir = Path(tmp)
            schema_path = tmpdir / "output.schema.json"
            output_path = tmpdir / "output.json"
            schema_path.write_text(json.dumps(schema, ensure_ascii=False), encoding="utf-8")

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
                raise LLMWorkerError(result.stderr or result.stdout or "codex exec failed")
            try:
                payload = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise LLMWorkerError(f"Could not read Codex JSON output: {exc}") from exc
            if not isinstance(payload, dict):
                raise LLMWorkerError("Codex JSON output was not an object")
            return payload

    def generate_text(self, prompt: str, *, model: str | None = None) -> str:
        with tempfile.TemporaryDirectory(prefix="arc-llm-worker-") as tmp:
            output_path = Path(tmp) / "output.txt"
            cmd = [
                "codex",
                "exec",
                "--skip-git-repo-check",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--output-last-message",
                str(output_path),
            ]
            if model:
                cmd.extend(["-m", model])
            cmd.append("-")

            result = subprocess.run(cmd, input=prompt, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                raise LLMWorkerError(result.stderr or result.stdout or "codex exec failed")
            try:
                return output_path.read_text(encoding="utf-8")
            except OSError as exc:
                raise LLMWorkerError(f"Could not read Codex text output: {exc}") from exc
