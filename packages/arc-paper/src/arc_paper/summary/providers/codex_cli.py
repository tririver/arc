from __future__ import annotations

from typing import Any, Callable, Mapping

from arc_llm.providers.codex_cli import CodexCliProvider as CodexPromptProvider
from arc_llm.runner import run_json

from ..model import resolve_summary_model
from ..schema import load_summary_schema, validate_summary
from .pipeline import apply_provider_provenance, generate_summary_with_section_pipeline


class CodexCliProvider:
    name = "codex-cli"

    def __init__(
        self,
        prompt_provider: CodexPromptProvider | None = None,
        *,
        env: Mapping[str, str] | None = None,
    ):
        self.prompt_provider = prompt_provider
        self.env = env

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
            provider=self.name,
            run_json=self._run_json,
            progress_callback=progress_callback,
        )
        summary = apply_provider_provenance(summary, task, method=self.name, model=resolved_model)
        validate_summary(summary)
        return summary

    def _run_json(self, prompt: str, schema: dict, model: str | None) -> dict:
        output_schema = schema or load_summary_schema()
        if self.prompt_provider is not None:
            return self.prompt_provider.generate_json(prompt, schema=output_schema, model=model)
        return run_json(
            prompt,
            schema=output_schema,
            provider=self.name,
            model=model,
            env=self.env,
            session_policy="stateless",
            call_label="arc-paper/summary",
        )
