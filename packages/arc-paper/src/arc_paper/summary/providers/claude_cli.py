from __future__ import annotations

from typing import Any, Callable, Mapping

from arc_llm.providers.claude_cli import ClaudeCliProvider as ClaudePromptProvider
from arc_llm.runner import run_json

from ..model import DEFAULT_SUMMARY_MODEL_TIER, resolve_summary_model
from ..schema import load_summary_schema, validate_summary
from ..checkpoint import current_provider_checkpoint
from .pipeline import apply_provider_provenance, generate_summary_with_section_pipeline


class ClaudeCliProvider:
    name = "claude-cli"

    def __init__(
        self,
        *,
        env: Mapping[str, str] | None = None,
        _test_prompt_provider: ClaudePromptProvider | None = None,
    ):
        self._test_prompt_provider = _test_prompt_provider
        self.env = env

    @property
    def prompt_provider(self) -> ClaudePromptProvider | None:
        return self._test_prompt_provider

    def generate_summary(
        self,
        task: dict,
        *,
        model: str | None = None,
        model_tier: str | None = DEFAULT_SUMMARY_MODEL_TIER,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict:
        resolved_model = resolve_summary_model(self.name, model, model_tier=model_tier)
        summary = generate_summary_with_section_pipeline(
            task,
            model=resolved_model,
            provider=self.name,
            run_json=lambda prompt, schema, selected_model: self._run_json(
                prompt, schema, selected_model, model_tier=model_tier
            ),
            progress_callback=progress_callback,
        )
        summary = apply_provider_provenance(summary, task, method=self.name, model=resolved_model)
        validate_summary(summary)
        return summary

    def _run_json(
        self, prompt: str, schema: dict, model: str | None, *, model_tier: str | None = None
    ) -> dict:
        output_schema = schema or load_summary_schema()
        if self._test_prompt_provider is not None:
            return self._test_prompt_provider.generate_json(prompt, schema=output_schema, model=model)
        artifact_dir, call_label = current_provider_checkpoint()
        return run_json(
            prompt,
            schema=output_schema,
            provider=self.name,
            model=model,
            model_tier=model_tier,
            env=self.env,
            session_policy="stateless",
            artifact_dir=artifact_dir,
            call_label=call_label or "arc-paper/summary",
        )
