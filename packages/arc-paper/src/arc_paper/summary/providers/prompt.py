from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from arc_llm.runner import run_json

from ..model import DEFAULT_SUMMARY_MODEL_TIER, resolve_summary_model
from ..schema import load_summary_schema, validate_summary
from ..checkpoint import current_provider_checkpoint, current_schema_canary_root
from .pipeline import apply_provider_provenance, generate_summary_with_section_pipeline


class PromptProviderSummaryAdapter:
    def __init__(
        self,
        prompt_provider=None,
        *,
        provider_name: str | None = None,
        env: Mapping[str, str] | None = None,
        process_chain: Sequence[str] | None = None,
    ) -> None:
        self.prompt_provider = prompt_provider
        self.name = provider_name or prompt_provider.name
        self.env = env
        self.process_chain = list(process_chain) if process_chain is not None else None

    def generate_summary(
        self,
        task: dict,
        *,
        model: str | None = None,
        model_tier: str | None = DEFAULT_SUMMARY_MODEL_TIER,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict:
        resolved_model = resolve_summary_model(self.name, model, model_tier=model_tier, env=self.env)
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
        if self.prompt_provider is not None:
            return self.prompt_provider.generate_json(prompt, schema=output_schema, model=model)
        artifact_dir, call_label = current_provider_checkpoint()
        return run_json(
            prompt,
            schema=output_schema,
            provider=self.name,
            model=model,
            model_tier=model_tier,
            env=self.env,
            process_chain=self.process_chain,
            session_policy="stateless",
            artifact_dir=artifact_dir,
            schema_canary_root=current_schema_canary_root(),
            call_label=call_label or "arc-paper/summary",
        )
