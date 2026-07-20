from arc_paper.summary.providers.prompt import PromptProviderSummaryAdapter
from arc_paper.summary.schema import load_summary_schema


class RecoveredKimiPromptProvider:
    name = "kimi-code-cli"

    def __init__(self):
        self.calls = []

    def generate_json(self, prompt, *, schema, model):
        self.calls.append({"prompt": prompt, "schema": schema, "model": model})
        # arc-llm owns relaxed JSON recovery. The summary adapter receives the
        # recovered mapping and must preserve it while attaching provenance.
        return {
            "schema_version": "arc.paper_llm_summary.v1",
            "paper_id": "arXiv:0911.3380",
            "title": "A Test Paper",
            "authors_short": "Alice and Bob",
            "high_value_summary": ["The paper computes a useful result."],
            "toc": [],
            "section_summaries": [],
            "reading_guide": [],
            "warnings": ["provider_json_recovered"],
            "provenance": {},
        }


def test_kimi_prompt_adapter_uses_default_model_schema_and_valid_provenance():
    prompt_provider = RecoveredKimiPromptProvider()
    adapter = PromptProviderSummaryAdapter(prompt_provider, env={})
    task = {
        "prompt_version": "paper-summary-v1",
        "system_prompt": "Summarize the paper.",
        "user_prompt": "Return JSON only.",
        "input_pack": {
            "paper_id": "arXiv:0911.3380",
            "source_hash": "a" * 64,
        },
        "output_schema": load_summary_schema(),
    }

    summary = adapter.generate_summary(task)

    assert prompt_provider.calls[0]["model"] == "default_model"
    assert prompt_provider.calls[0]["schema"]["$id"] == "arc.paper-summary-v1"
    assert summary["warnings"] == ["provider_json_recovered"]
    assert summary["provenance"]["method"] == "kimi-code-cli"
    assert summary["provenance"]["model"] == "default_model"
    assert summary["provenance"]["source_hash"] == "a" * 64
