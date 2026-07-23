import json

from arc_llm import runner as llm_runner
from arc_llm.budget import SharedBudget, shared_budget_context
from arc_llm.usage import LLMProviderResponse, LLMUsage
from arc_paper.summary.checkpoint import current_provider_checkpoint
from arc_paper.summary.providers.pipeline import generate_summary_with_section_pipeline


def test_section_pipeline_summarizes_sections_sequentially_and_uses_compact_final_pack(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    calls = []
    final_prompts = []
    events = []

    def run_json(prompt, schema, model):
        if schema.get("$id") == "arc.section-summary-v1":
            section_id = "S1" if '"section_id": "S1"' in prompt else "S2"
            calls.append(section_id)
            return {
                "section_id": section_id,
                "title": f"{section_id} title",
                "summary": f"{section_id} compact summary.",
                "warnings": [],
            }

        final_prompts.append(prompt)
        return {
            "schema_version": "arc.paper_summary_synthesis.v1",
            "paper_id": "arXiv:0911.3380",
            "title": "A Test Paper",
            "authors_short": "Alice and Bob",
            "high_value_summary": ["The final summary uses compact section summaries."],
            "reading_guide": [],
            "warnings": [],
        }

    task = {
        "pipeline": "section_then_paper",
        "prompt_version": "paper-summary-v1",
        "system_prompt": "system",
        "user_prompt": "user",
        "input_pack": {
            "paper_id": "arXiv:0911.3380",
            "metadata": {"title": "A Test Paper", "abstract": "A useful abstract."},
            "toc": [{"id": "S1", "title": "S1 title", "level": 2}, {"id": "S2", "title": "S2 title", "level": 2}],
            "sections": [
                {"section_id": "S1", "title": "S1 title", "text": "raw S1 text"},
                {"section_id": "S2", "title": "S2 title", "text": "raw S2 text"},
            ],
            "source_hash": "b" * 64,
        },
        "output_schema": {"$id": "arc.paper-summary-v1"},
    }

    result = generate_summary_with_section_pipeline(
        task,
        model="test-model",
        run_json=run_json,
        progress_callback=events.append,
    )

    assert result["title"] == "A Test Paper"
    assert result["schema_version"] == "arc.paper_llm_summary.v1"
    assert result["toc"] == [
        {"section_id": "S1", "title": "S1 title", "level": 2},
        {"section_id": "S2", "title": "S2 title", "level": 2},
    ]
    assert result["section_summaries"] == [
        {"section_id": "S1", "title": "S1 title", "summary": "S1 compact summary.", "warnings": []},
        {"section_id": "S2", "title": "S2 title", "summary": "S2 compact summary.", "warnings": []},
    ]
    assert calls == ["S1", "S2"]
    assert [event["event"] for event in events] == [
        "sections_started",
        "section_started",
        "section_completed",
        "section_started",
        "section_completed",
        "final_started",
        "final_completed",
    ]
    assert events[2]["sections_completed"] == 1
    final_pack = _extract_final_input_pack(final_prompts[0])
    assert "sections" not in final_pack
    assert "references" not in final_pack
    assert final_pack["section_summaries"][0]["summary"] == "S1 compact summary."
    assert "raw S1 text" not in final_prompts[0]
    assert "Do not output table of contents, section_summaries, or provenance" in final_prompts[0]

    calls.clear()
    final_prompts.clear()
    cached_events = []

    task_with_string_false_refresh = {**task, "refresh": "false"}
    generate_summary_with_section_pipeline(
        task_with_string_false_refresh,
        model="test-model",
        run_json=run_json,
        progress_callback=cached_events.append,
    )

    assert calls == []
    assert [event["event"] for event in cached_events[:3]] == [
        "sections_started",
        "section_cached",
        "section_cached",
    ]


def test_section_pipeline_keeps_section_cache_model_specific(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path))
    section_models = []

    def run_json(prompt, schema, model):
        if schema.get("$id") == "arc.section-summary-v1":
            section_models.append(model)
            return {
                "section_id": "S1",
                "title": "S1 title",
                "summary": f"{model} section summary.",
                "warnings": [],
            }

        return {
            "schema_version": "arc.paper_summary_synthesis.v1",
            "paper_id": "arXiv:0911.3380",
            "title": "A Test Paper",
            "authors_short": "Alice and Bob",
            "high_value_summary": ["The final summary uses compact section summaries."],
            "reading_guide": [],
            "warnings": [],
        }

    task = {
        "pipeline": "section_then_paper",
        "prompt_version": "paper-summary-v1",
        "system_prompt": "system",
        "user_prompt": "user",
        "input_pack": {
            "paper_id": "arXiv:0911.3380",
            "metadata": {"title": "A Test Paper", "abstract": "A useful abstract."},
            "toc": [{"id": "S1", "title": "S1 title", "level": 2}],
            "sections": [{"section_id": "S1", "title": "S1 title", "text": "raw S1 text"}],
            "source_hash": "c" * 64,
        },
        "output_schema": {"$id": "arc.paper-summary-v1"},
    }

    generate_summary_with_section_pipeline(task, model="cheap-model", run_json=run_json)
    result = generate_summary_with_section_pipeline(task, model="quality-model", run_json=run_json)

    assert section_models == ["cheap-model", "quality-model"]
    assert result["section_summaries"][0]["summary"] == "quality-model section summary."


def test_section_and_final_calls_share_budget_and_checkpoint_replay_is_free(
    monkeypatch, tmp_path,
):
    monkeypatch.setenv("ARC_PAPER_CACHE", str(tmp_path / "cache"))
    provider_calls = []

    class Provider:
        name = "codex-cli"

        def generate_json_result(self, prompt, *, schema, **kwargs):
            provider_calls.append(schema.get("$id"))
            if schema.get("$id") == "arc.section-summary-v1":
                section_id = "S1" if '"section_id": "S1"' in prompt else "S2"
                value = {
                    "section_id": section_id,
                    "title": f"{section_id} title",
                    "summary": f"{section_id} compact summary.",
                    "warnings": [],
                }
            else:
                value = {
                    "schema_version": "arc.paper_summary_synthesis.v1",
                    "paper_id": "arXiv:0911.3380",
                    "title": "A Test Paper",
                    "authors_short": "Alice and Bob",
                    "high_value_summary": ["A shared-budget result."],
                    "reading_guide": [],
                    "warnings": [],
                }
            return LLMProviderResponse(
                value,
                usage=LLMUsage(input_tokens=5, output_tokens=2),
            )

    monkeypatch.setattr(
        llm_runner, "select_provider", lambda *args, **kwargs: Provider(),
    )
    budget = SharedBudget.create(
        tmp_path / "budget.sqlite3",
        budget_id="summary-parent",
        max_calls=3,
        max_tokens=10_000,
    )
    task = {
        "pipeline": "section_then_paper",
        "prompt_version": "paper-summary-v1",
        "system_prompt": "system",
        "user_prompt": "user",
        "input_pack": {
            "paper_id": "arXiv:0911.3380",
            "metadata": {"title": "A Test Paper", "abstract": "Abstract."},
            "toc": [],
            "sections": [
                {"section_id": "S1", "title": "S1 title", "text": "one"},
                {"section_id": "S2", "title": "S2 title", "text": "two"},
            ],
            "source_hash": "d" * 64,
        },
        "output_schema": {"$id": "arc.paper-summary-v1"},
    }

    def run_budgeted(prompt, schema, model):
        artifact_dir, call_label = current_provider_checkpoint()
        assert artifact_dir is not None and call_label is not None
        return llm_runner.run_json(
            prompt,
            schema=schema,
            provider="codex-cli",
            env={},
            process_chain=[],
            artifact_dir=artifact_dir,
            call_label=call_label,
            idempotency_key=call_label,
        )

    with shared_budget_context(budget, output_reserve_tokens=100):
        generate_summary_with_section_pipeline(
            task, model="test-model", provider="codex-cli",
            run_json=run_budgeted,
        )
        first = budget.snapshot()
        generate_summary_with_section_pipeline(
            task, model="test-model", provider="codex-cli",
            run_json=run_budgeted,
        )
        replay = budget.snapshot()

    assert provider_calls == [
        "arc.section-summary-v1",
        "arc.section-summary-v1",
        "arc.paper-summary-synthesis-v1",
    ]
    assert first.charged_calls == 3
    assert first.charged_tokens == 21
    assert replay == first


def _extract_final_input_pack(prompt):
    marker = "Input pack:\n"
    start = prompt.index(marker) + len(marker)
    end = prompt.index("\n\nReturn JSON only.", start)
    return json.loads(prompt[start:end])
