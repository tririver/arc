import json

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

    generate_summary_with_section_pipeline(
        task,
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


def _extract_final_input_pack(prompt):
    marker = "Input pack:\n"
    start = prompt.index(marker) + len(marker)
    end = prompt.index("\n\nReturn JSON only.", start)
    return json.loads(prompt[start:end])
