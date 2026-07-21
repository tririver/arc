from __future__ import annotations

import json
from pathlib import Path

from arc_typeset import translate


def test_default_translated_paths_add_target_locale(tmp_path: Path) -> None:
    source = tmp_path / "report.md"

    assert translate.default_translated_markdown_path(source, "zh_CN") == tmp_path / "report.zh_CN.md"
    assert translate.default_translated_pdf_path(source, "zh_CN") == tmp_path / "report.zh_CN.pdf"


def test_translate_markdown_uses_low_tier_by_default_preserves_protected_blocks_and_writes_pdf(tmp_path: Path) -> None:
    source = tmp_path / "report.md"
    source.write_text(
        "# Report\n\n"
        "This paragraph explains the collapsed limit with $E=mc^2$.\n\n"
        "```python\n"
        "print('do not translate')\n"
        "```\n\n"
        "$$\n"
        "a^2+b^2=c^2\n"
        "$$\n",
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []
    pdf_calls: list[dict[str, object]] = []

    def fake_json_runner(prompt, *, schema=None, provider="auto", model=None, model_tier=None):
        calls.append(
            {
                "prompt": prompt,
                "schema": schema,
                "provider": provider,
                "model": model,
                "model_tier": model_tier,
            }
        )
        if "technical glossary" in prompt:
            return {"glossary": [{"source": "collapsed limit", "target": "坍缩极限"}]}
        payload = json.loads(prompt.split("BLOCKS_JSON:\n", 1)[1])
        return {
            "translations": [
                {"id": item["id"], "text": f"ZH:{item['text']}"}
                for item in payload["blocks"]
            ]
        }

    def fake_pdf_converter(**kwargs):
        pdf_calls.append(kwargs)
        output = Path(kwargs["output_path"])
        output.write_bytes(b"%PDF")
        return {"ok": True, "data": {"output_path": str(output), "pdf_size_bytes": 4}, "errors": [], "meta": {}}

    result = translate.translate_markdown(
        source,
        json_runner=fake_json_runner,
        pdf_converter=fake_pdf_converter,
    )

    assert result["ok"] is True
    output = Path(result["data"]["output_markdown_path"])
    assert output == tmp_path / "report.zh_CN.md"
    text = output.read_text(encoding="utf-8")
    assert "ZH:# Report" in text
    assert "ZH:This paragraph explains the collapsed limit with $E=mc^2$." in text
    assert "```python\nprint('do not translate')\n```" in text
    assert "$$\na^2+b^2=c^2\n$$" in text
    assert [call["model_tier"] for call in calls] == ["low", "low"]
    assert pdf_calls[0]["input_path"] == output
    assert pdf_calls[0]["output_path"] == tmp_path / "report.zh_CN.pdf"


def test_translate_markdown_preserves_block_newlines_when_llm_drops_them(tmp_path: Path) -> None:
    source = tmp_path / "report.md"
    source.write_text("# Report\n\nParagraph.\n\n## Next\n\n- item\n", encoding="utf-8")

    def fake_json_runner(prompt, *, schema=None, provider="auto", model=None, model_tier=None):
        if "technical glossary" in prompt:
            return {"glossary": []}
        payload = json.loads(prompt.split("BLOCKS_JSON:\n", 1)[1])
        translations = {
            "# Report\n": "# 报告",
            "Paragraph.\n": "段落。",
            "## Next\n": "## 下一节",
            "- item\n": "- 条目",
        }
        return {"translations": [{"id": item["id"], "text": translations[item["text"]]} for item in payload["blocks"]]}

    result = translate.translate_markdown(source, convert_pdf=False, json_runner=fake_json_runner)

    assert result["ok"] is True
    text = Path(result["data"]["output_markdown_path"]).read_text(encoding="utf-8")
    assert text == "# 报告\n\n段落。\n\n## 下一节\n\n- 条目\n"


def test_translate_markdown_fails_when_translation_ids_do_not_match(tmp_path: Path) -> None:
    source = tmp_path / "report.md"
    source.write_text("First paragraph.\n\nSecond paragraph.\n", encoding="utf-8")

    def fake_json_runner(prompt, *, schema=None, provider="auto", model=None, model_tier=None):
        if "technical glossary" in prompt:
            return {"glossary": []}
        payload = json.loads(prompt.split("BLOCKS_JSON:\n", 1)[1])
        return {"translations": [{"id": payload["blocks"][0]["id"], "text": "Only first"}]}

    result = translate.translate_markdown(source, convert_pdf=False, json_runner=fake_json_runner)

    assert result["ok"] is False
    assert result["error"]["code"] == "translation_output_invalid"
    assert "missing translation ids" in result["error"]["message"]
    assert not (tmp_path / "report.zh_CN.md").exists()


def test_translate_markdown_preserves_single_line_display_math_and_latex_environments(tmp_path: Path) -> None:
    source = tmp_path / "report.md"
    source.write_text(
        "Intro sentence.\n\n"
        "$$ E = mc^2 $$\n\n"
        "\\begin{align}\n"
        "a &= b \\\\\n"
        "\\end{align}\n\n"
        "Done sentence.\n",
        encoding="utf-8",
    )

    def fake_json_runner(prompt, *, schema=None, provider="auto", model=None, model_tier=None):
        if "technical glossary" in prompt:
            return {"glossary": []}
        payload = json.loads(prompt.split("BLOCKS_JSON:\n", 1)[1])
        return {"translations": [{"id": item["id"], "text": f"ZH:{item['text']}"} for item in payload["blocks"]]}

    result = translate.translate_markdown(source, convert_pdf=False, json_runner=fake_json_runner)

    assert result["ok"] is True
    text = Path(result["data"]["output_markdown_path"]).read_text(encoding="utf-8")
    assert "ZH:Intro sentence." in text
    assert "$$ E = mc^2 $$" in text
    assert "\\begin{align}\na &= b \\\\\n\\end{align}" in text
    assert "ZH:$$ E = mc^2 $$" not in text
    assert "ZH:\\begin{align}" not in text


def test_normalize_pipe_table_widths_gives_long_text_columns_more_pdf_width() -> None:
    markdown = (
        "| 排名 | 循环 | 标题 |\n"
        "|---:|---|---|\n"
        "| 1 | domain_idea_002 | 最小准单场暴胀中的标量化学势交换导致的坍缩极限暴胀子四点谱 |\n"
    )

    normalized = translate.normalize_markdown_for_pdf(markdown)

    separator = normalized.splitlines()[1]
    cells = [cell.strip() for cell in separator.strip("|").split("|")]
    assert len(cells[2].strip(":")) > len(cells[0].strip(":"))
    assert len(cells[2].strip(":")) > len(cells[1].strip(":"))


def test_normalize_markdown_for_pdf_restores_heading_and_list_block_spacing() -> None:
    markdown = "# 标题\n段落。\n## 下一节\n- 条目\n"

    normalized = translate.normalize_markdown_for_pdf(markdown)

    assert normalized == "# 标题\n\n段落。\n\n## 下一节\n\n- 条目\n"


def test_translate_markdown_runs_quality_pass_only_when_requested(tmp_path: Path) -> None:
    source = tmp_path / "report.md"
    source.write_text("A sentence.\n", encoding="utf-8")
    call_count = 0

    def fake_json_runner(prompt, *, schema=None, provider="auto", model=None, model_tier=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"glossary": []}
        if call_count == 2:
            payload = json.loads(prompt.split("BLOCKS_JSON:\n", 1)[1])
            return {"translations": [{"id": item["id"], "text": "草稿"} for item in payload["blocks"]]}
        return {"revised_markdown": "精修\n", "issues": []}

    result = translate.translate_markdown(
        source,
        quality=True,
        convert_pdf=False,
        json_runner=fake_json_runner,
    )

    assert result["ok"] is True
    assert call_count == 3
    assert Path(result["data"]["output_markdown_path"]).read_text(encoding="utf-8") == "精修\n"
    assert result["data"]["quality_pass"] is True


def test_translate_quality_scope_marks_prefix_limited_review(tmp_path: Path) -> None:
    source = tmp_path / "report.md"
    source.write_text("A" * 31000 + "\n", encoding="utf-8")
    call_count = 0

    def fake_json_runner(prompt, *, schema=None, provider="auto", model=None, model_tier=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"glossary": []}
        if call_count == 2:
            payload = json.loads(prompt.split("BLOCKS_JSON:\n", 1)[1])
            return {"translations": [{"id": item["id"], "text": item["text"]} for item in payload["blocks"]]}
        return {"revised_markdown": "reviewed\n", "issues": []}

    result = translate.translate_markdown(
        source,
        quality=True,
        convert_pdf=False,
        json_runner=fake_json_runner,
    )

    assert result["ok"] is True
    assert result["data"]["quality_scope"] == {
        "quality_scope": "first_30000_chars_only",
        "source_chars_checked": 30000,
        "draft_chars_checked": 30000,
        "full_document_checked": False,
    }
    output = Path(result["data"]["output_markdown_path"]).read_text(encoding="utf-8")
    assert output.startswith("reviewed\n")
    assert output.endswith("A" * 1000 + "\n")


def test_discover_batch_translation_candidates_requires_matching_md_pdf_and_skips_locale_outputs(tmp_path: Path) -> None:
    (tmp_path / "report.md").write_text("Report\n", encoding="utf-8")
    (tmp_path / "report.pdf").write_bytes(b"%PDF")
    (tmp_path / "notes.md").write_text("No PDF\n", encoding="utf-8")
    (tmp_path / "localized.zh_CN.md").write_text("Translated\n", encoding="utf-8")
    (tmp_path / "localized.zh_CN.pdf").write_bytes(b"%PDF")
    (tmp_path / "done.md").write_text("Done\n", encoding="utf-8")
    (tmp_path / "done.pdf").write_bytes(b"%PDF")
    (tmp_path / "done.zh_CN.pdf").write_bytes(b"%PDF")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "paper.md").write_text("Paper\n", encoding="utf-8")
    (nested / "paper.pdf").write_bytes(b"%PDF")

    candidates = translate.discover_batch_translation_candidates(tmp_path, target_locale="zh_CN")

    assert [Path(item["input_markdown_path"]).name for item in candidates] == ["report.md", "paper.md"]
    assert candidates[0]["output_markdown_path"].endswith("report.zh_CN.md")
    assert candidates[0]["output_pdf_path"].endswith("report.zh_CN.pdf")


def test_batch_translate_project_translates_discovered_candidates(tmp_path: Path) -> None:
    (tmp_path / "report.md").write_text("Report\n", encoding="utf-8")
    (tmp_path / "report.pdf").write_bytes(b"%PDF")
    calls: list[dict[str, object]] = []

    def fake_translator(**kwargs):
        calls.append(kwargs)
        return {
            "ok": True,
            "data": {
                "input_markdown_path": str(kwargs["input_path"]),
                "output_markdown_path": str(kwargs["output_path"]),
                "output_pdf_path": str(Path(kwargs["output_path"]).with_suffix(".pdf")),
            },
            "errors": [],
            "meta": {},
        }

    result = translate.batch_translate_project(tmp_path, translator=fake_translator)

    assert result["ok"] is True
    assert result["data"]["translated_count"] == 1
    assert calls[0]["input_path"] == tmp_path / "report.md"
    assert calls[0]["output_path"] == tmp_path / "report.zh_CN.md"
    assert calls[0]["model_tier"] == "low"
