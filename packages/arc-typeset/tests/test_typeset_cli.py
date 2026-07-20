from __future__ import annotations

import json

from arc_typeset import cli


def test_cli_json_wraps_dispatch_exception(monkeypatch, capsys):
    def fail(_args):
        raise RuntimeError("converter unavailable")

    monkeypatch.setattr(cli, "_dispatch", fail)

    assert cli.main(["md2pdf", "report.md", "--json"]) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["error"] == {
        "code": "command_failed",
        "message": "converter unavailable",
        "type": "RuntimeError",
    }


def test_cli_md2pdf_dispatches_to_converter(monkeypatch, tmp_path, capsys):
    source = tmp_path / "report.md"
    output = tmp_path / "report.pdf"
    source.write_text("# Report\n", encoding="utf-8")
    calls = {}

    def fake_convert_markdown_to_pdf(**kwargs):
        calls.update(kwargs)
        return {
            "ok": True,
            "data": {"input_path": str(source), "output_path": str(output), "pdf_size_bytes": 8},
            "errors": [],
            "meta": {},
        }

    monkeypatch.setattr(cli.md2pdf, "convert_markdown_to_pdf", fake_convert_markdown_to_pdf)

    assert cli.main(["md2pdf", str(source), "--output", str(output), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["data"]["output_path"] == str(output)
    assert calls["input_path"] == source
    assert calls["output_path"] == output


def test_cli_md2pdf_prints_output_path_for_human_success(monkeypatch, tmp_path, capsys):
    source = tmp_path / "report.md"
    output = tmp_path / "report.pdf"
    source.write_text("# Report\n", encoding="utf-8")

    monkeypatch.setattr(
        cli.md2pdf,
        "convert_markdown_to_pdf",
        lambda **kwargs: {
            "ok": True,
            "data": {"input_path": str(source), "output_path": str(output), "pdf_size_bytes": 8},
            "errors": [],
            "meta": {},
        },
    )

    assert cli.main(["md2pdf", str(source)]) == 0

    assert capsys.readouterr().out.strip() == str(output)


def test_cli_md2pdf_returns_nonzero_on_failure(monkeypatch, tmp_path, capsys):
    source = tmp_path / "report.md"
    source.write_text("# Report\n", encoding="utf-8")

    monkeypatch.setattr(
        cli.md2pdf,
        "convert_markdown_to_pdf",
        lambda **kwargs: {
            "ok": False,
            "error": {"code": "missing_dependency", "message": "xelatex not found on PATH"},
            "errors": [],
            "meta": {},
        },
    )

    assert cli.main(["md2pdf", str(source), "--json"]) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "missing_dependency"


def test_cli_translate_dispatches_to_translator_with_low_model_tier(monkeypatch, tmp_path, capsys):
    source = tmp_path / "report.md"
    output = tmp_path / "report.zh_CN.md"
    source.write_text("# Report\n", encoding="utf-8")
    calls = {}

    def fake_translate_markdown(**kwargs):
        calls.update(kwargs)
        return {
            "ok": True,
            "data": {
                "input_markdown_path": str(source),
                "output_markdown_path": str(output),
                "output_pdf_path": str(output.with_suffix(".pdf")),
            },
            "errors": [],
            "meta": {},
        }

    monkeypatch.setattr(cli.translate, "translate_markdown", fake_translate_markdown)

    assert cli.main(["translate", str(source), "--output", str(output), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert calls["input_path"] == source
    assert calls["output_path"] == output
    assert calls["target_language"] == "Chinese"
    assert calls["target_locale"] == "zh_CN"
    assert calls["model_tier"] == "low"
    assert calls["convert_pdf"] is True


def test_cli_batch_translate_dispatches_to_project_batch(monkeypatch, tmp_path, capsys):
    calls = {}

    def fake_batch_translate_project(**kwargs):
        calls.update(kwargs)
        return {
            "ok": True,
            "data": {"project_dir": str(tmp_path), "candidate_count": 1, "translated_count": 1},
            "errors": [],
            "meta": {},
        }

    monkeypatch.setattr(cli.translate, "batch_translate_project", fake_batch_translate_project)

    assert cli.main(["batch-translate", str(tmp_path), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert calls["project_dir"] == tmp_path
    assert calls["target_language"] == "Chinese"
    assert calls["target_locale"] == "zh_CN"
    assert calls["model_tier"] == "low"
