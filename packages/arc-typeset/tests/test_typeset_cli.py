from __future__ import annotations

import json

from arc_typeset import cli


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
