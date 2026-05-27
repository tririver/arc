from __future__ import annotations

import os
from pathlib import Path

from arc_typeset import md2pdf


def test_default_output_is_next_to_markdown(tmp_path: Path) -> None:
    source = tmp_path / "report.md"
    source.write_text("# Report\n", encoding="utf-8")

    assert md2pdf.default_output_path(source) == tmp_path / "report.pdf"


def test_pandoc_command_uses_xelatex_fonts_margin_and_resource_path(tmp_path: Path) -> None:
    source = tmp_path / "report.md"
    output = tmp_path / "out.pdf"
    source.write_text("# Report\n", encoding="utf-8")

    command = md2pdf.pandoc_command(
        source,
        output,
        margin="1.5cm",
        mainfont="Noto Sans CJK SC",
        cjk_mainfont="Noto Sans CJK SC",
        resource_paths=[source.parent, Path("/arc-dev")],
    )

    assert command == [
        "pandoc",
        str(source),
        "-o",
        str(output),
        "--pdf-engine=xelatex",
        f"--resource-path={os.pathsep.join([str(source.parent), '/arc-dev'])}",
        "-V",
        "geometry:margin=1.5cm",
        "-V",
        "mainfont=Noto Sans CJK SC",
        "-V",
        "CJKmainfont=Noto Sans CJK SC",
    ]


def test_convert_markdown_to_pdf_reports_missing_input(tmp_path: Path) -> None:
    result = md2pdf.convert_markdown_to_pdf(tmp_path / "missing.md")

    assert result["ok"] is False
    assert result["error"]["code"] == "input_not_found"


def test_convert_markdown_to_pdf_checks_pandoc_and_xelatex_before_running(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "report.md"
    source.write_text("# Report\n", encoding="utf-8")

    monkeypatch.setattr(md2pdf.shutil, "which", lambda name, path=None: None if name == "xelatex" else f"/bin/{name}")

    result = md2pdf.convert_markdown_to_pdf(source)

    assert result["ok"] is False
    assert result["error"]["code"] == "missing_dependency"
    assert "xelatex" in result["error"]["message"]


def test_convert_markdown_to_pdf_runs_pandoc_and_returns_output_metadata(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "report.md"
    source.write_text("# Report\n\nMath: $E=mc^2$.\n", encoding="utf-8")
    output = tmp_path / "report.pdf"
    calls = {}

    monkeypatch.setattr(md2pdf.shutil, "which", lambda name, path=None: f"/usr/bin/{name}")

    def fake_run(command, env=None, capture_output=False, text=False):
        calls["command"] = command
        calls["env"] = env
        output.write_bytes(b"%PDF test")
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(md2pdf.subprocess, "run", fake_run)

    result = md2pdf.convert_markdown_to_pdf(source, output_path=output)

    assert result["ok"] is True
    assert result["data"]["output_path"] == str(output)
    assert result["data"]["pdf_size_bytes"] == len(b"%PDF test")
    assert calls["command"][0] == "pandoc"
    assert "--pdf-engine=xelatex" in calls["command"]
    assert calls["env"]["PATH"]


def test_convert_markdown_to_pdf_returns_pandoc_failure(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "report.md"
    source.write_text("# Report\n", encoding="utf-8")

    monkeypatch.setattr(md2pdf.shutil, "which", lambda name, path=None: f"/usr/bin/{name}")
    monkeypatch.setattr(
        md2pdf.subprocess,
        "run",
        lambda command, env=None, capture_output=False, text=False: type(
            "Completed",
            (),
            {"returncode": 43, "stdout": "out", "stderr": "bad math"},
        )(),
    )

    result = md2pdf.convert_markdown_to_pdf(source)

    assert result["ok"] is False
    assert result["error"]["code"] == "conversion_failed"
    assert result["error"]["returncode"] == 43
    assert result["error"]["stderr"] == "bad math"
