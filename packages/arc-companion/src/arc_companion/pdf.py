from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
from typing import Callable


class PDFError(RuntimeError):
    """Raised when LaTeX compilation or PDF inspection fails."""


def compile_latex(tex_path: Path, pdf_path: Path, *, timeout_seconds: float = 300.0) -> None:
    executable = shutil.which("latexmk")
    if executable is None:
        raise PDFError("latexmk is required to build a companion PDF")
    command = [
        executable,
        "-xelatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-outdir={tex_path.parent}",
        tex_path.name,
    ]
    completed = subprocess.run(
        command,
        cwd=tex_path.parent,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    built = tex_path.with_suffix(".pdf")
    if completed.returncode != 0 or not built.is_file() or built.stat().st_size == 0:
        tail = "\n".join((completed.stdout + "\n" + completed.stderr).splitlines()[-30:])
        raise PDFError(f"XeLaTeX compilation failed:\n{tail}")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if built.resolve() != pdf_path.resolve():
        shutil.copy2(built, pdf_path)


def validate_pdf(pdf_path: Path, *, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> dict[str, object]:
    if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
        raise PDFError(f"PDF is missing or empty: {pdf_path}")
    tools = {name: shutil.which(name) for name in ("pdfinfo", "pdftotext", "pdffonts", "pdftoppm")}
    missing = [name for name, path in tools.items() if path is None]
    if missing:
        raise PDFError(f"PDF validation tools are required: {', '.join(missing)}")

    info = _run(runner, [str(tools["pdfinfo"]), str(pdf_path)])
    pages, encrypted = _parse_pdfinfo(info)
    if encrypted:
        raise PDFError("PDF is encrypted")

    text_path = pdf_path.with_suffix(".validation.txt")
    _run(runner, [str(tools["pdftotext"]), str(pdf_path), str(text_path)])
    if not text_path.is_file() or not text_path.read_text(encoding="utf-8", errors="ignore").strip():
        raise PDFError("PDF contains no searchable text")

    fonts = _run(runner, [str(tools["pdffonts"]), str(pdf_path)])
    font_count = _validate_embedded_fonts(fonts)

    render_paths: list[str] = []
    for page in range(1, pages + 1):
        raster_prefix = pdf_path.with_suffix("").with_name(f"{pdf_path.stem}.validation-page-{page}")
        _run(
            runner,
            [
                str(tools["pdftoppm"]),
                "-f",
                str(page),
                "-l",
                str(page),
                "-singlefile",
                "-png",
                "-r",
                "72",
                str(pdf_path),
                str(raster_prefix),
            ],
        )
        raster = Path(f"{raster_prefix}.png")
        if not raster.is_file() or raster.stat().st_size == 0:
            raise PDFError(f"PDF page {page} rendering check failed")
        render_paths.append(str(raster))
    return {
        "pdfinfo": info,
        "pages": pages,
        "encrypted": False,
        "fonts": fonts,
        "embedded_font_count": font_count,
        "text_path": str(text_path),
        "render_paths": render_paths,
    }


def _parse_pdfinfo(output: str) -> tuple[int, bool]:
    fields: dict[str, str] = {}
    for line in output.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            fields[key.strip().lower()] = value.strip()
    page_value = fields.get("pages", "")
    if not re.fullmatch(r"[0-9]+", page_value):
        raise PDFError("PDF metadata does not contain a valid page count")
    pages = int(page_value)
    if pages < 1:
        raise PDFError("PDF contains no pages")
    encrypted_value = fields.get("encrypted", "").lower()
    if not encrypted_value:
        raise PDFError("PDF metadata does not report encryption status")
    encrypted_token = encrypted_value.split(maxsplit=1)[0]
    if encrypted_token not in {"yes", "no"}:
        raise PDFError("PDF metadata contains an invalid encryption status")
    return pages, encrypted_token == "yes"


def _validate_embedded_fonts(output: str) -> int:
    lines = output.splitlines()
    separator = next(
        (
            index
            for index, line in enumerate(lines)
            if line.count("-") >= 3 and re.fullmatch(r"[\s-]+", line)
        ),
        None,
    )
    if separator is None:
        raise PDFError("Unable to parse PDF font report")
    rows = [line for line in lines[separator + 1 :] if line.strip()]
    if not rows:
        raise PDFError("PDF font report contains no fonts")
    parsed = 0
    for row in rows:
        match = re.search(r"\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$", row, re.IGNORECASE)
        if match is None:
            raise PDFError(f"Unable to parse PDF font row: {row.strip()}")
        parsed += 1
        if match.group(1).lower() != "yes":
            font_name = row.split(maxsplit=1)[0]
            raise PDFError(f"PDF font is not embedded: {font_name}")
    return parsed


def _run(runner: Callable[..., subprocess.CompletedProcess[str]], command: list[str]) -> str:
    completed = runner(command, text=True, capture_output=True, timeout=120, check=False)
    if completed.returncode != 0:
        raise PDFError(f"command failed: {' '.join(command)}\n{completed.stderr}")
    return completed.stdout
