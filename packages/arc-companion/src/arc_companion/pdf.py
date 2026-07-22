from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Callable, Mapping
import uuid

from .io import sha256_file


class PDFError(RuntimeError):
    """Raised when LaTeX compilation or PDF inspection fails."""


_LEGACY_RUN_PDF_PATH_KEY = "output_project_pdf"
_LEGACY_RUN_PDF_SHA256_KEY = "output_project_pdf_sha256"
_LEGACY_RUN_PDF_MANAGED_KEY = "project_pdf_managed_path"


def publish_run_root_pdf(
    pdf_path: Path,
    run_root: Path,
    *,
    managed_path: Path | None = None,
) -> dict[str, str]:
    """Atomically maintain a final-PDF delivery in the resolved run root.

    The immutable render remains authoritative.  This copy is a stable
    user-facing delivery path and can be recreated without rendering or model
    work when it is missing or damaged.
    """

    source = pdf_path.resolve()
    root = run_root.resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise PDFError(f"Cannot publish a missing or empty PDF: {source}")
    root.mkdir(parents=True, exist_ok=True)
    target = root / source.name
    expected_sha256 = sha256_file(source)
    managed = managed_path.absolute() if managed_path is not None else None
    replace_existing = target.exists() or target.is_symlink()
    if replace_existing:
        if (
            not target.is_symlink()
            and target.is_file()
            and target.stat().st_size > 0
            and sha256_file(target) == expected_sha256
        ):
            return {
                "output_run_pdf": str(target),
                "output_run_pdf_sha256": expected_sha256,
            }
        if managed is None or managed != target.absolute():
            raise PDFError(
                f"Refusing to overwrite an unmanaged run-root delivery PDF: {target}"
            )
    candidate = root / (
        f".{source.name}.arc-companion-delivery-{uuid.uuid4().hex[:12]}.tmp"
    )
    try:
        shutil.copy2(source, candidate)
        if sha256_file(candidate) != expected_sha256:
            raise PDFError("Run-root delivery PDF does not match the immutable render")
        if replace_existing:
            _publish_run_root_pdf_replace(candidate, target)
        else:
            try:
                _publish_run_root_pdf_create(candidate, target)
            except FileExistsError as exc:
                raise PDFError(
                    f"Refusing to overwrite an unmanaged run-root delivery PDF: {target}"
                ) from exc
    finally:
        candidate.unlink(missing_ok=True)
    if not target.is_file() or sha256_file(target) != expected_sha256:
        raise PDFError(
            "Published run-root delivery PDF does not match the immutable render"
        )
    return {
        "output_run_pdf": str(target),
        "output_run_pdf_sha256": expected_sha256,
    }


def normalize_run_root_pdf_state(state: Mapping[str, object]) -> dict[str, object]:
    """Translate early draft field names to the run-root delivery contract."""

    normalized = dict(state)
    if not normalized.get("output_run_pdf") and normalized.get(
        _LEGACY_RUN_PDF_PATH_KEY
    ):
        normalized["output_run_pdf"] = normalized[_LEGACY_RUN_PDF_PATH_KEY]
    if not normalized.get("output_run_pdf_sha256") and normalized.get(
        _LEGACY_RUN_PDF_SHA256_KEY
    ):
        normalized["output_run_pdf_sha256"] = normalized[
            _LEGACY_RUN_PDF_SHA256_KEY
        ]
    if not normalized.get("run_pdf_managed_path") and normalized.get(
        _LEGACY_RUN_PDF_MANAGED_KEY
    ):
        normalized["run_pdf_managed_path"] = normalized[
            _LEGACY_RUN_PDF_MANAGED_KEY
        ]
    for key in (
        _LEGACY_RUN_PDF_PATH_KEY,
        _LEGACY_RUN_PDF_SHA256_KEY,
        _LEGACY_RUN_PDF_MANAGED_KEY,
    ):
        normalized.pop(key, None)

    published = normalized.get("published")
    if isinstance(published, Mapping):
        normalized_published = dict(published)
        pdf = normalized_published.get("pdf")
        if isinstance(pdf, Mapping):
            normalized_pdf = normalize_run_root_pdf_state(pdf)
            normalized_published["pdf"] = normalized_pdf
        normalized["published"] = normalized_published
    return normalized


def managed_run_root_pdf_path(state: Mapping[str, object]) -> Path | None:
    """Return only a run-root PDF path already owned by published ARC state."""

    normalized = normalize_run_root_pdf_state(state)
    value = normalized.get("run_pdf_managed_path")
    if not value:
        published = normalized.get("published")
        if isinstance(published, Mapping):
            pdf = published.get("pdf")
            if isinstance(pdf, Mapping):
                value = pdf.get("output_run_pdf")
    if not value:
        value = normalized.get("output_run_pdf")
    return Path(str(value)) if value else None


def _publish_run_root_pdf_replace(source: Path, target: Path) -> None:
    """Fault-injection seam for the atomic user-facing PDF replacement."""

    source.replace(target)


def _publish_run_root_pdf_create(source: Path, target: Path) -> None:
    """Atomically create a delivery path without replacing a racing file."""

    os.link(source, target)


def _first_latex_error_context(value: str, *, before: int = 2, after: int = 7) -> str:
    """Return bounded context around the first TeX exclamation diagnostic."""

    lines = value.splitlines()
    for index, line in enumerate(lines):
        if line.lstrip().startswith("!"):
            start = max(0, index - before)
            end = min(len(lines), index + after + 1)
            return "\n".join(lines[start:end])
    return ""


_VISIBLE_LAYER_LABELS = {
    "译文": r"译\s*文",
    "伴读": r"伴\s*读",
    "本段解释": r"本\s*段\s*解\s*释",
}
_VISIBLE_LAYER_LABEL_DECORATION = r"[#>*\-–—]*"
_VISIBLE_LAYER_LABEL_OPEN = r"[【\[(（「『《〈]?"
_VISIBLE_LAYER_LABEL_CLOSE = r"[】\])）」』》〉]?"


def compile_latex(tex_path: Path, pdf_path: Path, *, timeout_seconds: float = 300.0) -> None:
    executable = shutil.which("latexmk")
    if executable is None:
        raise PDFError("latexmk is required to build a companion PDF")
    safe_source_stem = re.sub(r"[^A-Za-z0-9_-]+", "-", tex_path.stem).strip("-") or "document"
    jobname = f"arc-companion-{safe_source_stem[:48]}-{uuid.uuid4().hex[:12]}"
    command = [
        executable,
        "-xelatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-outdir={tex_path.parent}",
        f"-jobname={jobname}",
        tex_path.name,
    ]
    built = tex_path.parent / f"{jobname}.pdf"
    try:
        completed = subprocess.run(
            command,
            cwd=tex_path.parent,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        if completed.returncode != 0 or not built.is_file() or built.stat().st_size == 0:
            command_output = completed.stdout + "\n" + completed.stderr
            log_path = tex_path.parent / f"{jobname}.log"
            try:
                log_output = log_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                log_output = ""
            first_error = (
                _first_latex_error_context(log_output)
                or _first_latex_error_context(command_output)
            )
            tail = "\n".join(command_output.splitlines()[-30:])
            diagnostic = (
                f"First XeLaTeX error:\n{first_error}\n\n"
                if first_error else ""
            )
            raise PDFError(
                f"XeLaTeX compilation failed:\n{diagnostic}Command tail:\n{tail}"
            )
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built, pdf_path)
    finally:
        for sidecar in tex_path.parent.glob(f"{jobname}.*"):
            if sidecar.is_file():
                sidecar.unlink(missing_ok=True)


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
    extracted_text = text_path.read_text(encoding="utf-8", errors="ignore") if text_path.is_file() else ""
    if not extracted_text.strip():
        raise PDFError("PDF contains no searchable text")
    forbidden = _visible_layer_labels(extracted_text)
    if forbidden:
        raise PDFError(f"PDF contains removed visible layer labels: {', '.join(forbidden)}")

    fonts = _run(runner, [str(tools["pdffonts"]), str(pdf_path)])
    font_count = _validate_embedded_fonts(fonts)
    font_roles = _validate_font_roles(fonts)

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
                "144",
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
        "font_roles": font_roles,
        "text_path": str(text_path),
        "render_paths": render_paths,
    }


def _visible_layer_labels(extracted_text: str) -> list[str]:
    found: list[str] = []
    for label, label_pattern in _VISIBLE_LAYER_LABELS.items():
        pattern = re.compile(
            rf"^\s*{_VISIBLE_LAYER_LABEL_DECORATION}\s*"
            rf"{_VISIBLE_LAYER_LABEL_OPEN}\s*{label_pattern}\s*"
            rf"{_VISIBLE_LAYER_LABEL_CLOSE}\s*[:：\-–—]?\s*$"
        )
        if any(pattern.fullmatch(line) for line in extracted_text.splitlines()):
            found.append(label)
    return found


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


def _validate_font_roles(output: str) -> dict[str, list[str]]:
    names = [
        line.split(maxsplit=1)[0].split("+", 1)[-1]
        for line in output.splitlines()
        if re.search(r"\s+(?:yes|no)\s+(?:yes|no)\s+(?:yes|no)\s+\d+\s+\d+\s*$", line, re.IGNORECASE)
    ]
    sans = [name for name in names if re.search(r"sans|hei|gothic", name, re.IGNORECASE)]
    serif = [name for name in names if not re.search(r"sans|hei|gothic", name, re.IGNORECASE)]
    if not sans:
        raise PDFError("PDF font report contains no sans-serif body font")
    if not serif:
        raise PDFError("PDF font report contains no serif mathematics font")
    return {"sans": sans, "serif": serif}


def _run(runner: Callable[..., subprocess.CompletedProcess[str]], command: list[str]) -> str:
    completed = runner(command, text=True, capture_output=True, timeout=120, check=False)
    if completed.returncode != 0:
        raise PDFError(f"command failed: {' '.join(command)}\n{completed.stderr}")
    return completed.stdout
