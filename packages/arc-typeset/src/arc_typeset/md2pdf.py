from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
from typing import Any


DEFAULT_TEXLIVE_BIN = Path("/usr/local/texlive/2026/bin/x86_64-linux")
DEFAULT_MARGIN = "1.5cm"
DEFAULT_MAINFONT = "Noto Sans CJK SC"
DEFAULT_CJK_MAINFONT = "Noto Sans CJK SC"


def default_output_path(input_path: Path) -> Path:
    return input_path.with_suffix(".pdf")


def default_resource_paths(input_path: Path) -> list[Path]:
    paths = [input_path.parent.resolve(), Path.cwd().resolve()]
    deduped: list[Path] = []
    for path in paths:
        if path not in deduped:
            deduped.append(path)
    return deduped


def build_env(texlive_bin: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    if texlive_bin and texlive_bin.exists():
        env["PATH"] = f"{texlive_bin}{os.pathsep}{env.get('PATH', '')}"
    return env


def pandoc_command(
    input_path: Path,
    output_path: Path,
    *,
    margin: str = DEFAULT_MARGIN,
    mainfont: str = DEFAULT_MAINFONT,
    cjk_mainfont: str = DEFAULT_CJK_MAINFONT,
    resource_paths: list[Path] | None = None,
) -> list[str]:
    resources = resource_paths if resource_paths is not None else default_resource_paths(input_path)
    return [
        "pandoc",
        str(input_path),
        "-o",
        str(output_path),
        "--pdf-engine=xelatex",
        f"--resource-path={os.pathsep.join(str(path) for path in resources)}",
        "-V",
        f"geometry:margin={margin}",
        "-V",
        f"mainfont={mainfont}",
        "-V",
        f"CJKmainfont={cjk_mainfont}",
    ]


def convert_markdown_to_pdf(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    texlive_bin: str | Path | None = DEFAULT_TEXLIVE_BIN,
    margin: str = DEFAULT_MARGIN,
    mainfont: str = DEFAULT_MAINFONT,
    cjk_mainfont: str = DEFAULT_CJK_MAINFONT,
    resource_paths: list[str | Path] | None = None,
) -> dict[str, Any]:
    source = Path(input_path)
    if not source.exists():
        return _error("input_not_found", f"input Markdown not found: {source}")
    if not source.is_file():
        return _error("input_not_file", f"input path is not a file: {source}")

    output = Path(output_path) if output_path is not None else default_output_path(source)
    output.parent.mkdir(parents=True, exist_ok=True)

    texlive_path = Path(texlive_bin) if texlive_bin else None
    env = build_env(texlive_path)
    pandoc_path = shutil.which("pandoc", path=env.get("PATH"))
    if not pandoc_path:
        return _error("missing_dependency", "pandoc not found on PATH")
    xelatex_path = shutil.which("xelatex", path=env.get("PATH"))
    if not xelatex_path:
        return _error("missing_dependency", "xelatex not found on PATH")

    resources = [Path(path) for path in resource_paths] if resource_paths is not None else default_resource_paths(source)
    command = pandoc_command(
        source,
        output,
        margin=margin,
        mainfont=mainfont,
        cjk_mainfont=cjk_mainfont,
        resource_paths=resources,
    )
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        return {
            "ok": False,
            "error": {
                "code": "conversion_failed",
                "message": "PDF conversion failed",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
            "errors": [],
            "meta": {"command": command},
        }
    if not output.exists():
        return _error("output_not_created", f"pandoc completed but output PDF was not created: {output}")

    return {
        "ok": True,
        "data": {
            "input_path": str(source),
            "output_path": str(output),
            "pdf_size_bytes": output.stat().st_size,
            "engine": "xelatex",
        },
        "errors": [],
        "meta": {
            "command": command,
            "dependencies": {
                "pandoc": pandoc_path,
                "xelatex": xelatex_path,
            },
            "resource_path": [str(path) for path in resources],
        },
    }


def _error(code: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message}, "errors": [], "meta": {}}
