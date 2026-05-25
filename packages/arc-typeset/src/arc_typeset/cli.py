from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from . import md2pdf


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ARC typesetting utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    md2pdf_parser = sub.add_parser("md2pdf", description="Convert Markdown to PDF with Pandoc and XeLaTeX")
    md2pdf_parser.add_argument("input", help="Markdown file to convert")
    md2pdf_parser.add_argument("--output", help="Output PDF path")
    md2pdf_parser.add_argument(
        "--texlive-bin",
        default=str(md2pdf.DEFAULT_TEXLIVE_BIN),
        help='Optional TeX Live bin directory to prepend to PATH. Pass "" to disable.',
    )
    md2pdf_parser.add_argument("--margin", default=md2pdf.DEFAULT_MARGIN)
    md2pdf_parser.add_argument("--mainfont", default=md2pdf.DEFAULT_MAINFONT)
    md2pdf_parser.add_argument("--cjk-mainfont", default=md2pdf.DEFAULT_CJK_MAINFONT)
    md2pdf_parser.add_argument(
        "--resource-path",
        action="append",
        default=None,
        help="Pandoc resource path entry. May be passed multiple times.",
    )
    md2pdf_parser.add_argument("--json", action="store_true", help="Print structured JSON output")

    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "md2pdf":
        result = md2pdf.convert_markdown_to_pdf(
            input_path=Path(args.input),
            output_path=Path(args.output) if args.output else None,
            texlive_bin=Path(args.texlive_bin) if args.texlive_bin else None,
            margin=args.margin,
            mainfont=args.mainfont,
            cjk_mainfont=args.cjk_mainfont,
            resource_paths=[Path(path) for path in args.resource_path] if args.resource_path else None,
        )
        _emit(result, json_output=args.json)
        return 0 if result.get("ok") else 1
    raise AssertionError(f"Unhandled command: {args.command}")


def _emit(result: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return
    if result.get("ok"):
        print(result["data"]["output_path"])
        return
    error = result.get("error") or {}
    print(f"ERROR: {error.get('message', 'conversion failed')}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
