from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

from . import md2pdf


DEFAULT_TARGET_LANGUAGE = "Chinese"
DEFAULT_TARGET_LOCALE = "zh_CN"
DEFAULT_MODEL_TIER = "low"
DEFAULT_CHUNK_CHARS = 8000

JsonRunner = Callable[..., dict[str, Any]]
PdfConverter = Callable[..., dict[str, Any]]
Translator = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class MarkdownBlock:
    id: str
    text: str
    translatable: bool


def default_translated_markdown_path(input_path: str | Path, target_locale: str = DEFAULT_TARGET_LOCALE) -> Path:
    source = Path(input_path)
    return source.with_name(f"{source.stem}.{target_locale}.md")


def default_translated_pdf_path(input_path: str | Path, target_locale: str = DEFAULT_TARGET_LOCALE) -> Path:
    source = Path(input_path)
    return source.with_name(f"{source.stem}.{target_locale}.pdf")


def translate_markdown(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    target_language: str = DEFAULT_TARGET_LANGUAGE,
    target_locale: str = DEFAULT_TARGET_LOCALE,
    provider: str = "auto",
    model: str | None = None,
    model_tier: str = DEFAULT_MODEL_TIER,
    quality: bool = False,
    convert_pdf: bool = True,
    overwrite: bool = False,
    json_runner: JsonRunner | None = None,
    pdf_converter: PdfConverter | None = None,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
) -> dict[str, Any]:
    source = Path(input_path)
    if not source.exists():
        return _error("input_not_found", f"input Markdown not found: {source}")
    if not source.is_file():
        return _error("input_not_file", f"input path is not a file: {source}")

    output = Path(output_path) if output_path is not None else default_translated_markdown_path(source, target_locale)
    if output.exists() and not overwrite:
        return _error("output_exists", f"translated Markdown already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    markdown = source.read_text(encoding="utf-8")
    blocks = split_markdown_blocks(markdown)
    runner = json_runner or _run_json
    glossary = _build_glossary(
        blocks,
        target_language=target_language,
        provider=provider,
        model=model,
        model_tier=model_tier,
        runner=runner,
    )
    translations = _translate_blocks(
        blocks,
        glossary=glossary,
        target_language=target_language,
        provider=provider,
        model=model,
        model_tier=model_tier,
        runner=runner,
        chunk_chars=chunk_chars,
    )
    translated = render_translated_markdown(blocks, translations)
    quality_issues: list[Any] = []
    if quality:
        quality_result = runner(
            _quality_prompt(
                source_markdown=markdown,
                draft_markdown=translated,
                glossary=glossary,
                target_language=target_language,
            ),
            schema=_quality_schema(),
            provider=provider,
            model=model,
            model_tier=model_tier,
        )
        translated = str(quality_result.get("revised_markdown", translated))
        quality_issues = list(quality_result.get("issues") or [])

    output.write_text(translated, encoding="utf-8")

    pdf_result: dict[str, Any] | None = None
    pdf_output = output.with_suffix(".pdf")
    if convert_pdf:
        converter = pdf_converter or md2pdf.convert_markdown_to_pdf
        pdf_result = converter(input_path=output, output_path=pdf_output)
        if not pdf_result.get("ok"):
            return {
                "ok": False,
                "error": {
                    "code": "pdf_conversion_failed",
                    "message": "Markdown was translated, but PDF conversion failed.",
                },
                "data": _translation_data(source, output, pdf_output, glossary, quality, quality_issues, pdf_result),
                "errors": [],
                "meta": {},
            }

    return {
        "ok": True,
        "data": _translation_data(source, output, pdf_output if convert_pdf else None, glossary, quality, quality_issues, pdf_result),
        "errors": [],
        "meta": {
            "target_language": target_language,
            "target_locale": target_locale,
            "provider": provider,
            "model": model,
            "model_tier": model_tier,
            "chunk_chars": chunk_chars,
        },
    }


def discover_batch_translation_candidates(
    project_dir: str | Path,
    *,
    target_locale: str = DEFAULT_TARGET_LOCALE,
    overwrite: bool = False,
) -> list[dict[str, str]]:
    root = Path(project_dir)
    if not root.exists() or not root.is_dir():
        return []
    candidates: list[dict[str, str]] = []
    markdown_files = sorted(root.rglob("*.md"), key=lambda path: (len(path.relative_to(root).parts), str(path)))
    for markdown_path in markdown_files:
        if markdown_path.stem.endswith(f".{target_locale}"):
            continue
        pdf_path = markdown_path.with_suffix(".pdf")
        if not pdf_path.is_file() or pdf_path.stem.endswith(f".{target_locale}"):
            continue
        output_markdown = default_translated_markdown_path(markdown_path, target_locale)
        output_pdf = default_translated_pdf_path(markdown_path, target_locale)
        if not overwrite and (output_markdown.exists() or output_pdf.exists()):
            continue
        candidates.append(
            {
                "input_markdown_path": str(markdown_path),
                "input_pdf_path": str(pdf_path),
                "output_markdown_path": str(output_markdown),
                "output_pdf_path": str(output_pdf),
            }
        )
    return candidates


def batch_translate_project(
    project_dir: str | Path,
    *,
    target_language: str = DEFAULT_TARGET_LANGUAGE,
    target_locale: str = DEFAULT_TARGET_LOCALE,
    provider: str = "auto",
    model: str | None = None,
    model_tier: str = DEFAULT_MODEL_TIER,
    quality: bool = False,
    overwrite: bool = False,
    json_runner: JsonRunner | None = None,
    pdf_converter: PdfConverter | None = None,
    translator: Translator | None = None,
) -> dict[str, Any]:
    root = Path(project_dir)
    candidates = discover_batch_translation_candidates(root, target_locale=target_locale, overwrite=overwrite)
    translate_one = translator or translate_markdown
    results = []
    failures = []
    for candidate in candidates:
        result = translate_one(
            input_path=Path(candidate["input_markdown_path"]),
            output_path=Path(candidate["output_markdown_path"]),
            target_language=target_language,
            target_locale=target_locale,
            provider=provider,
            model=model,
            model_tier=model_tier,
            quality=quality,
            convert_pdf=True,
            overwrite=overwrite,
            json_runner=json_runner,
            pdf_converter=pdf_converter,
        )
        results.append(result)
        if not result.get("ok"):
            failures.append(result)
    return {
        "ok": not failures,
        "data": {
            "project_dir": str(root),
            "target_language": target_language,
            "target_locale": target_locale,
            "candidate_count": len(candidates),
            "translated_count": sum(1 for result in results if result.get("ok")),
            "failed_count": len(failures),
            "candidates": candidates,
            "results": results,
        },
        "errors": [],
        "meta": {"provider": provider, "model": model, "model_tier": model_tier, "quality": quality},
    }


def split_markdown_blocks(markdown: str) -> list[MarkdownBlock]:
    lines = markdown.splitlines(keepends=True)
    blocks: list[MarkdownBlock] = []
    current: list[str] = []
    counter = 1

    def add_block(text: str, *, translatable: bool) -> None:
        nonlocal counter
        if not text:
            return
        block_id = f"b{counter:04d}" if translatable else f"p{counter:04d}"
        counter += 1
        blocks.append(MarkdownBlock(id=block_id, text=text, translatable=translatable))

    def flush_current() -> None:
        nonlocal current
        if current:
            add_block("".join(current), translatable=True)
            current = []

    index = 0
    if lines and lines[0].strip() == "---":
        front_matter = [lines[0]]
        index = 1
        while index < len(lines):
            front_matter.append(lines[index])
            if lines[index].strip() == "---":
                index += 1
                break
            index += 1
        add_block("".join(front_matter), translatable=False)

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if _is_fence_start(stripped):
            flush_current()
            fence = stripped[:3]
            protected = [line]
            index += 1
            while index < len(lines):
                protected.append(lines[index])
                if lines[index].strip().startswith(fence):
                    index += 1
                    break
                index += 1
            add_block("".join(protected), translatable=False)
            continue
        if stripped in {"$$", "\\["}:
            flush_current()
            close = "$$" if stripped == "$$" else "\\]"
            protected = [line]
            index += 1
            while index < len(lines):
                protected.append(lines[index])
                if lines[index].strip() == close:
                    index += 1
                    break
                index += 1
            add_block("".join(protected), translatable=False)
            continue
        if not stripped:
            flush_current()
            add_block(line, translatable=False)
            index += 1
            continue
        current.append(line)
        index += 1
    flush_current()
    return blocks


def render_translated_markdown(blocks: list[MarkdownBlock], translations: dict[str, str]) -> str:
    rendered: list[str] = []
    for block in blocks:
        rendered.append(translations.get(block.id, block.text) if block.translatable else block.text)
    return "".join(rendered)


def _build_glossary(
    blocks: list[MarkdownBlock],
    *,
    target_language: str,
    provider: str,
    model: str | None,
    model_tier: str,
    runner: JsonRunner,
) -> list[dict[str, Any]]:
    source_text = "\n".join(block.text for block in blocks if block.translatable)
    result = runner(
        _glossary_prompt(source_text, target_language=target_language),
        schema=_glossary_schema(),
        provider=provider,
        model=model,
        model_tier=model_tier,
    )
    glossary = result.get("glossary") if isinstance(result, dict) else []
    return [item for item in glossary or [] if isinstance(item, dict)]


def _translate_blocks(
    blocks: list[MarkdownBlock],
    *,
    glossary: list[dict[str, Any]],
    target_language: str,
    provider: str,
    model: str | None,
    model_tier: str,
    runner: JsonRunner,
    chunk_chars: int,
) -> dict[str, str]:
    translations: dict[str, str] = {}
    current: list[MarkdownBlock] = []
    current_chars = 0
    for block in [item for item in blocks if item.translatable]:
        if current and current_chars + len(block.text) > chunk_chars:
            translations.update(
                _translate_chunk(
                    current,
                    glossary=glossary,
                    target_language=target_language,
                    provider=provider,
                    model=model,
                    model_tier=model_tier,
                    runner=runner,
                )
            )
            current = []
            current_chars = 0
        current.append(block)
        current_chars += len(block.text)
    if current:
        translations.update(
            _translate_chunk(
                current,
                glossary=glossary,
                target_language=target_language,
                provider=provider,
                model=model,
                model_tier=model_tier,
                runner=runner,
            )
        )
    return translations


def _translate_chunk(
    blocks: list[MarkdownBlock],
    *,
    glossary: list[dict[str, Any]],
    target_language: str,
    provider: str,
    model: str | None,
    model_tier: str,
    runner: JsonRunner,
) -> dict[str, str]:
    payload = {"blocks": [{"id": block.id, "text": block.text} for block in blocks]}
    result = runner(
        _translation_prompt(payload, glossary=glossary, target_language=target_language),
        schema=_translation_schema(),
        provider=provider,
        model=model,
        model_tier=model_tier,
    )
    output: dict[str, str] = {}
    for item in result.get("translations") or []:
        if isinstance(item, dict) and "id" in item and "text" in item:
            output[str(item["id"])] = str(item["text"])
    return output


def _run_json(prompt: str, *, schema: dict[str, Any] | None, provider: str, model: str | None, model_tier: str) -> dict[str, Any]:
    from arc_llm.runner import run_json

    return run_json(prompt, schema=schema, provider=provider, model=model, model_tier=model_tier)


def _glossary_prompt(source_text: str, *, target_language: str) -> str:
    return (
        f"Create a concise technical glossary for translating this Markdown research report into {target_language}. "
        "Preserve symbols, citation keys, equation labels, URLs, code identifiers, paper identifiers, and file paths. "
        "Return only JSON matching the schema.\n\n"
        "SOURCE_MARKDOWN_EXCERPT:\n"
        f"{source_text[:20000]}"
    )


def _translation_prompt(payload: dict[str, Any], *, glossary: list[dict[str, Any]], target_language: str) -> str:
    return (
        f"Translate these Markdown blocks into {target_language}. Preserve Markdown structure, LaTeX math, "
        "citation keys, URLs, code spans, equation labels, and file paths exactly. Do not summarize. "
        "Use the glossary consistently. Return JSON with one translation per input id.\n\n"
        f"GLOSSARY_JSON:\n{json.dumps({'glossary': glossary}, ensure_ascii=False)}\n\n"
        "BLOCKS_JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )


def _quality_prompt(
    *,
    source_markdown: str,
    draft_markdown: str,
    glossary: list[dict[str, Any]],
    target_language: str,
) -> str:
    return (
        f"Review and revise this {target_language} Markdown translation for scholarly accuracy. "
        "Preserve all Markdown structure, LaTeX math, citations, URLs, code, equation labels, and file paths. "
        "Return the full revised Markdown and a short issue list as JSON.\n\n"
        f"GLOSSARY_JSON:\n{json.dumps({'glossary': glossary}, ensure_ascii=False)}\n\n"
        f"SOURCE_MARKDOWN:\n{source_markdown[:30000]}\n\n"
        f"DRAFT_MARKDOWN:\n{draft_markdown[:30000]}"
    )


def _glossary_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "glossary": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "target": {"type": "string"},
                        "note": {"type": "string"},
                    },
                    "required": ["source", "target"],
                    "additionalProperties": True,
                },
            }
        },
        "required": ["glossary"],
        "additionalProperties": False,
    }


def _translation_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "translations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}, "text": {"type": "string"}},
                    "required": ["id", "text"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["translations"],
        "additionalProperties": False,
    }


def _quality_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "revised_markdown": {"type": "string"},
            "issues": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["revised_markdown", "issues"],
        "additionalProperties": False,
    }


def _translation_data(
    source: Path,
    output: Path,
    pdf_output: Path | None,
    glossary: list[dict[str, Any]],
    quality: bool,
    quality_issues: list[Any],
    pdf_result: dict[str, Any] | None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "input_markdown_path": str(source),
        "output_markdown_path": str(output),
        "target_markdown_size_bytes": output.stat().st_size if output.exists() else 0,
        "quality_pass": quality,
        "quality_issues": quality_issues,
        "glossary": glossary,
    }
    if pdf_output is not None:
        data["output_pdf_path"] = str(pdf_output)
        data["pdf_result"] = pdf_result
    return data


def _is_fence_start(stripped: str) -> bool:
    return stripped.startswith("```") or stripped.startswith("~~~")


def _error(code: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message}, "errors": [], "meta": {}}
