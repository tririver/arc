from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import time
from typing import Any, Callable
import uuid

from .content import ContentBundleError, load_reader_content
from .io import read_json, safe_name, sha256_file, write_json, write_text
from .latex import LatexError, render_companion_tex, validate_tex_fidelity
from .pdf import compile_latex, validate_pdf
from .results import err, ok
from .run_lock import BuildInProgressError, ProjectBuildLock


RENDER_MODE = "render_only"
PDF_RENDER_VERSION = "arc.companion.final-render.v10"


def render_content(
    project_dir: Path,
    *,
    format: str = "all",
    content_sha256: str | None = None,
    compiler: Callable[[Path, Path], None] = compile_latex,
    pdf_validator: Callable[[Path], dict[str, object]] = validate_pdf,
) -> dict[str, Any]:
    """Render an immutable reviewed-content object without loading an LLM runtime."""
    if format not in {"pdf", "web", "all"}:
        raise ValueError("format must be pdf, web, or all")
    root = project_dir.resolve()
    started = time.monotonic()
    lock = ProjectBuildLock(root / ".arc-companion" / "render.lock")
    try:
        lock.acquire()
    except BuildInProgressError as exc:
        return err("render_in_progress", str(exc), mode=RENDER_MODE, provider_calls=0)
    try:
        state = _state(root)
        digest = content_sha256 or str((state.get("published") or {}).get("content_sha256") or "")
        if not digest:
            return err(
                "content_bundle_not_found",
                "No last-complete reviewed-content object is published for this project",
                mode=RENDER_MODE,
                provider_calls=0,
            )
        try:
            envelope = load_reader_content(root, digest)
        except ContentBundleError as exc:
            return err(
                "content_bundle_invalid", str(exc), mode=RENDER_MODE,
                content_sha256=digest, provider_calls=0,
            )
        content = envelope["content"]
        phase_times: dict[str, float] = {"load_content": time.monotonic() - started}
        published: dict[str, Any] = {}
        if format in {"pdf", "all"}:
            phase = time.monotonic()
            published["pdf"] = _render_pdf(
                root, state=state, content=content, content_sha256=digest,
                compiler=compiler, pdf_validator=pdf_validator,
            )
            phase_times["pdf"] = time.monotonic() - phase
        if format in {"web", "all"}:
            phase = time.monotonic()
            from .web import publish_reader

            overrides = {"status": "complete", **content}
            web = publish_reader(
                root,
                state={
                    "schema_version": "arc.companion.state.v3",
                    "status": "complete",
                    "paper_id": state.get("paper_id"),
                    "translation_mode": content["translation_mode"],
                    "annotation_language": content["language"],
                    "updated_at": state.get("updated_at"),
                },
                final_overrides=overrides,
            )
            published["web"] = web
            phase_times["web"] = time.monotonic() - phase
        phase_times["total"] = time.monotonic() - started
        final_state = _publish_state(root, state, content_sha256=digest, outputs=published)
        data = {
            "mode": RENDER_MODE,
            "format": format,
            "content_sha256": digest,
            "provider_calls": 0,
            "phase_times_seconds": phase_times,
            "published": final_state["published"],
        }
        pdf = published.get("pdf") or {}
        web = published.get("web") or {}
        data.update({key: value for key, value in pdf.items() if key.startswith("output_")})
        data.update({key: value for key, value in web.items() if key.startswith("output_")})
        return ok(data)
    except Exception as exc:
        # The commit is the state write after every requested renderer succeeds.
        # Candidate files are atomic and state still points at the prior revision.
        return err(
            "render_failed", str(exc), mode=RENDER_MODE,
            content_sha256=content_sha256, provider_calls=0,
            elapsed_seconds=time.monotonic() - started,
        )
    finally:
        lock.release()


def _render_pdf(
    root: Path,
    *,
    state: dict[str, Any],
    content: dict[str, Any],
    content_sha256: str,
    compiler: Callable[[Path, Path], None],
    pdf_validator: Callable[[Path], dict[str, object]],
) -> dict[str, Any]:
    tex, source_manifest = render_companion_tex(
        content["document"], content["segments"], content["annotations"],
        output_dir=root, language=content["language"], metadata=content["metadata"],
        translations=content["translations"], glossary=content["glossary"],
        evidence_by_segment=content["reader_evidence_by_segment"],
        augmentation_scope="substantive", chapters=content["chapters"],
        chapter_guides=content["chapter_guides"],
    )
    fidelity_errors = validate_tex_fidelity(tex, content["document"], source_manifest)
    if fidelity_errors:
        raise LatexError("source fidelity validation failed: " + "; ".join(fidelity_errors))
    stem = f"{safe_name(str(state.get('paper_id') or 'paper'))}_companion_{safe_name(content['language'])}"
    # Every successful render is published at a new immutable path.  Therefore
    # no sequence of file replacements can damage the revision referenced by
    # the current state if this render fails before its single state commit.
    render_dir = (
        root / ".arc-companion" / "renders" / "pdf"
        / f"{content_sha256}-{uuid.uuid4().hex[:12]}"
    )
    render_dir.mkdir(parents=True, exist_ok=False)
    tex_path = render_dir / f"{stem}.tex"
    pdf_path = render_dir / f"{stem}.pdf"
    manifest_path = render_dir / "source-manifest.json"
    validation_path = render_dir / "validation.json"
    staging = f"arc-companion-rendering-{safe_name(stem)}-{uuid.uuid4().hex[:12]}"
    candidate_tex = render_dir / f"{staging}.tex"
    candidate_pdf = render_dir / f"{staging}.pdf"
    candidate_manifest = render_dir / f"{staging}-manifest.json"
    candidate_validation = render_dir / f"{staging}-validation.json"
    candidates = (candidate_tex, candidate_pdf, candidate_manifest, candidate_validation)
    try:
        write_text(candidate_tex, tex)
        compiler(candidate_tex, candidate_pdf)
        report = pdf_validator(candidate_pdf)
        write_json(candidate_manifest, source_manifest)
        write_json(candidate_validation, {
            "ok": True,
            "content_sha256": content_sha256,
            "render_version": PDF_RENDER_VERSION,
            "pdf": report,
            "fidelity_errors": [],
        })
        _publish_replace(candidate_tex, tex_path)
        _publish_replace(candidate_pdf, pdf_path)
        _publish_replace(candidate_manifest, manifest_path)
        _publish_replace(candidate_validation, validation_path)
    except BaseException:
        for path in (*candidates, tex_path, pdf_path, manifest_path, validation_path):
            path.unlink(missing_ok=True)
        try:
            render_dir.rmdir()
        except OSError:
            pass
        raise
    return {
        "content_sha256": content_sha256,
        "render_version": PDF_RENDER_VERSION,
        "output_tex": str(tex_path),
        "output_tex_sha256": sha256_file(tex_path),
        "output_pdf": str(pdf_path),
        "output_pdf_sha256": sha256_file(pdf_path),
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "validation_path": str(validation_path),
        "validation_sha256": sha256_file(validation_path),
    }


def _state(root: Path) -> dict[str, Any]:
    path = root / "state.json"
    try:
        value = read_json(path)
    except (OSError, ValueError) as exc:
        raise ContentBundleError(f"companion state is unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise ContentBundleError("companion state is not an object")
    return value


def _publish_replace(source: Path, target: Path) -> None:
    """Fault-injection seam for the immutable PDF publish sequence."""
    os.replace(source, target)


def _publish_state(
    root: Path,
    state: dict[str, Any],
    *,
    content_sha256: str,
    outputs: dict[str, Any],
) -> dict[str, Any]:
    published = dict(state.get("published") or {})
    published["content_sha256"] = content_sha256
    for lane, value in outputs.items():
        published[lane] = dict(value)
    revisions = list(state.get("revisions") or [])
    revision = {
        "content_sha256": content_sha256,
        "pdf_sha256": (published.get("pdf") or {}).get("output_pdf_sha256"),
        "web_sha256": (published.get("web") or {}).get("output_html_sha256"),
        "published_at": datetime.now(timezone.utc).isoformat(),
    }
    if not revisions or {k: v for k, v in revisions[-1].items() if k != "published_at"} != {k: v for k, v in revision.items() if k != "published_at"}:
        revisions.append(revision)
    merged = {**state, "schema_version": "arc.companion.state.v3", "published": published, "revisions": revisions}
    for value in outputs.values():
        merged.update(value)
    merged["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(root / "state.json", merged)
    return merged
