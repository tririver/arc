from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import time
from typing import Any, Callable
import uuid

from .content import (
    ContentBundleError,
    load_reader_content,
    migrate_legacy_reader_content,
)
from .io import (
    read_json,
    safe_name,
    sha256_file,
    sha256_json,
    write_json,
    write_text,
)
from .latex import (
    LatexError,
    render_companion_tex,
    validate_pdf_source_credit_text,
    validate_tex_fidelity,
)
from .pdf import (
    PDF_RENDER_VERSION as _PDF_RENDER_VERSION,
    PDF_VALIDATOR_VERSION,
    build_pdf_rejected_attempt,
    build_pdf_validation_receipt,
    compile_latex,
    find_adoptable_pdf_revision,
    managed_run_root_pdf_path,
    match_validated_pdf_revision,
    normalize_run_root_pdf_state,
    pdf_render_recipe_sha256,
    publish_run_root_pdf,
    validate_pdf,
)
from .results import err, ok
from .run_lock import BuildInProgressError, ProjectBuildLock
from .source_credit import source_credit_visible_projection


RENDER_MODE = "render_only"
PDF_RENDER_VERSION = _PDF_RENDER_VERSION


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
    build_lock = ProjectBuildLock(root / ".arc-companion-build.lock")
    try:
        lock.acquire()
    except BuildInProgressError as exc:
        return err("render_in_progress", str(exc), mode=RENDER_MODE, provider_calls=0)
    try:
        build_lock.acquire()
    except BuildInProgressError as exc:
        lock.release()
        return err("render_in_progress", str(exc), mode=RENDER_MODE, provider_calls=0)
    web_commit: tuple[Path, bytes | None] | None = None
    state_committed = False
    try:
        state = _state(root)
        published_digest = str(
            (state.get("published") or {}).get("content_sha256") or ""
        )
        digest = content_sha256 or published_digest
        requested_digest = digest
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
            try:
                envelope = migrate_legacy_reader_content(root, digest)
            except ContentBundleError:
                return err(
                    "content_bundle_invalid", str(exc), mode=RENDER_MODE,
                    content_sha256=digest, provider_calls=0,
                )
            digest = str(envelope["content_sha256"])
        content = envelope["content"]
        if format != "all" and requested_digest != published_digest:
            try:
                published_content = load_reader_content(
                    root, published_digest,
                )["content"]
            except ContentBundleError:
                try:
                    published_content = migrate_legacy_reader_content(
                        root, published_digest,
                    )["content"]
                except ContentBundleError:
                    published_content = None
            if not _source_credit_only_content_change(
                published_content, content,
            ):
                return err(
                    "content_digest_requires_full_render",
                    "Rendering a different reviewed-content digest requires format=all",
                    mode=RENDER_MODE,
                    content_sha256=digest,
                    published_content_sha256=published_digest or None,
                    provider_calls=0,
                )
        phase_times: dict[str, float] = {"load_content": time.monotonic() - started}
        published: dict[str, Any] = {}
        pdf_reused = False
        if format in {"pdf", "all"}:
            phase = time.monotonic()
            pdf_match = match_validated_pdf_revision(
                root,
                state,
                content_sha256=digest,
            )
            if not pdf_match.reusable:
                adopted = find_adoptable_pdf_revision(
                    root, content_sha256=digest,
                )
                if adopted.reusable:
                    pdf_match = adopted
            if pdf_match.reusable:
                pdf_reused = True
                prior_pdf = (
                    dict((state.get("published") or {}).get("pdf") or {})
                    if isinstance(state.get("published"), dict) else {}
                )
                published["pdf"] = {
                    **prior_pdf,
                    **dict(pdf_match.revision or {}),
                }
            else:
                published["pdf"] = _render_pdf(
                    root, state=state, content=content,
                    content_sha256=digest,
                    compiler=compiler, pdf_validator=pdf_validator,
                )
            phase_times["pdf"] = time.monotonic() - phase
            if format == "pdf" and pdf_reused:
                delivery_was_valid = _run_root_delivery_valid(
                    root, state,
                )
                delivery = publish_run_root_pdf(
                    Path(str(published["pdf"]["output_pdf"])),
                    root,
                    managed_path=managed_run_root_pdf_path(state),
                    expected_sha256=str(
                        published["pdf"]["output_pdf_sha256"]
                    ),
                )
                published["pdf"].update(delivery)
                current_pdf = (
                    dict((state.get("published") or {}).get("pdf") or {})
                    if isinstance(state.get("published"), dict) else {}
                )
                if all(
                    current_pdf.get(key) == value
                    and state.get(key) == value
                    for key, value in delivery.items()
                ) and delivery_was_valid:
                    phase_times["total"] = time.monotonic() - started
                    return ok({
                        "mode": RENDER_MODE,
                        "format": format,
                        "content_sha256": digest,
                        "provider_calls": 0,
                        "phase_times_seconds": phase_times,
                        "published": state.get("published"),
                        "output_pdf": published["pdf"]["output_pdf"],
                        "output_pdf_sha256": published["pdf"][
                            "output_pdf_sha256"
                        ],
                        **delivery,
                        "pdf_reuse_status": "hit",
                        "pdf_reuse_reason": pdf_match.reason,
                    })
                final_state = _publish_state(
                    root,
                    content_sha256=digest,
                    outputs={"pdf": published["pdf"]},
                    render_format=format,
                )
                phase_times["total"] = time.monotonic() - started
                return ok({
                    "mode": RENDER_MODE,
                    "format": format,
                    "content_sha256": digest,
                    "provider_calls": 0,
                    "phase_times_seconds": phase_times,
                    "published": final_state["published"],
                    "output_pdf": published["pdf"]["output_pdf"],
                    "output_pdf_sha256": published["pdf"][
                        "output_pdf_sha256"
                    ],
                    **delivery,
                    "pdf_reuse_status": "hit",
                    "pdf_reuse_reason": pdf_match.reason,
                })
        if format in {"web", "all"}:
            phase = time.monotonic()
            from .web import publish_reader

            index_path = root / "reader" / "index.html"
            previous_index = index_path.read_bytes() if index_path.is_file() else None
            overrides = {"status": "complete", **content}
            web = publish_reader(
                root,
                state={
                    "schema_version": "arc.companion.state.v3",
                    "status": "complete",
                    "paper_id": state.get("paper_id"),
                    "translation_mode": content["translation_mode"],
                    "annotation_language": content["language"],
                    "source_language": content.get("source_language") or "und",
                    "updated_at": state.get("updated_at"),
                },
                final_overrides=overrides,
            )
            web["content_sha256"] = digest
            web_commit = (index_path, previous_index)
            published["web"] = web
            phase_times["web"] = time.monotonic() - phase
        if format == "all":
            for key in (
                "source_credit_sha256",
                "source_credit_observation_sha256",
            ):
                if published["pdf"].get(key) != published["web"].get(key):
                    raise LatexError(
                        "PDF and Web source-credit projections use different "
                        f"{key}"
                    )
        # Commit the immutable render before touching the mutable delivery copy.
        # A later copy/state failure therefore cannot invalidate the last-good
        # canonical revision recorded by state.json.
        final_state = _publish_state(
            root, content_sha256=digest, outputs=published,
            render_format=format,
        )
        state_committed = True
        if "pdf" in published:
            published["pdf"].update(
                publish_run_root_pdf(
                    Path(str(published["pdf"]["output_pdf"])),
                    root,
                    managed_path=managed_run_root_pdf_path(state),
                    expected_sha256=str(
                        published["pdf"]["output_pdf_sha256"]
                    ),
                )
            )
            final_state = _publish_state(
                root,
                content_sha256=digest,
                outputs={"pdf": published["pdf"]},
                render_format=format,
            )
        phase_times["total"] = time.monotonic() - started
        data = {
            "mode": RENDER_MODE,
            "format": format,
            "content_sha256": digest,
            "provider_calls": 0,
            "phase_times_seconds": phase_times,
            "published": final_state["published"],
        }
        if "pdf" in published:
            data["pdf_reuse_status"] = "hit" if pdf_reused else "miss"
            data["pdf_reuse_reason"] = pdf_match.reason
        pdf = published.get("pdf") or {}
        web = published.get("web") or {}
        data.update({key: value for key, value in pdf.items() if key.startswith("output_")})
        data.update({key: value for key, value in web.items() if key.startswith("output_")})
        return ok(data)
    except BaseException as exc:
        rollback_error: Exception | None = None
        if web_commit is not None and not state_committed:
            try:
                _restore_web_index(*web_commit)
            except Exception as restore_exc:  # pragma: no cover - filesystem failure
                rollback_error = restore_exc
        if not isinstance(exc, Exception):
            raise
        # The commit is the state write after every requested renderer succeeds.
        # Candidate files are atomic and state still points at the prior revision.
        message = str(exc)
        if rollback_error is not None:
            message += f"; web rollback failed: {rollback_error}"
        return err(
            "render_failed", message, mode=RENDER_MODE,
            content_sha256=content_sha256, provider_calls=0,
            elapsed_seconds=time.monotonic() - started,
        )
    finally:
        build_lock.release()
        lock.release()


def render_pdf_content_unlocked(
    root: Path,
    *,
    state: dict[str, Any],
    content: dict[str, Any],
    content_sha256: str,
    compiler: Callable[[Path, Path], None],
    pdf_validator: Callable[[Path], dict[str, object]],
) -> dict[str, Any]:
    """Render one verified content object without acquiring project locks."""
    tex, source_manifest = render_companion_tex(
        content["document"], content["segments"], content["annotations"],
        output_dir=root, language=content["language"], metadata=content["metadata"],
        translations=content["translations"], glossary=content["glossary"],
        evidence_by_segment=content["reader_evidence_by_segment"],
        augmentation_scope="substantive", chapters=content["chapters"],
        chapter_guides=content["chapter_guides"],
        source_language=content.get("source_language") or "und",
        title_translations=content.get("title_translations"),
        source_credit=content["source_credit"],
        translation_reference=content.get("translation_reference"),
        project_root=root,
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
        write_json(candidate_manifest, source_manifest)
        report = pdf_validator(candidate_pdf)
        source_credit_pdf = (
            validate_pdf_source_credit_text(
                candidate_pdf, content["document"], source_manifest,
            )
            if pdf_validator is validate_pdf
            else {
                "schema_version": (
                    "arc.companion.source-credit-pdf-observation.v1"
                ),
                "canonical_sha256": content["source_credit_sha256"],
                "visible_projection_sha256": sha256_json(
                    source_credit_visible_projection(
                        content["source_credit"],
                        front_matter_block_ids=[
                            str(value)
                            for key, values in (
                                (content["document"].get("front_matter") or {}).get(
                                    "block_ids"
                                )
                                or {}
                            ).items()
                            if key in {"title", "authors", "affiliations"}
                            for value in values
                        ],
                    )
                ),
                "validation": "delegated-test-validator",
            }
        )
        expected_hashes = {
            "output_tex_sha256": sha256_file(candidate_tex),
            "output_pdf_sha256": sha256_file(candidate_pdf),
            "source_manifest_sha256": sha256_file(candidate_manifest),
        }
        receipt = build_pdf_validation_receipt(
            content_sha256=content_sha256,
            pdf_sha256=expected_hashes["output_pdf_sha256"],
            tex_sha256=expected_hashes["output_tex_sha256"],
            source_manifest_sha256=expected_hashes[
                "source_manifest_sha256"
            ],
            pdf_report=report,
            source_credit_pdf=source_credit_pdf,
            warnings=list(
                source_manifest.get("render_warnings") or []
            ),
            validator_version=(
                PDF_VALIDATOR_VERSION
                if pdf_validator is validate_pdf
                else "custom-validator"
            ),
            reusable=pdf_validator is validate_pdf,
        )
        write_json(candidate_validation, receipt)
        expected_hashes["validation_sha256"] = sha256_file(
            candidate_validation
        )
        _publish_replace(candidate_tex, tex_path)
        _publish_replace(candidate_pdf, pdf_path)
        _publish_replace(candidate_manifest, manifest_path)
        _publish_replace(candidate_validation, validation_path)
        published_hashes = {
            "output_tex_sha256": sha256_file(tex_path),
            "output_pdf_sha256": sha256_file(pdf_path),
            "source_manifest_sha256": sha256_file(manifest_path),
            "validation_sha256": sha256_file(validation_path),
        }
        if published_hashes != expected_hashes:
            raise LatexError(
                "immutable PDF revision changed during publication"
            )
    except BaseException as exc:
        try:
            attempt = build_pdf_rejected_attempt(
                exc,
                content_sha256=content_sha256,
                pdf_sha256=(
                    sha256_file(candidate_pdf)
                    if candidate_pdf.is_file() else None
                ),
                tex_sha256=(
                    sha256_file(candidate_tex)
                    if candidate_tex.is_file() else None
                ),
                source_manifest_sha256=(
                    sha256_file(candidate_manifest)
                    if candidate_manifest.is_file() else None
                ),
            )
            try:
                _write_pdf_attempt(root, attempt)
            except OSError:
                pass
        finally:
            for path in (
                *candidates,
                tex_path,
                pdf_path,
                manifest_path,
                validation_path,
            ):
                path.unlink(missing_ok=True)
            try:
                render_dir.rmdir()
            except OSError:
                pass
        raise
    return {
        "content_sha256": content_sha256,
        "render_version": PDF_RENDER_VERSION,
        "render_recipe_sha256": pdf_render_recipe_sha256(),
        "validator_version": (
            PDF_VALIDATOR_VERSION
            if pdf_validator is validate_pdf
            else "custom-validator"
        ),
        "output_tex": str(tex_path),
        "output_tex_sha256": expected_hashes["output_tex_sha256"],
        "output_pdf": str(pdf_path),
        "output_pdf_sha256": expected_hashes["output_pdf_sha256"],
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": expected_hashes[
            "source_manifest_sha256"
        ],
        "validation_path": str(validation_path),
        "validation_sha256": expected_hashes["validation_sha256"],
        "source_credit_sha256": content["source_credit_sha256"],
        "source_credit_observation_sha256": source_credit_pdf[
            "visible_projection_sha256"
        ],
    }


_render_pdf = render_pdf_content_unlocked


def _run_root_delivery_valid(
    root: Path,
    state: dict[str, Any],
) -> bool:
    normalized = normalize_run_root_pdf_state(state)
    published = normalized.get("published")
    pdf = (
        published.get("pdf")
        if isinstance(published, dict) else None
    )
    effective = {
        **normalized,
        **(dict(pdf) if isinstance(pdf, dict) else {}),
    }
    path_value = effective.get("output_run_pdf")
    expected = str(effective.get("output_run_pdf_sha256") or "")
    canonical = str(effective.get("output_pdf_sha256") or "")
    if not path_value or not expected or expected != canonical:
        return False
    path = Path(str(path_value))
    return (
        path.parent == root
        and not path.is_symlink()
        and path.is_file()
        and path.stat().st_size > 0
        and sha256_file(path) == expected
    )


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


def _allowlisted_source_credit_pdf(
    value: Any,
) -> dict[str, object]:
    source = dict(value) if isinstance(value, dict) else {}
    return {
        key: source.get(key)
        for key in (
            "schema_version",
            "canonical_sha256",
            "visible_projection_sha256",
            "validation",
        )
    }


def _write_pdf_attempt(
    root: Path,
    attempt: dict[str, object],
) -> Path:
    directory = (
        root / ".arc-companion" / "pdf-validation-attempts"
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{sha256_json(attempt)}.json"
    if not path.exists():
        write_json(path, attempt)
    return path


def _source_credit_only_content_change(old: Any, new: Any) -> bool:
    if not isinstance(old, dict) or not isinstance(new, dict):
        return False
    old_value = deepcopy(old)
    new_value = deepcopy(new)
    old_hash = str(old_value.pop("source_credit_sha256", "") or "")
    new_hash = str(new_value.pop("source_credit_sha256", "") or "")
    old_value.pop("source_credit", None)
    new_value.pop("source_credit", None)
    old_value["document"] = _neutral_credit_document(old_value.get("document"))
    new_value["document"] = _neutral_credit_document(new_value.get("document"))
    old_value["metadata"] = _neutral_credit_metadata(old_value.get("metadata"))
    new_value["metadata"] = _neutral_credit_metadata(new_value.get("metadata"))
    return bool(old_hash and new_hash and old_hash != new_hash and old_value == new_value)


def _neutral_credit_document(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    document = deepcopy(value)
    document.pop("parser_version", None)
    front = document.get("front_matter")
    if isinstance(front, dict):
        for key in (
            "author_records", "affiliation_records", "profiles",
            "author_profiles", "author_affiliations", "associations",
            "author_name_variants",
        ):
            front.pop(key, None)
        block_ids = front.get("block_ids")
        if isinstance(block_ids, dict):
            block_ids.pop("profiles", None)
    return document


def _neutral_credit_metadata(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    metadata = deepcopy(value)
    for key in ("authors", "author", "affiliations", "profiles"):
        metadata.pop(key, None)
    return metadata


def _publish_state(
    root: Path,
    *,
    content_sha256: str,
    outputs: dict[str, Any],
    render_format: str,
) -> dict[str, Any]:
    # Rendering is serialized with generation, but re-read at the commit point
    # so a state update made after the initial content lookup is never lost.
    state = normalize_run_root_pdf_state(_state(root))
    managed_run_pdf = managed_run_root_pdf_path(state)
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
    merged = {
        **state,
        "schema_version": "arc.companion.state.v3",
        "published": published,
        "revisions": revisions,
    }
    for value in outputs.values():
        merged.update(value)
    if "pdf" in outputs:
        if outputs["pdf"].get("output_run_pdf"):
            merged["run_pdf_managed_path"] = outputs["pdf"][
                "output_run_pdf"
            ]
        else:
            merged.pop("output_run_pdf", None)
            merged.pop("output_run_pdf_sha256", None)
            if managed_run_pdf is not None:
                merged["run_pdf_managed_path"] = str(managed_run_pdf)
    if _repairs_current_render_failure(
        state, content_sha256=content_sha256, render_format=render_format,
    ):
        merged["status"] = "complete"
        merged.pop("error", None)
        active_run = merged.get("active_run")
        if (
            isinstance(active_run, dict)
            and active_run.get("status") == "failed"
            and active_run.get("content_sha256") == content_sha256
            and _is_render_error(active_run.get("error"))
        ):
            active_run = {**active_run, "status": "complete"}
            active_run.pop("error", None)
            merged["active_run"] = active_run
    merged["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(root / "state.json", merged)
    return merged


def _repairs_current_render_failure(
    state: dict[str, Any], *, content_sha256: str, render_format: str,
) -> bool:
    """Return true only when an all-format render repairs this exact run."""

    if (
        render_format != "all"
        or state.get("status") != "failed"
        or state.get("content_sha256") != content_sha256
    ):
        return False
    return _is_render_error(state.get("error"))


def _is_render_error(error: Any) -> bool:
    if isinstance(error, dict):
        code = str(error.get("code") or "").casefold()
        return code in {
            "render_failed", "pdf_failed", "latex_failed", "typeset_failed",
            "companion_pdf_failed",
        }
    message = str(error or "")
    return message.startswith((
        "XeLaTeX compilation failed:",
        "source fidelity validation failed:",
        "PDF validation failed:",
        "PDF inspection failed:",
    ))


def _restore_web_index(path: Path, previous: bytes | None) -> None:
    """Restore the sole mutable web entry point after a later commit failure."""

    if previous is None:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.rollback")
    try:
        temporary.write_bytes(previous)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
