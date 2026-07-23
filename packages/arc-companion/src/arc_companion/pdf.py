from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
from typing import Any, Callable, Mapping
import uuid

from .artifact_ids import (
    ARTIFACT_ID_RECEIPT_NAME,
    ArtifactIdError,
    resolve_artifact_dir,
)
from .io import sha256_file


PDF_RENDER_VERSION = "arc.companion.final-render.v13"
PDF_VALIDATOR_VERSION = "arc.companion.pdf-validator.v2"
PDF_VALIDATION_RECEIPT_VERSION = (
    "arc.companion.pdf-validation-receipt.v2"
)
PDF_VALIDATION_ATTEMPT_VERSION = (
    "arc.companion.pdf-validation-attempt.v1"
)
PDF_RASTER_DPI = 144
PDF_DIAGNOSTIC_MAX_CHARS = 4096
PDF_STDERR_MAX_CHARS = 2048
PDF_FONT_ROLE_MAX_ITEMS = 128
PDF_WARNING_MAX_ITEMS = 128
PDF_WARNING_MAX_CHARS = 512
PDF_RECEIPT_MAX_BYTES = 512 * 1024
PDF_SOURCE_CREDIT_OBSERVATION_VERSION = (
    "arc.companion.source-credit-pdf-observation.v1"
)
_PDF_ATTEMPT_REASONS = {
    "pdf_error",
    "validation_failed",
    "pdf_missing",
    "tool_missing",
    "encrypted",
    "text_missing",
    "visible_layer_label",
    "raster_missing",
    "command_failed",
    "metadata_invalid",
    "fonts_invalid",
    "compilation_failed",
}
_PDF_ATTEMPT_STAGES = {
    "unknown",
    "input",
    "preflight",
    "compile",
    "metadata",
    "text",
    "fonts",
    "raster",
    "delivery",
}


class PDFError(RuntimeError):
    """Bounded typed PDF compilation or validation failure."""

    def __init__(
        self,
        message: str,
        *,
        reason: str = "pdf_error",
        stage: str = "unknown",
        page: int | None = None,
        executable: str | None = None,
        stderr: str | None = None,
    ) -> None:
        super().__init__(str(message)[:PDF_DIAGNOSTIC_MAX_CHARS])
        self.reason = reason
        self.stage = stage
        self.page = page
        self.executable = Path(executable).name if executable else None
        self.stderr = (
            str(stderr)[-PDF_STDERR_MAX_CHARS:] if stderr else None
        )

    def diagnostic(self) -> dict[str, object]:
        return {
            "reason": self.reason,
            "stage": self.stage,
            "page": self.page,
            "executable": self.executable,
            "stderr": self.stderr,
            "message": str(self),
        }


@dataclass(frozen=True)
class PDFReuseDecision:
    reusable: bool
    reason: str
    revision: Mapping[str, object] | None = None


def pdf_render_recipe_sha256() -> str:
    payload = {
        "render_version": PDF_RENDER_VERSION,
        "validator_version": PDF_VALIDATOR_VERSION,
        "raster_dpi": PDF_RASTER_DPI,
    }
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def allowlisted_pdf_summary(
    report: Mapping[str, object] | None,
) -> dict[str, object]:
    value = dict(report or {})
    roles = value.get("font_roles")
    roles = roles if isinstance(roles, Mapping) else {}
    def role_values(key: str) -> list[str]:
        raw = roles.get(key)
        if not isinstance(raw, list):
            return []
        return [
            item[:PDF_WARNING_MAX_CHARS]
            for item in raw[:PDF_FONT_ROLE_MAX_ITEMS]
            if isinstance(item, str)
        ]

    return {
        "validator": (
            value.get("validator")
            if isinstance(value.get("validator"), str) else None
        ),
        "result": (
            value.get("result")
            if isinstance(value.get("result"), str) else None
        ),
        "pages": value.get("pages") if type(value.get("pages")) is int else None,
        "pages_checked": (
            value.get("pages_checked")
            if type(value.get("pages_checked")) is int else None
        ),
        "dpi": value.get("dpi") if type(value.get("dpi")) is int else None,
        "pdf_bytes": (
            value.get("pdf_bytes")
            if type(value.get("pdf_bytes")) is int else None
        ),
        "text_bytes": (
            value.get("text_bytes")
            if type(value.get("text_bytes")) is int else None
        ),
        "raster_bytes": (
            value.get("raster_bytes")
            if type(value.get("raster_bytes")) is int else None
        ),
        "encrypted": (
            value.get("encrypted")
            if type(value.get("encrypted")) is bool else None
        ),
        "embedded_font_count": (
            value.get("embedded_font_count")
            if type(value.get("embedded_font_count")) is int else None
        ),
        "font_roles": {
            "sans": role_values("sans"),
            "serif": role_values("serif"),
        },
    }


def allowlisted_source_credit_pdf(
    value: Mapping[str, object] | None,
) -> dict[str, object]:
    source = dict(value or {})
    ordered = source.get("ordered_ids")
    counts = source.get("visible_counts")
    counts = counts if isinstance(counts, Mapping) else {}
    return {
        "schema_version": (
            source.get("schema_version")
            if isinstance(source.get("schema_version"), str) else None
        ),
        "canonical_sha256": source.get("canonical_sha256"),
        "searchable_text_sha256": source.get(
            "searchable_text_sha256"
        ),
        "ordered_ids": (
            [
                item[:PDF_WARNING_MAX_CHARS]
                for item in ordered[:PDF_FONT_ROLE_MAX_ITEMS]
                if isinstance(item, str)
            ]
            if isinstance(ordered, list) else []
        ),
        "visible_projection_sha256": source.get(
            "visible_projection_sha256"
        ),
        "visible_counts": {
            key: counts.get(key) if type(counts.get(key)) is int else None
            for key in ("authors", "affiliations", "profiles")
        },
    }


def build_pdf_validation_receipt(
    *,
    content_sha256: str,
    pdf_sha256: str,
    tex_sha256: str,
    source_manifest_sha256: str,
    pdf_report: Mapping[str, object] | None,
    source_credit_pdf: Mapping[str, object] | None = None,
    warnings: list[object] | tuple[object, ...] = (),
    preview: bool = False,
    validator_version: str = PDF_VALIDATOR_VERSION,
    reusable: bool | None = None,
) -> dict[str, object]:
    credit = dict(source_credit_pdf or {})
    return {
        "schema_version": PDF_VALIDATION_RECEIPT_VERSION,
        "result": "success",
        "scope": "preview" if preview else "final",
        "reusable": (not preview) if reusable is None else reusable,
        "render_version": PDF_RENDER_VERSION,
        "content_sha256": content_sha256,
        "render_recipe_sha256": pdf_render_recipe_sha256(),
        "validator_version": validator_version,
        "pdf_sha256": pdf_sha256,
        "tex_sha256": tex_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "pdf": allowlisted_pdf_summary(pdf_report),
        "fidelity_errors": [],
        "warnings": [
            str(value)[:PDF_WARNING_MAX_CHARS]
            for value in list(warnings)[:PDF_WARNING_MAX_ITEMS]
        ],
        "source_credit_pdf": allowlisted_source_credit_pdf(credit),
    }


def build_pdf_rejected_attempt(
    error: BaseException,
    *,
    content_sha256: str | None = None,
    pdf_sha256: str | None = None,
    tex_sha256: str | None = None,
    source_manifest_sha256: str | None = None,
    preview: bool = False,
) -> dict[str, object]:
    diagnostic = (
        error.diagnostic()
        if isinstance(error, PDFError)
        else {
            "reason": "validation_failed",
            "stage": "unknown",
            "page": None,
            "message": str(error)[:PDF_DIAGNOSTIC_MAX_CHARS],
        }
    )
    reason = str(diagnostic.get("reason") or "")
    stage = str(diagnostic.get("stage") or "")
    page = diagnostic.get("page")
    return {
        "schema_version": PDF_VALIDATION_ATTEMPT_VERSION,
        "result": "rejected",
        "scope": "preview" if preview else "final",
        "content_sha256": (
            content_sha256 if _is_sha256(content_sha256) else None
        ),
        "render_recipe_sha256": pdf_render_recipe_sha256(),
        "validator_version": PDF_VALIDATOR_VERSION,
        "pdf_sha256": pdf_sha256 if _is_sha256(pdf_sha256) else None,
        "tex_sha256": tex_sha256 if _is_sha256(tex_sha256) else None,
        "source_manifest_sha256": (
            source_manifest_sha256
            if _is_sha256(source_manifest_sha256) else None
        ),
        "reason": (
            reason if reason in _PDF_ATTEMPT_REASONS
            else "validation_failed"
        ),
        "stage": stage if stage in _PDF_ATTEMPT_STAGES else "unknown",
        "page": (
            page
            if type(page) is int and 1 <= page <= 1_000_000 else None
        ),
    }


_LEGACY_RUN_PDF_PATH_KEY = "output_project_pdf"
_LEGACY_RUN_PDF_SHA256_KEY = "output_project_pdf_sha256"
_LEGACY_RUN_PDF_MANAGED_KEY = "project_pdf_managed_path"


def publish_run_root_pdf(
    pdf_path: Path,
    run_root: Path,
    *,
    managed_path: Path | None = None,
    expected_sha256: str,
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
    actual_source_sha256 = sha256_file(source)
    if not _is_sha256(expected_sha256):
        raise PDFError(
            "Expected immutable render PDF hash is invalid",
            reason="expected_hash_invalid",
            stage="delivery",
        )
    if expected_sha256 != actual_source_sha256:
        raise PDFError(
            "Immutable render PDF hash changed before delivery",
            reason="source_hash_mismatch",
            stage="delivery",
        )
    managed = managed_path.absolute() if managed_path is not None else None
    replace_existing = target.exists() or target.is_symlink()
    if replace_existing:
        if (
            not target.is_symlink()
            and target.is_file()
            and target.stat().st_size > 0
            and sha256_file(target) == expected_sha256
        ):
            if sha256_file(source) != expected_sha256:
                raise PDFError(
                    "Immutable render PDF hash changed before delivery return",
                    reason="source_hash_mismatch",
                    stage="delivery",
                )
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
        if (
            sha256_file(source) != expected_sha256
            or sha256_file(candidate) != expected_sha256
        ):
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
    if (
        sha256_file(source) != expected_sha256
        or not target.is_file()
        or sha256_file(target) != expected_sha256
    ):
        raise PDFError(
            "Published run-root delivery PDF does not match the immutable render"
        )
    return {
        "output_run_pdf": str(target),
        "output_run_pdf_sha256": expected_sha256,
    }


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def _project_regular_file(
    project_dir: Path,
    value: object,
) -> Path | None:
    if not value:
        return None
    root = project_dir.resolve()
    raw = Path(str(value))
    if ".." in raw.parts:
        return None
    candidate = raw if raw.is_absolute() else root / raw
    try:
        relative = candidate.absolute().relative_to(root)
    except ValueError:
        return None
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return None
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError):
        return None
    if (
        not resolved.is_file()
        or resolved.is_symlink()
        or resolved.stat().st_size <= 0
    ):
        return None
    return resolved


_PDF_SUMMARY_KEYS = {
    "validator",
    "result",
    "pages",
    "pages_checked",
    "dpi",
    "pdf_bytes",
    "text_bytes",
    "raster_bytes",
    "encrypted",
    "embedded_font_count",
    "font_roles",
}
_PDF_RECEIPT_KEYS = {
    "schema_version",
    "result",
    "scope",
    "reusable",
    "render_version",
    "content_sha256",
    "render_recipe_sha256",
    "validator_version",
    "pdf_sha256",
    "tex_sha256",
    "source_manifest_sha256",
    "pdf",
    "fidelity_errors",
    "warnings",
    "source_credit_pdf",
}
_SOURCE_CREDIT_PDF_KEYS = {
    "schema_version",
    "canonical_sha256",
    "searchable_text_sha256",
    "ordered_ids",
    "visible_projection_sha256",
    "visible_counts",
}


def _read_bounded_regular_file(path: Path) -> bytes | None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return None
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_size <= 0:
            return None
        value = os.read(descriptor, PDF_RECEIPT_MAX_BYTES + 1)
    except OSError:
        return None
    finally:
        os.close(descriptor)
    return value


def _hash_regular_file_nofollow(
    path: Path,
) -> tuple[str, int] | None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return None
    digest = hashlib.sha256()
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_size <= 0:
            return None
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    except OSError:
        return None
    finally:
        os.close(descriptor)
    return digest.hexdigest(), details.st_size


def pdf_validation_receipt_is_closed(
    receipt: Mapping[str, object],
    *,
    scope: str,
    reusable: bool,
    validator_version: str,
) -> bool:
    """Validate the exact receipt and nested evidence contract for one scope."""

    if scope not in {"preview", "final"}:
        return False
    if set(receipt) != _PDF_RECEIPT_KEYS:
        return False
    if (
        receipt.get("schema_version")
        != PDF_VALIDATION_RECEIPT_VERSION
        or receipt.get("result") != "success"
        or receipt.get("scope") != scope
        or receipt.get("reusable") is not reusable
        or receipt.get("render_version") != PDF_RENDER_VERSION
    ):
        return False
    if any(
        not _is_sha256(receipt.get(key))
        for key in (
            "content_sha256",
            "render_recipe_sha256",
            "pdf_sha256",
            "tex_sha256",
            "source_manifest_sha256",
        )
    ):
        return False
    if receipt.get("validator_version") != validator_version:
        return False
    fidelity = receipt.get("fidelity_errors")
    warnings = receipt.get("warnings")
    if fidelity != [] or not isinstance(warnings, list):
        return False
    if (
        len(warnings) > PDF_WARNING_MAX_ITEMS
        or any(
            not isinstance(item, str)
            or len(item) > PDF_WARNING_MAX_CHARS
            for item in warnings
        )
    ):
        return False
    summary = receipt.get("pdf")
    if not isinstance(summary, Mapping) or set(summary) != _PDF_SUMMARY_KEYS:
        return False
    if (
        summary.get("validator") != validator_version
        or summary.get("result") != "success"
        or summary.get("dpi") != PDF_RASTER_DPI
        or summary.get("encrypted") is not False
    ):
        return False
    pages = summary.get("pages")
    pages_checked = summary.get("pages_checked")
    if (
        not isinstance(pages, int)
        or isinstance(pages, bool)
        or pages < 1
        or pages_checked != pages
    ):
        return False
    for key in (
        "pdf_bytes",
        "text_bytes",
        "raster_bytes",
        "embedded_font_count",
    ):
        value = summary.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            return False
    roles = summary.get("font_roles")
    if not isinstance(roles, Mapping) or set(roles) != {"sans", "serif"}:
        return False
    for key in ("sans", "serif"):
        values = roles.get(key)
        if (
            not isinstance(values, list)
            or not values
            or len(values) > PDF_FONT_ROLE_MAX_ITEMS
            or any(
                not isinstance(item, str)
                or len(item) > PDF_WARNING_MAX_CHARS
                for item in values
            )
        ):
            return False
    credit = receipt.get("source_credit_pdf")
    if (
        not isinstance(credit, Mapping)
        or set(credit) != _SOURCE_CREDIT_PDF_KEYS
        or credit.get("schema_version")
        != PDF_SOURCE_CREDIT_OBSERVATION_VERSION
        or not _is_sha256(credit.get("canonical_sha256"))
        or not _is_sha256(credit.get("searchable_text_sha256"))
        or not _is_sha256(credit.get("visible_projection_sha256"))
    ):
        return False
    ordered_ids = credit.get("ordered_ids")
    if (
        not isinstance(ordered_ids, list)
        or len(ordered_ids) > PDF_FONT_ROLE_MAX_ITEMS
        or any(
            not isinstance(item, str)
            or not item
            or len(item) > PDF_WARNING_MAX_CHARS
            for item in ordered_ids
        )
    ):
        return False
    counts = credit.get("visible_counts")
    if not isinstance(counts, Mapping) or set(counts) != {
        "authors", "affiliations", "profiles",
    }:
        return False
    if any(
        type(counts.get(key)) is not int or counts.get(key) < 0
        for key in ("authors", "affiliations", "profiles")
    ):
        return False
    visible_total = sum(int(counts[key]) for key in counts)
    if (
        len(ordered_ids)
        != min(visible_total, PDF_FONT_ROLE_MAX_ITEMS)
        or len(set(ordered_ids)) != len(ordered_ids)
    ):
        return False
    return True


def _current_receipt_is_closed(receipt: Mapping[str, object]) -> bool:
    return pdf_validation_receipt_is_closed(
        receipt,
        scope="final",
        reusable=True,
        validator_version=PDF_VALIDATOR_VERSION,
    )


def current_pdf_validation_receipt_matches(
    receipt: Mapping[str, object],
    *,
    content_sha256: str,
    render_recipe_sha256: str,
    validator_version: str,
    pdf_sha256: str,
    tex_sha256: str,
    source_manifest_sha256: str,
    pdf_bytes: int | None = None,
) -> bool:
    """Validate a closed current receipt against its authoritative state."""

    if not _current_receipt_is_closed(receipt):
        return False
    expected = {
        "content_sha256": content_sha256,
        "render_recipe_sha256": render_recipe_sha256,
        "validator_version": validator_version,
        "pdf_sha256": pdf_sha256,
        "tex_sha256": tex_sha256,
        "source_manifest_sha256": source_manifest_sha256,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        return False
    if (
        pdf_bytes is not None
        and dict(receipt["pdf"]).get("pdf_bytes") != pdf_bytes
    ):
        return False
    return True


def match_validated_pdf_revision(
    project_dir: Path,
    state: Mapping[str, object],
    *,
    content_sha256: str,
    render_recipe_sha256: str | None = None,
    validator_version: str = PDF_VALIDATOR_VERSION,
) -> PDFReuseDecision:
    """Purely match one successful immutable PDF revision."""

    if not _is_sha256(content_sha256):
        return PDFReuseDecision(False, "content_identity_invalid")
    normalized = normalize_run_root_pdf_state(state)
    published = normalized.get("published")
    if isinstance(published, Mapping) and "pdf" in published:
        raw_pdf = published.get("pdf")
        if not isinstance(raw_pdf, Mapping):
            return PDFReuseDecision(False, "pdf_state_invalid")
        effective = dict(raw_pdf)
        published_content = published.get("content_sha256")
    else:
        effective = {
            key: value
            for key, value in normalized.items()
            if key != "published"
        }
        published_content = effective.get("content_sha256")
    if not effective.get("output_pdf"):
        return PDFReuseDecision(False, "pdf_state_missing")
    fields = (
        ("output_tex", "output_tex_sha256"),
        ("output_pdf", "output_pdf_sha256"),
        ("source_manifest_path", "source_manifest_sha256"),
        ("validation_path", "validation_sha256"),
    )
    files: dict[str, Path] = {}
    for path_key, hash_key in fields:
        if not _is_sha256(effective.get(hash_key)):
            return PDFReuseDecision(False, f"{hash_key}_missing")
        path = _project_regular_file(
            Path(project_dir), effective.get(path_key),
        )
        if path is None:
            return PDFReuseDecision(False, f"{path_key}_unsafe")
        files[path_key] = path
    revision_parent = files["output_pdf"].parent
    render_identity_fields = (
        "render_identity",
        "render_stem",
        "render_identity_receipt_path",
        "render_identity_receipt_sha256",
    )
    present_render_fields = [
        key for key in render_identity_fields if key in effective
    ]
    identity_receipt_at_revision = (
        revision_parent / ARTIFACT_ID_RECEIPT_NAME
    )
    resolved_render = None
    if not present_render_fields:
        try:
            identity_receipt_at_revision.lstat()
        except FileNotFoundError:
            pass
        else:
            return PDFReuseDecision(
                False, "render_identity_incomplete",
            )
    if present_render_fields:
        if any(path.parent != revision_parent for path in files.values()):
            return PDFReuseDecision(
                False, "render_revision_parent_mismatch",
            )
        if len(present_render_fields) != len(render_identity_fields):
            return PDFReuseDecision(False, "render_identity_incomplete")
        if not _is_sha256(effective.get("render_identity")):
            return PDFReuseDecision(False, "render_identity_invalid")
        if not _is_sha256(effective.get("render_identity_receipt_sha256")):
            return PDFReuseDecision(
                False, "render_identity_receipt_sha256_missing",
            )
        identity_receipt = _project_regular_file(
            Path(project_dir),
            effective.get("render_identity_receipt_path"),
        )
        if (
            identity_receipt is None
            or identity_receipt.name != ARTIFACT_ID_RECEIPT_NAME
            or identity_receipt.parent != revision_parent
        ):
            return PDFReuseDecision(
                False, "render_identity_receipt_path_unsafe",
            )
        try:
            resolved = resolve_artifact_dir(
                Path(project_dir) / ".arc-companion" / "renders" / "pdf",
                revision_parent,
                expected_identity=str(effective["render_identity"]),
                kind="pdf-render",
                allow_legacy=False,
            )
        except (ArtifactIdError, OSError, ValueError):
            return PDFReuseDecision(False, "render_identity_mismatch")
        if (
            resolved.receipt_path != identity_receipt
            or resolved.receipt_sha256
            != effective["render_identity_receipt_sha256"]
        ):
            return PDFReuseDecision(
                False, "render_identity_receipt_hash_mismatch",
            )
        resolved_render = resolved
    receipt_path = files["validation_path"]
    try:
        receipt_bytes = _read_bounded_regular_file(receipt_path)
        if receipt_bytes is None:
            return PDFReuseDecision(False, "receipt_invalid")
        if len(receipt_bytes) > PDF_RECEIPT_MAX_BYTES:
            return PDFReuseDecision(False, "receipt_oversized")
        if hashlib.sha256(receipt_bytes).hexdigest() != effective[
            "validation_sha256"
        ]:
            return PDFReuseDecision(False, "receipt_hash_mismatch")
        receipt = json.loads(receipt_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return PDFReuseDecision(False, "receipt_invalid")
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema_version")
        != PDF_VALIDATION_RECEIPT_VERSION
    ):
        return PDFReuseDecision(False, "legacy_receipt")
    if not _current_receipt_is_closed(receipt):
        return PDFReuseDecision(False, "receipt_invalid")
    recipe = render_recipe_sha256 or pdf_render_recipe_sha256()
    if effective.get("render_version") != PDF_RENDER_VERSION:
        return PDFReuseDecision(False, "render_version_mismatch")
    if str(
        effective.get("content_sha256")
        or published_content
        or ""
    ) != content_sha256:
        return PDFReuseDecision(False, "content_mismatch")
    if (
        effective.get("content_sha256") != content_sha256
        or (
            published_content is not None
            and published_content != content_sha256
        )
    ):
        return PDFReuseDecision(False, "content_mismatch")
    if effective.get("render_recipe_sha256") != recipe:
        return PDFReuseDecision(False, "render_recipe_mismatch")
    if effective.get("validator_version") != validator_version:
        return PDFReuseDecision(False, "validator_mismatch")
    if resolved_render is not None and (
        resolved_render.payload != {
            "content_sha256": content_sha256,
            "render_recipe_sha256": recipe,
            "validator_version": validator_version,
            "stem": effective.get("render_stem"),
        }
        or resolved_render.nonce is None
    ):
        return PDFReuseDecision(
            False, "render_identity_payload_mismatch",
        )
    source_credit = dict(receipt["source_credit_pdf"])
    if (
        effective.get("source_credit_sha256")
        != source_credit["canonical_sha256"]
        or effective.get("source_credit_observation_sha256")
        != source_credit["visible_projection_sha256"]
    ):
        return PDFReuseDecision(
            False, "source_credit_identity_mismatch",
        )
    expected_receipt = {
        "content_sha256": content_sha256,
        "render_recipe_sha256": recipe,
        "validator_version": validator_version,
        "pdf_sha256": effective["output_pdf_sha256"],
        "tex_sha256": effective["output_tex_sha256"],
        "source_manifest_sha256": effective[
            "source_manifest_sha256"
        ],
    }
    for key, value in expected_receipt.items():
        if receipt.get(key) != value:
            return PDFReuseDecision(False, f"{key}_mismatch")
    verified_files: dict[str, tuple[str, int]] = {}
    for path_key, hash_key in fields[:-1]:
        verified = _hash_regular_file_nofollow(files[path_key])
        if verified is None:
            return PDFReuseDecision(
                False, f"{path_key}_unreadable",
            )
        if verified[0] != effective[hash_key]:
            return PDFReuseDecision(
                False, f"{hash_key}_mismatch",
            )
        verified_files[path_key] = verified
    if not current_pdf_validation_receipt_matches(
        receipt,
        content_sha256=content_sha256,
        render_recipe_sha256=recipe,
        validator_version=validator_version,
        pdf_sha256=str(effective["output_pdf_sha256"]),
        tex_sha256=str(effective["output_tex_sha256"]),
        source_manifest_sha256=str(
            effective["source_manifest_sha256"]
        ),
        pdf_bytes=verified_files["output_pdf"][1],
    ):
        return PDFReuseDecision(False, "receipt_invalid")
    return PDFReuseDecision(
        True,
        "exact_match",
        {
            "source_credit_sha256": dict(
                receipt["source_credit_pdf"]
            )["canonical_sha256"],
            "source_credit_observation_sha256": dict(
                receipt["source_credit_pdf"]
            )["visible_projection_sha256"],
            **{
            key: effective[key]
            for key in (
                "content_sha256",
                "render_identity",
                "render_stem",
                "render_identity_receipt_path",
                "render_identity_receipt_sha256",
                "render_recipe_sha256",
                "validator_version",
                "output_tex",
                "output_tex_sha256",
                "output_pdf",
                "output_pdf_sha256",
                "source_manifest_path",
                "source_manifest_sha256",
                "validation_path",
                "validation_sha256",
            )
            if key in effective
            },
        },
    )


def find_adoptable_pdf_revision(
    project_dir: Path,
    *,
    content_sha256: str,
) -> PDFReuseDecision:
    """Find a fully published current revision left before a state commit."""

    root = project_dir.resolve()
    renders = root / ".arc-companion" / "renders" / "pdf"
    if not renders.is_dir() or renders.is_symlink():
        return PDFReuseDecision(False, "adoptable_revision_missing")
    try:
        directories = sorted(
            (
                path for path in renders.iterdir()
                if path.is_dir() and not path.is_symlink()
            ),
            key=lambda path: path.name,
            reverse=True,
        )
    except OSError:
        return PDFReuseDecision(False, "adoptable_revision_unreadable")
    for directory in directories:
        try:
            allocation = resolve_artifact_dir(
                renders,
                directory,
                kind="pdf-render",
                allow_legacy=False,
            )
        except (ArtifactIdError, OSError, ValueError):
            continue
        manifest = directory / "source-manifest.json"
        receipt_path = directory / "validation.json"
        tex_files = [
            path for path in directory.glob("*.tex")
            if path.is_file() and not path.is_symlink()
        ]
        pdf_files = [
            path for path in directory.glob("*.pdf")
            if path.is_file() and not path.is_symlink()
        ]
        if (
            len(tex_files) != 1
            or len(pdf_files) != 1
            or _project_regular_file(root, manifest) is None
            or _project_regular_file(root, receipt_path) is None
        ):
            continue
        receipt_bytes = _read_bounded_regular_file(receipt_path)
        if (
            receipt_bytes is None
            or len(receipt_bytes) > PDF_RECEIPT_MAX_BYTES
        ):
            continue
        try:
            receipt = json.loads(receipt_bytes)
        except (UnicodeError, json.JSONDecodeError):
            continue
        if (
            not isinstance(receipt, Mapping)
            or not _current_receipt_is_closed(receipt)
            or receipt.get("content_sha256") != content_sha256
            or receipt.get("render_recipe_sha256")
            != pdf_render_recipe_sha256()
            or receipt.get("validator_version")
            != PDF_VALIDATOR_VERSION
        ):
            continue
        pdf_state = {
            "content_sha256": content_sha256,
            "render_identity": allocation.identity,
            "render_stem": (
                allocation.payload.get("stem")
                if allocation.payload is not None else None
            ),
            "render_identity_receipt_path": str(
                allocation.receipt_path
            ),
            "render_identity_receipt_sha256": (
                allocation.receipt_sha256
            ),
            "render_version": PDF_RENDER_VERSION,
            "render_recipe_sha256": pdf_render_recipe_sha256(),
            "validator_version": PDF_VALIDATOR_VERSION,
            "source_credit_sha256": dict(
                receipt["source_credit_pdf"]
            )["canonical_sha256"],
            "source_credit_observation_sha256": dict(
                receipt["source_credit_pdf"]
            )["visible_projection_sha256"],
            "output_tex": str(tex_files[0]),
            "output_tex_sha256": receipt["tex_sha256"],
            "output_pdf": str(pdf_files[0]),
            "output_pdf_sha256": receipt["pdf_sha256"],
            "source_manifest_path": str(manifest),
            "source_manifest_sha256": receipt[
                "source_manifest_sha256"
            ],
            "validation_path": str(receipt_path),
            "validation_sha256": hashlib.sha256(
                receipt_bytes
            ).hexdigest(),
        }
        decision = match_validated_pdf_revision(
            root,
            {
                "published": {
                    "content_sha256": content_sha256,
                    "pdf": pdf_state,
                },
            },
            content_sha256=content_sha256,
        )
        if decision.reusable:
            return PDFReuseDecision(
                True, "adoptable_revision", decision.revision,
            )
    return PDFReuseDecision(False, "adoptable_revision_missing")


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
        raise PDFError(
            "latexmk is required to build a companion PDF",
            reason="tool_missing",
            stage="compile",
            executable="latexmk",
        )
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
                f"XeLaTeX compilation failed:\n{diagnostic}Command tail:\n{tail}",
                reason="compilation_failed",
                stage="compile",
                executable=executable,
                stderr=completed.stderr,
            )
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built, pdf_path)
    finally:
        for sidecar in tex_path.parent.glob(f"{jobname}.*"):
            if sidecar.is_file():
                sidecar.unlink(missing_ok=True)


def validate_pdf(
    pdf_path: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, object]:
    if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
        raise PDFError(
            "PDF is missing or empty",
            reason="pdf_missing",
            stage="input",
        )
    tools = {name: shutil.which(name) for name in ("pdfinfo", "pdftotext", "pdffonts", "pdftoppm")}
    missing = [name for name, path in tools.items() if path is None]
    if missing:
        raise PDFError(
            "PDF validation tools are required: "
            + ", ".join(missing),
            reason="tool_missing",
            stage="preflight",
        )

    pdf_bytes = pdf_path.stat().st_size
    with tempfile.TemporaryDirectory(
        prefix="arc-companion-pdf-validation-"
    ) as temporary:
        workspace = Path(temporary)
        info = _run(
            runner,
            [str(tools["pdfinfo"]), str(pdf_path)],
            stage="metadata",
        )
        try:
            pages, encrypted = _parse_pdfinfo(info)
        except PDFError as exc:
            raise PDFError(
                str(exc),
                reason="metadata_invalid",
                stage="metadata",
            ) from exc
        if encrypted:
            raise PDFError(
                "PDF is encrypted",
                reason="encrypted",
                stage="metadata",
            )

        text_path = workspace / "extracted.txt"
        _run(
            runner,
            [
                str(tools["pdftotext"]),
                str(pdf_path),
                str(text_path),
            ],
            stage="text",
        )
        extracted_text = (
            text_path.read_text(
                encoding="utf-8", errors="ignore",
            )
            if text_path.is_file() else ""
        )
        if not extracted_text.strip():
            raise PDFError(
                "PDF contains no searchable text",
                reason="text_missing",
                stage="text",
            )
        forbidden = _visible_layer_labels(extracted_text)
        if forbidden:
            raise PDFError(
                "PDF contains removed visible layer labels: "
                + ", ".join(forbidden),
                reason="visible_layer_label",
                stage="text",
            )

        fonts = _run(
            runner,
            [str(tools["pdffonts"]), str(pdf_path)],
            stage="fonts",
        )
        try:
            font_count = _validate_embedded_fonts(fonts)
            font_roles = {
                role: [
                    str(value)[:PDF_WARNING_MAX_CHARS]
                    for value in values[:PDF_FONT_ROLE_MAX_ITEMS]
                ]
                for role, values in _validate_font_roles(fonts).items()
            }
        except PDFError as exc:
            raise PDFError(
                str(exc),
                reason="fonts_invalid",
                stage="fonts",
            ) from exc
        raster_bytes = 0
        for page in range(1, pages + 1):
            raster_prefix = workspace / f"page-{page}"
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
                    str(PDF_RASTER_DPI),
                    str(pdf_path),
                    str(raster_prefix),
                ],
                stage="raster",
                page=page,
            )
            raster = Path(f"{raster_prefix}.png")
            if not raster.is_file() or raster.stat().st_size == 0:
                raise PDFError(
                    f"PDF page {page} rendering check failed",
                    reason="raster_missing",
                    stage="raster",
                    page=page,
                )
            raster_bytes += raster.stat().st_size
    return {
        "validator": PDF_VALIDATOR_VERSION,
        "result": "success",
        "pages": pages,
        "pages_checked": pages,
        "dpi": PDF_RASTER_DPI,
        "pdf_bytes": pdf_bytes,
        "text_bytes": len(extracted_text.encode("utf-8")),
        "raster_bytes": raster_bytes,
        "encrypted": False,
        "embedded_font_count": font_count,
        "font_roles": font_roles,
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


def _run(
    runner: Callable[..., subprocess.CompletedProcess[str]],
    command: list[str],
    *,
    stage: str,
    page: int | None = None,
) -> str:
    completed = runner(command, text=True, capture_output=True, timeout=120, check=False)
    if completed.returncode != 0:
        executable = Path(command[0]).name
        stderr = str(completed.stderr or "")[-PDF_STDERR_MAX_CHARS:]
        raise PDFError(
            f"{executable} failed during PDF {stage} validation: "
            f"{stderr}",
            reason="command_failed",
            stage=stage,
            page=page,
            executable=executable,
            stderr=stderr,
        )
    return completed.stdout
