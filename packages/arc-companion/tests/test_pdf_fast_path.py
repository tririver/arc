from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from arc_companion.io import sha256_file, write_json
from arc_companion.pdf import (
    PDF_RENDER_VERSION,
    PDF_VALIDATOR_VERSION,
    build_pdf_rejected_attempt,
    build_pdf_validation_receipt,
    find_adoptable_pdf_revision,
    match_validated_pdf_revision,
    pdf_render_recipe_sha256,
)


CONTENT_SHA256 = "1" * 64
SOURCE_CREDIT_PDF = {
    "schema_version": "arc.companion.source-credit-pdf-observation.v1",
    "canonical_sha256": "a" * 64,
    "searchable_text_sha256": "c" * 64,
    "ordered_ids": ["author:1"],
    "visible_projection_sha256": "b" * 64,
    "visible_counts": {
        "authors": 1,
        "affiliations": 0,
        "profiles": 0,
    },
}
PDF_REPORT = {
    "validator": PDF_VALIDATOR_VERSION,
    "result": "success",
    "pages": 1,
    "pages_checked": 1,
    "dpi": 144,
    "pdf_bytes": 12,
    "text_bytes": 3,
    "raster_bytes": 9,
    "encrypted": False,
    "embedded_font_count": 2,
    "font_roles": {"sans": ["Noto Sans"], "serif": ["Latin Modern"]},
}


def _revision(project: Path) -> dict[str, object]:
    root = project / ".arc-companion" / "renders" / "pdf" / "revision"
    root.mkdir(parents=True)
    tex = root / "paper.tex"
    pdf = root / "paper.pdf"
    manifest = root / "source-manifest.json"
    receipt = root / "validation.json"
    tex.write_text("tex", encoding="utf-8")
    pdf.write_bytes(b"%PDF current")
    write_json(manifest, {"assets": []})
    receipt_value = build_pdf_validation_receipt(
        content_sha256=CONTENT_SHA256,
        pdf_sha256=sha256_file(pdf),
        tex_sha256=sha256_file(tex),
        source_manifest_sha256=sha256_file(manifest),
        pdf_report=PDF_REPORT,
        source_credit_pdf=SOURCE_CREDIT_PDF,
    )
    write_json(receipt, receipt_value)
    pdf_state = {
        "content_sha256": CONTENT_SHA256,
        "render_version": PDF_RENDER_VERSION,
        "render_recipe_sha256": pdf_render_recipe_sha256(),
        "validator_version": PDF_VALIDATOR_VERSION,
        "source_credit_sha256": SOURCE_CREDIT_PDF[
            "canonical_sha256"
        ],
        "source_credit_observation_sha256": SOURCE_CREDIT_PDF[
            "visible_projection_sha256"
        ],
        "output_tex": str(tex),
        "output_tex_sha256": sha256_file(tex),
        "output_pdf": str(pdf),
        "output_pdf_sha256": sha256_file(pdf),
        "source_manifest_path": str(manifest),
        "source_manifest_sha256": sha256_file(manifest),
        "validation_path": str(receipt),
        "validation_sha256": sha256_file(receipt),
    }
    return {
        "schema_version": "arc.companion.state.v3",
        "status": "complete",
        "content_sha256": CONTENT_SHA256,
        "published": {
            "content_sha256": CONTENT_SHA256,
            "pdf": pdf_state,
        },
    }


def test_match_validated_pdf_revision_accepts_exact_five_identity(
    tmp_path: Path,
) -> None:
    state = _revision(tmp_path)

    decision = match_validated_pdf_revision(
        tmp_path, state, content_sha256=CONTENT_SHA256,
    )

    assert decision.reusable
    assert decision.reason == "exact_match"


def test_fully_published_revision_is_adoptable_before_state_commit(
    tmp_path: Path,
) -> None:
    _revision(tmp_path)

    decision = find_adoptable_pdf_revision(
        tmp_path, content_sha256=CONTENT_SHA256,
    )

    assert decision.reusable
    assert decision.reason == "adoptable_revision"
    assert decision.revision["output_pdf_sha256"]


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("content_sha256", "2" * 64, "content_mismatch"),
        ("render_recipe_sha256", "3" * 64, "render_recipe_mismatch"),
        ("validator_version", "old", "validator_mismatch"),
        ("output_pdf_sha256", "4" * 64, "pdf_sha256_mismatch"),
        ("validation_sha256", "5" * 64, "receipt_hash_mismatch"),
    ],
)
def test_match_validated_pdf_revision_rejects_identity_mismatch(
    tmp_path: Path,
    field: str,
    value: str,
    reason: str,
) -> None:
    state = _revision(tmp_path)
    pdf_state = state["published"]["pdf"]
    if field == "content_sha256":
        state["content_sha256"] = value
        state["published"]["content_sha256"] = value
        pdf_state[field] = value
    else:
        pdf_state[field] = value

    decision = match_validated_pdf_revision(
        tmp_path, state, content_sha256=CONTENT_SHA256,
    )

    assert not decision.reusable
    assert decision.reason == reason


def test_match_validated_pdf_revision_rejects_legacy_receipt(
    tmp_path: Path,
) -> None:
    state = _revision(tmp_path)
    pdf_state = state["published"]["pdf"]
    receipt = Path(pdf_state["validation_path"])
    write_json(receipt, {"ok": True})
    pdf_state["validation_sha256"] = sha256_file(receipt)

    decision = match_validated_pdf_revision(
        tmp_path, state, content_sha256=CONTENT_SHA256,
    )

    assert not decision.reusable
    assert decision.reason == "legacy_receipt"


def test_legacy_receipt_classification_precedes_recipe_checks(
    tmp_path: Path,
) -> None:
    state = _revision(tmp_path)
    pdf_state = state["published"]["pdf"]
    receipt = Path(pdf_state["validation_path"])
    write_json(receipt, {"ok": True})
    pdf_state["validation_sha256"] = sha256_file(receipt)
    pdf_state["render_recipe_sha256"] = "2" * 64
    pdf_state["validator_version"] = "legacy"

    decision = match_validated_pdf_revision(
        tmp_path, state, content_sha256=CONTENT_SHA256,
    )

    assert decision.reason == "legacy_receipt"


def test_published_pdf_is_authoritative_and_not_filled_from_flat_state(
    tmp_path: Path,
) -> None:
    state = _revision(tmp_path)
    pdf_state = state["published"]["pdf"]
    state["render_recipe_sha256"] = pdf_state.pop(
        "render_recipe_sha256"
    )

    decision = match_validated_pdf_revision(
        tmp_path, state, content_sha256=CONTENT_SHA256,
    )

    assert not decision.reusable
    assert decision.reason == "render_recipe_mismatch"


@pytest.mark.parametrize("mutation", ["extra", "malformed_roles"])
def test_match_rejects_nonclosed_or_malformed_current_receipt(
    tmp_path: Path,
    mutation: str,
) -> None:
    state = _revision(tmp_path)
    pdf_state = state["published"]["pdf"]
    receipt_path = Path(pdf_state["validation_path"])
    receipt = deepcopy(
        __import__("json").loads(receipt_path.read_text(encoding="utf-8"))
    )
    if mutation == "extra":
        receipt["raw_output"] = "/private/path"
    else:
        receipt["pdf"]["font_roles"]["sans"] = "not-a-list"
    write_json(receipt_path, receipt)
    pdf_state["validation_sha256"] = sha256_file(receipt_path)

    decision = match_validated_pdf_revision(
        tmp_path, state, content_sha256=CONTENT_SHA256,
    )

    assert not decision.reusable
    assert decision.reason == "receipt_invalid"


def test_rejected_attempt_has_only_enumerated_diagnostics() -> None:
    attempt = build_pdf_rejected_attempt(
        RuntimeError("/private/path raw stderr"),
        content_sha256=CONTENT_SHA256,
    )

    assert "message" not in attempt
    assert "stderr" not in attempt
    assert "/private/path" not in repr(attempt)


def test_match_validated_pdf_revision_rejects_symlink_without_touching_target(
    tmp_path: Path,
) -> None:
    state = _revision(tmp_path)
    pdf_state = state["published"]["pdf"]
    original = Path(pdf_state["output_pdf"])
    external = tmp_path.parent / f"{tmp_path.name}-external.pdf"
    external.write_bytes(original.read_bytes())
    original.unlink()
    original.symlink_to(external)
    before = external.read_bytes()

    decision = match_validated_pdf_revision(
        tmp_path, state, content_sha256=CONTENT_SHA256,
    )

    assert not decision.reusable
    assert decision.reason == "output_pdf_unsafe"
    assert external.read_bytes() == before
