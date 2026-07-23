from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from arc_companion.artifact_ids import (
    allocate_artifact_dir,
    render_artifact_identity,
)
from arc_companion.content import store_reader_content
from arc_companion.gc import apply_gc
from arc_companion.io import sha256_file, sha256_json, write_json
from arc_companion.pdf import (
    PDF_RENDER_VERSION,
    PDF_VALIDATION_RECEIPT_VERSION,
    PDF_VALIDATOR_VERSION,
    build_pdf_validation_receipt,
    pdf_render_recipe_sha256,
    publish_run_root_pdf,
)
from arc_companion.render import render_content
import arc_companion.provenance as provenance


def _content() -> dict[str, object]:
    return {
        "document": {
            "blocks": [{
                "block_id": "b1", "type": "paragraph", "text": "Source.",
            }],
            "equations": [],
            "figures": [],
            "tables": [],
            "assets": [],
        },
        "chapters": [],
        "segments": [{"segment_id": "s1", "block_ids": ["b1"]}],
        "chapter_guides": {},
        "translations": {
            "s1": {"blocks": [{"block_id": "b1", "text": "Translation."}]},
        },
        "annotations": {"s1": {"explanation": "Note.", "commentary": ""}},
        "glossary": {"entries": []},
        "metadata": {"title": "Fixture"},
        "reader_evidence_by_segment": {"s1": []},
        "language": "en",
        "translation_mode": "enabled",
        "accepted_ledger_chains": {},
        "review_overlay_hashes": {},
    }


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    with_outputs: bool = True,
    with_recovery: bool = False,
) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "project"
    identity = "5" * 64
    checkpoint = allocate_artifact_dir(
        root / ".arc-companion" / "checkpoints",
        identity,
        kind="checkpoint",
        allow_legacy=False,
    )
    document_payload = {
        "paper_id": "local:fixture",
        "document": {"blocks": []},
    }
    write_json(checkpoint.path / "document.json", document_payload)
    write_json(checkpoint.path / "source-snapshot-receipt.json", {
        "schema_version": "arc.companion.source-snapshot-receipt.v2",
        "paper_id": "local:fixture",
        "fingerprint": identity,
        "checkpoint_identity": identity,
        "build_instance_id": "1" * 32,
        "build_request_sha256": "2" * 64,
        "build_source_fingerprint": identity,
        "document_payload_sha256": sha256_json(document_payload),
        "chapters_pack_sha256": None,
        "evidence_sha256": None,
        "domain_context_sha256": None,
        "translation_reference_manifest_path": None,
        "translation_reference_manifest_sha256": None,
        "translation_reference_source_id": None,
        "translation_reference_source_hash": None,
    })
    stored = store_reader_content(
        root, content=_content(), checkpoint_dir=checkpoint.path,
    )
    state: dict[str, object] = {
        "schema_version": "arc.companion.state.v3",
        "status": "complete",
        "paper_id": "local:fixture",
        "fingerprint": identity,
        "checkpoint_identity": identity,
        "checkpoint_dir": str(checkpoint.path),
        "checkpoint_identity_receipt_path": str(checkpoint.receipt_path),
        "checkpoint_identity_receipt_sha256": checkpoint.receipt_sha256,
        "published": {
            "content_sha256": stored["content_sha256"],
            "content_object_path": str(stored["path"]),
        },
    }
    write_json(root / "state.json", state)
    gc_receipt = apply_gc(root)
    state["artifact_gc"] = {
        "status": "complete",
        "receipt_path": gc_receipt["receipt_path"],
        "receipt_sha256": gc_receipt["receipt_sha256"],
        "candidate_set_sha256": gc_receipt["candidate_set_sha256"],
        "reclaimed_bytes": gc_receipt["reclaimed_bytes"],
    }
    if with_outputs:
        render = root / ".arc-companion" / "renders" / "pdf" / "fixture"
        render.mkdir(parents=True)
        paths = {
            "output_pdf": render / "fixture.pdf",
            "output_tex": render / "fixture.tex",
            "source_manifest_path": render / "source-manifest.json",
            "validation_path": render / "validation.json",
        }
        paths["output_pdf"].write_bytes(b"%PDF-fixture")
        paths["output_tex"].write_text("fixture", encoding="utf-8")
        write_json(paths["source_manifest_path"], {"assets": []})
        write_json(paths["validation_path"], {
            "schema_version": PDF_VALIDATION_RECEIPT_VERSION,
            "result": "success",
        })
        run_pdf = root / "fixture.pdf"
        run_pdf.write_bytes(paths["output_pdf"].read_bytes())
        reader = root / "reader"
        (reader / "data").mkdir(parents=True)
        index = reader / "index.html"
        manifest = reader / "data" / "manifest.json"
        snapshot = reader / "data" / "snapshot.json"
        index.write_text("<html></html>", encoding="utf-8")
        write_json(manifest, {
            "schema_version": "arc.companion.web-manifest.v3",
        })
        write_json(snapshot, {
            "schema_version": "arc.companion.reader-snapshot.v4",
        })
        pdf = {
            **{key: str(path) for key, path in paths.items()},
            **{
                key.replace("path", "sha256") if key.endswith("_path")
                else f"{key}_sha256": sha256_file(path)
                for key, path in paths.items()
            },
            "output_run_pdf": str(run_pdf),
            "output_run_pdf_sha256": sha256_file(run_pdf),
        }
        # Correct the two non-output field hash names.
        pdf["source_manifest_sha256"] = sha256_file(
            paths["source_manifest_path"],
        )
        pdf["validation_sha256"] = sha256_file(paths["validation_path"])
        web = {
            "output_html": str(index),
            "output_html_sha256": sha256_file(index),
            "web_manifest_path": str(manifest),
            "web_manifest_sha256": sha256_file(manifest),
            "reader_snapshot_path": str(snapshot),
            "reader_snapshot_sha256": sha256_file(snapshot),
        }
        state["published"] = {
            **dict(state["published"]),
            "pdf": pdf,
            "web": web,
        }
        monkeypatch.setattr(
            provenance,
            "match_validated_pdf_revision",
            lambda *_args, **_kwargs: SimpleNamespace(
                reusable=True, reason="exact",
            ),
        )
        monkeypatch.setattr(
            provenance, "validate_reader_project",
            lambda *_args, **_kwargs: {"ok": True},
        )
    if with_recovery:
        write_json(root / ".arc-companion" / "resume-transaction.json", {
            "schema_version": "arc.companion.resume-transaction.v3",
            "status": "complete",
            "checkpoint_path": str(checkpoint.path),
            "checkpoint_fingerprint": identity,
            "entries": [],
        })
    write_json(root / "state.json", state)
    return root, state


def _merge_state(root: Path, values: dict[str, object]) -> dict[str, object]:
    state = json.loads((root / "state.json").read_text())
    published = dict(state.get("published") or {})
    published["provenance"] = dict(values["published_provenance"])
    state["published"] = published
    write_json(root / "state.json", state)
    return state


def _install_owner_valid_pdf(
    root: Path, state: dict[str, object],
) -> None:
    published = dict(state["published"])
    content_sha256 = str(published["content_sha256"])
    payload = {
        "content_sha256": content_sha256,
        "render_recipe_sha256": pdf_render_recipe_sha256(),
        "validator_version": PDF_VALIDATOR_VERSION,
        "stem": content_sha256,
    }
    nonce = "7" * 32
    identity = render_artifact_identity(
        kind="pdf-render", payload=payload, nonce=nonce,
    )
    allocation = allocate_artifact_dir(
        root / ".arc-companion" / "renders" / "pdf",
        identity,
        kind="pdf-render",
        stem=content_sha256,
        payload=payload,
        nonce=nonce,
        allow_legacy=False,
    )
    tex = allocation.path / "paper.tex"
    pdf = allocation.path / "paper.pdf"
    manifest = allocation.path / "source-manifest.json"
    validation = allocation.path / "validation.json"
    tex.write_text("fixture tex", encoding="utf-8")
    pdf.write_bytes(b"%PDF owner-valid fixture")
    write_json(manifest, {"assets": []})
    credit = {
        "schema_version": "arc.companion.source-credit-pdf-observation.v1",
        "canonical_sha256": "a" * 64,
        "searchable_text_sha256": "b" * 64,
        "ordered_ids": [],
        "visible_projection_sha256": "c" * 64,
        "visible_counts": {
            "authors": 0,
            "affiliations": 0,
            "profiles": 0,
        },
    }
    write_json(validation, build_pdf_validation_receipt(
        content_sha256=content_sha256,
        pdf_sha256=sha256_file(pdf),
        tex_sha256=sha256_file(tex),
        source_manifest_sha256=sha256_file(manifest),
        pdf_report={
            "validator": PDF_VALIDATOR_VERSION,
            "result": "success",
            "pages": 1,
            "pages_checked": 1,
            "dpi": 144,
            "pdf_bytes": pdf.stat().st_size,
            "text_bytes": 1,
            "raster_bytes": 1,
            "encrypted": False,
            "embedded_font_count": 2,
            "font_roles": {
                "sans": ["Noto Sans"],
                "serif": ["Latin Modern"],
            },
        },
        source_credit_pdf=credit,
    ))
    pdf_state = {
        "content_sha256": content_sha256,
        "render_identity": identity,
        "render_stem": content_sha256,
        "render_identity_receipt_path": str(allocation.receipt_path),
        "render_identity_receipt_sha256": allocation.receipt_sha256,
        "render_version": PDF_RENDER_VERSION,
        "render_recipe_sha256": pdf_render_recipe_sha256(),
        "validator_version": PDF_VALIDATOR_VERSION,
        "source_credit_sha256": credit["canonical_sha256"],
        "source_credit_observation_sha256": credit[
            "visible_projection_sha256"
        ],
        "output_tex": str(tex),
        "output_tex_sha256": sha256_file(tex),
        "output_pdf": str(pdf),
        "output_pdf_sha256": sha256_file(pdf),
        "source_manifest_path": str(manifest),
        "source_manifest_sha256": sha256_file(manifest),
        "validation_path": str(validation),
        "validation_sha256": sha256_file(validation),
    }
    delivery = publish_run_root_pdf(
        pdf,
        root,
        expected_sha256=str(pdf_state["output_pdf_sha256"]),
    )
    pdf_state.update(delivery)
    published["pdf"] = pdf_state
    state["published"] = published
    state.update(delivery)
    write_json(root / "state.json", state)


def test_final_provenance_round_trip_is_owner_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, state = _fixture(tmp_path, monkeypatch)

    plan = provenance.plan_final_provenance(
        root, state=state, mode="build",
    )
    published = provenance.commit_final_provenance(
        root,
        plan=plan,
        state=state,
        state_merger=lambda values: _merge_state(root, dict(values)),
    )
    current = json.loads((root / "state.json").read_text())
    value = provenance.validate_published_provenance(root, current)

    assert value is not None
    assert value["final_id"] == published["final_id"]
    assert value["counts"]["schema_version"] == provenance.FINAL_COUNTS_VERSION
    assert value["attribution"] == {
        "status": "partial",
        "segment_count": 1,
        "review_receipt_count": 0,
        "review_calls": None,
    }
    assert {
        item["category"] for item in value["controls"]
    } == {"source_snapshot", "render_validation", "artifact_gc"}


def test_pdf_noop_reuse_upgrades_missing_provenance_with_real_owners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, state = _fixture(
        tmp_path, monkeypatch, with_outputs=False,
    )
    _install_owner_valid_pdf(root, state)
    import arc_companion.gc as gc_module

    monkeypatch.setattr(
        gc_module,
        "run_post_publication_gc",
        lambda *_args, **_kwargs: pytest.fail(
            "T20 GC called on exact PDF no-op provenance upgrade"
        ),
    )

    result = render_content(
        root,
        format="pdf",
        compiler=lambda *_args: pytest.fail("compiler called on PDF reuse"),
        pdf_validator=lambda *_args: pytest.fail(
            "validator called on PDF reuse"
        ),
    )

    assert result["ok"] is True
    assert result["data"]["pdf_reuse_status"] == "hit"
    current = json.loads((root / "state.json").read_text())
    value = provenance.validate_published_provenance(root, current)
    assert value is not None
    assert value["outputs"]["mode"] == "render_pdf"
    assert value["outputs"]["pdf"]["sha256"] == current["published"][
        "pdf"
    ]["output_pdf_sha256"]
    assert not current.get("provenance_warning")
    from arc_companion.observability import enrich_status

    status = enrich_status(root, current)
    assert status["provenance_status"] == "complete"
    assert status["provenance_counts_status"] == "complete"
    from arc_companion.package import package_project

    package = package_project(root)
    assert package["ok"] is True, package
    packaged_paths = {
        item["path"] for item in package["data"]["files"]
    }
    assert current["published"]["provenance"]["path"] in packaged_paths
    assert current["published"]["provenance"]["counts_path"] in packaged_paths
    assert not {
        item["path"] for item in value["controls"]
        if item["category"] != "render_validation"
    } & packaged_paths

    missing_group = deepcopy(current)
    missing_group["published"].pop("provenance")
    write_json(root / "state.json", missing_group)
    with pytest.raises(
        provenance.ProvenanceError, match="required published provenance",
    ):
        provenance.validate_published_provenance(root, missing_group)
    write_json(root / "state.json", current)

    bad_sha256 = "0" * 64
    current["published"]["provenance"]["sha256"] = bad_sha256
    write_json(root / "state.json", current)
    monkeypatch.setattr(
        provenance,
        "commit_final_provenance",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            provenance.ProvenanceError(
                "injected_provenance_failure", "injected repair failure",
            )
        ),
    )

    rejected = render_content(
        root,
        format="pdf",
        compiler=lambda *_args: pytest.fail("compiler called on PDF reuse"),
        pdf_validator=lambda *_args: pytest.fail(
            "validator called on PDF reuse"
        ),
    )

    assert rejected["ok"] is False
    assert rejected["error"]["code"] == "companion_provenance_failed"
    preserved = json.loads((root / "state.json").read_text())
    assert preserved["published"]["provenance"]["sha256"] == bad_sha256
    assert preserved["provenance_warning"]["code"] == (
        "injected_provenance_failure"
    )


def test_final_id_semantic_projection_excludes_project_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, state = _fixture(tmp_path, monkeypatch)
    plan = provenance.plan_final_provenance(
        root, state=state, mode="build",
    )
    provenance.commit_final_provenance(
        root,
        plan=plan,
        state=state,
        state_merger=lambda values: _merge_state(root, dict(values)),
    )
    current = json.loads((root / "state.json").read_text())
    value = provenance.validate_published_provenance(root, current)
    relocated = deepcopy(value)
    relocated["checkpoint"]["path"] = "relocated/checkpoint"
    relocated["reviewed_content"]["path"] = "relocated/content.json"
    relocated["counts"]["path"] = "relocated/counts.json"
    for ordinal, record in enumerate(relocated["controls"]):
        record["path"] = f"relocated/control-{ordinal}.json"
    for key, record in relocated["outputs"].items():
        if isinstance(record, dict):
            record["path"] = f"relocated/{key}"

    original = provenance._semantic_projection(
        fingerprint=value["fingerprint"],
        checkpoint=value["checkpoint"],
        reviewed_content=value["reviewed_content"],
        outputs=value["outputs"],
        controls=value["controls"],
        counts=value["counts"],
    )
    moved = provenance._semantic_projection(
        fingerprint=relocated["fingerprint"],
        checkpoint=relocated["checkpoint"],
        reviewed_content=relocated["reviewed_content"],
        outputs=relocated["outputs"],
        controls=relocated["controls"],
        counts=relocated["counts"],
    )

    assert provenance._sha_json(original) == value["final_id"]
    assert provenance._sha_json(moved) == value["final_id"]


def test_direct_resume_closes_missing_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import arc_companion.pipeline as pipeline_module

    root, _state = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        pipeline_module,
        "_resume_companion_unlocked",
        lambda *_args, **_kwargs: {
            "ok": True,
            "data": json.loads((root / "state.json").read_text()),
            "errors": [],
            "meta": {"resumed": True},
        },
    )

    result = pipeline_module.resume_companion(root)

    assert result["ok"] is True
    current = json.loads((root / "state.json").read_text())
    assert current["published"]["provenance"]["status"] == "complete"
    assert provenance.validate_published_provenance(root, current) is not None


def test_new_render_surfaces_provenance_failure_but_preserves_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import arc_companion.render as render_module

    root, _state = _fixture(
        tmp_path, monkeypatch, with_outputs=False,
    )
    monkeypatch.setattr(
        render_module,
        "render_companion_tex",
        lambda *_args, **_kwargs: ("fixture tex", {"assets": []}),
    )
    monkeypatch.setattr(
        render_module, "validate_tex_fidelity",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        render_module,
        "_new_publication_requires_provenance",
        lambda _state: True,
    )

    result = render_content(
        root,
        format="pdf",
        compiler=lambda _tex, pdf: pdf.write_bytes(b"%PDF new fixture"),
        pdf_validator=lambda _pdf: {"pages": 1},
    )

    assert result["ok"] is False
    assert result["error"]["code"] == "companion_provenance_failed"
    assert result["meta"]["publication_preserved"] is True
    current = json.loads((root / "state.json").read_text())
    assert Path(current["published"]["pdf"]["output_pdf"]).is_file()
    assert current["status"] == "complete"
    assert current["provenance_warning"]["code"] == "provenance_output_invalid"
    assert "provenance" not in current["published"]


@pytest.mark.parametrize(
    ("mode", "with_outputs"),
    [
        ("build", False),
        ("render_pdf", False),
        ("render_web", False),
        ("render_all", False),
    ],
)
def test_provenance_mode_requires_current_output_lanes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    with_outputs: bool,
) -> None:
    root, state = _fixture(
        tmp_path, monkeypatch, with_outputs=with_outputs,
    )
    with pytest.raises(
        provenance.ProvenanceError, match="required current output",
    ):
        provenance.plan_final_provenance(root, state=state, mode=mode)


def test_terminal_recovery_is_snapshotted_without_mutating_live_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, state = _fixture(tmp_path, monkeypatch, with_recovery=True)
    live = root / ".arc-companion" / "resume-transaction.json"
    before = live.read_bytes()

    plan = provenance.plan_final_provenance(
        root, state=state, mode="build",
    )
    provenance.commit_final_provenance(
        root,
        plan=plan,
        state=state,
        state_merger=lambda values: _merge_state(root, dict(values)),
    )

    assert live.read_bytes() == before
    history = root / str(plan.recovery_path)
    assert history.read_bytes() == before


def test_counts_and_provenance_tamper_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, state = _fixture(tmp_path, monkeypatch)
    plan = provenance.plan_final_provenance(
        root, state=state, mode="build",
    )
    provenance.commit_final_provenance(
        root,
        plan=plan,
        state=state,
        state_merger=lambda values: _merge_state(root, dict(values)),
    )
    current = json.loads((root / "state.json").read_text())
    mapping = current["published"]["provenance"]
    counts = root / mapping["counts_path"]
    value = json.loads(counts.read_text())
    value["segment_counts"]["total"] = 2
    write_json(counts, value)

    with pytest.raises(provenance.ProvenanceError):
        provenance.validate_published_provenance(root, current)


def test_partial_published_provenance_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, state = _fixture(tmp_path, monkeypatch)
    state["published"]["provenance"] = {"status": "complete"}

    with pytest.raises(
        provenance.ProvenanceError, match="published provenance state",
    ):
        provenance.validate_published_provenance(root, state)
