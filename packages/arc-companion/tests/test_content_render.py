from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from arc_companion.content import (
    CONTENT_RECEIPT_VERSION,
    READER_CONTENT_VERSION,
    ContentBundleError,
    load_reader_content,
    store_reader_content,
)
from arc_companion.io import read_json, sha256_file, write_json
from arc_companion.pipeline import validate_project
from arc_companion.render import render_content
from arc_companion.run_lock import ProjectBuildLock


def _content() -> dict:
    return {
        "document": {
            "blocks": [{"block_id": "b1", "type": "paragraph", "text": "Source."}],
            "equations": [], "figures": [], "tables": [], "assets": [],
        },
        "chapters": [],
        "segments": [{"segment_id": "s1", "block_ids": ["b1"]}],
        "chapter_guides": {},
        "translations": {"s1": {"blocks": [{"block_id": "b1", "text": "译文。"}]}},
        "annotations": {"s1": {"explanation": "Note.", "commentary": ""}},
        "glossary": {"entries": []},
        "metadata": {"title": "Fixture"},
        "reader_evidence_by_segment": {"s1": []},
        "language": "zh-CN",
        "translation_mode": "enabled",
        "accepted_ledger_chains": {},
        "review_overlay_hashes": {},
    }


def _project(tmp_path: Path) -> tuple[Path, str]:
    project = tmp_path / "project"
    stored = store_reader_content(project, content=_content())
    old_pdf = project / "old.pdf"
    old_pdf.write_bytes(b"old-pdf")
    write_json(project / "state.json", {
        "schema_version": "arc.companion.state.v3",
        "status": "failed",
        "paper_id": "local:fixture",
        "published": {
            "content_sha256": stored["content_sha256"],
            "pdf": {"output_pdf": str(old_pdf), "output_pdf_sha256": "old-hash"},
        },
    })
    return project, stored["content_sha256"]


def _render_fakes(monkeypatch, *, fail_validation: bool = False) -> None:
    import arc_companion.render as module

    monkeypatch.setattr(
        module,
        "render_companion_tex",
        lambda *args, **kwargs: ("fixture tex", {"assets": []}),
    )
    monkeypatch.setattr(module, "validate_tex_fidelity", lambda *args: [])

    def compiler(_tex: Path, pdf: Path) -> None:
        pdf.write_bytes(b"new-pdf")

    monkeypatch.setattr(module, "compile_latex", compiler)
    monkeypatch.setattr(module, "validate_pdf", lambda _path: (
        (_ for _ in ()).throw(RuntimeError("validator failed"))
        if fail_validation else {"pages": 1}
    ))


def test_reviewed_content_is_immutable_and_tampering_is_rejected(tmp_path: Path) -> None:
    project, digest = _project(tmp_path)
    path = project / ".arc-companion" / "objects" / "reader-content" / f"{digest}.json"
    value = read_json(path)
    value["content"]["annotations"]["s1"]["explanation"] = "tampered"
    write_json(path, value)

    try:
        load_reader_content(project, digest)
    except ContentBundleError as exc:
        assert "hash" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("tampered content was accepted")


def test_reviewed_content_uses_current_schema_and_receipt_versions(tmp_path: Path) -> None:
    project, digest = _project(tmp_path)
    value = read_json(
        project / ".arc-companion" / "objects" / "reader-content" / f"{digest}.json"
    )

    assert value["schema_version"] == READER_CONTENT_VERSION
    assert value["validation_receipt"]["schema_version"] == CONTENT_RECEIPT_VERSION
    assert value["validation_receipt"]["validator_version"] == CONTENT_RECEIPT_VERSION


def test_store_refreshes_matching_legacy_content_envelope(tmp_path: Path) -> None:
    project, digest = _project(tmp_path)
    path = project / ".arc-companion" / "objects" / "reader-content" / f"{digest}.json"
    legacy = read_json(path)
    legacy["schema_version"] = "arc.companion.reader-content.v1"
    legacy["validation_receipt"]["schema_version"] = (
        "arc.companion.reader-content-validation.v1"
    )
    legacy["validation_receipt"]["validator_version"] = (
        "arc.companion.reader-content-validation.v1"
    )
    path.write_text(json.dumps(legacy), encoding="utf-8")

    stored = store_reader_content(project, content=legacy["content"])

    assert stored["schema_version"] == READER_CONTENT_VERSION
    assert stored["validation_receipt"]["schema_version"] == CONTENT_RECEIPT_VERSION


def test_validation_receipt_checks_are_bound_to_bundle_identity(tmp_path: Path) -> None:
    project, digest = _project(tmp_path)
    path = project / ".arc-companion" / "objects" / "reader-content" / f"{digest}.json"
    value = read_json(path)
    value["validation_receipt"]["checks"] = ["forged_check"]
    write_json(path, value)

    result = render_content(project, format="pdf", content_sha256=digest)

    assert result["ok"] is False
    assert result["error"]["code"] == "content_bundle_invalid"
    assert result["meta"]["provider_calls"] == 0


def test_render_pdf_uses_separate_lock_and_publishes_versioned_output(
    tmp_path: Path, monkeypatch,
) -> None:
    project, digest = _project(tmp_path)
    _render_fakes(monkeypatch)
    calls = {"provider": 0}

    # A generation build may be active; render-only owns a different lock.
    with ProjectBuildLock(project / ".arc-companion-build.lock"):
        result = render_content(
            project, format="pdf", content_sha256=digest,
            compiler=lambda _tex, pdf: pdf.write_bytes(b"new-pdf"),
            pdf_validator=lambda _pdf: {"pages": 1},
        )

    assert result["ok"] is True
    assert result["data"]["provider_calls"] == calls["provider"] == 0
    output = Path(result["data"]["output_pdf"])
    assert output.read_bytes() == b"new-pdf"
    assert ".arc-companion/renders/pdf/" in output.as_posix()
    state = read_json(project / "state.json")
    assert state["published"]["content_sha256"] == digest
    assert state["published"]["pdf"]["output_pdf"] == str(output)


def test_render_failure_preserves_last_good_state_and_pdf(tmp_path: Path) -> None:
    project, digest = _project(tmp_path)
    before = (project / "state.json").read_bytes()
    old_pdf = project / "old.pdf"

    result = render_content(
        project, format="pdf", content_sha256=digest,
        compiler=lambda _tex, pdf: pdf.write_bytes(b"candidate"),
        pdf_validator=lambda _pdf: (_ for _ in ()).throw(RuntimeError("invalid PDF")),
    )

    assert result["ok"] is False
    assert result["error"]["code"] == "render_failed"
    assert (project / "state.json").read_bytes() == before
    assert old_pdf.read_bytes() == b"old-pdf"


def test_partial_publish_failure_cannot_overwrite_last_good_pdf(
    tmp_path: Path, monkeypatch,
) -> None:
    import arc_companion.render as module

    project, digest = _project(tmp_path)
    before = (project / "state.json").read_bytes()
    replacements = 0
    real_replace = module._publish_replace

    def fail_second(source: Path, target: Path) -> None:
        nonlocal replacements
        replacements += 1
        if replacements == 2:
            raise OSError("injected publish failure")
        real_replace(source, target)

    monkeypatch.setattr(module, "_publish_replace", fail_second)
    monkeypatch.setattr(module, "render_companion_tex", lambda *args, **kwargs: (
        "fixture tex", {"assets": []}
    ))
    monkeypatch.setattr(module, "validate_tex_fidelity", lambda *args: [])

    result = render_content(
        project, format="pdf", content_sha256=digest,
        compiler=lambda _tex, pdf: pdf.write_bytes(b"candidate"),
        pdf_validator=lambda _pdf: {"pages": 1},
    )

    assert result["ok"] is False
    assert (project / "state.json").read_bytes() == before
    assert (project / "old.pdf").read_bytes() == b"old-pdf"


def test_render_rejects_invalid_bundle_without_falling_back(tmp_path: Path) -> None:
    project, digest = _project(tmp_path)
    path = project / ".arc-companion" / "objects" / "reader-content" / f"{digest}.json"
    path.write_text(json.dumps({"schema_version": "wrong"}), encoding="utf-8")

    result = render_content(project, format="pdf", content_sha256=digest)

    assert result["ok"] is False
    assert result["error"]["code"] == "content_bundle_invalid"
    assert result["meta"]["provider_calls"] == 0


def test_cli_import_does_not_load_pipeline_or_llm_runtime() -> None:
    source_root = Path(__file__).parents[1] / "src"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source_root)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import arc_companion.cli; "
                "assert 'arc_companion.pipeline' not in sys.modules; "
                "assert not any(name == 'arc_llm' or name.startswith('arc_llm.') "
                "for name in sys.modules)"
            ),
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_validate_uses_published_last_good_while_active_run_failed(
    tmp_path: Path, monkeypatch,
) -> None:
    import arc_companion.pipeline as pipeline_module
    import arc_companion.web as web_module

    project, digest = _project(tmp_path)
    revision = project / ".arc-companion" / "renders" / "pdf" / "last-good"
    revision.mkdir(parents=True)
    paths = {
        "output_tex": revision / "paper.tex",
        "output_pdf": revision / "paper.pdf",
        "source_manifest_path": revision / "source-manifest.json",
        "validation_path": revision / "validation.json",
    }
    paths["output_tex"].write_text("fixture tex", encoding="utf-8")
    paths["output_pdf"].write_bytes(b"%PDF fixture")
    write_json(paths["source_manifest_path"], {"assets": []})
    write_json(paths["validation_path"], {"ok": True})
    published_pdf = {
        key: str(path) for key, path in paths.items()
    }
    published_pdf.update({
        key.replace("path", "sha256") if key.endswith("_path") else f"{key}_sha256": sha256_file(path)
        for key, path in paths.items()
    })
    # Correct the two output keys whose hash names append rather than replace.
    published_pdf["output_tex_sha256"] = sha256_file(paths["output_tex"])
    published_pdf["output_pdf_sha256"] = sha256_file(paths["output_pdf"])
    state = read_json(project / "state.json")
    state.update({
        "schema_version": "arc.companion.state.v3",
        "status": "failed",
        "checkpoint_dir": str(project / "new-active-checkpoint"),
        "published": {
            "content_sha256": digest,
            "pdf": published_pdf,
            "web": {"output_html": str(project / "reader" / "index.html")},
        },
    })
    for key in (*paths, "output_tex_sha256", "output_pdf_sha256",
                "source_manifest_sha256", "validation_sha256", "output_html"):
        state.pop(key, None)
    write_json(project / "state.json", state)
    monkeypatch.setattr(pipeline_module, "validate_tex_fidelity", lambda *args: [])
    observed: dict = {}

    def validate_web(_root: Path, *, state: dict) -> dict:
        observed.update(state)
        return {"ok": True}

    monkeypatch.setattr(web_module, "validate_reader_project", validate_web)

    result = validate_project(
        project, pdf_validator=lambda path: {"bytes": path.stat().st_size},
    )

    assert result["ok"] is True
    assert result["data"]["output_pdf"] == str(paths["output_pdf"])
    assert observed["output_pdf"] == str(paths["output_pdf"])
    assert observed["output_html"].endswith("reader/index.html")
