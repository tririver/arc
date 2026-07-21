from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from arc_companion.content import (
    ContentBundleError,
    load_reader_content,
    store_reader_content,
)
from arc_companion.io import read_json, write_json
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
