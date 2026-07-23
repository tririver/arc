from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import pytest

from arc_companion import cli
import arc_companion
from arc_companion.latex import (
    LatexError,
    _render_html_fragment,
    _render_table,
    render_companion_tex,
    validate_tex_fidelity,
)
from arc_companion.package import package_project
from arc_companion.source import SourceError, load_source_bundle, validate_complete_document
from arc_paper.parse.source import parse_source_input


ROOT = Path(__file__).resolve().parents[3]


def _current_parsed(source_id: str, *, document: dict | None = None, kind: str = "auto") -> dict:
    return {
        "paper_id": source_id,
        "source_hash": "a" * 64,
        "structure": {
            "schema_version": "arc.paper.structure.v1",
            "requested_document_kind": kind,
            "resolved_document_kind": "article",
            "chapters": [],
        },
        "index_entries": {"schema_version": "arc.paper.index_entries.v1", "entries": []},
        "document": document or {
            "blocks": [{"block_id": "b1", "type": "text", "text": "x"}],
            "integrity": {"status": "complete"},
        },
    }


def test_markdown_rich_block_escaped_details_are_reader_clean(tmp_path: Path) -> None:
    markdown_path = tmp_path / "book.md"
    markdown_path.write_text(
        "# Chapter\n\n<details>\n<summary>natural_image</summary>\n"
        "Meaningful geometric description.\n</details>\n\n"
        "<details><summary>Proof sketch</summary>Argument body.</details>\n",
        encoding="utf-8",
    )
    parsed = parse_source_input(source_path=markdown_path, source_id="local:test-book")
    visible_text = " ".join(
        str(item.get("text") or "") for item in parsed["document"]["blocks"]
    )
    assert "Meaningful geometric description" not in visible_text
    assert "natural_image" not in visible_text

    proof_block = next(
        item for item in parsed["document"]["blocks"]
        if "Proof sketch" in str(item.get("text") or "")
    )
    proof_rendered = _render_html_fragment(proof_block["html"], rendered_links=[])
    assert "Proof sketch" in proof_rendered
    assert "Argument body" in proof_rendered
    assert "details" not in proof_rendered and "summary" not in proof_rendered


def test_source_adapter_requests_complete_ar5iv_document() -> None:
    calls = {}

    def parse(**kwargs):
        calls.update(kwargs)
        return {"ok": True, "data": _current_parsed("arXiv:1")}

    bundle = load_source_bundle(
        "arXiv:1",
        refresh=False,
        recache=True,
        parse=parse,
        metadata_getter=lambda *args, **kwargs: {"ok": True, "data": {"title": "T"}},
        references_getter=lambda *args, **kwargs: {"ok": True, "data": []},
        citers_getter=lambda *args, **kwargs: {"ok": True, "data": []},
    )
    assert bundle.paper_id == "arXiv:1"
    assert calls == {
        "source": "ar5iv",
        "paper_id": "arXiv:1",
        "include_document": True,
        "refresh": False,
        "recache": True,
        "document_kind": "auto",
    }


@pytest.mark.parametrize("source_id", ["local:test-book", "isbn:9780521850827"])
def test_source_adapter_reparses_local_provenance_through_arc_paper(
    source_id: str, tmp_path: Path,
) -> None:
    calls: list[tuple[str, dict]] = []
    markdown = tmp_path / "book.md"
    markdown.write_text("# Chapter\n\nBody\n", encoding="utf-8")
    document = {
        "source": {"format": "markdown", "path": str(markdown)},
        "blocks": [{"block_id": "b1", "type": "text", "text": "x"}],
        "integrity": {"status": "complete"},
    }

    def get_cached(requested_id, **kwargs):
        calls.append(("cached", {"source_id": requested_id, **kwargs}))
        return {"ok": True, "data": _current_parsed(source_id, document=document)}

    def parse(**kwargs):
        calls.append(("parse", kwargs))
        return {"ok": True, "data": _current_parsed(source_id, document=document)}

    bundle = load_source_bundle(
        source_id,
        parsed_getter=get_cached,
        parse=parse,
        metadata_getter=lambda *args, **kwargs: pytest.fail("metadata must remain local"),
        references_getter=lambda *args, **kwargs: pytest.fail("references must remain local"),
        citers_getter=lambda *args, **kwargs: pytest.fail("citers must remain local"),
    )

    assert calls[0] == ("cached", {"source_id": source_id, "include_document": True})
    assert calls[1][0] == "parse"
    assert calls[1][1] == {
        "source": "markdown", "source_id": source_id, "markdown_path": str(markdown),
        "include_document": True, "refresh": False, "recache": False,
        "document_kind": "auto",
    }
    assert bundle.paper_id == source_id
    assert bundle.metadata == {"_arc_companion_metadata_source": "unavailable"}
    assert bundle.references == []
    assert bundle.citers == []
    assert bundle.related_evidence == ()


@pytest.mark.parametrize("source_id", ["doi:10.1000/example", "inspire:12345"])
def test_resolvable_colon_ids_are_not_mistaken_for_local_only_sources(source_id: str) -> None:
    parse_calls = []

    def parse(**kwargs):
        parse_calls.append(kwargs)
        return {"ok": True, "data": _current_parsed(source_id)}

    bundle = load_source_bundle(
        source_id,
        parsed_getter=lambda *args, **kwargs: {"ok": False, "error": {"message": "not cached"}},
        parse=parse,
        metadata_getter=lambda *args, **kwargs: {"ok": True, "data": {"title": "Remote"}},
        references_getter=lambda *args, **kwargs: {"ok": True, "data": []},
        citers_getter=lambda *args, **kwargs: {"ok": True, "data": []},
    )

    assert bundle.metadata["title"] == "Remote"
    assert parse_calls[0]["paper_id"] == source_id


def test_any_identifier_revalidates_an_existing_rich_cache_entry() -> None:
    def cached(*args, **kwargs):
        return {"ok": True, "data": _current_parsed(
            "doi:10.1000/cached", document={
                "metadata": {"authors": ["A. Author"], "year": 2024},
                "toc": [{"title": "Cached document title"}],
                "blocks": [{"block_id": "b1", "type": "text", "text": "x"}],
                "integrity": {"status": "complete"},
            },
        )}

    parse_calls = []
    def parse(**kwargs):
        parse_calls.append(kwargs)
        return cached()

    bundle = load_source_bundle(
        "doi:10.1000/cached", parsed_getter=cached,
        parse=parse,
        metadata_getter=lambda *args, **kwargs: pytest.fail("rich cache must avoid metadata fetch"),
        references_getter=lambda *args, **kwargs: pytest.fail("rich cache must avoid reference fetch"),
        citers_getter=lambda *args, **kwargs: pytest.fail("rich cache must avoid citer fetch"),
    )

    assert bundle.metadata["title"] == "Cached document title"
    assert bundle.metadata["authors"] == ["A. Author"]
    assert bundle.metadata["_arc_companion_metadata_source"] == {
        "authors": "document.metadata",
        "year": "document.metadata",
        "title": "document.toc",
    }
    assert parse_calls[0]["document_kind"] == "auto"


@pytest.mark.parametrize(
    ("failed_getter", "message"),
    [
        ("metadata", "Unable to load seed metadata: metadata offline"),
        ("references", "Unable to load seed references: references offline"),
    ],
)
def test_source_adapter_surfaces_required_seed_evidence_failures(
    failed_getter: str, message: str
) -> None:
    def parse(**kwargs):
        return {"ok": True, "data": _current_parsed("arXiv:1")}

    metadata = {"ok": True, "data": {"title": "T"}}
    references = {"ok": True, "data": []}
    if failed_getter == "metadata":
        metadata = {"ok": False, "error": {"message": "metadata offline"}}
    else:
        references = {"ok": False, "error": {"message": "references offline"}}

    with pytest.raises(SourceError, match=message):
        load_source_bundle(
            "arXiv:1",
            parse=parse,
            metadata_getter=lambda *args, **kwargs: metadata,
            references_getter=lambda *args, **kwargs: references,
            citers_getter=lambda *args, **kwargs: {"ok": True, "data": []},
        )


def test_source_adapter_records_optional_citer_failure_as_warning() -> None:
    def parse(**kwargs):
        return {"ok": True, "data": _current_parsed("arXiv:1")}

    bundle = load_source_bundle(
        "arXiv:1",
        parse=parse,
        metadata_getter=lambda *args, **kwargs: {"ok": True, "data": {"title": "T"}},
        references_getter=lambda *args, **kwargs: {"ok": True, "data": []},
        citers_getter=lambda *args, **kwargs: {
            "ok": False,
            "error": {"message": "INSPIRE citer endpoint unavailable"},
        },
    )

    assert bundle.citers == []
    assert bundle.diagnostics == ({
        "severity": "warning",
        "code": "citer_context_unavailable",
        "source": "arc-paper",
        "message": "Unable to load optional seed citers: INSPIRE citer endpoint unavailable",
    },)


def test_source_adapter_keeps_related_metadata_without_parsing_related_full_text() -> None:
    calls: list[dict] = []

    def parse(**kwargs):
        paper_id = kwargs["paper_id"]
        calls.append(dict(kwargs))
        data = {**_current_parsed(paper_id), "source_hash": "d" * 64, "sections": [{
            "section_id": f"{paper_id}-s1", "title": "Result", "text": "field theory"
        }]}
        data["document"] = {
            "blocks": [{"block_id": f"{paper_id}-b1", "type": "text", "text": "field theory"}],
            "integrity": {"status": "complete"},
        }
        return {"ok": True, "data": data}

    references = [
        {"arxiv_id": f"0801.{index:04d}", "title": f"Prior {index}",
         "abstract": f"Prior abstract {index}.", "citation_count": index}
        for index in range(9)
    ]
    citers = [
        {"arxiv_id": f"2501.{index:04d}", "title": f"Later {index}",
         "abstract": f"Later abstract {index}.", "citation_count": index}
        for index in range(9)
    ]
    bundle = load_source_bundle(
        "arXiv:1",
        parse=parse,
        metadata_getter=lambda *args, **kwargs: {"ok": True, "data": {"title": "T"}},
        references_getter=lambda *args, **kwargs: {"ok": True, "data": references},
        citers_getter=lambda *args, **kwargs: {"ok": True, "data": citers},
    )

    assert calls[0]["paper_id"] == "arXiv:1"
    assert calls[0]["include_document"] is True
    assert len(calls) == 1
    assert len(bundle.related_evidence) == 18
    assert {item["paper_id"] for item in bundle.related_evidence} >= {
        "0801.0000", "0801.0008", "2501.0000", "2501.0008",
    }
    assert {item["relation"] for item in bundle.related_evidence} == {"prior", "later"}
    assert all(item["evidence_level"] == "abstract_only" for item in bundle.related_evidence)
    assert all(item["source_descriptor"]["provider"] == "arc-paper" for item in bundle.related_evidence)
    assert all(item["source_descriptor"]["content_sha256"] for item in bundle.related_evidence)
    assert all(not item["blocks"] for item in bundle.related_evidence)


@pytest.mark.parametrize(
    "integrity",
    [
        {"status": "partial"},
        {"complete": False},
        {"status": "complete", "blocking_issues": ["missing image"]},
    ],
)
def test_incomplete_documents_are_rejected(integrity) -> None:
    with pytest.raises(SourceError):
        validate_complete_document({"blocks": [{"block_id": "b"}], "integrity": integrity})


def test_source_adapter_rejects_response_without_current_structure_contract() -> None:
    with pytest.raises(SourceError, match="current structure contract"):
        load_source_bundle(
            "arXiv:1",
            parsed_getter=lambda *args, **kwargs: {"ok": False, "error": {"message": "miss"}},
            parse=lambda **kwargs: {"ok": True, "data": {
                "paper_id": "arXiv:1", "document": {
                    "blocks": [{"block_id": "b1", "text": "x"}],
                    "integrity": {"status": "complete"},
                },
            }},
            metadata_getter=lambda *args, **kwargs: pytest.fail("must validate before metadata"),
            references_getter=lambda *args, **kwargs: pytest.fail("must validate before references"),
            citers_getter=lambda *args, **kwargs: pytest.fail("must validate before citers"),
        )


def test_source_adapter_rejects_changed_paired_pdf_hash(tmp_path: Path) -> None:
    markdown = tmp_path / "book.md"
    pdf = tmp_path / "book.pdf"
    markdown.write_text("# One\n", encoding="utf-8")
    pdf.write_bytes(b"changed pdf")
    document = {
        "source": {
            "format": "markdown", "path": str(markdown), "pdf_path": str(pdf),
            "pdf_sha256": "0" * 64,
        },
        "blocks": [{"block_id": "b1", "text": "x"}],
        "integrity": {"status": "complete"},
    }
    parsed = _current_parsed("local:book", document=document)
    parsed["structure"]["reconciliation"] = {
        "schema_version": "arc.paper.reconciliation.v1", "status": "complete",
        "proof_sha256": "proof", "source_hash": parsed["source_hash"],
        "pdf_sha256": "0" * 64,
        "section_coverage": {"status": "complete"},
        "block_coverage": {"status": "complete"},
    }
    with pytest.raises(SourceError, match="PDF hash changed"):
        load_source_bundle(
            "local:book",
            parsed_getter=lambda *args, **kwargs: {"ok": True, "data": parsed},
            parse=lambda **kwargs: {"ok": True, "data": parsed},
            metadata_getter=lambda *args, **kwargs: pytest.fail("local only"),
            references_getter=lambda *args, **kwargs: pytest.fail("local only"),
            citers_getter=lambda *args, **kwargs: pytest.fail("local only"),
        )


def test_cli_prints_default_language_notice_without_pausing(tmp_path: Path, monkeypatch, capsys) -> None:
    captured = {}

    def fake_build(options):
        captured["options"] = options
        return {"ok": True, "data": {"status": "complete"}, "errors": [], "meta": {"notice": "n"}}

    monkeypatch.setattr(cli, "build_companion", fake_build)
    code = cli.main(["build", "arXiv:1", "--project-dir", str(tmp_path), "--json"])
    streams = capsys.readouterr()
    assert code == 0
    assert "默认使用中文" in streams.err
    assert captured["options"].annotation_language == "zh-CN"
    assert captured["options"].language_was_defaulted is True
    assert captured["options"].workers == 24
    assert json.loads(streams.out)["ok"] is True


def test_cli_explicit_language_has_no_notice(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "build_companion", lambda options: {"ok": True, "data": {"status": "complete"}, "meta": {}})
    assert cli.main(["build", "arXiv:1", "--project-dir", str(tmp_path), "--annotation-language", "en"]) == 0
    assert "默认使用中文" not in capsys.readouterr().err


def test_cli_passes_sampled_source_language_without_detecting_it(
    tmp_path: Path, monkeypatch,
) -> None:
    captured = {}

    def fake_build(options):
        captured["options"] = options
        return {"ok": True, "data": {"status": "complete"}, "meta": {}}

    monkeypatch.setattr(cli, "build_companion", fake_build)
    assert cli.main([
        "build", "local:paper", "--project-dir", str(tmp_path),
        "--source-language", "ja-JP", "--annotation-language", "zh-CN",
    ]) == 0
    assert captured["options"].source_language == "ja-JP"


def test_cli_is_controller_only_and_keeps_internet_enabled(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    def fake_build(options):
        captured["options"] = options
        return {"ok": True, "data": {"status": "complete"}, "meta": {}}

    monkeypatch.setattr(cli, "build_companion", fake_build)
    assert cli.main([
        "build",
        "local:david-tong-qft-notes",
        "--project-dir",
        str(tmp_path),
        "--annotation-language",
        "zh-CN",
    ]) == 0
    assert not hasattr(captured["options"], "allow_mcp")
    assert captured["options"].allow_internet is True


def test_cli_passes_explicit_managed_child_budget(
    tmp_path: Path, monkeypatch,
) -> None:
    captured = {}

    def fake_build(options):
        captured["options"] = options
        return {"ok": True, "data": {"status": "complete"}, "meta": {}}

    monkeypatch.setattr(cli, "build_companion", fake_build)
    assert cli.main([
        "build", "local:test", "--project-dir", str(tmp_path),
        "--arc-paper-child-llm-max-calls", "3",
        "--arc-paper-child-llm-max-tokens", "4000",
        "--arc-paper-child-llm-output-reserve-tokens", "200",
    ]) == 0
    options = captured["options"]
    assert options.arc_paper_child_llm_max_calls == 3
    assert options.arc_paper_child_llm_max_tokens == 4000
    assert options.arc_paper_child_llm_output_reserve_tokens == 200


def test_cli_returns_nonzero_for_error_envelope(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "build_companion",
        lambda _options: {
            "ok": False,
            "data": None,
            "error": {"code": "build_failed", "message": "boom"},
            "meta": {},
        },
    )

    assert cli.main([
        "build",
        "local:test",
        "--project-dir",
        str(tmp_path),
        "--annotation-language",
        "en",
        "--json",
    ]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_cli_json_wraps_dispatch_exception(monkeypatch, capsys) -> None:
    def fail(_args):
        raise RuntimeError("pipeline unavailable")

    monkeypatch.setattr(cli, "_dispatch", fail)

    assert cli.main(["status", "--project-dir", "/tmp/project", "--json"]) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["error"] == {
        "code": "command_failed",
        "message": "pipeline unavailable",
        "type": "RuntimeError",
    }


def test_cli_render_web_publishes_reader(tmp_path: Path, monkeypatch, capsys) -> None:
    module = types.ModuleType("arc_companion.render")
    captured: dict[str, Path] = {}

    def render_content(project_dir: Path, *, format: str) -> dict:
        captured["project_dir"] = project_dir
        captured["format"] = format
        return {"ok": True, "data": {
            "mode": "render_only",
            "output_html": str(project_dir / "reader" / "index.html"),
            "provider_calls": 0,
        }, "errors": [], "meta": {}}

    module.render_content = render_content  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "arc_companion.render", module)

    assert cli.main(["render-web", "--project-dir", str(tmp_path)]) == 0
    assert captured["project_dir"] == tmp_path
    assert captured["format"] == "web"
    assert capsys.readouterr().out.strip() == str(tmp_path / "reader" / "index.html")


def test_non_json_build_prefers_pdf_over_web_output(capsys) -> None:
    cli._emit(
        {
            "ok": True,
            "data": {"output_pdf": "/project/paper.pdf", "output_html": "/project/reader/index.html"},
            "meta": {},
        },
        json_output=False,
    )
    assert capsys.readouterr().out.strip() == "/project/paper.pdf"


def test_non_json_build_prefers_run_root_pdf_delivery(capsys) -> None:
    cli._emit(
        {
            "ok": True,
            "data": {
                "output_run_pdf": "/project/paper.pdf",
                "output_pdf": "/project/.arc-companion/renders/rev/paper.pdf",
            },
            "meta": {},
        },
        json_output=False,
    )
    assert capsys.readouterr().out.strip() == "/project/paper.pdf"


def test_public_web_contract_delegates_lazily(tmp_path: Path, monkeypatch) -> None:
    module = types.ModuleType("arc_companion.web")
    module.build_reader_snapshot = lambda project_dir, **kwargs: {  # type: ignore[attr-defined]
        "project_dir": project_dir, **kwargs
    }
    module.publish_reader = lambda project_dir, **kwargs: {  # type: ignore[attr-defined]
        "project_dir": project_dir, **kwargs
    }
    module.validate_reader_project = lambda project_dir, **kwargs: {  # type: ignore[attr-defined]
        "project_dir": project_dir, **kwargs
    }
    monkeypatch.setitem(sys.modules, "arc_companion.web", module)

    assert arc_companion.build_reader_snapshot(tmp_path, state={"status": "complete"}) == {
        "project_dir": tmp_path, "state": {"status": "complete"}, "final_overrides": None,
    }
    assert arc_companion.publish_reader(tmp_path, snapshot={"chapters": []})["snapshot"] == {
        "chapters": []
    }
    assert arc_companion.validate_reader_project(tmp_path)["project_dir"] == tmp_path


def test_cli_passes_chapter_build_options(tmp_path: Path, monkeypatch) -> None:
    captured = {}
    legacy = tmp_path / "legacy"
    legacy.mkdir()

    def fake_build(options):
        captured["options"] = options
        return {"ok": True, "data": {"status": "preview_ready"}, "meta": {}}

    monkeypatch.setattr(cli, "build_companion", fake_build)
    assert cli.main([
        "build",
        "local:david-tong-qft-notes",
        "--project-dir",
        str(tmp_path),
        "--stop-after-first-chapter",
        "--skip-translation",
        "--document-kind",
        "book",
        "--idle-timeout-seconds",
        "90",
        "--recovery-policy",
        "manual",
        "--max-auto-replacements",
        "5",
        "--regenerate-segment",
        "commentary:ch-0001.seg-0002",
        "--regenerate-commentary",
        "--legacy-checkpoint",
        str(legacy),
    ]) == 0
    options = captured["options"]
    assert options.stop_after_first_chapter is True
    assert options.skip_translation is True
    assert options.document_kind == "book"
    assert options.idle_timeout_seconds == 90
    assert options.recovery_policy == "manual"
    assert options.max_auto_replacements == 5
    assert options.regenerate_segments == ("commentary:ch-0001.seg-0002",)
    assert options.regenerate_commentary is True
    assert options.legacy_checkpoint == legacy.resolve()


def test_cli_resume_requires_duplicate_charge_confirmation(tmp_path: Path, capsys) -> None:
    (tmp_path / "state.json").write_text('{"status":"needs_supervision"}', encoding="utf-8")
    assert cli.main([
        "resume", "--project-dir", str(tmp_path), "--action", "restart-generation", "--json",
    ]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "needs_supervision"
    assert result["error"]["code"] == "duplicate_charge_confirmation_required"


def test_cli_defaults_build_recovery_policy_and_resume_action_to_auto(
    tmp_path: Path, monkeypatch,
) -> None:
    import arc_companion.pipeline as pipeline_module

    captured: dict[str, object] = {}

    def fake_build(options):
        captured["recovery_policy"] = options.recovery_policy
        captured["max_auto_replacements"] = options.max_auto_replacements
        return {"ok": True, "data": {"status": "complete"}, "meta": {}}

    def fake_resume(project_dir, *, action, confirm_possible_duplicate_charge=False):
        captured["resume_project_dir"] = project_dir
        captured["resume_action"] = action
        captured["resume_confirmation"] = confirm_possible_duplicate_charge
        return {"ok": True, "data": {"status": "complete"}, "meta": {}}

    monkeypatch.setattr(cli, "build_companion", fake_build)
    monkeypatch.setattr(pipeline_module, "resume_companion", fake_resume)

    assert cli.main([
        "build", "local:auto-recovery", "--project-dir", str(tmp_path),
        "--annotation-language", "en", "--json",
    ]) == 0
    assert cli.main([
        "resume", "--project-dir", str(tmp_path), "--json",
    ]) == 0

    assert captured["recovery_policy"] == "auto"
    assert captured["max_auto_replacements"] == 3
    assert captured["resume_project_dir"] == tmp_path
    assert captured["resume_action"] == "auto"
    assert captured["resume_confirmation"] is False


def test_cli_prints_structured_evidence_warnings(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "build_companion",
        lambda options: {
            "ok": True,
            "data": {"status": "complete"},
            "meta": {"diagnostics": [{
                "severity": "warning",
                "code": "citer_context_unavailable",
                "message": "Optional citer context is unavailable",
            }]},
        },
    )

    assert cli.main([
        "build",
        "arXiv:1",
        "--project-dir",
        str(tmp_path),
        "--annotation-language",
        "en",
        "--json",
    ]) == 0
    streams = capsys.readouterr()
    assert "WARNING: Optional citer context is unavailable" in streams.err
    assert json.loads(streams.out)["meta"]["diagnostics"][0]["code"] == "citer_context_unavailable"


def test_companion_docs_describe_chaptered_stateful_cli_contract() -> None:
    manual = (ROOT / "plugins/arc/skills/arc/manuals/arc-companion.md").read_text(encoding="utf-8")
    workflow = (ROOT / "plugins/arc/skills/arc/workflows/companion.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    manual_text = " ".join(manual.split())
    workflow_text = " ".join(workflow.split())
    readme_text = " ".join(readme.split())

    assert "paper, lecture note, or book" in manual_text
    assert "rich source plus its paired PDF" in manual_text
    assert "PDF is authoritative" in workflow_text
    assert "CLI-only" in manual_text
    assert "There is no entry cap" in manual_text
    assert "whole-document glossary" in manual_text
    assert "Project terms to segments" in manual_text
    assert "chapter-glossary.json" not in manual_text
    assert "glossary setup turns" not in manual_text
    assert "searches, reads, writes, and cites sources within one generation turn" in manual_text
    assert "`--no-internet`" in manual_text
    assert "title linked to its URL and the locator visible" in manual_text
    assert "`first_chapter_ready`" in manual_text
    assert "`needs_supervision`" in manual_text
    assert "`--stop-after-first-chapter`" in manual_text
    assert "--action resume-native" in manual_text
    assert "--recovery-policy auto|manual" in manual_text
    assert "arc-companion resume --project-dir <dir> --json" in manual_text
    assert "--action restart-generation" in manual_text
    assert "--confirm-possible-duplicate-charge" in manual_text
    assert "after every 30 minutes" in manual_text
    assert "`arc-jobs watch <job-id> --until-review --json`" in readme_text
    assert "run-root delivery PDF" in manual_text
    assert "resolved `--project-dir`" in manual_text
    assert "run-root delivery PDF" in workflow_text
    assert "resolved `<project-dir>`" in workflow_text
    assert "run-root delivery PDF" in readme_text
    assert "chapter-aware" in readme_text
    assert "A real Index becomes the complete global glossary" in readme_text
    assert "stateful guide session" in workflow_text
    assert "strictly in segment order" in workflow_text
    assert "`--stop-after-first-chapter`" in workflow_text
    assert "remain supervised rather than being masked by replacement" in workflow_text
    assert "after each 30 minutes" in workflow_text
    assert "MCP is not part of this workflow" in workflow_text
    assert "beginning, middle, and end" in workflow_text
    assert "`EN_US`, `EN_UK`, and `en-GB` are `en`" in workflow_text
    assert "simplified and traditional Chinese" in workflow_text
    assert "Mixed-language or uncertain source text" in workflow_text
    assert "`translation_mode=skip`" in workflow_text
    assert "`--skip-translation`" in manual_text
    assert "The CLI deliberately performs no automatic language detection" in manual_text
    assert "translation session, provider call, ledger, checkpoint, review overlay" in manual_text
    assert "translations=None" in manual_text
    assert "The CLI itself does not perform language detection" in readme_text


def test_package_includes_only_validated_deliverables(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    tex = tmp_path / "paper.tex"
    pdf.write_bytes(b"%PDF fixture")
    tex.write_text("tex", encoding="utf-8")
    (tmp_path / "source-manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "validation.json").write_text('{"ok":true}', encoding="utf-8")
    (tmp_path / "state.json").write_text(json.dumps({
        "status": "complete",
        "paper_id": "arXiv:1",
        "fingerprint": "abc",
        "output_pdf": str(pdf),
        "output_tex": str(tex),
    }), encoding="utf-8")
    result = package_project(tmp_path)
    assert result["ok"]
    assert Path(result["data"]["archive_path"]).is_file()


def test_renderer_accepts_current_arc_paper_rich_contract(tmp_path: Path) -> None:
    from arc_paper.parse.source import parse_source_input

    parsed = parse_source_input(
        html_text="""
        <article class="ltx_document">
          <h1 class="ltx_title_document">Rich title</h1>
          <div class="ltx_role_affiliation">Physics Institute</div>
          <div class="ltx_abstract">Rich abstract</div>
          <section id="S1"><h2>Result</h2>
            <table id="S1.E1" class="ltx_equation"><tr><td><math alttext="x=y"></math></td>
              <td class="ltx_eqn_eqno"><span class="ltx_tag_equation">(11)</span></td></tr></table>
            <figure id="S1.T1" class="ltx_table"><figcaption><span class="ltx_tag_table">Table 3:</span> Values</figcaption>
              <table><tr><td rowspan="2">mass</td><td>one</td></tr><tr><td>two</td></tr></table></figure>
          </section>
          <ul class="ltx_bibliography"><li id="bib.one" class="ltx_bibitem"><span class="ltx_tag_bibitem">[1]</span> A reference.</li></ul>
        </article>
        """,
        source_id="rich",
    )
    document = parsed["document"]
    blocks = document["blocks"]
    segments = [{
        "segment_id": "all",
        "title": "All",
        "start_block_id": blocks[0]["block_id"],
        "end_block_id": blocks[-1]["block_id"],
        "block_ids": [item["block_id"] for item in blocks],
    }]
    tex, _ = render_companion_tex(
        document,
        segments,
        {"all": {"commentary": "Companion"}},
        output_dir=tmp_path,
        language="en",
    )
    assert "x=y" in tex
    assert "['x=y']" not in tex
    assert r"\tag{11}" in tex
    assert "Table 3: Values" in tex
    assert r"\multirow{2}{*}{mass} & one" in tex
    assert "\n & two" in tex
    assert "Physics Institute" in tex
    assert "Rich abstract" in tex


def test_table_renderer_preserves_sparse_positions_rowspan_and_colspan() -> None:
    entity = {
        "id": "table",
        "column_count": 3,
        "rows": [
            [
                {"text": "A", "row": 0, "column": 0, "rowspan": 2, "colspan": 1},
                {"text": "wide", "row": 0, "column": 1, "rowspan": 1, "colspan": 2},
            ],
            [
                {"text": "B", "row": 1, "column": 1, "rowspan": 1, "colspan": 1},
                {"text": "C", "row": 1, "column": 2, "rowspan": 1, "colspan": 1},
            ],
        ],
        "grid": [
            [
                {"text": "A", "source_row": 0, "source_column": 0},
                {"text": "wide", "source_row": 0, "source_column": 1},
                {"text": "wide", "source_row": 0, "source_column": 1},
            ],
            [
                {"text": "A", "source_row": 0, "source_column": 0},
                {"text": "B", "source_row": 1, "source_column": 1},
                {"text": "C", "source_row": 1, "source_column": 2},
            ],
        ],
    }
    tex = _render_table(entity)
    assert r"\begin{longtable}{lll}" in tex
    assert r"\multirow{2}{*}{A} & \multicolumn{2}{l}{wide}" in tex
    assert "\n & B & C" in tex

    with pytest.raises(LatexError, match="expected exactly 2"):
        _render_table({**entity, "column_count": 2})


def test_table_renderer_can_reconstruct_spans_from_canonical_grid_only() -> None:
    entity = {
        "grid": [
            [
                {"text": "A", "source_row": 0, "source_column": 0},
                {"text": "wide", "source_row": 0, "source_column": 1},
                {"text": "wide", "source_row": 0, "source_column": 1},
            ],
            [
                {"text": "A", "source_row": 0, "source_column": 0},
                {"text": "B", "source_row": 1, "source_column": 1},
                {"text": "C", "source_row": 1, "source_column": 2},
            ],
        ]
    }
    tex = _render_table(entity)
    assert r"\multirow{2}{*}{A}" in tex
    assert r"\multicolumn{2}{l}{wide}" in tex
    assert "\n & B & C" in tex


def test_heading_and_nested_list_fields_render_without_falling_back_to_prose(tmp_path: Path) -> None:
    document = {
        "blocks": [
            {"block_id": "h", "kind": "heading", "heading_level": 2, "title": "Detailed result"},
            {
                "block_id": "l",
                "kind": "prose",
                "list_kind": "ordered",
                "list_items": [
                    {"text": "First", "items": [{"text": "Nested"}]},
                    {"content": "Second"},
                ],
            },
        ],
        "integrity": {"status": "complete"},
    }
    segments = [{
        "segment_id": "all",
        "start_block_id": "h",
        "end_block_id": "l",
        "block_ids": ["h", "l"],
    }]
    tex, _ = render_companion_tex(
        document,
        segments,
        {"all": {"commentary": "note"}},
        output_dir=tmp_path,
        language="en",
    )
    assert r"\subsection*{Detailed result}" in tex
    assert r"\addcontentsline{toc}{subsection}{Detailed result}" in tex
    assert r"\begin{enumerate}" in tex
    assert r"\item First" in tex
    assert r"\begin{itemize}" in tex
    assert r"\item Nested" in tex
    assert r"\item Second" in tex


def test_html_ordered_list_with_years_keeps_automatic_numbering() -> None:
    tex = _render_html_fragment(
        "<ol><li>2020 result</li><li>2021 follow-up</li></ol>",
        rendered_links=[],
    )

    assert r"\begin{enumerate}" in tex
    assert r"\item 2020 result" in tex
    assert r"\item 2021 follow-up" in tex
    assert r"\begin{description}" not in tex


def test_html_renderer_preserves_inline_structure_without_front_or_reference_duplication(tmp_path: Path) -> None:
    document = {
        "front_matter": {"title": "One title", "authors": ["A. Author"]},
        "blocks": [
            {
                "block_id": "title",
                "kind": "heading",
                "text": "One title",
                "title": "One title",
                "html": '<h1 id="title">One title</h1>',
                "section_id": "",
            },
            {
                "block_id": "p1",
                "source_id": "p1",
                "section_id": "S1",
                "kind": "prose",
                "text": "An important result x_i cites [1] and site.",
                "html": (
                    '<p id="p1">An <em>important</em> result '
                    '<math alttext="x_i"></math> cites <a href="#bib1">[1]</a> '
                    'and <a href="https://example.test/a_b?x=1&amp;y=2">site</a>.</p>'
                ),
            },
            {
                "block_id": "bib1",
                "source_id": "bib1",
                "kind": "bibliography",
                "text": "[1] Reference text.",
                "html": '<li id="bib1"><span class="ltx_tag_bibitem">[1]</span> Reference text.</li>',
            },
        ],
        "bibliography": [{
            "id": "bib1",
            "label": "[1]",
            "text": "[1] Reference text.",
            "html": '<li id="bib1"><span class="ltx_tag_bibitem">[1]</span> Reference text.</li>',
        }],
        "links": [
            {"href": "#bib1", "target_id": "bib1", "text": "[1]"},
            {"href": "https://example.test/a_b?x=1&y=2", "target_id": "", "text": "site"},
        ],
        "integrity": {"status": "complete"},
    }
    segments = [{
        "segment_id": "all",
        "start_block_id": "title",
        "end_block_id": "bib1",
        "block_ids": ["title", "p1", "bib1"],
    }]
    annotation = {
        "commentary": "note",
        "commentary_sources": [{
            "title": "Companion source",
            "url": "https://example.test/source_a?view=full&lang=en",
            "locator": "Section 2 / p. 4",
        }],
    }
    tex, manifest = render_companion_tex(
        document,
        segments,
        {"all": annotation},
        output_dir=tmp_path,
        language="en",
    )
    assert tex.count("One title") == 1
    assert tex.count("Reference text") == 1
    assert r"\emph{important}" in tex
    assert r"\(x_i\)" in tex
    assert r"\hyperref[bib1]{[1]}" in tex
    assert r"\href{https://example.test/a\_b?x=1\&y=2}{site}" in tex
    assert (
        r"\href{https://example.test/source\_a?view=full\&lang=en}{Companion source}"
        in tex
    )
    assert "Section 2 / p. 4" in tex
    assert manifest["companion_layers"]["annotation_sources_by_segment"] == {
        "all": {"commentary_sources": annotation["commentary_sources"]}
    }
    assert validate_tex_fidelity(tex, document, manifest) == []


def test_structural_combined_creator_block_renders_author_and_affiliation_once(tmp_path: Path) -> None:
    document = {
        "front_matter": {
            "title": "Structured Title",
            "authors": ["An Author"],
            "affiliations": ["An Institute"],
            "block_ids": {
                "title": ["title"],
                "authors": ["creator"],
                "affiliations": ["creator"],
            },
        },
        "blocks": [
            {
                "block_id": "title", "kind": "heading", "text": "Structured Title",
                "source_role": "front_matter_title",
            },
            {
                "block_id": "creator", "kind": "prose", "text": "An Author An Institute",
                "source_role": "front_matter",
                "front_matter_roles": ["front_matter_authors", "front_matter_affiliations"],
            },
            {"block_id": "body", "kind": "prose", "text": "Body text."},
        ],
        "integrity": {"status": "complete"},
    }
    segments = [{"segment_id": "body", "block_ids": ["body"]}]

    tex, _ = render_companion_tex(
        document,
        segments,
        {"body": {"commentary": "Note."}},
        output_dir=tmp_path,
        language="en",
    )

    assert tex.count("Structured Title") == 1
    assert tex.count("An Author") == 1
    assert tex.count("An Institute") == 1
    assert "An Author An Institute" not in tex


def test_source_only_toc_acknowledgments_and_references_render_once_with_toc_structure(tmp_path: Path) -> None:
    document = {
        "front_matter": {},
        "blocks": [
            {
                "block_id": "toc-title", "kind": "heading", "level": 6,
                "text": "Contents", "title": "Contents", "source_role": "table_of_contents",
                "html": '<h6 class="ltx_title_contents">Contents</h6>',
            },
            {
                "block_id": "toc-list", "kind": "list", "source_role": "table_of_contents",
                "text": "1 Main 1.1 Detail 2 Other", "list_kind": "ordered", "items": [],
                "html": (
                    '<ol class="ltx_toclist"><li><a href="#S1">1 Main</a>'
                    '<ol><li><a href="#S1.SS1">1.1 Detail</a></li></ol>'
                    '</li><li>2 Other</li></ol>'
                ),
            },
            {
                "block_id": "S1", "kind": "heading", "level": 2, "section_id": "S1",
                "text": "1 Main", "title": "Main", "html": '<h2 id="S1">1 Main</h2>',
            },
            {
                "block_id": "body", "source_id": "S1.SS1", "kind": "prose",
                "section_id": "S1", "text": "Body text.",
            },
            {
                "block_id": "ack-title", "kind": "heading", "section_id": "Sx",
                "text": "Acknowledgments", "title": "Acknowledgments", "source_role": "acknowledgments",
            },
            {
                "block_id": "ack-body", "kind": "prose", "section_id": "Sx",
                "text": "We thank our colleagues.", "source_role": "acknowledgments",
            },
            {
                "block_id": "refs-title", "kind": "heading", "section_id": "bib",
                "text": "References", "title": "References", "source_role": "references",
            },
            {
                "block_id": "bib1", "kind": "bibliography", "section_id": "bib",
                "text": "[1] Reference work.", "source_role": "references",
            },
        ],
        "bibliography": [{"id": "bib1", "label": "[1]", "text": "[1] Reference work."}],
        "links": [
            {"href": "#S1", "target_id": "S1", "text": "1 Main"},
            {"href": "#S1.SS1", "target_id": "S1.SS1", "text": "1.1 Detail"},
        ],
        "integrity": {"status": "complete"},
    }
    segments = [{
        "segment_id": "body", "start_block_id": "S1", "end_block_id": "body",
        "block_ids": ["S1", "body"], "title": "Main",
    }]

    tex, manifest = render_companion_tex(
        document,
        segments,
        {"body": {"commentary": "Body note", "explanation": "Body note"}},
        translations={"body": {"blocks": [
            {"block_id": "S1", "text": "主节"}, {"block_id": "body", "text": "正文。"},
        ]}},
        output_dir=tmp_path,
        language="zh-CN",
    )

    assert tex.count(r"\paragraph*{Contents}") == 1
    assert tex.count("We thank our colleagues") == 1
    assert tex.count("Reference work") == 1
    assert tex.count(r"\begin{description}") == 2
    assert r"\begin{enumerate}" not in tex
    assert r"\item[] \hyperref[S1]{1 Main}" in tex
    assert r"\item[] \hyperref[S1.SS1]{1.1 Detail}" in tex
    assert r"\item[] 2 Other" in tex
    assert manifest["rendered_links"] == manifest["expected_links"]
    assert manifest["companion_layers"]["semantic_segment_ids"] == ["body"]
    assert validate_tex_fidelity(tex, document, manifest) == []

    manifest["rendered_links"].append(dict(manifest["rendered_links"][0]))
    assert "rendered 1 unregistered source link occurrence(s)" in validate_tex_fidelity(
        tex, document, manifest,
    )


def test_multirow_equations_preserve_each_number_and_label(tmp_path: Path) -> None:
    document = {
        "blocks": [{"block_id": "eq", "kind": "equation", "equation_id": "eq"}],
        "equations": [{
            "id": "eq",
            "tex": ["a=b", "c=d"],
            "printed_equation_numbers": ["(4a)", "(4b)"],
            "labels": ["eq:4a", "eq:4b"],
        }],
        "integrity": {"status": "complete"},
    }
    tex, manifest = render_companion_tex(
        document,
        [{"segment_id": "all", "start_block_id": "eq", "end_block_id": "eq", "block_ids": ["eq"]}],
        {"all": {"commentary": "note"}},
        output_dir=tmp_path,
        language="en",
    )
    assert r"\tag{4a}" in tex and r"\tag{4b}" in tex
    assert r"\label{eq:4a}" in tex and r"\label{eq:4b}" in tex
    assert validate_tex_fidelity(tex, document, manifest) == []


def test_preamble_has_portable_deterministic_cjk_font_fallback(tmp_path: Path) -> None:
    document = {
        "blocks": [{"block_id": "p", "kind": "prose", "text": "中文"}],
        "integrity": {"status": "complete"},
    }
    tex, _ = render_companion_tex(
        document,
        [{"segment_id": "all", "start_block_id": "p", "end_block_id": "p", "block_ids": ["p"]}],
        {"all": {"commentary": "伴读"}},
        output_dir=tmp_path,
        language="zh-CN",
    )
    candidates = ["Noto Serif CJK SC", "Source Han Serif SC", "Source Han Serif CN", "FandolSong-Regular"]
    positions = [tex.index(value) for value in candidates]
    assert positions == sorted(positions)
    assert r"\PackageError{arc-companion}{No supported CJK serif font found}" in tex
