import io
import json

import pytest

from arc_paper import cli


def test_cli_returns_nonzero_for_error_envelope(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "_dispatch",
        lambda _args: {"ok": False, "data": None, "error": {"code": "failed", "message": "boom"}},
    )

    assert cli.main(["get-title", "0911.3380", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_cli_treats_needs_llm_as_successful_handoff(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "_dispatch",
        lambda _args: {"ok": False, "status": "needs_llm", "llm_task": {"prompt": "..."}},
    )

    assert cli.main(["get-title", "0911.3380", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "needs_llm"


def test_cli_json_wraps_dispatch_exception(monkeypatch, capsys) -> None:
    def fail(_args):
        raise RuntimeError("service unavailable")

    monkeypatch.setattr(cli, "_dispatch", fail)

    assert cli.main(["get-title", "0911.3380", "--json"]) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["error"] == {
        "code": "command_failed",
        "message": "service unavailable",
        "type": "RuntimeError",
    }


def test_cli_get_title(monkeypatch, capsys):
    monkeypatch.setattr(cli.service, "get_title", lambda ids, refresh=False: {"ok": True, "data": "Title"})

    assert cli.main(["get-title", "arXiv:0911.3380", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["data"] == "Title"


def test_cli_extract_paper_ids(capsys):
    assert cli.main(["extract-paper-ids", "See", "0911.3380", "and", "doi:10.1234/2512.06790", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["data"] == ["arXiv:0911.3380", "doi:10.1234/2512.06790"]


def test_cli_safe_dir_name(capsys):
    assert cli.main(["safe-dir-name", "0911.3380", "astro-ph/0610514", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["data"] == "0911.3380_x_astro-ph_0610514"


def test_cli_rejects_removed_validate_note_check_command(capsys):
    with pytest.raises(SystemExit):
        cli.main(["validate-note-check", "run-dir", "--json"])

    assert "invalid choice" in capsys.readouterr().err


def test_cli_llm_infer_main_references(monkeypatch, capsys):
    def infer(text, provider="auto", model=None, refresh=False):
        return {
            "ok": True,
            "data": ["arXiv:0911.3380"],
            "errors": [],
            "meta": {"text": text, "provider": provider, "model": model, "refresh": refresh},
        }

    monkeypatch.setattr(cli.service, "llm_infer_main_references", infer)

    assert (
        cli.main(
            [
                "llm-infer-main-references",
                "CMB",
                "trispectrum",
                "--provider",
                "manual",
                "--model",
                "test-model",
                "--refresh",
                "--json",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["data"] == ["arXiv:0911.3380"]
    assert output["meta"]["text"] == "CMB trispectrum"
    assert output["meta"]["provider"] == "manual"
    assert output["meta"]["model"] == "test-model"
    assert output["meta"]["refresh"] is True


def test_cli_get_references_enrich(monkeypatch, capsys):
    def get_references(ids, refresh=False, enrich=False):
        return {"ok": True, "data": {"ids": ids, "refresh": refresh, "enrich": enrich}}

    monkeypatch.setattr(cli.service, "get_references", get_references)

    assert cli.main(["get-references", "0911.3380", "--enrich", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["data"]["ids"] == "0911.3380"
    assert output["data"]["enrich"] is True


def test_cli_get_citers_limit_sort(monkeypatch, capsys):
    def get_citers(ids, refresh=False, limit=1000, sort="mostrecent"):
        return {"ok": True, "data": {"ids": ids, "limit": limit, "sort": sort}}

    monkeypatch.setattr(cli.service, "get_citers", get_citers)

    assert cli.main(["get-citers", "0911.3380", "--limit", "7", "--sort", "mostcited", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["data"]["limit"] == 7
    assert output["data"]["sort"] == "mostcited"


def test_cli_get_metadata(monkeypatch, capsys):
    monkeypatch.setattr(cli.service, "get_metadata", lambda ids, refresh=False: {"ok": True, "data": {"title": ids}})

    assert cli.main(["get-metadata", "0911.3380", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["data"]["title"] == "0911.3380"


def test_cli_get_section(monkeypatch, capsys):
    monkeypatch.setattr(
        cli.service,
        "get_section",
        lambda ids, section, refresh=False: {"ok": True, "data": {"section_id": section}},
    )

    assert cli.main(["get-section", "arXiv:0911.3380", "--section", "S2", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["data"]["section_id"] == "S2"


def test_cli_search_full_text(monkeypatch, capsys):
    def search_full_text(ids, *, query, refresh=False, limit=20, context=1, case_sensitive=False):
        return {
            "ok": True,
            "data": [{"paper_id": ids, "snippet": query}],
            "meta": {
                "refresh": refresh,
                "limit": limit,
                "context": context,
                "case_sensitive": case_sensitive,
            },
        }

    monkeypatch.setattr(cli.service, "search_full_text", search_full_text)

    assert (
        cli.main(
            [
                "search-full-text",
                "0911.3380",
                "--query",
                "scalar trispectrum",
                "--limit",
                "5",
                "--context",
                "2",
                "--case-sensitive",
                "--refresh",
                "--json",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["data"][0]["paper_id"] == "0911.3380"
    assert output["data"][0]["snippet"] == "scalar trispectrum"
    assert output["meta"]["limit"] == 5
    assert output["meta"]["context"] == 2
    assert output["meta"]["case_sensitive"] is True
    assert output["meta"]["refresh"] is True


def test_cli_search_full_text_defaults_to_one_context_line(monkeypatch, capsys):
    def search_full_text(ids, *, query, refresh=False, limit=20, context=1, case_sensitive=False):
        return {"ok": True, "data": [], "meta": {"context": context}}

    monkeypatch.setattr(cli.service, "search_full_text", search_full_text)

    assert cli.main(["search-full-text", "0911.3380", "--query", "scalar exchange", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["meta"]["context"] == 1


def test_cli_parse_dispatches_html(monkeypatch, capsys):
    def parse_source(
        source_path=None,
        *,
        source="auto",
        source_id=None,
        paper_id=None,
        html_path=None,
        tex_path=None,
        pdf_path=None,
        refresh=False,
    ):
        return {
            "ok": True,
            "data": {
                "paper_id": source_id,
                "parser_version": 7,
                "source_hash": "hash",
                "toc": [],
                "sections": [],
                "equations": [],
            },
            "errors": [],
            "meta": {"html_path": str(html_path), "source": source, "refresh": refresh},
        }

    monkeypatch.setattr(cli.service, "parse_source", parse_source)

    assert cli.main(["parse", "--html", "paper.html", "--id", "local-html", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["data"]["paper_id"] == "local-html"
    assert output["meta"]["html_path"] == "paper.html"


def test_cli_parse_dispatches_tex_pdf(monkeypatch, capsys):
    def parse_source(
        source_path=None,
        *,
        source="auto",
        source_id=None,
        paper_id=None,
        html_path=None,
        tex_path=None,
        pdf_path=None,
        refresh=False,
    ):
        return {"ok": True, "data": {"paper_id": source_id, "tex_path": str(tex_path), "pdf_path": str(pdf_path)}, "errors": [], "meta": {}}

    monkeypatch.setattr(cli.service, "parse_source", parse_source)

    assert cli.main(["parse", "--tex", "note.tex", "--pdf", "book.pdf", "--id", "lecture-9", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["data"] == {"paper_id": "lecture-9", "tex_path": "note.tex", "pdf_path": "book.pdf"}


@pytest.mark.parametrize("option", ["--markdown", "--md"])
def test_cli_parse_dispatches_markdown_aliases(monkeypatch, capsys, option):
    def parse_source(source_path=None, **kwargs):
        return {
            "ok": True,
            "data": {"paper_id": kwargs["source_id"], "markdown_path": str(kwargs["markdown_path"])},
            "errors": [],
            "meta": {},
        }

    monkeypatch.setattr(cli.service, "parse_source", parse_source)

    assert cli.main(["parse", option, "note.md", "--id", "notes", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["data"] == {
        "paper_id": "notes",
        "markdown_path": "note.md",
    }


def test_cli_parse_accepts_markdown_pdf_source(monkeypatch, capsys):
    def parse_source(source_path=None, **kwargs):
        return {"ok": True, "data": kwargs, "errors": [], "meta": {}}

    monkeypatch.setattr(cli.service, "parse_source", parse_source)

    assert cli.main(["parse", "--source", "markdown-pdf", "note.md", "--pdf", "note.pdf", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["data"]["source"] == "markdown-pdf"
    assert output["data"]["pdf_path"] == "note.pdf"


def test_cli_parse_passes_explicit_document_kind(monkeypatch, capsys):
    def parse_source(source_path=None, **kwargs):
        return {"ok": True, "data": kwargs, "errors": [], "meta": {}}

    monkeypatch.setattr(cli.service, "parse_source", parse_source)

    assert cli.main(["parse", "--markdown", "book.md", "--document-kind", "book", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["data"]["document_kind"] == "book"


def test_cli_prints_parse_warnings_to_stderr(monkeypatch, capsys):
    def parse_source(
        source_path=None,
        *,
        source="auto",
        source_id=None,
        paper_id=None,
        html_path=None,
        tex_path=None,
        pdf_path=None,
        refresh=False,
    ):
        return {
            "ok": True,
            "data": {"paper_id": source_id},
            "errors": [],
            "meta": {
                "warnings": [
                    {
                        "code": "pdf_not_used",
                        "message": "PDF input was provided but pdftotext is not installed; PDF was not used.",
                        "pdf_path": "book.pdf",
                    }
                ]
            },
        }

    monkeypatch.setattr(cli.service, "parse_source", parse_source)

    assert cli.main(["parse", "--tex", "note.tex", "--pdf", "book.pdf", "--id", "lecture-9", "--json"]) == 0

    captured = capsys.readouterr()
    assert "WARNING: PDF input was provided but pdftotext is not installed; PDF was not used. (book.pdf)" in captured.err


def test_cli_parse_dispatches_ar5iv_paper(monkeypatch, capsys):
    def parse_source(
        source_path=None,
        *,
        source="auto",
        source_id=None,
        paper_id=None,
        html_path=None,
        tex_path=None,
        pdf_path=None,
        refresh=False,
    ):
        return {"ok": True, "data": {"paper_id": paper_id, "source": source, "refresh": refresh}, "errors": [], "meta": {}}

    monkeypatch.setattr(cli.service, "parse_source", parse_source)

    assert cli.main(["parse", "--paper-id", "0911.3380", "--source", "ar5iv", "--refresh", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["data"] == {"paper_id": "0911.3380", "source": "ar5iv", "refresh": True}


def test_cli_parsed_lookup_commands(monkeypatch, capsys):
    monkeypatch.setattr(cli.service, "get_parsed_source", lambda source_id: {"ok": True, "data": {"paper_id": source_id}})
    monkeypatch.setattr(cli.service, "get_parsed_source_toc", lambda source_id: {"ok": True, "data": [{"id": source_id}]})
    monkeypatch.setattr(
        cli.service,
        "get_parsed_source_equations",
        lambda source_id: {"ok": True, "data": [{"id": "eq_00001", "paper_id": source_id}]},
    )
    monkeypatch.setattr(
        cli.service,
        "get_parsed_source_equation",
        lambda source_id, equation_id: {"ok": True, "data": {"paper_id": source_id, "id": equation_id}},
    )
    monkeypatch.setattr(
        cli.service,
        "search_parsed_source",
        lambda source_id, *, query, limit=20, case_sensitive=False: {
            "ok": True,
            "data": [{"paper_id": source_id, "query": query}],
            "meta": {"limit": limit, "case_sensitive": case_sensitive},
        },
    )

    assert cli.main(["get-parsed", "lecture-9", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["data"]["paper_id"] == "lecture-9"
    assert cli.main(["get-parsed-toc", "lecture-9", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["data"][0]["id"] == "lecture-9"
    assert cli.main(["get-parsed-equations", "lecture-9", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["data"][0]["id"] == "eq_00001"
    assert cli.main(["get-parsed-equation", "lecture-9", "--equation-id", "eq_00001", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["data"]["id"] == "eq_00001"
    assert cli.main(["search-parsed", "lecture-9", "--query", "Friedmann", "--limit", "3", "--case-sensitive", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["data"] == [{"paper_id": "lecture-9", "query": "Friedmann"}]
    assert output["meta"]["limit"] == 3
    assert output["meta"]["case_sensitive"] is True


def test_cli_doctor_host(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "detect_host",
        lambda: type("Detected", (), {"host": "codex", "confidence": 1.0, "signals": ["test"]})(),
    )

    assert cli.main(["doctor", "host", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["data"]["host"] == "codex"


def test_cli_doctor_provider(monkeypatch, capsys):
    host = type("Detected", (), {"host": "codex", "confidence": 1.0, "signals": ["test"]})()
    monkeypatch.setattr(
        cli,
        "select_llm_provider",
        lambda: type("Selected", (), {"provider": "codex-cli", "host": host, "signals": ["test"]})(),
    )

    assert cli.main(["doctor", "provider", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["data"]["provider"] == "codex-cli"


def test_cli_doctor_cache(monkeypatch, capsys):
    monkeypatch.setattr(
        cli.service,
        "doctor_cache",
        lambda paper_id=None: {"ok": True, "data": {"paper": {"paper_id": paper_id}}},
    )

    assert cli.main(["doctor", "cache", "0911.3380", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["data"]["paper"]["paper_id"] == "0911.3380"


def test_cli_cache_list_dispatches(monkeypatch, capsys):
    monkeypatch.setattr(
        cli.service,
        "list_cached_papers",
        lambda *, ids=None, since=None, older_than=None: {
            "ok": True,
            "data": {"items": [{"paper_id": ids[0], "kinds": ["paper_dir"]}]},
            "errors": [],
            "meta": {"since": since, "older_than": older_than},
        },
    )

    assert cli.main(["cache", "list", "--id", "0911.3380", "--since", "1h", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["data"]["items"][0]["paper_id"] == "0911.3380"
    assert output["meta"]["since"] == "1h"


def test_cli_cache_remove_prompts_and_runs_after_y(monkeypatch, capsys):
    calls = []

    def remove_cached_papers(*, ids=None, since=None, older_than=None, all_items=False, dry_run=True):
        calls.append(dry_run)
        return {
            "ok": True,
            "data": {
                "items": [{"paper_id": ids[0], "kinds": ["source"], "paths": [{"path": "/cache/sources/lecture.json"}]}],
                "removed_count": 0 if dry_run else 1,
                "removed_paths": [] if dry_run else ["/cache/sources/lecture.json"],
            },
            "errors": [],
            "meta": {"dry_run": dry_run},
        }

    monkeypatch.setattr(cli.service, "remove_cached_papers", remove_cached_papers)
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO("y\n"))

    assert cli.main(["cache", "remove", "--id", "lecture-9", "--json"]) == 0

    captured = capsys.readouterr()
    assert "lecture-9" in captured.err
    assert "Remove cached papers? [y/N]" in captured.err
    assert calls == [True, False]
    output = json.loads(captured.out)
    assert output["data"]["removed_count"] == 1


def test_cli_cache_remove_yes_skips_prompt(monkeypatch, capsys):
    calls = []

    def remove_cached_papers(*, ids=None, since=None, older_than=None, all_items=False, dry_run=True):
        calls.append(dry_run)
        return {
            "ok": True,
            "data": {
                "items": [{"paper_id": ids[0], "kinds": ["source"], "paths": []}],
                "removed_count": 1,
                "removed_paths": ["/cache/sources/lecture.json"],
            },
            "errors": [],
            "meta": {"dry_run": dry_run},
        }

    monkeypatch.setattr(cli.service, "remove_cached_papers", remove_cached_papers)
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO(""))

    assert cli.main(["cache", "remove", "--id", "lecture-9", "--yes", "--json"]) == 0

    captured = capsys.readouterr()
    assert "Remove cached papers? [y/N]" not in captured.err
    assert calls == [False]


def test_cli_cache_remove_decline_cancels_after_preview(monkeypatch, capsys):
    calls = []

    def remove_cached_papers(*, ids=None, since=None, older_than=None, all_items=False, dry_run=True):
        calls.append(dry_run)
        return {
            "ok": True,
            "data": {
                "items": [{"paper_id": ids[0], "kinds": ["source"], "paths": []}],
                "removed_count": 0,
                "removed_paths": [],
            },
            "errors": [],
            "meta": {"dry_run": dry_run},
        }

    monkeypatch.setattr(cli.service, "remove_cached_papers", remove_cached_papers)
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO("n\n"))

    assert cli.main(["cache", "remove", "--id", "lecture-9", "--json"]) == 0

    captured = capsys.readouterr()
    assert "lecture-9" in captured.err
    assert calls == [True]
    output = json.loads(captured.out)
    assert output["data"]["cancelled"] is True
