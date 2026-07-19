from __future__ import annotations

import json
from pathlib import Path

import pytest

from arc_companion import cli
from arc_companion.domain import DomainContextError, load_domain_context
from arc_companion.pipeline import BuildOptions, _fingerprint
from arc_companion.prompts import annotation_prompt
from arc_companion.source import SourceBundle


def _write_manifest(project: Path) -> Path:
    domain_dir = project / "domain"
    domain_dir.mkdir(parents=True)
    (domain_dir / "one_summary.json").write_text(json.dumps({
        "domain_id": "domain-one", "domain_title": "One", "overview": "Preferred context",
    }), encoding="utf-8")
    (domain_dir / "one_pack.json").write_text(json.dumps({
        "domain_id": "domain-one", "papers": [
            {"paper_id": "arXiv:1111.1111", "role": "selected_foundation"},
            {"paper_id": "arXiv:2222.2222", "role": "domain_paper"},
        ],
    }), encoding="utf-8")
    manifest = domain_dir / "domain-manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": "arc.workflow.domain_manifest.v1",
        "domains": [{
            "domain_id": "domain-one",
            "summary_json_path": "domain/one_summary.json",
            "paper_json_pack_path": "domain/one_pack.json",
        }],
    }), encoding="utf-8")
    return manifest


def test_explicit_manifest_loads_summary_roles_and_paper_ids(tmp_path: Path) -> None:
    context = load_domain_context(domain_manifest=_write_manifest(tmp_path))
    assert context["source"] == "domain_manifest"
    assert context["paper_ids"] == ["arXiv:1111.1111", "arXiv:2222.2222"]
    assert context["domains"][0]["papers"][1]["role"] == "domain_paper"


def test_no_domain_option_does_not_discover_or_build(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    assert load_domain_context() is None


def test_explicit_domain_id_reads_existing_summary_and_graph_only(monkeypatch) -> None:
    from arc_domain import service

    monkeypatch.setattr(service, "get_domain_summary", lambda **kwargs: {
        "ok": True, "data": {"summary": {"domain_id": kwargs["domain_id"], "overview": "Existing"}}
    })
    monkeypatch.setattr(service, "get_domain_graph", lambda **kwargs: {
        "ok": True, "data": {"graph": {"nodes": [{"paper_id": "arXiv:3333.3333", "role": "domain_paper"}]}}
    })

    context = load_domain_context(domain_id="existing-domain")
    assert context["source"] == "domain_id"
    assert context["paper_ids"] == ["arXiv:3333.3333"]


def test_invalid_or_duplicate_manifest_is_rejected(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["domains"].append(dict(payload["domains"][0]))
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DomainContextError, match="duplicate domain_id"):
        load_domain_context(domain_manifest=manifest)


def test_cli_domain_options_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        cli.main(["build", "arXiv:1", "--domain-id", "one", "--domain-manifest", "manifest.json"])


def test_domain_context_is_in_annotation_prompt_and_fingerprint(tmp_path: Path) -> None:
    context = load_domain_context(domain_manifest=_write_manifest(tmp_path))
    prompt = annotation_prompt(
        {"segment_id": "s"}, [], language="zh-CN", metadata={}, evidence={}, glossary={},
        protected_names=[], paper_context={}, domain_context=context,
    )
    assert "EXPLICIT DOMAIN CONTEXT" in prompt
    assert "arXiv:2222.2222" in prompt
    assert "not as a closed corpus" in prompt

    document = {
        "blocks": [{"block_id": "b", "type": "text", "text": "x"}],
        "integrity": {"status": "complete", "document_hash": "x"},
    }
    bundle = SourceBundle(
        paper_id="arXiv:1", parsed={"document": document}, document=document,
        metadata={}, references=[], citers=[],
    )
    options = BuildOptions(paper_id="arXiv:1", project_dir=tmp_path / "run")
    evidence = {"related_papers": []}
    assert _fingerprint(bundle, options, evidence=evidence) != _fingerprint(
        bundle, options, evidence=evidence, domain_context=context
    )
