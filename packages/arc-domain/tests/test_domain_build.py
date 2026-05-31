from __future__ import annotations

import os
import re

import pytest

from arc_domain import foundation
from arc_domain import evidence
from arc_domain import network
from arc_domain import render
from arc_domain import service
from arc_domain import summary as domain_summary
from arc_domain.cache import DomainPaths, domain_id_for, read_json
from arc_domain import paper


SEED = "arXiv:2401.00001"
FOUNDATION = "arXiv:2301.00001"


@pytest.mark.skipif(
    os.environ.get("ARC_RUN_SLOW_DOMAIN_TESTS") != "1",
    reason="set ARC_RUN_SLOW_DOMAIN_TESTS=1 to run slow domain build tests",
)
def test_build_domain_writes_core_artifacts(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_DOMAIN_CACHE", str(tmp_path / "arc-domain"))
    _install_fake_paper_query(monkeypatch)
    _install_fake_domain_summary(monkeypatch)

    result = service.build_domain(SEED, intent="inflation observables", provider="manual", workers=1)

    assert result["ok"] is True
    data = result["data"]
    paths = DomainPaths.for_domain(data["domain_id"])
    assert paths.foundation_selection.exists()
    assert paths.domain_graph.exists()
    assert paths.evidence_pack.exists()
    assert paths.paper_json_pack.exists()
    assert paths.domain_summary.exists()
    assert paths.domain_summary_markdown.exists()
    assert paths.network_html.exists()
    assert data["foundation"]["selected_foundation"]["paper_id"] == FOUNDATION
    graph = read_json(paths.domain_graph)
    assert graph["foundation_paper"] == FOUNDATION
    assert any(node["paper_id"] == FOUNDATION and node["role"] == "selected_foundation" for node in graph["nodes"])
    parent = next(node for node in graph["nodes"] if node["paper_id"] == "arXiv:2301.00002")
    assert parent["role"] == "parent_foundation"
    assert parent["abstract"] == "Abstract for Parent Paper."
    assert parent["authors"] == ["Alice A.", "Bob B."]
    assert parent["citation_count"] == 1500
    domain = next(node for node in graph["nodes"] if node["paper_id"] == "arXiv:2402.00001")
    assert domain["in_graph_citer_count"] == 1
    assert domain["in_graph_citer_score"] == 1.0
    assert "citation_rate_score" in domain
    assert domain["reference_edge_count"] == 2
    assert domain["reference_edge_score"] == 1.0
    assert any(node["role"] == "common_reference" for node in graph["nodes"])
    paper_pack = read_json(paths.paper_json_pack)
    graph_ids = {node["paper_id"] for node in graph["nodes"]}
    assert paper_pack["schema_version"] == "arc.domain_paper_json_pack.v1"
    assert paper_pack["paper_count"] == len(graph_ids)
    assert {item["paper_id"] for item in paper_pack["papers"]} == graph_ids
    assert all("metadata" in item and "toc" in item and "references" in item for item in paper_pack["papers"])
    assert _toc_calls == graph_ids
    assert read_json(paths.domain_summary)["summary_method"] == "llm"
    assert data["domain_summary_markdown_path"] == str(paths.domain_summary_markdown)
    markdown = paths.domain_summary_markdown.read_text(encoding="utf-8")
    assert "## Key Papers" in markdown
    assert "Foundation paper:" in markdown
    assert "Best reference paper:" in markdown
    html = paths.network_html.read_text(encoding="utf-8")
    graph_data = re.search(r'<script id="graph-data" type="application/json">(.*?)</script>', html, re.S)
    assert graph_data
    assert "&quot;" not in graph_data.group(1)
    assert '"nodes"' in graph_data.group(1)
    assert "highlightConnectedEdges" in html
    assert "Authors:" in html
    assert "MathJax" in html
    assert "typesetMath(details)" in html
    assert "Ref edges:" in html
    assert "network.focus" not in html
    assert "navigationButtons: false" in html
    assert 'id="fit-network"' in html
    table_body = re.search(r"<tbody>(.*?)</tbody>", html, re.S).group(1)
    assert table_body.index(">Parent<") < table_body.index(">Common<") < table_body.index(">Domain<")
    assert re.search(r'<tr data-id="arXiv:2201.00001">.*?<td></td>.*?</tr>', table_body, re.S)
    assert re.search(r'<tr data-id="arXiv:2402.00001">.*?<td>[0-9.]+</td>.*?</tr>', table_body, re.S)


@pytest.mark.skipif(
    os.environ.get("ARC_RUN_SLOW_DOMAIN_TESTS") != "1",
    reason="set ARC_RUN_SLOW_DOMAIN_TESTS=1 to run slow domain build tests",
)
def test_status_and_cached_summary(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_DOMAIN_CACHE", str(tmp_path / "arc-domain"))
    _install_fake_paper_query(monkeypatch)
    _install_fake_domain_summary(monkeypatch)
    domain_id = domain_id_for(SEED, "intent")

    service.build_domain(SEED, intent="intent", domain_id=domain_id, provider="manual", workers=1)
    status = service.status(domain_id=domain_id)
    summary = service.get_domain_summary(domain_id=domain_id)
    graph = service.get_domain_graph(domain_id=domain_id)

    assert status["ok"] is True
    assert status["data"]["artifacts"]["domain_summary"]["exists"] is True
    assert status["data"]["artifacts"]["paper_json_pack"]["exists"] is True
    assert summary["ok"] is True
    assert graph["ok"] is True


def test_build_domain_passes_model_tier_to_all_llm_domain_steps(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_DOMAIN_CACHE", str(tmp_path / "arc-domain"))
    calls = []

    def identify_foundation(**kwargs):
        calls.append(("foundation", kwargs.get("model_tier")))
        return {"selection": {"selected_foundation": {"paper_id": FOUNDATION}}}

    def build_network(**kwargs):
        calls.append(("network", kwargs.get("model_tier")))
        return {"node_count": 1, "edge_count": 0, "graph_path": "graph.json"}

    def summarize_domain(**kwargs):
        calls.append(("summary", kwargs.get("model_tier")))
        return {"domain_summary_path": "summary.json", "summary": {}}

    monkeypatch.setattr(service, "_identify_foundation", identify_foundation)
    monkeypatch.setattr(service, "_build_network", build_network)
    monkeypatch.setattr(service, "render_network_html", lambda **kwargs: {"network_html_path": "network.html"})
    monkeypatch.setattr(service, "_build_paper_json_pack", lambda **kwargs: {"paper_json_pack_path": "pack.json"})
    monkeypatch.setattr(service, "_build_evidence_pack", lambda **kwargs: {"evidence_pack_path": "evidence.json"})
    monkeypatch.setattr(service, "_summarize_domain", summarize_domain)

    result = service.build_domain(SEED, intent="intent", provider="auto", model_tier="high", workers=1)

    assert result["ok"] is True
    assert calls == [("foundation", "high"), ("network", "high"), ("summary", "high")]


def test_domain_rejects_reused_domain_id_with_changed_seed_or_intent(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_DOMAIN_CACHE", str(tmp_path / "arc-domain"))
    domain_id = "shared-domain"

    first = service.init_domain(SEED, intent="first intent", domain_id=domain_id)
    second = service.build_domain(SEED, intent="second intent", domain_id=domain_id, provider="manual", workers=1)

    assert first["ok"] is True
    assert second["ok"] is False
    assert second["error"]["code"] == "domain_build_failed"
    assert "domain_id input mismatch" in second["error"]["message"]
    config = read_json(DomainPaths.for_domain(domain_id).config)
    assert config["input_fingerprint"]["identity"]["intent"] == "first intent"
    assert config["input_fingerprint"]["identity_hash"]


def test_domain_rejects_reused_domain_id_with_changed_llm_config(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_DOMAIN_CACHE", str(tmp_path / "arc-domain"))
    domain_id = "shared-domain"

    monkeypatch.setattr(service, "_identify_foundation", lambda **kwargs: {"selection": {"selected_foundation": {}}})
    monkeypatch.setattr(
        service,
        "_build_network",
        lambda **kwargs: {"node_count": 0, "edge_count": 0, "graph_path": "graph.json"},
    )
    monkeypatch.setattr(service, "render_network_html", lambda **kwargs: {"network_html_path": "network.html"})
    monkeypatch.setattr(service, "_build_paper_json_pack", lambda **kwargs: {"paper_json_pack_path": "pack.json"})
    monkeypatch.setattr(service, "_build_evidence_pack", lambda **kwargs: {"evidence_pack_path": "evidence.json"})
    monkeypatch.setattr(service, "_summarize_domain", lambda **kwargs: {"domain_summary_path": "summary.json", "summary": {}})

    first = service.build_domain(SEED, intent="intent", domain_id=domain_id, provider="manual", workers=1)
    second = service.identify_foundation(SEED, intent="intent", domain_id=domain_id, provider="auto")

    assert first["ok"] is True
    assert second["ok"] is False
    assert second["error"]["code"] == "foundation_identification_failed"
    assert "LLM configuration mismatch" in second["error"]["message"]


def test_domain_llm_helpers_pass_model_tier_to_run_json(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_DOMAIN_CACHE", str(tmp_path / "arc-domain"))
    captured = []

    def fake_run_json(prompt, *, schema, provider, model=None, model_tier=None):
        captured.append((schema["$id"], provider, model, model_tier))
        if schema["$id"] == "arc.domain-foundation-candidate-audit-v1":
            return {
                "schema_version": "arc.domain_foundation_candidate_audit.v1",
                "candidate_set_sufficient": True,
                "confidence": "high",
                "search_queries": [],
                "citation_directions": [],
                "reasoning": "ok",
                "warnings": [],
            }
        if schema["$id"] == "arc.domain-intent-ranking-v1":
            return {"ranked_paper_ids": [FOUNDATION], "reasoning": "ok"}
        return {
            "schema_version": "arc.domain_summary.v4",
            "domain_title": "Domain",
            "brief_introduction": "Brief",
            "task_focus": {"user_intent": "intent", "research_scope": "scope", "priority_rules": []},
            "foundation_paper": {"paper_id": FOUNDATION, "title": "Foundation", "reason": "test"},
            "best_reference_paper": {"paper_id": FOUNDATION, "title": "Foundation", "reason": "test"},
            "methodology": [],
            "known_solved_cases": [],
            "open_axes_for_new_work": [],
            "warnings": [],
        }

    monkeypatch.setattr(foundation, "run_json", fake_run_json)
    monkeypatch.setattr(network, "run_json", fake_run_json)
    monkeypatch.setattr(domain_summary, "run_json", fake_run_json)

    foundation._llm_audit_candidates(
        seed_metadata={"paper_id": SEED, "title": "Seed"},
        candidates=[],
        intent="intent",
        provider="auto",
        model=None,
        model_tier="high",
    )
    network._rank_by_intent(
        [{"paper_id": FOUNDATION, "title": "Foundation", "abstract": "intent", "citation_count": 1}],
        intent="intent",
        provider="auto",
        model=None,
        model_tier="high",
    )
    paths = DomainPaths.for_domain(domain_id_for(SEED, "intent"))
    paths.domain_dir.mkdir(parents=True, exist_ok=True)
    paths.domain_graph.write_text('{"nodes": []}', encoding="utf-8")
    paths.evidence_pack.write_text('{"papers": []}', encoding="utf-8")
    paths.foundation_selection.write_text(
        '{"selected_foundation": {"paper_id": "arXiv:2301.00001", "title": "Foundation", "reason": "test"}}',
        encoding="utf-8",
    )
    domain_summary.summarize_domain(paths=paths, provider="auto", model=None, model_tier="high")

    assert captured == [
        ("arc.domain-foundation-candidate-audit-v1", "auto", None, "high"),
        ("arc.domain-intent-ranking-v1", "auto", None, "high"),
        ("arc.domain-summary-v4", "auto", None, "high"),
    ]


def test_summarize_domain_reports_llm_failure_without_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_DOMAIN_CACHE", str(tmp_path / "arc-domain"))
    domain_id = domain_id_for(SEED, "intent")
    paths = DomainPaths.for_domain(domain_id)
    paths.domain_dir.mkdir(parents=True, exist_ok=True)
    paths.evidence_pack.write_text('{"papers": [], "warnings": []}', encoding="utf-8")
    paths.domain_graph.write_text('{"nodes": []}', encoding="utf-8")
    paths.foundation_selection.write_text(
        '{"selected_foundation": {"paper_id": "arXiv:2301.00001", "title": "Foundation", "reason": "seed"}}',
        encoding="utf-8",
    )

    def fail_summary(*args, **kwargs):
        raise RuntimeError("summary prompt too large")

    monkeypatch.setattr(domain_summary, "run_json", fail_summary)

    result = service.summarize_domain(SEED, intent="intent", domain_id=domain_id, provider="auto")

    assert result["ok"] is False
    assert result["error"]["code"] == "domain_summary_failed"
    assert "summary prompt too large" in result["error"]["message"]
    assert not paths.domain_summary.exists()
    assert not paths.domain_summary_markdown.exists()


def test_get_domain_summary_rejects_cached_deterministic_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_DOMAIN_CACHE", str(tmp_path / "arc-domain"))
    domain_id = domain_id_for(SEED, "intent")
    paths = DomainPaths.for_domain(domain_id)
    paths.domain_dir.mkdir(parents=True, exist_ok=True)
    paths.domain_summary.write_text(
        '{"schema_version": "arc.domain_summary.v4", "summary_method": "deterministic_fallback"}',
        encoding="utf-8",
    )

    result = service.get_domain_summary(domain_id=domain_id)

    assert result["ok"] is False
    assert result["error"]["code"] == "domain_summary_invalid"
    assert "deterministic fallback" in result["error"]["message"]


def test_foundation_prompt_marks_low_citation_candidates_as_low_priority():
    prompt = foundation._foundation_prompt(
        seed_metadata={"paper_id": SEED, "title": "Seed Paper"},
        candidates=[
            {
                "paper_id": "arXiv:2401.00001",
                "title": "Young Exact Paper",
                "citation_count": 41,
            }
        ],
        intent="exact topic",
    )

    assert "fewer than 100 citations" in prompt
    assert "low priority" in prompt


def test_deterministic_foundation_selection_deprioritizes_low_citation_candidates():
    selection = foundation._deterministic_selection(
        [
            {
                "paper_id": "arXiv:2401.00001",
                "title": "Young Exact Paper",
                "citation_count": 41,
                "witness_citation_overlap": 5,
                "intent_overlap": 1.0,
            },
            {
                "paper_id": "arXiv:2301.00001",
                "title": "Established Foundation",
                "citation_count": 150,
                "witness_citation_overlap": 5,
                "intent_overlap": 0.5,
            },
        ],
        intent="exact topic",
    )

    assert selection["selected_foundation"]["paper_id"] == "arXiv:2301.00001"


def test_foundation_selection_contract_includes_best_reference_paper():
    prompt = foundation._foundation_prompt(
        seed_metadata={"paper_id": SEED, "title": "Seed Paper"},
        candidates=[
            {
                "paper_id": "arXiv:2301.00001",
                "title": "Established Foundation",
                "citation_count": 180,
                "intent_overlap": 0.4,
                "witness_citation_overlap": 8,
            },
            {
                "paper_id": "arXiv:2401.00002",
                "title": "Clear Modern Method Paper",
                "citation_count": 80,
                "intent_overlap": 0.9,
                "witness_citation_overlap": 3,
            },
        ],
        intent="clear modern method",
    )
    selection = foundation._deterministic_selection(
        [
            {
                "paper_id": "arXiv:2301.00001",
                "title": "Established Foundation",
                "citation_count": 180,
                "intent_overlap": 0.4,
                "witness_citation_overlap": 8,
            },
            {
                "paper_id": "arXiv:2401.00002",
                "title": "Clear Modern Method Paper",
                "citation_count": 80,
                "intent_overlap": 0.9,
                "witness_citation_overlap": 3,
            },
        ],
        intent="clear modern method",
    )

    assert "best reference" in prompt.lower()
    assert "best_reference_paper" in foundation.FOUNDATION_SELECTION_SCHEMA["required"]
    assert selection["selected_foundation"]["paper_id"] == "arXiv:2301.00001"
    assert selection["best_reference_paper"]["paper_id"] == "arXiv:2401.00002"


def test_foundation_audit_adds_verified_llm_candidate(monkeypatch):
    calls = []

    def fake_infer(text, *, provider="auto", model=None, refresh=False):
        calls.append(text)
        return {
            "ok": True,
            "data": ["arXiv:2101.00001"],
            "meta": {
                "llm_used": True,
                "verified_references": [
                    {
                        "paper_id": "arXiv:2101.00001",
                        "verified_title": "Missing Foundation",
                        "evidence_urls": ["https://arxiv.org/abs/2101.00001"],
                        "reasoning": "verified by web search",
                    }
                ]
            },
        }

    monkeypatch.setattr(foundation.paper, "infer_main_references", fake_infer)
    monkeypatch.setattr(foundation.paper, "metadata", _metadata)

    expanded, report = foundation._expand_candidates_from_audit(
        candidates=[
            {
                "paper_id": "arXiv:2301.00001",
                "title": "Existing Candidate",
                "citation_count": 120,
                "witness_citation_overlap": 3,
                "intent_overlap": 0.4,
            }
        ],
        audit={
            "candidate_set_sufficient": False,
            "confidence": "complete",
            "search_queries": [
                {
                    "query": "missing foundation exact title",
                    "reason": "canonical paper is absent",
                    "confidence": "complete",
                }
            ],
            "citation_directions": ["check references of the missing foundation"],
            "warnings": [],
        },
        intent="missing foundation",
        provider="auto",
        model=None,
        refresh=False,
        workers=1,
    )

    assert len(calls) == 1
    assert "missing foundation exact title" in calls[0]
    assert "missing foundation" in calls[0]
    added = next(item for item in expanded if item["paper_id"] == "arXiv:2101.00001")
    assert added["llm_added"] is True
    assert added["source_role"] == "llm_added_foundation_candidate"
    assert added["llm_verified_evidence_urls"] == ["https://arxiv.org/abs/2101.00001"]
    assert report["added_candidate_count"] == 1
    assert report["searches"][0]["status"] == "added"


def test_foundation_audit_skips_uncertain_expansion(monkeypatch):
    calls = []

    def fake_infer(*args, **kwargs):
        calls.append(args)
        return {"ok": True, "data": ["arXiv:2101.00001"], "meta": {}}

    monkeypatch.setattr(foundation.paper, "infer_main_references", fake_infer)
    candidates = [{"paper_id": "arXiv:2301.00001", "title": "Existing Candidate"}]

    expanded, report = foundation._expand_candidates_from_audit(
        candidates=candidates,
        audit={
            "candidate_set_sufficient": False,
            "confidence": "medium",
            "search_queries": [
                {
                    "query": "maybe missing foundation",
                    "reason": "uncertain",
                    "confidence": "complete",
                }
            ],
            "citation_directions": [],
            "warnings": [],
        },
        intent="maybe",
        provider="auto",
        model=None,
        refresh=False,
        workers=1,
    )

    assert expanded == candidates
    assert calls == []
    assert report["added_candidate_count"] == 0
    assert report["searches"][0]["status"] == "skipped_uncertain_audit"


def test_foundation_expansion_requires_web_verified_references(monkeypatch):
    def fake_infer(text, *, provider="auto", model=None, refresh=False):
        return {
            "ok": True,
            "data": ["arXiv:2101.00001"],
            "meta": {"llm_used": False},
        }

    monkeypatch.setattr(foundation.paper, "infer_main_references", fake_infer)
    monkeypatch.setattr(foundation.paper, "metadata", _metadata)

    expanded, report = foundation._expand_candidates_from_audit(
        candidates=[{"paper_id": "arXiv:2301.00001", "title": "Existing Candidate"}],
        audit={
            "candidate_set_sufficient": False,
            "confidence": "complete",
            "search_queries": [
                {
                    "query": "missing foundation terms",
                    "reason": "canonical paper likely absent",
                    "confidence": "complete",
                }
            ],
            "citation_directions": [],
            "warnings": [],
        },
        intent="missing foundation",
        provider="auto",
        model=None,
        refresh=False,
        workers=1,
    )

    assert [item["paper_id"] for item in expanded] == ["arXiv:2301.00001"]
    assert report["added_candidate_count"] == 0
    assert report["searches"][0]["status"] == "reference_inference_unverified"


def test_foundation_verifier_request_omits_explicit_ids_from_reason_and_intent(monkeypatch):
    calls = []

    def fake_infer(text, *, provider="auto", model=None, refresh=False):
        calls.append(text)
        return {
            "ok": True,
            "data": ["arXiv:2101.00001"],
            "meta": {
                "llm_used": True,
                "verified_references": [
                    {
                        "paper_id": "arXiv:2101.00001",
                        "verified_title": "Missing Foundation",
                        "evidence_urls": ["https://arxiv.org/abs/2101.00001"],
                        "reasoning": "verified by web search",
                    }
                ],
            },
        }

    monkeypatch.setattr(foundation.paper, "infer_main_references", fake_infer)
    monkeypatch.setattr(foundation.paper, "metadata", _metadata)

    expanded, report = foundation._expand_candidates_from_audit(
        candidates=[{"paper_id": "arXiv:2301.00001", "title": "Existing Candidate"}],
        audit={
            "candidate_set_sufficient": False,
            "confidence": "complete",
            "search_queries": [
                {
                    "query": "missing foundation terms",
                    "reason": "maybe arXiv:2101.00001",
                    "confidence": "complete",
                }
            ],
            "citation_directions": [],
            "warnings": [],
        },
        intent="compare with arXiv:2201.00001",
        provider="auto",
        model=None,
        refresh=False,
        workers=1,
    )

    assert "arXiv:2101.00001" not in calls[0]
    assert "arXiv:2201.00001" not in calls[0]
    assert next(item for item in expanded if item["paper_id"] == "arXiv:2101.00001")["llm_added"] is True
    assert report["searches"][0]["status"] == "added"


def test_deterministic_fallback_preserves_llm_added_markers(monkeypatch):
    def fail_selection(*args, **kwargs):
        raise RuntimeError("selection failed")

    monkeypatch.setattr(foundation, "run_json", fail_selection)

    selection = foundation._llm_select_foundation(
        seed_metadata={"paper_id": SEED, "title": "Seed Paper"},
        candidates=[
            {
                "paper_id": "arXiv:2101.00001",
                "title": "Missing Foundation",
                "citation_count": 200,
                "witness_citation_overlap": 8,
                "intent_overlap": 0.9,
                "source_role": "llm_added_foundation_candidate",
                "llm_added": True,
                "llm_reference_query": "missing foundation terms",
            }
        ],
        intent="missing foundation",
        provider="auto",
        model=None,
    )

    selected = selection["selected_foundation"]
    assert selected["paper_id"] == "arXiv:2101.00001"
    assert selected["llm_added"] is True
    assert selected["source_role"] == "llm_added_foundation_candidate"
    assert selected["llm_reference_query"] == "missing foundation terms"


def test_candidate_audit_moves_uncertain_queries_to_warnings():
    audit = foundation._repair_candidate_audit(
        {
            "candidate_set_sufficient": False,
            "confidence": "complete",
            "search_queries": [
                {
                    "query": "confident missing foundation",
                    "reason": "complete certainty",
                    "confidence": "complete",
                },
                {
                    "query": "maybe missing foundation",
                    "reason": "not certain",
                    "confidence": "medium",
                },
            ],
            "citation_directions": [],
            "warnings": [],
        },
        method="llm",
    )

    assert [item["query"] for item in audit["search_queries"]] == ["confident missing foundation"]
    assert any("maybe missing foundation" in warning for warning in audit["warnings"])


def test_evidence_pack_sorts_same_role_by_paper_id(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_DOMAIN_CACHE", str(tmp_path / "arc-domain"))
    paths = DomainPaths.for_domain("sort_test")
    paths.domain_dir.mkdir(parents=True)
    paths.domain_graph.write_text(
        """
        {
          "foundation_paper": "arXiv:2301.00001",
          "nodes": [
            {"paper_id": "arXiv:2301.00003", "role": "domain_paper"},
            {"paper_id": "arXiv:2301.00001", "role": "selected_foundation"},
            {"paper_id": "arXiv:2301.00002", "role": "domain_paper"}
          ]
        }
        """,
        encoding="utf-8",
    )

    monkeypatch.setattr(
        evidence.paper,
        "metadata",
        lambda paper_id, refresh=False: {"paper_id": paper_id, "title": paper_id, "authors": [], "citation_count": 0},
    )
    monkeypatch.setattr(
        evidence.paper,
        "section",
        lambda paper_id, selector, refresh=False: {"section_id": selector, "title": selector, "text": "done"},
    )

    result = evidence.build_evidence_pack(paths=paths, workers=1)

    assert [item["paper_id"] for item in result["evidence_pack"]["papers"]] == [
        "arXiv:2301.00001",
        "arXiv:2301.00002",
        "arXiv:2301.00003",
    ]


def test_render_node_rank_uses_zero_domain_score_not_support_count():
    high_score = {"role": "domain_paper", "domain_score": 1.0, "support_count": 0, "citation_count": 0}
    zero_score_with_support = {
        "role": "domain_paper",
        "domain_score": 0.0,
        "support_count": 100,
        "citation_count": 0,
    }

    ranked = sorted([zero_score_with_support, high_score], key=render._node_rank_key)  # noqa: SLF001

    assert ranked == [high_score, zero_score_with_support]


@pytest.mark.skipif(
    os.environ.get("ARC_RUN_SLOW_DOMAIN_TESTS") != "1",
    reason="set ARC_RUN_SLOW_DOMAIN_TESTS=1 to run slow domain build tests",
)
def test_network_marks_llm_added_foundation(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_DOMAIN_CACHE", str(tmp_path / "arc-domain"))
    _install_fake_paper_query(monkeypatch)
    paths = DomainPaths.for_domain(domain_id_for(SEED, "intent"))
    paths.domain_dir.mkdir(parents=True, exist_ok=True)
    paths.foundation_selection.write_text(
        """
        {
          "selected_foundation": {
            "paper_id": "arXiv:2101.00001",
            "title": "Missing Foundation",
            "reason": "LLM verified missing canonical paper",
            "source_role": "llm_added_foundation_candidate",
            "llm_added": true,
            "llm_reference_query": "missing foundation exact title",
            "llm_verified_evidence_urls": ["https://arxiv.org/abs/2101.00001"]
          },
          "parent_foundations": []
        }
        """,
        encoding="utf-8",
    )

    result = service.build_network(SEED, intent="intent", domain_id=paths.domain_id, provider="manual", workers=1)

    assert result["ok"] is True
    graph = read_json(paths.domain_graph)
    node = next(item for item in graph["nodes"] if item["role"] == "selected_foundation")
    assert node["paper_id"] == "arXiv:2101.00001"
    assert node["llm_added"] is True
    assert node["source_role"] == "llm_added_foundation_candidate"
    assert node["llm_reference_query"] == "missing foundation exact title"


def test_foundation_repair_repairs_unknown_best_reference_to_selected_foundation():
    selection = foundation._repair_selection(
        {
            "selected_foundation": {
                "paper_id": "arXiv:0911.3380",
                "title": "Selected Foundation",
                "reason": "selected",
            },
            "best_reference_paper": {
                "paper_id": "arXiv:9999.99999",
                "title": "Unknown Paper",
                "reason": "not in candidates",
            },
            "parent_foundations": [],
            "rejected_candidates": [],
            "warnings": [],
        },
        [{"paper_id": "arXiv:0911.3380", "title": "Selected Foundation", "year": 2009}],
        method="llm",
    )

    assert selection["best_reference_paper"]["paper_id"] == "arXiv:0911.3380"
    assert "unknown id" in selection["best_reference_paper"]["reason"]


def test_foundation_repair_unknown_selected_foundation_uses_deterministic_ranking_without_removing_candidates():
    candidates = [
        {
            "paper_id": "arXiv:2401.00001",
            "title": "First Low Support Candidate",
            "citation_count": 10,
            "witness_citation_overlap": 1,
            "intent_overlap": 0.1,
        },
        {
            "paper_id": "arXiv:2301.00001",
            "title": "Deterministic Foundation",
            "citation_count": 200,
            "witness_citation_overlap": 6,
            "intent_overlap": 0.8,
        },
    ]

    selection = foundation._repair_selection(
        {
            "selected_foundation": {
                "paper_id": "arXiv:9999.99999",
                "title": "Unknown Paper",
                "reason": "not in candidates",
            },
            "best_reference_paper": {
                "paper_id": "arXiv:2301.00001",
                "title": "Deterministic Foundation",
                "reason": "readable",
            },
            "parent_foundations": [],
            "rejected_candidates": [],
            "warnings": [],
        },
        candidates,
        method="llm",
        intent="deterministic foundation",
    )

    assert [item["paper_id"] for item in candidates] == ["arXiv:2401.00001", "arXiv:2301.00001"]
    assert selection["selected_foundation"]["paper_id"] == "arXiv:2301.00001"
    assert "deterministic fallback" in selection["selected_foundation"]["reason"]
    assert "llm_selected_unknown_id:arXiv:9999.99999" in selection["warnings"]


def test_foundation_repair_rejects_later_parent_foundations():
    selection = foundation._repair_selection(
        {
            "selected_foundation": {
                "paper_id": "arXiv:0911.3380",
                "title": "Selected Foundation",
                "reason": "selected",
            },
            "parent_foundations": [
                {
                    "paper_id": "arXiv:1503.08043",
                    "title": "Later Parent",
                    "reason": "broad but later",
                },
                {
                    "paper_id": "arXiv:0901.00001",
                    "title": "Earlier Parent",
                    "reason": "earlier",
                },
            ],
            "rejected_candidates": [],
            "warnings": [],
        },
        [
            {"paper_id": "arXiv:0911.3380", "title": "Selected Foundation", "year": 2009},
            {"paper_id": "arXiv:1503.08043", "title": "Later Parent", "year": 2015},
            {"paper_id": "arXiv:0901.00001", "title": "Earlier Parent", "year": 2009},
        ],
        method="llm",
    )

    assert [item["paper_id"] for item in selection["parent_foundations"]] == ["arXiv:0901.00001"]
    rejected_ids = [item["paper_id"] for item in selection["rejected_candidates"]]
    assert "arXiv:1503.08043" in rejected_ids
    assert "later than the selected foundation year" in selection["rejected_candidates"][0]["reason"]


def test_domain_summary_contract_uses_best_reference_and_new_sections():
    prompt = domain_summary._summary_prompt(
        graph={
            "foundation_paper": FOUNDATION,
            "nodes": [{"paper_id": FOUNDATION, "role": "selected_foundation", "title": "Foundation Paper"}],
            "edges": [],
        },
        evidence={"papers": [], "warnings": []},
        selection={
            "selected_foundation": {"paper_id": FOUNDATION, "title": "Foundation Paper", "reason": "seed"},
            "best_reference_paper": {"paper_id": "arXiv:2401.00002", "title": "Best Ref", "reason": "clear"},
        },
    )

    assert "best reference" in prompt.lower()
    assert "foundation paper" in prompt.lower()
    assert "known solved cases" in prompt.lower()
    assert "open axes" in prompt.lower()
    assert "llm_summary" not in prompt
    assert "discover additional axes" in prompt.lower()
    assert "idea examples" not in prompt.lower()
    assert "research directions" not in prompt.lower()
    assert "open questions" not in prompt.lower()
    assert "report_remarks" not in domain_summary.DOMAIN_SUMMARY_SCHEMA["properties"]
    assert "foundation_paper" in domain_summary.DOMAIN_SUMMARY_SCHEMA["required"]
    assert "research_directions_and_questions" not in domain_summary.DOMAIN_SUMMARY_SCHEMA["properties"]
    assert "idea_examples" not in domain_summary.DOMAIN_SUMMARY_SCHEMA["properties"]
    assert "open_questions" not in domain_summary.DOMAIN_SUMMARY_SCHEMA["properties"]
    assert "mainstream_directions" not in domain_summary.DOMAIN_SUMMARY_SCHEMA["properties"]
    assert "frequently_asked_questions" not in domain_summary.DOMAIN_SUMMARY_SCHEMA["properties"]
    assert "research_guidance" not in domain_summary.DOMAIN_SUMMARY_SCHEMA["properties"]
    assert "reading_guide" not in domain_summary.DOMAIN_SUMMARY_SCHEMA["properties"]
    assert not hasattr(domain_summary, "_fallback_summary")


def test_domain_summary_markdown_omits_report_remarks_guidance_and_warnings():
    markdown = domain_summary.render_summary_markdown(
        {
            "domain_title": "Example Domain",
            "brief_introduction": "Compact intro.",
            "foundation_paper": {
                "paper_id": "arXiv:2301.00001",
                "title": "Foundation Paper",
                "reason": "Best field-defining citer source.",
            },
            "best_reference_paper": {
                "paper_id": "arXiv:2401.00002",
                "title": "Best Ref",
                "reason": "Clear methodology.",
            },
            "methodology": [{"claim": "Use residues.", "papers": ["arXiv:2401.00002"]}],
            "task_focus": {
                "user_intent": "Compute a tree-level scalar correlator.",
                "research_scope": "Example domain seeded by a paper.",
                "priority_rules": ["Satisfy the user request before following attached context."],
            },
            "known_solved_cases": [
                {
                    "solved_case": "Known four-point massive exchange.",
                    "why_it_is_solved": "The seed correlator is already derived.",
                    "transferable_form": "Use a precise observable, setup, and validation limits.",
                    "forbidden_reuse": "Do not propose the same seed correlator as new.",
                    "valid_new_axes": ["new observable", "new regime"],
                    "papers": ["arXiv:2401.00002"],
                }
            ],
            "open_axes_for_new_work": [
                {
                    "axis": "Observable",
                    "guidance": "Move beyond the solved four-point seed when the observable is genuinely distinct.",
                    "example_variations": ["higher-point correlator", "late-time statistic"],
                    "papers": ["arXiv:2401.00002"],
                }
            ],
            "warnings": ["missing conclusion text"],
        }
    )

    assert markdown.startswith("# Example Domain\n\nCompact intro.\n\n## Task Focus for Idea Generation")
    assert "This report lists prominent outstanding ideas" not in markdown
    assert "## Foundation Paper" not in markdown
    assert "## Best Reference Paper" not in markdown
    assert "## Key Papers" in markdown
    assert "- Foundation paper: arXiv:2301.00001: Foundation Paper" in markdown
    assert "- Best reference paper: arXiv:2401.00002: Best Ref" in markdown
    assert "## Research Guidance" not in markdown
    assert "## Research Directions and Questions" not in markdown
    assert "## Idea Examples" not in markdown
    assert "## Warnings" not in markdown
    assert "missing conclusion text" not in markdown
    assert "## Task Focus for Idea Generation" in markdown
    assert "Compute a tree-level scalar correlator." in markdown
    assert "## Known Solved Cases" in markdown
    assert "Do not propose the same seed correlator as new." in markdown
    assert "## Open Axes for New Work" in markdown
    assert "These axes are examples, not a complete list" in markdown
    assert "discover additional axes" in markdown


def _install_fake_paper_query(monkeypatch):
    global _toc_calls
    _toc_calls = set()
    monkeypatch.setattr(paper, "metadata", _metadata)
    monkeypatch.setattr(paper, "references", _references)
    monkeypatch.setattr(paper, "citers", _citers)
    monkeypatch.setattr(paper, "section", _section)
    monkeypatch.setattr(paper, "toc", _toc)


def _install_fake_domain_summary(monkeypatch):
    monkeypatch.setattr(domain_summary, "run_json", lambda *args, **kwargs: _summary_payload())


def _summary_payload():
    return {
        "schema_version": "arc.domain_summary.v4",
        "domain_title": "Example Domain",
        "brief_introduction": "Compact intro.",
        "task_focus": {
            "user_intent": "intent",
            "research_scope": "Example research scope.",
            "priority_rules": ["Satisfy the user request first."],
        },
        "foundation_paper": {
            "paper_id": FOUNDATION,
            "title": "Foundation Paper",
            "reason": "Best field-defining citer source.",
        },
        "best_reference_paper": {
            "paper_id": "arXiv:2401.00002",
            "title": "Best Ref",
            "reason": "Clear methodology.",
        },
        "methodology": [{"claim": "Use residues.", "papers": ["arXiv:2401.00002"]}],
        "known_solved_cases": [
            {
                "solved_case": "Known solved case.",
                "why_it_is_solved": "The central calculation is already available.",
                "transferable_form": "Use a precise observable and setup.",
                "forbidden_reuse": "Do not propose the same calculation as new.",
                "valid_new_axes": ["new observable"],
                "papers": ["arXiv:2401.00002"],
            }
        ],
        "open_axes_for_new_work": [
            {
                "axis": "Observable",
                "guidance": "Use a genuinely distinct observable.",
                "example_variations": ["new line ratio"],
                "papers": ["arXiv:2401.00002"],
            }
        ],
        "warnings": [],
    }


_toc_calls: set[str] = set()


def _metadata(paper_id, *, refresh=False):
    records = {
        SEED: ("Seed Paper", 2024, 12),
        FOUNDATION: ("Foundation Paper", 2023, 180),
        "arXiv:2301.00002": ("Parent Paper", 2020, 1500),
        "arXiv:2402.00001": ("Domain Paper 1", 2024, 8),
        "arXiv:2402.00002": ("Domain Paper 2", 2024, 7),
        "arXiv:2402.00003": ("Domain Paper 3", 2024, 6),
        "arXiv:2201.00001": ("Common Method Paper", 2022, 30),
    }
    title, year, citations = records.get(paper_id, (paper_id, 2024, 1))
    return {
        "paper_id": paper_id,
        "title": title,
        "abstract": f"Abstract for {title}.",
        "authors": ["Alice A.", "Bob B."],
        "year": year,
        "citation_count": citations,
        "identifiers": {"paper_id": paper_id, "arxiv": paper_id},
    }


def _references(paper_id, *, refresh=False, enrich=False):
    if paper_id == SEED:
        return [_metadata(FOUNDATION), _metadata("arXiv:2301.00002")]
    if paper_id == "arXiv:2501.00001":
        return [_metadata(FOUNDATION), _metadata("arXiv:2301.00002"), _metadata("arXiv:2201.00001")]
    if paper_id == "arXiv:2501.00002":
        return [_metadata(FOUNDATION), _metadata("arXiv:2201.00001")]
    if paper_id in {"arXiv:2402.00001", "arXiv:2402.00002", "arXiv:2402.00003"}:
        if paper_id == "arXiv:2402.00002":
            return [_metadata(FOUNDATION), _metadata("arXiv:2201.00001"), _metadata("arXiv:2402.00001")]
        return [_metadata(FOUNDATION), _metadata("arXiv:2201.00001")]
    return []


def _citers(paper_id, *, refresh=False, limit=1000, sort="mostrecent"):
    if paper_id == SEED:
        return [_metadata("arXiv:2501.00001"), _metadata("arXiv:2501.00002")]
    if paper_id == FOUNDATION:
        return [_metadata("arXiv:2402.00001"), _metadata("arXiv:2402.00002"), _metadata("arXiv:2402.00003")]
    return []


def _section(paper_id, selector, *, refresh=False):
    if "conclusion" in selector:
        return {"section_id": "S9", "title": "Conclusion", "text": f"Open questions remain for {paper_id}."}
    raise RuntimeError("missing")


def _toc(paper_id, *, refresh=False):
    _toc_calls.add(paper_id)
    return [{"section_id": "S1", "title": "Introduction"}]
