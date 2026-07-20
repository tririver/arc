from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

from arc_llm.json_schema import to_provider_json_schema
from arc_llm.host import HostDetection
from arc_llm.runner import LLMConfig, LLMNeedsLLM
from arc_domain import foundation
from arc_domain import evidence
from arc_domain import network
from arc_domain import render
from arc_domain import service
from arc_domain import summary as domain_summary
from arc_domain.cache import DomainPaths, domain_id_for, read_json
from arc_domain import paper
from arc_domain import cache as domain_cache


SEED = "arXiv:2401.00001"
FOUNDATION = "arXiv:2301.00001"


def _assert_openai_strict(node):
    if isinstance(node, dict):
        props = node.get("properties")
        if isinstance(props, dict):
            assert node.get("additionalProperties") is False
            assert set(node.get("required") or []) == set(props)
        for value in node.values():
            _assert_openai_strict(value)
    elif isinstance(node, list):
        for value in node:
            _assert_openai_strict(value)


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


def test_recency_config_change_invalidates_only_downstream_artifacts(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_DOMAIN_CACHE", str(tmp_path / "arc-domain"))
    domain_id = "recency-change"
    first = service.init_domain(SEED, intent="intent", domain_id=domain_id,
                                recent_window_days=365, as_of_date="2026-07-20")
    assert first["ok"] is True
    paths = DomainPaths.for_domain(domain_id)
    paths.foundation_selection.write_text("{}", encoding="utf-8")
    paths.domain_graph.write_text("{}", encoding="utf-8")
    paths.evidence_pack.write_text("{}", encoding="utf-8")

    service._ensure_domain(SEED, intent="intent", domain_id=domain_id,
                           recent_window_days=730, as_of_date="2026-07-20")

    assert paths.foundation_selection.exists()
    assert not paths.domain_graph.exists()
    assert not paths.evidence_pack.exists()
    config = read_json(paths.config)
    assert config["recency"] == {
        "recent_window_days": 730,
        "as_of_date": "2026-07-20",
        "window_start_date": "2024-07-20",
        "window_end_date": "2026-07-20",
    }
    assert read_json(paths.status)["recency"] == config["recency"]
    assert config["input_fingerprint"]["recency_hash"]


def test_calendar_two_year_window_handles_leap_boundaries():
    assert service.calendar_window_days("2024-02-29") == 731
    assert service._recency_config(731, "2024-02-29")["window_start_date"] == "2022-02-28"
    assert service.calendar_window_days("2026-03-01") == 730


def _needs_llm() -> LLMNeedsLLM:
    return LLMNeedsLLM(
        LLMConfig(
            provider="manual",
            model=None,
            host=HostDetection(host="unknown", confidence=0.0, signals=[]),
            signals=[],
        )
    )


def test_summarize_domain_propagates_needs_llm(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_DOMAIN_CACHE", str(tmp_path / "arc-domain"))
    domain_id = "needs-llm-summary"
    paths = DomainPaths.for_domain(domain_id)
    paths.domain_dir.mkdir(parents=True)
    paths.evidence_pack.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(service, "_summarize_domain", lambda **kwargs: (_ for _ in ()).throw(_needs_llm()))

    result = service.summarize_domain(SEED, intent="intent", domain_id=domain_id)

    assert result["ok"] is False
    assert result["status"] == "needs_llm"
    assert result["error"]["code"] == "needs_llm"
    assert result["llm_task"]["provider_resolved"] == "manual"


def test_build_domain_propagates_needs_llm_from_summary(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_DOMAIN_CACHE", str(tmp_path / "arc-domain"))
    monkeypatch.setattr(service, "_identify_foundation", lambda **kwargs: {"selection": {"selected_foundation": {}}})
    monkeypatch.setattr(service, "_build_network", lambda **kwargs: {"node_count": 0, "edge_count": 0, "graph_path": "g"})
    monkeypatch.setattr(service, "render_network_html", lambda **kwargs: {"network_html_path": "h"})
    monkeypatch.setattr(service, "_build_paper_json_pack", lambda **kwargs: {"paper_json_pack_path": "p"})
    monkeypatch.setattr(service, "_build_evidence_pack", lambda **kwargs: {"evidence_pack_path": "e"})
    monkeypatch.setattr(service, "_summarize_domain", lambda **kwargs: (_ for _ in ()).throw(_needs_llm()))

    result = service.build_domain(SEED, intent="intent")

    assert result["ok"] is False
    assert result["status"] == "needs_llm"


def test_domain_cache_prefers_package_override_then_arc_home(monkeypatch, tmp_path):
    monkeypatch.delenv("ARC_DOMAIN_CACHE", raising=False)
    monkeypatch.setenv("ARC_HOME", str(tmp_path / "arc-home"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert domain_cache.cache_root() == tmp_path / "arc-home/cache/arc-domain"
    monkeypatch.setenv("ARC_DOMAIN_CACHE", str(tmp_path / "override"))
    assert domain_cache.cache_root() == tmp_path / "override"


def test_domain_cache_uses_xdg_then_isolated_home_without_checkout_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("ARC_DOMAIN_CACHE", raising=False)
    monkeypatch.delenv("ARC_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "isolated-home"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert domain_cache.cache_root() == tmp_path / "xdg/arc/arc-domain"

    monkeypatch.delenv("XDG_CACHE_HOME")
    assert domain_cache.cache_root() == tmp_path / "isolated-home/.cache/arc/arc-domain"
    assert not domain_cache.cache_root().is_relative_to(Path(__file__).resolve().parents[3])


def test_domain_llm_helpers_pass_model_tier_to_run_json(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_DOMAIN_CACHE", str(tmp_path / "arc-domain"))
    captured = []

    def fake_run_json(prompt, *, schema, provider, model=None, model_tier=None, **kwargs):
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
        return _summary_payload()

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
        ("arc.domain-summary-v5", "auto", None, "high"),
    ]


def test_summarize_domain_valid_json_no_relaxed_warning(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_DOMAIN_CACHE", str(tmp_path / "arc-domain"))
    paths = DomainPaths.for_domain("summary_valid_test")
    paths.domain_dir.mkdir(parents=True)
    paths.domain_graph.write_text('{"nodes":[]}', encoding="utf-8")
    paths.evidence_pack.write_text('{"papers":[]}', encoding="utf-8")
    paths.foundation_selection.write_text(
        json.dumps(
            {
                "selected_foundation": {
                    "paper_id": FOUNDATION,
                    "title": "Foundation",
                    "reason": "selected",
                },
                "best_reference_paper": {
                    "paper_id": FOUNDATION,
                    "title": "Foundation",
                    "reason": "reference",
                },
                "intent": "intent",
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, Any] = {}

    def fake_run_json(prompt, **kwargs):
        captured.update(kwargs)
        return _summary_payload()

    monkeypatch.setattr(domain_summary, "run_json", fake_run_json)

    result = domain_summary.summarize_domain(paths=paths, provider="auto")
    summary = result["summary"]

    assert "validate_schema" not in captured
    assert captured["output_recovery"] == "warn"
    assert summary["summary_method"] == "llm"
    assert summary["schema_version"] == "arc.domain_summary.v5"
    assert summary["mathematical_opportunities"]["well_defined_problems"]
    assert "relaxed_payload" not in summary
    assert "domain_summary_relaxed" not in json.dumps(read_json(paths.status, {}))


def test_summarize_domain_recovers_v4_shape_without_inventing_opportunities(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_DOMAIN_CACHE", str(tmp_path / "arc-domain"))
    paths = DomainPaths.for_domain("summary_v4_recovery_test")
    paths.domain_dir.mkdir(parents=True)
    paths.domain_graph.write_text('{"nodes":[]}', encoding="utf-8")
    paths.evidence_pack.write_text('{"papers":[]}', encoding="utf-8")
    paths.foundation_selection.write_text(
        json.dumps({"selected_foundation": {"paper_id": FOUNDATION, "title": "Foundation", "reason": "selected"}}),
        encoding="utf-8",
    )
    payload = _summary_payload()
    payload["schema_version"] = "arc.domain_summary.v4"
    payload.pop("mathematical_opportunities")
    monkeypatch.setattr(domain_summary, "run_json", lambda *args, **kwargs: payload)

    summary = domain_summary.summarize_domain(paths=paths, provider="auto")["summary"]

    assert summary["schema_version"] == "arc.domain_summary.v5"
    assert summary["summary_method"] == "llm_relaxed"
    assert summary["methodology"]
    assert summary["open_axes_for_new_work"]
    assert summary["mathematical_opportunities"] == {"well_defined_problems": []}
    assert any("domain_summary_schema_relaxed" in warning for warning in summary["warnings"])


def test_summarize_domain_relaxed_v5_preserves_mathematical_opportunities(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_DOMAIN_CACHE", str(tmp_path / "arc-domain"))
    paths = DomainPaths.for_domain("summary_v5_recovery_test")
    paths.domain_dir.mkdir(parents=True)
    paths.domain_graph.write_text('{"nodes":[]}', encoding="utf-8")
    paths.evidence_pack.write_text('{"papers":[]}', encoding="utf-8")
    paths.foundation_selection.write_text(
        json.dumps({"selected_foundation": {"paper_id": FOUNDATION, "title": "Foundation", "reason": "selected"}}),
        encoding="utf-8",
    )
    payload = _summary_payload()
    payload["unexpected"] = "relaxed"
    monkeypatch.setattr(domain_summary, "run_json", lambda *args, **kwargs: payload)

    summary = domain_summary.summarize_domain(paths=paths, provider="auto")["summary"]
    problems = summary["mathematical_opportunities"]["well_defined_problems"]

    assert summary["summary_method"] == "llm_relaxed"
    assert len(problems) == 1
    assert problems[0]["problem"] == "Determine the first unsolved residue constraint."
    assert [method["origin"] for method in problems[0]["available_systematic_methods"]] == [
        "in_domain",
        "external_search_lead",
    ]
    assert summary["relaxed_payload"]["unexpected"] == "relaxed"


def test_summarize_domain_filters_unknown_target_papers_and_drops_unsupported_cards(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_DOMAIN_CACHE", str(tmp_path / "arc-domain"))
    paths = DomainPaths.for_domain("summary_target_evidence_test")
    paths.domain_dir.mkdir(parents=True)
    paths.domain_graph.write_text('{"nodes":[]}', encoding="utf-8")
    paths.evidence_pack.write_text('{"papers":[]}', encoding="utf-8")
    paths.foundation_selection.write_text(
        json.dumps({"selected_foundation": {"paper_id": FOUNDATION, "title": "Foundation", "reason": "selected"}}),
        encoding="utf-8",
    )
    payload = _summary_payload()
    supported = payload["mathematical_opportunities"]["well_defined_problems"][0]
    supported["target_domain_papers"] = [FOUNDATION, "arXiv:9999.99999"]
    unsupported = json.loads(json.dumps(supported))
    unsupported["problem"] = "Unsupported opportunity"
    unsupported["target_domain_papers"] = ["arXiv:8888.88888"]
    payload["mathematical_opportunities"]["well_defined_problems"].append(unsupported)
    monkeypatch.setattr(domain_summary, "run_json", lambda *args, **kwargs: payload)

    summary = domain_summary.summarize_domain(paths=paths, provider="auto")["summary"]
    problems = summary["mathematical_opportunities"]["well_defined_problems"]

    assert summary["summary_method"] == "llm_relaxed"
    assert len(problems) == 1
    assert problems[0]["target_domain_papers"] == [FOUNDATION]
    assert any("unknown_target_domain_papers_filtered" in warning for warning in summary["warnings"])
    assert any("dropped_without_target_evidence" in warning for warning in summary["warnings"])


def test_summarize_domain_relaxed_drops_incomplete_opportunity_without_inventing_evidence_status(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("ARC_DOMAIN_CACHE", str(tmp_path / "arc-domain"))
    paths = DomainPaths.for_domain("summary_incomplete_opportunity_test")
    paths.domain_dir.mkdir(parents=True)
    paths.domain_graph.write_text('{"nodes":[]}', encoding="utf-8")
    paths.evidence_pack.write_text('{"papers":[]}', encoding="utf-8")
    paths.foundation_selection.write_text(
        json.dumps({"selected_foundation": {"paper_id": FOUNDATION, "title": "Foundation", "reason": "selected"}}),
        encoding="utf-8",
    )
    payload = _summary_payload()
    payload["unexpected"] = "relaxed"
    payload["mathematical_opportunities"]["well_defined_problems"][0].pop("evidence_status")
    monkeypatch.setattr(domain_summary, "run_json", lambda *args, **kwargs: payload)

    summary = domain_summary.summarize_domain(paths=paths, provider="auto")["summary"]

    assert summary["mathematical_opportunities"] == {"well_defined_problems": []}
    assert any("invalid_mathematical_opportunity_dropped" in warning for warning in summary["warnings"])


def test_summarize_domain_accepts_extra_keys_with_warning(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_DOMAIN_CACHE", str(tmp_path / "arc-domain"))
    paths = DomainPaths.for_domain("summary_relaxed_test")
    paths.domain_dir.mkdir(parents=True)
    paths.domain_graph.write_text('{"nodes":[]}', encoding="utf-8")
    paths.evidence_pack.write_text('{"papers":[]}', encoding="utf-8")
    paths.foundation_selection.write_text(
        json.dumps(
            {
                "selected_foundation": {
                    "paper_id": FOUNDATION,
                    "title": "Foundation",
                    "reason": "selected",
                },
                "best_reference_paper": {
                    "paper_id": "arXiv:2401.00002",
                    "title": "Best Ref",
                    "reason": "reference",
                },
                "intent": "intent",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        domain_summary,
        "run_json",
        lambda *args, **kwargs: {
            "domain": "DeepSeek Domain",
            "core_methodology": ["Use exchange Witten diagrams."],
            "key_papers": [FOUNDATION],
            "open_axes": ["higher-point correlators"],
            "priority_rules": "satisfy user intent first",
            "sufficient": True,
        },
    )

    result = domain_summary.summarize_domain(paths=paths, provider="auto")
    summary = result["summary"]
    markdown = paths.domain_summary_markdown.read_text(encoding="utf-8")
    status = read_json(paths.status, {})

    assert summary["summary_method"] == "llm_relaxed"
    assert summary["schema_version"] == "arc.domain_summary.v5"
    assert summary["mathematical_opportunities"] == {"well_defined_problems": []}
    assert summary["domain_title"] == "DeepSeek Domain"
    assert summary["task_focus"]["priority_rules"] == ["satisfy user intent first"]
    assert summary["relaxed_payload"]["sufficient"] is True
    assert any("domain_summary_schema_relaxed" in item for item in summary["warnings"])
    assert "## Relaxed LLM Output Warning" in markdown
    assert "DeepSeek Domain" in markdown
    assert "domain_summary_relaxed" in json.dumps(status)


def test_summarize_domain_accepts_plain_text_recovery_with_warning(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_DOMAIN_CACHE", str(tmp_path / "arc-domain"))
    paths = DomainPaths.for_domain("summary_text_test")
    paths.domain_dir.mkdir(parents=True)
    paths.domain_graph.write_text('{"nodes":[]}', encoding="utf-8")
    paths.evidence_pack.write_text('{"papers":[]}', encoding="utf-8")
    paths.foundation_selection.write_text(
        json.dumps(
            {
                "selected_foundation": {
                    "paper_id": FOUNDATION,
                    "title": "Foundation",
                    "reason": "selected",
                },
                "intent": "intent",
            }
        ),
        encoding="utf-8",
    )
    raw_text = "Plain text briefing about massive scalar exchange."

    monkeypatch.setattr(
        domain_summary,
        "run_json",
        lambda *args, **kwargs: {
            "arc_llm_call_record": {
                "structured_output": {
                    "mode": "recovered",
                    "severity": "major",
                    "recovery_strategy": "natural_language_fallback",
                    "warnings": ["provider returned text"],
                    "raw_text_excerpt": raw_text,
                }
            }
        },
    )

    result = domain_summary.summarize_domain(paths=paths, provider="auto")
    summary = result["summary"]
    markdown = paths.domain_summary_markdown.read_text(encoding="utf-8")

    assert summary["summary_method"] == "llm_relaxed_text"
    assert summary["mathematical_opportunities"] == {"well_defined_problems": []}
    assert raw_text in summary["brief_introduction"]
    assert raw_text in markdown
    assert any("domain_summary_structured_recovery" in item for item in summary["warnings"])


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
    paths.domain_summary.write_text(
        '{"schema_version": "arc.domain_summary.v4", "summary_method": "llm", "overview": "stale"}',
        encoding="utf-8",
    )
    paths.domain_summary_markdown.write_text("# stale\n", encoding="utf-8")

    def fail_summary(*args, **kwargs):
        raise RuntimeError("summary prompt too large")

    monkeypatch.setattr(domain_summary, "run_json", fail_summary)

    result = service.summarize_domain(SEED, intent="intent", domain_id=domain_id, provider="auto")

    assert result["ok"] is True
    assert result["data"]["summary_available"] is False
    assert result["data"]["summary"] is None
    assert result["data"]["domain_summary_path"] is None
    assert result["data"]["domain_summary_markdown_path"] is None
    assert result["data"]["warnings"]
    assert result["data"]["warnings"][0]["code"] == "domain_summary_llm_failed"
    assert "summary prompt too large" in result["data"]["warnings"][0]["message"]
    assert not paths.domain_summary.exists()
    assert not paths.domain_summary_markdown.exists()
    status = read_json(paths.status)
    assert status["stage"] == "summary_warning_no_summary"
    assert status["summary_available"] is False
    assert status["domain_summary_path"] is None
    assert status["domain_summary_markdown_path"] is None
    assert status["warnings"]
    stale_read = service.get_domain_summary(domain_id=domain_id)
    assert stale_read["ok"] is False
    assert stale_read["error"]["code"] == "domain_summary_not_available"


def test_build_domain_continues_when_llm_summary_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_DOMAIN_CACHE", str(tmp_path / "arc-domain"))

    monkeypatch.setattr(service, "_identify_foundation", lambda **kwargs: {"selection": {"selected_foundation": {}}})
    monkeypatch.setattr(
        service,
        "_build_network",
        lambda **kwargs: {"node_count": 1, "edge_count": 0, "graph_path": "graph.json"},
    )
    monkeypatch.setattr(service, "render_network_html", lambda **kwargs: {"network_html_path": "network.html"})
    monkeypatch.setattr(service, "_build_paper_json_pack", lambda **kwargs: {"paper_json_pack_path": "pack.json"})
    monkeypatch.setattr(service, "_build_evidence_pack", lambda **kwargs: {"evidence_pack_path": "evidence.json"})

    def fail_summary(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(service, "_summarize_domain", fail_summary)

    result = service.build_domain(SEED, intent="intent", provider="auto", workers=1)

    assert result["ok"] is True
    data = result["data"]
    paths = DomainPaths.for_domain(data["domain_id"])
    assert data["domain_summary_path"] is None
    assert data["domain_summary_markdown_path"] is None
    assert data["summary"] is None
    assert data["summary_available"] is False
    assert data["warnings"][0]["code"] == "domain_summary_failed"
    assert data["paper_json_pack_path"] == "pack.json"
    assert data["evidence_pack_path"] == "evidence.json"
    assert not paths.domain_summary.exists()
    assert not paths.domain_summary_markdown.exists()


def test_build_domain_marks_summary_unavailable_when_summary_returns_no_summary(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_DOMAIN_CACHE", str(tmp_path / "arc-domain"))

    monkeypatch.setattr(service, "_identify_foundation", lambda **kwargs: {"selection": {"selected_foundation": {}}})
    monkeypatch.setattr(
        service,
        "_build_network",
        lambda **kwargs: {"node_count": 1, "edge_count": 0, "graph_path": "graph.json"},
    )
    monkeypatch.setattr(service, "render_network_html", lambda **kwargs: {"network_html_path": "network.html"})
    monkeypatch.setattr(service, "_build_paper_json_pack", lambda **kwargs: {"paper_json_pack_path": "pack.json"})
    monkeypatch.setattr(service, "_build_evidence_pack", lambda **kwargs: {"evidence_pack_path": "evidence.json"})
    monkeypatch.setattr(
        service,
        "_summarize_domain",
        lambda **kwargs: {
            "domain_summary_path": None,
            "domain_summary_markdown_path": None,
            "summary": None,
            "summary_available": False,
            "warnings": [{"code": "domain_summary_llm_failed", "message": "boom"}],
        },
    )

    result = service.build_domain(SEED, intent="intent", provider="auto", workers=1)

    assert result["ok"] is True
    assert result["data"]["summary_available"] is False
    assert result["data"]["summary"] is None
    assert result["data"]["warnings"][0]["code"] == "domain_summary_llm_failed"


def test_get_domain_summary_accepts_v4_as_read_only_legacy_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_DOMAIN_CACHE", str(tmp_path / "arc-domain"))
    domain_id = domain_id_for(SEED, "intent")
    paths = DomainPaths.for_domain(domain_id)
    paths.domain_dir.mkdir(parents=True, exist_ok=True)
    cached = {"schema_version": "arc.domain_summary.v4", "summary_method": "llm", "domain_title": "Legacy"}
    paths.domain_summary.write_text(json.dumps(cached), encoding="utf-8")
    before = paths.domain_summary.read_text(encoding="utf-8")

    result = service.get_domain_summary(domain_id=domain_id)

    assert result["ok"] is True
    assert result["data"]["summary"] == cached
    assert result["data"]["summary_schema_version"] == "arc.domain_summary.v4"
    assert result["data"]["summary_capabilities"] == {"mathematical_opportunities": False}
    assert paths.domain_summary.read_text(encoding="utf-8") == before


def test_get_domain_summary_reports_v5_mathematical_opportunity_capability(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_DOMAIN_CACHE", str(tmp_path / "arc-domain"))
    domain_id = domain_id_for(SEED, "intent")
    paths = DomainPaths.for_domain(domain_id)
    paths.domain_dir.mkdir(parents=True, exist_ok=True)
    cached = {
        "schema_version": "arc.domain_summary.v5",
        "summary_method": "llm",
        "mathematical_opportunities": {"well_defined_problems": []},
    }
    paths.domain_summary.write_text(json.dumps(cached), encoding="utf-8")

    result = service.get_domain_summary(domain_id=domain_id)

    assert result["ok"] is True
    assert result["data"]["summary_schema_version"] == "arc.domain_summary.v5"
    assert result["data"]["summary_capabilities"] == {"mathematical_opportunities": True}


@pytest.mark.parametrize(
    "mathematical_opportunities",
    [None, [], {}, {"well_defined_problems": {}}, {"well_defined_problems": [{}]}],
)
def test_get_domain_summary_rejects_malformed_v5_mathematical_opportunities(
    monkeypatch,
    tmp_path,
    mathematical_opportunities,
):
    monkeypatch.setenv("ARC_DOMAIN_CACHE", str(tmp_path / "arc-domain"))
    domain_id = domain_id_for(SEED, f"malformed-{mathematical_opportunities}")
    paths = DomainPaths.for_domain(domain_id)
    paths.domain_dir.mkdir(parents=True, exist_ok=True)
    cached = {
        "schema_version": "arc.domain_summary.v5",
        "summary_method": "llm",
        "mathematical_opportunities": mathematical_opportunities,
    }
    paths.domain_summary.write_text(json.dumps(cached), encoding="utf-8")

    result = service.get_domain_summary(domain_id=domain_id)

    assert result["ok"] is False
    assert result["error"]["code"] == "domain_summary_invalid"
    assert "invalid mathematical_opportunities contract" in result["error"]["message"]


@pytest.mark.parametrize("schema_version", [None, "arc.domain_summary.v3", "arc.domain_summary.v6"])
def test_get_domain_summary_rejects_missing_or_unknown_schema(monkeypatch, tmp_path, schema_version):
    monkeypatch.setenv("ARC_DOMAIN_CACHE", str(tmp_path / "arc-domain"))
    domain_id = domain_id_for(SEED, f"intent-{schema_version}")
    paths = DomainPaths.for_domain(domain_id)
    paths.domain_dir.mkdir(parents=True, exist_ok=True)
    cached = {"summary_method": "llm"}
    if schema_version is not None:
        cached["schema_version"] = schema_version
    paths.domain_summary.write_text(json.dumps(cached), encoding="utf-8")

    result = service.get_domain_summary(domain_id=domain_id)

    assert result["ok"] is False
    assert result["error"]["code"] == "domain_summary_invalid"
    assert "unsupported schema version" in result["error"]["message"]


@pytest.mark.parametrize("schema_version", ["arc.domain_summary.v4", "arc.domain_summary.v5"])
def test_get_domain_summary_rejects_cached_deterministic_fallback(monkeypatch, tmp_path, schema_version):
    monkeypatch.setenv("ARC_DOMAIN_CACHE", str(tmp_path / "arc-domain"))
    domain_id = domain_id_for(SEED, "intent")
    paths = DomainPaths.for_domain(domain_id)
    paths.domain_dir.mkdir(parents=True, exist_ok=True)
    paths.domain_summary.write_text(
        json.dumps({"schema_version": schema_version, "summary_method": "deterministic_fallback"}),
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


def test_foundation_selection_provider_schema_is_openai_strict():
    schema = to_provider_json_schema(foundation.FOUNDATION_SELECTION_SCHEMA)

    _assert_openai_strict(schema)


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


def test_candidate_audit_relaxed_call_records_schema_warning(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_run_json(prompt, **kwargs):
        captured.update(kwargs)
        return {
            "sufficient": False,
            "candidate_set_sufficient": False,
            "confidence": "complete",
            "search_queries": [
                {
                    "query": "missing same-scope foundation",
                    "reason": "complete evidence",
                    "confidence": "complete",
                }
            ],
            "citation_directions": ["references"],
            "reasoning": "extra key should not discard useful audit output",
            "warnings": [],
        }

    monkeypatch.setattr(foundation, "run_json", fake_run_json)

    audit = foundation._llm_audit_candidates(
        seed_metadata={"paper_id": SEED, "title": "Seed"},
        candidates=[{"paper_id": FOUNDATION, "title": "Foundation"}],
        intent="intent",
        provider="auto",
        model=None,
    )

    assert "validate_schema" not in captured
    assert captured["output_recovery"] == "warn"
    assert audit["audit_method"] == "llm_relaxed"
    assert audit["search_queries"][0]["query"] == "missing same-scope foundation"
    assert any("candidate_audit_schema_relaxed" in warning for warning in audit["warnings"])


def test_relaxed_foundation_audit_string_false_is_false():
    audit = foundation._repair_candidate_audit(
        {
            "candidate_set_sufficient": "false",
            "confidence": "complete",
            "search_queries": [
                {
                    "query": "confident missing foundation",
                    "reason": "complete certainty",
                    "confidence": "complete",
                }
            ],
            "citation_directions": [],
            "reasoning": "string false should remain false",
            "warnings": [],
        },
        method="llm_relaxed",
    )

    assert audit["candidate_set_sufficient"] is False
    assert audit["search_queries"][0]["query"] == "confident missing foundation"


def test_foundation_selection_relaxed_call_uses_foundation_paper_mapping(monkeypatch):
    captured: dict[str, Any] = {}
    low_candidate = {
        "paper_id": "arXiv:2401.00002",
        "title": "LLM Chosen Lower Support",
        "citation_count": 10,
        "witness_citation_overlap": 1,
        "intent_overlap": 0.9,
    }
    high_candidate = {
        "paper_id": FOUNDATION,
        "title": "Deterministic Higher Support",
        "citation_count": 500,
        "witness_citation_overlap": 9,
        "intent_overlap": 0.1,
    }

    def fake_run_json(prompt, **kwargs):
        captured.update(kwargs)
        return {
            "foundation_paper": {
                "paper_id": low_candidate["paper_id"],
                "title": low_candidate["title"],
                "reason": "DeepSeek used natural key",
            },
            "parent_foundations": [],
            "rejected_candidates": [],
            "reasoning": "missing schema_version should repair",
            "warnings": [],
        }

    monkeypatch.setattr(foundation, "run_json", fake_run_json)

    selection = foundation._llm_select_foundation(
        seed_metadata={"paper_id": SEED, "title": "Seed"},
        candidates=[low_candidate, high_candidate],
        intent="intent",
        provider="auto",
        model=None,
    )

    assert "validate_schema" not in captured
    assert captured["output_recovery"] == "warn"
    assert selection["selected_foundation"]["paper_id"] == low_candidate["paper_id"]
    assert selection["selection_method"] == "llm_relaxed"


def test_intent_ranking_relaxed_non_list_ids_do_not_fallback(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_run_json(prompt, **kwargs):
        captured.update(kwargs)
        return {"ranked_paper_ids": FOUNDATION, "reasoning": "string ids are malformed but recoverable"}

    monkeypatch.setattr(network, "run_json", fake_run_json)

    ranking = network._rank_by_intent(
        [{"paper_id": FOUNDATION, "title": "Foundation", "abstract": "scalar exchange", "citation_count": 5}],
        intent="scalar exchange",
        provider="auto",
        model=None,
    )

    assert "validate_schema" not in captured
    assert captured["output_recovery"] == "warn"
    assert ranking["method"] == "llm_relaxed"
    assert ranking["ranked_paper_ids"] == []
    assert "ranked_paper_ids was not a list" in ranking["reasoning"]


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


def test_network_caps_total_papers_including_recent_arxiv(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_DOMAIN_CACHE", str(tmp_path / "arc-domain"))
    paths = DomainPaths.for_domain("paper_cap_test")
    paths.domain_dir.mkdir(parents=True)
    parent_ids = [f"arXiv:2301.0000{index}" for index in range(2, 5)]
    paths.foundation_selection.write_text(
        json.dumps(
            {
                "selected_foundation": {"paper_id": FOUNDATION, "title": "Foundation", "reason": "seed"},
                "parent_foundations": [
                    {"paper_id": paper_id, "title": f"Parent {index}", "reason": "parent"}
                    for index, paper_id in enumerate(parent_ids, start=1)
                ],
            }
        ),
        encoding="utf-8",
    )

    def metadata(paper_id, *, refresh=False):
        return {
            "paper_id": paper_id,
            "title": paper_id,
            "abstract": f"Abstract for {paper_id}",
            "authors": [],
            "year": 2026,
            "published": "2026-05-01",
            "citation_count": int(paper_id.rsplit(".", 1)[-1]) if paper_id.startswith("arXiv:2605.") else 1,
            "identifiers": {"paper_id": paper_id, "arxiv": paper_id},
        }

    def citers(paper_id, *, refresh=False, limit=1000, sort="mostrecent"):
        if paper_id != FOUNDATION:
            return []
        return [metadata(f"arXiv:2605.{index:05d}") for index in range(140)]

    monkeypatch.setattr(network.paper, "metadata", metadata)
    monkeypatch.setattr(network.paper, "citers", citers)
    monkeypatch.setattr(network.paper, "references", lambda paper_id, **kwargs: [metadata(FOUNDATION)])

    result = network.build_network(
        seed_paper=SEED,
        intent="",
        paths=paths,
        provider="manual",
        workers=1,
    )

    assert result["node_count"] == 90
    roles = [node["role"] for node in result["graph"]["nodes"]]
    assert roles.count("selected_foundation") == 1
    assert roles.count("parent_foundation") == 3
    assert roles.count("domain_paper") == 86


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
    assert "mathematical_opportunities.well_defined_problems" in prompt
    assert "at most 6" in prompt
    assert "at most 3" in prompt
    assert "external_search_lead" in prompt
    assert "not complete proposals" in prompt.lower()
    assert "do not invent external citations" in prompt.lower()
    assert "llm_summary" not in prompt
    assert "discover additional axes" in prompt.lower()
    assert "idea examples" not in prompt.lower()
    assert "research directions" not in prompt.lower()
    assert "open questions" not in prompt.lower()
    assert "report_remarks" not in domain_summary.DOMAIN_SUMMARY_SCHEMA["properties"]
    assert "foundation_paper" in domain_summary.DOMAIN_SUMMARY_SCHEMA["required"]
    assert "mathematical_opportunities" in domain_summary.DOMAIN_SUMMARY_SCHEMA["required"]
    assert "research_directions_and_questions" not in domain_summary.DOMAIN_SUMMARY_SCHEMA["properties"]
    assert "idea_examples" not in domain_summary.DOMAIN_SUMMARY_SCHEMA["properties"]
    assert "open_questions" not in domain_summary.DOMAIN_SUMMARY_SCHEMA["properties"]
    assert "mainstream_directions" not in domain_summary.DOMAIN_SUMMARY_SCHEMA["properties"]
    assert "frequently_asked_questions" not in domain_summary.DOMAIN_SUMMARY_SCHEMA["properties"]
    assert "research_guidance" not in domain_summary.DOMAIN_SUMMARY_SCHEMA["properties"]
    assert "reading_guide" not in domain_summary.DOMAIN_SUMMARY_SCHEMA["properties"]
    assert not hasattr(domain_summary, "_fallback_summary")


def test_domain_summary_provider_schema_is_openai_strict():
    schema = to_provider_json_schema(domain_summary.DOMAIN_SUMMARY_SCHEMA)

    assert schema["$id"] == "arc.domain-summary-v5"
    assert schema["properties"]["schema_version"]["const"] == "arc.domain_summary.v5"
    problems = schema["properties"]["mathematical_opportunities"]["properties"]["well_defined_problems"]
    assert problems["maxItems"] == 6
    methods = problems["items"]["properties"]["available_systematic_methods"]
    assert methods["maxItems"] == 3
    assert methods["items"]["properties"]["origin"]["enum"] == ["in_domain", "external_search_lead"]
    assert problems["items"]["properties"]["evidence_status"]["enum"] == [
        "source_explicit",
        "source_grounded_inference",
    ]
    _assert_openai_strict(schema)


def test_domain_summary_prompt_compacts_large_evidence_and_warnings():
    huge_warning = "selection failed " + ("candidate details " * 20000)
    long_abstract = "abstract sentence. " * 8000
    long_conclusion = "conclusion sentence. " * 8000
    papers = [
        {
            "paper_id": f"arXiv:2401.{index:05d}",
            "role": "domain_paper",
            "title": f"Paper {index}",
            "abstract": long_abstract,
            "conclusion": {"text": long_conclusion},
            "warnings": [huge_warning],
        }
        for index in range(90)
    ]

    prompt = domain_summary._summary_prompt(
        graph={
            "foundation_paper": FOUNDATION,
            "nodes": [
                {
                    "paper_id": item["paper_id"],
                    "role": item["role"],
                    "title": item["title"],
                    "year": 2024,
                    "citation_count": index,
                    "selection_reason": "selected",
                }
                for index, item in enumerate(papers)
            ],
            "edges": [{"source": "a", "target": "b", "width": 1}] * 400,
        },
        evidence={"papers": papers, "warnings": [huge_warning]},
        selection={
            "selected_foundation": {"paper_id": FOUNDATION, "title": "Foundation Paper", "reason": "seed"},
            "best_reference_paper": {"paper_id": "arXiv:2401.00002", "title": "Best Ref", "reason": "clear"},
            "warnings": [huge_warning],
        },
    )

    assert len(prompt) < 1_000_000
    assert "[truncated]" in prompt
    assert prompt.count("candidate details") < 1000
    assert prompt.count("abstract sentence.") < 10000


def test_domain_summary_prompt_caps_detailed_papers_with_global_budget():
    long_abstract = "abstract sentence. " * 8000
    long_conclusion = "conclusion sentence. " * 8000
    papers = [
        {
            "paper_id": f"arXiv:2401.{index:05d}",
            "role": "domain_paper",
            "title": f"Paper {index}",
            "abstract": long_abstract,
            "conclusion": {"text": long_conclusion},
            "warnings": [],
        }
        for index in range(355)
    ]
    graph = {
        "foundation_paper": FOUNDATION,
        "nodes": [
            {
                "paper_id": item["paper_id"],
                "role": item["role"],
                "title": item["title"],
                "year": 2024,
                "citation_count": index,
                "selection_reason": "selected",
            }
            for index, item in enumerate(papers)
        ],
        "edges": [{"source": "a", "target": "b", "width": 1}] * 943,
    }
    selection = {
        "selected_foundation": {"paper_id": FOUNDATION, "title": "Foundation Paper", "reason": "seed"},
        "best_reference_paper": {"paper_id": "arXiv:2401.00002", "title": "Best Ref", "reason": "clear"},
    }

    compact = domain_summary._compact_summary_evidence(
        graph=graph,
        evidence={"papers": papers, "warnings": []},
        selection=selection,
        paper_limit=150,
        graph_node_limit=150,
        abstract_limit=domain_summary.SUMMARY_ABSTRACT_CHAR_LIMIT,
        conclusion_limit=domain_summary.SUMMARY_CONCLUSION_CHAR_LIMIT,
    )

    prompt = domain_summary._summary_prompt(
        graph=graph,
        evidence={"papers": papers, "warnings": []},
        selection=selection,
    )

    assert len(compact["papers"]) == 150
    assert compact["omitted_detail_counts"]["omitted_paper_count"] == 205
    assert len(compact["graph"]["nodes"]) == 150
    assert compact["graph"]["omitted_node_count"] == 205
    assert len(prompt) < 900_000
    assert "paper_detail_limit" in prompt
    assert "omitted_detail_counts" in prompt
    assert prompt.count("abstract sentence.") < 20_000


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
            "mathematical_opportunities": _summary_payload()["mathematical_opportunities"],
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
    assert "## Mathematical Opportunities" in markdown
    assert "Determine the first unsolved residue constraint." in markdown
    assert "Importance: It separates competing analytic structures." in markdown
    assert "Creative telescoping (external search lead)" in markdown
    assert "External-search methods are leads" in markdown
    assert "Bounded first calculation: Compute the first three nontrivial residues." in markdown
    assert "Kill criterion: Stop if the recurrence fails at the third residue." in markdown
    assert "Evidence status: source_grounded_inference" in markdown
    assert "## Known Solved Cases" in markdown
    assert "Do not propose the same seed correlator as new." in markdown
    assert "## Open Axes for New Work" in markdown
    assert "These axes are examples, not a complete list" in markdown
    assert "discover additional axes" in markdown


def test_domain_summary_markdown_accepts_legacy_v4_without_empty_opportunity_section():
    markdown = domain_summary.render_summary_markdown(
        {
            "schema_version": "arc.domain_summary.v4",
            "domain_title": "Legacy Domain",
            "brief_introduction": "Legacy briefing.",
            "methodology": [{"claim": "Use a legacy method.", "papers": []}],
        }
    )

    assert markdown.startswith("# Legacy Domain")
    assert "Use a legacy method." in markdown
    assert "## Mathematical Opportunities" not in markdown


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
        "schema_version": "arc.domain_summary.v5",
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
        "mathematical_opportunities": {
            "well_defined_problems": [
                {
                    "problem": "Determine the first unsolved residue constraint.",
                    "importance": "It separates competing analytic structures.",
                    "mathematical_object": "A meromorphic correlator residue.",
                    "assumptions_and_regime": ["tree level", "fixed external weights"],
                    "success_criterion": "Derive a closed constraint and reproduce a known limit.",
                    "available_systematic_methods": [
                        {
                            "method": "Residue recursion",
                            "origin": "in_domain",
                            "source_area": "Analytic correlators",
                            "required_adaptation": "Extend the recursion to the new residue family.",
                            "applicability_conditions": ["isolated simple poles"],
                            "validation_checks": ["recover the four-point limit"],
                        },
                        {
                            "method": "Creative telescoping",
                            "origin": "external_search_lead",
                            "source_area": "Symbolic summation",
                            "required_adaptation": "Map the residue sum to a holonomic sequence.",
                            "applicability_conditions": ["a finite recurrence exists"],
                            "validation_checks": ["compare low-order residues"],
                        },
                    ],
                    "bounded_first_calculation": "Compute the first three nontrivial residues.",
                    "feasibility": {
                        "ready_inputs": ["known seed correlator"],
                        "blocking_unknowns": ["large-order convergence"],
                        "kill_criterion": "Stop if the recurrence fails at the third residue.",
                    },
                    "target_domain_papers": [FOUNDATION],
                    "evidence_status": "source_grounded_inference",
                }
            ]
        },
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
