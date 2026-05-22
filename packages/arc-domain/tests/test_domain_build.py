from __future__ import annotations

import re

from arc_domain import service
from arc_domain.cache import DomainPaths, domain_id_for, read_json
from arc_domain import paper


SEED = "arXiv:2401.00001"
FOUNDATION = "arXiv:2301.00001"


def test_build_domain_writes_core_artifacts(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_DOMAIN_CACHE", str(tmp_path / "arc-domain"))
    _install_fake_paper_query(monkeypatch)

    result = service.build_domain(SEED, intent="inflation observables", provider="manual", workers=1)

    assert result["ok"] is True
    data = result["data"]
    paths = DomainPaths.for_domain(data["domain_id"])
    assert paths.foundation_selection.exists()
    assert paths.domain_graph.exists()
    assert paths.evidence_pack.exists()
    assert paths.domain_summary.exists()
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
    assert read_json(paths.domain_summary)["summary_method"] == "deterministic_fallback"
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


def test_status_and_cached_summary(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_DOMAIN_CACHE", str(tmp_path / "arc-domain"))
    _install_fake_paper_query(monkeypatch)
    domain_id = domain_id_for(SEED, "intent")

    service.build_domain(SEED, intent="intent", domain_id=domain_id, provider="manual", workers=1)
    status = service.status(domain_id=domain_id)
    summary = service.get_domain_summary(domain_id=domain_id)
    graph = service.get_domain_graph(domain_id=domain_id)

    assert status["ok"] is True
    assert status["data"]["artifacts"]["domain_summary"]["exists"] is True
    assert summary["ok"] is True
    assert graph["ok"] is True


def _install_fake_paper_query(monkeypatch):
    monkeypatch.setattr(paper, "metadata", _metadata)
    monkeypatch.setattr(paper, "references", _references)
    monkeypatch.setattr(paper, "citers", _citers)
    monkeypatch.setattr(paper, "section", _section)


def _metadata(paper_id, *, refresh=False):
    records = {
        SEED: ("Seed Paper", 2024, 12),
        FOUNDATION: ("Foundation Paper", 2023, 80),
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
