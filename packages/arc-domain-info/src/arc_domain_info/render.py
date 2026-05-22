from __future__ import annotations

import html
import json
import math
from textwrap import shorten
from typing import Any

from .cache import DomainPaths, read_json, update_status, write_text


ROLE_COLORS = {
    "selected_foundation": "#f4b400",
    "parent_foundation": "#e76f51",
    "domain_paper": "#4f83cc",
    "common_reference": "#43aa8b",
}
ROLE_LABELS = {
    "selected_foundation": "Selected foundation",
    "parent_foundation": "Parent foundation",
    "domain_paper": "Domain paper",
    "common_reference": "Common reference",
}


def render_network_html(*, paths: DomainPaths) -> dict[str, Any]:
    graph = read_json(paths.domain_graph, {})
    html_text = _render(graph)
    write_text(paths.network_html, html_text)
    update_status(paths, stage="html_done", network_html_path=str(paths.network_html))
    return {"domain_id": paths.domain_id, "network_html_path": str(paths.network_html)}


def _render(graph: dict[str, Any]) -> str:
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    data = {"nodes": _vis_nodes(nodes), "edges": _vis_edges(edges)}
    ranked_rows = "\n".join(_ranked_row(node) for node in sorted(nodes, key=_node_rank_key))
    node_count = len(nodes)
    edge_count = len(edges)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(graph.get("foundation_paper") or "ARC Domain Network")}</title>
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<script>
window.MathJax = {{
  tex: {{
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
    processEscapes: true
  }},
  options: {{
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
  }}
}};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<style>
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family: Arial, sans-serif; color:#1f2933; background:#eef2f6; }}
header {{ padding:14px 20px; background:#ffffff; border-bottom:1px solid #d6dde7; }}
h1 {{ margin:0; font-size:20px; font-weight:700; }}
.meta {{ margin-top:4px; color:#5f6b7a; font-size:13px; }}
.layout {{ display:grid; grid-template-columns:minmax(580px, 1fr) 560px; min-height:calc(100vh - 65px); }}
.graph-wrap {{ position:relative; background:#ffffff; min-height:calc(100vh - 65px); overflow:hidden; }}
#mynetwork {{ width:100%; height:calc(100vh - 65px); min-height:560px; background:radial-gradient(circle at 50% 42%, #ffffff 0%, #f8fafc 58%, #edf2f7 100%); }}
#network-message {{ position:absolute; left:18px; bottom:16px; max-width:520px; color:#7b8794; background:rgba(255,255,255,0.92); border:1px solid #d6dde7; border-radius:6px; padding:9px 12px; font-size:12px; display:none; }}
.legend {{ position:absolute; top:12px; left:12px; z-index:4; background:rgba(255,255,255,0.92); border:1px solid #d6dde7; border-radius:6px; padding:7px 9px; font-size:11px; line-height:1.45; box-shadow:0 2px 9px rgba(15,23,42,0.12); }}
.legend span {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }}
.fit-button {{ position:absolute; left:12px; bottom:12px; z-index:5; width:34px; height:34px; border:1px solid #c9d2de; border-radius:6px; background:rgba(255,255,255,0.94); color:#1f2937; font-size:18px; line-height:1; cursor:pointer; box-shadow:0 2px 9px rgba(15,23,42,0.12); }}
.fit-button:hover {{ background:#f2f6fb; }}
.fit-button:focus-visible {{ outline:2px solid #2563eb; outline-offset:2px; }}
aside {{ border-left:1px solid #d6dde7; background:#fbfcfd; overflow:auto; max-height:calc(100vh - 65px); }}
.panel {{ padding:14px 16px; border-bottom:1px solid #dfe5ec; }}
#details {{ min-height:150px; font-size:12px; line-height:1.5; }}
table {{ width:100%; border-collapse:collapse; font-size:12px; table-layout:fixed; }}
th, td {{ padding:8px 10px; border-bottom:1px solid #e2e6ec; text-align:left; vertical-align:top; }}
th:first-child, td:first-child {{ width:96px; }}
th:last-child, td:last-child {{ width:64px; }}
tr:hover, tr.is-selected {{ background:#eaf3ff; cursor:pointer; }}
.title {{ font-weight:700; line-height:1.3; overflow-wrap:anywhere; }}
.muted {{ color:#5f6b7a; }}
.chip {{ display:inline-block; border-radius:999px; padding:2px 7px; color:#ffffff; font-size:11px; line-height:1.4; }}
a {{ color:#2563eb; }}
@media (max-width: 980px) {{
  .layout {{ display:block; }}
  #mynetwork {{ height:70vh; min-height:480px; }}
  aside {{ max-height:none; }}
  aside {{ border-left:0; border-top:1px solid #d7dce3; }}
}}
</style>
</head>
<body>
<header>
  <h1>ARC Domain Network</h1>
  <div class="meta">Foundation: {html.escape(str(graph.get("foundation_paper") or ""))} | Nodes: {node_count} | Edges: {edge_count}</div>
</header>
<div class="layout">
<div class="graph-wrap">
  <div class="legend">
    <b>Legend</b><br>
    <span style="background:{ROLE_COLORS["selected_foundation"]}"></span>Selected foundation<br>
    <span style="background:{ROLE_COLORS["parent_foundation"]}"></span>Parent foundation<br>
    <span style="background:{ROLE_COLORS["domain_paper"]}"></span>Domain paper<br>
    <span style="background:{ROLE_COLORS["common_reference"]}"></span>Common reference
  </div>
  <div id="mynetwork" role="img" aria-label="Interactive domain citation network"></div>
  <button id="fit-network" class="fit-button" type="button" title="Re-scale and re-center" aria-label="Re-scale and re-center">[ ]</button>
  <div id="network-message"></div>
</div>
<aside>
  <div id="details" class="panel">Click a node or row.</div>
  <table>
    <thead><tr><th>Role</th><th>Paper</th><th>Score</th></tr></thead>
    <tbody>{ranked_rows}</tbody>
  </table>
</aside>
</div>
<script id="graph-data" type="application/json">{_script_json(data)}</script>
<script>
(function () {{
  const graphData = JSON.parse(document.getElementById('graph-data').textContent);
  const details = document.getElementById('details');
  const message = document.getElementById('network-message');
  const fitButton = document.getElementById('fit-network');
  const byId = Object.fromEntries(graphData.nodes.map(n => [n.id, n]));
  const rows = Object.fromEntries(Array.from(document.querySelectorAll('tr[data-id]')).map(row => [row.dataset.id, row]));
  let network = null;
  let highlightedEdges = [];

  function showMessage(text) {{
    message.textContent = text;
    message.style.display = 'block';
  }}

  if (typeof vis === 'undefined' || !vis.Network) {{
    showMessage('Interactive graph library failed to load. Check network access for vis-network, or reload the page.');
    return;
  }}

  const nodes = new vis.DataSet(graphData.nodes);
  const edges = new vis.DataSet(graphData.edges);
  const container = document.getElementById('mynetwork');
  const options = {{
    nodes: {{
      shape: 'dot',
      borderWidth: 1,
      borderWidthSelected: 4,
      font: {{ size: 15, face: 'Arial', strokeWidth: 3, strokeColor: '#ffffff' }},
      shadow: {{ enabled: true, color: 'rgba(15,23,42,0.22)', size: 9, x: 2, y: 2 }}
    }},
    edges: {{
      arrows: {{ to: {{ enabled: true, scaleFactor: 0.45 }} }},
      color: {{ color: '#a3adbb', highlight: '#1f2937', hover: '#64748b' }},
      width: 1.2,
      smooth: {{ enabled: true, type: 'dynamic', roundness: 0.45 }}
    }},
    interaction: {{
      hover: true,
      navigationButtons: false,
      dragNodes: true,
      dragView: true,
      zoomView: true,
      multiselect: false
    }},
    physics: {{
      enabled: true,
      solver: 'forceAtlas2Based',
      forceAtlas2Based: {{
        gravitationalConstant: -90,
        centralGravity: 0.018,
        springLength: 165,
        springConstant: 0.045,
        damping: 0.82,
        avoidOverlap: 0.85
      }},
      stabilization: {{ enabled: true, iterations: 500, updateInterval: 25, fit: true }},
      minVelocity: 2,
      maxVelocity: 12
    }},
    layout: {{ improvedLayout: true }}
  }};

  network = new vis.Network(container, {{ nodes, edges }}, options);
  network.once('stabilizationIterationsDone', function () {{
    network.storePositions();
    network.setOptions({{ physics: {{ enabled: false }} }});
    network.fit({{ animation: {{ duration: 450, easingFunction: 'easeInOutQuad' }} }});
  }});
  setTimeout(function () {{
    if (network) {{
      network.storePositions();
      network.setOptions({{ physics: {{ enabled: false }} }});
    }}
  }}, 3500);
  fitButton.addEventListener('click', function () {{
    network.fit({{ animation: {{ duration: 450, easingFunction: 'easeInOutQuad' }} }});
  }});

  network.on('selectNode', function (event) {{
    const id = event.nodes[0];
    show(id);
    markRow(id);
    highlightConnectedEdges(id);
  }});
  network.on('deselectNode', function () {{
    markRow(null);
    restoreEdges();
  }});

  document.querySelectorAll('tr[data-id]').forEach(row => {{
    row.addEventListener('click', () => {{
      const id = row.dataset.id;
      if (!byId[id]) return;
      network.selectNodes([id]);
      show(id);
      markRow(id);
      highlightConnectedEdges(id);
    }});
    row.addEventListener('mouseenter', () => {{
      if (byId[row.dataset.id]) network.selectNodes([row.dataset.id], false);
    }});
  }});

  function markRow(id) {{
    Object.values(rows).forEach(row => row.classList.toggle('is-selected', row.dataset.id === id));
  }}

  function highlightConnectedEdges(nodeId) {{
    restoreEdges();
    if (!network || !nodeId) return;
    highlightedEdges = network.getConnectedEdges(nodeId);
    const updates = highlightedEdges.map(edgeId => {{
      const edge = edges.get(edgeId);
      return {{
        id: edgeId,
        width: Math.max(3.4, Number(edge.width || 1.2) * 2.6),
        color: '#1f2937'
      }};
    }});
    if (updates.length) edges.update(updates);
  }}

  function restoreEdges() {{
    if (!highlightedEdges.length) return;
    const updates = highlightedEdges.map(edgeId => {{
      const edge = edges.get(edgeId);
      if (!edge) return null;
      return {{
        id: edgeId,
        width: edge.relation === 'cites_foundation' ? 2.2 : 1.15,
        color: edge.relation === 'cites_foundation' ? '#7c8796' : '#b0b8c4'
      }};
    }}).filter(Boolean);
    highlightedEdges = [];
    if (updates.length) edges.update(updates);
  }}

  function show(id) {{
    const node = byId[id];
    if (!node) return;
    details.innerHTML = `<div class="title">${{escapeHtml(node.raw_title || node.label || node.id)}}</div>
      <div class="muted">${{escapeHtml(node.paper_id || node.id)}} | ${{escapeHtml(node.role_label || node.role || '')}}</div>
      <div class="muted">Authors: ${{escapeHtml(formatAuthors(node.authors))}}</div>
      <p>${{escapeHtml((node.abstract || '').slice(0, 1000))}}</p>
      <div class="muted">Year: ${{node.year || ''}} | Citations: ${{node.citation_count || 0}}${{scoreText(node)}}</div>
      ${{domainScoreDetails(node)}}
      <div style="margin-top:8px;"><a href="${{arxivHref(node)}}" target="_blank">Open paper</a></div>`;
    typesetMath(details);
  }}

  function scoreText(node) {{
    if (node.role !== 'domain_paper' || !node.domain_score) return '';
    return ` | Score: ${{node.domain_score}}`;
  }}

  function domainScoreDetails(node) {{
    if (node.role !== 'domain_paper') return '';
    return `<div class="muted">Citation/year: ${{node.citation_per_year || ''}} | Graph citers: ${{node.in_graph_citer_count || 0}} | Ref edges: ${{node.reference_edge_count || 0}} | Intent overlap: ${{node.intent_overlap || ''}}</div>`;
  }}

  function typesetMath(element) {{
    if (window.MathJax && MathJax.typesetPromise) {{
      MathJax.typesetPromise([element]).catch(() => {{}});
    }}
  }}

  function formatAuthors(authors) {{
    if (!Array.isArray(authors) || authors.length === 0) return '';
    if (authors.length <= 6) return authors.join(', ');
    return authors.slice(0, 6).join(', ') + ', et al.';
  }}

  function arxivHref(node) {{
    const id = String(node.paper_id || node.id || '');
    if (id.startsWith('arXiv:')) return 'https://arxiv.org/abs/' + encodeURIComponent(id.slice(6));
    return '#';
  }}

  function escapeHtml(s) {{
    return String(s || '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
  }}
}})();
</script>
</body>
</html>
"""


def _ranked_row(node: dict[str, Any]) -> str:
    role = str(node.get("role") or "")
    color = ROLE_COLORS.get(role, "#7b8794")
    score = node.get("domain_score") if role == "domain_paper" else ""
    return (
        f'<tr data-id="{html.escape(str(node.get("id") or ""), quote=True)}">'
        f'<td><span class="chip" style="background:{color};">{html.escape(_role_label(role, compact=True))}</span></td>'
        f"<td><div class=\"title\">{html.escape(str(node.get('title') or node.get('paper_id') or ''))}</div>"
        f"<div class=\"muted\">{html.escape(str(node.get('paper_id') or ''))}</div></td>"
        f"<td>{html.escape(str(score or ''))}</td>"
        "</tr>"
    )


def _node_rank_key(node: dict[str, Any]) -> tuple[int, float, int]:
    role_order = {"selected_foundation": 0, "parent_foundation": 1, "common_reference": 2, "domain_paper": 3}
    score = float(node.get("domain_score") or node.get("support_count") or 0)
    citations = int(node.get("citation_count") or 0)
    return (role_order.get(node.get("role"), 9), -score, -citations)


def _vis_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_vis_node(node) for node in nodes]


def _vis_node(node: dict[str, Any]) -> dict[str, Any]:
    role = str(node.get("role") or "")
    paper_id = str(node.get("paper_id") or node.get("id") or "")
    title = str(node.get("title") or paper_id)
    size = _node_size(node)
    color = ROLE_COLORS.get(role, "#95a5a6")
    border = "#1f2937" if role == "selected_foundation" else "#ffffff"
    return {
        "id": str(node.get("id") or paper_id),
        "paper_id": paper_id,
        "label": shorten(title, width=48, placeholder="..."),
        "raw_title": title,
        "role": role,
        "role_label": _role_label(role),
        "abstract": str(node.get("abstract") or ""),
        "authors": node.get("authors") or [],
        "year": node.get("year"),
        "citation_count": int(node.get("citation_count") or 0),
        "citation_per_year": node.get("citation_per_year"),
        "domain_score": node.get("domain_score"),
        "citation_rate_score": node.get("citation_rate_score"),
        "recency": node.get("recency"),
        "recency_score": node.get("recency_score"),
        "intent_overlap": node.get("intent_overlap"),
        "intent_overlap_score": node.get("intent_overlap_score"),
        "intent_boost": node.get("intent_boost"),
        "in_graph_citer_count": node.get("in_graph_citer_count"),
        "in_graph_citer_score": node.get("in_graph_citer_score"),
        "reference_edge_count": node.get("reference_edge_count"),
        "reference_edge_score": node.get("reference_edge_score"),
        "support_count": node.get("support_count"),
        "title": _tooltip(node),
        "value": size,
        "size": size,
        "shape": "dot",
        "color": {"background": color, "border": border, "highlight": {"background": color, "border": "#111827"}},
        "borderWidth": 3 if role == "selected_foundation" else 1,
        "mass": 2.2 if role in {"selected_foundation", "parent_foundation"} else 1.0,
    }


def _vis_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for index, edge in enumerate(edges):
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if not source or not target:
            continue
        relation = str(edge.get("relation") or "cites")
        out.append(
            {
                "id": f"e{index}",
                "from": source,
                "to": target,
                "source": source,
                "target": target,
                "relation": relation,
                "arrows": "to",
                "width": 2.2 if relation == "cites_foundation" else 1.15,
                "color": "#7c8796" if relation == "cites_foundation" else "#b0b8c4",
            }
        )
    return out


def _node_size(node: dict[str, Any]) -> float:
    role = str(node.get("role") or "")
    if role == "selected_foundation":
        return 42
    if role == "parent_foundation":
        return 32
    if role == "common_reference":
        return min(34, 18 + float(node.get("support_count") or 0) * 0.9)
    citations = int(node.get("citation_count") or 0)
    score = float(node.get("domain_score") or 0)
    return min(34, 13 + math.log1p(max(citations, 0)) * 2.3 + score * 0.7)


def _role_label(role: str, *, compact: bool = False) -> str:
    label = ROLE_LABELS.get(role, role.replace("_", " ").title())
    if compact and label == "Selected foundation":
        return "Foundation"
    if compact and label == "Parent foundation":
        return "Parent"
    if compact and label == "Domain paper":
        return "Domain"
    if compact and label == "Common reference":
        return "Common"
    return label


def _tooltip(node: dict[str, Any]) -> str:
    parts = [
        f"<b>{html.escape(str(node.get('title') or node.get('paper_id') or ''))}</b>",
        html.escape(str(node.get("paper_id") or node.get("id") or "")),
        html.escape(_role_label(str(node.get("role") or ""))),
    ]
    if node.get("year"):
        parts.append(f"Year: {html.escape(str(node.get('year')))}")
    if node.get("citation_count") is not None:
        parts.append(f"Citations: {html.escape(str(node.get('citation_count')))}")
    return "<br>".join(part for part in parts if part)


def _script_json(data: Any) -> str:
    return (
        json.dumps(data, ensure_ascii=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
