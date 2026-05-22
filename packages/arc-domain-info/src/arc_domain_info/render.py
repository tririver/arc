from __future__ import annotations

import html
import json
import math
from typing import Any

from .cache import DomainPaths, read_json, update_status, write_text


ROLE_COLORS = {
    "selected_foundation": "#d64545",
    "parent_foundation": "#8a5cf6",
    "domain_paper": "#2f80ed",
    "common_reference": "#0f9f7f",
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
    positioned = _layout(nodes)
    data = {"nodes": positioned, "edges": edges}
    ranked_rows = "\n".join(_ranked_row(node) for node in sorted(nodes, key=_node_rank_key))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(graph.get("foundation_paper") or "ARC Domain Network")}</title>
<style>
body {{ margin:0; font-family: Arial, sans-serif; color:#1f2933; background:#f7f8fb; }}
header {{ padding:16px 22px; background:#ffffff; border-bottom:1px solid #d7dce3; }}
h1 {{ margin:0; font-size:20px; font-weight:700; }}
.layout {{ display:grid; grid-template-columns:minmax(520px, 1fr) 420px; min-height:calc(100vh - 57px); }}
#graph {{ width:100%; height:calc(100vh - 57px); background:#ffffff; }}
aside {{ border-left:1px solid #d7dce3; background:#fbfcfd; overflow:auto; }}
.panel {{ padding:14px 16px; border-bottom:1px solid #d7dce3; }}
.legend span {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }}
table {{ width:100%; border-collapse:collapse; font-size:12px; }}
th, td {{ padding:8px 10px; border-bottom:1px solid #e2e6ec; text-align:left; vertical-align:top; }}
tr:hover {{ background:#eef5ff; cursor:pointer; }}
.title {{ font-weight:700; }}
.muted {{ color:#5f6b7a; }}
@media (max-width: 980px) {{
  .layout {{ display:block; }}
  #graph {{ height:70vh; }}
  aside {{ border-left:0; border-top:1px solid #d7dce3; }}
}}
</style>
</head>
<body>
<header><h1>ARC Domain Network</h1><div class="muted">Foundation: {html.escape(str(graph.get("foundation_paper") or ""))}</div></header>
<div class="layout">
<svg id="graph" viewBox="0 0 1000 760" role="img" aria-label="Domain citation network"></svg>
<aside>
  <div class="panel legend">
    <div><span style="background:{ROLE_COLORS["selected_foundation"]}"></span>Selected foundation</div>
    <div><span style="background:{ROLE_COLORS["parent_foundation"]}"></span>Parent foundation</div>
    <div><span style="background:{ROLE_COLORS["domain_paper"]}"></span>Domain paper</div>
    <div><span style="background:{ROLE_COLORS["common_reference"]}"></span>Common reference</div>
  </div>
  <div id="details" class="panel">Click a node or row.</div>
  <table>
    <thead><tr><th>Role</th><th>Paper</th><th>Score</th></tr></thead>
    <tbody>{ranked_rows}</tbody>
  </table>
</aside>
</div>
<script id="graph-data" type="application/json">{html.escape(json.dumps(data, ensure_ascii=False))}</script>
<script>
const data = JSON.parse(document.getElementById('graph-data').textContent);
const svg = document.getElementById('graph');
const details = document.getElementById('details');
const byId = Object.fromEntries(data.nodes.map(n => [n.id, n]));
const color = {json.dumps(ROLE_COLORS)};
function el(name, attrs) {{
  const node = document.createElementNS('http://www.w3.org/2000/svg', name);
  Object.entries(attrs || {{}}).forEach(([k,v]) => node.setAttribute(k, v));
  return node;
}}
for (const edge of data.edges) {{
  const a = byId[edge.source], b = byId[edge.target];
  if (!a || !b) continue;
  svg.appendChild(el('line', {{x1:a.x, y1:a.y, x2:b.x, y2:b.y, stroke:'#b5bdc9', 'stroke-width':1.2, opacity:0.62}}));
}}
for (const node of data.nodes) {{
  const group = el('g', {{tabindex:0, role:'button', 'data-id':node.id}});
  const r = node.role === 'selected_foundation' ? 18 : Math.max(7, Math.min(16, 7 + Math.log1p(node.citation_count || 0) * 1.4));
  group.appendChild(el('circle', {{cx:node.x, cy:node.y, r, fill:color[node.role] || '#7b8794', stroke:'#ffffff', 'stroke-width':2}}));
  const label = el('text', {{x:node.x + r + 4, y:node.y + 4, 'font-size':12, fill:'#1f2933'}});
  label.textContent = (node.title || node.id).slice(0, 45);
  group.appendChild(label);
  group.addEventListener('click', () => show(node.id));
  svg.appendChild(group);
}}
document.querySelectorAll('tr[data-id]').forEach(row => row.addEventListener('click', () => show(row.dataset.id)));
function show(id) {{
  const node = byId[id];
  if (!node) return;
  details.innerHTML = `<div class="title">${{escapeHtml(node.title || node.id)}}</div>
    <div class="muted">${{escapeHtml(node.paper_id || node.id)}} | ${{escapeHtml(node.role || '')}}</div>
    <p>${{escapeHtml((node.abstract || '').slice(0, 900))}}</p>
    <div class="muted">Year: ${{node.year || ''}} | Citations: ${{node.citation_count || 0}} | Score: ${{node.domain_score || ''}}</div>`;
}}
function escapeHtml(s) {{ return String(s || '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c])); }}
</script>
</body>
</html>
"""


def _layout(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not nodes:
        return []
    center = (500, 380)
    foundation = [node for node in nodes if node.get("role") == "selected_foundation"]
    others = [node for node in nodes if node.get("role") != "selected_foundation"]
    positioned = []
    for node in foundation:
        item = dict(node)
        item["x"], item["y"] = center
        positioned.append(item)
    count = len(others)
    for index, node in enumerate(others):
        angle = 2 * math.pi * index / max(1, count)
        radius = 255 if node.get("role") == "domain_paper" else 330
        item = dict(node)
        item["x"] = round(center[0] + radius * math.cos(angle), 2)
        item["y"] = round(center[1] + radius * math.sin(angle), 2)
        positioned.append(item)
    return positioned


def _ranked_row(node: dict[str, Any]) -> str:
    return (
        f'<tr data-id="{html.escape(str(node.get("id") or ""), quote=True)}">'
        f"<td>{html.escape(str(node.get('role') or ''))}</td>"
        f"<td><div class=\"title\">{html.escape(str(node.get('title') or node.get('paper_id') or ''))}</div>"
        f"<div class=\"muted\">{html.escape(str(node.get('paper_id') or ''))}</div></td>"
        f"<td>{html.escape(str(node.get('domain_score') or node.get('support_count') or ''))}</td>"
        "</tr>"
    )


def _node_rank_key(node: dict[str, Any]) -> tuple[int, float, int]:
    role_order = {"selected_foundation": 0, "parent_foundation": 1, "domain_paper": 2, "common_reference": 3}
    score = float(node.get("domain_score") or node.get("support_count") or 0)
    citations = int(node.get("citation_count") or 0)
    return (role_order.get(node.get("role"), 9), -score, -citations)
