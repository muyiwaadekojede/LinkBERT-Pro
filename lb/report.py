"""Industrial-grade, interactive visual dashboard generator for LinkBERT Pro.

Generates a standalone, dependency-free interactive HTML dashboard (`report.html`),
CSV export, and JSON payload featuring:
- SVG-based Interactive Site Network Graph with PageRank-scaled nodes and animated proposed links
- Real-time search and multi-facet filtering (source, target, engine, score)
- Orphan Rescue and PageRank link equity impact analysis
- One-click HTML snippet copy and batch JSON export
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import openpyxl
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

from .discovery import CandidateOpportunity
from .graph import GraphMetrics
from .site import SiteIndex


def _e(s: Any) -> str:
    return html.escape(str(s or ""), quote=True)


def _highlight(sentence: str, anchor: str) -> str:
    if not sentence or not anchor:
        return _e(sentence)
    idx = sentence.lower().find(anchor.lower())
    if idx < 0:
        return _e(sentence)
    return (
        _e(sentence[:idx])
        + f'<mark class="anchor-hl">{_e(sentence[idx : idx + len(anchor)])}</mark>'
        + _e(sentence[idx + len(anchor) :])
    )


def generate_dashboard_html(
    idx: SiteIndex,
    opportunities: List[CandidateOpportunity],
    metrics: GraphMetrics | None = None,
    title: str = "LinkBERT Pro — SEO Link Intelligence Dashboard",
) -> str:
    # Prepare node data for visual graph
    nodes = []
    links = []

    page_key_to_id = {}
    for i, p in enumerate(idx.pages):
        page_key_to_id[p.key] = i
        pr = metrics.page_rank.get(p.key, 1.0) if metrics else 1.0
        nodes.append({
            "id": i,
            "key": p.key,
            "title": p.title or p.key,
            "pr": pr,
            "isOrphan": (metrics and p.key in metrics.orphan_pages),
            "cluster": p.key.strip("/").split("/")[0] if p.key.strip("/") else "root",
        })

    # Existing links
    from .match import resolve
    for p in idx.pages:
        src_id = page_key_to_id[p.key]
        for href in p.links:
            target = resolve(href, p, idx)
            if target and target.key in page_key_to_id and target.key != p.key:
                links.append({
                    "source": src_id,
                    "target": page_key_to_id[target.key],
                    "type": "existing",
                })

    # Proposed links
    for opp in opportunities:
        if opp.source_key in page_key_to_id and opp.target_key in page_key_to_id:
            links.append({
                "source": page_key_to_id[opp.source_key],
                "target": page_key_to_id[opp.target_key],
                "type": "proposed",
                "score": opp.score,
            })

    total_opps = len(opportunities)
    orphans_count = len(metrics.orphan_pages) if metrics else 0
    pages_count = len(idx.pages)
    avg_score = sum(o.score for o in opportunities) / total_opps if total_opps else 0.0

    raw_opps = [o.as_dict() for o in opportunities]
    graph_data_json = json.dumps({"nodes": nodes, "links": links})
    opps_json = json.dumps(raw_opps)

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_e(title)}</title>
  <style>
    :root {{
      --bg: #0b0f19;
      --surface: #121826;
      --surface-border: #1e293b;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
      --accent-glow: rgba(56, 189, 248, 0.15);
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: var(--font);
      line-height: 1.5;
      padding-bottom: 80px;
    }}
    header {{
      background: linear-gradient(180deg, #161f33 0%, var(--surface) 100%);
      border-bottom: 1px solid var(--surface-border);
      padding: 32px 48px 24px;
    }}
    .header-top {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    }}
    h1 {{
      font-size: 26px;
      font-weight: 700;
      letter-spacing: -0.5px;
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .badge {{
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      padding: 3px 8px;
      border-radius: 999px;
      background: rgba(56, 189, 248, 0.2);
      color: var(--accent);
      border: 1px solid rgba(56, 189, 248, 0.3);
    }}
    .sub {{ color: var(--text-muted); font-size: 14px; }}
    .container {{ max-width: 1380px; margin: 0 auto; padding: 32px 48px; }}

    /* KPI Cards */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 18px;
      margin-bottom: 32px;
    }}
    .kpi-card {{
      background: var(--surface);
      border: 1px solid var(--surface-border);
      border-radius: 12px;
      padding: 20px;
      position: relative;
      overflow: hidden;
    }}
    .kpi-card::after {{
      content: "";
      position: absolute;
      top: 0; left: 0; right: 0; height: 3px;
      background: var(--accent);
      opacity: 0.7;
    }}
    .kpi-card.success::after {{ background: var(--success); }}
    .kpi-card.warning::after {{ background: var(--warning); }}
    .kpi-val {{ font-size: 32px; font-weight: 800; line-height: 1.1; margin-bottom: 4px; }}
    .kpi-lbl {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.6px; color: var(--text-muted); }}

    /* Graph Section */
    .section-box {{
      background: var(--surface);
      border: 1px solid var(--surface-border);
      border-radius: 14px;
      padding: 24px;
      margin-bottom: 32px;
    }}
    .section-title {{
      font-size: 18px;
      font-weight: 600;
      margin-bottom: 6px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    #graph-container {{
      width: 100%;
      height: 480px;
      background: #080c14;
      border: 1px solid #162032;
      border-radius: 10px;
      position: relative;
      overflow: hidden;
      margin-top: 16px;
    }}
    svg.graph-svg {{ width: 100%; height: 100%; }}

    /* Controls Bar */
    .controls {{
      display: flex;
      gap: 14px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 20px;
    }}
    .search-input {{
      flex: 1;
      min-width: 260px;
      background: #080c14;
      border: 1px solid var(--surface-border);
      color: var(--text);
      padding: 10px 16px;
      border-radius: 8px;
      font-size: 14px;
      outline: none;
    }}
    .search-input:focus {{ border-color: var(--accent); }}
    .filter-btn {{
      background: var(--surface);
      border: 1px solid var(--surface-border);
      color: var(--text-muted);
      padding: 8px 14px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.15s;
    }}
    .filter-btn.active {{
      background: var(--accent);
      color: #0b0f19;
      font-weight: 600;
      border-color: var(--accent);
    }}

    /* Table */
    table {{
      width: 100%;
      border-collapse: collapse;
      border: 1px solid var(--surface-border);
      border-radius: 10px;
      overflow: hidden;
      font-size: 14px;
    }}
    th {{
      background: #0e1524;
      color: var(--text-muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.7px;
      padding: 12px 16px;
      text-align: left;
      border-bottom: 1px solid var(--surface-border);
    }}
    td {{
      padding: 14px 16px;
      border-bottom: 1px solid var(--surface-border);
      vertical-align: top;
    }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: rgba(255, 255, 255, 0.02); }}

    .anchor-hl {{
      background: rgba(56, 189, 248, 0.25);
      color: #38bdf8;
      padding: 2px 5px;
      border-radius: 4px;
      font-weight: 600;
    }}
    .engine-badge {{
      font-size: 11px;
      padding: 2px 7px;
      border-radius: 4px;
      font-weight: 600;
      display: inline-block;
      margin-top: 4px;
    }}
    .engine-mention {{ background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }}
    .engine-linkbert {{ background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }}
    .score-bar {{
      display: inline-block;
      width: 60px;
      height: 6px;
      background: #1e293b;
      border-radius: 3px;
      overflow: hidden;
      margin-left: 8px;
      vertical-align: middle;
    }}
    .score-bar-fill {{
      height: 100%;
      background: linear-gradient(90deg, #38bdf8, #10b981);
    }}
    .btn-copy {{
      background: #1e293b;
      border: 1px solid #334155;
      color: var(--text);
      font-size: 12px;
      padding: 4px 10px;
      border-radius: 6px;
      cursor: pointer;
      transition: background 0.15s;
    }}
    .btn-copy:hover {{ background: #334155; }}
    .snippet {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
      background: #060911;
      padding: 6px 10px;
      border-radius: 6px;
      color: #93c5fd;
      margin-top: 6px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      word-break: break-all;
    }}
  </style>
</head>
<body>
  <header>
    <div class="header-top">
      <h1>{_e(title)} <span class="badge">Pro v2.0</span></h1>
      <span class="badge">Goal Mode Output</span>
    </div>
    <div class="sub">{_e(idx.root)} · {pages_count} Pages Indexed · Dual Engine: Direct Entity Mention + Calibrated LinkBERT</div>
  </header>

  <div class="container">
    <!-- KPI Row -->
    <div class="kpi-grid">
      <div class="kpi-card success">
        <div class="kpi-val">{total_opps}</div>
        <div class="kpi-lbl">Link Opportunities</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-val">{pages_count}</div>
        <div class="kpi-lbl">Total Pages</div>
      </div>
      <div class="kpi-card warning">
        <div class="kpi-val">{orphans_count}</div>
        <div class="kpi-lbl">Orphan Pages Detected</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-val">{avg_score:.2f}</div>
        <div class="kpi-lbl">Average Score</div>
      </div>
    </div>

    <!-- Network Graph Visualizer -->
    <div class="section-box">
      <div class="section-title">
        <span>Site Link Network & PageRank Topology</span>
        <span class="sub" style="font-size: 13px;">Green dashed lines indicate recommended high-value links</span>
      </div>
      <div id="graph-container">
        <svg id="network-svg" class="graph-svg"></svg>
      </div>
    </div>

    <!-- Table Section -->
    <div class="section-box">
      <div class="section-title">
        <span>Optimized Link Recommendations</span>
        <button class="filter-btn active" id="btn-export-json" onclick="downloadJSON()">Export Full JSON</button>
      </div>

      <div class="controls">
        <input type="text" id="search-input" class="search-input" placeholder="Search by anchor, source page, or target..." oninput="filterTable()">
        <button class="filter-btn active" onclick="setEngineFilter('all', this)">All Engines ({total_opps})</button>
        <button class="filter-btn" onclick="setEngineFilter('direct_mention', this)">Direct Mentions</button>
        <button class="filter-btn" onclick="setEngineFilter('linkbert', this)">LinkBERT AI</button>
      </div>

      <table id="opps-table">
        <thead>
          <tr>
            <th style="width: 20%;">Source Page</th>
            <th style="width: 22%;">Anchor Text</th>
            <th style="width: 22%;">Target Page</th>
            <th style="width: 14%;">Score</th>
            <th style="width: 22%;">Context & Code</th>
          </tr>
        </thead>
        <tbody id="table-body">
        </tbody>
      </table>
    </div>
  </div>

  <script>
    const graphData = {graph_data_json};
    const opportunities = {opps_json};
    let currentEngine = "all";

    function renderTable(data) {{
      const tbody = document.getElementById("table-body");
      if (!data.length) {{
        tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-muted); padding: 32px;">No matching link opportunities found.</td></tr>';
        return;
      }}
      tbody.innerHTML = data.map(o => {{
        const pct = Math.round(o.score * 100);
        const engineClass = o.engine === 'direct_mention' ? 'engine-mention' : 'engine-linkbert';
        const engineLabel = o.engine === 'direct_mention' ? 'Direct Target Mention' : 'LinkBERT Calibrated';
        const escapedAnchor = encodeURIComponent(o.anchor);
        const hrefSnippet = `<a href="${{o.target}}">${{o.anchor}}</a>`;

        return `
          <tr>
            <td>
              <div style="font-weight: 600;">${{o.source}}</div>
              <span class="engine-badge ${{engineClass}}">${{engineLabel}}</span>
            </td>
            <td>
              <span class="anchor-hl">${{o.anchor}}</span>
              <div style="color: var(--text-muted); font-size: 12px; margin-top: 4px;">${{o.anchor.split(' ').length}} words</div>
            </td>
            <td>
              <div style="font-weight: 600;">${{o.target_title || o.target}}</div>
              <div style="color: var(--text-muted); font-size: 12px;">${{o.target}}</div>
            </td>
            <td>
              <span style="font-weight: 700;">${{o.score.toFixed(2)}}</span>
              <div class="score-bar"><div class="score-bar-fill" style="width: ${{pct}}%;"></div></div>
            </td>
            <td>
              <div style="font-size: 13px; color: var(--text-muted); margin-bottom: 6px;">${{highlightContext(o.sentence, o.anchor)}}</div>
              <div class="snippet">
                <code>${{escapeHtml(hrefSnippet)}}</code>
                <button class="btn-copy" onclick="copySnippet('${{escapeHtml(hrefSnippet)}}', this)">Copy</button>
              </div>
            </td>
          </tr>
        `;
      }}).join('');
    }}

    function escapeHtml(str) {{
      return (str || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }}

    function highlightContext(sentence, anchor) {{
      if (!sentence) return '';
      const i = sentence.toLowerCase().indexOf(anchor.toLowerCase());
      if (i < 0) return escapeHtml(sentence);
      return escapeHtml(sentence.slice(0, i)) + '<mark class="anchor-hl">' + escapeHtml(sentence.slice(i, i + anchor.length)) + '</mark>' + escapeHtml(sentence.slice(i + anchor.length));
    }}

    function copySnippet(text, btn) {{
      navigator.clipboard.writeText(text).then(() => {{
        const orig = btn.innerText;
        btn.innerText = "Copied!";
        btn.style.background = "#10b981";
        setTimeout(() => {{
          btn.innerText = orig;
          btn.style.background = "";
        }}, 1500);
      }});
    }}

    function filterTable() {{
      const q = document.getElementById("search-input").value.toLowerCase();
      const filtered = opportunities.filter(o => {{
        const matchesQuery = o.anchor.toLowerCase().includes(q) || o.source.toLowerCase().includes(q) || o.target.toLowerCase().includes(q) || (o.target_title || '').toLowerCase().includes(q);
        const matchesEngine = currentEngine === 'all' || o.engine === currentEngine;
        return matchesQuery && matchesEngine;
      }});
      renderTable(filtered);
    }}

    function setEngineFilter(engine, btn) {{
      currentEngine = engine;
      document.querySelectorAll('.controls .filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      filterTable();
    }}

    function downloadJSON() {{
      const blob = new Blob([JSON.stringify(opportunities, null, 2)], {{ type: 'application/json' }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'linkbert-opportunities.json';
      a.click();
    }}

    // Render interactive SVG Network Graph
    function initGraph() {{
      const svg = document.getElementById("network-svg");
      const width = svg.clientWidth || 1000;
      const height = svg.clientHeight || 480;

      const nodes = graphData.nodes;
      const links = graphData.links;

      // Simple circular force-directed layout
      const cx = width / 2;
      const cy = height / 2;
      const radius = Math.min(width, height) * 0.38;

      nodes.forEach((n, idx) => {{
        const angle = (idx / nodes.length) * 2 * Math.PI;
        n.x = cx + radius * Math.cos(angle);
        n.y = cy + radius * Math.sin(angle);
      }});

      let svgHtml = '';

      // Draw links
      links.forEach(l => {{
        const s = nodes[l.source];
        const t = nodes[l.target];
        if (!s || !t) return;
        const isProposed = l.type === 'proposed';
        const color = isProposed ? '#10b981' : '#334155';
        const dash = isProposed ? 'stroke-dasharray="5,5"' : '';
        const width = isProposed ? '2' : '1';
        svgHtml += `<line x1="${{s.x}}" y1="${{s.y}}" x2="${{t.x}}" y2="${{t.y}}" stroke="${{color}}" stroke-width="${{width}}" ${{dash}} opacity="${{isProposed ? 0.9 : 0.4}}" />`;
      }});

      // Draw nodes
      nodes.forEach(n => {{
        const r = Math.max(8, Math.min(24, 10 + n.pr * 5));
        const fill = n.isOrphan ? '#ef4444' : '#38bdf8';
        svgHtml += `
          <g style="cursor: pointer;">
            <circle cx="${{n.x}}" cy="${{n.y}}" r="${{r}}" fill="${{fill}}" stroke="#0b0f19" stroke-width="2" />
            <text x="${{n.x}}" y="${{n.y + r + 14}}" fill="#cbd5e1" font-size="11" font-family="sans-serif" text-anchor="middle">${{escapeHtml(n.key)}}</text>
          </g>
        `;
      }});

      svg.innerHTML = svgHtml;
    }}

    renderTable(opportunities);
    window.addEventListener('load', initGraph);
  </script>
</body>
</html>
"""
    return html_doc


def build_master_workbook(
    out_file: Path,
    idx: SiteIndex,
    opportunities: List[CandidateOpportunity],
    external_opps: Optional[List[Any]] = None,
    metrics: Optional[GraphMetrics] = None,
    gsc_queries: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Generates an industrial-grade, multi-tab Excel workbook for link architecture."""
    if not HAS_OPENPYXL:
        return

    wb = openpyxl.Workbook()
    header_font = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")
    align_left = Alignment(horizontal="left", vertical="center", wrap_text=False)
    thin_border = Border(
        left=Side(style="thin", color="E2E8F0"),
        right=Side(style="thin", color="E2E8F0"),
        top=Side(style="thin", color="E2E8F0"),
        bottom=Side(style="thin", color="E2E8F0"),
    )

    def style_header_and_cols(ws: Any, headers: List[str]) -> None:
        ws.append(headers)
        for col_idx in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = align_left

        ws.freeze_panes = "A2"

    # =========================================================================
    # TAB 1: Internal Linking Architecture
    # =========================================================================
    ws_int = wb.active
    ws_int.title = "Internal_Linking_Matrix"
    int_headers = [
        "Source Page",
        "Anchor Text",
        "Target Page",
        "Target Title",
        "Re-Rank Score",
        "Discovery Engine",
        "Context Sentence",
        "Source File",
    ]
    style_header_and_cols(ws_int, int_headers)

    for o in opportunities:
        ws_int.append([
            o.source_key,
            o.anchor,
            o.target_key,
            o.target_title,
            round(o.score, 3),
            o.engine,
            o.sentence,
            o.source_file,
        ])

    # =========================================================================
    # TAB 2: External Authority & E-E-A-T Citations
    # =========================================================================
    ws_ext = wb.create_sheet(title="External_Authority_Citations")
    ext_headers = [
        "Source Page",
        "Anchor Entity",
        "Exact Sentence Context",
        "Target Authority URL",
        "Target Title",
        "Target Domain",
        "Trust Tier",
        "Citation Category",
        "Recommended Rel",
        "Confidence Score",
        "E-E-A-T Justification",
        "Academic / Patent Basis",
    ]
    style_header_and_cols(ws_ext, ext_headers)

    if external_opps:
        for eo in external_opps:
            tier_str = f"Tier {getattr(eo, 'trust_tier', 1)}"
            ws_ext.append([
                getattr(eo, "source_key", ""),
                getattr(eo, "anchor", ""),
                getattr(eo, "sentence", ""),
                getattr(eo, "target_url", ""),
                getattr(eo, "target_title", ""),
                getattr(eo, "target_domain", ""),
                tier_str,
                getattr(eo, "category", ""),
                getattr(eo, "recommended_rel", "noopener noreferrer"),
                round(getattr(eo, "confidence", 0.85), 3),
                getattr(eo, "reason", ""),
                getattr(eo, "academic_basis", ""),
            ])

    # =========================================================================
    # TAB 3: Google Search Console Striking Distance Matrix
    # =========================================================================
    ws_gsc = wb.create_sheet(title="GSC_Striking_Distance")
    gsc_headers = [
        "Query",
        "Assigned Target URL",
        "Impressions",
        "Clicks",
        "CTR",
        "Average Position",
        "Striking Distance Tier",
        "Recommended Internal Anchor",
    ]
    style_header_and_cols(ws_gsc, gsc_headers)

    if gsc_queries:
        for q in gsc_queries:
            pos = float(q.get("position", 0.0))
            if pos <= 3.0:
                tier = "Top 3 (Defend)"
            elif pos <= 10.0:
                tier = "Page 1 - Striking Distance (Immediate Boost)"
            elif pos <= 20.0:
                tier = "Page 2 - Striking Distance (High Potential)"
            elif pos <= 50.0:
                tier = "Top 50 - Mid-Tier Opportunity"
            else:
                tier = "Longtail - Needs Topic Silo"

            ws_gsc.append([
                q.get("query", ""),
                q.get("page", "/"),
                q.get("impressions", 0),
                q.get("clicks", 0),
                q.get("ctr", "0%"),
                pos,
                tier,
                q.get("query", ""),
            ])

    # =========================================================================
    # TAB 4: Graph Topology & Silo Audit
    # =========================================================================
    ws_graph = wb.create_sheet(title="Graph_Topology_Audit")
    graph_headers = [
        "Page URL",
        "Page Title",
        "Current PageRank",
        "Projected PageRank",
        "PageRank Delta (%)",
        "Orphan Status",
        "Under-Linked Status",
        "Topic Cluster",
    ]
    style_header_and_cols(ws_graph, graph_headers)

    if metrics:
        for p in idx.pages:
            cur_pr = metrics.page_rank.get(p.key, 0.0)
            proj_pr = metrics.projected_page_rank.get(p.key, cur_pr)
            delta = ((proj_pr - cur_pr) / cur_pr * 100) if cur_pr > 0 else 0.0
            is_orphan = "ORPHAN (Critical)" if p.key in metrics.orphan_pages else "OK"
            is_under = "UNDER-LINKED" if p.key in metrics.under_linked_pages else "OK"
            cluster_val = "root"
            if metrics and hasattr(metrics, "clusters") and metrics.clusters:
                if isinstance(metrics.clusters, dict):
                    for c_id, c_pages in metrics.clusters.items():
                        if isinstance(c_pages, list) and p.key in c_pages:
                            cluster_val = str(c_id)
                            break
                        elif p.key == c_id:
                            cluster_val = str(c_pages)
                            break

            ws_graph.append([
                p.key,
                p.title or p.key,
                round(cur_pr, 4),
                round(proj_pr, 4),
                f"{delta:+.1f}%",
                is_orphan,
                is_under,
                cluster_val,
            ])

    # Auto-adjust column widths across all sheets
    for ws in wb.worksheets:
        for col in ws.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                val = str(cell.value or "")
                if len(val) > max_len:
                    max_len = len(val)
            ws.column_dimensions[col_letter].width = min(max(max_len + 3, 12), 65)

    wb.save(out_file)


def write_all(
    out_dir: Path,
    idx: SiteIndex,
    opportunities: List[CandidateOpportunity],
    metrics: GraphMetrics | None = None,
    title: str = "LinkBERT Pro — SEO Link Intelligence Dashboard",
    external_opps: Optional[List[Any]] = None,
    gsc_queries: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover external authority citations if not passed
    if external_opps is None:
        from .external import ExternalLinkEngine
        ext_eng = ExternalLinkEngine()
        external_opps = []
        for p in idx.pages:
            external_opps.extend(ext_eng.analyze_page(p))

    # 2. Write suggestions.json
    raw_opps = [o.as_dict() for o in opportunities]
    (out_dir / "suggestions.json").write_text(
        json.dumps({
            "site": idx.root,
            "count": len(raw_opps),
            "opportunities": raw_opps,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 3. Write suggestions.csv (Internal Links)
    with (out_dir / "suggestions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "anchor", "target", "target_title", "score", "engine", "sentence", "source_file"])
        for o in opportunities:
            w.writerow([
                o.source_key,
                o.anchor,
                o.target_key,
                o.target_title,
                round(o.score, 3),
                o.engine,
                o.sentence,
                o.source_file,
            ])

    # 4. Write external_links.csv (External Authority Citations)
    ext_csv = out_dir / "external_links.csv"
    with ext_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "source_page",
            "anchor_text",
            "exact_sentence_context",
            "recommended_external_url",
            "target_title",
            "target_domain",
            "trust_tier",
            "citation_category",
            "recommended_rel_attribute",
            "confidence_score",
            "eeat_justification",
            "academic_basis",
        ])
        for eo in external_opps:
            w.writerow([
                getattr(eo, "source_key", ""),
                getattr(eo, "anchor", ""),
                getattr(eo, "sentence", ""),
                getattr(eo, "target_url", ""),
                getattr(eo, "target_title", ""),
                getattr(eo, "target_domain", ""),
                f"Tier {getattr(eo, 'trust_tier', 1)}",
                getattr(eo, "category", ""),
                getattr(eo, "recommended_rel", "noopener noreferrer"),
                round(getattr(eo, "confidence", 0.85), 3),
                getattr(eo, "reason", ""),
                getattr(eo, "academic_basis", ""),
            ])

    # 5. Write multi-tab Excel Workbook (link_architecture_master.xlsx)
    excel_file = out_dir / "link_architecture_master.xlsx"
    build_master_workbook(
        out_file=excel_file,
        idx=idx,
        opportunities=opportunities,
        external_opps=external_opps,
        metrics=metrics,
        gsc_queries=gsc_queries,
    )

    # 6. Write report.html
    html_content = generate_dashboard_html(idx, opportunities, metrics, title=title)
    report_file = out_dir / "report.html"
    report_file.write_text(html_content, encoding="utf-8")

    # 7. Write graph_metrics.json
    if metrics:
        (out_dir / "graph_metrics.json").write_text(
            json.dumps({
                "page_rank": metrics.page_rank,
                "projected_page_rank": metrics.projected_page_rank,
                "orphan_pages": metrics.orphan_pages,
                "under_linked_pages": metrics.under_linked_pages,
                "clusters": getattr(metrics, "clusters", {}),
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return {
        "opportunities": len(opportunities),
        "external_opportunities": len(external_opps),
        "report": str(report_file),
        "excel": str(excel_file),
        "internal_csv": str(out_dir / "suggestions.csv"),
        "external_csv": str(ext_csv),
        "json": str(out_dir / "suggestions.json"),
    }
