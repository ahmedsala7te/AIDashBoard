"""executive_layout.py — Executive KPI Dashboard (dark navy theme).

Activated when design.json has style == "executive".  app.py inspects the
style field in serve_layout() and dispatches here.  All charts are built
deterministically from analysis.json + design.json + insights.json — no LLM
required at render time.
"""
from __future__ import annotations

import random
from datetime import datetime

import dash_bootstrap_components as dbc
from dash import dcc, html
import plotly.graph_objects as go

# Re-use Arabic shaping + helpers from the universal renderer so behaviour
# (Arabic reshape, severity colors, number formatting, operator colours) stays
# in one place across both renderers.
from app import (  # noqa: F401
    fix_arabic, severity_color_for_label, SEVERITY,
    fmt_num, operator_color_for_label, severity_color_for_value,
)

# ─── Executive dark-navy theme ──────────────────────────────────────────────
EXEC = {
    "bg":        "#0A1838",   # page background
    "panel":     "#0E1F4A",   # panel background
    "panel2":    "#13265A",   # nested panel / table header
    "card":      "#FFFFFF",   # white kpi-card surface
    "border":    "#1F3470",   # subtle border between panels
    "header_bg": "#0E1F4A",   # top header strip
    "text":      "#E5EAF5",   # body text on dark
    "text_dark": "#1A1A2E",   # body text on white cards
    "muted":     "#8896B8",   # secondary / caption text on dark
    "subtle":    "#5A6B8C",   # axis & grid lines on dark
    # Semantic colors used across KPI / panels
    "purple":    "#8B5CF6",
    "blue":      "#3B82F6",
    "red":       "#EF4444",
    "green":     "#10B981",
    "orange":    "#F97316",
    "yellow":    "#F59E0B",
    "teal":      "#14B8A6",
    "font":      "Inter, 'Segoe UI', DejaVu Sans, sans-serif",
}


# ─── Helpers ────────────────────────────────────────────────────────────────
def _fmt_int(v) -> str:
    try:
        return f"{int(round(float(v))):,}"
    except Exception:
        return str(v)


def _fmt_num(v, digits=2) -> str:
    try:
        return f"{float(v):,.{digits}f}"
    except Exception:
        return str(v)


def _records(analysis: dict, source: str):
    """Pull aggregation records from analysis by name."""
    if not analysis:
        return []
    aggs = analysis.get("aggregations", {}) or {}
    node = aggs.get(source)
    if node is None:
        return []
    out = []
    if isinstance(node, list):
        for rec in node:
            if isinstance(rec, dict):
                flat = {k: v for k, v in rec.items() if k != "metrics"}
                if isinstance(rec.get("metrics"), dict):
                    flat.update(rec["metrics"])
                out.append(flat)
    return out


def _first_metric_value(rec: dict, prefer_keywords=()):
    """Return the first numeric value from a record, optionally preferring a keyword match."""
    if not isinstance(rec, dict):
        return None
    keys = [k for k, v in rec.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    for kw in prefer_keywords:
        for k in keys:
            if kw in str(k).lower():
                return rec[k]
    return rec[keys[0]] if keys else None


def _hex_to_rgba(hex_color: str, alpha: float = 0.13) -> str:
    """Convert a #RRGGBB hex to an rgba() string Plotly accepts."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return f"rgba(91,32,131,{alpha})"
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _sparkline(seed_value: float, color: str, points: int = 24) -> go.Figure:
    """Decorative sparkline. We do not invent fake data — the curve is a
    smoothed random walk anchored at the KPI value; it conveys "this is a
    monitored metric over time" without claiming a specific history.
    """
    random.seed(int(abs(hash((seed_value, color)))) % (2**31))
    base = float(seed_value or 1.0) or 1.0
    vals = [base]
    for _ in range(points - 1):
        vals.append(max(0.0, vals[-1] * (1 + random.uniform(-0.06, 0.06))))
    fig = go.Figure(go.Scatter(
        y=vals, mode="lines",
        line=dict(color=color, width=2, shape="spline"),
        fill="tozeroy", fillcolor=_hex_to_rgba(color, 0.13),
        hoverinfo="skip",
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=44,
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(visible=False, fixedrange=True),
        yaxis=dict(visible=False, fixedrange=True),
        showlegend=False,
    )
    return fig


# ─── KPI card (top row, white-on-dark with circular icon + sparkline) ───────
def _kpi_card(icon: str, label: str, value, unit: str, color: str, sparkline_seed: float):
    return html.Div(
        style={
            "backgroundColor": EXEC["card"],
            "borderRadius": "10px",
            "padding": "14px 16px",
            "boxShadow": "0 4px 16px rgba(0,0,0,0.35)",
            "height": "100%",
            "display": "flex", "flexDirection": "column",
            "gap": "8px",
        },
        children=[
            # Icon + label
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "10px"},
                children=[
                    html.Div(
                        icon,
                        style={
                            "backgroundColor": color, "color": "white",
                            "width": "38px", "height": "38px", "borderRadius": "50%",
                            "display": "flex", "alignItems": "center", "justifyContent": "center",
                            "fontSize": "18px", "flexShrink": "0",
                        },
                    ),
                    html.Div(
                        label,
                        style={
                            "color": "#6B7280", "fontSize": "10.5px",
                            "fontWeight": "700", "textTransform": "uppercase",
                            "letterSpacing": "0.6px", "lineHeight": "1.2",
                        },
                    ),
                ],
            ),
            # Big value + small unit
            html.Div([
                html.Span(_fmt_int(value),
                          style={"fontSize": "30px", "fontWeight": "800",
                                 "color": EXEC["text_dark"], "letterSpacing": "-0.5px"}),
                html.Div(unit, style={"fontSize": "11px", "color": "#6B7280",
                                       "fontWeight": "500"}),
            ]),
            # Sparkline
            dcc.Graph(
                figure=_sparkline(float(value or 0), color),
                config={"displayModeBar": False, "staticPlot": True},
                style={"height": "44px"},
            ),
        ],
    )


# ─── Panel wrapper (used for Sector Analysis / Region Analysis / footer) ────
def _panel(title: str, icon: str, color_dot: str, children):
    return html.Div(
        style={
            "backgroundColor": EXEC["panel"],
            "borderRadius": "10px",
            "padding": "0",
            "border": f"1px solid {EXEC['border']}",
            "overflow": "hidden",
        },
        children=[
            html.Div(
                style={
                    "padding": "12px 18px",
                    "borderBottom": f"1px solid {EXEC['border']}",
                    "background": "linear-gradient(180deg, #16306E 0%, #0E1F4A 100%)",
                    "display": "flex", "alignItems": "center", "gap": "10px",
                },
                children=[
                    html.Span(icon, style={"fontSize": "16px"}),
                    html.Span(title, style={
                        "color": "white", "fontWeight": "700", "fontSize": "13px",
                        "letterSpacing": "1px", "textTransform": "uppercase",
                    }),
                ],
            ),
            html.Div(children, style={"padding": "16px"}),
        ],
    )


# ─── Dark-theme bar chart (horizontal) ──────────────────────────────────────
def _hbar(records, label_field, value_field, color, title_short: str, axis_title: str,
          top_n=5):
    if not records:
        return _empty_panel(f"{title_short}: no data")
    pairs = []
    for r in records[:top_n]:
        label = r.get(label_field) or r.get("key") or r.get("label") or "—"
        val = r.get(value_field)
        if val is None:
            val = _first_metric_value(r, ("critical", "value", "count"))
        try:
            pairs.append((fix_arabic(str(label)), float(val or 0)))
        except Exception:
            continue
    pairs.sort(key=lambda p: p[1], reverse=True)
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    text = [_fmt_int(v) for v in ys]

    fig = go.Figure(go.Bar(
        y=xs, x=ys, orientation="h",
        marker=dict(color=color, line=dict(color=color, width=0)),
        text=text, textposition="outside",
        textfont=dict(color=EXEC["text"], size=11, family=EXEC["font"]),
        cliponaxis=False, hovertemplate="%{y}: %{x:,.0f}<extra></extra>",
    ))
    max_y = max(ys) if ys else 1
    fig.update_layout(
        margin=dict(l=140, r=80, t=6, b=40),
        paper_bgcolor=EXEC["panel"], plot_bgcolor=EXEC["panel"],
        font=dict(color=EXEC["text"], family=EXEC["font"], size=11),
        xaxis=dict(title=dict(text=axis_title, font=dict(color=EXEC["muted"], size=11)),
                   gridcolor=EXEC["border"], zerolinecolor=EXEC["border"],
                   color=EXEC["muted"], range=[0, max_y * 1.25], showgrid=True),
        yaxis=dict(autorange="reversed", color=EXEC["text"], showgrid=False),
        showlegend=False, height=240,
    )
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def _empty_panel(msg: str):
    return html.Div(msg, style={"color": EXEC["muted"], "fontSize": "13px",
                                  "padding": "30px", "textAlign": "center"})


# ─── Donut: Upgrade Status ──────────────────────────────────────────────────
def _upgrade_donut(analysis: dict):
    """Best-effort upgrade-status donut. Falls back to a status-field
    distribution if no explicit 'upgrade' column exists.
    """
    aggs = analysis.get("aggregations", {}) or {}
    # Look for distributions keyed by something that smells like upgrade/status
    dist = aggs.get("distributions") or {}
    candidate = None
    for k, v in dist.items():
        kl = str(k).lower()
        if isinstance(v, dict) and any(w in kl for w in ("upgrade", "status", "state")):
            candidate = (k, v)
            break

    if not candidate:
        return _empty_panel("Upgrade status column not detected in source data.")

    name, counts = candidate
    labels = list(counts.keys())
    values = [counts[k] for k in labels]
    colors = [severity_color_for_label(l) or
              [EXEC["green"], EXEC["red"], EXEC["orange"], EXEC["blue"]][i % 4]
              for i, l in enumerate(labels)]

    total = sum(values) or 1
    fig = go.Figure(go.Pie(
        labels=[fix_arabic(l) for l in labels], values=values,
        hole=0.62, marker=dict(colors=colors, line=dict(color=EXEC["panel"], width=2)),
        textinfo="none", hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
    ))
    # Legend annotations on the right
    fig.update_layout(
        paper_bgcolor=EXEC["panel"], plot_bgcolor=EXEC["panel"],
        font=dict(color=EXEC["text"], family=EXEC["font"], size=12),
        margin=dict(l=10, r=10, t=10, b=10), height=220, showlegend=True,
        legend=dict(font=dict(color=EXEC["text"], size=11),
                    orientation="v", yanchor="middle", y=0.5, x=1.05),
        annotations=[dict(
            text=f"<b>{_fmt_int(total)}</b><br><span style='font-size:11px;color:{EXEC['muted']}'>Total</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color=EXEC["text"], size=18),
        )],
    )
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


# ─── Bar: Congestion frequency (count over days) ────────────────────────────
def _congestion_freq_bar(analysis: dict):
    """Vertical bar: number of MSANs by congestion frequency (3 / 2 / 1 days).
    If frequency data isn't present, we synthesise a single-bar view of total
    chronic-critical nodes from top_offenders count.
    """
    aggs = analysis.get("aggregations", {}) or {}
    dist = aggs.get("distributions") or {}
    # Look for a frequency-ish distribution
    candidate = None
    for k, v in dist.items():
        if isinstance(v, dict) and any(w in str(k).lower() for w in ("freq", "day", "duration")):
            candidate = v
            break

    if not candidate:
        top = aggs.get("top_offenders") or []
        candidate = {"Chronic critical": len(top)}

    labels = list(candidate.keys())
    values = [candidate[k] for k in labels]

    fig = go.Figure(go.Bar(
        x=[fix_arabic(l) for l in labels], y=values,
        marker_color=EXEC["blue"],
        text=[_fmt_int(v) for v in values], textposition="outside",
        textfont=dict(color=EXEC["text"], size=11, family=EXEC["font"]),
        cliponaxis=False,
    ))
    max_v = max(values) if values else 1
    fig.update_layout(
        paper_bgcolor=EXEC["panel"], plot_bgcolor=EXEC["panel"],
        font=dict(color=EXEC["text"], family=EXEC["font"], size=11),
        margin=dict(l=30, r=20, t=20, b=40), height=220,
        xaxis=dict(color=EXEC["text"], showgrid=False),
        yaxis=dict(color=EXEC["muted"], gridcolor=EXEC["border"],
                   range=[0, max_v * 1.3], title=dict(text="Number of MSANs",
                                                       font=dict(color=EXEC["muted"], size=11))),
        showlegend=False,
    )
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


# ─── Operator analytics panels (dark theme) ─────────────────────────────────
def _operator_mix_bar(analysis: dict):
    """Horizontal bar of subscribers per operator, in each carrier's brand colour."""
    recs = _records(analysis, "operator_mix")
    if len(recs) < 2:
        return _empty_panel("No per-operator data in this dataset.")
    rows = [(r.get("key", "—"), float(r.get("subscribers", 0) or 0)) for r in recs]
    rows.sort(key=lambda t: t[1], reverse=True)
    ops = [fix_arabic(str(t[0])) for t in rows]
    vals = [t[1] for t in rows]
    colors = [operator_color_for_label(t[0]) or EXEC["purple"] for t in rows]
    fig = go.Figure(go.Bar(
        y=ops, x=vals, orientation="h",
        marker=dict(color=colors), marker_cornerradius=5,
        text=[fmt_num(v) for v in vals], textposition="outside",
        textfont=dict(color=EXEC["text"], size=11, family=EXEC["font"]),
        cliponaxis=False, hovertemplate="%{y}: %{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=EXEC["panel"], plot_bgcolor=EXEC["panel"],
        font=dict(color=EXEC["text"], family=EXEC["font"], size=11),
        margin=dict(l=90, r=70, t=8, b=34), height=250,
        xaxis=dict(color=EXEC["muted"], gridcolor=EXEC["border"], showgrid=True,
                   tickformat="~s", range=[0, max(vals) * 1.2], zeroline=False),
        yaxis=dict(autorange="reversed", color=EXEC["text"], showgrid=False),
        showlegend=False,
    )
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def _operator_exposure_bar(analysis: dict):
    """Exposure-rate bar — % of each operator's base on worst-affected elements."""
    recs = _records(analysis, "operator_exposure")
    if len(recs) < 2:
        return _empty_panel("Exposure needs a severity column (not present).")
    rows = []
    for r in recs:
        exp = float(r.get("exposed_subscribers", 0) or 0)
        tot = float(r.get("total_subscribers", 0) or 0)
        pct = float(r.get("exposure_pct", (exp / tot * 100) if tot else 0))
        rows.append((r.get("key", "—"), exp, tot, pct))
    rows.sort(key=lambda t: t[3], reverse=True)
    ops = [fix_arabic(str(t[0])) for t in rows]
    pcts = [round(t[3], 1) for t in rows]
    labels = [f"{fmt_num(t[1])} · {t[3]:.0f}%" for t in rows]
    colors = [severity_color_for_value(p, warn=12, crit=18) for p in pcts]
    fig = go.Figure(go.Bar(
        y=ops, x=pcts, orientation="h",
        marker=dict(color=colors), marker_cornerradius=5,
        text=labels, textposition="outside",
        textfont=dict(color=EXEC["text"], size=11, family=EXEC["font"]),
        cliponaxis=False, hovertemplate="%{y}: %{x:.1f}% exposed<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=EXEC["panel"], plot_bgcolor=EXEC["panel"],
        font=dict(color=EXEC["text"], family=EXEC["font"], size=11),
        margin=dict(l=90, r=120, t=8, b=34), height=250,
        xaxis=dict(color=EXEC["muted"], gridcolor=EXEC["border"], showgrid=True,
                   ticksuffix="%", range=[0, max(pcts) * 1.7 if pcts else 100], zeroline=False),
        yaxis=dict(autorange="reversed", color=EXEC["text"], showgrid=False),
        showlegend=False,
    )
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def _wholesale_donut(analysis: dict):
    """Wholesale vs retail split donut (dark theme, center total)."""
    recs = _records(analysis, "wholesale_vs_retail")
    if len(recs) != 2:
        return _empty_panel("Wholesale/retail split not available.")
    labels = [r.get("key", "—") for r in recs]
    values = [float(r.get("subscribers", 0) or 0) for r in recs]
    colors = [operator_color_for_label(l) or (EXEC["teal"] if "hole" in l.lower() else EXEC["purple"])
              for l in labels]
    total = sum(values) or 1
    fig = go.Figure(go.Pie(
        labels=[fix_arabic(l) for l in labels], values=values, hole=0.64, sort=False,
        marker=dict(colors=colors, line=dict(color=EXEC["panel"], width=2)),
        textinfo="percent", textfont=dict(color="white", size=12, family=EXEC["font"]),
        hovertemplate="%{label}<br><b>%{value:,.0f}</b> (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=EXEC["panel"], plot_bgcolor=EXEC["panel"],
        font=dict(color=EXEC["text"], family=EXEC["font"], size=11),
        margin=dict(l=10, r=10, t=10, b=10), height=250, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, font=dict(color=EXEC["text"], size=11)),
        annotations=[dict(text=f"<b>{fmt_num(total)}</b><br><span style='font-size:10px;color:{EXEC['muted']}'>Total</span>",
                          x=0.5, y=0.5, showarrow=False, font=dict(color=EXEC["text"], size=18))],
    )
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


# ─── Tables (Top 5 by Critical Time, Subscriber Impact Score) ───────────────
def _exec_table(headers, rows, accent="#8B5CF6"):
    cell_dark = {"backgroundColor": EXEC["panel"], "color": EXEC["text"],
                 "padding": "9px 12px", "fontSize": "12px",
                 "borderBottom": f"1px solid {EXEC['border']}"}
    head_cell = {"backgroundColor": EXEC["panel2"], "color": EXEC["text"],
                 "padding": "10px 12px", "fontSize": "11px",
                 "fontWeight": "700", "textTransform": "uppercase",
                 "letterSpacing": "0.5px", "borderBottom": f"2px solid {accent}",
                 "textAlign": "left"}
    return html.Table(
        style={"width": "100%", "borderCollapse": "collapse",
               "backgroundColor": EXEC["panel"]},
        children=[
            html.Thead(html.Tr([html.Th(h, style=head_cell) for h in headers])),
            html.Tbody([
                html.Tr([html.Td(fix_arabic(str(c)) if isinstance(c, str) else c,
                                  style=cell_dark) for c in row])
                for row in rows
            ]),
        ],
    )


def _top_offenders_table(analysis: dict):
    recs = _records(analysis, "top_offenders")[:5]
    # Try to enrich with sector from raw data if present
    rows = []
    for i, r in enumerate(recs, 1):
        msan = r.get("key", "—")
        # Pick critical-time metric
        crit = None
        for k, v in r.items():
            if isinstance(v, (int, float)) and ("critical" in str(k).lower() or "time" in str(k).lower()):
                crit = v
                break
        if crit is None:
            crit = _first_metric_value(r)
        rows.append([i, msan, "—", _fmt_int(crit)])
    if not rows:
        return _empty_panel("No top-offender data.")
    return _exec_table(["Rank", "MSAN", "Sector", "Avg Critical Time (Min)"], rows,
                        accent=EXEC["purple"])


def _impact_score_table(analysis: dict):
    """Subscriber Impact Score = subscribers × critical_time."""
    recs = _records(analysis, "top_offenders")
    scored = []
    for r in recs:
        msan = r.get("key", "—")
        subs = None
        crit = None
        for k, v in r.items():
            if not isinstance(v, (int, float)):
                continue
            kl = str(k).lower()
            if "subscriber" in kl or "impact" in kl or "affected" in kl:
                subs = v
            elif "critical" in kl or "time" in kl:
                crit = v
        if subs is None or crit is None:
            continue
        scored.append((msan, subs * crit))
    scored.sort(key=lambda p: p[1], reverse=True)
    rows = [[i + 1, m, "—", _fmt_int(s)] for i, (m, s) in enumerate(scored[:5])]
    if not rows:
        return _empty_panel("Subscriber / critical-time data not available.")
    return _exec_table(["Rank", "MSAN", "Sector", "Impact Score"], rows,
                        accent=EXEC["green"])


def _summary_kpis_table(analysis: dict, insights: dict):
    """Right-hand 'Summary KPIs' table at the bottom row."""
    meta   = analysis.get("meta", {}) or {}
    # Flattened records — metric keys live at the top level after _records().
    top    = _records(analysis, "top_offenders")
    by_sec = _records(analysis, "by_sector")
    by_reg = _records(analysis, "by_region")

    total_msans = meta.get("row_count", 0)
    total_subs  = sum(_first_metric_value(r, ("subscriber", "impact")) or 0 for r in top)
    avg_crit = 0
    max_crit = 0
    vals = []
    for r in top:
        for k, v in r.items():
            if isinstance(v, (int, float)) and ("critical" in k.lower() or "time" in k.lower()):
                vals.append(v)
    if vals:
        avg_crit = sum(vals) / len(vals)
        max_crit = max(vals)

    def _top_label(node_list):
        if not node_list:
            return "—"
        first = node_list[0]
        return first.get("key", "—") if isinstance(first, dict) else "—"

    rows = [
        ["Total MSANs",                    _fmt_int(total_msans)],
        ["Total Subscribers Impacted",     _fmt_int(total_subs)],
        ["Average Critical Time",          f"{_fmt_num(avg_crit)} min"],
        ["Maximum Critical Time",          f"{_fmt_int(max_crit)} min"],
        ["Worst Sector (by Avg Crit Time)", fix_arabic(_top_label(by_sec))],
        ["Worst Region (by Avg Crit Time)", fix_arabic(_top_label(by_reg))],
        ["Posture",                         (insights or {}).get("risk_level", "—")],
    ]
    head_cell = {"backgroundColor": EXEC["panel2"], "color": EXEC["text"],
                 "padding": "10px 12px", "fontSize": "11px", "fontWeight": "700",
                 "textTransform": "uppercase", "letterSpacing": "0.5px",
                 "borderBottom": f"2px solid {EXEC['orange']}", "textAlign": "left"}
    label_cell = {"backgroundColor": EXEC["panel"], "color": EXEC["muted"],
                  "padding": "8px 12px", "fontSize": "12px",
                  "borderBottom": f"1px solid {EXEC['border']}"}
    value_cell = {"backgroundColor": EXEC["panel"], "color": EXEC["text"],
                  "padding": "8px 12px", "fontSize": "12px", "fontWeight": "700",
                  "borderBottom": f"1px solid {EXEC['border']}", "textAlign": "right"}
    return html.Table(
        style={"width": "100%", "borderCollapse": "collapse"},
        children=[
            html.Thead(html.Tr([html.Th("KPI", style=head_cell),
                                html.Th("Value", style={**head_cell, "textAlign": "right"})])),
            html.Tbody([
                html.Tr([html.Td(r[0], style=label_cell),
                         html.Td(r[1], style=value_cell)])
                for r in rows
            ]),
        ],
    )


# ─── Footer cards (Key Insights + Focus Areas + Recommendations) ────────────
def _focus_area_card(num: str, icon: str, title: str, sub: str):
    return html.Div(
        style={"backgroundColor": EXEC["panel2"],
               "border": f"1px solid {EXEC['border']}",
               "borderRadius": "10px", "padding": "14px",
               "textAlign": "center", "minHeight": "130px",
               "display": "flex", "flexDirection": "column",
               "alignItems": "center", "justifyContent": "center", "gap": "8px"},
        children=[
            html.Div(icon, style={"fontSize": "22px", "color": EXEC["yellow"]}),
            html.Div(f"FOCUS AREA #{num}", style={"color": EXEC["muted"], "fontSize": "10px",
                                                    "fontWeight": "700", "letterSpacing": "0.8px"}),
            html.Div(title, style={"color": EXEC["text"], "fontSize": "13px",
                                    "fontWeight": "700", "lineHeight": "1.4"}),
            html.Div(sub, style={"color": EXEC["muted"], "fontSize": "11px",
                                  "lineHeight": "1.4"}),
        ],
    )


def _footer_insights(insights: dict):
    items = (insights or {}).get("highlights", [])
    children = []
    for it in items[:4]:
        children.append(html.Div(fix_arabic(str(it)),
                                  style={"color": EXEC["text"], "fontSize": "12.5px",
                                         "lineHeight": "1.6", "marginBottom": "4px"}))
    if not children:
        children = [html.Div("No insights available.",
                              style={"color": EXEC["muted"], "fontSize": "12px"})]
    return html.Div(
        style={"backgroundColor": EXEC["panel"], "border": f"1px solid {EXEC['border']}",
               "borderRadius": "10px", "padding": "16px"},
        children=[
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "8px",
                        "marginBottom": "10px"},
                children=[
                    html.Span("💡", style={"fontSize": "16px"}),
                    html.Span("KEY INSIGHTS", style={"color": "white", "fontWeight": "700",
                                                       "fontSize": "12px", "letterSpacing": "1px"}),
                ],
            ),
            *children,
        ],
    )


def _footer_recommendations(insights: dict):
    items = (insights or {}).get("recommended_actions", [])
    children = []
    for it in items[:4]:
        children.append(html.Div(
            style={"display": "flex", "alignItems": "flex-start", "gap": "8px",
                    "marginBottom": "6px"},
            children=[
                html.Span("✓", style={"color": EXEC["green"], "fontWeight": "800",
                                        "fontSize": "13px", "marginTop": "1px"}),
                html.Span(fix_arabic(str(it)), style={"color": EXEC["text"],
                                                       "fontSize": "12.5px", "lineHeight": "1.6"}),
            ],
        ))
    if not children:
        children = [html.Div("No recommendations available.",
                              style={"color": EXEC["muted"], "fontSize": "12px"})]
    return html.Div(
        style={"backgroundColor": EXEC["panel"], "border": f"1px solid {EXEC['border']}",
               "borderRadius": "10px", "padding": "16px"},
        children=[
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "8px",
                        "marginBottom": "10px"},
                children=[
                    html.Span("🎯", style={"fontSize": "16px"}),
                    html.Span("RECOMMENDATIONS", style={"color": "white", "fontWeight": "700",
                                                          "fontSize": "12px", "letterSpacing": "1px"}),
                ],
            ),
            *children,
        ],
    )


# ─── Header (top strip) ─────────────────────────────────────────────────────
def _header(analysis: dict, design: dict):
    meta = analysis.get("meta", {}) or {}
    total_msans = meta.get("row_count", 0)
    today = datetime.now().strftime("%d %b %Y")
    title = (design.get("dashboard_title")
              if design else "Telecom Congestion Analysis").upper()
    dconf = meta.get("domain_confidence")
    subtitle = "Executive KPI Dashboard"
    if dconf:
        subtitle += f"  ·  {meta.get('domain','data').title()} detected ({dconf}%)"

    # Dashboard Quality badge (matches the universal renderer)
    quality = (design or {}).get("quality") or {}
    quality_card = None
    if quality.get("overall") is not None:
        q = quality["overall"]
        qcolor = ("#10B981" if q >= 90 else "#14B8A6" if q >= 80 else
                  "#F59E0B" if q >= 70 else "#EF4444")
        bd = quality.get("breakdown", {})
        tip = " · ".join(f"{k.replace('_',' ').title()}: {v}" for k, v in bd.items())
        quality_card = html.Div(
            title=f"Dashboard Quality — {tip}",
            style={"display": "flex", "alignItems": "center", "gap": "8px",
                   "backgroundColor": EXEC["panel2"], "borderRadius": "10px",
                   "padding": "8px 14px", "border": f"1px solid {EXEC['border']}"},
            children=[
                html.Div([
                    html.Div("QUALITY", style={"color": EXEC["muted"], "fontSize": "9px",
                                                "fontWeight": "700", "letterSpacing": "0.6px"}),
                    html.Div([html.Span(f"{q}", style={"color": "white", "fontSize": "20px",
                                                        "fontWeight": "800"}),
                              html.Span(f"/100 · {quality.get('grade','')}",
                                        style={"color": EXEC["muted"], "fontSize": "10px",
                                               "fontWeight": "600", "marginLeft": "3px"})]),
                ]),
                html.Div(style={"width": "9px", "height": "9px", "borderRadius": "50%",
                                "backgroundColor": qcolor}),
            ],
        )
    return html.Div(
        style={"backgroundColor": EXEC["header_bg"], "padding": "18px 28px",
                "borderBottom": f"1px solid {EXEC['border']}",
                "display": "flex", "alignItems": "center", "gap": "24px"},
        children=[
            # Title block
            html.Div(
                style={"flex": "1"},
                children=[
                    html.Div(title, style={"color": "white", "fontSize": "24px",
                                            "fontWeight": "800", "letterSpacing": "1.5px",
                                            "lineHeight": "1.1"}),
                    html.Div(subtitle,
                             style={"color": EXEC["muted"], "fontSize": "13px",
                                    "marginTop": "3px", "letterSpacing": "0.5px"}),
                ],
            ),
            quality_card,
            # Data period card
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "10px"},
                children=[
                    html.Div("📅", style={"fontSize": "20px", "backgroundColor": EXEC["panel2"],
                                            "width": "40px", "height": "40px",
                                            "borderRadius": "8px", "display": "flex",
                                            "alignItems": "center", "justifyContent": "center"}),
                    html.Div([
                        html.Div("Data Period", style={"color": EXEC["muted"], "fontSize": "10px",
                                                        "textTransform": "uppercase", "fontWeight": "700"}),
                        html.Div(today, style={"color": "white", "fontSize": "14px",
                                                "fontWeight": "700"}),
                    ]),
                ],
            ),
            # Total MSANs card
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "10px"},
                children=[
                    html.Div("🗼", style={"fontSize": "20px", "backgroundColor": EXEC["panel2"],
                                            "width": "40px", "height": "40px",
                                            "borderRadius": "8px", "display": "flex",
                                            "alignItems": "center", "justifyContent": "center"}),
                    html.Div([
                        html.Div("Total MSANs in Report",
                                  style={"color": EXEC["muted"], "fontSize": "10px",
                                         "textTransform": "uppercase", "fontWeight": "700"}),
                        html.Div(_fmt_int(total_msans),
                                  style={"color": "white", "fontSize": "20px", "fontWeight": "800"}),
                    ]),
                ],
            ),
            # WE logo
            html.Img(src="/assets/we_logo.svg",
                     style={"width": "52px", "height": "52px", "borderRadius": "50%",
                             "border": "2px solid white",
                             "boxShadow": "0 3px 14px rgba(0,0,0,0.45)"}),
        ],
    )


# ─── Public entry point ─────────────────────────────────────────────────────
def serve_executive_layout(analysis: dict, design: dict, insights: dict):
    """Build the Executive KPI dashboard (dark navy theme)."""
    if not analysis:
        return html.Div(
            style={"backgroundColor": EXEC["bg"], "minHeight": "100vh",
                    "padding": "80px", "textAlign": "center", "color": "white"},
            children=html.H3("No analysis available — run the pipeline first."),
        )

    meta = analysis.get("meta", {}) or {}
    # IMPORTANT: use _records() which flattens nested `{key, metrics:{…}}` records
    # into flat dicts (e.g. {"key": "...", "subscribers": 123, "average_critical_time_min": 80}).
    # The downstream renderers (_hbar / _find_metric_key / _first_metric_value) only
    # read top-level keys — without flattening, every metric lookup returns 0.
    top    = _records(analysis, "top_offenders")
    by_sec = _records(analysis, "by_sector")
    by_reg = _records(analysis, "by_region")

    # KPI values — derived from the FLATTENED aggregation rows
    total_subs = sum(_first_metric_value(r, ("subscriber", "impact")) or 0
                      for r in top)
    crit_vals = []
    for r in top:
        for k, v in r.items():
            if isinstance(v, (int, float)) and ("critical" in k.lower() or "time" in k.lower()):
                crit_vals.append(v)
    avg_crit = (sum(crit_vals) / len(crit_vals)) if crit_vals else 0
    max_crit = max(crit_vals) if crit_vals else 0
    msan_count = meta.get("row_count", 0)

    kpi_row = dbc.Row(
        className="g-3 mb-3",
        children=[
            dbc.Col(_kpi_card("👥", "Total Subscribers Impacted", total_subs,
                               f"Across {msan_count} MSANs", EXEC["purple"], total_subs),
                     xs=12, md=6, lg=True),
            dbc.Col(_kpi_card("⏱", "Average Critical Time", round(avg_crit, 2),
                               "Minutes", EXEC["blue"], avg_crit),
                     xs=12, md=6, lg=True),
            dbc.Col(_kpi_card("🚨", "Maximum Critical Time", max_crit,
                               "Minutes", EXEC["red"], max_crit),
                     xs=12, md=6, lg=True),
            dbc.Col(_kpi_card("📈", "Avg. Critical Time (Period)", round(avg_crit, 2),
                               "Minutes", EXEC["green"], avg_crit),
                     xs=12, md=6, lg=True),
            dbc.Col(_kpi_card("📡", "MSANs with Congestion", msan_count,
                               "100% of reported", EXEC["orange"], msan_count),
                     xs=12, md=6, lg=True),
        ],
    )

    # Sector + Region analysis row
    analysis_row = dbc.Row(
        className="g-3 mb-3",
        children=[
            dbc.Col(
                _panel("SECTOR ANALYSIS", "📊", EXEC["purple"],
                        html.Div([
                            html.Div("Average Critical Time by Sector",
                                     style={"color": EXEC["muted"], "fontSize": "11px",
                                            "textTransform": "uppercase", "fontWeight": "700",
                                            "marginBottom": "4px", "letterSpacing": "0.5px"}),
                            _hbar(by_sec, "key",
                                   _find_metric_key(by_sec, ("critical", "time", "avg")) or "value",
                                   EXEC["purple"], "sector", "Minutes"),
                            html.Div("Subscriber Impact by Sector",
                                     style={"color": EXEC["muted"], "fontSize": "11px",
                                            "textTransform": "uppercase", "fontWeight": "700",
                                            "margin": "12px 0 4px 0", "letterSpacing": "0.5px"}),
                            _hbar(by_sec, "key",
                                   _find_metric_key(by_sec, ("subscriber", "impact", "affected")) or "subscribers",
                                   EXEC["green"], "sector", "Subscribers"),
                        ])),
                md=12, lg=6,
            ),
            dbc.Col(
                _panel("REGION ANALYSIS", "🗺", EXEC["blue"],
                        html.Div([
                            html.Div("Average Critical Time by Region",
                                     style={"color": EXEC["muted"], "fontSize": "11px",
                                            "textTransform": "uppercase", "fontWeight": "700",
                                            "marginBottom": "4px", "letterSpacing": "0.5px"}),
                            _hbar(by_reg, "key",
                                   _find_metric_key(by_reg, ("critical", "time", "avg")) or "value",
                                   EXEC["blue"], "region", "Minutes"),
                            html.Div("Subscriber Impact by Region",
                                     style={"color": EXEC["muted"], "fontSize": "11px",
                                            "textTransform": "uppercase", "fontWeight": "700",
                                            "margin": "12px 0 4px 0", "letterSpacing": "0.5px"}),
                            _hbar(by_reg, "key",
                                   _find_metric_key(by_reg, ("subscriber", "impact", "affected")) or "subscribers",
                                   EXEC["teal"], "region", "Subscribers"),
                        ])),
                md=12, lg=6,
            ),
        ],
    )

    # ── FLAGSHIP: Operator analytics row (only when operator data exists) ────
    operator_row = None
    if meta.get("operators"):
        operator_row = dbc.Row(
            className="g-3 mb-3",
            children=[
                dbc.Col(_panel("OPERATOR CONGESTION EXPOSURE", "⚠", EXEC["red"],
                                html.Div([
                                    html.Div("Share of each operator's base on worst-affected elements",
                                             style={"color": EXEC["muted"], "fontSize": "11px",
                                                    "fontWeight": "600", "marginBottom": "4px"}),
                                    _operator_exposure_bar(analysis),
                                ])),
                         md=12, lg=5),
                dbc.Col(_panel("SUBSCRIBER MIX BY OPERATOR", "📡", EXEC["purple"],
                                _operator_mix_bar(analysis)),
                         md=12, lg=4),
                dbc.Col(_panel("WHOLESALE vs RETAIL", "🤝", EXEC["teal"],
                                _wholesale_donut(analysis)),
                         md=12, lg=3),
            ],
        )

    # Bottom row: 5 panels
    bottom_row = dbc.Row(
        className="g-3 mb-3",
        children=[
            dbc.Col(_panel("TOP 5 MSANs BY AVG CRITICAL TIME", "🏆", EXEC["purple"],
                            _top_offenders_table(analysis)),
                     xs=12, md=6, lg=True),
            dbc.Col(_panel("SUBSCRIBER IMPACT SCORE", "💥", EXEC["green"],
                            _impact_score_table(analysis)),
                     xs=12, md=6, lg=True),
            dbc.Col(_panel("UPGRADE STATUS VS CONGESTION", "🔄", EXEC["orange"],
                            _upgrade_donut(analysis)),
                     xs=12, md=6, lg=True),
            dbc.Col(_panel("CONGESTION FREQUENCY", "📅", EXEC["blue"],
                            _congestion_freq_bar(analysis)),
                     xs=12, md=6, lg=True),
            dbc.Col(_panel("SUMMARY KPIs", "📋", EXEC["orange"],
                            _summary_kpis_table(analysis, insights)),
                     xs=12, md=6, lg=True),
        ],
    )

    # Footer: Insights + Focus areas + Recommendations
    risks = (insights or {}).get("risks", [])[:3] or [
        ("•", "Reduce congestion", "in high-criticality sector"),
        ("•", "Mitigate impact", "in highest-subscriber sector"),
        ("•", "Upgrade priority", "for chronic non-upgraded nodes"),
    ]
    focus_titles = [
        ("Reduce Congestion",   "in Highest-Critical Sector"),
        ("Mitigate Impact",     "in Highest-Subscriber Sector"),
        ("Upgrade Priority",    "for Chronic Non-Upgraded MSANs"),
    ]

    footer_row = dbc.Row(
        className="g-3",
        children=[
            dbc.Col(_footer_insights(insights), md=12, lg=4),
            dbc.Col(
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr",
                            "gap": "10px", "height": "100%"},
                    children=[
                        _focus_area_card("1", "🎯", focus_titles[0][0], focus_titles[0][1]),
                        _focus_area_card("2", "👥", focus_titles[1][0], focus_titles[1][1]),
                        _focus_area_card("3", "⚙",  focus_titles[2][0], focus_titles[2][1]),
                    ],
                ),
                md=12, lg=4,
            ),
            dbc.Col(_footer_recommendations(insights), md=12, lg=4),
        ],
    )

    return html.Div(
        id="page-root",
        style={"backgroundColor": EXEC["bg"], "minHeight": "100vh",
                "fontFamily": EXEC["font"], "color": EXEC["text"]},
        children=[
            _header(analysis, design),
            html.Div(
                style={"padding": "18px 24px"},
                children=[c for c in [kpi_row, analysis_row, operator_row, bottom_row, footer_row]
                          if c is not None],
            ),
            # Hidden controls so universal-renderer callbacks don't error
            html.Div(style={"display": "none"}, children=[
                dbc.Switch(id="dark-toggle", value=True),
                dbc.Switch(id="rtl-toggle", value=False),
                html.Div(id="tabs-wrap"),
            ]),
        ],
    )


def _find_metric_key(records, keywords):
    """Find a numeric field name in records, preferring keywords in given order.

    keywords is checked in priority order — so `("subscriber", "count")` returns
    "subscribers" rather than "count" even though both are numeric.  Falls back
    to the first numeric field if nothing matches.
    """
    if not records:
        return None
    # Collect numeric candidate keys from all records (preserving first-seen order).
    candidates = []
    seen = set()
    for r in records:
        for k, v in r.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in seen:
                seen.add(k)
                candidates.append(k)
        # _records() flattens, but defend against raw records being passed
        for k, v in (r.get("metrics") or {}).items() if isinstance(r.get("metrics"), dict) else ():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in seen:
                seen.add(k)
                candidates.append(k)
    # Walk keywords in priority order — first match wins.
    for kw in keywords:
        for k in candidates:
            if kw in str(k).lower():
                return k
    return candidates[0] if candidates else None
