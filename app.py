"""
app.py — Universal Data-Driven Dash Dashboard (port 8050)

This is a DETERMINISTIC renderer. It reads the pipeline's JSON artifacts
(analysis.json / design.json / insights.json) and builds a complete, working
dashboard with REAL data every time — independent of LLM code-generation quality.

If output/figures_code.py produced valid figures, those are layered in; otherwise
each chart is built deterministically here from the analysis aggregations.

It is intentionally NOT overwritten by the pipeline, so the served dashboard is
always runnable. Agent 4's creative version is saved to output/app_generated.py.
"""
import os
import json
import io
import traceback
from datetime import datetime

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, ctx, no_update
import plotly.graph_objects as go

# Optional Arabic shaping
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    _ARABIC_OK = True
except Exception:
    _ARABIC_OK = False

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "output")
EXPORTS_DIR = os.path.join(OUTPUT_DIR, "exports")
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(EXPORTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

THEME = {
    # ── WE Telecom Egypt brand palette ──────────────────────────────────────
    "bg":      "#F7F5FC",   # very light lavender-white page background
    "card":    "#FFFFFF",   # pure white cards
    "header":  "#5B2083",   # WE primary purple
    "purple2": "#7B3BAF",   # lighter purple (hover / accent)
    "purple3": "#E8D8F8",   # pale purple (chip backgrounds, dividers)
    # Primary action colour (re-alias to WE purple)
    "blue1":   "#5B2083",
    "blue2":   "#7B3BAF",
    "blue3":   "#C7A8E8",
    # Alarm / severity colours — ITU standard kept for NOC charts
    "red":     "#DC2626",
    "orange":  "#EA580C",
    "yellow":  "#D97706",
    "green":   "#16A34A",
    "teal":    "#0891B2",   # impacted-subscriber annotation colour
    "gray":    "#6B7280",
    "text":    "#1A1A2E",   # very dark navy-black body text
    "font":    "DejaVu Sans",
}
DARK = {"bg": "#110A1E", "card": "#1E0F33", "text": "#E9D8F8", "grid": "#3A2355"}

CATEGORICAL_PALETTE = [
    "#5B2083", "#0891B2", "#16A34A", "#EA580C",
    "#D97706", "#C7A8E8", "#DC2626", "#6B7280",
]

# Telecom NOC alarm-severity colours (RAG / ITU-style alarm severities)
SEVERITY = {
    "critical": "#DC2626",  # red
    "major":    "#EA580C",  # orange
    "minor":    "#D97706",  # amber
    "warning":  "#FACC15",  # yellow
    "normal":   "#16A34A",  # green
    "healthy":  "#16A34A",
    "ok":       "#16A34A",
}


def severity_color_for_value(v, warn=80.0, crit=90.0):
    """RAG colour for a percentage-like metric (utilization/congestion)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return THEME["blue1"]
    if v >= crit:
        return SEVERITY["critical"]
    if v >= warn:
        return SEVERITY["major"]
    if v >= warn * 0.85:
        return SEVERITY["warning"]
    return SEVERITY["normal"]


def severity_color_for_label(label):
    """Map a status/severity label to its NOC colour.

    Covers two separate vocabularies:
    1. Alarm/network severity  → Critical/Major/Minor/Warning/Normal
    2. Operational/work-order status → Solved/Cancelled/New/Pending/InProgress/NotPlanned
    """
    key = str(label).strip().lower().replace("_", " ").replace("-", " ")

    # ── Alarm severity keywords (ITU / NOC standard) ──
    for sev, col in SEVERITY.items():
        if sev in key:
            return col

    # ── Operational / upgrade status keywords ──
    # Positive outcome → green
    if any(w in key for w in ("solved", "done", "completed", "resolved", "fixed",
                               "closed", "approved", "success", "healthy", "active")):
        return SEVERITY["normal"]   # #16A34A green

    # Negative / blocked → red
    if any(w in key for w in ("cancel", "rejected", "failed", "blocked",
                               "abort", "invalid", "removed")):
        return SEVERITY["critical"]  # #DC2626 red

    # In-flight / action needed → orange
    if any(w in key for w in ("progress", "pending", "waiting", "review",
                               "assigned", "escalat", "dispatch")):
        return SEVERITY["major"]    # #EA580C orange

    # New / open → blue (informational, not alarming)
    if any(w in key for w in ("new", "open", "created", "submitted", "raised")):
        return THEME["blue1"]       # WE purple (informational)

    # Not planned / no action → grey (neutral)
    if any(w in key for w in ("plan", "schedule", "defer", "hold", "tbd",
                               "not in", "n/a", "none", " - ")):
        return THEME["gray"]        # #6B7280 grey

    return None  # caller falls back to categorical palette


# ─── Helpers ──────────────────────────────────────────────────────────────────
def fix_arabic(text):
    """Reshape Arabic text so letters connect properly in Plotly charts.

    We use arabic_reshaper to form correct ligatures but deliberately skip
    get_display() (bidi visual-order reversal). Plotly renders via HTML/SVG
    which handles RTL directionality natively; applying get_display() on top
    causes a double-reversal that produces the garbled, reversed characters
    seen in the bar-chart labels.
    """
    if not isinstance(text, str):
        return str(text)
    has_arabic = any("؀" <= ch <= "ۿ" for ch in text)
    if not has_arabic or not _ARABIC_OK:
        return text
    try:
        return arabic_reshaper.reshape(text)  # connect letters; let Plotly handle RTL direction
    except Exception:
        return text


def load_json(name):
    path = os.path.join(OUTPUT_DIR, name)
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def load_artifacts():
    return (
        load_json("analysis.json"),
        load_json("design.json"),
        load_json("insights.json"),
    )


def try_load_llm_figures():
    """Best-effort exec of figures_code.py; return {} on any failure."""
    path = os.path.join(OUTPUT_DIR, "figures_code.py")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            code = f.read()
        sandbox = {}
        exec(code, sandbox)  # noqa: S102 — local, user-owned generated code
        figs = sandbox.get("figures", {})
        return figs if isinstance(figs, dict) else {}
    except Exception:
        return {}


# ─── Aggregation data extraction ──────────────────────────────────────────────
def resolve_data_source(analysis, data_source):
    """Resolve a chart's data_source string into a list of record dicts."""
    if not analysis:
        return []
    aggs = analysis.get("aggregations", {}) or {}
    if not data_source:
        return []
    key = str(data_source).replace("aggregations.", "").strip()
    parts = [p for p in key.split(".") if p]

    node = aggs
    for p in parts:
        if isinstance(node, dict) and p in node:
            node = node[p]
        else:
            node = None
            break

    # Fallback: search aggregations for a matching key
    if node is None and parts:
        target = parts[-1]
        if target in aggs:
            node = aggs[target]

    if node is None:
        # Final fallback: first list-shaped aggregation
        for v in aggs.values():
            if isinstance(v, list) and v:
                node = v
                break

    return _normalize_records(node)


def _normalize_records(node):
    """Coerce various aggregation shapes into a flat list of {field: value}."""
    if node is None:
        return []
    if isinstance(node, list):
        out = []
        for rec in node:
            if isinstance(rec, dict):
                flat = {k: v for k, v in rec.items() if k != "metrics"}
                if isinstance(rec.get("metrics"), dict):
                    flat.update(rec["metrics"])
                out.append(flat)
            else:
                out.append({"value": rec})
        return out
    if isinstance(node, dict):
        # distributions style {"value": count} or nested {desc: [...]}
        # If any value is a list, take the first list.
        for v in node.values():
            if isinstance(v, list):
                return _normalize_records(v)
        # Else treat as label->count map
        return [{"label": k, "value": v} for k, v in node.items()]
    return []


def pick_field(records, preferred, fallbacks):
    """Choose the best key present across records."""
    if not records:
        return None
    keys = set()
    for r in records:
        keys.update(r.keys())
    if preferred and preferred in keys:
        return preferred
    for fb in fallbacks:
        if fb in keys:
            return fb
    return None


def first_numeric_field(records, exclude=()):
    for r in records:
        for k, v in r.items():
            if k in exclude:
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return k
    return None


def extract_xy(records, x_field, y_field):
    x_key = pick_field(records, x_field, ["key", "period", "label", "name", "x"])
    # Guard: y_field must be a hashable string before using it as a dict key
    if y_field and isinstance(y_field, str):
        try:
            y_key = y_field if any(y_field in r for r in records if isinstance(r, dict)) else None
        except TypeError:
            y_key = None
    else:
        y_key = None
    if y_key is None:
        y_key = first_numeric_field(records, exclude={x_key} if x_key else set())
    if y_key is None:
        y_key = pick_field(records, y_field, ["value", "count", "total", "y"])

    xs, ys = [], []
    for r in records:
        xv = r.get(x_key) if x_key else None
        yv = r.get(y_key) if y_key else None
        if xv is None and yv is None:
            continue
        xs.append(fix_arabic(xv) if isinstance(xv, str) else xv)
        try:
            ys.append(float(yv) if yv is not None else 0.0)
        except (TypeError, ValueError):
            ys.append(0.0)
    return xs, ys, x_key, y_key


# ─── Figure builders ──────────────────────────────────────────────────────────
def base_layout(title, x_title, y_title):
    return dict(
        title=dict(text=fix_arabic(title or ""), font=dict(size=15, color=THEME["header"])),
        paper_bgcolor=THEME["card"],
        plot_bgcolor=THEME["card"],
        font=dict(family=THEME["font"], color=THEME["text"], size=12),
        margin=dict(l=220, r=160, t=60, b=80),
        xaxis=dict(title=fix_arabic(x_title or ""), showgrid=True, gridcolor="#E5E7EB", zeroline=False),
        yaxis=dict(title=fix_arabic(y_title or ""), showgrid=True, gridcolor="#E5E7EB", zeroline=False),
        showlegend=False,
        height=380,
    )


def empty_fig(message="No data available for this chart"):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False,
                       font=dict(size=14, color=THEME["gray"]), x=0.5, y=0.5, xref="paper", yref="paper")
    fig.update_layout(**base_layout("", "", ""))
    return fig


_PERCENT_HINTS = ("util", "congest", "percent", "%", "availab", "occupancy", "load", "rate")


def _is_percent_metric(spec, ys):
    """Decide whether the y values are a percentage (RAG by value) vs a count (RAG by rank)."""
    fields = " ".join(str(spec.get(k, "")) for k in ("y_field", "y_title", "title", "threshold_label")).lower()
    name_says_pct = any(h in fields for h in _PERCENT_HINTS)
    in_range = bool(ys) and max(ys) <= 100.0 and min(ys) >= 0
    # Only treat as percentage if the NAME implies it AND values fit 0–100.
    return name_says_pct and in_range


def compute_bar_colors(xs, ys, spec, highlight_top_n):
    """RAG/severity-aware bar colours for telecom-style charts."""
    scheme = (spec.get("color_scheme") or "").lower()
    colors = [THEME["blue1"]] * len(xs)
    warn, crit = 80.0, 90.0
    try:
        if spec.get("threshold_value") is not None:
            crit = float(spec["threshold_value"])
            warn = crit * 0.85
    except Exception:
        pass

    if scheme == "severity":
        # 1) label-based severity wins (e.g. bars literally named Critical/Major)
        label_hits = 0
        for i, xv in enumerate(xs):
            lc = severity_color_for_label(xv)
            if lc:
                colors[i] = lc
                label_hits += 1
        if label_hits:
            return colors
        # 2) percentage metric → colour by absolute thresholds (utilization etc.)
        if _is_percent_metric(spec, ys):
            for i, yv in enumerate(ys):
                colors[i] = severity_color_for_value(yv, warn, crit)
            return colors
        # 3) count metric → colour by rank so the worst offenders stand out
        order = sorted(range(len(ys)), key=lambda i: ys[i], reverse=True)
        n = len(order)
        for rank, i in enumerate(order):
            if rank < max(1, round(n * 0.2)):
                colors[i] = SEVERITY["critical"]
            elif rank < max(2, round(n * 0.5)):
                colors[i] = SEVERITY["major"]
            else:
                colors[i] = THEME["blue1"]
        return colors

    if highlight_top_n:
        order = sorted(range(len(ys)), key=lambda i: ys[i], reverse=True)[:int(highlight_top_n)]
        for i in order:
            colors[i] = THEME["red"]
    return colors


def get_secondary_annotations(records, x_key, sec_field):
    """Map x-label -> secondary metric (e.g. impacted subscribers) for teal annotation."""
    out = {}
    if not sec_field:
        return out
    for r in records:
        lbl = r.get(x_key)
        lbl = fix_arabic(lbl) if isinstance(lbl, str) else lbl
        if sec_field in r and r.get(sec_field) is not None:
            out[str(lbl)] = r.get(sec_field)
    return out


def build_figure(spec, analysis):
    """Deterministically build a Plotly figure from a chart spec + analysis."""
    try:
        ctype = (spec.get("chart_type") or "horizontal_bar").lower()
        records = resolve_data_source(analysis, spec.get("data_source"))
        title = spec.get("title", "")
        x_title = spec.get("x_title", "")
        y_title = spec.get("y_title", "")
        sort_order = spec.get("sort_order", "natural")
        top_n = spec.get("top_n")
        highlight_top_n = spec.get("highlight_top_n")

        if not records:
            return empty_fig()

        xs, ys, x_key, y_key = extract_xy(records, spec.get("x_field"), spec.get("y_field"))
        sec_map = get_secondary_annotations(records, x_key, spec.get("secondary_annotation_field"))

        # ── gauge (overall utilization / availability / network health) ──
        if ctype == "gauge":
            val = (sum(ys) / len(ys)) if ys else 0.0
            if len(ys) == 1:
                val = ys[0]
            looks_pct = val <= 100.0
            axis_max = 100 if looks_pct else (max(ys) * 1.2 if ys else 100)
            invert = bool(spec.get("invert_gauge"))  # True → high is good (health score)
            warn = float(spec.get("threshold_value") or (60 if invert else 80)) if looks_pct else axis_max * 0.8

            if invert:
                # Health score: green at top (high=good), red at bottom (low=bad)
                crit_lo = max(0.0, warn - 20)   # e.g. 40 — critical threshold
                bar_col = (SEVERITY["normal"] if val >= warn
                           else SEVERITY["major"] if val >= crit_lo
                           else SEVERITY["critical"])
                steps = [
                    {"range": [0, crit_lo],   "color": "#FEE2E2"},  # red zone
                    {"range": [crit_lo, warn], "color": "#FEF3C7"},  # amber zone
                    {"range": [warn, axis_max], "color": "#DCFCE7"}, # green zone
                ]
                thr_val = warn
            else:
                # Utilization / congestion: red at top (high=bad), green at bottom
                crit = warn * 1.125 if looks_pct else axis_max * 0.9
                bar_col = severity_color_for_value(val, warn, crit) if looks_pct else THEME["blue1"]
                steps = [
                    {"range": [0, warn],      "color": "#DCFCE7"},  # green zone
                    {"range": [warn, crit],   "color": "#FEF3C7"},  # amber zone
                    {"range": [crit, axis_max], "color": "#FEE2E2"}, # red zone
                ]
                thr_val = crit if looks_pct else warn * 1.125

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(val, 1),
                number={"suffix": "%" if looks_pct else "", "font": {"size": 40, "color": THEME["header"]}},
                gauge={
                    "axis": {"range": [0, axis_max], "tickcolor": THEME["gray"]},
                    "bar": {"color": bar_col, "thickness": 0.3},
                    "steps": steps,
                    "threshold": {"line": {"color": SEVERITY["critical"], "width": 4},
                                  "thickness": 0.8, "value": thr_val},
                },
            ))
            fig.update_layout(
                title=dict(text=fix_arabic(title or ""), font=dict(size=15, color=THEME["header"])),
                paper_bgcolor=THEME["card"], font=dict(family=THEME["font"], color=THEME["text"]),
                margin=dict(l=40, r=40, t=60, b=20), height=380,
            )
            return fig

        if not xs:
            return empty_fig()

        # Sort / Top-N for bar-like / ranking charts (carry annotations along)
        if ctype in ("horizontal_bar", "vertical_bar", "donut") and sort_order in ("desc", "asc"):
            paired = sorted(zip(xs, ys), key=lambda p: p[1], reverse=(sort_order == "desc"))
            xs, ys = [p[0] for p in paired], [p[1] for p in paired]
        if top_n and isinstance(top_n, int) and ctype in ("horizontal_bar", "vertical_bar"):
            xs, ys = xs[:top_n], ys[:top_n]

        layout = base_layout(title, x_title, y_title)

        # ── horizontal bar ──
        if ctype == "horizontal_bar":
            colors = compute_bar_colors(xs, ys, spec, highlight_top_n)
            bar_text = [f"{v:,.0f}" for v in ys]
            # textposition="outside" guarantees every label is readable even on tiny bars.
            # cliponaxis=False keeps labels visible when they overflow the plot area.
            fig = go.Figure(go.Bar(
                y=xs, x=ys, orientation="h", marker_color=colors,
                text=bar_text, textposition="outside",
                textfont=dict(color=THEME["text"], size=11, family=THEME["font"]),
                cliponaxis=False,
            ))
            # Make room on the right for outside labels + secondary annotations
            layout["yaxis"]["autorange"] = "reversed"
            layout["margin"] = dict(l=240, r=200, t=60, b=80)
            max_y = max(ys) if ys else 1
            layout["xaxis"]["range"] = [0, max_y * 1.25]  # ~25% headroom for labels
            # Height scales with number of bars so each one has room to breathe
            layout["height"] = max(380, 36 * len(xs) + 100)
            fig.update_layout(**layout)
            # Teal secondary annotations (e.g. impacted subscribers) at bar ends
            if sec_map:
                for xv, yv in zip(xs, ys):
                    sval = sec_map.get(str(xv))
                    if sval is not None:
                        try:
                            stxt = f"⊕ {float(sval):,.0f}"
                        except (TypeError, ValueError):
                            stxt = f"⊕ {sval}"
                        # Push annotation further right so it doesn't collide with the bar's own text
                        fig.add_annotation(
                            x=yv, y=xv, text=stxt, showarrow=False,
                            xanchor="left", xshift=55,
                            font=dict(color=THEME["teal"], size=11, family=THEME["font"]),
                        )

        # ── vertical bar ──
        elif ctype == "vertical_bar":
            colors = compute_bar_colors(xs, ys, spec, highlight_top_n)
            fig = go.Figure(go.Bar(
                x=xs, y=ys, marker_color=colors,
                text=[f"{v:,.0f}" for v in ys], textposition="outside",
                textfont=dict(color=THEME["text"], size=11, family=THEME["font"]),
                cliponaxis=False,
            ))
            max_y = max(ys) if ys else 1
            layout["yaxis"]["range"] = [0, max_y * 1.15]
            fig.update_layout(**layout)

        # ── grouped bar ──
        elif ctype == "grouped_bar":
            numeric_keys = [k for k in (records[0].keys() if records else [])
                            if k != x_key and all(isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)
                                                  for r in records if r.get(k) is not None)]
            numeric_keys = numeric_keys[:4] or ([y_key] if y_key else [])
            fig = go.Figure()
            for idx, nk in enumerate(numeric_keys):
                series = [float(r.get(nk) or 0) for r in records]
                fig.add_trace(go.Bar(name=fix_arabic(nk), x=xs, y=series,
                                     marker_color=CATEGORICAL_PALETTE[idx % len(CATEGORICAL_PALETTE)]))
            layout["barmode"] = "group"
            layout["showlegend"] = True
            fig.update_layout(**layout)

        # ── line / area ──
        elif ctype in ("line", "area"):
            fill = "tozeroy" if ctype == "area" else None
            fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers", fill=fill,
                                       line=dict(color=THEME["blue1"], width=3),
                                       marker=dict(size=6, color=THEME["blue1"])))
            fig.update_layout(**layout)

        # ── scatter ──
        elif ctype == "scatter":
            fig = go.Figure(go.Scatter(x=xs, y=ys, mode="markers",
                                       marker=dict(size=10, color=THEME["blue1"], opacity=0.7)))
            fig.update_layout(**layout)

        # ── donut ──
        elif ctype == "donut":
            total = sum(ys) or 1
            pulls = [0.05 if (v / total) < 0.10 else 0 for v in ys]
            # Use alarm-severity colours when slice labels are severities/states
            scheme = (spec.get("color_scheme") or "").lower()
            slice_colors = list(CATEGORICAL_PALETTE)
            if scheme == "severity":
                for i, lbl in enumerate(xs):
                    sc = severity_color_for_label(lbl)
                    if sc:
                        slice_colors[i % len(slice_colors)] = sc
            fig = go.Figure(go.Pie(labels=xs, values=ys, hole=0.52, pull=pulls,
                                   marker=dict(colors=slice_colors),
                                   textinfo="label+percent"))
            layout["showlegend"] = True
            layout["xaxis"] = {}
            layout["yaxis"] = {}
            fig.update_layout(**layout)

        # ── histogram ──
        elif ctype == "histogram":
            fig = go.Figure(go.Histogram(x=ys if ys else xs, marker_color=THEME["blue1"], nbinsx=20))
            fig.update_layout(**layout)

        # ── heatmap ──
        elif ctype == "heatmap":
            fig = go.Figure(go.Heatmap(z=[ys], x=xs, colorscale="Blues"))
            fig.update_layout(**layout)

        # ── table ──
        elif ctype == "table":
            header = [fix_arabic(x_key or "Label"), fix_arabic(y_key or "Value")]
            fig = go.Figure(go.Table(
                header=dict(values=header, fill_color=THEME["header"],
                            font=dict(color="white", size=13), align="left"),
                cells=dict(values=[xs, ys], fill_color=THEME["bg"], align="left"),
            ))
            fig.update_layout(title=fix_arabic(title), paper_bgcolor=THEME["card"],
                              font=dict(family=THEME["font"]), margin=dict(l=20, r=20, t=60, b=20), height=380)

        else:
            fig = go.Figure(go.Bar(y=xs, x=ys, orientation="h", marker_color=THEME["blue1"]))
            fig.update_layout(**layout)

        # ── overlays: threshold + average lines (axis charts only) ──
        if ctype in ("horizontal_bar", "vertical_bar", "line", "area", "scatter") and xs:
            if spec.get("has_threshold_line") and spec.get("threshold_value") is not None:
                tv = float(spec["threshold_value"])
                if ctype == "horizontal_bar":
                    fig.add_vline(x=tv, line_dash="dash", line_color=THEME["red"],
                                  annotation_text=fix_arabic(spec.get("threshold_label") or "Threshold"))
                else:
                    fig.add_hline(y=tv, line_dash="dash", line_color=THEME["red"],
                                  annotation_text=fix_arabic(spec.get("threshold_label") or "Threshold"))
            if spec.get("show_average_line") and ys:
                avg = sum(ys) / len(ys)
                if ctype == "horizontal_bar":
                    fig.add_vline(x=avg, line_dash="dot", line_color=THEME["gray"],
                                  annotation_text=f"Avg {avg:,.1f}")
                else:
                    fig.add_hline(y=avg, line_dash="dot", line_color=THEME["gray"],
                                  annotation_text=f"Avg {avg:,.1f}")

        return fig
    except Exception:
        traceback.print_exc()
        return empty_fig("Chart could not be rendered")


# ─── App ──────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="AI Analytics Dashboard",
    suppress_callback_exceptions=True,
    update_title=None,
)
server = app.server

# Inject NOC-style CSS (pulsing urgent banner, smooth card hover).
app.index_string = """<!DOCTYPE html>
<html>
<head>
{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
/* ── Google Fonts ─────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

body { font-family: 'Inter', 'DejaVu Sans', sans-serif; background: #F7F5FC; }

/* ── WE brand animations ─────────────────── */
@keyframes urgentPulse {
  0%, 100% { box-shadow: inset 0 0 0 0 rgba(255,255,255,0); }
  50%       { box-shadow: inset 0 0 50px 0 rgba(255,255,255,0.30); }
}
@keyframes fadeInDown {
  from { opacity:0; transform:translateY(-10px); }
  to   { opacity:1; transform:translateY(0); }
}
.urgent-pulse { animation: urgentPulse 1.8s ease-in-out infinite; }

/* ── Card hover lift ─────────────────────── */
.card {
  transition: transform .18s ease, box-shadow .18s ease;
  border: none !important;
}
.card:hover {
  transform: translateY(-3px);
  box-shadow: 0 10px 28px rgba(91,32,131,0.14) !important;
}

/* ── Tabs: WE purple underline ───────────── */
.nav-tabs .nav-link {
  color: #6B7280 !important;
  font-weight: 500;
  border: none !important;
  border-bottom: 3px solid transparent !important;
  border-radius: 0 !important;
  padding: 10px 20px;
  transition: color .15s, border-color .15s;
}
.nav-tabs .nav-link:hover {
  color: #5B2083 !important;
  border-bottom-color: #C7A8E8 !important;
}
.nav-tabs .nav-link.active {
  color: #5B2083 !important;
  font-weight: 700;
  border-bottom: 3px solid #5B2083 !important;
  background: transparent !important;
}
.nav-tabs { border-bottom: 1px solid #E8D8F8 !important; }

/* ── Dropdown: WE purple focus ───────────── */
.Select-control:focus-within, .Select--is-focused .Select-control {
  border-color: #5B2083 !important;
  box-shadow: 0 0 0 3px rgba(91,32,131,0.15) !important;
}

/* ── Scrollbar (subtle purple) ───────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #F7F5FC; }
::-webkit-scrollbar-thumb { background: #C7A8E8; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #5B2083; }

/* ── Buttons ─────────────────────────────── */
.btn-primary { background-color: #5B2083 !important; border-color: #5B2083 !important; }
.btn-primary:hover { background-color: #7B3BAF !important; border-color: #7B3BAF !important; }
</style>
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>"""


def kpi_card(kpi, dark=False):
    color_map = {
        "green":  THEME["green"],  "red":    THEME["red"],
        "blue":   THEME["header"], "orange": THEME["orange"],
        "yellow": THEME["yellow"], "teal":   THEME["teal"],
        "purple": THEME["header"],
    }
    accent      = color_map.get((kpi.get("color_hint") or "blue").lower(), THEME["header"])
    bg          = DARK["card"] if dark else THEME["card"]
    label_color = "#B39DDB" if dark else THEME["gray"]
    text_color  = DARK["text"] if dark else THEME["text"]
    return dbc.Card(
        dbc.CardBody([
            html.Div(
                style={"display": "flex", "alignItems": "flex-start", "justifyContent": "space-between"},
                children=[
                    html.Div([
                        html.Div(fix_arabic(str(kpi.get("label", ""))),
                                 style={"fontSize": "11px", "fontWeight": "600", "color": label_color,
                                        "textTransform": "uppercase", "letterSpacing": "0.5px"}),
                        html.Div(str(kpi.get("value", "—")),
                                 style={"fontSize": "28px", "fontWeight": "800", "color": accent,
                                        "lineHeight": "1.2", "marginTop": "4px", "color": text_color}),
                    ]),
                    html.Div(kpi.get("icon_hint", "📊"),
                             style={"fontSize": "28px", "opacity": "0.85",
                                    "backgroundColor": f"{accent}15",
                                    "borderRadius": "10px", "padding": "8px",
                                    "lineHeight": "1"}),
                ],
            ),
            html.Div(style={"height": "3px", "borderRadius": "2px",
                             "backgroundColor": accent, "marginTop": "14px",
                             "opacity": "0.7"}),
        ], style={"padding": "18px"}),
        style={"backgroundColor": bg, "borderRadius": "14px", "border": "none",
               "boxShadow": "0 2px 12px rgba(91,32,131,0.09)",
               "borderTop": f"3px solid {accent}"},
        className="h-100",
    )


def chart_card(spec, fig, dark=False):
    bg          = DARK["card"] if dark else THEME["card"]
    header_bg   = DARK["card"] if dark else "#F3EDF9"   # very light purple tint header
    title_color = DARK["text"] if dark else THEME["header"]
    insight     = spec.get("insight", "")
    title_text  = fix_arabic(spec.get("title", ""))
    return dbc.Card([
        # Card header: purple-tinted title strip
        html.Div(
            style={"backgroundColor": header_bg, "padding": "10px 18px",
                   "borderBottom": f"2px solid {THEME['header']}",
                   "borderRadius": "14px 14px 0 0",
                   "display": "flex", "alignItems": "center", "gap": "8px"},
            children=[
                html.Div(style={"width": "4px", "height": "18px", "borderRadius": "2px",
                                "backgroundColor": THEME["header"], "flexShrink": "0"}),
                html.Span(title_text,
                          style={"fontSize": "14px", "fontWeight": "700",
                                 "color": title_color, "letterSpacing": "0.2px"}),
            ],
        ),
        # Card body: chart + optional insight
        dbc.CardBody([
            dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "360px"}),
            html.Div(
                [html.Span("💡 ", style={"marginRight": "4px"}),
                 html.Span(fix_arabic(insight),
                           style={"fontSize": "12px", "color": THEME["header"],
                                  "fontStyle": "italic", "opacity": "0.85"})],
                style={"padding": "4px 2px 2px 2px"},
            ) if insight else None,
        ], style={"padding": "12px 16px"}),
    ],
    style={"backgroundColor": bg, "borderRadius": "14px", "border": "none",
           "boxShadow": "0 2px 16px rgba(91,32,131,0.08)"},
    className="mb-3 h-100",
    )


def insights_panel(insights, dark=False):
    if not insights:
        return None
    bg         = DARK["card"] if dark else THEME["card"]
    text_color = DARK["text"] if dark else THEME["text"]

    risk_level = insights.get("risk_level", "")
    risk_badge_color = {
        "CRITICAL": THEME["red"], "HIGH": THEME["orange"],
        "MEDIUM": THEME["yellow"], "LOW": THEME["green"],
    }.get(risk_level.upper() if risk_level else "", THEME["header"])

    def section(title, items, color, icon=""):
        if not items:
            return None
        return html.Div([
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "6px", "marginTop": "14px"},
                children=[
                    html.Div(style={"width": "3px", "height": "16px", "borderRadius": "2px",
                                    "backgroundColor": color}),
                    html.Span(f"{icon} {title}" if icon else title,
                              style={"fontSize": "13px", "fontWeight": "700", "color": color}),
                ],
            ),
            html.Ul([
                html.Li(fix_arabic(str(it)),
                        style={"fontSize": "13px", "color": text_color,
                               "lineHeight": "1.6", "marginBottom": "2px"})
                for it in items
            ], style={"paddingLeft": "18px", "marginTop": "6px"}),
        ])

    return dbc.Card([
        # Header strip with WE purple gradient
        html.Div(
            style={"background": f"linear-gradient(135deg, {THEME['header']}, {THEME['purple2']})",
                   "padding": "14px 20px", "borderRadius": "14px 14px 0 0",
                   "display": "flex", "alignItems": "center", "justifyContent": "space-between"},
            children=[
                html.Div([
                    html.Div("📋 Network Operations Brief",
                             style={"fontSize": "11px", "color": "#E8D8F8",
                                    "fontWeight": "600", "textTransform": "uppercase",
                                    "letterSpacing": "0.8px"}),
                    html.Div(fix_arabic(insights.get("summary_title", "Executive Summary")),
                             style={"fontSize": "16px", "fontWeight": "800",
                                    "color": "white", "marginTop": "2px"}),
                ]),
                html.Div(risk_level,
                         style={"backgroundColor": risk_badge_color, "color": "white",
                                "fontWeight": "700", "fontSize": "12px", "padding": "4px 12px",
                                "borderRadius": "20px", "boxShadow": "0 2px 6px rgba(0,0,0,0.2)"})
                if risk_level else None,
            ],
        ),
        dbc.CardBody([
            section("Highlights", insights.get("highlights"), THEME["green"], "✅"),
            section("Risks & Alerts", insights.get("risks"), THEME["orange"], "⚠"),
            section("Recommended Actions", insights.get("recommended_actions"), THEME["header"], "→"),
            # Urgent action callout
            html.Div(
                [html.Span("🚨 ", style={"fontSize": "16px"}),
                 html.Span(fix_arabic(str(insights.get("urgent_action", ""))),
                           style={"fontSize": "13px", "fontWeight": "600", "color": THEME["red"]})],
                style={"backgroundColor": "#FEF2F2", "border": f"1px solid {THEME['red']}",
                       "borderRadius": "8px", "padding": "10px 14px", "marginTop": "16px"},
            ) if insights.get("urgent_action") else None,
        ], style={"padding": "16px 20px"}),
    ],
    style={"backgroundColor": bg, "borderRadius": "14px", "border": "none",
           "boxShadow": "0 2px 16px rgba(91,32,131,0.10)"},
    className="mb-3",
    )


def build_tab_content(tab_index, analysis, design, insights, figures, dark=False):
    charts = [c for c in (design.get("charts", []) if design else []) if c.get("tab", 0) == tab_index]
    children = []

    # KPI row (first tab only, or all tabs if single)
    kpis = (analysis.get("kpis", []) if analysis else [])
    if kpis and (tab_index == 0 or (design and design.get("layout_hint") == "single_tab")):
        children.append(dbc.Row(
            [dbc.Col(kpi_card(k, dark), xs=6, md=4, lg=2, className="mb-3") for k in kpis[:6]],
            className="g-3 mb-2",
        ))

    # Charts grouped by priority (row)
    by_priority = {}
    for c in charts:
        by_priority.setdefault(c.get("priority", 1), []).append(c)

    for priority in sorted(by_priority.keys()):
        row_charts = by_priority[priority]
        cols = []
        for spec in row_charts:
            fig = figures.get(spec.get("id"))
            if fig is None:
                fig = build_figure(spec, analysis)
            width = int(spec.get("width_cols", 6))
            width = max(3, min(12, width))
            cols.append(dbc.Col(chart_card(spec, fig, dark), xs=12, md=12, lg=width))
        children.append(dbc.Row(cols, className="g-3"))

    # Insights panel on the last tab (or single tab)
    tab_count = len(design.get("tab_names", ["Overview"])) if design else 1
    if insights and (tab_index == tab_count - 1 or tab_count == 1):
        children.append(insights_panel(insights, dark))

    if not children:
        children.append(html.Div("No charts configured for this tab.",
                                  style={"padding": "40px", "textAlign": "center", "color": THEME["gray"]}))
    return html.Div(children, style={"padding": "16px"})


def welcome_layout():
    return dbc.Container(fluid=True,
                         style={"padding": "80px 60px", "background": THEME["bg"], "minHeight": "100vh"},
                         children=[
        dbc.Row(justify="center", children=[dbc.Col(md=7, lg=6, children=[
            dbc.Card(dbc.CardBody([
                # WE logo + title
                html.Div(
                    style={"textAlign": "center", "marginBottom": "28px"},
                    children=[
                        html.Img(src="/assets/we_logo.svg",
                                 style={"width": "90px", "height": "90px", "marginBottom": "18px"}),
                        html.H2("WE Network Analytics",
                                style={"color": THEME["header"], "fontWeight": "800",
                                       "fontSize": "26px", "marginBottom": "6px"}),
                        html.Div("OSS Technical Operations — Universal Dashboard",
                                 style={"color": THEME["gray"], "fontSize": "13px",
                                        "letterSpacing": "0.3px"}),
                    ],
                ),
                # Divider
                html.Hr(style={"borderColor": THEME["purple3"], "margin": "0 0 24px 0"}),
                # Instructions
                html.Div(
                    style={"textAlign": "center"},
                    children=[
                        html.Div("📁", style={"fontSize": "48px", "marginBottom": "12px"}),
                        html.P("Upload an Excel file in the control panel to get started.",
                               style={"color": THEME["text"], "fontSize": "15px",
                                      "fontWeight": "500", "marginBottom": "8px"}),
                        html.P("Open the Streamlit control panel on port 8501, upload your file, and run the pipeline.",
                               style={"color": THEME["gray"], "fontSize": "13px",
                                      "lineHeight": "1.6"}),
                    ],
                ),
                html.Div(id="welcome-status",
                         style={"textAlign": "center", "color": THEME["header"],
                                "fontSize": "13px", "fontWeight": "500", "marginTop": "20px"}),
                dcc.Interval(id="welcome-interval", interval=4000, n_intervals=0),
            ], style={"padding": "40px 36px"}),
            style={"borderRadius": "18px", "border": "none",
                   "boxShadow": "0 8px 40px rgba(91,32,131,0.14)"},
            ),
        ])]),
    ])


def serve_layout():
    analysis, design, insights = load_artifacts()
    figures = try_load_llm_figures() if analysis else {}

    if not analysis:
        return html.Div(style={"backgroundColor": THEME["bg"], "minHeight": "100vh"}, children=[welcome_layout()])

    meta = analysis.get("meta", {}) or {}
    domain = (meta.get("domain") or "data").lower()
    is_telecom = domain == "telecom"
    title = (design.get("dashboard_title") if design else None) or \
            f"{domain.title()} Analytics"
    tab_names = (design.get("tab_names") if design else None) or ["Overview"]
    urgent = analysis.get("urgent_flag", {}) or {}

    # NOC-style subtitle: generated timestamp + scope
    rows = meta.get("row_count", 0)
    scope_word = "network elements" if is_telecom else "records"
    subtitle = f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}  ·  {rows:,} {scope_word}"
    if is_telecom:
        risk = (insights or {}).get("risk_level", "")
        if risk:
            subtitle += f"  ·  Posture: {risk}"

    header = html.Div(
        style={"background": f"linear-gradient(135deg, {THEME['header']} 0%, {THEME['purple2']} 100%)",
               "color": "white", "minHeight": "68px",
               "padding": "0 28px", "display": "flex", "alignItems": "center",
               "justifyContent": "space-between",
               "boxShadow": "0 3px 16px rgba(91,32,131,0.30)"},
        children=[
            # Left: WE logo + title block
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "14px"},
                children=[
                    html.Img(src="/assets/we_logo.svg",
                             style={"width": "44px", "height": "44px", "borderRadius": "50%",
                                    "boxShadow": "0 2px 8px rgba(0,0,0,0.25)"}),
                    html.Div([
                        html.Div(fix_arabic(title),
                                 style={"fontSize": "18px", "fontWeight": "800",
                                        "color": "white", "letterSpacing": "0.2px",
                                        "lineHeight": "1.2"}),
                        html.Div(subtitle,
                                 style={"fontSize": "11px", "color": "#D8C4F0",
                                        "marginTop": "2px", "letterSpacing": "0.3px"}),
                    ]),
                ],
            ),
            # Right: controls
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "6px"},
                children=[
                    dbc.Switch(id="dark-toggle", label="🌙 Dark", value=False,
                               style={"display": "inline-block", "marginRight": "10px",
                                      "color": "white", "fontSize": "13px"}),
                    dbc.Switch(id="rtl-toggle", label="↔ RTL", value=False,
                               style={"display": "inline-block", "color": "white",
                                      "fontSize": "13px"}),
                ],
            ),
        ],
    )

    # Alarm-severity legend strip (telecom NOC convention)
    severity_legend = None
    if is_telecom:
        def _chip(label, color):
            return html.Span([
                html.Span(style={"display": "inline-block", "width": "10px", "height": "10px",
                                 "borderRadius": "50%", "backgroundColor": color, "marginRight": "5px"}),
                html.Span(label, style={"fontSize": "12px", "color": THEME["text"]}),
            ], style={"marginRight": "18px", "display": "inline-flex", "alignItems": "center"})
        severity_legend = html.Div(
            [html.Span("Alarm severity: ", style={"fontSize": "12px", "fontWeight": "600",
                                                  "color": THEME["gray"], "marginRight": "10px"}),
             _chip("Critical (≥90%)", SEVERITY["critical"]),
             _chip("Major (≥80%)", SEVERITY["major"]),
             _chip("Minor", SEVERITY["minor"]),
             _chip("Normal", SEVERITY["normal"]),
             _chip("Impacted subscribers ⊕", THEME["teal"])],
            style={"backgroundColor": THEME["card"], "padding": "8px 28px",
                   "borderBottom": "1px solid #E5E7EB"},
        )

    urgent_banner = None
    if urgent.get("exists"):
        sev = urgent.get("severity", "HIGH")
        urgent_banner = html.Div(
            f"🚨 {sev}: {fix_arabic(urgent.get('message', 'Urgent attention required'))}",
            className="urgent-pulse",
            style={"backgroundColor": THEME["red"], "color": "white", "fontWeight": "700",
                   "padding": "12px 28px", "textAlign": "center", "fontSize": "15px"},
        )

    tabs = dbc.Tabs(
        [dbc.Tab(build_tab_content(i, analysis, design, insights, figures, dark=False),
                 label=fix_arabic(name), tab_id=f"tab-{i}")
         for i, name in enumerate(tab_names)],
        id="main-tabs", active_tab="tab-0",
    )

    footer = html.Div(
        style={"backgroundColor": THEME["card"], "padding": "14px 28px", "borderTop": "1px solid #E5E7EB",
               "display": "flex", "alignItems": "center", "gap": "12px", "flexWrap": "wrap"},
        children=[
            dbc.Button("⬇ Download PNG", id="btn-png", color="primary", size="sm"),
            dbc.Button("📄 Export PDF", id="btn-pdf", color="secondary", size="sm"),
            html.Span("Auto-refresh:", style={"fontSize": "13px", "color": THEME["gray"], "marginLeft": "12px"}),
            dcc.Dropdown(
                id="refresh-rate",
                options=[{"label": "Off", "value": 0}, {"label": "30s", "value": 30000},
                         {"label": "1 min", "value": 60000}, {"label": "5 min", "value": 300000}],
                value=0, clearable=False, style={"width": "110px"},
            ),
            dcc.Upload(id="upload-new",
                       children=dbc.Button("📁 Upload New File", color="light", size="sm"),
                       multiple=True, style={"marginLeft": "auto"}),
            html.Span(id="upload-status", style={"fontSize": "12px", "color": THEME["teal"]}),
            dcc.Download(id="download-file"),
            dcc.Interval(id="auto-interval", interval=60000, disabled=True),
        ],
    )

    return html.Div(
        id="page-root",
        style={"backgroundColor": THEME["bg"], "minHeight": "100vh"},
        children=[header, urgent_banner, severity_legend, html.Div(tabs, id="tabs-wrap"), footer],
    )


app.layout = serve_layout


# ─── Callbacks ────────────────────────────────────────────────────────────────
@app.callback(
    Output("page-root", "style"),
    Output("tabs-wrap", "dir"),
    Input("dark-toggle", "value"),
    Input("rtl-toggle", "value"),
    prevent_initial_call=True,
)
def toggle_theme(dark, rtl):
    bg = DARK["bg"] if dark else THEME["bg"]
    style = {"backgroundColor": bg, "minHeight": "100vh"}
    return style, ("rtl" if rtl else "ltr")


@app.callback(
    Output("auto-interval", "interval"),
    Output("auto-interval", "disabled"),
    Input("refresh-rate", "value"),
)
def set_refresh(rate):
    if not rate:
        return 60000, True
    return rate, False


app.clientside_callback(
    "function(n){ if(n){ window.location.reload(); } return ''; }",
    Output("upload-status", "title"),
    Input("auto-interval", "n_intervals"),
    prevent_initial_call=True,
)


def _all_figures():
    analysis, design, insights = load_artifacts()
    figures = try_load_llm_figures() if analysis else {}
    out = []
    for spec in (design.get("charts", []) if design else []):
        fig = figures.get(spec.get("id")) or build_figure(spec, analysis)
        out.append((spec.get("title", "chart"), fig))
    return out


@app.callback(
    Output("download-file", "data"),
    Input("btn-png", "n_clicks"),
    Input("btn-pdf", "n_clicks"),
    prevent_initial_call=True,
)
def export_report(png_clicks, pdf_clicks):
    trigger = ctx.triggered_id
    figs = _all_figures()
    if not figs:
        return no_update
    try:
        from PIL import Image
        images = []
        for _, fig in figs:
            png_bytes = fig.to_image(format="png", width=1000, height=420, scale=2)
            images.append(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
        if not images:
            return no_update

        if trigger == "btn-pdf":
            pdf_path = os.path.join(EXPORTS_DIR, "report.pdf")
            images[0].save(pdf_path, save_all=True, append_images=images[1:])
            return dcc.send_file(pdf_path)
        else:
            total_h = sum(im.height for im in images)
            max_w = max(im.width for im in images)
            canvas = Image.new("RGB", (max_w, total_h), "white")
            y = 0
            for im in images:
                canvas.paste(im, (0, y))
                y += im.height
            png_path = os.path.join(EXPORTS_DIR, "report.png")
            canvas.save(png_path)
            return dcc.send_file(png_path)
    except Exception as e:
        print(f"Export failed: {e}")
        return no_update


@app.callback(
    Output("upload-status", "children"),
    Input("upload-new", "contents"),
    State("upload-new", "filename"),
    prevent_initial_call=True,
)
def handle_upload(contents_list, names_list):
    if not contents_list:
        return no_update
    import base64
    import subprocess
    import sys
    if not isinstance(contents_list, list):
        contents_list, names_list = [contents_list], [names_list]
    saved = []
    for content, name in zip(contents_list, names_list):
        try:
            header, b64 = content.split(",", 1)
            data = base64.b64decode(b64)
            dest = os.path.join(DATA_DIR, name)
            with open(dest, "wb") as f:
                f.write(data)
            saved.append(dest)
        except Exception as e:
            return f"Upload error: {e}"
    try:
        with open(os.path.join(OUTPUT_DIR, "uploaded_files.json"), "w") as f:
            json.dump(saved, f)
        log = open(os.path.join(OUTPUT_DIR, "pipeline.log"), "ab")
        subprocess.Popen([sys.executable, os.path.join(ROOT, "orchestrate.py")] + saved,
                         cwd=ROOT, stdout=log, stderr=log)
        return f"✅ Re-running pipeline on {len(saved)} file(s)…"
    except Exception as e:
        return f"Pipeline launch error: {e}"


# Welcome-screen status (only present before first run)
@app.callback(
    Output("welcome-status", "children"),
    Input("welcome-interval", "n_intervals"),
    prevent_initial_call=False,
)
def welcome_status(n):
    status = load_json("pipeline_status.json")
    if not status:
        return "⬜ Waiting for pipeline to start…"
    running = [k for k, v in status.items() if v.get("status") == "running"]
    done = [k for k, v in status.items() if v.get("status") == "done"]
    if running:
        return f"⏳ Running… ({', '.join(running)})"
    if len(done) == 5:
        return "✅ Complete — refresh this page to view your dashboard."
    return "⬜ Waiting for pipeline to start…"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
