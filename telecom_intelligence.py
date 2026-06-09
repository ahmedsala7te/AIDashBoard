"""telecom_intelligence.py — Operator / carrier analytics engine.

THE FLAGSHIP DOMAIN FEATURE.

WE (Telecom Egypt) is a *wholesale* access provider: a single MSAN / PE-router
carries subscribers belonging to several downstream operators (Vodafone, Orange,
Etisalat, Noor) plus WE's own retail customers (WE Data) and bitstream wholesale.
The single most important operational question is therefore:

    "When this network element congests / fails, WHICH operator's customers
     are affected, and how many?"

The rest of the pipeline historically collapsed every operator into one
`subscribers` number and threw this away. This module recovers it.

DESIGN PRINCIPLES
-----------------
1. UNIVERSAL FIRST. We detect operator columns by *pattern* (``*_sub``,
   ``*_subscribers``, ``*_lines`` …) so the engine works on any telecom
   subscriber dataset, not just WE's six known operators.
2. KNOWN OPERATORS GET NICE NAMES. A keyword map turns ``etisilat_sub`` (note
   the real-world typo) into "Etisalat", ``voda_sub`` into "Vodafone", etc.
3. TOTALS ARE NOT OPERATORS. Columns like ``total_voice_sub`` / ``subscribers``
   are aggregate totals and are excluded from the per-operator set.
4. PURE FUNCTIONS, NO SIDE EFFECTS. Everything returns plain dicts/lists so the
   orchestrator can merge results and the math stays deterministic.
"""
from __future__ import annotations

import re
import pandas as pd


# ─── Operator vocabulary ──────────────────────────────────────────────────────
# canonical_name -> keyword variants found in column names.
# Order matters: more specific keys first so "we_data" wins over a bare "we".
OPERATOR_KEYWORDS: list[tuple[str, list[str]]] = [
    ("Vodafone",  ["vodafone", "voda", "vfe", "vf_"]),
    ("Orange",    ["orange", "mobinil"]),
    ("Etisalat",  ["etisalat", "etisilat", "etislat"]),   # real exports misspell it
    ("Noor",      ["noor", "nour"]),
    ("WE Data",   ["we_data", "wedata", "we_sub", "te_data", "te_sub"]),
    ("WE Voice",  ["we_voice", "te_voice"]),
    ("Bitstream", ["bitstream", "bit_stream", "bstream"]),
    ("ADSL",      ["adsl"]),
    ("FTTH",      ["ftth", "fiber", "fibre"]),
]

# Tokens that mark a column as a TOTAL / aggregate, not an individual operator.
_TOTAL_TOKENS = ("total", "all", "sum", "grand", "overall", "aggregate")

# Suffixes that mark a column as a per-segment subscriber count.
_SUB_SUFFIXES = ("_sub", "_subs", "_subscriber", "_subscribers", "_lines",
                 "_customers", "_users", "_count")

# WE / Telecom Egypt's own retail brands (everything else is wholesale).
_RETAIL_OPERATORS = {"WE DATA", "WE VOICE", "WE", "TE", "TELECOM EGYPT"}


# ─── Small local helpers (kept here so the module is import-standalone) ────────
def _norm(name) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _round(v):
    try:
        f = float(v)
        return int(f) if f == int(f) else round(f, 2)
    except (TypeError, ValueError):
        return v


def _canonical_operator(col_name: str) -> str | None:
    """Map a column name to a canonical operator name, or None if it isn't one."""
    nm = _norm(col_name)
    for canonical, kws in OPERATOR_KEYWORDS:
        if any(kw in nm for kw in kws):
            return canonical
    return None


def _pretty_unknown(col_name: str) -> str:
    """Turn an unknown operator-ish column into a readable label.
    'mvno_x_sub' -> 'Mvno X'."""
    nm = _norm(col_name)
    for suf in ("_sub", "_subs", "_subscriber", "_subscribers", "_lines",
                "_customers", "_users", "_count"):
        if nm.endswith(suf):
            nm = nm[: -len(suf)]
            break
    return nm.replace("_", " ").strip().title() or col_name


def _is_numeric_or_coercible(series: pd.Series, threshold: float = 0.6) -> bool:
    """True if the column is numeric OR at least `threshold` of its non-null
    values parse as numbers. Telecom exports store counts as strings like
    '400.00' with '-' placeholders, so a plain dtype check misses them."""
    if pd.api.types.is_numeric_dtype(series):
        return True
    coerced = pd.to_numeric(series, errors="coerce")
    n_total = series.notna().sum()
    if n_total == 0:
        return False
    return (coerced.notna().sum() / n_total) >= threshold


def _num(series: pd.Series) -> pd.Series:
    """Coerce a possibly-string subscriber column to numeric, treating
    '-' / '' / 'N/A' placeholders as 0."""
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


# ─── Detection ────────────────────────────────────────────────────────────────
def detect_operator_columns(df: pd.DataFrame) -> dict[str, str]:
    """Find per-operator subscriber columns.

    Returns an ordered mapping {operator_label: column_name}.

    A column qualifies when it is numeric AND its name looks like a per-segment
    subscriber count (ends with a sub-suffix or matches a known operator keyword),
    AND it is not an aggregate total.
    """
    operators: dict[str, str] = {}
    for c in df.columns:
        nm = _norm(c)
        if any(tok in nm for tok in _TOTAL_TOKENS):
            continue  # skip totals like total_voice_sub
        if not _is_numeric_or_coercible(df[c]):
            continue
        canonical = _canonical_operator(c)
        looks_like_sub = nm.endswith(_SUB_SUFFIXES)
        if canonical is None and not looks_like_sub:
            continue
        label = canonical or _pretty_unknown(c)
        # If two columns map to the same operator, keep the one with the larger sum.
        if label in operators:
            prev = operators[label]
            if _num(df[c]).sum() <= _num(df[prev]).sum():
                continue
        operators[label] = c
    return operators


def find_total_subscriber_column(df: pd.DataFrame, operator_cols: dict[str, str]) -> str | None:
    """Find an explicit aggregate-total subscriber column, if present."""
    op_set = set(operator_cols.values())
    candidates = []
    for c in df.columns:
        if c in op_set or not _is_numeric_or_coercible(df[c]):
            continue
        nm = _norm(c)
        if nm in ("subscribers", "subscriber", "subs", "total_subscribers",
                  "total_sub", "total_subs") or (
                any(t in nm for t in _TOTAL_TOKENS) and nm.endswith(_SUB_SUFFIXES)):
            candidates.append(c)
    # Prefer the broadest total (largest sum)
    if candidates:
        return max(candidates, key=lambda c: _num(df[c]).sum())
    return None


def is_retail_operator(label: str) -> bool:
    return label.strip().upper() in _RETAIL_OPERATORS


# ─── Analytics ────────────────────────────────────────────────────────────────
def build_operator_analytics(
    df: pd.DataFrame,
    entity_col: str | None = None,
    severity_col: str | None = None,
    severity_high_is_bad: bool = True,
) -> dict:
    """Compute operator-level analytics.

    Parameters
    ----------
    df : the (already cleaned) dataframe
    entity_col : the network-element identifier column (for exposure ranking)
    severity_col : a congestion / critical-time / utilisation column. When present
        we compute each operator's CONGESTION EXPOSURE — how many of its
        subscribers sit on the worst-affected elements.
    severity_high_is_bad : True when a higher severity value means worse
        (critical time, utilisation). False when higher = healthier.

    Returns a dict with keys (any of which may be absent if not computable):
        operator_columns      {label: column}
        aggregations          {operator_mix, operator_exposure, wholesale_vs_retail}
        kpis                  list of KPI dicts
        summary               {n_operators, largest_operator, most_exposed_operator, ...}
    """
    op_cols = detect_operator_columns(df)
    if len(op_cols) < 2:
        return {}  # not an operator-segmented dataset; nothing to do

    out: dict = {"operator_columns": op_cols, "aggregations": {}, "kpis": [], "summary": {}}

    # ── 1. Operator mix: total subscribers per operator ──────────────────────
    totals = {}
    for label, col in op_cols.items():
        totals[label] = float(_num(df[col]).sum())
    grand_total = sum(totals.values()) or 1.0
    mix_recs = []
    for label, total in sorted(totals.items(), key=lambda kv: kv[1], reverse=True):
        mix_recs.append({
            "key": label,
            "metrics": {
                "subscribers": _round(total),
                "share_pct": _round(total / grand_total * 100),
                "segment": "retail" if is_retail_operator(label) else "wholesale",
            },
        })
    out["aggregations"]["operator_mix"] = mix_recs

    # ── 2. Wholesale vs retail split ─────────────────────────────────────────
    wholesale = sum(t for l, t in totals.items() if not is_retail_operator(l))
    retail = sum(t for l, t in totals.items() if is_retail_operator(l))
    if retail > 0 or wholesale > 0:
        out["aggregations"]["wholesale_vs_retail"] = [
            {"key": "Wholesale (other operators)", "metrics": {"subscribers": _round(wholesale)}},
            {"key": "Retail (WE own)", "metrics": {"subscribers": _round(retail)}},
        ]

    # ── 3. Congestion exposure per operator (needs a severity column) ────────
    most_exposed = None
    if severity_col and severity_col in df.columns and entity_col:
        sev = pd.to_numeric(df[severity_col], errors="coerce")
        if sev.notna().any():
            # "Affected" = elements in the worst quartile of severity
            if severity_high_is_bad:
                threshold = sev.quantile(0.75)
                affected_mask = sev >= threshold
            else:
                threshold = sev.quantile(0.25)
                affected_mask = sev <= threshold
            affected_mask = affected_mask.fillna(False)
            exposure_recs = []
            for label, col in op_cols.items():
                col_num = _num(df[col])
                exposed = float(col_num[affected_mask].sum())
                total = float(col_num.sum()) or 1.0
                exposure_recs.append({
                    "key": label,
                    "metrics": {
                        "exposed_subscribers": _round(exposed),
                        "total_subscribers": _round(total),
                        "exposure_pct": _round(exposed / total * 100),
                    },
                })
            exposure_recs.sort(key=lambda r: r["metrics"]["exposed_subscribers"], reverse=True)
            out["aggregations"]["operator_exposure"] = exposure_recs
            if exposure_recs and exposure_recs[0]["metrics"]["exposed_subscribers"] > 0:
                most_exposed = exposure_recs[0]["key"]

    # ── 4. KPIs (ordered most-valuable first; the strip caps at a few) ───────
    largest = mix_recs[0]["key"] if mix_recs else None
    largest_val = mix_recs[0]["metrics"]["subscribers"] if mix_recs else 0
    kpis = []
    # The single most operationally-valuable operator KPI: who's most exposed.
    if most_exposed:
        exp = out["aggregations"]["operator_exposure"][0]["metrics"]
        kpis.append({
            "label": "Most Congestion-Exposed",
            "value": f"{most_exposed} ({int(exp['exposed_subscribers']):,})",
            "color_hint": "red", "icon_hint": "⚠",
        })
    if largest:
        kpis.append({
            "label": "Largest Operator",
            "value": f"{largest} ({int(largest_val):,})",
            "color_hint": "blue", "icon_hint": "🏆",
        })
    if wholesale > 0:
        kpis.append({
            "label": "Wholesale Subscribers",
            "value": f"{int(wholesale):,}",
            "color_hint": "teal", "icon_hint": "🤝",
        })
    kpis.append({
        "label": "Operators Carried", "value": str(len(op_cols)),
        "color_hint": "blue", "icon_hint": "📡",
    })
    out["kpis"] = kpis

    # ── 5. Summary (for narrative hooks) ─────────────────────────────────────
    out["summary"] = {
        "n_operators": len(op_cols),
        "operators": list(op_cols.keys()),
        "largest_operator": largest,
        "largest_operator_subscribers": int(largest_val),
        "wholesale_subscribers": int(wholesale),
        "retail_subscribers": int(retail),
        "wholesale_pct": _round(wholesale / (wholesale + retail) * 100) if (wholesale + retail) else 0,
        "most_exposed_operator": most_exposed,
    }
    return out


def operator_story_fragment(summary: dict) -> str:
    """One-sentence narrative fragment for the Detective / Narrator / meta story."""
    if not summary or not summary.get("n_operators"):
        return ""
    parts = [f"carries {summary['n_operators']} operators"]
    if summary.get("largest_operator"):
        parts.append(
            f"led by {summary['largest_operator']} "
            f"({summary['largest_operator_subscribers']:,} subscribers)"
        )
    if summary.get("wholesale_pct"):
        parts.append(f"{summary['wholesale_pct']:.0f}% wholesale")
    if summary.get("most_exposed_operator"):
        parts.append(f"most congestion-exposed operator: {summary['most_exposed_operator']}")
    return "; ".join(parts) + "."
