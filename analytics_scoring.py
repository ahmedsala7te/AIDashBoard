"""analytics_scoring.py — Confidence, ranking, and quality scoring.

Implements the best ideas from the enterprise master-prompt as REAL computed
metrics (never hallucinated):

  • score_domain()            → domain detection with confidence % + runner-up
  • score_business()          → telecom sub-domain detection with confidence
  • classify_and_rank_kpis()  → KPI category + 0-100 importance, sorted
  • detect_correlations()     → strong numeric correlations for the findings panel
  • compute_dashboard_quality() → 0-100 quality score with a transparent breakdown

Every function is pure and deterministic so the scores are reproducible and the
"Prime Directive: truth over fluency" holds — these are measurements, not guesses.
"""
from __future__ import annotations

import re
import pandas as pd


# ─── Domain signals ───────────────────────────────────────────────────────────
DOMAIN_SIGNALS: dict[str, list[str]] = {
    "telecom": [
        "msan", "dslam", "olt", "ont", "bts", "nodeb", "enodeb", "gnodeb", "cell",
        "sector", "port", "subscriber", "congest", "critical", "utilization",
        "throughput", "bandwidth", "sinr", "rsrp", "rsrq", "latency", "trap",
        "outage", "exchange", "vlan", "interface", "hostname", "vendor", "bitstream",
    ],
    "banking": [
        "account", "loan", "deposit", "credit", "debit", "interest", "branch",
        "balance", "mortgage", "iban", "swift", "overdraft", "ledger",
    ],
    "finance": [
        "revenue", "profit", "cost", "margin", "ebitda", "cash", "asset",
        "liability", "invoice", "payment", "equity", "roi", "expense", "budget",
    ],
    "retail": [
        "sku", "product", "store", "sales", "inventory", "stock", "basket",
        "category", "discount", "pos", "promotion", "supplier",
    ],
    "hr": [
        "employee", "salary", "department", "hire", "attrition", "tenure",
        "manager", "leave", "payroll", "headcount", "recruit", "performance_review",
    ],
    "healthcare": [
        "patient", "diagnosis", "admission", "discharge", "treatment", "ward",
        "clinic", "icd", "provider", "prescription", "bed",
    ],
    "logistics": [
        "shipment", "delivery", "route", "warehouse", "carrier", "freight",
        "tracking", "eta", "fleet", "dispatch", "consignment",
    ],
    "sales": [
        "lead", "opportunity", "pipeline", "deal", "quota", "conversion", "close",
        "prospect", "win_rate",
    ],
    "cyber": [
        "threat", "vulnerability", "alert", "incident", "malware", "firewall",
        "cve", "attack", "severity", "intrusion", "phishing",
    ],
    "energy": [
        "meter", "consumption", "kwh", "grid", "load", "tariff", "generation",
        "voltage", "transformer",
    ],
}


def _column_blob_tokens(df: pd.DataFrame) -> list[str]:
    """Tokenised column names + a sample of categorical values, for signal matching."""
    toks: list[str] = []
    for c in df.columns:
        toks.append(str(c).lower())
    # sample a few categorical values (domain words often live in the data, not headers)
    try:
        for c in df.select_dtypes(include="object").columns[:8]:
            for v in df[c].dropna().astype(str).head(20):
                toks.append(v.lower())
    except Exception:
        pass
    return toks


def score_domain(df: pd.DataFrame) -> dict:
    """Score each domain by signal hits; return confidence % + ranked list.

    Confidence is the winning domain's SHARE of total signal hits — an honest
    relative measure ("of everything we recognised, 92% pointed at telecom").
    """
    blob = " ".join(_column_blob_tokens(df))
    raw: dict[str, int] = {}
    for domain, signals in DOMAIN_SIGNALS.items():
        hits = sum(1 for s in signals if s in blob)
        if hits:
            raw[domain] = hits
    if not raw:
        return {"domain": "general", "confidence": 0, "ranked": [], "rationale": "no domain signals matched"}
    total = sum(raw.values())
    ranked = sorted(raw.items(), key=lambda kv: kv[1], reverse=True)
    top_domain, top_hits = ranked[0]
    confidence = round(top_hits / total * 100)
    runner_up = ({"domain": ranked[1][0], "confidence": round(ranked[1][1] / total * 100)}
                 if len(ranked) > 1 else None)
    matched = [s for s in DOMAIN_SIGNALS[top_domain] if s in blob][:6]
    return {
        "domain": top_domain,
        "confidence": confidence,
        "runner_up": runner_up,
        "ranked": [{"domain": d, "confidence": round(h / total * 100)} for d, h in ranked[:4]],
        "rationale": f"matched signals: {', '.join(matched)}",
    }


# ─── Business sub-domain signals (telecom) ────────────────────────────────────
BUSINESS_SIGNALS: dict[str, list[str]] = {
    "congestion":   ["critical_time", "critical time", "chronic", "congest", "warning_time", "outage_duration"],
    "inventory":    ["vendor", "technology", "port", "olt", "ont", "interface", "capacity",
                     "free", "configured", "in_service", "rack", "shelf", "slot"],
    "alarms":       ["alarm", "cleared", "occurred", "raised", "alert", "trap"],
    "tickets":      ["ticket", "incident", "case", "complaint", "sla", "owner", "assignee", "resolved"],
    "performance":  ["throughput", "latency", "loss", "jitter", "bandwidth", "rsrp", "rsrq", "sinr", "cpu"],
}


def score_business(df: pd.DataFrame, domain: str) -> dict:
    """Telecom business sub-domain detection with confidence."""
    if domain != "telecom":
        return {"business": "general", "confidence": 0, "ranked": []}
    blob = " ".join(_column_blob_tokens(df))
    raw = {b: sum(1 for s in sig if s in blob) for b, sig in BUSINESS_SIGNALS.items()}
    raw = {b: h for b, h in raw.items() if h}
    if not raw:
        return {"business": "other_telecom", "confidence": 0, "ranked": []}
    total = sum(raw.values())
    ranked = sorted(raw.items(), key=lambda kv: kv[1], reverse=True)
    top, hits = ranked[0]
    return {
        "business": top,
        "confidence": round(hits / total * 100),
        "ranked": [{"business": b, "confidence": round(h / total * 100)} for b, h in ranked[:4]],
    }


# ─── KPI classification + ranking ─────────────────────────────────────────────
# category -> (keywords, base_weight 0-100). Higher weight = more boardroom-visible.
KPI_RULES: list[tuple[str, list[str], int]] = [
    ("Risk",        ["risk", "exposed", "anomaly", "chronic", "outage", "breach", "violation"], 95),
    ("Executive",   ["health", "total", "impacted", "overall", "score", "revenue", "availability"], 90),
    ("Customer",    ["subscriber", "operator", "customer", "churn", "user", "client", "wholesale"], 80),
    ("Operational", ["critical", "congest", "worst", "peak", "avg", "average", "pending",
                     "utilization", "throughput", "queue", "backlog"], 70),
    ("Financial",   ["revenue", "cost", "margin", "arpu", "profit", "expense", "budget"], 85),
    ("Quality",     ["error", "defect", "sla", "uptime", "success", "failure", "loss"], 75),
]


def classify_and_rank_kpis(kpis: list[dict]) -> list[dict]:
    """Attach a category + a 0-100 importance score to each KPI, then sort desc.

    Importance = category base weight, nudged up when the KPI is the network-wide
    total or names a worst/most-exposed entity (those draw the eye first).
    """
    scored = []
    for k in kpis:
        label = str(k.get("label", "")).lower()
        value = str(k.get("value", "")).lower()
        category, base = "Operational", 60
        for cat, kws, weight in KPI_RULES:
            if any(kw in label for kw in kws):
                category, base = cat, weight
                break
        # Nudges (deterministic, explainable)
        score = base
        if any(w in label for w in ("most", "worst", "largest", "peak", "top")):
            score += 5
        if any(w in label for w in ("total", "impacted", "monitored")):
            score += 3
        if k.get("color_hint") == "red":
            score += 4   # red = alarming = high attention
        score = max(0, min(100, score))
        kk = dict(k)
        kk["category"] = category
        kk["importance"] = score
        scored.append(kk)
    scored.sort(key=lambda x: x["importance"], reverse=True)
    return scored


# ─── Correlation detection ────────────────────────────────────────────────────
def detect_correlations(df: pd.DataFrame, max_pairs: int = 3,
                         min_abs_r: float = 0.6) -> list[dict]:
    """Find the strongest pairwise correlations among numeric columns.

    Skips near-duplicate pairs (|r|>0.98 usually means the same thing measured
    twice) and degenerate columns. Returns human-readable finding strings.
    """
    # Columns that produce spurious / uninteresting correlations: geographic
    # coordinates (lat≈lng is a map artifact) and identifiers/codes.
    _EXCLUDE = ("lat", "latitude", "lng", "long", "longitude", "lon",
                "id", "_id", "code", "index", "rank", "row")

    def _excluded(name: str) -> bool:
        nm = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
        return nm in _EXCLUDE or nm.endswith(("_id", "_code")) or nm in ("x", "y")

    # Coerce string-numeric columns (e.g. "400.00" with '-' placeholders) so we
    # don't miss correlations hidden in object-dtype columns.
    num = pd.DataFrame(index=df.index)
    for c in df.columns:
        if _excluded(c):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num[c] = df[c]
        else:
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().sum() >= 0.6 * max(df[c].notna().sum(), 1):
                num[c] = coerced
    # drop constant / near-empty columns
    keep = [c for c in num.columns if num[c].nunique(dropna=True) > 2 and num[c].notna().sum() > 5]
    if len(keep) < 2:
        return []
    try:
        corr = num[keep].corr(numeric_only=True)
    except Exception:
        return []
    seen, out = set(), []
    cols = list(corr.columns)
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.isna(r):
                continue
            ar = abs(r)
            if min_abs_r <= ar < 0.98:
                pairs.append((ar, r, cols[i], cols[j]))
    pairs.sort(reverse=True)
    for ar, r, a, b in pairs[:max_pairs]:
        direction = "rise together" if r > 0 else "move in opposite directions"
        out.append({
            "col_a": a, "col_b": b, "r": round(float(r), 2),
            "finding": f"{_pretty(a)} and {_pretty(b)} {direction} (r={r:+.2f}).",
        })
    return out


def _pretty(name) -> str:
    n = re.sub(r"\(.*?\)", " ", str(name)).replace("_", " ")
    return " ".join(n.split()).title()


# ─── Dashboard quality score ──────────────────────────────────────────────────
def compute_dashboard_quality(analysis: dict, design: dict) -> dict:
    """Score the finished dashboard 0-100 across transparent sub-dimensions.

    This is a real measurement of the artifact we produced — coverage of the
    data, completeness of the story, and visual balance — not a vanity number.
    """
    meta = analysis.get("meta", {}) or {}
    aggs = analysis.get("aggregations", {}) or {}
    charts = design.get("charts", []) or []
    kpis = analysis.get("kpis", []) or []
    insights = analysis.get("insights", []) or []
    columns = analysis.get("columns", []) or []

    # 1. Data coverage — how much of the meaningful data is represented.
    meaningful = [c for c in columns if c.get("semantic_role") in ("metric", "dimension", "id")]
    used_sources = {c.get("data_source", "").split(".")[0] for c in charts}
    coverage = min(100, round((len(aggs) and len(used_sources) / max(len(aggs), 1)) * 100)) if aggs else 0
    # blend with column richness so tiny datasets aren't over-rewarded
    col_factor = min(1.0, len(meaningful) / 6)
    data_coverage = round(coverage * 0.6 + col_factor * 40)

    # 2. Completeness — KPIs + insights + charts + narrative present.
    completeness = 0
    completeness += min(30, len(kpis) * 6)          # up to 5 KPIs
    completeness += min(25, len(insights) * 6)      # up to ~4 insights
    completeness += min(30, len(charts) * 6)        # up to 5 charts
    completeness += 15 if meta.get("story") else 0
    completeness = min(100, completeness)

    # 3. Visual variety — distinct chart types (ideal 3-5).
    types = {c.get("chart_type") for c in charts}
    variety = min(100, round(len(types) / 5 * 100))

    # 4. Readability — chart count in the 4-8 sweet spot; penalise extremes.
    n = len(charts)
    if 4 <= n <= 8:
        readability = 100
    elif n in (3, 9, 10):
        readability = 75
    elif n in (2, 11, 12):
        readability = 55
    else:
        readability = 35 if n else 0

    # 5. Relevance — domain-appropriate signal: telecom should have severity/operator charts.
    relevance = 70
    if meta.get("domain") == "telecom":
        has_sev = any(c.get("color_scheme") == "severity" for c in charts)
        has_ops = any("operator" in c.get("id", "") for c in charts)
        has_gauge = any(c.get("chart_type") == "gauge" for c in charts)
        relevance = 60 + (15 if has_sev else 0) + (15 if has_ops else 0) + (10 if has_gauge else 0)
    relevance = min(100, relevance)

    weights = {
        "data_coverage": 0.25, "completeness": 0.25, "visual_variety": 0.15,
        "readability": 0.15, "relevance": 0.20,
    }
    parts = {
        "data_coverage": data_coverage, "completeness": completeness,
        "visual_variety": variety, "readability": readability, "relevance": relevance,
    }
    overall = round(sum(parts[k] * w for k, w in weights.items()))

    grade = ("A" if overall >= 90 else "B" if overall >= 80 else
             "C" if overall >= 70 else "D" if overall >= 55 else "E")
    return {"overall": overall, "grade": grade, "breakdown": parts}
