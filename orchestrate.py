"""
orchestrate.py — Multi-Agent Pipeline Runner
Runs 5 AI agents in sequence/parallel to analyze any tabular data file
and generate a complete Dash dashboard.
"""
import sys
import os
import json
import re
import time
import subprocess
import threading
import traceback
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import numpy as np
import chardet

try:
    from langchain_ollama import OllamaLLM
except ImportError:
    from langchain_community.llms import Ollama as OllamaLLM

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "exports").mkdir(exist_ok=True)

STATUS_FILE = OUTPUT_DIR / "pipeline_status.json"
ANALYSIS_FILE = OUTPUT_DIR / "analysis.json"
DESIGN_FILE = OUTPUT_DIR / "design.json"
FIGURES_FILE = OUTPUT_DIR / "figures_code.py"
INSIGHTS_FILE = OUTPUT_DIR / "insights.json"
# app.py (ROOT) is the deterministic renderer we launch — never overwritten.
# Agent 4's generated app is saved here for inspection in the Agent Monitor.
APP_FILE = ROOT / "app.py"
GENERATED_APP_FILE = OUTPUT_DIR / "app_generated.py"
PIPELINE_LOG = OUTPUT_DIR / "pipeline.log"

# ─── Settings ─────────────────────────────────────────────────────────────────
def load_settings():
    settings_path = ROOT / "settings.json"
    if settings_path.exists():
        with open(settings_path) as f:
            return json.load(f)
    return {
        "ollama": {"host": "http://localhost:11434", "default_model": "qwen2.5-coder:32b"},
        "pipeline": {"max_retries": 1}
    }

SETTINGS = load_settings()
OLLAMA_HOST = SETTINGS.get("ollama", {}).get("host", "http://localhost:11434")
DEFAULT_MODEL = SETTINGS.get("ollama", {}).get("default_model", "qwen2.5-coder:32b")
MAX_RETRIES = SETTINGS.get("pipeline", {}).get("max_retries", 1)

# ─── Speed mode ───────────────────────────────────────────────────────────────
# Agents 3 & 4 produce code that the served dashboard (app.py) NEVER uses —
# `app.py` is the deterministic renderer; figures_code.py / app_generated.py
# are only saved for inspection. Skipping their LLM calls saves ~7 minutes/run.
# Override via env: PIPELINE_FAST=0 to re-enable creative code generation.
FAST_MODE = os.environ.get("PIPELINE_FAST", "1") not in ("0", "false", "False", "no")

# ─── Status Management ────────────────────────────────────────────────────────
PIPELINE_STATUS = {
    "agent_1": {"status": "idle", "start": None, "end": None, "duration_sec": None},
    "agent_2": {"status": "idle", "start": None, "end": None, "duration_sec": None},
    "agent_3": {"status": "idle", "start": None, "end": None, "duration_sec": None},
    "agent_4": {"status": "idle", "start": None, "end": None, "duration_sec": None},
    "agent_5": {"status": "idle", "start": None, "end": None, "duration_sec": None},
}
_status_lock = threading.Lock()

def write_status():
    with _status_lock:
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(PIPELINE_STATUS, f, indent=2, ensure_ascii=False)

def set_agent_status(agent_key: str, status: str, start=None, end=None, duration=None):
    with _status_lock:
        PIPELINE_STATUS[agent_key]["status"] = status
        if start:
            PIPELINE_STATUS[agent_key]["start"] = start
        if end:
            PIPELINE_STATUS[agent_key]["end"] = end
        if duration is not None:
            PIPELINE_STATUS[agent_key]["duration_sec"] = round(duration, 2)
    write_status()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {agent_key} → {status}")

def agent_timer(agent_key: str):
    """Context manager-style helper — call start/end manually."""
    start_iso = datetime.now(timezone.utc).isoformat()
    start_mono = time.monotonic()
    set_agent_status(agent_key, "running", start=start_iso)
    return start_iso, start_mono

def agent_done(agent_key: str, start_mono: float):
    end_iso = datetime.now(timezone.utc).isoformat()
    duration = time.monotonic() - start_mono
    set_agent_status(agent_key, "done", end=end_iso, duration=duration)

def agent_failed(agent_key: str, start_mono: float):
    end_iso = datetime.now(timezone.utc).isoformat()
    duration = time.monotonic() - start_mono
    set_agent_status(agent_key, "failed", end=end_iso, duration=duration)

# ─── File Loading ─────────────────────────────────────────────────────────────
def detect_encoding(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read(100_000)
    result = chardet.detect(raw)
    return result.get("encoding") or "utf-8"

def load_file(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    elif ext == ".csv":
        enc = detect_encoding(path)
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            return pd.read_csv(path, encoding="utf-8", errors="replace")
    elif ext == ".json":
        return pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def load_files(paths: list) -> dict:
    """Load multiple files; return dict keyed by filename."""
    result = {}
    for p in paths:
        name = Path(p).name
        try:
            result[name] = load_file(p)
            print(f"  Loaded {name}: {result[name].shape[0]} rows × {result[name].shape[1]} cols")
        except Exception as e:
            print(f"  ERROR loading {name}: {e}")
    return result

def merge_dataframes(dfs: dict) -> pd.DataFrame:
    """Merge multiple DataFrames; if only one, return it directly."""
    frames = list(dfs.values())
    if len(frames) == 1:
        return frames[0]
    try:
        return pd.concat(frames, ignore_index=True)
    except Exception:
        return frames[0]

# ─── Data Serialisation for LLM Prompt ───────────────────────────────────────
def df_to_prompt_summary(df: pd.DataFrame, max_rows: int = 10) -> str:
    """Produce a compact text representation of the DataFrame for the prompt."""
    lines = []
    lines.append(f"SHAPE: {df.shape[0]} rows × {df.shape[1]} columns")
    lines.append("")
    lines.append("COLUMNS:")
    for col in df.columns:
        dtype = str(df[col].dtype)
        nulls = df[col].isna().mean()
        sample = df[col].dropna().head(5).tolist()
        lines.append(f"  {col!r:40s} dtype={dtype:12s} null%={nulls:.1%}  sample={sample}")
    lines.append("")
    lines.append(f"FIRST {min(max_rows, len(df))} ROWS (JSON):")
    preview = df.head(max_rows).copy()
    for col in preview.columns:
        if preview[col].dtype == "object":
            preview[col] = preview[col].astype(str)
    lines.append(preview.to_json(orient="records", force_ascii=False, date_format="iso"))
    lines.append("")
    lines.append("NUMERIC STATS:")
    try:
        num = df.select_dtypes(include="number")
        if not num.empty:
            stats = num.describe().round(2).to_string()
            lines.append(stats[:4000])
    except Exception:
        pass
    lines.append("")
    lines.append("CATEGORICAL VALUE COUNTS (top 5 each):")
    try:
        for col in df.select_dtypes(exclude="number").columns[:12]:
            vc = df[col].astype(str).value_counts().head(5).to_dict()
            lines.append(f"  {col!r}: {vc}")
    except Exception:
        pass
    return "\n".join(lines)

# ─── LLM Helpers ──────────────────────────────────────────────────────────────
def make_llm(model: str, temperature: float, num_ctx: int) -> OllamaLLM:
    return OllamaLLM(
        base_url=OLLAMA_HOST,
        model=model,
        temperature=temperature,
        num_ctx=num_ctx,
    )

def extract_json(text: str) -> str:
    """Extract first {...} block from LLM output."""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text.strip()


def parse_json_lenient(text: str) -> dict:
    """Parse LLM JSON output, repairing the common mistakes small models make.

    Handles: markdown fences, invalid backslash escapes (e.g. Windows paths,
    stray '\\' inside strings — the exact 'Invalid \\escape' error), trailing
    commas, smart quotes, and unescaped control characters.
    """
    s = extract_json(text)

    # Attempt 1: as-is
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Normalise smart quotes that some models emit
    s2 = (s.replace("“", '"').replace("”", '"')
            .replace("‘", "'").replace("’", "'"))

    # Repair invalid backslash escapes: a backslash NOT followed by a valid JSON
    # escape char becomes a literal (doubled) backslash. This fixes the
    # "Invalid \escape" failure on values like "MSAN\CAIRO" or "8\10".
    s2 = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s2)

    # Remove trailing commas before a closing brace/bracket
    s2 = re.sub(r",(\s*[}\]])", r"\1", s2)

    try:
        return json.loads(s2)
    except json.JSONDecodeError:
        pass

    # Last resort: escape raw control characters inside the string
    s3 = s2.replace("\t", "\\t").replace("\r", "\\r")
    s3 = re.sub(r"(?<!\\)\n", "\\\\n", s3)
    return json.loads(s3)  # if this still fails, the retry layer catches it


# ─── Deterministic fallbacks (used if the LLM cannot produce valid output) ─────
TELECOM_KEYWORDS = [
    "msan", "dslam", "olt", "ont", "bts", "nodeb", "enodeb", "gnodeb", "bsc", "rnc",
    "msc", "cell", "site", "sector", "port", "exchange", "trunk", "subscriber",
    "utiliz", "congest", "throughput", "availab", "alarm", "fault", "outage", "sla",
    "latency", "packet", "rsrp", "rsrq", "sinr",
]


IMPACT_HINT = ("subscriber", "customer", "impact", "affected", "user", "client", "line")
SEVERITY_HINT = ("critical", "util", "congest", "load", "fault", "down", "outage")
STATUS_HINT = ("status", "state", "severity", "alarm", "upgrade", "reason", "type", "priority", "category")
DIM_HINT = ("region", "area", "governorate", "sector", "exchange", "zone", "city",
            "district", "vendor", "technology", "cluster", "site_type")
_NOISE_WORDS = ("archive", "average", "avg", "yesterday", "minus1d", "minus2d",
                "minus", "1d", "2d", "min", "report")


def _clean(s) -> str:
    return re.sub(r"\W+", "_", str(s).strip().lower()).strip("_") or "field"


def _round2(v):
    try:
        f = float(v)
        return int(f) if f == int(f) else round(f, 2)
    except (TypeError, ValueError):
        return v


def _metric_label(name) -> str:
    """Short human label for a metric column.

    Strips report noise words, parenthetical units like '(min)' / '(%)',
    and trailing/leading whitespace so we get clean axis titles.
    """
    n = str(name).replace("_", " ")
    # Remove parenthetical suffixes — e.g. "(min)", "(%)", "(count)"
    n = re.sub(r"\(.*?\)", " ", n)
    for w in _NOISE_WORDS:
        n = re.sub(rf"\b{w}\b", "", n, flags=re.IGNORECASE)
    n = " ".join(n.split())
    return n.title() or "Value"


def _pretty(name) -> str:
    return str(name).replace("_", " ").strip().title()


def _is_arabic_series(s) -> bool:
    try:
        return bool(s.astype(str).str.contains(r"[؀-ۿ]", regex=True).any())
    except Exception:
        return False


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with object columns coerced to numeric where ≥70% of
    non-null values parse as numbers.  Fixes Excel files where a single stray
    text cell makes pandas read the whole column as object dtype."""
    df = df.copy()
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            coerced = pd.to_numeric(df[c], errors="coerce")
            n_valid = coerced.notna().sum()
            n_total = df[c].notna().sum()
            if n_total > 0 and n_valid / n_total >= 0.70:
                df[c] = coerced          # replace with properly-typed numeric series
    return df


def _classify_columns(df: pd.DataFrame) -> dict:
    """Assign each column a role: constant/identifier/dimension/metric/datelike/text.

    Expects df to have already been through _coerce_numeric_columns so that
    columns like 'average_critical_time(min)' that Excel stored as mixed
    int/str are already float64 by the time we classify them.
    """
    n = len(df)
    roles = {}
    for c in df.columns:
        s = df[c]
        name = str(c).lower()
        if s.nunique(dropna=True) <= 1:
            roles[c] = "constant"
            continue
        if pd.api.types.is_numeric_dtype(s):
            v = s.dropna()
            if "date" in name and not v.empty and v.between(20000, 60000).mean() > 0.7:
                roles[c] = "datelike"
            else:
                roles[c] = "metric"
            continue
        if pd.api.types.is_datetime64_any_dtype(s):
            roles[c] = "datetime"
            continue
        nun = s.nunique()
        if nun >= max(n - 1, int(0.9 * n)):
            roles[c] = "identifier"
        elif nun <= 50:
            roles[c] = "dimension"
        else:
            roles[c] = "text"
    return roles


def _pick_named(cols, keywords):
    for kw in keywords:
        for c in cols:
            if kw in str(c).lower():
                return c
    return cols[0] if cols else None


def _recency_rank(c) -> float:
    n = str(c).lower()
    if "minus2" in n or "2d" in n:
        return 0
    if "minus1" in n or "1d" in n:
        return 1
    if "yesterday" in n or "today" in n:
        return 3
    return 2


def _heuristic_roles(df: pd.DataFrame) -> dict:
    """Detect column roles from names/types — used as the fallback for the AI picks."""
    df = _coerce_numeric_columns(df)   # ensure mixed-type columns are numeric before classifying
    roles = _classify_columns(df)
    ids = [c for c, r in roles.items() if r == "identifier"]
    dims = [c for c, r in roles.items() if r == "dimension"]
    metrics = [c for c, r in roles.items() if r == "metric"]
    # Hostnames are ALWAYS preferred over codes as the entity label — operations
    # staff recognise nodes by hostname (SHKMA147-M01H-C-EG), not by code (02-1-08).
    # Look across identifiers AND dimensions for any hostname-like column first.
    hostname_candidates = [c for c in df.columns
                           if any(kw in str(c).lower()
                                  for kw in ("hostname", "host_name", "node_name"))]
    _entity_kws = ["hostname", "host", "name", "code", "id", "msan", "node", "site", "cell"]
    entity = (
        (hostname_candidates[0] if hostname_candidates else None) or
        _pick_named(ids, _entity_kws) or
        _pick_named(dims, ["hostname", "host", "msan", "node", "site"]) or
        (ids[0] if ids else None)
    )
    impact = _pick_named([m for m in metrics if any(h in str(m).lower() for h in IMPACT_HINT)], IMPACT_HINT)
    crit_cols = [m for m in metrics if "critical" in str(m).lower()]
    sev_cols = [m for m in metrics if any(h in str(m).lower() for h in SEVERITY_HINT)]
    if crit_cols:
        primary_sev = sorted(crit_cols, key=_recency_rank, reverse=True)[0]
    elif sev_cols:
        primary_sev = sev_cols[0]
    elif impact:
        primary_sev = impact
    else:
        primary_sev = metrics[0] if metrics else None
    # Only consider dims that have 2+ distinct meaningful values as status candidates
    # (skip near-constant columns like reason="msan" where every row is identical)
    useful_dims = [d for d in dims if df[d].nunique() >= 2]
    status = _pick_named([d for d in useful_dims if any(h in str(d).lower() for h in STATUS_HINT)], STATUS_HINT)
    if not status and useful_dims:
        status = min(useful_dims, key=lambda c: df[c].nunique())
    return {"entity": entity, "impact": impact, "primary_sev": primary_sev,
            "crit_cols": crit_cols, "dims": dims, "status_dim": status, "metrics": metrics}


def _resolve_roles(df: pd.DataFrame, ai_roles: dict) -> dict:
    """Merge AI-chosen roles with heuristics. AI picks are validated against real columns.

    NOTE: When BOTH 'hostname'-like and 'code'-like columns exist for the same entity
    (e.g. msan_hostname='SHKMA147-M01H-C-EG' alongside msan_code='02-1-08-147'),
    we always prefer the hostname — operations staff recognise nodes by hostname,
    not by code, so it makes much better chart labels.
    """
    h = _heuristic_roles(df)
    if not isinstance(ai_roles, dict):
        return h
    cols = set(df.columns)

    def v(x):
        return x if x in cols else None

    def vlist(xs):
        return [x for x in (xs or []) if x in cols]

    # Force hostname-preference: if a hostname column exists, use it as entity
    # regardless of what the LLM (or heuristic) picked.
    hostname_col = next(
        (c for c in df.columns
         if any(kw in str(c).lower() for kw in ("hostname", "host_name", "node_name"))),
        None,
    )
    entity_pick = hostname_col or v(ai_roles.get("entity_column")) or h["entity"]

    return {
        "entity": entity_pick,
        "impact": v(ai_roles.get("impact_metric")) or h["impact"],
        "primary_sev": v(ai_roles.get("severity_metric")) or h["primary_sev"],
        "crit_cols": h["crit_cols"],  # keep heuristic set for the 3-day trend
        "dims": vlist(ai_roles.get("dimension_columns")) or h["dims"],
        "status_dim": v(ai_roles.get("status_column")) or h["status_dim"],
        "metrics": h["metrics"],
    }


def compute_analysis(df: pd.DataFrame, roles: dict = None) -> dict:
    """Compute a correct, telecom-aware analysis from the dataframe.

    All numbers are computed in Python. `roles` (optionally chosen by the AI) decides
    WHICH columns are entity/impact/severity/dimensions; the maths is deterministic.
    """
    # ── Step 0: Coerce mixed-type columns to numeric ──────────────────────────
    # Excel files often give us object-dtype columns that are actually numbers
    # (one stray text cell makes pandas read the whole column as object).
    # This must happen BEFORE classify/roles so metrics are found correctly.
    df = _coerce_numeric_columns(df)
    n = len(df)
    cols_blob = " ".join(str(c).lower() for c in df.columns)
    domain = "telecom" if any(k in cols_blob for k in TELECOM_KEYWORDS) else "other"
    if roles is None:
        roles = _heuristic_roles(df)

    entity = roles.get("entity")
    impact = roles.get("impact")
    primary_sev = roles.get("primary_sev")
    crit_cols = roles.get("crit_cols") or []
    dims = roles.get("dims") or []
    status_dim = roles.get("status_dim")

    is_time = primary_sev is not None and any(t in str(primary_sev).lower() for t in ("time", "min", "duration"))
    arabic_found = any(_is_arabic_series(df[c]) for c in df.columns if df[c].dtype == object)
    col_roles = _classify_columns(df)

    # ── Column descriptors ──
    columns = []
    for c in df.columns:
        r = col_roles[c]
        dtype = {"metric": "numeric", "datelike": "numeric", "datetime": "datetime",
                 "identifier": "identifier"}.get(r, "categorical")
        role_map = {"identifier": "id", "dimension": "dimension", "metric": "metric",
                    "datelike": "timestamp", "datetime": "timestamp", "constant": "label", "text": "label"}
        s = df[c]
        stats = {}
        if r == "metric" and not s.dropna().empty:
            stats = {"min": _round2(s.min()), "max": _round2(s.max()), "mean": _round2(s.mean())}
        columns.append({
            "original_name": str(c), "clean_name": _clean(c), "data_type": dtype,
            "semantic_role": role_map.get(r, "label"),
            "importance": "high" if c in (entity, impact, primary_sev) else ("medium" if r in ("metric", "dimension") else "low"),
            "has_arabic": _is_arabic_series(s) if s.dtype == object else False,
            "nullable_pct": _round2(s.isna().mean() * 100),
            "sample_values": [str(x) for x in s.dropna().head(5).tolist()],
            "stats": stats,
        })

    aggregations = {}
    skey = _clean(primary_sev) if primary_sev else None
    ikey = _clean(impact) if impact else None

    # ── Top offenders ──
    if entity and primary_sev:
        keep = [entity, primary_sev] + ([impact] if impact else [])
        sub = df[keep].dropna(subset=[primary_sev]).sort_values(primary_sev, ascending=False).head(10)
        recs = []
        for _, row in sub.iterrows():
            m = {skey: _round2(row[primary_sev])}
            if impact:
                m[ikey] = _round2(row[impact])
            recs.append({"key": str(row[entity]), "metrics": m})
        aggregations["top_offenders"] = recs

    # ── By dimension ──
    for d in sorted(dims, key=lambda c: (0 if any(h in str(c).lower() for h in ("region", "area", "governorate")) else
                                         (1 if "sector" in str(c).lower() else 2)))[:3]:
        recs = []
        for key, grp in df.groupby(d):
            m = {"count": int(len(grp))}
            if impact:
                m[ikey] = _round2(grp[impact].sum())
            if primary_sev:
                m[skey] = _round2(grp[primary_sev].mean())
            recs.append({"key": str(key), "metrics": m})
        sort_field = ikey if impact else "count"
        recs.sort(key=lambda rr: rr["metrics"].get(sort_field, 0), reverse=True)
        aggregations[f"by_{_clean(d)}"] = recs[:20]

    # ── Status distribution ──
    if status_dim:
        vc = df[status_dim].astype(str).value_counts().head(12)
        aggregations["distributions"] = {str(status_dim): {str(k): int(v) for k, v in vc.items()}}

    # ── 3-day critical-time trend ──
    if len(crit_cols) >= 2:
        ordered = sorted(crit_cols, key=_recency_rank)
        labels_map = {2: ["Earlier", "Latest"], 3: ["Day -2", "Day -1", "Latest"]}
        labs = labels_map.get(len(ordered), [f"P{i+1}" for i in range(len(ordered))])
        aggregations["time_series"] = [
            {"period": lab, "metrics": {"avg_critical_time": _round2(df[c].mean())}}
            for c, lab in zip(ordered, labs)
        ]

    # ── Network health gauge (time-based severity only) ──
    gauge_val = None
    if primary_sev and is_time:
        crit_share = min(100.0, float(df[primary_sev].mean()) / 1440.0 * 100)
        gauge_val = round(100 - crit_share, 1)
        aggregations["network_health"] = [{"key": "health", "metrics": {"health_pct": gauge_val}}]

    # ── KPIs ──
    kpis = [{"label": "MSANs Monitored" if domain == "telecom" else "Total Records",
             "value": f"{n:,}", "color_hint": "blue", "icon_hint": "🗼" if domain == "telecom" else "📊"}]
    worst_name = worst_val = worst_impact = None
    if entity and primary_sev:
        wi = df[primary_sev].idxmax()
        worst_name, worst_val = str(df.loc[wi, entity]), _round2(df[primary_sev].max())
        if impact:
            worst_impact = int(df.loc[wi, impact])
    if impact:
        kpis.append({"label": "Impacted Subscribers", "value": f"{int(df[impact].sum()):,}",
                     "color_hint": "red", "icon_hint": "⚠"})
    if worst_name:
        kpis.append({"label": "Worst Node", "value": worst_name, "color_hint": "red", "icon_hint": "🔴"})
    if primary_sev:
        unit = " min" if is_time else ""
        kpis.append({"label": f"Avg {_metric_label(primary_sev)}",
                     "value": f"{_round2(df[primary_sev].mean()):,}{unit}", "color_hint": "orange", "icon_hint": "📈"})
        kpis.append({"label": f"Peak {_metric_label(primary_sev)}",
                     "value": f"{_round2(df[primary_sev].max()):,}{unit}", "color_hint": "orange", "icon_hint": "🔺"})
    if gauge_val is not None:
        kpis.append({"label": "Network Health", "value": f"{gauge_val}%",
                     "color_hint": "green" if gauge_val >= 70 else ("orange" if gauge_val >= 40 else "red"),
                     "icon_hint": "📊"})
    if status_dim:
        pend = int(df[status_dim].astype(str).str.contains("new|pend|progress|open", case=False, na=False).sum())
        if pend:
            kpis.append({"label": "Pending Upgrades", "value": str(pend), "color_hint": "blue", "icon_hint": "🛠"})

    # ── Anomalies ──
    anomalies = []
    if primary_sev:
        m, sd = df[primary_sev].mean(), (df[primary_sev].std() or 0)
        for _, r in df[df[primary_sev] > m + 2 * sd].iterrows():
            anomalies.append(f"{r[entity] if entity else 'Row'} has {_metric_label(primary_sev)} {_round2(r[primary_sev])} (>2σ above mean)")
    if entity and df[entity].duplicated().any():
        anomalies.append("Duplicate node entries detected.")

    # ── Urgent flag ──
    urgent = {"exists": False, "message": None, "severity": None}
    if worst_name:
        sev_level = "CRITICAL" if (gauge_val is not None and gauge_val < 40) or anomalies else \
                    ("HIGH" if gauge_val is not None and gauge_val < 60 else "MEDIUM")
        imp_txt = f" impacting {worst_impact:,} subscribers" if worst_impact else ""
        urgent = {"exists": sev_level in ("CRITICAL", "HIGH"), "severity": sev_level,
                  "message": f"{worst_name} shows the highest {_metric_label(primary_sev)} ({worst_val}){imp_txt} — prioritise for escalation."}

    # ── Story + insights ──
    region_key = next((k for k in aggregations if k.startswith("by_") and any(h in k for h in ("region", "area"))), None)
    hotspot = aggregations[region_key][0]["key"] if region_key and aggregations.get(region_key) else None

    if domain == "telecom":
        story = (f"{n} chronic-critical MSANs analysed"
                 + (f", impacting {int(df[impact].sum()):,} subscribers" if impact else "")
                 + (f"; worst node {worst_name} ({_metric_label(primary_sev)} {worst_val})" if worst_name else "")
                 + (f"; hotspot region: {hotspot}." if hotspot else "."))
    else:
        story = f"{n} records across {len(df.columns)} columns analysed."

    insights = []
    if worst_name:
        insights.append(f"Worst node is {worst_name} with {_metric_label(primary_sev)} of {worst_val}"
                        + (f", impacting {worst_impact:,} subscribers." if worst_impact else "."))
    if impact:
        insights.append(f"Total impacted subscribers across all nodes: {int(df[impact].sum()):,}.")
    if hotspot:
        insights.append(f"{hotspot} is the highest-impact region and should be prioritised.")
    if "time_series" in aggregations and len(aggregations["time_series"]) >= 2:
        first = aggregations["time_series"][0]["metrics"]["avg_critical_time"]
        last = aggregations["time_series"][-1]["metrics"]["avg_critical_time"]
        trend = "worsening" if last > first else ("improving" if last < first else "stable")
        insights.append(f"Average critical time is {trend} ({first} → {last} min over the period).")
    if gauge_val is not None:
        insights.append(f"Network health score is {gauge_val}% (100 − average daily critical-time share).")
    while len(insights) < 4:
        insights.append(f"{n} elements monitored across {len(dims)} operational dimensions.")

    return {
        "meta": {
            "domain": domain, "grain": ("one MSAN network element" if domain == "telecom" else "one record"),
            "row_count": n, "column_count": len(df.columns),
            "languages_detected": (["english", "arabic"] if arabic_found else ["english"]),
            "story": story, "anomalies": anomalies[:6],
            "schema_design_rationale": "Hybrid: AI-selected column roles, Python-computed aggregations.",
        },
        "columns": columns, "aggregations": aggregations, "kpis": kpis[:6],
        "insights": insights[:5], "urgent_flag": urgent,
    }


def _first_metric_key(recs, prefer=None):
    if not recs:
        return "value"
    mk = list((recs[0].get("metrics") or {}).keys())
    if prefer:
        for k in mk:
            if prefer in k:
                return k
    return mk[0] if mk else "value"


def build_design(analysis: dict) -> dict:
    """Deterministically design a NOC-grade dashboard from the computed analysis."""
    aggs = analysis.get("aggregations", {}) or {}
    domain = (analysis.get("meta", {}) or {}).get("domain", "data")
    sev = "severity" if domain == "telecom" else "categorical"
    charts = []

    has_gauge = isinstance(aggs.get("network_health"), list) and aggs["network_health"]
    if has_gauge:
        charts.append({
            "id": "network_health", "title": "Network Health Score", "chart_type": "gauge",
            "tab": 0, "priority": 1, "width_cols": 4, "data_source": "network_health",
            "x_field": "key", "y_field": "health_pct", "color_scheme": "severity",
            "invert_gauge": True,  # high = good: green zone at top, red at bottom
            "has_threshold_line": True, "threshold_value": 60, "threshold_label": "Healthy ≥60%",
            "x_title": "", "y_title": "", "insight": "Higher is healthier (100 − avg daily critical-time share).",
        })

    if isinstance(aggs.get("top_offenders"), list) and aggs["top_offenders"]:
        recs = aggs["top_offenders"]
        yf = _first_metric_key(recs, "critical")
        if yf == "value":
            yf = _first_metric_key(recs)
        mkeys = list((recs[0].get("metrics") or {}).keys())
        sf = next((k for k in mkeys if any(h in k for h in ("subscriber", "impact", "affected"))), None)
        charts.append({
            "id": "top_offenders", "title": "Worst Critical MSANs (Top 10)", "chart_type": "horizontal_bar",
            "tab": 0, "priority": 1, "width_cols": 8 if has_gauge else 12, "data_source": "top_offenders",
            "x_field": "key", "y_field": yf, "color_scheme": sev, "sort_order": "desc",
            "highlight_top_n": 3, "secondary_annotation_field": sf,
            "x_title": _metric_label(yf), "y_title": "MSAN",
            "insight": "Teal ⊕ marks impacted subscribers per node." if sf else "",
        })

    by_keys = sorted([k for k in aggs if k.startswith("by_")],
                     key=lambda k: (0 if "region" in k or "area" in k else (1 if "sector" in k else 2)))
    for k in by_keys[:2]:
        recs = aggs[k]
        yf = _first_metric_key(recs, "subscriber")
        if yf == "value":
            yf = _first_metric_key(recs, "count") or _first_metric_key(recs)
        charts.append({
            "id": k, "title": k.replace("by_", "By ").replace("_", " ").title(),
            "chart_type": "horizontal_bar", "tab": 0, "priority": 2, "width_cols": 6,
            "data_source": k, "x_field": "key", "y_field": yf, "color_scheme": sev,
            "sort_order": "desc", "top_n": 10,           # cap at 10 — bottom bars become unreadable beyond that
            "highlight_top_n": 3, "x_title": _metric_label(yf), "y_title": "",
            "insight": "",
        })

    # Donut: only show if the distribution has real diversity.
    # A column where one value is >85% of the total ("reason" = "msan" 92.5%) is noise, not insight.
    dist = aggs.get("distributions")
    if isinstance(dist, dict) and dist:
        col = list(dist.keys())[0]
        counts = list(dist[col].values()) if isinstance(dist[col], dict) else []
        total = sum(counts) or 1
        top_share = (max(counts) / total) if counts else 1.0
        if top_share <= 0.85 and len(counts) >= 2:
            charts.append({
                "id": "status_dist", "title": f"{_pretty(col)} Breakdown", "chart_type": "donut",
                "tab": 0, "priority": 3, "width_cols": 6, "data_source": f"distributions.{col}",
                "x_field": "label", "y_field": "value", "color_scheme": sev,
                "x_title": "", "y_title": "", "insight": "",
            })

    if isinstance(aggs.get("time_series"), list) and len(aggs["time_series"]) >= 2:
        yf = _first_metric_key(aggs["time_series"])
        charts.append({
            "id": "time_series", "title": "Critical-Time Trend (3-Day)", "chart_type": "line",
            "tab": 0, "priority": 3, "width_cols": 6, "data_source": "time_series",
            "x_field": "period", "y_field": yf, "show_average_line": True,
            "x_title": "Reporting Day", "y_title": _metric_label(yf),
            "insight": "Rising line = worsening chronic condition.",
        })

    if not charts:
        charts.append({
            "id": "overview", "title": "Data Overview", "chart_type": "table", "tab": 0,
            "priority": 1, "width_cols": 12, "data_source": "top_offenders", "x_field": "key",
            "y_field": "value", "x_title": "", "y_title": "", "insight": "",
        })

    title = "Access Network Operations — MSAN Health" if domain == "telecom" else f"{domain.title()} Analytics Dashboard"
    return {"dashboard_title": title, "layout_hint": "single_tab",
            "tab_names": ["NOC Overview" if domain == "telecom" else "Overview"], "charts": charts}


def compute_insights(analysis: dict) -> dict:
    """Rule-based, telecom-aware executive summary from the computed analysis."""
    meta = analysis.get("meta", {}) or {}
    domain = meta.get("domain", "data")
    aggs = analysis.get("aggregations", {}) or {}
    kpis = analysis.get("kpis", []) or []
    urgent = analysis.get("urgent_flag", {}) or {}

    highlights = [f"• {k.get('label')}: {k.get('value')}" for k in kpis[:4]] or \
                 [f"• {meta.get('row_count', 0):,} records analysed"]

    risks, actions = [], []
    top = aggs.get("top_offenders") or []
    if domain == "telecom" and top:
        worst = top[0]
        wk = worst["key"]
        m = worst.get("metrics", {})
        sub = next((v for kk, v in m.items() if "subscriber" in kk or "impact" in kk), None)
        sevv = next((v for kk, v in m.items() if "critical" in kk or "time" in kk), None)
        risks.append(f"⚠ {wk} is the worst node"
                     + (f" (critical time {sevv} min)" if sevv is not None else "")
                     + (f", impacting {int(sub):,} subscribers." if sub else "."))
        actions.append(f"→ Escalate {wk} to field operations for an immediate site visit.")
        if len(top) >= 3:
            names = ", ".join(t["key"] for t in top[:3])
            actions.append(f"→ Raise capacity-augmentation work orders for the top chronic nodes: {names}.")

    region_key = next((k for k in aggs if k.startswith("by_") and ("region" in k or "area" in k)), None)
    if region_key and aggs.get(region_key):
        r0 = aggs[region_key][0]["key"]
        risks.append(f"⚠ {r0} is the highest-impact region — likely a congestion/fault cluster.")
        actions.append(f"→ Assign a regional task force to {r0}.")

    if not risks:
        risks = ["No severe anomalies detected in this dataset."]
    if not actions:
        actions = ["→ Review the dashboard charts for detailed breakdowns."]

    risk_level = urgent.get("severity") or ("MEDIUM" if meta.get("anomalies") else "LOW")
    title = "Network Operations Brief — Access Network Health" if domain == "telecom" \
            else f"Executive Summary: {domain.title()}"
    return {
        "summary_title": title, "highlights": highlights, "risks": risks[:4],
        "recommended_actions": actions[:4], "urgent_action": urgent.get("message"),
        "risk_level": risk_level,
    }


# Backwards-compatible aliases
fallback_analysis = compute_analysis
fallback_design = build_design
fallback_insights = compute_insights

def extract_python(text: str) -> str:
    """Extract Python code from LLM output (strip markdown fences)."""
    if "```python" in text:
        text = text.split("```python", 1)[1]
        text = text.split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1]
        text = text.split("```", 1)[0]
    return text.strip()

def call_llm_with_retry(
    llm: OllamaLLM,
    prompt: str,
    agent_key: str,
    max_retries: int = 1,
    parser=None,
) -> str:
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            p = prompt if attempt == 0 else prompt + f"\n\nPREVIOUS ATTEMPT FAILED: {last_error}\nFix the error and try again."
            response = llm.invoke(p)
            if parser:
                return parser(response)
            return response
        except Exception as e:
            last_error = str(e)
            print(f"  [{agent_key}] attempt {attempt+1} failed: {e}")
            if attempt == max_retries:
                raise
    return ""

# ─── AGENT 1 — THE DETECTIVE ──────────────────────────────────────────────────
AGENT1_SYSTEM = """You are Agent 1 — The Detective. Your job is deep schema discovery and data analysis.
You receive raw tabular data and must return ONLY a valid JSON object (no markdown, no explanation).
The JSON must start with {{ and end with }}.
Fix ALL Arabic text with proper reshaping mentally before writing outputs.
Round all floats to 2 decimal places.
Never hardcode column names — always derive them from the data."""

AGENT1_TEMPLATE = """Analyse this dataset thoroughly and return a single JSON object matching this exact schema:

{{
  "meta": {{
    "domain": "telecom|finance|healthcare|logistics|HR|sales|manufacturing|education|government|other",
    "grain": "what one row represents",
    "row_count": <int>,
    "column_count": <int>,
    "languages_detected": ["english", "arabic", ...],
    "story": "1-2 sentence executive briefing about this dataset",
    "anomalies": ["list of anomalies found"],
    "schema_design_rationale": "why this schema structure was chosen"
  }},
  "columns": [
    {{
      "original_name": "exact column name",
      "clean_name": "snake_case_version",
      "data_type": "numeric|categorical|datetime|text|identifier|boolean",
      "semantic_role": "metric|dimension|timestamp|id|label|status|geo",
      "importance": "high|medium|low",
      "has_arabic": true|false,
      "nullable_pct": 0.00,
      "sample_values": ["up to 5 representative values"],
      "stats": {{}}
    }}
  ],
  "aggregations": {{
    "by_<dimension>": [{{"key": "value", "metrics": {{}}}}],
    "time_series": [{{"period": "value", "metrics": {{}}}}],
    "distributions": {{"column_name": {{"value": count}}}},
    "top_n": {{"description": [{{"label": "", "value": 0}}]}},
    "correlations": [{{"col_a": "", "col_b": "", "direction": "positive|negative"}}]
  }},
  "kpis": [
    {{"label": "", "value": "", "color_hint": "green|red|blue|orange", "icon_hint": "📊|⚠|✅|📈"}}
  ],
  "insights": ["insight 1", "insight 2", "insight 3", "insight 4", "insight 5"],
  "urgent_flag": {{
    "exists": false,
    "message": null,
    "severity": null
  }}
}}

DATA SUMMARY:
{data_summary}

LANGUAGE HINT: {language_hint}

═══ TELECOM / NETWORK OPERATIONS (OSS) EXPERTISE ═══
If this data is telecom/network related (look for: MSAN, DSLAM, OLT, ONT, BTS, NodeB,
eNodeB, gNodeB, BSC, RNC, MSC, cell, site, sector, port, exchange, trunk, link, VLAN,
IP, subscriber, utilization, congestion, throughput, availability, alarm, fault, outage,
SLA, latency, packet loss, RSRP, RSRQ, SINR, KPI, region/area/governorate names), then
set domain="telecom" and act as a senior OSS / Technical Operations analyst:

  • GRAIN is usually one network element per row (one MSAN / cell / site / port / link).
  • Classify operationally — recognise these semantic roles:
      - identifiers: node/element name or ID (MSAN name, site ID, cell ID)
      - dimensions: region / area / exchange / governorate / vendor / technology
      - metrics: utilization %, congestion %, throughput, availability %,
                 affected/impacted subscribers, port count, days_in_state, alarm count
      - status: alarm severity (Critical/Major/Minor/Warning), state (chronic/congested/normal)
  • COMPUTE the KPIs operations leadership cares about (only those the data supports):
      - Total elements monitored, # Critical / Congested / Chronic elements
      - Worst-offender node (highest utilization/congestion) by name + value
      - Total IMPACTED SUBSCRIBERS (sum) — this is the business-impact headline
      - Average / peak utilization or congestion %
      - Capacity-at-risk: count of elements above 80% (warning) and above 90% (critical)
      - Chronic/recurring count: elements critical for 3+ consecutive days (escalation candidates)
      - Network health: % of elements in normal/healthy state
  • AGGREGATIONS to build when possible:
      - by_region (or by_area/by_exchange): congestion/critical count + impacted subs per area
      - top_n: worst N congested/critical nodes by utilization with subscriber impact
      - distributions: alarm severity breakdown; state breakdown (chronic/congested/normal)
      - time_series: if dated, daily trend of critical/congested counts
  • ANOMALIES to flag: nodes at/over 100% utilization, chronic nodes (3+ days),
    nodes with high subscriber impact, sudden congestion spikes, duplicate node entries.
  • urgent_flag: severity=CRITICAL if any chronic-critical nodes or >90% utilization nodes
    exist with significant subscriber impact; HIGH if congestion is widespread; name the
    actual worst node and impacted-subscriber number in the message.

For ALL domains:
- Populate aggregations with REAL data computed from the dataset
- Include at least 4-6 KPIs with real computed values (telecom: lead with impacted subscribers,
  critical/congested counts, worst node, capacity-at-risk)
- Identify the most meaningful dimensions for grouping (telecom: region/area/exchange)
- Set urgent_flag with the specific node name + number when an urgent network condition exists
- Return ONLY the JSON object starting with {{ and ending with }}"""

AGENT1_ROLES_TEMPLATE = """Given these column names and sample values, identify which column serves each role.
Return ONLY a valid JSON object — no markdown, no explanations.

{{
  "entity_column": "exact column name that identifies each network element (node/site/hostname/MSAN)",
  "impact_metric": "exact column name for subscriber or customer count affected (or null)",
  "severity_metric": "exact column name for the primary severity/utilization/critical-time metric (or null)",
  "dimension_columns": ["region/area column", "sector column", "other grouping columns"],
  "status_column": "exact column name for status/alarm/upgrade-state field (or null)"
}}

COLUMNS (dtype — up to 4 sample values):
{column_list}

Use ONLY exact column names from the list above.
Return ONLY the JSON object. First char = {{ last char = }}"""


def run_agent_1(df: pd.DataFrame, model: str, temperature: float, num_ctx: int,
                language_hint: str = "Auto-detect") -> dict:
    """Hybrid Agent 1: LLM identifies column roles; Python computes ALL aggregations."""
    agent_key = "agent_1"
    _start_iso, start_mono = agent_timer(agent_key)
    try:
        # ── Phase 1: ask the LLM only which columns fill which roles (tiny prompt) ──
        col_lines = []
        for c in df.columns:
            dtype = str(df[c].dtype)
            samples = [str(x) for x in df[c].dropna().head(4).tolist()]
            col_lines.append(f"  {c!r:45s} [{dtype}]  samples: {samples}")
        column_list = "\n".join(col_lines)

        prompt = AGENT1_ROLES_TEMPLATE.format(column_list=column_list)
        llm = make_llm(model, 0.0, 2048)

        ai_roles = {}
        try:
            ai_roles = call_llm_with_retry(llm, prompt, agent_key, MAX_RETRIES,
                                           parser=parse_json_lenient)
            print(f"  [agent_1] AI role picks: {ai_roles}")
        except Exception as role_err:
            print(f"  [agent_1] role classification failed ({role_err}); using heuristics only.")

        # ── Phase 2: validate AI picks, then compute analysis entirely in Python ──
        roles = _resolve_roles(df, ai_roles)
        print(f"  [agent_1] resolved → entity={roles.get('entity')}, "
              f"impact={roles.get('impact')}, sev={roles.get('primary_sev')}, "
              f"dims={roles.get('dims')}, status={roles.get('status_dim')}")

        result = compute_analysis(df, roles)
        result.setdefault("meta", {})["language_hint"] = language_hint

        with open(ANALYSIS_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        agent_done(agent_key, start_mono)
        return result
    except Exception as e:
        agent_failed(agent_key, start_mono)
        raise RuntimeError(f"Agent 1 failed: {e}") from e

# ─── AGENT 2 — THE ARCHITECT ──────────────────────────────────────────────────
AGENT2_TEMPLATE = """You are Agent 2 — The Architect. Design the optimal charts for this dataset.
Return ONLY a valid JSON object (no markdown, no explanation).
The JSON must start with {{ and end with }}.

Chart selection rules:
- Categorical + numeric → horizontal_bar
- Time column exists → line or area (ALWAYS include, priority 1)
- Status/category → donut
- Two numerics → scatter
- Single numeric distribution → histogram
- Groups over time → grouped_bar or line
- Ranking → horizontal_bar sorted desc, highlight top 3
- > 5 categories for part-of-whole → bar not donut
- > 20 categories → top 10 + others
- Single critical percentage KPI (overall utilization / availability / network health) → gauge

═══ TELECOM / NETWORK OPERATIONS DASHBOARD DESIGN ═══
If domain is telecom/network, design a NOC-style operations dashboard. Prioritise:
  PRIORITY 1 (top row) — the "what needs attention now" view:
    • A gauge for overall network health % OR average/peak utilization %
      (chart_type="gauge", set has_threshold_line=true, threshold_value=80, threshold_label="Capacity Alert",
       color_scheme="severity")
    • "Worst Offenders" horizontal_bar: top 10 congested/critical nodes by utilization,
      sort_order="desc", top_n=10, highlight_top_n=3, color_scheme="severity",
      has_threshold_line=true (threshold_value=90, threshold_label="Critical"),
      secondary_annotation_field set to the impacted-subscribers field if present (teal annotation)
  PRIORITY 2 — distribution & geography:
    • Alarm-severity / state donut (Critical/Major/Minor/Normal), color_scheme="severity"
    • Congestion/critical count by region/area → horizontal_bar, color_scheme="severity"
  PRIORITY 3 — trend & impact:
    • If dated: daily trend of critical/congested counts → line/area, show_average_line=true
    • Impacted subscribers by region → vertical_bar OR a worst-impact table
Use telecom titles operations staff recognise: "Worst Congested MSANs", "Chronic Critical Nodes",
"Capacity Utilization", "Network Health", "Impacted Subscribers by Region", "Alarm Severity Breakdown".
Always set SLA/capacity threshold lines (80% warning, 90% critical, or 95% availability target).

Design 4-10 charts. width_cols must sum to 12 per row. priority=1 means Row 1.

Return this JSON:
{{
  "dashboard_title": "meaningful title from domain and data",
  "layout_hint": "single_tab|two_tabs|three_tabs",
  "tab_names": ["Tab Name 1", "Tab Name 2"],
  "charts": [
    {{
      "id": "snake_case_unique_id",
      "title": "Human Readable Title",
      "chart_type": "horizontal_bar|vertical_bar|grouped_bar|line|area|scatter|donut|histogram|heatmap|gauge|table",
      "tab": 0,
      "priority": 1,
      "width_cols": 6,
      "data_source": "aggregations.by_dimension_name or aggregations.time_series etc",
      "x_field": "field_name",
      "y_field": "field_name",
      "color_field": null,
      "color_scheme": "severity|categorical|sequential|diverging",
      "secondary_annotation_field": null,
      "has_threshold_line": false,
      "threshold_value": null,
      "threshold_label": null,
      "show_average_line": false,
      "sort_order": "desc|asc|natural",
      "top_n": null,
      "highlight_top_n": null,
      "x_title": "X Axis Label",
      "y_title": "Y Axis Label",
      "insight": "one-sentence insight for this chart"
    }}
  ]
}}

ANALYSIS JSON:
{analysis_json}

Return ONLY the JSON object starting with {{ and ending with }}"""

def run_agent_2(analysis: dict, model: str) -> dict:
    """Hybrid Agent 2: deterministic NOC layout — knows the exact aggregation keys Python produced."""
    agent_key = "agent_2"
    _start_iso, start_mono = agent_timer(agent_key)
    try:
        result = build_design(analysis)
        with open(DESIGN_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        agent_done(agent_key, start_mono)
        return result
    except Exception as e:
        agent_failed(agent_key, start_mono)
        raise RuntimeError(f"Agent 2 failed: {e}") from e

# ─── AGENT 3 — THE CODER ──────────────────────────────────────────────────────
AGENT3_TEMPLATE = """You are Agent 3 — The Coder. Generate executable Python code that builds Plotly figures.

THEME = {{
  "bg":"#F4F6F9", "card":"#FFFFFF", "header":"#1B3A6B",
  "blue1":"#2563EB", "blue2":"#3B82F6", "blue3":"#93C5FD",
  "red":"#DC2626", "orange":"#EA580C", "yellow":"#D97706",
  "green":"#16A34A", "teal":"#0891B2", "gray":"#6B7280",
  "text":"#111827", "font":"DejaVu Sans"
}}

Output ONLY valid Python code (no markdown fences, no explanation).
The code must:
1. Import plotly.graph_objects, plotly.express, arabic_reshaper, bidi
2. Define fix_arabic() function
3. Define all data as Python dicts/lists (no pandas, no file I/O)
4. Build figures dict: figures["chart_id"] = go.Figure(...)
5. Every figure MUST have: paper_bgcolor, plot_bgcolor, font.family, font.color set
6. layout.margin = dict(l=220, r=160, t=60, b=80) for all charts
7. Always apply fix_arabic() to any Arabic labels
8. End with: print("FIGURES_READY")

Chart-specific rules:
- Donuts: hole=0.52, pull problematic slices outward 0.05
- Top N bars: marker_color list with THEME["red"] for top items
- Threshold lines: go.Scatter mode="lines" line_dash="dash" color THEME["red"]
- Average lines: go.Scatter mode="lines" line_dash="dot" color THEME["gray"]
- Secondary annotations: teal #0891B2
- GAUGES: there is NO go.Gauge. Use go.Indicator(mode="gauge+number", value=<num>,
  gauge={{"axis": {{"range": [0, 100]}}, "bar": {{"color": THEME["red"]}},
          "steps": [{{"range": [0,80], "color": "#DCFCE7"}}, {{"range": [80,90], "color": "#FEF3C7"}},
                    {{"range": [90,100], "color": "#FEE2E2"}}]}})
- Only use real Plotly classes: go.Bar, go.Scatter, go.Pie, go.Histogram, go.Heatmap,
  go.Indicator, go.Table. Never invent class names.

ANALYSIS JSON:
{analysis_json}

DESIGN JSON:
{design_json}

Generate the complete Python code now. Start with: import plotly.graph_objects as go"""

def run_agent_3(analysis: dict, design: dict, model: str) -> str:
    agent_key = "agent_3"
    start_iso, start_mono = agent_timer(agent_key)
    try:
        # FAST PATH — skip LLM entirely. The served dashboard (app.py) builds
        # every chart deterministically from analysis + design; figures_code.py
        # is unused at runtime. This saves ~3–4 min per run on small models.
        if FAST_MODE:
            stub = (
                "# Skipped (PIPELINE_FAST=1). The deterministic renderer in app.py\n"
                "# builds every chart from analysis.json + design.json — no LLM-generated\n"
                "# figure code is required for the served dashboard to work.\n"
                "figures = {}\nprint('FIGURES_READY')\n"
            )
            with open(FIGURES_FILE, "w", encoding="utf-8") as f:
                f.write(stub)
            print("  [agent_3] FAST_MODE: stub written (no LLM call).")
            agent_done(agent_key, start_mono)
            return stub

        llm = make_llm(model, 0.0, 8192)
        analysis_short = json.dumps(analysis, ensure_ascii=False)[:3000]
        design_short = json.dumps(design, ensure_ascii=False)[:3000]
        prompt = AGENT3_TEMPLATE.format(
            analysis_json=analysis_short,
            design_json=design_short
        )
        raw = call_llm_with_retry(llm, prompt, agent_key, MAX_RETRIES)
        code = extract_python(raw)
        if not code.startswith("import"):
            code = "import plotly.graph_objects as go\nimport plotly.express as px\n" + code
        if "print(\"FIGURES_READY\")" not in code and "print('FIGURES_READY')" not in code:
            code += '\nprint("FIGURES_READY")'

        # Validate by executing in a sandbox. If it errors or yields no figures,
        # fall back to an empty figures dict — the deterministic renderer in app.py
        # will build every chart itself, so the dashboard still works.
        valid = False
        try:
            sandbox = {}
            exec(compile(code, "<figures_code>", "exec"), sandbox)  # noqa: S102
            figs = sandbox.get("figures")
            valid = isinstance(figs, dict) and len(figs) > 0
        except Exception as ve:
            print(f"  [agent_3] note: LLM figure code not usable ({ve}) — "
                  f"the dashboard renderer will build these charts itself (this is OK).")

        if not valid:
            code = (
                "# LLM figure code was not used; the deterministic renderer in app.py\n"
                "# builds every chart from analysis.json + design.json. This is expected.\n"
                "figures = {}\nprint('FIGURES_READY')\n"
            )

        with open(FIGURES_FILE, "w", encoding="utf-8") as f:
            f.write(code)
        agent_done(agent_key, start_mono)
        return code
    except Exception as e:
        agent_failed(agent_key, start_mono)
        raise RuntimeError(f"Agent 3 failed: {e}") from e

# ─── AGENT 5 — THE NARRATOR ──────────────────────────────────────────────────
AGENT5_TEMPLATE = """You are Agent 5 — The Narrator. Write an executive summary for this dataset.
Return ONLY a valid JSON object (no markdown, no explanation).
The JSON must start with {{ and end with }}.

Tone rules by domain:
- telecom → network operations focus (see OSS guidance below)
- finance → risk and return focus
- HR → people and retention focus
- sales → growth and revenue focus
- healthcare → compliance and quality focus
- other → balanced strategic focus

═══ TELECOM / OSS NARRATIVE (when domain = telecom) ═══
Write as a senior Technical Operations / NOC manager briefing leadership. Be specific and
operational, not generic. Use the real node names and numbers from the analysis.
  • summary_title like: "Network Operations Brief — Access Network Health"
  • highlights: lead with BUSINESS IMPACT then health posture, e.g.
      "• 12 MSANs critical for 3+ consecutive days, impacting 8,400 subscribers"
      "• Worst node MSAN-CAIRO-014 at 98% utilization — capacity exhausted"
      "• 73% of access nodes healthy; 18% congested, 9% chronic-critical"
      "• Congestion concentrated in Giza region (6 of 12 critical nodes)"
  • risks: frame as service-impact / SLA risk with numbers, e.g.
      "⚠ 5 nodes above 90% utilization risk imminent outage for ~3,200 subscribers"
      "⚠ Chronic faults unresolved 3+ days indicate field/escalation backlog"
  • recommended_actions: concrete OSS field actions with the node named, e.g.
      "→ Raise emergency capacity-augmentation work order for MSAN-CAIRO-014 (98%)"
      "→ Escalate the 12 chronic nodes to field operations for site visit within 24h"
      "→ Re-home subscribers from congested Giza cluster to relieve peak load"
      "→ Open trouble tickets and verify alarm correlation for repeat-offender nodes"
  • urgent_action: name the single most critical node + its utilization + impacted subs,
    and the immediate operational step (augment / escalate / dispatch).
Vocabulary to use naturally: utilization, congestion, chronic/critical, capacity augmentation,
SLA, availability, impacted subscribers, escalation, work order/trouble ticket, re-homing,
peak-hour, field operations, NOC, proactive intervention.

Risk level rules:
- CRITICAL if urgent_flag severity = CRITICAL, OR (telecom) chronic-critical nodes exist
  or any node >90% utilization with meaningful subscriber impact
- HIGH if any metric > 2 standard deviations above mean, OR widespread congestion
- MEDIUM if anomalies exist
- LOW otherwise

Return:
{{
  "summary_title": "Executive Summary: <Domain> Analytics",
  "highlights": [
    "• Real number insight with specific values",
    "• Real number insight with specific values",
    "• Real number insight with specific values",
    "• Real number insight with specific values"
  ],
  "risks": [
    "⚠ Risk with real numbers and specific context"
  ],
  "recommended_actions": [
    "→ Specific actionable recommendation"
  ],
  "urgent_action": null,
  "risk_level": "LOW|MEDIUM|HIGH|CRITICAL"
}}

ANALYSIS JSON:
{analysis_json}

Return ONLY the JSON object. First char = {{ last char = }}"""

def run_agent_5(analysis: dict, model: str, temperature: float) -> dict:
    """Hybrid Agent 5: compute deterministic baseline first; LLM enriches the narrative.

    Numbers always come from Python (baseline). LLM only adds operational wording.
    If LLM fails the baseline is used directly — the dashboard always shows correct insights.
    In FAST_MODE the LLM call is skipped entirely (baseline already contains real numbers
    + risks + actions; saves ~1 min on small models).
    """
    agent_key = "agent_5"
    _start_iso, start_mono = agent_timer(agent_key)
    try:
        # ── Phase 1: deterministic baseline (always correct numbers) ──
        baseline = compute_insights(analysis)

        # FAST PATH — baseline narrative is already operational-grade.
        if FAST_MODE:
            with open(INSIGHTS_FILE, "w", encoding="utf-8") as f:
                json.dump(baseline, f, indent=2, ensure_ascii=False)
            print("  [agent_5] FAST_MODE: deterministic baseline only (no LLM call).")
            agent_done(agent_key, start_mono)
            return baseline

        # ── Phase 2: optional LLM narrative enrichment ──
        # Smaller context (2048 vs 4096) → ~2× faster on small models.
        llm = make_llm(model, temperature, 2048)
        prompt = AGENT5_TEMPLATE.format(
            analysis_json=json.dumps(analysis, ensure_ascii=False)[:2500]
        )
        try:
            llm_result = call_llm_with_retry(llm, prompt, agent_key, MAX_RETRIES,
                                             parser=parse_json_lenient)
            if isinstance(llm_result, dict) and "risk_level" in llm_result:
                # Merge: keep LLM's richer narrative but guarantee baseline fields exist
                baseline.update({k: v for k, v in llm_result.items() if v})
                result = baseline
                print(f"  [agent_5] LLM narrative applied (risk={result.get('risk_level')}).")
            else:
                result = baseline
                print(f"  [agent_5] LLM output invalid; using deterministic baseline.")
        except Exception as parse_err:
            print(f"  [agent_5] LLM enrichment failed ({parse_err}); using deterministic baseline.")
            result = baseline

        with open(INSIGHTS_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        agent_done(agent_key, start_mono)
        return result
    except Exception as e:
        agent_failed(agent_key, start_mono)
        raise RuntimeError(f"Agent 5 failed: {e}") from e

# ─── AGENT 4 — THE BUILDER ────────────────────────────────────────────────────
AGENT4_TEMPLATE = """You are Agent 4 — The Builder. Generate a complete, runnable Dash application.
Return ONLY valid Python code (no markdown fences, no explanation).
The code must start with: import dash

Requirements:
1. Use dash, dash-bootstrap-components (dbc.themes.FLATLY), plotly
2. Import figures from the figures dict (use exec() to load figures_code.py content)
3. Build layout dynamically from analysis JSON and design JSON — NEVER hardcode chart IDs or tab names
4. Include ALL these features:
   - Top header bar: dashboard_title + current date, #1B3A6B background, white text
   - Urgent banner if urgent_flag.exists: red banner with bold white message
   - dcc.Tabs with tab_names from design JSON
   - Per tab: KPI cards row + chart rows grouped by priority + insights panel
   - Footer: Download PNG button + Export PDF button + auto-refresh dropdown + upload new file
   - Dark/Light mode toggle
   - RTL toggle for Arabic support
5. KPI cards: white bg, bold colored value (use color_hint), gray label, box-shadow
6. Chart cards: white bg, border-radius 8px, subtle border, dcc.Graph
7. Callbacks for:
   - Dark mode toggle (toggle body class)
   - RTL toggle
   - Auto-refresh interval
   - Download PNG (using kaleido)
   - dcc.Upload for new file triggering pipeline re-run
8. App runs on host="0.0.0.0" port=8050 debug=False

ANALYSIS JSON (truncated):
{analysis_json}

DESIGN JSON (truncated):
{design_json}

INSIGHTS JSON:
{insights_json}

FIGURES CODE PATH: output/figures_code.py

Generate complete runnable app.py code now. Start with: import dash"""

def run_agent_4(analysis: dict, design: dict, figures_code: str, insights: dict, model: str) -> str:
    agent_key = "agent_4"
    start_iso, start_mono = agent_timer(agent_key)
    try:
        # FAST PATH — skip LLM entirely. app_generated.py is a "creative" version
        # of the dashboard, never served. The real served dashboard is app.py
        # (deterministic renderer, never overwritten). Saves ~3–4 min per run.
        if FAST_MODE:
            stub = (
                "# Skipped (PIPELINE_FAST=1).\n"
                "# The served dashboard is app.py — a deterministic renderer that reads\n"
                "# analysis.json + design.json + insights.json live. Set env\n"
                "# PIPELINE_FAST=0 to have Agent 4 also produce a creative variant here.\n"
            )
            with open(GENERATED_APP_FILE, "w", encoding="utf-8") as f:
                f.write(stub)
            print("  [agent_4] FAST_MODE: stub written (no LLM call).")
            agent_done(agent_key, start_mono)
            return stub

        llm = make_llm(model, 0.0, 8192)
        prompt = AGENT4_TEMPLATE.format(
            analysis_json=json.dumps(analysis, ensure_ascii=False)[:2500],
            design_json=json.dumps(design, ensure_ascii=False)[:2500],
            insights_json=json.dumps(insights, ensure_ascii=False)[:1500],
        )
        raw = call_llm_with_retry(llm, prompt, agent_key, MAX_RETRIES)
        code = extract_python(raw)
        if not code.startswith("import"):
            code = "import dash\n" + code
        # Save Agent 4's creative version for inspection. The SERVED dashboard is
        # the deterministic renderer in app.py (never overwritten), so a broken
        # LLM generation can never take the dashboard down.
        with open(GENERATED_APP_FILE, "w", encoding="utf-8") as f:
            f.write(code)
        agent_done(agent_key, start_mono)
        return code
    except Exception as e:
        agent_failed(agent_key, start_mono)
        raise RuntimeError(f"Agent 4 failed: {e}") from e

# ─── Dashboard Launcher ───────────────────────────────────────────────────────
_dash_proc = None

def _dash_already_running() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:8050", timeout=2)
        return True
    except Exception:
        return False

def launch_dashboard():
    global _dash_proc
    # The renderer reads JSON live via app.layout, so if it's already up we don't
    # need to restart it — the new artifacts appear on the next page refresh.
    if _dash_already_running():
        print("Dashboard already running at http://localhost:8050 — refresh to update.")
        return None
    if _dash_proc and _dash_proc.poll() is None:
        return _dash_proc
    app_path = str(APP_FILE)
    # IMPORTANT: redirect child output to a log file. Using PIPE without draining
    # it deadlocks the child once the OS pipe buffer (~64KB) fills.
    log = open(OUTPUT_DIR / "dash_server.log", "ab")
    _dash_proc = subprocess.Popen(
        [sys.executable, app_path],
        cwd=str(ROOT),
        stdout=log,
        stderr=log,
    )
    print(f"Dashboard launched at http://localhost:8050 (pid={_dash_proc.pid})")
    return _dash_proc

# ─── Main Orchestration ───────────────────────────────────────────────────────
def run_pipeline(
    file_paths: list,
    model: str = None,
    agent_1_ctx: int = 16384,
    agent_1_temp: float = 0.0,
    agent_5_temp: float = 0.3,
    language_hint: str = "Auto-detect",
):
    if model is None:
        model = DEFAULT_MODEL

    print(f"\n{'='*60}")
    print(f"  Universal Dashboard Pipeline")
    print(f"  Model: {model}  Files: {len(file_paths)}")
    print(f"{'='*60}\n")

    # Reset status
    for k in PIPELINE_STATUS:
        PIPELINE_STATUS[k] = {"status": "idle", "start": None, "end": None, "duration_sec": None}
    write_status()

    # Load data
    print("Loading files...")
    dfs = load_files(file_paths)
    if not dfs:
        raise ValueError("No files could be loaded.")
    df = merge_dataframes(dfs)
    print(f"Combined dataset: {df.shape[0]} rows × {df.shape[1]} cols\n")

    # Agent 1
    print("--- Agent 1: The Detective ---")
    analysis = run_agent_1(df, model, agent_1_temp, agent_1_ctx, language_hint)
    print(f"  Domain: {analysis.get('meta', {}).get('domain', 'unknown')}")
    print(f"  KPIs: {len(analysis.get('kpis', []))}")

    # Agent 2
    print("\n--- Agent 2: The Architect ---")
    design = run_agent_2(analysis, model)
    print(f"  Charts designed: {len(design.get('charts', []))}")
    print(f"  Layout: {design.get('layout_hint')}")

    # Agents 3 + 5 in parallel
    print("\n--- Agents 3 + 5 (parallel) ---")
    figures_code = None
    insights = None
    errors = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_3 = executor.submit(run_agent_3, analysis, design, model)
        future_5 = executor.submit(run_agent_5, analysis, model, agent_5_temp)
        for future in as_completed([future_3, future_5]):
            try:
                result = future.result()
                if future == future_3:
                    figures_code = result
                    print("  Agent 3 (Coder): done")
                else:
                    insights = result
                    print("  Agent 5 (Narrator): done")
            except Exception as e:
                errors.append(str(e))
                print(f"  ERROR: {e}")

    if errors:
        print(f"  Parallel stage errors: {errors}")

    if figures_code is None:
        figures_code = '# Agent 3 failed\nfigures = {}\nprint("FIGURES_READY")'
        with open(FIGURES_FILE, "w") as f:
            f.write(figures_code)

    if insights is None:
        insights = {
            "summary_title": "Analysis Complete",
            "highlights": ["Data processed successfully."],
            "risks": [],
            "recommended_actions": ["Review the dashboard for details."],
            "urgent_action": None,
            "risk_level": "LOW"
        }
        with open(INSIGHTS_FILE, "w", encoding="utf-8") as f:
            json.dump(insights, f, indent=2, ensure_ascii=False)

    # Agent 4
    print("\n--- Agent 4: The Builder ---")
    app_code = run_agent_4(analysis, design, figures_code, insights, model)
    print("  app.py generated")

    # Launch Dashboard
    print("\n--- Launching Dashboard ---")
    launch_dashboard()
    print("\nPipeline complete!")
    print(f"  Dashboard: http://localhost:8050")
    return {
        "analysis": analysis,
        "design": design,
        "insights": insights,
        "app_code": app_code
    }

# ─── CLI Entry Point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Check for uploaded_files.json (set by Streamlit)
    uploaded_json = OUTPUT_DIR / "uploaded_files.json"
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    elif uploaded_json.exists():
        with open(uploaded_json) as f:
            files = json.load(f)
    else:
        print("Usage: python orchestrate.py file1.xlsx file2.csv ...")
        print("Or set output/uploaded_files.json with a list of file paths.")
        sys.exit(1)

    # Optional config from environment
    model = os.environ.get("PIPELINE_MODEL", DEFAULT_MODEL)
    ctx = int(os.environ.get("PIPELINE_CTX", "16384"))
    temp1 = float(os.environ.get("PIPELINE_TEMP1", "0.0"))
    temp5 = float(os.environ.get("PIPELINE_TEMP5", "0.3"))
    lang = os.environ.get("PIPELINE_LANG", "Auto-detect")

    try:
        run_pipeline(files, model=model, agent_1_ctx=ctx,
                     agent_1_temp=temp1, agent_5_temp=temp5, language_hint=lang)
    except Exception as e:
        print(f"\nPIPELINE FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Remove our PID marker so the control panel knows no run is active.
        try:
            (OUTPUT_DIR / "pipeline.pid").unlink()
        except Exception:
            pass
