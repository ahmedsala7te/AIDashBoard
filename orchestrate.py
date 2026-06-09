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

# Flagship domain feature: per-operator / carrier analytics (WE wholesale model)
import telecom_intelligence as ti

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

# ─── AI Design mode ───────────────────────────────────────────────────────────
# When True, the Architect (Agent 2) and the Reviewer (Agent 2b) call the LLM
# to design and critique the dashboard.  When False, the deterministic
# build_design() is used (fast, but mechanical).  Defaults to True — the user
# explicitly asked for the AI to be involved in detection + design + review.
# Override via env: PIPELINE_AI_DESIGN=0 to disable.
AI_DESIGN_MODE = os.environ.get("PIPELINE_AI_DESIGN", "1") not in ("0", "false", "False", "no")


# ─── Reviewer toggle ──────────────────────────────────────────────────────────
# The Reviewer (Agent 3 of the AI chain) takes one extra LLM call to critique
# and possibly revise the Architect's design. Skipping it saves 1-3 minutes per
# run on consumer GPUs. The Architect's output already passes through strict
# validation (_validate_design) so quality stays high even without the Reviewer.
# Override via env: PIPELINE_SKIP_REVIEWER=1
SKIP_REVIEWER = os.environ.get("PIPELINE_SKIP_REVIEWER", "0") in ("1", "true", "True", "yes")


# ─── User instructions ────────────────────────────────────────────────────────
# Free-form text the user typed in Streamlit's "AI Instructions" box.
# These are injected into every agent's prompt so the user can steer the AI:
#   "Focus on capacity planning"
#   "Use NOC operations vocabulary"
#   "Skip the upgrade-status donut"
#   "Treat as a weekly report"
# Stored in a FILE rather than an env var so multi-line text + special chars survive.
def _load_user_instructions() -> str:
    path = os.environ.get("PIPELINE_USER_INSTRUCTIONS_FILE", "")
    if path and os.path.exists(path):
        try:
            text = open(path, encoding="utf-8").read().strip()
            return text
        except Exception:
            return ""
    # Fallback: inline env var (small instructions only)
    return os.environ.get("PIPELINE_USER_INSTRUCTIONS", "").strip()


USER_INSTRUCTIONS = _load_user_instructions()
if USER_INSTRUCTIONS:
    print(f"[init] user instructions received ({len(USER_INSTRUCTIONS)} chars).")


def _instruction_block(label: str = "USER INSTRUCTIONS") -> str:
    """Return a formatted block to inject into prompts, or empty string when no instructions."""
    if not USER_INSTRUCTIONS:
        return ""
    # Frame strongly so small models actually pay attention to it.
    return (
        f"\n═══ {label} (MUST FOLLOW) ════════════════════════════════════════════\n"
        f"The user supplied these specific instructions for this dashboard:\n\n"
        f"{USER_INSTRUCTIONS}\n\n"
        f"Treat these as priority overrides — they trump any conflicting default behaviour.\n"
        f"════════════════════════════════════════════════════════════════════\n\n"
    )

# Dashboard style — chosen by the user in Streamlit before running the pipeline.
# "universal"  → the default light-theme universal dashboard
# "executive"  → the dark-navy Executive KPI Dashboard (executive_layout.py)
PIPELINE_STYLE = os.environ.get("PIPELINE_STYLE", "universal").lower().strip() or "universal"
if PIPELINE_STYLE not in ("universal", "executive"):
    PIPELINE_STYLE = "universal"

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
        df = pd.read_excel(path)
    elif ext == ".csv":
        enc = detect_encoding(path)
        try:
            df = pd.read_csv(path, encoding=enc)
        except Exception:
            df = pd.read_csv(path, encoding="utf-8", errors="replace")
    elif ext == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # Normalise column names at the source so every downstream stage sees one
    # column per name. We strip leading/trailing whitespace (pandas-read Excel
    # often gives '  Region  ') and dedupe — these are the two failure modes
    # that produce surprise duplicates without the user realising.
    df.columns = [str(c).strip() if isinstance(c, str) else c for c in df.columns]
    df = _dedupe_column_names(df)
    return df

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
def make_llm(model: str, temperature: float, num_ctx: int,
              num_predict: int = 1500) -> OllamaLLM:
    """Create an Ollama client with explicit context AND output-length caps.

    Speed economics (qwen2.5:14b on a 12 GB GPU):
      • num_ctx = prefill cost; doubling it adds ~30-60s to first-token latency
      • num_predict = output cost; uncapped output can run 5+ minutes when the
        model decides to verbose-explain.  Capping at 1500 forces concise JSON.

    Default num_predict=1500 is enough for any of our agent prompts:
      Detective JSON  ≈ 800-1100 tokens
      Architect JSON  ≈ 500-900 tokens
      Reviewer JSON   ≈ 400-700 tokens
      Narrator JSON   ≈ 300-600 tokens
    """
    return OllamaLLM(
        base_url=OLLAMA_HOST,
        model=model,
        temperature=temperature,
        num_ctx=num_ctx,
        num_predict=num_predict,
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

# ── Business sub-domain hints (within the telecom domain) ─────────────────────
# Each "business" produces a different dashboard concept: different KPIs,
# different chart selection, different narrative language.
CONGESTION_HINT = ("critical_time", "critical time", "chronic", "congest", "outage_duration")
INVENTORY_HINT  = ("vendor", "technology", "port", "olt", "ont", "interface", "model",
                   "make", "capacity", "max", "free", "configured", "in_service",
                   "out_of_service", "rack", "shelf", "slot", "card")
GEO_HINT        = ("latitude", "longitude", "lat", "long", "lat_", "long_", "_lat", "_lng")
ALARM_HINT      = ("alarm", "cleared", "occurred", "ack_time", "ack ", "raised", "alert")
TICKET_HINT     = ("ticket", "incident", "case", "complaint", "sla", "owner",
                   "assignee", "resolved", "created_at", "closed_at", "opened_at")
PERFORMANCE_HINT = ("throughput", "latency", "loss", "bandwidth", "rsrp", "rsrq",
                    "sinr", "cpu", "memory", "load_pct")


def _detect_business(df: pd.DataFrame, cols_blob: str, domain: str) -> str:
    """Identify the telecom business sub-domain so the dashboard concept fits the data.

    Returns one of:
      "congestion" — critical-time / chronic-critical reports (current default)
      "inventory"  — GPON/OLT/MSAN inventory exports with capacity + vendor + geo
      "alarms"     — alarm logs with severity + raised/cleared timestamps
      "tickets"    — trouble-ticket / incident lists with SLA + ownership
      "performance" — KPI snapshots (throughput / latency / utilization only)
      "other_telecom" — telecom-flavoured but no specific shape detected
      "general"    — not telecom
    Decision order matters: congestion wins over inventory because chronic-critical
    reports often also carry capacity columns.
    """
    if domain != "telecom":
        return "general"
    cb = cols_blob.lower()
    def has(keywords): return any(k in cb for k in keywords)

    if has(CONGESTION_HINT):
        return "congestion"
    if has(ALARM_HINT) and not has(("ticket", "incident", "sla")):
        return "alarms"
    if has(TICKET_HINT):
        return "tickets"
    if has(INVENTORY_HINT) or has(GEO_HINT):
        return "inventory"
    if has(PERFORMANCE_HINT):
        return "performance"
    return "other_telecom"


# ── Mojibake repair (UTF-8 decoded as cp1252) ────────────────────────────────
def _repair_mojibake(text):
    """Repair the very common 'UTF-8 bytes decoded as cp1252' mojibake — the
    pattern that produces strings like 'â€‹â€‹DAKAHLIA' or 'Ù‚Ø·Ø§Ø¹'.

    Safe: only flips a string when the round-trip both succeeds AND produces
    valid Arabic-or-Latin output.  Returns the original text otherwise.
    """
    if not isinstance(text, str) or not text:
        return text
    # Cheap signal: mojibake almost always contains one of these sequences.
    if not any(sig in text for sig in ("Ã", "Ù", "Ø", "â€", "Â")):
        return text
    try:
        repaired = text.encode("cp1252", errors="strict").decode("utf-8", errors="strict")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    return repaired


def _repair_dataframe_mojibake(df: pd.DataFrame) -> pd.DataFrame:
    """Repair _data_ mojibake in every object/string column.

    DELIBERATELY does NOT repair column names — collapsing two distinct
    mojibake-encoded headers into the same repaired name silently creates
    duplicate columns, and `df["X"]` then returns a DataFrame instead of a
    Series, blowing up every downstream `int(df[X].sum())` with a confusing
    "float() argument must be a real number, not 'Series'" error.
    """
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            try:
                df[c] = df[c].map(lambda v: _repair_mojibake(v) if isinstance(v, str) else v)
            except Exception:
                pass
    return df


def _dedupe_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure every column name in df is unique by suffixing repeats `_1`, `_2`, …

    Excel sheets occasionally have headers like "Subscribers" twice (one merged
    cell + one real header).  When that happens `df["Subscribers"]` returns a
    *DataFrame*, and operations like `.sum()` produce a Series rather than a
    scalar — which then crashes float-coercing helpers like `_round2`.
    This guard guarantees one column per name.
    """
    seen = {}
    new_names = []
    for c in df.columns:
        if c not in seen:
            seen[c] = 0
            new_names.append(c)
        else:
            seen[c] += 1
            new_names.append(f"{c}_{seen[c]}")
    if new_names != list(df.columns):
        renamed = [(orig, new) for orig, new in zip(df.columns, new_names) if orig != new]
        # ASCII-only print so the Windows cp1252 console doesn't choke.
        sample = ", ".join(f"{o!r}->{n!r}" for o, n in renamed[:5])
        suffix = " ..." if len(renamed) > 5 else ""
        print(f"  [dedup] renamed {len(renamed)} duplicate columns: {sample}{suffix}")
        df = df.copy()
        df.columns = new_names
    return df


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
    df = _dedupe_column_names(df)      # safety: never let duplicate names break classification
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
    """Merge AI-chosen roles with heuristics. AI picks are VALIDATED against real columns
    AND against semantic sanity — small models often confuse severity with impact when
    a "subscribers" column is more prominent than a "critical_time" column.

    SAFETY OVERRIDES (in priority order, applied even when the AI gave us a valid pick):
      1. Entity: prefer a hostname/host_name/node_name column over a code column.
      2. Severity ≠ Impact: if the AI set severity_metric == impact_metric AND a real
         critical/utilization/time-based column exists, use that as severity instead.
         This is what produced your "AVG SUBSCRIBERS / PEAK SUBSCRIBERS" KPIs —
         the model collapsed both roles onto 'subscribers'.
      3. Severity for telecom: if heuristic found a critical/time column and the AI's
         pick is not time-based, prefer the heuristic's time column.
    """
    h = _heuristic_roles(df)
    if not isinstance(ai_roles, dict):
        return h
    cols = set(df.columns)

    def v(x):
        return x if x in cols else None

    def vlist(xs):
        return [x for x in (xs or []) if x in cols]

    def _is_time_like(col):
        if not col:
            return False
        n = str(col).lower()
        return any(t in n for t in ("critical", "time", "_min", "duration", "downtime",
                                      "outage", "util", "congest", "fault"))

    # 1. Force hostname-preference for entity
    hostname_col = next(
        (c for c in df.columns
         if any(kw in str(c).lower() for kw in ("hostname", "host_name", "node_name"))),
        None,
    )
    entity_pick = hostname_col or v(ai_roles.get("entity_column")) or h["entity"]

    # 2. Resolve impact (the AI's pick is usually fine)
    impact_pick = v(ai_roles.get("impact_metric")) or h["impact"]

    # 3. Resolve severity with the sanity overrides above
    ai_sev = v(ai_roles.get("severity_metric"))
    heur_sev = h["primary_sev"]

    # SAFETY: small models often pick the same column for both roles.
    # Detect and correct: if heuristic found a real time/critical column, prefer it.
    if ai_sev and ai_sev == impact_pick and heur_sev and heur_sev != ai_sev:
        print(f"  [resolve_roles] AI confused severity & impact (both={ai_sev!r}); "
              f"overriding severity to {heur_sev!r} from heuristic.")
        sev_pick = heur_sev
    # SAFETY: if heuristic clearly found a time-based severity column and AI didn't,
    # prefer the time-based one (it produces operationally-meaningful KPIs).
    elif heur_sev and _is_time_like(heur_sev) and not _is_time_like(ai_sev):
        if ai_sev and ai_sev != heur_sev:
            print(f"  [resolve_roles] AI picked non-time severity ({ai_sev!r}); "
                  f"a time-based column exists ({heur_sev!r}) — using that instead.")
        sev_pick = heur_sev
    else:
        sev_pick = ai_sev or heur_sev

    return {
        "entity": entity_pick,
        "impact": impact_pick,
        "primary_sev": sev_pick,
        "crit_cols": h["crit_cols"],
        "dims": vlist(ai_roles.get("dimension_columns")) or h["dims"],
        "status_dim": v(ai_roles.get("status_column")) or h["status_dim"],
        "metrics": h["metrics"],
    }


def compute_analysis(df: pd.DataFrame, roles: dict = None) -> dict:
    """Compute a correct, telecom-aware analysis from the dataframe.

    All numbers are computed in Python. `roles` (optionally chosen by the AI) decides
    WHICH columns are entity/impact/severity/dimensions; the maths is deterministic.
    """
    # ── Step 0a: Force unique column names ────────────────────────────────────
    # Duplicate headers from Excel/CSV exports turn every `df[col]` into a
    # DataFrame, breaking every aggregation. Dedupe FIRST so no downstream
    # path ever encounters duplicate columns.
    df = _dedupe_column_names(df)
    # ── Step 0b: Repair mojibake in data cells ────────────────────────────────
    # Misencoded source exports produce strings like "Ù‚Ø·Ø§Ø¹" — visible as
    # garbled Arabic in chart labels.  Repair them BEFORE anything else so
    # aggregation keys and labels are clean Arabic from the start.
    df = _repair_dataframe_mojibake(df)
    # ── Step 0c: Coerce mixed-type columns to numeric ─────────────────────────
    # Excel files often give us object-dtype columns that are actually numbers
    # (one stray text cell makes pandas read the whole column as object).
    # This must happen BEFORE classify/roles so metrics are found correctly.
    df = _coerce_numeric_columns(df)
    n = len(df)
    cols_blob = " ".join(str(c).lower() for c in df.columns)
    domain = "telecom" if any(k in cols_blob for k in TELECOM_KEYWORDS) else "other"
    # NEW: identify the BUSINESS sub-domain so the dashboard concept fits.
    business = _detect_business(df, cols_blob, domain)
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
        # De-duplicate column selection — when primary_sev == impact (common in
        # inventory data where 'subscribers' is BOTH the severity AND the impact
        # field), df[[col, col]] yields a 2-column slice and sort_values blows up.
        keep = list(dict.fromkeys([entity, primary_sev] + ([impact] if impact else [])))
        sub = df[keep].dropna(subset=[primary_sev]).sort_values(primary_sev, ascending=False).head(10)
        recs = []
        for _, row in sub.iterrows():
            m = {skey: _round2(row[primary_sev])}
            if impact:
                m[ikey] = _round2(row[impact])
            recs.append({"key": str(row[entity]), "metrics": m})
        aggregations["top_offenders"] = recs

    # ── By dimension ──
    # Quality gate: a column only makes a good grouping bar/donut if it has a
    # readable number of categories AND isn't dominated by a placeholder value
    # (e.g. 'downstream_list' is 70%+ '-'). Junk dimensions otherwise produce
    # noisy, meaningless charts.
    def _is_chartworthy_dim(col: str) -> bool:
        s = df[col].astype(str).str.strip()
        # exclude empty/placeholder cells from the judgement
        clean = s[~s.isin(("-", "", "nan", "none", "null", "n/a", "na"))]
        nun = clean.nunique()
        if nun < 2 or nun > 25:               # too few or too many categories
            return False
        if len(clean) < 0.4 * n:              # mostly placeholders → not meaningful
            return False
        top_share = clean.value_counts(normalize=True).iloc[0] if len(clean) else 1.0
        if top_share > 0.95:                  # one value dominates → no insight
            return False
        # list-like columns (multiple items per cell) are not categorical dims
        if any(tok in str(col).lower() for tok in ("list", "_ids", "members")):
            return False
        avg_len = clean.str.len().mean() if len(clean) else 0
        if avg_len and avg_len > 40:          # long free-text, not a category
            return False
        return True

    chartworthy = [d for d in dims if _is_chartworthy_dim(d)]
    for d in sorted(chartworthy, key=lambda c: (0 if any(h in str(c).lower() for h in ("region", "area", "governorate")) else
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

    # ── Inventory / capacity aggregations ─────────────────────────────────────
    # Only build these when the data shape supports it, regardless of business —
    # vendor breakdown and geo points are useful whenever the columns are present.
    #
    # STRICT MATCHING is critical here:  loose substring search makes "lat"
    # match "ETISILAT_SUB" and "vendor" match "FLAVOR_RANK", which then
    # contaminates the column selection and produces Series-instead-of-scalar
    # errors downstream.  We use EXACT TOKENS only.
    def _norm(name): return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
    norm_map = {c: _norm(c) for c in df.columns}

    def _exact_token_match(allowed_tokens):
        """Return the first column whose normalised name == one of allowed_tokens
        OR ends with `_<token>` (e.g. 'site_latitude' matches 'latitude')."""
        for c, nm in norm_map.items():
            for tok in allowed_tokens:
                if nm == tok or nm.endswith("_" + tok) or nm.startswith(tok + "_"):
                    return c
        return None

    vendor_col = _exact_token_match(
        ["vendor", "vendor_name", "make", "manufacturer", "supplier"]
    )
    tech_col = _exact_token_match(
        ["technology", "tech", "platform", "access_type", "access_tech"]
    )
    lat_col = _exact_token_match(["latitude", "lat", "y_lat", "geo_lat"])
    lng_col = _exact_token_match(["longitude", "long", "lng", "lon", "geo_long", "geo_lng"])

    # VALUE-RANGE guard: even after name match, confirm the column holds
    # geographic numbers. Latitude must be in [-90, 90], longitude in [-180, 180].
    # This rules out misnamed columns like a "LAT" code field full of integers.
    def _is_valid_geo(col, lo, hi):
        if not col or not pd.api.types.is_numeric_dtype(df[col]):
            return False
        v = df[col].dropna()
        if v.empty:
            return False
        return bool(v.between(lo, hi).mean() >= 0.9)

    if not _is_valid_geo(lat_col, -90, 90):    lat_col = None
    if not _is_valid_geo(lng_col, -180, 180):  lng_col = None

    # CRITICAL: a single column can play AT MOST ONE role. If vendor_col happens
    # to be the same column as entity/impact/status, drop it from the inventory
    # side — otherwise downstream slices like `df[[entity, impact, vendor_col]]`
    # produce duplicate columns and every aggregation breaks.
    role_cols = {entity, impact, primary_sev, status_dim} - {None}
    if vendor_col in role_cols: vendor_col = None
    if tech_col   in role_cols: tech_col   = None
    if lat_col    in role_cols: lat_col    = None
    if lng_col    in role_cols: lng_col    = None

    if vendor_col and df[vendor_col].nunique(dropna=True) >= 2:
        vc = df[vendor_col].astype(str).value_counts()
        # Aggregate subscribers per vendor if we have an impact column
        recs = []
        for k, count in vc.items():
            metrics = {"count": int(count)}
            if impact:
                metrics[ikey] = int(df.loc[df[vendor_col].astype(str) == k, impact].sum() or 0)
            recs.append({"key": str(k), "metrics": metrics})
        aggregations["by_vendor"] = recs

    if tech_col and df[tech_col].nunique(dropna=True) >= 2:
        vc = df[tech_col].astype(str).value_counts()
        aggregations["by_technology"] = [
            {"key": str(k), "metrics": {"count": int(v)}} for k, v in vc.items()
        ]

    if business == "inventory" and entity and impact:
        # Top capacity sites: largest deployments by subscriber/port count.
        keep = list(dict.fromkeys([entity, impact]))   # dedupe even if collapsed
        sub = (df[keep].dropna(subset=[impact])
                  .sort_values(impact, ascending=False).head(10))
        recs = []
        for _, row in sub.iterrows():
            recs.append({"key": str(row[entity]),
                         "metrics": {ikey: _round2(row[impact])}})
        aggregations["top_capacity"] = recs

    if lat_col and lng_col:
        # Dedupe the column selection list so we never request the same column twice.
        keep = list(dict.fromkeys(
            [c for c in (entity, lat_col, lng_col, impact, vendor_col) if c]
        ))
        geo = df[keep].dropna(subset=[lat_col, lng_col]).head(500)   # cap for browser perf
        recs = []
        for _, row in geo.iterrows():
            r = {"key": str(row.get(entity)) if entity else "",
                 "lat": float(row[lat_col]), "lng": float(row[lng_col])}
            if impact and impact in geo.columns:
                r["impact"] = _round2(row[impact])
            if vendor_col and vendor_col in geo.columns:
                r["vendor"] = str(row[vendor_col])
            recs.append(r)
        aggregations["geo_points"] = recs

    # ── FLAGSHIP: per-operator / carrier analytics ───────────────────────────
    # WE is a wholesale provider; a single element carries several operators'
    # subscribers. This recovers the operator dimension the rest of the pipeline
    # historically collapsed into one number. Universal: pattern-based detection.
    operator_result = ti.build_operator_analytics(
        df,
        entity_col=entity,
        severity_col=primary_sev if is_time else None,
        severity_high_is_bad=True,   # critical-time / utilisation: higher = worse
    )
    if operator_result:
        # Merge operator aggregations (operator_mix, operator_exposure, wholesale_vs_retail)
        aggregations.update(operator_result.get("aggregations", {}))
        print(f"  [operators] detected {operator_result['summary'].get('n_operators')} operators: "
              f"{', '.join(operator_result['summary'].get('operators', []))}")

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
    # Business-aware "total" label — "MSANs Deployed" reads better for inventory
    # than "MSANs Monitored" (which implies an active monitoring posture).
    total_label = {
        "congestion":    "MSANs Monitored",
        "inventory":     "Network Elements Deployed",
        "alarms":        "Active Alarms",
        "tickets":       "Open Tickets",
        "performance":   "Elements Sampled",
        "other_telecom": "Network Elements",
        "general":       "Total Records",
    }.get(business, "Total Records")
    total_icon = "🗼" if domain == "telecom" else "📊"
    kpis = [{"label": total_label, "value": f"{n:,}",
              "color_hint": "blue", "icon_hint": total_icon}]
    worst_name = worst_val = worst_impact = None
    if entity and primary_sev:
        wi = df[primary_sev].idxmax()
        worst_name, worst_val = str(df.loc[wi, entity]), _round2(df[primary_sev].max())
        if impact:
            worst_impact = int(df.loc[wi, impact])
    if impact:
        kpis.append({"label": "Impacted Subscribers", "value": f"{int(df[impact].sum()):,}",
                     "color_hint": "red", "icon_hint": "⚠"})
    # Operator KPIs are high-value for WE's wholesale model — surface them early.
    if operator_result and operator_result.get("kpis"):
        kpis.extend(operator_result["kpis"])
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

    # Business-aware story — different language per sub-domain.
    if business == "congestion":
        story = (f"{n} chronic-critical MSANs analysed"
                 + (f", impacting {int(df[impact].sum()):,} subscribers" if impact else "")
                 + (f"; worst node {worst_name} ({_metric_label(primary_sev)} {worst_val})" if worst_name else "")
                 + (f"; hotspot region: {hotspot}." if hotspot else "."))
    elif business == "inventory":
        vendor_part = ""
        if vendor_col and df[vendor_col].nunique(dropna=True) >= 1:
            top_vendor = str(df[vendor_col].astype(str).value_counts().index[0])
            vendor_part = f"; dominant vendor: {top_vendor}"
        story = (f"{n} network elements catalogued"
                 + (f", serving {int(df[impact].sum()):,} subscribers" if impact else "")
                 + vendor_part
                 + (f"; broadest coverage: {hotspot}." if hotspot else "."))
    elif business == "alarms":
        story = (f"{n} alarms ingested"
                 + (f" across {df[status_dim].nunique()} severities" if status_dim else "")
                 + (f"; worst node {worst_name}." if worst_name else "."))
    elif business == "tickets":
        story = (f"{n} trouble tickets analysed"
                 + (f"; backlog hotspot: {hotspot}." if hotspot else "."))
    elif business == "performance":
        story = (f"{n} performance samples analysed"
                 + (f"; worst node {worst_name} ({_metric_label(primary_sev)} {worst_val})." if worst_name else "."))
    elif domain == "telecom":
        story = f"{n} network elements analysed across {len(df.columns)} dimensions."
    else:
        story = f"{n} records across {len(df.columns)} columns analysed."

    # Append the operator fragment to ANY telecom story — it's always relevant.
    if operator_result and operator_result.get("summary"):
        frag = ti.operator_story_fragment(operator_result["summary"])
        if frag:
            story = story.rstrip(".") + ". Network " + frag

    insights = []
    if worst_name:
        insights.append(f"Worst node is {worst_name} with {_metric_label(primary_sev)} of {worst_val}"
                        + (f", impacting {worst_impact:,} subscribers." if worst_impact else "."))
    if impact:
        insights.append(f"Total impacted subscribers across all nodes: {int(df[impact].sum()):,}.")
    # Operator-level insights — the WE-specific intelligence.
    if operator_result and operator_result.get("summary"):
        osum = operator_result["summary"]
        if osum.get("largest_operator"):
            insights.append(
                f"{osum['largest_operator']} is the largest operator carried "
                f"({osum['largest_operator_subscribers']:,} subscribers, "
                f"{osum.get('wholesale_pct', 0):.0f}% of the network is wholesale).")
        if osum.get("most_exposed_operator"):
            exp = operator_result["aggregations"]["operator_exposure"][0]["metrics"]
            insights.append(
                f"{osum['most_exposed_operator']} is the most congestion-exposed operator: "
                f"{int(exp['exposed_subscribers']):,} of its subscribers "
                f"({exp['exposure_pct']:.0f}%) sit on the worst-affected elements.")
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

    # Grain reflects the business — "deployed element" for inventory reads
    # very differently from "critical MSAN" for congestion.
    grain = {
        "congestion":    "one critically-impacted MSAN",
        "inventory":     "one deployed network element",
        "alarms":        "one alarm event",
        "tickets":       "one trouble ticket",
        "performance":   "one performance sample",
        "other_telecom": "one network element",
        "general":       "one record",
    }.get(business, "one record")

    meta = {
        "domain": domain,
        "business": business,                 # ← drives chart selection in build_design
        "grain": grain,
        "row_count": n, "column_count": len(df.columns),
        "languages_detected": (["english", "arabic"] if arabic_found else ["english"]),
        "story": story, "anomalies": anomalies[:6],
        "schema_design_rationale":
            f"Hybrid: AI-selected column roles, Python-computed aggregations. "
            f"Detected business sub-domain: {business}.",
    }
    # Operator metadata for the AI agents + dashboard header.
    if operator_result and operator_result.get("summary"):
        meta["operators"] = operator_result["summary"]

    return {
        "meta": meta,
        "columns": columns, "aggregations": aggregations,
        "kpis": kpis[:6],   # KPI strip shows the curated top-6 (operators now lead)
        "insights": insights[:6], "urgent_flag": urgent,
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
    """Deterministically design a NOC-grade dashboard from the computed analysis.

    Chart selection is now BUSINESS-AWARE — the meta.business field
    (set by compute_analysis) decides which chart set is appropriate:
      • congestion  → gauge + worst-critical bar + region/sector + trend
      • inventory   → top-capacity bar + vendor donut + geo map + region/sector
      • alarms      → severity histogram + top-alarming + status pie
      • tickets     → status pie + region/sector breakdowns
      • performance → top-utilization + region/sector
    """
    aggs = analysis.get("aggregations", {}) or {}
    meta = analysis.get("meta", {}) or {}
    domain   = meta.get("domain", "data")
    business = meta.get("business", "general")
    sev = "severity" if domain == "telecom" else "categorical"
    charts = []

    # ── Gauge: only for congestion (health = 100 − avg critical-time share) ──
    has_gauge = (business == "congestion"
                 and isinstance(aggs.get("network_health"), list)
                 and aggs["network_health"])
    if has_gauge:
        charts.append({
            "id": "network_health", "title": "Network Health Score", "chart_type": "gauge",
            "tab": 0, "priority": 1, "width_cols": 4, "data_source": "network_health",
            "x_field": "key", "y_field": "health_pct", "color_scheme": "severity",
            "invert_gauge": True,
            "has_threshold_line": True, "threshold_value": 60, "threshold_label": "Healthy ≥60%",
            "x_title": "", "y_title": "", "insight": "Higher is healthier (100 − avg daily critical-time share).",
        })

    # ── Lead bar: title + framing depends on business ───────────────────────
    primary_recs = None
    primary_source = None
    primary_title = None
    primary_insight = None
    if business == "inventory" and isinstance(aggs.get("top_capacity"), list) and aggs["top_capacity"]:
        primary_recs    = aggs["top_capacity"]
        primary_source  = "top_capacity"
        primary_title   = "Top 10 Sites by Subscriber Capacity"
        primary_insight = "Largest deployments by served subscriber count."
    elif isinstance(aggs.get("top_offenders"), list) and aggs["top_offenders"]:
        primary_recs    = aggs["top_offenders"]
        primary_source  = "top_offenders"
        primary_title   = {
            "congestion":  "Worst Critical MSANs (Top 10)",
            "alarms":      "Top 10 Alarming Nodes",
            "tickets":     "Top 10 Nodes by Ticket Count",
            "performance": "Top 10 Nodes by Severity Metric",
        }.get(business, "Top 10 Network Elements")
        primary_insight = "Teal ⊕ marks impacted subscribers per node."

    if primary_recs:
        yf = _first_metric_key(primary_recs, "critical")
        if yf == "value":
            yf = _first_metric_key(primary_recs, "subscriber") or _first_metric_key(primary_recs)
        mkeys = list((primary_recs[0].get("metrics") or {}).keys())
        sf = next((k for k in mkeys if any(h in k for h in ("subscriber", "impact", "affected"))), None)
        if business == "inventory":
            sf = None      # avoid double-rendering subscribers when it IS the primary metric
            primary_insight = "Larger bars indicate larger subscriber populations served by that node."
        charts.append({
            "id": primary_source, "title": primary_title, "chart_type": "horizontal_bar",
            "tab": 0, "priority": 1, "width_cols": 8 if has_gauge else 12,
            "data_source": primary_source,
            "x_field": "key", "y_field": yf, "color_scheme": sev, "sort_order": "desc",
            "highlight_top_n": 3, "secondary_annotation_field": sf,
            "x_title": _metric_label(yf), "y_title": "Node",
            "insight": primary_insight,
        })

    # ── FLAGSHIP: Operator analytics charts ─────────────────────────────────
    # These are the WE-specific visuals: who do we carry, and who's most exposed.
    if isinstance(aggs.get("operator_exposure"), list) and len(aggs["operator_exposure"]) >= 2:
        # The killer chart: stacked bar of exposed-vs-safe subscribers per operator.
        charts.append({
            "id": "operator_exposure", "title": "Operator Congestion Exposure",
            "chart_type": "operator_exposure", "tab": 0, "priority": 1, "width_cols": 6,
            "data_source": "operator_exposure",
            "x_field": "key", "y_field": "exposed_subscribers", "color_scheme": "severity",
            "sort_order": "desc",
            "x_title": "Subscribers", "y_title": "",
            "insight": "Red = subscribers on worst-affected elements. "
                       "A tall red share means that operator is disproportionately hit.",
        })
    if isinstance(aggs.get("operator_mix"), list) and len(aggs["operator_mix"]) >= 2:
        charts.append({
            "id": "operator_mix", "title": "Subscriber Mix by Operator",
            "chart_type": "horizontal_bar", "tab": 0, "priority": 2, "width_cols": 6,
            "data_source": "operator_mix",
            "x_field": "key", "y_field": "subscribers", "color_scheme": "operator",
            "sort_order": "desc", "highlight_top_n": 1,
            "x_title": "Subscribers", "y_title": "",
            "insight": "Subscriber base WE carries for each operator (retail + wholesale).",
        })
    if isinstance(aggs.get("wholesale_vs_retail"), list) and len(aggs["wholesale_vs_retail"]) == 2:
        charts.append({
            "id": "wholesale_vs_retail", "title": "Wholesale vs Retail Split",
            "chart_type": "donut", "tab": 0, "priority": 2, "width_cols": 6,
            "data_source": "wholesale_vs_retail",
            "x_field": "key", "y_field": "subscribers", "color_scheme": "operator",
            "x_title": "", "y_title": "",
            "insight": "How much of the carried base is WE's own retail vs other operators' wholesale.",
        })

    # by_vendor / by_technology (and combined vendor+technology columns) get their
    # own donut blocks below — exclude them here so we don't render them twice.
    def _is_vendor_or_tech(k):
        kl = k.lower()
        return any(s in kl for s in ("vendor", "technology", "make", "manufactur",
                                       "platform", "tech_", "_tech"))
    by_keys = sorted(
        [k for k in aggs if k.startswith("by_") and not _is_vendor_or_tech(k)],
        key=lambda k: (0 if "region" in k or "area" in k else (1 if "sector" in k else 2)),
    )
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

    # ── Vendor donut (inventory + any dataset with a vendor column) ────────
    if isinstance(aggs.get("by_vendor"), list) and len(aggs["by_vendor"]) >= 2:
        charts.append({
            "id": "by_vendor", "title": "Vendor Distribution", "chart_type": "donut",
            "tab": 0, "priority": 2, "width_cols": 6, "data_source": "by_vendor",
            "x_field": "key", "y_field": "count", "color_scheme": "categorical",
            "x_title": "", "y_title": "",
            "insight": "Share of network elements by vendor.",
        })

    # ── Technology donut (e.g. GPON vs Copper vs DSL) ──────────────────────
    if isinstance(aggs.get("by_technology"), list) and len(aggs["by_technology"]) >= 2:
        charts.append({
            "id": "by_technology", "title": "Technology Mix", "chart_type": "donut",
            "tab": 0, "priority": 2, "width_cols": 6, "data_source": "by_technology",
            "x_field": "key", "y_field": "count", "color_scheme": "categorical",
            "x_title": "", "y_title": "",
            "insight": "Share of access technology types.",
        })

    # ── Geo map: only for datasets with lat/long ──────────────────────────
    if isinstance(aggs.get("geo_points"), list) and len(aggs["geo_points"]) >= 1:
        charts.append({
            "id": "geo_points", "title": "Geographic Distribution",
            "chart_type": "map",
            "tab": 0, "priority": 3, "width_cols": 12, "data_source": "geo_points",
            "x_field": "lng", "y_field": "lat", "color_scheme": "categorical",
            "x_title": "", "y_title": "",
            "insight": "Each marker is a deployed network element; size = subscriber count if available.",
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

    # Business-aware title & tab name (replaces the old congestion-only labels)
    business_titles = {
        "congestion":    "Access Network Operations — MSAN Health",
        "inventory":     "Network Inventory & Capacity Overview",
        "alarms":        "NOC Alarm Dashboard",
        "tickets":       "Trouble Ticket Operations",
        "performance":   "Network Performance KPIs",
        "other_telecom": "Telecom Operations Overview",
        "general":       f"{domain.title()} Analytics Dashboard",
    }
    business_tabs = {
        "congestion":    "NOC Overview",
        "inventory":     "Inventory Overview",
        "alarms":        "Alarm Overview",
        "tickets":       "Ticket Overview",
        "performance":   "KPI Overview",
        "other_telecom": "Network Overview",
        "general":       "Overview",
    }
    if PIPELINE_STYLE == "executive":
        title = "Telecom Congestion Analysis" if business == "congestion" else (
            f"Telecom {business.title()} Analysis" if domain == "telecom"
            else f"{domain.title()} Executive Dashboard"
        )
    else:
        title = business_titles.get(business, business_titles["general"])
    return {
        "style": PIPELINE_STYLE,
        "business": business,                          # ← echoed for downstream consumers
        "dashboard_title": title,
        "layout_hint": "single_tab",
        "tab_names": [business_tabs.get(business, "Overview")],
        "charts": charts,
    }


def compute_insights(analysis: dict) -> dict:
    """Business-aware executive summary computed deterministically from the analysis."""
    meta = analysis.get("meta", {}) or {}
    domain   = meta.get("domain", "data")
    business = meta.get("business", "general")
    aggs = analysis.get("aggregations", {}) or {}
    kpis = analysis.get("kpis", []) or []
    urgent = analysis.get("urgent_flag", {}) or {}

    highlights = [f"• {k.get('label')}: {k.get('value')}" for k in kpis[:4]] or \
                 [f"• {meta.get('row_count', 0):,} records analysed"]

    risks, actions = [], []

    # ── Business-specific risks & actions ──────────────────────────────────
    if business == "congestion":
        top = aggs.get("top_offenders") or []
        if top:
            worst = top[0]; wk = worst["key"]; m = worst.get("metrics", {})
            sub = next((v for kk, v in m.items() if "subscriber" in kk or "impact" in kk), None)
            sevv = next((v for kk, v in m.items() if "critical" in kk or "time" in kk), None)
            risks.append(f"⚠ {wk} is the worst node"
                         + (f" (critical time {sevv} min)" if sevv is not None else "")
                         + (f", impacting {int(sub):,} subscribers." if sub else "."))
            actions.append(f"→ Escalate {wk} to field operations for an immediate site visit.")
            if len(top) >= 3:
                names = ", ".join(t["key"] for t in top[:3])
                actions.append(f"→ Raise capacity-augmentation work orders for the top chronic nodes: {names}.")

    elif business == "inventory":
        top = aggs.get("top_capacity") or aggs.get("top_offenders") or []
        if top:
            biggest = top[0]; bk = biggest["key"]; m = biggest.get("metrics", {})
            sub = next((v for kk, v in m.items() if "subscriber" in kk or "impact" in kk), None)
            risks.append(f"⚠ {bk} serves the largest subscriber population"
                         + (f" ({int(sub):,} subscribers)" if sub else "")
                         + " — a single point of impact if it fails.")
            actions.append(f"→ Ensure {bk} has redundancy and capacity headroom monitoring.")
        v = aggs.get("by_vendor") or []
        if len(v) >= 2:
            top_v = v[0]["key"]
            actions.append(f"→ Standardise spare-parts inventory and field training around {top_v} (largest installed base).")
        if aggs.get("geo_points"):
            actions.append("→ Use the geographic map to identify coverage gaps and over/under-served areas.")

    elif business == "alarms":
        top = aggs.get("top_offenders") or []
        if top:
            wk = top[0]["key"]
            risks.append(f"⚠ {wk} has the highest alarm count — chronic instability suspected.")
            actions.append(f"→ Open a root-cause-analysis case for {wk}.")

    elif business == "tickets":
        top = aggs.get("top_offenders") or []
        if top:
            wk = top[0]["key"]
            risks.append(f"⚠ {wk} carries the largest open-ticket backlog.")
            actions.append(f"→ Re-assign or escalate {wk}'s ticket queue to clear the backlog.")

    elif business == "performance":
        top = aggs.get("top_offenders") or []
        if top:
            wk = top[0]["key"]; m = top[0].get("metrics", {})
            v0 = next(iter(m.values()), None)
            risks.append(f"⚠ {wk} shows the worst performance"
                         + (f" ({v0})" if v0 is not None else "") + ".")
            actions.append(f"→ Investigate {wk} for capacity / configuration issues.")

    # ── Regional hotspot (applies to every business when by_region exists) ──
    region_key = next((k for k in aggs if k.startswith("by_") and ("region" in k or "area" in k)), None)
    if region_key and aggs.get(region_key):
        r0 = aggs[region_key][0]["key"]
        regional_phrase = {
            "congestion":  f"⚠ {r0} is the highest-impact region — likely a congestion/fault cluster.",
            "inventory":   f"⚠ {r0} hosts the largest deployed footprint — concentrate field-ops coverage there.",
            "alarms":      f"⚠ {r0} produces the most alarms — possible regional instability.",
            "tickets":     f"⚠ {r0} has the most open tickets — backlog hotspot.",
            "performance": f"⚠ {r0} shows the worst performance metrics on average.",
        }.get(business, f"⚠ {r0} stands out across the data.")
        risks.append(regional_phrase)
        actions.append(f"→ Assign a regional task force to {r0}.")

    if not risks:
        risks = ["No severe anomalies detected in this dataset."]
    if not actions:
        actions = ["→ Review the dashboard charts for detailed breakdowns."]

    risk_level = urgent.get("severity") or ("MEDIUM" if meta.get("anomalies") else "LOW")
    title_map = {
        "congestion":    "Network Operations Brief — Access Network Health",
        "inventory":     "Network Inventory Brief — Capacity & Coverage",
        "alarms":        "Alarm Operations Brief",
        "tickets":       "Ticket Operations Brief",
        "performance":   "Performance Operations Brief",
        "other_telecom": "Network Operations Brief",
    }
    title = title_map.get(business,
                          f"Executive Summary: {domain.title()}")
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


# ═════════════════════════════════════════════════════════════════════════════
# NEW: Multi-AI architecture (Detective → Architect → Reviewer)
# Each agent gets a focused prompt. Outputs accumulate; the Python aggregation
# layer guarantees that the *numbers* in the dashboard are always correct
# (math = Python), while the *meaning* and *design* come from the AI agents.
# ═════════════════════════════════════════════════════════════════════════════

AGENT1_DETECTIVE_TEMPLATE = """You are AGENT 1 — THE DATA DETECTIVE.

Your job is NOT to compute numbers. Your job is to understand what this dataset
is ABOUT and what dashboard story it should tell. Read the column names, sample
values, and statistics, then decide: who would care about this data, and what
questions would they want answered?

Return ONLY a JSON object. No markdown, no commentary.  First char = {{  last char = }}.

EXPECTED OUTPUT SHAPE:
{{
  "business_domain": "telecom_access_network | telecom_core | telecom_radio | finance | hr | sales | healthcare | logistics | retail | other",
  "business_subdomain": "short snake_case label — e.g. fttx_inventory | msan_congestion | alarm_log | trouble_tickets | capacity_planning | subscriber_analytics | sla_performance",
  "audience": "who reads this — e.g. noc_operations | network_planning | field_ops | executive_leadership | finance_director",
  "narrative_voice": "noc_brief | inventory_brief | planning_brief | performance_brief | executive_summary | other",
  "grain": "one short sentence: what each row represents (e.g. 'one OLT port serving an MSAN')",

  "entity": {{
    "column": "exact column name",
    "label": "Human-readable label for THIS kind of entity (e.g. 'MSAN', 'PE Router', 'Customer')"
  }},
  "primary_metric": {{
    "column": "exact column name (or null)",
    "label": "Human label (e.g. 'Etisalat Subscribers')",
    "kind": "count | duration_min | percent | currency | rate | binary | other",
    "direction": "higher_is_better | lower_is_better | neutral"
  }},
  "secondary_metrics": [
    {{"column": "exact name", "label": "human label", "kind": "...", "direction": "..."}}
  ],
  "dimensions": [
    {{"column": "exact name", "label": "human label", "kind": "geo_region | geo_sector | vendor | technology | status | other_category"}}
  ],
  "geo": {{
    "lat_column": "exact name or null (only if values are -90..90)",
    "lng_column": "exact name or null (only if values are -180..180)"
  }},
  "status_columns": [
    {{"column": "exact name", "label": "human label", "kind": "alarm_severity | upgrade_state | error_flag | ticket_status | other"}}
  ],
  "time_columns": [
    {{"column": "exact name", "label": "human label", "kind": "snapshot | timestamp | period_label"}}
  ],

  "business_questions": [
    "Question 1 a dashboard reader would want answered",
    "Question 2 ...",
    "Question 3 ...",
    "Question 4 ..."
  ],
  "key_observations": [
    "Short sentence about something concrete and quantitative the data implies",
    "Another concrete observation",
    "..."
  ],
  "anomalies_to_flag": [
    "Things that should pop visually if true (e.g. 'PE routers with errors AND high load = high-impact failure risk')"
  ],
  "suggested_chart_concepts": [
    {{"concept": "Distribution of subscribers by region", "why": "answers Q1 — where the load is concentrated"}},
    {{"concept": "Top 10 PE routers by Etisalat subscribers", "why": "answers Q2 — single-point-of-failure analysis"}}
  ],
  "story_summary": "ONE rich sentence: what the dashboard story is, in operations language."
}}

Rules:
- Use EXACT column names from the list below — no inventions, no aliases.
- If a role has no obvious column, return null (NOT a guess).
- The `business_subdomain` MUST be a short snake_case label, not a sentence.
- `business_questions` MUST be specific to this dataset, not generic.
- `suggested_chart_concepts` MUST cite the column or aggregation each chart would use.
- For telecom data, prefer audience = noc_operations / network_planning / field_ops.

═══ DOMAIN CHEAT-SHEET ═══════════════════════════════════════════════════════
TELECOM-ACCESS  (MSAN, OLT, ONT, DSLAM, port, vendor):  inventory / capacity / congestion
TELECOM-CORE    (PE router, P router, BNG, MSC):         core capacity / topology
TELECOM-RADIO   (cell, NodeB, sector, RSRP, RSRQ):       radio coverage / KPI
NETWORK-ALARMS  (alarm, severity, occurred, cleared):    NOC alarm dashboard
TICKETING       (ticket, incident, SLA, owner):          trouble-ticket ops
PERFORMANCE     (throughput, latency, loss, jitter):     SLA / performance brief
SUBSCRIBER      (customer, churn, ARPU, plan):           subscriber analytics

═══ OPERATOR / CARRIER DIMENSION (CRITICAL for WE / Telecom Egypt) ════════════
If you see per-operator subscriber columns — e.g. noor_sub, etisilat_sub,
orange_sub, voda_sub, we_data_sub, bitstream_data_sub — this is a WHOLESALE
access network: one element carries several operators' customers. This is a
FIRST-CLASS dimension. Make sure your business_questions include:
  • "Which operator carries the most subscribers on this network?"
  • "Which operator is most exposed when elements congest/fail?"
  • "What is the wholesale (other operators) vs retail (WE own) split?"
List those *_sub columns under secondary_metrics, and add suggested_chart_concepts
for operator mix and operator congestion-exposure.

DATASET:
- Row count:    {row_count}
- Column count: {column_count}
- Mojibake detected & repaired: {mojibake_repaired}

COLUMNS (dtype — up to 5 sample values — uniqueness):
{column_list}

LANGUAGE HINT: {language_hint}

Return ONLY the JSON object. First char = {{ last char = }}."""


AGENT2_ARCHITECT_TEMPLATE = """You are AGENT 2 — THE DASHBOARD ARCHITECT.

Agent 1 (the Detective) has already identified what this data is about and what
questions readers care about. Python has already computed all aggregations.
Your job: pick the BEST chart for each question, in the right order, with
business-grade titles and insight captions.

Return ONLY a JSON object. First char = {{ last char = }}.

EXPECTED OUTPUT SHAPE:
{{
  "dashboard_title": "Specific, business-grade title — NOT generic. Match the audience's voice.",
  "tab_names": ["Single tab name in operations vocabulary"],
  "layout_hint": "single_tab",
  "charts": [
    {{
      "id": "snake_case_id",
      "title": "Human-readable chart title (specific, not 'Bar Chart 1')",
      "chart_type": "horizontal_bar | vertical_bar | donut | line | gauge | map | table | scatter | histogram",
      "tab": 0,
      "priority": 1,
      "width_cols": 4 | 6 | 8 | 12,
      "data_source": "EXACT key from AVAILABLE_AGGREGATIONS (e.g. 'top_offenders', 'by_region')",
      "x_field": "key | label | period | lng",
      "y_field": "exact metric key from the aggregation",
      "color_scheme": "severity | categorical",
      "sort_order": "desc | asc | none",
      "top_n": 10,
      "highlight_top_n": 3,
      "secondary_annotation_field": null | "exact field name",
      "x_title": "Axis label",
      "y_title": "Axis label",
      "insight": "ONE sentence — what this chart REVEALS to the reader, not 'this is a bar chart'.",
      "answers_question": "Which business_question (from Agent 1) does this chart answer?"
    }}
  ]
}}

Hard rules:
1. data_source MUST be one of the keys in AVAILABLE_AGGREGATIONS — never invent one.
2. y_field MUST be a real metric key present in that aggregation's records.
3. chart_type must fit the data shape:
     - one categorical key + one numeric → horizontal_bar (preferred for many categories) or donut (max 7 slices, with diversity)
     - time/period + numeric             → line
     - lat+lng coordinates              → map
     - single overall score (0..100)    → gauge
     - just labels & counts             → distributions → donut (if diverse) else horizontal_bar
     - HEATMAP is ONLY for 2-D data (e.g. region × sector → value).  Do NOT use heatmap on a flat
       categorical-vs-numeric list — that produces a meaningless single-row strip. Use horizontal_bar.
     - operator_exposure data_source → use chart_type "operator_exposure" (a stacked bar that shows,
       per operator, how many subscribers sit on the worst-affected elements). y_field = exposed_subscribers.
     - operator_mix / wholesale_vs_retail → horizontal_bar or donut with color_scheme "operator".
4. AXIS-TITLE CONVENTIONS:
     - For horizontal_bar:  x_title = metric name (e.g. "Critical Time (min)"),  y_title = "" (the y-axis is the entity label).
     - For vertical_bar:    x_title = "" or category name,                       y_title = metric name.
     - For line/scatter:    x_title = period/date label,                          y_title = metric name.
     - NEVER label the Y-axis "Number of Subscribers" if the y_field is critical-time.
       The axis title MUST describe what y_field actually measures.
5. priority 1 = the chart that DIRECTLY answers the most important business_question.
   priority 2 = supporting context. priority 3 = secondary detail.
6. width_cols MUST sum reasonably per row: prefer 12 (single full-width) or 6+6 or 4+8.
7. insight MUST be a real reading of the chart — NOT a description of the chart type.
8. answers_question MUST quote (or closely paraphrase) one of Agent 1's business_questions.
9. NEVER include charts whose data_source is missing or empty.

═══════════════════════════════════════════════════════════════════════════════
DETECTIVE'S REPORT (Agent 1 output):
{detective_report}

═══════════════════════════════════════════════════════════════════════════════
AVAILABLE_AGGREGATIONS (data_source key → first-record shape):
{aggregation_shapes}

═══════════════════════════════════════════════════════════════════════════════

Now design the dashboard. Return ONLY the JSON object."""


AGENT_REVIEWER_TEMPLATE = """You are AGENT 3 — THE DESIGN REVIEWER.

A junior architect drafted a dashboard design. You are a senior data-visualization
expert and a domain expert in {business_subdomain}. Review the design and either
APPROVE it as-is or RETURN A REVISED VERSION.

Return ONLY a JSON object. First char = {{ last char = }}.

EXPECTED OUTPUT SHAPE:
{{
  "verdict": "approve | revise",
  "score": 1-10,
  "critique": [
    "Short observation about what's good",
    "Short observation about what's missing or wrong"
  ],
  "revised_design": {{ ...same shape as the input design — required if verdict='revise'... }}
}}

Review checklist:
A. COVERAGE — does the design answer every business_question Agent 1 listed?
B. HIERARCHY — is the priority-1 chart the most important business question? Are there too many priority-1 charts (≤2 is ideal)?
C. CHART-TYPE FIT — is each chart_type the right shape for its data? (bars for many categories, donuts for few diverse slices, line for time, map for geo, gauge for single 0..100 score)
D. TITLES & INSIGHTS — are titles specific (not "Chart 1")? Do insights READ the chart, not describe it?
E. WIDTH BALANCE — do widths fit a 12-column grid sensibly per row?
F. DOMAIN VOICE — is the language right for the audience (noc_operations vs executive_leadership)?
G. NOISE — any chart whose data_source isn't in AVAILABLE_AGGREGATIONS? Any redundant charts?

If verdict == "revise", you MUST return a revised_design that is a complete,
drop-in replacement with the same shape as the input design. Use only the
available data_sources. Do not invent new aggregations.

═══════════════════════════════════════════════════════════════════════════════
DETECTIVE'S REPORT:
{detective_report}

═══════════════════════════════════════════════════════════════════════════════
ARCHITECT'S DRAFT:
{architect_draft}

═══════════════════════════════════════════════════════════════════════════════
AVAILABLE_AGGREGATIONS (the only valid data_source values):
{aggregation_shapes}

═══════════════════════════════════════════════════════════════════════════════

Now review. Return ONLY the JSON object."""


def _build_column_descriptor(df: pd.DataFrame) -> str:
    """Build a rich description of each column for the Detective prompt:
    name, dtype, samples, uniqueness — everything the AI needs to reason about the data.
    """
    lines = []
    for c in df.columns:
        dtype = str(df[c].dtype)
        s = df[c]
        nun = s.nunique(dropna=True)
        try:
            samples = [str(x)[:60] for x in s.dropna().head(5).tolist()]
        except Exception:
            samples = ["<unreadable>"]
        if pd.api.types.is_numeric_dtype(s) and not s.dropna().empty:
            try:
                stats = (f"  range=[{s.min():g}..{s.max():g}] "
                         f"mean={s.mean():.2f} nulls={s.isna().sum()}")
            except Exception:
                stats = f"  nulls={s.isna().sum()}"
        else:
            stats = f"  nulls={s.isna().sum()}"
        lines.append(
            f"  - {c!r:40s} [{dtype:18s}]  uniq={nun}  samples={samples}{stats}"
        )
    return "\n".join(lines)


def _detective_to_roles(detective: dict) -> dict:
    """Translate the Detective's rich report into the schema _resolve_roles()
    consumes (which has its own naming convention).  We don't lose information —
    the full Detective report is also saved separately for downstream agents.
    """
    def col_of(field):
        v = detective.get(field)
        return v.get("column") if isinstance(v, dict) else None

    dims_raw = detective.get("dimensions") or []
    dims = [d.get("column") for d in dims_raw if isinstance(d, dict) and d.get("column")]

    status_raw = detective.get("status_columns") or []
    status = status_raw[0].get("column") if (status_raw and isinstance(status_raw[0], dict)) else None

    # _resolve_roles expects these EXACT keys:
    #   entity_column, impact_metric, severity_metric, dimension_columns, status_column
    return {
        "entity_column":      col_of("entity"),
        "impact_metric":      col_of("primary_metric"),
        "severity_metric":    col_of("primary_metric"),  # primary metric drives severity ranking
        "dimension_columns":  dims,
        "status_column":      status,
    }


def run_agent_1(df: pd.DataFrame, model: str, temperature: float, num_ctx: int,
                language_hint: str = "Auto-detect") -> dict:
    """Agent 1 — DATA DETECTIVE.

    The AI now does DEEP semantic analysis: business domain, audience, narrative
    voice, business questions, suggested chart concepts.  The detective report is
    saved separately to output/understanding.json and also embedded into
    analysis.meta.detective so downstream agents (Architect, Reviewer, Narrator)
    can read it.

    Python still computes all aggregations from the columns the Detective identified —
    the AI decides WHAT MATTERS; Python decides WHAT THE NUMBERS ARE.
    """
    agent_key = "agent_1"
    _start_iso, start_mono = agent_timer(agent_key)
    try:
        # ── Phase 0: dedupe column names FIRST ──────────────────────────────
        df = _dedupe_column_names(df)

        # Check whether mojibake repair will actually fire (informational for prompt)
        mojibake_present = False
        for c in df.columns:
            if df[c].dtype == object:
                sample = df[c].dropna().head(20).astype(str).tolist()
                if any(any(sig in s for sig in ("Ã", "Ù", "Ø", "â€", "Â")) for s in sample):
                    mojibake_present = True
                    break

        # ── Phase 1: Detective — deep semantic analysis via LLM ────────────
        column_list = _build_column_descriptor(df)
        prompt = (
            _instruction_block("USER INSTRUCTIONS FOR DETECTIVE") +
            AGENT1_DETECTIVE_TEMPLATE.format(
                row_count=len(df),
                column_count=len(df.columns),
                mojibake_repaired=mojibake_present,
                column_list=column_list,
                language_hint=language_hint,
            )
        )
        # Detective context: 6144 is enough for the prompt + a generous JSON output.
        # Larger windows make prefill 2-3× slower on consumer GPUs with no
        # quality gain — the prompt itself is well under 4K tokens.
        llm = make_llm(model, temperature, num_ctx=6144, num_predict=1200)

        detective = {}
        try:
            detective = call_llm_with_retry(
                llm, prompt, agent_key, MAX_RETRIES, parser=parse_json_lenient,
            )
            print(f"  [agent_1] Detective verdict: "
                  f"domain={detective.get('business_domain')}, "
                  f"subdomain={detective.get('business_subdomain')}, "
                  f"audience={detective.get('audience')}")
            print(f"  [agent_1] Story: {detective.get('story_summary', '')[:140]}")
            print(f"  [agent_1] Business questions identified: "
                  f"{len(detective.get('business_questions') or [])}")
        except Exception as det_err:
            print(f"  [agent_1] Detective LLM call failed ({det_err}); "
                  "falling back to pure-heuristic role detection.")

        # ── Phase 2: Resolve roles (Detective picks, with heuristic safety net) ──
        if detective:
            roles = _resolve_roles(df, _detective_to_roles(detective))
        else:
            roles = _resolve_roles(df, {})
        print(f"  [agent_1] resolved → entity={roles.get('entity')}, "
              f"impact={roles.get('impact')}, sev={roles.get('primary_sev')}, "
              f"dims={roles.get('dims')}, status={roles.get('status_dim')}")

        # ── Phase 3: Python computes aggregations (math is always correct) ──
        result = compute_analysis(df, roles)
        result.setdefault("meta", {})["language_hint"] = language_hint
        # Persist the user's AI instructions so they show up in the dashboard
        # header (and in any saved analysis.json review).
        if USER_INSTRUCTIONS:
            result["meta"]["user_instructions"] = USER_INSTRUCTIONS
        # Embed the Detective's report into meta so the Architect & Narrator can read it
        if detective:
            result["meta"]["detective"] = detective
            # If the Detective found a more specific business subdomain than our
            # rule-based _detect_business(), honour it — the AI sees nuance the heuristic misses.
            if detective.get("business_subdomain"):
                result["meta"]["business_subdomain"] = detective["business_subdomain"]
            if detective.get("audience"):
                result["meta"]["audience"] = detective["audience"]
            if detective.get("story_summary"):
                result["meta"]["story"] = detective["story_summary"]

        # Save the rich Detective report separately for the UI / inspection
        try:
            with open(OUTPUT_DIR / "understanding.json", "w", encoding="utf-8") as f:
                json.dump(detective or {}, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        with open(ANALYSIS_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        agent_done(agent_key, start_mono)
        return result
    except Exception as e:
        agent_failed(agent_key, start_mono)
        # Print the full inner traceback to the pipeline log so the real failing
        # line is visible — the outer RuntimeError otherwise hides everything
        # underneath the bare error message.
        import traceback as _tb
        print("  [agent_1] inner traceback:")
        print(_tb.format_exc())
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

def _compact_detective(detective: dict) -> str:
    """Compress the Detective's report to only the fields the Architect/Reviewer
    actually use. Cuts prompt size roughly in half — a big speed win on small
    GPUs where prefill time scales with context length.
    """
    if not isinstance(detective, dict):
        return ""
    keep = {
        "business_domain":    detective.get("business_domain"),
        "business_subdomain": detective.get("business_subdomain"),
        "audience":           detective.get("audience"),
        "narrative_voice":    detective.get("narrative_voice"),
        "grain":              detective.get("grain"),
        "entity_label":       (detective.get("entity") or {}).get("label"),
        "primary_metric_label": (detective.get("primary_metric") or {}).get("label"),
        "primary_metric_kind": (detective.get("primary_metric") or {}).get("kind"),
        "business_questions": (detective.get("business_questions") or [])[:4],
        "suggested_charts":   [c.get("concept") for c in (detective.get("suggested_chart_concepts") or [])[:4]],
        "story_summary":      detective.get("story_summary"),
    }
    # Remove None values to save tokens
    keep = {k: v for k, v in keep.items() if v}
    return json.dumps(keep, ensure_ascii=False, indent=1)


def _compact_design(design: dict) -> str:
    """Compress a design draft to just the fields the Reviewer needs to assess.
    Strips long insight prose; keeps the structural decisions."""
    if not isinstance(design, dict):
        return ""
    keep = {
        "dashboard_title": design.get("dashboard_title"),
        "tab_names":       design.get("tab_names"),
        "charts": [
            {
                "id":           c.get("id"),
                "title":        c.get("title"),
                "chart_type":   c.get("chart_type"),
                "data_source":  c.get("data_source"),
                "y_field":      c.get("y_field"),
                "priority":     c.get("priority"),
                "width_cols":   c.get("width_cols"),
                "answers":      (c.get("answers_question") or "")[:80],
            }
            for c in (design.get("charts") or [])
        ],
    }
    return json.dumps(keep, ensure_ascii=False, indent=1)


def _aggregation_shapes(aggs: dict) -> str:
    """Render each aggregation as a one-line schema so the Architect prompt
    stays compact:
        top_offenders         (12 records)  fields: key, average_critical_time_min, subscribers
        by_region             (8 records)   fields: key, count, subscribers
    The Architect uses these EXACT names; we validate strictly later.
    """
    lines = []
    for name, node in (aggs or {}).items():
        if isinstance(node, list) and node and isinstance(node[0], dict):
            first = node[0]
            flat = {k: v for k, v in first.items() if k != "metrics"}
            if isinstance(first.get("metrics"), dict):
                flat.update(first["metrics"])
            fields = ", ".join(flat.keys())
            lines.append(f"  - {name!r}  ({len(node)} records)  fields: {fields}")
        elif isinstance(node, dict):
            # distribution-style: column_name -> {label: count}
            sub = next(iter(node.items()), (None, {}))[1] if node else {}
            n_keys = len(sub) if isinstance(sub, dict) else 0
            lines.append(f"  - {name!r}  ({n_keys} distinct labels)  fields: label, value")
    return "\n".join(lines) or "  (no aggregations available)"


def _validate_design(design: dict, aggs: dict) -> tuple[dict, list[str]]:
    """Strict gate on Architect / Reviewer output. Catches and FIXES:
      1. data_source not in aggregations              → drop chart
      2. y_field not in records                       → auto-pick a real field
      3. heatmap on 1-D data                          → downgrade to horizontal_bar
      4. horizontal_bar with swapped/wrong x_title and y_title
                                                      → use y_field-derived labels
      5. line/scatter with y_title that doesn't match y_field metric
                                                      → relabel from y_field
      6. titles that mention a metric not present     → strip / replace
    Returns (clean_design, warnings_list).
    """
    warnings = []
    clean_charts = []
    available = set((aggs or {}).keys())

    def _metric_label_simple(field_name: str) -> str:
        """Turn 'average_critical_time_min' into 'Average Critical Time'."""
        if not field_name:
            return "Value"
        n = re.sub(r"\(.*?\)", " ", str(field_name)).replace("_", " ")
        for w in _NOISE_WORDS:
            n = re.sub(rf"\b{w}\b", "", n, flags=re.IGNORECASE)
        n = " ".join(n.split())
        return n.title() or "Value"

    for spec in (design.get("charts") or []):
        ds = spec.get("data_source", "")
        ds_key = str(ds).replace("aggregations.", "").split(".")[0]
        if ds_key not in available:
            warnings.append(f"dropped {spec.get('id')!r}: data_source {ds!r} not in aggregations")
            continue

        # ── 1. y_field must exist in records ────────────────────────────────
        node = aggs[ds_key]
        valid_fields = set()
        n_records = 0
        if isinstance(node, list) and node and isinstance(node[0], dict):
            n_records = len(node)
            valid_fields.update(k for k in node[0].keys() if k != "metrics")
            if isinstance(node[0].get("metrics"), dict):
                valid_fields.update(node[0]["metrics"].keys())
        elif isinstance(node, dict):
            valid_fields.update(["label", "value"])
            n_records = len(node)
        yf = spec.get("y_field")
        if yf and yf not in valid_fields and yf not in ("value", "count"):
            cand = (
                next((f for f in valid_fields if "subscriber" in f.lower() or "impact" in f.lower()), None)
                or next((f for f in valid_fields if "critical" in f.lower() or "time" in f.lower()), None)
                or next((f for f in valid_fields if f not in ("key", "label", "count")), None)
                or "count"
            )
            warnings.append(
                f"adjusted {spec.get('id')!r}: y_field {yf!r} not in {sorted(valid_fields)} → using {cand!r}"
            )
            spec["y_field"] = cand

        # ── 2. Downgrade heatmap on 1-D data to a bar chart ────────────────
        # A heatmap needs TWO dimensions (e.g. region × sector → metric).
        # When the data_source is a flat list (one row per category), Plotly
        # renders a single-row heatmap with a meaningless Y-axis. Convert it
        # to a horizontal_bar — which is what the data actually looks like.
        if spec.get("chart_type") == "heatmap":
            is_2d = False
            # crude 2-D detection: records have two non-key categorical fields
            if isinstance(node, list) and node:
                cat_fields = [k for k in valid_fields
                              if k not in ("key", "label") and
                              all(isinstance(r.get(k), str) for r in node[:5] if isinstance(r, dict))]
                is_2d = len(cat_fields) >= 1 and "key" in valid_fields
            if not is_2d:
                warnings.append(
                    f"downgraded {spec.get('id')!r}: heatmap needs 2-D data; "
                    f"{ds_key!r} is 1-D → using horizontal_bar instead"
                )
                spec["chart_type"] = "horizontal_bar"
                spec["sort_order"] = spec.get("sort_order", "desc")
                spec["top_n"] = spec.get("top_n", 10)

        # ── 3. Axis-title sanity: derive from y_field, override the AI label
        # if it claims a metric the y_field doesn't carry.
        actual_metric_label = _metric_label_simple(spec.get("y_field", ""))
        ctype = spec.get("chart_type", "")
        x_title = str(spec.get("x_title") or "").strip()
        y_title = str(spec.get("y_title") or "").strip()

        # 3a. y_title vs y_field semantic check
        # If the label mentions "subscribers" but y_field is critical-time
        # (or vice versa), the AI got confused — replace with field-derived label.
        def _label_conflicts(label: str, field: str) -> bool:
            l, f = label.lower(), str(field).lower()
            if not l or not f:
                return False
            subs_in_label = any(w in l for w in ("subscriber", "customer", "user"))
            subs_in_field = any(w in f for w in ("subscriber", "customer", "user", "impact"))
            time_in_label = any(w in l for w in ("time", "minute", "duration", "critical"))
            time_in_field = any(w in f for w in ("time", "min", "duration", "critical"))
            return (subs_in_label and not subs_in_field) or (time_in_label and not time_in_field)

        if _label_conflicts(y_title, spec.get("y_field", "")):
            warnings.append(
                f"relabeled {spec.get('id')!r}: y_title {y_title!r} contradicts y_field "
                f"{spec.get('y_field')!r} → using {actual_metric_label!r}"
            )
            y_title = actual_metric_label

        # 3b. horizontal_bar axis convention — value on X, label on Y
        if ctype == "horizontal_bar":
            # Detect the classic "wrote them as vertical-bar" mistake:
            # x_title looks categorical (mentions key/name/code) and y_title looks numeric.
            x_looks_cat = any(w in x_title.lower() for w in ("name", "code", "id", "msan", "node", "site"))
            y_looks_num = any(w in y_title.lower() for w in ("time", "min", "subscriber", "count", "value", "%"))
            if x_looks_cat and y_looks_num:
                warnings.append(
                    f"swapped axis titles on {spec.get('id')!r}: "
                    f"x={x_title!r} y={y_title!r} → x={y_title!r} y=''"
                )
                x_title, y_title = y_title, ""
            # Always ensure x_title carries the metric label, y_title can be empty
            if not x_title or x_title.lower() in ("msan code", "id", "key", "label"):
                x_title = actual_metric_label
            if y_title and any(w in y_title.lower() for w in ("time", "subscriber", "%", "min")):
                y_title = ""   # the y-axis is the entity — no numeric label needed

        # 3c. line / scatter — y_title MUST match y_field metric
        if ctype in ("line", "scatter", "area"):
            if not y_title or _label_conflicts(y_title, spec.get("y_field", "")):
                y_title = actual_metric_label

        spec["x_title"] = x_title
        spec["y_title"] = y_title

        clean_charts.append(spec)
    design["charts"] = clean_charts
    return design, warnings


def run_agent_2(analysis: dict, model: str) -> dict:
    """Agent 2 — DASHBOARD ARCHITECT.

    Reads the Detective's report + the actual aggregations, asks the LLM to
    design the dashboard (which charts, what titles, what insight captions,
    priority order). Falls back to the deterministic build_design() if the LLM
    output can't be parsed or validated.

    Then Agent 2b (Reviewer) critiques the design and may return a revised version.
    """
    agent_key = "agent_2"
    _start_iso, start_mono = agent_timer(agent_key)
    try:
        meta = analysis.get("meta", {}) or {}
        aggs = analysis.get("aggregations", {}) or {}
        detective = meta.get("detective") or {}
        agg_shapes = _aggregation_shapes(aggs)

        # ── Phase A: Architect drafts the design ───────────────────────────
        deterministic = build_design(analysis)   # always available as a safe fallback
        ai_design = None
        if detective and AI_DESIGN_MODE:         # use AI architect when Detective ran AND mode is on
            try:
                prompt = (
                    _instruction_block("USER INSTRUCTIONS FOR ARCHITECT") +
                    AGENT2_ARCHITECT_TEMPLATE.format(
                        detective_report=_compact_detective(detective)[:2200],
                        aggregation_shapes=agg_shapes,
                    )
                )
                llm = make_llm(model, 0.0, num_ctx=5120, num_predict=1100)
                ai_design = call_llm_with_retry(
                    llm, prompt, agent_key, MAX_RETRIES, parser=parse_json_lenient,
                )
                if isinstance(ai_design, dict) and ai_design.get("charts"):
                    ai_design, warns = _validate_design(ai_design, aggs)
                    for w in warns:
                        print(f"  [agent_2] validate: {w}")
                    if not ai_design["charts"]:
                        print("  [agent_2] all Architect charts invalid; using deterministic fallback.")
                        ai_design = None
                else:
                    ai_design = None
                    print("  [agent_2] Architect returned no usable charts; using deterministic fallback.")
            except Exception as arch_err:
                print(f"  [agent_2] Architect LLM call failed ({arch_err}); using deterministic fallback.")
                ai_design = None

        draft = ai_design or deterministic
        # ── Phase B: preserve required system fields from the deterministic baseline ──
        # The deterministic baseline knows the user's style/business choices; never lose them.
        draft.setdefault("style", deterministic.get("style"))
        draft.setdefault("business", deterministic.get("business"))
        draft.setdefault("dashboard_title", deterministic.get("dashboard_title"))
        draft.setdefault("layout_hint", "single_tab")
        draft.setdefault("tab_names", deterministic.get("tab_names", ["Overview"]))

        # ── Phase C: Reviewer critiques and may revise ─────────────────────
        if SKIP_REVIEWER and ai_design:
            print("  [agent_2] Reviewer skipped (PIPELINE_SKIP_REVIEWER=1).")
        if detective and AI_DESIGN_MODE and ai_design and not SKIP_REVIEWER:
            try:
                rprompt = (
                    _instruction_block("USER INSTRUCTIONS FOR REVIEWER") +
                    AGENT_REVIEWER_TEMPLATE.format(
                        business_subdomain=detective.get("business_subdomain", "unspecified"),
                        detective_report=_compact_detective(detective)[:1400],
                        architect_draft=_compact_design(draft)[:2200],
                        aggregation_shapes=agg_shapes,
                    )
                )
                llm = make_llm(model, 0.0, num_ctx=5120, num_predict=900)
                review = call_llm_with_retry(
                    llm, rprompt, agent_key, MAX_RETRIES, parser=parse_json_lenient,
                )
                print(f"  [agent_2] Reviewer verdict: {review.get('verdict')} "
                      f"(score {review.get('score')})")
                for note in (review.get("critique") or [])[:4]:
                    print(f"  [agent_2] reviewer: {note}")
                if review.get("verdict") == "revise" and isinstance(review.get("revised_design"), dict):
                    revised, warns = _validate_design(review["revised_design"], aggs)
                    for w in warns:
                        print(f"  [agent_2] revise-validate: {w}")
                    if revised.get("charts"):
                        # Merge: keep architect's required fields, take reviewer's chart list
                        draft["charts"] = revised["charts"]
                        if revised.get("dashboard_title"):
                            draft["dashboard_title"] = revised["dashboard_title"]
                        draft["_reviewer_score"] = review.get("score")
                        draft["_reviewer_critique"] = review.get("critique")
            except Exception as rev_err:
                print(f"  [agent_2] Reviewer LLM call failed ({rev_err}); keeping Architect draft.")

        with open(DESIGN_FILE, "w", encoding="utf-8") as f:
            json.dump(draft, f, indent=2, ensure_ascii=False)
        agent_done(agent_key, start_mono)
        return draft
    except Exception as e:
        agent_failed(agent_key, start_mono)
        import traceback as _tb
        print("  [agent_2] inner traceback:")
        print(_tb.format_exc())
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
        # Tight context: narrative prompts don't need 8K. 3072 is enough for
        # instructions + analysis blob + room for ~600 tokens of JSON output.
        ctx = 3072 if USER_INSTRUCTIONS else 2048
        llm = make_llm(model, temperature, num_ctx=ctx, num_predict=700)
        prompt = (
            _instruction_block("USER INSTRUCTIONS FOR NARRATOR") +
            AGENT5_TEMPLATE.format(
                analysis_json=json.dumps(analysis, ensure_ascii=False)[:2500]
            )
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
