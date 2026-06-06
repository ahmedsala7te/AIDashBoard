"""
streamlit_app.py — Universal AI Analytics Dashboard Control Panel
Port: 8501
"""
import os
import sys
import json
import time
import subprocess
import threading
import zipfile
import io
import traceback
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "exports").mkdir(exist_ok=True)
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
SETTINGS_FILE = ROOT / "settings.json"
STATUS_FILE = OUTPUT_DIR / "pipeline_status.json"
ANALYSIS_FILE = OUTPUT_DIR / "analysis.json"
DESIGN_FILE = OUTPUT_DIR / "design.json"
INSIGHTS_FILE = OUTPUT_DIR / "insights.json"
FIGURES_FILE = OUTPUT_DIR / "figures_code.py"
HISTORY_FILE = OUTPUT_DIR / "run_history.json"
PID_FILE = OUTPUT_DIR / "pipeline.pid"

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WE Network Analytics",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── WE Brand CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', 'DejaVu Sans', sans-serif !important;
}
.main .block-container {
    background-color: #FFFFFF !important;
    padding-top: 1.2rem;
    max-width: 1400px;
}
body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: #F7F5FC !important;
}
[data-testid="stHeader"] { color: #1A1A2E !important; }

/* ── Force readable text everywhere in main area ── */
section.main p,
section.main span,
section.main div,
section.main label,
section.main h1, section.main h2, section.main h3,
section.main h4, section.main h5, section.main h6,
section.main li,
section.main strong,
.main .stMarkdown,
.main .stMarkdown * {
    color: #1A1A2E !important;
}

/* Markdown emphasis */
section.main em { color: #5B2083 !important; }

/* ── Input fields: white bg, dark text ── */
section.main [data-baseweb="input"] input,
section.main [data-baseweb="textarea"] textarea,
section.main [data-baseweb="select"] > div,
section.main [data-baseweb="select"] input,
section.main .stTextInput input,
section.main .stNumberInput input,
section.main .stTextArea textarea,
section.main .stSelectbox > div > div,
section.main .stDateInput input {
    background-color: #FFFFFF !important;
    color: #1A1A2E !important;
    border: 1.5px solid #E8D8F8 !important;
    border-radius: 8px !important;
}
section.main [data-baseweb="input"] input:focus,
section.main [data-baseweb="textarea"] textarea:focus,
section.main [data-baseweb="select"] > div:focus-within,
section.main .stTextInput input:focus,
section.main .stNumberInput input:focus,
section.main .stTextArea textarea:focus,
section.main .stDateInput input:focus {
    border-color: #5B2083 !important;
    box-shadow: 0 0 0 3px rgba(91,32,131,0.12) !important;
}

/* Input field labels — purple, bold */
section.main .stTextInput label,
section.main .stNumberInput label,
section.main .stSelectbox label,
section.main .stTextArea label,
section.main .stDateInput label,
section.main .stMultiSelect label,
section.main .stCheckbox label,
section.main .stRadio label,
section.main .stSlider label,
section.main .stFileUploader label {
    color: #5B2083 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}

/* Helper / caption text below inputs */
section.main [data-testid="stCaptionContainer"],
section.main small,
section.main .stCaption {
    color: #6B7280 !important;
}

/* Selectbox dropdown menu items */
[data-baseweb="popover"] li,
[data-baseweb="popover"] div {
    color: #1A1A2E !important;
}
[data-baseweb="popover"] li:hover {
    background-color: #F3EDF9 !important;
}

/* Number input +/- buttons */
section.main .stNumberInput button {
    background-color: #F3EDF9 !important;
    color: #5B2083 !important;
    border: 1px solid #E8D8F8 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #5B2083 0%, #3D1257 100%) !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * {
    color: white !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: white !important;
}
[data-testid="stSidebarNav"] a {
    color: rgba(255,255,255,0.85) !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    padding: 6px 12px !important;
}
[data-testid="stSidebarNav"] a:hover {
    color: white !important;
    background: rgba(255,255,255,0.15) !important;
}
/* Sidebar selectbox / input backgrounds */
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] input {
    background-color: rgba(255,255,255,0.12) !important;
    border-color: rgba(255,255,255,0.25) !important;
    color: white !important;
}
/* Sidebar divider */
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.2) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #5B2083, #7B3BAF) !important;
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 8px 20px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(91,32,131,0.3) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #7B3BAF, #5B2083) !important;
    box-shadow: 0 4px 16px rgba(91,32,131,0.45) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}
/* Danger / stop button — keep red */
.stButton > button[data-stop="true"] {
    background: linear-gradient(135deg, #DC2626, #B91C1C) !important;
    box-shadow: 0 2px 8px rgba(220,38,38,0.3) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #5B2083 !important;
    border-radius: 12px !important;
    padding: 16px !important;
    background: #F3EDF9 !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #7B3BAF !important;
    background: #EDE0F6 !important;
}
/* Make uploader's inner section readable */
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section * {
    background-color: transparent !important;
    color: #1A1A2E !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small,
[data-testid="stFileUploaderDropzoneInstructions"] div {
    color: #1A1A2E !important;
    font-weight: 500 !important;
}
/* Inner "Browse files" button inside uploader */
[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #5B2083, #7B3BAF) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
[data-testid="stFileUploader"] button:hover {
    background: linear-gradient(135deg, #7B3BAF, #5B2083) !important;
}

/* ── Section headers ── */
.section-header {
    font-size: 18px;
    font-weight: 800;
    color: #5B2083;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 2px solid #C7A8E8;
    letter-spacing: 0.2px;
}

/* ── Status badges ── */
.badge-done {
    background: #16A34A; color: white; padding: 3px 12px;
    border-radius: 20px; font-size: 12px; font-weight: 600;
}
.badge-running {
    background: #5B2083; color: white; padding: 3px 12px;
    border-radius: 20px; font-size: 12px; font-weight: 600;
    animation: we-pulse 1.5s infinite;
}
.badge-failed {
    background: #DC2626; color: white; padding: 3px 12px;
    border-radius: 20px; font-size: 12px; font-weight: 600;
}
.badge-idle {
    background: #6B7280; color: white; padding: 3px 12px;
    border-radius: 20px; font-size: 12px; font-weight: 600;
}
@keyframes we-pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(91,32,131,0.4); }
    50%       { opacity: 0.85; box-shadow: 0 0 0 6px rgba(91,32,131,0); }
}

/* ── Risk badges ── */
.risk-critical { background: #DC2626; color: white; padding: 3px 12px; border-radius: 20px; font-size: 12px; font-weight: 700; }
.risk-high     { background: #EA580C; color: white; padding: 3px 12px; border-radius: 20px; font-size: 12px; font-weight: 700; }
.risk-medium   { background: #D97706; color: white; padding: 3px 12px; border-radius: 20px; font-size: 12px; font-weight: 700; }
.risk-low      { background: #16A34A; color: white; padding: 3px 12px; border-radius: 20px; font-size: 12px; font-weight: 700; }

/* ── KPI cards ── */
.kpi-card {
    background: white;
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 2px 12px rgba(91,32,131,0.09);
    margin-bottom: 12px;
    border-top: 3px solid #5B2083;
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}
.kpi-card:hover {
    box-shadow: 0 6px 24px rgba(91,32,131,0.16);
    transform: translateY(-2px);
}
.kpi-value {
    font-size: 28px;
    font-weight: 800;
    color: #1A1A2E;
}
.kpi-label {
    font-size: 11px;
    font-weight: 600;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
}
.kpi-icon {
    font-size: 26px;
    float: right;
    background: rgba(91,32,131,0.08);
    border-radius: 10px;
    padding: 6px 8px;
    line-height: 1;
}

/* ── Agent cards ── */
.agent-card {
    background: white;
    border-radius: 12px;
    padding: 14px 18px;
    box-shadow: 0 2px 8px rgba(91,32,131,0.07);
    margin-bottom: 8px;
    transition: box-shadow 0.15s ease;
}
.agent-card-done    { border-left: 4px solid #16A34A; }
.agent-card-running { border-left: 4px solid #5B2083; }
.agent-card-failed  { border-left: 4px solid #DC2626; }
.agent-card-idle    { border-left: 4px solid #C7A8E8; }

/* ── Metric boxes (st.metric) ── */
[data-testid="metric-container"] {
    background: white;
    border-radius: 12px;
    padding: 14px 18px;
    box-shadow: 0 2px 10px rgba(91,32,131,0.08);
    border-top: 3px solid #5B2083;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #F7F5FC; }
::-webkit-scrollbar-thumb { background: #C7A8E8; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #5B2083; }

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid #E8D8F8 !important;
    border-radius: 10px !important;
    background: white !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] details,
[data-testid="stExpander"] summary,
[data-testid="stExpander"] > div,
[data-testid="stExpander"] > div > div {
    background: white !important;
}
[data-testid="stExpander"] summary {
    font-weight: 700 !important;
    color: #5B2083 !important;
    background: #F3EDF9 !important;
    padding: 10px 14px !important;
}
[data-testid="stExpander"] summary:hover {
    background: #EDE0F6 !important;
}
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary svg {
    color: #5B2083 !important;
    fill: #5B2083 !important;
}

/* ── Checkbox / Radio ── */
section.main .stCheckbox > label,
section.main .stRadio > label {
    color: #1A1A2E !important;
}
section.main .stCheckbox [data-baseweb="checkbox"] > div:first-child,
section.main .stRadio [data-baseweb="radio"] > div:first-child {
    border-color: #5B2083 !important;
}
section.main .stCheckbox [data-baseweb="checkbox"][aria-checked="true"] > div:first-child,
section.main .stRadio [data-baseweb="radio"][aria-checked="true"] > div:first-child {
    background-color: #5B2083 !important;
}

/* ── Tabs (Streamlit st.tabs) ── */
section.main [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 2px solid #E8D8F8 !important;
}
section.main [data-baseweb="tab"] {
    color: #6B7280 !important;
    font-weight: 600 !important;
}
section.main [data-baseweb="tab"][aria-selected="true"] {
    color: #5B2083 !important;
    border-bottom-color: #5B2083 !important;
}

/* ── Code / pre blocks ── */
section.main pre, section.main code {
    background-color: #F7F5FC !important;
    color: #1A1A2E !important;
    border: 1px solid #E8D8F8 !important;
    border-radius: 6px !important;
}

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #5B2083, #7B3BAF) !important;
}

/* ── Info / Warning / Error / Success boxes ── */
section.main [data-testid="stAlert"],
section.main [data-testid="stNotification"],
section.main [role="alert"] {
    border-radius: 10px !important;
    border-left-width: 4px !important;
    font-weight: 500 !important;
}
section.main [data-testid="stAlert"] *,
section.main [data-testid="stNotification"] *,
section.main [role="alert"] * {
    color: inherit !important;
}
/* Info — purple */
section.main [data-baseweb="notification"][kind="info"],
section.main [data-testid="stNotificationContentInfo"] {
    background-color: #F3EDF9 !important;
    border-left-color: #5B2083 !important;
    color: #3D1257 !important;
}
/* Success — green */
section.main [data-baseweb="notification"][kind="positive"],
section.main [data-testid="stNotificationContentSuccess"] {
    background-color: #DCFCE7 !important;
    border-left-color: #16A34A !important;
    color: #064E2D !important;
}
/* Warning — amber */
section.main [data-baseweb="notification"][kind="warning"],
section.main [data-testid="stNotificationContentWarning"] {
    background-color: #FEF3C7 !important;
    border-left-color: #D97706 !important;
    color: #78350F !important;
}
/* Error — red */
section.main [data-baseweb="notification"][kind="negative"],
section.main [data-testid="stNotificationContentError"] {
    background-color: #FEE2E2 !important;
    border-left-color: #DC2626 !important;
    color: #7F1D1D !important;
}

/* ── Section dividers (st.markdown("---")) ── */
section.main hr {
    border-color: #E8D8F8 !important;
    margin: 16px 0 !important;
}

/* ── KPI / metric labels white-on-purple readability ── */
section.main .kpi-card *,
section.main .agent-card * {
    color: #1A1A2E !important;
}
section.main .kpi-card .kpi-label,
section.main .agent-card .kpi-label {
    color: #6B7280 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Settings ─────────────────────────────────────────────────────────────────
def load_settings() -> dict:
    if SETTINGS_FILE.exists():
        with open(SETTINGS_FILE) as f:
            return json.load(f)
    return {
        "ollama": {"host": "http://localhost:11434", "default_model": "qwen2.5-coder:32b"},
        "pipeline": {"default_output_folder": "./output", "auto_open_dashboard": True, "max_retries": 1, "fast_mode": True},
        "agent_defaults": {"agent_1_context": 16384, "agent_1_temperature": 0.0, "agent_5_temperature": 0.3, "language_hint": "Auto-detect"},
        "arabic": {"enable_rtl": True, "font": "DejaVu Sans", "auto_fix_mojibake": True},
        "export": {"png_dpi": 150, "pdf_page_size": "A4", "include_insights_in_pdf": True},
    }

def save_settings(s: dict):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(s, f, indent=2, ensure_ascii=False)

# ─── Session State Init ───────────────────────────────────────────────────────
def init_session():
    defaults = {
        "uploaded_files": [],
        "pipeline_status": {
            k: {"status": "idle", "start": None, "end": None, "duration_sec": None}
            for k in ["agent_1", "agent_2", "agent_3", "agent_4", "agent_5"]
        },
        "pipeline_start_time": None,
        "pipeline_running": False,
        "pipeline_proc": None,
        "last_analysis": None,
        "last_design": None,
        "last_insights": None,
        "run_history": [],
        "settings": load_settings(),
        "current_page": "🏠 Home",
        "current_files_df": None,
        "selected_file_idx": 0,
        "data_explorer_page": 0,
        "dash_proc": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ─── Helpers ──────────────────────────────────────────────────────────────────
def read_status() -> dict:
    if STATUS_FILE.exists():
        try:
            with open(STATUS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return st.session_state.pipeline_status

def read_json(path: Path):
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def status_badge(status: str) -> str:
    icons = {"done": "✅", "running": "⏳", "failed": "❌", "idle": "⬜"}
    classes = {"done": "badge-done", "running": "badge-running", "failed": "badge-failed", "idle": "badge-idle"}
    icon = icons.get(status, "⬜")
    cls = classes.get(status, "badge-idle")
    return f'<span class="{cls}">{icon} {status.upper()}</span>'

def risk_badge(level: str) -> str:
    cls = f"risk-{level.lower()}" if level else "risk-low"
    return f'<span class="{cls}">{level or "N/A"}</span>'

def format_duration(sec) -> str:
    if sec is None:
        return "—"
    if sec < 60:
        return f"{sec:.1f}s"
    return f"{sec/60:.1f}m"

def load_history() -> list:
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE) as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history(history: list):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history[-50:], f, indent=2, ensure_ascii=False)

def append_history(file_name: str, risk_level: str, status_summary: str):
    history = load_history()
    history.append({
        "file_name": file_name,
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "risk_level": risk_level,
        "status": status_summary,
    })
    save_history(history)

def is_dash_running() -> bool:
    # Probe the port directly — authoritative regardless of who started the server.
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:8050", timeout=2)
        return True
    except Exception:
        return False

def start_dash_server():
    app_path = str(ROOT / "app.py")
    # Redirect child output to a log file — an undrained PIPE deadlocks the
    # Dash server once its log output fills the OS pipe buffer (~64KB).
    log = open(OUTPUT_DIR / "dash_server.log", "ab")
    proc = subprocess.Popen(
        [sys.executable, app_path],
        cwd=str(ROOT),
        stdout=log,
        stderr=log,
    )
    st.session_state.dash_proc = proc
    time.sleep(2)

def get_available_ollama_models(host: str) -> list:
    try:
        import urllib.request, json as _json
        url = host.rstrip("/") + "/api/tags"
        with urllib.request.urlopen(url, timeout=5) as r:
            data = _json.loads(r.read())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []

def run_pipeline_subprocess(file_paths: list, model: str, ctx: int, temp1: float, temp5: float, lang: str):
    env = os.environ.copy()
    env["PIPELINE_MODEL"] = model
    env["PIPELINE_CTX"] = str(ctx)
    env["PIPELINE_TEMP1"] = str(temp1)
    env["PIPELINE_TEMP5"] = str(temp5)
    env["PIPELINE_LANG"] = lang
    # Fast mode — orchestrate.py honours PIPELINE_FAST. Default "1" (fast).
    # Reads the saved setting so a user toggle in Settings flows through here.
    _fast = st.session_state.settings.get("pipeline", {}).get("fast_mode", True)
    env["PIPELINE_FAST"] = "1" if _fast else "0"
    env["PYTHONUNBUFFERED"] = "1"  # stream prints to the log file immediately
    env["PYTHONIOENCODING"] = "utf-8"  # avoid cp1252 crashes on Arabic output

    # Write file list for orchestrator
    with open(OUTPUT_DIR / "uploaded_files.json", "w") as f:
        json.dump(file_paths, f)

    # Truncate previous log so the live view only shows THIS run.
    (OUTPUT_DIR / "pipeline.log").write_text("", encoding="utf-8")

    # -u = unbuffered, so the live log in the UI updates in real time.
    cmd = [sys.executable, "-u", str(ROOT / "orchestrate.py")] + file_paths
    # Stream child output to a log file rather than an undrained PIPE (which would
    # deadlock the pipeline). Progress is tracked via output/pipeline_status.json.
    log = open(OUTPUT_DIR / "pipeline.log", "ab")
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT),
        stdout=log, stderr=log,
        env=env,
    )
    # Record the PID so the run can be stopped from ANY session (even after a
    # browser refresh that lost the in-memory process handle).
    try:
        PID_FILE.write_text(str(proc.pid), encoding="utf-8")
    except Exception:
        pass
    return proc


def _kill_pid_tree(pid: int):
    """Kill a process and its children, cross-platform, without extra deps."""
    if sys.platform.startswith("win"):
        # /T kills the whole tree, /F forces it.
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        import signal
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(1)
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            pass


def terminate_pipeline(reset_status: bool = True):
    """Stop the running pipeline from anywhere and reset to a clean idle state.

    Order matters: kill the process FIRST, then reset the status file — otherwise
    the still-alive orchestrator could re-write a 'running' status after we reset.
    """
    # 1) Kill the in-memory handle if we have it.
    proc = st.session_state.get("pipeline_proc")
    if proc is not None and hasattr(proc, "poll") and proc.poll() is None:
        try:
            _kill_pid_tree(proc.pid)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass

    # 2) Kill via the PID file (covers refreshed sessions with no handle).
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            _kill_pid_tree(pid)
        except Exception:
            pass
        try:
            PID_FILE.unlink()
        except Exception:
            pass

    time.sleep(0.5)

    # 3) Reset status to idle so reconciliation won't resurrect "running".
    if reset_status:
        idle = {k: {"status": "idle", "start": None, "end": None, "duration_sec": None}
                for k in ["agent_1", "agent_2", "agent_3", "agent_4", "agent_5"]}
        try:
            with open(STATUS_FILE, "w") as f:
                json.dump(idle, f, indent=2)
        except Exception:
            pass

    # 4) Clear session flags so the normal Run button returns.
    st.session_state.pipeline_running = False
    st.session_state.pipeline_proc = None
    st.session_state.pipeline_start_time = None

# ─── Sidebar Navigation ───────────────────────────────────────────────────────
PAGES = [
    "🏠 Home",
    "📁 Upload & Run",
    "📊 Dashboard Viewer",
    "🔍 Data Explorer",
    "🤖 Agent Monitor",
    "⚙️  Settings",
]

with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:20px 0 12px 0;">'
        '  <img src="/assets/we_logo.svg" '
        '       onerror="this.style.display=\'none\'" '
        '       style="width:72px;height:72px;border-radius:50%;'
        '              box-shadow:0 4px 16px rgba(0,0,0,0.3);margin-bottom:12px;" />'
        '  <p style="font-size:20px;font-weight:800;margin:0;color:white;'
        '             letter-spacing:0.5px;">WE Analytics</p>'
        '  <p style="font-size:11px;margin:4px 0 0 0;color:rgba(255,255,255,0.65);'
        '             letter-spacing:0.8px;text-transform:uppercase;">'
        '    OSS Technical Operations</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<hr style="border-color:rgba(255,255,255,0.2);margin:4px 0 12px 0;" />',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        PAGES,
        key="nav_radio",
        label_visibility="collapsed",
    )
    st.session_state.current_page = page

    st.markdown(
        '<hr style="border-color:rgba(255,255,255,0.2);margin:4px 0 12px 0;" />',
        unsafe_allow_html=True,
    )
    status = read_status()
    all_done = all(v.get("status") == "done" for v in status.values())
    any_running = any(v.get("status") == "running" for v in status.values())

    if any_running:
        st.markdown(
            '<div style="background:rgba(255,255,255,0.12);border-radius:10px;'
            'padding:8px 12px;margin-bottom:10px;">'
            '<span style="color:#E8D8F8;font-size:13px;font-weight:600;">⏳ Pipeline running…</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        # Always-available Stop & Reset — works from any page, even after refresh.
        if st.button("⏹ Stop & Reset", key="sidebar_stop", width="stretch"):
            terminate_pipeline()
            st.session_state["_just_stopped"] = True
            st.rerun()
    elif all_done:
        st.markdown(
            '<div style="background:rgba(22,163,74,0.2);border-radius:10px;'
            'padding:8px 12px;margin-bottom:10px;">'
            '<span style="color:#86EFAC;font-size:13px;font-weight:600;">✅ Last run: complete</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="background:rgba(255,255,255,0.07);border-radius:10px;'
            'padding:8px 12px;margin-bottom:10px;">'
            '<span style="color:rgba(255,255,255,0.5);font-size:13px;">⬜ No active pipeline</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    # Reset everything (clears outputs so the next file starts truly fresh)
    if st.button("🔄 Reset / New Session", key="sidebar_reset", width="stretch"):
        terminate_pipeline()
        for _f in [ANALYSIS_FILE, DESIGN_FILE, INSIGHTS_FILE, FIGURES_FILE,
                   OUTPUT_DIR / "app_generated.py", OUTPUT_DIR / "uploaded_files.json",
                   OUTPUT_DIR / "pipeline.log"]:
            try:
                Path(_f).unlink()
            except Exception:
                pass
        for _k in ["uploaded_files", "current_files_df", "last_analysis",
                   "last_design", "last_insights", "data_explorer_page"]:
            st.session_state[_k] = [] if _k == "uploaded_files" else (0 if _k == "data_explorer_page" else None)
        st.session_state["_just_reset"] = True
        st.rerun()

# ─── Reconcile running state from disk (survives browser refresh) ────────────
# If the status file shows an agent actively running but this (possibly fresh)
# session thinks nothing is running, resume monitoring so the UI keeps updating.
_recon = read_status()
_recon_running = any(v.get("status") == "running" for v in _recon.values())
if _recon_running and not st.session_state.pipeline_running:
    st.session_state.pipeline_running = True
    if not st.session_state.pipeline_start_time:
        # best-effort: derive start from agent_1's timestamp
        try:
            a1 = _recon.get("agent_1", {}).get("start")
            st.session_state.pipeline_start_time = (
                datetime.fromisoformat(a1).timestamp() if a1 else time.time()
            )
        except Exception:
            st.session_state.pipeline_start_time = time.time()

# ─── Pipeline completion check (NON-blocking) ────────────────────────────────
# IMPORTANT: do NOT sleep/rerun here — that would prevent the page body (and its
# live progress display) from ever rendering. We only detect completion/failure
# and finalise state. The per-page live section handles the actual polling loop.
def check_pipeline_finished():
    """Returns ('running'|'done'|'failed'|'idle'). Finalises state on completion."""
    if not st.session_state.pipeline_running:
        return "idle"

    status = read_status()
    all_done = all(v.get("status") == "done" for v in status.values())
    any_failed = any(v.get("status") == "failed" for v in status.values())

    # Detect a subprocess that died without finishing (e.g. Ollama unreachable,
    # import error) — status would otherwise stay "running" forever.
    proc = st.session_state.get("pipeline_proc")
    proc_dead = proc is not None and hasattr(proc, "poll") and proc.poll() is not None
    if proc_dead and not all_done:
        any_failed = True

    if all_done or any_failed:
        st.session_state.pipeline_running = False
        st.session_state.last_analysis = read_json(ANALYSIS_FILE)
        st.session_state.last_design = read_json(DESIGN_FILE)
        st.session_state.last_insights = read_json(INSIGHTS_FILE)
        risk = (st.session_state.last_insights or {}).get("risk_level", "UNKNOWN")
        files = st.session_state.get("uploaded_files", [])
        fname = Path(files[0]).name if files else "unknown"
        append_history(fname, risk, "done" if all_done else "failed")
        return "done" if all_done else "failed"
    return "running"

_pipeline_state = check_pipeline_finished()

# One-shot confirmation messages after Stop / Reset actions
if st.session_state.pop("_just_stopped", False):
    try:
        st.toast("⏹ Pipeline stopped and reset to idle.", icon="⏹")
    except Exception:
        pass
if st.session_state.pop("_just_reset", False):
    try:
        st.toast("🔄 Session reset — outputs cleared. Ready for a new file.", icon="🔄")
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — HOME
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    # WE branded page header
    st.markdown(
        '<div style="background:linear-gradient(135deg,#5B2083,#7B3BAF);'
        'border-radius:16px;padding:22px 28px;margin-bottom:20px;'
        'display:flex;align-items:center;gap:18px;'
        'box-shadow:0 4px 20px rgba(91,32,131,0.25);">'
        '  <img src="/assets/we_logo.svg" style="width:56px;height:56px;border-radius:50%;'
        '       box-shadow:0 2px 10px rgba(0,0,0,0.25);" />'
        '  <div>'
        '    <div style="font-size:22px;font-weight:800;color:white;letter-spacing:0.3px;">'
        '      WE Network Analytics</div>'
        '    <div style="font-size:13px;color:rgba(255,255,255,0.75);margin-top:3px;">'
        '      OSS Technical Operations — AI-Powered Pipeline Dashboard</div>'
        '  </div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Read current status
    status = read_status()
    analysis = st.session_state.last_analysis or read_json(ANALYSIS_FILE)
    insights = st.session_state.last_insights or read_json(INSIGHTS_FILE)

    # Agent Status Cards
    st.markdown("### Agent Status")
    agent_info = {
        "agent_1": {"name": "Agent 1 — The Detective", "desc": "Schema Discovery + Deep Analysis"},
        "agent_2": {"name": "Agent 2 — The Architect", "desc": "Chart Design"},
        "agent_3": {"name": "Agent 3 — The Coder", "desc": "Plotly Figure Builder"},
        "agent_4": {"name": "Agent 4 — The Builder", "desc": "Dash App Assembly"},
        "agent_5": {"name": "Agent 5 — The Narrator", "desc": "Executive Insights"},
    }

    cols = st.columns(5)
    for i, (key, info) in enumerate(agent_info.items()):
        ag = status.get(key, {})
        s = ag.get("status", "idle")
        dur = format_duration(ag.get("duration_sec"))
        ts = ag.get("end") or ag.get("start") or ""
        ts_fmt = ts[:16].replace("T", " ") if ts else "—"
        border_colors = {"done": "#16A34A", "running": "#5B2083", "failed": "#DC2626", "idle": "#C7A8E8"}
        bc = border_colors.get(s, "#6B7280")
        with cols[i]:
            st.markdown(
                f'<div class="agent-card agent-card-{s}">'
                f'<p style="font-weight:700;font-size:13px;margin:0;color:#5B2083">{info["name"]}</p>'
                f'<p style="font-size:11px;color:#6B7280;margin:2px 0 6px 0">{info["desc"]}</p>'
                f'{status_badge(s)}'
                f'<p style="font-size:11px;color:#9CA3AF;margin-top:6px">Last: {ts_fmt}<br>Duration: {dur}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Quick Summary Card
    st.markdown("### Quick Summary")
    if analysis and analysis.get("meta"):
        meta = analysis["meta"]
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f'<div class="kpi-card"><div class="kpi-value">🌐</div><div class="kpi-value" style="font-size:18px">{meta.get("domain","—").upper()}</div><div class="kpi-label">Domain</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="kpi-card"><div class="kpi-value">{meta.get("row_count",0):,}</div><div class="kpi-label">Rows</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="kpi-card"><div class="kpi-value">{meta.get("column_count",0)}</div><div class="kpi-label">Columns</div></div>', unsafe_allow_html=True)
        with c4:
            risk = (insights or {}).get("risk_level", "—")
            st.markdown(f'<div class="kpi-card"><div class="kpi-value">{risk_badge(risk)}</div><div class="kpi-label">Risk Level</div></div>', unsafe_allow_html=True)
        with c5:
            kpi_count = len(analysis.get("kpis", []))
            st.markdown(f'<div class="kpi-card"><div class="kpi-value">{kpi_count}</div><div class="kpi-label">KPIs Generated</div></div>', unsafe_allow_html=True)

        story = meta.get("story", "")
        if story:
            st.info(f"📖 **Story:** {story}")
    else:
        st.info("No analysis data yet. Upload a file and run the pipeline to get started.")

    # Open Dashboard Button
    st.markdown("### Dashboard")
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        st.link_button("🚀 Open Dashboard", "http://localhost:8050")
    if not is_dash_running():
        st.warning("⚠ Dash server is not running. Go to Dashboard Viewer to start it.")

    # Run History
    st.markdown("### Recent Runs")
    history = load_history()
    if history:
        recent = history[-5:][::-1]
        df_hist = pd.DataFrame(recent)
        if not df_hist.empty:
            st.dataframe(
                df_hist[["file_name", "run_date", "risk_level", "status"]].rename(
                    columns={"file_name": "File", "run_date": "Date", "risk_level": "Risk", "status": "Status"}
                ),
                use_container_width=True, hide_index=True,
            )
    else:
        st.caption("No runs yet.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — UPLOAD & RUN
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📁 Upload & Run":
    st.markdown('<div class="section-header">📁 Upload & Run Pipeline</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop your data files here",
        type=["xlsx", "xls", "csv", "json"],
        accept_multiple_files=True,
        help="Supports Excel, CSV, and JSON files from any domain.",
    )

    if uploaded:
        # Save uploaded files to data/ directory
        saved_paths = []
        for uf in uploaded:
            dest = DATA_DIR / uf.name
            with open(dest, "wb") as f:
                f.write(uf.getbuffer())
            saved_paths.append(str(dest))
        st.session_state.uploaded_files = saved_paths

        # Preview
        st.markdown("### File Preview")
        for i, (uf, path) in enumerate(zip(uploaded, saved_paths)):
            with st.expander(f"📄 {uf.name}  ({uf.size/1024:.1f} KB)", expanded=(i == 0)):
                try:
                    ext = Path(uf.name).suffix.lower()
                    if ext in (".xlsx", ".xls"):
                        df_prev = pd.read_excel(path)
                    elif ext == ".csv":
                        df_prev = pd.read_csv(path, encoding="utf-8", errors="replace")
                    else:
                        df_prev = pd.read_json(path)
                    st.session_state.current_files_df = df_prev
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Rows", f"{len(df_prev):,}")
                    c2.metric("Columns", str(len(df_prev.columns)))
                    c3.metric("Size", f"{uf.size/1024:.1f} KB")
                    st.write("**Columns:**", ", ".join(df_prev.columns.tolist()))
                    st.dataframe(df_prev.head(10), use_container_width=True)
                except Exception as e:
                    st.error(f"Preview failed: {e}")

    # Run Configuration — initialise defaults then let the expander override them
    settings = st.session_state.settings
    saved_model = settings.get("ollama", {}).get("default_model", "qwen2.5-coder:32b")

    # Pull the models ACTUALLY installed in the user's Ollama. Fall back to a
    # recommended list only if Ollama is unreachable.
    ollama_host = settings.get("ollama", {}).get("host", "http://localhost:11434")
    installed_models = get_available_ollama_models(ollama_host)
    ollama_unreachable = not installed_models
    if installed_models:
        ollama_models = list(installed_models)
        # keep the saved default selectable even if not currently listed
        if saved_model not in ollama_models:
            ollama_models = [saved_model] + ollama_models
    else:
        ollama_models = [saved_model, "qwen2.5-coder:32b", "deepseek-coder-v2", "llama3.1:8b"]
        # de-dupe while preserving order
        ollama_models = list(dict.fromkeys(ollama_models))

    model_to_use = saved_model
    ctx_window = int(settings.get("agent_defaults", {}).get("agent_1_context", 16384))
    temp_1 = float(settings.get("agent_defaults", {}).get("agent_1_temperature", 0.0))
    temp_5 = float(settings.get("agent_defaults", {}).get("agent_5_temperature", 0.3))
    lang_hint = settings.get("agent_defaults", {}).get("language_hint", "Auto-detect")

    st.markdown("### Run Configuration")
    if ollama_unreachable:
        st.warning("⚠ Could not reach Ollama at "
                   f"`{ollama_host}` — showing recommended models instead of your installed ones. "
                   "Start it with `ollama serve`, then reopen this panel.")
    with st.expander("⚙️ Advanced Options", expanded=True):
        if not ollama_unreachable:
            st.caption(f"✅ Found {len(installed_models)} installed model(s) in your Ollama: "
                       + ", ".join(installed_models))
        custom_model = st.text_input("Custom model name (overrides dropdown)", value="",
                                     placeholder="e.g. qwen2.5-coder:7b")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            selected_model = st.selectbox(
                "Model (from your Ollama)",
                ollama_models,
                index=ollama_models.index(saved_model) if saved_model in ollama_models else 0,
                help="This list is read live from your Ollama installation.",
            )
        with col_m2:
            lang_hint = st.selectbox("Language Hint", ["Auto-detect", "English", "Arabic", "Mixed"],
                                      index=["Auto-detect", "English", "Arabic", "Mixed"].index(lang_hint) if lang_hint in ["Auto-detect", "English", "Arabic", "Mixed"] else 0)
        model_to_use = custom_model.strip() if custom_model.strip() else selected_model

        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            ctx_window = st.slider("Agent 1 Context Window", 4096, 32768, value=ctx_window, step=1024)
        with col_c2:
            temp_1 = st.slider("Agent 1 Temperature", 0.0, 1.0, value=temp_1, step=0.05)
        with col_c3:
            temp_5 = st.slider("Agent 5 Temperature", 0.0, 1.0, value=temp_5, step=0.05)

        _output_folder = st.text_input("Output Folder", value=settings.get("pipeline", {}).get("default_output_folder", "./output"))

    # Pipeline control
    st.markdown("---")
    col_run, col_cancel = st.columns([1, 4])

    with col_run:
        run_clicked = st.button("🚀 Run Full Pipeline", type="primary", disabled=not bool(st.session_state.uploaded_files))

    if not st.session_state.uploaded_files:
        st.info("Upload at least one file to enable the pipeline.")

    if run_clicked and st.session_state.uploaded_files:
        # Reset agent statuses
        st.session_state.pipeline_running = True
        st.session_state.pipeline_start_time = time.time()

        # Reset status file
        idle_status = {k: {"status": "idle", "start": None, "end": None, "duration_sec": None}
                       for k in ["agent_1", "agent_2", "agent_3", "agent_4", "agent_5"]}
        with open(STATUS_FILE, "w") as f:
            json.dump(idle_status, f)

        proc = run_pipeline_subprocess(
            st.session_state.uploaded_files,
            model=model_to_use,
            ctx=ctx_window,
            temp1=temp_1,
            temp5=temp_5,
            lang=lang_hint,
        )
        st.session_state.pipeline_proc = proc
        st.success("Pipeline started! Monitoring progress...")
        st.rerun()

    # ── Live progress display ────────────────────────────────────────────────
    agent_names = {
        "agent_1": "Agent 1 · The Detective (analysing schema & data)",
        "agent_2": "Agent 2 · The Architect (designing charts)",
        "agent_3": "Agent 3 · The Coder (building figures)",
        "agent_4": "Agent 4 · The Builder (assembling dashboard)",
        "agent_5": "Agent 5 · The Narrator (writing insights)",
    }
    agents_order = ["agent_1", "agent_2", "agent_3", "agent_4", "agent_5"]

    if st.session_state.pipeline_running:
        st.markdown("### 🔴 Pipeline Running")
        status = read_status()
        done_count = sum(1 for k in agents_order if status.get(k, {}).get("status") == "done")
        running_agents = [k for k in agents_order if status.get(k, {}).get("status") == "running"]
        progress_val = done_count / len(agents_order)

        # Headline: what is happening right now
        if running_agents:
            current = running_agents[0]
            st.info(f"⏳ **{agent_names[current]}** is working…  "
                    f"_(large models can take 1–5 min per agent — this is normal)_")
        else:
            st.info("⏳ Starting up… launching the first agent.")

        st.progress(progress_val, text=f"Agents completed: {done_count}/5")

        elapsed = time.time() - (st.session_state.pipeline_start_time or time.time())
        st.caption(f"⏱ Total elapsed: {int(elapsed)}s")

        # Per-agent status cards
        cols = st.columns(5)
        icons = {"done": "✅", "running": "⏳", "failed": "❌", "idle": "⬜"}
        border = {"done": "#16A34A", "running": "#5B2083", "failed": "#DC2626", "idle": "#C7A8E8"}
        for i, k in enumerate(agents_order):
            ag = status.get(k, {})
            s = ag.get("status", "idle")
            dur = format_duration(ag.get("duration_sec"))
            # live elapsed for the running agent
            extra = ""
            if s == "running" and ag.get("start"):
                try:
                    started = datetime.fromisoformat(ag["start"])
                    secs = (datetime.now(started.tzinfo) - started).total_seconds()
                    extra = f"{int(secs)}s…"
                except Exception:
                    extra = "…"
            label = dur if s == "done" else (extra or "")
            with cols[i]:
                st.markdown(
                    f'<div class="agent-card agent-card-{s}" style="text-align:center">'
                    f'<div style="font-size:22px">{icons.get(s,"⬜")}</div>'
                    f'<div style="font-size:11px;font-weight:700;color:#5B2083">{k.replace("_"," ").title()}</div>'
                    f'<div style="font-size:11px">{status_badge(s)}</div>'
                    f'<div style="font-size:11px;color:#6B7280;margin-top:4px">{label}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Live log stream from the orchestrator (real subprocess output)
        st.markdown("#### 📜 Live Log")
        log_area = st.empty()
        log_path = OUTPUT_DIR / "pipeline.log"
        if log_path.exists():
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    log_text = f.read()
                tail = "\n".join(log_text.splitlines()[-25:]) or "(waiting for output…)"
            except Exception:
                tail = "(log not readable yet)"
        else:
            tail = "(waiting for the orchestrator to produce output…)"
        log_area.code(tail, language=None)

        cc1, cc2 = st.columns([1, 4])
        with cc1:
            if st.button("⛔ Stop & Reset", type="primary"):
                terminate_pipeline()
                st.session_state["_just_stopped"] = True
                st.rerun()
        with cc2:
            st.caption("Stops the running agents and resets to idle so you can start a new file.")

        # Poll: re-read status every 2s (this is the ONLY place that loops now)
        time.sleep(2)
        st.rerun()

    # ── Failure display ──────────────────────────────────────────────────────
    if _pipeline_state == "failed":
        st.error("❌ The pipeline stopped before finishing. See the log below for the cause "
                 "(most often: Ollama isn't running, or the selected model isn't pulled).")
        status = read_status()
        failed = [agent_names.get(k, k) for k, v in status.items() if v.get("status") == "failed"]
        if failed:
            st.write("**Failed at:** " + ", ".join(failed))
        log_path = OUTPUT_DIR / "pipeline.log"
        if log_path.exists():
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                log_text = f.read()
            with st.expander("📜 View full pipeline log", expanded=True):
                st.code("\n".join(log_text.splitlines()[-40:]), language=None)
        st.info("**Quick checks:**  ① Run `ollama serve` in a terminal.  "
                "② Pull the model: `ollama pull llama3.1:8b`.  "
                "③ Confirm the host in **Settings → Test Connection**.")

    # After successful completion
    if not st.session_state.pipeline_running and STATUS_FILE.exists():
        status = read_status()
        all_done = all(v.get("status") == "done" for v in status.values())
        if all_done:
            analysis = read_json(ANALYSIS_FILE)
            insights = read_json(INSIGHTS_FILE)
            total_time = sum(
                (v.get("duration_sec") or 0) for v in status.values()
            )
            st.success(f"✅ Pipeline complete in {format_duration(total_time)}!")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.link_button("🚀 Open Dashboard", "http://localhost:8050")
            with c2:
                if analysis:
                    with st.expander("📋 View Analysis JSON"):
                        st.json(analysis)
            with c3:
                # Download all outputs as zip
                try:
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                        zip_items = {
                            "analysis.json": OUTPUT_DIR / "analysis.json",
                            "design.json": OUTPUT_DIR / "design.json",
                            "insights.json": OUTPUT_DIR / "insights.json",
                            "figures_code.py": OUTPUT_DIR / "figures_code.py",
                            "app_generated.py": OUTPUT_DIR / "app_generated.py",
                            "app.py": ROOT / "app.py",
                        }
                        for arcname, fpath in zip_items.items():
                            if fpath.exists():
                                zf.write(fpath, arcname)
                    buf.seek(0)
                    st.download_button("📦 Download All Outputs", buf, "dashboard_outputs.zip", "application/zip")
                except Exception as e:
                    st.error(f"Zip failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — DASHBOARD VIEWER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Dashboard Viewer":
    st.markdown('<div class="section-header">📊 Dashboard Viewer</div>', unsafe_allow_html=True)

    # Controls row
    col_a, col_b, col_c = st.columns([1, 1, 4])
    with col_a:
        if st.button("🔄 Refresh"):
            st.rerun()
    with col_b:
        st.link_button("↗ Open in New Tab", "http://localhost:8050")
    analysis = read_json(ANALYSIS_FILE)
    if analysis:
        with col_c:
            domain = analysis.get("meta", {}).get("domain", "")
            story = analysis.get("meta", {}).get("story", "")
            st.caption(f"**Domain:** {domain}  |  {story[:100] + '...' if len(story) > 100 else story}")

    dash_ok = is_dash_running()
    if dash_ok:
        st.components.v1.iframe("http://localhost:8050", height=900, scrolling=True)
    else:
        st.warning("⚠ Dash server is not running on port 8050.")
        col_s1, col_s2 = st.columns([1, 4])
        with col_s1:
            if st.button("▶ Start Dashboard Server"):
                with st.spinner("Starting Dash server..."):
                    start_dash_server()
                    time.sleep(3)
                if is_dash_running():
                    st.success("Dashboard started!")
                    st.rerun()
                else:
                    st.error("Could not start dashboard. Check that app.py exists.")
        with col_s2:
            st.info(
                "The dashboard is generated after a pipeline run. "
                "Run the pipeline from **Upload & Run** first."
            )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — DATA EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Data Explorer":
    st.markdown('<div class="section-header">🔍 Data Explorer</div>', unsafe_allow_html=True)

    # Load data
    df_explore = st.session_state.get("current_files_df")
    files = st.session_state.get("uploaded_files", [])
    if df_explore is None and files:
        try:
            ext = Path(files[0]).suffix.lower()
            if ext in (".xlsx", ".xls"):
                df_explore = pd.read_excel(files[0])
            elif ext == ".csv":
                df_explore = pd.read_csv(files[0], encoding="utf-8", errors="replace")
            else:
                df_explore = pd.read_json(files[0])
        except Exception as e:
            st.error(f"Could not load file: {e}")

    analysis = read_json(ANALYSIS_FILE)

    if df_explore is None:
        st.info("No data loaded yet. Please upload a file first.")
    else:
        tab_raw, tab_cols, tab_agg = st.tabs(["📋 Raw Data", "🔬 Column Inspector", "📊 Aggregations"])

        # ── Sub-tab A: Raw Data ──────────────────────────────────────────────
        with tab_raw:
            st.markdown(f"**{len(df_explore):,} rows × {len(df_explore.columns)} columns**")
            col_filter, col_search, col_sort = st.columns(3)
            with col_filter:
                selected_cols = st.multiselect("Filter Columns", df_explore.columns.tolist(), default=df_explore.columns.tolist())
            with col_search:
                search_text = st.text_input("Search rows (text match)", "")
            with col_sort:
                sort_col = st.selectbox("Sort by", ["(none)"] + df_explore.columns.tolist())

            df_view = df_explore[selected_cols] if selected_cols else df_explore
            if search_text:
                mask = df_view.apply(lambda col: col.astype(str).str.contains(search_text, case=False, na=False)).any(axis=1)
                df_view = df_view[mask]
            if sort_col and sort_col != "(none)":
                df_view = df_view.sort_values(sort_col)

            PAGE_SIZE = 100
            total_pages = max(1, (len(df_view) - 1) // PAGE_SIZE + 1)
            page_num = st.session_state.get("data_explorer_page", 0)
            page_num = max(0, min(page_num, total_pages - 1))
            start = page_num * PAGE_SIZE
            end = start + PAGE_SIZE

            st.dataframe(df_view.iloc[start:end], use_container_width=True, height=450)

            col_p1, col_p2, col_p3 = st.columns([1, 2, 1])
            with col_p1:
                if st.button("◀ Prev") and page_num > 0:
                    st.session_state.data_explorer_page = page_num - 1
                    st.rerun()
            with col_p2:
                st.caption(f"Page {page_num + 1} of {total_pages}  |  {len(df_view):,} rows shown")
            with col_p3:
                if st.button("Next ▶") and page_num < total_pages - 1:
                    st.session_state.data_explorer_page = page_num + 1
                    st.rerun()

            csv_data = df_view.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download Filtered CSV", csv_data, "filtered_data.csv", "text/csv")

        # ── Sub-tab B: Column Inspector ──────────────────────────────────────
        with tab_cols:
            col_meta = {}
            if analysis and analysis.get("columns"):
                col_meta = {c["original_name"]: c for c in analysis["columns"]}

            importance_filter = st.multiselect("Filter by Importance", ["high", "medium", "low"], default=["high", "medium", "low"])
            type_filter = st.multiselect("Filter by Data Type", ["numeric", "categorical", "datetime", "text", "identifier", "boolean"], default=["numeric", "categorical", "datetime", "text", "identifier", "boolean"])

            for col in df_explore.columns:
                meta = col_meta.get(col, {})
                imp = meta.get("importance", "medium")
                dtype_meta = meta.get("data_type", "text")
                if imp not in importance_filter or dtype_meta not in type_filter:
                    continue

                with st.expander(f"**{col}**  —  {dtype_meta.upper()}  |  Importance: {imp.upper()}", expanded=False):
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Data Type", meta.get("data_type", str(df_explore[col].dtype)))
                    with m2:
                        st.metric("Semantic Role", meta.get("semantic_role", "—"))
                    with m3:
                        null_pct = df_explore[col].isna().mean() * 100
                        st.metric("Null %", f"{null_pct:.1f}%")
                    with m4:
                        st.metric("Arabic", "Yes" if meta.get("has_arabic") else "No")

                    # Null bar
                    st.progress(null_pct / 100, text=f"Null: {null_pct:.1f}%")

                    # Stats
                    if pd.api.types.is_numeric_dtype(df_explore[col]):
                        s = df_explore[col].describe()
                        cols_s = st.columns(4)
                        cols_s[0].metric("Min", f"{s['min']:.2f}")
                        cols_s[1].metric("Max", f"{s['max']:.2f}")
                        cols_s[2].metric("Mean", f"{s['mean']:.2f}")
                        cols_s[3].metric("Std", f"{s['std']:.2f}")
                    else:
                        top_vals = df_explore[col].value_counts().head(5)
                        st.write("**Top Values:**")
                        st.dataframe(top_vals.reset_index().rename(columns={col: "Value", "count": "Count"}), use_container_width=True, hide_index=True)

                    sample = meta.get("sample_values") or df_explore[col].dropna().head(5).tolist()
                    st.write("**Samples:**", str(sample[:5]))

        # ── Sub-tab C: Aggregations ──────────────────────────────────────────
        with tab_agg:
            if analysis and analysis.get("aggregations"):
                agg_keys = list(analysis["aggregations"].keys())
                sel_agg = st.selectbox("Select Aggregation", agg_keys)
                agg_data = analysis["aggregations"].get(sel_agg, {})

                if isinstance(agg_data, list):
                    try:
                        df_agg = pd.DataFrame(agg_data)
                        st.dataframe(df_agg, use_container_width=True, hide_index=True)
                        if len(df_agg.columns) >= 2:
                            import plotly.express as px
                            x_col = df_agg.columns[0]
                            y_cols = [c for c in df_agg.columns[1:] if pd.api.types.is_numeric_dtype(df_agg[c])]
                            if y_cols:
                                fig = px.bar(df_agg, x=x_col, y=y_cols[0], title=sel_agg,
                                             color_discrete_sequence=["#5B2083"])
                                fig.update_layout(paper_bgcolor="#F7F5FC", plot_bgcolor="#F7F5FC")
                                st.plotly_chart(fig, use_container_width=True)
                        csv_agg = df_agg.to_csv(index=False).encode("utf-8")
                        st.download_button(f"⬇ Download {sel_agg}.csv", csv_agg, f"{sel_agg}.csv", "text/csv")
                    except Exception as e:
                        st.json(agg_data)
                elif isinstance(agg_data, dict):
                    st.json(agg_data)
                else:
                    st.write(agg_data)
            else:
                st.info("No aggregation data available. Run the pipeline first.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — AGENT MONITOR
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖 Agent Monitor":
    st.markdown('<div class="section-header">🤖 Agent Monitor</div>', unsafe_allow_html=True)

    status = read_status()
    analysis = read_json(ANALYSIS_FILE)
    design = read_json(DESIGN_FILE)
    insights = read_json(INSIGHTS_FILE)
    figures_code = None
    if FIGURES_FILE.exists():
        with open(FIGURES_FILE, encoding="utf-8") as f:
            figures_code = f.read()

    agent_details = {
        "agent_1": {
            "name": "Agent 1 — The Detective",
            "desc": "Schema Discovery + Deep Analysis",
            "output": analysis,
            "output_label": "analysis.json",
        },
        "agent_2": {
            "name": "Agent 2 — The Architect",
            "desc": "Chart Design",
            "output": design,
            "output_label": "design.json",
        },
        "agent_3": {
            "name": "Agent 3 — The Coder",
            "desc": "Plotly Figure Builder",
            "output": figures_code,
            "output_label": "figures_code.py",
        },
        "agent_4": {
            "name": "Agent 4 — The Builder",
            "desc": "Dash App Assembly",
            "output": (OUTPUT_DIR / "app_generated.py").read_text(encoding="utf-8") if (OUTPUT_DIR / "app_generated.py").exists() else None,
            "output_label": "app_generated.py",
        },
        "agent_5": {
            "name": "Agent 5 — The Narrator",
            "desc": "Executive Insights",
            "output": insights,
            "output_label": "insights.json",
        },
    }

    for key, info in agent_details.items():
        ag = status.get(key, {})
        s = ag.get("status", "idle")
        start_t = (ag.get("start") or "")[:16].replace("T", " ")
        end_t = (ag.get("end") or "")[:16].replace("T", " ")
        dur = format_duration(ag.get("duration_sec"))
        out = info["output"]
        out_size = 0
        if isinstance(out, str):
            out_size = len(out.encode("utf-8"))
        elif out:
            out_size = len(json.dumps(out).encode("utf-8"))

        with st.expander(f"{info['name']}  ·  {status_badge(s)}", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Status", s.upper())
            c2.metric("Duration", dur)
            c3.metric("Started", start_t or "—")
            c4.metric("Output Size", f"{out_size/1024:.1f} KB" if out_size else "—")

            if out:
                with st.expander(f"📄 Preview: {info['output_label']}", expanded=False):
                    if isinstance(out, str):
                        st.code(out[:3000] + ("..." if len(out) > 3000 else ""), language="python" if ".py" in info["output_label"] else None)
                    else:
                        st.json(out)

            col_rerun, _ = st.columns([1, 3])
            with col_rerun:
                if st.button(f"🔁 Re-run {key}", key=f"rerun_{key}"):
                    st.info(f"Re-running {key} is available via the full pipeline. Use Upload & Run with the same file.")

    # Run history
    st.markdown("### Run History")
    history = load_history()
    if history:
        df_hist = pd.DataFrame(history[::-1])
        st.dataframe(
            df_hist.rename(columns={"file_name": "File", "run_date": "Date", "risk_level": "Risk", "status": "Status"}),
            use_container_width=True, hide_index=True,
        )
        if st.button("🗑 Clear History"):
            save_history([])
            st.success("History cleared.")
            st.rerun()
    else:
        st.caption("No run history yet.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6 — SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "⚙️  Settings":
    st.markdown('<div class="section-header">⚙️ Settings</div>', unsafe_allow_html=True)

    s = st.session_state.settings

    # ── Ollama Connection ────────────────────────────────────────────────────
    st.markdown("#### Ollama Connection")
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        ollama_host = st.text_input("Ollama Host", value=s.get("ollama", {}).get("host", "http://localhost:11434"))
    with col_h2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔌 Test Connection"):
            models = get_available_ollama_models(ollama_host)
            if models:
                st.success(f"✅ Connected! {len(models)} models available: {', '.join(models[:5])}")
            else:
                st.error("❌ Cannot connect to Ollama. Make sure it's running.")

    available_models_cached = get_available_ollama_models(ollama_host) or ["qwen2.5-coder:32b", "deepseek-coder-v2", "llama3.1:8b"]
    current_model = s.get("ollama", {}).get("default_model", "qwen2.5-coder:32b")
    if current_model not in available_models_cached:
        available_models_cached = [current_model] + available_models_cached
    default_model = st.selectbox("Default Model", available_models_cached,
                                  index=available_models_cached.index(current_model) if current_model in available_models_cached else 0)

    # ── Pipeline Defaults ────────────────────────────────────────────────────
    st.markdown("#### Pipeline Defaults")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        default_output = st.text_input("Default Output Folder", value=s.get("pipeline", {}).get("default_output_folder", "./output"))
    with col_p2:
        auto_open = st.checkbox("Auto-open dashboard after run", value=s.get("pipeline", {}).get("auto_open_dashboard", True))
    with col_p3:
        max_retries = st.number_input("Max retries per agent", min_value=0, max_value=5, value=int(s.get("pipeline", {}).get("max_retries", 1)))

    # Fast mode — skips Agents 3, 4, 5 LLM calls (~7+ min saved)
    fast_mode = st.checkbox(
        "⚡ Fast mode (recommended) — skip Agents 3, 4, 5 LLM calls",
        value=s.get("pipeline", {}).get("fast_mode", True),
        help=(
            "Agents 3 (figure code) and 4 (Dash app code) generate creative variants "
            "that the served dashboard never uses — app.py is the deterministic renderer "
            "that builds every chart from JSON. Agent 5's deterministic baseline already "
            "contains operational-grade insights with real numbers. Skipping these three "
            "LLM calls saves about 7+ minutes per run on small models with no loss in "
            "dashboard quality."
        ),
    )

    # ── Arabic Support ───────────────────────────────────────────────────────
    st.markdown("#### Arabic Support")
    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        enable_rtl = st.checkbox("Enable RTL", value=s.get("arabic", {}).get("enable_rtl", True))
    with col_a2:
        arabic_font = st.selectbox("Arabic Font", ["DejaVu Sans", "Noto Sans Arabic", "Cairo"],
                                    index=["DejaVu Sans", "Noto Sans Arabic", "Cairo"].index(
                                        s.get("arabic", {}).get("font", "DejaVu Sans")
                                    ))
    with col_a3:
        auto_fix = st.checkbox("Auto-fix mojibake", value=s.get("arabic", {}).get("auto_fix_mojibake", True))

    # ── Export Settings ──────────────────────────────────────────────────────
    st.markdown("#### Export Settings")
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        png_dpi = st.slider("PNG DPI", 72, 300, value=int(s.get("export", {}).get("png_dpi", 150)))
    with col_e2:
        pdf_size = st.selectbox("PDF Page Size", ["A4", "A3", "Letter"],
                                 index=["A4", "A3", "Letter"].index(s.get("export", {}).get("pdf_page_size", "A4")))
    with col_e3:
        include_insights = st.checkbox("Include Insights in PDF", value=s.get("export", {}).get("include_insights_in_pdf", True))

    # ── Save / Reset ─────────────────────────────────────────────────────────
    st.markdown("---")
    col_save, col_reset = st.columns([1, 5])
    with col_save:
        if st.button("💾 Save Settings", type="primary"):
            new_settings = {
                "ollama": {"host": ollama_host, "default_model": default_model},
                "pipeline": {
                    "default_output_folder": default_output,
                    "auto_open_dashboard": auto_open,
                    "max_retries": max_retries,
                    "fast_mode": fast_mode,
                },
                "agent_defaults": {
                    "agent_1_context": s.get("agent_defaults", {}).get("agent_1_context", 16384),
                    "agent_1_temperature": s.get("agent_defaults", {}).get("agent_1_temperature", 0.0),
                    "agent_5_temperature": s.get("agent_defaults", {}).get("agent_5_temperature", 0.3),
                    "language_hint": s.get("agent_defaults", {}).get("language_hint", "Auto-detect"),
                },
                "arabic": {"enable_rtl": enable_rtl, "font": arabic_font, "auto_fix_mojibake": auto_fix},
                "export": {"png_dpi": png_dpi, "pdf_page_size": pdf_size, "include_insights_in_pdf": include_insights},
            }
            save_settings(new_settings)
            st.session_state.settings = new_settings
            st.success("✅ Settings saved!")

    with col_reset:
        if st.button("↩ Reset to Defaults"):
            defaults = {
                "ollama": {"host": "http://localhost:11434", "default_model": "qwen2.5-coder:32b"},
                "pipeline": {"default_output_folder": "./output", "auto_open_dashboard": True, "max_retries": 1, "fast_mode": True},
                "agent_defaults": {"agent_1_context": 16384, "agent_1_temperature": 0.0, "agent_5_temperature": 0.3, "language_hint": "Auto-detect"},
                "arabic": {"enable_rtl": True, "font": "DejaVu Sans", "auto_fix_mojibake": True},
                "export": {"png_dpi": 150, "pdf_page_size": "A4", "include_insights_in_pdf": True},
            }
            save_settings(defaults)
            st.session_state.settings = defaults
            st.success("Settings reset to defaults.")
            st.rerun()
