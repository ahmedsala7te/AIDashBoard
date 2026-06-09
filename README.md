# Universal AI Analytics Dashboard

A fully local, AI-powered analytics platform that ingests **any** tabular data file
and automatically builds a professional interactive dashboard — with zero hardcoded
assumptions about your data.

---

## How It Works

```
You upload a file
       ↓
Agent 1 — The Detective   analyses schema, detects domain & language
       ↓
Agent 2 — The Architect   designs the optimal chart layout
       ↓         ↓
Agent 3 — The Coder    Agent 5 — The Narrator   (run in parallel)
(builds Plotly figures)  (writes executive insights)
       ↓         ↓
Agent 4 — The Builder   assembles the final Dash dashboard
       ↓
Dashboard live at http://localhost:8050
```

All five agents run locally via **Ollama** — no API keys, no internet required.

---

## Telecom Operator Intelligence (flagship)

WE / Telecom Egypt is a **wholesale access provider**: a single MSAN or PE-router
carries subscribers belonging to several downstream operators (Vodafone, Orange,
Etisalat, Noor) plus WE's own retail (WE Data) and bitstream wholesale. The
operationally critical question is *which operator's customers are affected when
an element congests* — so the engine analyses this dimension automatically.

`telecom_intelligence.py` provides, whenever per-operator subscriber columns are
present (detected by pattern — works on any operator set, not just WE's):

| Output | What it answers |
|---|---|
| **Subscriber Mix by Operator** | Who do we carry, and how much? (brand-coloured bars) |
| **Operator Congestion Exposure** | Which operator is most exposed when elements fail? (stacked bar: exposed vs healthy) |
| **Wholesale vs Retail Split** | How much of the base is WE's own vs other operators'? |
| **Operator KPIs** | Most congestion-exposed · largest operator · wholesale subscribers · operators carried |

Detection is universal (`*_sub`, `*_subscribers`, `*_lines` …), case-insensitive,
and tolerant of real-world quirks (string counts like `"400.00"`, `'-'`
placeholders, the common `etisilat`→`etisalat` misspelling). When a dataset has
no operator columns the feature is silently skipped — nothing breaks.

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.9 or higher |
| Ollama | Latest — [ollama.com/download](https://ollama.com/download) |
| RAM / VRAM | See model table below |

---

## Installation

### 1 — Clone / download the project

```bash
git clone <repo-url>
cd universal_dashboard
```

### 2 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3 — Pull an Ollama model

Choose one based on your hardware:

| Model | VRAM Required | Quality |
|---|---|---|
| `qwen2.5-coder:32b` | 24 GB+ | Best |
| `deepseek-coder-v2` | 16 GB+ | Strong |
| `llama3.1:8b` | 8 GB | Good fallback |

```bash
# Best (recommended):
ollama pull qwen2.5-coder:32b

# Strong alternative:
ollama pull deepseek-coder-v2

# Lightweight fallback:
ollama pull llama3.1:8b
```

### 4 — Start Ollama

```bash
ollama serve
```

---

## How to Run

### Start the Streamlit control panel

```bash
streamlit run streamlit_app.py
```

Then open: **http://localhost:8501**

> The Streamlit app is always the entry point. It controls everything.
> The Dash dashboard on **http://localhost:8050** is launched automatically
> after the pipeline completes.

---

## How to Use

1. **Open** http://localhost:8501
2. **Navigate** to **Upload & Run** in the sidebar
3. **Drop** any Excel, CSV, or JSON file onto the upload zone
4. *(Optional)* Expand **Advanced Options** to change the model or temperature
5. Click **Run Full Pipeline** and watch the five agents run live
6. When complete, click **Open Dashboard** to view your analytics
7. Explore data further in **Data Explorer** and read AI insights in **Agent Monitor**

---

## Supported File Types

| Format | Extensions |
|---|---|
| Microsoft Excel | `.xlsx`, `.xls` |
| CSV | `.csv` (any encoding — auto-detected) |
| JSON | `.json` (flat or array of records) |

**Multiple files:** Upload several at once — they are merged before analysis.

---

## Supported Domains

The pipeline adapts automatically to:

- Telecom (call records, subscribers, network KPIs)
- Finance (transactions, P&L, portfolio data)
- Healthcare (patient records, clinical outcomes)
- Logistics (shipments, routes, inventory)
- HR (employees, payroll, performance)
- Sales (CRM, orders, revenue)
- Manufacturing (production, quality, downtime)
- Education (students, grades, attendance)
- Government (budgets, permits, public records)
- Any other structured tabular data

---

## Folder Structure

```
universal_dashboard/
├── streamlit_app.py        Control panel (always start here)
├── orchestrate.py          Five-agent pipeline runner
├── app.py                  Dash dashboard (regenerated after each run)
├── requirements.txt        Python dependencies
├── settings.json           User configuration (editable via Settings page)
├── data/                   Drop input files here for CLI usage
├── output/
│   ├── analysis.json       Agent 1 output: schema + aggregations + KPIs
│   ├── design.json         Agent 2 output: chart specifications
│   ├── figures_code.py     Agent 3 output: executable Plotly code
│   ├── insights.json       Agent 5 output: executive narrative
│   ├── pipeline_status.json Live agent status (polled by Streamlit)
│   └── exports/
│       ├── report.png      Exported chart screenshots
│       └── report.pdf      Exported PDF report
└── README.md               This file
```

---

## CLI Usage (without Streamlit)

You can also run the pipeline directly:

```bash
python orchestrate.py path/to/file.xlsx
python orchestrate.py data/sales.csv data/returns.csv
```

Override the model:

```bash
PIPELINE_MODEL=llama3.1:8b python orchestrate.py data/file.csv
```

---

## Settings

All settings are editable in the **Settings** page of the Streamlit app
and saved to `settings.json`.

| Setting | Default | Description |
|---|---|---|
| Ollama Host | `http://localhost:11434` | Ollama server address |
| Default Model | `qwen2.5-coder:32b` | LLM used for all agents |
| Auto-open Dashboard | `true` | Launch Dash after pipeline |
| Max Retries | `1` | Retry failed agents once |
| Arabic RTL | `true` | Enable RTL layout for Arabic data |
| Auto-fix Mojibake | `true` | Fix garbled Arabic text automatically |
| PNG DPI | `150` | Chart export resolution |

---

## Troubleshooting

**Ollama not connecting**
- Run `ollama serve` in a terminal
- Check that http://localhost:11434 is reachable
- Use the **Test Connection** button in Settings

**Pipeline fails at Agent 3/4 (code generation)**
- The model may have produced invalid Python — check `output/figures_code.py`
- Try a more capable model (`qwen2.5-coder:32b` or `deepseek-coder-v2`)
- Increase context window in Run Configuration

**Dashboard shows blank page**
- Open http://localhost:8050 directly in the browser
- Check that `app.py` was generated in the project root
- Use the **Start Dashboard Server** button in Dashboard Viewer

**Arabic text appears garbled**
- Enable **Auto-fix Mojibake** in Settings
- The pipeline uses `ftfy` to fix encoding issues automatically

---

## Architecture Notes

- **100% local** — no data ever leaves your machine
- **Zero hardcoded assumptions** — every agent reasons from the data
- **Parallel execution** — Agents 3 and 5 run concurrently
- **Live status polling** — Streamlit polls `output/pipeline_status.json` every 2 seconds
- **Retry policy** — each agent retries once on failure with the error appended to the prompt
