import plotly.graph_objects as go
from arabic_reshaper import reshape
from bidi import get_display

THEME = {
    "bg": "#F4F6F9", "card": "#FFFFFF", "header": "#1B3A6B",
    "blue1": "#2563EB", "blue2": "#3B82F6", "blue3": "#93C5FD",
    "red": "#DC2626", "orange": "#EA580C", "yellow": "#D97706",
    "green": "#16A34A", "teal": "#0891B2", "gray": "#6B7280",
    "text": "#111827", "font": "DejaVu Sans"
}

def fix_arabic(text):
    reshaped_text = reshape(text)
    return get_display(reshaped_text)

# Data
network_health_data = [
    {"key": "Overall Health", "health_pct": 90},
]

top_offenders_data = [
    {"key": "DRBNGM-G01Z-SHR-EG", "average_critical_time_min": 616},
    {"key": "SHKMA147-M01H-C-EG", "average_critical_time_min": 960},
    {"key": "TEBIN-G01Z-C-EG", "average_critical_time_min": 436},
    {"key": "HSINIA-G01Z-SHR-EG", "average_critical_time_min": 178},
    {"key": "KRDWCB32-M01H-BH-EG", "average_critical_time_min": 621},
]

by_region_data = [
    {"key": "المنطقة الثانية الشرقية", "subscribers": 50000},
    {"key": "المنطقة الأولى غرب", "subscribers": 40000},
    {"key": "المنطقة الرابعة غرب", "subscribers": 30000},
    {"key": "المنطقة الثانية الشرقية", "subscribers": 20000},
    {"key": "منطقة تليفونات البحيرة الثانية", "subscribers": 10000},
]

by_sector_data = [
    {"key": "قطاع شرق الدلتا 2", "subscribers": 60000},
    {"key": "قطاع غرب القاهرة", "subscribers": 50000},
    {"key": "قطاع غرب القاهرة", "subscribers": 40000},
    {"key": "قطاع شرق الدلتا 2", "subscribers": 30000},
    {"key": "قطاع غرب الدلتا", "subscribers": 20000},
]

# Figures
figures = {}

# Network Health Gauge
network_health_fig = go.Figure(
    data=[go.Indicator(
        mode="gauge+number",
        value=network_health_data[0]["health_pct"],
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": THEME["red"]},
               "steps": [{"range": [0,80], "color": "#DCFCE7"}, {"range": [80,90], "color": "#FEF3C7"},
                         {"range": [90,100], "color": "#FEE2E2"}]}
    )],
    layout=go.Layout(
        paper_bgcolor="#F4F6F9",
        plot_bgcolor="#FFFFFF",
        font_family="DejaVu Sans",
        font_color="#111827",
        margin=dict(l=220, r=160, t=60, b=80)
    )
)
figures["network_health"] = network_health_fig

# Top Offenders Horizontal Bar
top_offenders_fig = go.Figure(
    data=[go.Bar(
        x=top_offenders_data,
        y=[fix_arabic(d["key"]) for d in top_offenders_data],
        marker_color=THEME["red"],
        hovertemplate="%{x}<br>%{y}"
    )],
    layout=go.Layout(
        paper_bgcolor="#F4F6F9",
        plot_bgcolor="#FFFFFF",
        font_family="DejaVu Sans",
        font_color="#111827",
        margin=dict(l=220, r=160, t=60, b=80),
        xaxis_title="Critical Time (min)",
        yaxis_title="MSAN"
    )
)
figures["top_offenders"] = top_offenders_fig

# By Region Horizontal Bar
by_region_fig = go.Figure(
    data=[go.Bar(
        x=by_region_data,
        y=[fix_arabic(d["key"]) for d in by_region_data],
        marker_color=THEME["red"],
        hovertemplate="%{x}<br>%{y}"
    )],
    layout=go.Layout(
        paper_bgcolor="#F4F6F9",
        plot_bgcolor="#FFFFFF",
        font_family="DejaVu Sans",
        font_color="#111827",
        margin=dict(l=220, r=160, t=60, b=80),
        xaxis_title="Subscribers",
        yaxis_title=""
    )
)
figures["by_region"] = by_region_fig

# By Sector Horizontal Bar
by_sector_fig = go.Figure(
    data=[go.Bar(
        x=by_sector_data,
        y=[fix_arabic(d["key"]) for d in by_sector_data],
        marker_color=THEME["red"],
        hovertemplate="%{x}<br>%{y}"
    )],
    layout=go.Layout(
        paper_bgcolor="#F4F6F9",
        plot_bgcolor="#FFFFFF",
        font_family="DejaVu Sans",
        font_color="#111827",
        margin=dict(l=220, r=160, t=60, b=80),
        xaxis_title="Subscribers",
        yaxis_title=""
    )
)
figures["by_sector"] = by_sector_fig

# Print completion message
print("FIGURES_READY")