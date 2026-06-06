import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import kaleido
import json
import os

# Load figures from figures_code.py
figures_code_path = 'output/figures_code.py'
with open(figures_code_path, 'r') as f:
    exec(f.read())

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# Load analysis and design JSONs
analysis_json_path = 'path_to_analysis.json'  # Replace with actual path
design_json_path = 'path_to_design.json'  # Replace with actual path

with open(analysis_json_path, 'r') as f:
    analysis_data = json.load(f)

with open(design_json_path, 'r') as f:
    design_data = json.load(f)

# Extract data sources and chart configurations
data_sources = {chart['id']: chart for chart in design_data['charts']}
chart_configs = {chart['id']: chart for chart in design_data['charts']}

# Create layout dynamically from analysis JSON and design JSON
layout = dbc.Container([
    dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(dbc.Row([dbc.Col(dbc.Label(analysis_data['meta']['dashboard_title'], style={'color': 'white'}), width=12)])),
                        dbc.CardBody(
                            [
                                dbc.Row([dbc.Col(dbc.Label(f"Current Date: {datetime.now().strftime('%Y-%m-%d')}", style={'color': 'white'}), width=12)]),
                                dbc.Row([dbc.Col(dbc.Alert("Urgent Banner", color="danger", style={"font-weight": "bold", "color": "white"}), width=12) if analysis_data['meta']['urgent_flag'] else dbc.Col([], width=12)])
                            ]
                        )
                    ],
                    body=True,
                    color="#1B3A6B",
                    inverse=True
                ),
                width=12
            )
        ]
    ),
    dbc.Row(
        [
            dbc.Tabs(
                id="tabs",
                children=[
                    dbc.Tab(label=tab_name, tab_id=f"tab-{i}")
                    for i, tab_name in enumerate(design_data['tab_names'])
                ],
                active_tab_class="active"
            )
        ]
    ),
    dbc.Row([
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardHeader(dbc.Row([dbc.Col(dbc.Label("KPI Cards", style={'color': 'white'}), width=12)])),
                    dbc.CardBody(
                        [
                            dbc.Row([
                                dbc.Col(dbc.Card(
                                    [
                                        dbc.CardHeader(dbc.Row([dbc.Col(dbc.Label(f"{kpi['label']}", style={'color': kpi['color_hint']}), width=4)])),
                                        dbc.CardBody(dbc.Row([dbc.Col(dbc.Label(kpi['value'], style={'font-weight': 'bold', 'color': kpi['color_hint']}), width=8)]))
                                    ],
                                    body=True,
                                    color="white",
                                    inverse=True
                                ))
                                for kpi in design_data['kpi_cards']
                            ])
                        ]
                    )
                ],
                body=True,
                color="#1B3A6B",
                inverse=True
            ),
            width=4
        ),
        dbc.Col(
            dbc.Row([
                dbc.Col(dbc.Card(
                    [
                        dbc.CardHeader(dbc.Row([dbc.Col(dbc.Label(chart['title'], style={'color': 'white'}), width=12)])),
                        dbc.CardBody(
                            dcc.Graph(id=chart['id'], figure=figures[chart['id']])
                        )
                    ],
                    body=True,
                    color="white",
                    inverse=True
                ))
                for chart in design_data['charts']
            ])
        ),
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardHeader(dbc.Row([dbc.Col(dbc.Label("Insights", style={'color': 'white'}), width=12)])),
                    dbc.CardBody(
                        [
                            dbc.Row([
                                dbc.Col(dbc.Card(
                                    [
                                        dbc.CardHeader(dbc.Row([dbc.Col(dbc.Label(insight['title'], style={'color': 'white'}), width=12)])),
                                        dbc.CardBody(dbc.Row([dbc.Col(dbc.Markdown(insight['content']), width=12)]))
                                    ],
                                    body=True,
                                    color="white",
                                    inverse=True
                                ))
                                for insight in design_data['insights']
                            ])
                        ]
                    )
                ],
                body=True,
                color="#1B3A6B",
                inverse=True
            ),
            width=4
        )
    ]),
    dbc.Row([
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardHeader(dbc.Row([dbc.Col(dbc.Label("Download", style={'color': 'white'}), width=12)])),
                    dbc.CardBody(
                        dcc.Download(id="download-png")
                    )
                ],
                body=True,
                color="#1B3A6B",
                inverse=True
            ),
            width=4
        ),
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardHeader(dbc.Row([dbc.Col(dbc.Label("Export PDF", style={'color': 'white'}), width=12)])),
                    dbc.CardBody(
                        dcc.Download(id="download-pdf")
                    )
                ],
                body=True,
                color="#1B3A6B",
                inverse=True
            ),
            width=4
        ),
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardHeader(dbc.Row([dbc.Col(dbc.Label("Auto-Refresh", style={'color': 'white'}), width=12)])),
                    dbc.CardBody(
                        dcc.Dropdown(
                            id="auto-refresh-dropdown",
                            options=[
                                {"label": "Off", "value": 0},
                                {"label": "5 seconds", "value": 5},
                                {"label": "30 seconds", "value": 30},
                                {"label": "1 minute", "value": 60}
                            ],
                            value=0
                        )
                    )
                ],
                body=True,
                color="#1B3A6B",
                inverse=True
            ),
            width=4
        )
    ]),
    dbc.Row([
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardHeader(dbc.Row([dbc.Col(dbc.Label("Dark Mode", style={'color': 'white'}), width=12)])),
                    dbc.CardBody(
                        dcc.Checklist(
                            id="dark-mode-toggle",
                            options=[
                                {"label": "Enable Dark Mode", "value": True}
                            ],
                            value=[False]
                        )
                    )
                ],
                body=True,
                color="#1B3A6B",
                inverse=True
            ),
            width=4
        ),
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardHeader(dbc.Row([dbc.Col(dbc.Label("RTL", style={'color': 'white'}), width=12)])),
                    dbc.CardBody(
                        dcc.Checklist(
                            id="rtl-toggle",
                            options=[
                                {"label": "Enable RTL", "value": True}
                            ],
                            value=[False]
                        )
                    )
                ],
                body=True,
                color="#1B3A6B",
                inverse=True
            ),
            width=4
        )
    ])
])

# Callbacks for Dark mode toggle, RTL toggle, Auto-refresh interval, Download PNG, dcc.Upload
@app.callback(
    Output("body", "className"),
    Input("dark-mode-toggle", "value")
)
def update_dark_mode(value):
    if value:
        return "dark-mode"
    else:
        return ""

@app.callback(
    Output("body", "className"),
    Input("rtl-toggle", "value")
)
def update_rtl(value):
    if value:
        return "rtl"
    else:
        return ""

@app.callback(
    Output("auto-refresh-dropdown", "value"),
    Input("auto-refresh-dropdown", "value")
)
def update_auto_refresh_interval(interval):
    return interval

@app.callback(
    Output("download-png", "data"),
    [Input("tabs", "active_tab")]
)
def download_png(active_tab):
    if active_tab:
        chart_id = design_data['tab_names'].index(active_tab) + 1
        fig = figures[design_data['charts'][chart_id - 1]['id']]
        return dcc.send_bytes(kaleido.dumps(fig, format="png"))

@app.callback(
    Output("download-pdf", "data"),
    [Input("tabs", "active_tab")]
)
def download_pdf(active_tab):
    if active_tab:
        chart_id = design_data['tab_names'].index(active_tab) + 1
        fig = figures[design_data['charts'][chart_id - 1]['id']]
        return dcc.send_bytes(kaleido.dumps(fig, format="pdf"))

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=False)