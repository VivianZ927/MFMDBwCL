import pickle
import os
import pandas as pd
import plotly.express as px
from dash import (
    Dash, html, dash_table, dcc, callback,
    Output, Input, State
)
from dash.exceptions import PreventUpdate

# =========================
# Config
# =========================
PICKLE_PATH = "26compressedMinerals.pkl"
DEFAULT_MINERAL = 'Lithium'
S_Y, E_Y = 2020, 2025
TOP_K = 5
latest_click_ts = 0
px.set_mapbox_access_token(os.environ.get("MAPBOX_TOKEN"))
UK_GEO = dict(
    scope="europe",
    # projection_type="mercator",
    center=dict(lat=53.799, lon=-1.75),
    lonaxis=dict(range=[-11, 3]),
    lataxis=dict(range=[49, 60]),
    showcountries=True,
    showland=True,
)

# =========================
# App
# =========================
app = Dash(title="Mineral Analysis", suppress_callback_exceptions=True)
server = app.server

# =========================
# Load data - OPTIMIZED
# =========================
print("Loading data...")
with open(PICKLE_PATH, "rb") as f:
    raw_base = pickle.load(f)

#
# def optimize_df(df):
#     df = df.copy()
#
#     # 1) downcast integers
#     int_cols = df.select_dtypes(include=['int64', 'int32']).columns
#     df[int_cols] = df[int_cols].apply(
#         pd.to_numeric, downcast='integer'
#     )
#
#     # 2) downcast floats
#     float_cols = df.select_dtypes(include=['float64']).columns
#     df[float_cols] = df[float_cols].apply(
#         pd.to_numeric, downcast='float'
#     )
#
#     # 3) convert suitable object columns to category
#     obj_cols = df.select_dtypes(include=['object']).columns
#     for col in obj_cols:
#         # heuristic: if relatively few uniques, use category
#         num_unique = df[col].nunique(dropna=True)
#         if num_unique / len(df) < 0.5:
#             df[col] = df[col].astype('category')
#
#     return df
#
#
# for _m, _df in raw_base.items():
#     # Categorical columns
#     for c in ['notation',"SamplingPoint", "GrossWT", "region", "Season"]:
#         if c in _df.columns:
#             _df[c] = _df[c].astype("category")
#
#     # Numeric columns
#     if "Year" in _df.columns:
#         _df["Year"] = pd.to_numeric(_df["Year"], errors="coerce", downcast='integer')
#     if "Month" in _df.columns:
#         _df["Month"] = pd.to_numeric(_df["Month"], errors="coerce", downcast='integer')
#
#     # Pre-compute datetime column for faster plotting
#     if "Year" in _df.columns and "Month" in _df.columns:
#         _df["datetime"] = pd.to_datetime(
#             {"year": _df["Year"], "month": _df["Month"], "day": 1},
#             errors="coerce"
#         )
#     raw_base[_m]=optimize_df(_df)
#
# print("Data loaded and optimized!")
# with open('26compressedMinerals.pkl', 'wb') as fp:
#     pickle.dump(raw_base, fp)
#     print('compressed df saved successfully to file')

# =========================
# Helpers - OPTIMIZED
# =========================
def get_minerals(dfs):
    minerals = []
    for mineral in dfs.keys():
        minerals.append(mineral.split("(")[0])
    return minerals


def get_regions():
    return ['NE', 'NW', 'SE', 'SW']


def get_watertypes(dfs, mineral):
    for c in dfs.keys():
        if mineral in c:
            return sorted(dfs[c]["GrossWT"].unique().tolist())
    return []


def define_base(dfs, mineral, start_y, end_y, regions, watertypes):
    """Optimized: Fast boolean indexing with minimal operations"""
    for c in dfs.keys():
        if mineral in c:
            base = dfs[c]
            break
    if base is None:
        return pd.DataFrame()

    # Single-pass boolean mask
    mask = (
            (base["Year"] >= start_y) &
            (base["Year"] <= end_y) &
            base["region"].isin(regions) &
            base["GrossWT"].isin(watertypes)
    )
    return base[mask]


def select_SPtopk_fast(base, mineral, k):
    """Optimized: Minimize operations, use efficient indexing"""
    """Get top k sampling points with statistics"""
    mcols = base.filter(like=mineral, axis=1).columns.tolist()
    if not mcols:
        return {}
    mcol = mcols[-1]
    # Filter out NaN values early
    base = base.dropna(subset=[mcol])
    if base.empty:
        return {}
    else:
        gb = ['notation',"SamplingPoint", "lat", "lon", "GrossWT"]

        # Compute all stats at once, then filter to top k
        all_stats = base.groupby(gb, observed=True, sort=False).agg(
            average_concentration=(mcol, 'mean'),
            std_concentration=(mcol, 'std'),
            num_observations=(mcol, 'count'),
            min_year=('Year', 'min'),
            max_year=('Year', 'max')
        ).reset_index()

        # Get top k by average concentration
        top_k_stats = all_stats.nlargest(k, 'average_concentration')

        # Filter base data to only include these top k
        filtered_base = base.merge(top_k_stats[gb], on=gb, how='inner')

        # Merge stats back in
        result = filtered_base.merge(top_k_stats, on=gb, how='left')
        unit = mcol.split("(")[-1]
        unit = unit.split(")")[0]
        result = result.rename(columns={mcol: "monthly concentration ({})".format(unit)})

        return {mineral: result}


def select_WTtopk(base, mineral, k):
    """Optimized: Single aggregation pass"""
    mcols = base.filter(like=mineral, axis=1).columns.tolist()
    if not mcols:
        return pd.DataFrame()
    mcol = mcols[-1]
    base = base.dropna(subset=[mcol])
    # Single aggregation
    means_df = (
        base.groupby("GrossWT", observed=True, sort=False)[mcol]
        .mean()
        .nlargest(k)
        .reset_index(name="Monthly Mean Concentration")
    )

    if means_df.empty:
        return pd.DataFrame()

    # Filter and merge
    result = base[base["GrossWT"].isin(means_df["GrossWT"])].merge(
        means_df, on="GrossWT", how="inner"
    )

    return result.drop_duplicates()


def build_geo_fig(point_agg, mineral, k, start_y, end_y):
    """Optimized: Direct operations on filtered data"""
    UK_CENTER = dict(lat=54.5, lon=-3.0)
    UK_ZOOM = 5
    if not point_agg or (isinstance(point_agg, dict) and not point_agg):
        fig = px.scatter_map()
        fig.update_maps(
            mapbox_style="streets",
            mapbox=dict(center=UK_CENTER, zoom=UK_ZOOM),
            # uirevision="uk_map"
        )
        return fig

    label = list(point_agg.keys())[0]
    units = [c for c in point_agg[label].columns if "(" in c]
    unit = units[0].split("(")[-1]
    unit = unit.split(")")[0]

    d = point_agg[label][["SamplingPoint", "lat", "lon", 'GrossWT', "average_concentration"]].drop_duplicates()
    d = d.dropna(subset=['average_concentration'])
    if d.empty:
        fig = px.scatter_map()
        fig.update_layout(
            mapbox_style="streets",
            mapbox=dict(center=UK_CENTER, zoom=UK_ZOOM),
            # uirevision="uk_map",
            title=f"UK – {mineral} (Top {k}) ({start_y}-{end_y}) - No valid data"
        )
        return fig

    fig = px.scatter_map(
        d,
        lat="lat",
        lon="lon",
        color="SamplingPoint",
        size="average_concentration",
        hover_name="SamplingPoint",
        hover_data={
            "GrossWT": True,
            "average_concentration": ":.2f",
            "lat": True,  # Hide lat/lon in hover
            "lon": True
        },
        opacity=0.65,
    )
    fig.update_layout(
        mapbox_style="streets",
        mapbox=dict(
            center=UK_CENTER,
            zoom=UK_ZOOM
        ),
        title=f"UK – {mineral} (Top {k}) ({start_y}-{end_y}) ({unit})",
        # uirevision="uk_map",  # Keeps position between similar updates
        showlegend=True,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig


def build_dot_fig(point_agg):
    """
    Build time-series (dot/line) figures for the selected top-k sampling points.

    Returns
    -------
    list[plotly.graph_objects.Figure]
        A list with a single figure (for Dash) or an empty list if no data.
    """
    if not point_agg:
        return []

    # We only ever expect one key: the mineral name
    label = next(iter(point_agg))
    df = point_agg[label]

    # Find the concentration column: "monthly concentration (unit)"
    conc_cols = [c for c in df.columns if c.startswith("monthly concentration")]
    if not conc_cols:
        return []

    conc_col = conc_cols[0]

    # Extract unit from "monthly concentration (unit)"
    # e.g. "monthly concentration (µg/L)" -> "µg/L"
    unit = conc_col.split("(", 1)[-1].split(")", 1)[0] if "(" in conc_col else ""

    # Ensure we have a datetime column; if not, build from Year/Month
    ma = df.copy()
    if "datetime" not in ma.columns and {"Year", "Month"}.issubset(ma.columns):
        ma["datetime"] = pd.to_datetime(
            {"year": ma["Year"], "month": ma["Month"], "day": 1},
            errors="coerce"
        )

    # If there's still no datetime, we can't plot over time
    if "datetime" not in ma.columns:
        return []

    # Rename the concentration column for a nicer y-axis label
    y_label = f"Detected Concentration ({unit})" if unit else "Detected Concentration"
    ma = ma.rename(columns={conc_col: y_label})

    conc = ma[y_label].dropna()
    if conc.empty:
        return []

    ymin, ymax = conc.min(), conc.max()

    fig = px.line(
        ma,
        x="datetime",
        y=y_label,
        color="SamplingPoint",
        symbol="SamplingPoint",
        title=f"Concentration over Time — {label} ({unit})" if unit else f"Concentration over Time — {label}",
    )

    # Set a padded y-range for nicer visuals
    if pd.notna(ymin) and pd.notna(ymax):
        pad = max(1.0, 0.05 * (ymax - ymin) if ymax > ymin else 1.0)
        fig.update_yaxes(range=[ymin - pad, ymax + pad])

    fig.update_traces(mode="lines+markers", marker=dict(size=9))
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title=y_label,
        margin=dict(l=40, r=20, t=40, b=40),
        uirevision="constant",
    )

    return [fig]


def build_table_data(point_agg):
    """Optimized: Vectorized operations"""
    tbs = {}

    for m, d in point_agg.items():
        units = [c for c in d.columns if "(" in c]
        units = units[0].split("(")[-1]
        unit = units.split(")")[0]
        tbl = (
            d[["notation","SamplingPoint", "GrossWT", "average_concentration", 'std_concentration', 'num_observations']]
            .drop_duplicates()
            .sort_values("average_concentration", ascending=False)
        )
        tbl = tbl.rename(columns={"average_concentration": "MonthlyMean({})".format(unit)})
        table_cap = "MonthlyMean({})".format(unit)
        # Vectorized formatting

        tbl[table_cap] = tbl[table_cap].round(2).astype(str)
        tbl = tbl.assign(MonthlyStandardDivation=tbl["std_concentration"].round(2).astype(str))
        tbl = tbl.assign(NumberOfObservations=tbl["num_observations"].astype(str))
        tbl = tbl.assign(WaterType=tbl["GrossWT"].astype(str))
        tbl = tbl.assign(Notation=tbl["notation"].astype(str))
        tbl = tbl[["Notation","SamplingPoint", "WaterType", table_cap, "MonthlyStandardDivation", "NumberOfObservations"]]
        tbs[m] = tbl.to_dict("records")
    return tbs


def build_chart_table(initial_table, initial_dot, k):
    """Build table and chart components"""
    table_list = []
    dot_charts = []

    for label, data in initial_table.items():
        table_list.append(
            html.Div(
                children=[
                    html.Div(
                        [f"Top {k} — {label}"],
                        style={"margin": 8, "fontSize": 12, "fontWeight": "bold"}
                    ),
                    dash_table.DataTable(
                        data=data,
                        page_size=10,

                        style_table={"minWidth": 360, "overflowX": "auto"},
                        style_cell={
                            "fontSize": 12,
                            "textAlign": "left",
                            "whiteSpace": "normal",
                            "height": "auto",
                            "padding": "8px"
                        },
                        style_header={
                            "fontWeight": "bold",
                            "backgroundColor": "#f0f0f0"
                        }
                    ),
                ],
                style={"minWidth": 360, "maxWidth": 800}
            )
        )

    for fig in initial_dot:
        dot_charts.append(
            dcc.Graph(
                figure=fig,
                style={"padding": 10, "width": "100%"},
                config={"displayModeBar": False}  # Hide mode bar for speed
            )
        )

    return table_list, dot_charts


def build_bar_chart(df_for_bar, mineral):
    """Optimized: Efficient aggregation"""
    # df_for_bar=df_for_bar.rename(
    #     columns={"average monthly concentration":"average monthly concentration for specified water type"})
    if df_for_bar.empty:
        return px.bar()

    mcols = df_for_bar.filter(like=mineral, axis=1).columns.tolist()
    if not mcols:
        return px.bar()
    mcol = mcols[-1]

    # Single aggregation with observed=True
    site_stats = (
        df_for_bar.groupby(["GrossWT", "SamplingPoint"], observed=True, sort=False)
        [mcol].mean()
        .reset_index(name="site_mean")
    )

    if site_stats.empty:
        return px.bar()

    # Calculate proportions
    type_totals = site_stats.groupby("GrossWT", observed=True)["site_mean"].transform("sum")
    site_stats["portion"] = site_stats["site_mean"] / type_totals

    # Sort by Type and portion
    site_stats = site_stats.sort_values(["GrossWT", "portion"], ascending=[True, False])
    site_stats = site_stats.rename(columns={"site_mean": "Average monthly concentration"})

    fig = px.bar(
        site_stats,
        x="GrossWT",
        y="Average monthly concentration",
        color="SamplingPoint",
        title="Water Type — Average Concentration (Top types)",
    )
    fig.update_layout(
        # uirevision="keep",
        margin=dict(l=40, r=20, t=40, b=40)
    )
    return fig


# def _preserve(prev_vals, new_options, select_all_if_empty=True):
#     """Keep only previously selected values that are still valid"""
#     if prev_vals is None:
#         return new_options if select_all_if_empty else []
#     kept = [v for v in prev_vals if v in new_options]
#     return kept or (new_options if select_all_if_empty else kept)


# =========================
# Initial state for layout
# =========================
minerals = get_minerals(raw_base)
DEFAULT_MINERAL = "Lithium"

init_regions = get_regions()
init_wtypes = get_watertypes(raw_base, DEFAULT_MINERAL)

BASE_INIT = define_base(raw_base, DEFAULT_MINERAL, S_Y, E_Y, init_regions, init_wtypes)
_sp_init = select_SPtopk_fast(BASE_INIT, DEFAULT_MINERAL, k=TOP_K)
geo_init = build_geo_fig(_sp_init, DEFAULT_MINERAL, k=TOP_K, start_y=S_Y, end_y=E_Y)

dot_init = build_dot_fig(_sp_init)
table_init = build_table_data(_sp_init) if _sp_init else {}
tables_init, dots_init = build_chart_table(table_init, dot_init, k=TOP_K)
_wt_init = select_WTtopk(BASE_INIT, DEFAULT_MINERAL, k=TOP_K)
bar_init = build_bar_chart(_wt_init, DEFAULT_MINERAL)

# =========================
# Layout
# =========================
header = html.Div(
    [
        html.Div(
            [
                html.H1("UK Mineral Analysis Dashboard",
                        style={"margin": 0, "fontSize": "24px", "fontWeight": 1000}),
                html.Img(src=app.get_asset_url("logo.png"), style={"height": "70px"})
            ],
            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"}
        )
    ],
    style={
        "padding": "12px 16px",
        "backgroundColor": "white",
        "color": "#499823",
        "position": "sticky",
        "top": 0,
        "zIndex": 999,
        "borderBottom": "1px solid #eee",
    }
)

sidebar = html.Div(
    [
        html.Button(
            'Update Dashboard',
            id='submit',
            n_clicks=0,
            style={
                "width": "100%",
                "padding": "10px",
                "backgroundColor": "#499823",
                "color": "white",
                "border": "none",
                "borderRadius": "4px",
                "cursor": "pointer",
                "fontWeight": "bold"
            }
        ),
        html.H4("Mineral", style={"marginTop": 10}),
        dcc.Dropdown(
            minerals,
            DEFAULT_MINERAL,
            id="mineral-dropdown",
            clearable=False,
            style={"marginBottom": 12}
        ),

        html.Hr(),

        html.H4("Water Types"),
        dcc.Checklist(
            init_wtypes,
            init_wtypes,
            id="watertype-checkbox",
            style={"marginBottom": 12}
        ),

        html.Hr(),

        html.H4("Regions"),
        dcc.Checklist(
            init_regions,
            init_regions,
            id="region-checkbox",
            style={"maxHeight": "240px", "overflowY": "auto", "marginBottom": 12}
        ),

        html.Hr(),
        html.H4("Top Number"),
        dcc.Input(
            id="topkn",
            type="number",
            value=TOP_K,  # Add default value
            min=1,
            max=20,
            step=1

        ),

        html.H4("Year Range"),
        html.Div(
            [
                html.Div(["Start:"], style={"width": 60, "alignSelf": "center"}),
                dcc.Input(
                    id="start-year",
                    type="number",
                    value=S_Y,
                    min=2000,
                    max=2025,
                    step=1,
                    debounce=True,
                    style={"flex": 1}
                ),
            ],
            style={"display": "flex", "gap": 8, "marginBottom": 6},
        ),
        html.Div(
            [
                html.Div(["End:"], style={"width": 60, "alignSelf": "center"}),
                dcc.Input(
                    id="end-year",
                    type="number",
                    value=E_Y,
                    min=2000,
                    max=2025,
                    step=1,
                    debounce=True,
                    style={"flex": 1}
                ),
            ],
            style={"display": "flex", "gap": 8, "marginBottom": 12},
        ),

        html.Div(
            "Adjust filters and click 'Update Dashboard' to refresh charts.",
            style={"fontSize": 11, "color": "#666", "marginTop": 12, "lineHeight": 1.4}
        )
    ],
    style={
        "width": "320px",
        "flexShrink": 0,
        "padding": "16px",
        "borderRight": "1px solid #eee",
        "position": "sticky",
        "top": 84,

        "overflowY": "auto",
        "background": "#fafafa",
    }
)

content = html.Div(
    [
        dcc.Loading(
            id="loading",
            type="default",
            children=[
                html.Div(
                    id="mineral-table",
                    children=tables_init,
                    style={"gap": "20px", "width": "100%", "height": "100%", "padding-left": "50px"}
                ),
                html.Div(id="mineral-dot-chart", children=dots_init),
                html.Div([dcc.Graph(figure=geo_init, id="mineral-geoscatter-chart", config={"displayModeBar": 'hover'},
                                    style={
                                        "width": "100%",
                                        "height": "90vh",
                                        "padding-bottom": "20px"
                                    }
                                    )]),
                html.Div([dcc.Graph(figure=bar_init, id="water-type-bar", config={"displayModeBar": 'hover'})]),
            ]
        )
    ],
    style={"padding": "20px", "flex": 1, "overflow": "auto"}
)

app.layout = html.Div(
    [
        header,
        html.Div([sidebar, content], style={"display": "flex", "gap": "0px"}),
    ],
    style={"display": "flex", "flexDirection": "column", "overflow": "hidden"}
)


# =========================
# Callbacks - OPTIMIZED
# =========================
@callback(
    Output("watertype-checkbox", "options"),
    Output("watertype-checkbox", "value"),
    Output("region-checkbox", "options"),
    Output("region-checkbox", "value"),
    Input("mineral-dropdown", "value"),
    State("watertype-checkbox", "value"),
    State("region-checkbox", "value"),
    prevent_initial_call=True,
)
def sync_mineral_filters(chosen_mineral, current_wtypes, current_regions):
    """Update options and preserve valid selections when mineral changes"""
    if chosen_mineral is None:
        chosen_mineral = DEFAULT_MINERAL

    # Get new options
    new_wtypes = get_watertypes(raw_base, chosen_mineral) or []
    new_regions = ['NE', 'NW', 'SE', 'SW']

    # Preserve valid selections
    valid_wtypes = [w for w in (current_wtypes or []) if w in new_wtypes]
    valid_regions = [r for r in (current_regions or []) if r in new_regions]

    # If nothing valid, select all
    final_wtypes = valid_wtypes if valid_wtypes else new_wtypes
    final_regions = valid_regions if valid_regions else new_regions

    return new_wtypes, final_wtypes, new_regions, final_regions



@callback(
Output("mineral-dot-chart", "children"),
    Output("mineral-table", "children"),
    Output("mineral-geoscatter-chart", "figure"),
    Output("water-type-bar", "figure"),
    Output("submit", "disabled"),
    Input("submit", "n_clicks"),

    State("mineral-dropdown", "value"),
    State("watertype-checkbox", "value"),
    State("region-checkbox", "value"),
    State("start-year", "value"),
    State("end-year", "value"),
    State("topkn", "value"),
    prevent_initial_call=True,  # optional but recommended
)
def update_all(n_click,chosen_mineral, chosen_wtypes, chosen_regions,
               start_y, end_y, topkn):
    """Update all visualizations when 'Update Dashboard' is clicked."""


    # No clicks at all -> do nothing
    if n_click is None:
        raise PreventUpdate

    # # ADD THIS: Print debug info to see what's being passed
    # print(f"Click #{n_click}: mineral={chosen_mineral}, wtypes={chosen_wtypes}, regions={chosen_regions}")

    # -------------------------
    # 1. Normalise inputs
    # -------------------------
    if chosen_mineral is None:
        chosen_mineral = DEFAULT_MINERAL

    if topkn is None:
        topkn = TOP_K
    topkn = int(topkn)
    if topkn < 1:
        topkn = 1

    if start_y is None:
        start_y = S_Y
    if end_y is None:
        end_y = E_Y
    start_y = int(start_y)
    end_y = int(end_y)
    if start_y > end_y:
        start_y, end_y = end_y, start_y

    all_wtypes = get_watertypes(raw_base, chosen_mineral) or []
    all_regions = get_regions()

    # If user unchecked everything, fall back to "all"
    wtypes = chosen_wtypes if chosen_wtypes else all_wtypes
    regions = chosen_regions if chosen_regions else all_regions

    # -------------------------
    # 2. Filter base dataframe
    # -------------------------
    base = define_base(raw_base, chosen_mineral, start_y, end_y, regions, wtypes)
    sp_dict = select_SPtopk_fast(base, chosen_mineral, topkn)

    # Helper: build an empty map compatible with px.scatter_map
    def _blank_map(title_suffix="No valid data"):
        empty_df = pd.DataFrame({"lat": [], "lon": []})
        fig = px.scatter_map(
            empty_df,
            lat="lat",
            lon="lon",
            zoom=5,
            center={"lat": 54.5, "lon": -3.0},
            map_style="open-street-map",
            title=f"UK – {chosen_mineral} (Top {topkn}) ({start_y}-{end_y}) - {title_suffix}",
        )
        return fig

    # -------------------------
    # 3. Handle no data
    # -------------------------
    if base.empty or not sp_dict:
        empty_note = html.Div(
            "No data matches the selected filters.",
            style={"padding": 20, "color": "#999", "textAlign": "center", "fontSize": 14},
        )
        empty_geo = _blank_map("No data")
        empty_bar = px.bar()  # empty bar chart



        # dot children, table children, geo fig, bar fig, button disabled?
        return [empty_note], [], empty_geo, empty_bar, False

    # -------------------------
    # 4. Build figures & tables
    # -------------------------
    dot_figs = build_dot_fig(sp_dict)
    geo_fig = build_geo_fig(sp_dict, chosen_mineral, k=topkn,
                            start_y=start_y, end_y=end_y)
    table_data = build_table_data(sp_dict)
    table_list, dot_charts = build_chart_table(table_data, dot_figs, k=topkn)

    # If there is no time-series figure, show a text message instead
    if not dot_charts:
        dot_charts = [html.Div(
            "No concentration measurements for the selected filters.",
            style={"padding": 20, "color": "#999", "textAlign": "center"}
        )]

    wt_df = select_WTtopk(base, chosen_mineral, topkn)
    bar_fig = build_bar_chart(wt_df, chosen_mineral)




    # dot children, table children, geo fig, bar fig, button disabled?
    return dot_charts, table_list, geo_fig, bar_fig, False


if __name__ == "__main__":
    app.run(debug=True)  # Disable hot reload for performance
