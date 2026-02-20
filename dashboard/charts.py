"""All Plotly chart-builder functions."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard.config import (
    COLOR_C, COLOR_D, COLOR_I, COLOR_L, COLOR_R,
    RADAR_DIMS, RADAR_NAMES,
)


# ── Radar ─────────────────────────────────────────────────────────────────────

def make_radar(engineer: str, scores: dict) -> go.Figure:
    values = [scores.get(d, 0.0) for d in RADAR_DIMS]
    values_closed = values + [values[0]]
    dims_closed   = RADAR_NAMES + [RADAR_NAMES[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=dims_closed,
        fill="toself",
        fillcolor="rgba(67,97,238,0.18)",
        line=dict(color=COLOR_D, width=2),
        name=engineer,
        hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], tickfont_size=9)),
        showlegend=False,
        margin=dict(t=30, b=20, l=30, r=30),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Impact timeline ───────────────────────────────────────────────────────────

def make_impact_timeline(ts_df: pd.DataFrame, engineers: list[str]) -> go.Figure:
    fig = go.Figure()
    palette = px.colors.qualitative.Plotly
    for i, eng in enumerate(engineers):
        edf = ts_df[ts_df["engineer"] == eng].sort_values("week")
        if edf.empty:
            continue
        fig.add_trace(go.Scatter(
            x=edf["week"], y=edf["impact"],
            mode="lines+markers",
            name=eng,
            line=dict(color=palette[i % len(palette)], width=2),
            hovertemplate=f"<b>{eng}</b><br>Week: %{{x|%b %d}}<br>Impact: %{{y:.1f}}<extra></extra>",
        ))
    fig.update_layout(
        xaxis_title="Week",
        yaxis_title="Impact (0–100)",
        yaxis=dict(range=[0, 100]),
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=20, b=60, l=40, r=20),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#2a2a3a"),
        yaxis_gridcolor="#2a2a3a",
    )
    return fig


# ── Delivery vs Collaboration timeline ────────────────────────────────────────

def make_delivery_collab_timeline(ts_df: pd.DataFrame, engineer: str) -> go.Figure:
    edf = ts_df[ts_df["engineer"] == engineer].sort_values("week")
    fig = go.Figure()
    if edf.empty:
        return fig
    fig.add_trace(go.Bar(
        x=edf["week"], y=edf["delivery"],
        name="Delivery", marker_color=COLOR_D, opacity=0.7,
        hovertemplate="Delivery: %{y:.1f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=edf["week"], y=edf["collaboration"],
        name="Collaboration", mode="lines+markers",
        line=dict(color=COLOR_C, width=2),
        hovertemplate="Collaboration: %{y:.1f}<extra></extra>",
    ))
    fig.update_layout(
        barmode="overlay",
        yaxis=dict(range=[0, 100], title="Score"),
        xaxis_title="Week",
        legend=dict(
            orientation="h",
            x=1, y=1, xanchor="right", yanchor="bottom",
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(t=20, b=40, l=40, r=20),
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#2a2a3a"),
        yaxis_gridcolor="#2a2a3a",
    )
    return fig


# ── Risk timeline ─────────────────────────────────────────────────────────────

def make_risk_timeline(ts_df: pd.DataFrame, engineer: str) -> go.Figure:
    edf = ts_df[ts_df["engineer"] == engineer].sort_values("week")
    fig = go.Figure()
    if edf.empty:
        return fig
    fig.add_trace(go.Bar(
        x=edf["week"], y=edf["risk_weighted_prs"],
        name="Risk-weighted PR Volume",
        marker_color="#f77f00", opacity=0.75,
        hovertemplate="Risk Vol: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=edf["week"], y=edf["reverts_issues"],
        name="Reverts/Issues", mode="lines+markers",
        line=dict(color="#ef233c", width=2),
        yaxis="y2",
        hovertemplate="Reverts: %{y}<extra></extra>",
    ))
    fig.update_layout(
        yaxis=dict(title="Risk Volume"),
        yaxis2=dict(title="Reverts/Issues", overlaying="y", side="right", rangemode="tozero"),
        legend=dict(
            orientation="h",
            x=1, y=1, xanchor="right", yanchor="bottom",
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(t=20, b=40, l=40, r=40),
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Week",
        xaxis=dict(gridcolor="#2a2a3a"),
        yaxis_gridcolor="#2a2a3a",
    )
    return fig


# ── Collaboration mini-graph (ego network) ────────────────────────────────────

def make_collab_mini_graph(
    G: nx.DiGraph, engineer: str, max_neighbors: int = 15
) -> go.Figure:
    if engineer not in G:
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(text="No graph data", xref="paper", yref="paper",
                              x=0.5, y=0.5, showarrow=False, font_size=14)],
            height=320, paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    neighbors = set(G.predecessors(engineer)) | set(G.successors(engineer))
    neighbors = sorted(
        neighbors,
        key=lambda n: G.degree(n, weight="weight"),
        reverse=True,
    )[:max_neighbors]
    sub = G.subgraph([engineer] + neighbors).copy()
    pos = nx.spring_layout(sub, seed=42, k=1.5)

    edge_traces = []
    for u, v, data in sub.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        w = data.get("weight", 1)
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=max(1, min(w / 3, 6)), color="#555"),
            hoverinfo="none",
        ))

    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node in sub.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        deg = sub.degree(node, weight="weight")
        node_text.append(f"{node}<br>Edge weight: {deg}")
        if node == engineer:
            node_color.append(COLOR_I)
            node_size.append(22)
        elif sub.has_edge(node, engineer):
            node_color.append(COLOR_C)
            node_size.append(14)
        else:
            node_color.append(COLOR_D)
            node_size.append(12)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=[n if n == engineer else "" for n in sub.nodes()],
        textposition="top center",
        marker=dict(size=node_size, color=node_color,
                    line=dict(width=1, color="#fff")),
        hovertext=node_text,
        hoverinfo="text",
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(t=10, b=10, l=10, r=10),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Full collaboration network ─────────────────────────────────────────────────

def make_full_network_graph(
    G: nx.DiGraph, pr_scores: dict, bc_scores: dict
) -> go.Figure:
    if len(G.nodes) == 0:
        return go.Figure()

    pos = nx.spring_layout(G, seed=7, k=1.2)
    pr_max = max(pr_scores.values(), default=1e-9)

    edge_traces = []
    for u, v, data in G.edges(data=True):
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        w = data.get("weight", 1)
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=max(0.5, min(w / 4, 4)), color="rgba(150,150,200,0.4)"),
            hoverinfo="none",
        ))

    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node in G.nodes():
        if node not in pos:
            continue
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        pr_val = pr_scores.get(node, 0)
        bc_val = bc_scores.get(node, 0)
        node_text.append(f"<b>{node}</b><br>PageRank: {pr_val:.4f}<br>Betweenness: {bc_val:.4f}")
        node_color.append(pr_val / pr_max)
        node_size.append(8 + 24 * (pr_val / pr_max))

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers",
        hovertext=node_text, hoverinfo="text",
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale="Plasma",
            showscale=True,
            colorbar=dict(title="PageRank (norm)", thickness=12, len=0.5),
            line=dict(width=1, color="#fff"),
        ),
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(t=10, b=10, l=10, r=10),
        height=480,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Dimension breakdown (snapshot fallback) ───────────────────────────────────

def make_dimension_breakdown(scored_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    df = scored_df.head(top_n).sort_values("custom_impact")
    fig = go.Figure()
    for col, name, color in [
        ("delivery",      "Delivery",      COLOR_D),
        ("leverage",      "Leverage",      COLOR_L),
        ("collaboration", "Collaboration", COLOR_C),
        ("reliability",   "Reliability",  COLOR_R),
    ]:
        fig.add_trace(go.Bar(
            y=df["engineer"], x=df[col],
            name=name, orientation="h",
            marker_color=color, opacity=0.85,
            hovertemplate=f"{name}: %{{x:.1f}}<extra>%{{y}}</extra>",
        ))
    fig.update_layout(
        barmode="stack",
        xaxis=dict(title="Dimension Scores (stacked)", range=[0, 400]),
        yaxis=dict(title="", tickfont_size=11),
        legend=dict(orientation="h", y=1.05, x=0),
        margin=dict(t=40, b=40, l=120, r=20),
        height=max(300, top_n * 28),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_gridcolor="#2a2a3a",
    )
    return fig


# ── Momentum scatter (snapshot fallback) ──────────────────────────────────────

def make_momentum_scatter(scored_df: pd.DataFrame) -> go.Figure:
    df = scored_df.copy()
    med_d = df["delivery"].median()
    med_c = df["collaboration"].median()

    fig = px.scatter(
        df,
        x="delivery", y="collaboration",
        size="leverage", color="reliability",
        hover_name="engineer", text="engineer",
        color_continuous_scale="RdYlGn",
        size_max=30,
        labels={
            "delivery": "Delivery Score", "collaboration": "Collaboration Score",
            "reliability": "Reliability (colour)", "leverage": "Leverage (size)",
        },
        height=420,
    )
    fig.update_traces(textposition="top center", textfont_size=9)
    fig.add_vline(x=med_d, line_dash="dot", line_color="#666",
                  annotation_text="median delivery", annotation_position="top right")
    fig.add_hline(y=med_c, line_dash="dot", line_color="#666",
                  annotation_text="median collab", annotation_position="top right")
    for txt, ax, ay in [
        ("⚡ High-output soloists",      95,  5),
        ("🌟 All-round contributors",    95, 95),
        ("🤝 Collaborative supporters",  5,  95),
        ("💤 Low signal",                5,   5),
    ]:
        fig.add_annotation(x=ax, y=ay, text=txt, showarrow=False,
                           font=dict(size=9, color="#888"), xref="x", yref="y")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 100], gridcolor="#2a2a3a"),
        yaxis=dict(range=[0, 100], gridcolor="#2a2a3a"),
        margin=dict(t=30, b=40, l=40, r=20),
        coloraxis_colorbar=dict(title="Reliability"),
    )
    return fig


# ── Trajectory chart ──────────────────────────────────────────────────────────

def make_trajectory_chart(
    ts_df: pd.DataFrame, top_engineers: list[str]
) -> go.Figure:
    fig = go.Figure()
    palette = px.colors.qualitative.Bold

    for i, eng in enumerate(top_engineers):
        edf = ts_df[ts_df["engineer"] == eng].sort_values("week")
        if len(edf) < 2:
            continue
        x_num = np.arange(len(edf))
        y = edf["impact"].values
        m, b = np.polyfit(x_num, y, 1)
        color = palette[i % len(palette)]

        fig.add_trace(go.Scatter(
            x=edf["week"], y=edf["impact"],
            mode="markers", showlegend=False,
            marker=dict(color=color, size=6, opacity=0.5),
        ))
        trend_y = m * x_num + b
        fig.add_trace(go.Scatter(
            x=edf["week"], y=trend_y,
            mode="lines", name=eng,
            line=dict(color=color, width=2, dash="dash" if abs(m) < 1 else "solid"),
            hovertemplate=f"<b>{eng}</b>  slope={m:+.2f}/wk<extra></extra>",
        ))

    fig.update_layout(
        xaxis_title="Week",
        yaxis_title="Impact (0–100)",
        yaxis=dict(range=[0, 100]),
        legend=dict(orientation="h", y=-0.22),
        margin=dict(t=20, b=70, l=40, r=20),
        height=340,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#2a2a3a"),
        yaxis_gridcolor="#2a2a3a",
    )
    return fig
