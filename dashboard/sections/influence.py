"""Feature A — Collaboration Influence Score section."""

from __future__ import annotations

import networkx as nx
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.charts import make_full_network_graph


def render_influence_section(
    scored_df: pd.DataFrame,
    G: nx.DiGraph,
    pr_scores: dict,
    bc_scores: dict,
) -> None:
    st.markdown("---")
    st.markdown("## 🕸 Feature A — Collaboration Influence Score")
    st.caption(
        "PageRank (60%) + Betweenness Centrality (40%) on the reviewer→author graph. "
        "High influence = structurally important bridge or heavily-depended-upon reviewer."
    )

    with st.expander("Full Collaboration Network", expanded=False):
        st.plotly_chart(
            make_full_network_graph(G, pr_scores, bc_scores),
            use_container_width=True,
            key="full_network",
        )

    # Bar chart
    infl_df = (
        scored_df[["engineer", "influence", "custom_impact"]]
        .sort_values("influence", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )
    infl_df.index += 1

    fig_infl = px.bar(
        infl_df,
        x="engineer",
        y="influence",
        color="custom_impact",
        color_continuous_scale="Viridis",
        labels={"influence": "Influence Score", "custom_impact": "Impact"},
        height=320,
    )
    fig_infl.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=60, l=40, r=20),
        xaxis_gridcolor="#2a2a3a",
        yaxis_gridcolor="#2a2a3a",
        coloraxis_colorbar=dict(title="Impact"),
    )
    st.plotly_chart(fig_infl, use_container_width=True, key="infl_bar")

    # Table
    infl_table = scored_df[["engineer", "influence", "custom_impact", "pr_count"]].copy()
    infl_table.columns = ["Engineer", "Influence Score", "Impact", "PRs"]
    infl_table = (
        infl_table.sort_values("Influence Score", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )
    infl_table.index += 1
    st.dataframe(
        infl_table.style
        .background_gradient(subset=["Influence Score"], cmap="Purples")
        .format("{:.2f}", subset=["Influence Score", "Impact"]),
        use_container_width=True,
        height=360,
    )
