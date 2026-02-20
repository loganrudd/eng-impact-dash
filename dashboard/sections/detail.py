"""Engineer detail panel — radar, timelines, ego graph."""

from __future__ import annotations

import networkx as nx
import pandas as pd
import streamlit as st

from dashboard.charts import (
    make_collab_mini_graph,
    make_delivery_collab_timeline,
    make_radar,
    make_risk_timeline,
    make_trajectory_chart,
)


def render_detail_panel(
    scored_df: pd.DataFrame,
    snap: dict,
    ts_df: pd.DataFrame,
    G: nx.DiGraph,
    trajectories: dict,
) -> None:
    st.markdown("---")
    st.markdown("## 🔍 Engineer Detail Panel")

    eng_list = scored_df["engineer"].tolist()
    selected = st.selectbox("Select engineer:", eng_list, index=0, key="eng_select")

    eng_row  = scored_df[scored_df["engineer"] == selected].iloc[0]
    eng_meta = snap["engineers"].get(selected, {})

    # Overview metrics bar
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Impact",        f"{eng_row.custom_impact:.1f}")
    m2.metric("Delivery",      f"{eng_row.delivery:.1f}")
    m3.metric("Leverage",      f"{eng_row.leverage:.1f}")
    m4.metric("Collaboration", f"{eng_row.collaboration:.1f}")
    m5.metric("Reliability",   f"{eng_row.reliability:.1f}")
    m6.metric("Influence 🕸",  f"{eng_row.influence:.1f}")

    traj = trajectories.get(selected, {})
    traj_color = (
        "#2dc653" if "Accelerating" in traj.get("label", "")
        else "#ef233c" if "Declining"    in traj.get("label", "")
        else "#888"
    )
    st.markdown(
        f"<div style='padding:8px 14px; border-left:4px solid {traj_color}; "
        f"background:rgba(255,255,255,0.04); border-radius:4px; margin-bottom:8px'>"
        f"<b>Trajectory:</b> {traj.get('label', '—')} &nbsp;·&nbsp; "
        f"slope <code>{traj.get('slope', 0):+.2f}</code> pts/wk &nbsp;·&nbsp; "
        f"total Δ <code>{traj.get('delta', 0):+.1f}</code> pts over window &nbsp;·&nbsp; "
        f"PRs merged: <b>{eng_meta.get('pr_count', '?')}</b> &nbsp;·&nbsp; "
        f"Reviews given: <b>{eng_meta.get('review_count', '?')}</b>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Row 1: Radar + trajectory
    col_radar, col_timeline = st.columns([1, 2])
    with col_radar:
        st.markdown("#### Skill Radar")
        radar_scores = {
            "delivery":      eng_row.delivery,
            "leverage":      eng_row.leverage,
            "collaboration": eng_row.collaboration,
            "reliability":   eng_row.reliability,
        }
        st.plotly_chart(
            make_radar(selected, radar_scores),
            use_container_width=True,
            key="radar",
        )
    with col_timeline:
        st.markdown("#### Impact Trajectory")
        eng_ts = ts_df[ts_df["engineer"] == selected]
        if eng_ts.empty:
            st.info("No weekly time-series data available for this engineer.")
        else:
            st.plotly_chart(
                make_trajectory_chart(ts_df, [selected]),
                use_container_width=True,
                key="impact_timeline",
            )

    # Row 2: Delivery/Collab + Risk
    col_dc, col_risk = st.columns(2)
    with col_dc:
        st.markdown("#### Delivery vs Collaboration Over Time")
        st.plotly_chart(
            make_delivery_collab_timeline(ts_df, selected),
            use_container_width=True,
            key="dc_timeline",
        )
    with col_risk:
        st.markdown("#### Risk Exposure Over Time")
        st.plotly_chart(
            make_risk_timeline(ts_df, selected),
            use_container_width=True,
            key="risk_timeline",
        )

    # Row 3: Ego graph
    st.markdown("#### Collaboration Network (ego view)")
    st.plotly_chart(
        make_collab_mini_graph(G, selected),
        use_container_width=True,
        key="mini_graph",
    )
