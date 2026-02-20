"""Sidebar controls — weights, toggles, filters."""

from __future__ import annotations

import streamlit as st


def render_sidebar(snap: dict) -> tuple[float, float, float, float, bool, int]:
    """
    Render the sidebar and return:
        w_d, w_l, w_c, w_r  – dimension weights (0.0 – 2.0)
        risk_adj             – bool toggle
        min_prs              – int filter
    """
    with st.sidebar:
        st.image("https://posthog.com/brand/posthog-logo.svg", width=140)
        st.markdown("## ⚙️ Controls")

        st.markdown("### Dimension Weights")
        w_d = st.slider("Delivery",      0.0, 2.0, 0.35, 0.05, key="wd")
        w_l = st.slider("Leverage",      0.0, 2.0, 0.25, 0.05, key="wl")
        w_c = st.slider("Collaboration", 0.0, 2.0, 0.20, 0.05, key="wc")
        w_r = st.slider("Reliability",   0.0, 2.0, 0.20, 0.05, key="wr")

        st.markdown("---")
        risk_adj = st.toggle(
            "Risk-Adjusted Scoring",
            value=False,
            help="Boosts scores of engineers with high Collaboration Influence Score",
        )

        st.markdown("---")
        st.markdown("### Filters")
        min_prs = st.number_input(
            "Min merged PRs", min_value=1, max_value=30, value=2
        )

        st.markdown("---")
        st.info(
            f"**Snapshot**: {snap.get('repo', '')}  \n"
            f"**Generated**: {snap.get('generated_at', '')[:10]}  \n"
            f"**PRs analysed**: {snap.get('total_prs_analyzed', '?')}  \n"
            f"**Window**: {snap.get('window_days', 90)} days"
        )

    return w_d, w_l, w_c, w_r, risk_adj, min_prs
