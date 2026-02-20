"""Full leaderboard table — all engineers, all dimensions."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_full_leaderboard(scored_df: pd.DataFrame) -> None:
    st.markdown("---")
    st.markdown("## 📊 Full Leaderboard")

    full_display = scored_df[[
        "engineer", "custom_impact", "delivery", "leverage",
        "collaboration", "reliability", "influence",
        "primary_strength", "trend_arrow", "pr_count",
    ]].copy()
    full_display.index = range(1, len(full_display) + 1)
    full_display.columns = [
        "Engineer", "Impact ⚡", "Delivery", "Leverage",
        "Collaboration", "Reliability", "Influence 🕸",
        "Strength", "Trend", "PRs",
    ]

    st.dataframe(
        full_display.style
        .background_gradient(subset=["Impact ⚡"], cmap="YlOrRd")
        .background_gradient(
            subset=["Delivery", "Leverage", "Collaboration", "Reliability"],
            cmap="Blues",
        )
        .background_gradient(subset=["Influence 🕸"], cmap="Purples")
        .format({
            "Impact ⚡":     "{:.1f}",
            "Delivery":      "{:.1f}",
            "Leverage":      "{:.1f}",
            "Collaboration": "{:.1f}",
            "Reliability":   "{:.1f}",
            "Influence 🕸":  "{:.1f}",
        }),
        use_container_width=True,
        height=600,
    )

    st.markdown("---")
    st.caption(
        "Built with Streamlit · Data: GitHub GraphQL API · "
        "Metrics defined in `generate_impact.py`"
    )
