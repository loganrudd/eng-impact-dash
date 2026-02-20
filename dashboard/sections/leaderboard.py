"""Top-5 leaderboard, trajectory chart, and momentum proxy."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.charts import (
    make_dimension_breakdown,
    make_momentum_scatter,
    make_trajectory_chart,
)


def render_leaderboard(
    scored_df: pd.DataFrame,
    trajectories: dict,
    snap: dict,
    ts_df: pd.DataFrame,
) -> None:
    st.markdown("---")
    st.markdown("## 🏆 Top 5 Most Impactful Engineers (Last 90 Days)")

    top5 = scored_df.head(5).reset_index(drop=True)

    # Scorecards
    cols = st.columns(5)
    for i, (_, row) in enumerate(top5.iterrows()):
        with cols[i]:
            delta_val = trajectories.get(row.engineer, {}).get("delta", 0.0)
            st.metric(
                label=f"#{i + 1} {row.engineer}",
                value=f"{row.custom_impact:.1f}",
                delta=f"{delta_val:+.1f} pts" if delta_val != 0 else "—",
                delta_color="normal",
            )

    # Leaderboard table
    st.markdown("### Leaderboard")
    display_top = top5[[
        "engineer", "custom_impact", "primary_strength",
        "trend_arrow", "delivery", "leverage", "collaboration",
        "reliability", "influence", "pr_count",
    ]].copy()
    display_top.index = range(1, len(display_top) + 1)
    display_top.columns = [
        "Engineer", "Impact ⚡", "Primary Strength", "Trend",
        "Delivery", "Leverage", "Collab", "Reliability",
        "Influence", "PRs",
    ]
    st.dataframe(
        display_top.style
        .background_gradient(subset=["Impact ⚡"], cmap="YlOrRd")
        .background_gradient(
            subset=["Delivery", "Leverage", "Collab", "Reliability"], cmap="Blues"
        )
        .format({
            "Impact ⚡": "{:.1f}", "Delivery": "{:.1f}",
            "Leverage": "{:.1f}", "Collab": "{:.1f}",
            "Reliability": "{:.1f}", "Influence": "{:.1f}",
        }),
        use_container_width=True,
        height=220,
    )

    # ── Trajectory / momentum section ─────────────────────────────────────────
    st.markdown("### 📈 Impact Trajectory — Top 10 Accelerating Engineers")

    multi_week_engs = {
        eng for eng, series in snap.get("time_series", {}).items()
        if len(series) >= 2
    }
    accel_engs = sorted(
        [
            (eng, trajectories.get(eng, {}).get("slope", 0.0))
            for eng in scored_df["engineer"]
            if eng in multi_week_engs
        ],
        key=lambda t: -t[1],
    )
    accel_engs = [eng for eng, slope in accel_engs if slope > 0][:10]
    has_trajectory_data = len(accel_engs) > 0

    expander_label = (
        "Engineers gaining momentum week-over-week"
        if has_trajectory_data
        else "Dimension breakdown & momentum proxy (single-snapshot view)"
    )

    with st.expander(expander_label, expanded=True):
        if has_trajectory_data:
            badge_cols = st.columns(min(len(accel_engs), 5))
            for i, eng in enumerate(accel_engs[:5]):
                t = trajectories.get(eng, {})
                with badge_cols[i]:
                    st.markdown(
                        f"**{eng}**  \n{t.get('label', '—')}  \n"
                        f"slope: `{t.get('slope', 0):+.2f}`/wk  \n"
                        f"Δ impact: `{t.get('delta', 0):+.1f}` pts"
                    )
            if len(accel_engs) > 5:
                badge_cols2 = st.columns(min(len(accel_engs) - 5, 5))
                for i, eng in enumerate(accel_engs[5:]):
                    t = trajectories.get(eng, {})
                    with badge_cols2[i]:
                        st.markdown(
                            f"**{eng}**  \n{t.get('label', '—')}  \n"
                            f"slope: `{t.get('slope', 0):+.2f}`/wk  \n"
                            f"Δ impact: `{t.get('delta', 0):+.1f}` pts"
                        )
            st.plotly_chart(
                make_trajectory_chart(ts_df, accel_engs),
                use_container_width=True,
                key="trajectory_chart",
            )
        else:
            st.caption(
                "Not yet enough multi-week history to compute week-over-week slopes. "
                "Showing **dimension breakdown** (what drives each engineer's score) "
                "and a **momentum proxy** scatter instead."
            )

            scored_df = scored_df.copy()
            scored_df["momentum_proxy"] = (
                (scored_df["delivery"] / 100) * (scored_df["leverage"] / 100)
            ) ** 0.5 * 100
            top_momentum = scored_df.nlargest(5, "momentum_proxy")

            proxy_cols = st.columns(5)
            for i, (_, row) in enumerate(top_momentum.iterrows()):
                with proxy_cols[i]:
                    st.metric(
                        label=row["engineer"],
                        value=f"{row['momentum_proxy']:.1f}",
                        delta=f"D:{row['delivery']:.0f}  L:{row['leverage']:.0f}",
                        delta_color="off",
                        help=(
                            "Momentum proxy = √(Delivery × Leverage) — "
                            "rewards fast shippers in core areas"
                        ),
                    )

            tab_breakdown, tab_scatter = st.tabs(
                ["📊 Dimension Breakdown (top 15)", "🎯 Delivery vs Collaboration"]
            )
            with tab_breakdown:
                st.plotly_chart(
                    make_dimension_breakdown(scored_df, top_n=15),
                    use_container_width=True,
                    key="dim_breakdown",
                )
            with tab_scatter:
                st.plotly_chart(
                    make_momentum_scatter(scored_df),
                    use_container_width=True,
                    key="momentum_scatter",
                )
