"""
Engineering Impact Dashboard — thin orchestrator
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Eng Impact Dashboard · PostHog",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dashboard.data import (
    build_engineers_df,
    build_timeseries_df,
    load_snapshot,
    recompute_scores,
)
from dashboard.metrics import (
    collab_influence_score,
    compute_graph_metrics,
    compute_trajectory,
    trend_arrow,
)
from dashboard.sections.sidebar import render_sidebar
from dashboard.sections.leaderboard import render_leaderboard
from dashboard.sections.detail import render_detail_panel
from dashboard.sections.influence import render_influence_section
from dashboard.sections.risk import render_risk_section
from dashboard.sections.full_leaderboard import render_full_leaderboard


def main() -> None:
    snap = load_snapshot()

    eng_raw_df    = build_engineers_df(snap["engineers"])
    ts_df         = build_timeseries_df(snap.get("time_series", {}))
    collab_raw    = snap.get("collaboration_graph", [])
    hot_paths     = set(snap.get("hot_paths", []))
    all_engineers = set(eng_raw_df["engineer"].tolist())

    # Feature A: graph metrics
    pr_scores, bc_scores, G = compute_graph_metrics(collab_raw)
    infl_scores = collab_influence_score(pr_scores, bc_scores, all_engineers)

    # Feature C: trajectories
    trajectories = {eng: compute_trajectory(ts_df, eng) for eng in all_engineers}

    # Sidebar controls
    w_d, w_l, w_c, w_r, risk_adj, min_prs = render_sidebar(snap)

    # Filter + recompute scores
    filtered_df = eng_raw_df[eng_raw_df["pr_count"] >= min_prs].copy()
    scored_df   = recompute_scores(filtered_df, w_d, w_l, w_c, w_r, risk_adj, infl_scores)

    # Annotate scored_df
    scored_df["trajectory"] = scored_df["engineer"].map(
        lambda e: trajectories.get(e, {}).get("label", "-> Stabilizing")
    )
    scored_df["trend_arrow"] = scored_df["trajectory"].apply(trend_arrow)
    scored_df["primary_strength"] = scored_df.apply(
        lambda row: max(
            {
                "Delivery":      row.delivery,
                "Leverage":      row.leverage,
                "Collaboration": row.collaboration,
                "Reliability":   row.reliability,
            },
            key=lambda k: {
                "Delivery":      row.delivery,
                "Leverage":      row.leverage,
                "Collaboration": row.collaboration,
                "Reliability":   row.reliability,
            }[k],
        ),
        axis=1,
    )
    scored_df["influence"] = scored_df["engineer"].map(lambda e: infl_scores.get(e, 0.0))

    # Header
    st.title("⚡ Engineering Impact Dashboard")
    st.caption(
        f"PostHog/posthog · Last {snap.get('window_days', 90)} days · "
        f"{len(scored_df)} engineers · {snap.get('total_prs_analyzed', '?')} merged PRs"
    )

    # Sections
    render_leaderboard(scored_df, trajectories, snap, ts_df)
    render_detail_panel(scored_df, snap, ts_df, G, trajectories)
    render_influence_section(scored_df, G, pr_scores, bc_scores)
    render_risk_section(scored_df, eng_raw_df, hot_paths)
    render_full_leaderboard(scored_df)


if __name__ == "__main__":
    main()
