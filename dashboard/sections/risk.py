"""Feature B — Change Risk Model section."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.metrics import pr_risk_score


def render_risk_section(
    scored_df: pd.DataFrame,
    eng_raw_df: pd.DataFrame,
    hot_paths: set[str],
) -> None:
    st.markdown("---")
    st.markdown("## 🔥 Feature B — Change Risk Model")
    st.caption(
        "Per-engineer average PR risk score: large diff (30%) + many files (20%) + "
        "core-dir touch (25%) + late merge hour (10%) + high review count (15%). "
        "Engineers shown shipping high-risk changes with low regression rate."
    )

    risk_rows: list[dict] = []
    cache_path = Path("cache/prs.json")

    if cache_path.exists():
        with cache_path.open() as f:
            raw_prs = json.load(f)

        pr_risk_by_author: dict[str, list[float]] = defaultdict(list)
        for pr in raw_prs:
            author = (pr.get("author") or {}).get("login", "")
            if author:
                pr_risk_by_author[author].append(pr_risk_score(pr, hot_paths))

        for eng in scored_df["engineer"]:
            risks    = pr_risk_by_author.get(eng, [])
            avg_risk = sum(risks) / len(risks) if risks else 0.0
            rel_vals = eng_raw_df.loc[eng_raw_df["engineer"] == eng, "reliability"].values
            rel_val  = float(rel_vals[0]) if len(rel_vals) > 0 else 55.0
            imp_vals = scored_df.loc[scored_df["engineer"] == eng, "custom_impact"].values
            imp_val  = float(imp_vals[0]) if len(imp_vals) > 0 else 0.0
            risk_rows.append({
                "engineer":    eng,
                "avg_risk":    round(avg_risk * 100, 1),
                "reliability": rel_val,
                "impact":      imp_val,
                "pr_count":    len(risks),
            })
    else:
        for eng in scored_df["engineer"]:
            rel_vals = eng_raw_df.loc[eng_raw_df["engineer"] == eng, "reliability"].values
            rel_val  = float(rel_vals[0]) if len(rel_vals) > 0 else 55.0
            imp_vals = scored_df.loc[scored_df["engineer"] == eng, "custom_impact"].values
            imp_val  = float(imp_vals[0]) if len(imp_vals) > 0 else 0.0
            pr_c     = eng_raw_df.loc[eng_raw_df["engineer"] == eng, "pr_count"].values
            risk_rows.append({
                "engineer":    eng,
                "avg_risk":    0.0,
                "reliability": rel_val,
                "impact":      imp_val,
                "pr_count":    int(pr_c[0]) if len(pr_c) > 0 else 0,
            })
        st.warning(
            "cache/prs.json not found — risk scores approximated. "
            "Run `generate_impact.py` to refresh cache."
        )

    risk_df = pd.DataFrame(risk_rows).sort_values("avg_risk", ascending=False)

    fig_risk = px.scatter(
        risk_df,
        x="avg_risk",
        y="reliability",
        size="pr_count",
        color="impact",
        hover_name="engineer",
        color_continuous_scale="RdYlGn",
        labels={
            "avg_risk":    "Avg PR Risk Score (0–100)",
            "reliability": "Reliability Score",
            "impact":      "Impact",
        },
        title="High-Risk Shippers vs Regression Rate — bubble = PR count",
        height=380,
    )
    fig_risk.add_vline(
        x=risk_df["avg_risk"].median(), line_dash="dot",
        line_color="gray", annotation_text="Median risk",
    )
    fig_risk.add_hline(
        y=risk_df["reliability"].median(), line_dash="dot",
        line_color="gray", annotation_text="Median reliability",
    )
    fig_risk.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_gridcolor="#2a2a3a",
        yaxis_gridcolor="#2a2a3a",
        margin=dict(t=50, b=40, l=40, r=20),
    )
    st.plotly_chart(fig_risk, use_container_width=True, key="risk_scatter")

    # High-risk, low-regression table
    high_risk_good = (
        risk_df[risk_df["avg_risk"] >= risk_df["avg_risk"].median()]
        .sort_values("reliability", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    high_risk_good.index += 1
    st.markdown("**⭐ High-Risk Shippers With Low Regression Rate**")
    st.dataframe(
        high_risk_good[["engineer", "avg_risk", "reliability", "impact", "pr_count"]]
        .rename(columns={
            "avg_risk":    "Risk Score",
            "reliability": "Reliability",
            "impact":      "Impact",
            "pr_count":    "PRs",
        })
        .style
        .background_gradient(subset=["Risk Score"],  cmap="Reds")
        .background_gradient(subset=["Reliability"], cmap="Greens")
        .format({
            "Risk Score": "{:.1f}",
            "Reliability": "{:.1f}",
            "Impact": "{:.1f}",
        }),
        use_container_width=True,
        height=320,
    )
