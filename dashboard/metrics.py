"""
Metric computation:
  - Feature A: graph-based influence (PageRank + betweenness)
  - Feature B: per-PR change risk score
  - Feature C: trajectory analysis (slope / label / delta)
  - Utility: trend_arrow
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st


# ── Feature A ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def compute_graph_metrics(
    collab_graph: list[dict],
) -> tuple[dict, dict, nx.DiGraph]:
    """
    Build weighted directed graph (reviewer → author),
    compute PageRank and Betweenness Centrality.
    """
    G = nx.DiGraph()
    for edge in collab_graph:
        r, a, w = edge["reviewer"], edge["author"], edge["weight"]
        if G.has_edge(r, a):
            G[r][a]["weight"] += w
        else:
            G.add_edge(r, a, weight=w)

    pr = nx.pagerank(G, weight="weight", alpha=0.85)

    G_ud = G.to_undirected()
    bc = nx.betweenness_centrality(G_ud, weight="weight", normalized=True)

    return pr, bc, G


def collab_influence_score(
    pr_scores: dict,
    bc_scores: dict,
    engineers: set[str],
) -> dict[str, float]:
    """
    Combine PageRank (60%) + Betweenness (40%) → Collaboration Influence
    Score 0–100, restricted to engineers in the main dataset.
    """
    filtered_pr = {k: v for k, v in pr_scores.items() if k in engineers}
    filtered_bc = {k: v for k, v in bc_scores.items() if k in engineers}

    pr_max = max(filtered_pr.values(), default=1e-9)
    bc_max = max(filtered_bc.values(), default=1e-9)

    return {
        eng: round(
            100 * (
                0.60 * filtered_pr.get(eng, 0) / pr_max
                + 0.40 * filtered_bc.get(eng, 0) / bc_max
            ),
            2,
        )
        for eng in engineers
    }


# ── Feature B ─────────────────────────────────────────────────────────────────

def pr_risk_score(pr: dict, hot_paths: set[str]) -> float:
    """
    Heuristic risk score 0–1 per PR.
    Components: large diff (30%), many files (20%), core-dir touch (25%),
    late merge hour (10%), high review count (15%).
    """
    churn = pr.get("additions", 0) + pr.get("deletions", 0)
    files = pr.get("changedFiles", 0)

    churn_risk = math.log1p(min(churn, 5000)) / math.log1p(5000)
    file_risk  = math.log1p(min(files, 100))  / math.log1p(100)

    pr_dirs = {
        f["path"].split("/")[0]
        for f in (pr.get("files", {}).get("nodes", []) or [])
        if f and f.get("path")
    }
    core_risk = 1.0 if pr_dirs & hot_paths else 0.0

    late_hour = 0.0
    merged_at = pr.get("mergedAt", "")
    if merged_at:
        try:
            hour = int(merged_at[11:13])
            late_hour = 1.0 if (hour >= 22 or hour < 6) else 0.0
        except (ValueError, IndexError):
            pass

    review_count = len(pr.get("reviews", {}).get("nodes", []) or [])
    review_risk  = math.log1p(min(review_count, 10)) / math.log1p(10)

    risk = (
        0.30 * churn_risk
        + 0.20 * file_risk
        + 0.25 * core_risk
        + 0.10 * late_hour
        + 0.15 * review_risk
    )
    return round(min(max(risk, 0.0), 1.0), 4)


# ── Feature C ─────────────────────────────────────────────────────────────────

def compute_trajectory(ts_df: pd.DataFrame, engineer: str) -> dict:
    """
    Fit a linear trend on weekly impact scores.
    Returns {'slope': float, 'label': str, 'delta': float}.
    """
    eng_df = ts_df[ts_df["engineer"] == engineer].sort_values("week")
    if len(eng_df) < 2:
        return {"slope": 0.0, "label": "Stabilizing", "delta": 0.0}

    x = np.arange(len(eng_df), dtype=float)
    y = eng_df["impact"].values.astype(float)
    slope, _ = np.polyfit(x, y, 1)
    delta = float(y[-1] - y[0])

    if slope > 1.5:
        label = "🚀 Accelerating"
    elif slope < -1.5:
        label = "📉 Declining"
    else:
        label = "➡ Stabilizing"

    return {"slope": round(slope, 3), "label": label, "delta": round(delta, 2)}


def trend_arrow(label: str) -> str:
    if "Accelerating" in label:
        return "↑"
    if "Declining" in label:
        return "↓"
    return "→"
