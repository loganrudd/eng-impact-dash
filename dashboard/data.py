"""Data loading and score-recomputation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st


@st.cache_data(ttl=300)
def load_snapshot(path: str = "impact_snapshot.json") -> dict:
    p = Path(path)
    if not p.exists():
        st.error(f"'{path}' not found. Run `python generate_impact.py` first.")
        st.stop()
    import json
    with p.open() as f:
        return json.load(f)


def build_engineers_df(engineers: dict) -> pd.DataFrame:
    rows = []
    for name, v in engineers.items():
        rows.append({
            "engineer":      name,
            "impact":        v["impact"],
            "delivery":      v["delivery"],
            "leverage":      v["leverage"],
            "collaboration": v["collaboration"],
            "reliability":   v["reliability"],
            "pr_count":      v.get("pr_count", 0),
            "review_count":  v.get("review_count", 0),
        })
    return pd.DataFrame(rows).sort_values("impact", ascending=False).reset_index(drop=True)


def build_timeseries_df(time_series: dict) -> pd.DataFrame:
    rows = []
    for eng, series in time_series.items():
        for entry in series:
            rows.append({"engineer": eng, **entry})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["week"] = pd.to_datetime(df["week"])
    return df


def recompute_scores(
    base_df: pd.DataFrame,
    w_delivery: float,
    w_leverage: float,
    w_collab: float,
    w_reliability: float,
    risk_adjusted: bool,
    influence_scores: Optional[dict] = None,
) -> pd.DataFrame:
    """Re-weight the base impact score given sidebar sliders."""
    df = base_df.copy()
    total_w = w_delivery + w_leverage + w_collab + w_reliability
    if total_w == 0:
        total_w = 1.0

    d = df["delivery"]    / 100
    l = df["leverage"]    / 100
    c = df["collaboration"] / 100
    r = df["reliability"] / 100

    base = (
        (w_delivery   / total_w) * d
        + (w_leverage / total_w) * l
        + (w_collab   / total_w) * c
        + (w_reliability / total_w) * r
    )
    product = d * l * c * r
    balance = product.apply(lambda p: p ** 0.25 if p > 0 else 0.0)
    df["custom_impact"] = (100 * base * (0.70 + 0.30 * balance)).clip(0, 100).round(2)

    if risk_adjusted and influence_scores:
        infl = pd.Series(influence_scores).reindex(df["engineer"]).fillna(0).values
        risk_factor = 1 + 0.10 * (infl / 100)
        df["custom_impact"] = (df["custom_impact"] * risk_factor).clip(0, 100).round(2)

    return df.sort_values("custom_impact", ascending=False).reset_index(drop=True)
