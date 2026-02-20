#!/usr/bin/env python3
"""
generate_impact.py

Pulls the last 90 days of GitHub data for PostHog/posthog via GraphQL API,
caches locally (cache/ directory, 4-hour TTL), and computes 4-dimension
engineering impact scores per engineer, plus time-series and collaboration
graph data.

Output: impact_snapshot.json

Usage:
    export GITHUB_TOKEN=ghp_...
    python generate_impact.py
"""

import os
import json
import math
import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import defaultdict
from typing import Optional

import requests
import pandas as pd

# ── Config ───────────────────────────────────────────────────────────────────
REPO_OWNER  = "PostHog"
REPO_NAME   = "posthog"
WINDOW_DAYS = 90
CACHE_DIR   = Path("cache")
CACHE_TTL   = 3600 * 4          # 4 hours
OUTPUT_FILE = "impact_snapshot.json"
GITHUB_API  = "https://api.github.com/graphql"
REQUEST_DELAY = 0.4             # seconds between paginated requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# GitHub GraphQL client
# ─────────────────────────────────────────────────────────────────────────────

def get_token() -> str:
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        raise EnvironmentError(
            "GITHUB_TOKEN environment variable is required but not set. "
            "Create a token at https://github.com/settings/tokens (read:org + repo scopes)."
        )
    return token


def gql(query: str, variables: dict, token: str) -> dict:
    """Execute a GraphQL query against GitHub API."""
    resp = requests.post(
        GITHUB_API,
        json={"query": query, "variables": variables},
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-Github-Next-Global-ID": "1",
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL errors: {data['errors']}")
    return data["data"]


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{key}.json"


def cache_load(key: str) -> Optional[list | dict]:
    p = cache_path(key)
    if not p.exists():
        return None
    age = time.time() - p.stat().st_mtime
    if age > CACHE_TTL:
        log.info(f"Cache expired for '{key}' (age={age/3600:.1f}h)")
        return None
    with p.open() as f:
        return json.load(f)


def cache_save(key: str, data) -> None:
    with cache_path(key).open("w") as f:
        json.dump(data, f)


# ─────────────────────────────────────────────────────────────────────────────
# GraphQL queries
# ─────────────────────────────────────────────────────────────────────────────

# Fetches merged PRs with files, reviews, labels
PR_QUERY = """
query FetchMergedPRs($q: String!, $after: String) {
  search(query: $q, type: ISSUE, first: 50, after: $after) {
    pageInfo { hasNextPage endCursor }
    nodes {
      ... on PullRequest {
        number
        title
        body
        author { login }
        createdAt
        mergedAt
        additions
        deletions
        changedFiles
        files(first: 100) {
          pageInfo { hasNextPage }
          nodes { path }
        }
        labels(first: 20) {
          nodes { name }
        }
        reviews(first: 100) {
          nodes {
            author { login }
            state
            submittedAt
          }
        }
        timelineItems(itemTypes: [PULL_REQUEST_COMMIT], first: 100) {
          nodes {
            ... on PullRequestCommit {
              commit {
                committedDate
                additions
                deletions
              }
            }
          }
        }
      }
    }
  }
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Data fetching
# ─────────────────────────────────────────────────────────────────────────────

def fetch_prs(token: str) -> list[dict]:
    """Fetch all merged PRs in the last WINDOW_DAYS days, with pagination."""
    cached = cache_load("prs")
    if cached is not None:
        log.info(f"Loaded {len(cached)} PRs from cache")
        return cached

    since = (
        datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)
    ).strftime("%Y-%m-%d")
    search_q = (
        f"repo:{REPO_OWNER}/{REPO_NAME} is:pr is:merged merged:>={since}"
    )

    all_prs: list[dict] = []
    after: Optional[str] = None
    page = 0

    while True:
        page += 1
        log.info(f"  Fetching PR page {page} (collected {len(all_prs)} so far)…")
        try:
            data = gql(PR_QUERY, {"q": search_q, "after": after}, token)
        except requests.HTTPError as exc:
            log.error(f"HTTP error on page {page}: {exc}")
            break

        search_result = data["search"]
        nodes = search_result["nodes"]
        # GraphQL can return null nodes in union types – filter them
        all_prs.extend([n for n in nodes if n and "number" in n])

        if not search_result["pageInfo"]["hasNextPage"]:
            break
        after = search_result["pageInfo"]["endCursor"]
        time.sleep(REQUEST_DELAY)

    log.info(f"Fetched {len(all_prs)} merged PRs total")
    cache_save("prs", all_prs)
    return all_prs


# ─────────────────────────────────────────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────────────────────────────────────────

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def log1p_norm(value: float, cap: float) -> float:
    return math.log1p(min(value, cap)) / math.log1p(cap)


def parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


# ─────────────────────────────────────────────────────────────────────────────
# Dimension 1 – Delivery
# ─────────────────────────────────────────────────────────────────────────────

def delivery_contribution(pr: dict) -> float:
    """Compute DeliveryContribution for a single merged PR (0..1)."""
    churn = pr.get("additions", 0) + pr.get("deletions", 0)
    files = pr.get("changedFiles", 0)

    created_at = parse_dt(pr.get("createdAt"))
    merged_at  = parse_dt(pr.get("mergedAt"))
    if created_at and merged_at:
        ttm_hours = (merged_at - created_at).total_seconds() / 3600.0
    else:
        ttm_hours = 72.0  # neutral assumption

    reviews = pr.get("reviews", {}).get("nodes", []) or []

    # Rework: any CHANGES_REQUESTED review
    rework_flag = int(
        any(r.get("state") == "CHANGES_REQUESTED" for r in reviews if r)
    )

    # StabilityAfterReview: approximate via post-review commit churn
    # We use commit timeline to split churn before/after first review.
    stability = _compute_stability_after_review(pr, reviews)

    # --- Normalized components ---
    churn_score  = log1p_norm(churn, 2000)
    file_score   = log1p_norm(files, 50)
    speed_score  = sigmoid((72 - min(ttm_hours, 240)) / 24)
    rework_score = 1.0 - rework_flag

    base = (
        0.35 * churn_score
        + 0.20 * file_score
        + 0.25 * speed_score
        + 0.20 * rework_score
    )
    multiplier = 0.85 + 0.15 * stability
    return clamp(base * multiplier)


def _compute_stability_after_review(pr: dict, reviews: list[dict]) -> float:
    """
    Estimate StabilityAfterReview.
    Uses commit timeline items to find additions+deletions of commits
    that occurred AFTER the first review submission.
    Falls back to 0.5 (neutral) if data is insufficient.
    """
    commits = []
    for node in (pr.get("timelineItems", {}).get("nodes", []) or []):
        if node and "commit" in node:
            c = node["commit"]
            dt = parse_dt(c.get("committedDate"))
            if dt:
                commits.append({
                    "date": dt,
                    "churn": c.get("additions", 0) + c.get("deletions", 0),
                })

    review_times = sorted(
        [parse_dt(r.get("submittedAt")) for r in reviews if r and r.get("submittedAt")],
        key=lambda d: d or datetime.min.replace(tzinfo=timezone.utc),
    )

    if not review_times or not commits:
        # No review or no commit data → neutral
        return 0.5

    first_review_dt = review_times[0]
    total_churn = sum(c["churn"] for c in commits)
    post_review_churn = sum(
        c["churn"] for c in commits if c["date"] > first_review_dt
    )

    if total_churn == 0:
        return 1.0

    ratio = post_review_churn / total_churn
    return clamp(1.0 - ratio)


def compute_delivery(prs_by_author: dict[str, list[dict]]) -> dict[str, float]:
    """Delivery score 0..100 per engineer."""
    scores: dict[str, float] = {}
    for author, prs in prs_by_author.items():
        if not prs:
            scores[author] = 0.0
            continue
        contributions = [delivery_contribution(p) for p in prs]
        scores[author] = 100.0 * (sum(contributions) / len(contributions))
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Dimension 2 – Leverage
# ─────────────────────────────────────────────────────────────────────────────

def compute_hot_paths(all_prs: list[dict]) -> set[str]:
    """
    Returns the top-10% directories by PR-touch frequency.
    A 'directory' is the first component of each file path.
    """
    dir_counts: dict[str, int] = defaultdict(int)
    for pr in all_prs:
        touched_dirs = {
            f["path"].split("/")[0]
            for f in (pr.get("files", {}).get("nodes", []) or [])
            if f and f.get("path")
        }
        for d in touched_dirs:
            dir_counts[d] += 1

    if not dir_counts:
        return set()

    sorted_dirs = sorted(dir_counts.items(), key=lambda x: -x[1])
    top10_cutoff = max(1, math.ceil(len(sorted_dirs) * 0.10))
    return {d for d, _ in sorted_dirs[:top10_cutoff]}


def build_pr_mention_map(all_prs: list[dict]) -> dict[int, int]:
    """
    Returns {pr_number: downstream_mention_count} for PRs that
    are referenced by body/title of later PRs within 30 days of merge.
    """
    # Build lookup {pr_number -> merged_at}
    merged_at_map: dict[int, datetime] = {}
    for pr in all_prs:
        dt = parse_dt(pr.get("mergedAt"))
        if dt:
            merged_at_map[pr["number"]] = dt

    mention_counts: dict[int, int] = defaultdict(int)

    for pr in all_prs:
        created = parse_dt(pr.get("createdAt"))
        text = (pr.get("title") or "") + " " + (pr.get("body") or "")
        # Find #N references
        import re
        refs = {int(m) for m in re.findall(r"#(\d+)", text)}
        for ref_num in refs:
            if ref_num == pr["number"]:
                continue
            if ref_num in merged_at_map:
                origin_merge = merged_at_map[ref_num]
                if created and created >= origin_merge:
                    delta_days = (created - origin_merge).days
                    if delta_days <= 30:
                        mention_counts[ref_num] += 1

    return mention_counts


def pr_leverage(
    pr: dict,
    hot_paths: set[str],
    mention_counts: dict[int, int],
) -> tuple[float, float]:
    """Returns (CoreScore, RefScore) for a single PR."""
    files = [
        f["path"]
        for f in (pr.get("files", {}).get("nodes", []) or [])
        if f and f.get("path")
    ]
    if files:
        core_touch = sum(
            1 for f in files if f.split("/")[0] in hot_paths
        ) / len(files)
    else:
        core_touch = 0.0

    downstream = min(mention_counts.get(pr["number"], 0), 5)
    ref_score = log1p_norm(downstream, 5)

    return core_touch, ref_score


def compute_leverage(
    prs_by_author: dict[str, list[dict]],
    hot_paths: set[str],
    mention_counts: dict[int, int],
) -> dict[str, float]:
    """Leverage score 0..100 per engineer."""
    scores: dict[str, float] = {}
    for author, prs in prs_by_author.items():
        if not prs:
            scores[author] = 0.0
            continue
        core_scores, ref_scores = [], []
        for pr in prs:
            c, r = pr_leverage(pr, hot_paths, mention_counts)
            core_scores.append(c)
            ref_scores.append(r)
        avg_core = sum(core_scores) / len(core_scores)
        avg_ref  = sum(ref_scores)  / len(ref_scores)
        scores[author] = 100.0 * (0.65 * avg_core + 0.35 * avg_ref)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Dimension 3 – Collaboration
# ─────────────────────────────────────────────────────────────────────────────

def compute_collaboration(
    all_prs: list[dict],
    prs_by_author: dict[str, list[dict]],
) -> dict[str, float]:
    """
    Collaboration score 0..100 per engineer.

    For each engineer we compute:
      ReviewScore   – reviews they gave on *other* merged PRs
      UnblockScore  – how fast their first-review versus repo median
      CrossTeamScore – how often their authored PRs touch unfamiliar dirs
    """
    # Repo-wide first-review latency (hours) for all PRs
    repo_first_review_hours: list[float] = []
    # For each PR: list of reviews sorted by time
    for pr in all_prs:
        created = parse_dt(pr.get("createdAt"))
        if not created:
            continue
        reviews = sorted(
            [r for r in (pr.get("reviews", {}).get("nodes", []) or []) if r and r.get("submittedAt")],
            key=lambda r: parse_dt(r["submittedAt"]) or datetime.min.replace(tzinfo=timezone.utc),
        )
        # Exclude self-reviews
        author_login = (pr.get("author") or {}).get("login", "")
        non_self = [r for r in reviews if (r.get("author") or {}).get("login") != author_login]
        if non_self:
            first_dt = parse_dt(non_self[0]["submittedAt"])
            if first_dt:
                repo_first_review_hours.append(
                    (first_dt - created).total_seconds() / 3600.0
                )

    repo_median_first = (
        float(pd.Series(repo_first_review_hours).median())
        if repo_first_review_hours
        else 24.0
    )

    # Historical top dirs per author (for CrossArea)
    author_dir_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for author, prs in prs_by_author.items():
        for pr in prs:
            for f in (pr.get("files", {}).get("nodes", []) or []):
                if f and f.get("path"):
                    d = f["path"].split("/")[0]
                    author_dir_counts[author][d] += 1

    def top3_dirs(author: str) -> set[str]:
        counts = author_dir_counts[author]
        sorted_dirs = sorted(counts.items(), key=lambda x: -x[1])
        return {d for d, _ in sorted_dirs[:3]}

    # Per-engineer accumulators
    review_acc:   dict[str, list[float]] = defaultdict(list)   # review scores
    unblock_acc:  dict[str, list[float]] = defaultdict(list)   # unblock deltas
    cross_acc:    dict[str, list[float]] = defaultdict(list)   # cross-area (own PRs)

    for pr in all_prs:
        author_login = (pr.get("author") or {}).get("login", "")
        created = parse_dt(pr.get("createdAt"))
        if not created:
            continue

        reviews = sorted(
            [r for r in (pr.get("reviews", {}).get("nodes", []) or []) if r and r.get("submittedAt")],
            key=lambda r: parse_dt(r["submittedAt"]) or datetime.min.replace(tzinfo=timezone.utc),
        )
        non_self_reviews = [
            r for r in reviews
            if (r.get("author") or {}).get("login") != author_login
        ]

        # ReviewScore for each reviewer
        for r in non_self_reviews:
            reviewer = (r.get("author") or {}).get("login")
            if not reviewer:
                continue
            review_dt = parse_dt(r["submittedAt"])
            if review_dt:
                hours_from_open = (review_dt - created).total_seconds() / 3600.0
                early_bonus = sigmoid((24 - hours_from_open) / 6)
                review_acc[reviewer].append(0.7 + 0.3 * early_bonus)
            else:
                review_acc[reviewer].append(0.7)

        # UnblockScore – first non-self reviewer
        if non_self_reviews:
            first_reviewer_login = (non_self_reviews[0].get("author") or {}).get("login")
            if first_reviewer_login:
                first_dt = parse_dt(non_self_reviews[0]["submittedAt"])
                if first_dt:
                    their_hours = (first_dt - created).total_seconds() / 3600.0
                    delta = clamp(
                        (repo_median_first - their_hours) / max(repo_median_first, 1),
                        -1.0, 1.0,
                    )
                    unblock_acc[first_reviewer_login].append(delta)

        # CrossArea – for the PR author's own merged PRs
        own_top3 = top3_dirs(author_login)
        pr_files = [
            f["path"] for f in (pr.get("files", {}).get("nodes", []) or []) if f and f.get("path")
        ]
        if pr_files and author_login:
            outside = sum(1 for f in pr_files if f.split("/")[0] not in own_top3)
            cross_flag = 1.0 if outside / len(pr_files) > 0.5 else 0.0
            cross_acc[author_login].append(cross_flag)

    # Aggregate
    all_engineers = set(prs_by_author.keys()) | set(review_acc.keys())
    scores: dict[str, float] = {}
    for eng in all_engineers:
        review_scores = review_acc.get(eng, [])
        unblock_deltas = unblock_acc.get(eng, [])
        cross_flags    = cross_acc.get(eng, [])

        ReviewScore = (
            sum(review_scores) / len(review_scores) if review_scores else 0.0
        )
        UnblockScore = (
            0.5 + 0.5 * (sum(unblock_deltas) / len(unblock_deltas))
            if unblock_deltas
            else 0.5  # neutral if no first-review data
        )
        CrossTeamScore = (
            sum(cross_flags) / len(cross_flags) if cross_flags else 0.0
        )

        raw = (
            0.50 * ReviewScore
            + 0.30 * UnblockScore
            + 0.20 * CrossTeamScore
        )
        scores[eng] = 100.0 * clamp(raw)

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Dimension 4 – Reliability
# ─────────────────────────────────────────────────────────────────────────────

_BUG_TITLE_RE = None

def _is_bugfix(pr: dict) -> bool:
    """True if PR looks like a bugfix (label or title keywords)."""
    global _BUG_TITLE_RE
    if _BUG_TITLE_RE is None:
        import re
        _BUG_TITLE_RE = re.compile(r"\b(fix|bug|regression|hotfix|patch)\b", re.IGNORECASE)
    labels = {lbl["name"].lower() for lbl in (pr.get("labels", {}).get("nodes", []) or []) if lbl}
    if "bug" in labels or "fix" in labels:
        return True
    title = pr.get("title") or ""
    return bool(_BUG_TITLE_RE.search(title))


def _is_revert(pr: dict) -> bool:
    """True if this PR is a revert."""
    title = (pr.get("title") or "").lower()
    return title.startswith("revert")


def compute_reliability(
    all_prs: list[dict],
    prs_by_author: dict[str, list[dict]],
) -> dict[str, float]:
    """Reliability score 0..100 per engineer."""
    import re

    # Build {pr_number -> {merged_at, author, files, is_bugfix}}
    pr_meta: dict[int, dict] = {}
    for pr in all_prs:
        files = {
            f["path"]
            for f in (pr.get("files", {}).get("nodes", []) or [])
            if f and f.get("path")
        }
        pr_meta[pr["number"]] = {
            "author":     (pr.get("author") or {}).get("login", ""),
            "merged_at":  parse_dt(pr.get("mergedAt")),
            "created_at": parse_dt(pr.get("createdAt")),
            "files":      files,
            "is_bugfix":  _is_bugfix(pr),
            "is_revert":  _is_revert(pr),
            "title":      pr.get("title") or "",
            "body":       pr.get("body") or "",
        }

    # For each engineer's PR, count:
    #   RevertHit      – revert PRs that reference it (within 30 days)
    #   PostMergeFixHit – bug PRs within 14 days touching ≥1 same file

    # Pre-build reverse indexes for efficiency
    # revert PRs and their referenced PR numbers
    revert_refs: list[tuple[datetime, int, set[int]]] = []
    bugfix_prs_sorted: list[dict] = []

    for pr in all_prs:
        meta = pr_meta[pr["number"]]
        if not meta["merged_at"]:
            continue
        if meta["is_revert"]:
            refs = {int(m) for m in re.findall(r"#(\d+)", meta["title"] + " " + meta["body"])}
            revert_refs.append((meta["merged_at"], pr["number"], refs - {pr["number"]}))
        if meta["is_bugfix"]:
            bugfix_prs_sorted.append({
                "number":    pr["number"],
                "merged_at": meta["merged_at"],
                "files":     meta["files"],
            })

    def count_revert_hits(pr_num: int, merged_at: datetime) -> int:
        count = 0
        for revert_dt, _, refs in revert_refs:
            if pr_num in refs and revert_dt > merged_at:
                delta = (revert_dt - merged_at).days
                if delta <= 30:
                    count += 1
            if count >= 2:
                break
        return min(count, 2)

    def count_postmerge_fixes(merged_at: datetime, files: set[str]) -> int:
        count = 0
        for bp in bugfix_prs_sorted:
            bm = bp["merged_at"]
            if not bm or bm <= merged_at:
                continue
            delta = (bm - merged_at).days
            if delta > 14:
                continue
            if files & bp["files"]:
                count += 1
            if count >= 3:
                break
        return min(count, 3)

    scores: dict[str, float] = {}
    for author, prs in prs_by_author.items():
        if not prs:
            scores[author] = 55.0
            continue

        total = len(prs)
        bugfix_count = sum(1 for p in prs if _is_bugfix(p))
        bugfix_share = bugfix_count / total

        revert_total    = 0
        postmerge_total = 0

        for pr in prs:
            merged_at = pr_meta[pr["number"]]["merged_at"]
            files     = pr_meta[pr["number"]]["files"]
            if merged_at:
                revert_total    += count_revert_hits(pr["number"], merged_at)
                postmerge_total += count_postmerge_fixes(merged_at, files)

        revert_penalty  = log1p_norm(min(revert_total, 2 * total), 2 * total) if total else 0
        fallout_penalty = log1p_norm(min(postmerge_total, 3 * total), 3 * total) if total else 0

        # Normalize penalties relative to PR count (per-PR averages)
        avg_revert  = log1p_norm(revert_total / total, 2)
        avg_fallout = log1p_norm(postmerge_total / total, 3)

        raw = clamp(
            0.55
            + 0.15 * clamp(bugfix_share / 0.4)
            - 0.20 * avg_revert
            - 0.20 * avg_fallout,
        )
        scores[author] = 100.0 * raw

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Overall Impact score
# ─────────────────────────────────────────────────────────────────────────────

def compute_impact(
    delivery: dict[str, float],
    leverage: dict[str, float],
    collaboration: dict[str, float],
    reliability: dict[str, float],
) -> dict[str, float]:
    """Compute overall Impact score 0..100 for each engineer."""
    all_engineers = (
        set(delivery) | set(leverage) | set(collaboration) | set(reliability)
    )
    scores: dict[str, float] = {}
    for eng in all_engineers:
        D = delivery.get(eng, 0.0) / 100.0
        L = leverage.get(eng, 0.0) / 100.0
        C = collaboration.get(eng, 0.0) / 100.0
        R = reliability.get(eng, 0.0) / 100.0

        base_impact = 0.35 * D + 0.25 * L + 0.20 * C + 0.20 * R
        product = D * L * C * R
        balance = product ** 0.25 if product > 0 else 0.0
        impact = base_impact * (0.70 + 0.30 * balance)
        scores[eng] = clamp(impact * 100, 0.0, 100.0)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Time-series computation
# ─────────────────────────────────────────────────────────────────────────────

def _iso_week_start(dt: datetime) -> str:
    """Return ISO week-start Monday as YYYY-MM-DD."""
    monday = dt - timedelta(days=dt.weekday())
    return monday.strftime("%Y-%m-%d")


def compute_time_series(
    all_prs: list[dict],
    hot_paths: set[str],
    mention_counts: dict[int, int],
) -> dict[str, list[dict]]:
    """
    Returns per-engineer weekly time series:
      week, delivery, collaboration, reliability, leverage,
      impact, risk_weighted_prs, reverts_issues
    """
    # Build a map of PR number → PR for fast lookup
    pr_by_num = {pr["number"]: pr for pr in all_prs}

    # Group PRs by (author, week)
    weekly_prs: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for pr in all_prs:
        author = (pr.get("author") or {}).get("login", "")
        merged_at = parse_dt(pr.get("mergedAt"))
        if author and merged_at:
            week = _iso_week_start(merged_at)
            weekly_prs[(author, week)].append(pr)

    # Gather all unique (author, week) combos and all weeks
    all_weeks = sorted({w for _, w in weekly_prs})
    all_authors = sorted({a for a, _ in weekly_prs})

    result: dict[str, list[dict]] = {}

    for author in all_authors:
        series = []
        for week in all_weeks:
            week_prs = weekly_prs.get((author, week), [])
            if not week_prs:
                continue

            prs_by_a = {author: week_prs}

            # Delivery
            d_scores  = compute_delivery(prs_by_a)
            d_val     = d_scores.get(author, 0.0)

            # Leverage (use global hot_paths / mention_counts)
            l_scores  = compute_leverage(prs_by_a, hot_paths, mention_counts)
            l_val     = l_scores.get(author, 0.0)

            # Collaboration (needs broader PR context for repo medians; approximate with week slice)
            c_scores  = compute_collaboration(week_prs, prs_by_a)
            c_val     = c_scores.get(author, 0.0)

            # Reliability (needs broader context for revert/fallout lookup)
            r_scores  = compute_reliability(all_prs, prs_by_a)
            r_val     = r_scores.get(author, 0.0)

            # Overall impact
            imp_scores = compute_impact(
                {author: d_val}, {author: l_val},
                {author: c_val}, {author: r_val}
            )
            imp_val = imp_scores.get(author, 0.0)

            # Risk-weighted PR volume: sum of churn_score across week PRs
            risk_vol = sum(
                log1p_norm(
                    p.get("additions", 0) + p.get("deletions", 0), 2000
                )
                for p in week_prs
            )

            # Reverts/issues this week (revert PRs merged this week by anyone referencing author's PRs)
            author_pr_nums = {p["number"] for p in week_prs}
            import re
            rev_issues = 0
            for pr in all_prs:
                if not _is_revert(pr):
                    continue
                pr_merged = parse_dt(pr.get("mergedAt"))
                if not pr_merged:
                    continue
                if _iso_week_start(pr_merged) != week:
                    continue
                refs = {
                    int(m)
                    for m in re.findall(r"#(\d+)", (pr.get("title") or "") + " " + (pr.get("body") or ""))
                }
                if refs & author_pr_nums:
                    rev_issues += 1

            series.append({
                "week":              week,
                "delivery":          round(d_val, 2),
                "collaboration":     round(c_val, 2),
                "reliability":       round(r_val, 2),
                "leverage":          round(l_val, 2),
                "impact":            round(imp_val, 2),
                "risk_weighted_prs": round(risk_vol, 3),
                "reverts_issues":    rev_issues,
                "pr_count":          len(week_prs),
            })

        if series:
            result[author] = series

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Collaboration graph
# ─────────────────────────────────────────────────────────────────────────────

def compute_collab_graph(all_prs: list[dict]) -> list[dict]:
    """
    Returns edges: [{reviewer, author, weight}]
    Edge exists when engineer A reviewed a merged PR by engineer B.
    weight = number of such reviews.
    """
    edge_counts: dict[tuple[str, str], int] = defaultdict(int)

    for pr in all_prs:
        author = (pr.get("author") or {}).get("login", "")
        if not author:
            continue
        reviews = pr.get("reviews", {}).get("nodes", []) or []
        for r in reviews:
            if not r:
                continue
            reviewer = (r.get("author") or {}).get("login", "")
            if reviewer and reviewer != author:
                edge_counts[(reviewer, author)] += 1

    return [
        {"reviewer": rev, "author": aut, "weight": w}
        for (rev, aut), w in sorted(edge_counts.items(), key=lambda x: -x[1])
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    token = get_token()

    log.info(f"Pulling last {WINDOW_DAYS} days of merged PRs for {REPO_OWNER}/{REPO_NAME}…")
    all_prs = fetch_prs(token)

    if not all_prs:
        log.error("No PRs fetched. Check your GITHUB_TOKEN and network access.")
        return

    log.info("Building per-author PR index…")
    prs_by_author: dict[str, list[dict]] = defaultdict(list)
    for pr in all_prs:
        author = (pr.get("author") or {}).get("login", "")
        if author:
            prs_by_author[author].append(pr)

    # Filter to authors with at least 2 merged PRs in the window
    prs_by_author = {
        a: prs for a, prs in prs_by_author.items() if len(prs) >= 2
    }
    log.info(f"Active engineers (≥2 merged PRs): {len(prs_by_author)}")

    log.info("Computing hot paths…")
    hot_paths = compute_hot_paths(all_prs)
    log.info(f"  Hot paths set size: {len(hot_paths)} dirs")

    log.info("Building PR mention map (downstream references)…")
    mention_counts = build_pr_mention_map(all_prs)

    log.info("Computing Delivery scores…")
    delivery_scores = compute_delivery(prs_by_author)

    log.info("Computing Leverage scores…")
    leverage_scores = compute_leverage(prs_by_author, hot_paths, mention_counts)

    log.info("Computing Collaboration scores…")
    collab_scores = compute_collaboration(all_prs, prs_by_author)

    log.info("Computing Reliability scores…")
    reliability_scores = compute_reliability(all_prs, prs_by_author)

    log.info("Computing overall Impact scores…")
    impact_scores = compute_impact(
        delivery_scores, leverage_scores, collab_scores, reliability_scores
    )

    log.info("Computing time-series data…")
    time_series = compute_time_series(all_prs, hot_paths, mention_counts)

    log.info("Building collaboration graph…")
    collab_graph = compute_collab_graph(all_prs)

    # ── Assemble output ──────────────────────────────────────────────────────
    all_engineers = set(prs_by_author.keys())

    engineers_out: dict[str, dict] = {}
    for eng in sorted(all_engineers):
        engineers_out[eng] = {
            "delivery":      round(delivery_scores.get(eng, 0.0), 2),
            "leverage":      round(leverage_scores.get(eng, 0.0), 2),
            "collaboration": round(collab_scores.get(eng, 0.0), 2),
            "reliability":   round(reliability_scores.get(eng, 0.0), 2),
            "impact":        round(impact_scores.get(eng, 0.0), 2),
            "pr_count":      len(prs_by_author.get(eng, [])),
            "review_count":  sum(
                1 for pr in all_prs
                for r in (pr.get("reviews", {}).get("nodes", []) or [])
                if r and (r.get("author") or {}).get("login") == eng
                and (pr.get("author") or {}).get("login") != eng
            ),
        }

    snapshot = {
        "generated_at":       datetime.now(timezone.utc).isoformat(),
        "repo":               f"{REPO_OWNER}/{REPO_NAME}",
        "window_days":        WINDOW_DAYS,
        "total_prs_analyzed": len(all_prs),
        "hot_paths":          sorted(hot_paths),
        "engineers":          engineers_out,
        "time_series":        time_series,
        "collaboration_graph": collab_graph,
    }

    # ── Save output ──────────────────────────────────────────────────────────
    output_path = Path(OUTPUT_FILE)
    with output_path.open("w") as f:
        json.dump(snapshot, f, indent=2, default=str)

    log.info(f"✓ Saved {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
    log.info(f"  Engineers scored: {len(engineers_out)}")
    log.info(f"  PRs analyzed:     {len(all_prs)}")

    # ── Quick summary table ──────────────────────────────────────────────────
    rows = [
        {
            "engineer":      eng,
            "impact":        v["impact"],
            "delivery":      v["delivery"],
            "leverage":      v["leverage"],
            "collaboration": v["collaboration"],
            "reliability":   v["reliability"],
            "prs":           v["pr_count"],
        }
        for eng, v in engineers_out.items()
    ]
    df = pd.DataFrame(rows).sort_values("impact", ascending=False).reset_index(drop=True)
    print("\n── Impact Rankings ──────────────────────────────────────────────────────")
    print(df.to_string(index=False, float_format="%.1f"))


if __name__ == "__main__":
    main()
