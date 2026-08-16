"""
Statistical analysis utilities: cluster comparisons (ANOVA), cardiovascular
risk tables by cluster, demographic profiling, and bootstrap confidence
intervals.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

DEFAULT_FEATURES = [
    "age", "bmi", "sbp_mean", "sbpv", "dbp_mean", "dbpv",
    "pulse_pressure", "albumin", "hemoglobin", "crp",
]


def compare_clusters(df: pd.DataFrame, cluster_col: str = "cluster", features: list[str] | None = None) -> pd.DataFrame:
    """
    One-way ANOVA of each numeric feature across cluster groups.

    Returns a DataFrame with columns: feature, f_statistic, p_value,
    significant (p < 0.05), and per-cluster means.
    """
    features = features or [c for c in DEFAULT_FEATURES if c in df.columns]
    clusters = sorted(df[cluster_col].dropna().unique())
    rows = []
    for feat in features:
        groups = [df.loc[df[cluster_col] == c, feat].dropna() for c in clusters]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            continue
        f_stat, p_val = stats.f_oneway(*groups)
        row = {"feature": feat, "f_statistic": round(float(f_stat), 3), "p_value": round(float(p_val), 5),
               "significant": bool(p_val < 0.05)}
        for c in clusters:
            row[f"mean_{c}"] = round(float(df.loc[df[cluster_col] == c, feat].mean()), 2)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("p_value")


def calculate_cv_risk(df: pd.DataFrame, cluster_col: str = "cluster", outcome_col: str = "cv_risk_event") -> pd.DataFrame:
    """Cardiovascular risk rate (with counts and Wilson 95% CI) per cluster."""
    rows = []
    for c, sub in df.groupby(cluster_col):
        n = len(sub)
        events = int(sub[outcome_col].sum())
        rate = events / n if n else np.nan
        low, high = _wilson_ci(events, n)
        rows.append({
            "cluster": c, "n": n, "events": events,
            "risk_rate": round(rate, 4), "risk_pct": round(rate * 100, 1),
            "ci_lower_pct": round(low * 100, 1), "ci_upper_pct": round(high * 100, 1),
        })
    result = pd.DataFrame(rows)
    order = {"Low BPV": 0, "Medium BPV": 1, "High BPV": 2}
    if set(result["cluster"]) <= set(order):
        result["_order"] = result["cluster"].map(order)
        result = result.sort_values("_order").drop(columns="_order")
    return result.reset_index(drop=True)


def _wilson_ci(events: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = events / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def demographic_profiles(df: pd.DataFrame, cluster_col: str = "cluster") -> pd.DataFrame:
    """Per-cluster demographic and comorbidity summary table."""
    rows = []
    for c, sub in df.groupby(cluster_col):
        rows.append({
            "cluster": c,
            "n": len(sub),
            "age_mean": round(sub["age"].mean(), 1),
            "age_sd": round(sub["age"].std(), 1),
            "pct_female": round((sub["sex"] == "Female").mean() * 100, 1) if "sex" in sub else np.nan,
            "diabetes_pct": round(sub.get("diabetes", pd.Series(dtype=float)).mean() * 100, 1),
            "hypertension_pct": round(sub.get("hypertension", pd.Series(dtype=float)).mean() * 100, 1),
            "cad_pct": round(sub.get("coronary_artery_disease", pd.Series(dtype=float)).mean() * 100, 1),
            "stroke_pct": round(sub.get("prior_stroke", pd.Series(dtype=float)).mean() * 100, 1),
            "sbpv_mean": round(sub["sbpv"].mean(), 2) if "sbpv" in sub else np.nan,
            "dbpv_mean": round(sub["dbpv"].mean(), 2) if "dbpv" in sub else np.nan,
        })
    return pd.DataFrame(rows).reset_index(drop=True)


def confidence_intervals(data: pd.Series | np.ndarray, n_boot: int = 2000, ci: float = 0.95, seed: int = 42, stat=np.mean) -> dict:
    """
    Bootstrap confidence interval for an arbitrary summary statistic
    (mean by default) of a 1-D sample.
    """
    data = np.asarray(pd.Series(data).dropna())
    if len(data) == 0:
        return {"estimate": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "n_boot": n_boot}
    rng = np.random.default_rng(seed)
    boot_stats = np.array([
        stat(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)
    ])
    alpha = 1 - ci
    lower, upper = np.percentile(boot_stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "estimate": round(float(stat(data)), 4),
        "ci_lower": round(float(lower), 4),
        "ci_upper": round(float(upper), 4),
        "n_boot": n_boot,
        "confidence_level": ci,
    }


def bootstrap_cluster_ci(df: pd.DataFrame, cluster_col: str, feature: str, n_boot: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Bootstrap CI of the mean of `feature`, computed separately per cluster."""
    rows = []
    for c, sub in df.groupby(cluster_col):
        ci = confidence_intervals(sub[feature], n_boot=n_boot, seed=seed)
        rows.append({"cluster": c, "feature": feature, **ci})
    return pd.DataFrame(rows)


def correlation_matrix(df: pd.DataFrame, features: list[str] | None = None, method: str = "pearson") -> pd.DataFrame:
    """Correlation matrix over numeric features."""
    features = features or [c for c in DEFAULT_FEATURES if c in df.columns]
    return df[features].corr(method=method)


def chi_square_association(df: pd.DataFrame, cluster_col: str, categorical_col: str) -> dict:
    """Chi-square test of independence between cluster assignment and a categorical variable."""
    table = pd.crosstab(df[cluster_col], df[categorical_col])
    chi2, p, dof, _ = stats.chi2_contingency(table)
    return {"chi2": round(float(chi2), 3), "p_value": round(float(p), 5), "dof": int(dof), "significant": bool(p < 0.05)}
