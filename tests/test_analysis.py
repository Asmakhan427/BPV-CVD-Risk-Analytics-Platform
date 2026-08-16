import numpy as np
import pandas as pd
import pytest

from bpv_cvd.analysis import (
    calculate_cv_risk,
    chi_square_association,
    compare_clusters,
    confidence_intervals,
    correlation_matrix,
    demographic_profiles,
)
from bpv_cvd.clustering import label_clusters_by_severity
from bpv_cvd.data_generator import generate_patient_data


@pytest.fixture(scope="module")
def df_with_clusters():
    df = generate_patient_data()
    df["cluster"] = df["true_cluster"]  # use generative ground truth for deterministic tests
    return df


def test_compare_clusters_returns_anova(df_with_clusters):
    result = compare_clusters(df_with_clusters, cluster_col="cluster")
    assert "p_value" in result.columns
    assert "f_statistic" in result.columns
    assert (result["p_value"] >= 0).all() and (result["p_value"] <= 1).all()


def test_calculate_cv_risk_rates(df_with_clusters):
    risk = calculate_cv_risk(df_with_clusters, cluster_col="cluster")
    assert set(risk["cluster"]) == {"High BPV", "Medium BPV", "Low BPV"}
    high_risk = risk[risk["cluster"] == "High BPV"]["risk_pct"].iloc[0]
    low_risk = risk[risk["cluster"] == "Low BPV"]["risk_pct"].iloc[0]
    assert high_risk > low_risk


def test_demographic_profiles(df_with_clusters):
    profiles = demographic_profiles(df_with_clusters, cluster_col="cluster")
    assert len(profiles) == 3
    assert "age_mean" in profiles.columns


def test_confidence_intervals_bounds():
    data = np.random.default_rng(0).normal(10, 2, 200)
    ci = confidence_intervals(data, n_boot=500)
    assert ci["ci_lower"] <= ci["estimate"] <= ci["ci_upper"]


def test_correlation_matrix_symmetric(df_with_clusters):
    corr = correlation_matrix(df_with_clusters)
    assert np.allclose(corr.values, corr.values.T, atol=1e-8)


def test_chi_square_association(df_with_clusters):
    result = chi_square_association(df_with_clusters, "cluster", "sex")
    assert "p_value" in result
    assert 0 <= result["p_value"] <= 1
