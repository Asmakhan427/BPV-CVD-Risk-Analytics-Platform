import pytest

from bpv_cvd.dashboard_data import (
    build_full_pipeline,
    get_cluster_profile_matrix,
    get_cluster_summary,
    get_metrics_table,
    get_patient_data,
    get_prediction_results,
    get_timeline_data,
    predict_patient_by_id,
)


@pytest.fixture(scope="module")
def bundle():
    return build_full_pipeline(n=83, seed=42)


def test_bundle_has_expected_keys(bundle):
    for key in ["raw", "df", "fitted_algorithms", "metrics_df", "model_results", "deployed_model"]:
        assert key in bundle


def test_get_cluster_summary(bundle):
    summary = get_cluster_summary(bundle)
    assert len(summary) == 3
    assert "risk_pct" in summary.columns


def test_get_patient_data_lookup(bundle):
    all_patients = get_patient_data(bundle)
    pid = all_patients.iloc[0]["patient_id"]
    single = get_patient_data(bundle, pid)
    assert len(single) == 1


def test_get_metrics_table(bundle):
    metrics = get_metrics_table(bundle)
    assert len(metrics) == 4


def test_get_prediction_results(bundle):
    results = get_prediction_results(bundle)
    assert len(results) == 3
    assert (results["roc_auc"].dropna() <= 1).all()


def test_get_timeline_data(bundle):
    pid = bundle["df"].iloc[0]["patient_id"]
    traj = get_timeline_data(bundle, pid, n_sessions=8)
    assert len(traj) == 8


def test_get_cluster_profile_matrix(bundle):
    profile = get_cluster_profile_matrix(bundle)
    assert len(profile) == 3


def test_predict_patient_by_id(bundle):
    pid = bundle["df"].iloc[0]["patient_id"]
    result = predict_patient_by_id(bundle, pid)
    assert 0 <= result["probability"] <= 1
