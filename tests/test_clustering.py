import numpy as np
import pytest

from bpv_cvd.clustering import (
    ALGORITHMS,
    EMClustering,
    KMeansClustering,
    PAMClustering,
    WardClustering,
    compare_algorithms,
    label_clusters_by_severity,
    run_all_algorithms,
)
from bpv_cvd.data_generator import generate_patient_data
from bpv_cvd.preprocessing import scale_features

FEATURES = ["sbpv", "dbpv", "sbp_mean", "dbp_mean", "pulse_pressure"]


@pytest.fixture(scope="module")
def X():
    df = generate_patient_data()
    scaled, _ = scale_features(df, features=FEATURES)
    return scaled[FEATURES].values


@pytest.mark.parametrize("cls", [KMeansClustering, PAMClustering, WardClustering, EMClustering])
def test_each_algorithm_fits_and_predicts(cls, X):
    model = cls(n_clusters=3, random_state=42)
    model.fit(X)
    labels = model.predict()
    assert len(labels) == len(X)
    assert len(set(labels)) == 3


@pytest.mark.parametrize("cls", [KMeansClustering, PAMClustering, WardClustering, EMClustering])
def test_each_algorithm_evaluate_metrics(cls, X):
    model = cls(n_clusters=3, random_state=42)
    model.fit(X)
    metrics = model.evaluate()
    assert "silhouette_score" in metrics
    assert "davies_bouldin_score" in metrics
    assert -1 <= metrics["silhouette_score"] <= 1


def test_run_all_algorithms_returns_four(X):
    results = run_all_algorithms(X)
    assert set(results.keys()) == set(ALGORITHMS.keys())


def test_compare_algorithms_dataframe(X):
    df = compare_algorithms(X)
    assert len(df) == 4
    assert "algorithm" in df.columns


def test_label_clusters_by_severity_orders_correctly():
    labels = np.array([0, 0, 1, 1, 2, 2])
    sbpv = np.array([5, 5, 10, 10, 18, 18])
    dbpv = np.array([2, 2, 5, 5, 8, 8])
    named = label_clusters_by_severity(labels, sbpv, dbpv)
    assert named[0] == "Low BPV"
    assert named[2] == "Medium BPV"
    assert named[4] == "High BPV"


def test_ward_dendrogram_data(X):
    model = WardClustering(n_clusters=3, random_state=42)
    model.fit(X)
    data = model.dendrogram_data()
    assert "icoord" in data and "dcoord" in data
