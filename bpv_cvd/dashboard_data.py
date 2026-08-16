"""
Data-shaping layer consumed by the Streamlit dashboard (and the FastAPI
service). Framework-agnostic on purpose -- no `streamlit` import here, so
these functions stay independently testable; the dashboard wraps
`build_full_pipeline()` with `st.cache_resource` for speed.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import analysis, clustering, prediction, preprocessing
from .data_generator import generate_patient_data, validate_distribution

logger = logging.getLogger(__name__)

CLUSTER_FEATURES = ["sbpv", "dbpv", "sbp_mean", "dbp_mean", "pulse_pressure"]


def build_full_pipeline(n: int = 83, seed: int = 42, n_clusters: int = 3) -> dict:
    """
    Run the entire analytical pipeline once: generate data, cluster with all
    four algorithms (Ward is the canonical/primary assignment used
    throughout the dashboard, matching the paper's best-performing method),
    train all three predictive models, and package everything the dashboard
    needs into one bundle dict.
    """
    raw = generate_patient_data(n=n, seed=seed)
    validation_report = validate_distribution(raw)

    processed = preprocessing.build_feature_matrix(raw)
    scaled, scaler = preprocessing.scale_features(processed, features=CLUSTER_FEATURES)
    X_cluster = scaled[CLUSTER_FEATURES].values

    fitted_algorithms = clustering.run_all_algorithms(X_cluster, n_clusters=n_clusters, random_state=seed)
    metrics_df = pd.DataFrame([m.evaluate() for m in fitted_algorithms.values()])

    ward_model = fitted_algorithms["Ward"]
    cluster_names = clustering.label_clusters_by_severity(ward_model.labels_, raw["sbpv"].values, raw["dbpv"].values)

    df = raw.copy()
    df["cluster"] = cluster_names
    for algo_name, model in fitted_algorithms.items():
        df[f"cluster_{algo_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}"] = (
            clustering.label_clusters_by_severity(model.labels_, raw["sbpv"].values, raw["dbpv"].values)
        )

    # PCA projection (used by 3D scatter / 2D projection views)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3, random_state=seed)
    pca_coords = pca.fit_transform(X_cluster)
    df["pca1"], df["pca2"], df["pca3"] = pca_coords[:, 0], pca_coords[:, 1], pca_coords[:, 2]

    # Predictive models
    X_train, X_test, y_train, y_test, feature_cols = preprocessing.train_test_split_data(df, seed=seed)
    model_results = prediction.train_all_models(X_train, y_train, X_test, y_test, random_state=seed)

    # Fit best model (Random Forest) on full data for single-patient prediction in the UI
    full_processed = preprocessing.build_feature_matrix(df)
    X_full = full_processed[feature_cols]
    y_full = full_processed["cv_risk_event"]
    deployed_model = prediction.RandomForestRiskModel(random_state=seed)
    deployed_model.train_model(X_full, y_full)

    return {
        "raw": raw,
        "df": df,
        "full_processed": full_processed,
        "validation_report": validation_report,
        "scaler": scaler,
        "X_cluster": X_cluster,
        "cluster_features": CLUSTER_FEATURES,
        "fitted_algorithms": fitted_algorithms,
        "metrics_df": metrics_df,
        "ward_model": ward_model,
        "model_results": model_results,
        "feature_cols": feature_cols,
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "deployed_model": deployed_model,
        "pca": pca,
    }


def get_cluster_summary(bundle: dict) -> pd.DataFrame:
    """Cluster sizes, CV-risk rates + 95% CI, and mean BPV per cluster."""
    df = bundle["df"]
    risk_table = analysis.calculate_cv_risk(df, cluster_col="cluster")
    profiles = analysis.demographic_profiles(df, cluster_col="cluster")
    summary = risk_table.merge(profiles, on="cluster", suffixes=("", "_demo"))
    return summary


def get_patient_data(bundle: dict, patient_id: str | None = None) -> pd.DataFrame:
    """Full patient table, or a single-row lookup when patient_id is given."""
    df = bundle["df"]
    if patient_id is None:
        return df
    return df[df["patient_id"] == patient_id]


def get_metrics_table(bundle: dict) -> pd.DataFrame:
    """Clustering algorithm comparison metrics table."""
    return bundle["metrics_df"]


def get_prediction_results(bundle: dict) -> pd.DataFrame:
    """Tidy comparison table of all trained CV-risk prediction models."""
    rows = []
    for name, res in bundle["model_results"].items():
        m = res["metrics"]
        rows.append({
            "model": name, "accuracy": m["accuracy"], "precision": m["precision"],
            "recall": m["recall"], "f1_score": m["f1_score"], "roc_auc": m["roc_auc"],
        })
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def get_timeline_data(bundle: dict, patient_id: str, n_sessions: int = 12, seed: int = 42) -> pd.DataFrame:
    """Synthetic per-session BP trajectory for one patient (for line-chart display)."""
    patient = get_patient_data(bundle, patient_id)
    if patient.empty:
        return pd.DataFrame()
    row = patient.iloc[0]
    rng = np.random.default_rng(abs(hash(patient_id)) % (2**32) if seed is None else seed + int(row.name))
    sessions = np.arange(1, n_sessions + 1)
    sbp = rng.normal(row["sbp_mean"], row["sbp_sd"], n_sessions)
    dbp = rng.normal(row["dbp_mean"], row["dbp_sd"], n_sessions)
    return pd.DataFrame({"session": sessions, "sbp": np.round(sbp, 1), "dbp": np.round(dbp, 1)})


def get_cluster_profile_matrix(bundle: dict, features: list[str] | None = None) -> pd.DataFrame:
    """Mean feature values per cluster, used for radar/parallel-coordinates charts."""
    features = features or ["sbpv", "dbpv", "age", "bmi", "pulse_pressure", "crp"]
    df = bundle["df"]
    return df.groupby("cluster")[features].mean().reset_index()


def get_patient_similarity_edges(bundle: dict, top_k: int = 3) -> pd.DataFrame:
    """Nearest-neighbor edges (by clustering feature distance) for a patient similarity network graph."""
    X = bundle["X_cluster"]
    ids = bundle["df"]["patient_id"].values
    clusters = bundle["df"]["cluster"].values
    dist = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    edges = []
    for i in range(len(ids)):
        nearest = np.argsort(dist[i])[:top_k]
        for j in nearest:
            edges.append({"source": ids[i], "target": ids[j], "source_cluster": clusters[i], "weight": 1 / (1 + dist[i, j])})
    return pd.DataFrame(edges)


def predict_patient_risk(bundle: dict, patient_features: dict) -> dict:
    """Predict CV risk for an arbitrary (possibly hypothetical) patient using the deployed model."""
    return prediction.predict_single_patient(bundle["deployed_model"], patient_features, bundle["feature_cols"])


def predict_patient_by_id(bundle: dict, patient_id: str) -> dict:
    """Predict CV risk for an existing cohort patient using its fully engineered feature row."""
    row = bundle["full_processed"][bundle["full_processed"]["patient_id"] == patient_id]
    if row.empty:
        return {"probability": float("nan"), "prediction": 0, "risk_level": "Unknown"}
    patient_features = row.iloc[0].to_dict()
    return predict_patient_risk(bundle, patient_features)
