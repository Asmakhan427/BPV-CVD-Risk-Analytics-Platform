"""
FastAPI service exposing the BPV-CVD analytics pipeline over HTTP.

Run with:  uvicorn api.main:app --reload --port 8000
Docs at:   http://localhost:8000/docs
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from bpv_cvd import dashboard_data as dd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BPV-CVD Risk Analytics API",
    description="Blood pressure variability & cardiovascular risk analytics for hemodialysis patients "
                "(synthetic data, based on Montoya et al. 2025).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)


@lru_cache(maxsize=1)
def _bundle():
    logger.info("Building analytics pipeline for API (cached)...")
    return dd.build_full_pipeline()


class PatientFeatures(BaseModel):
    age: float = Field(..., ge=18, le=100)
    bmi: float = Field(27.0, ge=10, le=60)
    dialysis_vintage_months: float = Field(24.0, ge=0, le=400)
    albumin: float = Field(3.8, ge=1.5, le=5.5)
    hemoglobin: float = Field(11.0, ge=5, le=18)
    crp: float = Field(5.0, ge=0, le=200)
    ultrafiltration_rate: float = Field(9.0, ge=0, le=25)
    sbp_mean: float = Field(140.0, ge=60, le=220)
    dbp_mean: float = Field(80.0, ge=30, le=140)
    sbpv: float = Field(10.0, ge=0, le=30)
    dbpv: float = Field(5.0, ge=0, le=20)
    diabetes: int = Field(0, ge=0, le=1)
    hypertension: int = Field(1, ge=0, le=1)
    coronary_artery_disease: int = Field(0, ge=0, le=1)
    prior_stroke: int = Field(0, ge=0, le=1)


def _engineer(features: PatientFeatures) -> dict:
    f = features.model_dump()
    f["sbp_sd"] = f["sbp_mean"] * f["sbpv"] / 100
    f["dbp_sd"] = f["dbp_mean"] * f["dbpv"] / 100
    f["pulse_pressure"] = f["sbp_mean"] - f["dbp_mean"]
    f["bpv_composite"] = f["sbpv"] * 0.6 + f["dbpv"] * 0.4
    f["sbpv_dbpv_ratio"] = f["sbpv"] / f["dbpv"] if f["dbpv"] else 0
    f["sbpv_age_interaction"] = f["sbpv"] * f["age"] / 100
    f["pp_sbpv_interaction"] = f["pulse_pressure"] * f["sbpv"] / 100
    f["comorbidity_burden"] = f["diabetes"] + f["hypertension"] + f["coronary_artery_disease"] + f["prior_stroke"]
    f["sex_code"] = 1
    f["avf_location_code"] = 0
    return f


@app.get("/api/health", tags=["System"])
def health_check():
    """Liveness/readiness check."""
    return {"status": "ok", "service": "bpv-cvd-analytics-api", "version": "1.0.0"}


@app.get("/api/data", tags=["Data"])
def get_data(limit: Optional[int] = None):
    """Return the full (synthetic) patient dataset."""
    df = _bundle()["df"]
    out = df.head(limit) if limit else df
    return out.to_dict(orient="records")


@app.get("/api/cluster/{patient_id}", tags=["Data"])
def get_patient_cluster(patient_id: str):
    """Return the BPV cluster assignment for a given patient."""
    df = _bundle()["df"]
    row = df[df["patient_id"] == patient_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found.")
    r = row.iloc[0]
    return {"patient_id": patient_id, "cluster": r["cluster"], "sbpv": r["sbpv"], "dbpv": r["dbpv"]}


@app.get("/api/clusters/summary", tags=["Clustering"])
def clusters_summary():
    """Cluster sizes, CV-risk rates (with 95% CI), and demographic profile per cluster."""
    return dd.get_cluster_summary(_bundle()).to_dict(orient="records")


@app.get("/api/metrics", tags=["Clustering"])
def clustering_metrics():
    """Internal validation metrics (silhouette, Davies-Bouldin, Calinski-Harabasz) for all four algorithms."""
    return dd.get_metrics_table(_bundle()).to_dict(orient="records")


@app.get("/api/models/performance", tags=["Prediction"])
def model_performance():
    """Comparative performance metrics for all trained CV-risk prediction models."""
    return dd.get_prediction_results(_bundle()).to_dict(orient="records")


@app.post("/api/predict", tags=["Prediction"])
def predict(features: PatientFeatures):
    """Predict cardiovascular risk probability for an arbitrary patient feature set."""
    bundle = _bundle()
    engineered = _engineer(features)
    result = dd.predict_patient_risk(bundle, engineered)
    return result


@app.get("/api/patients/{patient_id}/timeline", tags=["Data"])
def patient_timeline(patient_id: str, n_sessions: int = 12):
    """Synthetic per-session BP trajectory for a given patient."""
    bundle = _bundle()
    traj = dd.get_timeline_data(bundle, patient_id, n_sessions=n_sessions)
    if traj.empty:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found.")
    return traj.to_dict(orient="records")
