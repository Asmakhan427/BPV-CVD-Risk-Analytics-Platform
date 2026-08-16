"""
SQLite persistence layer for the BPV-CVD Risk Analytics Platform.

Schema
------
patients      : one row per patient (demographics, comorbidities, labs)
measurements  : longitudinal blood-pressure readings used to derive BPV
clusters      : cluster assignment per patient per algorithm run
predictions   : CV-risk model predictions per patient per model
"""
from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "bpv_cvd.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    age REAL, sex TEXT, bmi REAL, dialysis_vintage_months REAL,
    diabetes INTEGER, hypertension INTEGER, coronary_artery_disease INTEGER,
    prior_stroke INTEGER, avf_location TEXT, albumin REAL, hemoglobin REAL,
    crp REAL, ultrafiltration_rate REAL, sbp_mean REAL, sbp_sd REAL, sbpv REAL,
    dbp_mean REAL, dbp_sd REAL, dbpv REAL, pulse_pressure REAL,
    cv_risk_event INTEGER
);

CREATE TABLE IF NOT EXISTS measurements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,
    session_number INTEGER,
    sbp REAL, dbp REAL, measured_at TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE TABLE IF NOT EXISTS clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    cluster_label TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    probability REAL,
    risk_level TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);
"""


@contextmanager
def get_connection(db_path: Path = DEFAULT_DB_PATH):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.commit()
    logger.info("Initialized database at %s", db_path)


def save_patients(df: pd.DataFrame, db_path: Path = DEFAULT_DB_PATH) -> None:
    init_db(db_path)
    cols = [
        "patient_id", "age", "sex", "bmi", "dialysis_vintage_months", "diabetes",
        "hypertension", "coronary_artery_disease", "prior_stroke", "avf_location",
        "albumin", "hemoglobin", "crp", "ultrafiltration_rate", "sbp_mean", "sbp_sd",
        "sbpv", "dbp_mean", "dbp_sd", "dbpv", "pulse_pressure", "cv_risk_event",
    ]
    available = [c for c in cols if c in df.columns]
    with get_connection(db_path) as conn:
        df[available].to_sql("patients", conn, if_exists="replace", index=False)
        conn.commit()
    logger.info("Saved %d patients to database.", len(df))


def save_measurements(df: pd.DataFrame, n_sessions: int = 12, seed: int = 42, db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Generate & persist synthetic longitudinal BP measurements per patient (for BPV derivation / trajectory charts)."""
    rng = np.random.default_rng(seed)
    rows = []
    for _, r in df.iterrows():
        for s in range(1, n_sessions + 1):
            sbp = rng.normal(r["sbp_mean"], r["sbp_sd"])
            dbp = rng.normal(r["dbp_mean"], r["dbp_sd"])
            rows.append({
                "patient_id": r["patient_id"], "session_number": s,
                "sbp": round(float(sbp), 1), "dbp": round(float(dbp), 1),
                "measured_at": f"Session {s}",
            })
    m_df = pd.DataFrame(rows)
    init_db(db_path)
    with get_connection(db_path) as conn:
        m_df.to_sql("measurements", conn, if_exists="replace", index=False)
        conn.commit()
    return m_df


def save_clusters(patient_ids: list[str], algorithm: str, labels: list, db_path: Path = DEFAULT_DB_PATH) -> None:
    init_db(db_path)
    df = pd.DataFrame({"patient_id": patient_ids, "algorithm": algorithm, "cluster_label": labels})
    with get_connection(db_path) as conn:
        df.to_sql("clusters", conn, if_exists="append", index=False)
        conn.commit()


def save_predictions(patient_ids: list[str], model_name: str, probabilities: list, risk_levels: list, db_path: Path = DEFAULT_DB_PATH) -> None:
    init_db(db_path)
    df = pd.DataFrame({
        "patient_id": patient_ids, "model_name": model_name,
        "probability": probabilities, "risk_level": risk_levels,
    })
    with get_connection(db_path) as conn:
        df.to_sql("predictions", conn, if_exists="append", index=False)
        conn.commit()


def load_table(table: str, db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    with get_connection(db_path) as conn:
        try:
            return pd.read_sql(f"SELECT * FROM {table}", conn)
        except pd.errors.DatabaseError:
            return pd.DataFrame()
