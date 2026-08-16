"""
Preprocessing utilities: missing-value handling, feature scaling, feature
engineering (interaction terms), and train/test splitting.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

# Core numeric feature set used for clustering / modeling
NUMERIC_FEATURES = [
    "age",
    "bmi",
    "dialysis_vintage_months",
    "albumin",
    "hemoglobin",
    "crp",
    "ultrafiltration_rate",
    "sbp_mean",
    "sbp_sd",
    "sbpv",
    "dbp_mean",
    "dbp_sd",
    "dbpv",
    "pulse_pressure",
]

BPV_FEATURES = ["sbpv", "dbpv"]

CATEGORICAL_FEATURES = ["sex", "avf_location"]

BINARY_FEATURES = ["diabetes", "hypertension", "coronary_artery_disease", "prior_stroke"]

TARGET = "cv_risk_event"


def handle_missing_values(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """Impute missing numeric values (median/mean) in place of NaNs."""
    df = df.copy()
    numeric_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    if df[numeric_cols].isna().any().any():
        imputer = SimpleImputer(strategy=strategy)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        logger.info("Imputed missing values in columns: %s", numeric_cols)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode categorical string columns, appending `_code` columns."""
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[f"{col}_code"] = le.fit_transform(df[col].astype(str))
    return df


def add_interaction_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer clinically meaningful interaction / derived features."""
    df = df.copy()
    if {"sbpv", "dbpv"}.issubset(df.columns):
        df["bpv_composite"] = df["sbpv"] * 0.6 + df["dbpv"] * 0.4
        df["sbpv_dbpv_ratio"] = df["sbpv"] / df["dbpv"].replace(0, np.nan)
        df["sbpv_dbpv_ratio"] = df["sbpv_dbpv_ratio"].fillna(df["sbpv_dbpv_ratio"].median())
    if {"sbpv", "age"}.issubset(df.columns):
        df["sbpv_age_interaction"] = df["sbpv"] * df["age"] / 100.0
    if {"pulse_pressure", "sbpv"}.issubset(df.columns):
        df["pp_sbpv_interaction"] = df["pulse_pressure"] * df["sbpv"] / 100.0
    if {"diabetes", "hypertension"}.issubset(df.columns):
        df["comorbidity_burden"] = (
            df.get("diabetes", 0) + df.get("hypertension", 0)
            + df.get("coronary_artery_disease", 0) + df.get("prior_stroke", 0)
        )
    return df


def scale_features(df: pd.DataFrame, features: Iterable[str] | None = None, scaler: StandardScaler | None = None):
    """
    Standardize the given feature columns (zero mean, unit variance).

    Returns
    -------
    (scaled_df, fitted_scaler)
    """
    features = list(features) if features is not None else [c for c in NUMERIC_FEATURES if c in df.columns]
    df = df.copy()
    if scaler is None:
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(df[features])
    else:
        scaled_values = scaler.transform(df[features])
    scaled_df = df.copy()
    scaled_df[features] = scaled_values
    return scaled_df, scaler


def build_feature_matrix(df: pd.DataFrame, features: Iterable[str] | None = None) -> pd.DataFrame:
    """Full preprocessing pipeline -> numeric feature matrix ready for ML."""
    processed = handle_missing_values(df)
    processed = encode_categoricals(processed)
    processed = add_interaction_terms(processed)
    features = list(features) if features is not None else None
    return processed


def train_test_split_data(df: pd.DataFrame, target: str = TARGET, test_size: float = 0.25, seed: int = 42, stratify: bool = True):
    """Stratified (by target) train/test split returning (X_train, X_test, y_train, y_test)."""
    processed = build_feature_matrix(df)
    feature_cols = [c for c in NUMERIC_FEATURES if c in processed.columns] + [
        c for c in ["bpv_composite", "sbpv_dbpv_ratio", "sbpv_age_interaction", "pp_sbpv_interaction", "comorbidity_burden"]
        if c in processed.columns
    ]
    X = processed[feature_cols]
    y = processed[target]
    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=strat)
    return X_train, X_test, y_train, y_test, feature_cols
