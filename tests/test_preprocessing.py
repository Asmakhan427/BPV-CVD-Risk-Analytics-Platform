import numpy as np
import pandas as pd

from bpv_cvd.data_generator import generate_patient_data
from bpv_cvd.preprocessing import (
    NUMERIC_FEATURES,
    add_interaction_terms,
    build_feature_matrix,
    encode_categoricals,
    handle_missing_values,
    scale_features,
    train_test_split_data,
)


def test_handle_missing_values_imputes():
    df = generate_patient_data()
    df.loc[0, "age"] = np.nan
    out = handle_missing_values(df)
    assert not out["age"].isna().any()


def test_encode_categoricals_adds_codes():
    df = generate_patient_data()
    out = encode_categoricals(df)
    assert "sex_code" in out.columns
    assert "avf_location_code" in out.columns
    assert out["sex_code"].dtype.kind in "iu"


def test_add_interaction_terms():
    df = generate_patient_data()
    out = add_interaction_terms(df)
    assert "bpv_composite" in out.columns
    assert "comorbidity_burden" in out.columns
    assert (out["comorbidity_burden"] >= 0).all()


def test_scale_features_zero_mean_unit_var():
    df = generate_patient_data()
    features = [c for c in NUMERIC_FEATURES if c in df.columns]
    scaled, scaler = scale_features(df, features=features)
    means = scaled[features].mean()
    stds = scaled[features].std(ddof=0)
    assert np.allclose(means, 0, atol=1e-8)
    assert np.allclose(stds, 1, atol=1e-6)


def test_build_feature_matrix_no_missing():
    df = generate_patient_data()
    out = build_feature_matrix(df)
    assert not out[NUMERIC_FEATURES].isna().any().any()


def test_train_test_split_shapes():
    df = generate_patient_data()
    X_train, X_test, y_train, y_test, feature_cols = train_test_split_data(df, test_size=0.25, seed=42)
    assert len(X_train) + len(X_test) == len(df)
    assert set(X_train.columns) == set(feature_cols)
    assert len(y_train) == len(X_train)
