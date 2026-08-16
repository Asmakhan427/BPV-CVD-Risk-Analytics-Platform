import numpy as np
import pandas as pd
import pytest

from bpv_cvd.data_generator import CLUSTER_SPEC, generate_patient_data, validate_distribution


def test_generate_default_size():
    df = generate_patient_data()
    assert len(df) == 83


def test_generate_custom_size():
    df = generate_patient_data(n=50, seed=1)
    assert len(df) == 50


def test_reproducibility():
    df1 = generate_patient_data(seed=7)
    df2 = generate_patient_data(seed=7)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds_differ():
    df1 = generate_patient_data(seed=1)
    df2 = generate_patient_data(seed=2)
    assert not df1["sbpv"].equals(df2["sbpv"])


def test_required_columns_present():
    df = generate_patient_data()
    required = {"patient_id", "age", "sex", "sbpv", "dbpv", "cv_risk_event", "true_cluster"}
    assert required.issubset(df.columns)


def test_no_missing_values():
    df = generate_patient_data()
    assert not df.isna().any().any()


def test_cv_risk_event_binary():
    df = generate_patient_data()
    assert set(df["cv_risk_event"].unique()).issubset({0, 1})


def test_bpv_values_nonnegative():
    df = generate_patient_data()
    assert (df["sbpv"] > 0).all()
    assert (df["dbpv"] > 0).all()


def test_cluster_sizes_sum_to_n():
    df = generate_patient_data(n=83)
    assert sum(CLUSTER_SPEC[c]["n"] for c in CLUSTER_SPEC) == 83
    assert df["true_cluster"].value_counts().sum() == 83


def test_validate_distribution_passes_on_default():
    df = generate_patient_data()
    report = validate_distribution(df)
    assert report["passed"] is True
    assert report["n_total"] == 83
