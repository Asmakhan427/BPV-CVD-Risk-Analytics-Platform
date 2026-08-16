import pytest

from bpv_cvd.data_generator import generate_patient_data
from bpv_cvd.preprocessing import train_test_split_data
from bpv_cvd.prediction import (
    LogisticRegressionRiskModel,
    MODELS,
    RandomForestRiskModel,
    XGBoostRiskModel,
    predict_single_patient,
    train_all_models,
)


@pytest.fixture(scope="module")
def split_data():
    df = generate_patient_data()
    return train_test_split_data(df, test_size=0.25, seed=42)


@pytest.mark.parametrize("cls", [RandomForestRiskModel, XGBoostRiskModel, LogisticRegressionRiskModel])
def test_model_train_and_evaluate(cls, split_data):
    X_train, X_test, y_train, y_test, _ = split_data
    model = cls(random_state=42)
    model.train_model(X_train, y_train)
    metrics = model.evaluate_model(X_test, y_test)
    assert 0 <= metrics["accuracy"] <= 1
    assert "confusion_matrix" in metrics
    assert len(metrics["predictions"]) == len(y_test)


def test_feature_importance_shape(split_data):
    X_train, X_test, y_train, y_test, feature_cols = split_data
    model = RandomForestRiskModel(random_state=42)
    model.train_model(X_train, y_train)
    importance = model.feature_importance()
    assert set(importance["feature"]) == set(feature_cols)
    assert (importance["importance"] >= 0).all()


def test_train_all_models_returns_all(split_data):
    X_train, X_test, y_train, y_test, _ = split_data
    results = train_all_models(X_train, y_train, X_test, y_test)
    assert set(results.keys()) == set(MODELS.keys())


def test_predict_single_patient(split_data):
    X_train, X_test, y_train, y_test, feature_cols = split_data
    model = RandomForestRiskModel(random_state=42)
    model.train_model(X_train, y_train)
    patient = {f: X_test.iloc[0][f] for f in feature_cols}
    result = predict_single_patient(model, patient, feature_cols)
    assert 0 <= result["probability"] <= 1
    assert result["risk_level"] in {"Low", "Moderate", "High"}
