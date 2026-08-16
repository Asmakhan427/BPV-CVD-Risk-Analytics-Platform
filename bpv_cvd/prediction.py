"""
Predictive modeling for cardiovascular (CV) risk: Random Forest, XGBoost
(with a graceful GradientBoosting fallback if xgboost is unavailable), and
Logistic Regression. Every model wrapper exposes train_model(),
evaluate_model(), and feature_importance() with a common return contract.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class BaseRiskModel:
    """Common interface for all CV-risk predictive models."""

    name = "Base"

    def __init__(self, random_state: int = 42, **kwargs):
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names_: list[str] = []
        self._needs_scaling = False

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> "BaseRiskModel":
        self.feature_names_ = list(X_train.columns)
        X = self.scaler.fit_transform(X_train) if self._needs_scaling else X_train.values
        self.model.fit(X, y_train)
        return self

    def _transform(self, X: pd.DataFrame):
        return self.scaler.transform(X) if self._needs_scaling else X.values

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._transform(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._transform(X))[:, 1]

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Compute standard classification metrics + curves on held-out data."""
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        metrics = {
            "algorithm": self.name,
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4) if len(set(y_test)) > 1 else np.nan,
            "confusion_matrix": cm.tolist(),
            "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "pr_curve": {"precision": prec.tolist(), "recall": rec.tolist()},
            "predictions": y_pred.tolist(),
            "probabilities": y_proba.tolist(),
        }
        return metrics

    def feature_importance(self) -> pd.DataFrame:
        """Return a DataFrame of feature -> importance, sorted descending."""
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_[0])
        else:
            raise NotImplementedError(f"{self.name} does not expose feature importances.")
        return (
            pd.DataFrame({"feature": self.feature_names_, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5, scoring: str = "roc_auc") -> dict:
        Xt = self.scaler.fit_transform(X) if self._needs_scaling else X.values
        scores = cross_val_score(self.model, Xt, y, cv=cv, scoring=scoring)
        return {"scores": scores.tolist(), "mean": round(float(scores.mean()), 4), "std": round(float(scores.std()), 4)}

    def learning_curve_data(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> dict:
        Xt = self.scaler.fit_transform(X) if self._needs_scaling else X.values
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, Xt, y, cv=cv, scoring="roc_auc",
            train_sizes=np.linspace(0.3, 1.0, 6), random_state=self.random_state,
        )
        return {
            "train_sizes": train_sizes.tolist(),
            "train_mean": train_scores.mean(axis=1).tolist(),
            "train_std": train_scores.std(axis=1).tolist(),
            "test_mean": test_scores.mean(axis=1).tolist(),
            "test_std": test_scores.std(axis=1).tolist(),
        }


class RandomForestRiskModel(BaseRiskModel):
    name = "Random Forest"

    def __init__(self, random_state: int = 42, n_estimators: int = 300, max_depth: int | None = 5, **kwargs):
        super().__init__(random_state)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state,
            class_weight="balanced", min_samples_leaf=2,
        )
        self._needs_scaling = False


class XGBoostRiskModel(BaseRiskModel):
    name = "XGBoost"

    def __init__(self, random_state: int = 42, n_estimators: int = 200, max_depth: int = 3, learning_rate: float = 0.08, **kwargs):
        super().__init__(random_state)
        try:
            from xgboost import XGBClassifier

            self.model = XGBClassifier(
                n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                random_state=random_state, eval_metric="logloss", use_label_encoder=False,
            )
        except ImportError:
            logger.info("xgboost not available; falling back to GradientBoostingClassifier.")
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=random_state,
            )
        self._needs_scaling = False


class LogisticRegressionRiskModel(BaseRiskModel):
    name = "Logistic Regression"

    def __init__(self, random_state: int = 42, C: float = 1.0, **kwargs):
        super().__init__(random_state)
        self.model = LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=random_state)
        self._needs_scaling = True


MODELS = {
    "Random Forest": RandomForestRiskModel,
    "XGBoost": XGBoostRiskModel,
    "Logistic Regression": LogisticRegressionRiskModel,
}


def train_all_models(X_train, y_train, X_test, y_test, random_state: int = 42) -> dict:
    """Train all three risk models and return {name: {model, metrics, importance}}."""
    results = {}
    for name, cls in MODELS.items():
        m = cls(random_state=random_state)
        m.train_model(X_train, y_train)
        metrics = m.evaluate_model(X_test, y_test)
        try:
            importance = m.feature_importance()
        except NotImplementedError:
            importance = None
        results[name] = {"model": m, "metrics": metrics, "importance": importance}
        logger.info("%s trained: AUC=%s", name, metrics.get("roc_auc"))
    return results


def predict_single_patient(model: BaseRiskModel, patient_features: dict, feature_order: list[str]) -> dict:
    """Predict CV risk probability for a single patient dict of feature -> value."""
    row = pd.DataFrame([{f: patient_features.get(f, 0.0) for f in feature_order}])
    proba = float(model.predict_proba(row)[0])
    pred = int(model.predict(row)[0])
    if proba < 0.15:
        level = "Low"
    elif proba < 0.35:
        level = "Moderate"
    else:
        level = "High"
    return {"probability": round(proba, 4), "prediction": pred, "risk_level": level}
