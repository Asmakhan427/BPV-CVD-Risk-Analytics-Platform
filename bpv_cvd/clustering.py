"""
Clustering algorithms used to reproduce the four-method comparison in
Montoya et al. (2025): K-Means, PAM (K-Medoids), Ward's hierarchical
clustering, and Expectation-Maximization (Gaussian Mixture Model).

Each class exposes a common interface: fit(), predict(), evaluate(),
get_metrics() so the dashboard/analysis code can treat all four
interchangeably.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


class BaseClustering(ABC):
    """Common interface for all clustering algorithms in this package."""

    name: str = "Base"

    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels_: np.ndarray | None = None
        self.model = None
        self._X: np.ndarray | None = None

    @abstractmethod
    def fit(self, X: np.ndarray | pd.DataFrame) -> "BaseClustering":
        ...

    def predict(self, X: np.ndarray | pd.DataFrame | None = None) -> np.ndarray:
        """Return cluster labels. If X is None, returns labels from fit()."""
        if X is None:
            if self.labels_ is None:
                raise RuntimeError(f"{self.name}: call fit() before predict().")
            return self.labels_
        if hasattr(self.model, "predict"):
            return self.model.predict(np.asarray(X))
        raise NotImplementedError(f"{self.name} does not support predicting on new data.")

    def get_metrics(self) -> dict:
        """Alias for evaluate(), kept for API symmetry with the spec."""
        return self.evaluate()

    def evaluate(self) -> dict:
        """Compute standard internal-validation clustering metrics."""
        if self._X is None or self.labels_ is None:
            raise RuntimeError(f"{self.name}: call fit() before evaluate().")
        X = self._X
        labels = self.labels_
        n_found = len(set(labels))
        if n_found < 2:
            return {
                "algorithm": self.name,
                "n_clusters": n_found,
                "silhouette_score": np.nan,
                "davies_bouldin_score": np.nan,
                "calinski_harabasz_score": np.nan,
            }
        return {
            "algorithm": self.name,
            "n_clusters": n_found,
            "silhouette_score": round(float(silhouette_score(X, labels)), 4),
            "davies_bouldin_score": round(float(davies_bouldin_score(X, labels)), 4),
            "calinski_harabasz_score": round(float(calinski_harabasz_score(X, labels)), 2),
            "cluster_sizes": {str(k): int(v) for k, v in pd.Series(labels).value_counts().sort_index().items()},
        }

    def silhouette_samples_(self) -> np.ndarray:
        if self._X is None or self.labels_ is None:
            raise RuntimeError(f"{self.name}: call fit() before requesting silhouette samples.")
        return silhouette_samples(self._X, self.labels_)


class KMeansClustering(BaseClustering):
    """Standard K-Means (Lloyd's algorithm, k-means++ init)."""

    name = "K-Means"

    def fit(self, X: np.ndarray | pd.DataFrame) -> "KMeansClustering":
        X = np.asarray(X)
        self.model = KMeans(n_clusters=self.n_clusters, init="k-means++", n_init=10, random_state=self.random_state)
        self.labels_ = self.model.fit_predict(X)
        self._X = X
        self.centers_ = self.model.cluster_centers_
        return self


class PAMClustering(BaseClustering):
    """
    Partitioning Around Medoids (K-Medoids). Falls back to a pure-numpy PAM
    implementation if scikit-learn-extra is not installed, so the platform
    has no hard dependency on that optional package.
    """

    name = "PAM"

    def fit(self, X: np.ndarray | pd.DataFrame) -> "PAMClustering":
        X = np.asarray(X)
        try:
            from sklearn_extra.cluster import KMedoids

            self.model = KMedoids(n_clusters=self.n_clusters, random_state=self.random_state, method="pam", init="k-medoids++")
            self.labels_ = self.model.fit_predict(X)
            self.centers_ = self.model.cluster_centers_
        except ImportError:
            logger.info("scikit-learn-extra not available; using built-in PAM implementation.")
            self.labels_, self.centers_ = self._pam_numpy(X)
        self._X = X
        return self

    def _pam_numpy(self, X: np.ndarray, max_iter: int = 100):
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        medoid_idx = rng.choice(n, self.n_clusters, replace=False)
        dist = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)

        for _ in range(max_iter):
            labels = np.argmin(dist[:, medoid_idx], axis=1)
            new_medoid_idx = medoid_idx.copy()
            for k in range(self.n_clusters):
                members = np.where(labels == k)[0]
                if len(members) == 0:
                    continue
                sub_dist = dist[np.ix_(members, members)].sum(axis=1)
                new_medoid_idx[k] = members[np.argmin(sub_dist)]
            if np.array_equal(np.sort(new_medoid_idx), np.sort(medoid_idx)):
                break
            medoid_idx = new_medoid_idx

        labels = np.argmin(dist[:, medoid_idx], axis=1)
        return labels, X[medoid_idx]


class WardClustering(BaseClustering):
    """Ward's minimum-variance agglomerative hierarchical clustering."""

    name = "Ward"

    def fit(self, X: np.ndarray | pd.DataFrame) -> "WardClustering":
        X = np.asarray(X)
        self.linkage_matrix_ = linkage(X, method="ward")
        self.labels_ = fcluster(self.linkage_matrix_, t=self.n_clusters, criterion="maxclust") - 1
        self._X = X
        # centers = centroid of members (Ward has no native centers)
        self.centers_ = np.array([
            X[self.labels_ == k].mean(axis=0) if np.any(self.labels_ == k) else np.zeros(X.shape[1])
            for k in range(self.n_clusters)
        ])
        self.model = None
        return self

    def dendrogram_data(self, labels: list[str] | None = None) -> dict:
        """Return scipy dendrogram coordinate data (no plotting)."""
        if not hasattr(self, "linkage_matrix_"):
            raise RuntimeError("Ward: call fit() before dendrogram_data().")
        return dendrogram(self.linkage_matrix_, labels=labels, no_plot=True)

    def predict(self, X: np.ndarray | pd.DataFrame | None = None) -> np.ndarray:
        # Ward has no native out-of-sample predict; nearest-centroid fallback.
        if X is None:
            return self.labels_
        X = np.asarray(X)
        dists = np.linalg.norm(X[:, None, :] - self.centers_[None, :, :], axis=-1)
        return np.argmin(dists, axis=1)


class EMClustering(BaseClustering):
    """Expectation-Maximization via a Gaussian Mixture Model."""

    name = "EM (GMM)"

    def fit(self, X: np.ndarray | pd.DataFrame) -> "EMClustering":
        X = np.asarray(X)
        self.model = GaussianMixture(n_components=self.n_clusters, covariance_type="full", random_state=self.random_state, n_init=5)
        self.model.fit(X)
        self.labels_ = self.model.predict(X)
        self.centers_ = self.model.means_
        self._X = X
        return self

    def predict_proba(self, X: np.ndarray | pd.DataFrame | None = None) -> np.ndarray:
        X = self._X if X is None else np.asarray(X)
        return self.model.predict_proba(X)


ALGORITHMS = {
    "K-Means": KMeansClustering,
    "PAM": PAMClustering,
    "Ward": WardClustering,
    "EM (GMM)": EMClustering,
}


def run_all_algorithms(X: np.ndarray | pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> dict:
    """Fit all four clustering algorithms and return {name: fitted_instance}."""
    results = {}
    for name, cls in ALGORITHMS.items():
        model = cls(n_clusters=n_clusters, random_state=random_state)
        model.fit(X)
        results[name] = model
        logger.info("%s fit complete: %s", name, model.evaluate())
    return results


def compare_algorithms(X: np.ndarray | pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> pd.DataFrame:
    """Fit all algorithms and return a tidy metrics comparison DataFrame."""
    fitted = run_all_algorithms(X, n_clusters, random_state)
    rows = [m.evaluate() for m in fitted.values()]
    return pd.DataFrame(rows)


def label_clusters_by_severity(labels: np.ndarray, sbpv: np.ndarray, dbpv: np.ndarray) -> np.ndarray:
    """
    Map arbitrary integer cluster labels onto clinically ordered names
    (Low BPV / Medium BPV / High BPV) by ranking cluster-mean BPV composite.
    """
    composite = 0.6 * np.asarray(sbpv) + 0.4 * np.asarray(dbpv)
    df = pd.DataFrame({"label": labels, "composite": composite})
    order = df.groupby("label")["composite"].mean().sort_values().index.tolist()
    name_map = {}
    names = ["Low BPV", "Medium BPV", "High BPV"]
    for i, lbl in enumerate(order):
        name_map[lbl] = names[i] if i < len(names) else f"Cluster {i}"
    return df["label"].map(name_map).values
