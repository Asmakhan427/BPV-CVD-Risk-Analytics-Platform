"""
Publication-quality static visualizations (matplotlib / seaborn) for the
BPV-CVD Risk Analytics Platform. Every function returns a `matplotlib.figure.Figure`
so callers (CLI pipeline, tests, Streamlit via st.pyplot) can save or embed it.

All figures share one colorblind-safe palette (see `bpv_cvd.palette`) and a
consistent style so static and interactive (Plotly, in `dashboard_data.py`
consumers) figures read as one visual system.
"""
from __future__ import annotations

import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve, silhouette_samples

from . import palette as pal

matplotlib.use("Agg")
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", rc={
    "axes.edgecolor": pal.BASELINE,
    "axes.labelcolor": pal.INK_PRIMARY,
    "text.color": pal.INK_PRIMARY,
    "xtick.color": pal.INK_SECONDARY,
    "ytick.color": pal.INK_SECONDARY,
    "grid.color": pal.GRIDLINE,
    "figure.facecolor": pal.SURFACE,
    "axes.facecolor": pal.SURFACE,
})

CLUSTER_PALETTE = pal.CLUSTER_COLORS
CLUSTER_ORDER = pal.CLUSTER_ORDER


def _finalize(fig, title):
    fig.suptitle(title, fontsize=13, fontweight="bold", color=pal.INK_PRIMARY)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ---------------------------------------------------------------- Statistical

def plot_distribution(df: pd.DataFrame, column: str, title: str | None = None) -> plt.Figure:
    """Histogram + KDE for a single numeric column."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(df[column].dropna(), kde=True, ax=ax, color=pal.CATEGORICAL[0], edgecolor="white")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    return _finalize(fig, title or f"Distribution of {column}")


def plot_boxplot_by_cluster(df: pd.DataFrame, column: str, cluster_col: str = "cluster") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = [c for c in CLUSTER_ORDER if c in df[cluster_col].unique()]
    sns.boxplot(data=df, x=cluster_col, y=column, order=order, ax=ax,
                palette=[CLUSTER_PALETTE.get(c, pal.CATEGORICAL[0]) for c in order])
    ax.set_xlabel("BPV Cluster")
    ax.set_ylabel(column)
    return _finalize(fig, f"{column} by BPV Cluster")


def plot_violin_by_risk(df: pd.DataFrame, column: str, risk_col: str = "cv_risk_event") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_df = df.copy()
    plot_df["Risk Status"] = plot_df[risk_col].map({1: "CV Event", 0: "No Event"})
    sns.violinplot(data=plot_df, x="Risk Status", y=column, ax=ax,
                    palette=[pal.RISK_COLORS["No Event"], pal.RISK_COLORS["CV Event"]],
                    order=["No Event", "CV Event"])
    ax.set_xlabel("")
    ax.set_ylabel(column)
    return _finalize(fig, f"{column} by Cardiovascular Risk Status")


def plot_correlation_heatmap(df: pd.DataFrame, features: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 7))
    corr = df[features].corr()
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, center=0, ax=ax,
                linewidths=0.5, linecolor=pal.SURFACE, cbar_kws={"shrink": 0.8})
    return _finalize(fig, "Feature Correlation Matrix")


def plot_pairplot(df: pd.DataFrame, features: list[str], cluster_col: str = "cluster"):
    """Seaborn PairGrid colored by cluster (returns a seaborn.PairGrid, not a Figure)."""
    order = [c for c in CLUSTER_ORDER if c in df[cluster_col].unique()]
    g = sns.pairplot(df, vars=features, hue=cluster_col, hue_order=order,
                      palette=[CLUSTER_PALETTE.get(c, pal.CATEGORICAL[0]) for c in order],
                      diag_kind="kde", plot_kws={"alpha": 0.7, "s": 30})
    g.fig.suptitle("Pairwise Feature Relationships by Cluster", y=1.02, fontweight="bold")
    return g


def plot_qq(df: pd.DataFrame, column: str) -> plt.Figure:
    from scipy import stats as sstats
    fig, ax = plt.subplots(figsize=(6, 6))
    sstats.probplot(df[column].dropna(), dist="norm", plot=ax)
    ax.get_lines()[0].set_markerfacecolor(pal.CATEGORICAL[0])
    ax.get_lines()[0].set_markeredgecolor(pal.CATEGORICAL[0])
    ax.get_lines()[1].set_color(pal.STATUS["critical"])
    return _finalize(fig, f"Q-Q Plot: {column}")


def plot_confidence_intervals(ci_df: pd.DataFrame, label_col: str = "cluster", value_col: str = "estimate") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    y_pos = np.arange(len(ci_df))
    errors = [ci_df[value_col] - ci_df["ci_lower"], ci_df["ci_upper"] - ci_df[value_col]]
    ax.errorbar(ci_df[value_col], y_pos, xerr=errors, fmt="o", color=pal.CATEGORICAL[0],
                ecolor=pal.INK_SECONDARY, capsize=4, markersize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ci_df[label_col])
    ax.set_xlabel(value_col)
    return _finalize(fig, "Bootstrap 95% Confidence Intervals")


# ----------------------------------------------------------------- Clustering

def plot_scatter_2d(df: pd.DataFrame, x: str, y: str, cluster_col: str = "cluster") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    order = [c for c in CLUSTER_ORDER if c in df[cluster_col].unique()]
    for c in order:
        sub = df[df[cluster_col] == c]
        ax.scatter(sub[x], sub[y], label=c, color=CLUSTER_PALETTE.get(c, pal.CATEGORICAL[0]),
                   alpha=0.75, s=55, edgecolor="white", linewidth=0.5)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend(title="Cluster", frameon=False)
    return _finalize(fig, f"{y} vs {x} by BPV Cluster")


def plot_dendrogram(linkage_matrix: np.ndarray, labels: list[str] | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    dendrogram(linkage_matrix, labels=labels, ax=ax, color_threshold=0,
               above_threshold_color=pal.CATEGORICAL[0])
    ax.set_xlabel("Patient")
    ax.set_ylabel("Ward Distance")
    ax.tick_params(axis="x", labelsize=6, rotation=90)
    return _finalize(fig, "Ward's Method Dendrogram")


def plot_silhouette(X: np.ndarray, labels: np.ndarray, algorithm_name: str = "") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sil_values = silhouette_samples(X, labels)
    sil_avg = sil_values.mean()
    y_lower = 10
    uniq = sorted(set(labels))
    colors = [pal.CATEGORICAL[i % len(pal.CATEGORICAL)] for i in range(len(uniq))]
    for i, cluster in enumerate(uniq):
        vals = np.sort(sil_values[labels == cluster])
        y_upper = y_lower + len(vals)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, facecolor=colors[i], edgecolor=colors[i], alpha=0.8)
        ax.text(-0.02, y_lower + 0.5 * len(vals), str(cluster))
        y_lower = y_upper + 10
    ax.axvline(x=sil_avg, color=pal.STATUS["critical"], linestyle="--", label=f"Mean = {sil_avg:.3f}")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster")
    ax.set_yticks([])
    ax.legend(frameon=False)
    return _finalize(fig, f"Silhouette Plot — {algorithm_name}")


def plot_pca_projection(X: np.ndarray, labels: np.ndarray, n_components: int = 2) -> plt.Figure:
    pca = PCA(n_components=n_components, random_state=42)
    proj = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    uniq = sorted(set(labels))
    for i, c in enumerate(uniq):
        mask = labels == c
        ax.scatter(proj[mask, 0], proj[mask, 1], label=str(c),
                   color=pal.CATEGORICAL[i % len(pal.CATEGORICAL)], alpha=0.75, s=55, edgecolor="white")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var.)")
    ax.legend(title="Cluster", frameon=False)
    return _finalize(fig, "PCA Projection")


def plot_tsne_projection(X: np.ndarray, labels: np.ndarray, perplexity: float = 15) -> plt.Figure:
    perplexity = min(perplexity, max(5, len(X) // 3))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="pca")
    proj = tsne.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    uniq = sorted(set(labels))
    for i, c in enumerate(uniq):
        mask = labels == c
        ax.scatter(proj[mask, 0], proj[mask, 1], label=str(c),
                   color=pal.CATEGORICAL[i % len(pal.CATEGORICAL)], alpha=0.75, s=55, edgecolor="white")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(title="Cluster", frameon=False)
    return _finalize(fig, "t-SNE Projection")


# ---------------------------------------------------------------- Performance

def plot_metrics_comparison(metrics_df: pd.DataFrame, metric: str, algo_col: str = "algorithm") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = [pal.ALGORITHM_COLORS.get(a, pal.CATEGORICAL[0]) for a in metrics_df[algo_col]]
    ax.bar(metrics_df[algo_col], metrics_df[metric], color=colors, edgecolor="white")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xlabel("")
    return _finalize(fig, f"{metric.replace('_', ' ').title()} by Algorithm")


# ----------------------------------------------------------------- Prediction

def plot_roc_curve(y_true, y_proba, model_name: str = "") -> plt.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    from sklearn.metrics import auc
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(fpr, tpr, color=pal.CATEGORICAL[0], lw=2.5, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color=pal.BASELINE, lw=1.5, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", frameon=False)
    return _finalize(fig, f"ROC Curve — {model_name}")


def plot_confusion_matrix(y_true, y_pred, model_name: str = "") -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Event", "CV Event"], yticklabels=["No Event", "CV Event"], cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return _finalize(fig, f"Confusion Matrix — {model_name}")


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 12, model_name: str = "") -> plt.Figure:
    top = importance_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, max(4, 0.4 * len(top))))
    ax.barh(top["feature"], top["importance"], color=pal.CATEGORICAL[0], edgecolor="white")
    ax.set_xlabel("Importance")
    return _finalize(fig, f"Feature Importance — {model_name}")


def plot_precision_recall(y_true, y_proba, model_name: str = "") -> plt.Figure:
    from sklearn.metrics import precision_recall_curve, average_precision_score
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(rec, prec, color=pal.CATEGORICAL[2], lw=2.5, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left", frameon=False)
    return _finalize(fig, f"Precision-Recall Curve — {model_name}")


def plot_learning_curve(lc_data: dict, model_name: str = "") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sizes = lc_data["train_sizes"]
    ax.plot(sizes, lc_data["train_mean"], "o-", color=pal.CATEGORICAL[0], label="Training score")
    ax.fill_between(sizes, np.array(lc_data["train_mean"]) - np.array(lc_data["train_std"]),
                     np.array(lc_data["train_mean"]) + np.array(lc_data["train_std"]), alpha=0.15, color=pal.CATEGORICAL[0])
    ax.plot(sizes, lc_data["test_mean"], "o-", color=pal.CATEGORICAL[1], label="Cross-val score")
    ax.fill_between(sizes, np.array(lc_data["test_mean"]) - np.array(lc_data["test_std"]),
                     np.array(lc_data["test_mean"]) + np.array(lc_data["test_std"]), alpha=0.15, color=pal.CATEGORICAL[1])
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("ROC AUC")
    ax.legend(frameon=False)
    return _finalize(fig, f"Learning Curve — {model_name}")


# ------------------------------------------------------------------ Dashboard

def plot_pie_cluster_distribution(df: pd.DataFrame, cluster_col: str = "cluster") -> plt.Figure:
    counts = df[cluster_col].value_counts().reindex([c for c in CLUSTER_ORDER if c in df[cluster_col].unique()])
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = [CLUSTER_PALETTE.get(c, pal.CATEGORICAL[0]) for c in counts.index]
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", colors=colors,
           wedgeprops={"edgecolor": "white", "linewidth": 1.5}, textprops={"color": pal.INK_PRIMARY})
    return _finalize(fig, "Cluster Distribution")


def plot_radar_cluster_profiles(profile_df: pd.DataFrame, features: list[str], cluster_col: str = "cluster") -> plt.Figure:
    """Radar/spider chart comparing normalized cluster-mean profiles."""
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    norm = (profile_df[features] - profile_df[features].min()) / (profile_df[features].max() - profile_df[features].min() + 1e-9)
    for idx, row in profile_df.iterrows():
        values = norm.loc[idx].tolist()
        values += values[:1]
        c = row[cluster_col]
        color = CLUSTER_PALETTE.get(c, pal.CATEGORICAL[0])
        ax.plot(angles, values, color=color, linewidth=2, label=c)
        ax.fill(angles, values, color=color, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, fontsize=9)
    ax.set_yticklabels([])
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=False)
    return _finalize(fig, "Cluster Profile Comparison")
