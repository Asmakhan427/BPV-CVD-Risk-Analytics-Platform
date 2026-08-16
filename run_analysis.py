"""
Backend-only analysis runner: generates the synthetic cohort, runs all four
clustering algorithms, trains all predictive models, saves data to SQLite,
and writes publication-quality static figures to ./reports/figures.

Usage:
    python run_analysis.py
    python run_analysis.py --n 83 --seed 42 --outdir reports/figures
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from bpv_cvd import analysis, database, dashboard_data, visualization

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_analysis")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full BPV-CVD analysis pipeline.")
    parser.add_argument("--n", type=int, default=83, help="Cohort size (default: 83, matches the paper).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--outdir", type=str, default="reports/figures", help="Directory for saved figures.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("Step 1/5: Building full analysis pipeline (data, clustering, prediction)...")
    bundle = dashboard_data.build_full_pipeline(n=args.n, seed=args.seed)
    df = bundle["df"]

    logger.info("Step 2/5: Validating synthetic cohort against published statistics...")
    logger.info("Validation report: %s", bundle["validation_report"])

    logger.info("Step 3/5: Persisting patients + longitudinal measurements to SQLite...")
    database.save_patients(df)
    database.save_measurements(df)
    for algo_name, model in bundle["fitted_algorithms"].items():
        database.save_clusters(df["patient_id"].tolist(), algo_name, model.labels_.tolist())

    logger.info("Step 4/5: Generating publication-quality static figures...")
    figs = {
        "sbpv_distribution": visualization.plot_distribution(df, "sbpv", "Systolic BPV Distribution"),
        "dbpv_distribution": visualization.plot_distribution(df, "dbpv", "Diastolic BPV Distribution"),
        "sbpv_by_cluster": visualization.plot_boxplot_by_cluster(df, "sbpv"),
        "dbpv_by_cluster": visualization.plot_boxplot_by_cluster(df, "dbpv"),
        "bpv_by_risk_violin": visualization.plot_violin_by_risk(df, "sbpv"),
        "correlation_heatmap": visualization.plot_correlation_heatmap(df, analysis.DEFAULT_FEATURES),
        "scatter_sbpv_dbpv": visualization.plot_scatter_2d(df, "sbpv", "dbpv"),
        "dendrogram": visualization.plot_dendrogram(bundle["fitted_algorithms"]["Ward"].linkage_matrix_,
                                                      labels=df["patient_id"].tolist()),
        "silhouette_ward": visualization.plot_silhouette(bundle["X_cluster"], bundle["fitted_algorithms"]["Ward"].labels_, "Ward"),
        "pca_projection": visualization.plot_pca_projection(bundle["X_cluster"], bundle["fitted_algorithms"]["Ward"].labels_),
        "tsne_projection": visualization.plot_tsne_projection(bundle["X_cluster"], bundle["fitted_algorithms"]["Ward"].labels_),
        "metrics_silhouette": visualization.plot_metrics_comparison(bundle["metrics_df"], "silhouette_score"),
        "metrics_davies_bouldin": visualization.plot_metrics_comparison(bundle["metrics_df"], "davies_bouldin_score"),
        "pie_cluster_distribution": visualization.plot_pie_cluster_distribution(df),
    }

    best_model_name = dashboard_data.get_prediction_results(bundle).iloc[0]["model"]
    best_res = bundle["model_results"][best_model_name]
    import numpy as np
    y_test = bundle["y_test"]
    probs = np.array(best_res["metrics"]["probabilities"])
    preds = np.array(best_res["metrics"]["predictions"])
    figs["roc_curve"] = visualization.plot_roc_curve(y_test, probs, best_model_name)
    figs["confusion_matrix"] = visualization.plot_confusion_matrix(y_test, preds, best_model_name)
    figs["precision_recall"] = visualization.plot_precision_recall(y_test, probs, best_model_name)
    if best_res["importance"] is not None:
        figs["feature_importance"] = visualization.plot_feature_importance(best_res["importance"], model_name=best_model_name)

    profile_matrix = dashboard_data.get_cluster_profile_matrix(bundle)
    figs["radar_cluster_profiles"] = visualization.plot_radar_cluster_profiles(
        profile_matrix, ["sbpv", "dbpv", "age", "bmi", "pulse_pressure", "crp"]
    )

    for name, fig in figs.items():
        path = outdir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Saved %s", path)

    logger.info("Step 5/5: Summary")
    logger.info("Cluster CV-risk summary:\n%s", dashboard_data.get_cluster_summary(bundle).to_string(index=False))
    logger.info("Model performance:\n%s", dashboard_data.get_prediction_results(bundle).to_string(index=False))
    logger.info("Done. Figures saved to %s", outdir.resolve())


if __name__ == "__main__":
    main()
