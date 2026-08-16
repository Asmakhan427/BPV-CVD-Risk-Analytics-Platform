"""
Full pipeline runner: runs the analysis (see run_analysis.py) and, if
requested, generates the PDF report as well. This is the single command an
instructor / reviewer can run to regenerate every artifact from scratch.

Usage:
    python run_pipeline.py
    python run_pipeline.py --generate-report
    python run_pipeline.py --generate-report --report-out reports/bpv_cvd_report.pdf
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from bpv_cvd import dashboard_data, database, report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_pipeline")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full BPV-CVD pipeline end to end.")
    parser.add_argument("--n", type=int, default=83)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generate-report", action="store_true", help="Also generate the PDF report.")
    parser.add_argument("--report-out", type=str, default="reports/bpv_cvd_analytics_report.pdf")
    parser.add_argument("--figures-out", type=str, default="reports/figures")
    args = parser.parse_args()

    logger.info("Running full pipeline (n=%d, seed=%d)...", args.n, args.seed)
    bundle = dashboard_data.build_full_pipeline(n=args.n, seed=args.seed)

    database.save_patients(bundle["df"])
    database.save_measurements(bundle["df"])
    for algo_name, model in bundle["fitted_algorithms"].items():
        database.save_clusters(bundle["df"]["patient_id"].tolist(), algo_name, model.labels_.tolist())
    logger.info("Persisted patients, measurements, and cluster assignments to SQLite.")

    logger.info("Cluster risk summary:\n%s", dashboard_data.get_cluster_summary(bundle).to_string(index=False))
    logger.info("Model performance:\n%s", dashboard_data.get_prediction_results(bundle).to_string(index=False))

    if args.generate_report:
        out_path = Path(args.report_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_bytes = report.generate_pdf_report(bundle)
        out_path.write_bytes(pdf_bytes)
        logger.info("PDF report written to %s", out_path.resolve())

    logger.info("Pipeline complete. Launch the dashboard with: streamlit run app.py")


if __name__ == "__main__":
    main()
