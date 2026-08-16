"""
PDF report generation (ReportLab) for the BPV-CVD Risk Analytics Platform.
Builds an executive-summary style clinical/research report from the
analysis bundle produced by `dashboard_data.build_full_pipeline()`.
"""
from __future__ import annotations

import io
import logging
from datetime import date

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger(__name__)

BRAND_BLUE = colors.HexColor("#2a78d6")
BRAND_RED = colors.HexColor("#e34948")
INK = colors.HexColor("#0b0b0b")
MUTED = colors.HexColor("#52514e")

SECTIONS = ["executive_summary", "methodology", "results", "clinical_recommendations", "limitations"]


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("H1c", parent=ss["Heading1"], textColor=BRAND_BLUE, spaceAfter=10))
    ss.add(ParagraphStyle("H2c", parent=ss["Heading2"], textColor=INK, spaceBefore=14, spaceAfter=6))
    ss.add(ParagraphStyle("Bodyc", parent=ss["BodyText"], textColor=INK, leading=15))
    ss.add(ParagraphStyle("Mutedc", parent=ss["BodyText"], textColor=MUTED, fontSize=9))
    return ss


def generate_pdf_report(bundle: dict, sections: list[str] | None = None) -> bytes:
    """
    Build a PDF report (bytes) covering the requested sections. Valid
    section keys: executive_summary, methodology, results,
    clinical_recommendations, limitations (defaults to all).
    """
    from . import dashboard_data as dd

    sections = sections or SECTIONS
    styles = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=LETTER, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    story = []

    story.append(Paragraph("BPV-CVD Risk Analytics Platform", styles["H1c"]))
    story.append(Paragraph(
        "Blood Pressure Variability and Cardiovascular Risk Assessment After Arteriovenous "
        "Fistula Creation in Hemodialysis Patients — Machine Learning Clustering Analysis Report",
        styles["Bodyc"]))
    story.append(Paragraph(f"Generated {date.today().isoformat()} · Based on Montoya et al. (2025)", styles["Mutedc"]))
    story.append(Spacer(1, 16))

    df = bundle["df"]
    cluster_summary = dd.get_cluster_summary(bundle)
    pred_results = dd.get_prediction_results(bundle)

    if "executive_summary" in sections:
        story.append(Paragraph("Executive Summary", styles["H2c"]))
        overall_rate = df["cv_risk_event"].mean() * 100
        story.append(Paragraph(
            f"This report summarizes a machine-learning clustering analysis of blood pressure variability "
            f"(BPV) in {len(df)} hemodialysis patients following arteriovenous fistula (AVF) creation. "
            f"Three clinically distinct BPV phenotypes were identified (Low / Medium / High BPV) using "
            f"Ward's hierarchical clustering, the best-performing of four algorithms evaluated. Overall "
            f"cardiovascular (CV) event rate in the cohort was {overall_rate:.1f}%.", styles["Bodyc"]))
        story.append(Spacer(1, 8))

    if "methodology" in sections:
        story.append(Paragraph("Methodology", styles["H2c"]))
        story.append(Paragraph(
            "Four unsupervised clustering algorithms were compared: K-Means, Partitioning Around Medoids "
            "(PAM), Ward's hierarchical clustering, and Expectation-Maximization (Gaussian Mixture Model). "
            "Internal validation used silhouette score, Davies-Bouldin index, and Calinski-Harabasz score. "
            "Cardiovascular risk prediction used Random Forest, XGBoost/Gradient Boosting, and Logistic "
            "Regression classifiers, evaluated via ROC-AUC, precision, recall, and F1-score on a held-out "
            "test split.", styles["Bodyc"]))
        story.append(Spacer(1, 8))

    if "results" in sections:
        story.append(Paragraph("Results — Cluster Cardiovascular Risk", styles["H2c"]))
        table_data = [["Cluster", "N", "Events", "CV Risk %", "95% CI"]]
        for _, r in cluster_summary.iterrows():
            table_data.append([
                r["cluster"], str(int(r["n"])), str(int(r["events"])), f"{r['risk_pct']:.1f}%",
                f"[{r['ci_lower_pct']:.1f}, {r['ci_upper_pct']:.1f}]",
            ])
        t = Table(table_data, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BRAND_BLUE),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e1e0d9")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9f9f7")]),
        ]))
        story.append(t)
        story.append(Spacer(1, 10))

        story.append(Paragraph("Predictive Model Performance", styles["H2c"]))
        table_data = [["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]]
        for _, r in pred_results.iterrows():
            table_data.append([r["model"], f"{r['accuracy']:.3f}", f"{r['precision']:.3f}",
                                f"{r['recall']:.3f}", f"{r['f1_score']:.3f}", f"{r['roc_auc']:.3f}"])
        t2 = Table(table_data, hAlign="LEFT")
        t2.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BRAND_RED),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e1e0d9")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9f9f7")]),
        ]))
        story.append(t2)
        story.append(Spacer(1, 10))

    if "clinical_recommendations" in sections:
        story.append(Paragraph("Clinical Recommendations", styles["H2c"]))
        recs = [
            ("High BPV cluster", "Aggressive BP monitoring, frequent follow-up, consider antihypertensive "
             "adjustment and dialysis-session frequency review; evaluate for cardiovascular complications."),
            ("Medium BPV cluster", "Standard monitoring with lifestyle modification counseling; maintain "
             "current therapy and reassess quarterly."),
            ("Low BPV cluster", "Routine care; continue current management with annual reassessment."),
        ]
        for title, text in recs:
            story.append(Paragraph(f"<b>{title}:</b> {text}", styles["Bodyc"]))
        story.append(Spacer(1, 8))

    if "limitations" in sections:
        story.append(Paragraph("Limitations & Future Directions", styles["H2c"]))
        story.append(Paragraph(
            "This report is generated from a synthetic cohort calibrated to published summary statistics "
            "and is intended for methodological demonstration, not clinical decision-making on real "
            "patients. External validation on prospective multi-center hemodialysis cohorts, longer "
            "follow-up, and integration with ambulatory BP monitoring are recommended next steps.",
            styles["Bodyc"]))

    doc.build(story)
    return buf.getvalue()
