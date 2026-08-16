"""Page 8: Reports — PDF report generation and data export."""
import streamlit as st

from bpv_cvd.dashboard_data import get_cluster_summary, get_prediction_results
from bpv_cvd.report import SECTIONS, generate_pdf_report
from dash_common import configure_page, download_buttons, get_bundle, sidebar_nav_footer

configure_page("Reports")
st.title("Reports")

bundle = get_bundle()
df = bundle["df"]

st.subheader("Custom Report Builder")
section_labels = {
    "executive_summary": "Executive Summary",
    "methodology": "Methodology",
    "results": "Results",
    "clinical_recommendations": "Clinical Recommendations",
    "limitations": "Limitations",
}
selected = st.multiselect("Sections to include", options=SECTIONS,
                           format_func=lambda s: section_labels[s], default=SECTIONS)

if st.button("Generate PDF Report", type="primary"):
    with st.spinner("Building report..."):
        pdf_bytes = generate_pdf_report(bundle, sections=selected)
    st.session_state["report_pdf"] = pdf_bytes
    st.success("Report generated.")

if "report_pdf" in st.session_state:
    st.download_button("Download PDF Report", st.session_state["report_pdf"],
                        file_name="bpv_cvd_analytics_report.pdf", mime="application/pdf")

st.divider()

overall_rate = df["cv_risk_event"].mean() * 100
st.markdown(
    f"**{len(df)} patients**, 3 BPV clusters identified via Ward's hierarchical clustering. "
    f"Overall CV event rate: **{overall_rate:.1f}%**."
)

st.subheader("Cluster Risk Table")
cluster_summary = get_cluster_summary(bundle)
st.dataframe(cluster_summary, use_container_width=True)
download_buttons(cluster_summary, "cluster_risk_summary", key_prefix="cluster_summary")

st.subheader("Model Performance Table")
pred_results = get_prediction_results(bundle)
st.dataframe(pred_results, use_container_width=True)
download_buttons(pred_results, "model_performance", key_prefix="model_perf")

st.subheader("Full Patient Dataset")
download_buttons(df, "bpv_cvd_full_dataset", key_prefix="full_dataset")

sidebar_nav_footer()
