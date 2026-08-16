"""Page 6: Clinical Insights — cluster interpretation, management recommendations, guidelines."""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from bpv_cvd import analysis
from bpv_cvd import palette as pal
from dash_common import cluster_badge, configure_page, get_bundle, sidebar_nav_footer

configure_page("Clinical Insights")
st.title("Clinical Insights")

bundle = get_bundle()
df = bundle["df"]
risk_table = analysis.calculate_cv_risk(df, cluster_col="cluster")
profiles = analysis.demographic_profiles(df, cluster_col="cluster")

CLUSTER_DETAILS = {
    "High BPV": {
        "range": "SBPV > 13%, DBPV > 6%",
        "characteristics": "Highest BP variability; older patients, higher rates of diabetes and CAD.",
        "recommendation": "Aggressive BP monitoring, frequent follow-up, consider antihypertensive "
                           "adjustment, and evaluate for cardiovascular complications.",
    },
    "Medium BPV": {
        "range": "SBPV 8-13%, DBPV 4-6%",
        "characteristics": "Moderate BP variability, intermediate comorbidity burden.",
        "recommendation": "Standard monitoring, lifestyle modification counseling, quarterly review.",
    },
    "Low BPV": {
        "range": "SBPV < 8%, DBPV < 4%",
        "characteristics": "Lowest BP variability; younger patients, fewer comorbidities.",
        "recommendation": "Routine care, annual review.",
    },
}

for cluster in pal.CLUSTER_ORDER:
    detail = CLUSTER_DETAILS[cluster]
    risk_row = risk_table[risk_table["cluster"] == cluster]
    risk_pct = risk_row["risk_pct"].iloc[0] if not risk_row.empty else float("nan")
    n = risk_row["n"].iloc[0] if not risk_row.empty else 0

    with st.container(border=True):
        h1, h2 = st.columns([3, 1])
        with h1:
            st.markdown(f"### {cluster}")
            st.markdown(cluster_badge(cluster), unsafe_allow_html=True)
        with h2:
            st.metric("CV Risk Rate", f"{risk_pct:.1f}%", help=f"n = {n} patients")

        st.markdown(f"**Range:** {detail['range']}")
        st.markdown(f"**Characteristics:** {detail['characteristics']}")
        st.markdown(f"**Recommendation:** {detail['recommendation']}")

st.divider()

st.subheader("CV Risk by Cluster")
fig = px.bar(risk_table, x="cluster", y="risk_pct", color="cluster",
             category_orders={"cluster": pal.CLUSTER_ORDER}, color_discrete_map=pal.CLUSTER_COLORS,
             text=[f"{v:.1f}%" for v in risk_table["risk_pct"]], template=pal.PLOTLY_TEMPLATE)
fig.update_traces(textposition="outside")
fig.update_layout(height=380, showlegend=False, yaxis_title="CV Risk (%)")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Comparison with Montoya et al. (2025)")
compare_df = risk_table[["cluster", "risk_pct"]].rename(columns={"risk_pct": "This Cohort (%)"})
compare_df["Published (%)"] = compare_df["cluster"].map({
    "High BPV": 42.9, "Medium BPV": 16.7, "Low BPV": 12.0,
})
fig = go.Figure()
fig.add_bar(name="This Cohort", x=compare_df["cluster"], y=compare_df["This Cohort (%)"], marker_color=pal.CATEGORICAL[0])
fig.add_bar(name="Published", x=compare_df["cluster"], y=compare_df["Published (%)"], marker_color=pal.CATEGORICAL[1])
fig.update_layout(template=pal.PLOTLY_TEMPLATE, barmode="group", height=400, yaxis_title="CV Risk (%)")
st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("Limitations")
st.markdown(
    "- Data is synthetically generated; not a clinical decision tool.\n"
    "- Original cohort (n=83) is modest; external validation is needed.\n"
    "- Future work: cluster-transition tracking, integration with additional cardiovascular measures."
)

sidebar_nav_footer()
