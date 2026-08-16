"""
BPV-CVD Risk Analytics Platform — Streamlit entry point (Home page).

Run with:  streamlit run app.py
"""
import plotly.graph_objects as go
import streamlit as st

from bpv_cvd import palette as pal
from dash_common import (
    APP_TITLE,
    configure_page,
    get_bundle,
    sidebar_nav_footer,
    stat_card,
)

configure_page("Home")

st.sidebar.title(APP_TITLE)

bundle = get_bundle()
df = bundle["df"]
metrics_df = bundle["metrics_df"]

st.title(APP_TITLE)
st.caption(
    f"Blood pressure variability and cardiovascular risk clustering in hemodialysis patients "
    f"(synthetic data, n={len(df)}, based on Montoya et al. 2025)."
)

st.divider()

# ---------------------------------------------------------------- Stat cards
overall_rate = df["cv_risk_event"].mean() * 100
n_clusters = df["cluster"].nunique()

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(stat_card("Total Patients", f"{len(df)}"), unsafe_allow_html=True)
with c2:
    st.markdown(stat_card("Clusters Identified", f"{n_clusters}"), unsafe_allow_html=True)
with c3:
    st.markdown(stat_card("Best Algorithm", "Ward's Method",
                           delta=f"Silhouette {metrics_df.set_index('algorithm').loc['Ward','silhouette_score']:.3f}",
                           delta_color=pal.STATUS["good"]), unsafe_allow_html=True)
with c4:
    st.markdown(stat_card("Overall CV Risk Rate", f"{overall_rate:.1f}%",
                           delta=f"{int(df['cv_risk_event'].sum())} events / {len(df)} patients"),
                unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------------- Overview row
left, right = st.columns([1.1, 1])

with left:
    st.subheader("Cluster Distribution")
    counts = df["cluster"].value_counts().reindex(pal.CLUSTER_ORDER).dropna()
    fig = go.Figure(data=[go.Pie(
        labels=counts.index, values=counts.values, hole=0.45,
        marker=dict(colors=[pal.CLUSTER_COLORS[c] for c in counts.index], line=dict(color="white", width=2)),
        textinfo="label+percent",
    )])
    fig.update_layout(template=pal.PLOTLY_TEMPLATE, showlegend=False, height=380, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("CV Risk by Cluster")
    from bpv_cvd import analysis
    risk_table = analysis.calculate_cv_risk(df, cluster_col="cluster")
    fig2 = go.Figure()
    fig2.add_bar(
        x=risk_table["cluster"], y=risk_table["risk_pct"],
        marker_color=[pal.CLUSTER_COLORS.get(c, pal.CATEGORICAL[0]) for c in risk_table["cluster"]],
        error_y=dict(type="data", symmetric=False,
                      array=risk_table["ci_upper_pct"] - risk_table["risk_pct"],
                      arrayminus=risk_table["risk_pct"] - risk_table["ci_lower_pct"],
                      color=pal.INK_SECONDARY),
        text=[f"{v:.1f}%" for v in risk_table["risk_pct"]], textposition="outside",
    )
    fig2.update_layout(template=pal.PLOTLY_TEMPLATE, height=380, yaxis_title="CV Risk (%)", xaxis_title="",
                        margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

st.subheader("Quick Summary")
qc1, qc2, qc3 = st.columns(3)
qc1.metric("Mean SBPV", f"{df['sbpv'].mean():.2f}")
qc2.metric("Mean DBPV", f"{df['dbpv'].mean():.2f}")
qc3.metric("Mean Age", f"{df['age'].mean():.1f} yrs")

sidebar_nav_footer()
