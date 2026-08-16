"""Page 2: Data Explorer — raw data, summary stats, correlations, distributions."""
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

from bpv_cvd import palette as pal
from bpv_cvd.analysis import DEFAULT_FEATURES
from dash_common import configure_page, download_buttons, get_bundle, sidebar_nav_footer

configure_page("Data Explorer")
st.title("Data Explorer")

bundle = get_bundle()
df = bundle["df"]

tab1, tab2, tab3, tab4 = st.tabs(["Raw Data", "Summary Stats", "Correlations", "Distributions"])

with tab1:
    st.subheader("Patient-Level Dataset")
    clusters_filter = st.multiselect("Filter by cluster", options=pal.CLUSTER_ORDER, default=pal.CLUSTER_ORDER)
    sex_filter = st.multiselect("Filter by sex", options=sorted(df["sex"].unique()), default=sorted(df["sex"].unique()))
    filtered = df[df["cluster"].isin(clusters_filter) & df["sex"].isin(sex_filter)]
    st.dataframe(filtered, use_container_width=True, height=420)
    st.caption(f"Showing {len(filtered)} of {len(df)} patients")
    download_buttons(filtered, "bpv_cvd_patient_data", key_prefix="raw")

with tab2:
    st.subheader("Summary Statistics")
    numeric_cols = df.select_dtypes("number").columns.tolist()
    st.dataframe(df[numeric_cols].describe().T.round(2), use_container_width=True)

    st.subheader("Categorical Breakdown")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.caption("Sex")
        st.dataframe(df["sex"].value_counts().rename("count"))
    with cc2:
        st.caption("AVF Location")
        st.dataframe(df["avf_location"].value_counts().rename("count"))

with tab3:
    st.subheader("Interactive Correlation Heatmap")
    features = st.multiselect("Features", options=[c for c in DEFAULT_FEATURES if c in df.columns],
                               default=[c for c in DEFAULT_FEATURES if c in df.columns])
    if len(features) >= 2:
        corr = df[features].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         aspect="auto", template=pal.PLOTLY_TEMPLATE)
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Select at least two features.")

    st.subheader("Pairwise Feature Relationships (colored by cluster)")
    pair_features = st.multiselect("Pairplot features", options=[c for c in DEFAULT_FEATURES if c in df.columns],
                                    default=["sbpv", "dbpv", "age", "pulse_pressure"], key="pairplot_feats")
    if len(pair_features) >= 2:
        fig2 = px.scatter_matrix(df, dimensions=pair_features, color="cluster",
                                  category_orders={"cluster": pal.CLUSTER_ORDER},
                                  color_discrete_map=pal.CLUSTER_COLORS, template=pal.PLOTLY_TEMPLATE)
        fig2.update_traces(diagonal_visible=False, marker=dict(size=5, opacity=0.75, line=dict(width=0.5, color="white")))
        fig2.update_layout(height=650)
        st.plotly_chart(fig2, use_container_width=True)

with tab4:
    st.subheader("Feature Distributions by Cluster")
    feat = st.selectbox("Feature", options=[c for c in DEFAULT_FEATURES if c in df.columns], index=2)
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        fig3 = px.histogram(df, x=feat, color="cluster", marginal="rug", barmode="overlay", opacity=0.65,
                             category_orders={"cluster": pal.CLUSTER_ORDER}, color_discrete_map=pal.CLUSTER_COLORS,
                             template=pal.PLOTLY_TEMPLATE)
        fig3.update_layout(height=420, title=f"Distribution of {feat} (overlaid by cluster)")
        st.plotly_chart(fig3, use_container_width=True)
    with dcol2:
        fig4 = px.box(df, x="cluster", y=feat, color="cluster", category_orders={"cluster": pal.CLUSTER_ORDER},
                       color_discrete_map=pal.CLUSTER_COLORS, template=pal.PLOTLY_TEMPLATE, points="all")
        fig4.update_layout(height=420, showlegend=False, title=f"{feat} by Cluster (box plot)")
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Violin Plot — Feature by CV Risk Status")
    plot_df = df.copy()
    plot_df["Risk Status"] = plot_df["cv_risk_event"].map({1: "CV Event", 0: "No Event"})
    fig5 = px.violin(plot_df, x="Risk Status", y=feat, color="Risk Status", box=True, points="all",
                      color_discrete_map={"No Event": pal.STATUS["good"], "CV Event": pal.STATUS["critical"]},
                      template=pal.PLOTLY_TEMPLATE)
    fig5.update_layout(height=420, showlegend=False)
    st.plotly_chart(fig5, use_container_width=True)

sidebar_nav_footer()
