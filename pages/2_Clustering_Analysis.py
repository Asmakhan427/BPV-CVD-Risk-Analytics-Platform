"""Page 3: Clustering Analysis — algorithm comparison, projections, dendrogram, silhouette."""
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples

from bpv_cvd import clustering as clu
from bpv_cvd import palette as pal
from dash_common import configure_page, get_bundle, sidebar_nav_footer

configure_page("Clustering Analysis")
st.title("Clustering Analysis")

bundle = get_bundle()
df = bundle["df"]
X = bundle["X_cluster"]
features = bundle["cluster_features"]
fitted = bundle["fitted_algorithms"]
metrics_df = bundle["metrics_df"]

algo_name = st.selectbox("Algorithm", options=list(fitted.keys()), index=list(fitted.keys()).index("Ward"))
model = fitted[algo_name]
labels = model.labels_
label_names = clu.label_clusters_by_severity(labels, df["sbpv"].values, df["dbpv"].values)
plot_df = df.copy()
plot_df["algo_cluster"] = label_names

tabs = st.tabs([
    "Overview & Metrics", "2D / 3D Scatter", "Parallel Coordinates",
    "PCA / t-SNE", "Dendrogram", "Silhouette",
])

# --------------------------------------------------------------- Overview
with tabs[0]:
    st.subheader("Algorithm Comparison — Internal Validation Metrics")
    st.dataframe(metrics_df.set_index("algorithm").style.highlight_max(
        subset=["silhouette_score", "calinski_harabasz_score"], color=f"{pal.STATUS['good']}33"
    ).highlight_min(subset=["davies_bouldin_score"], color=f"{pal.STATUS['good']}33"), use_container_width=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        fig = px.bar(metrics_df, x="algorithm", y="silhouette_score", color="algorithm",
                     color_discrete_map=pal.ALGORITHM_COLORS, template=pal.PLOTLY_TEMPLATE,
                     title="Silhouette Score (higher is better)")
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
    with m2:
        fig = px.bar(metrics_df, x="algorithm", y="davies_bouldin_score", color="algorithm",
                     color_discrete_map=pal.ALGORITHM_COLORS, template=pal.PLOTLY_TEMPLATE,
                     title="Davies-Bouldin Index (lower is better)")
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
    with m3:
        # Stability index: mean silhouette sample std (lower std = more stable assignment)
        stability = []
        for name, mdl in fitted.items():
            sil = silhouette_samples(X, mdl.labels_)
            stability.append({"algorithm": name, "stability_index": round(float(1 - sil.std()), 3)})
        import pandas as pd
        stab_df = pd.DataFrame(stability)
        fig = px.bar(stab_df, x="algorithm", y="stability_index", color="algorithm",
                     color_discrete_map=pal.ALGORITHM_COLORS, template=pal.PLOTLY_TEMPLATE,
                     title="Stability Index (higher is better)")
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"Interpretation — {algo_name}")
    ev = model.evaluate()
    sep = "good" if ev["silhouette_score"] > 0.5 else "moderate" if ev["silhouette_score"] > 0.25 else "weak"
    st.markdown(
        f"- **Clusters found:** {ev['n_clusters']}\n"
        f"- **Silhouette score:** {ev['silhouette_score']:.3f} ({sep} separation)\n"
        f"- **Davies-Bouldin index:** {ev['davies_bouldin_score']:.3f}\n"
        f"- **Cluster sizes:** {ev.get('cluster_sizes')}"
    )

# --------------------------------------------------------------- Scatter
with tabs[1]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("2D Scatter — SBPV vs DBPV")
        fig = px.scatter(plot_df, x="sbpv", y="dbpv", color="algo_cluster",
                          category_orders={"algo_cluster": pal.CLUSTER_ORDER}, color_discrete_map=pal.CLUSTER_COLORS,
                          hover_data=["patient_id", "age", "sex"], template=pal.PLOTLY_TEMPLATE,
                          labels={"sbpv": "Systolic BPV (%)", "dbpv": "Diastolic BPV (%)"})
        fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("3D Scatter — PCA Components")
        fig3d = px.scatter_3d(plot_df, x="pca1", y="pca2", z="pca3", color="algo_cluster",
                               category_orders={"algo_cluster": pal.CLUSTER_ORDER}, color_discrete_map=pal.CLUSTER_COLORS,
                               hover_data=["patient_id"], template=pal.PLOTLY_TEMPLATE)
        fig3d.update_traces(marker=dict(size=5, line=dict(width=0.5, color="white")))
        fig3d.update_layout(height=480, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig3d, use_container_width=True)

# --------------------------------------------------------------- Parallel coords
with tabs[2]:
    st.subheader("Parallel Coordinates — Feature Profiles Across Clusters")
    cluster_num = {c: i for i, c in enumerate(pal.CLUSTER_ORDER)}
    pc_df = plot_df.copy()
    pc_df["cluster_num"] = pc_df["algo_cluster"].map(cluster_num)
    fig = go.Figure(data=go.Parcoords(
        line=dict(color=pc_df["cluster_num"],
                  colorscale=[[0, pal.CLUSTER_COLORS["Low BPV"]], [0.5, pal.CLUSTER_COLORS["Medium BPV"]], [1, pal.CLUSTER_COLORS["High BPV"]]],
                  showscale=False),
        dimensions=[dict(label=f, values=pc_df[f]) for f in features],
    ))
    fig.update_layout(template=pal.PLOTLY_TEMPLATE, height=480)
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------- PCA / tSNE
with tabs[3]:
    n_comp = st.slider("PCA components to display", 2, 3, 2)
    pca = PCA(n_components=max(n_comp, 2), random_state=42)
    proj = pca.fit_transform(X)
    pc1 = st.columns(2)
    with pc1[0]:
        fig = px.scatter(x=proj[:, 0], y=proj[:, 1], color=label_names,
                          category_orders={"color": pal.CLUSTER_ORDER}, color_discrete_map=pal.CLUSTER_COLORS,
                          labels={"x": f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                                  "y": f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", "color": "Cluster"},
                          template=pal.PLOTLY_TEMPLATE, title="PCA Projection")
        fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    with pc1[1]:
        with st.spinner("Computing t-SNE embedding..."):
            perplexity = min(20, max(5, len(X) // 3))
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="pca")
            tproj = tsne.fit_transform(X)
        fig = px.scatter(x=tproj[:, 0], y=tproj[:, 1], color=label_names,
                          category_orders={"color": pal.CLUSTER_ORDER}, color_discrete_map=pal.CLUSTER_COLORS,
                          labels={"x": "t-SNE 1", "y": "t-SNE 2", "color": "Cluster"},
                          template=pal.PLOTLY_TEMPLATE, title="t-SNE Projection")
        fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------- Dendrogram
with tabs[4]:
    st.subheader("Ward's Method Dendrogram")
    ward_model = fitted["Ward"]
    dendro_fig = ff.create_dendrogram(
        X, orientation="bottom", linkagefun=lambda x: ward_model.linkage_matrix_,
        color_threshold=None,
    )
    dendro_fig.update_layout(template=pal.PLOTLY_TEMPLATE, height=500, xaxis_title="Patient index", yaxis_title="Ward distance")
    for trace in dendro_fig["data"]:
        trace["marker"]["color"] = pal.CATEGORICAL[0]
        trace["line"]["color"] = pal.CATEGORICAL[0]
    st.plotly_chart(dendro_fig, use_container_width=True)

# --------------------------------------------------------------- Silhouette
with tabs[5]:
    st.subheader(f"Silhouette Plot — {algo_name}")
    sil_values = silhouette_samples(X, labels)
    sil_avg = sil_values.mean()
    uniq = sorted(set(labels))
    fig = go.Figure()
    y_lower = 0
    order = np.argsort([np.mean(sil_values[labels == c]) for c in uniq])
    for rank, ci in enumerate(order):
        c = uniq[ci]
        vals = np.sort(sil_values[labels == c])
        y_upper = y_lower + len(vals)
        color = pal.CATEGORICAL[rank % len(pal.CATEGORICAL)]
        fig.add_trace(go.Scatter(x=vals, y=list(range(y_lower, y_upper)), fill="tozerox", mode="lines",
                                  line=dict(width=0.5, color=color), fillcolor=color, name=f"Cluster {c}"))
        y_lower = y_upper + 10
    fig.add_vline(x=sil_avg, line_dash="dash", line_color=pal.STATUS["critical"],
                  annotation_text=f"Mean = {sil_avg:.3f}")
    fig.update_layout(template=pal.PLOTLY_TEMPLATE, height=500, xaxis_title="Silhouette Coefficient",
                       yaxis_title="Patient (grouped by cluster)", yaxis_showticklabels=False)
    st.plotly_chart(fig, use_container_width=True)

sidebar_nav_footer()
