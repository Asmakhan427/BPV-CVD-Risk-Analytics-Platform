"""Page 7: Model Performance — full comparison across predictive models, SHAP, CV, learning curves."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import auc

from bpv_cvd import palette as pal
from dash_common import configure_page, get_bundle, sidebar_nav_footer

configure_page("Model Performance")
st.title("Model Performance")

bundle = get_bundle()
model_results = bundle["model_results"]
X_train, y_train = bundle["X_train"], bundle["y_train"]
X_test, y_test = bundle["X_test"], bundle["y_test"]

st.subheader("Metrics Comparison Table")
rows = []
for name, res in model_results.items():
    m = res["metrics"]
    rows.append({"Model": name, "Accuracy": m["accuracy"], "Precision": m["precision"],
                 "Recall": m["recall"], "F1 Score": m["f1_score"], "ROC-AUC": m["roc_auc"]})
metrics_table = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
st.dataframe(metrics_table.style.background_gradient(subset=["ROC-AUC"], cmap="Blues"), use_container_width=True)

st.subheader("ROC Curves — All Models")
fig = go.Figure()
for i, (name, res) in enumerate(model_results.items()):
    fpr = res["metrics"]["roc_curve"]["fpr"]
    tpr = res["metrics"]["roc_curve"]["tpr"]
    a = auc(fpr, tpr)
    fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={a:.3f})",
                     line=dict(color=pal.CATEGORICAL[i % len(pal.CATEGORICAL)], width=2.5))
fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=pal.BASELINE, dash="dash"), showlegend=False)
fig.update_layout(template=pal.PLOTLY_TEMPLATE, height=460, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Precision-Recall Curves — All Models")
fig = go.Figure()
for i, (name, res) in enumerate(model_results.items()):
    prec = res["metrics"]["pr_curve"]["precision"]
    rec = res["metrics"]["pr_curve"]["recall"]
    fig.add_scatter(x=rec, y=prec, mode="lines", name=name, line=dict(color=pal.CATEGORICAL[i % len(pal.CATEGORICAL)], width=2.5))
fig.update_layout(template=pal.PLOTLY_TEMPLATE, height=460, xaxis_title="Recall", yaxis_title="Precision")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Feature Importance Across Models")
imp_cols = st.columns(len(model_results))
for col, (name, res) in zip(imp_cols, model_results.items()):
    with col:
        st.caption(name)
        if res["importance"] is not None:
            top = res["importance"].head(8)
            fig = px.bar(top, x="importance", y="feature", orientation="h", template=pal.PLOTLY_TEMPLATE,
                         color_discrete_sequence=[pal.CATEGORICAL[0]])
            fig.update_layout(height=320, yaxis={"categoryorder": "total ascending"}, margin=dict(l=0, r=0, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No native feature importance for this model.")

st.subheader("Cross-Validation (5-fold ROC-AUC)")
cv_rows = []
with st.spinner("Running cross-validation..."):
    for name, res in model_results.items():
        cv = res["model"].cross_validate(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]), cv=5)
        cv_rows.append({"Model": name, "Mean AUC": cv["mean"], "Std": cv["std"], "Folds": cv["scores"]})
cv_df = pd.DataFrame(cv_rows)
fig = go.Figure()
fig.add_bar(x=cv_df["Model"], y=cv_df["Mean AUC"], error_y=dict(type="data", array=cv_df["Std"]),
            marker_color=[pal.CATEGORICAL[i % len(pal.CATEGORICAL)] for i in range(len(cv_df))])
fig.update_layout(template=pal.PLOTLY_TEMPLATE, height=380, yaxis_title="ROC-AUC (5-fold CV)")
st.plotly_chart(fig, use_container_width=True)
st.dataframe(cv_df[["Model", "Mean AUC", "Std"]], use_container_width=True)

st.subheader("Learning Curves")
lc_model = st.selectbox("Model for learning curve", list(model_results.keys()))
with st.spinner("Computing learning curve..."):
    lc = model_results[lc_model]["model"].learning_curve_data(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))
fig = go.Figure()
fig.add_scatter(x=lc["train_sizes"], y=lc["train_mean"], mode="lines+markers", name="Training score",
                 line=dict(color=pal.CATEGORICAL[0]))
fig.add_scatter(x=lc["train_sizes"], y=lc["test_mean"], mode="lines+markers", name="Cross-val score",
                 line=dict(color=pal.CATEGORICAL[1]))
fig.update_layout(template=pal.PLOTLY_TEMPLATE, height=420, xaxis_title="Training set size", yaxis_title="ROC-AUC")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Model Explainability (SHAP)")
shap_model_name = st.selectbox("Model for SHAP analysis", ["Random Forest", "XGBoost"], key="shap_model")
try:
    import shap

    with st.spinner("Computing SHAP values..."):
        m = model_results[shap_model_name]["model"]
        explainer = shap.TreeExplainer(m.model)
        sample = X_test.sample(min(30, len(X_test)), random_state=42)
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
        mean_abs = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({"feature": sample.columns, "mean_abs_shap": mean_abs}).sort_values(
            "mean_abs_shap", ascending=False).head(12)
        fig = px.bar(shap_df, x="mean_abs_shap", y="feature", orientation="h", template=pal.PLOTLY_TEMPLATE,
                     color_discrete_sequence=[pal.CATEGORICAL[6]], title=f"Mean |SHAP value| — {shap_model_name}")
        fig.update_layout(height=420, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
except ImportError:
    st.warning("Install `shap` to enable model explainability plots (`pip install shap`).")
except Exception as e:  # pragma: no cover
    st.warning(f"SHAP computation unavailable for this model: {e}")

sidebar_nav_footer()
