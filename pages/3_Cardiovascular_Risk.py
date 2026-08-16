"""Page 4: Cardiovascular Risk Analysis — risk by cluster, ANOVA, prediction sandbox, ROC/CM."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import auc, confusion_matrix, roc_curve

from bpv_cvd import analysis
from bpv_cvd import palette as pal
from dash_common import configure_page, get_bundle, risk_badge, sidebar_nav_footer

configure_page("Cardiovascular Risk")
st.title("Cardiovascular Risk Analysis")

bundle = get_bundle()
df = bundle["df"]
model_results = bundle["model_results"]

tabs = st.tabs(["Risk by Cluster", "Statistical Tests", "Predict Patient Risk", "Model Diagnostics"])

# --------------------------------------------------------------- Risk by cluster
with tabs[0]:
    risk_table = analysis.calculate_cv_risk(df, cluster_col="cluster")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(risk_table, names="cluster", values="events", hole=0.45,
                     category_orders={"cluster": pal.CLUSTER_ORDER}, color="cluster",
                     color_discrete_map=pal.CLUSTER_COLORS, title="CV Events by Cluster")
        fig.update_layout(template=pal.PLOTLY_TEMPLATE, height=400)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_bar(x=risk_table["cluster"], y=risk_table["risk_pct"],
                    marker_color=[pal.CLUSTER_COLORS[c] for c in risk_table["cluster"]],
                    error_y=dict(type="data", symmetric=False,
                                 array=risk_table["ci_upper_pct"] - risk_table["risk_pct"],
                                 arrayminus=risk_table["risk_pct"] - risk_table["ci_lower_pct"]),
                    text=[f"{v:.1f}%" for v in risk_table["risk_pct"]], textposition="outside")
        fig.update_layout(template=pal.PLOTLY_TEMPLATE, height=400, title="CV Risk Rate (95% CI)", yaxis_title="Risk (%)")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(risk_table, use_container_width=True)

    st.subheader("SBPV vs DBPV, colored by CV Risk")
    plot_df = df.copy()
    plot_df["Risk Status"] = plot_df["cv_risk_event"].map({1: "CV Event", 0: "No Event"})
    fig = px.scatter(plot_df, x="sbpv", y="dbpv", color="Risk Status", symbol="cluster",
                      color_discrete_map={"No Event": pal.STATUS["good"], "CV Event": pal.STATUS["critical"]},
                      hover_data=["patient_id", "cluster"], template=pal.PLOTLY_TEMPLATE)
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("BPV by CV Risk Status (violin)")
    vc1, vc2 = st.columns(2)
    for col, feat, label in [(vc1, "sbpv", "Systolic BPV"), (vc2, "dbpv", "Diastolic BPV")]:
        with col:
            fig = px.violin(plot_df, x="Risk Status", y=feat, color="Risk Status", box=True, points="all",
                             color_discrete_map={"No Event": pal.STATUS["good"], "CV Event": pal.STATUS["critical"]},
                             template=pal.PLOTLY_TEMPLATE, title=label)
            fig.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------- Statistical tests
with tabs[1]:
    st.subheader("One-Way ANOVA — Feature Differences Across Clusters")
    anova_df = analysis.compare_clusters(df, cluster_col="cluster")
    st.dataframe(anova_df.style.apply(
        lambda r: [f"background-color:{pal.STATUS['good']}22" if r["significant"] else "" for _ in r], axis=1
    ), use_container_width=True)

    st.subheader("Risk Factor Correlation Matrix")
    features = ["sbpv", "dbpv", "age", "bmi", "pulse_pressure", "crp", "cv_risk_event"]
    corr = df[features].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, template=pal.PLOTLY_TEMPLATE)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Categorical Association (Chi-Square)")
    cat_col = st.selectbox("Categorical variable", ["sex", "avf_location", "diabetes", "hypertension"])
    result = analysis.chi_square_association(df, "cluster", cat_col)
    st.json(result)

# --------------------------------------------------------------- Predict
with tabs[2]:
    st.subheader("Interactive Cardiovascular Risk Predictor")

    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1:
        age = st.slider("Age", 21, 90, 60)
        bmi = st.slider("BMI", 16.0, 48.0, 27.0)
        dialysis_vintage = st.slider("Dialysis vintage (months)", 1, 180, 24)
    with pcol2:
        sbp_mean = st.slider("Mean SBP (mmHg)", 90, 190, 140)
        sbpv = st.slider("Systolic BPV (%)", 2.0, 22.0, 10.0)
        dbp_mean = st.slider("Mean DBP (mmHg)", 50, 110, 78)
        dbpv = st.slider("Diastolic BPV (%)", 1.0, 12.0, 5.0)
    with pcol3:
        albumin = st.slider("Albumin (g/dL)", 2.4, 4.8, 3.8)
        hemoglobin = st.slider("Hemoglobin (g/dL)", 7.5, 15.5, 11.0)
        crp = st.slider("CRP (mg/L)", 0.2, 60.0, 5.0)
        ufr = st.slider("Ultrafiltration rate (mL/kg/h)", 3.0, 18.0, 9.0)

    comorb1, comorb2, comorb3, comorb4 = st.columns(4)
    diabetes = comorb1.checkbox("Diabetes")
    hypertension = comorb2.checkbox("Hypertension", value=True)
    cad = comorb3.checkbox("Coronary artery disease")
    stroke = comorb4.checkbox("Prior stroke")

    patient_features = {
        "age": age, "bmi": bmi, "dialysis_vintage_months": dialysis_vintage,
        "albumin": albumin, "hemoglobin": hemoglobin, "crp": crp, "ultrafiltration_rate": ufr,
        "sbp_mean": sbp_mean, "sbp_sd": sbp_mean * sbpv / 100, "sbpv": sbpv,
        "dbp_mean": dbp_mean, "dbp_sd": dbp_mean * dbpv / 100, "dbpv": dbpv,
        "pulse_pressure": sbp_mean - dbp_mean,
        "diabetes": int(diabetes), "hypertension": int(hypertension),
        "coronary_artery_disease": int(cad), "prior_stroke": int(stroke),
        "bpv_composite": sbpv * 0.6 + dbpv * 0.4,
        "sbpv_dbpv_ratio": sbpv / dbpv if dbpv else 0,
        "sbpv_age_interaction": sbpv * age / 100,
        "pp_sbpv_interaction": (sbp_mean - dbp_mean) * sbpv / 100,
        "comorbidity_burden": int(diabetes) + int(hypertension) + int(cad) + int(stroke),
        "sex_code": 1, "avf_location_code": 0,
    }

    from bpv_cvd.dashboard_data import predict_patient_risk
    result = predict_patient_risk(bundle, patient_features)

    rc1, rc2 = st.columns([1, 2])
    with rc1:
        st.markdown(risk_badge(result["risk_level"]), unsafe_allow_html=True)
        st.metric("Predicted CV Risk Probability", f"{result['probability']*100:.1f}%")
    with rc2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=result["probability"] * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": pal.CATEGORICAL[0]},
                "steps": [
                    {"range": [0, 15], "color": pal.hex_to_rgba(pal.STATUS["good"], 0.2)},
                    {"range": [15, 35], "color": pal.hex_to_rgba(pal.STATUS["warning"], 0.2)},
                    {"range": [35, 100], "color": pal.hex_to_rgba(pal.STATUS["critical"], 0.2)},
                ],
            },
        ))
        fig.update_layout(template=pal.PLOTLY_TEMPLATE, height=280, margin=dict(t=20, b=10, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Contributions (Random Forest importance)")
    importance = bundle["deployed_model"].feature_importance().head(10)
    fig = px.bar(importance, x="importance", y="feature", orientation="h", template=pal.PLOTLY_TEMPLATE,
                 color_discrete_sequence=[pal.CATEGORICAL[0]])
    fig.update_layout(height=400, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------- Diagnostics
with tabs[3]:
    model_choice = st.selectbox("Model", list(model_results.keys()))
    res = model_results[model_choice]
    metrics = res["metrics"]
    mcols = st.columns(5)
    for col, (label, key) in zip(mcols, [("Accuracy", "accuracy"), ("Precision", "precision"),
                                          ("Recall", "recall"), ("F1", "f1_score"), ("ROC-AUC", "roc_auc")]):
        col.metric(label, f"{metrics[key]:.3f}")

    dcol1, dcol2 = st.columns(2)
    with dcol1:
        fpr, tpr = metrics["roc_curve"]["fpr"], metrics["roc_curve"]["tpr"]
        roc_auc_val = auc(fpr, tpr)
        fig = go.Figure()
        fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={roc_auc_val:.3f}", line=dict(color=pal.CATEGORICAL[0], width=3))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=pal.BASELINE, dash="dash"), showlegend=False)
        fig.update_layout(template=pal.PLOTLY_TEMPLATE, height=420, title="ROC Curve",
                           xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)
    with dcol2:
        cm = np.array(metrics["confusion_matrix"])
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                         x=["No Event", "CV Event"], y=["No Event", "CV Event"],
                         labels=dict(x="Predicted", y="Actual", color="Count"), template=pal.PLOTLY_TEMPLATE)
        fig.update_layout(height=420, title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)

    if res["importance"] is not None:
        st.subheader("Feature Importance")
        fig = px.bar(res["importance"].head(12), x="importance", y="feature", orientation="h",
                     template=pal.PLOTLY_TEMPLATE, color_discrete_sequence=[pal.CATEGORICAL[2]])
        fig.update_layout(height=420, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

sidebar_nav_footer()
