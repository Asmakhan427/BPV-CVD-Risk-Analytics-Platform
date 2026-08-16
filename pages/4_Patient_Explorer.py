"""Page 5: Patient Explorer — individual patient profile, trajectory, cluster standing."""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from bpv_cvd import palette as pal
from bpv_cvd.dashboard_data import get_timeline_data, predict_patient_by_id
from dash_common import cluster_badge, configure_page, get_bundle, risk_badge, sidebar_nav_footer

configure_page("Patient Explorer")
st.title("Patient Explorer")

bundle = get_bundle()
df = bundle["df"]

patient_id = st.selectbox("Select patient", options=df["patient_id"].tolist())
row = df[df["patient_id"] == patient_id].iloc[0]

st.divider()

# --------------------------------------------------------------- Profile card
c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    st.subheader(f"Patient {patient_id}")
    st.markdown(cluster_badge(row["cluster"]), unsafe_allow_html=True)
    st.write("")
    st.markdown(
        f"**Age:** {row['age']:.0f}  \n"
        f"**Sex:** {row['sex']}  \n"
        f"**BMI:** {row['bmi']:.1f}  \n"
        f"**AVF Location:** {row['avf_location']}  \n"
        f"**Dialysis Vintage:** {row['dialysis_vintage_months']:.0f} months"
    )
with c2:
    st.subheader("Comorbidities")
    comorbs = {
        "Diabetes": row["diabetes"], "Hypertension": row["hypertension"],
        "Coronary Artery Disease": row["coronary_artery_disease"], "Prior Stroke": row["prior_stroke"],
    }
    for k, v in comorbs.items():
        st.markdown(f"**{k}:** {'Yes' if v else 'No'}")
with c3:
    st.subheader("BPV Measurements")
    st.metric("Systolic BPV", f"{row['sbpv']:.2f}%")
    st.metric("Diastolic BPV", f"{row['dbpv']:.2f}%")
    st.metric("Mean SBP / DBP", f"{row['sbp_mean']:.0f} / {row['dbp_mean']:.0f} mmHg")

st.divider()

# --------------------------------------------------------------- Risk gauge + trajectory
c4, c5 = st.columns([1, 1.4])
with c4:
    st.subheader("CV Risk Assessment")
    result = predict_patient_by_id(bundle, patient_id)
    st.markdown(risk_badge(result["risk_level"]), unsafe_allow_html=True)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=result["probability"] * 100, number={"suffix": "%"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": pal.CATEGORICAL[0]},
               "steps": [{"range": [0, 15], "color": pal.hex_to_rgba(pal.STATUS["good"], 0.2)},
                         {"range": [15, 35], "color": pal.hex_to_rgba(pal.STATUS["warning"], 0.2)},
                         {"range": [35, 100], "color": pal.hex_to_rgba(pal.STATUS["critical"], 0.2)}]},
    ))
    fig.update_layout(template=pal.PLOTLY_TEMPLATE, height=280, margin=dict(t=10, b=10, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Actual recorded CV event: {'Yes' if row['cv_risk_event'] else 'No'}")

with c5:
    st.subheader("Blood Pressure Trajectory (synthetic dialysis-session series)")
    traj = get_timeline_data(bundle, patient_id)
    fig = go.Figure()
    fig.add_scatter(x=traj["session"], y=traj["sbp"], mode="lines+markers", name="SBP",
                     line=dict(color=pal.CATEGORICAL[0], width=2.5))
    fig.add_scatter(x=traj["session"], y=traj["dbp"], mode="lines+markers", name="DBP",
                     line=dict(color=pal.CATEGORICAL[1], width=2.5))
    fig.update_layout(template=pal.PLOTLY_TEMPLATE, height=350, xaxis_title="Dialysis session", yaxis_title="mmHg")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# --------------------------------------------------------------- Cluster comparison
st.subheader("Where This Patient Stands vs. Cluster Peers")
feat = st.selectbox("Feature", ["sbpv", "dbpv", "sbp_mean", "dbp_mean", "pulse_pressure", "age", "bmi"])
fig = px.box(df, x="cluster", y=feat, color="cluster", category_orders={"cluster": pal.CLUSTER_ORDER},
             color_discrete_map=pal.CLUSTER_COLORS, template=pal.PLOTLY_TEMPLATE, points="outliers")
fig.add_scatter(x=[row["cluster"]], y=[row[feat]], mode="markers",
                 marker=dict(size=16, color="black", symbol="star", line=dict(width=2, color="white")),
                 name=patient_id)
fig.update_layout(height=420, showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Treatment Recommendation")
recs = {
    "High BPV": "**Aggressive BP monitoring.** Frequent follow-ups, consider antihypertensive adjustment, "
                "evaluate dialysis-session frequency, and screen for cardiovascular complications.",
    "Medium BPV": "**Standard monitoring.** Lifestyle modification counseling and regular BP checks; "
                  "reassess quarterly.",
    "Low BPV": "**Routine care.** Maintain current management with annual review.",
}
st.info(recs.get(row["cluster"], ""))

sidebar_nav_footer()
