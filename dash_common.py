"""
Shared helpers for every page of the Streamlit dashboard: sidebar cohort
controls that drive the whole pipeline, cached pipeline loading keyed off
those controls, consistent page config, and stat-card / badge components.
Imported by app.py and every module under pages/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bpv_cvd import dashboard_data as dd  # noqa: E402
from bpv_cvd import palette as pal  # noqa: E402

APP_TITLE = "BPV-CVD Risk Analytics Platform"

N_CLUSTERS = 3  # fixed: clinical interpretation assumes Low/Medium/High BPV


def configure_page(page_title: str, layout: str = "wide") -> None:
    st.set_page_config(page_title=f"{page_title} - {APP_TITLE}", layout=layout,
                        initial_sidebar_state="expanded")
    _inject_css()
    _render_cohort_controls()


def _inject_css() -> None:
    st.markdown(f"""
    <style>
    .stat-card {{
        background: {pal.SURFACE};
        border: 1px solid {pal.GRIDLINE};
        border-radius: 10px;
        padding: 1rem 1.1rem;
        box-shadow: 0 1px 2px rgba(11,11,11,0.04);
    }}
    .stat-card .label {{
        color: {pal.INK_MUTED};
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.25rem;
    }}
    .stat-card .value {{
        color: {pal.INK_PRIMARY};
        font-size: 1.7rem;
        font-weight: 700;
        line-height: 1.1;
    }}
    .stat-card .delta {{
        font-size: 0.82rem;
        margin-top: 0.2rem;
    }}
    .risk-badge {{
        display: inline-block;
        padding: 0.15rem 0.65rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.82rem;
        color: white;
    }}
    section[data-testid="stSidebar"] .block-container {{
        padding-top: 1.2rem;
    }}
    </style>
    """, unsafe_allow_html=True)


def _render_cohort_controls() -> None:
    """Sidebar controls that parameterize the whole pipeline. Any change here
    triggers a rerun; get_bundle() picks up the new values and recomputes."""
    st.sidebar.subheader("Cohort settings")
    st.sidebar.slider("Patients", min_value=50, max_value=250, value=83, step=1, key="cohort_n")
    st.sidebar.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1, key="cohort_seed")
    st.sidebar.divider()


@st.cache_resource(show_spinner="Recomputing clustering and predictive models...")
def _run_pipeline(n: int, seed: int, n_clusters: int) -> dict:
    return dd.build_full_pipeline(n=n, seed=seed, n_clusters=n_clusters)


def get_bundle() -> dict:
    """Run (or fetch the cached result of) the pipeline for the current
    sidebar-selected cohort settings. Changing a control changes the cache
    key, so the whole dashboard recomputes and re-renders automatically."""
    n = st.session_state.get("cohort_n", 83)
    seed = st.session_state.get("cohort_seed", 42)
    return _run_pipeline(n=n, seed=seed, n_clusters=N_CLUSTERS)


def stat_card(label: str, value: str, delta: str | None = None, delta_color: str | None = None) -> str:
    delta_html = f'<div class="delta" style="color:{delta_color or pal.INK_MUTED}">{delta}</div>' if delta else ""
    return f"""
    <div class="stat-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {delta_html}
    </div>
    """


def risk_badge(level: str) -> str:
    color = {"Low": pal.STATUS["good"], "Moderate": pal.STATUS["warning"], "High": pal.STATUS["critical"]}.get(level, pal.INK_MUTED)
    return f'<span class="risk-badge" style="background:{color}">{level} Risk</span>'


def cluster_badge(cluster: str) -> str:
    color = pal.CLUSTER_COLORS.get(cluster, pal.INK_MUTED)
    return f'<span class="risk-badge" style="background:{color}">{cluster}</span>'


def sidebar_nav_footer() -> None:
    st.sidebar.markdown("---")
    st.sidebar.caption("BPV-CVD Risk Analytics Platform. Synthetic data. Not for clinical use.")


def download_buttons(df: pd.DataFrame, base_name: str, key_prefix: str = "") -> None:
    """Standard CSV / Excel / JSON download row for a dataframe."""
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("CSV", df.to_csv(index=False).encode("utf-8"),
                            file_name=f"{base_name}.csv", mime="text/csv", key=f"{key_prefix}_csv")
    with c2:
        import io
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="data")
        st.download_button("Excel", buf.getvalue(), file_name=f"{base_name}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"{key_prefix}_xlsx")
    with c3:
        st.download_button("JSON", df.to_json(orient="records", indent=2).encode("utf-8"),
                            file_name=f"{base_name}.json", mime="application/json", key=f"{key_prefix}_json")
