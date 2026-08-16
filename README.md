# BPV-CVD Risk Analytics Platform

**Interactive Machine Learning Dashboard for Blood Pressure Variability Analysis and Cardiovascular Risk Prediction in Hemodialysis Patients**

A complete, production-ready computational reproduction of the analytical approach described in:

> Montoya et al. (2025). *Blood pressure variability and cardiovascular risk assessment using machine learning clustering after arteriovenous fistula creation in hemodialysis patients.*

This project includes a full Python analytics package, an 8-page interactive Streamlit dashboard, a FastAPI service, SQLite persistence, PDF reporting, Docker packaging, and a pytest test suite.

> **Data disclaimer:** All patient data in this project is **synthetically generated** (`bpv_cvd/data_generator.py`) to statistically match the cluster sizes and cardiovascular event rates reported in the paper. No real patient data is used or required. This platform is a methodological/educational demonstration, **not a clinical decision-making tool**.

---

## Features

- **Dynamic dashboard** — sidebar controls (patient count, random seed) regenerate the cohort and rerun the full pipeline live; every page reflects the new data immediately
- **4 clustering algorithms** — K-Means, PAM (K-Medoids), Ward's hierarchical clustering, EM (Gaussian Mixture Model) — with a shared `fit / predict / evaluate / get_metrics` interface
- **3 predictive models** — Random Forest, XGBoost, Logistic Regression — for cardiovascular risk classification
- **8-page interactive dashboard** — Home, Data Explorer, Clustering Analysis, Cardiovascular Risk, Patient Explorer, Clinical Insights, Model Performance, Reports
- **30+ visualizations** — histograms, box/violin plots, correlation heatmaps, pairplots, 2D/3D scatter, dendrogram, silhouette plots, PCA/t-SNE, parallel coordinates, ROC/PR curves, confusion matrices, feature importance, SHAP, radar charts, gauge charts, and more
- **FastAPI backend** with auto-generated Swagger docs (`/docs`)
- **SQLite persistence** for patients, longitudinal BP measurements, cluster assignments, and predictions
- **PDF report generator** (ReportLab) with a configurable section builder
- **Docker & docker-compose** for one-command deployment
- **pytest** unit/integration test suite covering every backend module

---

## Project Structure

```
BPV-CVD Risk Analytics Platform/
├── bpv_cvd/                     # Core analytics package
│   ├── data_generator.py        # Synthetic cohort generation (n=83, matches paper stats)
│   ├── preprocessing.py         # Scaling, imputation, feature engineering, splitting
│   ├── clustering.py            # K-Means, PAM, Ward, EM implementations
│   ├── analysis.py               # ANOVA, CV risk tables, demographics, bootstrap CI
│   ├── prediction.py            # Random Forest / XGBoost / Logistic Regression models
│   ├── visualization.py         # Publication-quality static figures (matplotlib/seaborn)
│   ├── dashboard_data.py        # Cached data-shaping layer used by the dashboard/API
│   ├── database.py              # SQLite persistence layer
│   ├── report.py                # PDF report generation (ReportLab)
│   └── palette.py               # Shared colorblind-safe color tokens
├── app.py                       # Streamlit entry point (Home page)
├── pages/                       # Streamlit multipage app (auto-discovered)
│   ├── 1_Data_Explorer.py
│   ├── 2_Clustering_Analysis.py
│   ├── 3_Cardiovascular_Risk.py
│   ├── 4_Patient_Explorer.py
│   ├── 5_Clinical_Insights.py
│   ├── 6_Model_Performance.py
│   └── 7_Reports.py
├── dash_common.py                # Shared dashboard helpers (caching, stat cards, badges)
├── api/main.py                   # FastAPI service
├── tests/                        # pytest unit & integration tests
├── run_analysis.py               # Backend-only pipeline runner (saves figures + DB)
├── run_pipeline.py               # Full pipeline runner (+ optional PDF report)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── setup.py
```

---

## Quick Start

### Local development

```bash
# Create & activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the backend analysis pipeline (generates figures + SQLite DB)
python run_analysis.py

# Launch the dashboard
streamlit run app.py
```

The dashboard opens at **http://localhost:8501**.

### Run the FastAPI service

```bash
uvicorn api.main:app --reload --port 8000
```

Swagger docs at **http://localhost:8000/docs**.

### Full pipeline with PDF report

```bash
python run_pipeline.py --generate-report
```

### Docker

```bash
docker build -t bpv-dashboard .
docker run -p 8501:8501 bpv-dashboard

# or, for both dashboard + API:
docker-compose up
```

### Run tests

```bash
pytest
pytest --cov=bpv_cvd --cov-report=term-missing
```

---

## Dashboard Pages

Every page shares a sidebar **Cohort settings** panel (patient count, random seed). Changing either control regenerates the cohort and reruns clustering + prediction; all pages update immediately with the new results.

| Page | Contents |
|---|---|
| **Home** | Key statistics, cluster distribution, CV risk overview |
| **Data Explorer** | Raw data table, summary stats, interactive correlation heatmap, pairplot, distributions |
| **Clustering Analysis** | Algorithm selector, 2D/3D scatter, metrics comparison, parallel coordinates, PCA/t-SNE, dendrogram, silhouette plots |
| **Cardiovascular Risk** | Risk by cluster, ANOVA, interactive risk predictor with gauge chart, ROC/confusion matrix |
| **Patient Explorer** | Individual patient profile, BP trajectory, cluster standing, treatment recommendation |
| **Clinical Insights** | Cluster-by-cluster clinical interpretation and management recommendations |
| **Model Performance** | Full model comparison, ROC/PR curves, learning curves, cross-validation, SHAP |
| **Reports** | Custom PDF report builder + CSV/Excel/JSON data export |

---

## Clinical Interpretation Summary

| Cluster | BPV Range | CV Risk | Recommendation |
|---|---|---|---|
| **High BPV** | SBPV > 13%, DBPV > 6% | **42.9%** | Aggressive BP monitoring, frequent follow-up, consider antihypertensive adjustment and cardiac evaluation |
| **Medium BPV** | SBPV 8–13%, DBPV 4–6% | **16.7%** | Standard monitoring, lifestyle modification, quarterly review |
| **Low BPV** | SBPV < 8%, DBPV < 4% | **12.0%** | Routine care, annual review |

---

## Methodology Notes

- **Clustering**: features are standardized (`StandardScaler`) before clustering on `[SBPV, DBPV, mean SBP, mean DBP, pulse pressure]`. Ward's hierarchical clustering is used as the canonical/primary cluster assignment throughout the dashboard, reflecting the best-performing algorithm reported in the source paper. Cluster labels are ordered by mean BPV composite into `Low / Medium / High BPV` for clinical interpretability.
- **Validation metrics**: silhouette score, Davies-Bouldin index, Calinski-Harabasz score.
- **Prediction**: models are trained on a stratified 75/25 train/test split with class-balanced weighting; evaluated via accuracy, precision, recall, F1, and ROC-AUC.
- **PAM**: uses `scikit-learn-extra`'s `KMedoids` if installed, otherwise falls back to a built-in numpy PAM implementation — no hard dependency on a package that can be difficult to build on some Windows toolchains.

## Citation

If referencing the underlying methodology, cite the original paper:

> Montoya, et al. (2025). *Blood pressure variability and cardiovascular risk assessment using machine learning clustering after arteriovenous fistula creation in hemodialysis patients.*

## License

MIT — for educational and research demonstration purposes.
