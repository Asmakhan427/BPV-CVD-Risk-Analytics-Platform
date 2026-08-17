# BPV-CVD Risk Analytics Platform

**Machine learning platform for blood pressure variability (BPV) clustering and cardiovascular risk prediction in hemodialysis patients**, reproducing the analytical approach of Montoya et al. (2025) on a synthetic cohort.

[![Python](https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/container-Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/tests-48%20passing-brightgreen)](tests/)
[![License](https://img.shields.io/badge/license-MIT-green)](#license)


---

## Output

Overview 
<img width="1919" height="865" alt="image" src="https://github.com/user-attachments/assets/834f141a-df61-4c9a-bf73-32ce03c8b8fe" />
Clustering
 <img width="1915" height="813" alt="image" src="https://github.com/user-attachments/assets/5cfb92e2-3860-47c8-8b2f-46e22a7acaa7" />


Cardiovascular Risk  
 <img width="1919" height="791" alt="image" src="https://github.com/user-attachments/assets/73570dbc-0546-4825-b120-b76fde0238d8" />
Patient Explorer
 <img width="1919" height="825" alt="image" src="https://github.com/user-attachments/assets/eeba8506-6552-4947-8dc9-a3c996fe0e66" />
 

---

## Features

- **Dynamic dashboard** — sidebar controls (patient count, random seed) regenerate the cohort and rerun the full pipeline live
- **4 clustering algorithms** — K-Means, PAM, Ward's hierarchical clustering, EM (Gaussian Mixture) via a shared `fit / predict / evaluate` interface
- **3 predictive models** — Random Forest, XGBoost, Logistic Regression for cardiovascular risk classification
- **8-page dashboard** — Home, Data Explorer, Clustering Analysis, Cardiovascular Risk, Patient Explorer, Clinical Insights, Model Performance, Reports
- **30+ visualizations** — histograms, box/violin plots, correlation heatmaps, 2D/3D scatter, dendrogram, silhouette plots, PCA/t-SNE, ROC/PR curves, SHAP, radar and gauge charts
- **FastAPI service** with auto-generated Swagger docs
- **SQLite persistence**, **PDF report generator**, **Docker** packaging, **pytest** suite

## Quick Start

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux

pip install -r requirements.txt
streamlit run app.py           # dashboard at http://localhost:8501
```

Other entry points:

```bash
python run_analysis.py                       # backend-only: figures + SQLite DB
python run_pipeline.py --generate-report      # full pipeline + PDF report
uvicorn api.main:app --reload --port 8000     # API, docs at /docs
pytest                                        # test suite
docker-compose up                             # dashboard + API in containers
```

## Project Structure

<details>
<summary>Expand</summary>

```
bpv_cvd/               Core analytics package (data generation, preprocessing,
                        clustering, statistics, prediction, visualization,
                        SQLite persistence, PDF reporting)
app.py                 Streamlit entry point
pages/                 Streamlit multipage dashboard (7 pages)
dash_common.py         Shared dashboard helpers (cohort controls, caching, UI)
api/main.py            FastAPI service
tests/                 pytest unit & integration tests
run_analysis.py         Backend-only pipeline runner
run_pipeline.py         Full pipeline runner (+ optional PDF report)
Dockerfile / docker-compose.yml
```

</details>

## Clinical Interpretation Summary

| Cluster | BPV Range | CV Risk | Recommendation |
|---|---|---|---|
| **High BPV** | SBPV > 13%, DBPV > 6% | **42.9%** | Aggressive BP monitoring, frequent follow-up, antihypertensive adjustment, cardiac evaluation |
| **Medium BPV** | SBPV 8–13%, DBPV 4–6% | **16.7%** | Standard monitoring, lifestyle modification, quarterly review |
| **Low BPV** | SBPV < 8%, DBPV < 4% | **12.0%** | Routine care, annual review |

## Methodology Notes

- Clustering runs on standardized `[SBPV, DBPV, mean SBP, mean DBP, pulse pressure]`; Ward's hierarchical clustering is the canonical assignment (best-performing algorithm in the source paper), labeled `Low / Medium / High BPV` by mean BPV composite.
- Validation metrics: silhouette score, Davies-Bouldin index, Calinski-Harabasz score.
- Prediction models trained on a stratified 75/25 split, evaluated via accuracy, precision, recall, F1, and ROC-AUC.

## License

MIT — for educational and research demonstration purposes.
