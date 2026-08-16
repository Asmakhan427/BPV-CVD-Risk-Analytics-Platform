"""
BPV-CVD Risk Analytics Platform
================================

Machine learning analytics package supporting the study "Blood pressure
variability and cardiovascular risk assessment using machine learning
clustering after arteriovenous fistula creation in hemodialysis patients"
(Montoya et al., 2025).

Modules
-------
data_generator   : synthetic cohort generation matching published statistics
preprocessing    : scaling, imputation, feature engineering, splitting
clustering       : K-Means, PAM (K-Medoids), Ward, and EM (GMM) clustering
analysis         : ANOVA, cardiovascular risk tables, bootstrap CIs
prediction       : Random Forest / XGBoost / Logistic Regression CV-risk models
visualization    : publication-quality static figures (matplotlib/seaborn)
dashboard_data   : cached data-shaping helpers consumed by the Streamlit app
database         : SQLite persistence layer
report           : PDF report generation
palette          : shared colorblind-safe color tokens
"""

__version__ = "1.0.0"
__author__ = "BPV-CVD Analytics Team"
