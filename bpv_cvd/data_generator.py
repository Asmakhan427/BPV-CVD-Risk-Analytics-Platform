"""
Synthetic cohort generator.

Reproduces a patient-level dataset with the same sample size, cluster
structure, and cardiovascular (CV) event rates reported by Montoya et al.
(2025) for blood-pressure-variability (BPV) clustering after arteriovenous
fistula (AVF) creation in hemodialysis patients:

    * n = 83 patients
    * 3 latent BPV clusters (Low / Medium / High) identified by Ward's
      hierarchical clustering
    * CV risk by cluster: High BPV 42.9%, Medium BPV 16.7%, Low BPV 12.0%

The data is entirely synthetic (no real patient data is used or required)
but is calibrated so that summary statistics, cluster separability, and
downstream model behavior resemble the published cohort closely enough to
demonstrate the full analytical pipeline end to end.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Published/target cluster sizes and CV event rates for n=83
CLUSTER_SPEC = {
    "High BPV": {"n": 21, "cv_risk_rate": 0.429},
    "Medium BPV": {"n": 24, "cv_risk_rate": 0.167},
    "Low BPV": {"n": 38, "cv_risk_rate": 0.120},
}

AVF_LOCATIONS = ["Radiocephalic", "Brachiocephalic", "Brachiobasilic"]


@dataclass
class ClusterProfile:
    """Generative parameters for one latent BPV cluster."""

    name: str
    sbp_mean_range: tuple
    sbpv_range: tuple  # systolic BPV, coefficient of variation (%)
    dbp_mean_range: tuple
    dbpv_range: tuple  # diastolic BPV, coefficient of variation (%)
    age_mean: float
    diabetes_p: float
    hypertension_p: float
    cad_p: float
    stroke_p: float


PROFILES = {
    "High BPV": ClusterProfile(
        name="High BPV",
        sbp_mean_range=(135, 165),
        sbpv_range=(13.0, 19.0),
        dbp_mean_range=(75, 95),
        dbpv_range=(6.0, 10.5),
        age_mean=66.0,
        diabetes_p=0.55,
        hypertension_p=0.90,
        cad_p=0.38,
        stroke_p=0.19,
    ),
    "Medium BPV": ClusterProfile(
        name="Medium BPV",
        sbp_mean_range=(128, 150),
        sbpv_range=(8.0, 13.0),
        dbp_mean_range=(70, 88),
        dbpv_range=(4.0, 6.0),
        age_mean=59.0,
        diabetes_p=0.38,
        hypertension_p=0.75,
        cad_p=0.21,
        stroke_p=0.08,
    ),
    "Low BPV": ClusterProfile(
        name="Low BPV",
        sbp_mean_range=(110, 138),
        sbpv_range=(3.5, 8.0),
        dbp_mean_range=(65, 82),
        dbpv_range=(2.0, 4.0),
        age_mean=52.0,
        diabetes_p=0.24,
        hypertension_p=0.58,
        cad_p=0.08,
        stroke_p=0.03,
    ),
}


def _generate_cluster_block(profile: ClusterProfile, n: int, cv_rate: float, rng: np.random.Generator, id_offset: int) -> pd.DataFrame:
    """Generate n synthetic patients belonging to one latent BPV cluster."""
    sbp_mean = rng.uniform(*profile.sbp_mean_range, n)
    sbpv = np.clip(rng.normal(np.mean(profile.sbpv_range), (profile.sbpv_range[1] - profile.sbpv_range[0]) / 4, n),
                    profile.sbpv_range[0] * 0.85, profile.sbpv_range[1] * 1.1)
    sbp_sd = sbpv / 100.0 * sbp_mean

    dbp_mean = rng.uniform(*profile.dbp_mean_range, n)
    dbpv = np.clip(rng.normal(np.mean(profile.dbpv_range), (profile.dbpv_range[1] - profile.dbpv_range[0]) / 4, n),
                    profile.dbpv_range[0] * 0.85, profile.dbpv_range[1] * 1.1)
    dbp_sd = dbpv / 100.0 * dbp_mean

    age = np.clip(rng.normal(profile.age_mean, 11.0, n), 21, 90)
    sex = rng.choice(["Male", "Female"], n, p=[0.58, 0.42])
    bmi = np.clip(rng.normal(27.5, 4.8, n), 16, 48)
    dialysis_vintage = np.clip(rng.gamma(2.2, 14.0, n), 1, 180)

    diabetes = rng.binomial(1, profile.diabetes_p, n)
    hypertension = rng.binomial(1, profile.hypertension_p, n)
    cad = rng.binomial(1, profile.cad_p, n)
    stroke = rng.binomial(1, profile.stroke_p, n)

    avf_location = rng.choice(AVF_LOCATIONS, n, p=[0.5, 0.33, 0.17])

    albumin = np.clip(rng.normal(3.9 - 0.15 * (sbpv > 12), 0.35, n), 2.4, 4.8)
    hemoglobin = np.clip(rng.normal(11.2, 1.1, n), 7.5, 15.5)
    crp = np.clip(rng.lognormal(mean=np.log(4.0 + 3.0 * (sbpv / 10)), sigma=0.6, size=n), 0.2, 60)
    ultrafiltration_rate = np.clip(rng.normal(9.5, 2.5, n), 3, 18)

    pulse_pressure = sbp_mean - dbp_mean

    n_events = int(round(cv_rate * n))
    n_events = min(max(n_events, 0), n)
    cv_event = np.zeros(n, dtype=int)
    if n_events > 0:
        # Weighted (not deterministic top-K) sampling: higher SBPV/DBPV/age raise
        # the *probability* of an event without guaranteeing it, so within-cluster
        # risk is realistically noisy rather than perfectly separable by rank.
        score = sbpv + 0.5 * dbpv + 0.3 * age
        order = np.argsort(score)
        ranks = np.empty(n)
        ranks[order] = np.arange(n)
        weights = np.exp(2.0 * ranks / max(n - 1, 1))
        weights = weights / weights.sum()
        event_idx = rng.choice(n, size=n_events, replace=False, p=weights)
        cv_event[event_idx] = 1

    patient_id = [f"PT-{i:03d}" for i in range(id_offset, id_offset + n)]

    df = pd.DataFrame({
        "patient_id": patient_id,
        "age": np.round(age, 1),
        "sex": sex,
        "bmi": np.round(bmi, 1),
        "dialysis_vintage_months": np.round(dialysis_vintage, 1),
        "diabetes": diabetes,
        "hypertension": hypertension,
        "coronary_artery_disease": cad,
        "prior_stroke": stroke,
        "avf_location": avf_location,
        "albumin": np.round(albumin, 2),
        "hemoglobin": np.round(hemoglobin, 2),
        "crp": np.round(crp, 2),
        "ultrafiltration_rate": np.round(ultrafiltration_rate, 2),
        "sbp_mean": np.round(sbp_mean, 1),
        "sbp_sd": np.round(sbp_sd, 2),
        "sbpv": np.round(sbpv, 2),
        "dbp_mean": np.round(dbp_mean, 1),
        "dbp_sd": np.round(dbp_sd, 2),
        "dbpv": np.round(dbpv, 2),
        "pulse_pressure": np.round(pulse_pressure, 1),
        "true_cluster": profile.name,
        "cv_risk_event": cv_event,
    })
    return df


def generate_patient_data(n: int = 83, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic hemodialysis + AVF cohort matching the cluster
    sizes and cardiovascular event rates reported in Montoya et al. (2025).

    Parameters
    ----------
    n : int
        Total number of patients. Defaults to the published cohort size (83).
        Cluster sizes scale proportionally to CLUSTER_SPEC when n != 83.
    seed : int
        Random seed for full reproducibility.

    Returns
    -------
    pd.DataFrame
        One row per patient with demographic, clinical, BPV, and outcome
        columns. `true_cluster` is the generative ground-truth label kept
        for validation purposes only -- downstream clustering code must
        treat the data as unlabeled.
    """
    rng = np.random.default_rng(seed)

    if n == 83:
        sizes = {k: v["n"] for k, v in CLUSTER_SPEC.items()}
    else:
        total = sum(v["n"] for v in CLUSTER_SPEC.values())
        sizes = {k: max(1, round(v["n"] / total * n)) for k, v in CLUSTER_SPEC.items()}
        # adjust rounding drift onto the largest cluster
        drift = n - sum(sizes.values())
        sizes["Low BPV"] += drift

    blocks = []
    offset = 1
    for cname, spec in CLUSTER_SPEC.items():
        block = _generate_cluster_block(
            PROFILES[cname], sizes[cname], spec["cv_risk_rate"], rng, offset
        )
        offset += sizes[cname]
        blocks.append(block)

    df = pd.concat(blocks, ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle rows
    df["patient_id"] = [f"PT-{i:03d}" for i in range(1, len(df) + 1)]

    logger.info("Generated synthetic cohort: n=%d, clusters=%s", len(df), sizes)
    return df


def validate_distribution(df: pd.DataFrame, tolerance: float = 0.15) -> dict:
    """
    Validate that the generated cohort's summary statistics are consistent
    with the published cluster CV-risk rates.

    Returns a dict report with per-cluster observed rates, target rates, and
    a boolean `passed` flag (all clusters within `tolerance` absolute rate).
    """
    report = {"n_total": len(df), "clusters": {}, "passed": True}
    for cname, spec in CLUSTER_SPEC.items():
        sub = df[df["true_cluster"] == cname]
        if len(sub) == 0:
            report["clusters"][cname] = {"n": 0, "observed_rate": None, "target_rate": spec["cv_risk_rate"], "ok": False}
            report["passed"] = False
            continue
        observed = sub["cv_risk_event"].mean()
        ok = bool(abs(observed - spec["cv_risk_rate"]) <= tolerance)
        report["clusters"][cname] = {
            "n": len(sub),
            "observed_rate": round(float(observed), 3),
            "target_rate": spec["cv_risk_rate"],
            "ok": ok,
        }
        report["passed"] = bool(report["passed"] and ok)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = generate_patient_data()
    print(data.head())
    print(data.describe(include="all").T)
    print(validate_distribution(data))
