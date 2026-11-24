"""Step 2: explicit category encoding, training-only scaling, and K-means."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .data import NUMERIC, SEX_ENCODING


@dataclass
class FeatureEncoder:
    scaler: StandardScaler
    sex_weight: float = 1.0

    @classmethod
    def fit(cls, training: pd.DataFrame, sex_weight: float = 1.0):
        return cls(StandardScaler().fit(training[NUMERIC]), sex_weight)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        continuous = self.scaler.transform(frame[NUMERIC])
        binary = frame.sex_recorded.map(SEX_ENCODING).to_numpy(dtype=float) * self.sex_weight
        return np.column_stack([continuous, binary])

    def state(self) -> dict:
        return {
            "numeric_features": NUMERIC,
            "mean": self.scaler.mean_.tolist(),
            "scale": self.scaler.scale_.tolist(),
            "sex_encoding": SEX_ENCODING,
            "sex_indicator_weight": self.sex_weight,
            "coordinate_order": NUMERIC + ["sex_recorded_male"],
            "note": "The four continuous fields are standardized with training statistics; the weighted 0/1 category indicator is not standardized.",
        }


def elbow_scores(curve: pd.DataFrame) -> np.ndarray:
    """Gap below the endpoint chord after normalizing k and SSE to [0,1]."""
    inertia = curve.train_sse.to_numpy(dtype=float)
    if len(curve) < 3 or np.isclose(inertia[0], inertia[-1]):
        return np.zeros(len(curve))
    x = (curve.k.to_numpy() - curve.k.min()) / (curve.k.max() - curve.k.min())
    y = (inertia - inertia[-1]) / (inertia[0] - inertia[-1])
    return (1.0 - x) - y


def choose_k(curve: pd.DataFrame, tolerance: float = 0.02) -> dict:
    scores = elbow_scores(curve)
    winner = int(np.argmax(scores))
    if scores[winner] <= 1e-9:
        winner = 0
    near = curve.loc[(scores.max() - scores <= tolerance) & (scores > 0), "k"].astype(int).tolist()
    return {
        "selected_k": int(curve.iloc[winner].k),
        "method": "Largest gap below the endpoint chord on normalized training SSE; inspect the elbow visually.",
        "near_tie_tolerance": tolerance,
        "near_tie_candidates": near,
        "warning": "Several elbow candidates are close; k is a descriptive design choice, not a clinically validated number of risk levels." if len(near) > 1 else None,
    }


def fit_candidates(train: pd.DataFrame, test: pd.DataFrame | None, config: dict) -> dict:
    encoder = FeatureEncoder.fit(train, config["sex_indicator_weight"])
    x = encoder.transform(train)
    xt = encoder.transform(test) if test is not None else None
    models, rows = {}, []
    for k in range(1, min(config["maximum_k"], len(train) - 1) + 1):
        model = KMeans(n_clusters=k, random_state=config["random_state"], n_init=config["n_init"]).fit(x)
        models[k] = model
        rows.append({
            "k": k, "train_sse": float(model.inertia_),
            "train_mean_squared_distance": float(model.inertia_ / len(train)),
            "train_rms_distance": float(np.sqrt(model.inertia_ / len(train))),
            "train_silhouette": float(silhouette_score(x, model.labels_)) if k > 1 and len(np.unique(model.labels_)) > 1 else None,
            "test_sse": float(np.square(model.transform(xt).min(axis=1)).sum()) if xt is not None else None,
            "test_mean_squared_distance": float(np.square(model.transform(xt).min(axis=1)).mean()) if xt is not None else None,
        })
    curve = pd.DataFrame(rows)
    curve["elbow_gap"] = elbow_scores(curve)
    choice = choose_k(curve, config["elbow_near_tie_tolerance"])
    chosen = models[choice["selected_k"]]
    return {"encoder": encoder, "x_train": x, "x_test": xt, "models": models, "model": chosen, "curve": curve, "choice": choice}


def build_artifact(fit: dict, train: pd.DataFrame, config: dict) -> tuple[dict, pd.DataFrame]:
    model, x = fit["model"], fit["x_train"]
    # Ordering is an explicit descriptive heuristic, not learned medical risk:
    # average the group's standardized duration, HbA1c, and systolic-pressure means.
    burden_columns = [NUMERIC.index(c) for c in ["diabetes_duration_years", "hba1c_pct", "systolic_bp_mmhg"]]
    burden = model.cluster_centers_[:, burden_columns].mean(axis=1)
    order = np.argsort(burden, kind="stable")
    k = model.n_clusters
    tiers = ["Lower", "Middle", "Higher"] if k == 3 else [f"Rank {i + 1} of {k}" for i in range(k)]
    profiles = []
    for rank, raw_label in enumerate(order):
        group = train.iloc[np.flatnonzero(model.labels_ == raw_label)]
        distances2 = ((x[model.labels_ == raw_label] - model.cluster_centers_[raw_label]) ** 2).sum(axis=1)
        record = {
            "segment_id": f"G{rank + 1}", "raw_cluster_id": int(raw_label), "count": int(len(group)),
            "burden_tier": tiers[rank], "profile_label": f"{tiers[rank]} relative risk-factor burden",
            "descriptive_burden_index": float(burden[raw_label]),
            "information_package": "Enhanced eye-health information" if rank == k - 1 and k > 1 else "Standard eye-health information",
            "male_share_pct": float(100 * group.sex_recorded.eq("Male").mean()),
            "within_group_rms_distance": float(np.sqrt(distances2.mean())),
        }
        for c in NUMERIC:
            record[c + "_mean"] = float(group[c].mean())
            record[c + "_sd"] = float(group[c].std(ddof=0))
        profiles.append(record)
    artifact = {
        "schema_version": 1, "model_family": "KMeans", "selected_k": k,
        "training_rows": len(train), "preprocessing": fit["encoder"].state(),
        "centers": model.cluster_centers_.tolist(), "profiles": profiles,
        "training_ranges": {c: {"min": float(train[c].min()), "max": float(train[c].max())} for c in NUMERIC},
        "k_selection": fit["choice"],
        "risk_interpretation": "Exploratory relative risk-factor burden, not a diagnosis, a future disease probability, a measured quality-of-life score, or a decision to withhold eye care.",
        "burden_ordering": "Equal-weight average of the standardized cluster means for diabetes duration, HbA1c, and systolic BP; age and recorded sex/gender are not assigned ordinal medical-risk weights.",
        "random_state": config["random_state"], "n_init": config["n_init"],
    }
    return artifact, pd.DataFrame(profiles)
