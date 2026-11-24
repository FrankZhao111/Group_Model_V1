"""Step 3: in-sample fit, optional held-out evaluation, and sensitivity."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from .model import choose_k


def evaluate(fit: dict, train: pd.DataFrame, test: pd.DataFrame | None, artifact: dict, reference: pd.DataFrame, config: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    k = artifact["selected_k"]
    model, x = fit["model"], fit["x_train"]
    baseline = fit["models"][1]
    baseline_sse, model_sse = float(baseline.inertia_), float(model.inertia_)
    mapping = {p["raw_cluster_id"]: p["segment_id"] for p in artifact["profiles"]}
    assignments = train[["participant_id"]].copy()
    assignments["split"] = "train"
    assignments["raw_cluster_id"] = model.labels_
    assignments["segment_id"] = assignments.raw_cluster_id.map(mapping)
    assignments["distance_to_center"] = np.sqrt(((x - model.cluster_centers_[model.labels_]) ** 2).sum(axis=1))
    if test is not None:
        labels = model.predict(fit["x_test"])
        extra = test[["participant_id"]].copy()
        extra["split"] = "test"
        extra["raw_cluster_id"] = labels
        extra["segment_id"] = extra.raw_cluster_id.map(mapping)
        extra["distance_to_center"] = model.transform(fit["x_test"]).min(axis=1)
        assignments = pd.concat([assignments, extra], ignore_index=True)
    # These findings are joined only AFTER fitting. They never select k or labels.
    observed = assignments.merge(reference[["participant_id", "observed_retinopathy"]], on="participant_id", validate="one_to_one")
    retinal_summary = observed.groupby(["split", "segment_id"], as_index=False).agg(
        assessed_count=("observed_retinopathy", "count"),
        observed_retinopathy_count=("observed_retinopathy", lambda v: int(v.eq(1).sum())),
    )
    retinal_summary["observed_retinopathy_pct"] = np.where(retinal_summary.assessed_count.gt(0), 100 * retinal_summary.observed_retinopathy_count / retinal_summary.assessed_count, np.nan)

    no_category = KMeans(n_clusters=k, n_init=config["n_init"], random_state=config["random_state"]).fit(x[:, :4])
    repeat_ari = []
    for seed in [0, 1, 2, 3, 4]:
        other = KMeans(n_clusters=k, n_init=config["n_init"], random_state=seed).fit(x)
        repeat_ari.append(float(adjusted_rand_score(model.labels_, other.labels_)))
    test_choice = None
    test_baseline_sse = test_model_sse = None
    if test is not None:
        alternative = fit["curve"][["k", "test_sse"]].rename(columns={"test_sse": "train_sse"})
        test_choice = choose_k(alternative, config["elbow_near_tie_tolerance"])["selected_k"]
        test_baseline_sse = float(np.square(baseline.transform(fit["x_test"]).min(axis=1)).sum())
        test_model_sse = float(np.square(model.transform(fit["x_test"]).min(axis=1)).sum())
    metrics = {
        "observation_count": len(train) + (len(test) if test is not None else 0),
        "training_rows": len(train), "test_rows": len(test) if test is not None else 0,
        "split_rule": "Under 30: use all rows and skip the test split" if test is None else "Random 70/30 split; all preprocessing and centers fitted on training rows only",
        "test_evaluation_status": "Not available under the assignment's under-30 exception" if test is None else "Computed from training-fitted preprocessing and centers",
        "chosen_k": k, "elbow_selection": fit["choice"],
        "baseline_train_sse": baseline_sse, "model_train_sse": model_sse,
        "training_sse_reduction_pct": 100 * (1 - model_sse / baseline_sse) if baseline_sse else 0.0,
        "baseline_train_rms_distance": float(np.sqrt(baseline_sse / len(train))),
        "model_train_rms_distance": float(np.sqrt(model_sse / len(train))),
        "model_train_silhouette": fit["curve"].loc[fit["curve"].k.eq(k), "train_silhouette"].iloc[0] if k > 1 else None,
        "baseline_test_sse": test_baseline_sse, "model_test_sse": test_model_sse,
        "test_elbow_k": test_choice, "train_test_elbow_agreement": test_choice == k if test is not None else None,
        "smallest_training_cluster": min(p["count"] for p in artifact["profiles"]),
        "recorded_category_sensitivity_ari": float(adjusted_rand_score(model.labels_, no_category.labels_)),
        "random_seed_sensitivity": {"seeds": [0, 1, 2, 3, 4], "adjusted_rand_indices": repeat_ari, "minimum_ari": min(repeat_ari)},
        "metric_meaning": "Distances are in the five-coordinate encoded/scaled space. SSE reduction is not medical accuracy, risk reduction, quality-of-life improvement, or evidence of held-out performance.",
        "retinal_summary_meaning": "Post-fit, unweighted descriptive counts of existing findings. These are not future-risk estimates or a held-out validation set.",
    }
    return metrics, assignments, retinal_summary
