"""Step 4: a portable new-user decision function using the fitted centers."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import NUMERIC, validate_inputs


def load_artifact(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def encode_from_artifact(frame: pd.DataFrame, artifact: dict) -> np.ndarray:
    p = artifact["preprocessing"]
    numeric = (frame[NUMERIC].to_numpy(dtype=float) - np.array(p["mean"])) / np.array(p["scale"])
    category = frame.sex_recorded.map(p["sex_encoding"]).to_numpy(dtype=float) * p["sex_indicator_weight"]
    return np.column_stack([numeric, category])


def make_decision(user_input: dict, artifact: dict) -> dict:
    """Take one realistic intake record; return a group and information choice.

    No retinal examination label, study ID, or other user's answers are inputs.
    Missing/unsupported inputs raise a clear error instead of receiving a group.
    """
    frame = validate_inputs(pd.DataFrame([user_input]))
    encoded = encode_from_artifact(frame, artifact)
    centers = np.array(artifact["centers"], dtype=float)
    squared_distances = ((centers - encoded[0]) ** 2).sum(axis=1)
    raw = int(np.argmin(squared_distances))
    profile = next(p for p in artifact["profiles"] if p["raw_cluster_id"] == raw)
    outside = []
    for column, bounds in artifact["training_ranges"].items():
        value = float(frame.iloc[0][column])
        if value < bounds["min"] or value > bounds["max"]:
            outside.append(column)
    return {
        "input": {c: float(frame.iloc[0][c]) for c in NUMERIC} | {"sex_recorded": str(frame.iloc[0].sex_recorded)},
        "segment_id": profile["segment_id"], "raw_cluster_id": raw,
        "relative_burden_tier": profile["burden_tier"], "profile_label": profile["profile_label"],
        "decision": profile["information_package"],
        "baseline_decision": "Standard eye-health information",
        "distance_to_center": float(np.sqrt(squared_distances[raw])),
        "nearest_group_mean_values": {c: profile[c + "_mean"] for c in NUMERIC},
        "outside_observed_training_ranges": outside,
        "interpretation": artifact["risk_interpretation"],
    }
