"""Run the entire assignment workflow: python run_model.py."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

import numpy as np
import pandas as pd
import sklearn

from carecompass.business import quantify_value
from carecompass.data import load_data, split_data, check_reference
from carecompass.decision import make_decision
from carecompass.evaluation import evaluate
from carecompass.model import fit_candidates, build_artifact
from carecompass.plots import save_figures
from carecompass.reporting import write_results

ROOT = Path(__file__).resolve().parent


def write_json(path: Path, value: dict):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(config_path: Path = ROOT / "config.json", charts: bool = True) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / config["output_directory"]
    output.mkdir(parents=True, exist_ok=True)

    print("[1/6] Data and assignment split rule")
    data = load_data(ROOT / config["data_file"])
    reference = check_reference(data, ROOT / config["reference_file"])
    train, test = split_data(data, config["random_state"])
    print(f"Rows: {len(data)}; training: {len(train)}; test: {len(test) if test is not None else 0}; missing core values: 0; imputed: 0")
    if test is None:
        print("Under 30 observations: all rows train; held-out evaluation and a test elbow are unavailable.")
    split = data[["participant_id"]].copy()
    split["split"] = np.where(split.participant_id.isin(train.participant_id), "train", "test")
    split.to_csv(output / "data_split.csv", index=False)

    print("\n[2/6] Encode, scale, and fit the one-group baseline and candidate models")
    fit = fit_candidates(train, test, config)
    artifact, profiles = build_artifact(fit, train, config)
    print(fit["curve"][["k", "train_sse", "train_rms_distance", "elbow_gap"]].to_string(index=False))
    print(f"Training elbow rule selects k={artifact['selected_k']}; near candidates: {fit['choice']['near_tie_candidates']}")

    print("\n[3/6] Evaluate fit, spread, and sensitivity")
    metrics, assignments, retinal_summary = evaluate(fit, train, test, artifact, reference, config)
    fit["curve"].to_csv(output / "elbow.csv", index=False)
    profiles.to_csv(output / "cluster_profiles.csv", index=False)
    assignments.to_csv(output / "assignments.csv", index=False)
    retinal_summary.to_csv(output / "retinal_findings_descriptive_only.csv", index=False)
    transformed = pd.DataFrame(fit["x_train"], columns=artifact["preprocessing"]["coordinate_order"])
    transformed.insert(0, "participant_id", train.participant_id.to_numpy())
    transformed.to_csv(output / "transformed_training_inputs.csv", index=False)
    print(f"Training SSE: {metrics['baseline_train_sse']:.4f} baseline -> {metrics['model_train_sse']:.4f} v1 ({metrics['training_sse_reduction_pct']:.2f}% lower)")
    print("This reduction is a training-distance statistic, not medical accuracy or risk reduction.")
    print(f"Grouping agreement without recorded sex/gender (ARI): {metrics['recorded_category_sensitivity_ari']:.3f}")
    write_json(output / "model.json", artifact)
    write_json(output / "metrics.json", metrics)

    print("\n[4/6] Make a decision for a realistic, separate example input")
    example = json.loads((ROOT / config["example_file"]).read_text(encoding="utf-8"))
    decision = make_decision(example, artifact)
    write_json(output / "example_decision.json", decision)
    print(f"Example -> {decision['segment_id']}: {decision['profile_label']}; {decision['decision']}")

    print("\n[5/6] Quantify baseline versus v1 in business units")
    business = quantify_value(artifact, config["business_assumptions"])
    write_json(output / "business_value.json", business)
    print(f"ASSUMED navigation minutes/month: {business['baseline_monthly_navigation_minutes']:.2f} baseline -> {business['model_monthly_navigation_minutes']:.2f} v1")
    print(f"ASSUMED time saved: {business['assumed_monthly_navigation_minutes_saved']:.2f} minutes per {config['business_assumptions']['monthly_users']} monthly users.")

    print("\n[6/6] Write reproducible outputs and the team presentation checklist")
    if charts:
        save_figures(fit["curve"], assignments, data, metrics, artifact["profiles"], business, output / "figures")
    write_results(output, metrics, profiles, decision, business)
    write_json(output / "run_manifest.json", {"python_workflow": "run_model.py", "data_sha256": hashlib.sha256((ROOT / config["data_file"]).read_bytes()).hexdigest(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "config": config, "charts_generated": charts})
    print(f"Outputs written to {output}; the team must write its own submission slides.")
    return {"artifact": artifact, "metrics": metrics, "decision": decision, "business": business}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--no-charts", action="store_true", help="Run numerical evaluation without generating figures")
    args = parser.parse_args()
    run(args.config, charts=not args.no_charts)
