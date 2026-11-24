"""Reproducible chart code; the team must write its own submission slides."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_figures(curve, assignments, frame, metrics, profiles, business, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "svg.hashsalt": "carecompass-v1", "svg.fonttype": "none"})

    def save(fig, name):
        fig.tight_layout()
        svg_path = directory / (name + ".svg")
        fig.savefig(svg_path, metadata={"Date": None})
        # Keep generated vector files readable and free of trailing whitespace.
        svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
        fig.savefig(directory / (name + ".png"), dpi=150, metadata={"Software": "CareCompass model code"})
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(curve.k, curve.train_mean_squared_distance, "o-", label="Training: mean squared distance")
    if curve.test_mean_squared_distance.notna().any():
        ax.plot(curve.k, curve.test_mean_squared_distance, "s--", label="Test: distance to trained centers")
    ax.axvline(metrics["chosen_k"], color="tab:red", linestyle=":", label=f"Selected k={metrics['chosen_k']}")
    ax.set(xlabel="Number of clusters, k", ylabel="Mean squared distance in encoded/scaled space", title="Clustering elbow: one-group baseline to candidate models", xticks=curve.k)
    ax.legend(fontsize=8)
    if metrics["test_rows"] == 0:
        fig.text(0.5, 0.015, f"{metrics['training_rows']}-person demonstration: no test curve under the assignment exception.", ha="center", fontsize=8)
        fig.subplots_adjust(bottom=0.18)
    save(fig, "01_elbow")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    values = [metrics["baseline_train_rms_distance"], metrics["model_train_rms_distance"]]
    bars = ax.bar(["One group: baseline", f"{metrics['chosen_k']} groups: v1"], values, color=["#888888", "#2878a6"])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom")
    ax.set(ylabel="RMS distance across all five fitted coordinates", title="Training spread: sqrt(SSE / training rows)", ylim=(0, max(values) * 1.18))
    save(fig, "02_baseline_vs_v1_spread")

    plotted = frame.merge(assignments[["participant_id", "segment_id", "split"]], on="participant_id", validate="one_to_one")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for p in profiles:
        group = plotted[plotted.segment_id.eq(p["segment_id"])]
        ax.scatter(group.diabetes_duration_years, group.hba1c_pct, label=f"{p['segment_id']}: {p['burden_tier']} relative burden", s=48, alpha=.85)
    ax.set(xlabel="Diabetes duration (years)", ylabel="HbA1c (%)", title="Two-dimensional view of a five-coordinate model")
    ax.legend(fontsize=8)
    fig.text(.5, .01, "The fit also uses age, systolic pressure, and the encoded recorded category.", ha="center", fontsize=8)
    save(fig, "03_profiles_2d_view")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    values = [business["baseline_monthly_navigation_minutes"], business["model_monthly_navigation_minutes"]]
    bars = ax.bar(["Generic information", "Profile-based information"], values, color=["#888888", "#2878a6"])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom")
    ax.set(ylabel="Assumed total user-navigation minutes/month", title="Hypothetical business value; not a measured benefit", ylim=(0, max(values) * 1.15))
    save(fig, "04_assumed_business_value")
