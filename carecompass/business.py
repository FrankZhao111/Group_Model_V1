"""Step 5: explicitly hypothetical navigation-time value, not clinical benefit."""


def quantify_value(artifact: dict, assumptions: dict) -> dict:
    monthly = int(assumptions["monthly_users"])
    generic = float(assumptions["generic_information_navigation_minutes"])
    enhanced = float(assumptions["enhanced_information_navigation_minutes"])
    if monthly < 0 or min(generic, enhanced) < 0:
        raise ValueError("Business assumptions must use nonnegative users and minutes.")
    if not assumptions["future_group_mix_matches_training_sample"]:
        raise ValueError("Supply a separate forecast group mix before disabling the sample-mix assumption.")
    selected_count = sum(p["count"] for p in artifact["profiles"] if p["information_package"].startswith("Enhanced"))
    share = selected_count / artifact["training_rows"]
    expected_selected = monthly * share
    baseline_minutes = monthly * generic
    model_minutes = (monthly - expected_selected) * generic + expected_selected * enhanced
    return {
        "status": "Hypothetical scenario, not measured value or a clinical outcome",
        "assumptions": assumptions,
        "enhanced_information_training_users": selected_count,
        "training_users": artifact["training_rows"], "enhanced_information_share": share,
        "expected_enhanced_information_users_per_month": expected_selected,
        "baseline_monthly_navigation_minutes": baseline_minutes,
        "model_monthly_navigation_minutes": model_minutes,
        "assumed_monthly_navigation_minutes_saved": baseline_minutes - model_minutes,
        "assumed_minutes_saved_per_enhanced_user": generic - enhanced,
        "calculation": "Monthly users × training share assigned enhanced information × (generic minutes − enhanced minutes)",
        "not_measured": ["Actual time savings", "Treatment effects", "Quality-of-life improvement", "Avoided eye disease"],
    }
