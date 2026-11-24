"""Step 1: validate observations and apply the assignment's split rule."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

NUMERIC = ["age_years", "diabetes_duration_years", "hba1c_pct", "systolic_bp_mmhg"]
CORE = ["age_years", "sex_recorded", "diabetes_duration_years", "hba1c_pct", "systolic_bp_mmhg"]
SEX_ENCODING = {"Female": 0.0, "Male": 1.0}


def validate_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """Reject unknown or incompatible values; never silently mean-fill a user."""
    missing = [c for c in CORE if c not in frame]
    if missing:
        raise ValueError(f"Missing required input fields: {missing}")
    clean = frame.copy()
    for column in NUMERIC:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    if clean[NUMERIC].isna().any().any() or not np.isfinite(clean[NUMERIC]).all().all():
        raise ValueError("The four numerical inputs must be known, finite numbers; consult the source records instead of entering zero for unknown.")
    clean["sex_recorded"] = clean.sex_recorded.astype("string").str.strip().str.title()
    if not clean.sex_recorded.isin(SEX_ENCODING).all():
        raise ValueError("This source supports only its recorded Female/Male categories; an unknown or unsupported category cannot receive this model's profile.")
    if not clean.age_years.between(40, 79).all():
        raise ValueError("This demonstration covers ages 40-79 only.")
    if not (clean.diabetes_duration_years.ge(0) & clean.diabetes_duration_years.le(clean.age_years)).all():
        raise ValueError("Diabetes duration must be nonnegative and no greater than age.")
    if not (clean.hba1c_pct.gt(0) & clean.hba1c_pct.le(100)).all():
        raise ValueError("HbA1c must be supplied as a positive percentage, not a glucose concentration.")
    if not (clean.systolic_bp_mmhg.gt(0) & clean.systolic_bp_mmhg.le(300)).all():
        raise ValueError("Supply systolic pressure in mmHg, within the source instrument's positive-to-300 range.")
    return clean


def load_data(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "participant_id" not in frame or frame.participant_id.isna().any() or not frame.participant_id.is_unique:
        raise ValueError("Each observation must have one distinct, nonmissing participant ID.")
    if len(frame) < 3:
        raise ValueError("At least three observations are required for this demonstration.")
    return validate_inputs(frame)


def split_data(frame: pd.DataFrame, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Under 30 means no test set; exactly 30 uses the 70/30 split."""
    if len(frame) < 30:
        return frame.copy(), None
    train, test = train_test_split(frame, test_size=0.30, random_state=random_state, shuffle=True)
    return train.copy(), test.copy()


def check_reference(frame: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    """Keep examination findings separate and verify the same people/measurements."""
    reference = pd.read_csv(path)
    if not reference.participant_id.is_unique:
        raise ValueError("Duplicate IDs in the source reference.")
    joined = frame.merge(reference, on="participant_id", suffixes=("_input", "_reference"), validate="one_to_one")
    if len(joined) != len(frame):
        raise ValueError("A participant lacks source provenance.")
    for column in NUMERIC:
        if not np.allclose(joined[column + "_input"], joined[column + "_reference"], rtol=0, atol=1e-8):
            raise ValueError(f"Source mismatch in {column}")
    if not joined.sex_recorded_input.eq(joined.sex_recorded_reference).all():
        raise ValueError("Recorded category does not match source provenance.")
    return reference
