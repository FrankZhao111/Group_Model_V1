"""Optional source reproduction: python prepare_data.py.

The model runs offline from the committed CSV. This script downloads or reuses
the seven official NHANES source files to reproduce the selected 29 people.
"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data/raw"
REFERENCE = ROOT / "data/reference"
FIELDS = ["age_years", "sex_recorded", "diabetes_duration_years", "hba1c_pct", "systolic_bp_mmhg", "diastolic_bp_mmhg", "bmi_kg_m2", "smoking_status", "insulin_use", "diabetes_pill_use"]
COLUMNS = {
    "DEMO_E": ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDEXPRG", "WTMEC2YR", "SDMVPSU", "SDMVSTRA"],
    "DIQ_E": ["SEQN", "DIQ010", "DID040", "DIQ050", "DID070"],
    "GHB_E": ["SEQN", "LBXGH"],
    "BPX_E": ["SEQN", "BPXSY1", "BPXSY2", "BPXSY3", "BPXDI1", "BPXDI2", "BPXDI3"],
    "OPXRET_E": ["SEQN", "OPDUARMA", "OPDURL4"],
    "BMX_E": ["SEQN", "BMXBMI"],
    "SMQ_E": ["SEQN", "SMQ020", "SMQ040"],
}


def source_frames(raw_directory: Path):
    raw_directory.mkdir(parents=True, exist_ok=True)
    pinned = json.loads((REFERENCE / "expanded_source_manifest.json").read_text())
    expected = {r["file"]: r["sha256"] for r in pinned}
    frames = {}
    for name, columns in COLUMNS.items():
        path = raw_directory / f"{name}.xpt"
        if not path.exists():
            url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/{name}.xpt"
            with urlopen(url, timeout=60) as response:
                payload = response.read()
            if not payload.startswith(b"HEADER RECORD"):
                raise ValueError(f"{name}: response is not a SAS transport file")
            path.write_bytes(payload)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected[path.name]:
            raise ValueError(f"Source version differs from the recorded checksum: {path.name}. Review the changed source before changing the manifest.")
        frame = pd.read_sas(path, format="xport")[columns]
        # Some numeric zeros decode as ~5.4e-79. Normalize only negligible values.
        frame = frame.mask(frame.abs().lt(1e-70), 0)
        if not frame.SEQN.is_unique:
            raise ValueError(f"Repeated source identifiers in {name}")
        frames[name] = frame
    return frames


def reproduce(raw_directory: Path = RAW, verify_only: bool = False):
    frames = source_frames(raw_directory)
    pool = frames["DEMO_E"]
    for name in COLUMNS:
        if name != "DEMO_E":
            pool = pool.merge(frames[name], on="SEQN", how="left", validate="one_to_one")
    pool = pool[pool.RIDAGEYR.between(40, 79) & pool.DIQ010.eq(1) & ~pool.RIDEXPRG.eq(1)].copy()
    pool = pool[pool.DID040.between(0, 79) & pool.DID040.le(pool.RIDAGEYR)].copy()
    pool["age_years"] = pool.RIDAGEYR
    pool["sex_recorded"] = pool.RIAGENDR.map({1: "Male", 2: "Female"})
    pool["diabetes_duration_years"] = pool.RIDAGEYR - pool.DID040
    pool["hba1c_pct"] = pool.LBXGH
    sbp = pool[["BPXSY1", "BPXSY2", "BPXSY3"]].where(lambda x: x.gt(0) & x.le(300))
    dbp = pool[["BPXDI1", "BPXDI2", "BPXDI3"]].where(lambda x: x.ge(0) & x.le(300))
    pool["systolic_bp_mmhg"] = sbp.mean(axis=1).where(sbp.notna().sum(axis=1).ge(2))
    pool["diastolic_bp_mmhg"] = dbp.mean(axis=1).where(dbp.notna().sum(axis=1).ge(2))
    pool["bmi_kg_m2"] = pool.BMXBMI.where(pool.BMXBMI.gt(0))
    pool["insulin_use"] = pool.DIQ050.map({1: "Yes", 2: "No"})
    pool["diabetes_pill_use"] = pool.DID070.map({1: "Yes", 2: "No"})
    pool["smoking_status"] = pd.Series(pd.NA, index=pool.index, dtype="string")
    pool.loc[pool.SMQ020.eq(2), "smoking_status"] = "Never (<100 lifetime cigarettes)"
    pool.loc[pool.SMQ020.eq(1) & pool.SMQ040.eq(3), "smoking_status"] = "Former"
    pool.loc[pool.SMQ020.eq(1) & pool.SMQ040.isin([1, 2]), "smoking_status"] = "Current"
    pool = pool[pool.hba1c_pct.gt(0) & pool.systolic_bp_mmhg.notna()].sort_values("SEQN").reset_index(drop=True)
    pool.insert(0, "participant_id", pool.SEQN.map(lambda x: f"NHANES_E_{int(x)}"))
    pool["source_seqn"] = pool.SEQN.astype(int)
    pool["observed_retinopathy"] = pool.OPDUARMA
    assessed = pool[pool.observed_retinopathy.isin([0, 1])].reset_index(drop=True)
    original_16 = assessed.sample(n=16, random_state=42)
    remaining = assessed[~assessed.participant_id.isin(original_16.participant_id)]
    additional_13 = remaining.sample(n=13, random_state=43)
    selected = pd.concat([original_16, additional_13]).sort_values("SEQN").reset_index(drop=True)
    assert len(pool) == 512 and len(assessed) == 439 and len(selected) == 29
    assert selected.participant_id.nunique() == 29 and selected[FIELDS].notna().all().all()
    expected_ids = json.loads((REFERENCE / "expanded_cleaning_report.json").read_text())["selected_source_seqn"]
    if selected.SEQN.astype(int).tolist() != expected_ids:
        raise ValueError("Sampling IDs do not match the recorded selection")
    output = selected[["participant_id", *FIELDS]]
    current = pd.read_csv(ROOT / "CareCompass_EyeRisk_29_Patients.csv")
    pd.testing.assert_frame_equal(output.reset_index(drop=True), current, check_dtype=False, check_exact=False, atol=1e-8, rtol=0)
    if not verify_only:
        output.to_csv(ROOT / "CareCompass_EyeRisk_29_Patients.csv", index=False, float_format="%.9f")
        reference = selected[["participant_id", "source_seqn", *FIELDS, "observed_retinopathy", "OPDURL4", "BPXSY1", "BPXSY2", "BPXSY3", "WTMEC2YR", "SDMVPSU", "SDMVSTRA"]]
        reference.to_csv(REFERENCE / "patient_provenance_29.csv", index=False, float_format="%.9f")
        pool[["participant_id", "source_seqn", *FIELDS, "observed_retinopathy"]].to_csv(REFERENCE / "complete_pool512.csv", index=False, float_format="%.9f")
    print("Verified: 512 complete-input participants; 439 assessed; original 16 + additional 13 = 29. All ten selected fields observed, no imputation.")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW, help="Optional existing cache of the seven official XPT files")
    parser.add_argument("--verify-only", action="store_true", help="Verify source reproduction without rewriting the prepared data")
    args = parser.parse_args()
    reproduce(args.raw_dir, args.verify_only)
