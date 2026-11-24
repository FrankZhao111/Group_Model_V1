# Data dictionary and source scope

The model table is `CareCompass_EyeRisk_29_Patients.csv`. One row is one distinct, deidentified public-use NHANES participant from 2007-2008. It is a deliberately selected 29-person demonstration subset of a larger usable pool, not the entire survey and not 29 rows from the original literature-extraction CSV.

**Outcome for clustering: none.** The optional retinal assessment in the reference file is kept out of fitting, elbow selection, and the relative-burden ordering. It is joined after fitting for a separate descriptive table.

| Field | Unit or values | Model role | Source |
| --- | --- | --- | --- |
| participant_id | Public source-ID-derived string | Identifier only; excluded from distances | SEQN |
| age_years | Years | Continuous input; standardized | DEMO_E.RIDAGEYR |
| sex_recorded | Female/Male | Category input; Female=0, Male=1; not standardized | DEMO_E.RIAGENDR |
| diabetes_duration_years | Approximate whole years | Continuous input; standardized | Age minus DIQ_E.DID040 |
| hba1c_pct | % | Continuous input; standardized | GHB_E.LBXGH |
| systolic_bp_mmhg | mmHg | Continuous input; standardized | Mean of observed valid BPXSY1-3; at least two readings |
| diastolic_bp_mmhg | mmHg | Available but excluded from v1 | Mean of observed valid BPXDI1-3; at least two readings |
| bmi_kg_m2 | kg/m2 | Available but excluded from v1 | BMX_E.BMXBMI |
| smoking_status | Current/Former/Never (<100 lifetime cigarettes) | Available but excluded from v1 | SMQ020 and SMQ040 |
| insulin_use | Yes/No | Available but excluded; treatment context | DIQ_E.DIQ050 |
| diabetes_pill_use | Yes/No | Available but excluded; treatment context | DIQ_E.DID070 |
| observed_retinopathy | 0/1 or missing, reference only | Not a clustering outcome or feature; post-fit descriptive checking | OPXRET_E.OPDUARMA |

The source calls RIAGENDR `Gender` and releases Male/Female categories. It does not capture a full set of gender identities. The code does not assign an unsupported category a made-up value or treat the category number as medical severity.

## Cleaning and reproducible selection

`prepare_data.py` verifies the recorded raw-file checksums and joins source tables one-to-one on SEQN. It normalizes negligible zero-decoding artifacts, handles questionnaire unknown codes as missing, and respects smoking skip patterns. A pressure average is an average of measured readings, not imputation. The source protocol permits a diastolic value of zero, which is retained.

Eligibility: age 40-79; self-reported clinician-diagnosed diabetes; no recorded current pregnancy; a valid age at diagnosis no greater than current age; measured HbA1c; and at least two valid systolic readings. Age 80 is excluded because it is top-coded, preventing an exact age-minus-diagnosis-age calculation. Diabetes type is not verified, so this is not a confirmed Type 2-only sample.

These filters produce 512 people with complete core measurements; 439 also have an observed retinal assessment. The earlier 16 are sampled from those 439 with random_state=42 after sorting by source ID. Thirteen more are sampled without replacement from the remaining 423 with random_state=43. Selection is not balanced by disease status or filtered on optional-field completeness.

All 29 selected people have all ten available input fields observed. There is no imputation. The larger pool contains missing optional data and missing retinal assessments; those remain missing. An unavailable examination finding is not converted to no disease.

The 2007-2008 cycle and our filters differ from the full analytic sample in the 2005-2008 paper linked as CC0078 in the original project CSV. The sampling and manifest JSON files preserve that distinction.

## Limits

The data is cross-sectional, not a record of future eye-disease incidence or future quality-of-life decline. It does not estimate the effect of seeing a specialist. The 29-person unweighted demonstration is not nationally representative. Source identifiers, weights, group size, optional treatment fields, and retinal findings do not enter the v1 distance calculation.
