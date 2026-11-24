# CareCompass — Group Model V1

Standalone clustering assignment for CareCompass. The workflow uses **29 real NHANES participants and five user-input measurements**, compares a one-group baseline with K-means, assigns a separate example user to a profile, and calculates a clearly labeled hypothetical business value.

**Business question:** Which eye-health information package should CareCompass show based on the user's clinical risk-factor profile?

The v1 decision changes information emphasis. It does not diagnose eye disease, estimate a future disease probability, quantify quality-of-life loss, or decide whether someone should receive an eye examination.

## Start here

With Python 3.12 available, clone the repository and run:

```bash
git clone https://github.com/FrankZhao111/Group_Model_V1.git
cd Group_Model_V1
python -m pip install -r requirements.txt
python run_model.py
```

The model command uses the committed CSV and works without downloading source data or connecting to Railway. If Windows recognizes `py` rather than `python`, use `py` in those commands.

A virtual environment is recommended. For Windows PowerShell, without needing to activate it:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_model.py
```

Use the saved model on another input, without fitting again:

```bash
python predict.py --input examples/new_user.json
python predict.py --input examples/second_user.json
```

Run the behavioral checks:

```bash
python -m unittest discover -s tests -v
```

The pinned dependency versions were tested with Python 3.12. The program records the package versions actually used in `outputs/run_manifest.json`.

## Step 1 — Understand, verify, and split the data

The root CSV is the same 29-person file prepared for this project. One row is one distinct, deidentified public-use CDC NHANES 2007-2008 participant. This is a **declared demonstration subset**, not the full available survey or a claim that only 29 usable people exist.

The larger pool contains 512 people with complete inputs, of whom 439 have an observed retinal assessment. The earlier 16-person sample is retained and expanded with 13 more people without replacement. The sample is not chosen or balanced by retinal result. All ten available fields happen to be observed for these 29 people; no cells are imputed.

These participant records came from additional retrieval of public NHANES files. They were not individual patient rows in the original project literature CSV. The connection to that CSV is paper CC0078, which uses NHANES. The 16 earlier population-segment records from a different paper are not this model's observations.

**The assignment says under 30, so all 29 observations train.** There is no test set, test elbow, or train/test elbow confirmation for this version. Those results are marked unavailable, not passed. The code supports the required random 70/30 split if the dataset has 30 or more rows and fits scaling only on the training rows.

The clustering model has **no outcome variable**. Retinal examination findings remain in a separate reference table and are joined only after fitting for descriptive checking.

To reproduce the data from all seven official source files, optionally run:

```bash
python prepare_data.py
```

This optional command downloads about 20 MB into the ignored `data/raw/` cache if needed, verifies source checksums, reproduces the 29 selected IDs, checks all prepared values, and writes source reference tables. It does not invent patient profiles. To verify without rewriting the prepared tables, add `--verify-only`. The model can be run directly without this optional download.

Details: [data dictionary](docs/data_dictionary.md), [sources](docs/sources.md), and the source/sampling JSON files in `data/reference/`.

## Step 2 — Encode, scale, and fit

The five core user inputs are:

| Input | User supplies | Processing |
| --- | --- | --- |
| Age | Years | Standardize using training mean and population SD |
| Recorded sex/gender | Source-supported Female/Male category | Female=0, Male=1; leave the category indicator unstandardized |
| Diabetes duration | Years since first diagnosis | Standardize |
| Latest HbA1c | Lab result in % | Standardize |
| Recent systolic pressure | Average top number, mmHg | Standardize |

The source labels its two-category variable Gender; it does not provide a full set of gender identities. The binary values are category codes, not medical severity scores. The category has an explicit weight of 1.0, so two otherwise identical profiles in different categories differ by one squared-distance unit. The program reports a sensitivity comparison omitting this field.

Diastolic pressure, BMI, smoking category, insulin use, and diabetes-pill use remain available in the data but are excluded from v1. Source IDs, survey weights, population share, and retinal findings are excluded. Adding every available field is not necessary for this small assignment model.

The baseline is K-means with k=1. The program fits k=1 through k=6 with the same preprocessing, random_state=42, and 50 initializations. The training elbow rule selects the maximum gap below the endpoint chord after normalizing k and SSE. Close alternatives are explicitly reported. Inspect the plot rather than claiming a clinically optimal number of risk groups.

## Step 3 — Evaluate baseline versus v1

The current run's numerical results are generated in [outputs/RESULTS.md](outputs/RESULTS.md), with full precision in CSV/JSON files. They include:

- Training elbow values and the selected k, including close alternatives.
- One-group versus v1 training SSE and RMS distance across the five fitted coordinates.
- Group counts, raw-unit feature means, population SDs, and combined within-group spread.
- The grouping change when the recorded category is omitted, and checks across alternative random seeds.
- Separate, post-fit descriptive retinal-finding counts.

SSE reduction measures tighter grouping of the training inputs. It is **not medical accuracy, risk reduction, or held-out validation**. The current three-versus-four elbow choice is close, separation is modest, and the smallest group has only a few people. Those limitations should be part of the team's own interpretation.

The two-axis profile plot is a projection: the fit still uses all five encoded/scaled coordinates. See [method and formulas](docs/method_and_formulas.md) for exact definitions of scaling, distances, spread, the elbow rule, and sensitivity.

## Step 4 — Make the new-user decision

`carecompass/decision.py:make_decision` accepts the five raw answers and the saved model artifact. It applies the saved preprocessing, finds the nearest center, and returns the group, its relative-burden description, and the information package. `predict.py` exposes this function at the command line.

Example intake:

```json
{
  "age_years": 60,
  "sex_recorded": "Female",
  "diabetes_duration_years": 20,
  "hba1c_pct": 10.5,
  "systolic_bp_mmhg": 145
}
```

This is a separate realistic demonstration input, not an added training patient. Its exact decision is saved in `outputs/example_decision.json`.

After fitting, the groups are ordered by the equal-weight average of their standardized diabetes-duration, HbA1c, and systolic-pressure means. This is an explicit descriptive interpretation rule, not a fitted or validated future-risk score. With three groups, the labels describe lower, middle and higher **relative risk-factor burden**. Age and the recorded category do not receive ordinal clinical-severity weights in this ordering rule.

The highest relative-burden group receives enhanced eye-health information; others receive standard eye-health information. Both packages provide eye-health information. An unknown or unsupported answer returns a clear error; the function never treats unknown as normal. It reports when a numerical answer lies outside the observed training range.

## Step 5 — Quantify business value with explicit assumptions

The assignment permits reasonable invented business assumptions when actual figures are unavailable. This version uses user-navigation minutes, not claimed clinical benefit.

The default scenario in `config.json` assumes 100 monthly users, five minutes locating relevant information with the generic package, three minutes for enhanced-package recipients, and a future group mix matching the training sample. All navigation times and resulting benefits are hypothetical. No time saving, treatment effect, avoided eye disease, or quality-of-life improvement was measured.

`outputs/business_value.json` records the enhanced-group share, expected recipients, baseline total minutes, v1 total minutes, and the difference. The formula is monthly users × enhanced-group share × assumed minutes saved per recipient. Changing the assumptions and rerunning the script reproduces the revised value.

## Step 6 — Prepare the team-written submission

The original assignment is preserved in [docs/group_model_assignment.md](docs/group_model_assignment.md). Use the [team submission checklist](docs/team_submission_checklist.md) to check coverage. The team must write its own title slide and two business slides, and submit one PDF.

This repository provides code, data, reproducible figures, and a technical results record. It does not generate submission slides or a business headline for the team. Run the chart code yourselves before using its figures and write your own explanation of what the numbers mean. Standalone model execution is sufficient; app integration is not required.

## Project files

| File or directory | Purpose |
| --- | --- |
| `run_model.py` | Standalone entry point; runs and prints the assignment steps |
| `prepare_data.py` | Optional reproduction from official source files |
| `predict.py` | New-user decision from the saved artifact, without fitting again |
| `config.json` | Reproducible model settings and labeled business assumptions |
| `carecompass/data.py` | Input validation, source checking, under-30 split rule |
| `carecompass/model.py` | Encoding, scaling, K-means candidates, elbow choice, group summaries |
| `carecompass/evaluation.py` | Baseline comparison, optional test distances, sensitivity, post-fit checking |
| `carecompass/decision.py` | Portable intake-to-group decision function |
| `carecompass/business.py` | Hypothetical business-unit calculation |
| `carecompass/plots.py` | Chart-generating code |
| `carecompass/reporting.py` | Technical results generated directly from the numerical outputs |
| `tests/` | Boundary, leakage, saved-model prediction, validation, and value checks |
| `data/reference/` | Source provenance, checksums, sampling, and larger pool |
| `outputs/` | Reproducible model, metrics, group summaries, assignments, example decision, value and figures |

The saved artifact is ordinary JSON containing preprocessing statistics and centers. Prediction does not require loading a pickle or repeating the fit.
