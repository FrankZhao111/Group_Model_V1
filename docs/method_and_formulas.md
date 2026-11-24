# Method and formulas

## Encoding and scaling

The four continuous measurements are transformed with training statistics: z = (value - training mean) / training population standard deviation. The recorded category is encoded Female=0 and Male=1 with weight 1.0 and is left unstandardized. Thus two otherwise identical profiles in different recorded categories have an additional squared distance of 1. The category does not receive an ordinal medical risk score. Standardizing it to unit variance would give it a different weight; this v1 makes that choice explicit.

All fitted distances use five coordinates, even when a figure shows two axes. `outputs/transformed_training_inputs.csv` contains the exact coordinates. For this 29-person version, every row is a training row. For a dataset with 30 or more rows, `split_data` creates a random 70/30 split before fitting scaling or centers; test rows are transformed with the training statistics.

## K-means and the baseline

For each candidate k from 1 to 6, K-means uses random_state=42 and 50 initializations. A person's assigned center is the one minimizing the sum of squared coordinate differences.

SSE = sum across training people of squared distance to the assigned center. Mean squared distance = SSE / n. RMS distance = sqrt(SSE / n). This is the combined spread across all fitted coordinates, not the standard deviation of one particular clinical measurement. Raw-unit feature means and population SDs are reported separately for each group.

The one-group baseline uses the same preprocessing and all people share one center. Percentage SSE reduction = 100 × (1 - model SSE / baseline SSE). Adding groups generally reduces training SSE; this is a fit comparison, not accuracy or proof of generalization.

## Elbow choice

The code normalizes candidate k and SSE to the range 0-1 and measures how far the curve lies below the straight chord between its endpoints. It selects the k with the largest gap. Candidate gaps within 0.02 of the maximum are reported as close alternatives. The current data has a close 3-versus-4 choice; the plot and gap values should be read together. This heuristic does not establish a clinically meaningful number of risk levels.

For a held-out set, test SSE is the sum of squared distances from transformed test rows to the nearest **training-fitted** centers, separately for every candidate k. Test centers are not fitted. The test elbow is checked after selecting k from training. In the 29-person version, a test curve and train/test elbow agreement are unavailable, not reported as passing.

## Relative-burden ordering and decision

Group order is a declared interpretation rule: take the equal-weight mean of each group's standardized diabetes-duration, HbA1c, and systolic-pressure means. Rank groups by this descriptive index. With three groups, the labels are lower, middle, and higher **relative risk-factor burden**. Age and recorded sex/gender are not assigned increasing medical-severity weights by this ordering rule.

The decision function encodes a new user's five answers using the saved training statistics, computes distance to the saved centers, identifies the nearest group, and returns the information package. The highest relative-burden group gets enhanced eye-health information; other groups get standard eye-health information. Everyone remains eligible for eye-health information. This is not a diagnosis or an examination-urgency decision. Missing or unsupported inputs receive an explanatory error rather than a fabricated group. Values beyond observed training ranges are reported in the output.

## Descriptive checking and sensitivity

Retinal examination findings are joined only after fitting; the counts by group are not used to choose k, fit centers, or order the groups. They describe existing findings, not prospective risk or held-out prediction accuracy. The counts are unweighted and the smallest group has only a few people.

The code refits the same k without the recorded category and reports adjusted Rand agreement, which is insensitive to arbitrary cluster-number changes. One means identical grouping. It also checks five alternative random seeds. These checks describe sensitivity, not external validation.

## Business-unit calculation

Every value assumption is in `config.json`. The scenario assumes 100 monthly users, five minutes locating relevant content with generic information, and three minutes for recipients of the enhanced information package. It assumes future group shares equal training shares. None of these navigation times was measured.

Expected enhanced recipients = monthly users × enhanced-group training count / training n.

Baseline minutes = monthly users × generic minutes.

Model minutes = expected enhanced recipients × enhanced minutes + remaining users × generic minutes.

Assumed saved minutes = baseline minutes - model minutes. These assumptions do not estimate avoided disease, treatment effects, healthcare cost savings, or measured quality-of-life improvement.
