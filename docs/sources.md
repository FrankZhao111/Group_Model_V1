# Sources

## Data

All participant measurements come from public-use CDC NHANES 2007-2008 files. The committed source manifest records the official XPT download URLs and SHA-256 checksums. No user account credentials or database connections are required to reproduce the data.

- [Demographic variables and category coding](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/DEMO_E.htm)
- [Diabetes history and age at diagnosis](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/DIQ_E.htm)
- [Measured HbA1c](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/GHB_E.htm)
- [Blood-pressure measurements and protocol](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/BPX_E.htm)
- [BMI and body measurements](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/BMX_E.htm)
- [Smoking questionnaire and skip patterns](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/SMQ_E.htm)
- [Retinal imaging and worse-eye examination summaries](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/OPXRET_E.htm)

## Clinical context

[Zhang et al., Prevalence of Diabetic Retinopathy in the United States, 2005-2008, JAMA 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2945293/) is linked as CC0078 in the original project literature CSV. It provides association evidence for diabetes duration, HbA1c, systolic pressure, and recorded sex. Our small demonstration does not reproduce its sample, weighted estimates, regression, or clinical conclusions.

[NIDDK: Diabetic Eye Disease](https://www.niddk.nih.gov/health-information/diabetes/overview/preventing-problems/diabetic-eye-disease) provides background on diabetes duration, blood glucose and blood pressure. These sources motivate the input selection; they do not validate the clustering labels, the equal-weight burden index, or the assumed business benefit.

## Software and assignment

- [scikit-learn KMeans documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [scikit-learn StandardScaler documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- Installed and tested package versions are pinned in `requirements.txt`; each run records actual versions.
- The team's supplied assignment is preserved at `docs/group_model_assignment.md`. It allows chart-generating code and requires the team to write its own presentation slides.
