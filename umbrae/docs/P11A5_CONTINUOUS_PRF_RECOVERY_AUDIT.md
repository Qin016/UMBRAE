# P11-A.5 Continuous pRF Recovery Audit

| Asset | Official evidence | Recovery status | Available locally | Space | Units | Needed |
| --- | --- | --- | --- | --- | --- | --- |
| angle | cvnlab/nsddatapaper main/analysis_prf.m and official S3 object listing | AVAILABLE_OFFICIAL_PRECOMPUTED | True | subj01 func1pt8mm volume | degrees 0–360 | True |
| eccentricity | cvnlab/nsddatapaper main/analysis_prf.m and official S3 object listing | AVAILABLE_OFFICIAL_PRECOMPUTED | True | subj01 func1pt8mm volume | degrees visual angle | True |
| size | cvnlab/nsddatapaper main/analysis_prf.m and official S3 object listing | AVAILABLE_OFFICIAL_PRECOMPUTED | True | subj01 func1pt8mm volume | degrees, CSS effective sigma/sqrt(exponent) | True |
| quality | cvnlab/nsddatapaper main/analysis_prf.m and official S3 object listing | AVAILABLE_OFFICIAL_PRECOMPUTED | True | subj01 func1pt8mm volume | R2 percent | True |
| exponent | cvnlab/nsddatapaper main/analysis_prf.m and official S3 object listing | AVAILABLE_OFFICIAL_PRECOMPUTED | True | subj01 func1pt8mm volume | dimensionless | True |
| pRF center x/y | nsddatapaper PARAMETERSnotes.m formulas | AVAILABLE_OFFICIAL_RECOMPUTABLE | True | visual field | degrees visual angle | True |
| Gaussian sigma | analyzePRF.m defines size=sigma/sqrt(exponent) | AVAILABLE_OFFICIAL_RECOMPUTABLE | True | visual field | degrees visual angle | True |
| surface continuous maps | official S3 lh/rh.prf*.mgz listing | AVAILABLE_OFFICIAL_PRECOMPUTED | False | subj01 native FreeSurfer vertices | parameter-specific | False |
| surface geometry for mapping | NSD FreeSurfer subject release and nsdcode | AVAILABLE_OFFICIAL_PRECOMPUTED | False | subj01 native FreeSurfer | millimetres/vertices | False |

Primary sources: [cvnlab/nsddatapaper](https://github.com/cvnlab/nsddatapaper), [cvnlab/nsdcode](https://github.com/cvnlab/nsdcode), and [kendrickkay/analyzePRF](https://github.com/kendrickkay/analyzePRF). Official objects were recovered from the public `natural-scenes-dataset` S3 bucket; exact URLs and hashes are in `download_manifest.json`.
