# Per-child detection on Partly Cloudy (ds000228) — rules frozen before any subject data is analysed

Frozen 2026-11-24, before extracting any time series. Only sub-pixar001 was opened, to read array shapes.

Data: OpenNeuro ds000228 (Richardson et al. 2018), derivatives/fmriprep/*_swrf_bold.nii.gz (MNI, smoothed, as shared),
nuisance file *_ART_and_CompCor_nuisance_regressors.mat (R), ROI spheres derivatives/ROIs/*_9mm_sphere_xyz.mat (roi_XYZmm).

Networks (as in the paper): ToM = DMPFC, MMPFC, VMPFC, PC, LTPJ, RTPJ. Pain = AMCC, LInsula, RInsula, LS2, RS2, LMFG, RMFG.

Per subject:
1. ROI mean signal per region (voxels from roi_XYZmm mapped through the bold affine).
2. Regress all columns of R plus constant and linear trend from each region series.
3. Drop TRs 1-10 (rest + opening credits, per README). 158 TRs remain.
4. z-score each region, average regions to one network series.

Reference = mean network series of all adults (Child_Adult == adult), leaving the subject out if adult.

Statistic: Pearson r(subject network, adult reference), same network.
Null: 1000 circular shifts of the subject series (shift 10..148 TRs). Detected if p < 0.05 one-sided.
Empirical false-alarm check: same test against the time-reversed reference (expected ~5% detections).

Primary outcome: ToM detection rate per AgeGroup (3yo,4yo,5yo,7yo,8-12yo,adult).
Secondary: pain detection rate per AgeGroup; ToM specificity per subject = r(ToM, adult ToM) - r(ToM, adult pain).
Motion: number of single-TR spike columns in R; report detection by motion tercile within children.

No subject is excluded. Any failure, bug or deviation from these rules is reported, not hidden.
