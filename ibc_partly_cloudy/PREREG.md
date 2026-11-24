# Do personal ToM regions rescue the adults that group spheres miss? (IBC, ds002685)

Frozen 2026-11-24, before any IBC movie time series was extracted. Only sub-01's file headers and events were opened.

Follows [partly_cloudy/RESULTS.md](../partly_cloudy/RESULTS.md). There, 6 of 33 adults in ds000228 were missed on ToM using group spheres from one 5.6 min viewing.

Data: OpenNeuro ds002685 (Individual Brain Charting, Pinho et al.). 12 adults, all with ses-*/task-PainMovie (Partly Cloudy, two viewings: dir-ap and dir-pa) and task-TheoryOfMind (false-belief vs photo localiser, two runs) in the same session, plus ses-00 T1w.

Preprocessing (per subject):
1. Resample every BOLD volume to 3 mm isotropic.
2. Rigid motion correction of all four runs to the mean of the first PainMovie run (ANTs). Keep the 6 motion parameters.
3. Mean EPI -> T1 (rigid), T1 -> MNI152 2 mm (nilearn template, ANTs SyN). Inverse transforms bring MNI-space ROIs to native EPI space.

ROIs:
- Group: the ds000228 9 mm spheres (DMPFC, MMPFC, VMPFC, PC, LTPJ, RTPJ), warped to native space.
- Personal: localiser GLM (belief vs photo boxcars, canonical double-gamma HRF, 6 motion params, linear drift, both runs). In each 15 mm sphere around the same centres, take the top 50 voxels by belief > photo t.

Movie series: regress 6 motion params + constant + linear trend, z-score per region, average over regions.

Alignment to the ds000228 time base: one offset k (in TRs) is chosen to maximise the correlation between the IBC group-mean PAIN network (group spheres, all 24 viewings) and the ds000228 adult pain reference (158 TRs). It is fixed for everyone. The pain network is used only for alignment. ToM is the test.

Tests (each of 24 viewings, and both ROI types):
- r(IBC ToM series, ds000228 adult ToM reference). 1000 circular shifts, detected if p < .05.
- False-alarm check: the same test against the time-reversed reference.
- Test-retest: r(viewing 1, viewing 2) within person, per ROI type.

Primary outcome: the number of viewings detected with personal ROIs vs group spheres (McNemar on the 24 paired viewings).
Secondary: test-retest r by ROI type.

No subject is excluded. Pipeline failures are reported per subject.
