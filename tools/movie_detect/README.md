# movie_detect

This tool asks whether a movie drives a brain network in **one person**, and how many minutes of movie you need. It calibrates the false-alarm rate and puts confidence intervals on every rate.

```
pip install -r requirements.txt            # pinned: numpy 2.3.5 (+ nibabel 5.5.0, nilearn 0.14.1 for extract.py)
python extract.py --bold sub-01_space-MNI_bold.nii.gz --confounds sub-01_confounds.tsv \
       --spheres spheres.tsv --drop 10 --out tsv/sub-01.tsv      # one TSV per subject (or sub-01_view-2.tsv for repeats)
python movie_detect.py tsv/ --tr 2 --regions DMPFC,MMPFC,VMPFC,PC,LTPJ,RTPJ --out out/
```

## Outputs
- `per_subject.tsv`: r, p and detected for each subject, plus the share of windows detected at each length.
- `summary.json`: group rate with a Wilson 95% CI, calibrated cutoffs, the false-alarm rate achieved, and the minutes curve.
- `receipt.json`: arguments, versions, and SHA-256 of every input and output.

## Method
- The network series is the mean of z-scored regions, averaged over a subject's views when there are repeats.
- The reference is the leave-one-subject-out mean, taken from the reference group (`--reference-subjects`).
- The null is 2000 phase-randomised surrogates.
- The cutoff is set so that tests against 20 mismatched references (time-reversed, circularly shifted) give `--fa`, default 5%.
- `--views k` keeps only each subject's first k views, which lets you measure the repeat-viewing gain.

## Check
On the 33 ds000228 adults (Partly Cloudy, cleaned with the shared ART and CompCor regressors), the tool detects **27/33 (82%, CI 66-91%)**.
The full analysis ([final.py](../../partly_cloudy/final.py)) gives 26/33. The difference is one borderline adult, sub-pixar127 (p ≈ .05). Its result depends on which pool the cutoff is calibrated on: adults only here, all 155 people in the full analysis.
