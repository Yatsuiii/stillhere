# How often does a healthy listener miss a passive story test, by scan length? (frozen 2026-11-26, before any BOLD is downloaded)

Clinics that use passive language listening can only read a negative patient result if they know how often healthy people come out negative at the same scan length. This measures that miss rate on public data, model-free, the way a clinic would test: one person against a group of healthy references.

**Data.** Narratives (Nastase et al. 2021; OpenNeuro ds002345; fmriprep fsaverage5 derivatives and confounds from the fcp-indi mirror). Stories at least 13 minutes long, heard as one continuous run: black, forgot, merlin, sherlock, tunnel, 21styear, and the slumlord part of slumlordreach. Every listener of those stories, minus the dataset's own scan_exclude.json. Timing from the dataset's event_meta.json: keep volumes onset..onset+duration, then drop the first 6 (as the dataset's run_isc.py does). TR 1.5 s. Only code/*.json files and file listings were read before this file was frozen.

**Cleaning (per run).** Both hemispheres, fsaverage5. Regress, over the kept volumes: 6 motion parameters and their temporal derivatives, a_comp_cor_00 to 04, and fmriprep's cosine regressors; then z-score each vertex. Network series = mean over the network's vertices, z-scored. LANGUAGE (primary) = left TempPar, DefaultB_Temp, DefaultB_PFCv, DefaultB_PFCl of Schaefer-400 17-network (stillhere/heldout.py parcel_sets). HEARING (secondary) = that file's AUD set.

**Per-person test.** Reference = mean series of up to 20 other listeners of the same story (drawn once with numpy default_rng(20261126); all others if fewer than 20). Score = Pearson r between the listener's series and the reference over the first k minutes, k in {1, 2, 3, 5, 7, 10, 13}. p = (m + 1) / 1001 from 1000 phase-randomised surrogates of the listener's series (same k-minute window).

**Calibration (fixed false-alarm rate).** Wrong-story control: the same listener scored against the reference of each other primary story (20 listeners of that story, never the listener themself), same k minutes. Per k and network, the p cutoff c_k is the 5th percentile of all wrong-story p values, so wrong-story positives are 5% by construction. Detection at k = share of listener-runs with right-story p <= c_k. The uncalibrated p < .05 rate and its wrong-story rate are reported beside it.

**Primary result.** LANGUAGE miss rate (1 - detection) at 5 and 10 minutes, with Wilson 95% CIs and a bootstrap CI that resamples people (a person who heard two stories counts once per resample). The full curve over k and per-story rates are reported. No bar: this is a measurement, reported whatever it shows.

**Known limits, written now.** These are healthy young adults in a research scanner, not patients; motion and arousal in patients will make misses more common, so these rates are a floor. The reference is 20 listeners of the same story, which a clinic may not have. Some people heard several of these stories, so runs are not independent (handled by the person bootstrap).

Never "awareness", "consciousness" or any diagnostic wording. A pass means the language-network time course followed the healthy group's, nothing more.
