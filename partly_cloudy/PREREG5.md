# Repeat-viewing replication at larger n: Aly et al. 2018, OpenNeuro ds001545 (frozen 2026-11-24, before download)

Data: 30 adults, 3 runs each. Every run shows the same 90 s "Intact A" clip from The Grand Budapest Hotel twice, so each person has 6 viewings. TR = 1.5 s, 60 volumes per clip. The "Scramble Fix C" clip is also shown 6 times, in the same scrambled order each time.

Preprocessing: the same as the IBC route.
- Drop 3 volumes per run, as the README says.
- Resample to 3 mm and rigid motion correction to the run-1 mean.
- Mean EPI -> MNI152 2 mm (ANTs SyN, masked EPI), with Dice QC reported.
- The ds000228 ToM and pain 9 mm spheres are warped to native space.
- Per run: regress motion parameters, a constant and a linear trend.
- Clips come from the README volume table, using each run's events.tsv labels.

Per person, network = mean of z-scored region series. Viewings are put in presentation order, 1-6.
Reference: the leave-one-subject-out mean of all 6 viewings of every other subject.
Test for k = 1, 2, 3 and 6 viewings: z-score each of the first k viewings, average them, and test against the reference.
- Phase-randomised null with 2000 surrogates.
- Cutoff calibrated so that tests against mismatched references (reversed reference, 20 circular shifts) are detected 5% of the time, separately for each k.

Primary: ToM detection rate for k = 1, 2, 3 and 6 viewings (Intact A), with Wilson 95% CIs.
Also k = 1 for each single viewing position, 1 to 6, to see whether a first viewing differs from later ones.
Secondary: the pain network, and the Scramble Fix C clip.
No subject is excluded. Pipeline failures are reported.
Caveat fixed in advance: the clip is 90 s, so absolute rates will be lower than for 5 minute films. The question is how much repeating helps.
