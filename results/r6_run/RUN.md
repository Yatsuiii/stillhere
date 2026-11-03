# R6 scoring run (docs/rules/r6_alice_short_scan_rule.md)

- Rule, scorer and tests: c0ed5f2, committed 2026-11-03T18:48:01Z and pushed to origin/main by 18:53:48Z (the cloud's ruling of
  Nov 3, 19:05 UTC, item 3), before any R6 statistic existed.
- One scoring run, on the laptop's CPU: started 2026-11-03T18:53:58Z, ended 19:02:26Z, exit 0. No aborted attempt.
  `python scripts/r6_alice_short_scan.py score --root <analysis tree> --out <a folder outside the repository>`, from a checkout equal to
  origin/main at c0ed5f2 (the scorer's own check). Python 3.12.13, numpy 2.2.6, scipy 1.18.1, nibabel 5.4.2, nilearn 0.14.1.
- Inputs: checked by the scorer against the Alice rule's pins (stimulus, annotations, the two stock forecasts) and the brain files
  against the SHA-256 the Alice download recorded (results/r6_alice_short_scan.json, inputs_sha256).
- Output: results/r6_alice_short_scan.json (group numbers and the rule's line). The per-listener file stays outside the repository;
  its SHA-256 is 36986407f83eb8b9be68164c080ecfe659bd655afb51f75c97de65da2f316e5c.
- Outcome: FAIL, reported as the rule says. The scorer's printed line, which is the file's line:
  Exploratory; the Alice scans were read on Oct 30 for the held-out test. A replication attempt of R5: on a second public story outside TRIBE v2's training and evaluation sets (the Alice Datasets, chapter 1; 22 listeners), the minutes TRIBE's forecast ranked highest did not reliably give a closer language-area match than the minutes it ranked lowest (12 of 22 listeners, one-sided p = 0.34). R5's Paranoia result is not replicated here. We show it because we said we would.
