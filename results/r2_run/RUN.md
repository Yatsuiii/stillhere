# R2 and R5 scoring run (Kaggle CPU kernel)

The one scoring run of `docs/rules/r2_paranoia_rule.md` (section 11) and `docs/rules/r5_short_scan_rule.md`.

- Kernel: `redlotusthepotus/stillhere-r2-score`, version 2 (version 1 stopped at 08:41:59 UTC on the missing tar, before any
  download; 7f34b71). Pushed Nov 1 2026, 08:48:04 UTC; the code is `r2/kaggle_score/stillhere_r2_score.py` at 7f34b71.
- Started 08:48:07 UTC, ended 10:02:23 UTC (`r2_score_run.json`). Steps: download 213 s, score 4,134 s, R5 99 s; each returned 0.
- Clone: full (213 commits, not shallow), HEAD bfbda2b, code equal to the pins commit 354cab7. Inputs: bundle SHA-256
  455f2a5c853ccb5d997b3ac83e6885deeb670a6727e30c71df5fcd35f5424196, scoring root tar SHA-256
  ba6a090b67a6180fd44bc001ce78b6a8de763e867ef5ecdf4a61e69b4b9f5f89.
- Status COMPLETE; the output was downloaded on Nov 2 2026 at 04:01 UTC and checked against the kernel's `MANIFEST.sha256` (SHA-256
  e483fbff15765f5771f5807d45c9dfe622bee3c66e34300914bbd70d62e684bd). Every file matches except `score_driver.log`, whose last line
  ("done", 10:02:23 UTC) the kernel writes after the manifest; without that line its SHA-256 is the manifest's (b37f2c1f...).
  The kernel's own log is `kaggle_kernel_log.json` (SHA-256 27c85766fc6c5593d60f997f822f720ad52e197a48bb30bcbf090ad04518cf2e).
- `private/` (per-listener values) is not committed; its files' SHA-256 are in the manifest. The kernel was not rerun.
- Committed here: `results/r2_paranoia.json`, `results/r2_paranoia_SHA256.json` (every input the scoring read), `results/r2_bold_SHA256.json`
  (every GIFTI downloaded) and `results/r5_short_scan.json`, as the kernel wrote them.

Lines, as the scorers printed them:

- On another public story, outside TRIBE v2's training and evaluation sets (Paranoia, Finn et al. 2018; 22 listeners, the whole story:
  1,250 volumes = 20.8 minutes at TR 1 s), stock TRIBE v2's forecast was worth about 3 real volunteers, and at least 1. It reached 59%
  of what the other listeners' average reaches.
- What it had to beat: word rate and the sound envelope (beaten in 4 of 4 comparisons).
- Fine-tuned TRIBE v2: rule committed, not run (the stock runs' Llama features were not kept).
- z is not compared across datasets with different scan timing; the share is.
- Exploratory. On another public story (Paranoia, Finn et al. 2018; 22 listeners, outside TRIBE v2's training and evaluation sets),
  listeners' language areas followed TRIBE's forecast more closely in the 7 minutes it ranked highest than in the 7 minutes it ranked
  lowest (17 of 22 listeners, one-sided p = 0.0004). Ranking the same minutes by word rate alone: 8 of 22, p = 0.94.
