"""Sample-size and power numbers behind docs/rules/r10_lppcn_prereg.md (section 5).

Reads no brain data. Every input is either a fixed design choice from the rule or a number already committed in
results/r2_paranoia.json. Writes results/prereg_lppcn_power.json.

    python scripts/prereg_lppcn_power.py
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.05
N_AVAILABLE = 35           # Chinese listeners in ds003643 participants.tsv, each with 9 preprocessed runs
N_ORIGINAL = 22            # Paranoia (R2), the result being replicated
P0 = 0.70                  # sensitivity the test must beat (rule section 4, H2)
FA_MAX = 2                 # most wrong-story false alarms allowed out of n (rule section 4, guard)
SIMS = 20000
SEED = 20261125


def t_power(d, n):
    df = n - 1
    return float(1 - stats.nct.cdf(stats.t.ppf(1 - ALPHA, df), df, d * np.sqrt(n)))


def wilcoxon_power(d, n, rng):
    x = rng.normal(d, 1.0, size=(SIMS, n))
    hits = 0
    for row in x:
        hits += stats.wilcoxon(row, alternative="greater").pvalue < ALPHA
    return hits / SIMS


def kstar(n, p0):
    return next(k for k in range(n + 1) if stats.binom.sf(k - 1, n, p0) <= ALPHA)


def main():
    rng = np.random.default_rng(SEED)
    r2 = json.loads((ROOT / "results/r2_paranoia.json").read_text())
    ve = r2["models"]["stock"]["volunteer_equivalent_language"]
    mean_z, (lo, hi) = ve["z"], ve["ci95"]
    sd_z = (hi - lo) / (2 * 1.96) * np.sqrt(N_ORIGINAL)   # back out the listener SD from the bootstrap CI
    dz_paranoia = mean_z / sd_z

    # Simonsohn (2015) small telescopes: d33 is the effect the original n had 33% power to detect.
    df = N_ORIGINAL - 1
    d33 = brentq(lambda d: t_power(d, N_ORIGINAL) - 1 / 3, 0.01, 3)

    out = {
        "inputs": {"alpha_one_sided": ALPHA, "n_available": N_AVAILABLE, "n_original": N_ORIGINAL, "p0": P0,
                   "fa_max": FA_MAX, "sims": SIMS, "seed": SEED,
                   "paranoia_mean_language_z": mean_z, "paranoia_ci95": [lo, hi]},
        "mudrik_rule_n": 2.5 * N_ORIGINAL,
        "paranoia_listener_sd_backed_out": float(sd_z),
        "paranoia_dz": float(dz_paranoia),
        "d33_for_n22": float(d33),
        "h1_group": {},
        "h2_sensitivity": {},
        "guard_false_alarms": {},
    }
    for n in (N_ORIGINAL, N_AVAILABLE, 55):
        out["h1_group"][str(n)] = {
            "t_power_d0.5": t_power(0.5, n),
            "wilcoxon_power_d0.5_sim": wilcoxon_power(0.5, n, rng),
            "t_power_d33": t_power(d33, n),
        }
        k = kstar(n, P0)
        out["h2_sensitivity"][str(n)] = {
            "kstar": k,
            "power_at_true_sensitivity": {str(p): float(stats.binom.sf(k - 1, n, p)) for p in (0.80, 0.85, 0.90, 0.95)},
        }
    n = N_AVAILABLE
    out["guard_false_alarms"] = {
        "rule": f"at most {FA_MAX} of {n} ({100 * FA_MAX / n:.1f}%)",
        "prob_guard_passes_at_true_rate": {str(f): float(stats.binom.cdf(FA_MAX, n, f)) for f in (0.01, 0.03, 0.05, 0.07, 0.10)},
    }
    # Clopper-Pearson one-sided 95% lower bound on sensitivity for a few outcomes at n = 35
    out["cp_lower_95_one_sided"] = {str(k): float(stats.beta.ppf(ALPHA, k, n - k + 1)) for k in (35, 33, 31, 29, 28)}
    (ROOT / "results/prereg_lppcn_power.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
