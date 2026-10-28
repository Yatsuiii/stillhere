"""Cheap baselines beside TRIBE (docs/rules/cheap_baselines_rule.md; exploratory): HRF x word rate and HRF x envelope scored like TRIBE's
unpartialled prediction on the same parcels, volumes, cleaning and QC, and a paired comparison over listeners.

- compare(): paired d = TRIBE - baseline: n, k, Hodges-Lehmann median with a 95% listener bootstrap (10,000, default_rng(20261028)),
  two-sided Wilcoxon p, and the verdict ("TRIBE beats", "baseline beats", "no difference shown") from the interval.
- section8(): LPP-EN section 8 with the section-8 harness (heldout.clean, heldout.gauge with its circular-shift null, regressors at lag L).
- run_gauges() / pair_gauges(): the story wall and notthefall, built only from the frozen picker functions passed in as G (gauges) and
  C (clips): G.run_times, clip.covered, clip.regs, G.clean, G.confounds, C.keepmask and G.hearing (raw Fisher-z r over parcels of the
  cleaned BOLD rows with the filtered regressor rows). Nothing here changes a frozen result."""
import numpy as np

SEED, NBOOT = 20261028, 10_000
LABEL = "exploratory"
KINDS = ("tribe", "word_rate", "envelope")


def hl(d) -> float:
    d = np.asarray(d, float)
    i, j = np.triu_indices(len(d))
    return float(np.median((d[i] + d[j]) / 2))


def compare(tribe, base) -> dict:
    """TRIBE minus a baseline, paired over the same listeners (equal-length sequences, NaN pairs dropped)."""
    from scipy import stats
    a, b = np.asarray(tribe, float), np.asarray(base, float)
    ok = np.isfinite(a) & np.isfinite(b); d = a[ok] - b[ok]
    if len(d) < 2:
        return {"n": int(len(d)), "verdict": "not enough listeners"}
    rng = np.random.default_rng(SEED)
    bs = np.array([hl(d[rng.integers(0, len(d), len(d))]) for _ in range(NBOOT)])
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)); h = hl(d)
    p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    verdict = "TRIBE beats" if h > 0 and lo > 0 else "baseline beats" if h < 0 and hi < 0 else "no difference shown"
    return {"n": int(len(d)), "k_tribe_higher": int((d > 0).sum()), "hl": h, "hl_ci95": [lo, hi], "p_two_sided": p, "verdict": verdict}


# ---- section 8 ------------------------------------------------------------------------------------------------------------------
def section8(H, bold: dict, pp: dict, wps, env, n_vols: int, j1: int, L: float, hp: float = 0.03) -> dict:
    """Per-listener z on volumes J0..j1 (H = stillhere.heldout): LANG with TRIBE P_full, word rate, envelope; AUD with TRIBE P_audio, word
    rate, envelope; all unpartialled (zero controls), against the harness's circular shifts of the regressor."""
    subs = sorted(bold); cl = lambda A: H.clean(A, hp, H.J0, j1); tj = H.TR * np.arange(n_vols)
    reg = {"word_rate": cl(H.hrf_regressor(wps, tj - L)[None]), "envelope": cl(H.hrf_regressor(env, tj - L)[None])}
    zc = H.zeros_control(j1 - H.J0 + 1); out = {}
    for net, tribe_kind in (("LANG", "full"), ("AUD", "audio")):
        X = {"tribe": cl(pp[tribe_kind][net])}
        X.update({k: np.tile(v, (X["tribe"].shape[0], 1)) for k, v in reg.items()})
        B = {s: cl(bold[s][net]) for s in subs}
        z = {k: np.array([H.gauge(B[s], x, zc)[0] for s in subs]) for k, x in X.items()}
        out[net] = {"mean_z": {k: float(v.mean()) for k, v in z.items()},
                    "tribe_vs": {k: compare(z["tribe"], z[k]) for k in ("word_rate", "envelope")},
                    "per_listener": {k: v.tolist() for k, v in z.items()}}
    return {"label": LABEL, "L": L, "volumes": [H.J0, j1], "n": len(subs), **out}


# ---- story wall and notthefall (frozen picker functions passed in) ------------------------------------------------------------------
def _raw(G, C, r, v, w, ta, clip, tr):
    """The raw gauges for run r on volumes v, with `clip`'s TRIBE predictions at audio times w and its regressors at ta."""
    keep = C.keepmask(r["fd"], v); conf = G.confounds(r, v); wr, _, env = clip.regs(ta); out = {}
    for net, kind in (("LANG", "full"), ("AUD", "audio")):
        B = G.clean(r[net][:, v], tr, conf, keep); p = B.shape[0]
        out[net] = {"tribe": G.hearing(B, clip.pred(kind, net, w), tr, keep),
                    "word_rate": G.hearing(B, np.tile(wr[None], (p, 1)), tr, keep),
                    "envelope": G.hearing(B, np.tile(env[None], (p, 1)), tr, keep)}
    return out


def run_gauges(G, C, r, parts, clip, tr) -> dict:
    """One listener-run on its own clip, on the same scored volumes as the frozen T1/T3 (G.run_times, then clip.covered)."""
    v, w, ta = G.run_times(r, parts, tr); ok = clip.covered(w)
    return _raw(G, C, r, v[ok], w[ok], ta[ok], clip, tr)


def pair_gauges(G, C, r, parts, own, own_parts, other, other_parts, tr):
    """D19's own and wrong on the same volumes (the frozen G.pair_T1 volume rule), with the raw gauges in place of T1 (LANG only)."""
    v, w, ta = G.run_times(r, parts, tr); s0 = parts[0]["s0"]
    wo = w - s0 + own_parts[0]["s0"]; wx = w - s0 + other_parts[0]["s0"]; to = ta - s0 + own_parts[0]["s0"]; tx = ta - s0 + other_parts[0]["s0"]
    ok = G.in_story(wo, own_parts) & own.covered(wo) & G.in_story(wx, other_parts) & other.covered(wx)
    if ok.sum() < 4 * C.CFG["min_shift_trs"]:
        return None
    a = _raw(G, C, r, v[ok], wo[ok], to[ok], own, tr)["LANG"]; b = _raw(G, C, r, v[ok], wx[ok], tx[ok], other, tr)["LANG"]
    return a, b
