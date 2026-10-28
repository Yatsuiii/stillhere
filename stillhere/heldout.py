"""Held-out section scorer (PREREG §8): the stage numbers of LPP English section 8, written during the event from the frozen definitions.
12 of 215 lines follow pre-event code (language-area parcel list, lag and resampling, one alignment call); the rest was written from Oct 27.

New code that re-implements the pre-event harness behind the Oct 23 section-8 numbers (HELDOUT.md; p1/hold8.py, m1_en8.py,
m2_en8.py, m3_en8.py and their helpers noise_floor, hearing, understanding, timing): same parcels, cleaning, regressors, alignment,
gauges, circular-shift nulls, seeds and resample counts, so the two can be compared number by number. Inputs are cached pre-event data
products (TRIBE predictions from the k2 runs, BOLD parcel series, stimulus files, GPT-2 surprisal); this module takes paths and arrays
only and holds no data.

Definitions (TR 2.0 s; scored volumes j0..j1 inclusive):
- clean: per row, nilearn signal.clean with linear detrend, Butterworth high-pass at hp Hz and z-scoring (ddof 1).
- TRIBE alignment: volume j reads the prediction at stimulus time 2j - 5 s, linear between the 1 Hz samples, NaN outside them; parcel
  value = mean over its vertices. Parcels: Schaefer-400 17-network labels; AUD = SomMotB_Aud in both hemispheres (right-hemisphere
  vertices offset by 10242), LANG = left TempPar, DefaultB_Temp, DefaultB_PFCv and DefaultB_PFCl.
- stage regressors: per-second word counts or audio envelope convolved with the SPM HRF on a 0.1 s grid and read at 2j - L (L < 0
  reads a later stimulus time); word-rate derivative = r(2j - L + 0.5) - r(2j - L - 0.5). The M1-M3 harness uses word rate only,
  convolved with the 1 Hz SPM HRF and read at 2j, with no lag.
- gauge: Fisher-z mean over parcels of the partial r between each z-scored BOLD row and the matching regressor row, both residualised
  on [controls, 1]; z = (real - mean) / SD of the same score over circular shifts of the regressor by MIN..T-MIN-1 volumes.
- detection: z > 1.645."""
import collections
import difflib
import re

import numpy as np

TR, J0, MIN, FS, DETECT = 2.0, 10, 15, 10, 1.645
_HRF = {}


def _spm_hrf(dt: float) -> np.ndarray:
    if dt not in _HRF:
        from nilearn.glm.first_level import spm_hrf
        _HRF[dt] = spm_hrf(dt, oversampling=1, time_length=32.0)
    return _HRF[dt]


# ---- stimulus side -----------------------------------------------------------------------------------------------------------
def textgrid_intervals(path) -> list:
    """(text, xmin) for every non-empty interval of a Praat TextGrid (UTF-8 or UTF-16)."""
    raw = open(path, "rb").read()
    s = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8")
    return [(t.strip(), float(a)) for a, _, t in re.findall(r'xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "(.*?)"', s, flags=re.S) if t.strip()]


def word_onsets(intervals) -> np.ndarray:
    """Onsets of word intervals: pause marks "sp", "SP" and "<p>" dropped (the word-rate definition)."""
    return np.array([t0 for w, t0 in intervals if w not in ("sp", "SP", "<p>")])


def word_list(intervals) -> list:
    """(lower-cased word, onset), pause marks dropped case-insensitively (the baseline-feature definition)."""
    return [(w.lower(), t0) for w, t0 in intervals if w.lower() not in ("sp", "<p>")]


def envelope_per_second(wav) -> np.ndarray:
    """Mean absolute amplitude of the channel-mean signal in each whole second."""
    import soundfile as sf
    x, sr = sf.read(wav, dtype="float32", always_2d=True)
    m = np.abs(x.mean(1))
    return np.array([m[int(k * sr):int((k + 1) * sr)].mean() for k in range(int(len(m) / sr))])


def hrf_regressor(x_per_s, t) -> np.ndarray:
    """x_per_s[k] covers stimulus second [k, k+1); convolved with the SPM HRF on a 0.1 s grid and read at times t (s)."""
    xf = np.repeat(np.asarray(x_per_s, float), FS)
    r = np.convolve(xf, _spm_hrf(1.0 / FS))[: len(xf)] / FS
    return np.interp(np.asarray(t, float), np.arange(len(xf)) / FS + 0.5 / FS, r, left=r[0], right=r[-1])


def wordrate_1hz(onsets, n_vols: int) -> np.ndarray:
    """M1-M3 word rate: per-second counts convolved with the 1 Hz SPM HRF, read at 2j (no lag)."""
    T = int(np.ceil(n_vols * TR)) + 40
    wr = np.bincount(np.asarray(onsets).astype(int).clip(0, T - 1), minlength=T).astype(float)
    y = np.convolve(wr, _spm_hrf(1.0))[: len(wr)]
    return np.interp(np.arange(n_vols) * TR, np.arange(len(y)), y)


def per_second(vals, onsets, n_sec: int) -> np.ndarray:
    """A word feature as a per-second series: NaN filled with the mean, centred, summed into the second of each word's onset."""
    v = np.array(vals, float)
    v = np.where(np.isnan(v), np.nanmean(v), v)
    v = v - v.mean()
    x = np.zeros(n_sec)
    for t0, val in zip(onsets, v):
        x[min(int(t0), n_sec - 1)] += val
    return x


def tree_leaves(lines) -> list:
    """Leaves of bracketed parse trees: word, POS, top-down depth (brackets opened since the previous leaf) and bottom-up depth
    (brackets closed right after it, plus one)."""
    leaves = []
    for line in lines:
        toks = re.findall(r"\(|\)|[^()\s]+", line)
        opens, i = 0, 0
        while i < len(toks):
            if toks[i] != "(":
                i += 1
                continue
            opens += 1
            i += 1
            if i < len(toks) and toks[i] not in "()" and i + 1 < len(toks) and toks[i + 1] not in "()":
                leaf = {"w": toks[i + 1].lower(), "pos": toks[i], "td": opens, "bu": 1}
                i += 2
                if toks[i] == ")":
                    i += 1
                leaves.append(leaf)
                opens = 0
                while i < len(toks) and toks[i] == ")":
                    leaf["bu"] += 1
                    i += 1
            elif i < len(toks) and toks[i] not in "()":
                i += 1
    return leaves


def baseline_series(words, surprisal, tree_lines, n_sec: int) -> dict:
    """Per-second baseline regressors before HRF: log word frequency within the section, parse-tree bottom-up and top-down depth,
    GPT-2 surprisal. The tree is matched to the section at the book offset whose next 260 leaves best match the section's first 200
    words (among offsets whose first three leaves match), then word by word. Returns the series and the match ratios."""
    a = [w for w, _ in words]
    leaves = tree_leaves(tree_lines)
    lw, aw = [l["w"] for l in leaves], [w for w in a if w != "#"]
    best = (-1.0, None)
    for i in range(0, len(lw) - len(aw) + 60):
        if lw[i:i + 3] != aw[:3]:
            continue
        r = difflib.SequenceMatcher(None, aw[:200], lw[i:i + 260], autojunk=False).ratio()
        if r > best[0]:
            best = (r, i)
    sub = leaves[best[1]: best[1] + len(a) + 60]
    sm = difflib.SequenceMatcher(None, a, [l["w"] for l in sub], autojunk=False)
    feat = [None] * len(a)
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            feat[blk.a + k] = sub[blk.b + k]
    cnt = collections.Counter(a)
    t_on = [t0 for _, t0 in words]
    out = {"B_freq": per_second([np.log10(cnt[w] / len(a)) for w in a], t_on, n_sec),
           "B_bu": per_second([f["bu"] if f else np.nan for f in feat], t_on, n_sec),
           "B_td": per_second([f["td"] if f else np.nan for f in feat], t_on, n_sec),
           "surprisal": per_second(surprisal[1], surprisal[0], n_sec)}
    return out, {"tree_offset": best[1], "offset_match_first_200": best[0], "word_match": sm.ratio()}


# ---- prediction side ---------------------------------------------------------------------------------------------------------
def parcel_sets(annot_lh, annot_rh) -> dict:
    """AUD and LANG vertex index lists in TRIBE's fsaverage5 order (left 0..10241, right 10242..20483), in annotation label order."""
    import nibabel.freesurfer as fs
    lab = {}
    for h, p in (("lh", annot_lh), ("rh", annot_rh)):
        labels, _, names = fs.read_annot(p)
        lab[h] = (labels, [n.decode() for n in names])

    def pick(h, pats, off):
        labels, names = lab[h]
        return [np.where(labels == i)[0] + off for i, n in enumerate(names) if any(p in n for p in pats)]
    return {"AUD": pick("lh", ["SomMotB_Aud"], 0) + pick("rh", ["SomMotB_Aud"], 10242),
            "LANG": pick("lh", ["TempPar", "DefaultB_Temp", "DefaultB_PFCv", "DefaultB_PFCl"], 0)}


def align(P, starts, n_vols: int) -> np.ndarray:
    """(n_vols, V) float32: volume j reads stimulus time 2j - 5 s, linear between 1 Hz samples, NaN outside [starts[0], starts[-1]]."""
    starts = np.asarray(starts, float)
    t = np.arange(n_vols) * TR - 5.0
    out = np.full((n_vols, P.shape[1]), np.nan, np.float32)
    ok = (t >= starts[0]) & (t <= starts[-1])
    i0 = np.clip(np.searchsorted(starts, t[ok], side="right") - 1, 0, len(starts) - 1)
    i1 = np.minimum(i0 + 1, len(starts) - 1)
    w = (t[ok] - starts[i0])[:, None]
    out[ok] = (1 - w) * P[i0] + w * P[i1]
    return out


def pred_parcels(P, starts, n_vols: int, sets: dict) -> dict:
    al = align(P, starts, n_vols).T
    return {k: np.stack([al[v].mean(0) for v in vs]) for k, vs in sets.items()}


# ---- gauges -------------------------------------------------------------------------------------------------------------------
def clean(A, hp: float, j0: int, j1: int) -> np.ndarray:
    """Rows of A on volumes j0..j1: linear detrend, high-pass at hp Hz, z-score (ddof 1)."""
    from nilearn import signal
    return signal.clean(np.asarray(A)[:, j0:j1 + 1].T, detrend=True, high_pass=hp, t_r=TR, standardize="zscore_sample").T


def zrow(a) -> np.ndarray:
    """Rows z-scored with ddof 0."""
    return (a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True)


def _fz_rows(rb, rp) -> np.ndarray:
    """Fisher-z mean over parcels of the per-row Pearson r of rb (p, T) with rp (..., p, T)."""
    a = rb - rb.mean(-1, keepdims=True)
    b = rp - rp.mean(-1, keepdims=True)
    r = (a * b).mean(-1) / (rb.std(-1) * rp.std(-1))
    return np.arctanh(np.clip(r, -0.999, 0.999)).mean(-1)


def gauge(B, X, controls) -> tuple:
    """(z, real): the gauge of BOLD rows B (p, T) on regressor rows X (p, T) given control rows, against circular shifts of X."""
    T = B.shape[1]
    C = np.vstack(list(controls) + [np.ones((1, T))]).T
    H = C @ np.linalg.pinv(C.T @ C) @ C.T                                  # projector onto [controls, 1]
    rb = zrow(B)
    rb = rb - rb @ H
    shifts = np.r_[0, np.arange(MIN, T - MIN)]                             # 0 = the real score; np.roll(X, k, 1)[:, t] = X[:, (t - k) % T]
    Xs = np.asarray(X)[:, (np.arange(T)[None, :] - shifts[:, None]) % T].transpose(1, 0, 2)
    s = _fz_rows(rb, Xs - Xs @ H)
    return float((s[0] - s[1:].mean()) / s[1:].std()), float(s[0])


def zeros_control(T: int) -> list:
    return [np.zeros((1, T))]


def detect_pct(z) -> float:
    return float(100 * (np.asarray(z) > DETECT).mean())


# ---- stage numbers (HEARING, envelope, LANGUAGE, shuffled, baselines) -------------------------------------------------------------
def own_best_lag(aud_raw: list, env, n_vols: int, j1: int, hp: float, grid=None) -> tuple:
    """The envelope's own best lag on this section: the grid lag maximising the mean HRF x envelope HEARING-style z over the given
    listeners (the first 16 in sorted order), with the same regressor in every AUD parcel."""
    grid = np.arange(-4, 1.01, 0.25) if grid is None else grid
    tj = TR * np.arange(n_vols)
    BA = [clean(b, hp, J0, j1) for b in aud_raw]
    zz = []
    for lag in grid:
        reg = np.tile(clean(hrf_regressor(env, tj - lag)[None], hp, J0, j1), (BA[0].shape[0], 1))
        zz.append(np.mean([gauge(b, reg, zeros_control(b.shape[1]))[0] for b in BA]))
    return float(grid[int(np.argmax(zz))]), [float(x) for x in zz]


def stage_scores(bold: dict, pp: dict, wps, env, baselines: dict, n_vols: int, j1: int, L: float, hp: float = 0.03) -> dict:
    """Per-listener z on volumes J0..j1 with the regressors at lag L. bold: {listener: {"AUD", "LANG"} raw parcel series};
    pp: {"full", "audio", "shuf"} -> {"AUD", "LANG"} aligned prediction parcels."""
    from scipy import stats
    subs = sorted(bold)
    cl = lambda A: clean(A, hp, J0, j1)
    tj = TR * np.arange(n_vols)
    at = lambda x: hrf_regressor(x, tj - L)
    envr, wr = at(env), at(wps)
    dwr = hrf_regressor(wps, tj - L + 0.5) - hrf_regressor(wps, tj - L - 0.5)
    BA = {s: cl(bold[s]["AUD"]) for s in subs}
    BL = {s: cl(bold[s]["LANG"]) for s in subs}
    T = j1 - J0 + 1
    zc = zeros_control(T)
    paud = cl(pp["audio"]["AUD"])
    zh = np.array([gauge(BA[s], paud, zc)[0] for s in subs])
    ereg = np.tile(cl(envr[None]), (paud.shape[0], 1))
    ez = np.array([gauge(BA[s], ereg, zc)[0] for s in subs])
    tc = [cl(pp["audio"]["LANG"]).mean(0, keepdims=True), cl(wr[None]), cl(dwr[None])]
    pfl, psl = cl(pp["full"]["LANG"]), cl(pp["shuf"]["LANG"])
    zl = np.array([gauge(BL[s], pfl, tc)[0] for s in subs])
    zs = np.array([gauge(BL[s], psl, tc)[0] for s in subs])
    bctl = [cl(envr[None]), cl(wr[None]), cl(dwr[None])]
    base = {}
    for k, x in baselines.items():
        reg = np.tile(cl(at(x)[None]), (pfl.shape[0], 1))
        base[k] = np.array([gauge(BL[s], reg, bctl)[0] for s in subs])
    best = max(base, key=lambda k: base[k].mean())
    wil = lambda d: float(stats.wilcoxon(d, alternative="greater").pvalue)
    p2, p1, pb = wil(zl), wil(zl - zs), wil(zl - base[best])
    return {"L": L, "volumes": [J0, j1], "n": len(subs),
            "hearing": {"z": float(zh.mean()), "detect_pct": detect_pct(zh)},
            "envelope": {"z": float(ez.mean()), "detect_pct": detect_pct(ez), "p_tribe_gt_envelope": wil(zh - ez)},
            "language": {"z": float(zl.mean()), "detect_pct": detect_pct(zl), "p": p2, "pass": bool(zl.mean() >= 1 and p2 < 0.05)},
            "shuffled": {"z": float(zs.mean()), "p_real_gt_shuffled": p1, "pass": bool(p1 < 0.05)},
            "baselines": {k: float(v.mean()) for k, v in base.items()}, "best_baseline": best, "p_tribe_gt_best": pb,
            "tribe_beats_best": bool(zl.mean() > base[best].mean() and pb < 0.05),
            "per_listener": {"listeners": subs, "hearing": zh.tolist(), "envelope": ez.tolist(), "language": zl.tolist(), "shuffled": zs.tolist(),
                             **{f"base_{k}": v.tolist() for k, v in base.items()}}}


# ---- M1 control-equivalent, M2 hybrid, M3 wrong story -------------------------------------------------------------------------
def isc_curve(bold: dict, ctrl: dict, ks, reps: int, seed: int, hp: float = 0.03, j1: int = 154) -> dict:
    """M1: per listener and k, mean z (and detection share) over `reps` random sets of k other listeners (one set when k is all of
    them); the reference is the cleaned mean of their raw series, scored like TRIBE with the same controls. ctrl: {"AUD", "LANG"}
    control rows. Random sets come from one numpy Generator(seed), drawn network, listener, k, repeat in that order."""
    subs = sorted(bold)
    rng = np.random.default_rng(seed)
    cl = lambda A: clean(A, hp, J0, j1)
    out = {}
    for net in ("AUD", "LANG"):
        res = {k: [] for k in ks}
        for s in subs:
            y = cl(bold[s][net])
            others = [t for t in subs if t != s]
            for k in ks:
                zs = []
                for _ in range(1 if k == len(others) else reps):
                    pick = others if k == len(others) else list(rng.choice(others, k, replace=False))
                    zs.append(gauge(y, cl(np.mean([bold[t][net] for t in pick], 0)), ctrl[net])[0])
                res[k].append((float(np.mean(zs)), float(np.mean(np.array(zs) > DETECT))))
        out[net] = res
    return out


def hybrid(bold: dict, tribe: dict, ctrl: dict, wrong_bold: dict, wrong_tribe, ks, reps: int, seed: int, nboot: int,
           hp: float = 0.03, j1: int = 154) -> dict:
    """M2: per listener, mean z over `reps` random sets of k other listeners used alone (mean of their z-scored cleaned series) and
    with TRIBE added as one more z-scored member (weight 1/(k+1)); paired dz = hybrid - alone with a percentile bootstrap CI over
    listeners. LANGUAGE also gets the wrong-story check: controls drawn from wrong_bold listeners and wrong_tribe added. tribe:
    {"AUD", "LANG"} TRIBE rows (cleaned); wrong_tribe: LANG rows. One Generator(seed), consumed as the M2 run: per network the
    listener x k x repeat draws, then (LANGUAGE) the wrong-story draws, then the bootstraps."""
    subs = sorted(bold)
    rng = np.random.default_rng(seed)
    cl = lambda A: clean(A, hp, J0, j1)
    zc = {net: {s: zrow(cl(bold[s][net])) for s in subs} for net in ("AUD", "LANG")}
    wz = {t: zrow(cl(wrong_bold[t]["LANG"])) for t in sorted(wrong_bold)}

    def run(net, pool, pool_z, extra):
        res = {k: {"alone": [], "hybrid": []} for k in ks}
        for s in subs:
            y = cl(bold[s][net])
            cands = [t for t in pool if t != s]
            for k in ks:
                za, zh = [], []
                for _ in range(reps):
                    ctl = np.stack([pool_z[t] for t in rng.choice(cands, k, replace=False)])
                    za.append(gauge(y, ctl.mean(0), ctrl[net])[0])
                    zh.append(gauge(y, np.vstack([ctl, extra[None]]).mean(0), ctrl[net])[0])
                res[k]["alone"].append(float(np.mean(za)))
                res[k]["hybrid"].append(float(np.mean(zh)))
        return res
    out = {}
    for net in ("AUD", "LANG"):
        res = run(net, subs, zc[net], zrow(tribe[net]))
        if net == "LANG":
            fp = run("LANG", sorted(wrong_bold), wz, zrow(wrong_tribe))
            out["wrong_story"] = {k: {"alone_z": float(np.mean(v["alone"])), "alone_detect_pct": detect_pct(v["alone"]),
                                      "hybrid_z": float(np.mean(v["hybrid"])), "hybrid_detect_pct": detect_pct(v["hybrid"])} for k, v in fp.items()}
        rows = {}
        for k in ks:
            a, h = np.array(res[k]["alone"]), np.array(res[k]["hybrid"])
            d = h - a
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(nboot)]
            rows[k] = {"alone_z": float(a.mean()), "alone_detect_pct": detect_pct(a), "hybrid_z": float(h.mean()), "hybrid_detect_pct": detect_pct(h),
                       "dz": float(d.mean()), "dz_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
        out[net] = rows
    return out


def wrong_story_alone(bold: dict, own: dict, wrong: dict, ctrl: dict, wrong_group: dict, hp: float = 0.03, j1: int = 154) -> dict:
    """M3 (fix 4): TRIBE alone, own vs wrong-story prediction (own and wrong: {"AUD", "LANG"} cleaned rows), and the ISC analogue:
    leave-one-out own-story template vs the wrong story's group-mean BOLD (wrong_group: {"AUD", "LANG"} cleaned rows)."""
    subs = sorted(bold)
    cl = lambda A: clean(A, hp, J0, j1)
    out = {}
    for net in ("AUD", "LANG"):
        ys = {s: cl(bold[s][net]) for s in subs}
        zo = np.array([gauge(ys[s], own[net], ctrl[net])[0] for s in subs])
        zw = np.array([gauge(ys[s], wrong[net], ctrl[net])[0] for s in subs])
        io = np.array([gauge(ys[s], cl(np.mean([bold[t][net] for t in subs if t != s], 0)), ctrl[net])[0] for s in subs])
        iw = np.array([gauge(ys[s], wrong_group[net], ctrl[net])[0] for s in subs])
        out[net] = {"n": len(subs), "tribe_own_z": float(zo.mean()), "tribe_own_detect_pct": detect_pct(zo), "tribe_wrong_z": float(zw.mean()),
                    "tribe_wrong_detect_pct": detect_pct(zw), "tribe_wrong_detected": int((zw > DETECT).sum()),
                    "isc_own_z": float(io.mean()), "isc_own_detect_pct": detect_pct(io), "isc_wrong_z": float(iw.mean()),
                    "isc_wrong_detect_pct": detect_pct(iw), "isc_wrong_detected": int((iw > DETECT).sum())}
    return out


def volunteer_equivalent(tribe_z, ks, curve, nboot: int = 10_000, seed: int = 20261028) -> dict:
    """Same-clip controls TRIBE alone equals: its mean z (and the lower end of the listener-bootstrap 95% CI of that mean) placed on
    the ISC(k) mean-z curve by linear interpolation (below the curve's first point -> 0, above its last -> the last k); the claim uses
    the lower end, rounded down."""
    z = np.asarray(tribe_z, float)
    rng = np.random.default_rng(seed)
    bs = np.array([z[rng.integers(0, len(z), len(z))].mean() for _ in range(nboot)])
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    def place(v):
        if v < curve[0]:
            return 0.0
        if v >= curve[-1]:
            return float(ks[-1])
        return float(np.interp(v, curve, ks))
    return {"z": float(z.mean()), "ci95": [lo, hi], "point_k": place(z.mean()), "lower_k": place(lo), "claim_k": int(np.floor(place(lo)))}
