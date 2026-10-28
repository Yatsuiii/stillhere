"""Emo-FilM scorer (StillHere Pilot, exploratory): TRIBE v2's forecast for Tears of Steel against the Emo-FilM viewers (ds004892).

The rule is docs/rules/emofilm_rule.md (version 2: the opening three minutes of the cut), committed with this code before any Emo-FilM
brain data was read. This module takes arrays and paths only and holds no data. It reuses the held-out section scorer's gauge, null,
HRF regressor and volunteer-equivalent rule (stillhere.heldout) and the cheap-baselines comparison (stillhere.cheap_baselines), with
the Emo-FilM TR (1.3 s from the sidecar) in place of LPP's 2.0 s.

Per run: the volume-count check (EXPECTED_VOLS +- VOL_TOL); film time 0 = the run's own logged film onset (film_onset; the run opens
with 90 s of fixation); scored volumes = film time (j x TR + slice offset - film onset) of at least DROP_S, with TRIBE's sample (film
time - TRIBE_OFFSET, a fixed offset: no lag search) inside OPENING_S and inside the prediction; FD (Power) censoring and the D7 run
rule; the BOLD cleaned (detrend + high-pass 0.03 Hz + z-score) over the whole run, then cropped to the scored volumes
(clean_full_then_crop); HEARING, LANGUAGE, VISUAL gauges and the unpartialled TRIBE and baseline gauges against circular shifts of the
regressor. Group: median z with a viewer bootstrap CI, share detected, TRIBE minus each baseline paired over viewers, and ISC(k) on a
common film-time grid with TRIBE placed on that curve."""
import numpy as np

from . import cheap_baselines as CB
from . import heldout as H

DROP_S, TRIBE_OFFSET, FD_MAX, FD_SHARE, HP = 10.0, 5.0, 0.5, 0.30, 0.03
OPENING_S = (0.0, 180.0)        # release seconds = film time (the cut starts at release 0): the first three minutes, the scored window
BRIDGE_S = (25.0, 40.25)        # the bridge conversation: SAM 3.1 edits and the video only, never scored (11 or 12 volumes)
BASELINES = ("word_rate", "envelope", "brightness", "motion")
MATCHED = (("LANG", "word_rate"), ("AUD", "envelope"), ("VIS", "brightness"), ("VIS", "motion"))    # the primary comparisons
SEED, NBOOT, ISC_KS, ISC_REPS = 20261028, 10_000, tuple(range(1, 16)), 20
EXPECTED_VOLS, VOL_TOL = 607, 2  # Tears of Steel scan length in TRs, washouts included (data descriptor Table 1); runs off by > 2 drop
LABEL = "exploratory"


# ---- run selection ------------------------------------------------------------------------------------------------------------
def film_onset(rows) -> float:
    """Film time 0 of one run: the onset (s from the run's first volume) of the single 'film' row of its events file (rows of dicts
    with onset, duration and trial_type: rest, film, rest)."""
    films = [r for r in rows if r["trial_type"] == "film"]
    if len(films) != 1:
        raise ValueError(f"{len(films)} film rows in the events file (need exactly 1)")
    return float(films[0]["onset"])


def volume_check(n_vols: int, expected: int = EXPECTED_VOLS, tol: int = VOL_TOL) -> dict:
    """The run's volume count against the designed scan length: kept only if it is off by at most `tol` TRs."""
    off = int(n_vols) - int(expected)
    return {"n_vols": int(n_vols), "expected": int(expected), "off_by": off, "keep": abs(off) <= tol}


# ---- timing -------------------------------------------------------------------------------------------------------------------
def film_volumes(n_vols: int, tr: float, onset: float, slice_offset: float, starts, win=OPENING_S) -> tuple:
    """(volume indices, film times, TRIBE sample times) of the scored volumes: film time at least DROP_S, and the TRIBE sample (film
    time - TRIBE_OFFSET) inside the window `win` and inside the prediction's sample times [starts[0], starts[-1]]."""
    j = np.arange(n_vols); ft = j * tr + slice_offset - onset; ts = ft - TRIBE_OFFSET
    st = np.asarray(starts, float)
    ok = (ft >= DROP_S) & (ts >= win[0]) & (ts <= win[1]) & (ts >= st[0]) & (ts <= st[-1])
    return j[ok], ft[ok], ts[ok]


def tribe_at(P, starts, t) -> np.ndarray:
    """P (n, V), row i predicting stimulus time starts[i] (s, 1 Hz), read at times t inside [starts[0], starts[-1]] with linear
    interpolation (as heldout.align) -> (V, len(t))."""
    st = np.asarray(starts, float); t = np.asarray(t, float)
    i0 = np.clip(np.searchsorted(st, t, side="right") - 1, 0, len(st) - 1); i1 = np.minimum(i0 + 1, len(st) - 1)
    w = np.where(i1 > i0, (t - st[i0]) / np.where(i1 > i0, st[i1] - st[i0], 1.0), 0.0)[None, :]
    return ((1 - w) * np.asarray(P)[i0].T + w * np.asarray(P)[i1].T).astype(float)


def common_grid(tr: float, win=OPENING_S) -> np.ndarray:
    """The ISC grid: film times from DROP_S to the window's last TRIBE sample + TRIBE_OFFSET, in steps of one TR."""
    return np.arange(DROP_S, win[1] + TRIBE_OFFSET + 1e-9, tr)


def parcel_means(vert_ts, sets: dict) -> dict:
    """vert_ts (V, T) -> {name: (n_parcels, T)}, each parcel the mean of its vertices."""
    return {k: np.stack([np.asarray(vert_ts)[v].mean(0) for v in vs]) for k, vs in sets.items()}


def visual_sets(annot_lh, annot_rh) -> list:
    """VisCent and VisPeri parcels (both hemispheres) in TRIBE's fsaverage5 vertex order (right hemisphere offset 10242)."""
    import nibabel.freesurfer as fs
    out = []
    for p, off in ((annot_lh, 0), (annot_rh, 10242)):
        labels, _, names = fs.read_annot(p)
        out += [np.where(labels == i)[0] + off for i, n in enumerate(names) if any(s in n.decode() for s in ("VisCent", "VisPeri"))]
    return out


def project(bold_img, meshes: dict) -> np.ndarray:
    """MNI volume -> fsaverage5 (left then right) exactly as the LPP harness: vol_to_surf, ball radius 3 mm, linear, mid meshes.
    meshes = {"left": (mid, inner), "right": (mid, inner)}."""
    from nilearn import surface
    return np.vstack([surface.vol_to_surf(bold_img, surf_mesh=meshes[h][0], inner_mesh=meshes[h][1], radius=3.0, kind="ball",
                                          interpolation="linear") for h in ("left", "right")])


# ---- motion -------------------------------------------------------------------------------------------------------------------
def fd_power(params) -> np.ndarray:
    """Power FD from MCFLIRT parameters (n, 6): rotations (rad) x, y, z then translations (mm) x, y, z; rotations x 50 mm."""
    p = np.asarray(params, float); d = np.abs(np.diff(p, axis=0)); d[:, :3] *= 50.0
    return np.r_[0.0, d.sum(1)]


def run_rows(session_counts: list, run_index: int) -> slice:
    """Rows of one run inside a session motion file that concatenates the session's runs in acquisition order."""
    start = int(np.sum(session_counts[:run_index])); return slice(start, start + int(session_counts[run_index]))


def run_pass(fd_scored) -> bool:
    """D7: mean FD <= 0.5 mm and at most 30% of scored frames with FD > 0.5 mm."""
    f = np.asarray(fd_scored, float)
    return bool(f.mean() <= FD_MAX and (f > FD_MAX).mean() <= FD_SHARE)


# ---- cleaning and gauges ------------------------------------------------------------------------------------------------------
def clean(A, tr: float, keep=None) -> np.ndarray:
    """Rows of A (p, T): detrend, Butterworth high-pass HP Hz, z-score (ddof 1), on the kept frames only -> (p, n_kept)."""
    from nilearn import signal
    A = np.asarray(A, float); sm = None if keep is None or np.all(keep) else np.where(keep)[0]
    return signal.clean(A.T, detrend=True, high_pass=HP, t_r=tr, standardize="zscore_sample", sample_mask=sm).T


def clean_full_then_crop(A_full, tr: float, keep_full, idx) -> tuple:
    """BOLD rows of the whole run (p, n_vols), washouts included, cleaned on the run's uncensored frames (`clean`), then cropped to the
    scored volumes `idx` (film_volumes). Returns (cleaned rows at the scored uncensored volumes (p, n), mask over idx of those volumes)."""
    keep_full = np.asarray(keep_full, bool); idx = np.asarray(idx)
    C = clean(A_full, tr, keep_full); pos = np.cumsum(keep_full) - 1; k = keep_full[idx]
    return C[:, pos[idx[k]]], k


def gauge_null(B, X, controls) -> dict:
    """The section-8 gauge (stillhere.heldout.gauge) with its null kept: z, real score, and detection = real above the 95th
    percentile of the circular-shift null (shifts of the regressor rows by heldout.MIN .. T - MIN - 1 frames)."""
    T = B.shape[1]
    C = np.vstack(list(controls) + [np.ones((1, T))]).T; Hm = C @ np.linalg.pinv(C.T @ C) @ C.T
    rb = H.zrow(B); rb = rb - rb @ Hm
    shifts = np.r_[0, np.arange(H.MIN, T - H.MIN)]
    Xs = np.asarray(X)[:, (np.arange(T)[None, :] - shifts[:, None]) % T].transpose(1, 0, 2)
    s = H._fz_rows(rb, Xs - Xs @ Hm); null = s[1:]
    z, real = float((s[0] - null.mean()) / null.std()), float(s[0])
    assert np.isclose(z, H.gauge(B, X, controls)[0])                       # same numbers as the section-8 gauge
    return {"z": z, "real": real, "detected": bool(real > np.percentile(null, 95))}


def viewer_scores(bold: dict, pred: dict, wr, keep, tr: float, visual: bool) -> dict:
    """One viewer. bold: {AUD, LANG[, VIS]} parcel rows already cleaned over the whole run and cropped (clean_full_then_crop), at the
    scored uncensored volumes; pred: {"audio": {AUD, LANG}, "full": {LANG[, VIS]}} and wr (HRF x word rate) on all scored volumes;
    keep: the uncensored scored volumes. The predictions and controls exist only inside the forecast window, so they are cleaned on the
    kept scored frames."""
    c = lambda a: clean(a, tr, keep)
    out = {"HEARING": gauge_null(bold["AUD"], c(pred["audio"]["AUD"]), H.zeros_control(int(np.sum(keep)))),
           "LANGUAGE": gauge_null(bold["LANG"], c(pred["full"]["LANG"]),
                                  [c(pred["audio"]["LANG"].mean(0, keepdims=True)), c(np.atleast_2d(wr))])}
    out["VISUAL"] = gauge_null(bold["VIS"], c(pred["full"]["VIS"]), H.zeros_control(int(np.sum(keep)))) if visual else "not run"
    return out


# ---- group --------------------------------------------------------------------------------------------------------------------
def group(zs, nboot: int = NBOOT, seed: int = SEED) -> dict:
    """Median z over viewers, its viewer bootstrap 95% CI, and whether the CI excludes 0 (the rule's only criterion)."""
    z = np.asarray(zs, float); rng = np.random.default_rng(seed)
    bs = np.array([np.median(z[rng.integers(0, len(z), len(z))]) for _ in range(nboot)])
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    return {"label": LABEL, "n": int(len(z)), "median_z": float(np.median(z)), "ci95": [lo, hi], "ci_excludes_0": bool(lo > 0 or hi < 0)}


def to_grid(series, film_t, keep, grid) -> np.ndarray:
    """Cleaned parcel rows (p, n_kept) at their film times -> the common grid by linear interpolation; NaN outside the kept span."""
    t = np.asarray(film_t, float)[np.asarray(keep, bool)]
    return np.stack([np.interp(grid, t, r, left=np.nan, right=np.nan) for r in np.asarray(series)])


def isc_curve(grid_series: dict, controls: list, ks=ISC_KS, reps: int = ISC_REPS, seed: int = SEED) -> dict:
    """grid_series: {viewer: (p, G)} on the common grid (NaN where missing); controls: rows on the grid (empty for HEARING). For each
    k, `reps` random sets of k other viewers per viewer; template = their nanmean; gauge z on frames valid in both; mean z per k."""
    rng = np.random.default_rng(seed); names = sorted(grid_series); curve = []
    for k in ks:
        zs = []
        for v in names:
            others = [o for o in names if o != v]
            for _ in range(reps):
                pick = rng.choice(len(others), size=min(k, len(others)), replace=False)
                tmpl = np.nanmean(np.stack([grid_series[others[i]] for i in pick]), 0); y = grid_series[v]
                ok = np.isfinite(y).all(0) & np.isfinite(tmpl).all(0)
                ctl = [np.atleast_2d(c)[:, ok] for c in controls] or H.zeros_control(int(ok.sum()))
                zs.append(gauge_null(y[:, ok], tmpl[:, ok], ctl)["z"])
        curve.append(float(np.mean(zs)))
    return {"ks": list(ks), "mean_z": curve}


def volunteer_equivalent(tribe_z, curve: dict) -> dict:
    """The section-8 rule (stillhere.heldout.volunteer_equivalent) on this curve, labelled exploratory."""
    return {"label": LABEL, **H.volunteer_equivalent(tribe_z, curve["ks"], curve["mean_z"])}


# ---- baselines (docs/rules/cheap_baselines_rule.md conventions) ---------------------------------------------------------------
def baseline_regressors(per_s: dict, ft) -> dict:
    """{baseline: per-second series covering stimulus second [k, k+1)} -> {baseline: HRF x series read at film times ft} (the SPM HRF
    on a 0.1 s grid, heldout.hrf_regressor; no lag, as the base rule's word rate)."""
    return {b: H.hrf_regressor(per_s[b], ft) for b in BASELINES}


def word_rate_per_s(onsets, n_s: int) -> np.ndarray:
    """Per-second word onset counts (onsets in film seconds) over [0, n_s)."""
    on = np.asarray(onsets, float); on = on[(on >= 0) & (on < n_s)]
    return np.bincount(on.astype(int), minlength=n_s).astype(float)


def baseline_gauges(bold: dict, pred: dict, regs: dict, keep, tr: float, visual: bool) -> dict:
    """One viewer, the cheap-baselines gauges: raw (unpartialled) gauge z against circular shifts for TRIBE and each baseline, on LANG
    (TRIBE full), AUD (TRIBE audio-only) and VIS (TRIBE full; only if the video path ran). bold: parcel rows cleaned over the whole run
    and cropped (clean_full_then_crop); pred: {"audio": {AUD, LANG}, "full": {LANG[, VIS]}} and regs: {baseline: one row}, on all scored
    volumes, cleaned here on the kept frames exactly as the TRIBE gauges; one baseline row is tiled over the parcels."""
    c = lambda a: clean(a, tr, keep); zc = H.zeros_control(int(np.sum(keep))); out = {}
    for net, kind in (("LANG", "full"), ("AUD", "audio"), ("VIS", "full")):
        if net == "VIS" and not visual:
            out[net] = "not run"; continue
        B = np.asarray(bold[net], float); X = {"tribe": c(pred[kind][net])}
        X.update({b: np.tile(c(np.atleast_2d(regs[b])), (B.shape[0], 1)) for b in BASELINES})
        out[net] = {k: gauge_null(B, x, zc)["z"] for k, x in X.items()}
    return out


def compare_baselines(per_viewer: list) -> dict:
    """TRIBE minus each baseline, paired over viewers (cheap_baselines.compare: n, k, Hodges-Lehmann with a 95% viewer bootstrap,
    two-sided Wilcoxon p, verdict), per network. The MATCHED pairs are the primary comparisons; the rest are reported beside them."""
    out = {"label": LABEL, "primary": [f"{n}:{b}" for n, b in MATCHED]}
    for net in ("LANG", "AUD", "VIS"):
        rows = [v[net] for v in per_viewer if isinstance(v.get(net), dict)]
        if not rows:
            out[net] = "not run"; continue
        out[net] = {b: CB.compare([r["tribe"] for r in rows], [r[b] for r in rows]) for b in BASELINES}
    return out


def frame_stats(frames) -> tuple:
    """Per-frame mean Rec. 709 luma (brightness) and mean absolute luma difference from the previous frame (motion; 0 for the first),
    for frames (n, h, w, 3) uint8 at the release's native 24 fps and resolution, the frames TRIBE's video path samples from."""
    f = np.asarray(frames, float); y = 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]
    bright = y.reshape(len(y), -1).mean(1)
    motion = np.r_[0.0, np.abs(np.diff(y, axis=0)).reshape(len(y) - 1, -1).mean(1)]
    return bright, motion


def per_second(x, fps: float) -> np.ndarray:
    """Mean of a per-frame series in each whole second (the 1 Hz grid the HRF regressors use)."""
    x = np.asarray(x, float); n = int(len(x) // fps)
    return np.array([x[int(round(i * fps)):int(round((i + 1) * fps))].mean() for i in range(n)])
