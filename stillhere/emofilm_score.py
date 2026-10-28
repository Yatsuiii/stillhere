"""Emo-FilM scorer (StillHere Pilot, exploratory): TRIBE v2's forecast for Tears of Steel against the Emo-FilM viewers (ds004892).

The rule is docs/rules/emofilm_rule.md, committed with this code before any Emo-FilM brain data was read. This module takes arrays and
paths only and holds no data. It reuses the held-out section scorer's gauge, null and volunteer-equivalent rule (stillhere.heldout),
with the Emo-FilM TR (1.3 s from the sidecar) in place of LPP's 2.0 s.

Per viewer: scored volumes = film time (j x TR + slice offset - film onset) from DROP_S to the end of the cut, with TRIBE's sample
(film time - TRIBE_OFFSET) inside the prediction; FD (Power) censoring and the D7 run rule; detrend + high-pass 0.03 Hz + z-score;
HEARING, LANGUAGE, VISUAL gauges against circular shifts of the prediction. Group: median z with a viewer bootstrap CI, share detected,
and ISC(k) on a common film-time grid with TRIBE placed on that curve."""
import numpy as np

from . import heldout as H

CUT_S, DROP_S, TRIBE_OFFSET, FD_MAX, FD_SHARE, HP = 588.0, 10.0, 5.0, 0.5, 0.30, 0.03
SEED, NBOOT, ISC_KS, ISC_REPS = 20261028, 10_000, tuple(range(1, 16)), 20
LABEL = "exploratory"


# ---- timing -------------------------------------------------------------------------------------------------------------------
def film_volumes(n_vols: int, tr: float, onset: float, slice_offset: float, n_pred_s: int) -> tuple:
    """(volume indices, film times, TRIBE sample times) of the scored volumes: film time in [DROP_S, CUT_S] and the TRIBE sample
    (film time - TRIBE_OFFSET) inside the 1 Hz prediction [0, n_pred_s - 1]."""
    j = np.arange(n_vols); ft = j * tr + slice_offset - onset; ts = ft - TRIBE_OFFSET
    ok = (ft >= DROP_S) & (ft <= CUT_S) & (ts >= 0) & (ts <= n_pred_s - 1)
    return j[ok], ft[ok], ts[ok]


def tribe_at(P, t) -> np.ndarray:
    """P (n_s, V) at 1 Hz, read at times t (s) with linear interpolation -> (V, len(t))."""
    t = np.asarray(t, float); i0 = np.clip(np.floor(t).astype(int), 0, len(P) - 1); i1 = np.minimum(i0 + 1, len(P) - 1)
    w = (t - i0)[None, :]
    return ((1 - w) * P[i0].T + w * P[i1].T).astype(float)


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
    """One viewer. bold: {AUD, LANG[, VIS]} parcel rows on the scored volumes; pred: {"audio": {AUD, LANG}, "full": {LANG[, VIS]}} on
    the same volumes; wr: HRF x word rate on the same volumes; keep: uncensored frames. Gauges cleaned on the kept frames."""
    c = lambda a: clean(a, tr, keep)
    out = {"HEARING": gauge_null(c(bold["AUD"]), c(pred["audio"]["AUD"]), H.zeros_control(int(np.sum(keep)))),
           "LANGUAGE": gauge_null(c(bold["LANG"]), c(pred["full"]["LANG"]),
                                  [c(pred["audio"]["LANG"].mean(0, keepdims=True)), c(np.atleast_2d(wr))])}
    out["VISUAL"] = gauge_null(c(bold["VIS"]), c(pred["full"]["VIS"]), H.zeros_control(int(np.sum(keep)))) if visual else "not run"
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
