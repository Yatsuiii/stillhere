"""Face map (docs/rules/face_map_rule.md; exploratory): TRIBE v2's forecast of a SAM 3.1 face edit (R = intact - faces removed; C =
intact - area matched, per Schaefer-400 parcel) against where Emo-FilM viewers' BOLD follows the face-area fraction on screen (M_i: the
Fisher z partial correlation with F, controlling brightness, motion, envelope and word rate). Gate 0 (collinearity, stimulus only),
gate 1 (leave-one-out reliability), then B1 and B2 on the 374 parcels outside the project's auditory and language sets, each needing an
exact one-sided Wilcoxon, the binomial count k*(N) and a spin permutation. Arrays and paths in, numbers out; holds no data."""
import numpy as np

from . import emofilm_score as E
from . import heldout as H

FRAME_PX, ALPHA, VIF_MAX, N_SPIN, SPIN_SEED = 1280 * 534, 0.05, 5.0, 1000, 20261029
OUTCOMES = ("NOT RUN", "UNINFORMATIVE", "FAIL", "FAIL not face-specific", "PASS")
LABEL = "exploratory"


# ---- parcels ------------------------------------------------------------------------------------------------------------------
def all_parcels(annot_lh, annot_rh) -> dict:
    """The 400 Schaefer parcels in annotation order (left 1..200, then right with the 10242 vertex offset): vertex lists, hemisphere,
    the tested mask (outside the project's AUD and LANG sets, the patterns of heldout.parcel_sets) and the visual-network mask
    (VisCent, VisPeri; emofilm_score.visual_sets)."""
    import nibabel.freesurfer as fs
    verts, hemi, names = [], [], []
    for h, (p, off) in enumerate(((annot_lh, 0), (annot_rh, 10242))):
        labels, _, nm = fs.read_annot(p)
        for i, n in enumerate(nm):
            if i == 0 or not (labels == i).any():
                continue                                               # the medial-wall background
            verts.append(np.where(labels == i)[0] + off); hemi.append(h); names.append(n.decode())
    hemi = np.array(hemi)
    aud = np.array(["SomMotB_Aud" in n for n in names])
    lang = np.array([h == 0 and any(s in n for s in ("TempPar", "DefaultB_Temp", "DefaultB_PFCv", "DefaultB_PFCl")) for n, h in zip(names, hemi)])
    vis = np.array([any(s in n for s in ("VisCent", "VisPeri")) for n in names])
    return {"vertices": verts, "hemi": hemi, "names": names, "tested": ~(aud | lang), "visual": vis}


# ---- regressors and maps ------------------------------------------------------------------------------------------------------
def face_fraction(masked_px_per_frame) -> np.ndarray:
    """Face-area fraction per release frame: the sum of the frame's face-mask pixels over the release frame area."""
    return np.array([float(np.sum(px)) for px in masked_px_per_frame]) / FRAME_PX


def face_per_s(masked_px_per_frame, fps: float = 24.0) -> np.ndarray:
    """The brightness code path, first half: the per-frame series averaged per second (emofilm_score.per_second)."""
    return E.per_second(face_fraction(masked_px_per_frame), fps)


def face_regressor(per_s, ft) -> np.ndarray:
    """The brightness code path, second half: the SPM HRF regressor read at film time (heldout.hrf_regressor, no lag)."""
    return H.hrf_regressor(per_s, ft)


def forecast_map(P_a, P_b, starts, parcels: dict, tr: float, keep=None) -> np.ndarray:
    """Per parcel, the mean over the intact rule's common grid (or the grid points in `keep`) of (forecast a - forecast b), each read
    at film time - TRIBE_OFFSET."""
    grid = E.common_grid(tr); t = grid - E.TRIBE_OFFSET
    if keep is not None:
        t = t[np.asarray(keep, bool)]
    D = E.tribe_at(P_a, starts, t) - E.tribe_at(P_b, starts, t)                         # (V, G)
    return np.array([D[v].mean(0).mean() for v in parcels["vertices"]])


def frames_fully_controlled(faces_per_frame) -> np.ndarray:
    """Per frame, True when every face with pixels in TRIBE's view has a placed control (edits_v2.json: px_in_view, control_px)."""
    return np.array([all(f["control_px"] > 0 for f in faces if f["px_in_view"] > 0) for faces in faces_per_frame])


def controlled_grid(frame_ok, tr: float, fps: float = 24.0, clip_s: float = 4.0) -> np.ndarray:
    """The descriptive's grid points: the prediction is read at film time - TRIBE_OFFSET, and TRIBE's video stage encodes the clip_s
    seconds ending there (docs/rules/film_stimulus_rule.md), so a point is kept when every frame in (s - clip_s, s] is fully controlled."""
    ok = np.asarray(frame_ok, bool); keep = []
    for s in E.common_grid(tr) - E.TRIBE_OFFSET:
        i = np.arange(int(np.floor((s - clip_s) * fps)) + 1, int(np.floor(s * fps)) + 1); i = i[(i >= 0) & (i < len(ok))]
        keep.append(bool(len(i)) and bool(ok[i].all()))
    return np.array(keep)


def partial_fz(B, x, controls) -> np.ndarray:
    """Fisher z of the partial correlation of each row of B (p, T) with x (1, T), given the control rows and an intercept."""
    B = np.asarray(B, float); x = np.atleast_2d(np.asarray(x, float)); T = B.shape[1]
    C = np.vstack([np.atleast_2d(c) for c in controls] + [np.ones((1, T))]).T; Hm = C @ np.linalg.pinv(C.T @ C) @ C.T
    rb = B - B @ Hm; rx = x - x @ Hm
    if rx.std() < 1e-12 * max(1.0, np.abs(x).max()):
        raise ValueError("the regressor has no variance left after the controls")
    rb = (rb - rb.mean(1, keepdims=True)) / rb.std(1, keepdims=True); rx = (rx - rx.mean()) / rx.std()
    r = (rb * rx).mean(1)
    return np.arctanh(np.clip(r, -0.999999, 0.999999))


def degenerate_rows(A) -> np.ndarray:
    """Per row of A (p, T): True when it has a non-finite value or no variance (SD below 1e-12 of its largest magnitude, partial_fz's
    test for the regressor). partial_fz divides each BOLD row by its SD, so one such row gives a NaN M_i and gate 1 fails for good:
    the runners refuse on any (never drop one)."""
    A = np.asarray(A, float)
    with np.errstate(invalid="ignore"):
        return ~np.isfinite(A).all(1) | ~(A.std(1) >= 1e-12 * np.maximum(1.0, np.abs(A).max(1)))


def viewer_map(B_all, F_row, controls) -> np.ndarray:
    """M_i over all 400 parcels: B_all (400, n_kept) cleaned BOLD rows; F_row and controls cleaned on the same kept frames."""
    return partial_fz(B_all, F_row, controls)


# ---- statistics ---------------------------------------------------------------------------------------------------------------
def kstar(n: int, alpha: float = ALPHA) -> int:
    """The smallest k with P(X >= k) < alpha for X ~ Binomial(n, 0.5)."""
    from scipy import stats
    return next(k for k in range(n + 1) if stats.binom.sf(k - 1, n, 0.5) < alpha)


def wilcoxon_greater(x) -> float:
    """One-sided exact Wilcoxon signed-rank p (greater than 0)."""
    from scipy import stats
    x = np.asarray(x, float)
    return float(stats.wilcoxon(x, alternative="greater", method="exact").pvalue)


def spearman(a, b) -> float:
    from scipy import stats
    return float(stats.spearmanr(a, b).statistic)


def centroids(sphere_lh, sphere_rh, parcels: dict) -> np.ndarray:
    """Each parcel's normalised mean sphere coordinate, in its own hemisphere's sphere (vertex indices without the 10242 offset)."""
    out = []
    for v, h in zip(parcels["vertices"], parcels["hemi"]):
        xyz = (sphere_lh if h == 0 else sphere_rh)[v - 10242 * h].mean(0); out.append(xyz / np.linalg.norm(xyz))
    return np.array(out)


def spin_perms(cents, hemi, n: int = N_SPIN, seed: int = SPIN_SEED) -> np.ndarray:
    """n permutations of the parcels: a uniform random rotation for the left hemisphere, its mirror for the right, rotated centroids
    matched one to one to the original ones within each hemisphere (Hungarian). Row r maps parcel j to parcel perms[r, j]."""
    from scipy.optimize import linear_sum_assignment
    rng = np.random.default_rng(seed); mirror = np.diag([-1.0, 1.0, 1.0]); hemi = np.asarray(hemi); out = np.empty((n, len(cents)), int)
    for r in range(n):
        q, rr = np.linalg.qr(rng.standard_normal((3, 3))); q = q @ np.diag(np.sign(np.diag(rr)))
        if np.linalg.det(q) < 0:
            q[:, 0] = -q[:, 0]
        for h, rot in ((0, q), (1, mirror @ q @ mirror)):
            idx = np.where(hemi == h)[0]; c = cents[idx]
            _, col = linear_sum_assignment(-(c @ rot.T) @ c.T)                           # rotated centroid i takes original slot col[i]
            out[r, idx[col]] = idx
    return out


def spin_p(observed: float, stat, perms) -> float:
    """p = (1 + rotations at or above the observed statistic) / (1 + rotations); stat(perm) gives the statistic for one rotation."""
    null = np.array([stat(p) for p in perms])
    return float((1 + np.sum(null >= observed)) / (1 + len(perms)))


def criteria(vals, spin: float) -> dict:
    v = np.asarray(vals, float); n = len(v); k = int((v > 0).sum()); ks = kstar(n); p = wilcoxon_greater(v)
    return {"n": n, "k_above_0": k, "kstar": ks, "wilcoxon_p": p, "spin_p": spin, "median": float(np.median(v)),
            "pass": bool(p < ALPHA and k >= ks and spin < ALPHA)}


def reliability(M, tested) -> dict:
    """Gate 1: per viewer, rho across the tested parcels between M_i and the mean of the other viewers' maps; exact one-sided Wilcoxon."""
    M = np.asarray(M, float)[:, tested]; n = len(M)
    rhos = [spearman(M[i], M[np.arange(n) != i].mean(0)) for i in range(n)]
    p = wilcoxon_greater(rhos)
    return {"rhos_median": float(np.median(rhos)), "wilcoxon_p": p, "pass": bool(p < ALPHA)}


def evaluate(R, C, M, parcels: dict, perms) -> dict:
    """Gate 1, B1, B2 and the outcome (the first of UNINFORMATIVE, FAIL, FAIL not face-specific, PASS that applies; NOT RUN is decided
    before any BOLD is read). M: (viewers, 400). On a PASS, the lead check: d_i's median on the tested parcels outside the visual network."""
    R, C, M = (np.asarray(a, float) for a in (R, C, M)); S = np.asarray(parcels["tested"], bool)
    out = {"label": LABEL, "viewers": int(len(M)), "parcels_tested": int(S.sum()), "gate1": reliability(M, S)}
    if not out["gate1"]["pass"]:
        return {**out, "outcome": "UNINFORMATIVE"}
    rR = np.array([spearman(R[S], m[S]) for m in M]); rC = np.array([spearman(C[S], m[S]) for m in M]); d = rR - rC
    Mbar = M.mean(0)
    s1 = lambda p: spearman(R[p][S], Mbar[S])
    s2 = lambda p: spearman(R[p][S], Mbar[S]) - spearman(C[p][S], Mbar[S])
    ident = np.arange(len(R))
    out["B1"] = criteria(rR, spin_p(s1(ident), s1, perms))
    out["B2"] = criteria(d, spin_p(s2(ident), s2, perms))
    if not out["B1"]["pass"]:
        return {**out, "outcome": "FAIL"}
    if not out["B2"]["pass"]:
        return {**out, "outcome": "FAIL not face-specific"}
    med, lead = lead_check(R, C, M, parcels)
    return {**out, "outcome": "PASS", "B2_median_outside_visual": med, "film_leads_video": lead,
            "caption": None if lead else "carried by visual-network parcels"}


def lead_check(R, C, M, parcels: dict) -> tuple:
    """On a PASS: the median over viewers of d_i on the tested parcels outside the visual network, and whether it is above 0."""
    nv = np.asarray(parcels["tested"], bool) & ~np.asarray(parcels["visual"], bool)
    d = np.array([spearman(R[nv], m[nv]) - spearman(C[nv], m[nv]) for m in np.asarray(M, float)])
    return float(np.median(d)), bool(np.median(d) > 0)


# ---- gate 0 -------------------------------------------------------------------------------------------------------------------
def vif(F, others) -> float:
    """Variance inflation factor of F against the other regressors (each a row), with an intercept."""
    F = np.asarray(F, float); X = np.vstack([np.atleast_2d(o) for o in others] + [np.ones((1, len(F)))]).T
    fit = X @ np.linalg.lstsq(X, F, rcond=None)[0]; r2 = 1 - np.sum((F - fit) ** 2) / np.sum((F - F.mean()) ** 2)
    return float(np.inf if r2 >= 1 else 1 / (1 - r2))


def gate0(per_s: dict, tr: float) -> dict:
    """Collinearity gate on the common grid: per_s holds the per-second series face, brightness, motion, envelope, word_rate; each
    becomes its HRF regressor at the grid's film times; VIF(F) must be at most VIF_MAX, else NOT RUN."""
    grid = E.common_grid(tr)
    regs = {k: H.hrf_regressor(per_s[k], grid) for k in ("face", "brightness", "motion", "envelope", "word_rate")}
    v = vif(regs["face"], [regs[k] for k in ("brightness", "motion", "envelope", "word_rate")])
    return {"label": LABEL, "vif_face": v, "vif_max": VIF_MAX, "grid_points": int(len(grid)), "pass": bool(v <= VIF_MAX),
            "outcome_if_failed": "NOT RUN"}
