"""The scorer. Arrays in, one JSON-ready dict out. Definitions are in PREREG.md section 1; this file is frozen by SHA-256 (FROZEN.sha256).

Shapes: a person's BOLD is (areas, volumes). The forecast is (areas, volumes), (volumes,) for one series used in every area, or a dict
of per-person forecasts. A baseline or control is (volumes,) or (areas, volumes)."""
import numpy as np
from scipy import signal, stats

DETECT = 1.645


# ---- inputs ----------------------------------------------------------------------------------------------------------------------
def _rows(x, p: int, T: int, what: str) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        a = np.tile(a, (p, 1))
    if a.shape != (p, T):
        raise ValueError(f"{what}: expected shape ({p}, {T}) or ({T},), got {np.asarray(x).shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{what}: contains NaN or inf")
    return a


def _subjects(bold) -> tuple:
    if isinstance(bold, dict):
        subs = sorted(bold)
        arrs = [np.asarray(bold[s], float) for s in subs]
    else:
        arrs = [np.asarray(b, float) for b in bold]
        subs = [f"sub-{i:03d}" for i in range(len(arrs))]
    if len(arrs) < 3:
        raise ValueError(f"need at least 3 people, got {len(arrs)}")
    if any(a.ndim == 1 for a in arrs):
        arrs = [a[None] if a.ndim == 1 else a for a in arrs]
    p, T = arrs[0].shape
    for s, a in zip(subs, arrs):
        _rows(a, p, T, f"bold[{s}]")
    return subs, np.stack(arrs), p, T


# ---- cleaning and the gauge ------------------------------------------------------------------------------------------------------
def clean(A, tr: float, high_pass=None) -> np.ndarray:
    """Rows of A: linear detrend, optional 5th-order Butterworth high-pass at high_pass Hz (zero phase), z-score with ddof 1."""
    A = signal.detrend(np.atleast_2d(np.asarray(A, float)), axis=-1, type="linear")
    if high_pass:
        sos = signal.butter(5, high_pass, btype="highpass", fs=1.0 / tr, output="sos")
        A = signal.sosfiltfilt(sos, A, axis=-1)
    sd = A.std(-1, ddof=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (A - A.mean(-1, keepdims=True)) / sd


def _projector(controls, T: int) -> np.ndarray:
    C = np.vstack([np.atleast_2d(c) for c in controls] + [np.ones((1, T))]).T if controls else np.ones((T, 1))
    return C @ np.linalg.pinv(C.T @ C) @ C.T


def _fz(rb, rp) -> np.ndarray:
    a = rb - rb.mean(-1, keepdims=True)
    b = rp - rp.mean(-1, keepdims=True)
    den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    r = np.where(den > 0, (a * b).sum(-1) / np.where(den > 0, den, 1.0), 0.0)
    return np.arctanh(np.clip(r, -0.999, 0.999)).mean(-1)


def gauge(B, X, H, min_shift: int) -> tuple:
    """(z, fisher_z): B (p, T) BOLD rows, X (p, T) forecast rows, H the projector onto [controls, 1]. Both are residualised on H;
    the real score is compared with every circular shift of X by min_shift..T-min_shift volumes."""
    T = B.shape[1]
    rb = B - B @ H
    shifts = np.r_[0, np.arange(min_shift, T - min_shift + 1)]
    idx = (np.arange(T)[None, :] - shifts[:, None]) % T
    Xs = np.asarray(X)[:, idx].transpose(1, 0, 2)
    s = _fz(rb, Xs - Xs @ H)
    null = s[1:]
    sd = null.std()
    return float((s[0] - null.mean()) / sd) if sd > 0 else 0.0, float(s[0])


# ---- volunteer curve and placement -----------------------------------------------------------------------------------------------
def _place(v: float, ks, curve) -> tuple:
    """(k, at_ceiling) for a z value on an increasing-k curve; the curve is made monotone (running max) before interpolation."""
    c = np.maximum.accumulate(np.asarray(curve, float))
    if v < c[0]:
        return 0.0, False
    if v >= c[-1]:
        return float(ks[-1]), True
    return float(np.interp(v, c, ks)), False


def _holm(ps: dict) -> dict:
    order = sorted(ps, key=ps.get)
    out, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, ps[k] * (len(order) - i)))
        out[k] = running
    return out


def _wilcoxon_greater(d) -> float:
    d = np.asarray(d, float)
    if np.allclose(d, 0):
        return 1.0
    return float(stats.wilcoxon(d, alternative="greater").pvalue)


# ---- the score -------------------------------------------------------------------------------------------------------------------
def score(bold, prediction, baselines=None, controls=None, *, tr: float, high_pass=None, ks=(1, 2, 3, 4, 6, 8, 12), reps: int = 20,
          min_shift: int = 15, seed: int = 0, nboot: int = 10_000, alpha: float = 0.05) -> dict:
    """Score a forecast against each person, beside cheap baselines, and say how many real volunteers it is worth.

    bold: {person: (areas, volumes)} or a list/array of them. prediction: (areas, volumes), (volumes,), or {person: ...}.
    baselines: {name: (volumes,) or (areas, volumes)}. controls: list of regressors partialled out of everything.
    ks: the volunteer counts of the curve; values above n-1 are dropped. Returns a JSON-ready dict."""
    subs, B, p, T = _subjects(bold)
    n = len(subs)
    if min_shift < 1 or T - 2 * min_shift < 20:
        raise ValueError(f"min_shift {min_shift} leaves too few circular shifts for {T} volumes")
    if isinstance(prediction, dict):
        missing = [s for s in subs if s not in prediction]
        if missing:
            raise ValueError(f"prediction has no forecast for {missing[:3]}")
        P = {s: clean(_rows(prediction[s], p, T, f"prediction[{s}]"), tr, high_pass) for s in subs}
    else:
        one = clean(_rows(prediction, p, T, "prediction"), tr, high_pass)
        P = {s: one for s in subs}
    base = {k: clean(_rows(v, p, T, f"baseline {k}"), tr, high_pass) for k, v in (baselines or {}).items()}
    ctrl = []
    if controls:
        ctrl = [clean(np.asarray(c, float)[None] if np.ndim(c) == 1 else np.asarray(c, float), tr, high_pass) for c in controls]
        for c in ctrl:
            if c.shape[1] != T or not np.isfinite(c).all():
                raise ValueError("control: wrong length or not finite")
    ctrl_rows = [r[None] for c in ctrl for r in c] if controls else []
    H = _projector(ctrl_rows, T)
    H_beyond = _projector(ctrl_rows + [b.mean(0, keepdims=True) for b in base.values()], T)
    Bc = {s: clean(B[i], tr, high_pass) for i, s in enumerate(subs)}

    ks = tuple(int(k) for k in ks if 1 <= int(k) <= n - 1)
    if not ks:
        raise ValueError("no usable k: need 1 <= k <= n - 1")
    rng = np.random.default_rng(seed)

    rows = []
    for s in subs:
        y = Bc[s]
        zm, rm = gauge(y, P[s], H, min_shift)
        zb = {k: gauge(y, b, H, min_shift)[0] for k, b in base.items()}
        zbeyond = gauge(y, P[s], H_beyond, min_shift)[0] if base else zm
        others = [t for t in subs if t != s]
        curve = []
        for k in ks:
            draws = 1 if k == len(others) else reps
            zz = []
            for _ in range(draws):
                pick = others if k == len(others) else [others[j] for j in rng.choice(len(others), k, replace=False)]
                zz.append(gauge(y, clean(np.mean([Bc[t] for t in pick], 0), tr), H, min_shift)[0])
            curve.append(float(np.mean(zz)))
        k_s, ceil = _place(zm, ks, curve)
        rows.append({"subject": s, "z_model": zm, "fisher_z_model": rm, "z_beyond_baselines": zbeyond,
                     **{f"z_{k}": v for k, v in zb.items()}, "curve_z": curve, "volunteer_equivalent": k_s, "at_ceiling": ceil})

    zm = np.array([r["z_model"] for r in rows])
    curve_mean = np.mean([r["curve_z"] for r in rows], 0)
    bs = np.array([zm[rng.integers(0, n, n)].mean() for _ in range(nboot)])
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    pk, pceil = _place(zm.mean(), ks, curve_mean)
    lk, _ = _place(lo, ks, curve_mean)
    hk, _ = _place(hi, ks, curve_mean)

    raw_p = {k: _wilcoxon_greater(zm - np.array([r[f"z_{k}"] for r in rows])) for k in base}
    holm = _holm(raw_p) if raw_p else {}
    zbe = np.array([r["z_beyond_baselines"] for r in rows])
    p_beyond = _wilcoxon_greater(zbe)
    bl = {k: {"z_mean": float(np.mean([r[f"z_{k}"] for r in rows])), "p_model_greater": raw_p[k], "p_holm": holm[k],
              "beaten": bool(zm.mean() > np.mean([r[f"z_{k}"] for r in rows]) and holm[k] < alpha and p_beyond < alpha)} for k in base}
    beats = sorted(k for k, v in bl.items() if v["beaten"])
    return {
        "format": "veq-score/1",
        "n": n, "areas": p, "volumes": T,
        "settings": {"tr": tr, "high_pass": high_pass, "ks": list(ks), "reps": reps, "min_shift": min_shift, "seed": seed, "nboot": nboot,
                     "alpha": alpha, "controls": len(ctrl_rows)},
        "model": {"z_mean": float(zm.mean()), "detect_pct": float(100 * (zm > DETECT).mean()),
                  "p_greater_than_zero": _wilcoxon_greater(zm)},
        "baselines": bl,
        "beyond_baselines": {"z_mean": float(zbe.mean()), "detect_pct": float(100 * (zbe > DETECT).mean()),
                             "p_greater_than_zero": p_beyond} if base else None,
        "curve": {"ks": list(ks), "z_mean": [float(v) for v in curve_mean]},
        "volunteer_equivalent": {"group": {"z_mean": float(zm.mean()), "z_ci95": [lo, hi], "point_k": pk, "ci95_k": [lk, hk],
                                           "claim_k": int(np.floor(lk)), "at_ceiling": pceil},
                                 "per_subject_median_k": float(np.median([r["volunteer_equivalent"] for r in rows]))},
        "verdict": {"beats": beats, "beats_all_baselines": bool(base) and len(beats) == len(base),
                    "worth_at_least": int(np.floor(lk))},
        "flags": {"subjects_at_ceiling": [r["subject"] for r in rows if r["at_ceiling"]]},
        "per_subject": rows,
    }
