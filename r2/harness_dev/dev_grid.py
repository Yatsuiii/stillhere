"""Dev set only (LPP English section 1): high-pass cutoff and forecast lag, full run, against the leave-one-out group ceiling."""
import sys, glob, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "../timing")
import noise_floor as N, hearing as Hh, understanding as Uu, timing as T
MIN = 15; TR = 2.0
sets = {"AUD": [v + (10242 if h == "rh" else 0) for h, v in N.parcels()["AUD"]], "LANG": [v for h, v in N.parcels()["LANG"]]}
subs = [f.split("/")[-1][:-4] for f in sorted(glob.glob("cache_en/sub-*.npz"))]
raw = {s: np.load(f"cache_en/{s}.npz")["LANG"] for s in subs}; n = raw[subs[0]].shape[1]
tj = 2.0 * np.arange(n); L = -1.25
onsets = N.textgrid_words("data/lppEN_section1.TextGrid"); wps = np.bincount(onsets.astype(int), minlength=int(onsets.max()) + 20).astype(float)
wr = T.hrf_regressor(wps, tj - L); dwr = T.hrf_regressor(wps, tj - L + 0.5) - T.hrf_regressor(wps, tj - L - 0.5)
D = "../d22/d3"
def load(f, shift):
    d = np.load(f); return Hh.parcel_ts(Hh.align(d["P"], d["starts"].astype(float) + shift, n).T, sets)["LANG"]
def z_of(y, x, ctl):
    real = Uu.partial_score(y, x, ctl); null = np.array([Uu.partial_score(y, np.roll(x, k, 1), ctl) for k in range(MIN, y.shape[1] - MIN)]); return (real - null.mean()) / null.std()
j0, j1 = 12, 275
for hp in (0.01, 0.02, 0.03):
    cl = lambda a: N.clean(a[:, j0:j1 + 1], hp)
    B = {s: cl(raw[s]) for s in subs}; S = np.stack([B[s] for s in subs])
    res = []
    for shift in (-2, -1, 0, 1, 2):
        Pf, Pa = load(f"{D}/out_stock/d3_out/pred_en_full.npz", shift), load(f"{D}/out_stock/d3_out/pred_en_audio.npz", shift)
        Tf, Ta = load(f"{D}/out_tuned/d3_out/pred_en_full.npz", shift), load(f"{D}/out_tuned/d3_out/pred_en_audio.npz", shift)
        ctl = lambda aud: [cl(aud).mean(0, keepdims=True), cl(wr[None]), cl(dwr[None])]
        st = np.mean([z_of(B[s], cl(Pf), ctl(Pa)) for s in subs]); tu = np.mean([z_of(B[s], cl(Tf), ctl(Ta)) for s in subs])
        if shift == 0: ceil = np.mean([z_of(B[s], (S.sum(0) - B[s]) / 48, ctl(Pa)) for s in subs])
        res.append((shift, st, tu))
    print(f"hp {hp}: ceiling {ceil:.2f}; " + " | ".join(f"shift {s:+d} s stock {a:.2f} ({a / ceil:.0%}) tuned {b:.2f} ({b / ceil:.0%})" for s, a, b in res), flush=True)
