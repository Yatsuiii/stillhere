"""Dev set only (LPP English section 1, PREREG's development set): how much the 145-volume window and the zero-shot readout
leave on the table. D3's LANGUAGE computation unchanged except the window; plus the leave-one-out group ceiling."""
import sys, glob, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "../timing")
import noise_floor as N, hearing as Hh, understanding as Uu, timing as T
MIN = 15
sets = {"AUD": [v + (10242 if h == "rh" else 0) for h, v in N.parcels()["AUD"]], "LANG": [v for h, v in N.parcels()["LANG"]]}
subs = [f.split("/")[-1][:-4] for f in sorted(glob.glob("cache_en/sub-*.npz"))]; assert len(subs) == 49
raw = {s: np.load(f"cache_en/{s}.npz")["LANG"] for s in subs}; n = raw[subs[0]].shape[1]
tj = 2.0 * np.arange(n); L = -1.25
onsets = N.textgrid_words("data/lppEN_section1.TextGrid"); wps = np.bincount(onsets.astype(int), minlength=int(onsets.max()) + 20).astype(float)
wr = T.hrf_regressor(wps, tj - L); dwr = T.hrf_regressor(wps, tj - L + 0.5) - T.hrf_regressor(wps, tj - L - 0.5)
D = "../d22/d3"
P = {k: Hh.parcel_ts(Hh.align(np.load(f)["P"], np.load(f)["starts"].astype(float), n).T, sets)["LANG"]
     for k, f in {"stock_full": f"{D}/out_stock/d3_out/pred_en_full.npz", "stock_aud": f"{D}/out_stock/d3_out/pred_en_audio.npz",
                  "tuned_full": f"{D}/out_tuned/d3_out/pred_en_full.npz", "tuned_aud": f"{D}/out_tuned/d3_out/pred_en_audio.npz"}.items()}
fin = np.all(np.isfinite(np.vstack(list(P.values()))), 0); last = int(np.where(fin)[0].max())
def z_of(y, x, ctl):
    real = Uu.partial_score(y, x, ctl); null = np.array([Uu.partial_score(y, np.roll(x, k, 1), ctl) for k in range(MIN, y.shape[1] - MIN)]); return (real - null.mean()) / null.std()
print(f"volumes in the run {n}, forecast finite to {last}")
for j0, j1 in ((10, 154), (10, last)):
    cl = lambda a: N.clean(a[:, j0:j1 + 1], 0.03)
    B = {s: cl(raw[s]) for s in subs}
    ctl = lambda aud: [cl(aud).mean(0, keepdims=True), cl(wr[None]), cl(dwr[None])]
    out = {}
    out["stock"] = [z_of(B[s], cl(P["stock_full"]), ctl(P["stock_aud"])) for s in subs]
    out["tuned"] = [z_of(B[s], cl(P["tuned_full"]), ctl(P["tuned_aud"])) for s in subs]
    out["average"] = [z_of(B[s], cl((P["stock_full"] + P["tuned_full"]) / 2), ctl((P["stock_aud"] + P["tuned_aud"]) / 2)) for s in subs]
    # ceiling: the other 48 listeners' mean LANG series as the forecast, same controls (stock audio forecast, word rate)
    S = np.stack([B[s] for s in subs])
    out["loo48_ceiling"] = [z_of(B[s], (S.sum(0) - B[s]) / 48, ctl(P["stock_aud"])) for s in subs]
    print(f"volumes {j0}-{j1} ({j1 - j0 + 1} TRs, {2 * (j1 - j0 + 1) / 60:.1f} min): " + " | ".join(f"{k} {np.mean(v):.3f}" for k, v in out.items()), flush=True)
