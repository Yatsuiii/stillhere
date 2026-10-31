"""R3 scoring (docs/rules/r3_variants_rule.md): two forecast variants against the default, through PREREG's variant rule (line 38:
chosen on one language, a variant must beat the default on the other, paired one-sided Wilcoxon p < 0.05 and +0.2 z). Development
on LPP English section 1 (49 listeners), confirmation on LPP French section 1 (28 listeners); never section 8, never sections 2-7 or 9.

Default: the stock forecast through D3's harness (d22/d3/d3_table.py, which is timing/chk8.py's computation): per listener, LANGUAGE z
is the partial r of the 19 LANG parcels on the full forecast given the LANG-mean audio-only forecast and the word-rate regressor and
its derivative at the language's l_env (EN -1.25 s, FR -1.77 s, chk8.py's values), volumes 10-154 cleaned at 0.03 Hz, against a
circular-shift null of at least 15 TRs; HEARING z is p1/hearing.py score_z at lag 0 on the 7 AUD parcels with the audio-only forecast.
Variants, each applied to the full and the audio-only forecast alike, scored by the same harness:
  (a) the average of the stock and fine-tuned forecasts (same seconds required);
  (b) the stock forecast smoothed on the surface, k passes of p1/develop3.py's 1-ring operator S = D^-1 (A + I) on fsaverage5's pial
      mesh (hemispheres apart), forecast only; k in K_GRID, chosen on English as the k with the highest mean LANGUAGE z (ties: the
      smaller k).
A variant is adopted only if its mean English LANGUAGE z is above the default's and, on French, its per-listener LANGUAGE z minus the
default's has a mean of at least +0.20 with a paired one-sided Wilcoxon p below 0.05. Before any variant is scored the run reproduces
D3's English numbers (stock and tuned) and chk8's French stock LANGUAGE from section 3's forecasts, or refuses.
Run from the analysis tree's p1/ (the harness modules are kept outside this repository). Writes OUT/r3_variants.json (per-listener z,
outside this repository) and results/r3_variants.json (group numbers and the decisions).
usage: cd ../stillhere/p1 && python ../../stillhere-pilot-public/lpp_variants/score_variants.py --out ../lpp_variants/score"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K_GRID = (1, 2, 3)
BAR_Z, BAR_P = 0.20, 0.05
D3_EN = {"stock": (1.7252, 1.4723), "tuned": (1.9283, 1.4495)}   # d22/d3/D3-TABLE-2026-10-26.md, all 49, (LANGUAGE, HEARING)
CHK8_FR_STOCK_LANGUAGE = 1.11                                     # timing/chk8_out.txt, FR, section 3's Kaggle forecasts
FR_HOLDOUT = ("FR001", "FR005", "FR006", "FR014", "FR016", "FR022", "FR023", "FR024", "FR025")   # in no fine-tuning target (PREREG)
PINS = {"../d22/d3/out_stock/d3_out/pred_en_full.npz": "925ac173683ac8fe49674b8441ccd7f4eab4d5693406fbe17bfe7fa6dc2e146d",   # D3 table
        "../d22/d3/out_stock/d3_out/pred_en_audio.npz": "96f48a9c9710bb7f2bf1db4d7937adce5c560d269653b08124fb8b26f70ade33",
        "../d22/d3/out_tuned/d3_out/pred_en_full.npz": "2f22aac924ac0fc84f159f244a42c5a03753229b29594e5177a06ab87cf399da",
        "../d22/d3/out_tuned/d3_out/pred_en_audio.npz": "006072d9e4836f09721de6c7303717879b8d07f8801e0508fce562ca0a1aacbf",
        "../k2/out/k2_out/pred_fr_full.npz": "b9268ee548718e2e966cd128d6d4ab22bca04f471c487620d57c08143cd5158f",            # section 3
        "../k2/out/k2_out/pred_fr_audio.npz": "83c7e967f332bce85ea485ecbd0f74ec671f334091188c17a0fd32885b922793"}
FR_PINS = {"../lpp_variants/forecast/out/pred_fr_stock_full.npz": "f61ce1b83965b71f48038d8aa214d1711a7788c52b33661e5da8cd5541a3718c",   # the
           "../lpp_variants/forecast/out/pred_fr_stock_audio.npz": "bdadf0885afd16002df944be9bc4bb2c27a16d13e6671566ee2205f2ac4eaec4",  # rule's
           "../lpp_variants/forecast/out/pred_fr_tuned_full.npz": "805f404f8a2031db872c0d4282239952952983c3ab67a15d0476a532c74154e0",   # second
           "../lpp_variants/forecast/out/pred_fr_tuned_audio.npz": "d4d1ed6b06e19043108294758df1e1d0e07708f6a11fe0ed48954b08e3b2ea2e"}  # commit
CFG = {"EN": dict(cache="cache_en", tg="data/lppEN_section1.TextGrid", l_env=-1.25, n=49,
                  stock=("../d22/d3/out_stock/d3_out/pred_en_full.npz", "../d22/d3/out_stock/d3_out/pred_en_audio.npz"),
                  tuned=("../d22/d3/out_tuned/d3_out/pred_en_full.npz", "../d22/d3/out_tuned/d3_out/pred_en_audio.npz")),
       "FR": dict(cache="cache", tg="data/lppFR_section1.TextGrid", l_env=-1.77, n=28,
                  stock=("../lpp_variants/forecast/out/pred_fr_stock_full.npz", "../lpp_variants/forecast/out/pred_fr_stock_audio.npz"),
                  tuned=("../lpp_variants/forecast/out/pred_fr_tuned_full.npz", "../lpp_variants/forecast/out/pred_fr_tuned_audio.npz"),
                  section3=("../k2/out/k2_out/pred_fr_full.npz", "../k2/out/k2_out/pred_fr_audio.npz"))}


# ---- pure parts (tested without the harness) -------------------------------------------------------------------------------------
def ring_operator(faces_lh, faces_rh, n_lh=10242):
    """p1/develop3.py's S = D^-1 (A + I): each vertex becomes the equal-weight mean of itself and its 1-ring neighbours; the two
    hemispheres are not connected."""
    import scipy.sparse as sp
    rows, cols = [], []
    for off, f in ((0, np.asarray(faces_lh)), (n_lh, np.asarray(faces_rh))):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            rows += [f[:, a] + off, f[:, b] + off]; cols += [f[:, b] + off, f[:, a] + off]
    n = n_lh + int(np.asarray(faces_rh).max()) + 1
    A = sp.coo_matrix((np.ones(sum(len(r) for r in rows)), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    A.data[:] = 1
    return sp.diags(1 / (np.asarray(A.sum(1)).ravel() + 1)) @ (A + sp.eye(n))


def smooth(vert_ts, S, k: int):
    """k passes of S on a (vertices, time) array; k = 0 returns it unchanged."""
    for _ in range(k):
        vert_ts = S @ vert_ts
    return vert_ts


def average(a: dict, b: dict) -> dict:
    """Variant (a): the mean of two forecasts of the same mode over the same seconds."""
    assert np.array_equal(a["starts"], b["starts"]), "the two forecasts cover different seconds"
    return {"P": (a["P"].astype("float64") + b["P"].astype("float64")) / 2, "starts": a["starts"]}


def choose_k(en_means: dict) -> int:
    """Variant (b)'s k: the highest mean English LANGUAGE z; ties go to the smaller k."""
    return min(en_means, key=lambda k: (-en_means[k], k))


def decide(en_default: float, en_variant: float, fr_default: dict, fr_variant: dict) -> dict:
    """PREREG line 38 with the development step: adopted only if English improves and French clears +0.20 z and p < 0.05."""
    from scipy import stats
    ids = sorted(fr_default)
    assert sorted(fr_variant) == ids, "the variant and the default score different French listeners"
    d = np.array([fr_variant[s] - fr_default[s] for s in ids])
    p = float(stats.wilcoxon(d, alternative="greater").pvalue) if np.any(d != 0) else 1.0
    dev_gain = en_variant - en_default
    out = {"en_gain": dev_gain, "fr_mean_difference": float(d.mean()), "fr_higher": int((d > 0).sum()), "fr_n": len(ids), "fr_p": p,
           "dev_ok": bool(dev_gain > 0), "bar_ok": bool(d.mean() >= BAR_Z and p < BAR_P)}
    out["adopted"] = out["dev_ok"] and out["bar_ok"]
    return out


# ---- the harness (analysis tree) -------------------------------------------------------------------------------------------------
class Harness:
    """d22/d3/d3_table.py's per-listener LANGUAGE and HEARING z for one language, with chk8.py's settings for that language."""

    def __init__(self, lang: str):
        import glob
        sys.path.insert(0, "."); sys.path.insert(0, "../timing")
        import hearing as Hh, noise_floor as N, timing as T, understanding as Uu
        self.N, self.Hh, self.Uu = N, Hh, Uu
        c = CFG[lang]
        self.sets = {"AUD": [v + (10242 if h == "rh" else 0) for h, v in N.parcels()["AUD"]], "LANG": [v for h, v in N.parcels()["LANG"]]}
        self.subs = [f.split("/")[-1][:-4] for f in sorted(glob.glob(f"{c['cache']}/sub-*.npz"))]
        assert len(self.subs) == c["n"], f"{lang}: {len(self.subs)} listeners, expected {c['n']}"
        self.raw = {s: np.load(f"{c['cache']}/{s}.npz") for s in self.subs}
        self.n = self.raw[self.subs[0]]["LANG"].shape[1]
        self.B = {s: self.cl(self.raw[s]["LANG"]) for s in self.subs}
        tj, L = 2.0 * np.arange(self.n), c["l_env"]
        onsets = N.textgrid_words(c["tg"]); wps = np.bincount(onsets.astype(int), minlength=int(onsets.max()) + 20).astype(float)
        self.wr = T.hrf_regressor(wps, tj - L)
        self.dwr = T.hrf_regressor(wps, tj - L + 0.5) - T.hrf_regressor(wps, tj - L - 0.5)

    def cl(self, a):
        return self.N.clean(a[:, 10:155], 0.03)

    def z_of(self, y, x, ctl):
        real = self.Uu.partial_score(y, x, ctl)
        null = np.array([self.Uu.partial_score(y, np.roll(x, k, 1), ctl) for k in range(15, y.shape[1] - 15)])
        return (real - null.mean()) / null.std()

    def vert(self, d):
        return self.Hh.align(d["P"], d["starts"].astype(float), self.n).T

    def score(self, full: dict, aud: dict, S=None, k: int = 0) -> dict:
        Pf = self.Hh.parcel_ts(smooth(self.vert(full), S, k), self.sets)
        Pa = self.Hh.parcel_ts(smooth(self.vert(aud), S, k), self.sets)
        ctl = [self.cl(Pa["LANG"]).mean(0, keepdims=True), self.cl(self.wr[None]), self.cl(self.dwr[None])]
        lang = {s: float(self.z_of(self.B[s], self.cl(Pf["LANG"]), ctl)) for s in self.subs}
        hear = {}
        for s in self.subs:
            r, m, sd = self.Hh.score_z(self.raw[s]["AUD"], Pa["AUD"], 0); hear[s] = float((r - m) / sd)
        return {"LANGUAGE": lang, "HEARING": hear}


def sha256(p) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def check_pins(pins: dict, what: str) -> dict:
    if not pins:
        raise SystemExit(f"refused: the {what} are not pinned yet (the rule's second commit comes before scoring)")
    bad = [p for p, want in pins.items() if sha256(p) != want]
    if bad:
        raise SystemExit(f"refused: {bad} differ from the pinned SHA-256")
    return dict(pins)


def means(r: dict, ids=None) -> dict:
    return {g: float(np.mean([v for s, v in r[g].items() if ids is None or s in ids])) for g in ("LANGUAGE", "HEARING")}


def run(out_dir, check_only=False):
    """check_only: the scorer checks alone (numbers already on record), no variant scored and nothing written."""
    assert not os.path.abspath(out_dir).startswith(REPO), "OUT must be outside the repo (per-listener z)"
    from nilearn import datasets, surface
    fsa = datasets.fetch_surf_fsaverage("fsaverage5")
    S = ring_operator(surface.load_surf_mesh(fsa["pial_left"])[1], surface.load_surf_mesh(fsa["pial_right"])[1])
    load = lambda pair: tuple(dict(np.load(p)) for p in pair)
    per, pub = {}, {"rule": "docs/rules/r3_variants_rule.md", "bar": {"z": BAR_Z, "p": BAR_P}, "k_grid": list(K_GRID),
                    "inputs_sha256": check_pins(PINS, "English and section 3 forecasts")}
    if not check_only:
        pub["inputs_sha256"].update(check_pins(FR_PINS, "laptop French forecasts"))
    # scorer checks, before any variant is scored
    en = Harness("EN"); es, et = load(CFG["EN"]["stock"]), load(CFG["EN"]["tuned"])
    per["EN_default"] = en.score(*es); per["EN_tuned"] = en.score(*et)
    for name, key in (("stock", "EN_default"), ("tuned", "EN_tuned")):
        got = means(per[key]); want = D3_EN[name]                      # compared as D3's table printed them (4 decimals)
        assert (f"{got['LANGUAGE']:.4f}", f"{got['HEARING']:.4f}") == tuple(f"{w:.4f}" for w in want), \
            f"EN {name} {got} does not reproduce D3 {want}"
    fr = Harness("FR")
    fr3 = means(fr.score(*load(CFG["FR"]["section3"])))["LANGUAGE"]       # as chk8 printed it (2 decimals)
    assert f"{fr3:.2f}" == f"{CHK8_FR_STOCK_LANGUAGE:.2f}", f"FR section 3 stock LANGUAGE {fr3:.4f} does not reproduce chk8's 1.11"
    pub["scorer_checks"] = {"EN_stock_d3": D3_EN["stock"], "EN_tuned_d3": D3_EN["tuned"], "FR_section3_stock_language": round(fr3, 4)}
    print("scorer checks pass:", json.dumps(pub["scorer_checks"]), flush=True)
    if check_only:
        return pub
    # development: English
    per["EN_a"] = en.score(*(average(x, y) for x, y in zip(es, et)))
    for k in K_GRID:
        per[f"EN_b{k}"] = en.score(*es, S=S, k=k)
    k_star = choose_k({k: means(per[f"EN_b{k}"])["LANGUAGE"] for k in K_GRID})
    # confirmation: French (laptop stock is the default; same environment as the fine-tuned forecasts)
    fs, ft = load(CFG["FR"]["stock"]), load(CFG["FR"]["tuned"])
    per["FR_default"] = fr.score(*fs)
    per["FR_a"] = fr.score(*(average(x, y) for x, y in zip(fs, ft)))
    per[f"FR_b{k_star}"] = fr.score(*fs, S=S, k=k_star)
    per["FR_tuned"] = fr.score(*ft)
    pub["k_chosen"] = k_star
    pub["means"] = {key: {"all": means(r), **({"fr_holdout9": means(r, {f"sub-{h}" for h in FR_HOLDOUT})} if key.startswith("FR") else {})}
                    for key, r in per.items()}
    dec = {}
    for tag, en_key, fr_key in (("a_average", "EN_a", "FR_a"), (f"b_smooth_k{k_star}", f"EN_b{k_star}", f"FR_b{k_star}")):
        dec[tag] = decide(means(per["EN_default"])["LANGUAGE"], means(per[en_key])["LANGUAGE"],
                          per["FR_default"]["LANGUAGE"], per[fr_key]["LANGUAGE"])
    pub["decisions"] = dec
    pub["adopted"] = [t for t, d in dec.items() if d["adopted"]] or "none"
    os.makedirs(out_dir, exist_ok=True)
    json.dump({**pub, "per_listener": per}, open(f"{out_dir}/r3_variants.json", "w"), indent=1)
    json.dump(pub, open(f"{REPO}/results/r3_variants.json", "w"), indent=1)
    print(json.dumps({"k_chosen": k_star, "adopted": pub["adopted"],
                      "decisions": {t: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()} for t, d in dec.items()}}))
    return pub


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    run(a.out, check_only=a.check)
