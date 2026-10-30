"""Alice Datasets held-out (docs/rules/alice_heldout_rule.md; exploratory): section 8's claim, unchanged, on a different public
audiobook, outside TRIBE v2's training and evaluation sets (OpenNeuro ds002322, chapter 1, CC0).

Section 8's per-listener math is called unchanged (stillhere/heldout.py: clean, gauge, stage_scores, volunteer_equivalent;
stillhere/cheap_baselines.py: section8, compare), one listener at a time, because each listener's audio started at its own scan time
(events.tsv onset, 0.003 to 0.837 s): that listener's forecast and every stimulus-time regressor are shifted by its onset (the forecast
through `starts + onset`, the regressors through the lag `L + onset`, which hrf_regressor reads at 2j - L - onset). Only the group step
is formed here, with section 8's formulas (group_stage, own_best_lag_onsets, isc_curve_onsets); with every onset equal they reproduce
heldout.stage_scores, own_best_lag and isc_curve exactly (tests/test_alice_heldout.py, synthetic data only).

Steps, in this order:
  download   the 23 derivatives and their events.tsv (only once the rule is on origin and the cloud has said go)
  surprisal  GPT-2 small surprisal of the timed words, by p1/surprisal8.py's method (needs torch, transformers and the cached gpt2)
  score      one scoring run; writes OUT/alice_heldout.json (per-listener z, so OUT is outside this repo) and
             results/alice_heldout.json (group numbers, the outcome line, and per listener only protocol, onset and coverage)
usage: python scripts/alice_heldout.py {download|surprisal|score} --root ../stillhere [--out ../stillhere/alice/score]"""
import argparse
import csv
import hashlib
import json
import os
import sys
import urllib.request

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from stillhere import cheap_baselines as CB  # noqa: E402
from stillhere import heldout as H  # noqa: E402

BASE = "https://s3.amazonaws.com/openneuro.org/ds002322"
SUBS = ("18", "22", "23", "24", "26", "28", "30", "31", "35", "36", "37", "39", "41", "42", "43", "44", "45", "47", "48", "49", "50",
        "51", "53")
NO_DERIVATIVE = ("38", "46", "52")
EXCLUDED = {"53": "its published derivative fails the gzip CRC check, and our MD5 matches the checksum OpenNeuro's storage reports, "
                  "so the damage is at the source; from inside derivative volume 11, every later volume is displaced by 300 voxels, "
                  "and no repair can be verified (rule section 2, amended by the cloud ruling of Oct 30, 19:40 UTC)"}
SCORED = tuple(s for s in SUBS if s not in EXCLUDED)
PROTOCOL = {s: "single-echo, 44 slices" if int(s) <= 37 else "three-echo (echo 2), 33 slices" for s in SUBS}
STIM = "DownTheRabbitHoleFinal_mono_exp120_NR16_pad.wav"
STIM_SHA = "bb8003578d0667cd45fbce932c45dcfa3abee4be40da98c0a83dea80516aeefe"
ANN_SHA = "6e25b0d85a1457a75c608f2a298dbef97172f83270eb35a82b43c92229d0e46c"
SKIP = 10                    # the dataset dropped raw volumes 0-9: derivative volume i is raw volume i + 10
COVER, MIN_N = 0.90, 10      # an area counts for a listener only with >= 90% of its vertices covered; < 10 listeners: not scored
DEV_LAG, M1_KS, M1_REPS, M1_SEED = -1.25, (1, 2, 3, 4, 6, 8, 12, 15), 20, 11
TREE_LABEL = "parse from the dataset's annotations, not LPP's"
LINES = {   # docs/rules/alice_heldout_rule.md section 7, word for word
    "claim": "On a different public audiobook, outside TRIBE v2's training and evaluation sets (the Alice Datasets, {n} listeners), the "
             "forecast was worth about {point} real volunteers, and at least {claim}.",
    "below_one": "On a different public audiobook, outside TRIBE v2's training and evaluation sets (the Alice Datasets, {n} listeners), "
                 "the forecast tracked the listeners' language areas (z {z}, p = {p}), but at its lower bound it was worth less than one "
                 "real volunteer (about {point} at the point estimate).",
    "no_bar": "On a different public audiobook, outside TRIBE v2's training and evaluation sets (the Alice Datasets, {n} listeners), the "
              "forecast did not reach the bar in language areas (z {z}, p = {p}). We show it because we said we would.",
    "coverage": "Alice Datasets: rule committed, not run (coverage): {n} listeners had the language areas in view, fewer than 10.",
    "not_run": "Alice Datasets: rule committed, not run.",
    "beat": "What it had to beat: word rate and the sound envelope (beaten in {k} of {K} comparisons).",
}


def sha(p) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# ---- words (the runner's rule: timed rows only) ------------------------------------------------------------------------------------
def timed_words(ann_path) -> list:
    """[(lower-cased word, onset, row)] for annotations.tsv rows with both timings, in file order; untimed rows are dropped."""
    rows = list(csv.DictReader(open(ann_path), delimiter="\t"))
    return [(r["Words"].strip().lower(), float(r["Word Onset"]), r) for r in rows if r["Word Onset"].strip() and r["Word Offset"].strip()]


def baselines(words, surprisal: dict, n_sec: int) -> dict:
    """Section 8's four baseline regressors (heldout.baseline_series' definitions) from the timed words: log frequency within the
    chapter; the parse tree's bottom-up and top-down node counts, here from the dataset's own parse (TREE_LABEL); GPT-2 surprisal."""
    import collections
    a = [w for w, _, _ in words]
    t_on = [t for _, t, _ in words]
    cnt = collections.Counter(a)
    num = lambda r, k: float(r[k]) if r.get(k, "").strip() not in ("", "nan", "NaN") else np.nan
    return {"B_freq": H.per_second([np.log10(cnt[w] / len(a)) for w in a], t_on, n_sec),
            "B_bu": H.per_second([num(r, "CFG BU nodeCount") for _, _, r in words], t_on, n_sec),
            "B_td": H.per_second([num(r, "CFG TD nodeCount") for _, _, r in words], t_on, n_sec),
            "surprisal": H.per_second(surprisal["surprisal_bits"], surprisal["onsets"], n_sec)}


def gpt2_surprisal(words) -> dict:
    """p1/surprisal8.py's method: GPT-2 small, " " + lower-cased word, 512-token windows advancing by half, bits summed per word."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2"); lm = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    ids, owner = [], []
    for i, (w, _, _) in enumerate(words):
        t_ids = tok(" " + w.lower(), add_special_tokens=False)["input_ids"]; ids += t_ids; owner += [i] * len(t_ids)
    ids = np.array(ids); ctx = 512; surp = np.zeros(len(ids)); bos = tok.bos_token_id
    for start in range(0, len(ids), ctx // 2):
        lo = max(0, start - ctx // 2); hi = min(len(ids), start + ctx // 2); seg = np.concatenate([[bos], ids[lo:hi]])
        with torch.no_grad():
            lp = torch.log_softmax(lm(torch.tensor(seg)[None]).logits[0], -1)
        for k in range(start, hi):
            if k >= lo:
                surp[k] = -lp[k - lo, seg[k - lo + 1]].item() / np.log(2)
    ws = np.zeros(len(words))
    for s_, o in zip(surp, owner):
        ws[o] += s_
    return {"onsets": [t for _, t, _ in words], "surprisal_bits": ws.tolist()}


# ---- volumes to parcels ------------------------------------------------------------------------------------------------------------
_M = {}


def mid_meshes():
    """Section 8's LPP projection meshes (p1/hearing.py): fsaverage5, halfway between pial and white, white as the inner mesh."""
    from nilearn import datasets, surface
    if not _M:
        fsa = datasets.fetch_surf_fsaverage("fsaverage5")
        for h in ("left", "right"):
            pc, pf = surface.load_surf_mesh(fsa[f"pial_{h}"]); wc, _ = surface.load_surf_mesh(fsa[f"white_{h}"])
            _M[h] = ((0.5 * pc + 0.5 * wc, pf), fsa[f"white_{h}"])
    return _M


def project(bold_path) -> np.ndarray:
    """(20484, T): p1/hearing.py's project, unchanged (vol_to_surf, 3 mm ball, linear; left then right, TRIBE's vertex order)."""
    import nibabel as nb
    from nilearn import surface
    img = nb.load(bold_path); M = mid_meshes()
    return np.vstack([surface.vol_to_surf(img, surf_mesh=M[h][0], inner_mesh=M[h][1], radius=3.0, kind="ball", interpolation="linear")
                      for h in ("left", "right")])


def covered(vert_ts) -> np.ndarray:
    """A vertex is covered when its series is finite and has non-zero variance (outside the field of view it is constant or NaN)."""
    v = np.asarray(vert_ts, float)
    out = np.isfinite(v).all(1)
    out[out] = v[out].var(1) > 0
    return out


def area_shares(cov, sets) -> dict:
    """Covered share of each area's vertices for one listener."""
    return {net: float(np.mean(cov[np.concatenate(vs)])) for net, vs in sets.items()}


def kept_parcels(covs: dict, sets: dict, eligible: dict) -> dict:
    """Per area, the parcels with at least one covered vertex in every listener eligible for that area (from masks only)."""
    return {net: [i for i, vs in enumerate(sets[net]) if all(covs[s][vs].any() for s in eligible[net])] for net in sets}


def parcel_series(vert_ts, cov, sets, keep) -> dict:
    """{area: (kept parcels, T)}: each parcel the mean over its covered vertices (NaN-free); a parcel with no covered vertex in this
    listener is filled with zeros (only possible for an area the listener is not eligible for; its scores are never used)."""
    out = {}
    for net, idx in keep.items():
        rows = []
        for i in idx:
            v = sets[net][i][cov[sets[net][i]]]
            rows.append(np.asarray(vert_ts, float)[v].mean(0) if len(v) else np.zeros(np.asarray(vert_ts).shape[1]))
        out[net] = np.stack(rows)
    return out


def to_raw(series: dict, skip: int = SKIP) -> dict:
    """Derivative volume i at raw volume i + skip; raw volumes 0..skip-1 are NaN and never read (section 8's windows start at J0)."""
    return {net: np.hstack([np.full((A.shape[0], skip), np.nan), A]) for net, A in series.items()}


def subset(sets: dict, keep: dict) -> dict:
    return {net: [sets[net][i] for i in idx] for net, idx in keep.items()}


# ---- the group step, with section 8's formulas -------------------------------------------------------------------------------------
def own_best_lag_onsets(aud_raw: list, onsets: list, env, n_vols: int, j1: int, hp: float = 0.03, grid=None) -> tuple:
    """heldout.own_best_lag with each listener's regressor read at its own onset (equal onsets: identical to it)."""
    grid = np.arange(-4, 1.01, 0.25) if grid is None else grid
    tj = H.TR * np.arange(n_vols)
    BA = [H.clean(b, hp, H.J0, j1) for b in aud_raw]
    zz = []
    for lag in grid:
        zz.append(np.mean([H.gauge(b, np.tile(H.clean(H.hrf_regressor(env, tj - lag - o)[None], hp, H.J0, j1), (BA[0].shape[0], 1)),
                                   H.zeros_control(b.shape[1]))[0] for b, o in zip(BA, onsets)]))
    return float(grid[int(np.argmax(zz))]), [float(x) for x in zz]


def group_stage(per: dict, L: float, j1: int, aud_subs: list, lang_subs: list) -> dict:
    """heldout.stage_scores' group step over per-listener stage results (each from stage_scores on that listener alone): hearing and
    envelope over the listeners eligible for HEARING, the rest over those eligible for LANGUAGE."""
    from scipy import stats
    g = lambda subs, k: np.array([per[s]["per_listener"][k][0] for s in subs])
    zh, ez = g(aud_subs, "hearing"), g(aud_subs, "envelope")
    zl, zs = g(lang_subs, "language"), g(lang_subs, "shuffled")
    base = {k[5:]: g(lang_subs, k) for k in per[lang_subs[0]]["per_listener"] if k.startswith("base_")}
    best = max(base, key=lambda k: base[k].mean())
    wil = lambda d: float(stats.wilcoxon(d, alternative="greater").pvalue)
    p2, p1, pb = wil(zl), wil(zl - zs), wil(zl - base[best])
    return {"L": L, "volumes": [H.J0, j1], "n": len(lang_subs), "n_hearing": len(aud_subs),
            "hearing": {"z": float(zh.mean()), "detect_pct": H.detect_pct(zh)},
            "envelope": {"z": float(ez.mean()), "detect_pct": H.detect_pct(ez), "p_tribe_gt_envelope": wil(zh - ez)},
            "language": {"z": float(zl.mean()), "detect_pct": H.detect_pct(zl), "p": p2, "pass": bool(zl.mean() >= 1 and p2 < 0.05)},
            "shuffled": {"z": float(zs.mean()), "p_real_gt_shuffled": p1, "pass": bool(p1 < 0.05)},
            "baselines": {k: float(v.mean()) for k, v in base.items()}, "best_baseline": best, "p_tribe_gt_best": pb,
            "tribe_beats_best": bool(zl.mean() > base[best].mean() and pb < 0.05),
            "per_listener": {"hearing_listeners": aud_subs, "hearing": zh.tolist(), "envelope": ez.tolist(), "listeners": lang_subs,
                             "language": zl.tolist(), "shuffled": zs.tolist(), **{f"base_{k}": v.tolist() for k, v in base.items()}}}


def isc_curve_onsets(bold: dict, ctrl_by: dict, ks, reps: int, seed: int, subs_by: dict, hp: float = 0.03, j1: int = 154) -> dict:
    """heldout.isc_curve with each listener's own control rows and each area over its eligible listeners; the random sets come from one
    Generator(seed), drawn network, listener, k, repeat in that order (equal controls, everyone eligible: identical to it)."""
    rng = np.random.default_rng(seed)
    cl = lambda A: H.clean(A, hp, H.J0, j1)
    out = {}
    for net in ("AUD", "LANG"):
        subs = sorted(subs_by[net]); res = {k: [] for k in ks}
        for s in subs:
            y = cl(bold[s][net]); others = [t for t in subs if t != s]
            for k in ks:
                zs = []
                for _ in range(1 if k == len(others) else reps):
                    pick = others if k == len(others) else list(rng.choice(others, k, replace=False))
                    zs.append(H.gauge(y, cl(np.mean([bold[t][net] for t in pick], 0)), ctrl_by[s][net])[0])
                res[k].append((float(np.mean(zs)), float(np.mean(np.array(zs) > H.DETECT))))
        out[net] = res
    return out


def group_cheap(per: dict, subs_by: dict) -> dict:
    """cheap_baselines.section8's group step over per-listener results: mean z and compare() per area, over its eligible listeners."""
    out = {}
    for net in ("LANG", "AUD"):
        subs = subs_by[net]
        z = {k: np.array([per[s][net]["per_listener"][k][0] for s in subs]) for k in CB.KINDS}
        out[net] = {"mean_z": {k: float(v.mean()) for k, v in z.items()}, "tribe_vs": {k: CB.compare(z["tribe"], z[k]) for k in ("word_rate", "envelope")},
                    "per_listener": {"listeners": subs, **{k: v.tolist() for k, v in z.items()}}}
    return out


def outcome(n: int, ve: dict = None, lang: dict = None, beats: tuple = None) -> list:
    """The pre-written lines for what happened (rule section 7); numbers copied from the output."""
    if n < MIN_N:
        return [LINES["coverage"].format(n=n)]
    if ve["claim_k"] >= 1:
        lines = [LINES["claim"].format(n=n, point=round(ve["point_k"]), claim=ve["claim_k"])]
    elif lang["p"] < 0.05:
        lines = [LINES["below_one"].format(n=n, z=f"{lang['z']:.2f}", p=f"{lang['p']:.2g}", point=round(ve["point_k"]))]
    else:
        lines = [LINES["no_bar"].format(n=n, z=f"{lang['z']:.2f}", p=f"{lang['p']:.2g}")]
    return lines + [LINES["beat"].format(k=beats[0], K=beats[1])]


# ---- the steps ---------------------------------------------------------------------------------------------------------------------
def download(root):
    d = f"{root}/alice/bold"; os.makedirs(d, exist_ok=True); got = {}
    for s in SUBS:
        for rel in (f"derivatives/sub-{s}/sub-{s}_task-alice_bold_preprocessed.nii.gz", f"sub-{s}/sub-{s}_task-alice_events.tsv"):
            dst = f"{d}/{os.path.basename(rel)}"
            if not os.path.exists(dst):
                urllib.request.urlretrieve(f"{BASE}/{rel}", dst)
            got[os.path.basename(rel)] = sha(dst)
        print(f"sub-{s}: derivative and events", flush=True)
    json.dump(got, open(f"{d}/SHA256.json", "w"), indent=1)


def onset_of(events_path) -> tuple:
    r = next(csv.DictReader(open(events_path), delimiter="\t"))
    return float(r["onset"]), r["stim_file"], float(r["duration"])


def score(root, out_dir):
    assert not os.path.abspath(out_dir).startswith(REPO), "OUT must be outside the repo (per-listener z)"
    A = f"{root}/alice"; os.makedirs(out_dir, exist_ok=True)
    assert sha(f"{A}/stimulus/{STIM}") == STIM_SHA and sha(f"{A}/annotations/annotations.tsv") == ANN_SHA
    sets = H.parcel_sets(f"{root}/atlas/lh.Schaefer2018_400Parcels_17Networks_order.annot",
                         f"{root}/atlas/rh.Schaefer2018_400Parcels_17Networks_order.annot")
    P = {k: np.load(f"{A}/forecast/out/pred_alice_{k}.npz") for k in ("full", "audio", "shuf")}
    ev = {s: onset_of(f"{A}/bold/sub-{s}_task-alice_events.tsv") for s in SCORED}
    vert, covs, table = {}, {}, {}
    for s in SCORED:
        vert[s] = project(f"{A}/bold/sub-{s}_task-alice_bold_preprocessed.nii.gz"); covs[s] = covered(vert[s])
        sh = area_shares(covs[s], sets)
        table[s] = {"protocol": PROTOCOL[s], "onset_s": ev[s][0], "stim_file": ev[s][1], "volumes": int(vert[s].shape[1]),
                    "covered_share": {k: round(v, 4) for k, v in sh.items()}, "language": sh["LANG"] >= COVER, "hearing": sh["AUD"] >= COVER}
        print(f"sub-{s}: {vert[s].shape[1]} volumes, covered LANG {sh['LANG']:.3f} AUD {sh['AUD']:.3f}", flush=True)
    elig = {"LANG": [s for s in SCORED if table[s]["language"]], "AUD": [s for s in SCORED if table[s]["hearing"]]}
    res = {"label": "exploratory", "rule": "docs/rules/alice_heldout_rule.md", "dataset": "OpenNeuro ds002322 v1.0.4 (CC0)",
           "listeners": table, "no_derivative": list(NO_DERIVATIVE),
           "excluded": {f"sub-{s}": r for s, r in EXCLUDED.items()}, "tree_baselines": TREE_LABEL,
           "stim_file_note": {s: v["stim_file"] for s, v in table.items() if v["stim_file"] != STIM},
           "not_run": "section 8's M2 (hybrid) and M3 (wrong story): they need a second story's listener group and are not part of the claim",
           "inputs_sha256": {"stimulus": STIM_SHA, "annotations": ANN_SHA, **{f"pred_{k}": sha(f"{A}/forecast/out/pred_alice_{k}.npz") for k in P}}}
    n_lang = len(elig["LANG"])
    if n_lang < MIN_N:
        res["lines"] = outcome(n_lang)
        return _write(res, res, out_dir)
    keep = kept_parcels(covs, sets, elig); ksets = subset(sets, keep)
    res["parcels_kept"] = {net: f"{len(idx)} of {len(sets[net])}" for net, idx in keep.items()}
    n = SKIP + min(v.shape[1] for v in vert.values())
    bold = {s: {net: A_[:, :n] for net, A_ in to_raw(parcel_series(vert[s], covs[s], sets, keep)).items()} for s in SCORED}
    words = timed_words(f"{A}/annotations/annotations.tsv"); on = np.array([t for _, t, _ in words])
    n_sec = int(max(on.max(), n * H.TR)) + 40
    wps = np.bincount(on.astype(int), minlength=n_sec).astype(float)
    env = H.envelope_per_second(f"{A}/stimulus/{STIM}")
    base = baselines(words, json.load(open(f"{A}/surprisal.json")), n_sec)
    o = {s: ev[s][0] for s in SCORED}
    pp = {s: {k: H.pred_parcels(d["P"], d["starts"] + o[s], n, ksets) for k, d in P.items()} for s in SCORED}
    last = min(int((float(P["full"]["starts"][-1]) + o[s] + 5) // H.TR) for s in SCORED)
    full = {"stage": {}, "cheap": {}}
    for tag, j1 in (("145", 154), ("full", min(last, n - 1))):
        aud16 = sorted(elig["AUD"])[:16]
        Lown, _ = own_best_lag_onsets([bold[s]["AUD"] for s in aud16], [o[s] for s in aud16], env, n, j1, 0.03)
        for lab, L in (("primary", DEV_LAG), ("own_lag", Lown)):
            per = {s: H.stage_scores({s: bold[s]}, pp[s], wps, env, base, n, j1, L + o[s]) for s in SCORED}
            full["stage"][f"{tag}_{lab}"] = group_stage(per, L, j1, sorted(elig["AUD"]), sorted(elig["LANG"]))
        per = {s: CB.section8(H, {s: bold[s]}, pp[s], wps, env, n, j1, DEV_LAG + o[s]) for s in SCORED}
        full["cheap"][tag] = group_cheap(per, elig)
    T145 = 154 - H.J0 + 1
    cl = lambda X: H.clean(X, 0.03, H.J0, 154)
    ctrl = {s: {"AUD": H.zeros_control(T145), "LANG": [cl(pp[s]["audio"]["LANG"]).mean(0, keepdims=True),
                                                       cl(H.wordrate_1hz(on + o[s], n)[None])]} for s in SCORED}
    m1 = isc_curve_onsets(bold, ctrl, [k for k in M1_KS if k < n_lang - 1] + [n_lang - 1], M1_REPS, M1_SEED, elig)
    full["m1"] = {net: {str(k): v for k, v in r.items()} for net, r in m1.items()}
    ks = [k for k in M1_KS if k < n_lang - 1] + [n_lang - 1]
    curve = [float(np.mean([z for z, _ in m1["LANG"][k]])) for k in ks]
    ve = H.volunteer_equivalent(full["stage"]["145_primary"]["per_listener"]["language"], ks, curve)
    verdicts = [c["verdict"] for t in full["cheap"].values() for net in ("LANG", "AUD") for c in t[net]["tribe_vs"].values()]
    pub = {**res, "volunteer_equivalent_language": ve,
           "stage": {k: {kk: vv for kk, vv in v.items() if kk != "per_listener"} for k, v in full["stage"].items()},
           "cheap_baselines": {t: {net: {kk: vv for kk, vv in r[net].items() if kk != "per_listener"} for net in ("LANG", "AUD")}
                               for t, r in full["cheap"].items()},
           "m1_language_curve": dict(zip(map(str, ks), curve)),
           "lines": outcome(n_lang, ve, full["stage"]["145_primary"]["language"], (verdicts.count("TRIBE beats"), len(verdicts)))}
    return _write(pub, {**pub, **full}, out_dir)


def _write(pub, full, out_dir):
    json.dump(full, open(f"{out_dir}/alice_heldout.json", "w"), indent=1)
    json.dump(pub, open(f"{REPO}/results/alice_heldout.json", "w"), indent=1)
    for line in pub["lines"]:
        print(line, flush=True)
    return pub


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("download", "surprisal", "score"))
    ap.add_argument("--root", required=True); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.step == "download":
        download(a.root)
    elif a.step == "surprisal":
        w = timed_words(f"{a.root}/alice/annotations/annotations.tsv")
        json.dump(gpt2_surprisal(w), open(f"{a.root}/alice/surprisal.json", "w"))
        print(len(w), "words")
    else:
        score(a.root, a.out or f"{a.root}/alice/score")
