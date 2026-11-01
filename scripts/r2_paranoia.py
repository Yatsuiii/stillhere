"""R2 (docs/rules/r2_paranoia_rule.md; exploratory): stock and fine-tuned TRIBE v2 against the Paranoia listeners (Finn et al. 2018,
OpenNeuro ds001338), the whole story: the three runs cleaned one at a time and joined, one window, one line per model.

Section 8's math and Alice's group step, at TR 1 s. heldout.py's functions that read its module TR (clean's t_r, align's and
wordrate_1hz's volume times) run with TR 1 s (`tr1`); every gauge is heldout.gauge's definition with the projector onto [controls, 1]
applied in its factored form and R2's 30-TR shift minimum (`gauge`; the same numbers to rounding, 14 to 33 times faster at 1,250
volumes, which the M1 draws need); Alice's group functions (group_stage, group_cheap, kept_parcels, parcel_series, covered,
area_shares) are called unchanged, and the lines take alice_heldout.outcome's number formats. With one run, every per-run function here reproduces heldout.py and
alice_heldout.py at TR 1 s (tests/test_r2_paranoia.py, synthetic data only).

Timing (rule section 5): run r's volume j is at scan time j s and story time j - 2 s; TRIBE's 5 s offset is unchanged; regressors are
made at 1 Hz and read at story time j - 2 - L. The forecast read is FORECAST_READ (below): rule section 5 reads "at lag L volume j reads
story time j - 2 - 5 - L s of run r's forecast", while section 8 and Alice move only the regressors with L. Set to "section8" by the rule's
dated clarification of Nov 1 (Yatsu's decision, about 05:55 UTC, before any Paranoia brain data was downloaded): L moves only the
word-rate regressor and its derivative; every forecast is read at story time j - 2 - 5 s.

Steps, in the rule's order:
  surprisal  GPT-2 small surprisal of each story's word table (alice_heldout.gpt2_surprisal, p1/surprisal8.py's method)
  download   the 132 fsaverage5 GIFTI files and the three events.tsv (refuses until the forecasts' commit has filled the pins; run
             only after the cloud's go)
  score      one scoring run; writes OUT/r2_paranoia.json (per-listener z, so OUT is outside this repo) and results/r2_paranoia.json
             (group numbers, the lines, and per listener only volumes, coverage and the areas scored)
usage: python scripts/r2_paranoia.py {surprisal|download|score} --root ../stillhere [--out ../stillhere/r2/score]"""
import argparse
import collections
import contextlib
import csv
import hashlib
import json
import os
import string
import sys
import urllib.request

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
import alice_heldout as AH  # noqa: E402
from stillhere import heldout as H  # noqa: E402

RULE = "docs/rules/r2_paranoia_rule.md"
DATASET = "OpenNeuro ds001338 v1.0.0 (Finn et al. 2018; CC BY-NC 4.0), fMRIPrep 22.0.2 fsaverage5 derivative"
RAW = "https://s3.amazonaws.com/openneuro.org/ds001338/"
DER = "https://openneuro-derivatives.s3.amazonaws.com/fmriprep/ds001338-fmriprep/"
SUBS = tuple(json.load(open(os.path.join(REPO, "r2/data_check.json")))["participants"])
STORIES = ("story1", "story2", "story3")
MODELS = {"stock": "stock TRIBE v2", "tuned": "fine-tuned TRIBE v2"}
PREFIX = {"stock": "", "tuned": "Fine-tuned TRIBE v2: "}
MODES = ("full", "audio", "shuf")
TR, ONSET, J0, MIN, HP = 1.0, 2.0, 20, 30, 0.03   # section 5: TR 1 s, story onset 2 s, first 20 volumes dropped, 30-TR null minimum
DEV_LAG, COVER, MIN_N = AH.DEV_LAG, AH.COVER, AH.MIN_N
M1_KS, M1_REPS, M1_SEED = AH.M1_KS, AH.M1_REPS, AH.M1_SEED
LAG_GRID = np.arange(-4, 1.01, 0.25)              # heldout.own_best_lag's grid
READS = {"section8": "every forecast read at story time j - 2 - 5 s; the lag moves only the regressors (section 8 and Alice)",
         "lagged": "rule section 5 as written: a gauge at lag L reads LANGUAGE's forecasts at story time j - 2 - 5 - L s; HEARING's "
                   "audio-only forecast at lag 0 (section 6); M1's controls at lag 0 (section 8's M1 has no lag)"}
FORECAST_READ = "section8"   # rule r2_paranoia_rule.md, dated clarification of Nov 1 (Yatsu, about 05:55 UTC, no Paranoia data downloaded)

# Filled by the forecasts' commit (rule order: forecasts, then a commit with these SHA-256 and nothing else, then the download)
STIM_PINS = {}         # story -> SHA-256 of stimuli/<story>_audio.wav
WORD_PINS = {}         # story -> SHA-256 of forecast/out/words_<story>.tsv
SURPRISAL_PIN = ""     # SHA-256 of r2/surprisal.json
FORECAST_PINS = {}     # f"{model}_{story}_{mode}" -> SHA-256 of forecast/out/pred_<that>.npz

LINES = {   # docs/rules/r2_paranoia_rule.md section 7, word for word
    "claim": "On another public story, outside TRIBE v2's training and evaluation sets (Paranoia, Finn et al. 2018; {n} listeners, the "
             "whole story: {T} volumes = {M} minutes at TR 1 s), {model}'s forecast was worth about {point} real volunteers, and at least "
             "{claim}. It reached {share}% of what the other listeners' average reaches.",
    "below_one": "On another public story, outside TRIBE v2's training and evaluation sets (Paranoia, Finn et al. 2018; {n} listeners, "
                 "the whole story: {T} volumes = {M} minutes at TR 1 s), {model}'s forecast tracked the listeners' language areas (z {z}, "
                 "p = {p}), but at its lower bound it was worth less than one real volunteer (about {point} at the point estimate). It "
                 "reached {share}% of what the other listeners' average reaches.",
    "no_bar": "On another public story, outside TRIBE v2's training and evaluation sets (Paranoia, Finn et al. 2018; {n} listeners, the "
              "whole story: {T} volumes = {M} minutes at TR 1 s), {model}'s forecast did not reach the bar in language areas (z {z}, p = "
              "{p}); it reached {share}% of what the other listeners' average reaches. We show it because we said we would.",
    "coverage": "Paranoia: rule committed, not run (coverage): {n} listeners had the language areas in view, fewer than 10.",
    "not_run": "Paranoia: rule committed, not run.",
    "beat": "What it had to beat: word rate and the sound envelope (beaten in {k} of {K} comparisons).",
    "pair": "z is not compared across datasets with different scan timing; the share is.",
}


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


# ---- heldout at TR 1 s -------------------------------------------------------------------------------------------------------------
@contextlib.contextmanager
def tr1():
    """heldout's module TR at R2's 1 s for the call (clean's t_r, align's and wordrate_1hz's volume times), restored after."""
    old = H.TR; H.TR = TR
    try:
        yield
    finally:
        H.TR = old


def clean(A, j0: int, j1: int) -> np.ndarray:
    """heldout.clean at t_r 1 s: rows of A on volumes j0..j1, linear detrend, 0.03 Hz high-pass, z-score (ddof 1)."""
    with tr1():
        return H.clean(A, HP, j0, j1)


def cj(rows: dict, W: dict) -> np.ndarray:
    """Rows per run ({run: (p, n_run)}): each run cleaned on its own on volumes J0..W[run], then joined in story order (W's order)."""
    return np.hstack([clean(rows[r], J0, j1) for r, j1 in W.items()])


def gauge(B, X, controls) -> tuple:
    """heldout.gauge with R2's 30-TR shift minimum: (z, real), the Fisher-z mean over parcels of the partial r of z-scored BOLD rows B
    with regressor rows X given [controls, 1], against every circular shift of X by MIN..T-MIN-1. The projector C pinv(C'C) C' is applied
    as (A C) pinv(C'C) C', the same algebra without the T x T matrix."""
    T = B.shape[1]
    C = np.vstack(list(controls) + [np.ones((1, T))]).T
    M = np.linalg.pinv(C.T @ C)
    proj = lambda A: ((A @ C) @ M) @ C.T
    rb = H.zrow(B)
    rb = rb - proj(rb)
    shifts = np.r_[0, np.arange(MIN, T - MIN)]
    Xs = np.asarray(X)[:, (np.arange(T)[None, :] - shifts[:, None]) % T].transpose(1, 0, 2)
    s = H._fz_rows(rb, Xs - proj(Xs))
    return float((s[0] - s[1:].mean()) / s[1:].std()), float(s[0])


def story_times(n: int, lag: float = 0.0) -> np.ndarray:
    """Story time (s) at which volume j's regressors are read: scan time j s, story onset 2 s, lag L (L < 0 reads a later time)."""
    return np.arange(n) * TR - ONSET - lag


def pred_parcels(P, starts, n: int, sets: dict, shift: float = 0.0) -> dict:
    """{area: (parcels, n)}: volume j reads story time j - 2 - 5 - shift of the run's forecast (heldout.align at TR 1 s, linear between
    1 Hz samples, NaN outside them), each parcel the mean over its vertices."""
    with tr1():
        return H.pred_parcels(P, np.asarray(starts, float) + ONSET + shift, n, sets)


def wordrate_1hz(onsets, n: int) -> np.ndarray:
    """heldout.wordrate_1hz at TR 1 s with the story's onsets moved to scan time (M1's word-rate control: read at scan time j, no lag)."""
    with tr1():
        return H.wordrate_1hz(np.asarray(onsets, float) + ONSET, n)


def fshift(L: float) -> float:
    """How far a LANGUAGE gauge at lag L moves its forecasts (FORECAST_READ)."""
    if FORECAST_READ not in READS:
        raise SystemExit(f"refused: FORECAST_READ is {FORECAST_READ!r}; rule section 5 and section 8's practice differ on whether the "
                         f"lag moves the forecast, and a commit must set one of {sorted(READS)} before scoring")
    return float(L) if FORECAST_READ == "lagged" else 0.0


# ---- data --------------------------------------------------------------------------------------------------------------------------
def gifti_name(s, story, hemi) -> str:
    return f"{s}_task-{story}_hemi-{hemi}_space-fsaverage5_bold.func.gii"


def _structure(img) -> str:
    for meta in [img.meta] + [d.meta for d in img.darrays[:1]]:
        v = dict(meta or {}).get("AnatomicalStructurePrimary")
        if v:
            return v
    return None


def load_run(bold_dir, s, story) -> np.ndarray:
    """(20484, T) float32: the run's left then right fsaverage5 GIFTI, TRIBE's vertex order. Refuses a hemisphere that is not 10,242
    vertices of CortexLeft (L) or CortexRight (R), and hemispheres that differ in volume count."""
    import nibabel as nb
    out = []
    for hemi, want in (("L", "CortexLeft"), ("R", "CortexRight")):
        img = nb.load(os.path.join(bold_dir, gifti_name(s, story, hemi)))
        a = np.column_stack([d.data for d in img.darrays]).astype(np.float32)
        if a.shape[0] != 10242 or _structure(img) != want:
            raise SystemExit(f"refused: {gifti_name(s, story, hemi)} holds {a.shape[0]} vertices of {_structure(img)}, not 10242 of {want}")
        out.append(a)
    if out[0].shape[1] != out[1].shape[1]:
        raise SystemExit(f"refused: {s} {story}: the hemispheres differ in volume count")
    return np.vstack(out)


def medial_from_labels(labels_lh, names_lh, labels_rh, names_rh) -> np.ndarray:
    """(20484,) True for Schaefer-400's unlabelled vertices (the medial wall): no 17Networks_ parcel, left then right."""
    def unl(labels, names):
        names = [n.decode() if isinstance(n, bytes) else n for n in names]
        return np.array([not (0 <= i < len(names) and names[i].startswith("17Networks_")) for i in labels])
    return np.concatenate([unl(labels_lh, names_lh), unl(labels_rh, names_rh)])


def medial_wall(annot_lh, annot_rh) -> np.ndarray:
    import nibabel.freesurfer as fs
    (ll, _, nl), (lr, _, nr) = fs.read_annot(annot_lh), fs.read_annot(annot_rh)
    return medial_from_labels(ll, nl, lr, nr)


def coverage(runs: dict, medial) -> np.ndarray:
    """A vertex counts as covered for a listener only if it is covered (finite, not constant: alice_heldout.covered) in every run and is
    not on the medial wall."""
    cov = ~np.asarray(medial, bool)
    for v in runs.values():
        cov &= AH.covered(v)
    return cov


def norm_word(w) -> str:
    """A word as the word baselines read it: lower case, without the punctuation speech-to-text attaches at its edges (the LPP TextGrid
    and the Alice annotations, where these baselines were defined, carry none); a word that is only punctuation stays as it is."""
    w = str(w).strip().lower()
    return w.strip(string.punctuation + "\u201c\u201d\u2018\u2019\u2026\u2013\u2014") or w


def word_table(path) -> list:
    """[(word as the baselines read it, onset s)] from a word table (ExtractWordsFromAudio's tab-separated format), in file order."""
    with open(path, newline="") as f:
        return [(norm_word(r["text"]), float(r["start"])) for r in csv.DictReader(f, delimiter="\t")]


def baselines(words, surprisal: dict, n_sec: int) -> dict:
    """The rule's word baselines for one story, defined as section 8 defined them (heldout.baseline_series): log word frequency within
    the story's word table, and GPT-2 small surprisal; no parse-tree baselines (the dataset has no parses)."""
    a = [w for w, _ in words]
    t_on = [t for _, t in words]
    cnt = collections.Counter(a)
    return {"B_freq": H.per_second([np.log10(cnt[w] / len(a)) for w in a], t_on, n_sec),
            "surprisal": H.per_second(surprisal["surprisal_bits"], surprisal["onsets"], n_sec)}


# ---- the per-listener gauges on the joined window -----------------------------------------------------------------------------------
def joined_forecasts(pp: dict, W: dict) -> dict:
    """{(mode, area): cleaned joined rows} from {run: {mode: {area: (p, n_run)}}}."""
    r0 = next(iter(W))
    return {(k, net): cj({r: pp[r][k][net] for r in W}, W) for k in pp[r0] for net in pp[r0][k]}


def regressors(stories: dict, W: dict, L: float) -> dict:
    """Section 8's stage regressors at lag L, made per run at 1 Hz from that story's words and sound, read at story time j - 2 - L,
    cleaned per run and joined: word rate, its derivative, the envelope, and each word baseline. stories: {run: {"n", "wps", "env",
    "base"}}."""
    at = lambda key, d=0.0: cj({r: H.hrf_regressor(stories[r][key], story_times(stories[r]["n"], L) + d)[None] for r in W}, W)
    out = {"wr": at("wps"), "dwr": cj({r: (H.hrf_regressor(stories[r]["wps"], story_times(stories[r]["n"], L) + 0.5)
                                         - H.hrf_regressor(stories[r]["wps"], story_times(stories[r]["n"], L) - 0.5))[None] for r in W}, W),
           "env": at("env")}
    for k in stories[next(iter(W))]["base"]:
        out[f"base_{k}"] = cj({r: H.hrf_regressor(stories[r]["base"][k], story_times(stories[r]["n"], L))[None] for r in W}, W)
    return out


def listener_stage(s, b: dict, F_aud: dict, F_lang: dict, reg: dict) -> dict:
    """heldout.stage_scores' z for listener s on the joined window, in the shape alice_heldout.group_stage reads. b: {"AUD", "LANG"}
    cleaned joined BOLD rows; F_aud: the HEARING forecasts (lag 0); F_lang: the LANGUAGE forecasts at this lag; reg: regressors()."""
    T = b["AUD"].shape[1]
    zc = H.zeros_control(T)
    paud = F_aud[("audio", "AUD")]
    zh = gauge(b["AUD"], paud, zc)[0]
    ez = gauge(b["AUD"], np.tile(reg["env"], (paud.shape[0], 1)), zc)[0]
    tc = lang_controls(F_lang, reg)
    pfl = F_lang[("full", "LANG")]
    out = {"listeners": [s], "hearing": [zh], "envelope": [ez], "language": [gauge(b["LANG"], pfl, tc)[0]],
           "shuffled": [gauge(b["LANG"], F_lang[("shuf", "LANG")], tc)[0]]}
    bctl = [reg["env"], reg["wr"], reg["dwr"]]
    for k in [k for k in reg if k.startswith("base_")]:
        out[k] = [gauge(b["LANG"], np.tile(reg[k], (pfl.shape[0], 1)), bctl)[0]]
    return {"per_listener": out}


def lang_controls(F_lang: dict, reg: dict) -> list:
    """LANGUAGE's controls: the LANG-mean audio-only forecast, word rate and its derivative at the gauge's lag."""
    return [F_lang[("audio", "LANG")].mean(0, keepdims=True), reg["wr"], reg["dwr"]]


def listener_cheap(b: dict, F_aud: dict, F_lang: dict, reg: dict) -> dict:
    """cheap_baselines.section8's per-listener z on the joined window: LANG with the full forecast, AUD with the audio-only forecast,
    word rate and envelope, all unpartialled, in the shape alice_heldout.group_cheap reads."""
    zc = H.zeros_control(b["AUD"].shape[1]); out = {}
    for net, X in (("LANG", F_lang[("full", "LANG")]), ("AUD", F_aud[("audio", "AUD")])):
        xs = {"tribe": X, "word_rate": np.tile(reg["wr"], (X.shape[0], 1)), "envelope": np.tile(reg["env"], (X.shape[0], 1))}
        out[net] = {"per_listener": {k: [gauge(b[net], x, zc)[0]] for k, x in xs.items()}}
    return out


def own_best_lag(aud: list, stories: dict, W: dict, grid=LAG_GRID) -> tuple:
    """heldout.own_best_lag on the joined window: the grid lag maximising the mean envelope HEARING-style z over the given listeners'
    cleaned joined AUD rows, the envelope regressor made per run at that lag, cleaned per run and joined."""
    zz = []
    for lag in grid:
        reg = cj({r: H.hrf_regressor(stories[r]["env"], story_times(stories[r]["n"], lag))[None] for r in W}, W)
        zz.append(np.mean([gauge(b, np.tile(reg, (b.shape[0], 1)), H.zeros_control(b.shape[1]))[0] for b in aud]))
    return float(grid[int(np.argmax(zz))]), [float(x) for x in zz]


def isc_draws(subs_by: dict, ks, reps: int, seed: int) -> dict:
    """heldout.isc_curve's random sets, drawn from one Generator(seed) in its order (area, listener, k, repeat; one set, no draw, when k
    is all the others)."""
    rng = np.random.default_rng(seed); out = {}
    for net in ("AUD", "LANG"):
        subs = sorted(subs_by[net]); out[net] = {}
        for s in subs:
            others = [t for t in subs if t != s]; out[net][s] = {}
            for k in ks:
                out[net][s][k] = [others] if k == len(others) else [list(rng.choice(others, k, replace=False)) for _ in range(reps)]
    return out


def isc_curve(bold: dict, Bj: dict, ctrl: dict, draws: dict, W: dict, nets=("AUD", "LANG")) -> dict:
    """heldout.isc_curve (M1) on the joined window: per listener and k, mean z and detection share over the drawn sets; the reference is
    the mean of the set's raw parcel series, run by run, cleaned per run and joined, scored with that listener's controls. bold: {s:
    {run: {area: raw rows}}}; Bj: {s: {area: cleaned joined rows}}; ctrl: {s: {area: control rows}}."""
    out = {}
    for net in nets:
        res = {}
        for s, by_k in draws[net].items():
            for k, picks in by_k.items():
                zs = [gauge(Bj[s][net], cj({r: np.mean([bold[t][r][net] for t in pick], 0) for r in W}, W), ctrl[s][net])[0] for pick in picks]
                res.setdefault(k, []).append((float(np.mean(zs)), float(np.mean(np.array(zs) > H.DETECT))))
        out[net] = res
    return out


def reference(Bj: dict, subs: list, controls: list) -> dict:
    """The leave-one-out reference (rule section 6): per listener, the other listeners' average cleaned joined LANG series used as the
    forecast, with LANGUAGE's gauge, controls and null."""
    S = {s: Bj[s]["LANG"] for s in subs}
    return {s: gauge(S[s], np.mean([S[t] for t in subs if t != s], 0), controls)[0] for s in subs}


def share(model_z: dict, ref_z: dict) -> dict:
    """The model's mean LANGUAGE z over the reference's listeners divided by the reference's mean z there (both reported)."""
    subs = sorted(ref_z)
    m, r = float(np.mean([model_z[s] for s in subs])), float(np.mean([ref_z[s] for s in subs]))
    return {"n": len(subs), "model_mean_z": m, "reference_mean_z": r, "share": m / r}


def last_volume(pp_runs: dict, n: int) -> int:
    """The run's last volume with brain data (n volumes) and every forecast finite in every parcel."""
    ok = np.ones(n, bool)
    for by_net in pp_runs.values():
        for A in by_net.values():
            ok &= np.isfinite(A).all(0)
    idx = np.where(ok[:n])[0]
    return int(idx.max())


def outcome(model: str, n: int, T: int = 0, ve: dict = None, lang: dict = None, sh: float = None, beats: tuple = None) -> list:
    """The rule's lines for one model (section 7); numbers copied from the output. Fine-tuned lines start with its prefix."""
    if n < MIN_N:
        return [LINES["coverage"].format(n=n)]
    f = {"n": n, "T": f"{T:,}", "M": f"{T * TR / 60:.1f}", "model": MODELS[model], "share": f"{100 * sh:.0f}"}
    if ve["claim_k"] >= 1:
        line = LINES["claim"].format(point=round(ve["point_k"]), claim=ve["claim_k"], **f)
    elif lang["p"] < 0.05:
        line = LINES["below_one"].format(z=f"{lang['z']:.2f}", p=f"{lang['p']:.2g}", point=round(ve["point_k"]), **f)
    else:
        line = LINES["no_bar"].format(z=f"{lang['z']:.2f}", p=f"{lang['p']:.2g}", **f)
    return [PREFIX[model] + line, PREFIX[model] + LINES["beat"].format(k=beats[0], K=beats[1])]


# ---- the steps ---------------------------------------------------------------------------------------------------------------------
def check_pins(root) -> dict:
    """Refuses unless the forecasts' commit has filled every pin and every input matches it."""
    if not (STIM_PINS and WORD_PINS and SURPRISAL_PIN and FORECAST_PINS):
        raise SystemExit("refused: the stimuli', word tables', surprisal file's and forecasts' SHA-256 are not pinned yet (the forecasts' "
                         "commit comes before the download and the scoring run)")
    R = f"{root}/r2"
    got = {**{f"stimulus {s}": sha(f"{R}/stimuli/{s}_audio.wav") for s in STORIES},
           **{f"words {s}": sha(f"{R}/forecast/out/words_{s}.tsv") for s in STORIES}, "surprisal": sha(f"{R}/surprisal.json"),
           **{f"pred {k}": sha(f"{R}/forecast/out/pred_{k}.npz") for k in FORECAST_PINS}}
    want = {**{f"stimulus {s}": STIM_PINS.get(s) for s in STORIES}, **{f"words {s}": WORD_PINS.get(s) for s in STORIES},
            "surprisal": SURPRISAL_PIN, **{f"pred {k}": v for k, v in FORECAST_PINS.items()}}
    need = {f"{m}_{s}_{k}" for m in MODELS for s in STORIES for k in MODES}
    bad = sorted(k for k in want if got[k] != want[k]) + sorted(f"pred {k} not pinned" for k in need - set(FORECAST_PINS))
    if bad:
        raise SystemExit(f"refused: inputs differ from the pins: {bad}")
    return got


def download(root):
    check_pins(root)
    d = f"{root}/r2/bold"; os.makedirs(d, exist_ok=True); got = {}
    for story in STORIES:
        name = f"task-{story}_events.tsv"
        urllib.request.urlretrieve(RAW + name, f"{d}/{name}"); got[name] = sha(f"{d}/{name}")
    for s in SUBS:
        for story in STORIES:
            for hemi in ("L", "R"):
                name = gifti_name(s, story, hemi); dst = f"{d}/{name}"
                if not os.path.exists(dst):
                    urllib.request.urlretrieve(f"{DER}{s}/func/{name}", dst)
                got[name] = sha(dst)
        print(f"{s}: 6 files", flush=True)
    json.dump(got, open(f"{d}/SHA256.json", "w"), indent=1)


def story_onset(events_path, story) -> float:
    r = next(csv.DictReader(open(events_path), delimiter="\t"))
    if float(r["onset"]) != ONSET or r["stim_file"] != f"{story}_audio.wav":
        raise SystemExit(f"refused: {events_path}: onset {r['onset']}, stim_file {r['stim_file']} (the rule's timing assumes 2 s, {story}_audio.wav)")
    return float(r["onset"])


def score(root, out_dir):
    assert not os.path.abspath(out_dir).startswith(REPO), "OUT must be outside the repo (per-listener z)"
    fshift(0.0)                                                       # refuses until FORECAST_READ is set
    pins = check_pins(root)
    R = f"{root}/r2"; os.makedirs(out_dir, exist_ok=True)
    for story in STORIES:
        story_onset(f"{R}/bold/task-{story}_events.tsv", story)
    sets = H.parcel_sets(f"{root}/atlas/lh.Schaefer2018_400Parcels_17Networks_order.annot",
                         f"{root}/atlas/rh.Schaefer2018_400Parcels_17Networks_order.annot")
    medial = medial_wall(f"{root}/atlas/lh.Schaefer2018_400Parcels_17Networks_order.annot",
                         f"{root}/atlas/rh.Schaefer2018_400Parcels_17Networks_order.annot")
    vert, covs, table = {}, {}, {}
    for s in SUBS:
        vert[s] = {story: load_run(f"{R}/bold", s, story) for story in STORIES}
        covs[s] = coverage(vert[s], medial)
        sh = AH.area_shares(covs[s], sets)
        table[s] = {"volumes": {story: int(v.shape[1]) for story, v in vert[s].items()},
                    "covered_share": {**{k: round(v, 4) for k, v in sh.items()}, "cortex": round(float(covs[s][~medial].mean()), 4)},
                    "language": sh["LANG"] >= COVER, "hearing": sh["AUD"] >= COVER}
        print(f"{s}: volumes {table[s]['volumes']}, covered LANG {sh['LANG']:.3f} AUD {sh['AUD']:.3f}", flush=True)
    elig = {"LANG": sorted(s for s in SUBS if table[s]["language"]), "AUD": sorted(s for s in SUBS if table[s]["hearing"])}
    res = {"label": "exploratory", "rule": RULE, "dataset": DATASET, "listeners": table, "forecast_read": {FORECAST_READ: READS[FORECAST_READ]},
           "not_run": "section 8's M2 (hybrid) and M3 (wrong story)", "inputs_sha256": pins}
    n_lang = len(elig["LANG"])
    if n_lang < MIN_N:
        res["lines"] = outcome("stock", n_lang)
        return _write(res, res, out_dir)
    keep = AH.kept_parcels(covs, sets, elig); ksets = AH.subset(sets, keep)
    res["parcels_kept"] = {net: f"{len(idx)} of {len(sets[net])}" for net, idx in keep.items()}
    n = {story: min(vert[s][story].shape[1] for s in SUBS) for story in STORIES}
    bold = {s: {story: {net: A[:, :n[story]] for net, A in AH.parcel_series(vert[s][story], covs[s], sets, keep).items()}
                for story in STORIES} for s in SUBS}
    del vert
    surp = json.load(open(f"{R}/surprisal.json"))
    stories = {}
    for story in STORIES:
        words = word_table(f"{R}/forecast/out/words_{story}.tsv"); on = np.array([t for _, t in words])
        n_sec = int(max(on.max(), n[story] * TR)) + 40
        stories[story] = {"n": n[story], "onsets": on, "wps": np.bincount(on.astype(int), minlength=n_sec).astype(float),
                          "env": H.envelope_per_second(f"{R}/stimuli/{story}_audio.wav"), "base": baselines(words, surp[story], n_sec)}
    P = {m: {story: {k: np.load(f"{R}/forecast/out/pred_{m}_{story}_{k}.npz") for k in MODES} for story in STORIES} for m in MODELS}
    pp = lambda m, story, shift: {k: pred_parcels(d["P"], d["starts"], n[story], ksets, shift) for k, d in P[m][story].items()}
    W = {story: min(n[story] - 1, min(last_volume(pp(m, story, 0.0), n[story]) for m in MODELS)) for story in STORIES}
    T = sum(j1 - J0 + 1 for j1 in W.values())
    res["window"] = {"volumes": {story: [J0, j1] for story, j1 in W.items()}, "T": T, "minutes": round(T * TR / 60, 1)}
    scored = sorted(set(elig["AUD"]) | set(elig["LANG"]))
    Bj = {s: {net: cj({story: bold[s][story][net] for story in W}, W) for net in ("AUD", "LANG")} for s in scored}

    def F(m, shift):
        runs = {story: pp(m, story, shift) for story in W}
        for story, by in runs.items():
            for k, d in by.items():
                for net, A in d.items():
                    if not np.isfinite(A[:, J0:W[story] + 1]).all():
                        raise SystemExit(f"refused: {m} {story} {k} {net} read outside its forecast at shift {shift}")
        return joined_forecasts(runs, W)
    aud16 = elig["AUD"][:16]
    Lown, lag_curve = own_best_lag([Bj[s]["AUD"] for s in aud16], stories, W)
    ks = [k for k in M1_KS if k < n_lang - 1] + [n_lang - 1]
    draws = isc_draws(elig, ks, M1_REPS, M1_SEED)
    m1_aud = None
    full, pub = {}, {**res, "own_best_lag": Lown, "models": {}}
    lines = []
    for m in MODELS:
        F0 = F(m, 0.0)
        stage = {}
        for lab, L in (("primary", DEV_LAG), ("own_lag", Lown)):
            reg = regressors(stories, W, L)
            FL = F0 if fshift(L) == 0.0 else F(m, fshift(L))
            per = {s: listener_stage(s, Bj[s], F0, FL, reg) for s in scored}
            stage[lab] = AH.group_stage(per, L, None, elig["AUD"], elig["LANG"])
            stage[lab]["volumes"] = res["window"]["volumes"]
            if lab == "primary":
                tc = lang_controls(FL, reg)
                ref = reference(Bj, elig["LANG"], tc)
                shr = share(dict(zip(elig["LANG"], stage[lab]["per_listener"]["language"])), ref)
                cheap = AH.group_cheap({s: listener_cheap(Bj[s], F0, FL, reg) for s in scored}, elig)
        wr1 = cj({st: wordrate_1hz(stories[st]["onsets"], n[st])[None] for st in W}, W)
        ctrl = {s: {"AUD": H.zeros_control(T), "LANG": [F0[("audio", "LANG")].mean(0, keepdims=True), wr1]} for s in scored}
        if m1_aud is None:
            m1_aud = isc_curve(bold, Bj, ctrl, draws, W, nets=("AUD",))["AUD"]
        m1 = {"AUD": m1_aud, **isc_curve(bold, Bj, ctrl, draws, W, nets=("LANG",))}
        curve = [float(np.mean([z for z, _ in m1["LANG"][k]])) for k in ks]
        ve = H.volunteer_equivalent(stage["primary"]["per_listener"]["language"], ks, curve)
        verdicts = [c["verdict"] for net in ("LANG", "AUD") for c in cheap[net]["tribe_vs"].values()]
        ml = outcome(m, n_lang, T, ve, stage["primary"]["language"], shr["share"], (verdicts.count("TRIBE beats"), len(verdicts)))
        lines += ml
        full[m] = {"stage": stage, "reference": {**shr, "per_listener": ref}, "cheap": cheap, "m1": {net: {str(k): v for k, v in r.items()} for net, r in m1.items()},
                   "volunteer_equivalent_language": ve}
        pub["models"][m] = {"model": MODELS[m], "volunteer_equivalent_language": ve, "reference": shr,
                            "stage": {k: {kk: vv for kk, vv in v.items() if kk != "per_listener"} for k, v in stage.items()},
                            "cheap_baselines": {net: {kk: vv for kk, vv in cheap[net].items() if kk != "per_listener"} for net in ("LANG", "AUD")},
                            "m1_language_curve": dict(zip(map(str, ks), curve)), "lines": ml}
    pub["lines"] = lines + [LINES["pair"]]
    pub["own_lag_curve"] = dict(zip(map(str, LAG_GRID.tolist()), lag_curve))
    return _write(pub, {**pub, "full": full}, out_dir)


def _write(pub, full, out_dir):
    json.dump(full, open(f"{out_dir}/r2_paranoia.json", "w"), indent=1, default=float)
    json.dump(pub, open(f"{REPO}/results/r2_paranoia.json", "w"), indent=1, default=float)
    for line in pub["lines"]:
        print(line, flush=True)
    return pub


def surprisal(root):
    R = f"{root}/r2"
    out = {story: AH.gpt2_surprisal([(w, t, None) for w, t in word_table(f"{R}/forecast/out/words_{story}.tsv")]) for story in STORIES}
    json.dump(out, open(f"{R}/surprisal.json", "w"))
    print({story: len(v["onsets"]) for story, v in out.items()}, "words", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("surprisal", "download", "score"))
    ap.add_argument("--root", required=True); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    {"surprisal": lambda: surprisal(a.root), "download": lambda: download(a.root),
     "score": lambda: score(a.root, a.out or f"{a.root}/r2/score")}[a.step]()
