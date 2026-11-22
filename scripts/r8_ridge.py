"""R8 (docs/rules/r8_ridge_rule.md; exploratory): a ridge model on word vectors, trained on Narratives, forecasting Paranoia in the R2
forecast file format so that scripts/r2_paranoia.py scores it unchanged (--verify-own).
  prep DATA      Narratives training stories: words, envelope, and the cleaned group-mean fsaverage5 BOLD per story (files deleted after use)
  fit DATA       features, leave-one-story-out alpha, ridge weights (full model and envelope-only model)
  predict DATA OWN   the nine Paranoia forecasts into OWN (pred_stock_<story>_<full|audio|shuf>.npz), from the R2 release word tables
DATA is outside this repository. Healthy volunteers only; not a diagnosis or medical advice."""
import csv
import io
import json
import os
import sys
import tempfile
import urllib.request

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from stillhere import heldout as H  # noqa: E402

B = "https://fcp-indi.s3.amazonaws.com/data/Projects/narratives"
O = "https://s3.amazonaws.com/openneuro.org/ds002345"
STORIES = ["pieman", "lucy", "tunnel", "slumlordreach", "merlin", "prettymouth", "black", "forgot"]
MAX_LISTENERS, TR_N, HP, DROP_S = 20, 1.5, 0.01, 10
DELAYS = [2, 3, 4, 5, 6, 7, 8]
ALPHAS = np.logspace(1, 6, 11)
SEED, OFFSET = 20261122, 5
NV = 10242


def get(url, dst=None):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                data = r.read()
            if dst:
                open(dst, "wb").write(data)
            return data
        except Exception as e:  # network retry
            last = e
    raise SystemExit(f"fetch failed: {url}: {last}")


def words_from_align(text):
    out = []
    for row in csv.reader(io.StringIO(text)):
        if len(row) >= 4 and row[2] not in ("", None):
            out.append((row[0], float(row[2])))
    return out


def listeners(story):
    part = list(csv.DictReader(io.StringIO(get(f"{O}/participants.tsv").decode()), delimiter="\t"))
    excl = json.loads(get(f"{B}/code/scan_exclude.json").decode()).get(story, {})
    subs = [p["participant_id"] for p in part if story in p["task"].split(",") and p["participant_id"] not in excl]
    return subs


def bold_keys(sub, story):
    xml = get(f"https://fcp-indi.s3.amazonaws.com/?prefix=data/Projects/narratives/derivatives/fmriprep/{sub}/func/{sub}_task-{story}_&list-type=2").decode()
    import re
    keys = sorted(set(re.findall(r"<Key>([^<]*space-fsaverage5_hemi-[LR]\.func\.gii)</Key>", xml)))
    runs = sorted({k.split("_space-")[0] for k in keys})
    if not runs:
        return None
    first = runs[0]
    return [f"{first}_space-fsaverage5_hemi-{h}.func.gii" for h in "LR"]


def load_gii(path):
    import nibabel as nib
    g = nib.load(path)
    return np.column_stack([d.data for d in g.darrays]).astype(np.float32)  # vertices x volumes


def clean_tr(A):
    from nilearn import signal
    return signal.clean(A.T, detrend=True, high_pass=HP, t_r=TR_N, standardize="zscore_sample").T.astype(np.float32)


def prep(data):
    os.makedirs(f"{data}/train", exist_ok=True)
    log = {}
    for story in STORIES:
        dst = f"{data}/train/{story}.npz"
        if os.path.exists(dst):
            continue
        words = words_from_align(get(f"{B}/stimuli/gentle/{story}/align.csv").decode())
        with tempfile.TemporaryDirectory() as td:
            get(f"{B}/stimuli/{story}_audio.wav", f"{td}/a.wav")
            env = H.envelope_per_second(f"{td}/a.wav")
        acc, n, used = None, None, []
        for sub in listeners(story):
            if len(used) == MAX_LISTENERS:
                break
            keys = bold_keys(sub, story)
            if not keys:
                continue
            with tempfile.TemporaryDirectory() as td:
                parts = []
                for k in keys:
                    get(f"https://fcp-indi.s3.amazonaws.com/{k}", f"{td}/h.gii")
                    parts.append(load_gii(f"{td}/h.gii"))
            X = np.vstack(parts)
            if X.shape[0] != 2 * NV:
                continue
            X = clean_tr(X)
            if acc is None:
                acc, n = X.astype(np.float64), X.shape[1]
            else:
                m = min(n, X.shape[1]); acc = acc[:, :m] + X[:, :m]; n = m
            used.append(sub)
            print(story, sub, X.shape, flush=True)
        mean = (acc / len(used)).astype(np.float32)
        np.savez_compressed(dst, bold=mean, onsets=np.array([t for _, t in words]), words=np.array([w for w, _ in words]), env=env)
        log[story] = {"listeners": used, "volumes": int(mean.shape[1]), "words": len(words)}
        json.dump(log, open(f"{data}/train/{story}.json", "w"), indent=1)


def vectors():
    import spacy
    nlp = spacy.load("en_core_web_md")
    return nlp.vocab, spacy.__version__, nlp.meta["version"]


def word_features(words, onsets, n_sec, vocab):
    V = np.zeros((n_sec, 300)); C = np.zeros(n_sec)
    for w, t in zip(words, onsets):
        k = int(t)
        if 0 <= k < n_sec:
            lex = vocab[str(w).lower()]
            if lex.has_vector:
                V[k] += lex.vector
            C[k] += 1
    nz = C > 0
    V[nz] /= C[nz, None]
    return np.column_stack([V, C])


def env_features(env, n_sec):
    e = np.zeros(n_sec); m = min(n_sec, len(env)); e[:m] = env[:m]
    return e[:, None]


def delay(X):
    T = X.shape[0]
    out = []
    for d in DELAYS:
        Y = np.zeros_like(X); Y[d:] = X[:T - d]; out.append(Y)
    return np.hstack(out)


def bold_1hz(bold, n_sec):
    t_tr = np.arange(bold.shape[1]) * TR_N
    t = np.arange(n_sec) + 0.5
    keep = t <= t_tr[-1]
    Y = np.stack([np.interp(t[keep], t_tr, row) for row in bold], 1)
    return Y, keep


def ridge_fit(X, Y, alpha):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    d = s / (s ** 2 + alpha)
    return Vt.T @ (d[:, None] * (U.T @ Y))


def fit(data):
    vocab, spv, mv = vectors()
    sets = {}
    for story in STORIES:
        z = np.load(f"{data}/train/{story}.npz")
        n_sec = int(z["bold"].shape[1] * TR_N)
        Y, keep = bold_1hz(z["bold"], n_sec)
        Wf = word_features(z["words"], z["onsets"], n_sec, vocab)[keep]
        Ef = env_features(z["env"], n_sec)[keep]
        sl = slice(DROP_S, None)
        sets[story] = {"Y": Y[sl], "W": Wf[sl], "E": Ef[sl]}
    res = {"spacy": spv, "vectors": mv, "stories": {s: int(v["Y"].shape[0]) for s, v in sets.items()}}
    for name in ("W", "E"):
        mu = np.vstack([v[name] for v in sets.values()]).mean(0); sd = np.vstack([v[name] for v in sets.values()]).std(0); sd[sd == 0] = 1
        Xs = {s: delay((v[name] - mu) / sd) for s, v in sets.items()}
        Ys = {s: H.zrow(v["Y"].T).T for s, v in sets.items()}
        cv = []
        for a in ALPHAS:
            rs = []
            for held in STORIES:
                Xtr = np.vstack([Xs[s] for s in STORIES if s != held]); Ytr = np.vstack([Ys[s] for s in STORIES if s != held])
                Wt = ridge_fit(Xtr, Ytr, a); P = Xs[held] @ Wt
                Pz = (P - P.mean(0)) / (P.std(0) + 1e-9); Yz = (Ys[held] - Ys[held].mean(0)) / (Ys[held].std(0) + 1e-9)
                rs.append(float(np.nanmean((Pz * Yz).mean(0))))
            cv.append(float(np.mean(rs)))
            print(name, f"alpha {a:.3g} cv r {cv[-1]:.4f}", flush=True)
        best = float(ALPHAS[int(np.argmax(cv))])
        Wt = ridge_fit(np.vstack(list(Xs.values())), np.vstack(list(Ys.values())), best)
        np.savez(f"{data}/ridge_{name}.npz", W=Wt.astype(np.float32), mu=mu, sd=sd)
        res[name] = {"alpha": best, "cv_r": dict(zip([f"{a:.3g}" for a in ALPHAS], cv))}
    json.dump(res, open(f"{data}/ridge_fit.json", "w"), indent=1)
    print(json.dumps({k: (v if k != "W" and k != "E" else {"alpha": v["alpha"], "best_cv_r": max(v["cv_r"].values())}) for k, v in res.items()}))


def read_word_table(path):
    rows = list(csv.reader(open(path), delimiter="\t"))
    hdr = rows[0]
    wi = next(i for i, h in enumerate(hdr) if h.lower() in ("word", "text", "token"))
    ti = next(i for i, h in enumerate(hdr) if h.lower() in ("onset", "start", "time", "onset_s"))
    return [r[wi] for r in rows[1:]], np.array([float(r[ti]) for r in rows[1:]])


def predict(data, own):
    vocab, _, _ = vectors()
    os.makedirs(own, exist_ok=True)
    M = {n: np.load(f"{data}/ridge_{n}.npz") for n in ("W", "E")}
    rel = f"{data}/ours_release"
    for story in ("story1", "story2", "story3"):
        ref = np.load(f"{rel}/pred_stock_{story}_full.npz")
        n_sec = int(ref["P"].shape[0]) + OFFSET + 10
        words, on = read_word_table(f"{rel}/words_{story}.tsv")
        env = H.envelope_per_second(f"{data}/r2/stimuli/{story}_audio.wav")
        perm = np.random.default_rng(SEED).permutation(len(words))
        feats = {"full": ("W", word_features(words, on, n_sec, vocab)),
                 "shuf": ("W", word_features([words[i] for i in perm], on, n_sec, vocab)),
                 "audio": ("E", env_features(env, n_sec))}
        for mode, (name, X) in feats.items():
            m = M[name]
            P = delay((X - m["mu"]) / m["sd"]) @ m["W"]
            P = P[OFFSET:OFFSET + ref["P"].shape[0]]
            np.savez(f"{own}/pred_stock_{story}_{mode}.npz", P=P.astype(np.float32), starts=ref["starts"])
            print(story, mode, P.shape, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"prep": lambda: prep(sys.argv[2]), "fit": lambda: fit(sys.argv[2]), "predict": lambda: predict(sys.argv[2], sys.argv[3])}[cmd]()
