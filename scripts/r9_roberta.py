"""R9 (docs/rules/r9_roberta_rule.md; exploratory): R8's ridge with RoBERTa-base word features (spaCy en_core_web_trf 3.8.0).
  fit DATA           features for the 8 Narratives stories (DATA/train from R8's prep), leave-one-story-out alpha, ridge weights
  predict DATA OWN   the nine Paranoia forecasts into OWN; the audio forecasts use R8's envelope model (DATA/ridge_E.npz)
Healthy volunteers only; not a diagnosis or medical advice."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r8_ridge as R  # noqa: E402

H = R.H


def load_nlp():
    import spacy
    nlp = spacy.load("en_core_web_trf", exclude=["tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])
    return nlp, spacy.__version__, nlp.meta["version"]


def word_vectors(nlp, words):
    from spacy.tokens import Doc
    out = []
    step = 2000  # words per document; the transformer itself works in its own strided windows
    for i in range(0, len(words), step):
        chunk = [str(w) if str(w).strip() else "_" for w in words[i:i + step]]
        doc = nlp(Doc(nlp.vocab, words=chunk))
        st = doc._.trf_data.last_hidden_layer_state
        data = np.asarray(st.dataXd) if hasattr(st, "dataXd") else np.asarray(st.data)
        lens = np.asarray(st.lengths)
        idx = np.concatenate([[0], np.cumsum(lens)])
        for j in range(len(chunk)):
            a, b = idx[j], idx[j + 1]
            out.append(data[a:b].mean(0) if b > a else np.zeros(data.shape[1]))
    return np.vstack(out)


def bin_features(vecs, onsets, n_sec):
    V = np.zeros((n_sec, vecs.shape[1])); C = np.zeros(n_sec)
    for v, t in zip(vecs, onsets):
        k = int(t)
        if 0 <= k < n_sec:
            V[k] += v; C[k] += 1
    nz = C > 0
    V[nz] /= C[nz, None]
    return np.column_stack([V, C])


def fit(data):
    nlp, spv, mv = load_nlp()
    sets = {}
    for story in R.STORIES:
        z = np.load(f"{data}/train/{story}.npz")
        n_sec = int(z["bold"].shape[1] * R.TR_N)
        Y, keep = R.bold_1hz(z["bold"], n_sec)
        vec = word_vectors(nlp, list(z["words"]))
        Wf = bin_features(vec, z["onsets"], n_sec)[keep]
        sl = slice(R.DROP_S, None)
        sets[story] = {"Y": Y[sl], "W": Wf[sl]}
        print(story, Wf.shape, flush=True)
    res = {"spacy": spv, "model": f"en_core_web_trf {mv}", "stories": {s: int(v["Y"].shape[0]) for s, v in sets.items()}}
    allW = np.vstack([v["W"] for v in sets.values()])
    mu = allW.mean(0); sd = allW.std(0); sd[sd == 0] = 1
    Xs = {s: R.delay((v["W"] - mu) / sd) for s, v in sets.items()}
    Ys = {s: H.zrow(v["Y"].T).T for s, v in sets.items()}
    cv = {a: [] for a in R.ALPHAS}
    for held in R.STORIES:
        Xtr = np.vstack([Xs[s] for s in R.STORIES if s != held]); Ytr = np.vstack([Ys[s] for s in R.STORIES if s != held])
        U, sv, Vt = np.linalg.svd(Xtr, full_matrices=False)
        UtY = U.T @ Ytr
        del Xtr, Ytr, U
        Yz = (Ys[held] - Ys[held].mean(0)) / (Ys[held].std(0) + 1e-9)
        for a in R.ALPHAS:
            P = Xs[held] @ (Vt.T @ ((sv / (sv ** 2 + a))[:, None] * UtY))
            Pz = (P - P.mean(0)) / (P.std(0) + 1e-9)
            cv[a].append(float(np.nanmean((Pz * Yz).mean(0))))
        print("fold", held, flush=True)
    cvm = [float(np.mean(cv[a])) for a in R.ALPHAS]
    for a, r in zip(R.ALPHAS, cvm):
        print(f"alpha {a:.3g} cv r {r:.4f}", flush=True)
    best = float(R.ALPHAS[int(np.argmax(cvm))])
    Wt = R.ridge_fit(np.vstack(list(Xs.values())), np.vstack(list(Ys.values())), best)
    np.savez(f"{data}/r9_W.npz", W=Wt.astype(np.float32), mu=mu, sd=sd)
    res["W"] = {"alpha": best, "cv_r": dict(zip([f"{a:.3g}" for a in R.ALPHAS], cvm))}
    json.dump(res, open(f"{data}/r9_fit.json", "w"), indent=1)
    print(json.dumps({"alpha": best, "best_cv_r": max(cvm)}))


def predict(data, own):
    nlp, _, _ = load_nlp()
    os.makedirs(own, exist_ok=True)
    M = {"W": np.load(f"{data}/r9_W.npz"), "E": np.load(f"{data}/ridge_E.npz")}
    rel = f"{data}/ours_release"
    for story in ("story1", "story2", "story3"):
        ref = np.load(f"{rel}/pred_stock_{story}_full.npz")
        n_sec = int(ref["P"].shape[0]) + R.OFFSET + 10
        words, on = R.read_word_table(f"{rel}/words_{story}.tsv")
        env = H.envelope_per_second(f"{data}/r2/stimuli/{story}_audio.wav")
        perm = np.random.default_rng(R.SEED).permutation(len(words))
        feats = {"full": ("W", bin_features(word_vectors(nlp, words), on, n_sec)),
                 "shuf": ("W", bin_features(word_vectors(nlp, [words[i] for i in perm]), on, n_sec)),
                 "audio": ("E", R.env_features(env, n_sec))}
        for mode, (name, X) in feats.items():
            m = M[name]
            P = R.delay((X - m["mu"]) / m["sd"]) @ m["W"]
            P = P[R.OFFSET:R.OFFSET + ref["P"].shape[0]]
            np.savez(f"{own}/pred_stock_{story}_{mode}.npz", P=P.astype(np.float32), starts=ref["starts"])
            print(story, mode, P.shape, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"fit": lambda: fit(sys.argv[2]), "predict": lambda: predict(sys.argv[2], sys.argv[3])}[cmd]()
