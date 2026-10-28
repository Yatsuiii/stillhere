"""notthefall (D20) cheap-baseline kernel (stillhere/d20_baselines.py, scripts/make_d20_baselines.py), synthetic data only.

Always: it refuses without a frozen result, and with an unscored one; the built code dataset imports flat (as on Kaggle) and the kernel
is private, CPU, and attaches the frozen scoring output. When the frozen D20 code dataset and nilearn are on this machine: on that code
dataset with synthetic listener keys and a fake S3, it refuses before any fetch while the frozen result is missing; then the real frozen
d20_score.main scores the world; then the kernel re-reads it, reproduces the frozen listeners and T1, recovers a word-rate (LANG) and
envelope (AUD) effect planted in intact only (intact > shortscram) while raw TRIBE shows none, and leaves the frozen output byte-identical."""
import datetime
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from stillhere import d20_baselines as B

REPO = Path(__file__).resolve().parents[1]
FROZEN = Path(os.environ.get("STILLHERE_D20_SCORE_DIR", REPO.parent / "stillhere" / "d22" / "d20" / "score" / "kds_score"))
T_OK = datetime.datetime(2026, 10, 29, 17, 0, tzinfo=datetime.timezone.utc)
TR, WAV, S0, K = 1.5, 200, 6.0, 190.0
NV = int((WAV + 12) / TR)
PAIRS = [f"sub-{100 + i}" for i in range(12)]                                 # heard intact and shortscram
ONLY = ["sub-301", "sub-302"]                                                  # heard intact only
PLANT = {"intact": 0.8, "shortscram": 0.0, "longscram": 0.0}                   # word rate in LANG, envelope in AUD


def _mk():
    spec = importlib.util.spec_from_file_location("mk_d20_baselines", REPO / "scripts" / "make_d20_baselines.py")
    mk = importlib.util.module_from_spec(spec); spec.loader.exec_module(mk); return mk


def test_refuses_without_a_scored_frozen_result(tmp_path):
    inp = tmp_path / "input"; inp.mkdir()
    r = B.main(now=T_OK, net=object(), inputs=str(inp), out=str(tmp_path / "o1"))
    assert r["status"].startswith("refused: 0 frozen tierD_d20_score.json attached") and r["label"] == "exploratory"
    assert json.loads((tmp_path / "o1" / "tierD_d20_baselines.json").read_text())["status"] == r["status"]
    d = inp / "stillhere-d20-score" / "d20"; d.mkdir(parents=True)
    (d / "tierD_d20_score.json").write_text(json.dumps({"kernel": "stillhere-d20-score", "status": "pre-registered, not run (alignment)"}))
    r = B.main(now=T_OK, net=object(), inputs=str(inp), out=str(tmp_path / "o2"))
    assert r["status"] == "refused: frozen result has status 'pre-registered, not run (alignment)', not 'scored': no baseline"


def test_built_code_dataset_imports_flat_and_kernel_attaches_the_frozen_output(tmp_path):
    h = _mk().build(str(tmp_path)); kds = tmp_path / "kds"
    assert sorted(os.listdir(kds)) == ["MANIFEST.sha256", "cheap_baselines.py", "d20_baselines.py", "dataset-metadata.json"]
    assert hashlib.sha256((kds / "MANIFEST.sha256").read_bytes()).hexdigest() == h
    assert (kds / "d20_baselines.py").read_bytes() == (REPO / "stillhere" / "d20_baselines.py").read_bytes()
    meta = json.loads((tmp_path / "kernel" / "kernel-metadata.json").read_text())
    assert meta["id"] == "redlotusthepotus/stillhere-d20-baselines" and meta["is_private"] and not meta["enable_gpu"] and meta["enable_internet"]
    assert meta["kernel_sources"] == ["redlotusthepotus/stillhere-d20-pred-asr", "redlotusthepotus/stillhere-d20-score"]
    assert "redlotusthepotus/stillhere-d20-score-code" in meta["dataset_sources"]
    code = f"import sys; sys.path.insert(0, {str(kds)!r}); import d20_baselines as B; print(B.CB.__file__)"
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": os.path.dirname(os.path.dirname(np.__file__))}   # numpy only, no stillhere
    p = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert p.returncode == 0 and p.stdout.strip() == str(kds / "cheap_baselines.py"), p.stderr


# ---- synthetic world on the frozen code dataset --------------------------------------------------------------------------------------
def _smooth(x, k):
    return np.convolve(x, np.ones(k) / k, "same")


def _preds(S, seed):
    r = np.random.default_rng(seed); Pa = np.zeros((WAV, 20484), np.float32); Pf = np.zeros_like(Pa)
    for vs in S["AUD"] + S["LANG"]:
        a, u = _smooth(r.standard_normal(WAV), 9), _smooth(r.standard_normal(WAV), 9)
        Pa[:, vs] = a[:, None]; Pf[:, vs] = (a + 1.5 * u)[:, None]
    return {"full": Pf, "audio": Pa}


def _regs(seed):
    """A version's regressors.npz: word onsets with a slowly varying rate (the speech-to-text pass) and its audio's envelope."""
    r = np.random.default_rng(seed); lam = np.clip(3 + 6 * _smooth(r.standard_normal(WAV), 15), 0, 8)
    on = np.concatenate([np.sort(s + r.uniform(0, 1, r.poisson(lam[s]))) for s in range(int(S0), WAV - 1)])
    return dict(env=(3 * np.abs(_smooth(r.standard_normal(WAV), 5))).astype(np.float32), wps=np.bincount(on.astype(int), minlength=WAV).astype(float),
                onsets=on, durations=np.full(len(on), 0.15), wav_dur=float(WAV))


def _kernel_out(base, kernel, versions, preds, regs, words, pin):
    os.makedirs(f"{base}/d20_out"); man = {}
    for v in versions:
        d = f"{base}/pred/{v}"; os.makedirs(d)
        for k in ("full", "audio"):
            np.savez_compressed(f"{d}/pred_{k}.npz", P=preds[v][k], starts=np.arange(WAV))
            json.dump({"stock_ckpt_sha256": pin}, open(f"{base}/d20_out/tierD_pred_{v}_{k}.json", "w"))
        np.savez_compressed(f"{d}/regressors.npz", **regs[v])
        if words:
            open(f"{d}/words.tsv", "w").write("text\n" + "\n".join(words[v]) + "\n")
        man.update({f"{v}/{f}": hashlib.sha256(open(f"{d}/{f}", "rb").read()).hexdigest() for f in os.listdir(d)})
    json.dump({"kernel": kernel, "pred_files_sha256": man}, open(f"{base}/d20_out/tierD_outputs.json", "w"))


def _gii(X):
    import nibabel as nb
    return nb.gifti.GiftiImage(darrays=[nb.gifti.GiftiDataArray(X[:, t], intent="NIFTI_INTENT_TIME_SERIES") for t in range(X.shape[1])]).to_bytes()


class _Net:
    """Fake S3 for the allow-listed keys. Audio starts at scan time 0 and the story at S0. Every version's BOLD carries its TRIBE
    prediction at the same gain; intact alone adds HRF x word rate in LANG and HRF x envelope in AUD."""
    def __init__(self, DS, S, clips, preds, transcript):
        self.DS, self.S, self.clips, self.preds, self.transcript, self.calls, self.cache = DS, S, clips, preds, transcript, [], {}

    def bold(self, lst, task):
        if (lst, task) not in self.cache:
            v = task.replace("notthefall", ""); r = np.random.default_rng([int(lst[4:]), len(task)])
            t = np.arange(NV) * TR; z = lambda s: (s - s.mean()) / (s.std() + 1e-9)
            wr, _, env = self.clips[v].regs(t); X = np.zeros((20484, NV), np.float32)
            for i, vs in enumerate(self.S["AUD"] + self.S["LANG"]):
                aud = i < len(self.S["AUD"])
                s = z(np.interp(t - 5.0, np.arange(WAV), self.preds[v]["audio" if aud else "full"][:, vs[0]]))
                X[vs] = 0.5 * s + PLANT[v] * z(env if aud else wr) + r.standard_normal(NV)
            fd = np.abs(0.05 * r.standard_normal(NV)); fd[0] = 0
            hdr = [f"{c}{s}" for c in self.DS.M24 for s in ("", "_derivative1")] + [f"a_comp_cor_{i:02d}" for i in range(5)] + ["framewise_displacement"]
            cf = np.column_stack([0.01 * r.standard_normal((NV, len(hdr) - 1)), fd])
            conf = "\t".join(hdr) + "\n" + "".join("\t".join(f"{x:.6f}" for x in row) + "\n" for row in cf)
            self.cache[(lst, task)] = (_gii(X[:10242]), _gii(X[10242:]), conf.encode())
        return self.cache[(lst, task)]

    def save(self, url, dst):
        key = url.removeprefix("fake://"); self.calls.append(key)
        if key == self.DS.GENTLE_KEY:
            b = "".join(f"{w},{w},,\n" if i < 20 else f"{w},{w},{6 + 0.18 * i:.2f},{6.15 + 0.18 * i:.2f}\n" for i, w in enumerate(self.transcript))
        else:
            lst, task = key.split("/")[-1].split("_")[:2]; task = task.removeprefix("task-")
            if key.endswith("_bold.json"):
                b = json.dumps({"RepetitionTime": TR, "ManufacturersModelName": "Prisma"})
            elif key.endswith("_events.tsv"):
                st = f"{task}_audio.wav"; b = f"onset\tduration\ttrial_type\tstim_file\n0.0\t{S0}\tmusic\t{st}\n{S0}\t{K}\tstory\t{st}\n"
            else:
                L, R, cf = self.bold(lst, task); b = L if "hemi-L" in key else R if "hemi-R" in key else cf
        open(dst, "wb").write(b if isinstance(b, bytes) else b.encode())


def _keys(DS):
    out = [dict(key=DS.GENTLE_KEY, base="fake://", size=None, kind="align")]
    for lst in PAIRS + ONLY:
        for t in ("notthefallintact",) + (("notthefallshortscram",) if lst in PAIRS else ()):
            fp = f"data/Projects/narratives/derivatives/fmriprep/{lst}/func/{lst}_task-{t}"
            out += [dict(key=k, base="fake://", size=None, kind=kind, listener=lst, task=t) for kind, k in (
                ("boldL", f"{fp}_space-fsaverage5_hemi-L.func.gii"), ("boldR", f"{fp}_space-fsaverage5_hemi-R.func.gii"),
                ("confounds", f"{fp}_desc-confounds_regressors.tsv"), ("sidecar", f"ds002345/{lst}/func/{lst}_task-{t}_bold.json"),
                ("events", f"ds002345/{lst}/func/{lst}_task-{t}_events.tsv"))]
    return out


def _world(tmp_path, monkeypatch):
    """/kaggle/input as the baseline kernel sees it, minus the frozen output: the frozen code dataset (synthetic keys, MANIFEST rewritten)
    and the two prediction outputs. Shares: intact 0.90, shortscram 0.87 (scored), longscram 0.849 (not scored)."""
    inp = tmp_path / "input"; kds = inp / "stillhere-d20-score-code"; shutil.copytree(FROZEN, kds)
    monkeypatch.setattr(sys, "path", [str(kds)] + sys.path)
    for m in ("d20_score", "gauges", "clips", "timing", "d20_lib"):
        monkeypatch.delitem(sys.modules, m, raising=False)
    DS = importlib.import_module("d20_score")
    (kds / "d20_keys.json").write_text(json.dumps(_keys(DS)))
    files = sorted(f.name for f in kds.iterdir() if f.is_file() and f.name not in ("MANIFEST.sha256", "dataset-metadata.json"))
    (kds / "MANIFEST.sha256").write_text("".join(f"{hashlib.sha256((kds / f).read_bytes()).hexdigest()}  {f}\n" for f in files))
    S = DS.sets_of(f"{kds}/sets.npz"); pin = DS.C.CFG["stock_ckpt_sha256"]
    transcript = [f"w{i}" for i in np.random.default_rng(1).integers(0, 400, 1000)]
    words = {v: transcript[:n] + [f"zz{i}" for i in range(1000 - n)] for v, n in (("intact", 900), ("shortscram", 870), ("longscram", 849))}
    preds = {v: _preds(S, s) for v, s in (("intact", 11), ("shortscram", 12), ("longscram", 13))}
    regs = {v: _regs(s) for v, s in (("intact", 21), ("shortscram", 22), ("longscram", 23))}
    _kernel_out(f"{inp}/stillhere-d20-pred-asr", "stillhere-d20-pred-asr", tuple(preds), preds, regs, words, pin)
    _kernel_out(f"{inp}/stillhere-d20-pred", "stillhere-d20-pred", ("intact",), preds, regs, None, pin)
    clips = {v: DS.G.Clip(DS.VERSIONS[v], {k: {"P": preds[v][k], "starts": np.arange(WAV)} for k in ("full", "audio")}, regs[v], S) for v in preds}
    return inp, _Net(DS, S, clips, preds, transcript), hashlib.sha256((kds / "MANIFEST.sha256").read_bytes()).hexdigest(), DS


def _hashes(d):
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob("*")) if p.is_file()}


@pytest.mark.skipif(not (FROZEN / "d20_score.py").exists() or any(importlib.util.find_spec(m) is None for m in ("nilearn", "scipy", "nibabel", "pandas")),
                    reason="needs the frozen D20 code dataset, nilearn, scipy, nibabel and pandas")
def test_planted_baseline_recovered_and_frozen_output_untouched(tmp_path, monkeypatch):
    inp, net, man, DS = _world(tmp_path, monkeypatch)
    r = B.main(now=T_OK, net=net, inputs=str(inp), out=str(tmp_path / "b0"), manifest=man)
    assert r["status"].startswith("refused: 0 frozen") and not net.calls                    # frozen run not done yet: no fetch at all

    fz = DS.main("a" * 64, p8="pass", now=T_OK, net=net, root=str(inp / "stillhere-d20-score-code"), inputs=str(inp),
                 out=str(inp / "stillhere-d20-score" / "d20"))                              # the frozen D20 run, code unchanged
    assert fz["status"] == "scored" and sorted(fz["T1_per_listener"]) == ["intact", "shortscram"]
    frozen = _hashes(inp / "stillhere-d20-score"); n0 = len(net.calls)

    r = B.main(now=T_OK, net=net, inputs=str(inp), out=str(tmp_path / "b1"), manifest=man)
    assert _hashes(inp / "stillhere-d20-score") == frozen                                  # frozen output byte-identical
    assert r["status"] == "scored" and r["label"] == "exploratory" and r["scored_versions"] == ["intact", "shortscram"]
    assert r["same_listeners_as_frozen"] and r["T1_max_abs_diff_vs_frozen"] <= 1e-9
    assert set(net.calls[n0:]) <= {x["key"] for x in json.loads((inp / "stillhere-d20-score-code" / "d20_keys.json").read_text())}
    assert sorted(r["per_listener"]["intact"]) == sorted(PAIRS + ONLY) and sorted(r["per_listener"]["shortscram"]) == PAIRS
    T = r["tests"]
    assert set(T) == {f"{n}_{k}" for n in ("LANG", "AUD") for k in B.CB.KINDS}
    for name in ("LANG_word_rate", "AUD_envelope"):
        p = T[name]["intact_gt_shortscram"]
        assert p["status"] == "pass" and p["n_pairs"] == 12 and p["k_intact_higher"] == 12 and p["hl"] > 0, (name, p)
    assert T["LANG_tribe"]["intact_gt_shortscram"]["status"] != "pass"                     # same TRIBE gain in every version
    assert r["frozen_primary_intact_gt_shortscram"] == fz["primary_intact_gt_shortscram"]
    assert any(x.startswith("[exploratory] frozen D20 primary") for x in r["lines"])

    n1 = len(net.calls)
    r = B.main(now=T_OK, net=net, inputs=str(inp), out=str(tmp_path / "b2"), manifest="0" * 64)
    assert r["status"] == "refused: the frozen result names a different code MANIFEST" and len(net.calls) == n1
    r = B.main(now=datetime.datetime(2026, 10, 29, 15, 59, tzinfo=datetime.timezone.utc), net=net, inputs=str(inp), out=str(tmp_path / "b3"), manifest=man)
    assert r["status"].startswith("refused: before") and len(net.calls) == n1
