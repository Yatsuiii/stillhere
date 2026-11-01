"""R2 and R5's one scoring run on a Kaggle CPU (docs/rules/r2_paranoia_rule.md sections 9-11; docs/rules/r5_short_scan_rule.md).

The repository arrives as a git bundle and is cloned in full (the clone must not be shallow; HEAD must be EXPECT_HEAD, whose scoring code
equals the pins commit's); the scoring root (stimuli, the 9 stock forecasts and their stage records, the word tables, surprisal, the code
records and the atlas) arrives as one tar whose SHA-256 is checked, and r2_paranoia's check_pins checks every pinned file in it. Then, once
each and in order: r2_paranoia.py download (the events files and the 132 fsaverage5 GIFTIs, each size checked against r2/data_check.json,
SHA256.json written), score, and r5_short_scan.py. Any step that fails stops the run; nothing is rerun. Outputs: results/ (the files the
scorers write into the repository's results/, and the download's SHA256.json), private/ (per-listener values, never committed), the logs,
the run record and a MANIFEST.sha256. Prints counts, hashes, timings and the scorers' lines only."""
import glob, hashlib, json, os, shutil, subprocess, sys, tarfile, time
from pathlib import Path

EXPECT_HEAD = "bfbda2baf93dbd54f0b0462242f5edf53e52d09d"     # the pins commit 354cab7 plus two test-only changes
PINS_COMMIT = "354cab775f7fa24437f0cb5aa5a77ee691bce7fb"
ROOT_TAR_SHA = "ba6a090b67a6180fd44bc001ce78b6a8de763e867ef5ecdf4a61e69b4b9f5f89"
PKGS = "numpy==2.2.6 scipy==1.18.1 nibabel==5.4.2 nilearn==0.14.1 soundfile==0.14.0 pandas==3.0.6 scikit-learn==1.9.1 joblib==1.6.0"
IN = Path(os.environ.get("R2S_IN", "/kaggle/input")); OUT = Path(os.environ.get("R2S_OUT", "/kaggle/working"))
TMP = Path(os.environ.get("R2S_TMP", "/tmp")); DRY = os.environ.get("R2S_DRY") == "1"
OUT.mkdir(parents=True, exist_ok=True); LOG = OUT / "score_driver.log"


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def log(*a):
    line = time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + " ".join(str(x) for x in a)
    print(line, flush=True); open(LOG, "a").write(line + "\n")


def run(cmd, logname=None, cwd=None, env=None, timeout=36000):
    t0 = time.time()
    with open(OUT / logname, "w") if logname else open(os.devnull, "w") as f:
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        tail = []
        for line in p.stdout:
            f.write(line); tail = (tail + [line.rstrip()])[-40:]
            if logname:
                print(line, end="", flush=True)
        rc = p.wait(timeout=timeout)
    return rc, round(time.time() - t0), tail


def one(pattern):
    hits = glob.glob(str(IN / "**" / pattern), recursive=True)
    assert len(hits) == 1, f"{len(hits)} files match {pattern}"
    return Path(hits[0])


res = {"rule": ["docs/rules/r2_paranoia_rule.md", "docs/rules/r5_short_scan_rule.md"], "dry": DRY,
       "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "steps": {}}
try:
    bundle, tarp = one("stillhere.bundle"), one("r2_score_root.tar")
    res["inputs"] = {"bundle_sha256": sha(bundle), "root_tar_sha256": sha(tarp)}
    assert res["inputs"]["root_tar_sha256"] == ROOT_TAR_SHA, "the scoring root's tar differs from the one pinned in this kernel"
    repo, root = TMP / "repo", TMP / "root"
    shutil.rmtree(repo, ignore_errors=True); shutil.rmtree(root, ignore_errors=True)
    rc, _, tail = run(["git", "clone", "-q", "-b", "r2-score-bundle", str(bundle), str(repo)])
    assert rc == 0, f"git clone failed: {tail}"
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True)
    head, shallow = git("rev-parse", "HEAD").stdout.strip(), git("rev-parse", "--is-shallow-repository").stdout.strip()
    same = git("diff", "--quiet", PINS_COMMIT, "HEAD", "--", "scripts", "stillhere", "r2", "docs").returncode == 0
    res["clone"] = {"head": head, "shallow": shallow, "commits": int(git("rev-list", "--count", "HEAD").stdout), "code_equals_pins_commit": same}
    assert head == EXPECT_HEAD and shallow == "false" and same, f"clone check failed: {res['clone']}"
    log("clone", json.dumps(res["clone"]))
    root.mkdir(parents=True)
    with tarfile.open(tarp) as t:
        t.extractall(root, filter="data")
    log("root extracted", sum(1 for _ in root.rglob("*") if _.is_file()), "files")
    if DRY:
        py = os.environ.get("R2S_PY", sys.executable)
    else:
        rc, s, tail = run(["bash", "-c", f"pip install -q uv && uv venv {TMP}/env --python 3.12 -q && uv pip install -q --python {TMP}/env/bin/python {PKGS}"])
        assert rc == 0, f"environment failed: {tail}"
        py = f"{TMP}/env/bin/python"
        res["pip_freeze"] = subprocess.run(["bash", "-c", f"uv pip freeze --python {py}"], capture_output=True, text=True).stdout.split()
        log("environment", s, "s")
    extra = os.environ.get("R2S_EXTRA_PATH", "") if DRY else ""                   # the dry run's stand-in packages; on Kaggle, the venv's
    env = {**os.environ, "PYTHONPATH": f"{repo}:{repo}/scripts" + (f":{extra}" if extra else ""), "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"}
    if DRY:                                                                # stand-in: the checks, never the download or the scoring
        steps = [("check_pins", [py, "-c", f"import sys; sys.argv=['x']; import r2_paranoia as R2, r5_short_scan as R5; "
                                           f"print(len(R2.check_pins('{root}')), len(R5.check_inputs('{root}')))"])]
    else:
        steps = [("download", [py, "scripts/r2_paranoia.py", "download", "--root", str(root)]),
                 ("score", [py, "scripts/r2_paranoia.py", "score", "--root", str(root), "--out", str(TMP / "out_r2")]),
                 ("r5", [py, "scripts/r5_short_scan.py", str(root), str(TMP / "out_r5")])]
    for name, cmd in steps:
        log("step", name, "start")
        rc, s, tail = run(cmd, f"{name}.log", cwd=str(repo), env=env)
        res["steps"][name] = {"rc": rc, "seconds": s}
        log("step", name, "rc", rc, s, "s")
        if rc != 0:
            res["stopped"] = {"step": name, "tail": tail[-15:]}
            break
finally:
    res["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rdir, pdir = OUT / "results", OUT / "private"; rdir.mkdir(exist_ok=True); pdir.mkdir(exist_ok=True)
    for f in glob.glob(str(TMP / "repo/results/r2_*")) + glob.glob(str(TMP / "repo/results/r5_short_scan*")):
        shutil.copy(f, rdir)
    if (TMP / "root/r2/bold/SHA256.json").exists():
        shutil.copy(TMP / "root/r2/bold/SHA256.json", rdir / "r2_bold_SHA256.json")
    for d in ("out_r2", "out_r5"):
        if (TMP / d).exists():
            shutil.copytree(TMP / d, pdir / d, dirs_exist_ok=True)
    json.dump(res, open(OUT / "r2_score_run.json", "w"), indent=1)
    with open(OUT / "MANIFEST.sha256", "w") as m:
        for p in sorted(x for x in OUT.rglob("*") if x.is_file() and x.name != "MANIFEST.sha256"):
            m.write(f"{sha(p)}  {p.relative_to(OUT)}\n")
    log("done", json.dumps({k: res[k] for k in ("steps",) if k in res}), "stopped" if "stopped" in res else "")
