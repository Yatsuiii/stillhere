"""R2 stock forecasts on a Kaggle T4, part @@PART@@ (docs/rules/r2_paranoia_rule.md, amendment of Nov 1 2026): stock TRIBE v2 on the three
Paranoia stories (ds001338 v1.0.0 stimuli, fetched from OpenNeuro and checked against their pinned SHA-256). Part A: the audio-only and
full forecasts of story1-3. Part B: the full forecast with the word texts shuffled across their onsets (seed 7, tribe_common.write_tsv)
of story1-3, reading part A's output. Moved here from the laptop (its RAM); the stages are r2/run_r2_forecast.py's, all on the T4: the
harness's helpers (tribe_common as the D22 kernels ran it, embedded byte for byte: build_events in the training order, load_model with the
text lane through accelerate, apply_weights('stock') with the checkpoint pin asserted, stitched_predict) and the runner's checks, its
functions embedded byte for byte (no exact-shift copies, last word before the audio's end, full-text counts descriptive, the shuffled table
keeps every onset and duration and permutes the texts; finite, one row per second).
Words: each story's word table is TRIBE v2's own speech-to-text, made by the kernel stillhere-r2-asr (attached to part A; SHA-256
asserted), placed as stim.tsv beside stim.wav, so ExtractWordsFromAudio reads it and runs no speech-to-text here (checked: the table is
unchanged after the events are built). Within a story every stage uses the same stim.wav path, so the sound features are computed once:
part A exports its sound cache and part B loads it (files part B adds to it are counted). Pinned like the film kernels: the package lock,
the Hugging Face revisions (a stage that loads another revision stops the kernel), one T4. Writes pred_stock_<story>_<mode>.npz,
stage_stock_<story>_<mode>.json (the runner's format), words_<story>.tsv, r2_forecast_part@@PART@@.json and MANIFEST.sha256; prints counts,
shapes, hashes and timings, never word text."""
import glob, hashlib, json, os, shutil, subprocess, time, urllib.request
from pathlib import Path
PART = "@@PART@@"
STORIES = ("story1", "story2", "story3")
STAGES = [f"stock_{s}_{m}" for s in STORIES for m in (("audio", "full") if PART == "A" else ("shuf",))]
STIM_SHA = "@@STIM_SHA@@"
FULLTEXT_SHA = "c762b04ed6a1141391c9537b9eddb3e77bb18824d1adff968e300d7d70aae82c"
WORDS_SHA = "@@WORDS_SHA@@"
URL = "https://s3.amazonaws.com/openneuro.org/ds001338/stimuli/{}"
PIN = "9c79ffff6b642b7b0c71d558c935fb3fa33f2788bfb509feead94fafbba2f321"          # stock TRIBE v2 checkpoint
HF_PINS = "@@HF_PINS@@"
LOCK = "@@LOCK@@"
COMMON = "@@COMMON@@"
COMMON_SHA = "@@COMMON_SHA@@"
CHECKS = "@@CHECKS@@"
STAGE = "@@STAGE@@"
DRY = bool(os.environ.get("R2K_DRY"))                                   # local dry run: a stand-in python and helpers, local inputs
OUT = Path(os.environ.get("R2K_OUT", "/kaggle/working")); OUT.mkdir(parents=True, exist_ok=True); LOG = OUT / "driver.log"
TMP = Path(os.environ.get("R2K_TMP", "/tmp")); ROOT = TMP / "r2"; CACHE = TMP / "cache"
INPUT = os.environ.get("R2K_INPUT", "/kaggle/input"); PY = os.environ.get("R2K_PY") or str(TMP / "tribe-env" / "bin" / "python")
T0 = time.time()


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); open(LOG, "a").write(s + "\n")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def sh(cmd, env=None, timeout=None):
    t = time.time(); r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env={**os.environ, **(env or {})}, timeout=timeout)
    log(f"$ {cmd[:110]}  rc={r.returncode}  {time.time() - t:.0f}s")
    if r.returncode: log("STDERR tail:", r.stderr[-1500:])
    return r


def revisions():
    out = {}
    for d in glob.glob(str(TMP / "hf/hub/models--*")):
        ref = Path(d) / "refs" / "main"
        if ref.exists(): out[Path(d).name[len("models--"):].replace("--", "/")] = ref.read_text().strip()
    return out


def one(pattern):
    hits = glob.glob(f"{INPUT}/**/{pattern}", recursive=True)
    assert len(hits) == 1, f"{len(hits)} attached copies of {pattern} (need 1)"
    return hits[0]


assert hashlib.sha256(COMMON.encode()).hexdigest() == COMMON_SHA, "tribe_common differs from its pin"
GPU = "dry run" if DRY else sh("nvidia-smi --query-gpu=name --format=csv,noheader -i 0").stdout.strip()
log("part", PART, "stages", STAGES, "gpu", GPU)
tok = None
for h in glob.glob(f"{INPUT}/**/hf_token", recursive=True): tok = Path(h).read_text().strip()
if not tok and not DRY: log("no HF token; exiting"); raise SystemExit(1)
ENVV = {"HF_HOME": str(TMP / "hf"), "HF_HUB_DISABLE_TELEMETRY": "1", "UV_CACHE_DIR": str(TMP / "uv"), "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "HF_TOKEN": tok or "", "CUDA_VISIBLE_DEVICES": "0"}
FREEZE = []
if not DRY:
    Path("/tmp/r2_lock.txt").write_text(LOCK)
    sh("pip install -q uv", timeout=600); sh(f"uv venv {TMP / 'tribe-env'} --python 3.11 -q", ENVV, 600)
    r = sh(f'uv pip install -q --python {PY} -c /tmp/r2_lock.txt "tribev2 @ git+https://github.com/facebookresearch/tribev2.git@af58661" '
           "accelerate soundfile pandas", ENVV, 1800)
    assert r.returncode == 0, "install failed"
    FREEZE = sh(f"uv pip freeze --python {PY}", ENVV).stdout.strip().splitlines()
(TMP / "k2code").mkdir(parents=True, exist_ok=True)
(TMP / "k2code" / "tribe_common.py").write_text(os.environ.get("R2K_FAKE_COMMON") and open(os.environ["R2K_FAKE_COMMON"]).read() or COMMON)
(TMP / "k2code" / "r2_checks.py").write_text(CHECKS); (TMP / "r2_stage.py").write_text(STAGE)

# stimuli (public audio), the full text (for counts only) and the word tables, every file hash-checked
res = {"part": PART, "rule": "docs/rules/r2_paranoia_rule.md (amendment of Nov 1 2026)", "gpu": GPU, "hf_pins": HF_PINS, "stimuli": {},
       "words": {}, "stages": {}}
ROOT.mkdir(parents=True, exist_ok=True)
for s in STORIES:
    w = ROOT / s / "stim.wav"; w.parent.mkdir(parents=True, exist_ok=True)
    src = os.environ.get("R2K_STIM")
    shutil.copyfile(f"{src}/{s}_audio.wav", w) if src else urllib.request.urlretrieve(URL.format(f"{s}_audio.wav"), w)
    assert sha(w) == STIM_SHA[s], f"{s}: audio differs from the pinned SHA-256"
    res["stimuli"][s] = STIM_SHA[s]
ft = ROOT / "story_fulltext.txt"
shutil.copyfile(f"{os.environ['R2K_STIM']}/story_fulltext.txt", ft) if os.environ.get("R2K_STIM") else urllib.request.urlretrieve(URL.format("story_fulltext.txt"), ft)
assert sha(ft) == FULLTEXT_SHA, "full text differs from its pin"
for s in STORIES:
    src = one(f"{s}_stim.tsv") if PART == "A" else one(f"words_{s}.tsv")
    assert sha(src) == WORDS_SHA[s], f"{s}: word table differs from the speech-to-text kernel's"
    shutil.copyfile(src, OUT / f"words_{s}.tsv"); res["words"][s] = WORDS_SHA[s]
    if PART == "A":
        shutil.copyfile(src, ROOT / s / "stim.tsv")                    # ExtractWordsFromAudio reads it instead of running speech-to-text
if PART == "A":
    res["speech_to_text"] = json.load(open(one("r2_asr.json")))
    res["speech_to_text"].pop("pip_freeze", None)
else:                                                                 # part A's sound cache, every file checked against its manifest
    man = one("export/MANIFEST.sha256"); exp = Path(man).parent
    bad = [l.split(None, 1)[1].strip() for l in open(man) if sha(exp / l.split(None, 1)[1].strip()) != l.split()[0]]
    assert not bad, f"{len(bad)} files of part A's export differ from its manifest"
    shutil.copytree(exp / "cache", CACHE, dirs_exist_ok=True)
    res["part_a_export_manifest_sha256"] = sha(man)
before = {str(p) for p in CACHE.rglob("*") if p.is_file()}

for stage in STAGES:
    a = time.time(); r = sh(f"{PY} {TMP / 'r2_stage.py'} {stage} {ROOT} {OUT} {CACHE} {TMP / 'k2code'}", ENVV, 3 * 3600)
    line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
    rec = json.loads(line[-1][7:]) if line else {"error_line": ([l for l in r.stderr.strip().splitlines() if "Error" in l] or ["?"])[-1][:300]}
    rec["wall_s"] = round(time.time() - a, 1); rec["ok"] = bool(line)
    off = {k: v for k, v in revisions().items() if k in HF_PINS and v != HF_PINS[k]}
    if off: rec["ok"] = False; rec["revision_mismatch"] = off
    res["stages"][stage] = rec; log("STAGE", json.dumps(rec))
    if not rec["ok"]:
        log(f"stage {stage} failed; stopping"); break
added = sorted({str(p) for p in CACHE.rglob("*") if p.is_file()} - before)
res["cache_files_added_by_dir"] = {}
for p in added:
    d = Path(p).relative_to(CACHE).parts[0]; res["cache_files_added_by_dir"][d] = res["cache_files_added_by_dir"].get(d, 0) + 1
if PART == "A":                                                       # for part B: the sound features of the three stim.wav paths
    exp = OUT / "export"; exp.mkdir(exist_ok=True)
    for d in sorted(os.listdir(CACHE)) if CACHE.exists() else []:
        if d.startswith("neuralset.extractors.audio."):
            shutil.copytree(CACHE / d, exp / "cache" / d)
    with open(exp / "MANIFEST.sha256", "w") as fh:
        for p in sorted(x for x in exp.rglob("*") if x.is_file() and x.name != "MANIFEST.sha256"):
            fh.write(f"{sha(p)}  {p.relative_to(exp)}\n")
res["whisperx_installed"] = sorted({Path(p).name for p in glob.glob(str(TMP / "uv/**/whisperx-*.dist-info"), recursive=True)})   # expect none
res["hf_revisions"] = revisions(); res["pip_freeze"] = FREEZE; res["pip_freeze_sha256"] = hashlib.sha256("\n".join(FREEZE).encode()).hexdigest()
res["all_ok"] = len(res["stages"]) == len(STAGES) and all(v["ok"] for v in res["stages"].values())
res["kernel_seconds"] = round(time.time() - T0)
json.dump(res, open(OUT / f"r2_forecast_part{PART}.json", "w"), indent=1)
with open(OUT / "MANIFEST.sha256", "w") as fh:
    for p in sorted(x for x in OUT.rglob("*") if x.is_file() and x.name not in ("MANIFEST.sha256", "driver.log")):
        fh.write(f"{sha(p)}  {p.relative_to(OUT)}\n")
log("RESULT", json.dumps({k: v for k, v in res.items() if k not in ("stages", "pip_freeze", "speech_to_text")}))
