"""Opening forecast (StillHere Pilot, public data only; rule docs/rules/emofilm_rule.md version 2): stock TRIBE v2 full and audio-only
predictions for release 0 to 190 s of Tears of Steel (Blender Foundation 2012, CC BY 3.0): the scored opening three minutes (0 to 180 s)
plus a 10 s margin so prediction samples exist through 180 s (ffmpeg -ss 0 -t 190). It also covers the bridge conversation (25.00 to
40.25 s) used for the SAM 3.1 edits and the video. The kernel keeps its bridge name. Version: intact. One T4 (CUDA_VISIBLE_DEVICES=0). Stages run as separate processes so each extractor's
memory is freed before the next: events (speech-to-text on the segment's audio, the harness's own ExtractWordsFromAudio), video
(V-JEPA2), audio (w2v-BERT), text (Llama-3.2-3B, 'accelerate' as in the stock predictions), predict (cached features, model forward).
Per stage: wall seconds, minutes per film minute, torch peak allocated and reserved GB, nvidia-smi peak GB. If the video stage runs out of
memory at the stock batch size it is retried at smaller batch sizes (memory only; features unchanged). Prints and writes numbers only."""
import glob, hashlib, json, os, shutil, subprocess, textwrap, threading, time, urllib.request
from pathlib import Path
TOS_URL = "https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov"
TOS_SHA = "efa9062d9cdb7a338e40ad530dfdf234806743f29ae6a1a136b97ece4e588e8f"
PIN = "9c79ffff6b642b7b0c71d558c935fb3fa33f2788bfb509feead94fafbba2f321"          # stock TRIBE v2 checkpoint (d22/d3, stillhere-pred)
OUT = Path("/kaggle/working/film_am"); OUT.mkdir(parents=True, exist_ok=True); LOG = OUT / "driver.log"
ENV = Path("/tmp/tribe-env"); PY = ENV / "bin" / "python"; T0 = time.time()


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); open(LOG, "a").write(s + "\n")


def sh(cmd, env=None, timeout=None):
    t = time.time(); r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env={**os.environ, **(env or {})}, timeout=timeout)
    log(f"$ {cmd[:110]}  rc={r.returncode}  {time.time() - t:.0f}s")
    if r.returncode: log("STDERR tail:", r.stderr[-1500:])
    return r


smi = {"rows": [], "stop": False}
def sampler():
    while not smi["stop"]:
        r = subprocess.run("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits", shell=True, capture_output=True, text=True)
        t = time.time()
        for line in r.stdout.strip().splitlines():
            i, m = [x.strip() for x in line.split(",")]; smi["rows"].append((t, int(i), int(m)))
        time.sleep(0.5)


log("gpus:", sh("nvidia-smi -L").stdout.strip().replace("\n", " | "))
tok = None
for h in glob.glob("/kaggle/input/**/hf_token", recursive=True): tok = Path(h).read_text().strip()
if not tok: log("no HF token; exiting"); raise SystemExit(0)
ENVV = {"HF_HOME": "/tmp/hf", "HF_HUB_DISABLE_TELEMETRY": "1", "UV_CACHE_DIR": "/tmp/uv", "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "HF_TOKEN": tok, "CUDA_VISIBLE_DEVICES": "0"}
sh("pip install -q uv", timeout=600); sh(f"uv venv {ENV} --python 3.11 -q", ENVV, 600)
sh(f'uv pip install -q --python {PY} "tribev2 @ git+https://github.com/facebookresearch/tribev2.git@af58661" accelerate soundfile pandas', ENVV, 1800)
SEG_FILE, SEG_SHA = "area_matched_v2.mp4", "9684c7a4cd4db8b380779a11e522e33347ae6d246ab9c84c883e13f15589b983"   # stillhere-film-stimuli (docs/rules/film_stimulus_rule.md)
segs = glob.glob(f"/kaggle/input/**/{SEG_FILE}", recursive=True)
assert len(segs) == 1, f"{len(segs)} copies of {SEG_FILE} attached (need 1)"
shutil.copyfile(segs[0], "/tmp/seg.mp4")
seg_sha = hashlib.sha256(open("/tmp/seg.mp4", "rb").read()).hexdigest(); log("segment", SEG_FILE, "sha256 ok:", seg_sha == SEG_SHA)
assert seg_sha == SEG_SHA, "segment differs from the committed stimuli"
dur = sh("ffprobe -v error -show_entries format=duration -of csv=p=0 /tmp/seg.mp4").stdout.strip(); log("segment seconds", dur)

exps = [p for p in glob.glob("/kaggle/input/**/export/MANIFEST.sha256", recursive=True)]
assert len(exps) == 1, f"{len(exps)} intact exports attached (need 1)"
EXP = Path(exps[0]).parent
bad = [l.split()[1] for l in open(EXP / "MANIFEST.sha256") if hashlib.sha256(open(EXP / l.split()[1], "rb").read()).hexdigest() != l.split()[0]]
assert not bad, f"{len(bad)} intact export files differ from their manifest"
shutil.copyfile(EXP / "ev.pkl", "/tmp/ev.pkl"); shutil.copyfile(EXP / "seg.wav", "/tmp/seg.wav")
for d in os.listdir(EXP / "cache"):
    shutil.copytree(EXP / "cache" / d, f"/tmp/cache/{d}")
n_cache = sum(1 for _ in Path("/tmp/cache").rglob("*") if _.is_file())
log("intact export loaded:", sum(1 for _ in open(EXP / "MANIFEST.sha256")), "files hash-checked | cache files", n_cache,
    "| manifest", hashlib.sha256(open(EXP / "MANIFEST.sha256", "rb").read()).hexdigest()[:16])
STAGE = textwrap.dedent('''
import sys, json, time, pickle, hashlib, glob, os
import torch
stage, bs = sys.argv[1], int(sys.argv[2])
from tribev2.demo_utils import TribeModel
cu = {"data.num_workers": 0, "data.features_to_use": ["text", "audio", "video"], "data.text_feature.device": "accelerate", "data.overlap_trs_train": 50}
if bs: cu["data.video_feature.image.batch_size"] = bs
if stage == "predict_audio": cu["data.features_to_use"] = ["audio"]
t0 = time.time()
m = TribeModel.from_pretrained("facebook/tribev2", cache_folder="/tmp/cache", config_update=cu)
base = torch.cuda.memory_allocated() / 1e9
torch.cuda.reset_peak_memory_stats(); t1 = time.time(); out = {"stage": stage, "load_s": round(t1 - t0, 1), "brain_model_gb": round(base, 2)}
if stage == "events":
    ev = m.get_events_dataframe(video_path="/tmp/seg.mp4"); pickle.dump(ev, open("/tmp/ev.pkl", "wb"))
    out["n_events"] = {k: int(v) for k, v in ev.type.value_counts().items()}
    ev[ev.type == "Word"][[c for c in ("start", "duration", "text") if c in ev.columns]].to_csv("/kaggle/working/bridge/words.csv", index=False)   # word-rate baseline (rule)
    ck = glob.glob("/tmp/hf/hub/models--facebook--tribev2/snapshots/*/best.ckpt")[0]; hh = hashlib.sha256(); f = open(ck, "rb")
    for b in iter(lambda: f.read(1 << 24), b""): hh.update(b)
    out["ckpt_sha256"] = hh.hexdigest()
else:
    ev = pickle.load(open("/tmp/ev.pkl", "rb"))
    if stage in ("video", "audio", "text"):
        getattr(m.data, f"{stage}_feature").prepare(ev)
    elif stage in ("predict", "predict_audio"):
        preds, segs = m.predict(events=ev, verbose=False); out["preds_shape"] = list(preds.shape)
        starts = [float(getattr(sg, "start", i)) for i, sg in enumerate(segs)]
        import numpy as np
        np.savez_compressed(f"/kaggle/working/film_am/pred_{'full' if stage == 'predict' else 'audio'}.npz", P=np.asarray(preds, dtype="float32"), starts=np.asarray(starts),
                            starts_from_segments=all(hasattr(sg, "start") for sg in segs))   # False means row index, i.e. seconds
torch.cuda.synchronize()
out.update(run_s=round(time.time() - t1, 1), torch_peak_alloc_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
           torch_peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 1e9, 2))
print("RESULT", json.dumps(out), flush=True)
''')
Path("/tmp/stage.py").write_text(STAGE)
th = threading.Thread(target=sampler, daemon=True); th.start()
res = {"stimulus": {"url": TOS_URL, "sha256": TOS_SHA, "segment": SEG_FILE, "segment_sha256": seg_sha,
                    "segment_seconds": float(dur)}, "stages": {}}
film_min = float(dur) / 60.0
for stage in ("video", "audio", "text", "predict", "predict_audio"):            # events: the intact run's, loaded above
    tries = [0] if stage != "video" else [0, 2, 1]                           # 0 = the stock batch size
    for bs in tries:
        a = time.time(); r = sh(f"{PY} /tmp/stage.py {stage} {bs}", ENVV, 3 * 3600); b = time.time()
        peak = max([m for t, i, m in smi["rows"] if a <= t <= b and i == 0] or [0]) / 1024
        line = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
        rec = {"wall_s": round(b - a, 1), "min_per_film_min": round((b - a) / 60 / film_min, 2), "smi_peak_gb_gpu0": round(peak, 2),
               "video_batch_size": bs or "stock", "ok": bool(line)}
        if line: rec.update(json.loads(line[-1][7:]))
        else: rec["error_line"] = ([l for l in r.stderr.strip().splitlines() if "Error" in l] or ["?"])[-1][:200]
        res["stages"][stage] = rec; log("STAGE", json.dumps(rec))
        if line or "out of memory" not in r.stderr.lower(): break
    if not res["stages"][stage]["ok"]:
        log(f"stage {stage} failed; stopping"); break
smi["stop"] = True
cks = glob.glob("/tmp/hf/hub/models--facebook--tribev2/snapshots/*/best.ckpt"); hh = hashlib.sha256()
if cks:
    with open(cks[0], "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): hh.update(b)
res["ckpt_sha256"] = hh.hexdigest() if cks else None; res["ckpt_matches_pin"] = res["ckpt_sha256"] == PIN   # the events stage hashed it in intact
ok = [s for s in res["stages"].values() if s["ok"]]
res["total_wall_s"] = round(sum(s["wall_s"] for s in ok), 1); res["total_min_per_film_min"] = round(res["total_wall_s"] / 60 / film_min, 2)
res["max_smi_peak_gb"] = max([s["smi_peak_gb_gpu0"] for s in ok] or [0]); res["kernel_seconds"] = round(time.time() - T0)
res["intact_export_manifest_sha256"] = hashlib.sha256(open(EXP / "MANIFEST.sha256", "rb").read()).hexdigest()
res["cache_files_added_by_stages"] = sum(1 for _ in Path("/tmp/cache").rglob("*") if _.is_file()) - n_cache
res["video"] = SEG_FILE; res["label"] = "exploratory (forecast of an edited cut, no BOLD)"
log("cache files added by the stages (video only expected):", res["cache_files_added_by_stages"])
json.dump(res, open(OUT / f"tierD_{name}_forecast.json", "w"), indent=1); log("RESULT", json.dumps({k: v for k, v in res.items() if k != "stages"}))
