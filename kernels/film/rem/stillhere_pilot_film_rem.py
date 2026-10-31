"""Opening forecast (StillHere Pilot, public data only; rule docs/rules/emofilm_rule.md version 2): stock TRIBE v2 full and audio-only
predictions for release 0 to 190 s of Tears of Steel (Blender Foundation 2012, CC BY 3.0): the scored opening three minutes (0 to 180 s)
plus a 10 s margin so prediction samples exist through 180 s (ffmpeg -ss 0 -t 190). It also covers the bridge conversation (25.00 to
40.25 s) used for the SAM 3.1 edits and the video. The kernel keeps its bridge name. Version: intact. One T4 (CUDA_VISIBLE_DEVICES=0). Stages run as separate processes so each extractor's
memory is freed before the next: events (speech-to-text on the segment's audio, the harness's own ExtractWordsFromAudio), video
(V-JEPA2), audio (w2v-BERT), text (Llama-3.2-3B, 'accelerate' as in the stock predictions), predict (cached features, model forward).
Route (cloud rulings of Oct 31, 05:15 and 05:30 UTC): events are built in the harness's order (tribe_common.build_events: words from the
whole audio first, then chunking), not tribev2's get_events_dataframe, which chunks first and so duplicated words past 60 s; the word
table is checked before any feature stage (no start at or past the segment end, no block of words repeated at one exact shift).
Predictions come from the reference harness, tribe_common.stitched_predict (byte for byte, below): 100 s windows at a 50 s stride, one
sorted row per second. Pinned: the package lock (LOCK), the Hugging Face revisions of every model the stages load (HF_PINS; a stage that
loads another revision stops the kernel), and one T4; the JSON records pip freeze, the revisions and the GPU.
Per stage: wall seconds, minutes per film minute, torch peak allocated and reserved GB, nvidia-smi peak GB. If the video stage runs out of
memory at the stock batch size it is retried at smaller batch sizes (memory only; features unchanged). Prints and writes numbers only."""
import glob, hashlib, json, os, shutil, subprocess, textwrap, threading, time, urllib.request
from pathlib import Path
TOS_URL = "https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov"
TOS_SHA = "efa9062d9cdb7a338e40ad530dfdf234806743f29ae6a1a136b97ece4e588e8f"
PIN = "9c79ffff6b642b7b0c71d558c935fb3fa33f2788bfb509feead94fafbba2f321"          # stock TRIBE v2 checkpoint (d22/d3, stillhere-pred)
HF_PINS = {'facebook/tribev2': 'f894e783020944dcd96e5568550afe2aa9743f9f', 'facebook/vjepa2-vitg-fpc64-256': '875c192b7b704b87d1e1d99345769632dd5f739a', 'facebook/w2v-bert-2.0': 'da985ba0987f70aaeb84a80f2851cfac8c697a7b', 'meta-llama/Llama-3.2-3B': '13afe5124825b4f3751f836b40dafda64c1ed062'}
LOCK = 'accelerate==1.15.0\nannotated-doc==0.0.5\nannotated-types==0.8.0\nanyio==4.15.1\nblis==1.3.4\ncatalogue==2.0.10\ncertifi==2026.10.23\ncffi==2.1.1\ncharset-normalizer==3.5.2\nclick==8.1.8\ncloudpathlib==0.26.0\ncloudpickle==3.1.2\nconfection==1.3.4\ncontourpy==1.3.3\ncycler==0.12.1\ncymem==2.0.16\ndecorator==5.3.1\neinops==0.8.2\nexca==0.5.20\nfilelock==4.2.5\nfonttools==4.68.0\nfsspec==2026.9.0\ngtts==2.5.4\nh11==0.16.0\nhf-xet==1.8.0\nhttpcore==1.0.9\nhttpx==0.28.1\nhuggingface-hub==1.16.1\nidna==3.20\nimageio==2.38.1\nimageio-ffmpeg==0.7.0\nimportlib-resources==7.1.0\njinja2==3.1.6\njoblib==1.6.0\njulius==0.2.8\nkiwisolver==1.5.1\nlangdetect==1.0.9\nlazy-loader==0.6\nlevenshtein==0.27.5\nlightning-utilities==0.15.3\nmarkdown-it-py==4.2.0\nmarkupsafe==3.0.4\nmatplotlib==3.11.2\nmdurl==0.1.2\nmne==1.13.2\nmne-bids==0.20.0\nmoviepy==2.2.1\nmpmath==1.3.0\nmurmurhash==1.0.16\nnarwhals==2.28.0\nnetworkx==3.6.1\nneuralset==0.0.2\nneuraltrain==0.0.2\nnibabel==5.4.2\nnumpy==2.2.6\nnvidia-cublas-cu12==12.4.5.8\nnvidia-cuda-cupti-cu12==12.4.127\nnvidia-cuda-nvrtc-cu12==12.4.127\nnvidia-cuda-runtime-cu12==12.4.127\nnvidia-cudnn-cu12==9.1.0.70\nnvidia-cufft-cu12==11.2.1.3\nnvidia-curand-cu12==10.3.5.147\nnvidia-cusolver-cu12==11.6.1.9\nnvidia-cusparse-cu12==12.3.1.170\nnvidia-cusparselt-cu12==0.6.2\nnvidia-nccl-cu12==2.21.5\nnvidia-nvjitlink-cu12==12.4.127\nnvidia-nvtx-cu12==12.4.127\norjson==3.13.0\npackaging==26.3\npandas==3.1.0\npillow==11.3.0\npip==26.2.2\nplatformdirs==4.14.0\npolars==2.1.0\npolars-runtime-32==2.1.0\npooch==1.9.0\npreshed==3.0.15\nproglog==0.1.12\npsutil==7.2.2\npyarrow==26.0.0\npycparser==3.12\npydantic==2.14.2\npydantic-core==2.50.2\npygments==2.21.0\npyparsing==3.3.3\npyprep==0.10.0\npython-dateutil==3.0.0\npython-dotenv==1.2.4\npyyaml==6.0.3\nrapidfuzz==3.14.6\nregex==2026.9.29\nrequests==2.34.2\nrich==15.0.0\nsafetensors==0.8.0\nscikit-learn==1.9.1\nscipy==1.17.1\nsetuptools==84.2.1\nshellingham==1.5.4\nsix==1.17.0\nsmart-open==9.0.1\nsoundfile==0.14.0\nspacy==3.8.14\nspacy-legacy==3.0.12\nspacy-loggers==1.0.5\nsrsly==2.5.5\nsubmitit==1.5.4\nsympy==1.13.1\nthinc==8.3.14\nthreadpoolctl==3.7.0\ntokenizers==0.23.3\ntorch==2.6.0\ntorchmetrics==1.9.0\ntorchvision==0.21.0\ntqdm==4.70.1\ntransformers==5.17.0\ntribev2 @ git+https://github.com/facebookresearch/tribev2.git@af58661791a351a448a489042a28f6c37e1c14b7\ntriton==3.2.0\ntyper==0.27.3\ntyping-extensions==4.16.0\ntyping-inspection==0.4.4\ntzdata==2026.6\nujson==6.0.0\nurllib3==2.8.0\nwasabi==1.1.3\nweasel==1.0.1\nwrapt==2.5.2\nx-transformers==1.27.20\n'
STITCH = 'def stitched_predict(m, ev, aud_end):\n    """50 s overlapping windows; per second keep the value from the window where it is closest to the centre."""\n    loader = m.data.get_loaders(events=ev, split_to_build="all")["all"]\n    best = {}\n    n_win = 0\n    with torch.inference_mode():\n        for batch in loader:\n            y = m._model(batch.to(m._model.device)).cpu().numpy()      # (B, 20484, T)\n            T = y.shape[-1]\n            for b, seg in enumerate(batch.segments):\n                n_win += 1\n                for k in range(T):\n                    t = int(round(seg.start + k)); d = abs(k + 0.5 - T / 2)\n                    if 0 <= t < aud_end and (t not in best or d < best[t][0]):\n                        best[t] = (d, y[b, :, k])\n    starts = np.array(sorted(best))\n    return starts, np.stack([best[t][1] for t in starts]), n_win'
OUT = Path("/kaggle/working/film_rem"); OUT.mkdir(parents=True, exist_ok=True); LOG = OUT / "driver.log"
ENV = Path("/tmp/tribe-env"); PY = ENV / "bin" / "python"; T0 = time.time()


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); open(LOG, "a").write(s + "\n")


def sh(cmd, env=None, timeout=None):
    t = time.time(); r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env={**os.environ, **(env or {})}, timeout=timeout)
    log(f"$ {cmd[:110]}  rc={r.returncode}  {time.time() - t:.0f}s")
    if r.returncode: log("STDERR tail:", r.stderr[-1500:])
    return r


def revisions():
    """The commit each downloaded model resolved to (its refs/main), from the Hugging Face cache."""
    out = {}
    for d in glob.glob("/tmp/hf/hub/models--*"):
        ref = Path(d) / "refs" / "main"
        if ref.exists(): out[Path(d).name[len("models--"):].replace("--", "/")] = ref.read_text().strip()
    return out


smi = {"rows": [], "stop": False}
def sampler():
    while not smi["stop"]:
        r = subprocess.run("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits", shell=True, capture_output=True, text=True)
        t = time.time()
        for line in r.stdout.strip().splitlines():
            i, m = [x.strip() for x in line.split(",")]; smi["rows"].append((t, int(i), int(m)))
        time.sleep(0.5)


log("gpus:", sh("nvidia-smi -L").stdout.strip().replace("\n", " | "))
GPU = sh("nvidia-smi --query-gpu=name --format=csv,noheader -i 0").stdout.strip()
tok = None
for h in glob.glob("/kaggle/input/**/hf_token", recursive=True): tok = Path(h).read_text().strip()
if not tok: log("no HF token; exiting"); raise SystemExit(0)
ENVV = {"HF_HOME": "/tmp/hf", "HF_HUB_DISABLE_TELEMETRY": "1", "UV_CACHE_DIR": "/tmp/uv", "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "HF_TOKEN": tok, "CUDA_VISIBLE_DEVICES": "0"}
Path("/tmp/film_lock.txt").write_text(LOCK); Path("/tmp/k2code").mkdir(exist_ok=True)
Path("/tmp/k2code/stitch.py").write_text("import numpy as np, torch\n\n\n" + STITCH + "\n")
sh("pip install -q uv", timeout=600); sh(f"uv venv {ENV} --python 3.11 -q", ENVV, 600)
sh(f'uv pip install -q --python {PY} -c /tmp/film_lock.txt "tribev2 @ git+https://github.com/facebookresearch/tribev2.git@af58661" accelerate soundfile pandas', ENVV, 1800)
FREEZE = sh(f"uv pip freeze --python {PY}", ENVV).stdout.strip().splitlines()
SEG_FILE, SEG_SHA = "faces_removed_v2.mp4", "ddaecc4c989fc9f8b267af5ff318897e170f39108ab679c419f66e3498222bf3"   # stillhere-film-stimuli (docs/rules/film_stimulus_rule.md)
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
cache_before = set(os.listdir("/tmp/cache")); files_before = {p for p in Path("/tmp/cache").rglob("*") if p.is_file()}
log("intact export loaded:", sum(1 for _ in open(EXP / "MANIFEST.sha256")), "files hash-checked | cache files", n_cache,
    "| manifest", hashlib.sha256(open(EXP / "MANIFEST.sha256", "rb").read()).hexdigest()[:16])
STAGE = textwrap.dedent('''
import sys, json, time, pickle, hashlib, glob, os
import torch
stage, bs, seg_s = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
from tribev2.demo_utils import TribeModel
cu = {"data.num_workers": 0, "data.features_to_use": ["text", "audio", "video"], "data.text_feature.device": "accelerate", "data.overlap_trs_train": 50}
if bs: cu["data.video_feature.image.batch_size"] = bs
if stage == "predict_audio": cu["data.features_to_use"] = ["audio"]
t0 = time.time()
m = TribeModel.from_pretrained("facebook/tribev2", cache_folder="/tmp/cache", config_update=cu)
base = torch.cuda.memory_allocated() / 1e9
torch.cuda.reset_peak_memory_stats(); t1 = time.time(); out = {"stage": stage, "load_s": round(t1 - t0, 1), "brain_model_gb": round(base, 2)}
if stage == "events":
    # tribe_common.build_events' order with the video lane: words from the whole audio first, then chunking (tribev2's own
    # get_events_dataframe chunks first, which duplicated words past 60 s)
    import pandas as pd
    from collections import Counter
    from neuralset.events.transforms import AddContextToWords, AddSentenceToWords, AddText, ChunkEvents, ExtractAudioFromVideo, RemoveMissing
    from neuralset.events.utils import standardize_events
    from tribev2.eventstransforms import ExtractWordsFromAudio                     # tribev2's own imports (demo_utils.py) are these
    ev = standardize_events(pd.DataFrame([{"type": "Video", "filepath": "/tmp/seg.mp4", "start": 0, "timeline": "default", "subject": "default"}]))
    for t in [ExtractAudioFromVideo(), ExtractWordsFromAudio(language="english"), AddText(), AddSentenceToWords(max_unmatched_ratio=0.05),
              AddContextToWords(sentence_only=False, max_context_len=1024, split_field=""), RemoveMissing(),
              ChunkEvents(event_type_to_chunk="Audio", max_duration=60, min_duration=30), ChunkEvents(event_type_to_chunk="Video", max_duration=60, min_duration=30)]:
        ev = t(ev)
    ev = standardize_events(ev); pickle.dump(ev, open("/tmp/ev.pkl", "wb"))
    out["n_events"] = {k: int(v) for k, v in ev.type.value_counts().items()}
    wd = ev[ev.type == "Word"]
    wd[[c for c in ("start", "duration", "text") if c in wd.columns]].to_csv("/kaggle/working/bridge/words.csv", index=False)   # word-rate baseline (rule)
    by = {}
    for txt, s in zip(wd.text, wd.start): by.setdefault(txt, []).append(round(float(s), 3))
    shifts = Counter(round(b - a, 3) for ts in by.values() for i, a in enumerate(ts) for b in ts[i + 1:] if b > a)
    worst = max(shifts.items(), key=lambda kv: kv[1], default=(None, 0))
    out["words"] = {"n": int(len(wd)), "last_start_s": float(wd.start.max()) if len(wd) else None, "segment_seconds": seg_s,
                    "exact_shift_copies": int(worst[1]) if worst[1] >= max(5, 0.1 * len(wd)) else 0, "worst_shift_s": worst[0], "worst_shift_count": int(worst[1])}
    out["words_ok"] = bool(len(wd) and out["words"]["last_start_s"] < seg_s and out["words"]["exact_shift_copies"] == 0)
    ck = glob.glob("/tmp/hf/hub/models--facebook--tribev2/snapshots/*/best.ckpt")[0]; hh = hashlib.sha256(); f = open(ck, "rb")
    for b in iter(lambda: f.read(1 << 24), b""): hh.update(b)
    out["ckpt_sha256"] = hh.hexdigest()
else:
    ev = pickle.load(open("/tmp/ev.pkl", "rb"))
    if stage in ("video", "audio", "text"):
        getattr(m.data, f"{stage}_feature").prepare(ev)
    elif stage in ("predict", "predict_audio"):
        sys.path.insert(0, "/tmp/k2code"); import numpy as np
        from stitch import stitched_predict                                  # tribe_common.stitched_predict, byte for byte
        starts, P, nwin = stitched_predict(m, ev, float(ev[ev.type == "Audio"].stop.max()))
        out.update(preds_shape=list(P.shape), n_windows=nwin, one_row_per_second=bool(len(starts) and starts[0] == 0 and (np.diff(starts) == 1).all()))
        np.savez_compressed(f"/kaggle/working/film_rem/pred_{'full' if stage == 'predict' else 'audio'}.npz", P=P.astype("float32"), starts=starts)
torch.cuda.synchronize()
out.update(run_s=round(time.time() - t1, 1), torch_peak_alloc_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
           torch_peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 1e9, 2))
print("RESULT", json.dumps(out), flush=True)
''')
Path("/tmp/stage.py").write_text(STAGE)
th = threading.Thread(target=sampler, daemon=True); th.start()
res = {"stimulus": {"url": TOS_URL, "sha256": TOS_SHA, "segment": SEG_FILE, "segment_sha256": seg_sha,
                    "segment_seconds": float(dur)}, "stages": {}, "gpu": GPU, "hf_pins": HF_PINS,
       "route": {"events": "words first, then chunking (tribe_common.build_events order)", "predict": "tribe_common.stitched_predict"}}
film_min = float(dur) / 60.0
for stage in ("video", "audio", "text", "predict", "predict_audio"):            # events: the intact run's, loaded above
    tries = [0] if stage != "video" else [0, 2, 1]                           # 0 = the stock batch size
    for bs in tries:
        a = time.time(); r = sh(f"{PY} /tmp/stage.py {stage} {bs} {dur}", ENVV, 3 * 3600); b = time.time()
        peak = max([m for t, i, m in smi["rows"] if a <= t <= b and i == 0] or [0]) / 1024
        line = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
        rec = {"wall_s": round(b - a, 1), "min_per_film_min": round((b - a) / 60 / film_min, 2), "smi_peak_gb_gpu0": round(peak, 2),
               "video_batch_size": bs or "stock", "ok": bool(line)}
        if line: rec.update(json.loads(line[-1][7:]))
        else: rec["error_line"] = ([l for l in r.stderr.strip().splitlines() if "Error" in l] or ["?"])[-1][:200]
        off = {k: v for k, v in revisions().items() if k in HF_PINS and v != HF_PINS[k]}
        if off: rec["ok"] = False; rec["revision_mismatch"] = off
        if rec.get("words_ok") is False: rec["ok"] = False
        res["stages"][stage] = rec; log("STAGE", json.dumps(rec))
        if line or "out of memory" not in r.stderr.lower(): break
    if "words" in res["stages"][stage]: res["words"] = res["stages"][stage]["words"]
    if not res["stages"][stage]["ok"]:
        log(f"stage {stage} failed; stopping"); break
smi["stop"] = True
cks = glob.glob("/tmp/hf/hub/models--facebook--tribev2/snapshots/*/best.ckpt"); hh = hashlib.sha256()
if cks:
    with open(cks[0], "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): hh.update(b)
res["ckpt_sha256"] = hh.hexdigest() if cks else None; res["ckpt_matches_pin"] = res["ckpt_sha256"] == PIN   # the events stage hashed it in intact
res["whisperx"] = sorted({Path(p).name for p in glob.glob("/tmp/uv/**/whisperx-*.dist-info", recursive=True)})   # uvx's own pick: recorded, not pinned
res["hf_revisions"] = revisions(); res["pip_freeze"] = FREEZE; res["pip_freeze_sha256"] = hashlib.sha256("\n".join(FREEZE).encode()).hexdigest()
ok = [s for s in res["stages"].values() if s["ok"]]
res["total_wall_s"] = round(sum(s["wall_s"] for s in ok), 1); res["total_min_per_film_min"] = round(res["total_wall_s"] / 60 / film_min, 2)
res["max_smi_peak_gb"] = max([s["smi_peak_gb_gpu0"] for s in ok] or [0]); res["kernel_seconds"] = round(time.time() - T0)
res["intact_export_manifest_sha256"] = hashlib.sha256(open(EXP / "MANIFEST.sha256", "rb").read()).hexdigest()
res["cache_files_added_by_stages"] = sum(1 for _ in Path("/tmp/cache").rglob("*") if _.is_file()) - n_cache
res["cache_dirs_added"] = sorted(set(os.listdir("/tmp/cache")) - cache_before)
res["cache_files_added_by_dir"] = {}                                       # every file the stages added, by its top cache folder
for p in sorted({p for p in Path("/tmp/cache").rglob("*") if p.is_file()} - files_before):
    top = p.relative_to("/tmp/cache").parts[0]; res["cache_files_added_by_dir"][top] = res["cache_files_added_by_dir"].get(top, 0) + 1
res["video"] = SEG_FILE; res["label"] = "exploratory (forecast of an edited cut, no BOLD)"
log("cache files added by the stages (video only expected):", res["cache_files_added_by_stages"])
json.dump(res, open(OUT / "tierD_film_rem_forecast.json", "w"), indent=1); log("RESULT", json.dumps({k: v for k, v in res.items() if k not in ("stages", "pip_freeze")}))
