"""R2 speech-to-text on a Kaggle T4 (docs/rules/r2_paranoia_rule.md, section 3: words from TRIBE v2's own speech-to-text,
ExtractWordsFromAudio as tribev2 af58661 ships it). The shipped command (uvx whisperx, large-v3, float16, batch 16, the English align
model) runs out of GPU memory on the laptop's 6 GB GPU for a full story, so this kernel runs tribev2's own _get_transcript_from_audio,
unchanged, on a 16 GB T4, and writes each table as the transform writes it (to_csv, tab-separated, no index). The laptop forecasts then
read these tables as stim.tsv, which the transform reads instead of rerunning. Stimuli: ds001338 v1.0.0 stimuli/, fetched from OpenNeuro
and checked against the pinned SHA-256 (the forecast runner's). Public audio only, no brain data; prints counts, hashes and timings only."""
import glob, hashlib, json, os, shutil, subprocess, time, urllib.request
from pathlib import Path

STIM = {"story1": "e0a674401cd64580aeb50327af4393332f387df87ddbf0a65d68a96eb55a6c3e",
        "story2": "e8b784671a748ad03f91227d1152d6219d51f713a46d6d6e41c63c17deb98e44",
        "story3": "a739113b62f1aef92a2ead0fa069d97c4e05e1478123d26f42d223b05f970034"}
URL = "https://s3.amazonaws.com/openneuro.org/ds001338/stimuli/{}_audio.wav"
OUT = Path("/kaggle/working"); ENV = "/tmp/env"; PY = f"{ENV}/bin/python"
ENVV = {**os.environ, "HF_HOME": "/tmp/hf", "HF_HUB_DISABLE_TELEMETRY": "1", "UV_CACHE_DIR": "/tmp/uv", "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "CUDA_VISIBLE_DEVICES": "0"}
ENVV.pop("MPLBACKEND", None)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()


def sh(cmd, timeout=1800):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=ENVV, timeout=timeout)
    print(f"$ {cmd[:120]}  rc={r.returncode}", flush=True)
    if r.returncode:
        print(r.stderr[-3000:], flush=True); raise SystemExit(f"failed: {cmd[:120]}")
    return r


res = {"rule": "docs/rules/r2_paranoia_rule.md", "tribev2": "af58661", "gpu": sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader").stdout.strip(),
       "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stories": {}}
sh("pip install -q uv", 600); sh(f"uv venv {ENV} --python 3.11 -q", 600)
sh(f'uv pip install -q --python {PY} "tribev2 @ git+https://github.com/facebookresearch/tribev2.git@af58661" pandas==3.0.6', 1800)
res["pip_freeze"] = sh(f"uv pip freeze --python {PY}").stdout.strip().splitlines()
RUN = ("import sys; from pathlib import Path; from tribev2.eventstransforms import ExtractWordsFromAudio as E\n"
       "w = Path(sys.argv[1]); t = E._get_transcript_from_audio(w, 'english'); t.to_csv(w.with_suffix('.tsv'), sep='\\t', index=False)\n"
       "print(len(t))")
Path("/tmp/asr.py").write_text(RUN)
for s, h in STIM.items():
    d = Path(f"/tmp/stim/{s}"); d.mkdir(parents=True, exist_ok=True); w = d / "stim.wav"
    urllib.request.urlretrieve(URL.format(s), w)
    assert sha(w) == h, f"{s}: audio differs from the pinned SHA-256"
    t0 = time.time(); n = sh(f"{PY} /tmp/asr.py {w}", 3600).stdout.strip().splitlines()[-1]
    shutil.copyfile(d / "stim.tsv", OUT / f"{s}_stim.tsv")
    res["stories"][s] = {"wav_sha256": h, "n_words": int(n), "tsv_sha256": sha(OUT / f"{s}_stim.tsv"), "seconds": round(time.time() - t0)}
    print(s, json.dumps(res["stories"][s]), flush=True)
res["whisperx"] = sorted({Path(p).name for p in glob.glob("/tmp/uv/**/whisperx-*.dist-info", recursive=True)})   # uvx's own pick: recorded
res["code_sha256"] = sha(__file__) if os.path.exists(__file__) else None
res["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
json.dump(res, open(OUT / "r2_asr.json", "w"), indent=1)
print("done", json.dumps({k: v for k, v in res.items() if k != "pip_freeze"}), flush=True)
