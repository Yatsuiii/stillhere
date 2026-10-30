"""Builds the three film forecast kernels from the one template (pilot/film/k_bridge/bridge_forecast_tpl.py; rulings of Oct 29, 06:00 UTC,
items 2 and 3b; docs/rules/film_stimulus_rule.md). Every model stage is unchanged.
- intact (stillhere-pilot-bridge-intact): reads seg_intact.mp4 from the private dataset stillhere-film-stimuli (SHA-256 asserted; the
  segment was cut once on the laptop with this kernel's former command) instead of cutting it here, so the edited cuts derive from the
  exact frames it forecasts; after the stages it exports the events pickle (transcript), the extracted audio, words.csv and the audio
  and text feature caches, with a MANIFEST.sha256.
- film-rem / film-am (stillhere-pilot-film-rem, -am): the same kernel with only the video file changed (faces_removed_v2.mp4 /
  area_matched_v2.mp4 from the same dataset, SHA-256 asserted); they load the intact run's export (attached kernel output), every file
  hash-asserted, so the transcript and the audio and text features are the intact run's; the events stage is the load; the audio and
  text stages then hit the loaded cache (files added to it are counted and logged); each saves its own output.
usage: python make_film_kernels.py STIM_MANIFEST"""
import hashlib
import json
import os
import sys

H = os.path.dirname(os.path.abspath(__file__)); FILM = os.path.dirname(H)
TPL = f"{FILM}/k_bridge/bridge_forecast_tpl.py"
TOS_SHA = "efa9062d9cdb7a338e40ad530dfdf234806743f29ae6a1a136b97ece4e588e8f"
MAN = {l.split()[1]: l.split()[0] for l in open(sys.argv[1])}
OWNER = "redlotusthepotus"


def build(version):
    s = open(TPL).read()

    def sub(old, new, count=1):
        nonlocal s
        assert s.count(old) == count, (version, old[:70], s.count(old))
        s = s.replace(old, new)
    sub('TOS_SHA = "@@TOS_SHA@@"', f'TOS_SHA = "{TOS_SHA}"')
    sub("import glob, hashlib, json, os, subprocess, textwrap, threading, time, urllib.request",
        "import glob, hashlib, json, os, shutil, subprocess, textwrap, threading, time, urllib.request")
    seg_file = {"intact": "seg_intact.mp4", "rem": "faces_removed_v2.mp4", "am": "area_matched_v2.mp4"}[version]
    get_seg = (f'SEG_FILE, SEG_SHA = "{seg_file}", "{MAN[seg_file]}"   # stillhere-film-stimuli (docs/rules/film_stimulus_rule.md)\n'
               'segs = glob.glob(f"/kaggle/input/**/{SEG_FILE}", recursive=True)\n'
               'assert len(segs) == 1, f"{len(segs)} copies of {SEG_FILE} attached (need 1)"\n'
               'shutil.copyfile(segs[0], "/tmp/seg.mp4")\n'
               'seg_sha = hashlib.sha256(open("/tmp/seg.mp4", "rb").read()).hexdigest(); log("segment", SEG_FILE, "sha256 ok:", seg_sha == SEG_SHA)\n'
               'assert seg_sha == SEG_SHA, "segment differs from the committed stimuli"\n')
    i = s.index('urllib.request.urlretrieve(TOS_URL, "/tmp/tos.mov")'); j = s.index('dur = sh("ffprobe')
    s = s[:i] + get_seg + s[j:]
    sub('"segment": "ffmpeg -ss 0 -t 190, re-encoded H.264 crf 18 + AAC"', '"segment": SEG_FILE')
    if version == "intact":
        sub('json.dump(res, open(OUT / "tierD_bridge_forecast.json", "w"), indent=1)',
            '''EXP = OUT / "export"; EXP.mkdir(exist_ok=True)                       # for film-rem and film-am: transcript, audio, text features
shutil.copyfile("/tmp/ev.pkl", EXP / "ev.pkl"); shutil.copyfile("/tmp/seg.wav", EXP / "seg.wav"); shutil.copyfile(OUT / "words.csv", EXP / "words.csv")
for d in sorted(os.listdir("/tmp/cache")):
    if d.startswith(("neuralset.extractors.audio.", "neuralset.extractors.text.")):
        shutil.copytree(f"/tmp/cache/{d}", EXP / "cache" / d)
with open(EXP / "MANIFEST.sha256", "w") as fh:
    for p in sorted(x for x in EXP.rglob("*") if x.is_file() and x.name != "MANIFEST.sha256"):
        fh.write(f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  {p.relative_to(EXP)}\\n")
res["export"] = {"files": sum(1 for _ in open(EXP / "MANIFEST.sha256")), "cache_dirs": sorted(os.listdir(EXP / "cache")) if (EXP / "cache").exists() else [],
                 "manifest_sha256": hashlib.sha256(open(EXP / "MANIFEST.sha256", "rb").read()).hexdigest()}
log("export", json.dumps({k: v for k, v in res["export"].items() if k != "cache_dirs"}), "| cache dirs", len(res["export"]["cache_dirs"]))
json.dump(res, open(OUT / "tierD_bridge_forecast.json", "w"), indent=1)''')
        kid, title, ksrc = "stillhere-pilot-bridge-intact", "stillhere-pilot-bridge-intact", []
    else:
        name = {"rem": "film_rem", "am": "film_am"}[version]
        sub('OUT = Path("/kaggle/working/bridge")', f'OUT = Path("/kaggle/working/{name}")')
        sub('np.savez_compressed(f"/kaggle/working/bridge/pred_', f'np.savez_compressed(f"/kaggle/working/{name}/pred_')
        load = ('''exps = [p for p in glob.glob("/kaggle/input/**/export/MANIFEST.sha256", recursive=True)]
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
''')
        k = s.index("\nSTAGE = textwrap.dedent"); s = s[:k + 1] + load + s[k + 1:]
        sub('for stage in ("events", "video", "audio", "text", "predict", "predict_audio"):',
            'for stage in ("video", "audio", "text", "predict", "predict_audio"):            # events: the intact run\'s, loaded above')
        sub('json.dump(res, open(OUT / "tierD_bridge_forecast.json", "w"), indent=1)',
            '''res["intact_export_manifest_sha256"] = hashlib.sha256(open(EXP / "MANIFEST.sha256", "rb").read()).hexdigest()
res["cache_files_added_by_stages"] = sum(1 for _ in Path("/tmp/cache").rglob("*") if _.is_file()) - n_cache
res["video"] = SEG_FILE; res["label"] = "exploratory (forecast of an edited cut, no BOLD)"
log("cache files added by the stages (video only expected):", res["cache_files_added_by_stages"])
json.dump(res, open(OUT / f"tierD_{name}_forecast.json", "w"), indent=1)''')
        sub('res["ckpt_matches_pin"] = res["stages"].get("events", {}).get("ckpt_sha256") == PIN',
            '''cks = glob.glob("/tmp/hf/hub/models--facebook--tribev2/snapshots/*/best.ckpt"); hh = hashlib.sha256()
if cks:
    with open(cks[0], "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): hh.update(b)
res["ckpt_sha256"] = hh.hexdigest() if cks else None; res["ckpt_matches_pin"] = res["ckpt_sha256"] == PIN   # the events stage hashed it in intact''')
        kid = title = f"stillhere-pilot-film-{version}"; ksrc = [f"{OWNER}/stillhere-pilot-bridge-intact"]
    compile(s, kid, "exec")
    d = f"{H}/{version}"; os.makedirs(d, exist_ok=True)
    code = f"{kid.replace('-', '_')}.py"
    open(f"{d}/{code}", "w").write(s)
    json.dump({"id": f"{OWNER}/{kid}", "title": title, "code_file": code, "language": "python", "kernel_type": "script", "is_private": True,
               "enable_gpu": True, "machine_shape": "NvidiaTeslaT4", "enable_internet": True,
               "dataset_sources": [f"{OWNER}/stillhere-hf", f"{OWNER}/stillhere-film-stimuli"], "competition_sources": [], "model_sources": [],
               "kernel_sources": ksrc}, open(f"{d}/kernel-metadata.json", "w"))
    return f"{d}/{code}", hashlib.sha256(s.encode()).hexdigest()


for v in ("intact", "rem", "am"):
    p, h = build(v); print(v, os.path.relpath(p, FILM), h[:16])
