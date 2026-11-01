"""R2 stock forecasts on a Kaggle T4, part B (docs/rules/r2_paranoia_rule.md, amendment of Nov 1 2026): stock TRIBE v2 on the three
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
stage_stock_<story>_<mode>.json (the runner's format), words_<story>.tsv, r2_forecast_partB.json and MANIFEST.sha256; prints counts,
shapes, hashes and timings, never word text."""
import glob, hashlib, json, os, shutil, subprocess, time, urllib.request
from pathlib import Path
PART = "B"
STORIES = ("story1", "story2", "story3")
STAGES = [f"stock_{s}_{m}" for s in STORIES for m in (("audio", "full") if PART == "A" else ("shuf",))]
STIM_SHA = {'story1': 'e0a674401cd64580aeb50327af4393332f387df87ddbf0a65d68a96eb55a6c3e', 'story2': 'e8b784671a748ad03f91227d1152d6219d51f713a46d6d6e41c63c17deb98e44', 'story3': 'a739113b62f1aef92a2ead0fa069d97c4e05e1478123d26f42d223b05f970034'}
FULLTEXT_SHA = "c762b04ed6a1141391c9537b9eddb3e77bb18824d1adff968e300d7d70aae82c"
WORDS_SHA = {'story1': '1a98288a1b8d48a468ce0bb6b18cc82a50db6a829d2c5fe0789b9339c185f8aa', 'story2': 'd3d07e7b93a35dfd44748c16e3f626021d98e6d573206658da5ca06f30b82bb9', 'story3': 'db9968150bc263c7d1f82f9cf88c31ec41e20f6e486b21e4d717768144a06e6f'}
URL = "https://s3.amazonaws.com/openneuro.org/ds001338/stimuli/{}"
PIN = "9c79ffff6b642b7b0c71d558c935fb3fa33f2788bfb509feead94fafbba2f321"          # stock TRIBE v2 checkpoint
HF_PINS = {'facebook/tribev2': 'f894e783020944dcd96e5568550afe2aa9743f9f', 'facebook/vjepa2-vitg-fpc64-256': '875c192b7b704b87d1e1d99345769632dd5f739a', 'facebook/w2v-bert-2.0': 'da985ba0987f70aaeb84a80f2851cfac8c697a7b', 'meta-llama/Llama-3.2-3B': '13afe5124825b4f3751f836b40dafda64c1ed062'}
LOCK = 'accelerate==1.15.0\nannotated-doc==0.0.5\nannotated-types==0.8.0\nanyio==4.15.1\nblis==1.3.4\ncatalogue==2.0.10\ncertifi==2026.10.23\ncffi==2.1.1\ncharset-normalizer==3.5.2\nclick==8.1.8\ncloudpathlib==0.26.0\ncloudpickle==3.1.2\nconfection==1.3.4\ncontourpy==1.3.3\ncycler==0.12.1\ncymem==2.0.16\ndecorator==5.3.1\neinops==0.8.2\nexca==0.5.20\nfilelock==4.2.5\nfonttools==4.68.0\nfsspec==2026.9.0\ngtts==2.5.4\nh11==0.16.0\nhf-xet==1.8.0\nhttpcore==1.0.9\nhttpx==0.28.1\nhuggingface-hub==1.16.1\nidna==3.20\nimageio==2.38.1\nimageio-ffmpeg==0.7.0\nimportlib-resources==7.1.0\njinja2==3.1.6\njoblib==1.6.0\njulius==0.2.8\nkiwisolver==1.5.1\nlangdetect==1.0.9\nlazy-loader==0.6\nlevenshtein==0.27.5\nlightning-utilities==0.15.3\nmarkdown-it-py==4.2.0\nmarkupsafe==3.0.4\nmatplotlib==3.11.2\nmdurl==0.1.2\nmne==1.13.2\nmne-bids==0.20.0\nmoviepy==2.2.1\nmpmath==1.3.0\nmurmurhash==1.0.16\nnarwhals==2.28.0\nnetworkx==3.6.1\nneuralset==0.0.2\nneuraltrain==0.0.2\nnibabel==5.4.2\nnumpy==2.2.6\nnvidia-cublas-cu12==12.4.5.8\nnvidia-cuda-cupti-cu12==12.4.127\nnvidia-cuda-nvrtc-cu12==12.4.127\nnvidia-cuda-runtime-cu12==12.4.127\nnvidia-cudnn-cu12==9.1.0.70\nnvidia-cufft-cu12==11.2.1.3\nnvidia-curand-cu12==10.3.5.147\nnvidia-cusolver-cu12==11.6.1.9\nnvidia-cusparse-cu12==12.3.1.170\nnvidia-cusparselt-cu12==0.6.2\nnvidia-nccl-cu12==2.21.5\nnvidia-nvjitlink-cu12==12.4.127\nnvidia-nvtx-cu12==12.4.127\norjson==3.13.0\npackaging==26.3\npandas==3.1.0\npillow==11.3.0\npip==26.2.2\nplatformdirs==4.14.0\npolars==2.1.0\npolars-runtime-32==2.1.0\npooch==1.9.0\npreshed==3.0.15\nproglog==0.1.12\npsutil==7.2.2\npyarrow==26.0.0\npycparser==3.12\npydantic==2.14.2\npydantic-core==2.50.2\npygments==2.21.0\npyparsing==3.3.3\npyprep==0.10.0\npython-dateutil==3.0.0\npython-dotenv==1.2.4\npyyaml==6.0.3\nrapidfuzz==3.14.6\nregex==2026.9.29\nrequests==2.34.2\nrich==15.0.0\nsafetensors==0.8.0\nscikit-learn==1.9.1\nscipy==1.17.1\nsetuptools==84.2.1\nshellingham==1.5.4\nsix==1.17.0\nsmart-open==9.0.1\nsoundfile==0.14.0\nspacy==3.8.14\nspacy-legacy==3.0.12\nspacy-loggers==1.0.5\nsrsly==2.5.5\nsubmitit==1.5.4\nsympy==1.13.1\nthinc==8.3.14\nthreadpoolctl==3.7.0\ntokenizers==0.23.3\ntorch==2.6.0\ntorchmetrics==1.9.0\ntorchvision==0.21.0\ntqdm==4.70.1\ntransformers==5.17.0\ntribev2 @ git+https://github.com/facebookresearch/tribev2.git@af58661791a351a448a489042a28f6c37e1c14b7\ntriton==3.2.0\ntyper==0.27.3\ntyping-extensions==4.16.0\ntyping-inspection==0.4.4\ntzdata==2026.6\nujson==6.0.0\nurllib3==2.8.0\nwasabi==1.1.3\nweasel==1.0.1\nwrapt==2.5.2\nx-transformers==1.27.20\n'
COMMON = '"""Shared TRIBE helpers for the K2 smoke notebook and the local test. PUBLIC audio only (LPP section 1).\nThrowaway pre-event code, to be disclosed as discarded. Never print word text: LPP text is copyrighted."""\nimport re, shutil\nfrom pathlib import Path\nimport numpy as np, pandas as pd, soundfile as sf, torch\n\n\ndef textgrid_words(path):\n    """Words tier of an LPP TextGrid: (text, start, duration) for non-empty, non-pause intervals."""\n    b = open(path, "rb").read()\n    s = b.decode("utf-16") if b[:2] in (b"\\xff\\xfe", b"\\xfe\\xff") else b.decode("utf-8")   # FR is UTF-16, CN is UTF-8\n    it = re.findall(r\'xmin = ([\\d.]+)\\s*xmax = ([\\d.]+)\\s*text = "(.*?)"\', s, flags=re.S)\n    return [(t.strip(), float(a), float(b) - float(a)) for a, b, t in it if t.strip() and t.strip().lower() not in ("sp", "<p>")]\n\n\ndef write_tsv(words, wav_path, shuffle_seed=None, t_max=None):\n    """Write <wav>.tsv in the format ExtractWordsFromAudio reads (skips ASR). shuffle_seed: permute the word\n    texts across the original onsets (the P8 shuffled-word control)."""\n    w = [x for x in words if t_max is None or x[1] < t_max]\n    texts = [x[0] for x in w]\n    if shuffle_seed is not None:\n        np.random.default_rng(shuffle_seed).shuffle(texts)\n    df = pd.DataFrame({"text": texts, "start": [x[1] for x in w], "duration": [x[2] for x in w],\n                       "sequence_id": 0, "sentence": ""})\n    df.to_csv(Path(wav_path).with_suffix(".tsv"), sep="\\t", index=False)\n    return len(df)\n\n\ndef prepare_dir(d, src_wav, words, shuffle_seed=None, seconds=None):\n    """Make dir d with stim.wav (optionally cut to `seconds`) and stim.tsv."""\n    d = Path(d); d.mkdir(parents=True, exist_ok=True)\n    wav = d / "stim.wav"\n    if seconds is None:\n        shutil.copy(src_wav, wav)\n    else:\n        x, sr = sf.read(src_wav, dtype="int16", always_2d=True)\n        sf.write(wav, x[: int(seconds * sr)], sr, subtype="PCM_16")\n    n = write_tsv(words, wav, shuffle_seed=shuffle_seed, t_max=seconds)\n    return wav, n\n\n\ndef build_events(wav, language="french", with_text=True):\n    """Training order (words first, then chunking) to avoid the >60 s duplication bug (PRs #29/#81)."""\n    from neuralset.events.transforms import AddContextToWords, AddSentenceToWords, AddText, ChunkEvents, RemoveMissing\n    from neuralset.events.utils import standardize_events\n    from tribev2.eventstransforms import ExtractWordsFromAudio\n    ev = standardize_events(pd.DataFrame([{"type": "Audio", "filepath": str(wav), "start": 0,\n                                           "timeline": "default", "subject": "default"}]))\n    ts = []\n    if with_text:\n        ts += [ExtractWordsFromAudio(language=language), AddText(), AddSentenceToWords(max_unmatched_ratio=0.05),\n               AddContextToWords(sentence_only=False, max_context_len=1024, split_field=""), RemoveMissing()]\n    ts += [ChunkEvents(event_type_to_chunk="Audio", max_duration=60, min_duration=30)]\n    for t in ts:\n        ev = t(ev)\n    return standardize_events(ev)\n\n\ndef event_checks(ev, n_tsv_words, tsv_after60):\n    aud = ev[ev.type == "Audio"]; wd = ev[ev.type == "Word"]\n    aud_end = float(aud.stop.max())\n    return {"n_audio_events": int(len(aud)), "audio_end_s": aud_end, "n_words": int(len(wd)), "n_tsv_words": int(n_tsv_words),\n            "words_equal_tsv": bool(len(wd) == n_tsv_words) if len(wd) else None,\n            "n_words_after_60s": int((wd.start > 60).sum()) if len(wd) else 0, "tsv_after_60s": int(tsv_after60),\n            "last_word_before_audio_end": bool(wd.start.max() < aud_end) if len(wd) else None}\n\n\ndef load_model(mode_features, cache, text_device="accelerate", extra=None):\n    from tribev2.demo_utils import TribeModel\n    cu = {"data.num_workers": 0, "data.features_to_use": mode_features, "data.text_feature.device": text_device,\n          "data.overlap_trs_train": 50}\n    cu.update(extra or {})\n    return TribeModel.from_pretrained("facebook/tribev2", cache_folder=str(cache), config_update=cu)\n\n\ndef stitched_predict(m, ev, aud_end):\n    """50 s overlapping windows; per second keep the value from the window where it is closest to the centre."""\n    loader = m.data.get_loaders(events=ev, split_to_build="all")["all"]\n    best = {}\n    n_win = 0\n    with torch.inference_mode():\n        for batch in loader:\n            y = m._model(batch.to(m._model.device)).cpu().numpy()      # (B, 20484, T)\n            T = y.shape[-1]\n            for b, seg in enumerate(batch.segments):\n                n_win += 1\n                for k in range(T):\n                    t = int(round(seg.start + k)); d = abs(k + 0.5 - T / 2)\n                    if 0 <= t < aud_end and (t not in best or d < best[t][0]):\n                        best[t] = (d, y[b, :, k])\n    starts = np.array(sorted(best))\n    return starts, np.stack([best[t][1] for t in starts]), n_win\n\n\ndef sha256(p):\n    import hashlib\n    h = hashlib.sha256(); f = open(p, "rb")\n    for b in iter(lambda: f.read(1 << 24), b""): h.update(b)\n    return h.hexdigest()\n\n\nSTOCK_PIN = "9c79ffff6b642b7b0c71d558c935fb3fa33f2788bfb509feead94fafbba2f321"\nTUNED_PIN = "96923860eb47ee8e8f65e597f2fc216992470852dc07af95112d80d32746421e"\n\n\ndef apply_weights(m, model):\n    """Stock: check the released best.ckpt against the pin. Tuned: also load tuned.ckpt (hash-checked) strictly into the same model."""\n    import glob, os\n    ck = glob.glob(os.environ.get("HF_HOME", "/tmp/hf") + "/hub/models--facebook--tribev2/snapshots/*/best.ckpt")[0]\n    info = {"stock_ckpt_sha256": sha256(ck)}\n    assert info["stock_ckpt_sha256"] == STOCK_PIN, "stock checkpoint differs from the pin"\n    if model == "tuned":\n        tp = glob.glob("/kaggle/input/**/tuned.ckpt", recursive=True)\n        assert len(tp) == 1, "tuned.ckpt not found exactly once"\n        info["tuned_ckpt_sha256"] = sha256(tp[0]); assert info["tuned_ckpt_sha256"] == TUNED_PIN, "tuned checkpoint differs from the pin"\n        sd = torch.load(tp[0], map_location="cpu", weights_only=False)["state_dict"]\n        sd = {k[len("model."):]: v for k, v in sd.items()}\n        before = {k: v.detach().cpu().clone() for k, v in m._model.state_dict().items()}\n        m._model.load_state_dict(sd, strict=True)\n        after = m._model.state_dict()\n        info["tuned_tensors"] = len(sd); info["tensors_changed_vs_stock"] = int(sum(not torch.equal(before[k], after[k].detach().cpu()) for k in before))\n        assert info["tensors_changed_vs_stock"] > 0\n    info["model"] = model; info["training_mode"] = bool(m._model.training)\n    return info\n\n\ndef builtin_predict(m, ev):\n    preds, segs = m.predict(events=ev, verbose=False)\n    return np.array([s.start for s in segs]), preds\n\n\ndef gpu_stats():\n    if not torch.cuda.is_available():\n        return {}\n    return {"peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}\n\n\ndef parcels(annot_dir):\n    """Schaefer-400/17 on fsaverage5 as vertex-index lists in TRIBE order (lh first). AUD 7, LANG 19."""\n    import nibabel.freesurfer as fs\n    labs = {}\n    for h in ("lh", "rh"):\n        l, ct, names = fs.read_annot(f"{annot_dir}/{h}.Schaefer2018_400Parcels_17Networks_order.annot")\n        labs[h] = (l, [n.decode() for n in names])\n    def pick(h, pats, off):\n        l, names = labs[h]\n        return [np.where(l == i)[0] + off for i, n in enumerate(names) if any(p in n for p in pats)]\n    out = {"AUD": pick("lh", ["SomMotB_Aud"], 0) + pick("rh", ["SomMotB_Aud"], 10242),\n           "LANG": pick("lh", ["TempPar", "DefaultB_Temp", "DefaultB_PFCv", "DefaultB_PFCl"], 0)}\n    nets = {}\n    for h, off in (("lh", 0), ("rh", 10242)):\n        l, names = labs[h]\n        for i, n in enumerate(names):\n            if n.startswith("17Networks_"):      # e.g. 17Networks_LH_SomMotB_Aud_1 -> network key SomMotB\n                nets.setdefault(n.split("_")[2], []).append(np.where(l == i)[0] + off)\n    out["NETS"] = nets\n    return out\n\n\ndef corr_cols(a, b):\n    a = (a - a.mean(0)) / a.std(0); b = (b - b.mean(0)) / b.std(0)\n    return (a * b).mean(0)\n\n\ndef separability(Pa, Pb, parc):\n    """Per set/network: mean over parcels of r(parcel-mean series of Pa, of Pb) and 1 - r^2."""\n    res = {}\n    sets = {"AUD": parc["AUD"], "LANG": parc["LANG"], **{f"net_{k}": v for k, v in parc["NETS"].items()}}\n    for name, vs in sets.items():\n        A = np.stack([Pa[:, v].mean(1) for v in vs], 1); B = np.stack([Pb[:, v].mean(1) for v in vs], 1)\n        r = float(np.tanh(np.arctanh(np.clip(corr_cols(A, B), -0.999, 0.999)).mean()))\n        res[name] = {"r": round(r, 4), "one_minus_r2": round(1 - r * r, 4), "n_parcels": len(vs)}\n    return res\n'
COMMON_SHA = '8df0380c20b62a28263800e7ddcba0df805cbc4b919dbc4e702cccdfb1afd7ba'
CHECKS = '"""r2/run_r2_forecast.py\'s check functions, byte for byte."""\nimport csv\nimport re\nfrom collections import Counter\n\nSTORIES = ("story1", "story2", "story3")\n\n\ndef words_table(path) -> list:\n    """(text, start, duration) rows of a word table in ExtractWordsFromAudio\'s format (tab separated, pandas quoting)."""\n    with open(path, newline="") as f:\n        return [(r["text"], float(r["start"]), float(r["duration"])) for r in csv.DictReader(f, delimiter="\\t")]\n\n\ndef shift_copies(texts, starts) -> dict:\n    """The film kernels\' check for the over-60 s duplication: the most common exact shift (ms) between repeats of the same word; a block\n    is flagged when that shift recurs at least max(5, 10% of the words) times."""\n    by = {}\n    for t, s in zip(texts, starts):\n        by.setdefault(t, []).append(round(float(s), 3))\n    shifts = Counter(round(b - a, 3) for ts in by.values() for i, a in enumerate(ts) for b in ts[i + 1:] if b > a)\n    worst = max(shifts.items(), key=lambda kv: kv[1], default=(None, 0))\n    return {"exact_shift_copies": int(worst[1]) if worst[1] >= max(5, 0.1 * len(texts)) else 0, "worst_shift_s": worst[0],\n            "worst_shift_count": int(worst[1])}\n\n\ndef fulltext_counts(path) -> dict:\n    """Word counts of the dataset\'s full text, per part: the citation header between the first two dash rules is dropped, parts are\n    split at the ---...--- marker lines, and a word is a whitespace token with a letter or digit. Counts only, never text."""\n    lines = open(path, encoding="utf-8").read().splitlines()\n    rules = [i for i, l in enumerate(lines) if re.fullmatch(r"-{10,}", l.strip())]\n    body = lines[rules[1] + 1:] if len(rules) >= 2 else lines\n    parts, cur = [], []\n    for l in body:\n        if re.fullmatch(r"-{3}[^-].*[^-]-{3}", l.strip()):\n            parts.append(cur); cur = []\n        else:\n            cur.append(l)\n    parts.append(cur)\n    n = [sum(1 for w in " ".join(p).split() if re.search(r"\\w", w)) for p in parts]\n    return dict(zip(STORIES, n)) if len(n) == len(STORIES) else {"total": sum(n)}\n\n\ndef check_shuffled(tsv, table) -> dict:\n    """The shuffled table keeps every onset and duration and permutes the texts (section 8\'s control)."""\n    a, b = words_table(tsv), words_table(table)\n    same_times = [(x[1], x[2]) for x in a] == [(x[1], x[2]) for x in b]\n    perm = sorted(x[0] for x in a) == sorted(x[0] for x in b)\n    assert same_times and perm, "the shuffled table does not keep the word table\'s timings and texts"\n    return {"same_onsets_and_durations": same_times, "texts_permuted": perm, "texts_moved": sum(x[0] != y[0] for x, y in zip(a, b))}\n'
STAGE = '"""One R2 stock stage inside the kernel\'s tribe environment (r2_forecast_tpl.py runs it as a separate process per stage, so each stage\'s\nmemory is freed): r2/run_r2_forecast.py\'s main for a stock stage, with every model on the T4.\nusage: python r2_stage.py STAGE ROOT OUT CACHE CODE"""\nimport hashlib, json, sys, time\nfrom pathlib import Path\nstage, root, out, cache, code = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5]\nsys.path.insert(0, code)\nimport numpy as np, torch  # noqa: E401,E402\nimport tribe_common as C  # noqa: E402\nfrom r2_checks import check_shuffled, fulltext_counts, shift_copies, words_table  # noqa: E402\nFEATS = {"audio": ["audio"], "full": ["text", "audio", "video"], "shuf": ["text", "audio", "video"]}       # the runner\'s\nSHUFFLE_SEED, PIN = 7, "9c79ffff6b642b7b0c71d558c935fb3fa33f2788bfb509feead94fafbba2f321"\n\n\ndef sha(p):\n    h = hashlib.sha256()\n    with open(p, "rb") as f:\n        for b in iter(lambda: f.read(1 << 24), b""):\n            h.update(b)\n    return h.hexdigest()\n\n\nmodel, story, mode = stage.split("_")\nassert model == "stock" and mode in FEATS, stage\nwav, tsv, table = root / story / "stim.wav", root / story / "stim.tsv", out / f"words_{story}.tsv"\nt0 = time.time()\nif torch.cuda.is_available():\n    torch.cuda.reset_peak_memory_stats()\nres = {"stage": stage, "model": model, "story": story, "mode": mode, "wav_sha256": sha(wav)}\nif mode == "shuf":\n    n = C.write_tsv(words_table(table), wav, shuffle_seed=SHUFFLE_SEED)\n    res["prepare"] = {"shuffle_seed": SHUFFLE_SEED, "n_tsv_words": n, "word_table_sha256": sha(table)}\nelif mode == "full":\n    assert sha(tsv) == sha(table), "stim.tsv is not the story\'s word table"\n    res["prepare"] = {"speech_to_text": "read from stim.tsv: the story\'s word table from the kernel stillhere-r2-asr (TRIBE v2\'s own "\n                                        "speech-to-text, tribev2 af58661\'s function unchanged, on a Kaggle T4)"}\nelse:\n    res["prepare"] = {}\nbefore = sha(tsv) if mode != "audio" else None\nev = C.build_events(wav, "english", with_text=(mode != "audio"))\nif mode != "audio":\n    assert sha(tsv) == before, "the word table changed while the events were built"\n    rows = words_table(tsv)\n    w = C.event_checks(ev, len(rows), sum(1 for r in rows if r[1] > 60))\n    wd = ev[ev.type == "Word"]\n    w.update(shift_copies(list(wd.text), list(wd.start)))\n    w["words_ok"] = bool(len(wd) and w["last_word_before_audio_end"] and w["exact_shift_copies"] == 0)\n    assert w["words_ok"], f"word checks failed: {({k: w[k] for k in (\'n_words\', \'last_word_before_audio_end\', \'exact_shift_copies\')})}"\n    if mode == "full":\n        w["fulltext_words"] = fulltext_counts(root / "story_fulltext.txt").get(story)\n    else:\n        w["shuffled"] = check_shuffled(tsv, table)\n    w["word_table_sha256"] = sha(table)\n    res["words"] = w; end = w["audio_end_s"]\nelse:\n    end = res["audio_end_s"] = float(ev[ev.type == "Audio"].stop.max())\nm = C.load_model(FEATS[mode], cache, text_device="accelerate")\nres["weights"] = C.apply_weights(m, "stock")\nassert res["weights"]["stock_ckpt_sha256"] == PIN, "stock checkpoint differs from the pin"\nstarts, P, nwin = C.stitched_predict(m, ev, end)\nres.update({"shape": list(P.shape), "n_windows": nwin, "finite": bool(np.isfinite(P).all()),\n            "contiguous_seconds": bool(len(starts) and (np.diff(starts) == 1).all() and starts[0] == 0), **C.gpu_stats(),\n            "seconds": round(time.time() - t0),\n            "placement": {"route": "Kaggle T4 (rule amendment of Nov 1)", "audio_extractor": "cuda", "brain_model": "cuda",\n                          "text_features": "accelerate"}})\nassert res["finite"] and res["contiguous_seconds"], "forecast not finite or not one row per second"\nf = out / f"pred_{stage}.npz"\nnp.savez_compressed(f, P=P.astype("float32"), starts=starts)\nres["pred_sha256"] = sha(f)\njson.dump(res, open(out / f"stage_{stage}.json", "w"), indent=1)\nprint("RESULT " + json.dumps({k: v for k, v in res.items() if k != "weights"}), flush=True)\n'
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
