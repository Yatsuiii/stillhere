"""Stock TRIBE v2's audio-only and full forecasts of any English story recording, for scripts/rank_minutes.sh: run_r2_forecast.py's stock
stages with the Paranoia pins removed. The harness is R2's, imported unchanged (tribe_common from the R2 kernel at tribev2 af58661,
build_events, stitched 50 s windows, the sound extractor and TRIBE's brain model on the CPU, Llama 3.2's layers on a fixed 3 GiB GPU
budget, the stock checkpoint checked against its pin), and so are its word checks.

Words: TRIBE v2's own speech-to-text (ExtractWordsFromAudio as tribev2 af58661 ships it: uvx whisperx large-v3, float16, batch 16), run
once by the full stage, which writes the story's word table. Two recorded departures, both off unless set:
  STILLHERE_WORDS_TSV=path   a word table in that transform's format (tab separated: text, start, duration, sequence_id, sentence),
                             read instead of running speech-to-text, as run_r2_forecast.prepare reads one already there;
  STILLHERE_ASR_BATCH=n      whisperx's batch size in place of 16 (a 6 GB GPU runs out of memory at 16 on a full story).
Writes OUT/pred_stock_<part>_{audio,full}.npz, OUT/words_<part>.tsv and OUT/stage_<part>_<mode>.json; prints counts, shapes and timings,
never word text. Run from DATA/r2/ (scripts/paranoia_files.py layout), beside alice/ and k2/.
usage: python run_story_forecast.py WAV PART MODE OUT   (MODE: audio, then full)"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_r2_forecast as F  # noqa: E402  (its helpers and FEATS; none of its Paranoia stages run)


def asr_batch(n: str) -> dict:
    """Rewrite whisperx's --batch_size in TRIBE's own command (the transform imports subprocess inside the call, so this module-level
    wrapper is the one it uses); everything else in the command is unchanged."""
    run = subprocess.run

    def wrapped(cmd, *a, **k):
        if isinstance(cmd, list) and cmd[:2] == ["uvx", "whisperx"] and "--batch_size" in cmd:
            cmd = list(cmd); cmd[cmd.index("--batch_size") + 1] = n
        return run(cmd, *a, **k)
    subprocess.run = wrapped
    return {"whisperx_batch_size": int(n), "shipped_batch_size": 16}


def main(wav, part, mode, out):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", part) or mode not in ("audio", "full"):
        raise SystemExit("usage: run_story_forecast.py WAV PART MODE OUT (PART: letters, digits, _ or -; MODE: audio or full)")
    sys.path.insert(0, str(F.ROOT / "alice/forecast"))
    import numpy as np
    import torch
    import run_alice_forecast as R
    import run_alice_forecast_tuned as RT
    C = R.C
    out = Path(out); work = out / "work" / f"{part}_{mode}"; R.CACHE = out / "cache"
    for d in (out, work, R.CACHE):
        d.mkdir(parents=True, exist_ok=True)
    if mode == "full" and not (out / f"stage_{part}_audio.json").exists():
        raise SystemExit("run the audio stage first")
    t0 = time.time(); torch.cuda.reset_peak_memory_stats()
    shutil.copyfile(wav, work / "stim.wav")
    res = {"stage": f"stock_{part}_{mode}", "part": part, "mode": mode, "wav_sha256": F.sha(wav)}
    feats = F.FEATS[mode]
    if mode == "full":
        given = os.environ.get("STILLHERE_WORDS_TSV")
        if given:
            shutil.copyfile(given, work / "stim.tsv")
            res["words_source"] = {"table": os.path.basename(given), "table_sha256": F.sha(given)}
        else:
            res["words_source"] = {"speech_to_text": "read from an earlier attempt's stim.tsv" if (work / "stim.tsv").exists()
                                   else "ran (ExtractWordsFromAudio)", "whisperx": F.whisperx_version()}
            if os.environ.get("STILLHERE_ASR_BATCH"):
                res["words_source"].update(asr_batch(os.environ["STILLHERE_ASR_BATCH"]))
    ev = C.build_events(work / "stim.wav", "english", with_text=("text" in feats))
    if "text" in feats:
        res["words"] = F.word_checks(ev, work / "stim.tsv", C)
        shutil.copyfile(work / "stim.tsv", out / f"words_{part}.tsv")
        res["words"]["word_table_sha256"] = F.sha(out / f"words_{part}.tsv")
        end = res["words"]["audio_end_s"]
    else:
        end = res["audio_end_s"] = float(ev[ev.type == "Audio"].stop.max())
    R.PLACEMENT.clear(); R.pin_llama()
    m = RT._stock_loader(feats)
    ck = glob.glob(os.environ.get("HF_HOME", "/tmp/hf") + "/hub/models--facebook--tribev2/snapshots/*/best.ckpt")[0]
    res["weights"] = {"stock_ckpt_sha256": F.sha(ck)}
    assert res["weights"]["stock_ckpt_sha256"] == RT.STOCK_PIN, "stock checkpoint differs from the pin"
    starts, P, nwin = C.stitched_predict(m, ev, end)
    res.update({"shape": list(P.shape), "n_windows": nwin, "finite": bool(np.isfinite(P).all()),
                "contiguous_seconds": bool((np.diff(starts) == 1).all() and starts[0] == 0), **C.gpu_stats(),
                "seconds": round(time.time() - t0), "placement": {"audio_extractor": "cpu", "brain_model": "cpu", **R.PLACEMENT}})
    if "text" in feats and not R.PLACEMENT:
        res["placement"]["text_features"] = "read from this runner's feature cache"
    assert res["finite"] and res["contiguous_seconds"], "forecast not finite or not one row per second"
    f = out / f"pred_stock_{part}_{mode}.npz"
    np.savez_compressed(f, P=P.astype("float32"), starts=starts)
    res["pred_sha256"] = F.sha(f)
    json.dump(res, open(out / f"stage_{part}_{mode}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "weights"}), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit(__doc__.split("usage: ")[1])
    main(*sys.argv[1:])
