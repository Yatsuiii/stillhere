"""The laptop check of S3b's hook on the real TRIBE v2 text lane (docs/rules/s3b_late_rule.md), before any S3b forecast. Public audio
only: the first 60 s of LPP-FR section 1 (PREREG's development section), no brain data. Stock TRIBE v2 with the Alice placement (brain
model and sound extractor on the CPU, Llama in float16 across the GPU and CPU), tribe_common's build_events and stitched_predict.
Checks, printed as one JSON line:
  hook off: two plain forecasts agree (run-to-run), and a plain forecast after both hooks have run equals the first;
  recording: one vector per word event of the clip;
  word_mean: the hook set words (count > 0); every text frame of every window that bears a word is parallel to the clip mean
  (cosine >= 0.9999); the forecast differs from the plain one.
usage (tribe venv): python s3b_hook_check.py ANALYSIS_ROOT WORK_DIR"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT, WORK = Path(sys.argv[1]), Path(sys.argv[2])
sys.path[:0] = [str(ROOT / "k2"), str(ROOT / "alice/forecast"), str(Path(__file__).resolve().parent)]
import tribe_common as C  # noqa: E402
import run_alice_forecast as R  # noqa: E402
import s3b_control as S  # noqa: E402

R.CACHE = WORK / "cache"; R.CACHE.mkdir(parents=True, exist_ok=True)
d, n_tsv = C.prepare_dir(WORK / "clip", ROOT / "p1/data/task-lppFR_section_1.wav", C.textgrid_words(ROOT / "p1/data/lppFR_section1.TextGrid"),
                         seconds=60)
ev = C.build_events(d, "french", with_text=True); end = float(ev[ev.type == "Audio"].stop.max())
R.pin_llama()
m = R.load_model_cpu_brain(["text", "audio", "video"]); ext = m.data.text_feature


def predict():
    return C.stitched_predict(m, ev, end)[1].astype("float64")


def text_windows():
    return [b.data["text"].float().cpu().numpy() for b in m.data.get_loaders(events=ev, split_to_build="all")["all"]]


P0 = predict(); P0b = predict()
with S.recording(ext) as seen:
    text_windows()
v = S.clip_mean(seen)
with S.word_mean(ext, v) as cnt:
    T = text_windows(); P2 = predict()
P3 = predict()
flat = v.reshape(-1).astype("float64"); cos = []
for t in T:                                                            # (B, 2, D, frames)
    for b in range(t.shape[0]):
        for k in range(t.shape[-1]):
            f = t[b, ..., k].reshape(-1).astype("float64")
            if np.any(f != 0):
                cos.append(float(f @ flat / (np.linalg.norm(f) * np.linalg.norm(flat))))
n_words = int((ev.type == "Word").sum())
out = {"clip": "lppFR_section1, first 60 s", "word_events": n_words, "tsv_words": n_tsv, "words_recorded": len(seen),
       "hook_word_calls": cnt["words"], "mean_vector_shape": list(v.shape), "word_frames_checked": len(cos),
       "min_cosine_to_mean": min(cos) if cos else None,
       "hook_off_run_to_run_max_abs": float(np.abs(P0 - P0b).max()), "hook_off_after_hooks_max_abs": float(np.abs(P3 - P0).max()),
       "control_vs_plain_corr": float(np.corrcoef(P2.ravel(), P0.ravel())[0, 1]), "control_vs_plain_max_abs": float(np.abs(P2 - P0).max()),
       "class_restored": "_get_timed_arrays" not in type(ext).__dict__ or type(ext).__dict__["_get_timed_arrays"].__name__ != "patched"}
out["pass"] = bool(out["words_recorded"] == n_words and out["hook_word_calls"] > 0 and cos and out["min_cosine_to_mean"] >= 0.9999
                   and out["hook_off_after_hooks_max_abs"] <= out["hook_off_run_to_run_max_abs"] and out["control_vs_plain_max_abs"] > 0
                   and out["class_restored"])
print(json.dumps(out), flush=True)
