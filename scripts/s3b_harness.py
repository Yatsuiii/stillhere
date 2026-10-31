"""S3b's prediction step (docs/rules/s3b_late_rule.md; amendment to D26 (c), cloud ruling of Oct 31, 21:00 UTC): harness_pred.py with
its approximate timing mode replaced by timing_exact, section 9's PULA_timing: each word's Llama vector replaced by the clip-mean vector,
word timing unchanged (the hook in s3b_control.py, on HuggingFaceText._get_timed_arrays: a recording pass over the clip's windows keeps one
vector per word, then stitched_predict runs with every word set to their mean). Modes full and audio run first with the hook off, exactly
as harness_pred; then the gate: the fresh full forecasts of LPP-FR section 8 and pieman against the scored ones (S3B_REFS/<stem>/
pred_full.npz) by pred_stage.compare on D26's bar. timing_exact runs only if both pass. Same load_model, apply_weights (stock pin
asserted), build_events and stitched_predict; same output layout, with pred_timing_exact.npz in place of pred_timing.npz. torch is
imported only inside run().

usage (inside the kernel's venv): python harness_pred.py JOB_JSON LANGS_JSON OUT_DIR   (env S3B_REFS: the scored references)"""
import json, os, sys, time
import numpy as np


def run(job_json, langs_json, out_dir):
    import torch
    sys.path[:0] = ["/tmp/k2code", "/tmp"]; import tribe_common as C, pred_stage as S   # the embedded harness and pred_stage, byte for byte
    jobs = json.load(open(job_json)); langs = json.load(open(langs_json)); os.makedirs(out_dir, exist_ok=True)
    import s3b_control as SC                                            # /tmp/s3b_control.py, written by the notebook from the commit
    res = {"clips": {}, "modes": {}, "method": "harness", "s3b_control_sha256": sha256_file("/tmp/s3b_control.py")}
    MODES = {"full": S.MODES["full"], "audio": S.MODES["audio"], "timing_exact": S.MODES["full"]}
    for mode, feats in MODES.items():
        if mode == "timing_exact":
            res["gate"] = gate(out_dir, S)
            print(f"S3B GATE {json.dumps(res['gate'])}", flush=True)
            if not res["gate"]["pass"]:
                break
        t0m = time.time(); m = C.load_model(feats, "/tmp/cache", text_device="accelerate"); info = C.apply_weights(m, "stock")
        assert info["stock_ckpt_sha256"] == S.STOCK_PIN
        res["modes"][mode] = {"ckpt_sha256": info["stock_ckpt_sha256"], "training_mode": info["training_mode"], "features_to_use": feats}
        for j in jobs:
            stem = j["stem"]; r = res["clips"].setdefault(stem, {})
            try:
                ev = C.build_events(f"/tmp/conf_{stem}/stim.wav", langs[stem], with_text=("text" in feats))
                end = float(ev[ev.type == "Audio"].stop.max()); r["harness_audio_end_s"] = end
                if mode == "timing_exact":
                    ext = m.data.text_feature
                    with SC.recording(ext) as seen:                         # every window once, as stitched_predict reads them
                        for _ in m.data.get_loaders(events=ev, split_to_build="all")["all"]:
                            pass
                    wd = ev[ev.type == "Word"]; n_ev = int((wd.start < end).sum()); v = SC.clip_mean(seen)   # words inside the audio
                    with SC.word_mean(ext, v) as cnt:
                        st, P, nw = C.stitched_predict(m, ev, end)
                    r.update(timing_exact_word_events=n_ev, timing_exact_words_recorded=len(seen), timing_exact_hook_words=cnt["words"])
                    if len(seen) != n_ev or cnt["words"] == 0:
                        raise RuntimeError("control incomplete: words recorded or set differ from the clip's word events")
                else:
                    st, P, nw = C.stitched_predict(m, ev, end)
                d = f"{out_dir}/{stem}"; os.makedirs(d, exist_ok=True)
                np.savez(f"{d}/pred_{mode}.npz", P=P.astype("float32"), starts=st)
                r.update({f"{mode}_shape": list(P.shape), f"{mode}_finite": bool(np.isfinite(P).all()), "n_windows": nw,
                          "contiguous_seconds": bool((np.diff(st) == 1).all() and st[0] == 0), f"{mode}_ckpt_sha256": info["stock_ckpt_sha256"]})
                print(f"ckpt {stem} {mode} {info['stock_ckpt_sha256']}", flush=True)
            except Exception as e:
                r[f"{mode}_error"] = type(e).__name__
        res["modes"][mode]["seconds"] = round(time.time() - t0m); del m; torch.cuda.empty_cache()
    for j in jobs:                                                      # provenance once every mode is done: the checkpoint of each prediction
        stem = j["stem"]; r = res["clips"].get(stem, {})
        if not all(f"{k}_shape" in r for k in ("full", "audio")):
            continue
        by_mode = {k: r[f"{k}_ckpt_sha256"] for k in ("full", "audio", "timing_exact") if f"{k}_ckpt_sha256" in r}
        json.dump({"stem": stem, "ckpt_sha256": S.STOCK_PIN if all(v == S.STOCK_PIN for v in by_mode.values()) else None, "ckpt_sha256_by_mode": by_mode,
                   "model": "stock TRIBE v2 (facebook/tribev2 best.ckpt, tribev2 af58661)",
                   "method": "full harness: tribe_common load_model (overlap_trs_train 50) and stitched_predict on the stock loader, extractor "
                             "features from the cache feat_stage filled (cloud ruling of Oct 31, 04:30 UTC)",
                   "features_route": j.get("route"), "features_sha256": j["feats_sha256"], "features_source": j["feats_source"],
                   "features_file_read": False, "source_check": j.get("source_check"), "audio_end_s": r["harness_audio_end_s"],
                   "timing_mode": "exact (S3b run late, docs/rules/s3b_late_rule.md): each word's Llama vector replaced by the clip-mean "
                                  "vector, timing unchanged (s3b_control.py on HuggingFaceText._get_timed_arrays)" if "timing_exact_shape" in r else
                                  "none: the S3b gate did not pass, so no timing_exact forecast was made",
                   "s3b_control_sha256": res["s3b_control_sha256"], "s3b_gate": res.get("gate"),
                   "subject": "average subject (subject_id 0, as in the D22 stock forward)",
                   "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, open(f"{out_dir}/{stem}/provenance.json", "w"), indent=1)
    json.dump(res, open(f"{out_dir}/../picker_pred_out/predict_summary.json", "w"), indent=1)
    ok = sum(all(f"{k}_shape" in r for k in MODES) for r in res["clips"].values())
    print(f"predict: {ok} of {len(jobs)} clips have full, audio and timing_exact (full harness, S3b); gate "
          f"{'PASS' if (res.get('gate') or {}).get('pass') else 'FAIL or not reached'}; modes " +
          ", ".join(f"{k} {v.get('seconds')} s (training_mode {v['training_mode']})" for k, v in res["modes"].items()), flush=True)


def sha256_file(p):
    import hashlib
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def gate(out_dir, S):
    """D26's bar on the hook-off full forecasts of the two self-check clips against the scored ones; both must pass."""
    refs = os.environ.get("S3B_REFS"); out = {"clips": {}}
    for stem in ("lppFR_section8", "pieman"):
        a, b = f"{out_dir}/{stem}/pred_full.npz", f"{refs}/{stem}/pred_full.npz" if refs else None
        if b is None or not (os.path.isfile(a) and os.path.isfile(b)):
            out["clips"][stem] = {"pass": False, "error": "missing fresh or scored forecast"}
        else:
            out["clips"][stem] = {**S.compare(np.load(a), np.load(b)), "scored_sha256": sha256_file(b), "fresh_sha256": sha256_file(a)}
    out["pass"] = all(c.get("pass") for c in out["clips"].values())
    return out


if __name__ == "__main__":
    run(*sys.argv[1:4])
