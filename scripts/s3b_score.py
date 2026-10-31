"""S3b's one scoring run (docs/rules/s3b_late_rule.md, section 4), from the analysis tree's d22/picker/ (the picker code is kept there).
Uses the picker's own functions unchanged: score_picker.load_clip and series, gauges.row_predictors (PULA on the full forecast, and on
the forecast passed as kind "timing", which here is S3b's exact control), bars.families and bars.rank_test. T1, the counted rows and the
families are the single scoring call's (scored/, checked against results/picker/MANIFEST.sha256); P_full and P_audio are the scored
forecasts (merged PRED). Refuses unless: SCORED.lock names the manifest; every counted row's clip has a control forecast from the part
that owns it, made after that part's gate passed, in its verified PRED folder, from the stock checkpoint, with the exact control's provenance, and its SHA-256 equals
CONTROL_PINS (a commit after the forecasts, before this run); and PULA recomputed here equals the scored PULA for every row (the same code
on the same inputs). Writes OUT/s3b_late.json (per row) and results/picker/s3b_late.json (the test and its line).
usage (from d22/picker): python s3b_score.py --merged ../picker_pred/merged/PRED --s3b DIR [DIR ...] --out OUT"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.getcwd(); sys.path.insert(0, HERE)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import bars as B  # noqa: E402
import gauges as G  # noqa: E402
import score_picker as SP  # noqa: E402

RULE = "docs/rules/s3b_late_rule.md"
LABEL = "pre-registered, run late (Oct 31-Nov 2, after S1-S3 were seen)"
CONTROL_PINS = {}        # stem -> SHA-256 of pred_timing_exact.npz; filled by the commit after the forecasts, before this run
PARTS = os.path.join(HERE, "../picker_pred/parts.json")
LINES = {
    True: "S3b ({label}): TRIBE's predicted response strength still ranked the stories with listeners' language readouts after a forecast "
          "that keeps each word's timing but not its content was partialled out (partial rho {rho:.2f}, one-sided p {p:.4f}, {families} story "
          "families in {blocks} blocks). It's the content, not just the timing.",
    False: "S3b ({label}): once a forecast that keeps each word's timing but not its content was partialled out, TRIBE's predicted response "
           "strength no longer ranked the stories reliably (partial rho {rho:.2f}, one-sided p {p:.4f}, {families} story families in "
           "{blocks} blocks). S1-S3 stand as reported; the content-over-timing line is not used."}


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def scored_inputs():
    """The single scoring call's files, each checked against the committed MANIFEST.sha256."""
    want = dict(reversed(line.split()) for line in open(f"{REPO}/results/picker/MANIFEST.sha256") if line.strip() and not line.startswith("#"))
    for n in ("SCORED.lock", "tierD_picker_targets.json", "tierD_picker_predictors.json"):
        if sha(f"{HERE}/scored/{n}") != want[n]:
            raise SystemExit(f"refused: scored/{n} differs from results/picker/MANIFEST.sha256")
    lock = open(f"{HERE}/scored/SCORED.lock").read().split()[0]
    if sha(f"{HERE}/manifest/picker_manifest.json") != lock:
        raise SystemExit("refused: the manifest is not the one SCORED.lock names")
    return (json.load(open(f"{HERE}/manifest/picker_manifest.json")), json.load(open(f"{HERE}/scored/tierD_picker_targets.json"))["per_clip"],
            json.load(open(f"{HERE}/scored/tierD_picker_predictors.json"))["predictors"])


def controls(dirs, stems) -> dict:
    """stem -> the owning part's control forecast, after the checks in the docstring."""
    owner = {s: p["part"] for p in json.load(open(PARTS))["parts"] for s in p["owns"]}
    found = {}
    for d in dirs:
        summ = json.load(open(next(os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs if f == "predict_summary.json")))
        part = json.load(open(os.path.join(d, "colab_run.json")))["s3b_part"]
        if not (summ.get("gate") or {}).get("pass"):
            continue
        for stem in stems:
            if owner.get(stem) != part:
                continue
            hits = [os.path.join(r, "pred_timing_exact.npz") for r, _, fs in os.walk(d) if os.path.basename(r) == stem and "pred_timing_exact.npz" in fs]
            hits = [h for h in hits if os.path.basename(os.path.dirname(os.path.dirname(h))) == "PRED"]   # the kernel's verified folder
            if len(hits) == 1:
                prov = json.load(open(os.path.join(os.path.dirname(hits[0]), "provenance.json")))
                if prov.get("ckpt_sha256_by_mode", {}).get("timing_exact") == SP.C.CFG["stock_ckpt_sha256"] and prov["timing_mode"].startswith("exact"):
                    found[stem] = hits[0]
    missing = sorted(set(stems) - set(found))
    if missing:
        raise SystemExit(f"refused: no checked control forecast for {len(missing)} counted clip(s): {missing}")
    if not CONTROL_PINS:
        raise SystemExit("refused: the control forecasts' SHA-256 are not pinned yet (a commit after the forecasts, before scoring)")
    bad = [s for s, p in found.items() if sha(p) != CONTROL_PINS.get(s)]
    if bad:
        raise SystemExit(f"refused: control forecast(s) differ from the pins: {bad}")
    return found


def overlay(merged, found, ovl) -> str:
    """ovl/<stem>/: the scored pred_full, pred_audio and provenance, and the control as pred_timing (the kind gauges reads)."""
    for stem, ctl in found.items():
        d = os.path.join(ovl, stem); os.makedirs(d, exist_ok=True)
        for n, src in (("pred_full.npz", f"{merged}/{stem}/pred_full.npz"), ("pred_audio.npz", f"{merged}/{stem}/pred_audio.npz"),
                       ("provenance.json", f"{merged}/{stem}/provenance.json"), ("pred_timing.npz", ctl)):
            if not os.path.islink(os.path.join(d, n)):
                os.symlink(os.path.abspath(src), os.path.join(d, n))
    return ovl


def score(merged, found, ovl_dir):
    """S3b on the given control forecasts {stem: path}: (the public record, per-row values)."""
    M, per_clip, scored_pred = scored_inputs()
    cnt = [r for r in M["rows"] if r["counted"]]; R = {r["rid"]: r for r in cnt}
    ovl = overlay(merged, found, ovl_dir)
    import numpy as np
    z = np.load(f"{HERE}/sets.npz"); sets = {k: [z[f"{k}_{i}"] for i in range(int(z[f"{k}_n"]))] for k in ("AUD", "LANG")}
    clips = {c: SP.load_clip(SP.C.stem_of(c), "collected", ovl, sets) for c in sorted({r["clip"] for r in cnt})}
    out, vals = {}, {}
    for q, r in R.items():
        S = [SP.series("collected", x["file"]) for x in r["listeners"]]
        P = G.row_predictors(clips[r["clip"]], S[0][2], [s[1][r["clip"]] for s in S])
        if abs(P["PULA"] - scored_pred[q]["PULA"]) > 1e-12:
            raise SystemExit(f"refused: PULA recomputed for {q} differs from the scored value")
        out[q] = {"PULA": P["PULA"], "PULA_timing": P["PULA_timing_approx"], "T1": per_clip[q]["T1"]}
        vals[q] = dict(per_clip[q], PULA_timing=P["PULA_timing_approx"])
    s3b = B.rank_test(B.families(cnt, vals), "PULA", "T1", ("PULA_timing",))
    pub = {"label": LABEL, "rule": RULE, "manifest_sha256": sha(f"{HERE}/manifest/picker_manifest.json"),
           "control_sha256": {s: sha(p) for s, p in found.items()}, "S3b": s3b, "line": LINES[s3b["pass_"]].format(label=LABEL, **s3b)}
    return pub, out


def main(a):
    M, _, _ = scored_inputs()
    found = controls(a.s3b, sorted({SP.C.stem_of(r["clip"]) for r in M["rows"] if r["counted"]}))
    pub, rows = score(a.merged, found, os.path.join(a.out, "overlay"))
    os.makedirs(a.out, exist_ok=True)
    json.dump({**pub, "rows": rows}, open(f"{a.out}/s3b_late.json", "w"), indent=1, default=float)
    json.dump(pub, open(f"{REPO}/results/picker/s3b_late.json", "w"), indent=1, default=float)
    print(pub["line"], flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True); ap.add_argument("--s3b", nargs="+", required=True); ap.add_argument("--out", required=True)
    main(ap.parse_args())
