"""S3b's one scoring run (docs/rules/s3b_late_rule.md, section 4), with the changes the cloud's review of Oct 31, 23:15 UTC asked for, all
made before any S3b forecast was read (rule section 7). Run from the analysis tree's d22/picker/ (the picker code is kept there); it uses
the picker's own functions unchanged: score_picker.load_clip and series, gauges.row_predictors (PULA on the full forecast, and on the
forecast passed as kind "timing", which here is S3b's exact control), bars.families and bars.rank_test. T1, the counted rows and the
families are the single scoring call's (scored/, checked against results/picker/MANIFEST.sha256); P_full and P_audio are the scored
forecasts (merged PRED).

Refuses, with a named reason, unless:
- parts.json has its pinned SHA-256, its owns lists are disjoint and cover every counted stem, and each part has exactly one folder;
- every file read from a part's folder is in CONTROL_PINS (a commit after the forecasts, before this run) with that SHA-256, and agrees
  with the notebook's own MANIFEST.sha256;
- each part's run record names section 6's notebook and kernel hashes, the hook's hash and the gate references, with kernel exit code 0
  on a T4; and its gate record passed, on the references pinned in section 6;
- every counted stem has a control forecast in its owning part's verified PRED folder, from the stock checkpoint, with the exact control's
  provenance, the right shape, finite values and the scored forecast's second grid;
- PULA recomputed here equals the scored PULA for every row (the same code on the same inputs).
--out must be a new, empty directory. --check-only stops after the checks, before any listener data or statistic. A refusal writes the
rule's not-run line with its reason (to OUT, and in a scoring run also to results/picker/s3b_late.json). Beside S3b, descriptive and
never gating: each counted clip's fresh full forecast against the scored one by D26's comparison (pred_stage.compare).

usage (from d22/picker):
  python s3b_score.py --merged ../picker_pred/merged/PRED --s3b DIR1 DIR2 DIR3 --print-pins             (the pins, for their commit)
  python s3b_score.py --merged ../picker_pred/merged/PRED --s3b DIR1 DIR2 DIR3 --out NEW --check-only    (first)
  python s3b_score.py --merged ../picker_pred/merged/PRED --s3b DIR1 DIR2 DIR3 --out NEW                 (the one scoring run)"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.getcwd()
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RULE = "docs/rules/s3b_late_rule.md"
LABEL = "pre-registered, run late (Oct 31-Nov 2, after S1-S3 were seen)"
WHY_LATE = ("The scored parts were fresh extractions, so per-word vectors existed when the scored forecasts were made; S3b was left out "
            "because no control was built, not because the data were missing.")
BESIDE = ("decided after the scored record was read, including S4, the wall and the approximate frame-level version (descriptive, partial "
          "rho 0.56, p 0.0012)")
HOOK = "6bf3e6714f829068cf76b02f42334986dce5e3cbadb12d417d610990618aeac4"
STOCK_PIN = "9c79ffff6b642b7b0c71d558c935fb3fa33f2788bfb509feead94fafbba2f321"
REFS = {"lppFR_section8": "53c77a37a42b32e2ab64346979492e59aba550351683524f7d099add8bc88163",
        "pieman": "196be7a95b71f9331fa98dc6798a16f6b5e24d4e1799e5b3f9afc9db61d1fed5"}
KERNELS = {"1": "d2bcf8303f2cb7d376190e659ffd67dacd66a2bdce1c70851213d8444dd871e0",
           "2": "be45a2b03e25150d14cbf10e9e16b4d582f5aeb92524fc29e305fb99b31d7a21",
           "3": "61987fb388668b0574d52e0ad0fea09f29abafffd280aca1674a259bf589a368"}
NOTEBOOKS = {"1": "7bb8b4dcfed20a3bb2fc8716a95c8e6c8b829779a8e833d35276a6ebc12f2f8b",
             "2": "b75ad8a22d6f7d692143042037c8fa2a48a1c7dca6f1ed885d9f790eaf142517",
             "3": "0b027cf4ffbfa36e148db4c572a1695ef5f127e01a2de11ae667971b8248794d"}
PARTS = os.path.join(HERE, "../picker_pred/parts.json")
PARTS_SHA = "ab9086dfcf47d51aaaf15c44ce3e320ae1c1b6aba09dfe5bee6a14a12f5bf07d"
SUMMARY = "picker_pred_out/predict_summary.json"     # the part's gate record
CONTROL_PINS = {}        # "part<N>/<path in the part's folder>" -> SHA-256; filled by the commit after the forecasts, before this run
PRED_STAGE = os.path.join(HERE, "../picker_pred")    # pred_stage.compare, D26's comparison
LINES = {   # section 5, word for word
    True: "S3b (pre-registered, run late: Oct 31-Nov 2, after S1-S3 were seen): TRIBE's predicted response strength still ranked the "
          "stories with listeners' language readouts after a forecast that keeps each word's timing but not its content was partialled out "
          "(partial rho {rho:.2f}, one-sided p {p:.4f}, {families} story families in {blocks} blocks). It's the content, not just the timing.",
    False: "S3b (pre-registered, run late: Oct 31-Nov 2, after S1-S3 were seen): once a forecast that keeps each word's timing but not its "
           "content was partialled out, TRIBE's predicted response strength no longer ranked the stories reliably (partial rho {rho:.2f}, "
           "one-sided p {p:.4f}, {families} story families in {blocks} blocks). S1-S3 stand as reported; the content-over-timing line is not "
           "used."}
NOT_RUN = "S3b: pre-registered, not run ({reason})."


class Refusal(Exception):
    """A named reason S3b is not run."""


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def load_json(p, what):
    try:
        return json.load(open(p))
    except (OSError, ValueError) as e:
        raise Refusal(f"{what} unreadable ({type(e).__name__})")


def picker():
    """The picker's modules, from d22/picker (imported only for scoring)."""
    sys.path.insert(0, HERE)
    import bars as B, gauges as G, score_picker as SP
    return B, G, SP


def compare(a, b):
    sys.path.insert(0, PRED_STAGE)
    import pred_stage
    return pred_stage.compare(a, b)


def scored_inputs():
    """The single scoring call's files, each checked against the committed MANIFEST.sha256."""
    want = {}
    for line in open(f"{REPO}/results/picker/MANIFEST.sha256"):
        if line.strip() and not line.startswith("#"):
            h, n = line.split(); want[n] = h
    for n in ("SCORED.lock", "tierD_picker_targets.json", "tierD_picker_predictors.json"):
        if not os.path.isfile(f"{HERE}/scored/{n}") or sha(f"{HERE}/scored/{n}") != want.get(n):
            raise Refusal(f"scored/{n} differs from results/picker/MANIFEST.sha256")
    lock = open(f"{HERE}/scored/SCORED.lock").read().split()[0]
    if sha(f"{HERE}/manifest/picker_manifest.json") != lock:
        raise Refusal("the manifest is not the one SCORED.lock names")
    return (json.load(open(f"{HERE}/manifest/picker_manifest.json")), json.load(open(f"{HERE}/scored/tierD_picker_targets.json"))["per_clip"],
            json.load(open(f"{HERE}/scored/tierD_picker_predictors.json"))["predictors"])


def counted_stems(M) -> list:
    sys.path.insert(0, HERE)
    import clips
    return sorted({clips.stem_of(r["clip"]) for r in M["rows"] if r["counted"]})


def owners(stems) -> dict:
    """stem -> part (str), from parts.json after its SHA-256 is checked; owns lists disjoint and covering every counted stem."""
    if not os.path.isfile(PARTS) or sha(PARTS) != PARTS_SHA:
        raise Refusal("parts.json differs from its pinned SHA-256")
    own = {}
    for p in load_json(PARTS, "parts.json").get("parts", []):
        for s in p.get("owns", []):
            if s in own:
                raise Refusal(f"parts.json gives {s} to parts {own[s]} and {p['part']}")
            own[s] = str(p["part"])
    missing = [s for s in stems if s not in own]
    if missing:
        raise Refusal(f"parts.json owns no part for {missing}")
    return own


def folders(dirs) -> dict:
    """part (str) -> folder, from each folder's run record; one folder per part."""
    out = {}
    for d in dirs:
        p = str(load_json(os.path.join(d, "colab_run.json"), f"{d}/colab_run.json").get("s3b_part"))
        if p in out:
            raise Refusal(f"two folders for part {p}: {out[p]} and {d}")
        out[p] = d
    return out


def pinned(part, root, rel) -> str:
    """Path of a file in a part's folder, once its SHA-256 equals its pin and its line in the notebook's MANIFEST.sha256."""
    key, p = f"part{part}/{rel}", os.path.join(root, rel)
    if key not in CONTROL_PINS:
        raise Refusal(f"{key} is not pinned")
    if not os.path.isfile(p):
        raise Refusal(f"{key} is missing")
    got = sha(p)
    if got != CONTROL_PINS[key]:
        raise Refusal(f"{key} differs from its pin")
    if rel != "MANIFEST.sha256":
        listed = {}
        for line in open(pinned(part, root, "MANIFEST.sha256")):
            if line.strip():
                h, n = line.split(None, 1); listed[n.strip()] = h
        if listed.get(rel) != got:
            raise Refusal(f"{key} differs from the notebook's MANIFEST.sha256")
    return p


def check_part(part, root) -> dict:
    """The part's run record and gate record against section 6's hashes."""
    if part not in KERNELS:
        raise Refusal(f"unknown part {part}")
    run = load_json(pinned(part, root, "colab_run.json"), f"part {part} run record")
    want = {"notebook_code_cells_sha256": NOTEBOOKS[part], "kernel_sha256": KERNELS[part], "s3b_control_sha256": HOOK,
            "s3b_refs_sha256": REFS, "kernel_rc": 0}
    bad = [k for k, v in want.items() if run.get(k) != v] + (["s3b_part"] if str(run.get("s3b_part")) != part else [])
    if bad or "T4" not in str(run.get("gpu")):
        raise Refusal(f"part {part}'s run record differs from section 6 ({', '.join(bad) or 'gpu'})")
    summ = load_json(pinned(part, root, SUMMARY), f"part {part} gate record")
    g = summ.get("gate") or {}
    if summ.get("s3b_control_sha256") != HOOK:
        raise Refusal(f"part {part}'s prediction step names another hook")
    for stem, ref in REFS.items():
        c = (g.get("clips") or {}).get(stem) or {}
        if c.get("scored_sha256") != ref:
            raise Refusal(f"part {part}'s gate compared {stem} with another reference")
        if not c.get("pass"):
            raise Refusal(f"part {part}'s gate did not pass on {stem}")
    if g.get("pass") is not True:
        raise Refusal(f"part {part}'s gate did not pass")
    return {"started_utc": run.get("started_utc"), "ended_utc": run.get("ended_utc"), "gate": g, "clips": summ.get("clips") or {}}


def controls(dirs, stems, own) -> dict:
    """stem -> its owning part's control, fresh full forecast and gate-checked part, every file pinned."""
    by = folders(dirs)
    need = sorted({own[s] for s in stems})
    if sorted(by) != need:
        raise Refusal(f"folders for parts {sorted(by)}, need {need}")
    parts = {p: check_part(p, by[p]) for p in need}
    found = {}
    for s in stems:
        p = own[s]; d = by[p]
        prov = load_json(pinned(p, d, f"PRED/{s}/provenance.json"), f"{s} provenance")
        ok = (prov.get("stem") == s and prov.get("ckpt_sha256") == STOCK_PIN and (prov.get("ckpt_sha256_by_mode") or {}).get("timing_exact") == STOCK_PIN
              and str(prov.get("timing_mode", "")).startswith("exact") and prov.get("s3b_control_sha256") == HOOK
              and (prov.get("s3b_gate") or {}).get("pass") is True)
        if not ok:
            raise Refusal(f"{s}: the control's provenance is not the exact control from the stock checkpoint after a passing gate")
        c = parts[p]["clips"].get(s) or {}
        if not (c.get("timing_exact_words_recorded") == c.get("timing_exact_word_events") and (c.get("timing_exact_hook_words") or 0) > 0):
            raise Refusal(f"{s}: words recorded or set differ from the clip's word events")
        found[s] = {"part": p, "control": pinned(p, d, f"PRED/{s}/pred_timing_exact.npz"), "fresh_full": pinned(p, d, f"PRED/{s}/pred_full.npz"),
                    "words": {k: c.get(f"timing_exact_{k}") for k in ("word_events", "words_recorded", "hook_words")}}
    return {"parts": parts, "found": found}


def validate(found, merged) -> dict:
    """Each control's shape, values and second grid against the scored full forecast of its clip; and, descriptive only, D26's comparison of
    the fresh full forecast with the scored one. No listener data is read."""
    rep = {}
    for s, f in sorted(found.items()):
        ref = f"{merged}/{s}/pred_full.npz"
        if not os.path.isfile(ref):
            raise Refusal(f"{s}: no scored full forecast in {merged}")
        C, F = np.load(f["control"]), np.load(ref)
        P, st = C["P"], C["starts"]
        shape = P.ndim == 2 and P.shape == F["P"].shape and P.shape[1] == 20484
        finite = bool(np.isfinite(P).all())
        grid = bool(np.array_equal(st, F["starts"]) and len(st) == len(P) and st[0] == 0 and (np.diff(st) == 1).all())
        if not (shape and finite and grid):
            raise Refusal(f"{s}: the control fails the {'shape' if not shape else 'finite' if not finite else 'second grid'} check")
        rep[s] = {"part": f["part"], "shape": list(P.shape), "finite": finite, "second_grid": grid, "words": f["words"],
                  "control_sha256": sha(f["control"]), "scored_full_sha256": sha(ref),
                  "fresh_full_vs_scored": compare(np.load(f["fresh_full"]), F)}
    return rep


def overlay(merged, found, ovl) -> str:
    """ovl/<stem>/: links to the scored pred_full, pred_audio and provenance, and the control as pred_timing (the kind gauges reads).
    Existing links are replaced; every pred_timing under ovl must resolve to a pinned control file."""
    pins = {v for k, v in CONTROL_PINS.items() if k.endswith("/pred_timing_exact.npz")}
    for stem, f in found.items():
        d = os.path.join(ovl, stem); os.makedirs(d, exist_ok=True)
        for n, src in (("pred_full.npz", f"{merged}/{stem}/pred_full.npz"), ("pred_audio.npz", f"{merged}/{stem}/pred_audio.npz"),
                       ("provenance.json", f"{merged}/{stem}/provenance.json"), ("pred_timing.npz", f["control"])):
            if os.path.lexists(os.path.join(d, n)):
                os.unlink(os.path.join(d, n))
            os.symlink(os.path.abspath(src), os.path.join(d, n))
        if sha(os.path.realpath(os.path.join(d, "pred_timing.npz"))) != CONTROL_PINS[f"part{f['part']}/PRED/{stem}/pred_timing_exact.npz"]:
            raise Refusal(f"{stem}: the overlay's control is not its pinned file")
    for r, _, fs in os.walk(ovl):
        if "pred_timing.npz" in fs and sha(os.path.realpath(os.path.join(r, "pred_timing.npz"))) not in pins:
            raise Refusal(f"{r}: a control in the overlay is not a pinned file")
    return ovl


def score(merged, found, ovl_dir):
    """S3b on the checked controls: (the public record, per-row values)."""
    B, G, SP = picker()
    M, per_clip, scored_pred = scored_inputs()
    cnt = [r for r in M["rows"] if r["counted"]]; R = {r["rid"]: r for r in cnt}
    ovl = overlay(merged, found, ovl_dir)
    z = np.load(f"{HERE}/sets.npz"); sets = {k: [z[f"{k}_{i}"] for i in range(int(z[f"{k}_n"]))] for k in ("AUD", "LANG")}
    clips = {c: SP.load_clip(SP.C.stem_of(c), "collected", ovl, sets) for c in sorted({r["clip"] for r in cnt})}
    out, vals = {}, {}
    for q, r in R.items():
        S = [SP.series("collected", x["file"]) for x in r["listeners"]]
        P = G.row_predictors(clips[r["clip"]], S[0][2], [s[1][r["clip"]] for s in S])
        if abs(P["PULA"] - scored_pred[q]["PULA"]) > 1e-12:
            raise Refusal(f"PULA recomputed for {q} differs from the scored value")
        out[q] = {"PULA": P["PULA"], "PULA_timing": P["PULA_timing_approx"], "T1": per_clip[q]["T1"]}
        vals[q] = dict(per_clip[q], PULA_timing=P["PULA_timing_approx"])
    s3b = B.rank_test(B.families(cnt, vals), "PULA", "T1", ("PULA_timing",))
    pub = {"label": LABEL, "rule": RULE, "why_late": WHY_LATE, "beside": BESIDE, "manifest_sha256": sha(f"{HERE}/manifest/picker_manifest.json"),
           "control_sha256": {s: sha(f["control"]) for s, f in found.items()}, "S3b": s3b, "line": LINES[s3b["pass_"]].format(**s3b)}
    return pub, out


def pins(dirs, stems, own) -> dict:
    """What CONTROL_PINS must hold: every file this scorer reads from the part folders (no check is made here)."""
    by, out = folders(dirs), {}
    for p, d in sorted(by.items()):
        rels = ["MANIFEST.sha256", "colab_run.json", SUMMARY]
        rels += [f"PRED/{s}/{n}" for s in stems if own.get(s) == p for n in ("pred_timing_exact.npz", "provenance.json", "pred_full.npz")]
        out.update({f"part{p}/{r}": sha(os.path.join(d, r)) for r in rels if os.path.isfile(os.path.join(d, r))})
    return out


def main(a, results=None):
    results = results or f"{REPO}/results/picker/s3b_late.json"
    if a.print_pins:
        M, _, _ = scored_inputs(); stems = counted_stems(M)
        print(json.dumps(pins(a.s3b, stems, owners(stems)), indent=1, sort_keys=True)); return 0
    if os.path.exists(a.out) and os.listdir(a.out):
        raise SystemExit(f"--out {a.out} is not a new, empty directory")
    os.makedirs(a.out, exist_ok=True)
    try:
        if not CONTROL_PINS:
            raise Refusal("the control forecasts' SHA-256 are not pinned")
        M, _, _ = scored_inputs(); stems = counted_stems(M)
        got = controls(a.s3b, stems, owners(stems))
        check = {"parts": {p: {k: v[k] for k in ("started_utc", "ended_utc", "gate")} for p, v in got["parts"].items()},
                 "clips": validate(got["found"], a.merged)}
        miss = sorted(s for s, c in check["clips"].items() if not c["fresh_full_vs_scored"].get("pass"))
        check["fresh_full_vs_scored_misses"] = miss
        json.dump(check, open(f"{a.out}/s3b_check.json", "w"), indent=1, default=float)
        print(f"check passed: {len(check['clips'])} controls from parts {sorted(check['parts'])}; fresh full vs scored (descriptive) misses D26's "
              f"bar on {len(miss)} clip(s){': ' + ', '.join(miss) if miss else ''}", flush=True)
        if a.check_only:
            return 0
        pub, rows = score(a.merged, got["found"], os.path.join(a.out, "overlay"))
        pub["fresh_full_vs_scored"] = {"descriptive": True, "misses": miss,
                                       "clips": {s: c["fresh_full_vs_scored"] for s, c in check["clips"].items()}}
    except (Refusal, KeyError, StopIteration) as e:
        why = str(e) if isinstance(e, Refusal) else f"an input lacks an expected entry ({type(e).__name__} {e})"
        pub, rows = {"label": LABEL, "rule": RULE, "why_late": WHY_LATE, "S3b": None, "refused": why, "line": NOT_RUN.format(reason=why)}, {}
        json.dump(pub, open(f"{a.out}/s3b_late.json", "w"), indent=1)
        if not a.check_only:
            json.dump(pub, open(results, "w"), indent=1)
        print(pub["line"], flush=True)
        return 2
    json.dump({**pub, "rows": rows}, open(f"{a.out}/s3b_late.json", "w"), indent=1, default=float)
    json.dump(pub, open(results, "w"), indent=1, default=float)
    print(pub["line"], flush=True)
    return 0


def args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True); ap.add_argument("--s3b", nargs="+", required=True); ap.add_argument("--out")
    ap.add_argument("--check-only", action="store_true"); ap.add_argument("--print-pins", action="store_true")
    a = ap.parse_args(argv)
    if not a.print_pins and not a.out:
        ap.error("--out is required")
    return a


if __name__ == "__main__":
    sys.exit(main(args()))
