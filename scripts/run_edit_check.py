"""Edit check with Spark Open (docs/rules/edit_check_rule.md; exploratory): the frame set of build_edit_check_frames.py (every file checked
against its MANIFEST first), the committed question, Glimmer's parser and one retry, sent to Muse Spark Open on Baseten through the typed
egress only (BASETEN_API_KEY in the environment for the run only; each frame on the public-frame allowlist). Bars: intact frames with a
face (bridge + new, 62) yes >= 80%; faces removed (62) no >= 90%; area matched (62) yes >= 80%; no-mask intact frames (31) at most 3 yes.
An unparsed answer or an error counts against the bar every time. Writes results/edit_check_spark_open.json; prints counts, rates, spend.
usage: python scripts/run_edit_check.py FRAMES_DIR [--workers 4]"""
import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("sofc", ROOT / "scripts/run_spark_open_face_check.py")
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)       # the committed ask/parse/retry and spend accounting

VERSIONS = ("intact", "faces_removed", "area_matched")


def jobs(meta):
    out = [(v, s, f) for s in ("bridge", "new") for v in VERSIONS for f in meta["sets"][s]]
    return out + [("intact", "no_mask", f) for f in meta["sets"]["no_mask"]]


def one(fr, v, s, f):
    row = F.one(fr, v, {"t_release": f.get("film_s"), "name": f["name"], "sam_faces": 0 if s == "no_mask" else 1})
    if row:
        row["set"] = s
    return row


def rates(rows):
    pick = lambda v, sets: [r for r in rows if r["version"] == v and r["set"] in sets]
    yes = lambda rs: sum(r["answer"] == "yes" for r in rs)
    no = lambda rs: sum(r["answer"] == "no" for r in rs)
    face = ("bridge", "new")
    it, fr_, am, nm = pick("intact", face), pick("faces_removed", face), pick("area_matched", face), pick("intact", ("no_mask",))
    r = {"intact_with_face_yes": [yes(it), len(it)], "faces_removed_no": [no(fr_), len(fr_)], "area_matched_yes": [yes(am), len(am)],
         "no_mask_yes_or_unparsed": [len(nm) - no(nm), len(nm)],
         "by_set": {s: {v: [yes(pick(v, (s,))), no(pick(v, (s,))), len(pick(v, (s,)))] for v in VERSIONS} for s in face},
         "retried": sum("first_answer" in x for x in rows), "unparsed": sum(x["answer"] == "unparsed" for x in rows),
         "errors": sum(x["answer"] == "error" for x in rows)}
    r["pass"] = {"intact": len(it) == 62 and yes(it) >= 0.80 * 62, "faces_removed": len(fr_) == 62 and no(fr_) >= 0.90 * 62,
                 "area_matched": len(am) == 62 and yes(am) >= 0.80 * 62, "no_mask": len(nm) == 31 and len(nm) - no(nm) <= 3}
    r["pass"]["all"] = all(r["pass"].values())
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("frames"); ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
    fr = Path(a.frames)
    bad = [l.split()[1] for l in open(fr / "MANIFEST.sha256") if hashlib.sha256((fr / l.split()[1]).read_bytes()).hexdigest() != l.split()[0]]
    assert not bad, f"{len(bad)} frames differ from the manifest"
    meta = json.load(open(fr / "frames.json")); t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        rows = [r for r in ex.map(lambda j: one(fr, *j), jobs(meta)) if r]
    res = {"label": "exploratory", "rule": "docs/rules/edit_check_rule.md", "model": F.MODEL, "decides": "no (Glimmer decides on Saturday)",
           "frames_manifest_sha256": hashlib.sha256((fr / "MANIFEST.sha256").read_bytes()).hexdigest(), "spend_usd": round(F.spend[0], 4),
           "cap_usd": F.CAP, "seconds": round(time.time() - t0, 1), "rates": rates(rows), "rows": rows}
    json.dump(res, open(ROOT / "results/edit_check_spark_open.json", "w"), indent=1)
    r = res["rates"]
    print(f"rows {len(rows)}; intact yes {r['intact_with_face_yes']}; removed no {r['faces_removed_no']}; control yes {r['area_matched_yes']}; "
          f"no-mask yes {r['no_mask_yes_or_unparsed']} (at most 3); retried {r['retried']}, unparsed {r['unparsed']}, errors {r['errors']}; "
          f"pass {r['pass']}; spend ${res['spend_usd']:.4f}")


if __name__ == "__main__":
    main()
