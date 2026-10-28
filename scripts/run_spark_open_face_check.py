"""Spark Open face-edit check (docs/rules/spark_open_face_check_rule.md; exploratory): the 93 frames of glimmer_edit_frames (every file
checked against its MANIFEST first), Glimmer's question, Glimmer's parser and one retry for an unparsed answer, sent to Muse Spark Open on
Baseten through the typed egress only (BASETEN_API_KEY set in the environment for the run only). Writes results/spark_open_face_check.json
and prints counts, rates, timings and spend only. usage: python scripts/run_spark_open_face_check.py FRAMES_DIR [--workers 4]"""
import argparse
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import httpx  # noqa: E402

from stillhere import egress, meta_client, planner  # noqa: E402

MODEL, CAP, PRICE = "baseten/muse-spark-open-1026", 10.0, (1.25e-6, 4.25e-6)
VERSIONS = ("intact", "faces_removed", "area_matched")
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["answer"], "properties": {"answer": {"type": "string"}}}
_tl = threading.local(); _real_post = httpx.post; lock = threading.Lock(); spend = [0.0]


def _recording_post(*a, **k):                         # keeps each response's usage block (meta_client logs no text)
    r = _real_post(*a, **k)
    try:
        _tl.usage = r.json().get("usage") or {}
    except ValueError:
        _tl.usage = {}
    return r


httpx.post = _recording_post


def parse(txt):
    """Glimmer's parser (pilot/probes/k1e), unchanged."""
    return "yes" if txt.startswith("yes") or " yes" in txt[:20] else "no" if txt.startswith("no") or " no" in txt[:20] else "unparsed"


def ask(frame, prompt):
    _tl.usage = {}
    req = egress.Request(model=MODEL, system=[planner.public("frame_check_system")], user=[egress.PublicFrame(path=str(frame)), planner.public(prompt)],
                         schema_name="frame_answer", json_schema=SCHEMA)
    out = meta_client.call(req, timeout=120, effort=planner.EFFORT)
    u = _tl.usage or {}; pt, ct = u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0
    with lock:
        spend[0] += pt * PRICE[0] + ct * PRICE[1]
    return parse(str(out["content"].get("answer", "")).strip().lower()), pt, ct


def one(fr, v, f):
    with lock:
        if spend[0] >= CAP:
            return None
    row = {"version": v, "t_release": f["t_release"], "name": f["name"], "sam_faces": f["sam_faces"]}; t0 = time.time()
    try:
        ans, pt, ct = ask(fr / v / f["name"], "frame_check_question"); row.update(answer=ans, prompt_tokens=pt, completion_tokens=ct)
        if ans == "unparsed":                                       # one retry, yes or no only; the first answer stays in the row
            ans2, pt2, ct2 = ask(fr / v / f["name"], "frame_check_retry")
            row.update(answer=ans2, first_answer="unparsed", retry_prompt_tokens=pt2, retry_completion_tokens=ct2)
    except Exception as e:
        row.update(answer="error", error=type(e).__name__)
    row["s"] = round(time.time() - t0, 2)
    return row


def rates(rows):
    by = {v: [r for r in rows if r["version"] == v] for v in VERSIONS}
    intact = [r for r in by["intact"] if r["sam_faces"] > 0]
    yes = lambda rs: sum(r["answer"] == "yes" for r in rs) / len(rs) if rs else None
    no_strict = lambda rs: sum(r["answer"] == "no" for r in rs) / len(rs) if rs else None     # unparsed or error counts as yes
    r = {"intact_with_face_yes": yes(intact), "faces_removed_no": no_strict(by["faces_removed"]), "area_matched_yes": yes(by["area_matched"]),
         "n": {v: len(by[v]) for v in VERSIONS}, "n_intact_with_face": len(intact),
         "unparsed_after_retry": sum(x["answer"] == "unparsed" for x in rows), "errors": sum(x["answer"] == "error" for x in rows),
         "retried": sum("first_answer" in x for x in rows)}
    r["pass"] = {"intact": r["intact_with_face_yes"] is not None and r["intact_with_face_yes"] >= 0.80,
                 "faces_removed": r["faces_removed_no"] is not None and r["faces_removed_no"] >= 0.90,
                 "area_matched": r["area_matched_yes"] is not None and r["area_matched_yes"] >= 0.80}
    r["pass"]["all"] = all(r["pass"].values()) and all(r["n"][v] == 31 for v in VERSIONS)
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("frames"); ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
    fr = Path(a.frames)
    bad = [l.split()[1] for l in open(fr / "MANIFEST.sha256") if hashlib.sha256((fr / l.split()[1]).read_bytes()).hexdigest() != l.split()[0]]
    assert not bad, f"{len(bad)} frames differ from the manifest"
    frames = json.load(open(fr / "frames.json"))["frames"]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        rows = [r for r in ex.map(lambda vf: one(fr, *vf), [(v, f) for v in VERSIONS for f in frames]) if r]
    res = {"label": "exploratory", "rule": "docs/rules/spark_open_face_check_rule.md (stillhere-app 9831f1e)", "model": MODEL,
           "frames_manifest_sha256": hashlib.sha256((fr / "MANIFEST.sha256").read_bytes()).hexdigest(), "spend_usd": round(spend[0], 4), "cap_usd": CAP,
           "seconds": round(time.time() - t0, 1), "rates": rates(rows), "rows": rows}
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(res, open(ROOT / "results/spark_open_face_check.json", "w"), indent=1)
    r = res["rates"]
    print(f"rows {len(rows)}; intact-with-face yes {r['intact_with_face_yes']:.3f} (bar 0.80); faces-removed no {r['faces_removed_no']:.3f} (bar 0.90); "
          f"area-matched yes {r['area_matched_yes']:.3f} (bar 0.80); retried {r['retried']}, unparsed {r['unparsed_after_retry']}, errors {r['errors']}; "
          f"pass {r['pass']}; spend ${res['spend_usd']:.4f}; {res['seconds']} s")


if __name__ == "__main__":
    main()
