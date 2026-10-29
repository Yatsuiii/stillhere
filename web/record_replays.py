"""Records the hosted page's fallback runs ("recorded run") from real live calls, once, into web/replay/: a plan (Spark 1.3) and its
build check, audit and film masks, a declined plan, and Fork P's three example plans on both backends (results/planner_v21_examples.json,
already recorded live). Needs MODEL_API_KEY (or the laptop's Muse config) and FIREWORKS_API_KEY in the environment for the run only;
the film step needs opencv and meta-sam-parser. Prints counts, decisions and costs only.
usage: python web/record_replays.py"""
import base64
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from web import guard, stations  # noqa: E402
from web.store import MemoryStore, set_store  # noqa: E402

OUT = REPO / "web/replay"
DECLINE = "Use this on my patient to see if they understand."              # the plan's decline example (cloud plan v2, item 4)
LABEL = {"spark-1.3": "Muse Spark 1.3", "spark-open": "Muse Spark Open (Fireworks)"}


def jpeg_data_url(data_url: str) -> str:
    import cv2
    import numpy as np
    png = base64.b64decode(data_url.split(",", 1)[1])
    ok, buf = cv2.imencode(".jpg", cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def main():
    set_store(MemoryStore()); sid = "0" * 32
    ex = json.loads((REPO / "results/planner_v21_examples.json").read_text())
    runs = []
    for e in ex["examples"]:
        for r in e["runs"]:
            runs.append({"request": e["text"], "backend": r["backend"], "backend_label": LABEL.get(r["backend"], r["backend"]),
                         "source": "results/planner_v21_examples.json", "card": {"decision": r["decision"], "route": {k: r["route"][k] for k in ("kind", "clip_id", "condition", "seed", "defaults")
                                                                                           if k in r["route"]},
                                                                               "lines": [guard.clean(l) for l in r["card"].split("\n") if l.strip()]}})
    p = stations.plan(ex["examples"][0]["text"], "spark-1.3", sid); assert p["mode"] == "live", p
    d = stations.plan(DECLINE, "spark-1.3", sid); assert d["mode"] == "live", d
    runs.append({"request": DECLINE, "backend": "spark-1.3", "backend_label": LABEL["spark-1.3"], "source": "web/record_replays.py",
                 "card": d["card"]})
    print("plan:", p["card"]["decision"], "| decline example:", d["card"]["decision"])
    rt = p["card"]["route"]
    b = stations.build(p["run_id"], sid, rt["clip_id"], rt["condition"], rt["seed"]); assert b["mode"] == "live", b
    b.pop("audio_b64", None); b.pop("receipt_id", None)
    print("build:", b["condition"], "pass", b["pass"], [i["value"] for i in b["checklist"]])
    a = stations.audit(p["run_id"], sid); assert a["mode"] == "live", a
    a.pop("receipt_id", None); print("audit notes:", len(a["notes"]))
    f = stations.film(p["run_id"], sid, "face"); assert f["mode"] == "live", f
    for fr in f["frames"]:
        for k in ("intact", "faces_removed", "area_matched"):
            if fr.get(k):
                fr[k] = jpeg_data_url(fr[k])
    f.pop("receipt_id", None); print("film frames:", [(x["id"], x.get("n_masks", 0)) for x in f["frames"]])
    OUT.mkdir(exist_ok=True)
    meta = {"recorded_utc": stations._now(), "label": "recorded run"}
    json.dump({**meta, "runs": runs}, open(OUT / "plan.json", "w"), indent=1)
    json.dump({**meta, "default": b, f"{rt['clip_id']}:{rt['condition']}": b}, open(OUT / "build.json", "w"), indent=1)
    json.dump({**meta, **a}, open(OUT / "audit.json", "w"), indent=1)
    json.dump({**meta, **{k: v for k, v in f.items() if k != "mode"}}, open(OUT / "film.json", "w"))
    print("spend:", stations.ledger.remaining())


if __name__ == "__main__":
    main()
