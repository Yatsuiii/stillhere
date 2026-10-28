"""Builds the tier-D file the demo page reads from the analysis tree (read only), stdlib only:
- results/section8.json: the held-out LPP English section 8 numbers the page shows (event-built recompute, PREREG §8), labelled
  "recomputed with code built during the event; first measured Oct 22-23".
The per-volunteer sedation file (results/cohort.json) is no longer built: the sedation case study carries only frozen group statistics,
which come from results/gate1_real.json. The file records the SHA-256 of its source.
usage: python scripts/build_demo_results.py [ANALYSIS_ROOT] (default ../stillhere)"""
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def section8(root: Path) -> dict:
    src = root / "d22/sec8/heldout_section8.json"; d = json.loads(src.read_text()); s = d["stage"]["145_primary"]
    trs = s["volumes"][1] - s["volumes"][0] + 1; m1 = d["m1"]["LANG"]; ks = sorted((int(k) for k in m1 if int(k) <= 15))
    ve = d["volunteer_equivalent"]["language"]
    return {"label": "recomputed with code built during the event; first measured Oct 22-23", "format": "stillhere-section8/1",
            "dataset": "LPP English section 8 (held out)", "listeners": d["listeners"], "trs": trs, "tr_s": 2.0,
            "stage": {"language_z": s["language"]["z"], "language_p": s["language"]["p"], "shuffled_z": s["shuffled"]["z"],
                      "p_real_gt_shuffled": s["shuffled"]["p_real_gt_shuffled"], "hearing_z": s["hearing"]["z"], "envelope_z": s["envelope"]["z"],
                      "surprisal_z": s["baselines"]["surprisal"], "p_tribe_gt_best_reference": s["p_tribe_gt_best"]},
            "isc_curve_language": [{"k": k, "z": sum(x[0] for x in m1[str(k)]) / len(m1[str(k)])} for k in ks],
            "volunteer_equivalent_language": {"z": ve["z"], "ci95": ve["ci95"], "point_k": ve["point_k"], "lower_k": ve["lower_k"], "claim_k": ve["claim_k"]},
            "hybrid_language": [{"k": int(k), "alone_z": v["alone_z"], "hybrid_z": v["hybrid_z"], "dz": v["dz"], "dz_ci95": v["dz_ci95"]}
                                for k, v in sorted(d["m2"]["LANG"].items(), key=lambda kv: int(kv[0]))],
            "hybrid_wrong_story_language": [{"k": int(k), "detect_pct": v["hybrid_detect_pct"]} for k, v in sorted(d["m2"]["wrong_story"].items(), key=lambda kv: int(kv[0]))],
            "m3_wrong_story": {g: {"n": d["m3"][k]["n"], "detected": d["m3"][k]["tribe_wrong_detected"], "pct": d["m3"][k]["tribe_wrong_detect_pct"]}
                               for g, k in (("hearing", "AUD"), ("language", "LANG"))},
            "sources": {src.relative_to(root).as_posix(): sha(src)}}


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else REPO.parent / "stillhere").resolve()
    for name, doc in (("section8", section8(root)),):
        p = REPO / f"results/{name}.json"; p.write_text(json.dumps(doc, indent=1) + "\n"); print(p.relative_to(REPO), sha(p))
