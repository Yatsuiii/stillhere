"""Renders the D20 scoring kernel WITH the exploratory cheap baselines (docs/rules/cheap_baselines_rule.md), for use in place of the plain
stub on Thursday only if the cloud thread agrees. The frozen d20_score.main runs unchanged from the hashed stillhere-d20-score-code
dataset (it checks its own MANIFEST); a hook on gauges.listener_T1_T3 records the raw cheap gauges (TRIBE unpartialled, HRF x word rate,
HRF x envelope; LANG and AUD) on the same in-memory series for every run the frozen code scores, then returns the frozen value unchanged.
Records are matched back to listeners by their exact T1 value in the frozen result. Per regressor, D20's own paired_test (intact >
shortscram, and intact > longscram if scored) is written to tierD_d20_baselines.json beside the frozen tierD_d20_score.json.
usage: python scripts/make_d20_baselines_kernel.py FROZEN_KERNEL_DIR OUT_KERNEL_DIR POSTED_PICKER_HASH"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOOK = '''
import json

# ---- exploratory cheap baselines (docs/rules/cheap_baselines_rule.md); the frozen scoring call below is unchanged ----
_REC = []
_orig_t1_t3 = G.listener_T1_T3


def _hooked(r, parts, clip, tr):
    out = _orig_t1_t3(r, parts, clip, tr)                          # the frozen value, returned unchanged
    try:
        g = run_gauges(G, C, r, parts, clip, tr)
    except Exception as e:                                         # never disturbs the frozen call
        g = {"error": type(e).__name__}
    _REC.append({"stem": clip.stem, "T1": out[0], "gauges": g})
    return out


G.listener_T1_T3 = _hooked
res = d20_score.main(POSTED, p8="pass")                            # Gate 1 P8 passed (stillhere-gate1 primary)
G.listener_T1_T3 = _orig_t1_t3
out = {"label": LABEL, "rule": "docs/rules/cheap_baselines_rule.md", "frozen_status": res.get("status"), "regressors": list(KINDS)}
if res.get("status") == "scored":
    by = {}
    for v, dct in res["T1_per_listener"].items():
        for lst, t1 in dct.items():
            hit = [x for x in _REC if x["stem"] == d20_score.VERSIONS[v] and x["T1"] == t1 and "error" not in x["gauges"]]
            if len(hit) == 1:
                by.setdefault(v, {})[lst] = hit[0]["gauges"]
    tests = {}
    for net in ("LANG", "AUD"):
        for k in KINDS:
            sc = {v: {lst: g[net][k] for lst, g in d.items()} for v, d in by.items()}
            t = {"intact_gt_shortscram": d20_score.paired_test(sc.get("intact", {}), sc.get("shortscram", {}))}
            if "longscram" in sc:
                t["intact_gt_longscram"] = d20_score.paired_test(sc["intact"], sc["longscram"])
            tests[f"{net}_{k}"] = t
    out.update(matched_runs={v: len(d) for v, d in by.items()}, tests=tests)
    for name, t in tests.items():
        p = t["intact_gt_shortscram"]
        print(f"[{LABEL}] D20 with {name}: n {p.get('n_pairs')}, p {p.get('p')}, HL {p.get('hl')} {p.get('hl_ci95')}", flush=True)
json.dump(out, open("/kaggle/working/d20/tierD_d20_baselines.json", "w"), indent=1, default=float)
'''


def render(frozen_dir, out_dir, posted):
    stub = open(f"{frozen_dir}/stillhere_d20_score.py").read()
    head = stub.split("d20_score.main(")[0].replace("@@POSTED_HASH@@", posted)          # frozen imports and the posted hash, no call yet
    lib = open(f"{HERE}/stillhere/cheap_baselines.py").read()
    body = (head + "import gauges as G, clips as C\n"
            + "\n# ---- stillhere/cheap_baselines.py, inlined ----\n" + lib.split('"""', 2)[2] + HOOK)
    os.makedirs(out_dir, exist_ok=True)
    open(f"{out_dir}/stillhere_d20_score.py", "w").write(body)
    shutil.copy(f"{frozen_dir}/kernel-metadata.json", f"{out_dir}/kernel-metadata.json")
    return f"{out_dir}/stillhere_d20_score.py"


if __name__ == "__main__":
    print(render(*sys.argv[1:4]))
