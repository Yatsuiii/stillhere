"""Builds the desk's REAL Gate 1 results document from the frozen run's tier-D files (reads only; counts and group values, no subject
IDs except the D14 demo volunteer). Checks first that the run read the posted input package (MANIFEST 8c93e18b...) and that the D24
lag-sensitive marks are complete (L - 1 s, L + 1 s and dL = 0 all merged); records every source file's SHA-256 in the document.
usage: python scripts/build_desk_results.py [GATE1_OUT_DIR] [OUT_JSON]"""
import hashlib
import json
import sys
from pathlib import Path

from stillhere import frozen

POSTED_MANIFEST = "8c93e18bcc000701b2120fc35156f2b2423ff5d92867cfac21289f0bf6fb04cb"
root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "stillhere/d22/gate1/out")
out = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "results/gate1_real.json")
files = {"gate1": root / "primary/tierD/tierD_primary_gate1.json", "d7_headphone": root / "primary/tierD/tierD_primary_d7_headphone.json",
         "inputs": root / "primary/tierD/tierD_primary_inputs.json", "merged": root / "merged/tierD_merged_gate1.json"}
doc = {k: json.loads(p.read_text()) for k, p in files.items()}
sha = {f"{p.parent.parent.name}/{p.name}" if p.parent.name == "tierD" else f"{p.parent.name}/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest()
       for p in files.values()}

assert doc["inputs"]["manifest_sha256"] == POSTED_MANIFEST, "the run did not read the posted input package"
g, m = doc["gate1"], doc["merged"]
assert g["variant"] == "primary" and m["marks_complete"] and sorted(m["marks_from"]) == ["dl0", "lm1", "lp1"], "D24 marks incomplete"
assert g["n_trs"] == g["volumes"][1] - g["volumes"][0] + 1, "scored-volume count does not match the volume range"
res = frozen.compose(g, doc["d7_headphone"], g["n_trs"], "real", lag_sensitive=m["lag_sensitive"], sources=sha)
frozen.Frozen(res)                                                       # the desk can load it
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
print(f"wrote {out} ({hashlib.sha256(out.read_bytes()).hexdigest()[:16]}): sedation line {g['sedation_stage_line']}, "
      f"lag-sensitive {sorted(k for k, v in res['lag_sensitive'].items() if v)}")
