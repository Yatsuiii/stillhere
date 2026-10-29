"""worker/regions.json: the frozen proof's LANG (language) and AUD (hearing) parcel sets (stillhere/heldout.py parcel_sets on the
Schaefer-400/17 atlas) as vertex index lists in TRIBE's fsaverage5 order, so the sandboxed worker needs neither the atlas nor nibabel.
usage: PYTHONPATH=.:<site-packages with nibabel> python3 worker/make_regions.py ATLAS_DIR"""
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from stillhere import heldout  # noqa: E402

atlas = Path(sys.argv[1]); files = [atlas / f"{h}.Schaefer2018_400Parcels_17Networks_order.annot" for h in ("lh", "rh")]
sets = heldout.parcel_sets(*files)
out = {"about": "stillhere/heldout.py parcel_sets: LANG = language parcels, AUD = hearing parcels; one vertex index list per parcel, "
                "TRIBE fsaverage5 order (left 0..10241, right 10242..20483)",
       "atlas_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
       "LANG": [[int(i) for i in v] for v in sets["LANG"]], "AUD": [[int(i) for i in v] for v in sets["AUD"]]}
(REPO / "worker/regions.json").write_text(json.dumps(out, separators=(",", ":")) + "\n")
print(f"LANG {len(out['LANG'])} parcels / {sum(map(len, out['LANG']))} vertices; AUD {len(out['AUD'])} / {sum(map(len, out['AUD']))}")
