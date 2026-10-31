"""R3, descriptive only (dated note of Oct 31 on docs/rules/r3_variants_rule.md; cloud ruling of 21:30 UTC, item 8): each forecast's
spread, forecast only, no brain data: per vertex the SD over time of the 1 Hz forecast, averaged over the LANG and the AUD vertices
(p1/noise_floor.parcels, as the harness sets them), for the stock, fine-tuned and averaged (candidate (a): the raw mean of the two,
second by second) forecasts, full and audio-only, English (D3's) and French (the laptop's), the files score_variants.py pins.
Run from the analysis tree's p1/. Writes results/r3_forecast_sd.json."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
import noise_floor as N  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{REPO}/lpp_variants")
import score_variants as SV  # noqa: E402

P = N.parcels()
SETS = {"LANG": np.concatenate([v for h, v in P["LANG"]]), "AUD": np.concatenate([v + (10242 if h == "rh" else 0) for h, v in P["AUD"]])}
FILES = {"EN": {"stock": "../d22/d3/out_stock/d3_out/pred_en_{m}.npz", "tuned": "../d22/d3/out_tuned/d3_out/pred_en_{m}.npz"},
         "FR": {"stock": "../lpp_variants/forecast/out/pred_fr_stock_{m}.npz", "tuned": "../lpp_variants/forecast/out/pred_fr_tuned_{m}.npz"}}
pins = {**SV.PINS, **SV.FR_PINS}
out = {"label": "descriptive, forecast only; decides nothing", "rule": "docs/rules/r3_variants_rule.md", "unit": "mean over vertices of the SD over time"}
for lang, fs in FILES.items():
    for m in ("full", "audio"):
        A = {}
        for k, f in fs.items():
            p = f.format(m=m); assert SV.sha256(p) == pins[p], f"{p} differs from score_variants.py's pin"
            d = np.load(p); A[k] = d["P"].astype("float64"); A[k + "_starts"] = d["starts"]
        assert np.array_equal(A["stock_starts"], A["tuned_starts"])
        A["average"] = (A["stock"] + A["tuned"]) / 2
        for k in ("stock", "tuned", "average"):
            sd = A[k].std(0)
            out.setdefault(lang, {}).setdefault(m, {})[k] = {s: round(float(sd[v].mean()), 5) for s, v in SETS.items()}
json.dump(out, open(f"{REPO}/results/r3_forecast_sd.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k in ("EN", "FR")}))
