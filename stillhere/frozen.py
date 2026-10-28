"""Frozen Gate 1 results for the analysis desk (SPARK-PLAN A, "Lookups only"). One JSON document per Gate 1 run, composed from the
tier-D output of the private stillhere-gate1 kernel (tierD_gate1.json plus the included counts of tierD_d7_headphone.json) or from the
SYNTHETIC test fixtures. Nothing is computed here beyond flattening and the minutes of listening. Counts only: subject IDs are dropped,
except the D14 demo volunteer. Pure stdlib, so the fixture script can import it next to numpy/scipy."""
import json
import math
from pathlib import Path
from typing import NamedTuple

FORMAT = "stillhere-desk-results/1"
TR_S = 2.0
STATES = ("awake", "light", "deep", "recovery")


class Analysis(NamedTuple):
    prereg: str       # PREREG reference, shown on declines
    holm: bool        # decided under Holm (D10): passed = holm_pass
    isc: bool         # compares with LOO-ISC, so its card states the LOO reference size as same-clip controls (D18)


# The registered analyses the desk can look up. "sedation" is the D11 composite: whether the §7 / D21 Part B stage line is licensed.
ANALYSES = {
    "sedation": Analysis("§6 Gate 1 table, §7, D11, D21 Part B", False, False),
    "P7": Analysis("§6 P7, D10", False, True),
    "P1": Analysis("§6 P1", False, False),
    "P3": Analysis("§6 P3", False, False),
    "P8": Analysis("§6 P8, D12", False, False),
    "P2a": Analysis("§6 P2a, D10", True, False),
    "P2b": Analysis("§6 P2b, D17", True, True),
    "P4": Analysis("§6 P4, D17", True, True),
    "P5_H": Analysis("§6 P5, D13", True, False),
    "P5_L": Analysis("§6 P5, D13", True, False),
    "P6": Analysis("§6 P6", True, False),
}
_GATE1_KEYS = ("bars", "awake_hearing_gate", "deep_effective_n", "rows", "sedation_stage_line", "come_back_on_waking", "full_headline")


def compose(gate1: dict, d7: dict, trs_per_state: int, source: str, lag_sensitive: dict | None = None, sources: dict | None = None) -> dict:
    """gate1: gate1_bars.gate1() output (group statistics only: no subject ID is carried); d7: {"included": {state: [subject IDs]}, "deep_effective_n": int}.
    lag_sensitive: gate1_merge's marks (D24: pass or fail differs at L +/- 1 s or dL = 0), keyed by bar or gate; sources: file -> SHA-256.
    A real document also carries the timing notes every quoted Gate 1 result needs (D24 sets L; D25 records the early scoring)."""
    if source not in ("synthetic", "real"):
        raise ValueError("source must be 'synthetic' or 'real'")
    inc = {st: len(d7["included"][st]) for st in STATES}
    doc = {"format": FORMAT, "source": source, "trs_per_state": int(trs_per_state), "tr_s": TR_S, "included": inc,
           "deep_effective_n": int(d7["deep_effective_n"]), "isc_loo_reference": max(inc["awake"] - 1, 0),
           "gate1": {k: gate1[k] for k in _GATE1_KEYS}}
    if lag_sensitive is not None:
        doc["lag_sensitive"] = {aid: bool(lag_sensitive["sedation_stage_line" if aid == "sedation" else aid]) for aid in ANALYSES}
    if sources is not None:
        doc["sources"] = dict(sources)
    if source == "real":
        doc["timing_notes"] = ["D24", "D25"]
    return doc


def _flat(prefix: str, x, out: dict):
    if isinstance(x, dict):
        for k, v in x.items():
            _flat(f"{prefix}.{k}", v, out)
    elif isinstance(x, list) and len(x) == 2:                   # a 95% interval
        _flat(f"{prefix}_lo", x[0], out)
        _flat(f"{prefix}_hi", x[1], out)
    elif isinstance(x, bool) or x is None:
        out[prefix] = x
    elif isinstance(x, (int, float)):
        out[prefix] = x if math.isfinite(x) else None            # an infinite ratio has no number to show
    # strings (row texts) are not tier-D values and are left out


class Frozen:
    def __init__(self, doc: dict):
        if doc.get("format") != FORMAT or doc.get("source") not in ("synthetic", "real"):
            raise ValueError("not a desk results document")
        self.doc, self.g = doc, doc["gate1"]
        self.synthetic = doc["source"] == "synthetic"
        self.minutes = doc["trs_per_state"] * doc["tr_s"] / 60

    @classmethod
    def load(cls, path) -> "Frozen":
        return cls(json.loads(Path(path).read_text()))

    def passed(self, aid: str) -> bool:
        if aid == "sedation":
            return bool(self.g["sedation_stage_line"])
        bar = self.g["bars"][aid]
        return bool(bar["holm_pass"] if ANALYSES[aid].holm else bar["pass"])

    def lag_sensitive(self, aid: str) -> bool:
        """D24 mark from gate1_merge; False when the document has no marks (the synthetic fixtures)."""
        return bool(self.doc.get("lag_sensitive", {}).get(aid, False))

    @property
    def timing_notes(self) -> list:
        return list(self.doc.get("timing_notes", []))

    def controls(self, aid: str) -> int:
        return self.doc["isc_loo_reference"] if ANALYSES[aid].isc else 0

    @property
    def included_awake(self) -> int:
        return self.doc["included"]["awake"]

    def lookup(self, aid: str) -> dict:
        """run_registered: the precomputed values of one analysis, flattened to identifier keys "<aid>.<field>". Tier-D only."""
        if aid not in ANALYSES:
            raise KeyError(aid)
        out = {f"{aid}.passed": self.passed(aid), f"{aid}.lag_sensitive": self.lag_sensitive(aid), f"{aid}.trs": self.doc["trs_per_state"],
               f"{aid}.minutes": self.minutes}
        if aid == "sedation":
            p1 = self.g["bars"]["P1"]
            src = {k: self.g[k] for k in ("sedation_stage_line", "come_back_on_waking", "full_headline", "awake_hearing_gate", "deep_effective_n")}
            src.update(k=p1["k"], n=p1["n"], rows_applying=len(self.g["rows"]))
        else:
            src = self.g["bars"][aid]
        _flat(aid, src, out)
        return out
