"""Writes results/d20_notthefall.json, the page's derived summary of notthefall (PREREG D20, D27), from the two frozen outputs, which
it only reads: the stillhere-d20-score run's tierD_d20_score.json and the stillhere-d20-baselines run's tierD_d20_baselines.json.
It keeps only what the page shows (ruling of Oct 29, 17:35 UTC, item 3): status, N, Hodges-Lehmann median with its 95% CI and p for the
primary; the secondary in its fail form; the two stand-ins' p values and matches_frozen; and the SHA-256 of both frozen files. No k of N
(D23(b)) and no per-listener value.
usage: python scripts/make_d20_summary.py SCORE_JSON BASELINES_JSON"""
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MINUTES, TRS = 9.1, 365                         # story length: 547.0 s at TR 1.5 s (public events files), as PREREG D20's line has it


def test(t: dict) -> dict:
    return {"status": t["status"], "n": t["n_pairs"], "hl": t["hl"], "ci95": list(t["hl_ci95"]), "p": t["p"]}


def main(score_path: str, base_path: str) -> dict:
    sb, bb = Path(score_path).read_bytes(), Path(base_path).read_bytes()
    s, b = json.loads(sb), json.loads(bb)
    assert s["status"] == "scored" and b["frozen_status"] == "scored" and b["code_manifest_sha256"] == s["code_manifest_sha256"]
    prim, sec = test(s["primary_intact_gt_shortscram"]), test(s["secondary_intact_gt_longscram"])
    sec["line"] = (f"Intact vs the coarser scramble (descriptive): HL {sec['hl']:.4f} [95% CI {sec['ci95'][0]:.4f}, {sec['ci95'][1]:.4f}], "
                   f"p = {sec['p']:.3g}, {sec['n']} listeners, {MINUTES} minutes = {TRS} TRs per version; "
                   + ("did not reach p < 0.05." if sec["status"] != "pass" else "reached p < 0.05."))
    wr, env = b["tests"]["LANG_word_rate"]["intact_gt_shortscram"], b["tests"]["LANG_envelope"]["intact_gt_shortscram"]
    out = {"format": "stillhere-d20-summary/1",
           "label": "pre-registered (PREREG D20, D27): derived from the frozen outputs, only what the page shows",
           "test": "language readout, intact > sentences shuffled (notthefall shortscram), one-sided Wilcoxon, p < 0.05",
           "primary": prim, "secondary": sec, "minutes": MINUTES, "trs_per_version": TRS,
           "baselines": {"label": "exploratory", "rule": "docs/rules/cheap_baselines_rule.md", "matches_frozen": b["matches_frozen"],
                         "areas": "language", "n": wr["n_pairs"], "word_rate_p": wr["p"], "envelope_p": env["p"]},
           "sources": {"tierD_d20_score.json": hashlib.sha256(sb).hexdigest(), "tierD_d20_baselines.json": hashlib.sha256(bb).hexdigest()},
           "scored_utc": s["started_utc"]}
    json.dump(out, open(REPO / "results/d20_notthefall.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    o = main(sys.argv[1], sys.argv[2])
    print("primary", o["primary"]["status"], "n", o["primary"]["n"], "| secondary", o["secondary"]["status"], "| matches_frozen",
          o["baselines"]["matches_frozen"])
