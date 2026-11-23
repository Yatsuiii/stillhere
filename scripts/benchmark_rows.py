"""Builds the public benchmark tables from committed results only (no brain data is read).
usage: python scripts/benchmark_rows.py OUT_DIR
Writes heldout.json, wall.json and hybrid.json, each {rows, meta}, with this script's own text in meta."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results"
SELF = Path(__file__).read_text()


def load(p):
    return json.loads((R / p).read_text())


# Source studies for each Narratives story, from the reference list of Nastase et al. 2021 (Sci Data 8:250), read via Crossref.
NARR = {
    "pieman": ("Lerner, Honey, Silbert & Hasson 2011, J Neurosci (and later Hasson lab papers)", "Hasson lab, Princeton"),
    "prettymouth": ("Yeshurun et al. 2017, Psychol Sci", "Hasson lab, Princeton"),
    "milkyway": ("Yeshurun, Nguyen & Hasson 2017, PNAS", "Hasson lab, Princeton"),
    "merlin": ("Zadbood, Chen, Leong, Norman & Hasson 2017, Cereb Cortex", "Hasson and Norman labs, Princeton"),
    "schema": ("Baldassano, Hasson & Norman 2018, J Neurosci", "Hasson and Norman labs, Princeton"),
    "shapes": ("Nguyen, Vanderwal & Hasson 2019, NeuroImage", "Hasson lab, Princeton"),
    "21styear": ("Chang, Lazaridi, Yeshurun, Norman & Hasson 2021, J Cogn Neurosci", "Hasson and Norman labs, Princeton"),
    "tunnel": ("Lositsky et al. 2016, eLife", "Princeton"),
    "slumlordreach": ("First released in Nastase et al. 2021, Sci Data", "Hasson lab, Princeton"),
    "pni": ("Nastase, Liu, Hillman, Norman & Hasson 2020, NeuroImage", "Hasson and Norman labs, Princeton"),
}


def narr_source(clip):
    for key in ("schema", "shapes", "milkyway"):
        if clip.startswith(key):
            return NARR[key]
    if clip in ("black", "forgot", "bronx", "piemanpni"):
        return NARR["pni"]
    return NARR.get(clip, ("not stated", "not stated"))


def wall_rows():
    w, b = load("picker/tierD_picker_wall.json"), load("picker/tierD_wall_baselines.json")["per_clip"]
    rows = []
    for t in w["tiles"]:
        lang = b.get(f"{t['block']}/{t['clip']}", {}).get("LANG", {})
        if t["dataset"] == "LPP":
            study, lab = "Li, Hale & Pallier, Le Petit Prince corpus (OpenNeuro ds003643)", "Li, Hale and Pallier"
        else:
            study, lab = narr_source(t["clip"])
        rows.append({
            "story": t["clip"], "dataset": t["dataset"], "language": t["language"], "listeners": t["listeners"],
            "minutes": round(t["minutes"], 1), "followed": "yes" if t["lit"] else "no",
            "effect": round(t["hl_d"], 3), "effect_lo": round(t["hl_ci95"][0], 3), "effect_hi": round(t["hl_ci95"][1], 3),
            "tribe_fit": round(lang.get("tribe", float("nan")), 3) if lang else None,
            "word_rate_fit": round(lang.get("word_rate", float("nan")), 3) if lang else None,
            "counted": "yes" if t["counted"] else f"no ({t['touched_reason'] or 'not counted'})",
            "source_study": study, "lab": lab,
        })
    return rows


def heldout_rows():
    p, a, s, e, d, g = (load(x) for x in ("r2_paranoia.json", "alice_heldout.json", "section8.json", "emofilm_film_score.json",
                                          "d20_notthefall.json", "gate1_real.json"))
    pv = p["models"]["stock"]["volunteer_equivalent_language"]
    pd = p["models"]["stock"]["stage"]["primary"]["language"]["detect_pct"]
    av = a["volunteer_equivalent_language"]
    ad = a["stage"]["full_primary"]["language"]["detect_pct"]
    sv = s["volunteer_equivalent_language"]
    lz = e["gauges"]["LANGUAGE"]
    p4 = g["gate1"]["bars"]["P4"]["L"]
    return [
        {"test": "Paranoia", "owners": "Finn, Corlett, Chen, Bandettini & Constable (Finn et al. 2018)", "dataset": "OpenNeuro ds001338",
         "people": 22, "minutes": 20.8, "worth": round(pv["point_k"], 1), "worth_low": round(pv["lower_k"], 1),
         "detected_pct": round(pd), "status": "exploratory, rule committed before download",
         "result": "Worth about 3 real listeners. Detected 21 of 22 listeners with zero volunteers. Beat word rate for 22 of 22."},
        {"test": "Le Petit Prince, section 8", "owners": "Li, Hale & Pallier", "dataset": "OpenNeuro ds003643",
         "people": 49, "minutes": 4.8, "worth": round(sv["point_k"], 1), "worth_low": round(sv["lower_k"], 1),
         "detected_pct": None, "status": "held-out section",
         "result": "Worth about 5 real listeners. Wrong-story false alarms 10%, over the 7% limit."},
        {"test": "Alice", "owners": "Bhattasali, Brennan, Luh, Franzluebbers & Hale", "dataset": "OpenNeuro ds002322",
         "people": 22, "minutes": 12.5, "worth": round(av["point_k"], 1), "worth_low": round(av["lower_k"], 1),
         "detected_pct": round(ad), "status": "exploratory, rule committed before download",
         "result": "Tracked listeners (z 1.42) but worth less than one real listener at the lower bound."},
        {"test": "Not the Fall, intact vs shuffled", "owners": "Chien & Honey 2020 (data in Narratives)", "dataset": "OpenNeuro ds002345",
         "people": d["primary"]["n"], "minutes": d["minutes"], "worth": None, "worth_low": None, "detected_pct": None,
         "status": "pre-registered",
         "result": f"Passed: intact matched better than sentence-shuffled (p = {d['primary']['p']:.3f}). Coarser scramble failed "
                   f"(p = {d['secondary']['p']:.3f}). Word rate alone: p = {d['baselines']['word_rate_p']:.2f}."},
        {"test": "Sedation, awake vs deep", "owners": "Kandeepan, Rudas, Gomez, Stojanoski, Valluri, Owen, Naci, Soddu & Nichols",
         "dataset": "OpenNeuro ds003171", "people": 12, "minutes": 4.8, "worth": None, "worth_low": None, "detected_pct": None,
         "status": "pre-registered",
         "result": f"Main bar failed (9 of 12, needed 10). Telling awake from deep per person: AUC {p4['auc']:.2f} vs "
                   f"{p4['isc_auc']:.2f} with real volunteers."},
        {"test": "Emo-FilM, Tears of Steel", "owners": "Morgenroth, Moia, Vilaclara, Fournier, Muszynski, Ploumitsakou, "
         "Almato Bellavista, Vuilleumier & Van De Ville", "dataset": "OpenNeuro ds004892", "people": e["viewers"]["scored"],
         "minutes": 3.0, "worth": None, "worth_low": None, "detected_pct": round(100 * lz["share_detected"]),
         "status": "exploratory",
         "result": "Followed hearing and visual areas. In language areas no better than word rate, and worth less than one viewer."},
    ]


def hybrid_rows():
    s = load("section8.json")
    fa = {r["k"]: r["detect_pct"] for r in s["hybrid_wrong_story_language"]}
    rows = [{"reference": f"{r['k']} volunteer{'s' if r['k'] > 1 else ''} alone", "volunteers": r["k"], "forecast": "no",
             "z": round(r["alone_z"], 2), "false_alarm_pct": None} for r in s["hybrid_language"]]
    rows += [{"reference": f"Forecast + {r['k']} volunteer{'s' if r['k'] > 1 else ''}", "volunteers": r["k"], "forecast": "yes",
              "z": round(r["hybrid_z"], 2), "false_alarm_pct": round(fa.get(r["k"], float("nan")), 1)} for r in s["hybrid_language"]]
    rows += [{"reference": "15 volunteers alone", "volunteers": 15, "forecast": "no",
              "z": round([c["z"] for c in s["isc_curve_language"] if c["k"] == 15][0], 2), "false_alarm_pct": None},
             {"reference": "Forecast alone", "volunteers": 0, "forecast": "yes",
              "z": round(s["volunteer_equivalent_language"]["z"], 2), "false_alarm_pct": round(s["m3_wrong_story"]["language"]["pct"], 1)}]
    return rows


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    meta = [{"label": "benchmark_rows.py", "value": SELF, "type": "code", "lang": "python"},
            {"label": "Input", "value": "StillHere results/ at commit 4ea23d0 (github.com/Yatsuiii/stillhere)"}]
    for name, fn in (("heldout", heldout_rows), ("wall", wall_rows), ("hybrid", hybrid_rows)):
        rows = fn()
        (out / f"{name}.json").write_text(json.dumps({"rows": rows, "meta": meta}, indent=1, allow_nan=False))
        print(name, len(rows))
