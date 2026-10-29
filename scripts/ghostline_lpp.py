"""Ghost-line data for LPP English section 8, as docs/rules/display_rule_ghostline.md fixes it (a display, no new statistic).

Reads, under ROOT (the analysis folder): p1/cache_en8 (the 49 listeners' parcel series the frozen held-out proof already read),
k2/out_en8/k2_out/pred_en8_full.npz (the stock TRIBE v2 forecast it used), atlas/ (Schaefer-400) and d22/sec8/heldout_section8.json
(its per-listener LANGUAGE z, used only to pick the median listener). Writes results/ghostline_lpp_s8.json and a reference render
OUTDIR/ghostline_lpp_s8.png outside the repository. Refuses before 2026-10-29 16:00 UTC, if the rule file has no commit or has
uncommitted changes, or if the per-listener file's SHA-256 differs. Prints counts, shapes and timings only.
usage: python scripts/ghostline_lpp.py --root ../stillhere --outdir ../stillhere/ghostline"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from stillhere import heldout as H  # noqa: E402

RULE = "docs/rules/display_rule_ghostline.md"
NOT_BEFORE = "2026-10-29T16:00:00Z"
J0, J1, SHORT_J1 = 10, 154, 69                                          # proof window; short view = its first 120 s
ALL_IDS = tuple(f"sub-EN{k}" for k in "057 058 059 061 062 063 064 065 067 068 069 070 072 073 074 075 076 077 078 079 081 082 083 084 086 087 088 089 091 092 093 094 095 096 097 098 099 100 101 103 104 105 106 108 109 110 113 114 115".split())  # the proof's 49, sorted
FIVE_SEED, FIVE = 20261029, ("sub-EN068", "sub-EN079", "sub-EN091", "sub-EN113", "sub-EN114")
PER_LISTENER_SHA = "73aa5139b5057f00f9ce3d9360ee371a4f0516314150a052befe88edecfe88b5"
ORDER = ["forecast", "chip", "listeners", "one_listener"]
TITLE = "Le Petit Prince, section 8 (LPP English)"
CHIP = {"short": "Checked: held-out LPP",
        "long": "held out; first measured Oct 22-23; LPP is one of TRIBE v2's own evaluation sets"}
FROZEN_LINE = "~5 volunteers' worth (at least 2). Group level."
LABELS = {"forecast": "TRIBE v2 forecast", "group": "49 public listeners (average)", "five": "5 public listeners (average)",
          "one": "1 public listener"}


def zline(rows) -> np.ndarray:
    """Mean over rows (already cleaned on the window), z-scored over the window with ddof 1."""
    m = np.asarray(rows, float).mean(0)
    return (m - m.mean()) / m.std(ddof=1)


def lines(listener_lang: dict, forecast_lang, five=FIVE, one=None, hp: float = 0.03) -> dict:
    """The rule's four lines from raw LANG parcel rows (parcels x volumes) on the full scan; cleaning on volumes J0..J1 as the proof."""
    def cl(A):
        return H.clean(A, hp, J0, J1)
    per = {s: zline(cl(listener_lang[s])) for s in sorted(listener_lang)}
    out = {"forecast": zline(cl(forecast_lang)), "group": zline(np.stack(list(per.values()))),
           "five": zline(np.stack([per[s] for s in five]))}
    if one is not None:
        out["one"] = per[one]
    return out


def r3(v) -> list:
    return [round(float(x), 3) for x in v]


def median_listener(ids, z) -> str:
    """The 25th of 49 in ascending frozen LANGUAGE z, ties broken by ID."""
    ranked = sorted(zip(z, ids), key=lambda t: (t[0], t[1]))
    return ranked[(len(ranked) - 1) // 2][1]


def rule_commit() -> tuple:
    dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--", RULE], capture_output=True, text=True).stdout.strip()
    log = subprocess.run(["git", "-C", REPO, "log", "-1", "--format=%h %cI", "--", RULE], capture_output=True, text=True).stdout.split()
    if dirty or not log:
        raise SystemExit("refused: the rule file has no commit or has uncommitted changes")
    return log[0], log[1]


def render(doc: dict, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t, L = np.array(doc["time_s"]), doc["lines"]
    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=150)
    ax.plot(t, L["forecast"]["values"], ls="--", color="0.55", lw=2.2, label=L["forecast"]["label"], zorder=1)
    ax.text(0.01, 0.97, doc["chip"]["short"], transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6"), zorder=5)
    ax.plot(t, L["group"]["values"], color="#1f6fb4", lw=2.0, label=L["group"]["label"], zorder=2)
    ax.plot(t, L["five"]["values"], color="#d9822b", lw=1.6, label=L["five"]["label"], zorder=3)
    ax.plot(t, L["one"]["values"], color="#2a9d55", lw=0.8, label=L["one"]["label"], zorder=4)
    ax.set_title(doc["title"])
    ax.set_xlabel("seconds of the section's scan")
    ax.set_ylabel("z")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.text(0.99, 0.97, doc["frozen_line"]["text"], transform=ax.transAxes, ha="right", va="top", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    t0 = time.time()
    now = datetime.datetime.now(datetime.timezone.utc)
    if now < datetime.datetime.fromisoformat(NOT_BEFORE.replace("Z", "+00:00")):
        raise SystemExit(f"refused: before {NOT_BEFORE}")
    commit, committed = rule_commit()
    assert not os.path.abspath(a.outdir).startswith(REPO), "OUTDIR must be outside the repository"
    R = a.root
    pl_path = f"{R}/d22/sec8/heldout_section8.json"
    if hashlib.sha256(open(pl_path, "rb").read()).hexdigest() != PER_LISTENER_SHA:
        raise SystemExit("refused: per-listener file SHA-256 differs")
    pl = json.load(open(pl_path))["stage"]["145_primary"]["per_listener"]
    one = median_listener(pl["listeners"], pl["language"])
    files = sorted(glob.glob(f"{R}/p1/cache_en8/sub-*.npz"))
    bold = {os.path.basename(f)[:-4]: np.load(f)["LANG"] for f in files}
    assert tuple(sorted(bold)) == ALL_IDS == tuple(sorted(pl["listeners"])) and set(FIVE) <= set(bold)
    n = bold[sorted(bold)[0]].shape[1]
    sets = H.parcel_sets(f"{R}/atlas/lh.Schaefer2018_400Parcels_17Networks_order.annot",
                         f"{R}/atlas/rh.Schaefer2018_400Parcels_17Networks_order.annot")
    assert len(sets["LANG"]) == bold[sorted(bold)[0]].shape[0]
    P = np.load(f"{R}/k2/out_en8/k2_out/pred_en8_full.npz")
    fc = H.pred_parcels(P["P"], P["starts"], n, sets)["LANG"]
    assert not np.isnan(fc[:, J0:J1 + 1]).any(), "forecast does not cover the window"
    ln = lines(bold, fc, FIVE, one)
    doc = {"format": "stillhere-ghostline/1", "label": "display only; no new statistic",
           "rule": {"path": RULE, "commit": commit, "committed": committed},
           "dataset": "LPP English section 8 (ds003643), 49 listeners of the frozen held-out proof",
           "title": TITLE, "region": "frozen proof's language parcels (heldout.parcel_sets LANG)", "n_parcels": len(sets["LANG"]),
           "volumes": [J0, J1], "tr_s": H.TR, "time_s": [round(H.TR * j, 1) for j in range(J0, J1 + 1)],
           "short_view": {"volumes": [J0, SHORT_J1], "index": [0, SHORT_J1 - J0 + 1], "note": "crop of the full-window lines"},
           "draw_order": ORDER, "chip": CHIP,
           "lines": {"forecast": {"label": LABELS["forecast"], "style": "gray dashed", "values": r3(ln["forecast"])},
                     "group": {"label": LABELS["group"], "style": "solid colour", "values": r3(ln["group"])},
                     "five": {"label": LABELS["five"], "style": "solid colour", "listeners": list(FIVE), "seed": FIVE_SEED,
                              "values": r3(ln["five"])},
                     "one": {"label": LABELS["one"], "style": "thin solid", "values": r3(ln["one"])}},
           "provenance": {"one_listener_id": one, "selection": "median frozen LANGUAGE z (145_primary), ties by ID",
                          "per_listener_sha256": PER_LISTENER_SHA, "forecast": "k2 pred_en8_full.npz (stock TRIBE v2)",
                          "alignment": "volume j reads stimulus time 2j - 5 s", "cleaning": "heldout.clean, 0.03 Hz, volumes 10-154"},
           "frozen_line": {"text": FROZEN_LINE, "source": "results/section8.json volunteer_equivalent_language (point 5.0, lower 2.58)",
                           "only_beside": "this LPP ghost line"},
           "extracted_utc": now.isoformat(timespec="seconds")}
    out = os.path.join(REPO, "results/ghostline_lpp_s8.json")
    json.dump(doc, open(out, "w"), indent=1)
    os.makedirs(a.outdir, exist_ok=True)
    render(doc, os.path.join(a.outdir, "ghostline_lpp_s8.png"))
    print(f"{len(bold)} listeners, {len(sets['LANG'])} parcels, {J1 - J0 + 1} volumes, 4 lines; rule {commit}; "
          f"wrote {out} ({os.path.getsize(out)} bytes) and the render; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
