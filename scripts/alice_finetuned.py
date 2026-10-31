"""R1 scoring (docs/rules/alice_finetuned_rule.md; exploratory): the fine-tuned TRIBE v2 forecasts against the Alice Datasets
listeners, scored by scripts/alice_heldout.py's score step, called unchanged.

alice_heldout.score reads ROOT/alice/forecast/out/pred_alice_{full,audio,shuf}.npz and writes results/alice_heldout.json. This script
gives it an overlay root, outside the repository: links to the same atlas, BOLD, events, stimulus, annotations and surprisal, with
alice/forecast/out linked to the fine-tuned forecasts. It replaces only the writer, so the stock result is never overwritten and every
number comes from the same code. It refuses to score unless the three forecasts match the SHA-256 pinned in FORECAST_PINS (the rule's
second commit, before scoring). After scoring it adds the rule's descriptive paired comparison with the stock run's per-listener
scores (read from the stock run's output file; stock is not scored again).

Writes OUT/alice_finetuned.json (per-listener z, so OUT is outside this repo) and results/alice_finetuned.json (group numbers and lines).
usage: python scripts/alice_finetuned.py --root ../stillhere [--out ../stillhere/alice/score_finetuned]"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alice_heldout as AH  # noqa: E402

RULE = "docs/rules/alice_finetuned_rule.md"
MODEL = {"name": "fine-tuned TRIBE v2", "weights_sha256": "96923860eb47ee8e8f65e597f2fc216992470852dc07af95112d80d32746421e",
         "trained_on": "LPP sections 2-7 and 9 (33 English and 19 French listeners) and Narratives; never the Alice Datasets",
         "frozen": "2026-10-26 (D3)"}
PREFIX = "Fine-tuned TRIBE v2: "
MODES = ("full", "audio", "shuf")
FORECAST_PINS = {"full": "dbdf2ee7558f195d3d1488791d1ac3c5c0ebb4faee798357a170ff8d682a7194",   # the rule's second commit, after the
                 "audio": "5593dd410b871671c8046e2af620d76333dfb02d408706ecaa4bf0cf8b24ec2e",  # forecasts and before scoring
                 "shuf": "fdbfa5b03f27cbf34eaeb57f4864936b295c65a848a3781eab5b1e68745322ec"}
STOCK_PINS = {"full": "0a47d825a521f643fc5b298942fde7049679b070381bc97330ffd11a25d30ae3",     # docs/rules/alice_heldout_rule.md
              "audio": "d747d74f16231b249e1288da14f48ed0b57e434bd356edfb2836cdc7f166859b",    # section 3: the stock run's forecasts
              "shuf": "7c4ba627bc0b6d801e263fa17d887f18a4028dec84fbec9850c750a4d66f41bf"}
LINKS = ("atlas", "alice/bold", "alice/stimulus", "alice/annotations", "alice/surprisal.json")


def overlay(root, pred_dir, ovl) -> str:
    """ovl with the scorer's inputs linked from root and alice/forecast/out linked to pred_dir; returns ovl. Existing links must agree."""
    want = {rel: os.path.abspath(os.path.join(root, rel)) for rel in LINKS}
    want["alice/forecast/out"] = os.path.abspath(pred_dir)
    for rel, src in want.items():
        assert os.path.exists(src), f"missing input: {src}"
        dst = os.path.join(ovl, rel); os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.islink(dst):
            assert os.path.realpath(dst) == os.path.realpath(src), f"{dst} links elsewhere"
        else:
            os.symlink(src, dst)
    return ovl


def check_pins(pred_dir, pins) -> dict:
    if not pins:
        raise SystemExit("refused: the fine-tuned forecasts' SHA-256 are not pinned yet (the rule's second commit comes before scoring)")
    got = {k: AH.sha(os.path.join(pred_dir, f"pred_alice_{k}.npz")) for k in MODES}
    bad = [k for k in MODES if got[k] != pins.get(k)]
    if bad:
        raise SystemExit(f"refused: forecast(s) {bad} differ from the pinned SHA-256")
    return got


def stock_scores(path, published=None) -> dict:
    """The stock run's output file, only if it is the corrected run of record: its forecast pins are the stock rule's (both stock runs
    share them, so the pins alone cannot tell the first run's file apart) and its claim-window group LANGUAGE and HEARING z equal
    results/alice_heldout.json's (dated note of Oct 31, cloud ruling of 21:30 UTC, item 9). Returns the claim-window stage."""
    d = json.load(open(path))
    got = {k: (d.get("inputs_sha256") or {}).get(f"pred_{k}") for k in MODES}
    if got != STOCK_PINS:
        raise SystemExit(f"refused: {path} is not the stock run on the pinned stock forecasts")
    pub = json.load(open(published or f"{AH.REPO}/results/alice_heldout.json"))["stage"]["145_primary"]
    st = d["stage"]["145_primary"]
    if any((st.get(a) or {}).get("z") != pub[a]["z"] for a in ("language", "hearing")):
        raise SystemExit(f"refused: {path} is not the stock run of record (its claim-window group z differ from results/alice_heldout.json)")
    return st


def paired(stock: dict, tuned: dict) -> dict:
    """The rule's descriptive comparison (section 5): per listener, fine-tuned minus stock, on the listeners both runs scored, for
    LANGUAGE and HEARING at the claim window and the development lag. Two-sided Wilcoxon; it decides nothing."""
    from scipy import stats
    out = {}
    for area, subs_key in (("language", "listeners"), ("hearing", "hearing_listeners")):
        s = dict(zip(stock["per_listener"][subs_key], stock["per_listener"][area]))
        t = dict(zip(tuned["per_listener"][subs_key], tuned["per_listener"][area]))
        both = sorted(set(s) & set(t))
        d = np.array([t[k] - s[k] for k in both])
        out[area] = {"n": len(both), "stock_mean_z": float(np.mean([s[k] for k in both])),
                     "finetuned_mean_z": float(np.mean([t[k] for k in both])), "mean_difference": float(d.mean()),
                     "finetuned_higher": int((d > 0).sum()),
                     "p_two_sided": float(stats.wilcoxon(d, alternative="two-sided").pvalue) if np.any(d != 0) else 1.0}
    return out


def make_writer(stock_full_path):
    def write(pub, full, out_dir):
        lines = [PREFIX + line for line in pub["lines"]]
        for d in (pub, full):
            d["rule"] = RULE; d["model"] = MODEL; d["lines"] = lines
        if "145_primary" in full.get("stage", {}):
            stock = stock_scores(stock_full_path)
            cmp = {"label": "descriptive, decides nothing", "window": "145_primary",
                   **paired(stock, full["stage"]["145_primary"])}
            pub["vs_stock"] = full["vs_stock"] = cmp
        json.dump(full, open(f"{out_dir}/alice_finetuned.json", "w"), indent=1)
        json.dump(pub, open(f"{AH.REPO}/results/alice_finetuned.json", "w"), indent=1)
        for line in lines:
            print(line, flush=True)
        return pub
    return write


def run(root, out_dir, pred_dir=None, ovl=None, pins=None, stock_full=None):
    pred_dir = pred_dir or f"{root}/alice/forecast/out_tuned"
    ovl = ovl or f"{root}/alice/finetuned_overlay"
    stock_full = stock_full or f"{root}/alice/score/alice_heldout.json"
    assert not os.path.abspath(ovl).startswith(AH.REPO), "the overlay must be outside the repo"
    check_pins(pred_dir, FORECAST_PINS if pins is None else pins)
    stock_scores(stock_full)                                  # refuse before scoring, not after
    overlay(root, pred_dir, ovl)
    AH._write = make_writer(stock_full)
    return AH.score(ovl, out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.root, a.out or f"{a.root}/alice/score_finetuned")
