"""R1's forecast provenance (dated note of Oct 31 on docs/rules/alice_finetuned_rule.md; cloud ruling of 21:30 UTC, item 11): the three
fine-tuned Alice stage records as the runner wrote them (counts, hashes and placement only: no brain data, no word text), each with its
UTC start and end from the launcher's log, the SHA-256 of the runner, the stock runner it imports and tribe_common.py, and an assert that
no Llama placement was used (every Llama feature was read from the stock runs' cache, so Llama was never loaded).
usage: python scripts/alice_finetuned_provenance.py --root ../stillhere   (writes results/alice_finetuned_provenance.json)"""
import argparse
import hashlib
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODES = ("audio", "full", "shuf")
PINS = {"full": "dbdf2ee7558f195d3d1488791d1ac3c5c0ebb4faee798357a170ff8d682a7194", "audio": "5593dd410b871671c8046e2af620d76333dfb02d408706ecaa4bf0cf8b24ec2e",
        "shuf": "fdbfa5b03f27cbf34eaeb57f4864936b295c65a848a3781eab5b1e68745322ec"}            # scripts/alice_finetuned.py FORECAST_PINS
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()


def main(root):
    F = f"{root}/alice/forecast"; log = open(f"{F}/run_tuned.log").read(); out = {"stages": {}}
    for m in MODES:
        st = json.load(open(f"{F}/out_tuned/stage_alice_{m}.json"))
        text = json.dumps(st)
        assert "per_listener" not in text and "bold" not in text.lower(), "a stage record holds brain data"
        assert set(st["placement"]) <= {"audio_extractor", "brain_model", "text_features"}, "a Llama placement was recorded"
        if m != "audio":
            assert st["placement"]["text_features"] == "read from the infra cache", "Llama features were computed, not read"
        assert st["pred_sha256"] == PINS[m] == sha(f"{F}/out_tuned/pred_alice_{m}.npz"), "forecast differs from the pin"
        t0 = re.search(rf"^== alice_{m} (\S+Z)$", log, re.M).group(1); t1 = re.search(rf"^== alice_{m} rc 0 (\S+Z)$", log, re.M).group(1)
        out["stages"][m] = {"start_utc": t0, "end_utc": t1, "record_sha256": sha(f"{F}/out_tuned/stage_alice_{m}.json"), "record": st}
    out["code_sha256"] = {"alice/forecast/run_alice_forecast_tuned.py": sha(f"{F}/run_alice_forecast_tuned.py"),
                          "alice/forecast/run_alice_forecast.py": sha(f"{F}/run_alice_forecast.py"),
                          "k2/tribe_common.py (analysis tree)": sha(f"{root}/k2/tribe_common.py"),
                          "alice/forecast/run_all_tuned.sh (analysis tree)": sha(f"{F}/run_all_tuned.sh")}
    for rel in ("alice/forecast/run_alice_forecast_tuned.py", "alice/forecast/run_alice_forecast.py"):
        assert sha(f"{REPO}/{rel}") == out["code_sha256"][rel], f"{rel}: the repository's copy differs from the one that ran"
    out["llama_placement_used"] = False
    json.dump(out, open(f"{REPO}/results/alice_finetuned_provenance.json", "w"), indent=1)
    print({m: (v["start_utc"], v["end_utc"]) for m, v in out["stages"].items()}, {k: v[:12] for k, v in out["code_sha256"].items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); main(ap.parse_args().root)
