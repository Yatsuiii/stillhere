"""Runs the exploratory Voice Transcribe checks (rules in stillhere/manipulation_check.py, committed before this ran): first the diarisation
validation on the known-turns two-reader clip, then one transcription per stimulus condition. All calls go through
meta_client.transcribe(mode="DIARIZATION") with typed, allowlisted public audio (5 calls). Writes results/diarization_validation.json and
results/conditions_manipulation_check.json. usage: python scripts/run_manipulation_check.py"""
import json
from pathlib import Path

from stillhere import egress, meta_client
from stillhere import manipulation_check as M

R = Path(__file__).resolve().parents[1]
A, OUT = R / "assets", R / "results"


def main():
    truth = json.loads((A / "diarization" / "two_readers_6turns.truth.json").read_text())
    d = meta_client.transcribe(egress.PublicAudio(path=str(A / "diarization" / truth["file"])), mode="DIARIZATION")
    turns = [{"speaker": t.get("speaker"), "startMs": t.get("startMs"), "endMs": t.get("endMs")} for t in d.get("turns") or []]
    dia = {"label": "exploratory", **M.diarization(turns, truth["turns"]), "turns": turns, "truth": truth["turns"],
           "audio_ms": d.get("audioDurationMs"), "gate_allowed": None}
    dia["gate_allowed"] = bool(dia["pass"])
    (OUT / "diarization_validation.json").write_text(json.dumps(dia, indent=1) + "\n")
    man = json.loads((A / "conditions" / "alice_ch01_75s_manifest.json").read_text()); exp = man["expected_words"]; res = {}
    for name in ("intact", "sentence_scramble", "word_scramble", "reversed"):
        out = meta_client.transcribe(egress.PublicAudio(path=str(A / "conditions" / man["files"][name]["file"])), mode="DIARIZATION")
        res[name] = {**M.check_condition(name, out.get("transcript", ""), exp.get(name, exp["intact"]), exp["intact"]),
                     "transcript": out.get("transcript", "")}
    (OUT / "conditions_manipulation_check.json").write_text(json.dumps(
        {"label": "exploratory", "source": man["source"], "seed": man["seed"], "conditions": res}, indent=1) + "\n")
    print("diarization:", {k: dia[k] for k in ("n_labels", "n_turns", "agreement", "speaker_changes", "true_speaker_changes", "pass")})
    for name, v in res.items():
        print(name, {k: (round(v[k], 3) if isinstance(v[k], float) else v[k]) for k in ("recall", "order_vs_original", "order_vs_expected", "n_words_heard", "pass")})


if __name__ == "__main__":
    main()
