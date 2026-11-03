"""scripts/rank_minutes.py (R5's ranking for a lab's own story), r2/run_story_forecast.py's speech-to-text batch option and
scripts/rank_minutes.sh's refusals, on synthetic inputs and stand-in pythons. The check against R5's own window scores needs the Paranoia
data (STILLHERE_PARANOIA_DIR: reproduce_paranoia.sh's DATA_DIR with ours_release/)."""
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def story(T=460, loud=None, seed=0):
    """A 4-vertex story: two language parcels; the full forecast is the audio-only one plus a part only the full one has, made large over
    story seconds `loud` (a range); words at 2 per second."""
    rng = np.random.default_rng(seed)
    k = np.convolve(rng.standard_normal((T + 20) * 4), np.ones(5) / 5, "same").reshape(T + 20, 4)[:T]
    Pa = k.astype("float32"); extra = rng.standard_normal((T, 4)).astype("float32") * 0.2
    if loud:
        extra[loud[0]:loud[1]] *= 8
    starts = np.arange(T, dtype=float)
    return {"full": (Pa + extra, starts), "audio": (Pa, starts), "onsets": np.arange(0, T, 0.5), "env": np.abs(rng.standard_normal(T + 60))}


LANG = [np.array([0, 1]), np.array([2, 3])]


@pytest.fixture(scope="module")
def RM():
    pytest.importorskip("nibabel"); pytest.importorskip("nilearn")
    return load("rank_minutes", "scripts/rank_minutes.py")


def test_the_window_whose_forecast_carries_most_beyond_sound_and_word_rate_ranks_first(RM):
    # volume j reads story time j - 7; window 3 covers volumes 200-259, so story seconds 193-252 of the forecast
    res = RM.rank({"s": story(loud=(193, 253))}, LANG)
    p = res["parts"]["s"]; w = p["windows"]
    assert p["window_volumes"][0] == 20 and len(w) == 7 and p["kept_each_end"] == 2 and res["minutes_each_side"] == 2
    assert next(x for x in w if x["rank"] == 1)["i"] == 3 and 3 in p["tribe_top"] and 3 not in p["tribe_bottom"]
    assert w[0]["story_s"] == [18, 78] and w[0]["story"] == "0:18-1:18" and w[3]["story"] == "3:18-4:18"     # R5's per-second mapping
    assert sorted(x["rank"] for x in w) == list(range(1, 8)) and all(abs(x["words_per_s"] - 2) < 1e-9 for x in w)
    assert "was not tested" in res["caveat"] and "Not a diagnosis" in res["caveat"] and res["lang_parcels"] == 2


def test_the_picks_are_the_k_best_and_k_worst_ranks_and_the_run_repeats_exactly(RM):
    s = story(seed=3)
    a, b = RM.rank({"s": s}, LANG), RM.rank({"s": s}, LANG)
    assert a == b
    p = a["parts"]["s"]; order = sorted(p["windows"], key=lambda x: x["rank"]); k = p["kept_each_end"]
    assert [x["pula"] for x in order] == sorted((x["pula"] for x in p["windows"]), reverse=True)
    assert p["tribe_top"] == sorted(x["i"] for x in order[:k]) and p["tribe_bottom"] == sorted(x["i"] for x in order[-k:])


def test_a_story_too_short_for_one_window_after_the_first_20_seconds_is_refused(RM):
    with pytest.raises(SystemExit, match="shorter than one 60-s window"):
        RM.rank({"s": story(T=60)}, LANG)


def test_parts_are_ranked_each_on_their_own_as_r5_ranks_paranoias_three(RM):
    res = RM.rank({"a": story(loud=(73, 133), seed=1), "b": story(T=300, loud=(13, 73), seed=2)}, LANG)
    assert list(res["parts"]) == ["a", "b"] and res["parts"]["b"]["windows"][0]["rank"] == 1
    assert next(x for x in res["parts"]["a"]["windows"] if x["rank"] == 1)["i"] == 1
    assert res["minutes_each_side"] == sum(len(p["tribe_top"]) for p in res["parts"].values())


@pytest.mark.skipif(not os.environ.get("STILLHERE_PARANOIA_DIR"), reason="needs the Paranoia data (STILLHERE_PARANOIA_DIR)")
def test_r5s_own_window_scores_and_picks_come_back_through_rank(RM):
    res = RM.check_r5(os.environ["STILLHERE_PARANOIA_DIR"])
    assert res["lang_parcels"] == 19 and all(res["same_picks"].values()) and res["pass"], res


def test_the_speech_to_text_batch_option_changes_only_whispers_batch(monkeypatch):
    RS = load("run_story_forecast", "r2/run_story_forecast.py")
    seen = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, *a, **k: seen.append(cmd))
    assert RS.asr_batch("4") == {"whisperx_batch_size": 4, "shipped_batch_size": 16}
    cmd = ["uvx", "whisperx", "stim.wav", "--model", "large-v3", "--batch_size", "16", "--output_dir", "o"]
    subprocess.run(cmd); subprocess.run(["uvx", "other", "--batch_size", "16"])
    assert seen[0] == ["uvx", "whisperx", "stim.wav", "--model", "large-v3", "--batch_size", "4", "--output_dir", "o"]
    assert seen[1] == ["uvx", "other", "--batch_size", "16"] and cmd[6] == "16"


def sh(*args, env=None):
    e = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/nonexistent"), **(env or {})}
    return subprocess.run(["bash", str(REPO / "scripts/rank_minutes.sh"), *map(str, args)], cwd=REPO, env=e, capture_output=True,
                          text=True, timeout=60)


@pytest.mark.skipif(shutil.which("bash") is None or shutil.which("realpath") is None, reason="needs bash and coreutils")
def test_the_launcher_refuses_before_any_download(tmp_path):
    wav = tmp_path / "s.wav"; wav.write_bytes(b"RIFF")
    p = sh(wav, REPO / "data_here")
    assert p.returncode == 2 and "outside this repository" in p.stderr and not (REPO / "data_here").exists()
    bad = tmp_path / "py"; bad.write_text("#!/usr/bin/env bash\nexit 1\n"); bad.chmod(0o755)
    p = sh(wav, tmp_path / "data", env={"STILLHERE_SCORE_PYTHON": str(bad)})
    assert p.returncode == 2 and "requirements-score.txt" in p.stderr and not (tmp_path / "data").exists()
    p = sh(tmp_path / "missing.wav", tmp_path / "data")
    assert p.returncode != 0 and not (tmp_path / "data").exists()
    p = sh(wav)
    assert p.returncode == 2 and "usage" in p.stderr
