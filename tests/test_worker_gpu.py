"""Opt-in (STILLHERE_GPU_TESTS=1, in the TRIBE environment on the laptop GPU): a live run reproduces the cached forecast for the same
recipe, including a run that follows another job in the same process (the case a per-job cache purge once broke). Tolerance from the
repeat measurements in docs/worker.md: run-to-run max abs difference was at most 0.0004 in raw units (region SD about 0.12 to 0.15)."""
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
pytestmark = pytest.mark.skipif(os.environ.get("STILLHERE_GPU_TESTS") != "1", reason="needs the laptop GPU and the TRIBE environment")
TOL_MAXABS, TOL_R = 0.002, 0.9999


def test_live_runs_reproduce_the_cache(tmp_path):
    import numpy as np

    from worker import recipe as R
    from worker.engine import Engine
    idx = R.load_index(); cache = json.loads((REPO / "data/forecast_cache.json").read_text())["entries"]
    e = Engine(tmp_path / "feature_cache").load()
    for rec in ({"clip_id": "alice_ch01", "condition": "word_scramble", "seed": idx["default_seed"]},
                {"clip_id": "alice_ch02", "condition": "intact", "seed": 0}):
        r = e.run(R.build_condition(rec, idx), tmp_path / "job", lambda *a: None)
        c = cache[R.cache_key(rec, idx)]
        assert r["n_words_text_lane"] == c["n_words_text_lane"]
        for k in ("language", "hearing"):
            x, y = np.array(r["regions"][k]), np.array(c["regions"][k])
            assert np.max(np.abs(x - y)) <= TOL_MAXABS, (rec, k, float(np.max(np.abs(x - y))))
            assert np.corrcoef(x, y)[0, 1] >= TOL_R, (rec, k)
