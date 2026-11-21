"""The failure injections of PREREG.md section 2, written before the scorer. Each builds a forecast whose right answer is known and
checks the scorer gives it. Synthetic data only."""
import json

import numpy as np
import pytest

import veq

KW = dict(tr=1.5, ks=(1, 2, 3, 4, 6, 8, 11), reps=10, min_shift=15, seed=3, nboot=2000)


def run(world, pred, **kw):
    return veq.score(world["bold"], pred, baselines=world["baselines"], **{**KW, **kw})


def test_f1_null_forecast(world):
    noise = np.random.default_rng(1).standard_normal((world["p"], world["T"]))
    r = run(world, noise)
    assert abs(r["model"]["z_mean"]) < 1
    assert r["volunteer_equivalent"]["group"]["claim_k"] == 0
    assert r["verdict"]["beats"] == []


def test_f2_word_rate_in_disguise(world):
    rng = np.random.default_rng(2)
    pred = np.tile(world["W"], (world["p"], 1)) + 0.1 * rng.standard_normal((world["p"], world["T"]))
    r = run(world, pred)
    assert not r["baselines"]["word_rate"]["beaten"]
    zb = np.array([s["z_beyond_baselines"] for s in r["per_subject"]])
    assert (np.abs(zb) < 1.645).mean() > 0.5


def test_f3_real_signal(world):
    rng = np.random.default_rng(3)
    pred = world["S"] + 0.5 * rng.standard_normal(world["S"].shape)
    r = run(world, pred)
    assert set(r["verdict"]["beats"]) == {"word_rate", "envelope"}
    assert r["verdict"]["beats_all_baselines"]
    zb = np.array([s["z_beyond_baselines"] for s in r["per_subject"]])
    assert (zb > 1.645).mean() > 0.5


def test_f4_cleaner_forecast_worth_more(world):
    rng = np.random.default_rng(4)
    noise = rng.standard_normal(world["S"].shape)
    lo = run(world, world["S"] + 3.0 * noise)["volunteer_equivalent"]["group"]["point_k"]
    hi = run(world, world["S"] + 0.3 * noise)["volunteer_equivalent"]["group"]["point_k"]
    assert hi >= lo
    assert hi > 0


def test_f5_misaligned_forecast_fails(world):
    rng = np.random.default_rng(5)
    pred = np.roll(world["S"] + 0.5 * rng.standard_normal(world["S"].shape), 20, axis=1)
    r = run(world, pred)
    assert abs(r["model"]["z_mean"]) < 1.5


def test_f6_leak_hits_ceiling_and_is_flagged(world):
    leaked = "sub-00"
    r = run(world, world["bold"][leaked])
    row = next(s for s in r["per_subject"] if s["subject"] == leaked)
    assert row["at_ceiling"]
    assert leaked in r["flags"]["subjects_at_ceiling"]


def test_f7_scale_and_offset_change_nothing(world):
    rng = np.random.default_rng(6)
    pred = world["S"] + 0.5 * rng.standard_normal(world["S"].shape)
    a = run(world, pred)
    b = run(world, 10 * pred + 4.0)
    assert np.allclose([s["z_model"] for s in a["per_subject"]], [s["z_model"] for s in b["per_subject"]], atol=1e-9)
    assert np.isclose(a["volunteer_equivalent"]["group"]["point_k"], b["volunteer_equivalent"]["group"]["point_k"], atol=1e-9)


def test_f8_bad_inputs_raise(world):
    pred = world["S"]
    with pytest.raises(ValueError):
        veq.score(world["bold"], pred[:, :-1], **KW)
    nan = pred.copy()
    nan[0, 5] = np.nan
    with pytest.raises(ValueError):
        veq.score(world["bold"], nan, **KW)
    two = dict(list(world["bold"].items())[:2])
    with pytest.raises(ValueError):
        veq.score(two, pred, **KW)


def test_f9_deterministic(world):
    rng = np.random.default_rng(9)
    pred = world["S"] + rng.standard_normal(world["S"].shape)
    a = json.dumps(run(world, pred), sort_keys=True)
    b = json.dumps(run(world, pred), sort_keys=True)
    assert a == b
