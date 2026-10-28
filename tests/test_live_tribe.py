"""Live laptop TRIBE run: refusals (unlisted audio, a bare path, a clip over the limit, a non-stock checkpoint) and the output schema
with TRIBE stubbed. No model, no GPU, no network."""
import hashlib
import json
import wave

import numpy as np
import pytest

from stillhere import egress, live_tribe


def _wav(path, seconds, rate=16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(np.zeros(int(seconds * rate), dtype=np.int16).tobytes())
    return path


@pytest.fixture
def listed(tmp_path, monkeypatch):
    clip = _wav(tmp_path / "public.wav", 20)
    long_clip = _wav(tmp_path / "long.wav", 30)
    allow = tmp_path / "allow.txt"
    allow.write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  test\n" for p in (clip, long_clip)))
    monkeypatch.setattr(egress, "AUDIO_ALLOWLIST", allow)
    return clip, long_clip


def _stub(P):
    return lambda wav, cache: (P, {"ckpt_sha256": live_tribe.STOCK_CKPT_SHA256, "device": "stub", "peak_vram_gb": None,
                                   "load_s": 0.0, "predict_s": 0.0})


def test_refuses_unlisted_audio_a_bare_path_and_a_long_clip(listed, tmp_path):
    clip, long_clip = listed
    other = _wav(tmp_path / "other.wav", 5)
    with pytest.raises(ValueError):                          # pydantic wraps EgressRefused (a ValueError)
        egress.PublicAudio(path=str(other))
    with pytest.raises(egress.EgressRefused):
        live_tribe.run(str(clip), tmp_path, tmp_path, predict=_stub(np.zeros((20, live_tribe.N_VERTICES))), parcels=[np.arange(3)])
    with pytest.raises(egress.EgressRefused):
        live_tribe.run(egress.PublicAudio(path=str(long_clip)), tmp_path, tmp_path,
                       predict=_stub(np.zeros((30, live_tribe.N_VERTICES))), parcels=[np.arange(3)])


def test_refuses_a_non_stock_checkpoint(tmp_path):
    fake = tmp_path / "best.ckpt"; fake.write_bytes(b"not the stock weights")
    with pytest.raises(egress.EgressRefused):
        live_tribe.verify_ckpt(fake)


def test_output_schema_and_readout_with_tribe_stubbed(listed, tmp_path):
    clip, _ = listed
    rng = np.random.default_rng(0); P = rng.standard_normal((20, live_tribe.N_VERTICES)).astype(np.float32)
    parcels = [np.arange(0, 50), np.arange(10242, 10300)]
    r = live_tribe.run(egress.PublicAudio(path=str(clip)), tmp_path, tmp_path, predict=_stub(P), parcels=parcels)
    want = float(np.mean([P[:, v].mean(1).std() for v in parcels]))
    assert r["predicted_aud_response_strength"] == round(want, 6) and r["n_aud_parcels"] == 2
    assert r["seconds_predicted"] == 20 and r["clip_seconds"] == 20.0 and r["label"] == live_tribe.LABEL
    assert "never" not in r["label"] and "nothing about language or understanding" in r["wording"]
    assert r["ckpt_sha256"] == live_tribe.STOCK_CKPT_SHA256 and r["audio_sha256"] == hashlib.sha256(clip.read_bytes()).hexdigest()
    json.dumps(r)                                            # tier-D numbers and strings only: serialisable as is


def test_bad_prediction_shape_is_refused(listed, tmp_path):
    clip, _ = listed
    with pytest.raises(ValueError):
        live_tribe.run(egress.PublicAudio(path=str(clip)), tmp_path, tmp_path, predict=_stub(np.zeros((20, 100))), parcels=[np.arange(3)])
