"""Baseten (the hackathon's Muse Spark Open, model id meta-models/muse-spark-open-1026) behind the typed wrapper: the third allowed host;
public text, scanned questions and allowlisted public film frames only, never tier-D; key from BASETEN_API_KEY only, refused before any
network call when unset and never logged."""
import hashlib
import json

import httpx
import pytest

from stillhere import egress, meta_client

MODEL = "baseten/muse-spark-open-1026"
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["a"], "properties": {"a": {"type": "string", "enum": ["x", "y"]}}}


def public(text, tmp_path, monkeypatch):
    allow = tmp_path / "allow.txt"; allow.write_text(hashlib.sha256(text.encode()).hexdigest() + "\n")
    monkeypatch.setattr(egress, "ALLOWLIST", allow)
    return egress.PublicText(text=text)


def test_baseten_is_an_allowed_host_and_model():
    provider, base, model_id, _ = meta_client.route(MODEL)
    assert provider == "baseten" and httpx.URL(base).host == "inference.baseten.co" in meta_client.HOSTS
    assert model_id == "meta-models/muse-spark-open-1026" and MODEL in egress.HOSTED_MODELS
    assert meta_client.HOSTS == {"api.meta.ai", "api.fireworks.ai", "inference.baseten.co"}


def test_tier_d_never_goes_to_baseten(tmp_path, monkeypatch):
    sys_ = public("You plan studies.", tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        egress.Request(model=MODEL, system=[sys_], user=[egress.TierD(values={"p1_k": 9})], schema_name="s", json_schema=SCHEMA)


def test_baseten_key_only_from_env_refused_before_network(tmp_path, monkeypatch):
    sys_ = public("You plan studies.", tmp_path, monkeypatch)
    req = egress.Request(model=MODEL, system=[sys_], user=[sys_], schema_name="s", json_schema=SCHEMA)
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    monkeypatch.setattr(httpx, "post", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network reached")))
    with pytest.raises(egress.EgressRefused):
        meta_client.call(req)


def test_baseten_call_uses_its_model_id_and_never_logs_the_key(tmp_path, monkeypatch):
    sys_ = public("You plan studies.", tmp_path, monkeypatch)
    req = egress.Request(model=MODEL, system=[sys_], user=[sys_], schema_name="s", json_schema=SCHEMA)
    monkeypatch.setenv("BASETEN_API_KEY", "bt-secret-test-key")
    log = tmp_path / "usage.jsonl"; monkeypatch.setattr(meta_client, "USAGE_LOG", log)
    seen = {}

    class R:
        def raise_for_status(self): pass
        def json(self): return {"usage": {"prompt_tokens": 3, "completion_tokens": 2}, "choices": [{"message": {"content": json.dumps({"a": "x"})}}]}

    def post(url, json=None, timeout=None, headers=None):
        seen.update(url=url, body=json, auth=headers["Authorization"]); return R()
    monkeypatch.setattr(httpx, "post", post)
    assert meta_client.call(req) == {"content": {"a": "x"}}
    assert seen["url"] == "https://inference.baseten.co/v1/chat/completions" and seen["body"]["model"] == "meta-models/muse-spark-open-1026"
    assert seen["auth"] == "Bearer bt-secret-test-key"
    line = log.read_text()
    assert "bt-secret-test-key" not in line and json.loads(line)["provider"] == "baseten"
