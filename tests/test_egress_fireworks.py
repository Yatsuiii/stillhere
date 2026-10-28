"""Fireworks (Meta's open-weights Muse Spark Open) behind the typed wrapper: public text, scanned questions and allowlisted public film
frames only; never tier-D; key from FIREWORKS_API_KEY only, refused before any network call when unset and never logged; only the
allowlisted hosts; parallel_tool_calls switchable per request."""
import hashlib
import json

import httpx
import pytest

from stillhere import egress, meta_client



def public(text, tmp_path, monkeypatch):
    allow = tmp_path / "allow.txt"; allow.write_text(hashlib.sha256(text.encode()).hexdigest() + "\n")
    monkeypatch.setattr(egress, "ALLOWLIST", allow)
    return egress.PublicText(text=text)


SCHEMA = {"type": "object", "additionalProperties": False, "required": ["a"], "properties": {"a": {"type": "string", "enum": ["x", "y"]}}}


def test_tier_d_never_goes_to_fireworks(tmp_path, monkeypatch):
    sys_ = public("You plan studies.", tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        egress.Request(model="fireworks/muse-spark-open", system=[sys_], user=[egress.TierD(values={"p1_k": 9})], schema_name="s", json_schema=SCHEMA)
    egress.Request(model="muse-spark-1.3", system=[sys_], user=[egress.TierD(values={"p1_k": 9})], schema_name="s", json_schema=SCHEMA)


def test_fireworks_key_only_from_env_refused_before_network(tmp_path, monkeypatch):
    sys_ = public("You plan studies.", tmp_path, monkeypatch)
    req = egress.Request(model="fireworks/muse-spark-open", system=[sys_], user=[sys_], schema_name="s", json_schema=SCHEMA)
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.setattr(httpx, "post", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network reached")))
    with pytest.raises(egress.EgressRefused):
        meta_client.call(req)


def test_hosts_routing_parallel_switch_and_key_not_logged(tmp_path, monkeypatch):
    sys_ = public("You plan studies.", tmp_path, monkeypatch)
    assert {httpx.URL(meta_client.route(m)[1]).host for m in ("muse-spark-1.3", "fireworks/muse-spark-open")} <= meta_client.HOSTS
    seen = {}
    class R:
        def raise_for_status(self): pass
        def json(self): return {"usage": {"prompt_tokens": 1, "completion_tokens": 1},
                                "choices": [{"message": {"tool_calls": [{"function": {"name": "t", "arguments": "{}"}}]}}]}
    def post(url, json=None, timeout=None, headers=None):
        seen.update(url=url, body=json, headers=headers); return R()
    monkeypatch.setattr(httpx, "post", post); monkeypatch.setenv("FIREWORKS_API_KEY", "fw-test-secret-123")
    log = tmp_path / "usage.jsonl"; monkeypatch.setattr(meta_client, "USAGE_LOG", log)
    tool = egress.Tool(name="t", description=sys_, params={"a": ["x"]})
    meta_client.call(egress.Request(model="fireworks/muse-spark-open", system=[sys_], user=[sys_], tools=[tool], parallel_tool_calls=False))
    assert seen["url"].startswith("https://api.fireworks.ai/") and seen["body"]["parallel_tool_calls"] is False
    assert seen["body"]["model"] == "accounts/fireworks/models/muse-spark-open"
    assert "fw-test-secret-123" not in log.read_text() and json.loads(log.read_text())["provider"] == "fireworks"


def test_public_frames_only_when_allowlisted(tmp_path, monkeypatch):
    f = tmp_path / "frame.jpg"; f.write_bytes(b"\xff\xd8 public frame"); g = tmp_path / "other.jpg"; g.write_bytes(b"\xff\xd8 other")
    allow = tmp_path / "frames.txt"; allow.write_text(hashlib.sha256(f.read_bytes()).hexdigest() + "  test frame\n")
    monkeypatch.setattr(egress, "FRAME_ALLOWLIST", allow)
    fr = egress.PublicFrame(path=str(f))
    with pytest.raises(ValueError):
        egress.PublicFrame(path=str(g))
    sys_ = public("Is any human face recognisable? Answer yes or no.", tmp_path, monkeypatch)
    content = meta_client._user_content([sys_, fr])
    assert content[0]["type"] == "text" and content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_table_enums_only_validation_table_ids_and_allowed_on_fireworks(tmp_path, monkeypatch):
    sys_ = public("You plan studies.", tmp_path, monkeypatch)
    te = egress.TableEnums(enums={"population": ["healthy_adults"], "conditions": ["intact", "sentence_scramble"], "rejected": ["schema_invalid"]})
    egress.Request(model="fireworks/muse-spark-open", system=[sys_], user=[te], schema_name="s", json_schema=SCHEMA)
    for bad in ({"population": ["children_6y"]}, {"x": ["0.83"]}, {"x": ["sub-01"]}):
        with pytest.raises(ValueError):
            egress.TableEnums(enums=bad)
    assert json.loads(meta_client._render(te)) == {"enums": te.enums}
