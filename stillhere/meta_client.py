"""Meta model client (Chat Completions with tools or strict json_schema output). Accepts only an egress.Request; builds the message
text, tool definitions and response format from its typed parts alone. Three providers, all serving Meta models: Meta's Model API
(muse-spark-1.x; key from MODEL_API_KEY or Muse Code's credential store), Fireworks serverless for Meta's open-weights Muse Spark Open
(egress id "fireworks/muse-spark-open"; key from the FIREWORKS_API_KEY environment variable only) and Baseten's Model API for the
hackathon's Muse Spark Open (egress id "baseten/muse-spark-open-1026"; key from BASETEN_API_KEY only). Keys are read at call time, never
logged or saved. Only the hosts in HOSTS are ever contacted.
Each call appends one usage line (time, provider, model, schema name, token counts, latency) to a local log; no request or response text."""
import base64
import json
import os
import subprocess
import time
from pathlib import Path

import httpx

from . import egress

API = "https://api.meta.ai/v1"
USAGE_LOG = Path(os.environ.get("STILLHERE_USAGE_LOG", Path.home() / ".local/state/stillhere/meta_usage.jsonl"))


def _key() -> str:
    k = os.environ.get("MODEL_API_KEY")
    if k:
        return k
    auth = Path.home() / ".config/muse/auth.json"
    return json.loads(auth.read_text())["providers"]["meta"]["api_key"]


def _fireworks_key() -> str:
    k = os.environ.get("FIREWORKS_API_KEY")
    if not k:
        raise egress.EgressRefused("FIREWORKS_API_KEY is not set: Fireworks calls are refused")
    return k


def _baseten_key() -> str:
    k = os.environ.get("BASETEN_API_KEY")
    if not k:
        raise egress.EgressRefused("BASETEN_API_KEY is not set: Baseten calls are refused")
    return k


# egress model id -> (provider, base URL, the provider's model id, key reader); Meta model ids not listed here go to Meta's API.
PROVIDERS = {"fireworks/muse-spark-open": ("fireworks", "https://api.fireworks.ai/inference/v1", "accounts/fireworks/models/muse-spark-open",
                                           _fireworks_key),
             "baseten/muse-spark-open-1026": ("baseten", "https://inference.baseten.co/v1", "meta-models/muse-spark-open-1026", _baseten_key)}


HOSTS = frozenset({"api.meta.ai", "api.fireworks.ai", "inference.baseten.co"})   # the only hosts a model call may reach


def route(model: str) -> tuple:
    r = PROVIDERS.get(model, ("meta", API, model, _key))
    if httpx.URL(r[1]).host not in HOSTS:
        raise egress.EgressRefused(f"host not on the allowlist: {httpx.URL(r[1]).host}")
    return r


def _render(part) -> str:
    if isinstance(part, egress.TierD):
        return json.dumps({"values": part.values, "enums": part.enums}, sort_keys=True)
    if isinstance(part, egress.TableEnums):
        return json.dumps({"enums": part.enums}, sort_keys=True)
    return part.text                                      # PublicText or PublicQuestion, both already checked


def _user_content(parts):
    """Plain text when there is no frame; otherwise OpenAI-style content parts, each frame as a base64 data URI."""
    if not any(isinstance(p, egress.PublicFrame) for p in parts):
        return "\n\n".join(_render(p) for p in parts)
    out = []
    for p in parts:
        if isinstance(p, egress.PublicFrame):
            mime = "image/png" if p.path.lower().endswith(".png") else "image/jpeg"
            out.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64," + base64.b64encode(Path(p.path).read_bytes()).decode()}})
        else:
            out.append({"type": "text", "text": _render(p)})
    return out


def _tool(t: egress.Tool) -> dict:
    params = {"type": "object", "additionalProperties": False, "required": list(t.params),
              "properties": {k: {"type": "string", "enum": v} for k, v in t.params.items()}}
    return {"type": "function", "function": {"name": t.name, "description": t.description.text, "parameters": params}}


def _args(a) -> dict:
    return json.loads(a) if isinstance(a, str) else (a or {})


def call(req: egress.Request, timeout: float = 60.0, effort: str | None = None, temperature: float | None = None) -> dict:
    """One API call. A request with tools returns {"tool_calls": [{"name", "arguments"}]}; a json_schema request returns
    {"content": <parsed answer>}. temperature is sent only when given (otherwise the provider's default applies)."""
    egress.check(req)
    provider, base, model_id, key = route(req.model)
    body = {"model": model_id,
            "messages": [{"role": "system", "content": "\n\n".join(p.text for p in req.system)},
                         {"role": "user", "content": _user_content(req.user)}]}
    if req.tools:
        body["tools"] = [_tool(t) for t in req.tools]
        if not req.parallel_tool_calls:
            body["parallel_tool_calls"] = False
    else:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": req.schema_name, "schema": req.json_schema, "strict": True}}
    if temperature is not None:
        body["temperature"] = float(temperature)
    if effort:
        body["reasoning_effort"] = effort          # only when asked; the parameter name is not confirmed in the docs yet
    t0 = time.time()
    r = httpx.post(f"{base}/chat/completions", json=body, timeout=timeout, headers={"Authorization": f"Bearer {key()}"})
    r.raise_for_status()
    data = r.json(); u = data.get("usage", {}); msg = data["choices"][0]["message"]
    out = ({"tool_calls": [{"name": c["function"]["name"], "arguments": _args(c["function"].get("arguments"))} for c in msg.get("tool_calls") or []]}
           if req.tools else {"content": json.loads(msg["content"])})
    USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with USAGE_LOG.open("a") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "provider": provider, "model": req.model,
                            "schema": req.schema_name or "tools", "temperature": temperature,
                            "tool_calls": len(out.get("tool_calls", [])), "prompt_tokens": u.get("prompt_tokens"),
                            "completion_tokens": u.get("completion_tokens"), "seconds": round(time.time() - t0, 2)}) + "\n")
    return out


def chat(req: egress.Request, timeout: float = 60.0, effort: str | None = None) -> dict:
    return call(req, timeout, effort)["content"]


ASR_URL = f"{API}/asr/transcribe"
ASR_MODEL = "muse-voice-transcribe-1.0"                      # the Standard model id; a -contributor id is never used
ASR_MODES = frozenset({"DIARIZATION"})


def transcribe(audio: egress.PublicAudio, timeout: float = 300.0, mode: str | None = None):
    """One Voice Transcribe call on an allowlisted public clip; logs time, model, mode, audio size and latency, never the transcript.
    mode=None: Muse Code's `muse voice transcribe` with contributor mode off; returns the plain text.
    mode="DIARIZATION": the batch endpoint POST /v1/asr/transcribe with muse-voice-transcribe-1.0; returns its JSON (transcript,
    audioDurationMs, turns with speaker, startMs, endMs, transcript)."""
    if not isinstance(audio, egress.PublicAudio):
        raise egress.EgressRefused(f"only egress.PublicAudio may be transcribed, got {type(audio).__name__}")
    if mode is not None and mode not in ASR_MODES:
        raise egress.EgressRefused(f"transcription mode not allowed: {mode}")
    t0 = time.time(); extra = {}
    if mode is None:
        r = subprocess.run(["muse", "voice", "transcribe", audio.path], capture_output=True, text=True, timeout=timeout, check=True,
                           env={**os.environ, "MUSE_CONTRIBUTOR_MODE": "0"})
        out = r.stdout
    else:
        req = json.dumps({"mode": mode, "model": ASR_MODEL, "audioEncoding": "WAV"})
        r = httpx.post(ASR_URL, params={"sessionId": "stillhere-pilot"}, timeout=timeout, headers={"Authorization": f"Bearer {_key()}"},
                       files={"request": (None, req, "application/json"), "audio": (Path(audio.path).name, Path(audio.path).read_bytes(), "audio/wav")})
        r.raise_for_status(); out = r.json(); extra = {"mode": mode, "audio_ms": out.get("audioDurationMs"), "turns": len(out.get("turns") or [])}
    USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with USAGE_LOG.open("a") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model": ASR_MODEL, "schema": "transcribe",
                            "audio_bytes": Path(audio.path).stat().st_size, **extra, "seconds": round(time.time() - t0, 2)}) + "\n")
    return out
