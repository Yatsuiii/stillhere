"""Spend caps and per-visitor limits, kept in the store so they survive restarts and hold across serverless instances.

Spend: every paid call first reserves its worst-case price (INCRBYFLOAT returns the new total atomically, so two calls cannot both
slip under a cap), then settles to the price it really cost. A provider stops for good at its total cap (Meta $80, Fireworks $50,
Baseten $50) and for the day at its daily cap (Meta $20, Fireworks $12, Baseten $12): provider-side caps may not exist. Per visitor session: at most RUNS_PER_HOUR runs in
any rolling hour, and each run may use each live station once (BUILD: once per allowed seed)."""
import datetime
import json
import os
import secrets
import time

from web.store import get_store

TOTAL_CAP = {"meta": float(os.environ.get("CAP_META_USD", 80)), "fireworks": float(os.environ.get("CAP_FIREWORKS_USD", 50)),
             "baseten": float(os.environ.get("CAP_BASETEN_USD", 50))}                   # cloud plan of Oct 29, 19:20 UTC, 1c
DAILY_CAP = {"meta": float(os.environ.get("DAILY_CAP_META_USD", 20)), "fireworks": float(os.environ.get("DAILY_CAP_FIREWORKS_USD", 12)),
             "baseten": float(os.environ.get("DAILY_CAP_BASETEN_USD", 12))}
RUNS_PER_HOUR = 5
RUN_TTL_S = 2 * 3600
STATIONS = ("plan", "build", "audit", "forecast", "film")


class Refused(Exception):
    """A live call that must not happen; `code` says why and the page shows the recorded run instead."""

    def __init__(self, code: str):
        super().__init__(code); self.code = code


def _day() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


def reserve(provider: str, usd: float) -> dict:
    if provider not in TOTAL_CAP:
        raise Refused("provider_not_offered")
    s, day = get_store(), _day()
    tot = s.incrbyfloat(f"spend:{provider}:total", usd)
    d = s.incrbyfloat(f"spend:{provider}:{day}", usd, ttl=3 * 86400)
    if tot > TOTAL_CAP[provider] or d > DAILY_CAP[provider]:
        s.incrbyfloat(f"spend:{provider}:total", -usd); s.incrbyfloat(f"spend:{provider}:{day}", -usd)
        raise Refused("spend_cap")
    return {"provider": provider, "reserved": usd, "day": day}


def settle(r: dict, actual_usd: float) -> None:
    """Replace the reservation by the real price (never below zero; an unknown price keeps the reservation)."""
    if actual_usd is None:
        return
    delta = max(actual_usd, 0.0) - r["reserved"]; s = get_store()
    s.incrbyfloat(f"spend:{r['provider']}:total", delta); s.incrbyfloat(f"spend:{r['provider']}:{r['day']}", delta)


def remaining() -> dict:
    s, day = get_store(), _day()
    out = {}
    for p in TOTAL_CAP:
        tot, d = float(s.get(f"spend:{p}:total") or 0), float(s.get(f"spend:{p}:{day}") or 0)
        out[p] = {"total_left_usd": round(max(TOTAL_CAP[p] - tot, 0), 2), "today_left_usd": round(max(DAILY_CAP[p] - d, 0), 2)}
    return out


def start_run(sid: str, now: float = None) -> str:
    """A new run for a session, or Refused('session_limit') past RUNS_PER_HOUR runs in the last hour."""
    s, now = get_store(), now or time.time()
    recent = [float(t) for t in s.lrange(f"runs:{sid}") if now - float(t) < 3600]
    if len(recent) >= RUNS_PER_HOUR:
        raise Refused("session_limit")
    s.rpush(f"runs:{sid}", repr(now), ttl=3600)
    run_id = secrets.token_hex(8)
    s.set(f"run:{run_id}", json.dumps({"sid": sid, "t": now}), ttl=RUN_TTL_S)
    return run_id


def use(run_id: str, sid: str, station: str, max_uses: int = 1) -> None:
    """One use of a live station by a run that belongs to this session."""
    s = get_store(); raw = s.get(f"run:{run_id}") if run_id else None
    if not raw or json.loads(raw)["sid"] != sid:
        raise Refused("no_run")
    if s.incrbyfloat(f"use:{run_id}:{station}", 1, ttl=RUN_TTL_S) > max_uses:
        raise Refused("station_used")
