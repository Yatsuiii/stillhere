"""The jobs table for the laptop worker (docs/jobs_api.md). The page queues a recipe; the worker claims it over HTTPS with its job
token, posts real stage events and a region-value result. Results are kept by cache key; data/forecast_cache.json (the worker fork's
precomputed default-seed forecasts) is the fallback whenever the worker is offline or a job expires."""
import datetime
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import sys
import time
from pathlib import Path

from web import ledger
from web.store import StoreError, get_store

REPO = Path(__file__).resolve().parents[1]
MENU = REPO / "stillhere/live_menu.json"
CACHE = REPO / "data/forecast_cache.json"
CONDITIONS = ("intact", "sentence_scramble", "word_scramble", "reversed")
RESEED = {"intact": 0, "reversed": 0}
JOB_EXPIRY_S = float(os.environ.get("JOB_EXPIRY_S", 120))   # measured jobs: 39-56 s warm on the 4050 (worker fork)
STAGES = ("load", "build", "text_lane", "audio_lane", "video_lane", "predict", "regions", "upload")
LANES = ("w2v-bert", "llama-3.2", "v-jepa-2", None)
STATUSES = ("start", "end", "off")
FAIL_CODES = ("timeout", "oom", "model_error", "bad_recipe", "cache_key_mismatch", "busy")
MAX_POINTS = 60
QUEUE_MAX = 6                                               # jobs waiting at most: about five minutes of the laptop's work
QUEUED_MAX_S = 720                                          # a queued job this old expires even with the worker online: a claim whose
                                                            # LPOP answer was lost, or whose running write failed, leaves it off the
                                                            # queue but "queued" (second refuter's review of Oct 31, item C)
ADDRESS_WAITING = 2                                         # of them, from one network address at most (fourth review, item 1)
WAITING_TTL_S = 600                                         # an address's waiting count ends 10 minutes after its latest queued job
                                                            # (each queued job sets it again, a refused try never: the refuters' items
                                                            # 5 and B), so a count a lost answer leaks is short (cloud review, item 8)
BUSY_LINE = "The laptop's queue is full, so a saved forecast is shown where one exists."
FAILED_LINE = "The live forecast did not finish, so a saved forecast is shown where one exists."
ADDRESS_BUSY_LINE = "This network address already has forecasts waiting on the laptop, so a saved forecast is shown where one exists."
SESSION_BUSY_LINE = "This session's other forecast is still on the laptop, so a saved forecast is shown where one exists."
_clock = time.time
_online = {}                                                # per store: (when read, the answer), for worker_online(max_age_s)


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load(p: Path):
    return json.loads(p.read_text()) if p.is_file() else None


def menu_ids() -> list:
    m = _load(MENU) or {}
    return [c["id"] for c in m.get("clips", [])]


def _recipe():
    from worker import recipe as R                         # the worker fork's module: one recipe check and one cache key for both sides
    return R


def recipe(clip_id, condition, seed) -> dict:
    """A validated recipe, or ValueError. Only live-menu clips, the four conditions and an integer seed (seed 0 where it is ignored)."""
    if clip_id not in menu_ids():
        raise ValueError("clip_not_on_live_menu")
    R = _recipe()
    try:
        return R.normalize({"clip_id": clip_id, "condition": condition, "seed": seed}, R.load_index())
    except R.RecipeRefused:
        raise ValueError("bad_recipe")


def cache_key(r: dict) -> str:
    R = _recipe(); return R.cache_key(r, R.load_index())


def cached(key: str, from_store: bool = True):
    """A worker's result for this cache key, else the committed precompute's, else None; from_store False: the precompute only (an
    answer that must cost no store command, such as a refusal past the address's bucket)."""
    try:
        v = get_store().get(f"result:{key}") if from_store else None
    except StoreError:                                      # the store is failing: the committed precompute still answers
        v = None
    if v:
        return {**json.loads(v), "source": "worker"}
    d = _load(CACHE) or {}                                  # the worker fork's precompute: entries by cache key, at the top level or under
    for pre in (d.get("entries"), d.get("results"), d):    # "entries" / "results", beside its "models" and "default_seed"
        if isinstance(pre, dict) and isinstance(pre.get(key), dict):
            return {**pre[key], "source": "cache"}
    return None


def default_seed() -> int:
    return int((_load(MENU) or {}).get("default_seed", 20261028))


def worker_online(max_age_s: float = 0) -> bool:
    """Whether the worker's heartbeat is fresh. With max_age_s (the status call), an answer read in the last max_age_s seconds in this
    instance is reused, so a flood of status calls costs the store one command per instance in that time (fifth review, item 7)."""
    s, now = get_store(), _clock()
    got = _online.get(id(s))
    if max_age_s and got and now - got[0] < max_age_s:
        return got[1]
    v = s.get("worker:last_seen") is not None
    _online[id(s)] = (now, v)
    return v


def _waiting(addr: str) -> str:
    return f"waiting:{addr}"


def _unwait(addr: str, job_id: str) -> None:
    """One fewer job waiting for this address (claimed by the worker, expired, or never queued), at most once per job id: SET NX on
    the id's own key, kept as long as the job's record, so the give-back, the claim and two polls expiring the job at once take it
    off once between them (cloud review of Oct 31, 12:20 UTC, item 8); never below zero."""
    s = get_store()
    if addr and s.set(f"unwait:{job_id}", "1", ttl=6 * 3600, nx=True):
        v = s.incrbyfloat(_waiting(addr), -1)
        if v < 0:
            s.incrbyfloat(_waiting(addr), -v)


def submit(r: dict, sid: str, claimed: "ledger.Claimed", addr: str = None) -> dict:
    """Cached result, a queued job, or the cached default-seed forecast when the worker is offline, the queue is full, or the
    network address `addr` already has ADDRESS_WAITING jobs waiting; past FORECAST's hourly cap (or the address's slice of it), which
    counts only jobs that reach the laptop, ledger.Refused and nothing written. The hour and the address's waiting count go into
    `claimed`, so the caller, ending without a queued job (busy, a lost queue race, a refusal, a store error), gives them back; once
    RPUSH has queued the job they stand (claimed.paid), as the worker will claim and run it."""
    key = cache_key(r); hit = cached(key)
    if hit:
        return {"state": "cached", "result": hit}
    s, sh = get_store(), hashlib.sha256(sid.encode()).hexdigest()[:16]
    busy = s.get(f"sjob:{sh}")
    other = (status(busy) or {}) if busy else {}
    if other.get("status") in ("queued", "running"):           # one job per session: followed only by a request for its own recipe
        return {"state": "busy", "job_id": busy} if other.get("recipe") == r else \
            {"state": "busy", "why": SESSION_BUSY_LINE, "fallback": fallback(r)}
    if not worker_online():                                # no queue without a worker: the saved run, labelled (ruling of 17:35 UTC, 7)
        return {"state": "offline", "fallback": fallback(r)}
    full = lambda line=BUSY_LINE: {"state": "busy", "why": line, "fallback": fallback(r)}
    waiting = s.lrange("jobs:queue")
    if len(waiting) >= QUEUE_MAX:
        return full()
    job_id = secrets.token_hex(8)
    # One atomic count per address: added here, removed once per job id on claim, expiry or give-back. The give-back goes in before
    # the INCR (cloud review of Oct 31, 12:20 UTC, item 8), so an INCR whose answer is lost is given back too. The trade-off, kept on
    # purpose: an INCR that never reached the store is given back as well, so the count runs one low, at most one extra waiting job
    # per lost INCR, never money (refuter's review of Oct 31, item 6). With no job queued the caller gives it back with the rest.
    if addr:
        claimed.add(lambda: _unwait(addr, job_id))
        if s.incrbyfloat(_waiting(addr), 1, ttl=WAITING_TTL_S) > ADDRESS_WAITING:
            return full(ADDRESS_BUSY_LINE)
    ledger.station_hour("forecast", claimed=claimed, addr=addr)
    s.set(f"job:{job_id}", json.dumps({"id": job_id, "recipe": r, "cache_key": key, "status": "queued", "created_utc": _now(),
                                       "sid_hash": sh, "created_t": time.time(), "addr": addr}), ttl=6 * 3600)
    if s.rpush("jobs:queue", job_id) > QUEUE_MAX:          # visitors passed the check at once: RPUSH's length is atomic, so each one
        s.delete(f"job:{job_id}"); s.lrem("jobs:queue", job_id); return full()       # past the cap takes its own id back out
    claimed.paid = True                                    # queued: the counts stand, whatever the next write does (item 8)
    if addr:                                               # the count's 600 s again, for a queued job only: a refused try never
        s.expire(_waiting(addr), WAITING_TTL_S)            # keeps a leaked count alive (second refuter's review of Oct 31, item B)
    s.set(f"sjob:{sh}", job_id, ttl=600)
    return {"state": "queued", "job_id": job_id}


def fallback(r: dict, from_store: bool = True):
    """The saved forecast for this clip and condition at the default seed, or None; when the visitor asked for another seed it says
    so (ruling of Oct 29, 17:35 UTC, item 7). from_store: as for cached()."""
    d = {**r, "seed": RESEED.get(r["condition"], default_seed())}
    hit = cached(cache_key(d), from_store)
    if not hit:
        return None
    note = None if d["seed"] == r["seed"] else f"No saved run for seed {r['seed']}; shown: the saved run at seed {d['seed']}."
    return {**hit, "recipe": d, "requested_seed": r["seed"], "note": note}


def status(job_id: str):
    s = get_store(); raw = s.get(f"job:{job_id}")
    if not raw:
        return None
    j = json.loads(raw); age = time.time() - j.get("created_t", 0)
    stale = (j["status"] == "running" and time.time() - j["claimed_t"] > JOB_EXPIRY_S) or \
        (j["status"] == "queued" and (age > QUEUED_MAX_S or (age > JOB_EXPIRY_S and not worker_online())))
    if stale:                                              # a running job past its time, or a queued one with no worker online or stranded
        was = j["status"]; j["status"] = "expired"; s.set(f"job:{job_id}", json.dumps(j), ttl=6 * 3600)
        if was == "queued":
            _unwait(j.get("addr"), job_id)
    out = {"status": j["status"], "recipe": j["recipe"], "events": [json.loads(e) for e in s.lrange(f"events:{job_id}")]}
    if j.get("worker_label"):
        out["worker_label"] = j["worker_label"]
    if j["status"] == "done":
        out["result"] = cached(j["cache_key"])
    if j["status"] in ("expired", "failed"):
        out.update(fallback=fallback(j["recipe"]), why=FAILED_LINE)
    return out


# ---- worker side -------------------------------------------------------------------------------------------------------------
def authorized(header: str) -> bool:
    tok = os.environ.get("STILLHERE_WORKER_TOKEN") or ""
    got = header[7:] if header and header.startswith("Bearer ") else ""
    return bool(tok) and hmac.compare_digest(got.encode(), tok.encode())


def _heartbeat() -> None:
    """The worker is online for JOB_EXPIRY_S after any authorized call (claim, event, result, fail): a busy worker does not poll
    claim, and a 30 s heartbeat from claim alone had it offline from 30 s into every 35-56 s job, so other visitors were told it was
    offline and queued jobs expired on a poll (cloud review of Oct 31, 12:20 UTC, item 3)."""
    get_store().set("worker:last_seen", _now(), ttl=JOB_EXPIRY_S)


def claim(body: dict):
    s = get_store(); _heartbeat()
    while True:
        job_id = s.lpop("jobs:queue")
        if job_id is None:
            return None
        raw = s.get(f"job:{job_id}")
        if not raw:
            continue
        j = json.loads(raw)
        if j["status"] != "queued":
            continue
        j.update(status="running", claimed_utc=_now(), claimed_t=time.time())
        lbl = body.get("worker_label") if isinstance(body, dict) else None   # what the page says while it runs (worker/labels.py)
        if isinstance(lbl, str) and re.fullmatch(r"[A-Za-z0-9 ._()-]{1,40}", lbl):
            j["worker_label"] = lbl
        try:                                               # first, so a failed write below never leaves the count up (item 8); best
            _unwait(j.get("addr"), job_id)                 # effort, so a failure here never strands the popped job as "queued" (it
        except StoreError as e:                            # would poll "Queued" for 6 h): the count over-counts at worst (refuter's
            print(f"stillhere: store error ({e}) in claim", file=sys.stderr, flush=True)   # review of Oct 31, item 4)
        s.set(f"job:{job_id}", json.dumps(j), ttl=6 * 3600)
        return {"id": job_id, "recipe": j["recipe"], "cache_key": j["cache_key"], "claimed_utc": j["claimed_utc"], "deadline_s": 40}


def _running(job_id: str):
    raw = get_store().get(f"job:{job_id}")
    j = json.loads(raw) if raw else None
    if not j or j["status"] != "running" or time.time() - j["claimed_t"] > JOB_EXPIRY_S:
        return None
    return j


def event(job_id: str, e: dict) -> int:
    _heartbeat()
    if not _running(job_id):
        return 409
    if e.get("stage") not in STAGES or e.get("lane") not in LANES or e.get("status") not in STATUSES:
        return 422
    det = e.get("detail") or {}
    keep = {k: det[k] for k in ("n_words", "precision") if k in det}
    if "n_words" in keep and not (isinstance(keep["n_words"], int) and 0 <= keep["n_words"] < 100000):
        return 422
    if "precision" in keep and not (isinstance(keep["precision"], str) and len(keep["precision"]) <= 64):
        return 422
    get_store().rpush(f"events:{job_id}", json.dumps({"stage": e["stage"], "lane": e["lane"], "status": e["status"],
                                                       "t_utc": str(e.get("t_utc", _now()))[:32], "detail": keep}), ttl=6 * 3600)
    return 204


def _series(x) -> bool:
    return isinstance(x, list) and 0 < len(x) <= MAX_POINTS and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                                                     and math.isfinite(v) for v in x)


def result(job_id: str, body: dict) -> int:
    _heartbeat(); j = _running(job_id)
    if not j:
        return 409
    regs = body.get("regions") or {}
    if body.get("cache_key") != j["cache_key"] or set(regs) != {"language", "hearing"} or not all(_series(v) for v in regs.values()) \
            or not _series(body.get("word_rate") or [0]) or len({len(v) for v in regs.values()}) != 1:
        return 422
    keep = {k: body.get(k) for k in ("cache_key", "tr_s", "seconds", "regions", "word_rate", "n_words_text_lane", "models", "input_sha256",
                                      "peak_vram_gb", "worker_version", "gpu", "timings_s", "computed_utc")}
    if len(json.dumps(keep)) > 20000:
        return 422
    s = get_store(); s.set(f"result:{j['cache_key']}", json.dumps({**keep, "recipe": j["recipe"]}))
    j["status"] = "done"; s.set(f"job:{job_id}", json.dumps(j), ttl=6 * 3600)
    return 204


def fail(job_id: str, body: dict) -> int:
    _heartbeat(); j = _running(job_id)
    if not j:
        return 409
    if body.get("reason") not in FAIL_CODES:
        return 422
    j.update(status="failed", reason=body["reason"]); get_store().set(f"job:{job_id}", json.dumps(j), ttl=6 * 3600)
    return 204
