"""Spend caps and per-visitor limits, kept in the store so they survive restarts and hold across serverless instances.

Spend: every paid call first reserves its worst-case price (INCRBYFLOAT returns the new total atomically, so two calls cannot both
slip under a cap), then settles to the price it really cost. A provider stops for good at its total cap (Meta $80, Fireworks $50,
Baseten $50) and for the day at its daily cap (Meta $20, Fireworks $12, Baseten $12): provider-side caps may not exist. One visitor
(its session, and its network address) keeps at most a small share of each daily cap per UTC day. Runs: at most RUNS_PER_HOUR per
session and RUNS_PER_HOUR_PER_ADDRESS per address in any sliding hour; each run may use each live station once (BUILD: once per
allowed seed); and every visitor together gets STATION_RUNS_PER_HOUR live uses of each station per sliding hour.

Order (third review, items 1 and 4): a live call takes the visitor's own counts first (its runs, its run's use, a read of its daily
shares and the spend caps), then the station's shared hour, then its spend reservation, and a refusal gives back everything the
request took (Claimed), so a visitor refused for its own limits never spends a shared count, and a full shared count never costs a
visitor its own. Every count is added first and checked after (INCRBYFLOAT, or a sorted set of event times for the sliding hours),
so two requests at once cannot both slip under a limit, and a take-back can only make another request refuse early."""
import datetime
import hashlib
import hmac
import ipaddress
import json
import os
import secrets
import sys
import time

from web.store import StoreError, get_store

TOTAL_CAP = {"meta": float(os.environ.get("CAP_META_USD", 80)), "fireworks": float(os.environ.get("CAP_FIREWORKS_USD", 50)),
             "baseten": float(os.environ.get("CAP_BASETEN_USD", 50))}                   # cloud plan of Oct 29, 19:20 UTC, 1c
DAILY_CAP = {"meta": float(os.environ.get("DAILY_CAP_META_USD", 20)), "fireworks": float(os.environ.get("DAILY_CAP_FIREWORKS_USD", 12)),
             "baseten": float(os.environ.get("DAILY_CAP_BASETEN_USD", 12))}
# The event's Meta credit (web/README.md, Money): Meta stations stay live through Nov 9; after META_CREDIT_END_UTC, Meta spend gets one
# more total cap of $20, then recorded runs (cloud ruling of Oct 29, 19:40 UTC, item 3). Override the end with the env var.
META_CREDIT_END_UTC = "2026-11-05T02:00:00Z"
POST_CREDIT_CAP_META = float(os.environ.get("POST_CREDIT_CAP_META_USD", 20))
RUNS_PER_HOUR = 5                  # per session cookie, in any sliding hour
RUNS_PER_HOUR_PER_ADDRESS = 20     # per network address (an IPv6 /64): four visitors' runs behind one venue's, office's or carrier's NAT
RUN_TTL_S = 2 * 3600
STATIONS = ("plan", "build", "audit", "forecast", "film")
# What one visitor may keep on the ledger per UTC day, as a share of each provider's configured daily cap (DAILY_CAP, env
# DAILY_CAP_*_USD): 20% for a session and the same for an address (fifth review, item 1: a session's own eighth turned a visitor's
# second FILM into a recorded run). At the configured caps (Meta $20, Fireworks and Baseten $12) that is $4.00 and $2.40. Three
# reshuffle-heavy walks fit once SAM_USD_PER_FRAME is set (FILM then settles to its price); unset, FILM keeps $1.50 a run, so a walk
# with FILM is $2.20 and a second FILM in one day does not fit. Every cookie from one address together keeps at most a fifth of a day.
SESSION_DAILY_SHARE, ADDRESS_DAILY_SHARE = 0.20, 0.20
SESSION_DAILY_USD = {p: c * SESSION_DAILY_SHARE for p, c in DAILY_CAP.items()}
ADDRESS_DAILY_USD = {p: c * ADDRESS_DAILY_SHARE for p, c in DAILY_CAP.items()}
# Live uses of each station per sliding hour, all visitors together, and (ADDRESS_SLICE) the most one network address may take of
# it, at most a quarter, so no address can switch a station off for another. BUILD 70: a run builds at most 1 +
# len(stations.RESHUFFLE_SEEDS) = 7 times, ten such runs an hour; its slice 17 holds two. PLAN and AUDIT 120 with a slice of 10 each
# (fifth review, item 8: four addresses held PLAN's 40 all day): both settle to their tokens, so the daily caps bound their money, and
# an address cannot AUDIT more runs than it plans. FILM 10 with a slice of 2 (item 1). FORECAST counts only jobs that reach the
# laptop: 30 is about half of what it finishes in an hour. An hour at every cap can now reserve more than a Meta day (PLAN 120 x $0.05
# + BUILD 70 x $0.10 + FILM 10 x $1.50 = $28.00 of $20, PLAN's $6.00 settling to cents), so with FILM unpriced the daily cap is the
# stop for many addresses at once, and the daily shares the stop for one; Fireworks: PLAN $6.00 + AUDIT 120 x $0.03 = $9.60 of $12.
STATION_RUNS_PER_HOUR = {"plan": 120, "build": 70, "audit": 120, "forecast": 30, "film": 10}
ADDRESS_SLICE = {"plan": 10, "build": 17, "audit": 10, "forecast": 7, "film": 2}
VISITOR_TTL_S = 86400              # an issued visitor id lives as long as its cookie (Max-Age)
SHARED = "a-"                      # the visitor of a request with no issued cookie: "a-" and its address's name (never a cookie)
HOUR = 3600


class Refused(Exception):
    """A live call that must not happen; `code` says why and the page shows the recorded run instead."""

    def __init__(self, code: str):
        super().__init__(code); self.code = code


def _day() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


def meta_credit_ended(now: datetime.datetime = None) -> bool:
    end = os.environ.get("META_CREDIT_END_UTC", META_CREDIT_END_UTC)
    now = now or datetime.datetime.now(datetime.timezone.utc)
    return bool(end) and now >= datetime.datetime.fromisoformat(end.replace("Z", "+00:00"))


class Claimed:
    """What one request has claimed from the counts so far. give_back() returns all of it, latest first, for a request refused before
    its paid call; once the call is made (paid) the counts stand. As a context manager it gives back on any exception before the paid
    call (a refusal, a store error, a bug) and lets the exception go on."""

    def __init__(self):
        self._back, self.paid = [], False                   # paid: set just before the provider call; from then on the counts stand

    def __enter__(self):
        return self

    def __exit__(self, kind, err, tb):
        if kind is not None and not self.paid:
            self.give_back()
        return False

    def add(self, undo) -> None:
        self._back.append(undo)

    def give_back(self) -> None:
        while self._back:
            try:
                self._back.pop()()
            except Exception as e:                             # the store is failing too: that count ends with its window
                print(f"stillhere: give-back failed ({type(e).__name__})", file=sys.stderr, flush=True)


def _caps(provider: str, run_id: str = None) -> list:
    """(key, cap, refusal code, ttl) for every bound a call to this provider counts toward: the provider's total and daily caps, Meta's
    after-credit cap, and for a run its session's and address's daily shares."""
    s, day = get_store(), _day()
    caps = [(f"spend:{provider}:total", TOTAL_CAP[provider], "spend_cap", None),
            (f"spend:{provider}:{day}", DAILY_CAP[provider], "spend_cap", 3 * 86400)]
    if provider == "meta" and meta_credit_ended():
        caps.append(("spend:meta:after_credit", POST_CREDIT_CAP_META, "post_credit_cap", None))
    who = json.loads(s.get(f"run:{run_id}") or "null") if run_id else None
    if who:
        caps.append((f"vspend:{provider}:{who['sid']}:{day}", SESSION_DAILY_USD[provider],
                     "shared_daily" if who["sid"].startswith(SHARED) else "session_daily", 2 * 86400))
        if who.get("ip"):
            caps.append((f"vspend:{provider}:ip:{who['ip']}:{day}", ADDRESS_DAILY_USD[provider], "ip_daily", 2 * 86400))
    return caps


ORDER = ("post_credit_cap", "spend_cap", "session_daily", "shared_daily", "ip_daily")


def affordable(run_id: str, provider: str, usd: float) -> None:
    """Read only, before any shared count is claimed: Refused when `usd` more would pass a spend cap or the run's session's or address's
    daily share (one MGET). reserve() makes the binding check."""
    if provider not in TOTAL_CAP:
        raise Refused("provider_not_offered")
    caps = _caps(provider, run_id)
    have = get_store().mget([k for k, *_ in caps])
    over = {code for (k, cap, code, ttl), v in zip(caps, have) if float(v or 0) + usd > cap + 1e-9}
    if over:
        raise Refused(next(c for c in ORDER if c in over))


def reserve(provider: str, usd: float, run_id: str = None) -> dict:
    """The call's worst-case price, added to every bound it counts toward, or Refused (and nothing added): the provider's total or
    daily cap ('spend_cap'), Meta's after-credit cap ('post_credit_cap'), and for a run's call its session's and its address's daily
    shares ('session_daily', or 'shared_daily' for the address's cookieless visitor; 'ip_daily')."""
    if provider not in TOTAL_CAP:
        raise Refused("provider_not_offered")
    s, day, caps = get_store(), _day(), _caps(provider, run_id)
    over = {code for k, cap, code, ttl in caps if s.incrbyfloat(k, usd, ttl=ttl) > cap + 1e-9}   # every key, then the check
    if over:
        for k, *_ in caps:
            s.incrbyfloat(k, -usd)
        raise Refused(next(c for c in ORDER if c in over))
    return {"provider": provider, "reserved": usd, "day": day, "caps": [(k, cap) for k, cap, *_ in caps]}


def settle(r: dict, actual_usd: float) -> None:
    """Replace the reservation by the real price (never below zero; an unknown price keeps the reservation). A bounded call cannot
    cost more than it reserved (stations: max_tokens, SAM's reservation at least its price); if one ever does, the excess is logged
    and each bound is held at its cap, so the next call is refused instead of the ledger passing a cap."""
    if actual_usd is None:
        return
    delta = max(actual_usd, 0.0) - r["reserved"]; s = get_store()
    if delta > 1e-12:
        print(f"stillhere: a {r['provider']} call cost more than its reservation", file=sys.stderr, flush=True)
    for k, cap in r["caps"]:
        v = s.incrbyfloat(k, delta)
        if delta > 0 and v > cap:
            s.incrbyfloat(k, cap - v)


def remaining() -> dict:
    s, day = get_store(), _day()
    out = {}
    for p in TOTAL_CAP:
        tot, d = float(s.get(f"spend:{p}:total") or 0), float(s.get(f"spend:{p}:{day}") or 0)
        out[p] = {"total_left_usd": round(max(TOTAL_CAP[p] - tot, 0), 2), "today_left_usd": round(max(DAILY_CAP[p] - d, 0), 2)}
    return out


def _window(key: str, limit: int, now: float = None):
    """One more event in the true sliding hour, or None past `limit`: the store keeps each event's time in a sorted set (events an
    hour old leave first), so no 3600 s span ever holds more than `limit` and an event passes the moment the oldest is an hour old.
    A refused event is claimed back at once. Returns the event's member, so a caller can give it back (store.zrem)."""
    s, now = get_store(), now or time.time()
    m = f"{now!r}:{os.urandom(4).hex()}"
    if s.zwindow(key, m, now, HOUR) > limit:
        s.zrem(key, m); return None
    return m


def _address(ip: str) -> str:
    """The store's name for a network address: an IPv6 address counts as its /64 (one subscriber's block), an IPv4-mapped one as its
    IPv4 address. Keyed HMAC-SHA256 under STILLHERE_WORKER_TOKEN, a server-only secret the app already holds, when it is set (else
    plain SHA-256), so the store never holds an address and its names cannot be reversed by trying every IPv4 address."""
    try:
        a = ipaddress.ip_address(ip.strip())
        a = a.ipv4_mapped or a if a.version == 6 else a
        net = str(ipaddress.ip_network(f"{a}/64", strict=False)) if a.version == 6 else str(a)
    except ValueError:
        net = ip
    salt = os.environ.get("STILLHERE_WORKER_TOKEN", "").encode()
    return (hmac.new(salt, net.encode(), "sha256") if salt else hashlib.sha256(net.encode())).hexdigest()[:16]


def issue_visitor(ip: str):
    """A fresh visitor id for each page's first call (POST /api/visitor), issued by the server alone: SET NX on the id's own key
    marks it as ours (a cookie the server never issued is never honored). An address mints at most RUNS_PER_HOUR_PER_ADDRESS ids in
    the sliding hour; past that, None, and its visitors share the address's cookieless session, whose refusals say so."""
    if not _window(f"win:ids:ip:{_address(ip or '')}", RUNS_PER_HOUR_PER_ADDRESS):
        return None
    s = get_store()
    while True:
        sid = secrets.token_hex(16)
        if s.set(f"visitor:{sid}", "1", ttl=VISITOR_TTL_S, nx=True):
            return sid


def visitor(cookie: str, ip: str) -> str:
    """The visitor a request counts as: its cookie when the server issued it, else its network address (SHARED and the address's
    name), so cookieless requests and cookies the server never issued all share the address's one session, never a fresh visitor
    each; their refusals say so ('shared_limit', 'shared_daily'). With the store failing, the address too (the station then gives its
    recorded run at its first count)."""
    try:
        if cookie and get_store().get(f"visitor:{cookie}") is not None:
            return cookie
    except StoreError:
        pass
    return SHARED + _address(ip or "")


def start_run(sid: str, now: float = None, ip: str = None, claimed: Claimed = None) -> str:
    """A new run for a session, or Refused past RUNS_PER_HOUR runs in the sliding hour for the session ('session_limit', or
    'shared_limit' for an address's cookieless visitor) or past RUNS_PER_HOUR_PER_ADDRESS for its network address ('ip_limit'; ip
    None: a caller with no address, such as a test or a script). A refused start costs neither count; with `claimed`, a later refusal
    gives the run back too. The run keeps its address's name, so its spend counts toward the address's daily share."""
    s, now = get_store(), now or time.time()
    mine = _window(f"win:runs:{sid}", RUNS_PER_HOUR, now)
    if not mine:
        raise Refused("shared_limit" if sid.startswith(SHARED) else "session_limit")
    addr = _address(ip) if ip is not None else None
    theirs = _window(f"win:runs:ip:{addr}", RUNS_PER_HOUR_PER_ADDRESS, now) if addr else None
    if addr and not theirs:
        s.zrem(f"win:runs:{sid}", mine); raise Refused("ip_limit")
    run_id = secrets.token_hex(8)
    s.set(f"run:{run_id}", json.dumps({"sid": sid, "t": now, "ip": addr}), ttl=RUN_TTL_S)
    if claimed is not None:
        def unstart():
            s.zrem(f"win:runs:{sid}", mine); s.delete(f"run:{run_id}")
            if theirs:
                s.zrem(f"win:runs:ip:{addr}", theirs)
        claimed.add(unstart)
    return run_id


def address_slice(station: str) -> int:
    """The most of a station's hour one network address may take (ADDRESS_SLICE)."""
    return ADDRESS_SLICE[station]


def station_hour(station: str, claimed: Claimed = None, addr: str = None) -> None:
    """One of the station's live uses in the sliding hour: first from the address's slice ('ip_station'; addr None: no address),
    then from STATION_RUNS_PER_HOUR for all visitors together ('station_cap'), so one address can never switch a station off for
    another. Take it after every per-visitor check; with `claimed`, a later refusal gives both back."""
    t = claimed if claimed is not None else Claimed()
    with t:
        if addr:
            m = _window(f"win:cap:{station}:ip:{addr}", address_slice(station))
            if not m:
                raise Refused("ip_station")
            t.add(lambda: get_store().zrem(f"win:cap:{station}:ip:{addr}", m))
        g = _window(f"win:cap:{station}", STATION_RUNS_PER_HOUR[station])
        if not g:
            raise Refused("station_cap")
        t.add(lambda: get_store().zrem(f"win:cap:{station}", g))


def use(run_id: str, sid: str, station: str, max_uses: int = 1, claimed: Claimed = None):
    """One use of a live station by a run that belongs to this session (a refused use is claimed back at once); with `claimed`, a later
    refusal gives it back too. Returns the run's address name, for station_hour(), which comes after."""
    s = get_store(); raw = s.get(f"run:{run_id}") if run_id else None
    who = json.loads(raw) if raw else None
    if not who or who["sid"] != sid:
        raise Refused("no_run")
    k = f"use:{run_id}:{station}"
    if s.incrbyfloat(k, 1, ttl=RUN_TTL_S) > max_uses:
        s.incrbyfloat(k, -1); raise Refused("station_used")
    if claimed is not None:
        claimed.add(lambda: s.incrbyfloat(k, -1))
    return who.get("ip")
