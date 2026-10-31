"""The backend's key-value store. Upstash Redis over its REST API when UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN (or
Vercel's KV_REST_API_URL and KV_REST_API_TOKEN) are set; otherwise an in-process stand-in with the same interface, for local runs and
tests. Values are strings; callers store JSON. Nothing typed by a visitor is ever written here."""
import os
import threading
import time

import httpx


class StoreError(Exception):
    """The shared store failed (an outage, a spent quota, a timeout, an error answer); its message is the cause's type only."""


class MemoryStore:
    """Single-process stand-in: the same commands and the same atomicity (one lock) as the Redis calls below."""

    def __init__(self):
        self._d, self._exp, self._lock = {}, {}, threading.Lock()

    def _live(self, k):
        e = self._exp.get(k)
        if e is not None and e <= time.time():
            self._d.pop(k, None); self._exp.pop(k, None)
        return self._d.get(k)

    def _ttl(self, k, ttl):
        if ttl:
            self._exp[k] = time.time() + ttl

    def get(self, k):
        with self._lock:
            v = self._live(k)
            return None if v is None or isinstance(v, (list, dict)) else v

    def mget(self, keys):
        return [self.get(k) for k in keys]

    def set(self, k, v, ttl=None, nx=False):
        with self._lock:
            if nx and self._live(k) is not None:
                return False
            self._d[k] = str(v); self._exp.pop(k, None); self._ttl(k, ttl)
            return True

    def incrbyfloat(self, k, amount, ttl=None):
        """INCRBYFLOAT; the ttl set on a key it creates (EXPIRE NX)."""
        with self._lock:
            return self._incr(k, amount, ttl)

    def incrbyfloat_many(self, items):
        """INCRBYFLOAT for each (key, amount, ttl), the ttl set on a key it creates, all under one lock; the new values."""
        with self._lock:
            return [self._incr(k, amount, ttl) for k, amount, ttl in items]

    def _incr(self, k, amount, ttl):
        v = float(self._live(k) or 0) + float(amount); new = self._live(k) is None
        self._d[k] = repr(v)
        if new:
            self._ttl(k, ttl)
        return v

    def expire(self, k, ttl):
        """EXPIRE k ttl: a live key's ttl set again."""
        with self._lock:
            if self._live(k) is not None:
                self._exp[k] = time.time() + ttl

    def zwindow(self, k, member, score, span):
        """ZREMRANGEBYSCORE k -inf score-span, ZADD k score member, ZCARD k, EXPIRE k span: how many members are left in
        (score - span, score], this one included."""
        with self._lock:
            z = self._live(k)
            if not isinstance(z, dict):
                z = self._d[k] = {}
            for m in [m for m, t in z.items() if t <= score - span]:
                del z[m]
            z[str(member)] = float(score); self._exp[k] = time.time() + span
            return len(z)

    def zrem(self, k, member):
        with self._lock:
            z = self._live(k)
            return int(isinstance(z, dict) and z.pop(str(member), None) is not None)

    def rpush(self, k, v, ttl=None):
        with self._lock:
            lst = self._live(k)
            if lst is None:
                lst = self._d[k] = []; self._ttl(k, ttl)
            lst.append(str(v)); return len(lst)

    def lpop(self, k):
        with self._lock:
            lst = self._live(k)
            return lst.pop(0) if lst else None

    def lrem(self, k, v):
        """Redis LREM k 1 v: removes the first copy of v; the number removed."""
        with self._lock:
            lst = self._live(k) or []
            if str(v) in lst:
                lst.remove(str(v)); return 1
            return 0

    def lrange(self, k, start=0, stop=-1):
        with self._lock:
            lst = self._live(k) or []
            return list(lst[start:] if stop == -1 else lst[start:stop + 1])

    def delete(self, k):
        with self._lock:
            self._d.pop(k, None); self._exp.pop(k, None)


class UpstashStore:
    """Upstash Redis REST: one POST per command (a pipeline where a command needs its TTL set in the same round trip)."""

    def __init__(self, url, token):
        self.url, self.h = url.rstrip("/"), {"Authorization": f"Bearer {token}"}

    def _cmd(self, *args):
        try:
            r = httpx.post(self.url, json=[str(a) for a in args], headers=self.h, timeout=10)
            r.raise_for_status(); return r.json()["result"]
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as e:
            raise StoreError(type(e).__name__) from e

    def _pipe(self, *cmds):
        try:
            r = httpx.post(self.url + "/pipeline", json=[[str(a) for a in c] for c in cmds], headers=self.h, timeout=10)
            r.raise_for_status(); out = r.json()
            if any("error" in x for x in out):                     # a pipeline answers each command alone
                raise ValueError("a command in the pipeline failed")
            return [x.get("result") for x in out]
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as e:
            raise StoreError(type(e).__name__) from e

    def get(self, k):
        return self._cmd("GET", k)

    def mget(self, keys):
        return self._cmd("MGET", *keys) if keys else []

    def set(self, k, v, ttl=None, nx=False):
        args = ["SET", k, v] + (["EX", int(ttl)] if ttl else []) + (["NX"] if nx else [])
        return self._cmd(*args) == "OK"

    def incrbyfloat(self, k, amount, ttl=None):
        if not ttl:
            return float(self._cmd("INCRBYFLOAT", k, amount))
        v, _ = self._pipe(["INCRBYFLOAT", k, amount], ["EXPIRE", k, int(ttl), "NX"])
        return float(v)

    def expire(self, k, ttl):
        self._cmd("EXPIRE", k, int(ttl))

    def incrbyfloat_many(self, items):
        """INCRBYFLOAT for each (key, amount, ttl), EXPIRE NX after each with a ttl, in one pipeline: one round trip, so a request
        that fails moves no key, where key by key a failure left the earlier keys moved (cloud review of Oct 31, 12:20 UTC, item 11).
        An answer lost after Upstash ran it moves them all, as for any one command. The new values."""
        cmds, at = [], []
        for k, amount, ttl in items:
            at.append(len(cmds)); cmds.append(["INCRBYFLOAT", k, amount])
            if ttl:
                cmds.append(["EXPIRE", k, int(ttl), "NX"])
        out = self._pipe(*cmds) if cmds else []
        return [float(out[i]) for i in at]

    def zwindow(self, k, member, score, span):
        return int(self._pipe(["ZREMRANGEBYSCORE", k, "-inf", repr(score - span)], ["ZADD", k, repr(score), member], ["ZCARD", k],
                              ["EXPIRE", k, int(span)])[2])

    def zrem(self, k, member):
        return int(self._cmd("ZREM", k, member))

    def rpush(self, k, v, ttl=None):
        if not ttl:
            return int(self._cmd("RPUSH", k, v))
        n, _ = self._pipe(["RPUSH", k, v], ["EXPIRE", k, int(ttl), "NX"])
        return int(n)

    def lpop(self, k):
        return self._cmd("LPOP", k)

    def lrem(self, k, v):
        return int(self._cmd("LREM", k, 1, v))

    def lrange(self, k, start=0, stop=-1):
        return self._cmd("LRANGE", k, start, stop) or []

    def delete(self, k):
        self._cmd("DEL", k)


_store = None


def get_store():
    global _store
    if _store is None:
        url = os.environ.get("UPSTASH_REDIS_REST_URL") or os.environ.get("KV_REST_API_URL")
        tok = os.environ.get("UPSTASH_REDIS_REST_TOKEN") or os.environ.get("KV_REST_API_TOKEN")
        _store = UpstashStore(url, tok) if url and tok else MemoryStore()
    return _store


def shared() -> bool:
    """True when the store is shared across instances (Upstash): caps, limits and the jobs table only hold then."""
    return isinstance(get_store(), UpstashStore)


def set_store(s):
    """Tests and the local runner swap the store."""
    global _store
    _store = s
