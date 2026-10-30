"""The backend's key-value store. Upstash Redis over its REST API when UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN (or
Vercel's KV_REST_API_URL and KV_REST_API_TOKEN) are set; otherwise an in-process stand-in with the same interface, for local runs and
tests. Values are strings; callers store JSON. Nothing typed by a visitor is ever written here."""
import os
import threading
import time

import httpx


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
            return None if v is None or isinstance(v, list) else v

    def set(self, k, v, ttl=None, nx=False):
        with self._lock:
            if nx and self._live(k) is not None:
                return False
            self._d[k] = str(v); self._exp.pop(k, None); self._ttl(k, ttl)
            return True

    def incrbyfloat(self, k, amount, ttl=None):
        with self._lock:
            v = float(self._live(k) or 0) + float(amount); new = self._live(k) is None
            self._d[k] = repr(v)
            if new:
                self._ttl(k, ttl)
            return v

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
        r = httpx.post(self.url, json=[str(a) for a in args], headers=self.h, timeout=10)
        r.raise_for_status(); return r.json()["result"]

    def _pipe(self, *cmds):
        r = httpx.post(self.url + "/pipeline", json=[[str(a) for a in c] for c in cmds], headers=self.h, timeout=10)
        r.raise_for_status(); return [x.get("result") for x in r.json()]

    def get(self, k):
        return self._cmd("GET", k)

    def set(self, k, v, ttl=None, nx=False):
        args = ["SET", k, v] + (["EX", int(ttl)] if ttl else []) + (["NX"] if nx else [])
        return self._cmd(*args) == "OK"

    def incrbyfloat(self, k, amount, ttl=None):
        if not ttl:
            return float(self._cmd("INCRBYFLOAT", k, amount))
        v, _ = self._pipe(["INCRBYFLOAT", k, amount], ["EXPIRE", k, int(ttl), "NX"])
        return float(v)

    def rpush(self, k, v, ttl=None):
        if not ttl:
            return int(self._cmd("RPUSH", k, v))
        n, _ = self._pipe(["RPUSH", k, v], ["EXPIRE", k, int(ttl), "NX"])
        return int(n)

    def lpop(self, k):
        return self._cmd("LPOP", k)

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
