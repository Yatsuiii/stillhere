"""The hosted Pilot app as one WSGI callable (`app`), standard library only around the repo's own modules: the stations rail page
(web/public), the station API, receipts, and the jobs API for the laptop worker (docs/jobs_api.md). Local: `python web/serve.py`.
Vercel: the repository root's api/index.py exports the same `app` (docs/deploy_checklist.md). Keys stay in the server's environment; no
response carries one, and request bodies are never logged: an unexpected error logs only its type and the path."""
import hashlib
import json
import mimetypes
import os
import re
import sys
import threading
import time
import urllib.parse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
_bl = os.environ.get("STILLHERE_QUESTION_BLOCKLIST")      # read by stillhere/egress.py at import: a relative path is the repository's
if _bl and not os.path.isabs(_bl):
    os.environ["STILLHERE_QUESTION_BLOCKLIST"] = str(REPO / _bl)
if os.environ.get("STILLHERE_LOCAL_GPU") == "1" and (os.environ.get("VERCEL_ENV") or os.environ.get("VERCEL")):
    raise SystemExit("STILLHERE_LOCAL_GPU is for a local server only (scripts/run_local_gpu.sh), never on Vercel")
if os.environ.get("VERCEL"):                               # only /tmp is writable there; the usage log holds counts and times, no text
    os.environ.setdefault("STILLHERE_USAGE_LOG", "/tmp/stillhere_usage.jsonl")

from web import guard, jobs, ledger, stations  # noqa: E402
from web.store import StoreError  # noqa: E402

PUBLIC = REPO / "web/public"
mimetypes.add_type("font/woff2", ".woff2")                     # not in every runtime's table (Vercel's sends octet-stream)
MEDIA = {"/media/old_bug.wav": REPO / stations.OLD_BUG["audio"]}
MAX_BODY = 8192
ROUTES = []
COOKIE = "sh_sid"


class TokenBucket:
    """Each network address's request budget in this instance's memory, checked before any store command: `burst` at once, then one
    every `refill_s` seconds. On Vercel every warm instance keeps its own, so a flood spread over N instances gets N budgets: this
    bounds the store commands one address can cost per instance, not in all; a Vercel Firewall rate limit is the global bound."""

    def __init__(self, burst: int, refill_s: float, clock=time.monotonic):
        self.burst, self.refill_s, self.clock, self._b, self._lock = burst, refill_s, clock, {}, threading.Lock()

    def take(self, key: str) -> bool:
        with self._lock:
            now = self.clock()
            if len(self._b) > 10000:                           # forget addresses whose budget is full again
                self._b = {k: v for k, v in self._b.items() if now - v[1] < self.burst * self.refill_s}
            tokens, t = self._b.get(key, (self.burst, now))
            tokens = min(self.burst, tokens + (now - t) / self.refill_s)
            ok = tokens >= 1
            self._b[key] = (tokens - 1 if ok else tokens, now)
            return ok


# 20 at once covers one ordinary walk's POSTs (the visitor call, PLAN, seven BUILDs, AUDIT, up to four FORECASTs, FILM); then one
# every 10 s (360 an hour) is over three times what RUNS_PER_HOUR runs can use (fifth review, item 7)
POST_LIMIT = (20, 10.0)
BUCKET = TokenBucket(*POST_LIMIT)                                  # the page's POSTs, not the worker's
# The GETs that reach the store with no CDN in front (no-store): a receipt and a job's poll get their own bucket per address, 60 at
# once, then one a second, above the page's 1.5 s poll, so a loop of random ids cannot spend the store's budget (cloud review of
# Oct 31, 12:20 UTC, item 5)
GET_LIMIT = (60, 1.0)
GET_BUCKET = TokenBucket(*GET_LIMIT)


def slowed(req) -> dict:
    """The body of a 429 past a bucket: for a station's POST its recorded run with the why line, as for any other refusal and with no
    store command (stations.slowed; cloud review of Oct 31, 12:20 UTC, item 6); bare for the visitor call and the GETs."""
    name = req.path.rsplit("/", 1)[1]
    return stations.slowed(name, req.json()) if req.method == "POST" and name != "visitor" else {"error": "slow_down"}


def route(method: str, pattern: str, cache: str = None, bucket: str = None):
    """`pattern` matches the whole path (fullmatch: "$" also matched before a trailing newline, so "/api/plan" and a newline reached
    the route and missed its bucket; refuter's review of Oct 31, item 1). `cache`: a Cache-Control for an answer that is the same for
    every visitor (see respond). `bucket`: the name of the token bucket the request passes before the route runs, so before any store
    command ("BUCKET" or "GET_BUCKET"), read per request so a test can swap it."""
    def deco(fn):
        ROUTES.append((method, re.compile(pattern), fn, cache, bucket)); return fn
    return deco


class Req:
    def __init__(self, env):
        self.env, self.method, self.path = env, env["REQUEST_METHOD"], env.get("PATH_INFO") or "/"
        ck = dict(p.strip().split("=", 1) for p in env.get("HTTP_COOKIE", "").split(";") if "=" in p)
        self.cookie = ck.get(COOKIE) if re.fullmatch(r"[0-9a-f]{32}", ck.get(COOKIE, "")) else None
        self.secure = env.get("HTTP_X_FORWARDED_PROTO", env.get("wsgi.url_scheme")) == "https"
        self.ip = (env.get("HTTP_X_REAL_IP") or env.get("REMOTE_ADDR") or "").strip()   # Vercel sets x-real-ip to the client's address
        self._sid, self.new_cookie = None, None

    @property
    def sid(self) -> str:
        """The visitor (ledger.visitor): the cookie if the server issued it, else the address. Looked up only when a route needs it."""
        if self._sid is None:
            self._sid = ledger.visitor(self.cookie, self.ip)
        return self._sid

    def issue(self) -> None:
        """The page's first call: a request without an issued cookie gets a visitor id as its cookie (ledger.issue_visitor)."""
        if self.cookie is None or self.sid != self.cookie:
            self.new_cookie = self._sid = ledger.issue_visitor(self.ip)

    def own_origin(self, origin: str) -> bool:
        """The page's own origin: the Host this request was addressed to, with the scheme the cookie's Secure flag reads, so a Vercel
        preview, the production domain, and 127.0.0.1 or localhost under serve.py all count with no list to keep; and Vercel's own
        names for this deployment (its env vars), should the function see another Host. X-Forwarded-Host is a client's to send, so it
        never counts."""
        scheme = "https" if self.secure else "http"
        mine = {f"{scheme}://{self.env['HTTP_HOST']}"} if self.env.get("HTTP_HOST") else set()
        mine |= {f"https://{os.environ[k]}" for k in ("VERCEL_URL", "VERCEL_BRANCH_URL", "VERCEL_PROJECT_PRODUCTION_URL") if os.environ.get(k)}
        return origin.lower() in {o.lower() for o in mine}

    def json(self) -> dict:
        n = int(self.env.get("CONTENT_LENGTH") or 0)
        if n > MAX_BODY:
            raise stations.Bad("body_too_large")
        try:
            d = json.loads(self.env["wsgi.input"].read(n) or b"{}")
        except ValueError:
            raise stations.Bad("bad_json")
        if not isinstance(d, dict):
            raise stations.Bad("bad_json")
        return d


def respond(start, status: int, body=None, ctype="application/json", req: Req = None, extra=(), cache: str = None):
    """`cache`: the Cache-Control of an answer that is the same for every visitor (status, static files). One a cache may reuse
    without asking again never sets the session cookie, which a shared cache would hand to the next visitor; the pages (no-cache) do,
    so the cookie is set before any script runs. JSON is otherwise no-store."""
    text = {200: "OK", 202: "Accepted", 204: "No Content", 400: "Bad Request", 401: "Unauthorized", 403: "Forbidden", 404: "Not Found",
            405: "Method Not Allowed", 409: "Conflict", 415: "Unsupported Media Type", 422: "Unprocessable Entity", 429: "Too Many Requests",
            500: "Internal Server Error", 503: "Service Unavailable"}
    data = b"" if body is None else body if isinstance(body, bytes) else json.dumps(body).encode()
    hdrs = [("Content-Type", ctype), ("Content-Length", str(len(data))), ("X-Content-Type-Options", "nosniff"),
            ("Referrer-Policy", "no-referrer"), ("X-Robots-Tag", "noindex")] + list(extra)
    cc = cache or ("no-store" if ctype == "application/json" else None)
    if cc:
        hdrs.append(("Cache-Control", cc))
    if req is not None and req.new_cookie and (not cache or any(d in cache for d in ("no-cache", "private", "no-store"))):
        hdrs.append(("Set-Cookie", f"{COOKIE}={req.new_cookie}; Path=/; HttpOnly; SameSite=Lax; Max-Age={ledger.VISITOR_TTL_S}"
                     + ("; Secure" if req.secure else "")))
    start(f"{status} {text.get(status, '')}", hdrs)
    return [data]


# ---- page API -------------------------------------------------------------------------------------------------------------------
@route("POST", "/api/visitor", bucket="BUCKET")
def visitor(req, m):
    """The page's first call (third review, item 5): the visitor id, issued only here, so crawlers and anything else that never runs
    the page's script spend none, and no page load reaches the store."""
    req.json()
    try:
        req.issue()
    except StoreError as e:                                 # the page goes on as the address's cookieless visitor
        print(f"stillhere: store error ({e}) in visitor", file=sys.stderr, flush=True)
    return 200, {}


@route("GET", "/api/status", cache="s-maxage=5")                  # the same for every visitor: the CDN may answer it for 5 s
def status(req, m):
    """Always whole: with the store failing (fourth review, item 4), worker offline and live off, so the page says what it will do."""
    try:
        online, live = jobs.worker_online(max_age_s=5), {"meta": stations.live_ok("meta"), "fireworks": stations.live_ok("fireworks")}
    except StoreError as e:
        print(f"stillhere: store error ({e}) in status", file=sys.stderr, flush=True)
        online, live = False, {"meta": False, "fireworks": False}
    return 200, {"worker_online": online, "worker_hours_et": stations.hours_et(), "saved_line": stations.saved_line(), "live": live, "runs_per_hour": ledger.RUNS_PER_HOUR, "menu": stations.menu(),
        "local_gpu": stations.local_gpu()}


@route("POST", "/api/plan", bucket="BUCKET")
def plan(req, m):
    b = req.json(); return 200, stations.plan(b.get("text"), b.get("backend", "spark-1.3"), req.sid, ip=req.ip)


@route("POST", "/api/audit", bucket="BUCKET")
def audit(req, m):
    return 200, stations.audit(req.json().get("run_id"), req.sid)


@route("POST", "/api/build", bucket="BUCKET")
def build(req, m):
    b = req.json()
    if b.get("old_bug"):
        return 200, stations.old_bug()
    return 200, stations.build(b.get("run_id"), req.sid, b.get("clip_id"), b.get("condition"), b.get("seed"))


@route("POST", "/api/forecast", bucket="BUCKET")
def forecast(req, m):
    b = req.json(); out = stations.forecast(b.get("run_id"), req.sid, b.get("clip_id"), b.get("condition"), b.get("seed"))
    return (202 if out.get("state") == "queued" else 200), out


@route("GET", "/api/forecast/([0-9a-f]{16})", bucket="GET_BUCKET")
def forecast_status(req, m):
    try:
        out = jobs.status(m.group(1))
    except StoreError as e:                                # the job is in the store: 503, so one store error never ends a running
        print(f"stillhere: store error ({e}) in forecast_status", file=sys.stderr, flush=True)   # job's display (the page polls again;
        return 503, paused_forecast(req.env.get("QUERY_STRING", ""))   # second refuter's review of Oct 31, item E), and a body that is
                                                                         # the page's final answer after 20 in a row, never a 500
    if not out:
        return 404, {"error": "no_job"}
    cid = out["recipe"]["clip_id"]
    return 200, {**out, "chip": stations.chip(cid), "end_line": stations.end_line(cid)}


def paused_forecast(query: str) -> dict:
    """A job's final state while the store cannot answer: "failed" with the paused line, and the saved forecast for the recipe the
    page sends with its poll (?clip_id=&condition=&seed=), checked against the live menu like any recipe; without one it can trust,
    the line alone."""
    out = {"status": "failed", "events": [], "why": stations.why_line("store_down", shown=stations.SAVED_SHOWN)}
    q = {k: v[0] for k, v in urllib.parse.parse_qs(query).items()}
    try:
        r = jobs.recipe(q.get("clip_id"), q.get("condition"), int(q.get("seed", "")))
    except ValueError:
        return out
    return {**out, "recipe": r, "fallback": jobs.fallback(r), "chip": stations.chip(r["clip_id"]), "end_line": stations.end_line(r["clip_id"])}


@route("GET", "/api/evidence")
def evidence(req, m):
    return 200, stations.evidence()


@route("POST", "/api/film", bucket="BUCKET")
def film(req, m):
    b = req.json(); return 200, stations.film(b.get("run_id"), req.sid, b.get("concept"), ip=req.ip)


@route("GET", "/api/receipt/([0-9a-f]{16})", bucket="GET_BUCKET")
def receipt(req, m):
    r = stations.get_receipt(m.group(1))
    return (200, r) if r else (404, {"error": "no_receipt"})


# ---- worker API (docs/jobs_api.md) --------------------------------------------------------------------------------------------------
def _worker(req):
    return jobs.authorized(req.env.get("HTTP_AUTHORIZATION", ""))


@route("POST", "/api/worker/claim")
def claim(req, m):
    if not _worker(req):
        return 401, None
    j = jobs.claim(req.json() or {})
    return (200, {"job": j}) if j else (204, None)


@route("POST", "/api/worker/jobs/([0-9a-f]{16})/(event|result|fail)")
def worker_job(req, m):
    if not _worker(req):
        return 401, None
    n = int(req.env.get("CONTENT_LENGTH") or 0)
    if n > 32768:
        return 422, None
    try:
        body = json.loads(req.env["wsgi.input"].read(n) or b"{}")
    except ValueError:
        return 422, None
    fn = {"event": jobs.event, "result": jobs.result, "fail": jobs.fail}[m.group(2)]
    return fn(m.group(1), body if isinstance(body, dict) else {}), None


# ---- static -------------------------------------------------------------------------------------------------------------------
ASSET = re.compile(r'(?:(?:href|src)="|url\()(/[^"?#)]*)(?=[")])')


def versioned(text: str) -> str:
    """A page or style sheet naming each local asset as path?v=<first 8 hex of the SHA-256 of what that path serves>, made as it is
    served, so a deploy that changes a file changes its URL and the day-long cache on assets never pairs a new page with an old
    script, style sheet, image or font. Pages are not versioned: they are served no-cache."""
    def v(m):
        got = None if m.group(1) == "/" or m.group(1).endswith(".html") else static(m.group(1))
        return m.group(0) if got is None else f"{m.group(0)}?v={hashlib.sha256(got[0]).hexdigest()[:8]}"
    return ASSET.sub(v, text)


def static(path: str):
    if path == "/pilot.css":                                                         # the static page's own CSS, one source
        body, ctype = guard._bp.CSS.encode(), "text/css"
    else:
        if path in MEDIA:
            f = MEDIA[path]
        else:
            rel = "index.html" if path in ("/", "") else path.lstrip("/")
            f = (PUBLIC / rel).resolve()
            if not f.is_relative_to(PUBLIC.resolve()):
                return None
        if not f.is_file():
            return None
        body, ctype = f.read_bytes(), mimetypes.guess_type(f.name)[0] or "application/octet-stream"
    if ctype in ("text/html", "text/css"):
        body = versioned(body.decode()).encode()
    return body, ctype + ("; charset=utf-8" if ctype.startswith("text/") else "")


def refused_post(req):
    """(code, body) for a POST to refuse, else None. It needs Content-Type application/json, which a cross-site form or a fetch
    without a preflight cannot send, and an Origin, when the browser sends one, that is this page's own (the worker sends none)."""
    if (req.env.get("CONTENT_TYPE") or "").split(";")[0].strip().lower() != "application/json":
        return 415, {"error": "content_type"}
    origin = req.env.get("HTTP_ORIGIN")
    if origin is not None and not req.own_origin(origin):
        return 403, {"error": "origin"}
    return None


LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


def foreign_host(env) -> bool:
    """Under web/serve.py (STILLHERE_LOCAL_SERVER), a request whose Host is not this machine's: a page on another site whose name was
    rebound to 127.0.0.1 would otherwise count as this page's own origin (Req.own_origin reads the Host) and could queue jobs on the
    visitor's GPU. Not used on Vercel, where the Host is the deployment's name."""
    if os.environ.get("STILLHERE_LOCAL_SERVER") != "1":
        return False
    host = (env.get("HTTP_HOST") or "").strip().lower()
    host = host[1:host.index("]")] if host.startswith("[") and "]" in host else host.rsplit(":", 1)[0] if host.count(":") == 1 else host
    return host not in LOCAL_HOSTS


def app(env, start):
    if foreign_host(env):
        return respond(start, 403, {"error": "host"})
    req, where = Req(env), "static"
    try:
        for method, rx, fn, cache, limit in ROUTES:
            m = rx.fullmatch(req.path)
            if m and req.method == method:
                where, b = rx.pattern, globals()[limit] if limit else None
                code, body = (method == "POST" and refused_post(req)) or (
                    (429, slowed(req)) if b and not b.take(ledger._address(req.ip)) else fn(req, m))
                return respond(start, code, body, req=req, cache=cache)
        if req.path.startswith("/api/"):
            return respond(start, 404 if not any(rx.fullmatch(req.path) for _, rx, *_ in ROUTES) else 405, {"error": "not_found"}, req=req)
        if req.method != "GET":
            return respond(start, 405, {"error": "method"}, req=req)
        s = static(req.path)                                   # no store call: a page loads whatever the store's state
    except stations.Bad as e:
        return respond(start, 400, {"error": str(e)}, req=req)
    except Exception as e:                                 # never the message or a static path: either may quote the request
        print(f"stillhere: {type(e).__name__} on {req.method} {where}", file=sys.stderr, flush=True)
        return respond(start, 500, {"error": "server_error"}, req=req)
    if s is None:
        return respond(start, 404, b"not found", ctype="text/plain; charset=utf-8", req=req)
    csp = ("default-src 'self'; img-src 'self' data:; media-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; "
           "frame-ancestors 'none'; base-uri 'none'")
    cache = "no-cache" if s[1].startswith("text/html") else "max-age=86400"   # assets and media: named by content hash where linked
    return respond(start, 200, s[0], ctype=s[1], req=req, cache=cache, extra=[("Content-Security-Policy", csp)])
