"""The hosted Pilot app as one WSGI callable (`app`), standard library only around the repo's own modules: the stations rail page
(web/public), the station API, receipts, and the jobs API for the laptop worker (docs/jobs_api.md). Local: `python web/serve.py`.
Vercel: api/index.py exports the same `app`. Keys stay in the server's environment; no response carries one, and request bodies are
never logged."""
import json
import mimetypes
import re
import secrets
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from web import guard, jobs, ledger, stations  # noqa: E402

PUBLIC = REPO / "web/public"
MEDIA = {"/media/old_bug.wav": REPO / stations.OLD_BUG["audio"]}
MAX_BODY = 8192
ROUTES = []
COOKIE = "sh_sid"


def route(method: str, pattern: str):
    def deco(fn):
        ROUTES.append((method, re.compile(f"^{pattern}$"), fn)); return fn
    return deco


class Req:
    def __init__(self, env):
        self.env, self.method, self.path = env, env["REQUEST_METHOD"], env.get("PATH_INFO") or "/"
        ck = dict(p.strip().split("=", 1) for p in env.get("HTTP_COOKIE", "").split(";") if "=" in p)
        sid = ck.get(COOKIE, "")
        self.new_sid = not re.fullmatch(r"[0-9a-f]{32}", sid)
        self.sid = secrets.token_hex(16) if self.new_sid else sid
        self.secure = env.get("HTTP_X_FORWARDED_PROTO", env.get("wsgi.url_scheme")) == "https"

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


def respond(start, status: int, body=None, ctype="application/json", req: Req = None, extra=()):
    text = {200: "OK", 202: "Accepted", 204: "No Content", 400: "Bad Request", 401: "Unauthorized", 404: "Not Found",
            405: "Method Not Allowed", 409: "Conflict", 422: "Unprocessable Entity", 429: "Too Many Requests", 503: "Service Unavailable"}
    data = b"" if body is None else body if isinstance(body, bytes) else json.dumps(body).encode()
    hdrs = [("Content-Type", ctype), ("Content-Length", str(len(data))), ("X-Content-Type-Options", "nosniff"),
            ("Referrer-Policy", "no-referrer"), ("X-Robots-Tag", "noindex")] + list(extra)
    if ctype == "application/json":
        hdrs.append(("Cache-Control", "no-store"))
    if req is not None and req.new_sid:
        hdrs.append(("Set-Cookie", f"{COOKIE}={req.sid}; Path=/; HttpOnly; SameSite=Lax; Max-Age=86400" + ("; Secure" if req.secure else "")))
    start(f"{status} {text.get(status, '')}", hdrs)
    return [data]


# ---- page API -------------------------------------------------------------------------------------------------------------------
@route("GET", "/api/status")
def status(req, m):
    return 200, {"worker_online": jobs.worker_online(), "worker_hours_et": stations.hours_et(),
                 "live": {"meta": stations.live_ok("meta"), "fireworks": stations.live_ok("fireworks")},
                 "budget": ledger.remaining(), "runs_per_hour": ledger.RUNS_PER_HOUR, "menu": stations.menu()}


@route("POST", "/api/plan")
def plan(req, m):
    b = req.json(); return 200, stations.plan(b.get("text"), b.get("backend", "spark-1.3"), req.sid)


@route("POST", "/api/audit")
def audit(req, m):
    return 200, stations.audit(req.json().get("run_id"), req.sid)


@route("POST", "/api/build")
def build(req, m):
    b = req.json()
    if b.get("old_bug"):
        return 200, stations.old_bug()
    return 200, stations.build(b.get("run_id"), req.sid, b.get("clip_id"), b.get("condition"), b.get("seed"))


@route("POST", "/api/forecast")
def forecast(req, m):
    b = req.json(); out = stations.forecast(b.get("run_id"), req.sid, b.get("clip_id"), b.get("condition"), b.get("seed"))
    return (202 if out.get("state") == "queued" else 200), out


@route("GET", "/api/forecast/([0-9a-f]{16})")
def forecast_status(req, m):
    out = jobs.status(m.group(1))
    return (200, {**out, "chip": stations.CHIP, "end_line": stations.GRAY_LINE}) if out else (404, {"error": "no_job"})


@route("GET", "/api/evidence")
def evidence(req, m):
    return 200, stations.evidence()


@route("POST", "/api/film")
def film(req, m):
    b = req.json(); return 200, stations.film(b.get("run_id"), req.sid, b.get("concept"))


@route("GET", "/api/receipt/([0-9a-f]{16})")
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
    req.json(); j = jobs.claim({})
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
def static(path: str):
    if path == "/pilot.css":                                                         # the static page's own CSS, one source
        return guard._bp.CSS.encode(), "text/css; charset=utf-8"
    if path in MEDIA:
        f = MEDIA[path]
    else:
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        f = (PUBLIC / rel).resolve()
        if not f.is_relative_to(PUBLIC.resolve()):
            return None
    if not f.is_file():
        return None
    ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
    return f.read_bytes(), ctype + ("; charset=utf-8" if ctype.startswith("text/") else "")


def app(env, start):
    req = Req(env)
    for method, rx, fn in ROUTES:
        m = rx.match(req.path)
        if m:
            if req.method != method:
                continue
            try:
                code, body = fn(req, m)
            except stations.Bad as e:
                code, body = 400, {"error": str(e)}
            return respond(start, code, body, req=req)
    if req.path.startswith("/api/"):
        return respond(start, 404 if not any(rx.match(req.path) for _, rx, _ in ROUTES) else 405, {"error": "not_found"}, req=req)
    if req.method != "GET":
        return respond(start, 405, {"error": "method"}, req=req)
    s = static(req.path)
    if s is None:
        return respond(start, 404, b"not found", ctype="text/plain; charset=utf-8", req=req)
    csp = ("default-src 'self'; img-src 'self' data:; media-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; "
           "frame-ancestors 'none'; base-uri 'none'")
    return respond(start, 200, s[0], ctype=s[1], req=req, extra=[("Content-Security-Policy", csp)])
