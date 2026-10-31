"""Local production-mode smoke test of the hosted app, with no account (docs/deploy_checklist.md, step 8a). It copies exactly the
files `vercel deploy` would upload (tracked files minus .vercelignore) into a temporary folder, serves the root api/index.py `app` from
there as Vercel does (working directory = project root, VERCEL=1, a HOME with nothing in it), and checks two runs:

  A. keys absent: every station answers with its labelled recorded run, the worker API refuses every call, nothing spends.
  B. dummy keys present, the model APIs blocked (HTTPS_PROXY points at a closed port), and a local stand-in for Upstash's REST API:
     PLAN and FILM try their live calls, fail, and fall back with "The live call failed"; AUDIT and BUILD, which need a live plan,
     show their recordings; the ledger's reservations go through the REST store and are released; one jobs-API round trip runs
     through it (the harness writes the run record a live plan would).

In both runs: no key or token appears in any response body, header or page file; a marker sent in every request body never appears
in the server's stdout or stderr. Prints counts and pass/fail only; the dummy keys are generated here and never printed.
usage: PYTHONPATH=<site-packages with httpx, pydantic, numpy, cv2, meta_sam_parser> python web/smoke_local.py"""
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAGES = ("/", "/app.js", "/app.css", "/pilot.css", "/receipt.html", "/receipt.js")


def upload_set() -> list:
    tracked = subprocess.run(["git", "-C", str(REPO), "ls-files"], capture_output=True, text=True, check=True).stdout.split()
    ignored = set(subprocess.run(["git", "-C", str(REPO), "ls-files", "-c", "-i", "--exclude-from=.vercelignore"], capture_output=True,
                                 text=True, check=True).stdout.split())
    return [f for f in tracked if f not in ignored and (REPO / f).is_file()]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]


class RedisStub(BaseHTTPRequestHandler):
    """Upstash's REST API (POST / with a command array, POST /pipeline with an array of them), enough commands for web/store.py."""
    db, exp, seen, lock = {}, {}, [], threading.Lock()

    def log_message(self, *a):
        pass

    def _live(self, k):
        if k in self.exp and self.exp[k] <= time.time():
            self.db.pop(k, None); self.exp.pop(k, None)
        return self.db.get(k)

    def run(self, c):
        op, k = c[0].upper(), c[1] if len(c) > 1 else None
        self.seen.append((op, k.split(":")[0] if k else None))
        if op == "GET":
            v = self._live(k); return None if isinstance(v, (list, dict)) else v
        if op == "MGET":
            return [None if isinstance(self._live(x), (list, dict)) else self._live(x) for x in c[1:]]
        if op == "SET":
            nx, ex = "NX" in c, int(c[c.index("EX") + 1]) if "EX" in c else None
            if nx and self._live(k) is not None:
                return None
            self.db[k] = c[2]; self.exp.pop(k, None)
            if ex:
                self.exp[k] = time.time() + ex
            return "OK"
        if op == "ZREMRANGEBYSCORE":
            z = self._live(k) or {}; hi = float(c[3]); gone = [m for m, t in z.items() if t <= hi]
            for m in gone:
                del z[m]
            return len(gone)
        if op == "ZADD":
            z = self._live(k)
            if not isinstance(z, dict):
                z = self.db[k] = {}
            new = c[3] not in z; z[c[3]] = float(c[2]); return int(new)
        if op == "ZCARD":
            return len(self._live(k) or {})
        if op == "ZREM":
            return int((self._live(k) or {}).pop(c[2], None) is not None)
        if op == "INCRBYFLOAT":
            v = float(self._live(k) or 0) + float(c[2]); self.db[k] = repr(v); return repr(v)
        if op == "EXPIRE":
            if self._live(k) is None or ("NX" in c and k in self.exp):
                return 0
            self.exp[k] = time.time() + int(c[2]); return 1
        if op == "RPUSH":
            self.db.setdefault(k, []).append(c[2]); return len(self.db[k])
        if op == "LREM":
            lst = self._live(k) or []
            if c[3] in lst:
                lst.remove(c[3]); return 1
            return 0
        if op == "LPOP":
            lst = self._live(k); return lst.pop(0) if lst else None
        if op == "LRANGE":
            lst = self._live(k) or []; a, b = int(c[2]), int(c[3]); return lst[a:] if b == -1 else lst[a:b + 1]
        if op == "DEL":
            return int(self.db.pop(k, None) is not None)
        raise ValueError(op)

    def do_POST(self):
        if self.headers.get("Authorization") != f"Bearer {os.environ['SMOKE_UPSTASH_TOKEN']}":
            self.send_response(401); self.end_headers(); return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        with self.lock:
            out = [{"result": self.run(c)} for c in body] if self.path == "/pipeline" else {"result": self.run(body)}
        data = json.dumps(out).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data)))
        self.end_headers(); self.wfile.write(data)


class Run:
    def __init__(self, root: Path, env: dict, marker: str, secrets_: list):
        self.root, self.env, self.marker, self.secrets, self.port = root, env, marker, secrets_, free_port()
        self.seen, self.fails, self.cookie = b"", [], ""

    def __enter__(self):
        code = ("import os, runpy, sys; from wsgiref.simple_server import make_server, WSGIServer; import socketserver;"
                "sys.path.insert(0, os.getcwd()); app = runpy.run_path('api/index.py')['app'];"
                "S = type('S', (socketserver.ThreadingMixIn, WSGIServer), {'daemon_threads': True});"
                f"make_server('127.0.0.1', {self.port}, app, server_class=S).serve_forever()")
        self.log = open(self.root.parent / f"server_{self.port}.log", "w+")
        self.proc = subprocess.Popen([sys.executable, "-c", code], cwd=self.root, env=self.env, stdout=self.log, stderr=subprocess.STDOUT)
        for _ in range(100):
            try:
                self.get("/"); break
            except (urllib.error.URLError, ConnectionError):
                time.sleep(0.2)
        self.call("POST", "/api/visitor", {})               # as the page does: its first call gets the visitor cookie
        return self

    def __exit__(self, *a):
        self.proc.terminate(); self.proc.wait(10)
        self.log.seek(0); self.logtext = self.log.read(); self.log.close()

    def call(self, method, path, body=None, headers=None):
        data = None if body is None else json.dumps({**body, "_smoke": self.marker}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=data, method=method,
                                     headers={"Content-Type": "application/json", "Cookie": self.cookie, **(headers or {})})
        try:
            r = urllib.request.urlopen(req, timeout=180); code, raw, hdrs = r.status, r.read(), dict(r.headers)
        except urllib.error.HTTPError as e:
            code, raw, hdrs = e.code, e.read(), dict(e.headers)
        if "Set-Cookie" in hdrs and not self.cookie:
            self.cookie = hdrs["Set-Cookie"].split(";")[0]
        self.seen += raw + json.dumps(hdrs).encode()
        try:
            return code, json.loads(raw) if raw and hdrs.get("Content-Type", "").startswith("application/json") else raw
        except ValueError:
            return code, raw

    def get(self, path):
        return self.call("GET", path)

    def check(self, ok, what):
        if not ok:
            self.fails.append(what)
        return ok


def stations(r: Run, live: bool) -> dict:
    """One call to each station; returns {station: (mode, why)}."""
    out = {}
    code, d = r.call("POST", "/api/plan", {"text": "Plan an exploratory stock TRIBE forecast of The Black Cat, intact versus sentences "
                                                   "shuffled, in the language network.", "backend": "spark-1.3"})
    out["plan spark-1.3"] = (d.get("mode"), d.get("why")); r.check(code == 200 and d.get("label") == "recorded run", "plan label")
    code, d = r.call("POST", "/api/plan", {"text": "Plan it on The Last Leaf, played backwards.", "backend": "spark-open"})
    out["plan spark-open"] = (d.get("mode"), d.get("why"))
    code, d = r.call("POST", "/api/audit", {"run_id": None})
    out["audit"] = (d.get("mode"), d.get("why")); r.check(d.get("label", "").startswith("Advisory check, not measured"), "audit label")
    code, d = r.call("POST", "/api/build", {"run_id": None, "clip_id": "story_black_cat", "condition": "sentence_scramble", "seed": 20261028})
    out["build story_black_cat"] = (d.get("mode"), d.get("why")); r.check(bool(d.get("checklist")) and d.get("story"), "build replay + card")
    code, d = r.call("POST", "/api/build", {"old_bug": True})
    out["build old bug"] = (d.get("mode"), None); r.check(d.get("pass") is False, "old bug")
    code, d = r.call("POST", "/api/film", {"concept": "face"})
    out["film"] = (d.get("mode"), d.get("why")); r.check(len(d.get("frames") or []) == 3 and "CC BY 3.0" in d.get("credit", ""), "film replay")
    code, d = r.call("POST", "/api/forecast", {"clip_id": "story_black_cat", "condition": "sentence_scramble", "seed": 3})
    fb = d.get("fallback") or {}
    out["forecast seed 3"] = (d.get("state"), fb.get("note")); r.check(fb.get("source") == "cache" and "seed 3" in (fb.get("note") or ""), "forecast fallback")
    code, d = r.get("/api/evidence"); r.check(code == 200 and "html" in d, "evidence")
    for p in PAGES:
        code, _ = r.get(p); r.check(code == 200, f"page {p}")
    return out


def main():
    marker = "SMOKE-BODY-" + secrets.token_hex(8)
    dummy = {k: f"dummy-{k.lower()}-" + secrets.token_hex(12) for k in ("MODEL_API_KEY", "FIREWORKS_API_KEY", "BASETEN_API_KEY",
                                                                         "STILLHERE_WORKER_TOKEN", "UPSTASH_REDIS_REST_TOKEN")}
    os.environ["SMOKE_UPSTASH_TOKEN"] = dummy["UPSTASH_REDIS_REST_TOKEN"]
    tmp = Path(tempfile.mkdtemp(prefix="stillhere_smoke_")); root, home = tmp / "project", tmp / "home"
    files = upload_set()
    for f in files:
        (root / f).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(REPO / f, root / f)
    home.mkdir()
    base = {"PATH": "/usr/bin:/bin", "HOME": str(home), "PYTHONPATH": os.environ.get("PYTHONPATH", ""), "VERCEL": "1",
            "VERCEL_ENV": "production", "STILLHERE_QUESTION_BLOCKLIST": "web/server_question_blocklist.txt",
            "WORKER_HOURS_ET": "10:00-14:00 ET, Nov 4-9", "PYTHONDONTWRITEBYTECODE": "1"}
    report, ok = {"uploaded_files": len(files), "uploaded_bytes": sum((REPO / f).stat().st_size for f in files)}, True

    with Run(root, dict(base), marker, list(dummy.values())) as a:
        code, st = a.get("/api/status")
        a.check(st["live"] == {"meta": False, "fireworks": False} and not st["worker_online"], "A: live off")
        a.check(st["saved_line"].endswith("10:00-14:00 ET, Nov 4-9."), "A: hours line")
        report["A stations"] = stations(a, live=False)
        a.check(all(v[0] in ("recorded", "offline") for v in report["A stations"].values()), "A: all recorded")
        code, _ = a.call("POST", "/api/worker/claim", {}, {"Authorization": "Bearer anything"}); a.check(code == 401, "A: worker 401")
    redis = ThreadingHTTPServer(("127.0.0.1", free_port()), RedisStub); threading.Thread(target=redis.serve_forever, daemon=True).start()
    envb = {**base, **dummy, "UPSTASH_REDIS_REST_URL": f"http://127.0.0.1:{redis.server_address[1]}", "HTTPS_PROXY": "http://127.0.0.1:9",
            "NO_PROXY": "127.0.0.1,localhost"}
    with Run(root, envb, marker, list(dummy.values())) as b:
        code, st = b.get("/api/status")
        b.check(st["live"] == {"meta": True, "fireworks": True}, "B: live on with keys and a shared store")
        report["B stations"] = stations(b, live=True)
        b.check(report["B stations"]["plan spark-1.3"] == ("recorded", "The live call failed, so this station shows its recorded run."),
                "B: plan failed -> recorded")
        W = {"Authorization": f"Bearer {dummy['STILLHERE_WORKER_TOKEN']}"}
        b.check(b.call("POST", "/api/worker/claim", {}, {"Authorization": "Bearer wrong"})[0] == 401, "B: bad token 401")
        b.check(b.call("POST", "/api/worker/claim", {}, W)[0] == 204, "B: claim 204 (heartbeat)")
        # A live forecast belongs to a live plan's run, and every plan here fails by design, so the harness writes one run record and
        # its stored plan into the REST store for this session (what a live PLAN writes), then asks for a seed with no saved forecast.
        rid = secrets.token_hex(8)
        RedisStub.db[f"run:{rid}"] = json.dumps({"sid": b.cookie.split("=", 1)[1], "t": time.time()})
        RedisStub.db[f"runspec:{rid}"] = json.dumps({"spec": {"decline_reason": "none"}, "route": {
            "kind": "live", "clip_id": "story_telltale_heart", "condition": "sentence_scramble", "seed": 20261028}})
        code, d = b.call("POST", "/api/forecast", {"run_id": rid, "clip_id": "story_telltale_heart", "condition": "sentence_scramble",
                                                   "seed": 4})
        if b.check(code == 202 and d.get("state") == "queued", "B: job queued"):
            code, j = b.call("POST", "/api/worker/claim", {}, W); job = j["job"]
            b.call("POST", f"/api/worker/jobs/{job['id']}/event", {"stage": "build", "lane": None, "status": "start"}, W)
            n = 37
            res = {"cache_key": job["cache_key"], "tr_s": 1.0, "seconds": n, "regions": {"language": [0.1] * n, "hearing": [0.2] * n},
                   "word_rate": [2] * n, "n_words_text_lane": 90, "models": {}, "input_sha256": "0" * 64, "peak_vram_gb": 3.5,
                   "worker_version": "smoke", "gpu": "smoke", "timings_s": {"total": 1.0}, "computed_utc": "2026-10-30T00:00:00Z"}
            b.check(b.call("POST", f"/api/worker/jobs/{job['id']}/result", res, W)[0] == 204, "B: result 204")
            code, s = b.get(f"/api/forecast/{d['job_id']}")
            b.check(s.get("status") == "done" and len(s.get("events", [])) == 1, "B: job done via the REST store")
        spend = {k: v for k, v in RedisStub.db.items() if k.startswith("spend:")}
        report["B ledger keys in the REST store"] = sorted(spend)
        b.check(any(k.startswith("spend:meta:") for k in spend) and all(abs(float(v)) < 1e-9 for v in spend.values()),
                "B: reservations made and released")
        report["B REST commands"] = sorted({op for op, _ in RedisStub.seen})
    redis.shutdown()

    for name, r in (("A", a), ("B", b)):
        leaked = [k for k, v in dummy.items() if v.encode() in r.seen]          # every response, header and page fetched above
        r.check(not leaked, f"{name}: no key in any response, header or page")
        r.check(marker not in r.logtext, f"{name}: no request body in the server log")
        r.check(not any(v in r.logtext for v in dummy.values()), f"{name}: no key in the server log")
        report[f"{name} server log lines"] = len(r.logtext.splitlines())
        report[f"{name} server log, other than request lines"] = [l for l in r.logtext.splitlines() if '" 2' not in l and '" 4' not in l]
        report[f"{name} failures"] = r.fails; ok &= not r.fails
    shutil.rmtree(tmp)
    print(json.dumps(report, indent=1))
    print("SMOKE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
