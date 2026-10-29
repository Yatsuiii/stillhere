"""StillHere live worker (cloud plan v2, item 5; docs/worker.md, docs/jobs_api.md). Runs on the laptop inside the sandboxed service
(worker/stillhere-worker.service): pulls jobs from the page's backend over HTTPS (the laptop accepts no inbound connection), one at a
time, TRIBE v2 loaded once (worker/engine.py), posts real stage events and uploads region values only. A job is a recipe and nothing
else ({clip_id, condition, seed}, worker/recipe.py); anything outside the schema, any clip not on data/live_clips.json, and any job
whose cache key the worker does not reproduce is refused without running. Its only secret is its job token.

  serve       pull and run jobs until stopped (STILLHERE_JOBS_URL; token from $CREDENTIALS_DIRECTORY/job_token or --token-file)
  precompute  the 12 clips x 4 conditions at the default seed into data/forecast_cache.json (resumable; same engine as serve)
  self-test   from inside the service: hidden paths must not be listable, model folders must be read-only; prints PASS or FAIL lines
  once        run one recipe locally and print its timings (no network)
Prints counts, timings and status codes only (never the token, never a response body)."""
import argparse
import hmac
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from worker import recipe as R  # noqa: E402

POLL_S = float(os.environ.get("STILLHERE_POLL_S", "2"))
WORKER_NAME = "laptop-4050"
STATE = Path(os.environ.get("STATE_DIRECTORY", HERE.parent / ".worker_state"))
HIDDEN = [p for p in os.environ.get("STILLHERE_SELFTEST_HIDDEN", "").split(":") if p]
READONLY = [p for p in os.environ.get("STILLHERE_SELFTEST_READONLY", "").split(":") if p]


def log(msg: str):
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + msg, flush=True)


class Jobs:
    """The worker side of docs/jobs_api.md. HTTPS only (plain http is accepted for localhost tests)."""

    def __init__(self, base: str, token: str, timeout: float = 30):
        if not (base.startswith("https://") or base.startswith(("http://127.0.0.1", "http://localhost"))):
            raise SystemExit("STILLHERE_JOBS_URL must be https")
        self.base, self.token, self.timeout = base.rstrip("/"), token, timeout

    def post(self, path: str, body: dict):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read()
                return r.status, (json.loads(raw) if raw else None)
        except urllib.error.HTTPError as e:
            return e.code, None

    def claim(self, models: dict):
        return self.post("/api/worker/claim", {"worker": WORKER_NAME, "models": models})

    def event(self, job_id: str, stage: str, lane, status: str, detail: dict):
        self.post(f"/api/worker/jobs/{job_id}/event", {"stage": stage, "lane": lane, "status": status,
                                                       "t_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "detail": detail})

    def result(self, job_id: str, res: dict):
        return self.post(f"/api/worker/jobs/{job_id}/result", res)[0]

    def fail(self, job_id: str, reason: str):
        return self.post(f"/api/worker/jobs/{job_id}/fail", {"reason": reason})[0]


def check_job(job: dict, index: dict) -> tuple:
    """(recipe, key) if the job is acceptable, else raises R.RecipeRefused with the fail reason as its message."""
    if not isinstance(job, dict) or not isinstance(job.get("id"), str) or not isinstance(job.get("recipe"), dict):
        raise R.RecipeRefused("bad_recipe")
    try:
        rec = R.normalize(job["recipe"], index)
    except R.RecipeRefused:
        raise R.RecipeRefused("bad_recipe")
    key = R.cache_key(rec, index)
    if not hmac.compare_digest(key, str(job.get("cache_key", ""))):
        raise R.RecipeRefused("cache_key_mismatch")
    return rec, key


def handle(job: dict, engine, jobs: Jobs, index: dict):
    jid = job.get("id", "")
    try:
        rec, key = check_job(job, index)
    except R.RecipeRefused as e:
        log(f"job refused: {e}"); jobs.fail(jid, str(e)); return
    try:
        built = R.build_condition(rec, index)
        res = engine.run(built, STATE / "jobs", lambda st, lane, status, detail: jobs.event(jid, st, lane, status, detail))
    except Exception as e:                                              # reason codes only, never the traceback
        import torch
        reason = "oom" if isinstance(e, torch.cuda.OutOfMemoryError) else "model_error"
        log(f"job failed: {reason} ({type(e).__name__})"); jobs.fail(jid, reason); return
    code = jobs.result(jid, {"cache_key": key, **res})
    log(f"job done: {rec['clip_id']} {rec['condition']} total {res['timings_s']['total']} s; result {code}")


def read_token(path) -> str:
    p = Path(path) if path else Path(os.environ.get("CREDENTIALS_DIRECTORY", "/nonexistent")) / "job_token"
    tok = p.read_text().strip()
    if len(tok) < 32:
        raise SystemExit("job token missing or too short")
    return tok


def serve(args):
    from worker.engine import Engine
    index = R.load_index(); jobs = Jobs(os.environ["STILLHERE_JOBS_URL"], read_token(args.token_file))
    engine = Engine(STATE / "feature_cache").load(); models = engine.models()
    log(f"engine loaded in {engine.load_s} s on {engine.gpu}; polling every {POLL_S} s")
    backoff = POLL_S
    while True:
        try:
            code, body = jobs.claim(models)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            code, body = None, None
            log(f"claim failed ({type(e).__name__}); retrying in {backoff:.0f} s")
        if code == 200 and body and body.get("job"):
            backoff = POLL_S
            handle(body["job"], engine, jobs, index)
            continue
        if code is not None and code not in (200, 204):
            log(f"claim answered {code}")
        # --max-polls (tests only; 0 = forever) counts every poll that brought no job, failed claims included
        if args.max_polls and args.max_polls > 0:
            args.max_polls -= 1
            if args.max_polls == 0:
                return
        if code is None:
            time.sleep(backoff); backoff = min(backoff * 2, 60)
        else:
            backoff = POLL_S; time.sleep(POLL_S)


def precompute(args):
    from worker.engine import Engine
    index = R.load_index(); out = Path(args.out)
    cache = json.loads(out.read_text()) if out.exists() else {}
    entries = cache.get("entries", {})
    engine = Engine(STATE / "feature_cache").load()
    todo = [{"clip_id": c["id"], "condition": cond, "seed": 0 if cond in R.SEEDLESS else index["default_seed"]}
            for c in index["clips"] for cond in R.CONDITIONS]
    for i, rec in enumerate(todo, 1):
        key = R.cache_key(rec, index)
        if key in entries:
            continue
        res = engine.run(R.build_condition(rec, index), STATE / "jobs", lambda *a: None)
        entries[key] = {"cache_key": key, "recipe": rec, **res}
        doc = {"about": "Cached TRIBE v2 forecasts for the live page (docs/worker.md): the 12 clips x 4 conditions at the default seed, "
                        "computed by worker/tribe_worker.py precompute with the same engine as live jobs. Region values only. "
                        "Forecast, not a scan.",
               "models": index["models"], "default_seed": index["default_seed"], "entries": entries}
        tmp = out.with_suffix(".tmp"); tmp.write_text(json.dumps(doc, indent=1) + "\n"); tmp.replace(out)
        log(f"{i}/{len(todo)} {rec['clip_id']} {rec['condition']}: {res['timings_s']}")


def self_test(args):
    """Run inside the service. Each hidden path must fail to list; each read-only path must refuse a write; the GPU must be visible."""
    ok = True
    for p in HIDDEN:
        try:
            os.listdir(p); log(f"FAIL: could list {p}"); ok = False
        except OSError as e:
            log(f"PASS: cannot list {p} ({type(e).__name__})")
    for p in READONLY:
        try:
            t = Path(p) / ".write_probe"; t.write_text("x"); t.unlink(); log(f"FAIL: could write in {p}"); ok = False
        except OSError as e:
            log(f"PASS: {p} is read-only ({type(e).__name__})")
    try:
        import torch
        log(("PASS" if torch.cuda.is_available() else "FAIL") + ": GPU visible to the service")
        ok &= torch.cuda.is_available()
    except Exception as e:
        log(f"FAIL: torch import ({type(e).__name__})"); ok = False
    for name in ("HOME", "USER"):
        log(f"info: {name}={os.environ.get(name, '')}")
    log("SELF-TEST " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


def once(args):
    from worker.engine import Engine
    index = R.load_index(); rec = R.normalize(json.loads(args.recipe), index)
    engine = Engine(STATE / "feature_cache").load()
    res = engine.run(R.build_condition(rec, index), STATE / "jobs", lambda st, lane, status, d: log(f"event {st} {status}"))
    log(f"{rec}: {res['timings_s']}, words {res['n_words_text_lane']}, peak {res['peak_vram_gb']} GB, key {R.cache_key(rec, index)[:12]}")


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve"); s.add_argument("--token-file"); s.add_argument("--max-polls", type=int, default=0)
    p = sub.add_parser("precompute"); p.add_argument("--out", default=str(HERE.parent / "data/forecast_cache.json"))
    sub.add_parser("self-test")
    o = sub.add_parser("once"); o.add_argument("recipe")
    a = ap.parse_args(argv)
    {"serve": serve, "precompute": precompute, "self-test": self_test, "once": once}[a.cmd](a)


if __name__ == "__main__":
    main()
